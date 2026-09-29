from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass, field, replace
import math
import os
import threading
import time
import uuid
from typing import Any
import zlib

import cv2
from fastapi import FastAPI, HTTPException, Query, Request
import numpy as np
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from .depth_anything_v3 import (
    DepthAnythingV3Error,
    DepthAnythingV3MetricEstimator,
    MetricDepthPrediction,
)
from .phase1_inference import Phase1InferenceError, Phase1JetsonRuntime
from .phase1_contract import (
    MAX_JPEG_BYTES, Phase1FlightSession, phase1_server_response,
    validate_input_age, validate_request_identity,
)
from .native_depth import native_depth_status
from .small_obstacle_perception import make_native_demo_contract
from .vertical_avoidance import (
    DepthCorridorStats,
    VerticalAvoidanceConfig,
    VerticalAvoidanceDecision,
    VerticalAvoidanceState,
    VerticalDirection,
    VerticalObstacleAvoidanceCore,
)


PHASE1_EVENT_TYPES = frozenset(
    {
        "FIRE_SMOKE",
        "HUMAN_VIOLENCE",
        "LITTERING",
        "INTRUSION_ATTEMPT",
        "CALL_FOR_HELP",
        "VEHICLE_ACCIDENT",
    }
)

NATIVE_DEPTH_ENCODINGS = frozenset({"16UC1", "32FC1"})
MAX_D435I_REQUEST_BYTES = 16 * 1024 * 1024


@dataclass(slots=True)
class JetsonDepthRuntime:
    estimator: DepthAnythingV3MetricEstimator
    policy: VerticalObstacleAvoidanceCore

    def evaluate_jpeg(
        self,
        payload: bytes,
        *,
        focal_px: float,
        current_z_ned_m: float,
        gimbal_forward: bool,
    ) -> tuple[VerticalAvoidanceDecision, float]:
        if not gimbal_forward:
            return self.policy._stale("gimbal_not_forward"), 0.0
        image_bgr = _decode_jpeg(payload)
        return self.evaluate_bgr(
            image_bgr,
            focal_px=focal_px,
            current_z_ned_m=current_z_ned_m,
            gimbal_forward=gimbal_forward,
        )

    def evaluate_bgr(
        self,
        image_bgr: np.ndarray,
        *,
        focal_px: float,
        current_z_ned_m: float,
        gimbal_forward: bool,
    ) -> tuple[VerticalAvoidanceDecision, float]:
        if not gimbal_forward:
            return self.policy._stale("gimbal_not_forward"), 0.0
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        started = time.monotonic()
        prediction = self.estimator.estimate(image_rgb, fx_px=focal_px)
        inference_s = time.monotonic() - started
        decision = self.policy.evaluate(
            _masked_depth(prediction),
            frame_age_s=inference_s,
            current_altitude_m=max(0.0, -current_z_ned_m),
            confidence=_relative_confidence(prediction),
        )
        return decision, inference_s * 1000.0


@dataclass(slots=True)
class JetsonNativeDepthRuntime:
    """Evaluate D435 metric depth on Jetson without monocular inference."""

    policy: VerticalObstacleAvoidanceCore
    _lock: Any = field(default_factory=threading.Lock, repr=False)
    geometry_tracker: Any = field(default=None, repr=False)
    _nominal_avoidance_altitude_m: float | None = field(default=None, repr=False)
    last_diagnostic: dict[str, Any] | None = field(default=None, repr=False)
    geometry_max_age_s: float = 0.25

    def evaluate(
        self,
        depth_m: np.ndarray,
        *,
        current_z_ned_m: float,
        frame_age_s: float,
        gimbal_forward: bool,
        forward_speed_mps: float = 0.0,
        pipeline_latency_s: float = 0.0,
        pose_age_s: float = 0.0,
        pair_skew_s: float = 0.0,
        angular_rate_rad_s: float = 0.0,
        geometry_metadata: dict[str, Any] | None = None,
    ) -> tuple[VerticalAvoidanceDecision, float]:
        started = time.monotonic()
        # Stateful hysteresis is one stream per service. Concurrent HTTP
        # workers must not race its counters. Queue/preprocess time consumes
        # the original observation lifetime, never a fresh timeout.
        with self._lock:
            if not gimbal_forward:
                return self.policy._stale("gimbal_not_forward"), 0.0
            values = np.asarray(depth_m, dtype=np.float32).copy()
            # Only Gazebo's +inf represents a ray beyond its clipping plane.
            # D435 16UC1 missing samples are zero and remain invalid.
            values[np.isposinf(values)] = np.float32(self.policy.config.max_depth_m)
            values[values <= 0.0] = np.nan
            elapsed = time.monotonic() - started
            # Evaluate a provisional state: an over-budget result must not
            # commit an expired CLEAR/release into the hysteresis state.
            trial = copy(self.policy)
            geometry = None
            trial_geometry = deepcopy(self.geometry_tracker)
            if self.policy.config.require_geometry:
                geometry_started = time.monotonic()
                trial_geometry, geometry = _observe_native_geometry(
                    depth_m, geometry_metadata, trial_geometry,
                    frame_age_s=frame_age_s + elapsed,
                    pose_age_s=pose_age_s + elapsed,
                    current_z_ned_m=current_z_ned_m,
                    avoidance_active=self.policy.active or self.policy._roof_guard_active,
                    nominal_altitude_m=(self._nominal_avoidance_altitude_m
                                        if self._nominal_avoidance_altitude_m is not None
                                        else max(0.0, -current_z_ned_m)),
                    metadata_elapsed_s=elapsed,
                    geometry_max_age_s=self.geometry_max_age_s,
                )
                geometry["geometry_ms"] = (time.monotonic()-geometry_started)*1000.0
            elapsed = time.monotonic() - started
            decision = trial.evaluate(
                values,
                frame_age_s=frame_age_s + elapsed,
                current_altitude_m=max(0.0, -current_z_ned_m),
                forward_speed_mps=forward_speed_mps,
                pipeline_latency_s=pipeline_latency_s + elapsed,
                pose_age_s=pose_age_s + elapsed,
                pair_skew_s=pair_skew_s,
                angular_rate_rad_s=angular_rate_rad_s,
                geometry=geometry,
            )
            new_episode = trial.active and not self.policy.active and not self.policy._roof_guard_active
            if self.policy.config.scenario_demo:
                # Begin at the first obstacle confirmation, before an active
                # escape. Previous-building bounds must not enter BRAKE's
                # accumulated far boundary, even on that first HOLD sample.
                new_episode = (not self.policy.active and not self.policy._roof_guard_active
                               and self.policy.trigger_count == 0
                               and (trial.trigger_count > 0 or trial.active))
                if new_episode and geometry is not None:
                    from .depth_geometry import DepthGeometryObservation
                    geometry = DepthGeometryObservation(
                        geometry_valid=geometry.get("geometry_valid") is True,
                        reason="scenario_new_obstacle_episode",
                    ).as_payload()
            decision = replace(decision, geometry=geometry)
            if self.policy.config.scenario_demo:
                prefix = "scenario_front_detect_2m_pass_3m_v1:" if self.policy.config.front_climb_demo else "scenario_demo_minimal_v1:"
                decision = replace(decision, reason=prefix+decision.reason)
            elapsed = time.monotonic() - started
            pose_deadline = min(
                self.policy.config.max_pose_age_s,
                getattr(getattr(trial_geometry, "config", None), "pose_timeout_s",
                        self.policy.config.max_pose_age_s),
            )
            if (
                frame_age_s + elapsed > self.policy.config.stale_timeout_s
                or pose_age_s + elapsed > pose_deadline
                or (geometry_metadata is not None
                    and geometry_metadata.get("attitude_age_s") is not None
                    and geometry_metadata["attitude_age_s"] + elapsed
                    > pose_deadline)
            ):
                decision = self.policy._stale("native_processing_deadline_exceeded")
            else:
                if new_episode:
                    self._nominal_avoidance_altitude_m = max(0.0, -current_z_ned_m)
                self.policy = trial
                # A prior building's passage must not authorize a new one.
                self.geometry_tracker = None if new_episode else trial_geometry
            self.last_diagnostic = {
                "state": decision.state.value, "reason": decision.reason,
                "geometry": decision.geometry,
                "inference_ms": (time.monotonic()-started)*1000.0,
            }
        return decision, (time.monotonic() - started) * 1000.0


def _observe_native_geometry(
    depth_m, metadata, tracker, *, frame_age_s, pose_age_s,
    current_z_ned_m, avoidance_active, nominal_altitude_m,
    metadata_elapsed_s=0.0,
    geometry_max_age_s=0.25,
):
    """Build calibrated inputs; receipt ages share Jetson's monotonic clock.

    Unknown calibration is a fail-closed result, never an assumed horizontal
    camera. Raw missing depth is retained for the geometric free-space test.
    """
    invalid = {"geometry_valid": False, "roof_clearance_verified": False,
               "roof_passage_verified": False, "obstacle_extent_valid": False,
               "roof_vertical_gap_m": None, "roof_height_m": None,
               "geometry_reason": "geometry_metadata_missing_or_invalid"}
    if not metadata or metadata.get("calibrated") is not True:
        return tracker, invalid
    try:
        from .depth_geometry import (
            DepthIntrinsics, CameraExtrinsics, VehiclePose, DepthGeometryTracker,
            DepthGeometryConfig,
        )
        attitude_age = float(metadata["attitude_age_s"]) + metadata_elapsed_s
        info_age = float(metadata["camera_info_age_s"]) + metadata_elapsed_s
        if (not math.isfinite(attitude_age) or not 0.0 <= attitude_age <= 0.5
                or not math.isfinite(info_age) or not 0.0 <= info_age <= 5.0):
            return tracker, {**invalid, "geometry_reason": "geometry_metadata_stale"}
        intrinsics = DepthIntrinsics(
            width=metadata["width"], height=metadata["height"],
            fx=metadata["fx"], fy=metadata["fy"],
            cx=metadata["cx"], cy=metadata["cy"], rectified=True,
        )
        roll, pitch, yaw = [float(value) for value in metadata["mount_rpy"]]
        cr, cp, cy = math.cos(roll/2), math.cos(pitch/2), math.cos(yaw/2)
        sr, sp, sy = math.sin(roll/2), math.sin(pitch/2), math.sin(yaw/2)
        mount_q = (cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                   cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy)
        extrinsics = CameraExtrinsics.from_mount_quaternion(
            mount_q, tuple(metadata["mount_translation"]),
        )
        now = time.monotonic()
        pose = VehiclePose(
            position_ned_m=tuple(metadata["position_ned_m"]),
            quaternion_body_to_ned_wxyz=tuple(metadata["quaternion"]),
            timestamp_s=now-max(pose_age_s, attitude_age),
        )
        if tracker is None:
            tracker = DepthGeometryTracker(DepthGeometryConfig(pose_timeout_s=geometry_max_age_s))
        heading = float(metadata["heading_rad"])
        observation = tracker.observe(
            depth_m, intrinsics=intrinsics, extrinsics=extrinsics, pose=pose,
            depth_timestamp_s=now-frame_age_s, now_s=now,
            route_direction_ned=(math.cos(heading), math.sin(heading), 0.0),
            avoidance_active=avoidance_active,
            nominal_altitude_m=nominal_altitude_m,
        )
        result = observation.as_payload()
        result["geometry_reason"] = result.pop("reason", "unknown")
        return tracker, result
    except (ValueError, TypeError, KeyError, OverflowError):
        return tracker, invalid


def _decode_jpeg(payload: bytes) -> np.ndarray:
    encoded = np.frombuffer(payload, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("request body is not a valid JPEG image")
    return image_bgr


def _decode_d435i_request(
    payload: bytes,
    *,
    jpeg_size: int,
    depth_width: int,
    depth_height: int,
    depth_step: int,
    depth_encoding: str,
    depth_bigendian: bool,
    compressed: bool,
    decode_jpeg: bool = True,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Decode a bounded JPEG + packed native-depth request body.

    The uncompressed body is ``jpeg || sensor_msgs/Image.data``. Metadata is
    carried as validated query parameters so the Jetson service does not need
    the optional multipart package.
    """

    if len(payload) > MAX_D435I_REQUEST_BYTES:
        raise ValueError("D435i request body is too large")
    if compressed:
        try:
            decoder = zlib.decompressobj()
            decoded = decoder.decompress(
                payload, MAX_D435I_REQUEST_BYTES + 1
            )
            if decoder.unconsumed_tail or len(decoded) > MAX_D435I_REQUEST_BYTES:
                raise ValueError("D435i decompressed body is too large")
            decoded += decoder.flush()
            payload = decoded
        except zlib.error as exc:
            raise ValueError("D435i request compression is invalid") from exc
        if len(payload) > MAX_D435I_REQUEST_BYTES:
            raise ValueError("D435i decompressed body is too large")
    if depth_encoding not in NATIVE_DEPTH_ENCODINGS:
        raise ValueError(
            "depth encoding must be 16UC1 millimetres or 32FC1 metres"
        )
    itemsize = 2 if depth_encoding == "16UC1" else 4
    packed_row_bytes = depth_width * itemsize
    if depth_step < packed_row_bytes:
        raise ValueError("depth step is smaller than one packed row")
    depth_size = depth_step * depth_height
    if jpeg_size <= 0 or jpeg_size + depth_size != len(payload):
        raise ValueError("D435i request sizes do not match the body")

    image_bgr = _decode_jpeg(payload[:jpeg_size]) if decode_jpeg else None
    dtype = np.dtype(
        (">u2" if depth_bigendian else "<u2")
        if depth_encoding == "16UC1"
        else (">f4" if depth_bigendian else "<f4")
    )
    raw = np.frombuffer(payload, dtype=np.uint8, offset=jpeg_size)
    packed = np.ascontiguousarray(
        raw.reshape(depth_height, depth_step)[:, :packed_row_bytes]
    )
    depth_m = packed.view(dtype).reshape(depth_height, depth_width).astype(
        np.float32, copy=False
    )
    if depth_encoding == "16UC1":
        depth_m = depth_m * np.float32(0.001)
    return image_bgr, depth_m


def _masked_depth(prediction: MetricDepthPrediction) -> np.ndarray:
    depth = prediction.depth_m.copy()
    sky = getattr(prediction, "sky_mask", None)
    valid = np.isfinite(depth) & (depth > 0.0)
    if sky is not None:
        # For vertical corridor selection, a confident sky pixel represents an
        # open ray rather than an unknown ray. DA3's own metric model uses the
        # same convention by assigning sky its maximum depth.
        depth[np.asarray(sky, dtype=bool)] = 80.0
        valid |= np.asarray(sky, dtype=bool)
    depth[~valid] = np.nan
    return depth


def _relative_confidence(
    prediction: MetricDepthPrediction,
) -> np.ndarray | None:
    """Convert DA3's uncalibrated positive confidence to a robust frame mask.

    DA3 confidence is used relatively by the official model and is not a [0, 1]
    probability. Keeping the top 80% of finite, non-sky pixels avoids inventing
    a probability calibration while still rejecting the least reliable pixels.
    """

    if prediction.confidence is None:
        return None
    values = prediction.confidence
    sky = getattr(prediction, "sky_mask", None)
    valid = np.isfinite(values) & np.isfinite(prediction.depth_m)
    valid &= prediction.depth_m > 0.0
    if not np.any(valid):
        if sky is None or not np.any(sky):
            return np.zeros(values.shape, dtype=np.float32)
        result = np.full(values.shape, np.nan, dtype=np.float32)
        result[np.asarray(sky, dtype=bool)] = 1.0
        return result
    threshold = float(np.percentile(values[valid], 20.0))
    result = np.full(values.shape, np.nan, dtype=np.float32)
    result[valid] = np.where(values[valid] >= threshold, 1.0, 0.0)
    if sky is not None:
        result[np.asarray(sky, dtype=bool)] = 1.0
    return result


def decision_payload(
    decision: VerticalAvoidanceDecision, *, inference_ms: float
) -> dict[str, Any]:
    corridors: DepthCorridorStats | None = decision.corridors
    return {
        "state": decision.state.value,
        "direction": decision.direction.value,
        "vertical_velocity_ned_mps": decision.velocity_ned_mps[2],
        "lateral_velocity_body_mps": decision.velocity_ned_mps[1],
        "front_distance_m": (
            _finite_or_none(corridors.center.near_distance_m)
            if corridors is not None
            else None
        ),
        "upper_clearance_m": (
            _finite_or_none(corridors.upper.near_distance_m)
            if corridors is not None
            else None
        ),
        "lower_clearance_m": (
            _finite_or_none(corridors.lower.near_distance_m)
            if corridors is not None
            else None
        ),
        "left_clearance_m": (
            _finite_or_none(corridors.left.near_distance_m)
            if corridors is not None and corridors.left is not None
            else None
        ),
        "right_clearance_m": (
            _finite_or_none(corridors.right.near_distance_m)
            if corridors is not None and corridors.right is not None
            else None
        ),
        "reason": decision.reason,
        "inference_ms": inference_ms,
        "effective_trigger_distance_m": decision.effective_trigger_distance_m,
        "effective_release_distance_m": decision.effective_release_distance_m,
        "descent_corridor_clear": decision.descent_corridor_clear,
        "corridor_quality": ({name: {
            "valid_fraction": getattr(corridors, name).valid_fraction,
            "obstacle_fraction": getattr(corridors, name).obstacle_fraction,
        } for name in ("upper", "center", "lower")} if corridors else {}),
        "geometry_valid": False,
        "roof_clearance_verified": False,
        "roof_passage_verified": False,
        "obstacle_extent_valid": False,
        "roof_vertical_gap_m": None,
        "roof_height_m": None,
        "obstacle_far_north_m": None,
        "obstacle_far_east_m": None,
        **({key: value for key, value in decision.geometry.items()
            if key in {
                "geometry_valid", "roof_clearance_verified", "roof_vertical_gap_m",
                "roof_height_m", "roof_passage_verified", "obstacle_extent_valid",
                "obstacle_far_north_m", "obstacle_far_east_m", "geometry_reason",
                "geometry_ms", "roof_observation_age_s", "sampled_points",
                "roof_patch_count", "footprint_free_fraction", "observed_front_distance_m",
            }} if decision.geometry else {}),
    }


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def create_app(
    runtime: JetsonDepthRuntime | None = None,
    phase1_runtime: Phase1JetsonRuntime | Any | None = None,
    *,
    observation_only: bool = False,
    phase1_profile: str | None = None,
    geometry_max_age_s: float = 0.25,
) -> FastAPI:
    if not math.isfinite(geometry_max_age_s) or geometry_max_age_s <= 0.0:
        raise ValueError("geometry_max_age_s must be finite and positive")
    # Fix runtime ownership before constructing models or accepting any input.
    # A flight session lock alone cannot protect temporal windows/event cursors
    # if a compatibility endpoint can invoke the same runtime outside that lock.
    # Keep the selected value in this closure, not mutable request/app state.
    if phase1_profile is None:
        phase1_profile = os.environ.get("JOLGWA_PHASE1_PROFILE", "flight_exclusive")
    if not isinstance(phase1_profile, str) or phase1_profile not in {
        "flight_exclusive", "legacy_combined",
    }:
        raise ValueError("phase1_profile_invalid: expected flight_exclusive or legacy_combined")

    production_defaults = runtime is None
    if runtime is None:
        process_res = int(os.environ.get("JOLGWA_DA3_PROCESS_RES", "504"))
        stale_timeout = float(
            os.environ.get("JOLGWA_DEPTH_STALE_TIMEOUT_S", "2.0")
        )
        runtime = JetsonDepthRuntime(
            estimator=DepthAnythingV3MetricEstimator(
                device=os.environ.get("JOLGWA_DA3_DEVICE", "cuda"),
                process_res=process_res,
            ),
            policy=VerticalObstacleAvoidanceCore(
                VerticalAvoidanceConfig(stale_timeout_s=stale_timeout)
            ),
        )
    if observation_only:
        phase1_runtime = None
    if (
        phase1_runtime is None
        and production_defaults
        and not observation_only
        and os.environ.get("JOLGWA_PHASE1_ENABLED", "true").lower()
        in {"1", "true", "yes"}
    ):
        phase1_runtime = Phase1JetsonRuntime(
            os.environ.get(
                "JOLGWA_PHASE1_ROOT",
                "/home/jetson/jolgwa/phase1-demo-local",
            ),
            config_path=os.environ.get("JOLGWA_PHASE1_CONFIG") or None,
            device=os.environ.get("JOLGWA_PHASE1_DEVICE", "0"),
        )

    native_depth_config = VerticalAvoidanceConfig(
        trigger_distance_m=float(
            os.environ.get("JOLGWA_D435_TRIGGER_DISTANCE_M")
            or os.environ.get("JOLGWA_D435I_TRIGGER_DISTANCE_M", "3.0")
        ),
        release_distance_m=float(
            os.environ.get("JOLGWA_D435_RELEASE_DISTANCE_M")
            or os.environ.get("JOLGWA_D435I_RELEASE_DISTANCE_M", "5.5")
        ),
        minimum_standoff_m=float(
            os.environ.get("JOLGWA_D435_MINIMUM_STANDOFF_M", "2.5")
        ),
        emergency_margin_m=float(
            os.environ.get("JOLGWA_D435_EMERGENCY_MARGIN_M", "0.75")
        ),
        reaction_time_s=float(
            os.environ.get("JOLGWA_D435_REACTION_TIME_S", "0.25")
        ),
        max_deceleration_mps2=float(
            os.environ.get("JOLGWA_D435_MAX_DECELERATION_MPS2", "1.0")
        ),
        distance_uncertainty_m=float(
            os.environ.get("JOLGWA_D435_DISTANCE_UNCERTAINTY_M", "0.3")
        ),
        max_dynamic_trigger_distance_m=float(
            os.environ.get("JOLGWA_D435_MAX_DYNAMIC_TRIGGER_M", "15.0")
        ),
        trigger_samples=int(
            os.environ.get("JOLGWA_D435I_TRIGGER_FRAMES", "3")
        ),
        release_samples=int(
            os.environ.get("JOLGWA_D435I_RELEASE_FRAMES", "3")
        ),
        min_evade_climb_m=float(
            os.environ.get("JOLGWA_D435I_MIN_EVADE_CLIMB_M", "0.0")
        ),
        post_clear_climb_m=float(
            os.environ.get("JOLGWA_D435I_POST_CLEAR_CLIMB_M", "0.0")
        ),
        require_geometry=os.environ.get("JOLGWA_D435_REQUIRE_GEOMETRY", "true").lower()
        in {"1", "true", "yes"},
        scenario_demo=os.environ.get("JOLGWA_SCENARIO_PROFILE", "") in {"DEMO_MINIMAL_V1", "FRONT_DETECT_2M_PASS_3M_V1"},
        front_climb_demo=os.environ.get("JOLGWA_SCENARIO_PROFILE", "") == "FRONT_DETECT_2M_PASS_3M_V1",
        roof_minimum_gap_m=1.0,
        stale_timeout_s=float(
            os.environ.get("JOLGWA_D435I_STALE_TIMEOUT_S", "0.5")
        ),
        vertical_speed_mps=float(
            os.environ.get("JOLGWA_D435I_VERTICAL_SPEED_MPS", "0.6")
        ),
        prefer_lateral_escape=False,
        outdoor_climb_default=True,
        allow_descent=False,
        min_depth_m=float(
            os.environ.get("JOLGWA_D435I_MIN_DEPTH_M", "0.15")
        ),
        max_depth_m=float(
            os.environ.get("JOLGWA_D435I_MAX_DEPTH_M", "20.0")
        ),
        min_valid_fraction=float(
            os.environ.get("JOLGWA_D435I_MIN_VALID_FRACTION", "0.25")
        ),
        # Percentile + occupancy reject isolated speckle while retaining more
        # narrow returns than the generic 10% summary. Emergency-distance
        # detections bypass temporal voting (at low speed, all <=3m detections).
        near_percentile=float(
            os.environ.get("JOLGWA_D435I_NEAR_PERCENTILE", "2.0")
        ),
        min_obstacle_fraction=float(
            os.environ.get("JOLGWA_D435I_MIN_OBSTACLE_FRACTION", "0.02")
        ),
        min_pixel_confidence=0.0,
        min_frame_confidence=0.0,
        max_pair_skew_s=float(
            os.environ.get("JOLGWA_D435_MAX_PAIR_SKEW_S", "0.075")
        ),
        max_pose_age_s=float(
            os.environ.get("JOLGWA_D435_MAX_POSE_AGE_S", "0.5")
        ),
        max_angular_rate_rad_s=float(
            os.environ.get("JOLGWA_D435_MAX_ANGULAR_RATE_RAD_S", "1.5")
        ),
    )

    if observation_only:
        native_depth_config = replace(native_depth_config, require_geometry=True)
    app = FastAPI(title="Jolgwa Jetson Compute", version="1.0")
    # Identity lasts exactly one backend runtime, not one request or client.
    # This is provenance, not an authentication or flight-authorization token.
    native_backend_epoch = str(uuid.uuid4())
    app.state.observer_session = None

    @app.middleware("http")
    async def isolate_observer(request: Request, call_next):
        # Prevent even stateful policy updates across flight/observer sessions.
        # This is a routing/interlock mechanism, not authentication.
        session = request.headers.get("X-Jolgwa-Observer-Session", "")
        if not observation_only and session:
            return JSONResponse({"detail": "observer_requires_dedicated_backend"}, status_code=409)
        if observation_only and request.url.path != "/health":
            if request.url.path != "/v1/d435i-perception-decision" or request.method != "POST":
                return JSONResponse({"detail": "observer_endpoint_only"}, status_code=403)
            try:
                import uuid
                session = str(uuid.UUID(session))
            except ValueError:
                return JSONResponse({"detail": "observer_session_required"}, status_code=403)
            if request.query_params.get("run_phase1") != "false":
                return JSONResponse({"detail": "observer_disallows_phase1"}, status_code=403)
            if request.query_params.get("depth_encoding") != "16UC1":
                return JSONResponse({"detail": "observer_requires_raw_d435_16UC1"}, status_code=422)
            if app.state.observer_session not in (None, session):
                return JSONResponse({"detail": "observer_session_conflict_restart_backend"}, status_code=409)
            app.state.observer_session = session
        return await call_next(request)
    app.state.runtime = runtime
    app.state.phase1_runtime = phase1_runtime
    app.state.phase1_flight_session = Phase1FlightSession()
    app.state.phase1_flight_lock = threading.Lock()
    app.state.phase1_clock = time.monotonic
    if os.environ.get("JOLGWA_SCENARIO_PROFILE") == "FRONT_DETECT_2M_PASS_3M_V1":
        from .front_climb_profile import configure
        native_depth_config = configure(native_depth_config)
    app.state.native_depth_runtime = JetsonNativeDepthRuntime(
        VerticalObstacleAvoidanceCore(native_depth_config),
        geometry_max_age_s=geometry_max_age_s,
    )

    @app.on_event("startup")
    def eager_model_warmup() -> None:
        if observation_only:
            return
        if os.environ.get("JOLGWA_DA3_EAGER_LOAD", "true").lower() not in {
            "1",
            "true",
            "yes",
        }:
            return
        runtime.estimator.load()
        # Compile/allocate the steady-state single-view path before accepting
        # safety requests so the first flight decision has normal latency.
        warmup = np.zeros((360, 640, 3), dtype=np.uint8)
        runtime.estimator.estimate(warmup, fx_px=320.0)
        if phase1_runtime is not None:
            phase1_runtime.warmup()

    @app.get("/health")
    def health() -> dict[str, Any]:
        from .depth_geometry import DepthGeometryConfig

        tracker = app.state.native_depth_runtime.geometry_tracker
        geometry_config = tracker.config if tracker is not None else DepthGeometryConfig(
            pose_timeout_s=geometry_max_age_s)
        result = {
            "status": "ok",
            "native_demo_contract": make_native_demo_contract(
                native_depth_config, native_backend_epoch),
            "observation_only": observation_only,
            "phase1_profile": "observation_only" if observation_only else phase1_profile,
            # These are routing permissions, not model/temporal readiness.
            "phase1_access": {
                "dedicated_flight_enabled": (
                    not observation_only and phase1_runtime is not None
                    and phase1_profile == "flight_exclusive"
                ),
                "combined_phase1_enabled": (
                    not observation_only and phase1_runtime is not None
                    and phase1_profile == "legacy_combined"
                ),
                "native_depth_only_enabled": True,
                "profile_change_requires_backend_restart": True,
            },
            "device": runtime.estimator.device,
            "model_id": runtime.estimator.model_id,
            "model_loaded": runtime.estimator.loaded,
            "process_res": runtime.estimator.process_res,
            "event_policy": "all_phase1_confirmed_events",
            "native_depth": {
                "enabled": True,
                "corridor_backend": native_depth_status(),
                "sensor": "Intel RealSense D435 (d435i-compatible API)",
                "encodings": sorted(NATIVE_DEPTH_ENCODINGS),
                "trigger_distance_m": native_depth_config.trigger_distance_m,
                "release_distance_m": native_depth_config.release_distance_m,
                "minimum_standoff_m": native_depth_config.minimum_standoff_m,
                "emergency_margin_m": native_depth_config.emergency_margin_m,
                "reaction_time_s": native_depth_config.reaction_time_s,
                "max_deceleration_mps2": (
                    native_depth_config.max_deceleration_mps2
                ),
                "distance_uncertainty_m": (
                    native_depth_config.distance_uncertainty_m
                ),
                "max_dynamic_trigger_distance_m": (
                    native_depth_config.max_dynamic_trigger_distance_m
                ),
                "trigger_frames": native_depth_config.trigger_samples,
                "release_frames": native_depth_config.release_samples,
                "min_evade_climb_m": native_depth_config.min_evade_climb_m,
                "post_clear_climb_m": native_depth_config.post_clear_climb_m,
                "require_geometry": native_depth_config.require_geometry,
                "roof_minimum_gap_m": native_depth_config.roof_minimum_gap_m,
                "geometry_bounds": {
                    "angular_uncertainty_deg": math.degrees(geometry_config.angular_uncertainty_rad),
                    "position_uncertainty_m": geometry_config.position_uncertainty_m,
                    "depth_uncertainty_m": geometry_config.depth_uncertainty_m,
                    "body_half_extents_frd_m": geometry_config.body_half_extents_frd_m,
                    "online_error_bound_verified": False,
                    "scope": "Configured assumptions; not an online calibration or flight safety certificate",
                },
                "last_diagnostic": app.state.native_depth_runtime.last_diagnostic,
                "near_percentile": native_depth_config.near_percentile,
                "min_obstacle_fraction": (
                    native_depth_config.min_obstacle_fraction
                ),
                "min_valid_fraction": native_depth_config.min_valid_fraction,
                "max_pair_skew_s": native_depth_config.max_pair_skew_s,
                "max_pose_age_s": native_depth_config.max_pose_age_s,
                "geometry_max_age_s": geometry_max_age_s,
                "max_angular_rate_rad_s": (
                    native_depth_config.max_angular_rate_rad_s
                ),
                "known_sensor_limits": [
                    "reflective_or_transparent_surfaces",
                    "thin_or_low_texture_obstacles",
                    "forward_camera_blind_sides",
                ],
            },
            "phase1_loaded": bool(
                phase1_runtime is not None
                and getattr(phase1_runtime, "loaded", False)
            ),
        }
        if phase1_runtime is not None:
            result["phase1"] = phase1_runtime.health()
        return result

    @app.post("/v1/depth-decision")
    async def depth_decision(
        request: Request,
        focal_px: float = Query(gt=0.0),
        current_z_ned_m: float = Query(),
        frame_timestamp_ns: int = Query(ge=0),
        gimbal_forward: bool = Query(True),
        sent_monotonic_s: float | None = Query(None),
    ) -> dict[str, Any]:
        del frame_timestamp_ns, sent_monotonic_s
        payload = await request.body()
        if not payload:
            raise HTTPException(status_code=400, detail="JPEG body is empty")
        try:
            decision, inference_ms = runtime.evaluate_jpeg(
                payload,
                focal_px=focal_px,
                current_z_ned_m=current_z_ned_m,
                gimbal_forward=gimbal_forward,
            )
        except (ValueError, DepthAnythingV3Error) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return decision_payload(decision, inference_ms=inference_ms)

    @app.post("/v1/perception-decision")
    async def perception_decision(
        request: Request,
        focal_px: float = Query(gt=0.0),
        current_z_ned_m: float = Query(),
        frame_timestamp_ns: int = Query(ge=0),
        gimbal_forward: bool = Query(True),
        sent_monotonic_s: float | None = Query(None),
    ) -> dict[str, Any]:
        if phase1_profile != "legacy_combined":
            raise HTTPException(409, "phase1_legacy_disabled_flight_exclusive")
        del frame_timestamp_ns, sent_monotonic_s
        if phase1_runtime is None:
            raise HTTPException(
                status_code=503, detail="Phase 1 inference is disabled"
            )
        payload = await request.body()
        if not payload:
            raise HTTPException(status_code=400, detail="JPEG body is empty")
        try:
            image_bgr = _decode_jpeg(payload)
            decision, depth_ms = runtime.evaluate_bgr(
                image_bgr,
                focal_px=focal_px,
                current_z_ned_m=current_z_ned_m,
                gimbal_forward=gimbal_forward,
            )
            phase1 = phase1_runtime.evaluate_bgr(image_bgr)
        except (ValueError, DepthAnythingV3Error, Phase1InferenceError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        events = [
            event
            for event in phase1.events
            if event.get("state") == "CONFIRMED"
            and event.get("event_type") in PHASE1_EVENT_TYPES
        ]
        return {
            "safety": decision_payload(decision, inference_ms=depth_ms),
            "events": events,
            "phase1": {
                "inference_ms": phase1.inference_ms,
                "detections": list(phase1.detections),
                "modules": phase1.modules,
            },
        }

    @app.post("/v1/d435i-perception-decision")
    async def d435i_perception_decision(
        request: Request,
        jpeg_size: int = Query(gt=0),
        depth_width: int = Query(gt=0, le=4096),
        depth_height: int = Query(gt=0, le=4096),
        depth_step: int = Query(gt=0, le=65536),
        depth_encoding: str = Query(),
        depth_bigendian: bool = Query(False),
        current_x_ned_m: float = Query(0.0),
        current_y_ned_m: float = Query(0.0),
        current_z_ned_m: float = Query(),
        current_heading_rad: float = Query(0.0),
        current_vx_ned_mps: float = Query(0.0),
        current_vy_ned_mps: float = Query(0.0),
        current_vz_ned_mps: float = Query(0.0),
        current_angular_rate_rad_s: float = Query(0.0, ge=0.0),
        frame_timestamp_ns: int = Query(gt=0),
        depth_timestamp_ns: int = Query(gt=0),
        pair_skew_s: float = Query(ge=0.0, le=0.075),
        frame_age_s: float = Query(ge=0.0),
        rgb_age_s: float = Query(0.0, ge=0.0),
        depth_age_s: float = Query(0.0, ge=0.0),
        pose_age_s: float = Query(0.0, ge=0.0),
        gimbal_forward: bool = Query(True),
        compressed: bool = Query(True),
        run_phase1: bool = Query(True),
        geometry_calibrated: bool = Query(False),
        depth_fx: float | None = Query(None),
        depth_fy: float | None = Query(None),
        depth_cx: float | None = Query(None),
        depth_cy: float | None = Query(None),
        depth_info_width: int | None = Query(None),
        depth_info_height: int | None = Query(None),
        attitude_qw: float | None = Query(None),
        attitude_qx: float | None = Query(None),
        attitude_qy: float | None = Query(None),
        attitude_qz: float | None = Query(None),
        attitude_age_s: float | None = Query(None),
        camera_info_age_s: float | None = Query(None),
        camera_info_timestamp_ns: int | None = Query(None),
        attitude_timestamp_us: int | None = Query(None),
        position_timestamp_us: int | None = Query(None),
        camera_mount_x_m: float | None = Query(None),
        camera_mount_y_m: float | None = Query(None),
        camera_mount_z_m: float | None = Query(None),
        camera_mount_roll_rad: float | None = Query(None),
        camera_mount_pitch_rad: float | None = Query(None),
        camera_mount_yaw_rad: float | None = Query(None),
    ) -> dict[str, Any]:
        # Enforce before body/decompression/depth-policy/model work, including
        # before the first flight request has bound its temporal source session.
        if run_phase1 and phase1_profile != "legacy_combined":
            raise HTTPException(409, "phase1_legacy_disabled_flight_exclusive")
        received_at = time.monotonic()
        del (
            frame_timestamp_ns,
            depth_timestamp_ns,
            current_vz_ned_mps,
        )
        geometry_metadata = {
            "calibrated": geometry_calibrated,
            "width": depth_info_width, "height": depth_info_height,
            "fx": depth_fx, "fy": depth_fy, "cx": depth_cx, "cy": depth_cy,
            "quaternion": (attitude_qw, attitude_qx, attitude_qy, attitude_qz),
            "attitude_age_s": attitude_age_s,
            "camera_info_age_s": camera_info_age_s,
            "position_ned_m": (current_x_ned_m, current_y_ned_m, current_z_ned_m),
            "heading_rad": current_heading_rad,
            "mount_translation": (camera_mount_x_m, camera_mount_y_m, camera_mount_z_m),
            "mount_rpy": (camera_mount_roll_rad, camera_mount_pitch_rad, camera_mount_yaw_rad),
        }
        forward_speed_mps = max(
            0.0,
            current_vx_ned_mps * math.cos(current_heading_rad)
            + current_vy_ned_mps * math.sin(current_heading_rad),
        )
        pipeline_latency_s = max(frame_age_s, rgb_age_s, depth_age_s)
        payload = await request.body()
        if not payload:
            raise HTTPException(status_code=400, detail="D435i body is empty")

        def evaluate_request():
            image_bgr, depth_m = _decode_d435i_request(
                payload,
                jpeg_size=jpeg_size,
                depth_width=depth_width,
                depth_height=depth_height,
                depth_step=depth_step,
                depth_encoding=depth_encoding,
                depth_bigendian=depth_bigendian,
                compressed=compressed,
                decode_jpeg=run_phase1,
            )
            decision, depth_ms = app.state.native_depth_runtime.evaluate(
                depth_m,
                current_z_ned_m=current_z_ned_m,
                frame_age_s=pipeline_latency_s + (time.monotonic() - received_at),
                gimbal_forward=gimbal_forward,
                forward_speed_mps=forward_speed_mps,
                pipeline_latency_s=pipeline_latency_s + (time.monotonic() - received_at),
                pose_age_s=pose_age_s + (time.monotonic() - received_at),
                pair_skew_s=pair_skew_s,
                angular_rate_rad_s=current_angular_rate_rad_s,
                geometry_metadata={
                    **geometry_metadata,
                    "attitude_age_s": (None if attitude_age_s is None else
                                       attitude_age_s + time.monotonic() - received_at),
                },
            )
            phase1 = None
            if run_phase1 and phase1_runtime is not None:
                assert image_bgr is not None
                phase1 = phase1_runtime.evaluate_bgr(image_bgr)
            return decision, depth_ms, phase1

        try:
            decision, depth_ms, phase1 = await run_in_threadpool(
                evaluate_request
            )
        except (ValueError, DepthAnythingV3Error, Phase1InferenceError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        service_elapsed_s = time.monotonic() - received_at
        observation_age_s = pipeline_latency_s + service_elapsed_s
        geometry_pose_deadline = min(
            native_depth_config.max_pose_age_s,
            geometry_max_age_s if native_depth_config.require_geometry else native_depth_config.max_pose_age_s,
        )
        # The combined compatibility endpoint may wait for slow Phase1 after
        # safety finished. Never return that earlier CLEAR as current safety.
        if (
            observation_age_s > native_depth_config.stale_timeout_s
            or pose_age_s + service_elapsed_s > geometry_pose_deadline
            or (attitude_age_s is not None and attitude_age_s + service_elapsed_s
                > geometry_pose_deadline)
        ):
            decision = replace(
                decision,
                state=VerticalAvoidanceState.STALE,
                direction=VerticalDirection.STOP,
                velocity_ned_mps=(0.0, 0.0, 0.0),
                reason="native_response_deadline_exceeded",
                descent_corridor_clear=False,
                geometry={**(decision.geometry or {}), "geometry_valid": False,
                          "roof_clearance_verified": False,
                          "roof_passage_verified": False},
            )
        events = [] if phase1 is None else [
            event
            for event in phase1.events
            if event.get("state") == "CONFIRMED"
            and event.get("event_type") in PHASE1_EVENT_TYPES
        ]
        return {
            "observation_only": observation_only,
            "safety": {
                **decision_payload(decision, inference_ms=depth_ms),
                "native_demo_contract": make_native_demo_contract(
                    native_depth_config, native_backend_epoch, decision),
                "sensor_source": "realsense-d435i-native-depth",
                "service_elapsed_ms": service_elapsed_s * 1000.0,
                "observation_age_s": observation_age_s,
            },
            "events": events,
            "phase1": {
                "enabled": phase1 is not None,
                "inference_ms": 0.0 if phase1 is None else phase1.inference_ms,
                "detections": [] if phase1 is None else list(phase1.detections),
                "modules": {} if phase1 is None else phase1.modules,
            },
        }

    @app.post("/v1/phase1-decision")
    async def phase1_decision(
        request: Request,
        frame_timestamp_ns: int = Query(gt=0),
        frame_age_s: float = Query(ge=0.0),
        session_id: str = Query(min_length=36, max_length=36),
        request_sequence: int = Query(gt=0),
    ) -> dict[str, Any]:
        # A legacy app may already contain unleased frames. Never let it later
        # masquerade as an exclusive flight source by binding a fresh session.
        if phase1_profile != "flight_exclusive":
            raise HTTPException(409, "phase1_flight_disabled_legacy_combined")
        clock = app.state.phase1_clock
        received_at = clock()
        if phase1_runtime is None:
            raise HTTPException(
                status_code=503, detail="Phase 1 inference is disabled"
            )
        try:
            validate_request_identity(session_id, request_sequence, frame_timestamp_ns)
            validate_input_age(frame_age_s)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        # Do not queue temporal inference behind another client or old frame.
        # This lock is independent of the native depth worker and its deadline.
        lock = app.state.phase1_flight_lock
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "phase1_inference_busy")
        try:
            try:
                app.state.phase1_flight_session.consume(session_id, request_sequence, frame_timestamp_ns)
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            payload = bytearray()
            async for chunk in request.stream():
                payload.extend(chunk)
                if len(payload) > MAX_JPEG_BYTES:
                    raise HTTPException(413, "phase1_jpeg_too_large")
            if not payload:
                raise HTTPException(400, "JPEG body is empty")

            def evaluate_request():
                validate_input_age(frame_age_s+clock()-received_at)
                image = _decode_jpeg(bytes(payload))
                if image.shape[0]*image.shape[1] > 4096*2160:
                    raise ValueError("phase1_jpeg_invalid_dimensions")
                validate_input_age(frame_age_s+clock()-received_at)
                return phase1_runtime.evaluate_bgr(image)

            result, failure = None, ""
            try:
                result = await run_in_threadpool(evaluate_request)
            except Exception as exc:
                failure = str(exc) if isinstance(exc, ValueError) else "phase1_inference_failed"
            return phase1_server_response(
                result, session_id=session_id, request_sequence=request_sequence,
                frame_timestamp_ns=frame_timestamp_ns, input_frame_age_s=frame_age_s,
                elapsed_s=clock()-received_at, failure_reason=failure)
        finally:
            lock.release()

    @app.post("/v1/event-decision")
    async def event_decision(request: Request) -> dict[str, Any]:
        try:
            event = await request.json()
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail="invalid JSON"
            ) from exc
        if not isinstance(event, dict):
            raise HTTPException(
                status_code=400, detail="event must be an object"
            )
        event_type = str(event.get("event_type", ""))
        confirmed = event.get("state") == "CONFIRMED"
        respond = confirmed and event_type in PHASE1_EVENT_TYPES
        return {
            "respond": respond,
            "reason": (
                "confirmed_phase1_event"
                if respond
                else "not_an_approved_confirmed_event"
            ),
        }

    return app


app = create_app()


def create_observer_app() -> FastAPI:
    """Separate native-D435 policy memory; no DA3/event model warmup."""
    return create_app(observation_only=True)


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Jetson native perception backend")
    parser.add_argument("--mode", choices=("observe", "control"), default="observe")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    uvicorn.run(
        "jolgwa_uav.jetson_compute_service:" + ("create_observer_app" if args.mode == "observe" else "create_app"),
        factory=True,
        host=args.host or os.environ.get("JOLGWA_COMPUTE_HOST", "127.0.0.1" if args.mode == "observe" else "0.0.0.0"),
        port=args.port or int(os.environ.get("JOLGWA_COMPUTE_PORT", "8766" if args.mode == "observe" else "8765")),
        workers=1,
    )


if __name__ == "__main__":
    main()

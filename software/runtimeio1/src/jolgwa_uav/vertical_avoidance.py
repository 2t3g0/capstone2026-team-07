from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import Enum
import math
from typing import Any

import numpy as np

from .native_depth import native_corridor_stats


class VerticalAvoidanceState(str, Enum):
    CLEAR = "CLEAR"
    HOLD = "HOLD"
    EVADE = "EVADE"
    STALE = "STALE"


class VerticalDirection(str, Enum):
    FORWARD = "FORWARD"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"
    STOP = "STOP"


@dataclass(frozen=True, slots=True)
class DepthRoiStats:
    """Robust summary of one depth-map corridor."""

    valid_fraction: float
    near_distance_m: float
    median_distance_m: float
    obstacle_fraction: float
    valid_samples: int
    total_samples: int


@dataclass(frozen=True, slots=True)
class DepthCorridorStats:
    upper: DepthRoiStats
    center: DepthRoiStats
    lower: DepthRoiStats
    left: DepthRoiStats | None = None
    right: DepthRoiStats | None = None


@dataclass(frozen=True, slots=True)
class VerticalAvoidanceDecision:
    state: VerticalAvoidanceState
    direction: VerticalDirection
    velocity_ned_mps: tuple[float, float, float]
    reason: str
    corridors: DepthCorridorStats | None = None
    effective_trigger_distance_m: float | None = None
    effective_release_distance_m: float | None = None
    descent_corridor_clear: bool = False
    geometry: dict[str, Any] | None = None

    @property
    def must_hold(self) -> bool:
        return self.state in {
            VerticalAvoidanceState.HOLD,
            VerticalAvoidanceState.STALE,
        }


@dataclass(frozen=True, slots=True)
class VerticalAvoidanceConfig:
    trigger_distance_m: float = 3.0
    release_distance_m: float = 4.0
    minimum_standoff_m: float = 2.5
    emergency_margin_m: float = 0.0
    reaction_time_s: float = 0.25
    max_deceleration_mps2: float = 1.0
    distance_uncertainty_m: float = 0.3
    max_dynamic_trigger_distance_m: float = 15.0
    trigger_samples: int = 3
    release_samples: int = 1
    min_evade_climb_m: float = 0.0
    post_clear_climb_m: float = 0.0
    require_geometry: bool = False
    scenario_demo: bool = False  # Dedicated profile; geometry stays truthful.
    front_climb_demo: bool = False  # Only FRONT_DETECT_2M_PASS_3M_V1 ignores upper/lower validity.
    roof_minimum_gap_m: float = 1.0
    stale_timeout_s: float = 0.5
    vertical_speed_mps: float = 0.6
    lateral_speed_mps: float = 0.8
    prefer_lateral_escape: bool = False
    outdoor_climb_default: bool = True
    allow_descent: bool = False
    min_altitude_m: float = 1.5
    max_altitude_m: float = 15.0
    altitude_margin_m: float = 0.5
    min_depth_m: float = 0.1
    max_depth_m: float = 80.0
    min_valid_fraction: float = 0.55
    min_obstacle_fraction: float = 0.05
    min_pixel_confidence: float = 0.25
    min_frame_confidence: float = 0.5
    near_percentile: float = 10.0
    max_pair_skew_s: float = 0.075
    max_pose_age_s: float = 0.5
    max_angular_rate_rad_s: float = 1.5
    horizontal_roi_fraction: float = 0.50
    upper_roi: tuple[float, float] = (0.05, 0.32)
    center_roi: tuple[float, float] = (0.36, 0.64)
    lower_roi: tuple[float, float] = (0.68, 0.95)
    side_vertical_roi: tuple[float, float] = (0.25, 0.75)
    left_roi: tuple[float, float] = (0.02, 0.22)
    right_roi: tuple[float, float] = (0.78, 0.98)

    def __post_init__(self) -> None:
        if self.front_climb_demo and not self.scenario_demo:
            raise ValueError("front climb relaxation requires the dedicated scenario profile")
        finite_values = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if isinstance(getattr(self, field.name), (int, float))
        }
        if not all(
            math.isfinite(float(value)) for value in finite_values.values()
        ):
            raise ValueError("all numeric configuration values must be finite")
        if not 0.0 < self.trigger_distance_m < self.release_distance_m:
            raise ValueError(
                "distance thresholds must satisfy 0 < trigger < release"
            )
        if self.minimum_standoff_m < 0.0:
            raise ValueError("minimum_standoff_m cannot be negative")
        if self.emergency_margin_m < 0.0:
            raise ValueError("emergency_margin_m cannot be negative")
        if self.reaction_time_s < 0.0:
            raise ValueError("reaction_time_s cannot be negative")
        if self.max_deceleration_mps2 <= 0.0:
            raise ValueError("max_deceleration_mps2 must be positive")
        if self.distance_uncertainty_m < 0.0:
            raise ValueError("distance_uncertainty_m cannot be negative")
        if self.max_dynamic_trigger_distance_m < self.trigger_distance_m:
            raise ValueError(
                "max_dynamic_trigger_distance_m must cover trigger_distance_m"
            )
        if self.trigger_samples < 1:
            raise ValueError("trigger_samples must be at least one")
        if self.release_samples < 1:
            raise ValueError("release_samples must be at least one")
        if (
            self.stale_timeout_s <= 0.0
            or self.vertical_speed_mps <= 0.0
            or self.lateral_speed_mps <= 0.0
        ):
            raise ValueError("timing and speed values must be positive")
        if not 0.0 <= self.min_altitude_m < self.max_altitude_m:
            raise ValueError("altitude limits must satisfy 0 <= min < max")
        if self.altitude_margin_m < 0.0:
            raise ValueError("altitude_margin_m cannot be negative")
        if self.min_evade_climb_m < 0.0:
            raise ValueError("min_evade_climb_m cannot be negative")
        if self.post_clear_climb_m < 0.0:
            raise ValueError("post_clear_climb_m cannot be negative")
        if self.roof_minimum_gap_m <= 0.0:
            raise ValueError("roof_minimum_gap_m must be positive")
        if self.min_altitude_m + self.altitude_margin_m >= self.max_altitude_m:
            raise ValueError("altitude margin leaves no usable altitude band")
        if not 0.0 < self.min_depth_m < self.max_depth_m:
            raise ValueError("depth limits must satisfy 0 < min < max")
        for name in (
            "min_valid_fraction",
            "min_obstacle_fraction",
            "min_pixel_confidence",
            "min_frame_confidence",
            "horizontal_roi_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if (
            self.min_valid_fraction == 0.0
            or self.horizontal_roi_fraction == 0.0
        ):
            raise ValueError(
                "valid and ROI fractions must be greater than zero"
            )
        if not 0.0 <= self.near_percentile <= 50.0:
            raise ValueError("near_percentile must be in [0, 50]")
        if self.max_pair_skew_s <= 0.0 or self.max_pose_age_s <= 0.0:
            raise ValueError("sensor timing limits must be positive")
        if self.max_angular_rate_rad_s <= 0.0:
            raise ValueError("max_angular_rate_rad_s must be positive")
        for name in (
            "upper_roi",
            "center_roi",
            "lower_roi",
            "side_vertical_roi",
            "left_roi",
            "right_roi",
        ):
            start, end = getattr(self, name)
            if not 0.0 <= start < end <= 1.0:
                raise ValueError(
                    f"{name} must be an ordered normalized interval"
                )


class VerticalObstacleAvoidanceCore:
    """Convert a metric depth map into a bounded spatial escape command.

    The central corridor must report an obstacle for ``trigger_samples``
    consecutive valid frames before an EVADE command is emitted. A possible
    obstacle is held immediately while temporal agreement is collected. Once
    active, the core remains active through the 3--4 m hysteresis band and
    releases at 4 m.

    ``current_altitude_m`` is positive height above the configured local floor.
    NED vertical velocity is negative for UP and positive for DOWN. LEFT and
    RIGHT use camera/body FRD Y; a variable-yaw PX4 bridge must rotate that
    component into world NED.
    """

    def __init__(self, config: VerticalAvoidanceConfig | None = None) -> None:
        self.config = config or VerticalAvoidanceConfig()
        self._trigger_count = 0
        self._active = False
        self._evade_start_altitude_m: float | None = None
        self._clear_candidate_altitude_m: float | None = None
        self._release_count = 0
        self._roof_guard_active = False
        self._roof_gap_observed = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def trigger_count(self) -> int:
        return self._trigger_count

    def reset(self) -> None:
        self._trigger_count = 0
        self._active = False
        self._evade_start_altitude_m = None
        self._clear_candidate_altitude_m = None
        self._release_count = 0
        self._roof_guard_active = False
        self._roof_gap_observed = False

    def evaluate(
        self,
        depth_m: Any,
        *,
        frame_age_s: float,
        current_altitude_m: float,
        confidence: Any | None = None,
        forward_speed_mps: float = 0.0,
        pipeline_latency_s: float = 0.0,
        pose_age_s: float = 0.0,
        pair_skew_s: float = 0.0,
        angular_rate_rad_s: float = 0.0,
        geometry: dict[str, Any] | None = None,
    ) -> VerticalAvoidanceDecision:
        invalid_reason = self._validate_metadata(
            frame_age_s=frame_age_s,
            current_altitude_m=current_altitude_m,
            forward_speed_mps=forward_speed_mps,
            pipeline_latency_s=pipeline_latency_s,
            pose_age_s=pose_age_s,
            pair_skew_s=pair_skew_s,
            angular_rate_rad_s=angular_rate_rad_s,
        )
        if invalid_reason is not None:
            return self._stale(invalid_reason)
        if self.config.require_geometry and (
            geometry is None or geometry.get("geometry_valid") is not True
        ):
            return self._stale("depth_geometry_unavailable")

        try:
            depth = np.asarray(depth_m, dtype=np.float32)
        except (TypeError, ValueError):
            return self._stale("depth_not_numeric")
        if depth.ndim != 2 or depth.size == 0:
            return self._stale("depth_shape_invalid")

        valid = (
            np.isfinite(depth)
            & (depth >= self.config.min_depth_m)
            & (depth <= self.config.max_depth_m)
        )
        # Native metric depth has no learned confidence. None has always meant
        # confidence 1; avoid allocating/filling a full float image in this path.
        if confidence is not None:
            confidence_map, confidence_reason = self._confidence_map(
                confidence, depth.shape
            )
            if confidence_reason is not None:
                return self._stale(confidence_reason)
            valid &= confidence_map >= self.config.min_pixel_confidence
        effective_trigger = self.effective_trigger_distance_m(
            forward_speed_mps=forward_speed_mps,
            pipeline_latency_s=max(frame_age_s, pipeline_latency_s),
        )
        effective_release = effective_trigger + (
            self.config.release_distance_m - self.config.trigger_distance_m
        )
        corridors = self._corridor_stats(depth, valid, effective_trigger)
        deficient = [
            name
            for name in (("center",) if self.config.front_climb_demo else ("upper", "center", "lower"))
            if getattr(corridors, name).valid_fraction
            < self.config.min_valid_fraction
        ]
        if deficient:
            return self._stale(
                "depth_roi_invalid:" + ",".join(deficient), corridors
            )

        center = corridors.center
        obstacle = (
            center.near_distance_m <= effective_trigger
            and center.obstacle_fraction >= self.config.min_obstacle_fraction
        )
        released = (
            center.near_distance_m >= effective_release
            and center.obstacle_fraction < self.config.min_obstacle_fraction
        )

        # A forward ROI opening is not evidence of clearance above a roof.
        # Keep the guard throughout the elevated pass, including after the
        # first CLEAR. Only geometrically verified passage releases it.
        roof_release_blocked = False
        if self.config.require_geometry and not self.config.scenario_demo and geometry is not None:
            if geometry.get("roof_passage_verified") is True:
                self._roof_guard_active = False
                self._roof_gap_observed = False
            roof_gap = geometry.get("roof_vertical_gap_m")
            gap_observed = (
                isinstance(roof_gap, (float, int))
                and not isinstance(roof_gap, bool)
                and math.isfinite(roof_gap)
            )
            roof_clear = (
                geometry.get("roof_clearance_verified") is True
                and gap_observed
                and roof_gap > self.config.roof_minimum_gap_m
            )
            if self._roof_guard_active:
                if gap_observed:
                    self._roof_gap_observed = True
                elif self._roof_gap_observed:
                    # Evidence loss is not a measured need to climb. In
                    # particular, an expired roof map after an elevated pass
                    # must not start another climb over empty foreground.
                    # Retain the session and invalidate release counters;
                    # repeated unknown frames cannot re-arm initial roof
                    # discovery. Fresh finite gap evidence resumes normal
                    # gap policy, and only verified passage releases the guard.
                    self._active = True
                    self._release_count = 0
                    self._clear_candidate_altitude_m = None
                    return self._hold(
                        "roof_clearance_evidence_lost", corridors,
                        effective_trigger_distance_m=effective_trigger,
                        effective_release_distance_m=effective_release,
                    )
            if self._roof_guard_active and not roof_clear:
                roof_release_blocked = True
                released = False
                self._active = True

        if self._active:
            if released:
                self._release_count += 1
                if self._clear_candidate_altitude_m is None:
                    self._clear_candidate_altitude_m = current_altitude_m
            else:
                # Release evidence must be consecutive. A returning obstacle
                # or hysteresis-band sample invalidates the earlier candidate.
                self._release_count = 0
                self._clear_candidate_altitude_m = None
            climbed_enough = (
                self._evade_start_altitude_m is None
                or current_altitude_m
                >= self._evade_start_altitude_m
                + self.config.min_evade_climb_m
            )
            cleared_with_margin = (
                self._clear_candidate_altitude_m is not None
                and current_altitude_m
                >= self._clear_candidate_altitude_m
                + self.config.post_clear_climb_m
            )
            release_confirmed = (
                released
                and self._release_count >= self.config.release_samples
            )
            if release_confirmed and climbed_enough and cleared_with_margin:
                self._active = False
                if self.config.scenario_demo:
                    # This profile finishes the episode on forward release;
                    # it never waits for the optional roof certificate.
                    self._roof_guard_active = False
                    self._roof_gap_observed = False
                self._trigger_count = 0
                self._evade_start_altitude_m = None
                self._clear_candidate_altitude_m = None
                self._release_count = 0
                return self._clear(
                    "front_corridor_released",
                    corridors,
                    effective_trigger_distance_m=effective_trigger,
                    effective_release_distance_m=effective_release,
                    descent_corridor_clear=(
                        corridors.lower.near_distance_m >= effective_release
                        and corridors.lower.obstacle_fraction
                        < self.config.min_obstacle_fraction
                    ),
                )
            escape = self._evade(
                corridors,
                current_altitude_m,
                effective_release_distance_m=effective_release,
                effective_trigger_distance_m=effective_trigger,
            )
            if roof_release_blocked:
                escape = replace(escape, reason="roof_clearance_pending:" + escape.reason)
            return escape

        if not obstacle:
            self._trigger_count = 0
            return self._clear(
                "front_corridor_clear",
                corridors,
                effective_trigger_distance_m=effective_trigger,
                effective_release_distance_m=effective_release,
                descent_corridor_clear=(
                    corridors.lower.near_distance_m >= effective_release
                    and corridors.lower.obstacle_fraction
                    < self.config.min_obstacle_fraction
                ),
            )

        self._trigger_count += 1
        emergency_distance = min(
            effective_trigger,
            self.config.minimum_standoff_m + self.config.emergency_margin_m,
        )
        emergency = (
            self.config.emergency_margin_m > 0.0
            and center.near_distance_m <= emergency_distance
        )
        if self._trigger_count < self.config.trigger_samples and not emergency:
            return self._hold(
                f"obstacle_confirmation_{self._trigger_count}_of_"
                f"{self.config.trigger_samples}",
                corridors,
                effective_trigger_distance_m=effective_trigger,
                effective_release_distance_m=effective_release,
            )

        # Once a robust ROI enters the emergency envelope, bypass temporal
        # confirmation.  The first obstacle frame still produces HOLD at the
        # normal trigger distance, while a sudden geometry discontinuity near
        # the required standoff starts the verified escape immediately.
        self._active = True
        self._roof_guard_active = self.config.require_geometry
        self._roof_gap_observed = False
        self._evade_start_altitude_m = current_altitude_m
        self._clear_candidate_altitude_m = None
        self._release_count = 0
        return self._evade(
            corridors,
            current_altitude_m,
            effective_release_distance_m=effective_release,
            effective_trigger_distance_m=effective_trigger,
        )

    def effective_trigger_distance_m(
        self, *, forward_speed_mps: float, pipeline_latency_s: float
    ) -> float:
        """Return a speed- and latency-aware stop trigger.

        ``minimum_standoff_m`` is the clearance remaining after braking, not
        the point at which braking starts.  The static trigger remains a floor
        for low-speed operation and the configured cap prevents a malformed
        velocity estimate from exceeding the sensor's useful range.
        """

        speed = max(0.0, float(forward_speed_mps))
        latency = max(0.0, float(pipeline_latency_s))
        stopping_distance = speed * speed / (
            2.0 * self.config.max_deceleration_mps2
        )
        dynamic = (
            self.config.minimum_standoff_m
            + speed * (self.config.reaction_time_s + latency)
            + stopping_distance
            + self.config.distance_uncertainty_m
        )
        return min(
            self.config.max_dynamic_trigger_distance_m,
            max(self.config.trigger_distance_m, dynamic),
        )

    def _validate_metadata(
        self,
        *,
        frame_age_s: float,
        current_altitude_m: float,
        forward_speed_mps: float,
        pipeline_latency_s: float,
        pose_age_s: float,
        pair_skew_s: float,
        angular_rate_rad_s: float,
    ) -> str | None:
        if not math.isfinite(frame_age_s) or frame_age_s < 0.0:
            return "frame_age_invalid"
        if frame_age_s > self.config.stale_timeout_s:
            return "frame_stale"
        if not math.isfinite(current_altitude_m) or current_altitude_m < 0.0:
            return "altitude_invalid"
        for value, invalid_reason in (
            (forward_speed_mps, "forward_speed_invalid"),
            (pipeline_latency_s, "pipeline_latency_invalid"),
            (pose_age_s, "pose_age_invalid"),
            (pair_skew_s, "pair_skew_invalid"),
            (angular_rate_rad_s, "angular_rate_invalid"),
        ):
            if not math.isfinite(value) or value < 0.0:
                return invalid_reason
        if pose_age_s > self.config.max_pose_age_s:
            return "pose_stale"
        if pair_skew_s > self.config.max_pair_skew_s:
            return "rgb_depth_skew"
        if angular_rate_rad_s > self.config.max_angular_rate_rad_s:
            return "angular_rate_too_high"
        return None

    def _confidence_map(
        self, confidence: Any | None, shape: tuple[int, int]
    ) -> tuple[np.ndarray, str | None]:
        if confidence is None:
            return np.ones(shape, dtype=np.float32), None
        try:
            values = np.asarray(confidence, dtype=np.float32)
        except (TypeError, ValueError):
            return np.empty(shape, dtype=np.float32), "confidence_not_numeric"
        if values.ndim == 0:
            scalar = float(values)
            if not math.isfinite(scalar) or not 0.0 <= scalar <= 1.0:
                return np.empty(shape, dtype=np.float32), "confidence_invalid"
            if scalar < self.config.min_frame_confidence:
                return np.empty(shape, dtype=np.float32), "confidence_too_low"
            return np.full(shape, scalar, dtype=np.float32), None
        if values.shape != shape:
            return (
                np.empty(shape, dtype=np.float32),
                "confidence_shape_invalid",
            )
        finite = np.isfinite(values)
        if not finite.any():
            return np.empty(shape, dtype=np.float32), "confidence_invalid"
        if np.any(values[finite] < 0.0) or np.any(values[finite] > 1.0):
            return np.empty(shape, dtype=np.float32), "confidence_invalid"
        mean_confidence = float(np.mean(values[finite]))
        if mean_confidence < self.config.min_frame_confidence:
            return np.empty(shape, dtype=np.float32), "confidence_too_low"
        return values, None

    def _corridor_stats(
        self,
        depth: np.ndarray,
        valid: np.ndarray,
        obstacle_distance_m: float,
    ) -> DepthCorridorStats:
        native = native_corridor_stats(depth, valid, self.config, obstacle_distance_m)
        if native is not None:
            return native
        return DepthCorridorStats(
            upper=self._roi_stats(
                depth, valid, self.config.upper_roi, obstacle_distance_m
            ),
            center=self._roi_stats(
                depth, valid, self.config.center_roi, obstacle_distance_m
            ),
            lower=self._roi_stats(
                depth, valid, self.config.lower_roi, obstacle_distance_m
            ),
            left=self._region_stats(
                depth,
                valid,
                self.config.side_vertical_roi,
                self.config.left_roi,
                obstacle_distance_m,
            ),
            right=self._region_stats(
                depth,
                valid,
                self.config.side_vertical_roi,
                self.config.right_roi,
                obstacle_distance_m,
            ),
        )

    def _roi_stats(
        self,
        depth: np.ndarray,
        valid: np.ndarray,
        vertical_interval: tuple[float, float],
        obstacle_distance_m: float,
    ) -> DepthRoiStats:
        height, width = depth.shape
        y0, y1 = _pixel_interval(height, vertical_interval)
        roi_width = max(
            1, int(round(width * self.config.horizontal_roi_fraction))
        )
        x0 = max(0, (width - roi_width) // 2)
        x1 = min(width, x0 + roi_width)
        depth_roi = depth[y0:y1, x0:x1]
        valid_roi = valid[y0:y1, x0:x1]
        return self._stats_from_samples(
            depth_roi, valid_roi, obstacle_distance_m
        )

    def _region_stats(
        self,
        depth: np.ndarray,
        valid: np.ndarray,
        vertical_interval: tuple[float, float],
        horizontal_interval: tuple[float, float],
        obstacle_distance_m: float,
    ) -> DepthRoiStats:
        height, width = depth.shape
        y0, y1 = _pixel_interval(height, vertical_interval)
        x0, x1 = _pixel_interval(width, horizontal_interval)
        return self._stats_from_samples(
            depth[y0:y1, x0:x1],
            valid[y0:y1, x0:x1],
            obstacle_distance_m,
        )

    def _stats_from_samples(
        self,
        depth_roi: np.ndarray,
        valid_roi: np.ndarray,
        obstacle_distance_m: float,
    ) -> DepthRoiStats:
        total = int(valid_roi.size)
        samples = depth_roi[valid_roi]
        count = int(samples.size)
        if count == 0:
            return DepthRoiStats(0.0, math.inf, math.inf, 0.0, 0, total)
        return DepthRoiStats(
            valid_fraction=count / total,
            near_distance_m=float(
                np.percentile(samples, self.config.near_percentile)
            ),
            median_distance_m=float(np.median(samples)),
            obstacle_fraction=float(
                np.count_nonzero(samples <= obstacle_distance_m)
                / count
            ),
            valid_samples=count,
            total_samples=total,
        )

    def _evade(
        self,
        corridors: DepthCorridorStats,
        altitude_m: float,
        *,
        effective_release_distance_m: float,
        effective_trigger_distance_m: float,
    ) -> VerticalAvoidanceDecision:
        upward_allowed = self.config.scenario_demo or altitude_m < (
            self.config.max_altitude_m - self.config.altitude_margin_m
        )
        downward_allowed = self.config.allow_descent and altitude_m > (
            self.config.min_altitude_m + self.config.altitude_margin_m
        )
        if self.config.front_climb_demo:
            # Direction is an observation request, not independent authority.
            # The approved Manager/Controller envelope owns the +2m limit.
            return VerticalAvoidanceDecision(
                state=VerticalAvoidanceState.EVADE, direction=VerticalDirection.UP,
                velocity_ned_mps=(0.0, 0.0, -self.config.vertical_speed_mps),
                reason="front_obstacle_bounded_climb", corridors=corridors,
                effective_trigger_distance_m=effective_trigger_distance_m,
                effective_release_distance_m=effective_release_distance_m)
        upper_clear = (
            corridors.upper.near_distance_m >= effective_release_distance_m
            and corridors.upper.obstacle_fraction
            < self.config.min_obstacle_fraction
        )
        lower_clear = (
            corridors.lower.near_distance_m >= effective_release_distance_m
            and corridors.lower.obstacle_fraction
            < self.config.min_obstacle_fraction
        )
        lateral = self._lateral_escape(
            corridors,
            effective_release_distance_m=effective_release_distance_m,
            effective_trigger_distance_m=effective_trigger_distance_m,
        )
        if self.config.outdoor_climb_default and upper_clear and upward_allowed:
            return VerticalAvoidanceDecision(
                state=VerticalAvoidanceState.EVADE,
                direction=VerticalDirection.UP,
                velocity_ned_mps=(0.0, 0.0, -self.config.vertical_speed_mps),
                reason="front_obstacle_outdoor_climb_default",
                corridors=corridors,
                effective_trigger_distance_m=effective_trigger_distance_m,
                effective_release_distance_m=effective_release_distance_m,
            )
        if self.config.prefer_lateral_escape and lateral is not None:
            return lateral

        candidates: list[tuple[float, VerticalDirection]] = []
        if upper_clear and upward_allowed:
            candidates.append(
                (corridors.upper.near_distance_m, VerticalDirection.UP)
            )
        if lower_clear and downward_allowed:
            candidates.append(
                (corridors.lower.near_distance_m, VerticalDirection.DOWN)
            )
        if not candidates:
            if lateral is not None:
                return lateral
            reasons: list[str] = []
            if not upper_clear:
                reasons.append("upper_corridor_blocked")
            elif not upward_allowed:
                reasons.append("altitude_ceiling")
            if not lower_clear:
                reasons.append("lower_corridor_blocked")
            elif not self.config.allow_descent:
                reasons.append("descent_disabled")
            elif not downward_allowed:
                reasons.append("altitude_floor")
            return self._hold(
                ";".join(reasons) or "no_vertical_escape",
                corridors,
                effective_trigger_distance_m=effective_trigger_distance_m,
                effective_release_distance_m=effective_release_distance_m,
            )

        # Prefer the more open corridor. A tie deliberately prefers climbing,
        # avoiding unnecessary motion toward the configured floor.
        _, direction = max(
            candidates,
            key=lambda item: (
                item[0],
                1 if item[1] is VerticalDirection.UP else 0,
            ),
        )
        ned_z = (
            -self.config.vertical_speed_mps
            if direction is VerticalDirection.UP
            else self.config.vertical_speed_mps
        )
        return VerticalAvoidanceDecision(
            state=VerticalAvoidanceState.EVADE,
            direction=direction,
            velocity_ned_mps=(0.0, 0.0, ned_z),
            reason="front_obstacle_vertical_escape",
            corridors=corridors,
            effective_trigger_distance_m=effective_trigger_distance_m,
            effective_release_distance_m=effective_release_distance_m,
        )

    def _lateral_escape(
        self,
        corridors: DepthCorridorStats,
        *,
        effective_release_distance_m: float,
        effective_trigger_distance_m: float,
    ) -> VerticalAvoidanceDecision | None:
        if corridors.left is None or corridors.right is None:
            return None
        candidates: list[tuple[float, VerticalDirection]] = []
        for stats, direction in (
            (corridors.left, VerticalDirection.LEFT),
            (corridors.right, VerticalDirection.RIGHT),
        ):
            clear = (
                stats.valid_fraction >= self.config.min_valid_fraction
                and stats.near_distance_m
                >= effective_release_distance_m
                and stats.obstacle_fraction
                < self.config.min_obstacle_fraction
            )
            if clear:
                candidates.append((stats.near_distance_m, direction))
        if not candidates:
            return None

        # Prefer the clearer side. A tie uses LEFT deterministically.
        _, direction = max(
            candidates,
            key=lambda item: (
                item[0],
                1 if item[1] is VerticalDirection.LEFT else 0,
            ),
        )
        ned_y = (
            -self.config.lateral_speed_mps
            if direction is VerticalDirection.LEFT
            else self.config.lateral_speed_mps
        )
        return VerticalAvoidanceDecision(
            state=VerticalAvoidanceState.EVADE,
            direction=direction,
            # This Y component is body-right for a forward camera. AirSim's
            # fixed yaw=0 path is identical to NED Y; PX4 bridges must rotate
            # it into world NED using vehicle yaw.
            velocity_ned_mps=(0.0, ned_y, 0.0),
            reason="front_obstacle_lateral_escape",
            corridors=corridors,
            effective_trigger_distance_m=effective_trigger_distance_m,
            effective_release_distance_m=effective_release_distance_m,
        )

    @staticmethod
    def _clear(
        reason: str,
        corridors: DepthCorridorStats,
        *,
        effective_trigger_distance_m: float | None = None,
        effective_release_distance_m: float | None = None,
        descent_corridor_clear: bool = False,
    ) -> VerticalAvoidanceDecision:
        return VerticalAvoidanceDecision(
            state=VerticalAvoidanceState.CLEAR,
            direction=VerticalDirection.FORWARD,
            velocity_ned_mps=(0.0, 0.0, 0.0),
            reason=reason,
            corridors=corridors,
            effective_trigger_distance_m=effective_trigger_distance_m,
            effective_release_distance_m=effective_release_distance_m,
            descent_corridor_clear=descent_corridor_clear,
        )

    @staticmethod
    def _hold(
        reason: str,
        corridors: DepthCorridorStats | None = None,
        *,
        effective_trigger_distance_m: float | None = None,
        effective_release_distance_m: float | None = None,
    ) -> VerticalAvoidanceDecision:
        return VerticalAvoidanceDecision(
            state=VerticalAvoidanceState.HOLD,
            direction=VerticalDirection.STOP,
            velocity_ned_mps=(0.0, 0.0, 0.0),
            reason=reason,
            corridors=corridors,
            effective_trigger_distance_m=effective_trigger_distance_m,
            effective_release_distance_m=effective_release_distance_m,
        )

    def _stale(
        self, reason: str, corridors: DepthCorridorStats | None = None
    ) -> VerticalAvoidanceDecision:
        # Invalid observations cannot contribute to temporal confirmation.
        # An already active avoidance is retained until a valid 4 m release.
        self._trigger_count = 0
        self._release_count = 0
        self._clear_candidate_altitude_m = None
        return VerticalAvoidanceDecision(
            state=VerticalAvoidanceState.STALE,
            direction=VerticalDirection.STOP,
            velocity_ned_mps=(0.0, 0.0, 0.0),
            reason=reason,
            corridors=corridors,
        )


def _pixel_interval(
    length: int, normalized: tuple[float, float]
) -> tuple[int, int]:
    start = min(length - 1, max(0, int(math.floor(length * normalized[0]))))
    end = min(length, max(start + 1, int(math.ceil(length * normalized[1]))))
    return start, end

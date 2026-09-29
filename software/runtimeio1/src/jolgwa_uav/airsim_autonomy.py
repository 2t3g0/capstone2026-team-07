from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Protocol

import cv2
import numpy as np

from .remote_compute import (
    JetsonComputeClient,
    RemoteComputeError,
    RemotePerceptionDecision,
    RemoteSafetyDecision,
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


class AirSimAutonomyError(RuntimeError):
    pass


class _AsyncTask(Protocol):
    def join(self) -> Any: ...


class _Compute(Protocol):
    def health(self) -> dict[str, Any]: ...

    def infer_rgb(
        self,
        jpeg: bytes,
        *,
        focal_px: float,
        current_z_ned_m: float,
        frame_timestamp_ns: int,
        gimbal_forward: bool = True,
    ) -> RemoteSafetyDecision: ...

    def approve_event(self, event: dict[str, Any]) -> bool: ...

    def infer_perception(
        self,
        jpeg: bytes,
        *,
        focal_px: float,
        current_z_ned_m: float,
        frame_timestamp_ns: int,
        gimbal_forward: bool = True,
    ) -> RemotePerceptionDecision: ...


@dataclass(frozen=True, slots=True)
class AirSimAutonomyConfig:
    host: str = "127.0.0.1"
    port: int = 41451
    vehicle_name: str = "PatrolDrone"
    camera_name: str = "front_center"
    camera_hfov_deg: float = 90.0
    takeoff_altitude_m: float = 4.0
    min_altitude_m: float = 2.0
    max_altitude_m: float = 12.0
    cruise_speed_mps: float = 1.0
    vertical_step_m: float = 2.0
    vertical_speed_mps: float = 0.8
    lateral_step_m: float = 2.0
    lateral_speed_mps: float = 0.8
    path_recovery_gain: float = 0.6
    path_recovery_max_speed_mps: float = 0.6
    path_recovery_tolerance_m: float = 0.25
    command_watchdog_s: float = 0.75
    capture_duration_s: float = 5.0
    jpeg_quality: int = 85
    evade_repeat_s: float = 2.0

    def __post_init__(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if not 1.0 <= self.camera_hfov_deg < 180.0:
            raise ValueError("camera_hfov_deg must be in [1, 180)")
        if not 0.0 < self.min_altitude_m < self.max_altitude_m:
            raise ValueError("altitude limits are invalid")
        if not (
            self.min_altitude_m
            <= self.takeoff_altitude_m
            <= self.max_altitude_m
        ):
            raise ValueError("takeoff altitude must be inside altitude limits")
        for name in (
            "cruise_speed_mps",
            "vertical_step_m",
            "vertical_speed_mps",
            "lateral_step_m",
            "lateral_speed_mps",
            "path_recovery_gain",
            "path_recovery_max_speed_mps",
            "path_recovery_tolerance_m",
            "command_watchdog_s",
            "capture_duration_s",
            "evade_repeat_s",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")


@dataclass(frozen=True, slots=True)
class AirSimRgbFrame:
    image_bgr: np.ndarray
    timestamp_ns: int

    @property
    def focal_px(self) -> float:
        height, width = self.image_bgr.shape[:2]
        del height
        return float(width) / 2.0


class Phase1EventFeed:
    """Follow only newly confirmed events from the newest Phase 1 run."""

    def __init__(self, run_root: str | Path) -> None:
        self.run_root = Path(run_root)
        self._path: Path | None = None
        self._position = 0
        self._seen: set[str] = set()

    def poll(self) -> list[dict[str, Any]]:
        latest = self._latest_run()
        if latest is None:
            return []
        path = latest / "events.jsonl"
        if path != self._path:
            self._path = path
            self._position = path.stat().st_size
            return []
        size = path.stat().st_size
        if size < self._position:
            self._position = 0
        with path.open("r", encoding="utf-8") as stream:
            stream.seek(self._position)
            lines = stream.readlines()
            self._position = stream.tell()
        result: list[dict[str, Any]] = []
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            event_id = str(event.get("event_id", "")).strip()
            if (
                event_id
                and event_id not in self._seen
                and event.get("state") == "CONFIRMED"
                and event.get("event_type") in PHASE1_EVENT_TYPES
            ):
                self._seen.add(event_id)
                result.append(event)
        return result

    def _latest_run(self) -> Path | None:
        if not self.run_root.exists():
            return None
        candidates = [
            item
            for item in self.run_root.iterdir()
            if item.is_dir() and (item / "events.jsonl").exists()
        ]
        return (
            max(candidates, key=lambda item: item.stat().st_mtime)
            if candidates
            else None
        )


class AirSimAutonomyController:
    """Single command owner for a SimpleFlight AirSim patrol demo.

    Scene RGB is the only obstacle-control image. The Jetson returns the
    metric-depth policy decision. Simulator depth is deliberately absent from
    this class and may only be collected by a separate evaluator.
    """

    def __init__(
        self,
        client: Any,
        sdk: Any,
        compute: _Compute,
        config: AirSimAutonomyConfig | None = None,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client
        self.sdk = sdk
        self.compute = compute
        self.config = config or AirSimAutonomyConfig()
        self._monotonic = monotonic
        self._mode = "UNINITIALIZED"
        self._event_hold_until = 0.0
        self._last_evade_at = float("-inf")
        self._pending_events: list[dict[str, Any]] = []
        self._seen_event_ids: set[str] = set()
        self._path_y_ned_m: float | None = None
        self._path_z_ned_m: float | None = None

    @property
    def mode(self) -> str:
        return self._mode

    def initialize(self, *, reset: bool = False) -> None:
        name = self.config.vehicle_name
        if not bool(self.client.ping()):
            raise AirSimAutonomyError("AirSim ping failed")
        self.compute.health()
        if reset:
            self.client.reset()
        self.client.enableApiControl(True, vehicle_name=name)
        if not bool(self.client.isApiControlEnabled(vehicle_name=name)):
            raise AirSimAutonomyError("AirSim API control was not granted")
        if not bool(self.client.armDisarm(True, vehicle_name=name)):
            raise AirSimAutonomyError("AirSim vehicle could not be armed")
        self.client.takeoffAsync(vehicle_name=name).join()
        self.client.moveToZAsync(
            -self.config.takeoff_altitude_m,
            self.config.vertical_speed_mps,
            timeout_sec=15.0,
            vehicle_name=name,
        ).join()
        self._remember_straight_path()
        self._mode = "HOLD"
        self._hover()

    def close(self, *, land: bool = False) -> None:
        name = self.config.vehicle_name
        try:
            self._hover()
            if land:
                self.client.landAsync(vehicle_name=name).join()
                self.client.armDisarm(False, vehicle_name=name)
        finally:
            self.client.enableApiControl(False, vehicle_name=name)

    def current_z_ned_m(self) -> float:
        return self.current_position_ned_m()[2]

    def current_position_ned_m(self) -> tuple[float, float, float]:
        state = self.client.getMultirotorState(
            vehicle_name=self.config.vehicle_name
        )
        position = state.kinematics_estimated.position
        return (
            float(getattr(position, "x_val", 0.0)),
            float(getattr(position, "y_val", 0.0)),
            float(position.z_val),
        )

    def read_scene(self) -> AirSimRgbFrame:
        request = self.sdk.ImageRequest(
            self.config.camera_name,
            self.sdk.ImageType.Scene,
            pixels_as_float=False,
            compress=True,
        )
        responses = self._sim_get_images([request])
        if len(responses) != 1:
            raise AirSimAutonomyError(
                "AirSim did not return exactly one Scene image"
            )
        response = responses[0]
        encoded = np.frombuffer(
            bytes(response.image_data_uint8), dtype=np.uint8
        )
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise AirSimAutonomyError("AirSim returned an invalid Scene image")
        return AirSimRgbFrame(image, int(response.time_stamp))

    def _sim_get_images(self, requests: list[Any]) -> list[Any]:
        try:
            return list(
                self.client.simGetImages(
                    requests, vehicle_name=self.config.vehicle_name
                )
            )
        except Exception as exc:
            # Microsoft's official AirSim 1.8.1 environments use the third
            # `external` RPC argument. Cosys-AirSim's current Python client
            # intentionally omits it. Use the wire-compatible overload only
            # for the explicit argument-count mismatch.
            if "invalid number of arguments" not in str(exc):
                raise
            rpc = getattr(self.client, "client", None)
            response_type = getattr(self.sdk, "ImageResponse", None)
            if rpc is None or response_type is None:
                raise
            raw = rpc.call(
                "simGetImages", requests, self.config.vehicle_name, False
            )
            return [response_type.from_msgpack(item) for item in raw]

    def handle_event(self, event: dict[str, Any]) -> bool:
        try:
            approved = self.compute.approve_event(event)
        except RemoteComputeError:
            self._fail_closed("event_gate_unavailable")
            return False
        if not approved:
            return False
        return self._begin_approved_event(event)

    def _begin_approved_event(self, event: dict[str, Any]) -> bool:
        event_id = str(event.get("event_id", "")).strip()
        event_type = str(event.get("event_type", ""))
        if (
            not event_id
            or event_id in self._seen_event_ids
            or event.get("state") != "CONFIRMED"
            or event_type not in PHASE1_EVENT_TYPES
        ):
            return False
        self._seen_event_ids.add(event_id)
        self._pending_events.append(dict(event))
        self._event_hold_until = max(
            self._event_hold_until,
            self._monotonic() + self.config.capture_duration_s,
        )
        self._hover()
        self._mode = "EVENT_HOLD"
        return True

    def drain_events(self) -> tuple[dict[str, Any], ...]:
        events = tuple(self._pending_events)
        self._pending_events.clear()
        return events

    def complete_event_capture(self, *, succeeded: bool) -> None:
        if not succeeded:
            self._event_hold_until = float("inf")
            self._fail_closed("event_capture_failed")

    def step(self) -> RemoteSafetyDecision | None:
        if self._monotonic() < self._event_hold_until:
            if self._mode != "EVENT_HOLD":
                self._hover()
                self._mode = "EVENT_HOLD"
            return None

        frame = self.read_scene()
        ok, encoded = cv2.imencode(
            ".jpg",
            frame.image_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
        )
        if not ok:
            self._fail_closed("jpeg_encode_failed")
            return None
        try:
            perception = self.compute.infer_perception(
                encoded.tobytes(),
                focal_px=self._focal_px(frame.image_bgr.shape[1]),
                current_z_ned_m=self.current_z_ned_m(),
                frame_timestamp_ns=frame.timestamp_ns,
                gimbal_forward=True,
            )
        except RemoteComputeError:
            self._fail_closed("jetson_unavailable")
            return None

        decision = perception.safety
        accepted_event = False
        for event in perception.events:
            accepted_event = (
                self._begin_approved_event(event) or accepted_event
            )
        if accepted_event:
            return decision

        if decision.state == "CLEAR":
            self._cruise()
        elif decision.state == "EVADE":
            self._evade(decision.direction)
        else:
            self._hover()
            self._mode = decision.state
        return decision

    def _focal_px(self, width: int) -> float:
        return (width / 2.0) / np.tan(
            np.deg2rad(self.config.camera_hfov_deg) / 2.0
        )

    def _cruise(self) -> None:
        _, y, z = self.current_position_ned_m()
        if self._path_y_ned_m is None or self._path_z_ned_m is None:
            self._remember_straight_path()
        assert self._path_y_ned_m is not None
        assert self._path_z_ned_m is not None
        cross_track_error = self._path_y_ned_m - y
        lateral_velocity = float(
            np.clip(
                self.config.path_recovery_gain * cross_track_error,
                -self.config.path_recovery_max_speed_mps,
                self.config.path_recovery_max_speed_mps,
            )
        )
        if abs(cross_track_error) <= self.config.path_recovery_tolerance_m:
            lateral_velocity = 0.0
        self.client.moveByVelocityZAsync(
            self.config.cruise_speed_mps,
            lateral_velocity,
            self._path_z_ned_m,
            self.config.command_watchdog_s,
            drivetrain=self.sdk.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=self.sdk.YawMode(is_rate=False, yaw_or_rate=0.0),
            vehicle_name=self.config.vehicle_name,
        )
        recovering_altitude = (
            abs(z - self._path_z_ned_m)
            > self.config.path_recovery_tolerance_m
        )
        self._mode = (
            "RECOVER_PATH"
            if lateral_velocity != 0.0 or recovering_altitude
            else "CRUISE"
        )

    def _evade(self, direction: str) -> None:
        now = self._monotonic()
        if now - self._last_evade_at < self.config.evade_repeat_s:
            self._hover()
            self._mode = "EVADE_HOLD"
            return
        z_now = self.current_z_ned_m()
        if direction in {"LEFT", "RIGHT"}:
            y_velocity = (
                -self.config.lateral_speed_mps
                if direction == "LEFT"
                else self.config.lateral_speed_mps
            )
            duration = (
                self.config.lateral_step_m
                / self.config.lateral_speed_mps
            )
            self._hover()
            self.client.moveByVelocityZAsync(
                0.0,
                y_velocity,
                z_now,
                duration,
                drivetrain=self.sdk.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=self.sdk.YawMode(
                    is_rate=False, yaw_or_rate=0.0
                ),
                vehicle_name=self.config.vehicle_name,
            ).join()
            self._hover()
            self._last_evade_at = now
            self._mode = f"EVADE_{direction}"
            return
        if direction == "UP":
            target = max(
                -self.config.max_altitude_m,
                z_now - self.config.vertical_step_m,
            )
        elif direction == "DOWN":
            target = min(
                -self.config.min_altitude_m,
                z_now + self.config.vertical_step_m,
            )
        else:
            self._fail_closed("invalid_evade_direction")
            return
        if abs(target - z_now) < 0.1:
            self._fail_closed("altitude_limit")
            return
        self._hover()
        self.client.moveToZAsync(
            target,
            self.config.vertical_speed_mps,
            timeout_sec=8.0,
            vehicle_name=self.config.vehicle_name,
        ).join()
        self._hover()
        self._last_evade_at = now
        self._mode = f"EVADE_{direction}"

    def _remember_straight_path(self) -> None:
        _, y, z = self.current_position_ned_m()
        self._path_y_ned_m = y
        self._path_z_ned_m = z

    def _hover(self) -> None:
        name = self.config.vehicle_name
        try:
            self.client.cancelLastTask(vehicle_name=name)
        except TypeError:
            self.client.cancelLastTask(name)
        self.client.hoverAsync(vehicle_name=name).join()

    def _fail_closed(self, reason: str) -> None:
        self._hover()
        self._mode = f"STALE:{reason}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Move AirSim straight and apply Jetson RGB depth decisions"
    )
    parser.add_argument("--airsim-host", default="127.0.0.1")
    parser.add_argument("--airsim-port", type=int, default=41451)
    parser.add_argument("--jetson-url", default="http://192.168.50.112:8765")
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--land", action="store_true")
    parser.add_argument(
        "--event-output",
        type=Path,
        default=Path(r"C:\work\jolgwajetson\outputs\event_captures"),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    sdk = importlib.import_module("cosysairsim")
    config = AirSimAutonomyConfig(host=args.airsim_host, port=args.airsim_port)
    airsim_client = sdk.MultirotorClient(
        ip=config.host,
        port=config.port,
        timeout_value=10,
    )
    compute = JetsonComputeClient(args.jetson_url)
    controller = AirSimAutonomyController(airsim_client, sdk, compute, config)
    ros_package_root = (
        Path(__file__).resolve().parents[2]
        / "ros2_ws"
        / "src"
        / "jolgwa_ros"
    )
    if str(ros_package_root) not in sys.path:
        sys.path.insert(0, str(ros_package_root))
    from jolgwa_ros.airsim_camera import AirSimSceneCapture
    from jolgwa_ros.event_clip import EventClipPolicy, EventClipRecorder

    capture = AirSimSceneCapture(
        host=config.host,
        port=config.port,
        vehicle_name=config.vehicle_name,
        camera_name=config.camera_name,
        fps=25.0,
    ).start()
    recorder = EventClipRecorder(
        args.event_output,
        EventClipPolicy(duration_s=config.capture_duration_s, fps=25),
    )
    deadline = time.monotonic() + args.duration_s
    try:
        controller.initialize(reset=args.reset)
        while time.monotonic() < deadline:
            decision = controller.step()
            for event in controller.drain_events():
                track_ids = event.get("track_ids") or []
                clip = recorder.record(
                    capture,
                    event_id=str(event["event_id"]),
                    event_type=str(event["event_type"]),
                    track_id=",".join(str(value) for value in track_ids),
                    confidence=float(event.get("confidence", 0.0)),
                    source="airsim-rgb-phase1-on-jetson",
                    extra_metadata={
                        "map": "AirSimNH",
                        "vehicle": config.vehicle_name,
                        "camera": config.camera_name,
                        "bbox": event.get("bbox", []),
                        "model_sha256": event.get("model_sha256", ""),
                        "phase1_inference_host": "jetson",
                    },
                )
                controller.complete_event_capture(succeeded=clip.succeeded)
                print(
                    json.dumps(
                        {
                            "event": event,
                            "clip_succeeded": clip.succeeded,
                            "clip_path": clip.video_path,
                            "frames_written": clip.frames_written,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if decision is not None:
                position_ned_m = controller.current_position_ned_m()
                print(
                    json.dumps(
                        {
                            "mode": controller.mode,
                            "position_ned_m": list(position_ned_m),
                            "state": decision.state,
                            "direction": decision.direction,
                            "front_m": decision.front_distance_m,
                            "left_m": decision.left_clearance_m,
                            "right_m": decision.right_clearance_m,
                            "upper_m": decision.upper_clearance_m,
                            "lower_m": decision.lower_clearance_m,
                            "inference_ms": decision.inference_ms,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        capture.stop()
        controller.close(land=args.land)
        compute.close()


if __name__ == "__main__":
    main()

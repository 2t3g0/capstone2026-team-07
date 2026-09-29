"""Send paired D435 RGB/depth to Jetson and publish its decisions only.

The d435i node/topic names are retained for compatibility.  A physical D435
does not have an IMU; pose and angular velocity come from PX4.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
import os
import threading
from .field_diagnostics import record as field_record
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import uuid

import rclpy
from jolgwa_interfaces.msg import EventObservation, SafetyDecision
from px4_msgs import msg as px4_msg
from px4_msgs.msg import VehicleLocalPosition
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import String
from rcl_interfaces.msg import ParameterDescriptor
from jolgwa_uav.phase1_contract import (
    MAX_JPEG_BYTES, MAX_RESULT_AGE_S, current_phase1_response,
    validate_input_age, validate_phase1_response, validate_request_identity,
)
from jolgwa_uav.compute_endpoint_guard import open_compute_request, compute_guard_from_environment

from .avoidance_bridge import avoidance_body_velocity, body_frd_to_ned
from .depth_pose_motion_gate import (
    LEGACY_RECEIPT, checked_mode, bridge_unavailable_reason,
)
from .event_control import PHASE1_EVENT_TYPES
from .evidence_clock import local_evidence_clock_id, evidence_deadline_error
from .phase1_bridge import normalized_bbox, select_target_bbox
from .phase1_input_profile import phase1_input_profile
from .topic_names import (
    EVENT_OBSERVATION,
    FORWARD_CAMERA_COMPRESSED,
    OBSTACLE_SAFETY_DECISION,
    PX4_LOCAL_POSITION,
)


# Older px4_msgs releases do not expose this message.
VehicleAngularVelocity = getattr(px4_msg, "VehicleAngularVelocity", None)
VehicleOdometry = getattr(px4_msg, "VehicleOdometry", None)
VehicleAttitude = getattr(px4_msg, "VehicleAttitude", None)


@dataclass(frozen=True)
class PositionSnapshot:
    state: tuple[float, float, float, float, float, float, float]
    received_at: float
    timestamp_us: int


@dataclass(frozen=True)
class AttitudeSnapshot:
    quaternion: tuple[float, float, float, float]
    received_at: float
    timestamp_us: int


@dataclass(frozen=True)
class CameraInfoSnapshot:
    values: dict
    frame_id: str
    received_at: float


@dataclass(frozen=True)
class GeometrySnapshot:
    query_values: dict
    attitude_received_at: float
    camera_info_received_at: float
    position: PositionSnapshot | None = None


def _px4_timestamp_us(message) -> int:
    return int(getattr(message, "timestamp_sample", 0) or getattr(message, "timestamp", 0))


def _depth_intrinsics(message: CameraInfo) -> dict:
    """Accept only the actual rectified depth grid, never RGB intrinsics/FOV."""
    width, height = int(message.width), int(message.height)
    k = tuple(float(value) for value in message.k)
    if width <= 0 or height <= 0 or len(k) != 9 or not all(math.isfinite(v) for v in k):
        raise ValueError("depth_camera_info_invalid_intrinsics")
    if (
        k[0] <= 0.0 or k[4] <= 0.0
        or not 0.0 <= k[2] < width or not 0.0 <= k[5] < height
        or any(abs(k[index]) > 1e-9 for index in (1, 3, 6, 7))
        or abs(k[8] - 1.0) > 1e-9
    ):
        raise ValueError("depth_camera_info_invalid_intrinsics")
    distortion = tuple(float(value) for value in message.d)
    if any(not math.isfinite(v) or abs(v) > 1e-9 for v in distortion):
        raise ValueError("depth_camera_info_not_rectified")
    if int(message.binning_x) > 1 or int(message.binning_y) > 1:
        raise ValueError("depth_camera_info_binning_unsupported")
    roi = message.roi
    if (
        int(roi.x_offset) != 0 or int(roi.y_offset) != 0
        or int(roi.width) not in (0, width) or int(roi.height) not in (0, height)
    ):
        raise ValueError("depth_camera_info_roi_unsupported")
    stamp_ns = _stamp_ns(message)
    if stamp_ns <= 0 or not str(message.header.frame_id):
        raise ValueError("depth_camera_info_frame_invalid")
    return {
        "depth_fx": k[0], "depth_fy": k[4], "depth_cx": k[2], "depth_cy": k[5],
        "depth_info_width": width, "depth_info_height": height,
        "camera_info_timestamp_ns": stamp_ns,
    }


def _stamp_ns(message) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _nearest_synchronized_pair(
    colors: list[tuple[CompressedImage, float]],
    depths: list[tuple[Image, float]],
    *,
    now: float,
    sensor_timeout_s: float,
    max_pair_skew_s: float,
    eager_pair_skew_s: float,
    pair_holdback_s: float,
) -> tuple[int, int] | None:
    """Return the newest color frame that has a sufficiently close depth frame.

    Gazebo and a physical D435 deliver the two ROS topics on independent
    callbacks.  Pairing a color frame only when its callback runs can compare it
    with the previous depth frame and create a false skew fault.  Keeping both
    streams briefly lets the corresponding depth callback arrive first.
    """
    fresh_colors = [
        (index, item)
        for index, item in enumerate(colors)
        if now - item[1] <= sensor_timeout_s
    ]
    fresh_depths = [
        (index, item)
        for index, item in enumerate(depths)
        if now - item[1] <= sensor_timeout_s
    ]
    for color_index, (color, color_received_at) in reversed(fresh_colors):
        color_stamp = _stamp_ns(color)
        if color_stamp <= 0:
            continue
        candidates: list[tuple[float, int]] = []
        for depth_index, (depth, depth_received_at) in fresh_depths:
            depth_stamp = _stamp_ns(depth)
            if depth_stamp <= 0:
                continue
            skew_s = abs(color_stamp - depth_stamp) / 1_000_000_000.0
            candidates.append((skew_s, depth_index))
        if not candidates:
            continue
        skew_s, depth_index = min(candidates)
        color_age_s = now - color_received_at
        if skew_s <= max_pair_skew_s and (
            skew_s <= eager_pair_skew_s or color_age_s >= pair_holdback_s
        ):
            return color_index, depth_index
    return None


class JetsonD435iBridgeNode(Node):
    """Pair sensor frames locally; Jetson owns safety and event inference."""

    def __init__(self, *, node_name="jetson_d435i_bridge", default_jetson_url="http://192.168.50.112:8765") -> None:
        super().__init__(node_name)
        self.declare_parameter("jetson_url", default_jetson_url)
        self.declare_parameter("color_topic", FORWARD_CAMERA_COMPRESSED)
        for name, default in (("phase1_color_topic", ""),
                              ("phase1_input_mode", "camera"),
                              ("phase1_evidence_topic", ""),
                              ("phase1_replay_frame_id", "")):
            self.declare_parameter(name, default, ParameterDescriptor(read_only=True))
        self.declare_parameter("depth_topic", "/camera/d435i/depth/image")
        self.declare_parameter("depth_camera_info_topic", "/camera/d435i/depth/camera_info")
        self.declare_parameter("geometry_required", True)
        self.declare_parameter("safety_sequence_seed", 0, ParameterDescriptor(read_only=True))
        # Strict is an evidence-insufficiency interlock, NOT a physical-flight
        # enable switch. Legacy preserves receipt-paired SITL/diagnostic trials.
        self.declare_parameter("depth_pose_motion_mode", LEGACY_RECEIPT,
                               ParameterDescriptor(read_only=True))
        # No physical mount calibration has been supplied.  Even nominal values
        # must not silently grant flight authority without an explicit profile.
        self.declare_parameter("geometry_calibrated", False)
        for name in ("x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad"):
            self.declare_parameter("camera_mount_" + name, 0.0)
        self.declare_parameter("request_timeout_s", 0.5)
        self.declare_parameter("min_publish_interval_s", 0.05)
        self.declare_parameter("phase1_enabled", True)
        self.declare_parameter("phase1_publish_interval_s", 1.0)
        self.declare_parameter("phase1_request_timeout_s", 10.0)
        self.declare_parameter("phase1_watchdog_timeout_s", 15.0)
        self.declare_parameter("sensor_timeout_s", 0.5)
        self.declare_parameter("position_timeout_s", 0.5)
        self.declare_parameter("max_pair_skew_s", 0.075)
        self.declare_parameter("eager_pair_skew_s", 0.010)
        self.declare_parameter("pair_holdback_s", 0.080)
        self.declare_parameter("require_angular_velocity", True)
        self.declare_parameter("gimbal_forward", True)
        self.declare_parameter("capture_duration_s", 5.0)

        self._jetson_url = str(
            self.get_parameter("jetson_url").value
        ).rstrip("/")
        self._compute_guard = compute_guard_from_environment(self._jetson_url)
        self._request_timeout_s = float(
            self.get_parameter("request_timeout_s").value
        )
        self._min_interval_s = float(
            self.get_parameter("min_publish_interval_s").value
        )
        self._phase1_enabled = bool(
            self.get_parameter("phase1_enabled").value
        )
        self._phase1_input = phase1_input_profile(
            color_topic=str(self.get_parameter("color_topic").value),
            phase1_topic=str(self.get_parameter("phase1_color_topic").value),
            mode=str(self.get_parameter("phase1_input_mode").value),
            evidence_topic=str(self.get_parameter("phase1_evidence_topic").value),
            replay_frame_id=str(self.get_parameter("phase1_replay_frame_id").value),
            environment=os.environ,
        )
        self._phase1_evidence_publisher = (self.create_publisher(
            String, self._phase1_input.evidence_topic, qos_profile_sensor_data)
            if self._phase1_input.evidence_topic else None)
        self._phase1_interval_s = float(
            self.get_parameter("phase1_publish_interval_s").value
        )
        self._phase1_request_timeout_s = float(
            self.get_parameter("phase1_request_timeout_s").value
        )
        self._phase1_watchdog_timeout_s = float(
            self.get_parameter("phase1_watchdog_timeout_s").value
        )
        self._sensor_timeout_s = float(
            self.get_parameter("sensor_timeout_s").value
        )
        self._position_timeout_s = float(
            self.get_parameter("position_timeout_s").value
        )
        self._max_pair_skew_s = float(
            self.get_parameter("max_pair_skew_s").value
        )
        self._eager_pair_skew_s = float(
            self.get_parameter("eager_pair_skew_s").value
        )
        self._pair_holdback_s = float(
            self.get_parameter("pair_holdback_s").value
        )
        self._require_angular_velocity = bool(
            self.get_parameter("require_angular_velocity").value
        )
        self._gimbal_forward = bool(
            self.get_parameter("gimbal_forward").value
        )
        self._capture_duration_s = float(
            self.get_parameter("capture_duration_s").value
        )
        self._geometry_required = bool(self.get_parameter("geometry_required").value)
        self._depth_pose_motion_mode = checked_mode(
            str(self.get_parameter("depth_pose_motion_mode").value))
        self._geometry_calibrated = bool(self.get_parameter("geometry_calibrated").value)
        self._camera_mount = {
            "camera_mount_" + name: float(self.get_parameter("camera_mount_" + name).value)
            for name in ("x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad")
        }
        if not all(math.isfinite(value) for value in self._camera_mount.values()):
            raise ValueError("camera mount extrinsics must be finite")
        timing_values = (
            self._request_timeout_s,
            self._min_interval_s,
            self._phase1_interval_s,
            self._phase1_request_timeout_s,
            self._phase1_watchdog_timeout_s,
            self._sensor_timeout_s,
            self._position_timeout_s,
            self._max_pair_skew_s,
            self._eager_pair_skew_s,
            self._pair_holdback_s,
            self._capture_duration_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in timing_values):
            raise ValueError("Jetson D435 bridge timing values must be finite and positive")
        if self._eager_pair_skew_s > self._max_pair_skew_s:
            raise ValueError("eager_pair_skew_s must not exceed max_pair_skew_s")
        if self._pair_holdback_s >= self._sensor_timeout_s:
            raise ValueError("pair_holdback_s must be below sensor_timeout_s")

        self._create_output_publishers()
        self.create_subscription(
            VehicleLocalPosition,
            PX4_LOCAL_POSITION,
            self._on_local_position,
            qos_profile_sensor_data,
        )
        if VehicleAngularVelocity is not None:
            self.create_subscription(
                VehicleAngularVelocity,
                "/fmu/out/vehicle_angular_velocity",
                self._on_angular_velocity,
                qos_profile_sensor_data,
            )
        if VehicleAttitude is not None:
            self.create_subscription(
                VehicleAttitude, "/fmu/out/vehicle_attitude",
                self._on_attitude, qos_profile_sensor_data,
            )
        if VehicleOdometry is not None:
            self.create_subscription(
                VehicleOdometry,
                "/fmu/out/vehicle_odometry",
                self._on_vehicle_odometry,
                qos_profile_sensor_data,
            )
        # Keep DDS delivery latest-only.  Pairing history lives in the bounded
        # deques below, not in an extra backlog of sensor subscription callbacks.
        camera_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.BEST_EFFORT
        )
        self.create_subscription(
            CameraInfo, str(self.get_parameter("depth_camera_info_topic").value),
            self._on_depth_camera_info, camera_qos,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("depth_topic").value),
            self._on_depth,
            camera_qos,
        )
        self.create_subscription(
            CompressedImage,
            str(self.get_parameter("color_topic").value),
            self._on_color,
            camera_qos,
        )
        if self._phase1_input.separate:
            self.create_subscription(CompressedImage, self._phase1_input.topic,
                                     self._on_phase1_color, camera_qos)

        self._lock = threading.Lock()
        self._safety_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="jetson-d435i-safety"
        )
        self._phase1_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="jetson-d435i-phase1"
        )
        self._color_frames: deque[tuple[CompressedImage, float]] = deque(maxlen=12)
        self._depth_frames: deque[tuple[Image, float]] = deque(maxlen=12)
        self._depth_received_at = float("-inf")
        self._position_received_at = float("-inf")
        self._position_timestamp_us = 0
        self._attitude_q: tuple[float, float, float, float] | None = None
        self._attitude_received_at = float("-inf")
        self._attitude_timestamp_us = 0
        self._depth_camera_info: dict | None = None
        self._camera_info_frame_id = ""
        self._camera_info_received_at = float("-inf")
        self._camera_info_error = "depth_camera_info_missing"
        # At most 64 metadata samples, and selection expires them at the
        # existing 0.5s leases.  Camera/PX4 topics can arrive in different orders.
        self._position_history: deque[PositionSnapshot] = deque(maxlen=64)
        self._attitude_history: deque[AttitudeSnapshot] = deque(maxlen=64)
        self._camera_info_history: deque[CameraInfoSnapshot] = deque(maxlen=64)
        self._last_safety_request_at = float("-inf")
        self._last_safety_decision_at = float("-inf")
        self._safety_request_active = False
        # The deadline belongs to the captured snapshot, not the HTTP response.
        self._completed_safety_result: tuple[dict, float, float, float] | None = None
        self._completed_safety_error: str | None = None
        self._last_phase1_request_at = float("-inf")
        self._last_phase1_result_at = float("-inf")
        self._phase1_request_active = False
        self._completed_phase1_result: dict | None = None
        self._completed_phase1_error: str | None = None
        self._phase1_session_id = str(uuid.uuid4())
        self._phase1_request_sequence = 0
        self._phase1_source_stamp = 0
        self._phase1_published_sequence = 0
        self._phase1_published_stamp = 0
        self._event_evidence_clock_id = local_evidence_clock_id()
        self._phase1_started_at = time.monotonic()
        self._last_phase1_warning = ""
        self._x_ned_m = math.nan
        self._y_ned_m = math.nan
        self._z_ned_m = math.nan
        self._yaw_rad = math.nan
        self._vx_ned_mps = math.nan
        self._vy_ned_mps = math.nan
        self._vz_ned_mps = math.nan
        self._angular_rate_rad_s = math.nan
        self._angular_velocity_received_at = float("-inf")
        self._sequence = int(self.get_parameter("safety_sequence_seed").value)
        if not 0 <= self._sequence <= 0xFFFFFFFF:
            raise ValueError("safety_sequence_seed must fit uint32")
        self._last_stale_reason = ""
        self._active_event: dict | None = None
        self._active_event_until = float("-inf")
        self._last_bbox: tuple[float, float, float, float] | None = None
        self._closing = False
        # Workers wake the ROS executor immediately instead of waiting up to
        # 100 ms for a timer tick.  All ROS publications still run in its callback.
        self._completion_guard = self.create_guard_condition(self._watchdog)
        # A system/NTP clock step must not pause freshness checks or inference
        # scheduling.  Message stamps deliberately remain on the ROS clock.
        self._watchdog_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(0.02, self._watchdog, clock=self._watchdog_clock)

    def _create_output_publishers(self) -> None:
        self._safety_publisher = self.create_publisher(
            SafetyDecision, OBSTACLE_SAFETY_DECISION, qos_profile_sensor_data
        )
        self._event_publisher = self.create_publisher(
            EventObservation, EVENT_OBSERVATION, qos_profile_sensor_data
        )

    def _safety_request_headers(self) -> dict[str, str]:
        return {"Content-Type": "application/octet-stream"}

    def destroy_node(self):
        with self._lock:
            self._closing = True
        self._safety_executor.shutdown(wait=False, cancel_futures=True)
        self._phase1_executor.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()

    def _on_local_position(self, message: VehicleLocalPosition) -> None:
        if not all(
            (
                bool(message.xy_valid),
                bool(message.z_valid),
                bool(message.v_xy_valid),
                bool(message.v_z_valid),
            )
        ):
            if self._geometry_required:
                with self._lock:
                    self._position_received_at = float("-inf")
                    self._position_timestamp_us = 0
                    self._position_history.clear()
            return
        state = tuple(
            float(value)
            for value in (
                message.x,
                message.y,
                message.z,
                message.heading,
                message.vx,
                message.vy,
                message.vz,
            )
        )
        if not all(math.isfinite(value) for value in state):
            if self._geometry_required:
                with self._lock:
                    self._position_received_at = float("-inf")
                    self._position_timestamp_us = 0
                    self._position_history.clear()
            return
        stamp_us = _px4_timestamp_us(message)
        with self._lock:
            if self._geometry_required and stamp_us <= 0:
                self._position_received_at = float("-inf")
                self._position_timestamp_us = 0
                self._position_history.clear()
                return
            if self._geometry_required and stamp_us <= self._position_timestamp_us:
                return
            (
                self._x_ned_m,
                self._y_ned_m,
                self._z_ned_m,
                self._yaw_rad,
                self._vx_ned_mps,
                self._vy_ned_mps,
                self._vz_ned_mps,
            ) = state
            self._position_received_at = time.monotonic()
            self._position_timestamp_us = stamp_us
            self._position_history.append(PositionSnapshot(state, self._position_received_at, stamp_us))
        self._schedule_synchronized_pair()

    def _on_attitude(self, message) -> None:
        """PX4 quaternion is body FRD to local NED, in w/x/y/z order."""
        stamp_us = _px4_timestamp_us(message)
        q = tuple(float(value) for value in message.q)
        norm = math.sqrt(sum(value * value for value in q))
        valid = (
            len(q) == 4 and all(math.isfinite(value) for value in q)
            and math.isfinite(norm) and abs(norm - 1.0) <= 0.01 and stamp_us > 0
        )
        with self._lock:
            if not valid:
                self._attitude_q = None
                self._attitude_received_at = float("-inf")
                self._attitude_history.clear()
            elif stamp_us > self._attitude_timestamp_us:
                self._attitude_q = tuple(value / norm for value in q)
                self._attitude_timestamp_us = stamp_us
                self._attitude_received_at = time.monotonic()
                self._attitude_history.append(AttitudeSnapshot(
                    self._attitude_q, self._attitude_received_at, stamp_us
                ))
        self._schedule_synchronized_pair()

    def _on_depth_camera_info(self, message: CameraInfo) -> None:
        try:
            values = _depth_intrinsics(message)
        except (TypeError, ValueError) as exc:
            with self._lock:
                self._depth_camera_info = None
                self._camera_info_received_at = float("-inf")
                self._camera_info_error = str(exc)
                self._camera_info_history.clear()
        else:
            with self._lock:
                old_stamp = (self._depth_camera_info or {}).get("camera_info_timestamp_ns", 0)
                if values["camera_info_timestamp_ns"] > old_stamp:
                    self._depth_camera_info = values
                    self._camera_info_frame_id = str(message.header.frame_id)
                    self._camera_info_received_at = time.monotonic()
                    self._camera_info_error = ""
                    self._camera_info_history.append(CameraInfoSnapshot(
                        values, self._camera_info_frame_id, self._camera_info_received_at
                    ))
        self._schedule_synchronized_pair()

    def _geometry_unavailable_reason_locked(self, now: float) -> str:
        motion_reason = bridge_unavailable_reason(
            getattr(self, "_depth_pose_motion_mode", LEGACY_RECEIPT))
        if motion_reason:
            return motion_reason
        if not self._geometry_calibrated:
            return "camera_mount_uncalibrated"
        if self._depth_camera_info is None:
            return self._camera_info_error or "depth_camera_info_missing"
        if now - self._camera_info_received_at > self._sensor_timeout_s:
            return "depth_camera_info_stale"
        if self._attitude_q is None:
            return "px4_attitude_missing_or_invalid"
        if now - self._attitude_received_at > self._position_timeout_s:
            return "px4_attitude_stale"
        if self._position_timestamp_us <= 0:
            return "px4_position_timestamp_invalid"
        return ""

    def _geometry_snapshot_locked(
        self, depth: Image, depth_received_at: float, now: float
    ) -> tuple[GeometrySnapshot | None, str]:
        reason = self._geometry_unavailable_reason_locked(now)
        if reason:
            return None, reason if self._geometry_required else ""
        for history, timeout in (
            (self._position_history, self._position_timeout_s),
            (self._attitude_history, self._position_timeout_s),
            (self._camera_info_history, self._sensor_timeout_s),
        ):
            while history and now - history[0].received_at > timeout:
                history.popleft()
        infos = [sample for sample in self._camera_info_history if (
            int(depth.width), int(depth.height)
        ) == (sample.values["depth_info_width"], sample.values["depth_info_height"])]
        if not infos:
            return None, "depth_camera_info_resolution_mismatch" if self._geometry_required else ""
        infos = [sample for sample in infos if sample.frame_id == str(depth.header.frame_id)]
        if not infos:
            return None, "depth_camera_info_frame_mismatch" if self._geometry_required else ""
        info_sample = min(infos, key=lambda item: abs(_stamp_ns(depth) - item.values["camera_info_timestamp_ns"]))
        info = info_sample.values
        if abs(_stamp_ns(depth) - info["camera_info_timestamp_ns"]) / 1e9 > self._max_pair_skew_s:
            return None, "depth_camera_info_timestamp_skew" if self._geometry_required else ""
        # This is a bounded receipt-time approximation, not a claim that camera
        # simulation timestamps and PX4 agent-adjusted clocks are interchangeable.
        positions = [sample for sample in self._position_history
                     if abs(depth_received_at - sample.received_at) <= self._max_pair_skew_s]
        attitudes = [sample for sample in self._attitude_history
                     if abs(depth_received_at - sample.received_at) <= self._max_pair_skew_s]
        if not positions or not attitudes:
            return None, "depth_pose_receipt_skew" if self._geometry_required else ""
        candidates = [
            (max(abs(depth_received_at - pos.received_at), abs(depth_received_at - att.received_at)),
             abs(pos.timestamp_us - att.timestamp_us), pos, att)
            for pos in positions for att in attitudes
            if abs(pos.timestamp_us - att.timestamp_us) / 1e6 <= self._max_pair_skew_s
        ]
        if not candidates:
            return None, "px4_attitude_position_skew" if self._geometry_required else ""
        _, _, position, attitude = min(candidates, key=lambda item: item[:2])
        query = dict(info)
        query.update(self._camera_mount)
        query.update(dict(zip(("attitude_qw", "attitude_qx", "attitude_qy", "attitude_qz"), attitude.quaternion)))
        query.update({
            "geometry_calibrated": "true",
            "attitude_timestamp_us": attitude.timestamp_us,
            "position_timestamp_us": position.timestamp_us,
        })
        return GeometrySnapshot(query, attitude.received_at, info_sample.received_at, position), ""

    def _on_angular_velocity(self, message) -> None:
        self._store_angular_velocity(message.xyz)

    def _on_vehicle_odometry(self, message) -> None:
        self._store_angular_velocity(message.angular_velocity)

    def _store_angular_velocity(self, values) -> None:
        xyz = tuple(float(value) for value in values)
        if len(xyz) != 3 or not all(math.isfinite(value) for value in xyz):
            return
        with self._lock:
            self._angular_rate_rad_s = math.sqrt(
                sum(value * value for value in xyz)
            )
            self._angular_velocity_received_at = time.monotonic()

    def _on_depth(self, message: Image) -> None:
        with self._lock:
            received_at = time.monotonic()
            self._depth_frames.append((message, received_at))
            self._depth_received_at = received_at
            field_record(self, 'camera_depth_receive', stamp_ns=_stamp_ns(message), valid=True)
        self._schedule_synchronized_pair()

    def _on_color(self, message: CompressedImage) -> None:
        if "jpeg" not in str(message.format).lower() and "jpg" not in str(
            message.format
        ).lower():
            self._publish_stale("camera_frame_not_jpeg")
            return
        received_at = time.monotonic()
        with self._lock:
            self._color_frames.append((message, received_at))
            field_record(self, 'camera_rgb_receive', stamp_ns=_stamp_ns(message), valid=True)
        if not getattr(getattr(self, "_phase1_input", None), "separate", False):
            self._schedule_phase1(message, received_at)
        self._schedule_synchronized_pair()

    def _on_phase1_color(self, message: CompressedImage) -> None:
        """Replay never enters depth pairing, geometry, or safety freshness."""
        if not getattr(getattr(self, "_phase1_input", None), "separate", False):
            return
        if (getattr(getattr(message, "header", None), "frame_id", None)
                != self._phase1_input.replay_frame_id):
            self.get_logger().warning("Phase1 replay frame ID mismatch; discarded")
            return
        if "jpeg" not in message.format.lower() and "jpg" not in message.format.lower():
            self.get_logger().warning("Phase1 replay frame is not JPEG; discarded")
            return
        self._schedule_phase1(message, time.monotonic())

    def _schedule_phase1(
        self, color: CompressedImage, received_at: float
    ) -> None:
        if not self._phase1_enabled or _stamp_ns(color) <= 0:
            return
        now = time.monotonic()
        with self._lock:
            if self._phase1_request_active or (
                now - self._last_phase1_request_at
                < self._phase1_interval_s
            ):
                return
            self._phase1_request_active = True
            self._last_phase1_request_at = now
        self._phase1_executor.submit(
            self._request_phase1_worker,
            color,
            received_at,
        )

    def _latest_complete_pair_locked(
        self, now: float
    ) -> tuple[tuple[int, int, GeometrySnapshot | None] | None, str]:
        """Newest usable color/depth/metadata snapshot, without consuming waits."""
        geometry_by_depth = {}
        pending_reason = ""
        for color_index in range(len(self._color_frames) - 1, -1, -1):
            color, color_received_at = self._color_frames[color_index]
            color_stamp = _stamp_ns(color)
            if color_stamp <= 0:
                continue
            candidates = sorted(
                (abs(color_stamp - _stamp_ns(depth)) / 1e9, depth_index)
                for depth_index, (depth, _) in enumerate(self._depth_frames)
                if _stamp_ns(depth) > 0
            )
            for skew_s, depth_index in candidates:
                if skew_s > self._max_pair_skew_s:
                    break
                if skew_s > self._eager_pair_skew_s and now - color_received_at < self._pair_holdback_s:
                    continue
                if depth_index not in geometry_by_depth:
                    depth, received_at = self._depth_frames[depth_index]
                    geometry_by_depth[depth_index] = self._geometry_snapshot_locked(depth, received_at, now)
                geometry, reason = geometry_by_depth[depth_index]
                if not reason:
                    return (color_index, depth_index, geometry), ""
                if not pending_reason:
                    pending_reason = reason
        return None, pending_reason

    def _schedule_synchronized_pair(self) -> None:
        now = time.monotonic()
        reason = ""
        angular_rate_rad_s = None
        angular_velocity_received_at = None
        with self._lock:
            for frames in (self._color_frames, self._depth_frames):
                while frames and now - frames[0][1] > self._sensor_timeout_s:
                    frames.popleft()
            if (
                self._closing
                or self._safety_request_active
                or self._completed_safety_result is not None
                or self._completed_safety_error is not None
                or now - self._last_safety_request_at < self._min_interval_s
            ):
                return
            reason = bridge_unavailable_reason(
                getattr(self, "_depth_pose_motion_mode", LEGACY_RECEIPT))
            if reason:
                pass  # Strict cannot be bypassed with geometry_required=false.
            elif now - self._position_received_at > self._position_timeout_s:
                reason = "px4_position_stale"
            elif (
                math.isfinite(self._angular_rate_rad_s)
                and now - self._angular_velocity_received_at <= self._position_timeout_s
            ):
                angular_rate_rad_s = self._angular_rate_rad_s
                angular_velocity_received_at = self._angular_velocity_received_at
            elif getattr(self, "_require_angular_velocity", False):
                reason = "px4_angular_velocity_stale"
            if not reason and self._geometry_required:
                # Explicit invalid/stale metadata revokes authority immediately.
                # Only incomplete matching of otherwise healthy streams can wait.
                reason = self._geometry_unavailable_reason_locked(now)
            if not reason:
                pair, pending_reason = self._latest_complete_pair_locked(now)
                if pair is None:
                    if (
                        not pending_reason
                        or (not self._last_stale_reason
                            and now - self._last_safety_decision_at < self._sensor_timeout_s)
                    ):
                        # Keep the previous decision's original lease, but do
                        # not publish it again or extend it while metadata waits.
                        return
                    reason = pending_reason
            if not reason:
                color_index, depth_index, geometry_snapshot = pair
                color, color_received_at = self._color_frames[color_index]
                depth, depth_received_at = self._depth_frames[depth_index]
                if geometry_snapshot is not None and geometry_snapshot.position is not None:
                    position = geometry_snapshot.position
                    (x_ned_m, y_ned_m, z_ned_m, yaw_rad,
                     vx_ned_mps, vy_ned_mps, vz_ned_mps) = position.state
                    position_received_at = position.received_at
                else:
                    # Legacy geometry opt-out is for compute/transport tests.
                    x_ned_m, y_ned_m, z_ned_m = self._x_ned_m, self._y_ned_m, self._z_ned_m
                    yaw_rad = self._yaw_rad
                    vx_ned_mps, vy_ned_mps, vz_ned_mps = self._vx_ned_mps, self._vy_ned_mps, self._vz_ned_mps
                    position_received_at = self._position_received_at
                oldest_received_at = min(color_received_at, depth_received_at)
                for _ in range(color_index + 1):
                    self._color_frames.popleft()
                for _ in range(depth_index + 1):
                    self._depth_frames.popleft()
                self._safety_request_active = True
                self._last_safety_request_at = now
        if reason:
            self._publish_stale_once(reason)
            return
        self._safety_executor.submit(
            self._request_safety_worker,
            color,
            depth,
            oldest_received_at,
            z_ned_m,
            yaw_rad,
            color_received_at=color_received_at,
            depth_received_at=depth_received_at,
            position_received_at=position_received_at,
            x_ned_m=x_ned_m,
            y_ned_m=y_ned_m,
            vx_ned_mps=vx_ned_mps,
            vy_ned_mps=vy_ned_mps,
            vz_ned_mps=vz_ned_mps,
            angular_rate_rad_s=angular_rate_rad_s,
            angular_velocity_received_at=angular_velocity_received_at,
            geometry_snapshot=geometry_snapshot,
        )

    def _request_safety_worker(
        self,
        color: CompressedImage,
        depth: Image,
        oldest_received_at: float,
        z_ned_m: float,
        yaw_rad: float,
        *,
        color_received_at: float | None = None,
        depth_received_at: float | None = None,
        position_received_at: float | None = None,
        x_ned_m: float | None = None,
        y_ned_m: float | None = None,
        vx_ned_mps: float | None = None,
        vy_ned_mps: float | None = None,
        vz_ned_mps: float | None = None,
        angular_rate_rad_s: float | None = None,
        angular_velocity_received_at: float | None = None,
        geometry_snapshot: GeometrySnapshot | None = None,
    ) -> None:
        try:
            now = time.monotonic()
            expires_at = oldest_received_at + self._sensor_timeout_s
            if position_received_at is not None:
                expires_at = min(
                    expires_at, position_received_at + self._position_timeout_s
                )
            if angular_velocity_received_at is not None:
                expires_at = min(
                    expires_at,
                    angular_velocity_received_at + self._position_timeout_s,
                )
            if self._geometry_required and geometry_snapshot is None:
                raise ValueError("required geometry snapshot is missing")
            if geometry_snapshot is not None:
                expires_at = min(
                    expires_at,
                    geometry_snapshot.attitude_received_at + self._position_timeout_s,
                    geometry_snapshot.camera_info_received_at + self._sensor_timeout_s,
                )
            if now >= expires_at:
                raise TimeoutError("RGB-D/pose snapshot expired before request")
            rgb_age_s = max(
                0.0,
                now
                - (
                    oldest_received_at
                    if color_received_at is None
                    else color_received_at
                ),
            )
            depth_age_s = max(
                0.0,
                now
                - (
                    oldest_received_at
                    if depth_received_at is None
                    else depth_received_at
                ),
            )
            pose_age_s = (
                None
                if position_received_at is None
                else max(0.0, now - position_received_at)
            )
            payload = self._request_safety_jetson(
                color,
                depth,
                frame_age_s=max(rgb_age_s, depth_age_s),
                z_ned_m=z_ned_m,
                rgb_age_s=rgb_age_s,
                depth_age_s=depth_age_s,
                pose_age_s=pose_age_s,
                x_ned_m=x_ned_m,
                y_ned_m=y_ned_m,
                heading_rad=yaw_rad,
                vx_ned_mps=vx_ned_mps,
                vy_ned_mps=vy_ned_mps,
                vz_ned_mps=vz_ned_mps,
                angular_rate_rad_s=angular_rate_rad_s,
                geometry_snapshot=geometry_snapshot,
            )
            field_record(self, 'depth_compute_complete', input_age_s=time.monotonic()-oldest_received_at,
                processing_s=time.monotonic()-now, valid=True)
            safety = payload.get("safety")
            if not isinstance(safety, dict):
                raise ValueError("Jetson response has no safety object")
            with self._lock:
                self._completed_safety_result = (payload, yaw_rad, expires_at, oldest_received_at)
                self._completed_safety_error = None
        except Exception as exc:
            with self._lock:
                self._completed_safety_error = str(exc)
                self._completed_safety_result = None
        finally:
            with self._lock:
                self._safety_request_active = False
            self._notify_completion()

    def _notify_completion(self) -> None:
        with self._lock:
            if not self._closing:
                self._completion_guard.trigger()

    def _request_safety_jetson(
        self,
        color: CompressedImage,
        depth: Image,
        *,
        frame_age_s: float,
        z_ned_m: float,
        rgb_age_s: float | None = None,
        depth_age_s: float | None = None,
        pose_age_s: float | None = None,
        x_ned_m: float | None = None,
        y_ned_m: float | None = None,
        heading_rad: float | None = None,
        vx_ned_mps: float | None = None,
        vy_ned_mps: float | None = None,
        vz_ned_mps: float | None = None,
        angular_rate_rad_s: float | None = None,
        geometry_snapshot: GeometrySnapshot | None = None,
    ) -> dict:
        # Recheck at the actual HTTP boundary. A queued/fabricated snapshot or
        # remote geometry_valid flag cannot supply missing exposure proof.
        motion_reason = bridge_unavailable_reason(
            getattr(self, "_depth_pose_motion_mode", LEGACY_RECEIPT))
        if motion_reason:
            raise ValueError(motion_reason)
        encode_started_at = time.monotonic()
        # This endpoint is localhost-only.  Compressing a D435 depth frame on
        # the same Jetson consumed about 22 ms per request and shortened the
        # original RGB-D/pose lease without adding transport safety.  The
        # service already validates and supports this bounded plain format.
        jpeg = memoryview(color.data)
        raw_depth = memoryview(depth.data)
        body = b"".join((jpeg, raw_depth))
        encode_elapsed_s = max(0.0, time.monotonic() - encode_started_at)
        frame_age_s += encode_elapsed_s
        if rgb_age_s is not None:
            rgb_age_s += encode_elapsed_s
        if depth_age_s is not None:
            depth_age_s += encode_elapsed_s
        if pose_age_s is not None:
            pose_age_s += encode_elapsed_s
        remaining_s = self._sensor_timeout_s - frame_age_s
        if pose_age_s is not None:
            remaining_s = min(remaining_s, self._position_timeout_s - pose_age_s)
        geometry_query = {}
        if self._geometry_required and geometry_snapshot is None:
            raise ValueError("required geometry snapshot is missing")
        if geometry_snapshot is not None:
            now = encode_started_at + encode_elapsed_s
            geometry_query = dict(geometry_snapshot.query_values)
            attitude_age_s = max(0.0, now - geometry_snapshot.attitude_received_at)
            camera_info_age_s = max(0.0, now - geometry_snapshot.camera_info_received_at)
            geometry_query.update({"attitude_age_s": attitude_age_s, "camera_info_age_s": camera_info_age_s})
            remaining_s = min(
                remaining_s, self._position_timeout_s - attitude_age_s,
                self._sensor_timeout_s - camera_info_age_s,
            )
        if not math.isfinite(remaining_s) or remaining_s <= 0.0:
            raise TimeoutError("RGB-D/pose snapshot expired during encoding")
        query_values = {
            "jpeg_size": len(jpeg),
            "depth_width": int(depth.width),
            "depth_height": int(depth.height),
            "depth_step": int(depth.step),
            "depth_encoding": str(depth.encoding),
            "depth_bigendian": str(bool(depth.is_bigendian)).lower(),
            "current_z_ned_m": z_ned_m,
            "frame_timestamp_ns": _stamp_ns(color),
            "depth_timestamp_ns": _stamp_ns(depth),
            "pair_skew_s": abs(_stamp_ns(color) - _stamp_ns(depth))
            / 1_000_000_000.0,
            "frame_age_s": frame_age_s,
            "gimbal_forward": str(self._gimbal_forward).lower(),
            "compressed": "false",
            "run_phase1": "false",
            "geometry_required": str(self._geometry_required).lower(),
            "geometry_calibrated": "false",
        }
        query_values.update(geometry_query)
        # Newer Jetson services validate the full RGB-D/pose snapshot.  Keep
        # the legacy frame_age_s and current_z_ned_m parameters above so this
        # bridge can still talk to an older d435i-compatible service, whose
        # FastAPI endpoint safely ignores the additional query parameters.
        optional_query_values = {
            "current_x_ned_m": x_ned_m,
            "current_y_ned_m": y_ned_m,
            "current_heading_rad": heading_rad,
            "current_vx_ned_mps": vx_ned_mps,
            "current_vy_ned_mps": vy_ned_mps,
            "current_vz_ned_mps": vz_ned_mps,
            "rgb_age_s": rgb_age_s,
            "depth_age_s": depth_age_s,
            "pose_age_s": pose_age_s,
            "current_angular_rate_rad_s": angular_rate_rad_s,
        }
        query_values.update(
            {
                key: value
                for key, value in optional_query_values.items()
                if value is not None
            }
        )
        query = urlencode(query_values)
        request = Request(
            f"{self._jetson_url}/v1/d435i-perception-decision?{query}",
            data=body,
            headers=self._safety_request_headers(),
            method="POST",
        )
        # urllib bounds socket operations, not total wall time.  The original
        # snapshot deadline is checked again on the ROS executor before publish.
        with open_compute_request(request, timeout=min(self._request_timeout_s, remaining_s),
                location=os.environ.get('JOLGWA_SIM_COMPUTE_LOCATION','jetson'), opener=urlopen,
                guard=getattr(self,'_compute_guard',None)) as response:
            value = json.loads(response.read().decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Jetson response was not an object")
        return value

    def _request_phase1_worker(
        self,
        color: CompressedImage,
        received_at: float,
    ) -> None:
        try:
            requested_at = time.monotonic()
            validate_input_age(requested_at-received_at)
            frame_stamp = _stamp_ns(color)
            with self._lock:
                if frame_stamp <= self._phase1_source_stamp:
                    raise ValueError("phase1_duplicate_or_regressing_source")
                self._phase1_request_sequence += 1
                sequence = self._phase1_request_sequence
                self._phase1_source_stamp = frame_stamp
            # Out-of-band local metadata: never accept these fields from JSON.
            posted_input: dict[str, str] = {}
            payload = self._request_phase1_jetson(
                color,
                frame_age_s=time.monotonic()-received_at,
                session_id=self._phase1_session_id,
                request_sequence=sequence,
                posted_input=posted_input,
            )
            completed_at = time.monotonic()
            payload = validate_phase1_response(
                payload, session_id=self._phase1_session_id, request_sequence=sequence,
                frame_timestamp_ns=frame_stamp, frame_received_at=received_at,
                requested_at=requested_at, completed_at=completed_at)
            digest, input_frame_id = (posted_input.get(name) for name in (
                "input_sha256", "input_frame_id"))
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)
                    or not isinstance(input_frame_id, str)):
                raise ValueError("phase1_missing_local_post_identity")
            payload["_local_evidence"].update(
                input_sha256=digest, input_frame_id=input_frame_id)
            with self._lock:
                self._completed_phase1_result = payload
                self._completed_phase1_error = None
        except Exception as exc:
            with self._lock:
                self._completed_phase1_result = None
                self._completed_phase1_error = str(exc)
        finally:
            with self._lock:
                self._phase1_request_active = False
            self._notify_completion()

    def _request_phase1_jetson(
        self,
        color: CompressedImage,
        *,
        frame_age_s: float,
        session_id: str,
        request_sequence: int,
        posted_input: dict[str, str],
    ) -> dict:
        started = time.monotonic()
        posted_input.clear()
        input_frame_id = str(getattr(color.header, "frame_id", ""))
        profile = getattr(self, "_phase1_input", None)
        if (getattr(profile, "separate", False)
                and input_frame_id != profile.replay_frame_id):
            raise ValueError("phase1_replay_frame_id_mismatch")
        validate_request_identity(session_id, request_sequence, _stamp_ns(color))
        validate_input_age(frame_age_s)
        data = bytes(color.data)
        if not data or len(data) > MAX_JPEG_BYTES:
            raise ValueError("phase1_jpeg_invalid_size")
        input_sha256 = hashlib.sha256(data).hexdigest()
        # Byte snapshot and SHA time consume the original input lease.
        frame_age_s += time.monotonic()-started
        validate_input_age(frame_age_s)
        posted_input.update(input_sha256=input_sha256, input_frame_id=input_frame_id)
        query = urlencode(
            {
                "frame_timestamp_ns": _stamp_ns(color),
                "frame_age_s": frame_age_s,
                "session_id": session_id,
                "request_sequence": request_sequence,
            }
        )
        request = Request(
            f"{self._jetson_url}/v1/phase1-decision?{query}",
            data=data,
            headers={"Content-Type": "image/jpeg"},
            method="POST",
        )
        with open_compute_request(
            request, timeout=min(self._phase1_request_timeout_s, MAX_RESULT_AGE_S-frame_age_s),
            location=os.environ.get('JOLGWA_SIM_COMPUTE_LOCATION','jetson'), opener=urlopen,
            guard=getattr(self,'_compute_guard',None)
        ) as response:
            value = json.loads(response.read().decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Jetson Phase 1 response was not an object")
        return value

    def _publish_safety(self, payload: dict, yaw_rad: float) -> None:
        state_name = str(payload.get("state", "")).upper()
        if state_name != "STALE":
            motion_reason = bridge_unavailable_reason(
                getattr(self, "_depth_pose_motion_mode", LEGACY_RECEIPT))
            if motion_reason:
                raise ValueError(motion_reason)
        states = {
            "CLEAR": SafetyDecision.STATE_CLEAR,
            "HOLD": SafetyDecision.STATE_HOLD,
            "EVADE": SafetyDecision.STATE_EVADE,
            "STALE": SafetyDecision.STATE_STALE,
        }
        if state_name not in states:
            raise ValueError(f"invalid Jetson safety state: {state_name!r}")
        observation_age = payload.get("observation_age_s", 0.0)
        if isinstance(observation_age, bool):
            raise ValueError("Jetson observation_age_s must be a finite nonnegative number")
        observation_age = float(observation_age)
        if not math.isfinite(observation_age) or observation_age < 0.0:
            raise ValueError("Jetson observation_age_s must be a finite nonnegative number")
        if state_name != "STALE" and observation_age > self._sensor_timeout_s:
            raise ValueError("Jetson observation is expired")
        geometry = {}
        for name in ("geometry_valid", "roof_clearance_verified", "obstacle_extent_valid", "roof_passage_verified"):
            value = payload.get(name, False)
            if not isinstance(value, bool):
                raise ValueError(f"Jetson {name} must be boolean")
            geometry[name] = value
        for name in ("roof_vertical_gap_m", "roof_height_m", "obstacle_far_north_m", "obstacle_far_east_m"):
            raw = payload.get(name)
            value = math.nan if raw is None else float(raw)
            if raw is not None and (isinstance(raw, bool) or not math.isfinite(value)):
                raise ValueError(f"invalid Jetson geometry metric {name}")
            geometry[name] = value
        if self._geometry_required and state_name != "STALE" and not geometry["geometry_valid"]:
            raise ValueError("required geometry result is missing or invalid")
        if geometry["roof_clearance_verified"] and not (
            geometry["geometry_valid"] and math.isfinite(geometry["roof_vertical_gap_m"])
            and geometry["roof_vertical_gap_m"] > 1.0 and math.isfinite(geometry["roof_height_m"])
        ):
            raise ValueError("roof clearance verification is inconsistent")
        if geometry["obstacle_extent_valid"] and not (
            geometry["geometry_valid"] and math.isfinite(geometry["obstacle_far_north_m"])
            and math.isfinite(geometry["obstacle_far_east_m"])
        ):
            raise ValueError("obstacle extent verification is inconsistent")
        if geometry["roof_passage_verified"] and not (
            geometry["geometry_valid"] and geometry["obstacle_extent_valid"]
        ):
            raise ValueError("roof passage verification is inconsistent")
        velocity = (0.0, 0.0, 0.0)
        if state_name == "EVADE":
            body_velocity = avoidance_body_velocity(
                str(payload.get("direction", "")),
                lateral_velocity_body_mps=float(
                    payload.get("lateral_velocity_body_mps", 0.0)
                ),
                vertical_velocity_ned_mps=float(
                    payload.get("vertical_velocity_ned_mps", 0.0)
                ),
            )
            velocity = body_frd_to_ned(body_velocity, yaw_rad)
        metrics: dict[str, float] = {}
        for name in (
            "front_distance_m",
            "upper_clearance_m",
            "lower_clearance_m",
            "effective_trigger_distance_m",
            "effective_release_distance_m",
        ):
            raw = payload.get(name)
            if raw is None:
                metrics[name] = math.nan
                continue
            value = float(raw)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"invalid Jetson safety metric {name}: {raw!r}")
            metrics[name] = value
        descent_corridor_clear = payload.get("descent_corridor_clear", False)
        if not isinstance(descent_corridor_clear, bool):
            raise ValueError("Jetson descent_corridor_clear must be boolean")
        self._publish(
            state=states[state_name],
            confidence=0.0 if state_name == "STALE" else 1.0,
            velocity_ned_mps=velocity,
            reason=str(payload.get("reason", "")) or state_name.lower(),
            front_distance_m=metrics["front_distance_m"],
            upper_clearance_m=metrics["upper_clearance_m"],
            lower_clearance_m=metrics["lower_clearance_m"],
            effective_trigger_distance_m=metrics[
                "effective_trigger_distance_m"
            ],
            effective_release_distance_m=metrics[
                "effective_release_distance_m"
            ],
            descent_corridor_clear=descent_corridor_clear,
            observation_age_s=observation_age,
            **geometry,
        )

    def _publish_events(self, payload: dict) -> None:
        now = time.monotonic()
        payload = current_phase1_response(payload, now=now)
        sequence, stamp = payload["request_sequence"], payload["frame_timestamp_ns"]
        if (payload.get("session_id") != self._phase1_session_id
                or sequence <= self._phase1_published_sequence
                or stamp <= self._phase1_published_stamp):
            raise ValueError("phase1_duplicate_or_regressing_publication")
        self._phase1_published_sequence, self._phase1_published_stamp = sequence, stamp
        self._publish_phase1_evidence(payload, now)
        expires_at = payload["_local_evidence"]["expires_at"]
        events = payload.get("events", [])
        detections = payload.get("phase1", {}).get("detections", [])
        if isinstance(events, list):
            for event in events:
                if not isinstance(event, dict):
                    continue
                if (
                    event.get("state") != "CONFIRMED"
                    or event.get("event_type") not in PHASE1_EVENT_TYPES
                ):
                    continue
                event = dict(event)
                event["_evidence_expires_monotonic_s"] = expires_at
                event["_evidence_clock_id"] = self._event_evidence_clock_id
                bbox = normalized_bbox(event.get("bbox"))
                if bbox is None:
                    bbox = (0.0, 0.0, 1.0, 1.0)
                self._active_event = event
                self._active_event_until = expires_at
                self._last_bbox = bbox
                self._publish_event(event, bbox, visible=True, state="CONFIRMED")
        if self._active_event is None or now >= self._active_event_until:
            self._active_event = None
            self._last_bbox = None
            return
        bbox = select_target_bbox(
            self._active_event,
            detections if isinstance(detections, list) else [],
            previous_bbox=self._last_bbox,
        )
        if bbox is not None:
            self._last_bbox = bbox
        self._publish_event(
            self._active_event,
            bbox or (0.0, 0.0, 0.0, 0.0),
            visible=bbox is not None,
            state="ACTIVE",
        )

    def _publish_event(self, event, bbox, *, visible: bool, state: str) -> None:
        expires_at = event.get("_evidence_expires_monotonic_s", 0.0)
        if type(expires_at) not in (float, int) or not math.isfinite(expires_at):
            return
        expires_ns = int(expires_at*1_000_000_000)
        clock_id = event.get("_evidence_clock_id", "")
        if evidence_deadline_error(clock_id, expires_ns,
                local_clock_id=self._event_evidence_clock_id,
                now_ns=int(time.monotonic()*1_000_000_000)):
            return
        message = EventObservation()
        message.header.stamp = self.get_clock().now().to_msg()
        message.event_id = str(event["event_id"])
        message.event_type = str(event["event_type"])
        message.state = state
        message.track_id = ",".join(
            str(value) for value in event.get("track_ids", [])
        ) or "untracked"
        message.confidence = float(event.get("confidence", 0.0))
        message.bbox_x1, message.bbox_y1, message.bbox_x2, message.bbox_y2 = bbox
        message.target_visible = visible
        message.source = getattr(getattr(self, "_phase1_input", None),
                                 "source", "jetson-phase1-d435i")
        message.evidence_clock_id = clock_id
        message.evidence_expires_monotonic_ns = expires_ns
        self._event_publisher.publish(message)

    def _publish_phase1_evidence(self, payload: dict, now: float) -> None:
        publisher = getattr(self, "_phase1_evidence_publisher", None)
        if publisher is None:
            return
        # Diagnostic best-effort only. No disk I/O or renewed observation lease.
        try:
            encoded = json.dumps({"kind": "phase1_validated",
                "received_monotonic_s": now,
                "source": self._phase1_input.source, "payload": payload},
                allow_nan=False, separators=(",", ":"))
            if len(encoded.encode("utf-8")) > 262144:
                raise ValueError("trial evidence exceeds 256KiB")
        except (TypeError, ValueError) as exc:
            self.get_logger().warning(f"Phase1 trial evidence discarded: {exc}")
            return
        try:
            publisher.publish(String(data=encoded))
        except Exception as exc:
            # Only the optional diagnostic publication is isolated. Do not put
            # event processing, evidence validation or safety IO in this catch.
            self.get_logger().warning(
                f"Phase1 trial evidence publication discarded: {type(exc).__name__}: {exc}"
            )

    def _watchdog(self) -> None:
        now = time.monotonic()
        with self._lock:
            completed_safety = self._completed_safety_result
            completed_safety_error = self._completed_safety_error
            completed_phase1 = self._completed_phase1_result
            completed_phase1_error = self._completed_phase1_error
            self._completed_safety_result = None
            self._completed_safety_error = None
            self._completed_phase1_result = None
            self._completed_phase1_error = None
            depth_age = now - self._depth_received_at
            position_age = now - self._position_received_at
            decision_age = now - self._last_safety_decision_at
            phase1_age = now - max(
                self._phase1_started_at, self._last_phase1_result_at
            )
            geometry_reason = self._geometry_unavailable_reason_locked(now) if self._geometry_required else ""
        if completed_safety is not None:
            payload, yaw_rad, expires_at, observed_at = completed_safety
            if now >= expires_at:
                self._publish_stale_once("jetson_d435i_response_expired")
            elif (
                position_age <= self._position_timeout_s
                and depth_age <= self._sensor_timeout_s
                and not geometry_reason
            ):
                try:
                    safety = dict(payload["safety"])
                    service_age = safety.get("observation_age_s", 0.0)
                    if isinstance(service_age, bool) or not math.isfinite(float(service_age)) or float(service_age) < 0.0:
                        raise ValueError("invalid service observation age")
                    safety["observation_age_s"] = max(float(service_age), now - observed_at)
                    safety["remaining_validity_s"] = max(0.0, expires_at - now)
                    self._publish_safety(safety, yaw_rad)
                except (KeyError, TypeError, ValueError) as exc:
                    self.get_logger().warning(f"Invalid Jetson safety response: {exc}")
                    self._publish_stale_once("jetson_d435i_response_invalid")
                else:
                    with self._lock:
                        # Do not grant an old sample another full freshness
                        # interval just because it completed an HTTP round trip.
                        self._last_safety_decision_at = expires_at - self._sensor_timeout_s
                    decision_age = now - self._last_safety_decision_at
        elif completed_safety_error is not None:
            self.get_logger().warning(
                f"Jetson D435i safety request failed: {completed_safety_error}"
            )
            self._publish_stale("jetson_d435i_unavailable")
        if completed_phase1 is not None:
            try:
                current = current_phase1_response(completed_phase1, now=time.monotonic())
                self._publish_events(current)
            except (KeyError, TypeError, ValueError) as exc:
                self._warn_phase1_once(f"Jetson Phase 1 evidence rejected: {exc}")
            else:
                with self._lock:
                    self._last_phase1_result_at = current["_local_evidence"]["completed_at"]
                    self._last_phase1_warning = ""
                phase1_age = now-self._last_phase1_result_at
                if current.get("status") != "OBSERVED":
                    self._warn_phase1_once("Jetson Phase 1 coverage is incomplete")
        elif completed_phase1_error is not None:
            self._warn_phase1_once(
                f"Jetson Phase 1 request failed: {completed_phase1_error}"
            )
        if position_age > self._position_timeout_s:
            self._publish_stale_once("px4_position_stale")
        elif depth_age > self._sensor_timeout_s:
            self._publish_stale_once("d435i_depth_stale")
        elif geometry_reason:
            self._publish_stale_once(geometry_reason)
        elif decision_age > self._sensor_timeout_s:
            self._publish_stale_once("jetson_d435i_decision_stale")
        if (
            self._phase1_enabled
            and phase1_age > self._phase1_watchdog_timeout_s
        ):
            self._warn_phase1_once("Jetson Phase 1 result is stale")
        if self._active_event is not None and now >= self._active_event_until:
            self._active_event = None
            self._last_bbox = None
        # A pair may become eligible after the holdback or rate limit while no
        # new sensor callback arrives.  Resume from the newest buffered pair;
        # never queue requests behind an in-flight request.
        self._schedule_synchronized_pair()

    def _warn_phase1_once(self, reason: str) -> None:
        if reason != self._last_phase1_warning:
            self.get_logger().warning(reason)
            self._last_phase1_warning = reason

    def _publish_stale_once(self, reason: str) -> None:
        if reason != self._last_stale_reason:
            self._publish_stale(reason)

    def _publish_stale(self, reason: str) -> None:
        self._publish(
            state=SafetyDecision.STATE_STALE,
            confidence=0.0,
            velocity_ned_mps=(0.0, 0.0, 0.0),
            reason=reason,
        )

    def _publish(
        self,
        *,
        state: int,
        confidence: float,
        velocity_ned_mps: tuple[float, float, float],
        reason: str,
        front_distance_m: float = math.nan,
        upper_clearance_m: float = math.nan,
        lower_clearance_m: float = math.nan,
        effective_trigger_distance_m: float = math.nan,
        effective_release_distance_m: float = math.nan,
        descent_corridor_clear: bool = False,
        observation_age_s: float = 0.0,
        geometry_valid: bool = False,
        roof_clearance_verified: bool = False,
        roof_vertical_gap_m: float = math.nan,
        roof_height_m: float = math.nan,
        obstacle_extent_valid: bool = False,
        obstacle_far_north_m: float = math.nan,
        obstacle_far_east_m: float = math.nan,
        roof_passage_verified: bool = False,
    ) -> None:
        message = SafetyDecision()
        message.stamp = self.get_clock().now().to_msg()
        message.source = "jetson-realsense-d435i"
        self._sequence = (self._sequence + 1) & 0xFFFFFFFF
        message.sequence = self._sequence
        message.state = int(state)
        message.confidence = float(confidence)
        message.velocity_ned_mps = [float(value) for value in velocity_ned_mps]
        message.max_speed_mps = float(
            math.sqrt(sum(value * value for value in velocity_ned_mps))
        )
        message.front_distance_m = float(front_distance_m)
        message.upper_clearance_m = float(upper_clearance_m)
        message.lower_clearance_m = float(lower_clearance_m)
        message.effective_trigger_distance_m = float(
            effective_trigger_distance_m
        )
        message.effective_release_distance_m = float(
            effective_release_distance_m
        )
        message.descent_corridor_clear = bool(descent_corridor_clear)
        message.observation_age_s = float(observation_age_s)
        message.geometry_valid = bool(geometry_valid)
        message.roof_clearance_verified = bool(roof_clearance_verified)
        message.roof_vertical_gap_m = float(roof_vertical_gap_m)
        message.roof_height_m = float(roof_height_m)
        message.obstacle_extent_valid = bool(obstacle_extent_valid)
        message.obstacle_far_north_m = float(obstacle_far_north_m)
        message.obstacle_far_east_m = float(obstacle_far_east_m)
        message.roof_passage_verified = bool(roof_passage_verified)
        message.reason = reason
        self._last_stale_reason = reason if state == SafetyDecision.STATE_STALE else ""
        self._safety_publisher.publish(message)
        field_record(self, 'safety_publish', sequence=int(message.sequence),
            input_age_s=float(message.observation_age_s), state=int(message.state),
            reason=message.reason, valid=message.state != SafetyDecision.STATE_STALE)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = JetsonD435iBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

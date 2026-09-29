from __future__ import annotations

import json
import math
from pathlib import Path
import re
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import rclpy
from jolgwa_interfaces.msg import SafetyDecision
from px4_msgs.msg import VehicleLocalPosition
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

from .avoidance_bridge import avoidance_body_velocity, body_frd_to_ned
from .topic_names import (
    FORWARD_CAMERA_COMPRESSED,
    JETSON_SAFETY_DECISION,
    PX4_LOCAL_POSITION,
)


class JetsonSafetyBridgeNode(Node):
    """Publish Depth Anything avoidance decisions for the PX4 command owner."""

    def __init__(self) -> None:
        super().__init__("jetson_safety_bridge")
        self.declare_parameter("jetson_url", "http://192.168.50.112:8765")
        self.declare_parameter("camera_topic", FORWARD_CAMERA_COMPRESSED)
        self.declare_parameter("camera_focal_px", 640.0)
        self.declare_parameter("request_timeout_s", 2.0)
        self.declare_parameter("min_publish_interval_s", 0.1)
        self.declare_parameter("position_timeout_s", 0.5)
        self.declare_parameter("gimbal_forward", True)
        self.declare_parameter("evidence_dir", "")
        self.declare_parameter("evidence_max_pairs", 100)

        self._jetson_url = str(
            self.get_parameter("jetson_url").value
        ).rstrip("/")
        self._focal_px = float(self.get_parameter("camera_focal_px").value)
        self._request_timeout_s = float(
            self.get_parameter("request_timeout_s").value
        )
        self._min_interval_s = float(
            self.get_parameter("min_publish_interval_s").value
        )
        self._position_timeout_s = float(
            self.get_parameter("position_timeout_s").value
        )
        self._gimbal_forward = bool(
            self.get_parameter("gimbal_forward").value
        )
        evidence_dir = str(self.get_parameter("evidence_dir").value).strip()
        self._evidence_dir = Path(evidence_dir) if evidence_dir else None
        self._evidence_max_pairs = int(
            self.get_parameter("evidence_max_pairs").value
        )
        self._last_evidence_signature: tuple[str, str] | None = None
        self._evidence_count = 0
        if (
            self._focal_px <= 0.0
            or self._request_timeout_s <= 0.0
            or self._min_interval_s <= 0.0
            or self._position_timeout_s <= 0.0
            or self._evidence_max_pairs < 1
        ):
            raise ValueError(
                "Jetson bridge timing and focal parameters must be positive"
            )
        if self._evidence_dir is not None:
            self._evidence_dir.mkdir(parents=True, exist_ok=True)

        self._publisher = self.create_publisher(
            SafetyDecision,
            JETSON_SAFETY_DECISION,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            VehicleLocalPosition,
            PX4_LOCAL_POSITION,
            self._on_local_position,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CompressedImage,
            str(self.get_parameter("camera_topic").value),
            self._on_image,
            qos_profile_sensor_data,
        )
        self._z_ned_m = math.nan
        self._yaw_rad = math.nan
        self._position_received_at = float("-inf")
        self._last_request_at = float("-inf")
        self._sequence = 0

    def _on_local_position(self, message: VehicleLocalPosition) -> None:
        if not bool(message.z_valid) or not bool(message.xy_valid):
            return
        z_ned_m = float(message.z)
        yaw_rad = float(message.heading)
        if math.isfinite(z_ned_m) and math.isfinite(yaw_rad):
            self._z_ned_m = z_ned_m
            self._yaw_rad = yaw_rad
            self._position_received_at = time.monotonic()

    def _on_image(self, message: CompressedImage) -> None:
        now = time.monotonic()
        if now - self._last_request_at < self._min_interval_s:
            return
        self._last_request_at = now
        if now - self._position_received_at > self._position_timeout_s:
            self._publish_stale("px4_position_stale")
            return
        if "jpeg" not in str(message.format).lower() and "jpg" not in str(
            message.format
        ).lower():
            self._publish_stale("camera_frame_not_jpeg")
            return
        try:
            payload = self._request_depth(bytes(message.data), message)
            self._save_evidence_pair(bytes(message.data), message, payload)
            self._publish_payload(payload)
        except Exception as exc:
            self.get_logger().warning(f"Jetson depth request failed: {exc}")
            self._publish_stale("jetson_depth_unavailable")

    def _request_depth(
        self, jpeg: bytes, message: CompressedImage
    ) -> dict:
        stamp = message.header.stamp
        timestamp_ns = int(stamp.sec) * 1_000_000_000 + int(
            stamp.nanosec
        )
        if timestamp_ns <= 0:
            timestamp_ns = self.get_clock().now().nanoseconds
        query = urlencode(
            {
                "focal_px": self._focal_px,
                "current_z_ned_m": self._z_ned_m,
                "frame_timestamp_ns": timestamp_ns,
                "gimbal_forward": str(self._gimbal_forward).lower(),
            }
        )
        request = Request(
            f"{self._jetson_url}/v1/depth-decision?{query}",
            data=jpeg,
            headers={"Content-Type": "image/jpeg"},
            method="POST",
        )
        with urlopen(request, timeout=self._request_timeout_s) as response:
            value = json.loads(response.read().decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Jetson response was not an object")
        return value

    def _save_evidence_pair(
        self,
        jpeg: bytes,
        message: CompressedImage,
        payload: dict,
    ) -> None:
        if (
            self._evidence_dir is None
            or self._evidence_count >= self._evidence_max_pairs
        ):
            return
        state = str(payload.get("state", "UNKNOWN")).upper()
        reason = str(payload.get("reason", ""))
        signature = (state, _reason_bucket(reason))
        if signature == self._last_evidence_signature:
            return
        self._last_evidence_signature = signature
        safe_reason = re.sub(r"[^a-zA-Z0-9_-]+", "_", signature[1])
        stem = (
            f"{self._evidence_count:03d}_{time.time_ns()}_"
            f"{state.lower()}_{safe_reason}"
        )
        image_path = self._evidence_dir / f"{stem}.jpg"
        json_path = self._evidence_dir / f"{stem}.json"
        image_path.write_bytes(jpeg)
        stamp = message.header.stamp
        metadata = {
            "image": str(image_path),
            "camera_topic": str(self.get_parameter("camera_topic").value),
            "camera_format": str(message.format),
            "frame_timestamp_ns": int(stamp.sec) * 1_000_000_000
            + int(stamp.nanosec),
            "focal_px": self._focal_px,
            "current_z_ned_m": self._z_ned_m,
            "yaw_rad": self._yaw_rad,
            "jetson_response": payload,
        }
        json_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._evidence_count += 1

    def _publish_payload(self, payload: dict) -> None:
        state_name = str(payload.get("state", "")).upper()
        state_mapping = {
            "CLEAR": SafetyDecision.STATE_CLEAR,
            "HOLD": SafetyDecision.STATE_HOLD,
            "EVADE": SafetyDecision.STATE_EVADE,
            "STALE": SafetyDecision.STATE_STALE,
        }
        if state_name not in state_mapping:
            raise ValueError(f"invalid Jetson safety state: {state_name!r}")

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
            velocity = body_frd_to_ned(body_velocity, self._yaw_rad)

        self._publish(
            state=state_mapping[state_name],
            confidence=0.0 if state_name == "STALE" else 1.0,
            velocity_ned_mps=velocity,
            reason=str(payload.get("reason", "")) or state_name.lower(),
        )

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
    ) -> None:
        message = SafetyDecision()
        message.stamp = self.get_clock().now().to_msg()
        message.source = "jetson-depth-anything-v3"
        self._sequence = (self._sequence + 1) & 0xFFFFFFFF
        message.sequence = self._sequence
        message.state = int(state)
        message.confidence = float(confidence)
        magnitude = math.sqrt(sum(value * value for value in velocity_ned_mps))
        message.max_speed_mps = float(magnitude)
        message.velocity_ned_mps = [
            float(value) for value in velocity_ned_mps
        ]
        message.reason = reason
        self._publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = JetsonSafetyBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _reason_bucket(reason: str) -> str:
    if reason.startswith("obstacle_confirmation"):
        return "obstacle_confirmation"
    if "outdoor_climb" in reason:
        return "outdoor_climb"
    if "altitude_ceiling" in reason:
        return "altitude_ceiling"
    if "corridor_blocked" in reason:
        return "corridor_blocked"
    if "released" in reason:
        return "released"
    if "clear" in reason:
        return "clear"
    return reason or "unknown"

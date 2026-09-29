from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any

from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage


@dataclass(frozen=True)
class RosCapturedFrame:
    sequence: int
    captured_at: float
    image: Any


class RosCompressedImageCapture:
    """Expose a ROS JPEG topic through the event recorder capture contract."""

    def __init__(self, node, topic: str) -> None:
        topic = str(topic).strip()
        if not topic:
            raise ValueError("ROS compressed-image topic must not be empty")
        self._node = node
        self.topic = topic
        self._condition = threading.Condition()
        self._running = False
        self._subscription = None
        self._sequence = 0
        self._latest: tuple[int, float, bytes] | None = None
        self.last_error = ""

    def start(self) -> "RosCompressedImageCapture":
        with self._condition:
            if self._running:
                return self
            self._running = True
            self._subscription = self._node.create_subscription(
                CompressedImage,
                self.topic,
                self._on_image,
                qos_profile_sensor_data,
            )
        return self

    def stop(self) -> None:
        with self._condition:
            self._running = False
            subscription = self._subscription
            self._subscription = None
            self._condition.notify_all()
        if subscription is not None:
            try:
                self._node.destroy_subscription(subscription)
            except Exception:
                pass

    def _on_image(self, message: CompressedImage) -> None:
        image_format = str(message.format).lower()
        payload = bytes(message.data)
        with self._condition:
            if not self._running:
                return
            if ("jpeg" not in image_format and "jpg" not in image_format) or not payload:
                self.last_error = "ROS camera frame is not a non-empty JPEG"
                return
            self._sequence += 1
            self._latest = (self._sequence, time.monotonic(), payload)
            self.last_error = ""
            self._condition.notify_all()

    def read_after(
        self, sequence: int, *, timeout_s: float = 0.5
    ) -> RosCapturedFrame | None:
        import cv2
        import numpy as np

        deadline = time.monotonic() + max(0.0, float(timeout_s))
        latest = None
        with self._condition:
            while self._running:
                if self._latest is not None and self._latest[0] > sequence:
                    latest = self._latest
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(remaining)
        if latest is None:
            return None
        frame_sequence, captured_at, payload = latest
        image = cv2.imdecode(
            np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            self.last_error = "ROS JPEG decode failed"
            return None
        self.last_error = ""
        return RosCapturedFrame(frame_sequence, captured_at, image)


class FixedForwardGimbal:
    """Fixed Gazebo camera backend retaining the gimbal interface."""

    enabled = False

    def __init__(self) -> None:
        self.last_yaw_deg = 0.0
        self.last_pitch_deg = 0.0
        self.last_error = ""

    def set_attitude(self, yaw_deg: float, pitch_deg: float) -> None:
        if not math.isfinite(yaw_deg) or not math.isfinite(pitch_deg):
            raise ValueError("fixed-camera tracking angles must be finite")
        # The camera is physically fixed.  Keep the requested values only as
        # diagnostics; no Gazebo or PX4 command is emitted.
        self.last_yaw_deg = float(yaw_deg)
        self.last_pitch_deg = float(pitch_deg)

    def stop(self) -> None:
        return None

    def close(self) -> None:
        return None

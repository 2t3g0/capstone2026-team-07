"""Inject a documented incident clip into the Gazebo forward RGB stream.

The original Gazebo header is preserved so the composited RGB remains paired
with native simulator depth.  Depth pixels are never modified by this node.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import rclpy
from px4_msgs.msg import VehicleLocalPosition
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from .topic_names import PX4_LOCAL_POSITION


class IncidentMedia:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise ValueError(f"incident media does not exist: {self.path}")
        self._still = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        self._capture = None
        self._fps = 0.0
        self._frame_count = 0
        self._current_index = -1
        self._current_frame = None
        if self._still is None:
            capture = cv2.VideoCapture(str(self.path))
            if not capture.isOpened():
                raise ValueError(f"cannot open incident media: {self.path}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if not math.isfinite(fps) or fps <= 0.0 or frame_count <= 0:
                capture.release()
                raise ValueError("incident video has invalid FPS or frame count")
            self._capture = capture
            self._fps = fps
            self._frame_count = frame_count

    @property
    def is_video(self) -> bool:
        return self._capture is not None

    @property
    def duration_s(self) -> float:
        if self._capture is None:
            return math.inf
        return self._frame_count / self._fps

    def frame_at(self, elapsed_s: float):
        if self._still is not None:
            return self._still.copy()
        assert self._capture is not None
        index = int(max(0.0, elapsed_s) * self._fps) % self._frame_count
        if index == self._current_index and self._current_frame is not None:
            return self._current_frame.copy()
        if index <= self._current_index:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._current_index = -1
            self._current_frame = None
        frame = None
        while self._current_index < index:
            ok, frame = self._capture.read()
            if not ok or frame is None:
                self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._current_index = -1
                self._current_frame = None
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    raise RuntimeError("incident video frame decode failed")
            self._current_index += 1
        if frame is None:
            ok, frame = self._capture.read()
            if not ok or frame is None:
                raise RuntimeError("incident video frame decode failed")
            self._current_index += 1
        self._current_frame = frame
        return frame.copy()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


def composite_inset(base, incident, margin_fraction: float):
    if not 0.0 <= margin_fraction < 0.45:
        raise ValueError("inset margin must be in [0, 0.45)")
    height, width = base.shape[:2]
    margin_x = round(width * margin_fraction)
    margin_y = round(height * margin_fraction)
    target_width = width - 2 * margin_x
    target_height = height - 2 * margin_y
    if target_width <= 0 or target_height <= 0:
        raise ValueError("incident inset has no drawable area")
    # Fill the target region without geometrically stretching the source.
    # A 16:9 real incident clip fed into a 4:3 D435 stream was previously
    # squeezed enough to make the Phase-1 detector miss an otherwise stable
    # fire.  Centre-cropping is also what a physical display/camera framing
    # change would do, and keeps people, vehicles and flames undistorted.
    incident_height, incident_width = incident.shape[:2]
    target_aspect = target_width / target_height
    incident_aspect = incident_width / incident_height
    if incident_aspect > target_aspect:
        cropped_width = max(1, round(incident_height * target_aspect))
        crop_x = max(0, (incident_width - cropped_width) // 2)
        incident = incident[:, crop_x : crop_x + cropped_width]
    elif incident_aspect < target_aspect:
        cropped_height = max(1, round(incident_width / target_aspect))
        crop_y = max(0, (incident_height - cropped_height) // 2)
        incident = incident[crop_y : crop_y + cropped_height, :]
    resized = cv2.resize(
        incident,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )
    output = base.copy()
    output[
        margin_y : margin_y + target_height,
        margin_x : margin_x + target_width,
    ] = resized
    return output


class IncidentVideoCompositorNode(Node):
    def __init__(self) -> None:
        super().__init__("incident_video_compositor")
        self.declare_parameter("enabled", False)
        self.declare_parameter(
            "input_topic", "/camera/front/gazebo/compressed"
        )
        self.declare_parameter(
            "output_topic", "/camera/front/compressed"
        )
        self.declare_parameter("media_path", "")
        self.declare_parameter("trigger_x_m", 24.0)
        self.declare_parameter("minimum_altitude_m", 5.0)
        self.declare_parameter("active_duration_s", 15.0)
        self.declare_parameter("inset_margin_fraction", 0.08)
        self.declare_parameter("jpeg_quality", 88)
        self.declare_parameter(
            "status_topic", "/jolgwa/simulation/incident/status"
        )

        self._enabled = bool(self.get_parameter("enabled").value)
        self._trigger_x_m = float(self.get_parameter("trigger_x_m").value)
        self._minimum_altitude_m = float(
            self.get_parameter("minimum_altitude_m").value
        )
        self._active_duration_s = float(
            self.get_parameter("active_duration_s").value
        )
        self._margin_fraction = float(
            self.get_parameter("inset_margin_fraction").value
        )
        self._jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        if not math.isfinite(self._trigger_x_m):
            raise ValueError("trigger_x_m must be finite")
        if self._minimum_altitude_m < 0.0 or self._active_duration_s <= 5.0:
            raise ValueError(
                "incident altitude must be non-negative and duration above 5 seconds"
            )
        if not 1 <= self._jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if not 0.0 <= self._margin_fraction < 0.45:
            raise ValueError("inset_margin_fraction must be in [0, 0.45)")

        self._media = None
        if self._enabled:
            self._media = IncidentMedia(
                str(self.get_parameter("media_path").value)
            )
        self._lock = threading.Lock()
        self._x_m = math.nan
        self._altitude_m = math.nan
        self._active_started_at = None
        self._completed = False
        self._last_status_state = ""
        self._last_status_at = float("-inf")
        self._publisher = self.create_publisher(
            CompressedImage,
            str(self.get_parameter("output_topic").value),
            qos_profile_sensor_data,
        )
        self._status_publisher = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), 10
        )
        self.create_subscription(
            VehicleLocalPosition,
            PX4_LOCAL_POSITION,
            self._on_position,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CompressedImage,
            str(self.get_parameter("input_topic").value),
            self._on_image,
            qos_profile_sensor_data,
        )
        self._publish_status("WAITING" if self._enabled else "DISABLED")

    def _on_position(self, message: VehicleLocalPosition) -> None:
        if not bool(message.xy_valid) or not bool(message.z_valid):
            return
        x_m = float(message.x)
        altitude_m = -float(message.z)
        if math.isfinite(x_m) and math.isfinite(altitude_m):
            with self._lock:
                self._x_m = x_m
                self._altitude_m = altitude_m

    def _activation_elapsed(self, now: float) -> float | None:
        if not self._enabled or self._completed:
            return None
        with self._lock:
            x_m = self._x_m
            altitude_m = self._altitude_m
            started_at = self._active_started_at
            if (
                started_at is None
                and x_m >= self._trigger_x_m
                and altitude_m >= self._minimum_altitude_m
            ):
                started_at = now
                self._active_started_at = now
        if started_at is None:
            return None
        elapsed = now - started_at
        if elapsed >= self._active_duration_s:
            with self._lock:
                self._completed = True
            self._publish_status("COMPLETED")
            return None
        self._publish_status("ACTIVE")
        return elapsed

    def _on_image(self, message: CompressedImage) -> None:
        now = time.monotonic()
        elapsed = self._activation_elapsed(now)
        if elapsed is None:
            self._publisher.publish(message)
            return
        try:
            base = cv2.imdecode(
                np.frombuffer(bytes(message.data), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if base is None:
                raise RuntimeError("Gazebo JPEG decode failed")
            assert self._media is not None
            incident = self._media.frame_at(elapsed)
            composed = composite_inset(base, incident, self._margin_fraction)
            ok, encoded = cv2.imencode(
                ".jpg",
                composed,
                [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
            )
            if not ok:
                raise RuntimeError("incident JPEG encode failed")
        except Exception as exc:
            self.get_logger().error(f"incident compositor failed: {exc}")
            self._publish_status("ERROR", detail=str(exc))
            self._publisher.publish(message)
            return
        output = CompressedImage()
        output.header = message.header
        output.format = "jpeg"
        output.data = encoded.tobytes()
        self._publisher.publish(output)

    def _publish_status(self, state: str, *, detail: str = "") -> None:
        now = time.monotonic()
        if state == self._last_status_state and now - self._last_status_at < 1.0:
            return
        self._last_status_state = state
        self._last_status_at = now
        message = String()
        message.data = json.dumps(
            {
                "state": state,
                "media_path": str(self.get_parameter("media_path").value),
                "trigger_x_m": self._trigger_x_m,
                "active_duration_s": self._active_duration_s,
                "rgb_depth_policy": "composited_rgb_with_unmodified_native_depth",
                "detail": detail,
            },
            ensure_ascii=False,
        )
        self._status_publisher.publish(message)

    def destroy_node(self):
        if self._media is not None:
            self._media.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = IncidentVideoCompositorNode()
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

"""Normalize Gazebo RGB/depth camera topics for perception consumers."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Range


SOURCE_RGB_TOPIC = "/uav/camera/rgb/image_raw"
SOURCE_DEPTH_TOPIC = "/uav/camera/depth/image_raw"
PERCEPTION_RGB_TOPIC = "/jolgwa/perception/rgb/image_raw"
PERCEPTION_DEPTH_TOPIC = "/jolgwa/perception/depth/image_raw"
PERCEPTION_FRONT_RANGE_TOPIC = "/jolgwa/perception/front_range"


@dataclass(frozen=True)
class FrameMetadata:
    timestamp_ns: int
    frame_id: str
    width: int
    height: int
    encoding: str


@dataclass(frozen=True)
class DepthResult:
    metadata: FrameMetadata
    front_min_distance_m: float
    valid_sample_count: int


class FrameValidationError(ValueError):
    """Raised when a frame cannot enter the common perception contract."""


def _timestamp_ns(message: Image) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(
        message.header.stamp.nanosec
    )


def validate_frame_metadata(
    message: Image,
    *,
    now_ns: int,
    stale_after_s: float,
    future_tolerance_s: float,
    previous_timestamp_ns: int | None = None,
) -> FrameMetadata:
    """Validate dimensions, source identity, and acquisition time."""

    if stale_after_s <= 0.0 or future_tolerance_s < 0.0:
        raise ValueError("freshness limits are invalid")
    if message.width <= 0 or message.height <= 0:
        raise FrameValidationError("frame dimensions must be positive")
    if not message.header.frame_id.strip():
        raise FrameValidationError("frame_id is required")
    if not message.encoding.strip():
        raise FrameValidationError("encoding is required")

    timestamp_ns = _timestamp_ns(message)
    if timestamp_ns <= 0:
        raise FrameValidationError("timestamp is required")
    if now_ns <= 0:
        raise FrameValidationError("ROS clock is not initialized")
    if (
        previous_timestamp_ns is not None
        and timestamp_ns < previous_timestamp_ns
    ):
        raise FrameValidationError("timestamp regressed")

    age_ns = now_ns - timestamp_ns
    if age_ns > int(stale_after_s * 1_000_000_000):
        raise FrameValidationError("frame is stale")
    if age_ns < -int(future_tolerance_s * 1_000_000_000):
        raise FrameValidationError("timestamp is too far in the future")

    return FrameMetadata(
        timestamp_ns=timestamp_ns,
        frame_id=message.header.frame_id,
        width=int(message.width),
        height=int(message.height),
        encoding=message.encoding,
    )


def _depth_format(encoding: str, is_bigendian: bool) -> tuple[str, int, float]:
    byte_order = ">" if is_bigendian else "<"
    if encoding == "32FC1":
        return byte_order + "f", 4, 1.0
    if encoding == "16UC1":
        return byte_order + "H", 2, 0.001
    raise FrameValidationError(
        f"unsupported depth encoding {encoding!r}; expected 32FC1 metres "
        "or 16UC1 millimetres"
    )


def compute_front_min_distance(
    message: Image,
    *,
    roi_fraction: float,
    min_distance_m: float,
    max_distance_m: float,
) -> tuple[float, int]:
    """Return the minimum finite metric depth in the central forward ROI."""

    if not 0.0 < roi_fraction <= 1.0:
        raise ValueError("roi_fraction must be in (0, 1]")
    if not 0.0 < min_distance_m < max_distance_m:
        raise ValueError("depth range must satisfy 0 < min < max")

    value_format, bytes_per_pixel, unit_scale = _depth_format(
        message.encoding, bool(message.is_bigendian)
    )
    packed_row_size = int(message.width) * bytes_per_pixel
    if message.step < packed_row_size:
        raise FrameValidationError("depth step is smaller than one packed row")

    data = bytes(message.data)
    expected_size = int(message.step) * int(message.height)
    if len(data) != expected_size:
        raise FrameValidationError(
            f"depth data has {len(data)} bytes; expected {expected_size}"
        )

    roi_width = max(1, int(message.width * roi_fraction))
    roi_height = max(1, int(message.height * roi_fraction))
    x_start = (int(message.width) - roi_width) // 2
    y_start = (int(message.height) - roi_height) // 2

    minimum = math.inf
    valid_count = 0
    for y in range(y_start, y_start + roi_height):
        row_offset = y * int(message.step)
        for x in range(x_start, x_start + roi_width):
            offset = row_offset + x * bytes_per_pixel
            raw_value = struct.unpack_from(value_format, data, offset)[0]
            distance_m = float(raw_value) * unit_scale
            if not math.isfinite(distance_m):
                continue
            if distance_m < min_distance_m or distance_m > max_distance_m:
                continue
            valid_count += 1
            minimum = min(minimum, distance_m)

    if valid_count == 0:
        raise FrameValidationError(
            "depth ROI contains no finite in-range samples"
        )
    return minimum, valid_count


class GazeboPerceptionNode(Node):
    """Read-only adapter from Gazebo camera topics to canonical topics."""

    def __init__(self) -> None:
        super().__init__("gazebo_perception_node")
        self.declare_parameter("source_rgb_topic", SOURCE_RGB_TOPIC)
        self.declare_parameter("source_depth_topic", SOURCE_DEPTH_TOPIC)
        self.declare_parameter("output_rgb_topic", PERCEPTION_RGB_TOPIC)
        self.declare_parameter("output_depth_topic", PERCEPTION_DEPTH_TOPIC)
        self.declare_parameter(
            "output_front_range_topic", PERCEPTION_FRONT_RANGE_TOPIC
        )
        self.declare_parameter("stale_after_s", 1.0)
        self.declare_parameter("future_tolerance_s", 0.1)
        self.declare_parameter("front_roi_fraction", 0.6)
        self.declare_parameter("min_distance_m", 0.15)
        self.declare_parameter("max_distance_m", 30.0)
        self.declare_parameter("horizontal_fov_rad", 1.3962634)

        self._stale_after_s = float(self.get_parameter("stale_after_s").value)
        self._future_tolerance_s = float(
            self.get_parameter("future_tolerance_s").value
        )
        self._roi_fraction = float(
            self.get_parameter("front_roi_fraction").value
        )
        self._min_distance_m = float(
            self.get_parameter("min_distance_m").value
        )
        self._max_distance_m = float(
            self.get_parameter("max_distance_m").value
        )
        self._horizontal_fov_rad = float(
            self.get_parameter("horizontal_fov_rad").value
        )
        if not 0.0 < self._horizontal_fov_rad <= math.pi:
            raise ValueError("horizontal_fov_rad must be in (0, pi]")

        # Validate static parameters before subscriptions become active.
        if not 0.0 < self._roi_fraction <= 1.0:
            raise ValueError("front_roi_fraction must be in (0, 1]")
        if not 0.0 < self._min_distance_m < self._max_distance_m:
            raise ValueError("depth range must satisfy 0 < min < max")

        self._last_timestamps: dict[str, int] = {}
        self._rgb_publisher = self.create_publisher(
            Image,
            str(self.get_parameter("output_rgb_topic").value),
            qos_profile_sensor_data,
        )
        self._depth_publisher = self.create_publisher(
            Image,
            str(self.get_parameter("output_depth_topic").value),
            qos_profile_sensor_data,
        )
        self._range_publisher = self.create_publisher(
            Range,
            str(self.get_parameter("output_front_range_topic").value),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("source_rgb_topic").value),
            self._on_rgb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("source_depth_topic").value),
            self._on_depth,
            qos_profile_sensor_data,
        )

    def _validate(self, stream: str, message: Image) -> FrameMetadata:
        metadata = validate_frame_metadata(
            message,
            now_ns=self.get_clock().now().nanoseconds,
            stale_after_s=self._stale_after_s,
            future_tolerance_s=self._future_tolerance_s,
            previous_timestamp_ns=self._last_timestamps.get(stream),
        )
        self._last_timestamps[stream] = metadata.timestamp_ns
        return metadata

    def _on_rgb(self, message: Image) -> None:
        try:
            self._validate("rgb", message)
        except FrameValidationError as exc:
            self.get_logger().warning(f"Rejected Gazebo RGB frame: {exc}")
            return
        self._rgb_publisher.publish(message)

    def _on_depth(self, message: Image) -> None:
        try:
            metadata = self._validate("depth", message)
            front_min, _ = compute_front_min_distance(
                message,
                roi_fraction=self._roi_fraction,
                min_distance_m=self._min_distance_m,
                max_distance_m=self._max_distance_m,
            )
        except FrameValidationError as exc:
            self.get_logger().warning(f"Rejected Gazebo depth frame: {exc}")
            return

        self._depth_publisher.publish(message)
        range_message = Range()
        range_message.header = message.header
        range_message.radiation_type = Range.INFRARED
        range_message.field_of_view = (
            self._horizontal_fov_rad * self._roi_fraction
        )
        range_message.min_range = self._min_distance_m
        range_message.max_range = self._max_distance_m
        range_message.range = front_min
        # The copied header is the common timestamp/frame metadata contract.
        assert metadata.frame_id == range_message.header.frame_id
        self._range_publisher.publish(range_message)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GazeboPerceptionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

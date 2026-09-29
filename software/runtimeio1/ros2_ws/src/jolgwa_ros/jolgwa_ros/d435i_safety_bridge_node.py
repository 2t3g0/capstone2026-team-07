"""Publish fail-closed obstacle decisions from RealSense metric depth."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import rclpy
from jolgwa_interfaces.msg import SafetyDecision
from px4_msgs.msg import VehicleLocalPosition
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from .topic_names import (
    LOCAL_D435I_DEBUG_DECISION,
    OBSTACLE_SAFETY_DECISION,
    PX4_LOCAL_POSITION,
)


LOCAL_D435I_SAFETY_SOURCE = "local-realsense-d435i-v1"


D435I_DEPTH_TOPIC = "/camera/d435i/depth/image"


class DepthFrameError(ValueError):
    """Raised when a depth frame cannot safely be interpreted in metres."""


@dataclass(frozen=True, slots=True)
class NativeDepthConfig:
    trigger_distance_m: float = 3.0
    release_distance_m: float = 4.0
    trigger_frames: int = 3
    min_depth_m: float = 0.15
    max_depth_m: float = 20.0
    min_valid_fraction: float = 0.25
    min_obstacle_fraction: float = 0.05
    near_percentile: float = 10.0
    horizontal_roi_fraction: float = 0.50
    center_vertical_roi: tuple[float, float] = (0.32, 0.68)
    upper_vertical_roi: tuple[float, float] = (0.05, 0.30)
    vertical_speed_mps: float = 0.6
    max_altitude_m: float = 15.0
    altitude_margin_m: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.trigger_distance_m < self.release_distance_m:
            raise ValueError("distance thresholds must satisfy 0 < trigger < release")
        if self.trigger_frames < 1:
            raise ValueError("trigger_frames must be positive")
        if not 0.0 < self.min_depth_m < self.max_depth_m:
            raise ValueError("depth range must satisfy 0 < min < max")
        if not 0.0 < self.min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be in (0, 1]")
        if not 0.0 <= self.min_obstacle_fraction <= 1.0:
            raise ValueError("min_obstacle_fraction must be in [0, 1]")
        if not 0.0 <= self.near_percentile <= 50.0:
            raise ValueError("near_percentile must be in [0, 50]")
        if not 0.0 < self.horizontal_roi_fraction <= 1.0:
            raise ValueError("horizontal_roi_fraction must be in (0, 1]")
        if self.vertical_speed_mps <= 0.0:
            raise ValueError("vertical_speed_mps must be positive")
        if not 0.0 <= self.altitude_margin_m < self.max_altitude_m:
            raise ValueError("altitude limits are invalid")
        for interval in (self.center_vertical_roi, self.upper_vertical_roi):
            if not 0.0 <= interval[0] < interval[1] <= 1.0:
                raise ValueError("ROI intervals must be ordered in [0, 1]")


@dataclass(frozen=True, slots=True)
class CorridorSummary:
    near_distance_m: float
    median_distance_m: float
    valid_fraction: float
    obstacle_fraction: float


@dataclass(frozen=True, slots=True)
class NativeDepthDecision:
    state: str
    direction: str
    velocity_ned_mps: tuple[float, float, float]
    reason: str
    center: CorridorSummary | None = None
    upper: CorridorSummary | None = None


def decode_metric_depth(message: Image) -> np.ndarray:
    """Decode Gazebo 32FC1 metres or RealSense 16UC1 millimetres."""

    width = int(message.width)
    height = int(message.height)
    step = int(message.step)
    if width <= 0 or height <= 0:
        raise DepthFrameError("depth dimensions must be positive")
    if not message.header.frame_id.strip():
        raise DepthFrameError("depth frame_id is required")

    big_endian = bool(message.is_bigendian)
    if message.encoding == "32FC1":
        dtype = np.dtype(">f4" if big_endian else "<f4")
        scale = 1.0
    elif message.encoding == "16UC1":
        dtype = np.dtype(">u2" if big_endian else "<u2")
        scale = 0.001
    else:
        raise DepthFrameError(
            f"unsupported depth encoding {message.encoding!r}; expected "
            "32FC1 metres or 16UC1 millimetres"
        )

    packed_row_bytes = width * dtype.itemsize
    if step < packed_row_bytes:
        raise DepthFrameError("depth step is smaller than one packed row")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    expected_bytes = step * height
    if raw.size != expected_bytes:
        raise DepthFrameError(
            f"depth data has {raw.size} bytes; expected {expected_bytes}"
        )
    packed = np.ascontiguousarray(
        raw.reshape(height, step)[:, :packed_row_bytes]
    )
    values = packed.view(dtype).reshape(height, width).astype(
        np.float32, copy=False
    )
    if scale != 1.0:
        values = values * np.float32(scale)
    return values


def _corridor(
    depth_m: np.ndarray,
    config: NativeDepthConfig,
    vertical_roi: tuple[float, float],
) -> CorridorSummary:
    height, width = depth_m.shape
    y0 = min(height - 1, int(math.floor(height * vertical_roi[0])))
    y1 = max(y0 + 1, int(math.ceil(height * vertical_roi[1])))
    roi_width = max(1, int(round(width * config.horizontal_roi_fraction)))
    x0 = max(0, (width - roi_width) // 2)
    x1 = min(width, x0 + roi_width)
    roi = depth_m[y0:y1, x0:x1]
    # Gazebo 32FC1 uses +inf for rays beyond the configured far clip. Those
    # are valid clear-space observations. RealSense 16UC1 missing pixels are
    # decoded as zero and remain invalid / fail-closed when too numerous.
    finite_valid = (
        np.isfinite(roi) & (roi > 0.0) & (roi <= config.max_depth_m)
    )
    far_clear = np.isposinf(roi)
    valid = finite_valid | far_clear
    samples = np.where(
        far_clear[valid], np.float32(config.max_depth_m), roi[valid]
    )
    valid_fraction = float(samples.size / roi.size)
    if samples.size == 0:
        return CorridorSummary(math.inf, math.inf, valid_fraction, 0.0)
    # Positive sub-MinZ values are treated as dangerously close, not invalid.
    samples = np.maximum(samples, np.float32(config.min_depth_m))
    return CorridorSummary(
        near_distance_m=float(
            np.percentile(samples, config.near_percentile)
        ),
        median_distance_m=float(np.median(samples)),
        valid_fraction=valid_fraction,
        obstacle_fraction=float(
            np.count_nonzero(samples <= config.trigger_distance_m)
            / samples.size
        ),
    )


class NativeDepthSafetyCore:
    """Three-frame 3 m trigger with 4 m hysteresis and upward avoidance."""

    def __init__(self, config: NativeDepthConfig | None = None) -> None:
        self.config = config or NativeDepthConfig()
        self._trigger_count = 0
        self._active = False

    def evaluate(
        self, depth_m: np.ndarray, *, current_altitude_m: float
    ) -> NativeDepthDecision:
        if depth_m.ndim != 2 or depth_m.size == 0:
            return self.stale("depth_shape_invalid")
        if not math.isfinite(current_altitude_m) or current_altitude_m < 0.0:
            return self.stale("altitude_invalid")

        center = _corridor(depth_m, self.config, self.config.center_vertical_roi)
        upper = _corridor(depth_m, self.config, self.config.upper_vertical_roi)
        if center.valid_fraction < self.config.min_valid_fraction:
            return self.stale(
                f"depth_center_invalid:{center.valid_fraction:.3f}"
            )
        if upper.valid_fraction < self.config.min_valid_fraction:
            return self.stale(
                f"depth_upper_invalid:{upper.valid_fraction:.3f}"
            )

        obstacle = (
            center.near_distance_m <= self.config.trigger_distance_m
            and center.obstacle_fraction >= self.config.min_obstacle_fraction
        )
        released = (
            center.near_distance_m >= self.config.release_distance_m
            and center.obstacle_fraction < self.config.min_obstacle_fraction
        )
        distance = center.near_distance_m

        if self._active:
            if released:
                self._active = False
                self._trigger_count = 0
                return NativeDepthDecision(
                    "CLEAR", "FORWARD", (0.0, 0.0, 0.0),
                    f"d435i_front_released;front={distance:.3f}m",
                    center, upper,
                )
            return self._evade(center, upper, current_altitude_m)

        if not obstacle:
            self._trigger_count = 0
            return NativeDepthDecision(
                "CLEAR", "FORWARD", (0.0, 0.0, 0.0),
                f"d435i_front_clear;front={distance:.3f}m",
                center, upper,
            )

        self._trigger_count += 1
        if self._trigger_count < self.config.trigger_frames:
            return NativeDepthDecision(
                "HOLD", "STOP", (0.0, 0.0, 0.0),
                f"d435i_obstacle_confirmation_{self._trigger_count}_of_"
                f"{self.config.trigger_frames};front={distance:.3f}m",
                center, upper,
            )
        self._active = True
        return self._evade(center, upper, current_altitude_m)

    def _evade(
        self,
        center: CorridorSummary,
        upper: CorridorSummary,
        altitude_m: float,
    ) -> NativeDepthDecision:
        if altitude_m >= self.config.max_altitude_m - self.config.altitude_margin_m:
            return NativeDepthDecision(
                "HOLD", "STOP", (0.0, 0.0, 0.0),
                f"d435i_altitude_ceiling;front={center.near_distance_m:.3f}m",
                center, upper,
            )
        upper_clear = (
            upper.near_distance_m >= self.config.release_distance_m
            and upper.obstacle_fraction < self.config.min_obstacle_fraction
        )
        if not upper_clear:
            return NativeDepthDecision(
                "HOLD", "STOP", (0.0, 0.0, 0.0),
                "d435i_upper_corridor_blocked;"
                f"front={center.near_distance_m:.3f}m;"
                f"upper={upper.near_distance_m:.3f}m",
                center, upper,
            )
        return NativeDepthDecision(
            "EVADE", "UP", (0.0, 0.0, -self.config.vertical_speed_mps),
            "d435i_front_obstacle_outdoor_climb_default;"
            f"front={center.near_distance_m:.3f}m;"
            f"upper={upper.near_distance_m:.3f}m",
            center, upper,
        )

    @staticmethod
    def stale(reason: str) -> NativeDepthDecision:
        return NativeDepthDecision(
            "STALE", "STOP", (0.0, 0.0, 0.0), reason
        )


class D435iSafetyBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("d435i_safety_bridge")
        self.declare_parameter("depth_topic", D435I_DEPTH_TOPIC)
        self.declare_parameter("position_timeout_s", 0.5)
        self.declare_parameter("depth_timeout_s", 0.5)
        self.declare_parameter("trigger_distance_m", 3.0)
        self.declare_parameter("release_distance_m", 4.0)
        self.declare_parameter("trigger_frames", 3)
        self.declare_parameter("min_depth_m", 0.15)
        self.declare_parameter("max_depth_m", 20.0)
        self.declare_parameter("min_valid_fraction", 0.25)
        self.declare_parameter("min_obstacle_fraction", 0.05)
        self.declare_parameter("near_percentile", 10.0)
        self.declare_parameter("horizontal_roi_fraction", 0.50)
        self.declare_parameter("vertical_speed_mps", 0.6)
        self.declare_parameter("max_altitude_m", 15.0)
        self.declare_parameter("altitude_margin_m", 0.5)
        # This bridge historically published a diagnostic-only topic.  Moving
        # its output into the flight-control safety path must remain an
        # explicit launch-time choice so an existing debug deployment cannot
        # silently gain control authority after an upgrade.
        self.declare_parameter("authoritative_output", False)

        self._position_timeout_s = float(
            self.get_parameter("position_timeout_s").value
        )
        self._depth_timeout_s = float(
            self.get_parameter("depth_timeout_s").value
        )
        if self._position_timeout_s <= 0.0 or self._depth_timeout_s <= 0.0:
            raise ValueError("sensor timeouts must be positive")
        config = NativeDepthConfig(
            trigger_distance_m=float(
                self.get_parameter("trigger_distance_m").value
            ),
            release_distance_m=float(
                self.get_parameter("release_distance_m").value
            ),
            trigger_frames=int(self.get_parameter("trigger_frames").value),
            min_depth_m=float(self.get_parameter("min_depth_m").value),
            max_depth_m=float(self.get_parameter("max_depth_m").value),
            min_valid_fraction=float(
                self.get_parameter("min_valid_fraction").value
            ),
            min_obstacle_fraction=float(
                self.get_parameter("min_obstacle_fraction").value
            ),
            near_percentile=float(
                self.get_parameter("near_percentile").value
            ),
            horizontal_roi_fraction=float(
                self.get_parameter("horizontal_roi_fraction").value
            ),
            vertical_speed_mps=float(
                self.get_parameter("vertical_speed_mps").value
            ),
            max_altitude_m=float(
                self.get_parameter("max_altitude_m").value
            ),
            altitude_margin_m=float(
                self.get_parameter("altitude_margin_m").value
            ),
        )
        self._core = NativeDepthSafetyCore(config)
        self._authoritative_output = bool(
            self.get_parameter("authoritative_output").value
        )
        output_topic = (
            OBSTACLE_SAFETY_DECISION
            if self._authoritative_output
            else LOCAL_D435I_DEBUG_DECISION
        )
        self._publisher = self.create_publisher(
            SafetyDecision, output_topic, qos_profile_sensor_data
        )
        if self._authoritative_output:
            self.get_logger().warning(
                "LOCAL D435 metric-depth safety output is authoritative; "
                "fresh depth and PX4 position are mandatory"
            )
        self.create_subscription(
            VehicleLocalPosition,
            PX4_LOCAL_POSITION,
            self._on_local_position,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("depth_topic").value),
            self._on_depth,
            qos_profile_sensor_data,
        )
        self.create_timer(0.1, self._watchdog)
        self._z_ned_m = math.nan
        self._position_received_at = float("-inf")
        self._depth_received_at = float("-inf")
        self._sequence = 0
        self._last_stale_reason = ""

    def _on_local_position(self, message: VehicleLocalPosition) -> None:
        if not bool(message.z_valid):
            return
        z_ned_m = float(message.z)
        if math.isfinite(z_ned_m):
            self._z_ned_m = z_ned_m
            self._position_received_at = time.monotonic()

    def _on_depth(self, message: Image) -> None:
        now = time.monotonic()
        self._depth_received_at = now
        if now - self._position_received_at > self._position_timeout_s:
            self._publish(self._core.stale("px4_position_stale"))
            return
        try:
            depth_m = decode_metric_depth(message)
            decision = self._core.evaluate(
                depth_m, current_altitude_m=max(0.0, -self._z_ned_m)
            )
        except DepthFrameError as exc:
            decision = self._core.stale(f"d435i_depth_invalid:{exc}")
        self._publish(decision)

    def _watchdog(self) -> None:
        now = time.monotonic()
        if now - self._position_received_at > self._position_timeout_s:
            self._publish_stale_once("px4_position_stale")
        elif now - self._depth_received_at > self._depth_timeout_s:
            self._publish_stale_once("d435i_depth_stale")

    def _publish_stale_once(self, reason: str) -> None:
        if reason == self._last_stale_reason:
            return
        self._publish(self._core.stale(reason))

    def _publish(self, decision: NativeDepthDecision) -> None:
        state_mapping = {
            "CLEAR": SafetyDecision.STATE_CLEAR,
            "HOLD": SafetyDecision.STATE_HOLD,
            "EVADE": SafetyDecision.STATE_EVADE,
            "STALE": SafetyDecision.STATE_STALE,
        }
        message = SafetyDecision()
        message.stamp = self.get_clock().now().to_msg()
        message.source = (
            LOCAL_D435I_SAFETY_SOURCE
            if self._authoritative_output
            else "local-realsense-d435i-debug"
        )
        self._sequence = (self._sequence + 1) & 0xFFFFFFFF
        message.sequence = self._sequence
        # Evaluation runs synchronously in the depth callback.  Watchdog
        # publications are explicitly STALE, so zero here cannot turn an old
        # frame into CLEAR.
        message.observation_age_s = 0.0
        message.state = state_mapping[decision.state]
        message.confidence = 0.0 if decision.state == "STALE" else 1.0
        message.velocity_ned_mps = [
            float(value) for value in decision.velocity_ned_mps
        ]
        message.max_speed_mps = float(
            math.sqrt(sum(value * value for value in decision.velocity_ned_mps))
        )
        message.reason = decision.reason
        self._last_stale_reason = (
            decision.reason if decision.state == "STALE" else ""
        )
        self._publisher.publish(message)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = D435iSafetyBridgeNode()
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

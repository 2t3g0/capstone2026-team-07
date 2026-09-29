from __future__ import annotations

import time
import uuid
from pathlib import Path

import rclpy
from jolgwa_interfaces.msg import EventObservation
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from .event_control import PHASE1_EVENT_TYPES
from .phase1_bridge import (
    JsonlTail,
    latest_phase1_run,
    normalized_bbox,
    select_target_bbox,
)
from .topic_names import EVENT_OBSERVATION


class Phase1EventBridgeNode(Node):
    """Publish a live ROS observation stream from Phase 1 demo artifacts."""

    def __init__(self) -> None:
        super().__init__("phase1_event_bridge")
        self.declare_parameter(
            "run_root", "runs/vision/phase1-demo"
        )
        self.declare_parameter("capture_duration_s", 5.0)
        self.declare_parameter("source", "phase1-demo-local")
        self._run_root = Path(str(self.get_parameter("run_root").value))
        self._duration_s = float(
            self.get_parameter("capture_duration_s").value
        )
        self._source = str(self.get_parameter("source").value)
        self._run_dir: Path | None = None
        self._events: JsonlTail | None = None
        self._observations: JsonlTail | None = None
        self._active_event: dict | None = None
        self._active_until = 0.0
        self._last_bbox = None
        self._publisher = self.create_publisher(
            EventObservation, EVENT_OBSERVATION, qos_profile_sensor_data
        )
        self.create_timer(0.05, self._poll)

    def _poll(self) -> None:
        run_dir = latest_phase1_run(self._run_root)
        if run_dir != self._run_dir:
            self._run_dir = run_dir
            self._events = (
                JsonlTail(run_dir / "events.jsonl", start_at_end=True)
                if run_dir
                else None
            )
            self._observations = (
                JsonlTail(
                    run_dir / "observations.jsonl", start_at_end=True
                )
                if run_dir
                else None
            )
            self._active_event = None
            self.get_logger().info(
                f"following Phase 1 run: {run_dir or 'none'}"
            )
        if self._events is None:
            return

        for event in self._events.read():
            if (
                event.get("state") != "CONFIRMED"
                or event.get("event_type") not in PHASE1_EVENT_TYPES
            ):
                continue
            event.setdefault("event_id", str(uuid.uuid4()))
            bbox = normalized_bbox(event.get("bbox"))
            if bbox is None:
                bbox = (0.0, 0.0, 1.0, 1.0)
            self._publish(event, bbox, visible=True, state="CONFIRMED")
            if self._active_event is None or time.monotonic() >= self._active_until:
                self._active_event = event
                self._active_until = time.monotonic() + self._duration_s
                self._last_bbox = bbox

        if self._active_event is None:
            return
        if time.monotonic() >= self._active_until:
            self._active_event = None
            self._last_bbox = None
            return
        if self._observations is None:
            return
        for observation in self._observations.read():
            if float(observation.get("timestamp_s", -1.0)) < float(
                self._active_event.get("timestamp_s", -1.0)
            ):
                continue
            bbox = select_target_bbox(
                self._active_event,
                observation.get("detections", []),
                previous_bbox=self._last_bbox,
            )
            if bbox is not None:
                self._last_bbox = bbox
            self._publish(
                self._active_event,
                bbox or (0.0, 0.0, 0.0, 0.0),
                visible=bbox is not None,
                state="ACTIVE",
            )

    def _publish(self, event, bbox, *, visible: bool, state: str) -> None:
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
        message.source = self._source
        self._publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Phase1EventBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

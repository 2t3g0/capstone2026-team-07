"""Reuse the v2 photo bench and publish only new, exact model evidence."""
import json
import os
import threading
import time
from pathlib import Path
from http.server import ThreadingHTTPServer
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from jolgwa_interfaces.msg import SafetyDecision
from std_msgs.msg import String
from .scenario_incident_bench import Bench, handler_for
from .scenario_runtime import INCIDENT_TOPIC
from .topic_names import FORWARD_CAMERA_COMPRESSED, OBSTACLE_SAFETY_DECISION
from .scenario_observe_view import CameraObservation, handler_with_observation
from .evidence_clock import local_evidence_clock_id


class ScenarioIncidentNode(Node):
    def __init__(self):
        super().__init__("scenario_incident")
        self.declare_parameter("phase1_root", "/home/jetson/jolgwa-models/phase1-demo-local-v2-20260912")
        self.declare_parameter("output_root", "~/outputs/scenario_incidents")
        self.declare_parameter("device", "0")
        self.declare_parameter("camera_topic", "/camera/camera/color/image_raw/compressed")
        self.declare_parameter("depth_topic", "/camera/camera/depth/image_rect_raw")
        self.observation = CameraObservation()
        self.last_observe_depth_at = float('-inf')
        self.last_observe_rgb_at = float('-inf')
        self.bench = Bench(Path(self.get_parameter("output_root").value).expanduser(), cooldown_s=0.0)
        self.publisher = self.create_publisher(String, INCIDENT_TOPIC, 10)
        self.sent = set()
        self.create_subscription(CompressedImage, self.get_parameter("camera_topic").value,
                                 self.receive, qos_profile_sensor_data)
        self.create_subscription(Image, self.get_parameter("depth_topic").value,
                                 self.receive_depth, qos_profile_sensor_data)
        self.create_subscription(SafetyDecision, OBSTACLE_SAFETY_DECISION,
                                 self.receive_safety, qos_profile_sensor_data)
        self.create_timer(.05, self.publish_new)
        self.worker = threading.Thread(target=self.bench.worker,
            args=(self.get_parameter("phase1_root").value, self.get_parameter("device").value), daemon=True)
        self.worker.start()
        # Existing read-only viewer endpoints; no camera/start-flight interlock.
        handler = handler_with_observation(handler_for(self.bench), self.observation, self.bench)
        self.server = ThreadingHTTPServer(("127.0.0.1", 8878), handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        # During a viewer-only hot restart, retain the original launch lifetime.
        # Normal launch-owned nodes need no extra lifetime guard.
        self.launch_parent = os.environ.get("JOLGWA_OBSERVE_LAUNCH_PID", "")
        self.launch_start = os.environ.get("JOLGWA_OBSERVE_LAUNCH_START", "")
        if self.launch_parent:
            self.create_timer(1., self.check_launch_parent)

    def check_launch_parent(self):
        try:
            stat = Path('/proc', self.launch_parent, 'stat').read_text()
            same_launch = stat.rsplit(')', 1)[1].split()[19] == self.launch_start
        except (OSError, IndexError):
            same_launch = False
        if not same_launch and rclpy.ok():
            rclpy.shutdown()

    def receive(self, message):
        stamp = message.header.stamp.sec*10**9+message.header.stamp.nanosec
        delay = (self.get_clock().now().nanoseconds-stamp)/1e9
        now = time.monotonic()
        if now - self.last_observe_rgb_at >= .1:
            self.last_observe_rgb_at = now
            self.observation.rgb(message, delay if stamp > 0 else float('inf'))
        if stamp <= 0 or not 0 <= delay <= .5:
            return
        self.bench.submit(bytes(message.data), stamp)

    def receive_depth(self, message):
        now = time.monotonic()
        if now - self.last_observe_depth_at < .1:
            return
        self.last_observe_depth_at = now
        stamp = message.header.stamp.sec*10**9+message.header.stamp.nanosec
        age = (self.get_clock().now().nanoseconds-stamp)/1e9 if stamp > 0 else float('inf')
        self.observation.depth(message, age)

    def receive_safety(self, message):
        stamp = message.stamp.sec*10**9+message.stamp.nanosec
        age = (self.get_clock().now().nanoseconds-stamp)/1e9 if stamp > 0 else float('inf')
        self.observation.safety(message, age)

    def publish_new(self):
        with self.bench.lock:
            events = [dict(event) for event in self.bench.events]
            model_hash = self.bench.inference.get("model_hashes", {}).get("abnormal_behavior", "")
        for event in events:
            if event["event_id"] in self.sent:
                continue
            self.sent.add(event["event_id"])
            if len(self.sent) > 4096:
                continue
            event["clock_id"] = local_evidence_clock_id()
            event["model_sha256"] = model_hash
            # The unchanged bench records callback receipt. Retain the known
            # ROS publication age so queued images cannot become new evidence.
            stamp = event.get("source_timestamp_ns", 0)
            age = (self.get_clock().now().nanoseconds-stamp)/1e9
            if stamp <= 0 or age < 0:
                continue
            event["camera_received_monotonic_s"] = min(
                event["camera_received_monotonic_s"], time.monotonic()-age)
            message = String()
            message.data = json.dumps(event, allow_nan=False)
            self.publisher.publish(message)

    def destroy_node(self):
        self.server.shutdown()
        self.server.server_close()
        self.bench.stop.set()
        self.worker.join(timeout=2)
        if not self.worker.is_alive():
            self.bench.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ScenarioIncidentNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

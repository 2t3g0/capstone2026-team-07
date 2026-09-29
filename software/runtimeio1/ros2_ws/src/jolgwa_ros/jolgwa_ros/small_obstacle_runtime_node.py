"""Physical limited demo runtime; boots observe, one USB-owning worker only."""
import argparse
import json
from pathlib import Path
import time
import uuid

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data
from rcl_interfaces.msg import ParameterDescriptor
from std_msgs.msg import String
from px4_msgs.msg import VehicleLocalPosition, VehicleAttitude, VehicleAngularVelocity

from jolgwa_uav.observer_fc_telemetry import TOPIC_PREFIX
from jolgwa_uav.observer_fc_timesync import ObserveTimesync
from jolgwa_uav.small_obstacle_worker import SmallObstacleWorker
from jolgwa_uav.small_obstacle_perception import DEMO_TOPIC
from jolgwa_uav.small_obstacle_control import DemoControlServer
from .observer_usb_fc_node import ObserverUsbFcNode
from .evidence_clock import local_evidence_clock_id


class SmallObstacleRuntimeNode(ObserverUsbFcNode):
    def __init__(self, *, mode="observe", control_socket=None):
        if mode != "observe":
            raise ValueError("auto_demo_boot_prohibited_use_later_explicit_request")
        # Do not call ObserverUsbFcNode.__init__: it starts a different USB owner.
        # Inherit its proven report/poll/publication/dependency-expiry path only.
        Node.__init__(self, "small_obstacle_runtime")
        readonly = ParameterDescriptor(read_only=True)
        for name, default in (("mode", "observe"), ("observe_timesync_enabled", False),
                ("observe_timesync_protocol", "targeted_v2"),
                ("observe_timesync_legacy_exclusive_link_attested", False),
                ("observe_timesync_expected_flight_sw_version", 0),
                ("observe_timesync_expected_custom_version_hex", ""),
                ("demo_journal_dir", "outputs/small_obstacle_demo")):
            self.declare_parameter(name, default, readonly)
        if self.get_parameter("mode").value != "observe":
            raise ValueError("auto_demo_boot_prohibited")
        timing = ObserveTimesync(
            enabled=self.get_parameter("observe_timesync_enabled").value,
            protocol=self.get_parameter("observe_timesync_protocol").value,
            legacy_exclusive_link_attested=self.get_parameter("observe_timesync_legacy_exclusive_link_attested").value,
            expected_flight_sw_version=self.get_parameter("observe_timesync_expected_flight_sw_version").value,
            expected_custom_version_hex=self.get_parameter("observe_timesync_expected_custom_version_hex").value)
        if not timing.enabled:
            raise ValueError("explicit_bounded_timesync_required_no_legacy_fallback")
        self.timing_clock_id = local_evidence_clock_id()
        if not self.timing_clock_id:
            raise ValueError("local_monotonic_clock_identity_required")
        self.state = self.transport = self.journal = None
        self.worker = None
        self._next_worker_status_ns = self._next_demo_status_ns = 0
        self.positions = self.create_publisher(VehicleLocalPosition, TOPIC_PREFIX+"/vehicle_local_position", qos_profile_sensor_data)
        self.attitudes = self.create_publisher(VehicleAttitude, TOPIC_PREFIX+"/vehicle_attitude", qos_profile_sensor_data)
        self.rates = self.create_publisher(VehicleAngularVelocity, TOPIC_PREFIX+"/vehicle_angular_velocity", qos_profile_sensor_data)
        self.status = self.create_publisher(String, TOPIC_PREFIX+"/status", 1)
        self.timing_publisher = self.create_publisher(String, TOPIC_PREFIX+"/timing_evidence", qos_profile_sensor_data)
        self.demo_status = self.create_publisher(String, "/jolgwa/demo/status", 1)
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(String, DEMO_TOPIC, self._on_demo_perception, qos)
        directory = Path(str(self.get_parameter("demo_journal_dir").value))
        path = directory/("demo_fc_"+uuid.uuid4().hex+".jsonl")
        self.worker = SmallObstacleWorker(timing=timing, journal_path=path,
            evidence_clock_id=self.timing_clock_id, mode="observe")
        self.create_timer(.01, self.poll)
        # Claim the private control endpoint before opening the exclusive USB.
        self.control_server = DemoControlServer(self.worker, socket_path=control_socket)
        self.control_server.start()
        try:
            self.worker.start()
        except BaseException:
            self.control_server.close()
            raise

    def _on_demo_perception(self, message):
        if self.worker is not None:
            self.worker.submit_perception(message.data)

    def poll(self):
        super().poll()  # exact original private pose/evidence publishing checks
        now = time.monotonic_ns()
        if now >= self._next_demo_status_ns:
            state = self.worker.last_demo_status()
            # Diagnostic only. Never refresh its worker source timestamp.
            message = String()
            message.data = json.dumps(state, allow_nan=False)
            self.demo_status.publish(message)
            self._next_demo_status_ns = now+100_000_000

    def destroy_node(self):
        server = getattr(self, "control_server", None)
        if server is not None:
            server.close()
        return super().destroy_node()


def main(args=None):
    parser = argparse.ArgumentParser(description="Observe-first physical small-obstacle runtime")
    parser.add_argument("--mode", choices=("observe",), default="observe")
    parser.add_argument("--control-socket", type=Path)
    selected, ros_args = parser.parse_known_args(args)
    rclpy.init(args=ros_args)
    node = SmallObstacleRuntimeNode(mode=selected.mode, control_socket=selected.control_socket)
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

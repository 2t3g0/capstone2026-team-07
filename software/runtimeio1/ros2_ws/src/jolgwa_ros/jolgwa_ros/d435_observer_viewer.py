"""Same-host text display, with its own steady-clock expiry watchdog."""
import time
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from std_msgs.msg import String
from .sensor_observer import OBSERVER_TOPIC, ObserverReportLease


class D435ObserverViewer(Node):
    def __init__(self):
        super().__init__("d435_observer_viewer")
        self._lease = ObserverReportLease()
        self._last_signature = None
        self._last_log_at = float("-inf")
        self.create_subscription(String, OBSERVER_TOPIC, self._receive, 1)
        self._steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(0.1, self._tick, clock=self._steady_clock)

    def _receive(self, message):
        self._lease.receive(message.data, now=time.monotonic())

    def _tick(self):
        now = time.monotonic()
        report = self._lease.current(now=now)
        signature = report["assessment"], report["reason"]
        if signature != self._last_signature or now - self._last_log_at >= 1.0:
            self.get_logger().info(f"[DISPLAY ONLY] {signature[0]} | {signature[1]}")
            self._last_signature, self._last_log_at = signature, now


def main(args=None):
    rclpy.init(args=args)
    node = D435ObserverViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

"""Loopback-only GET status API. No FC publishers, services, or action clients."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import threading
import time
from urllib.parse import parse_qs, urlparse
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from px4_msgs.msg import VehicleLocalPosition
from .observer_status import StatusStore
from .sensor_observer import OBSERVER_TOPIC
from .evidence_clock import local_evidence_clock_id
from jolgwa_uav.observer_radio_bridge import RADIO_STATUS_TOPIC, make_radio_envelope


class D435StatusServer(Node):
    def __init__(self):
        super().__init__("d435_readonly_status")
        self.store, self.lock = StatusStore(), threading.RLock()
        self.last_stats_at = float("-inf")
        self.pose_stamp = self.rgb_stamp = 0
        self.fc_diagnostic = None
        self.fc_at = float('-inf')
        self.create_subscription(String, OBSERVER_TOPIC, self.report, 1)
        self.create_subscription(Image, "/camera/camera/depth/image_rect_raw", self.depth, qos_profile_sensor_data)
        self.create_subscription(CompressedImage, "/camera/camera/color/image_raw/compressed", self.rgb, qos_profile_sensor_data)
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position", self.pose, qos_profile_sensor_data)
        self.create_subscription(String, '/jolgwa/observer/fc/status', self.fc_status, 1)
        self.declare_parameter('observer_radio_enabled', False, ParameterDescriptor(read_only=True))
        self.radio_publisher = None
        if self.get_parameter('observer_radio_enabled').value is True:
            self.radio_clock_id = local_evidence_clock_id()
            if not self.radio_clock_id:
                raise ValueError('radio status requires same-host monotonic clock identity')
            self.radio_publisher = self.create_publisher(String, RADIO_STATUS_TOPIC, 1)
            # Local latest-only snapshots at 10 Hz avoid two 2 Hz timers
            # phase-locking to expired 250 ms pose evidence. Air TX stays 2 Hz.
            self.create_timer(.1, self.publish_radio)

    def publish_radio(self):
        # This is a private display topic, not /fmu/in or SafetyDecision. Keep
        # original generation time so a slow DDS callback never renews a lease.
        if self.radio_publisher is None:
            return
        with self.lock:
            generated_s = time.monotonic()
            envelope = make_radio_envelope(self.snapshot(generated_s),
                clock_id=self.radio_clock_id, generated_s=generated_s)
        message = String()
        message.data = json.dumps(envelope, allow_nan=False)
        self.radio_publisher.publish(message)

    def fc_status(self, msg):
        try:
            value = json.loads(msg.data)
            if (value.get('mode') != 'OBSERVE_ONLY' or value.get('flight_commands_enabled') is not False
                    or value.get('source') != 'physical_px4_usb_mavlink'
                    or not isinstance(value.get('reason'), str)):
                return
            json.dumps(value, allow_nan=False)
        except (ValueError, TypeError, AttributeError):
            return
        with self.lock:
            self.fc_diagnostic = value
            self.fc_at = time.monotonic()

    def snapshot(self, now):
        value = self.store.snapshot(now)
        if self.fc_diagnostic is not None and 0 <= now-self.fc_at < .25:
            value['fc'] = self.fc_diagnostic
            if not self.store.pose_at > now-.25 and value['report']['assessment'] == 'UNKNOWN':
                value['report']['reason'] += '; fc: ' + self.fc_diagnostic['reason']
        else:
            value['fc'] = {'source':'physical_px4_usb_mavlink','reason':'fc_diagnostic_missing_or_stale'}
        return value

    def report(self, msg):
        with self.lock:
            self.store.lease.receive(msg.data, now=time.monotonic())

    def depth(self, msg):
        now = time.monotonic()
        if now-self.last_stats_at >= .1:
            with self.lock:
                self.store.depth(msg, now)
            self.last_stats_at = now

    def rgb(self, msg):
        with self.lock:
            self.store.rgb(msg, time.monotonic())

    def pose(self, msg):
        with self.lock:
            if (msg.timestamp > self.pose_stamp and msg.xy_valid and msg.z_valid
                    and all(math.isfinite(x) for x in (msg.x,msg.y,msg.z))):
                self.store.pose_at = time.monotonic()
                self.pose_stamp = msg.timestamp
            else:
                self.store.pose_at = float("-inf")


def main(args=None):
    rclpy.init(args=args)
    node = D435StatusServer()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            nonce = parse_qs(parsed.query).get("nonce", [""])[0]
            if parsed.path != "/status" or len(nonce)>128:
                self.send_error(404)
                return
            with node.lock:
                value = node.snapshot(time.monotonic())
            value["nonce"] = nonce
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1",8877), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

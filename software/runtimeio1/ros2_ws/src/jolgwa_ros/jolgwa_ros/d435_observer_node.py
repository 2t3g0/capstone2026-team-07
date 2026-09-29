"""Manual-flight observer: sensor subscriptions and diagnostic output ONLY."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid
from urllib.request import urlopen

import rclpy
from jolgwa_interfaces.msg import SafetyDecision
from std_msgs.msg import String

from .jetson_d435i_bridge_node import JetsonD435iBridgeNode, _stamp_ns
from .sensor_observer import OBSERVER_TOPIC, make_report, unknown_report
from jolgwa_uav.observer_jpeg import validate_observer_jpeg


class D435ObserverNode(JetsonD435iBridgeNode):
    """Reuses synchronization and inference, not the flight-output publishers."""

    def __init__(self):
        self._session = str(uuid.uuid4())
        self._journal = None
        self._journal_error = False
        self._last_report_log_at = float("-inf")
        self._last_report_signature = None
        self._sensor_stamps = {"rgb": 0, "depth": 0}
        self._sensor_faults = {}
        super().__init__(node_name="d435_observer", default_jetson_url="http://127.0.0.1:8766")
        # Not ROS parameters: neither command-line nor dynamic parameter
        # changes can switch this executable into an active controller.
        self._phase1_enabled = False
        self._geometry_required = True
        self._require_angular_velocity = True
        self._position_timeout_s = min(self._position_timeout_s, 0.25)
        self.declare_parameter("observer_log_dir", "outputs/d435_observer")
        log_dir = str(self.get_parameter("observer_log_dir").value).strip()
        try:
            if not log_dir:
                raise ValueError("observer_log_dir must not be empty")
            directory = Path(log_dir).expanduser().resolve()
            directory.mkdir(parents=True, exist_ok=True)
            self._journal_path = directory / ("observer_" + self._session + ".jsonl")
            self._journal = self._journal_path.open("x", encoding="utf-8", buffering=1)
        except (OSError, ValueError):
            super().destroy_node()
            raise
        self.get_logger().warning(
            "OBSERVE_ONLY: no SafetyDecision, event, or FC-command output. "
            f"Report={OBSERVER_TOPIC}; journal={self._journal_path}. "
            "CLEAR is a local observation, not permission to fly."
        )
        self._emit_report(unknown_report("waiting_for_fresh_calibrated_sensors"))

    def _create_output_publishers(self):
        # No SafetyDecision/EventObservation publisher is even constructed.
        self._safety_publisher = None
        self._event_publisher = None
        self._observer_publisher = self.create_publisher(String, OBSERVER_TOPIC, 10)

    def _safety_request_headers(self):
        return {
            "Content-Type": "application/octet-stream",
            "X-Jolgwa-Observer-Session": self._session,
        }

    def _accept_frame(self, name, message):
        stamp = _stamp_ns(message)
        if stamp <= self._sensor_stamps[name]:
            reason = name + "_timestamp_not_advancing_restart_on_clock_reset"
            self._sensor_faults[name] = reason
            with self._lock:
                (self._depth_frames if name == "depth" else self._color_frames).clear()
                self._completed_safety_result = None
                if name == "depth":
                    self._depth_received_at = float("-inf")
            self._publish_stale_once(reason)
            return False
        self._sensor_stamps[name] = stamp
        self._sensor_faults.pop(name, None)
        return True

    def _on_depth(self, message):
        if message.encoding != "16UC1":
            self._sensor_faults["depth"] = "observer_requires_raw_d435_16UC1"
            self._publish_stale_once(self._sensor_faults["depth"])
            return
        if self._accept_frame("depth", message):
            super()._on_depth(message)

    def _on_color(self, message):
        try:
            validate_observer_jpeg(message.data, message.format)
        except (ValueError, TypeError, BufferError, OverflowError) as exc:
            self._sensor_faults["rgb"] = str(exc)
            with self._lock:
                self._color_frames.clear()
                self._completed_safety_result = None
            self._publish_stale_once(self._sensor_faults["rgb"])
            return
        if self._accept_frame("rgb", message):
            super()._on_color(message)

    def _request_safety_jetson(self, color, depth, **kwargs):
        if depth.encoding != "16UC1":
            raise ValueError("observer_requires_raw_d435_16UC1")
        # Check BEFORE any stateful POST, including against old backends which
        # do not understand/reject the observer session header yet.
        started = time.monotonic()
        with urlopen(self._jetson_url + "/health", timeout=min(0.25, self._request_timeout_s)) as response:
            health = json.loads(response.read().decode("utf-8"))
        if not isinstance(health, dict) or health.get("observation_only") is not True:
            raise ValueError("observer_requires_dedicated_updated_backend")
        elapsed = time.monotonic() - started
        for key in ("frame_age_s", "rgb_age_s", "depth_age_s", "pose_age_s"):
            if kwargs.get(key) is not None:
                kwargs[key] += elapsed
        payload = super()._request_safety_jetson(color, depth, **kwargs)
        if payload.get("observation_only") is not True:
            raise ValueError("observer_requires_dedicated_updated_backend")
        payload["safety"]["observer_input"] = {
            "rgb_timestamp_ns": _stamp_ns(color),
            "depth_timestamp_ns": _stamp_ns(depth),
            "depth_frame_id": str(depth.header.frame_id),
            "depth_width": int(depth.width), "depth_height": int(depth.height),
            "depth_encoding": str(depth.encoding),
            "position_ned_m": [kwargs.get(key) for key in ("x_ned_m", "y_ned_m", "z_ned_m")],
            "heading_rad": kwargs.get("heading_rad"),
            "angular_rate_rad_s": kwargs.get("angular_rate_rad_s"),
            "pair_skew_s": abs(_stamp_ns(color) - _stamp_ns(depth)) / 1e9,
        }
        return payload

    def _publish_safety(self, payload, yaw_rad):
        report = (unknown_report(";".join(self._sensor_faults.values())) if self._sensor_faults
                  else make_report(payload, timeout_s=self._sensor_timeout_s))
        report["input"] = payload.get("observer_input", {})
        self._last_stale_reason = report["reason"] if report["assessment"] == "UNKNOWN" else ""
        self._emit_report(report)

    def _publish(self, *, state, reason, **kwargs):
        # The base watchdog uses _publish for STALE only. Unexpected calls must
        # fail closed, never fall through to the production publisher.
        if state != SafetyDecision.STATE_STALE:
            raise RuntimeError("observer_cannot_publish_flight_safety")
        self._last_stale_reason = reason
        self._emit_report(unknown_report(reason))

    def _publish_event(self, *args, **kwargs):
        raise RuntimeError("observer_cannot_publish_events")

    def _request_phase1_jetson(self, *args, **kwargs):
        raise RuntimeError("observer_cannot_request_phase1")

    def _emit_report(self, report):
        now = time.monotonic()
        self._sequence += 1
        report.update({
            "sequence": self._sequence, "session": self._session,
            "utc": datetime.now(timezone.utc).isoformat(),
            "published_monotonic_s": now,
            "sensor": "D435 (no camera IMU); PX4 pose/angular velocity",
            "scope": "current forward view; not a full route or swept-volume proof",
            "time_basis": "ROS receipt-time approximation, not hardware capture latency",
            "configuration": {"geometry_calibrated": self._geometry_calibrated,
                              "camera_mount": self._camera_mount,
                              "sensor_timeout_s": self._sensor_timeout_s,
                              "position_timeout_s": self._position_timeout_s},
        })
        if self._journal_error:
            report.update(assessment="UNKNOWN", reason="observer_journal_unavailable", valid_for_s=0.0,
                          climb_needed=None, next_observation_step_m=None)
        serialized = json.dumps(report, ensure_ascii=False, allow_nan=False)
        if self._journal is not None and not self._journal_error:
            try:
                self._journal.write(serialized + "\n")
            except OSError:
                self._journal_error = True
                report.update(assessment="UNKNOWN", reason="observer_journal_unavailable", valid_for_s=0.0,
                              climb_needed=None, next_observation_step_m=None)
                serialized = json.dumps(report, ensure_ascii=False, allow_nan=False)
                self.get_logger().error("Observer journal failed; diagnostic output is UNKNOWN")
        message = String()
        if report["assessment"] != "UNKNOWN" and time.monotonic() >= now + report["valid_for_s"]:
            report.update(assessment="UNKNOWN", reason="observer_output_deadline_exceeded", valid_for_s=0.0,
                          climb_needed=None, next_observation_step_m=None)
            serialized = json.dumps(report, ensure_ascii=False, allow_nan=False)
        message.data = serialized
        self._observer_publisher.publish(message)
        signature = (report["assessment"], report["reason"])
        if signature != self._last_report_signature or now - self._last_report_log_at >= 1.0:
            metrics = report["metrics"]
            self.get_logger().info(
                f"[OBSERVE_ONLY] {report['assessment']} | front={metrics.get('front_distance_m')} m "
                f"upper_ROI={metrics.get('upper_clearance_m')} m | {report['reason']}"
            )
            self._last_report_signature = signature
            self._last_report_log_at = now

    def destroy_node(self):
        try:
            if self._journal is not None:
                self._emit_report(unknown_report("observer_stopped"))
                self._journal.close()
                self._journal = None
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = D435ObserverNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

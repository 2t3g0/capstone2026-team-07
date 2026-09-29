"""Separate native D435 demo perception producer. Never publishes FC commands."""
from collections import OrderedDict
from dataclasses import replace
import json
import math
from pathlib import Path
import threading
import time
import uuid
from urllib.parse import urlparse
from urllib.request import urlopen

import rclpy
from std_msgs.msg import String
from rclpy.qos import qos_profile_sensor_data

from .d435_observer_node import D435ObserverNode
from .jetson_d435i_bridge_node import JetsonD435iBridgeNode, _stamp_ns, _px4_timestamp_us
from .evidence_clock import local_evidence_clock_id
from jolgwa_uav.small_obstacle_perception import (BackendHealthCache, DEMO_TOPIC,
    OriginalPoseBindings, PerceptionEnvelope, NS)


class D435DemoPerceptionNode(D435ObserverNode):
    def __init__(self):
        self._session = str(uuid.uuid4())
        self._journal = None
        self._journal_error = False
        self._sensor_stamps = {"rgb": 0, "depth": 0}
        self._sensor_faults = {}
        self._demo_clock = local_evidence_clock_id()
        if not self._demo_clock:
            raise ValueError("demo_perception_requires_same_host_linux_clock")
        self._demo_proofs = OriginalPoseBindings(self._demo_clock)
        self._demo_envelope = PerceptionEnvelope(self._demo_clock, self._session)
        self._demo_health = BackendHealthCache()
        self._demo_stop = threading.Event()
        self._demo_thread = None
        self._demo_context = threading.local()
        self._demo_rates, self._demo_rate_lock = OrderedDict(), threading.Lock()
        self._last_demo_output = float("-inf")
        self._last_demo_reason = ""
        # Reuse the real synchronization/HTTP pipeline without changing the
        # original observer's executable, output topic, or default behavior.
        JetsonD435iBridgeNode.__init__(self, node_name="d435_demo_perception",
                                     default_jetson_url="http://127.0.0.1:8768")
        self._phase1_enabled = False
        self._geometry_required = True
        self._require_angular_velocity = True
        self._sensor_timeout_s = min(self._sensor_timeout_s, .5)
        self._position_timeout_s = min(self._position_timeout_s, .25)
        self._max_pair_skew_s = min(self._max_pair_skew_s, .075)
        self.declare_parameter("observer_log_dir", "outputs/d435_demo_perception")
        directory = Path(str(self.get_parameter("observer_log_dir").value)).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        self._journal = (directory/("demo_perception_"+self._session+".jsonl")).open("x", encoding="utf-8", buffering=1)
        parsed = urlparse(self._jetson_url)
        if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost")
                or parsed.port in (None, 8766) or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ("", "/")):
            self.destroy_node()
            raise ValueError("dedicated_local_backend_required_do_not_share_live_observer_8766")
        self.create_subscription(String, "/jolgwa/observer/fc/timing_evidence", self._on_demo_timing, qos_profile_sensor_data)
        self._demo_thread = threading.Thread(target=self._health_loop, name="demo-native-health", daemon=True)
        self._demo_thread.start()
        self._invalid("waiting_for_native_demo_contract_and_original_inputs")

    def create_subscription(self, message_type, topic, callback, qos_profile, **kwargs):
        # No simulator DDS inputs or odometry substitute are silently selected.
        aliases = {"/fmu/out/vehicle_local_position": "vehicle_local_position",
                   "/fmu/out/vehicle_local_position_v1": "vehicle_local_position",
                   "/fmu/out/vehicle_attitude": "vehicle_attitude",
                   "/fmu/out/vehicle_angular_velocity": "vehicle_angular_velocity",
                   "/fmu/out/vehicle_odometry": "vehicle_odometry"}
        if topic in aliases:
            topic = "/jolgwa/observer/fc/"+aliases[topic]
        return super().create_subscription(message_type, topic, callback, qos_profile, **kwargs)

    def _create_output_publishers(self):
        self._safety_publisher = self._event_publisher = None
        self._demo_publisher = self.create_publisher(String, DEMO_TOPIC, qos_profile_sensor_data)

    def _on_demo_timing(self, message):
        try:
            if len(message.data.encode()) > 65536: raise ValueError("timing_evidence_too_large")
            self._demo_proofs.receive(json.loads(message.data), time.monotonic_ns())
        except (ValueError, TypeError, KeyError) as exc:
            self._demo_proofs.invalidate_pending()
            self._invalid("original_pose_evidence:"+str(exc))

    def _on_angular_velocity(self, message):
        stamp = _px4_timestamp_us(message)
        xyz = tuple(float(item) for item in message.xyz)
        if stamp <= 0 or len(xyz) != 3 or not all(math.isfinite(item) for item in xyz):
            self._invalid("angular_velocity_source_invalid")
            return
        with self._demo_rate_lock:
            self._demo_rates[stamp] = xyz
            while len(self._demo_rates) > 32: self._demo_rates.popitem(last=False)
        super()._on_angular_velocity(message)

    def _on_vehicle_odometry(self, message):
        # The original adapter supports an odometry fallback. This producer
        # requires the actual ATTITUDE_QUATERNION-bound angular-rate sample.
        self._invalid("odometry_rate_substitute_not_supported")

    def _health_loop(self):
        while not self._demo_stop.is_set():
            started = time.monotonic_ns()
            try:
                with urlopen(self._jetson_url+"/health", timeout=.25) as response:
                    data = response.read(131073)
                if len(data) > 131072: raise ValueError("backend_health_too_large")
                health = json.loads(data)
            except Exception:
                health = None
            self._demo_health.update(health, started, time.monotonic_ns())
            self._demo_stop.wait(1.)

    def _request_safety_worker(self, color, depth, oldest_received_at, z_ned_m, yaw_rad, **kwargs):
        try:
            geometry = kwargs.get("geometry_snapshot")
            if geometry is None or geometry.position is None:
                raise ValueError("exact_geometry_pose_snapshot_required")
            pos_stamp = geometry.query_values["position_timestamp_us"]
            att_stamp = geometry.query_values["attitude_timestamp_us"]
            proofs, epoch = self._demo_proofs.match(pos_stamp, att_stamp, time.monotonic_ns())
            with self._demo_rate_lock:
                xyz = self._demo_rates.get(att_stamp)
            if xyz is None: raise ValueError("exact_attitude_rate_source_match_missing")
            pos_at = min(geometry.position.received_at, proofs[0]["source_local_earliest_ns"]/NS)
            att_at = min(geometry.attitude_received_at, proofs[1]["source_local_earliest_ns"]/NS)
            geometry = replace(geometry, position=replace(geometry.position, received_at=pos_at), attitude_received_at=att_at)
            kwargs.update(geometry_snapshot=geometry, position_received_at=pos_at,
                          angular_velocity_received_at=att_at, angular_rate_rad_s=math.sqrt(sum(item*item for item in xyz)))
            observed_ns = int(oldest_received_at*NS)
            expiry = min(observed_ns+500_000_000, int((pos_at+.25)*NS), int((att_at+.25)*NS),
                         int((geometry.camera_info_received_at+.5)*NS), *(proof["evidence_expires_monotonic_ns"] for proof in proofs))
            binding = {"rgb_timestamp_ns": _stamp_ns(color), "depth_timestamp_ns": _stamp_ns(depth),
                       "depth_frame_id": str(depth.header.frame_id), "depth_encoding": str(depth.encoding),
                       "position_timestamp_us": pos_stamp, "attitude_timestamp_us": att_stamp,
                       "angular_velocity_timestamp_us": att_stamp,
                       "position_ned_m": list(geometry.position.state[:3]),
                       "velocity_ned_mps": list(geometry.position.state[4:]), "heading_rad": yaw_rad,
                       "fc_session_id": epoch[0], "fc_link_identity": epoch[1],
                       "observed_monotonic_ns": observed_ns, "expires_monotonic_ns": expiry,
                       "original_pose_evidence": proofs, "camera_acquisition_time_proven": False}
            if time.monotonic_ns() >= expiry: raise ValueError("demo_snapshot_expired_before_request")
            self._demo_context.binding = binding
            JetsonD435iBridgeNode._request_safety_worker(self, color, depth, oldest_received_at, z_ned_m, yaw_rad, **kwargs)
        except Exception as exc:
            with self._lock:
                self._completed_safety_error = str(exc)
                self._completed_safety_result = None
                self._safety_request_active = False
            self._notify_completion()
        finally:
            self._demo_context.binding = None

    def _request_safety_jetson(self, color, depth, **kwargs):
        # Health is sampled separately: no extra GET or wait in the frame loop.
        epoch = self._demo_health.current(time.monotonic_ns())
        binding = self._demo_context.binding
        if not binding or time.monotonic_ns() >= binding["expires_monotonic_ns"]:
            raise ValueError("original_demo_input_expired")
        payload = JetsonD435iBridgeNode._request_safety_jetson(self, color, depth, **kwargs)
        if payload.get("observation_only") is not True:
            raise ValueError("demo_requires_observation_only_backend")
        if self._demo_health.current(time.monotonic_ns()) != epoch:
            raise ValueError("backend_epoch_changed_during_request")
        payload["safety"]["_demo_original_binding"] = binding
        payload["safety"]["_demo_backend_epoch"] = epoch
        return payload

    def _publish_safety(self, payload, yaw_rad):
        try:
            if self._sensor_faults: raise ValueError(";".join(self._sensor_faults.values()))
            binding = payload["_demo_original_binding"]
            self._demo_proofs.match(binding["position_timestamp_us"], binding["attitude_timestamp_us"], time.monotonic_ns())
            envelope = self._demo_envelope.build(payload, binding, time.monotonic_ns(), self._demo_health.current(time.monotonic_ns()))
            self._emit_demo(envelope)
            self._last_stale_reason = ""
        except (ValueError, TypeError, KeyError) as exc:
            self._invalid(str(exc))

    def _emit_report(self, report):
        # Inherited STALE/error routes never fall through to the old report or
        # SafetyDecision output. They explicitly revoke this private evidence.
        self._invalid(report.get("reason", "demo_perception_unavailable"))

    def _invalid(self, reason):
        now = time.monotonic()
        if reason == self._last_demo_reason and now-self._last_demo_output < .2:
            return
        self._emit_demo({"schema_version": 1, "mode": "DEMO_PERCEPTION_ONLY", "flight_commands_enabled": False,
                         "producer_session": self._session, "evidence_clock_id": self._demo_clock,
                         "published_monotonic_ns": time.monotonic_ns(), "valid": False, "reason": str(reason)})

    def _emit_demo(self, envelope):
        now = time.monotonic_ns()
        if envelope.get("valid") and now >= envelope["input_binding"]["expires_monotonic_ns"]:
            return self._invalid("demo_output_original_deadline_expired")
        if self._journal_error:
            envelope = {"schema_version": 1, "mode": "DEMO_PERCEPTION_ONLY", "flight_commands_enabled": False,
                        "producer_session": self._session, "evidence_clock_id": self._demo_clock,
                        "published_monotonic_ns": now, "valid": False, "reason": "demo_journal_unavailable"}
        body = json.dumps(envelope, allow_nan=False)
        if self._journal is not None and not self._journal_error:
            try: self._journal.write(body+"\n")
            except OSError:
                self._journal_error = True
                return self._invalid("demo_journal_unavailable")
        if envelope.get("valid") and time.monotonic_ns() >= envelope["input_binding"]["expires_monotonic_ns"]:
            return self._invalid("demo_journal_exceeded_original_deadline")
        message = String()
        message.data = body
        self._demo_publisher.publish(message)
        self._last_demo_output, self._last_demo_reason = time.monotonic(), envelope["reason"]

    def destroy_node(self):
        self._demo_stop.set()
        if self._demo_thread is not None: self._demo_thread.join(timeout=.5)
        # Parent marks closing and cancels queued work. A completing HTTP worker
        # cannot trigger a destroyed ROS guard condition after that boundary.
        try:
            self._invalid("demo_perception_stopped")
            JetsonD435iBridgeNode.destroy_node(self)
        finally:
            if self._journal is not None:
                self._journal.close()
                self._journal = None


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = D435DemoPerceptionNode()
        rclpy.spin(node)
    finally:
        if node is not None: node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


if __name__ == "__main__":
    main()

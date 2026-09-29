"""Explicit limited-demo MAVLink link; no ROS/uORB impersonation.

One caller owns read, accept and write. Never open alongside the observer.
Default observe can request telemetry ONLY. Auto-demo additionally requires an
explicit, one-session authorization; no arm/takeoff/land/RTL/descent API exists.

PX4 v1.17 d6f12ad1c4f70ad3230afd7d86e971421e02fef4 source contract:
src/modules/mavlink/streams/{HEARTBEAT,SYS_STATUS,ODOMETRY}.hpp
ODOMETRY is vehicle_odometry (sample timestamp, NED, real reset_counter).
RC presence means receiver present/enabled/healthy, NOT proof of pilot attention.
Status uses original USB receipt (wire messages have no source timestamp).
Pose uses existing TIMESYNC bounds, NOT camera exposure synchronization.
"""
from dataclasses import dataclass
import json
import math
import time

from .observer_fc_telemetry import TelemetryState, finite
from .observer_fc_timesync import KINDS, NS, valid_ns
from .observer_fc_transport import UsbTelemetryTransport
from .small_obstacle_demo_guard import Telemetry

STATUS_NS = 750_000_000
POSE_NS = 250_000_000
PAIR_NS = 75_000_000
RC_RECEIVER = 1 << 16
# PX4 explicitly rejects SET_MESSAGE_INTERVAL for HEARTBEAT (message 0).
# request_heartbeat uses fixed REQUEST_MESSAGE instead; no rate parameter change.
EXTRA_INTERVALS = {1: 200000, 245: 200000, 331: 50000}
HEARTBEAT_REQUEST_NS = 200_000_000
RX_FIELDS = {
    "HEARTBEAT": ("autopilot", "type", "base_mode", "custom_mode", "system_status"),
    "SYS_STATUS": ("onboard_control_sensors_present", "onboard_control_sensors_enabled", "onboard_control_sensors_health"),
    "EXTENDED_SYS_STATE": ("landed_state", "vtol_state"),
    "ODOMETRY": ("time_usec", "reset_counter", "frame_id", "child_frame_id", "estimator_type"),
    "COMMAND_ACK": ("command", "result", "target_system", "target_component"),
}


def _diagnostic_scalar(value):
    return value if type(value) is int and -(1 << 64) < value < (1 << 64) else None


def _diagnostic_type(value):
    for kind in (int, float, bool, str, list, dict, type(None)):
        if type(value) is kind:
            return kind.__name__
    return "other"


@dataclass(frozen=True)
class FcSnapshot:
    telemetry: Telemetry | None
    reason: str
    fc_epoch: str
    expires_ns: int
    pose_source_us: int
    main_mode: int | None
    actual_offboard: bool
    status_at_ns: int
    receiver_healthy: bool
    armed: bool
    airborne: bool
    ground_reference_ready: bool
    last_ack_command: int | None = None
    last_ack_result: int | None = None
    ack_received_ns: int = 0
    timing_evidence: tuple[dict, ...] = ()
    offboard_departure_count: int = 0


class SmallObstacleFcTelemetry:
    """Actual bounded fields for the limited guard; absent data stays absent."""
    def __init__(self, state: TelemetryState, clock_id: str):
        if not isinstance(state, TelemetryState) or not isinstance(clock_id, str) or not clock_id:
            raise ValueError("state_and_clock_identity_required")
        self.state, self.clock_id = state, clock_id
        self.identity = (state.timing.session_id, state.timing.link_identity)
        self.fc_epoch = "|".join(self.identity)
        self.fault = ""
        self.extra, self.receipts = {}, {}
        self.last_now_ns = 0
        self.odom_stamp = 0
        self.odom_expires_ns = 0
        self.reset_counter = None
        self.ground_z = None
        self.ground_since_ns = None
        self.ground_pose_us = 0
        self.ground_checked_ns = 0
        self.pose_cache = None
        self.last_ack = (None, None, 0)
        self.last_main_mode = None
        self.offboard_departure_count = 0
        self.ever_authorized = False  # Diagnostic context only, never authority.
        self.rx_diagnostics = {}
        self.last_reset_diagnostic = None
        self._rx_rejection = ""

    def _retire(self, reason):
        self.fault = self.fault or reason
        self.ground_z = self.ground_since_ns = None
        self.pose_cache = None
        return False

    def _clock(self, now_ns):
        if not valid_ns(now_ns) or now_ns < self.last_now_ns:
            return self._retire("local_clock_regressed_or_invalid")
        self.last_now_ns = now_ns
        timing = self.state.timing
        if (timing.session_id, timing.link_identity) != self.identity:
            return self._retire("fc_link_epoch_changed")
        if self.state.fault or timing.mapper.fault:
            return self._retire(self.state.fault or timing.mapper.fault)
        return not self.fault

    def _reject(self, reason):
        self._rx_rejection = reason
        return False

    def accept(self, kind, value, system_id, component_id, received_ns, now_ns):
        """Record bounded metadata for five fixed kinds, without granting trust."""
        self._rx_rejection = ""
        old_counter, old_ground = self.reset_counter, self.ground_z is not None
        accepted = self._accept(kind, value, system_id, component_id, received_ns, now_ns)
        if kind in RX_FIELDS:
            fields = value if isinstance(value, dict) else {}
            previous = self.rx_diagnostics.get(kind, {})
            record = dict(
                received_count=min(previous.get("received_count", 0)+1, (1 << 63)-1),
                accepted_count=min(previous.get("accepted_count", 0)+int(accepted), (1 << 63)-1),
                rejected_count=min(previous.get("rejected_count", 0)+int(not accepted), (1 << 63)-1),
                accepted=accepted,
                reason="" if accepted else self.fault or self._rx_rejection or "rejected",
                received_monotonic_ns=_diagnostic_scalar(received_ns),
                processed_monotonic_ns=_diagnostic_scalar(now_ns),
                system_id=_diagnostic_scalar(system_id), component_id=_diagnostic_scalar(component_id),
                expected_source=type(system_id) is int and type(component_id) is int and (system_id, component_id) == (1, 1),
                fields={key: _diagnostic_scalar(fields.get(key)) for key in RX_FIELDS[kind]},
                field_types={key: _diagnostic_type(fields.get(key)) for key in RX_FIELDS[kind]})
            self.rx_diagnostics[kind] = record
            if kind == "ODOMETRY" and self.fault == "actual_odometry_reset_changed" and self.last_reset_diagnostic is None:
                hb, ext = self.extra.get("HEARTBEAT", {}), self.extra.get("EXTENDED_SYS_STATE", {})
                self.last_reset_diagnostic = dict(record, previous_reset_counter=old_counter,
                    incoming_reset_counter=_diagnostic_scalar(fields.get("reset_counter")),
                    ground_reference_was_accepted=old_ground, ever_authorized=self.ever_authorized,
                    heartbeat_original_received_ns=self.receipts.get("HEARTBEAT", 0),
                    extended_state_original_received_ns=self.receipts.get("EXTENDED_SYS_STATE", 0),
                    heartbeat_base_mode=hb.get("base_mode"), landed_state=ext.get("landed_state"),
                    policy="TERMINAL_NO_IN_PLACE_EPOCH_REACQUISITION")
        return accepted

    def diagnostic(self):
        """Detached read-only diagnostics; no sample, epoch or deadline update."""
        return json.loads(json.dumps(dict(scope="BOUNDED_RX_METADATA_NOT_AUTHORITY",
            messages=self.rx_diagnostics, last_odometry_reset=self.last_reset_diagnostic,
            accepted_original_receipts_ns={key: self.receipts.get(key, 0) for key in RX_FIELDS},
            status_lease_ns=STATUS_NS, pose_lease_ns=POSE_NS, pair_lease_ns=PAIR_NS), allow_nan=False))

    def _accept(self, kind, value, system_id, component_id, received_ns, now_ns):
        if not self._clock(now_ns):
            return self._reject("fault_latched")
        if (type(system_id) is not int or type(component_id) is not int
                or (system_id, component_id) != (1, 1)):
            return self._reject("unexpected_source")
        if not valid_ns(received_ns) or not received_ns <= now_ns:
            return self._retire("invalid_original_receipt")
        if not isinstance(value, dict):
            return self._reject("message_object_required")
        if kind in ("SYS_STATUS", "EXTENDED_SYS_STATE", "ODOMETRY", "COMMAND_ACK"):
            # Multiple FIFO messages in one USB read share one ORIGINAL receipt.
            # Apply later content in that read without extending its deadline.
            if received_ns < self.receipts.get(kind, 0):
                return self._reject("original_receipt_regressed")
            if now_ns-received_ns >= (POSE_NS if kind == "ODOMETRY" else STATUS_NS):
                self.extra.pop(kind, None)
                return self._reject("original_receipt_expired")
            if kind == "SYS_STATUS":
                names = ("onboard_control_sensors_present", "onboard_control_sensors_enabled",
                         "onboard_control_sensors_health")
                if not all(type(value.get(k)) is int and 0 <= value[k] < 2**32 for k in names):
                    self.extra.pop(kind, None)
                    return self._reject("invalid_sensor_bits")
                safe = {k: value[k] for k in names}
            elif kind == "EXTENDED_SYS_STATE":
                if type(value.get("landed_state")) is not int or value["landed_state"] not in (1, 2, 3, 4):
                    self.extra.pop(kind, None)
                    return self._reject("invalid_landed_state")
                safe = {"landed_state": value["landed_state"]}
            elif kind == "COMMAND_ACK":
                # ACK is diagnostic only, never considered a mode transition.
                if (value.get("target_system"), value.get("target_component")) != (245, 191):
                    return self._reject("ack_target_mismatch")
                if value.get("command") != 176 or type(value.get("result")) is not int:
                    return self._reject("ack_diagnostic_only_not_mode_command")
                self.last_ack = (176, value["result"], received_ns)
                self.receipts[kind] = received_ns
                return True
            else:
                return self._accept_odometry(value, received_ns, now_ns)
            self.extra[kind], self.receipts[kind] = safe, received_ns
            return True
        if kind == "HEARTBEAT":
            if (received_ns < self.receipts.get(kind, 0)
                    or now_ns-received_ns >= STATUS_NS):
                return self._reject("heartbeat_receipt_regressed_or_expired")
            if not all(type(value.get(k)) is int for k in ("base_mode", "custom_mode", "system_status")):
                self.extra.pop(kind, None)
                return self._reject("invalid_heartbeat_fields")
        accepted = self.state.accept(kind, value, system_id, component_id,
                                     now_ns/NS, received_ns=received_ns)
        if accepted:
            self.receipts[kind] = received_ns
            if kind == "HEARTBEAT":
                self.extra[kind] = {k: value[k] for k in ("base_mode", "custom_mode", "system_status")}
                mode = (value["custom_mode"] >> 16) & 255 if value["base_mode"] & 1 else None
                if self.last_main_mode == 6 and mode != 6:
                    self.offboard_departure_count += 1
                self.last_main_mode = mode
        elif kind == "HEARTBEAT":
            self.extra.pop(kind, None)
            self._rx_rejection = "heartbeat_rejected_by_telemetry_state"
        if self.state.fault or self.state.timing.mapper.fault:
            self._retire(self.state.fault or self.state.timing.mapper.fault)
        return accepted

    def _accept_odometry(self, value, received_ns, now_ns):
        stamp, counter = value.get("time_usec"), value.get("reset_counter")
        if (type(stamp) is not int or stamp <= 0 or type(counter) is not int
                or not 0 <= counter <= 255 or value.get("frame_id") != 1
                or value.get("child_frame_id") != 1 or value.get("estimator_type") != 8):
            self.odom_expires_ns = 0
            return self._reject("invalid_odometry_fields_or_frame")
        if stamp < self.odom_stamp:
            return self._retire("odometry_clock_regressed")
        if stamp == self.odom_stamp:
            return self._reject("duplicate_odometry_source_timestamp")
        if self.reset_counter is not None and counter != self.reset_counter:
            return self._retire("actual_odometry_reset_changed")
        mapper = self.state.timing.mapper
        if not self.state.timing.enabled or not mapper.synchronized(now_ns):
            self.odom_expires_ns = 0
            return self._reject("odometry_timesync_not_ready")
        source = stamp*1000
        delta = max(abs(source-mapper.anchor_remote_ns), abs(source+1000-mapper.anchor_remote_ns))
        earliest = source+mapper.offset_low_ns-mapper._drift_margin(delta)
        if earliest > received_ns:
            return self._retire("odometry_source_in_future")
        if now_ns-earliest > mapper.SOURCE_AGE_MAX_NS:
            self.odom_expires_ns = 0
            return self._reject("odometry_source_age_excess")
        self.odom_expires_ns = min(earliest+POSE_NS, received_ns+POSE_NS, mapper.sync_expires_ns)
        if now_ns >= self.odom_expires_ns:
            return self._reject("odometry_original_deadline_expired")
        self.odom_stamp, self.reset_counter = stamp, counter
        self.receipts["ODOMETRY"] = received_ns
        return True

    def snapshot(self, now_ns):
        valid_clock = self._clock(now_ns)
        hb, sys, ext = (self.extra.get(k, {}) for k in ("HEARTBEAT", "SYS_STATUS", "EXTENDED_SYS_STATE"))
        status_at = min((self.receipts.get(k, 0) for k in ("HEARTBEAT", "SYS_STATUS", "EXTENDED_SYS_STATE")))
        status_fresh = status_at > 0 and 0 <= now_ns-status_at < STATUS_NS
        receiver = bool(sys) and all(sys[k] & RC_RECEIVER for k in sys)
        mode = ((hb["custom_mode"] >> 16) & 255) if hb and hb["base_mode"] & 1 else None
        armed = bool(hb.get("base_mode", 0) & 128)
        airborne = ext.get("landed_state") in (2, 3, 4)
        reason = self.fault if not valid_clock else ""
        if not reason and not status_fresh: reason = "status_missing_or_stale"
        if not reason and not receiver: reason = "rc_receiver_not_present_enabled_healthy"
        if not reason and hb.get("system_status") != (4 if armed else 3): reason = "px4_not_active_or_standby"
        if not reason and not self.state.timing.enabled: reason = "bounded_timesync_required"
        pos = self.state.position(now_ns/NS) if not reason else None
        if not reason and (pos is None or not pos["valid"]): reason = self.state.position_reason(now_ns/NS)
        proofs = [self.state.timing_evidence(k, now_ns/NS, self.clock_id) for k in KINDS] if not reason else []
        if not reason and (len(proofs) != 3 or any(p is None for p in proofs)): reason = "original_pose_evidence_missing"
        if not reason and (now_ns >= self.odom_expires_ns or self.reset_counter is None): reason = "odometry_reset_evidence_missing_or_stale"
        if not reason:
            receipts = [self.receipts.get(k, 0) for k in KINDS] + [self.receipts["ODOMETRY"]]
            source_stamps = [self.state.stamps[k] for k in KINDS] + [self.odom_stamp]
            if max(receipts)-min(receipts) > PAIR_NS or (max(source_stamps)-min(source_stamps))*1000 > PAIR_NS:
                reason = "pose_odometry_pair_skew"
        expiry = 0
        telemetry = None
        if not reason:
            expiry = min(status_at+STATUS_NS, self.odom_expires_ns,
                         *(p["evidence_expires_monotonic_ns"] for p in proofs),
                         *(self.receipts[k]+POSE_NS for k in KINDS))
            if now_ns >= expiry:
                reason = "original_pose_deadline_expired"
            else:
                stamp = pos["timestamp"]
                if self.pose_cache is None or self.pose_cache[0] != stamp:
                    proof = next(p for p in proofs if p["kind"] == "LOCAL_POSITION_NED")
                    self.pose_cache = (stamp, tuple(pos[k] for k in ("x", "y", "z")),
                        tuple(pos[k] for k in ("vx", "vy", "vz")), pos["heading"],
                        proof["source_local_earliest_ns"]/NS)
                _, xyz, velocity, yaw, observed = self.pose_cache
                if not armed and ext.get("landed_state") == 1 and math.hypot(*velocity) <= .1:
                    if self.ground_z is None and stamp > self.ground_pose_us:
                        if now_ns-self.ground_checked_ns >= POSE_NS:
                            self.ground_since_ns = None
                        self.ground_since_ns = now_ns if self.ground_since_ns is None else self.ground_since_ns
                        self.ground_checked_ns = now_ns
                        self.ground_pose_us = stamp
                        if now_ns-self.ground_since_ns >= 500_000_000:
                            self.ground_z = xyz[2]
                elif self.ground_z is None:
                    self.ground_since_ns = None
                if self.ground_z is None:
                    reason = "same_epoch_stationary_ground_reference_missing"
                else:
                    telemetry = Telemetry(self.fc_epoch, observed, status_at/NS, xyz, velocity, yaw,
                        self.ground_z, self.fc_epoch, True, True, armed, airborne, bool(receiver))
        elif self.ground_z is None:
            self.ground_since_ns = None
        return FcSnapshot(telemetry, reason, self.fc_epoch, expiry,
            pos["timestamp"] if pos else 0, mode, mode == 6, status_at,
            bool(receiver and status_fresh), armed, airborne, self.ground_z is not None,
            *self.last_ack, tuple(p for p in proofs if p is not None), self.offboard_departure_count)


class SmallObstacleFcTransport(UsbTelemetryTransport):
    """Opt-in command transport. The existing observe transport is untouched."""
    def __init__(self, journal, device="/dev/jolgwa-pixhawk6c", *, timing=None, mode="observe"):
        if mode not in ("observe", "auto-demo"):
            raise ValueError("explicit_demo_mode_required")
        super().__init__(journal, device, timing=timing)
        self.demo_mode = mode
        self.fc = None
        self.authorization = None
        self.authorization_used = False
        self.retired = False
        self.last_velocity_ns = 0
        self.first_velocity_ns = 0
        self.offboard_requested = False
        self.offboard_seen = False
        self.handover_sent = False
        self.last_heartbeat_request_ns = None

    def bind_telemetry(self, fc):
        if self.fc is not None or not isinstance(fc, SmallObstacleFcTelemetry) or fc.state.timing is not self.timing:
            raise ValueError("single_matching_telemetry_owner_required")
        self.fc = fc

    def close(self):
        self.retired = True
        super().close()

    def set_mode(self, mode):
        """Local selection only; no TX, no reset of used tokens/retired epoch."""
        if mode not in ("observe", "auto-demo"):
            raise ValueError("explicit_demo_mode_required")
        if self.authorization is not None:
            self.retired = True
        self.authorization = None
        self.demo_mode = mode

    def request_demo_stream(self, message_id):
        if type(message_id) is not int or message_id not in EXTRA_INTERVALS:
            raise ValueError("not_a_fixed_demo_telemetry_stream")
        msg = self.common.MAVLink_command_long_message(1, 1, 511, 0,
            float(message_id), float(EXTRA_INTERVALS[message_id]), 0., 0., 0., 0., 0.)
        self._write_demo(msg, "tx_demo_telemetry_request", None, None, {"message_id": message_id})

    def request_heartbeat(self):
        """Fixed telemetry-only REQUEST_MESSAGE(0), at most 5 Hz, no waiting.

        PX4 v1.17 receiver::handle_request_message_command invokes the existing
        heartbeat stream's send(). ACK neither renews its receipt nor proves RX.
        No API accepts an arbitrary command, target or requested message here.
        """
        if (self.retired or self.fc is None or self.fc.fault
                or self.fc.state.fault or self.timing.mapper.fault):
            return False
        now = time.monotonic_ns()
        last = getattr(self, "last_heartbeat_request_ns", None)
        if not valid_ns(now) or (last is not None and now < last):
            self.retired = True
            raise TimeoutError("heartbeat_request_clock_regressed")
        if last is not None and now-last < HEARTBEAT_REQUEST_NS:
            return False
        msg = self.common.MAVLink_command_long_message(1, 1, 512, 0, 0., 0., 0., 0., 0., 0., 0.)
        result = self._write_demo(msg, "tx_demo_heartbeat_request", None,
            now+HEARTBEAT_REQUEST_NS, {"message_id": 0, "command": 512, "original_requested_ns": now})
        # Completion spacing prevents a slow encode/write from causing a burst.
        self.last_heartbeat_request_ns = time.monotonic_ns()
        return result

    def authorize(self, token, fc_epoch, expires_ns):
        now = time.monotonic_ns()
        if self.demo_mode != "auto-demo" or self.authorization_used or self.retired:
            raise PermissionError("explicit_unused_auto_demo_authorization_required")
        if not isinstance(token, str) or not 1 <= len(token) <= 128 or not valid_ns(expires_ns) or not now < expires_ns <= now+60*NS:
            raise ValueError("invalid_authorization")
        snapshot = self.fc.snapshot(now) if self.fc else None
        if (snapshot is None or snapshot.reason or snapshot.fc_epoch != fc_epoch
                or not snapshot.armed or not snapshot.airborne or snapshot.main_mode != 3):
            raise PermissionError("fresh_pilot_posctl_hover_required")
        self.authorization = (token, fc_epoch, expires_ns)
        self.authorized_departure_count = snapshot.offboard_departure_count
        self.authorization_used = True
        self.fc.ever_authorized = True

    def _check(self, snapshot, expires_ns, now):
        if self.demo_mode != "auto-demo" or self.retired or not self.authorization or self.fc is None:
            raise PermissionError("flight_output_not_authorized")
        current = self.fc.snapshot(now)
        if current.offboard_departure_count != self.authorized_departure_count:
            self.retired = True
            raise PermissionError("actual_offboard_departure_terminal")
        if current.actual_offboard and not self.offboard_requested:
            self.retired = True
            raise PermissionError("offboard_not_entered_by_this_demo")
        if current.actual_offboard:
            self.offboard_seen = True
        if self.offboard_seen and not current.actual_offboard:
            self.retired = True
            raise PermissionError("actual_offboard_departure_terminal")
        if (not isinstance(snapshot, FcSnapshot) or snapshot.reason or current.reason
                or snapshot.fc_epoch != self.authorization[1] or current.fc_epoch != snapshot.fc_epoch
                or not current.armed or not current.airborne or not current.receiver_healthy
                or current.main_mode not in (3, 6)):
            self.retired = True
            raise PermissionError("actual_fc_state_invalid")
        if (not valid_ns(expires_ns) or not now < expires_ns <= now+500_000_000
                or now >= min(snapshot.expires_ns, current.expires_ns, self.authorization[2])):
            raise TimeoutError("original_command_or_fc_deadline_expired")

    def _write_demo(self, message, kind, snapshot, expires_ns, metadata):
        start = time.monotonic_ns()
        request_only_deadline = snapshot is None and expires_ns is not None
        if request_only_deadline and (not valid_ns(expires_ns) or start >= expires_ns):
            raise TimeoutError("original_demo_request_deadline_expired")
        if snapshot is not None:
            self._check(snapshot, expires_ns, start)
        packet = message.pack(self.encoder)
        self.journal.write(json.dumps({"kind": kind, "original_started_ns": start,
            "fc_epoch": self.fc.fc_epoch if self.fc else None, "expires_ns": expires_ns,
            "packet_hex": packet.hex(), **metadata})+"\n")
        self.journal.flush()
        before = time.monotonic_ns()
        if before < start:
            self.retired = True
            raise TimeoutError("clock_regressed_before_serial_write")
        if request_only_deadline and before >= expires_ns:
            self.retired = True
            raise TimeoutError("original_demo_request_deadline_expired")
        if snapshot is not None:
            self._check(snapshot, expires_ns, before)
        count = self.port.write(packet)
        if count != len(packet):
            self.retired = True
            raise OSError("partial_demo_packet_write")
        after = time.monotonic_ns()
        self.encoder.seq = (self.encoder.seq+1) % 256
        self.tx_count += 1
        if request_only_deadline and (after < before or after >= expires_ns):
            self.retired = True
            raise TimeoutError("serial_write_completed_outside_original_deadline")
        if snapshot is not None and (after < before or after >= min(expires_ns, snapshot.expires_ns, self.authorization[2])):
            self.retired = True
            raise TimeoutError("serial_write_completed_outside_original_deadline")
        return True

    def send_velocity(self, snapshot, vx, vy, vz, yaw, *, expires_ns):
        now = time.monotonic_ns()
        self._check(snapshot, expires_ns, now)
        if not finite((vx, vy, vz, yaw)) or math.hypot(vx, vy) > .5 or not -.6 <= vz <= 0 or not -math.pi <= yaw <= math.pi:
            raise ValueError("demo_velocity_or_yaw_out_of_bounds_no_descent")
        if now-self.last_velocity_ns < 50_000_000:
            return False
        # Ignore position/acceleration/yaw-rate; finite velocity + absolute yaw.
        mask = 1 | 2 | 4 | 64 | 128 | 256 | 2048
        msg = self.common.MAVLink_set_position_target_local_ned_message(
            (snapshot.pose_source_us//1000) & 0xffffffff, 1, 1, 1, mask,
            0., 0., 0., vx, vy, vz, 0., 0., 0., yaw, 0.)
        result = self._write_demo(msg, "tx_demo_velocity_ned", snapshot, expires_ns,
                                  {"velocity_ned_mps": [vx, vy, vz], "yaw_rad": yaw})
        if not self.first_velocity_ns or now-self.last_velocity_ns >= POSE_NS:
            self.first_velocity_ns = now
        self.last_velocity_ns = now
        return result

    def request_offboard(self, snapshot, *, expires_ns):
        self._check(snapshot, expires_ns, time.monotonic_ns())
        if self.offboard_requested:
            return False
        now = time.monotonic_ns()
        if (not self.first_velocity_ns or now-self.first_velocity_ns < NS
                or now-self.last_velocity_ns >= POSE_NS):
            raise PermissionError("actual_setpoint_prestream_not_ready")
        msg = self.common.MAVLink_command_long_message(1, 1, 176, 0, 1., 6., 0., 0., 0., 0., 0.)
        result = self._write_demo(msg, "tx_demo_request_offboard", snapshot, expires_ns, {})
        self.offboard_requested = True
        return result

    def handover_posctl(self, snapshot, *, expires_ns):
        self._check(snapshot, expires_ns, time.monotonic_ns())
        if not snapshot.actual_offboard or not self.offboard_requested:
            raise PermissionError("only_our_actual_offboard_can_request_handover")
        if self.handover_sent:
            return False
        msg = self.common.MAVLink_command_long_message(1, 1, 176, 0, 1., 3., 0., 0., 0., 0., 0.)
        result = self._write_demo(msg, "tx_demo_request_posctl", snapshot, expires_ns, {})
        self.handover_sent = True
        self.retired = True
        return result

    def stop(self):
        """One POSCTL request only if fresh and still our Offboard; else TX0."""
        if self.retired or self.demo_mode != "auto-demo" or not self.authorization or self.fc is None:
            self.retired = True
            return False
        now = time.monotonic_ns()
        snapshot = self.fc.snapshot(now)
        if (snapshot.reason or not snapshot.actual_offboard or not self.offboard_requested
                or now >= min(snapshot.expires_ns, self.authorization[2])):
            self.retired = True
            return False
        try:
            return self.handover_posctl(snapshot, expires_ns=min(snapshot.expires_ns, self.authorization[2]))
        finally:
            self.retired = True

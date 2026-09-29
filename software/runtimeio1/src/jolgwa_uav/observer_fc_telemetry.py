"""MAVLink telemetry semantics for OBSERVE ONLY; no flight-control output.

ROS conversion uses private observer topics, never impersonates /fmu/out.
LOCAL_POSITION_NED numbers alone do not prove estimator validity.
"""
import math
from .observer_fc_timesync import ObserveTimesync, KINDS, NS, valid_identity

STREAM_INTERVALS = {31: 20000, 32: 20000, 230: 50000}
TOPIC_PREFIX = "/jolgwa/observer/fc"
DIAGNOSTIC_MESSAGE_KINDS = frozenset({
    "HEARTBEAT", "ATTITUDE_QUATERNION", "LOCAL_POSITION_NED",
    "ESTIMATOR_STATUS", "AUTOPILOT_VERSION", "GPS_RAW_INT", "TIMESYNC"})


class RelativeSourceTimePolicy:
    """Detect relative queue growth without claiming absolute one-way delay.

    Each message kind keeps its own minimum receipt/source offset for the
    lifetime of the connection. The minimum is deliberately never aged out:
    doing so would make an old, steadily queued stream appear fresh again.
    """

    def __init__(self, *, maximum_offset_growth_s=.20, maximum_clock_jump_s=.25):
        self.maximum_offset_growth_s = float(maximum_offset_growth_s)
        self.maximum_clock_jump_s = float(maximum_clock_jump_s)
        self.stamps = {}
        self.minimum_offsets = {}
        self.fault = ""

    def assess(self, kind, stamp_us, received_monotonic_s):
        if type(stamp_us) is not int or stamp_us <= 0:
            return False, "invalid_source_timestamp"
        if not isinstance(received_monotonic_s, (int, float)) or not math.isfinite(
            received_monotonic_s
        ):
            return False, "invalid_receipt_time"
        if self.fault:
            return False, "fault_latched"

        previous = self.stamps.get(kind, 0)
        if stamp_us < previous:
            self.fault = "fc_clock_regressed_restart_required"
            return False, "source_clock_regressed"
        if stamp_us == previous:
            return False, "duplicate_source_timestamp"

        offset = received_monotonic_s - stamp_us / 1e6
        best = self.minimum_offsets.get(kind, offset)
        # A source clock that suddenly advances relative to the receiver is a
        # new epoch (or corrupt/replayed time), not improved latency.
        if offset < best - self.maximum_clock_jump_s:
            self.stamps[kind] = stamp_us
            self.fault = "fc_clock_jump_restart_required"
            return False, "source_clock_jump"

        self.minimum_offsets[kind] = min(best, offset)
        self.stamps[kind] = stamp_us
        if offset - self.minimum_offsets[kind] > self.maximum_offset_growth_s:
            return False, "source_offset_excess"
        return True, ""


def diagnostic_number(value):
    """Expose only finite scalar timing metadata, never arbitrary packet data."""
    if type(value) is int:
        return value if -(1 << 64) < value < (1 << 64) else None
    if type(value) is float and math.isfinite(value):
        return value
    return None


def request_parameters(command, message_id, interval=0):
    """The complete outbound command allowlist. No arbitrary COMMAND_LONG."""
    if type(command) is not int or type(message_id) is not int or type(interval) is not int:
        raise ValueError("integer telemetry request required")
    if command == 512 and message_id == 148 and interval == 0:
        return [148., 0., 0., 0., 0., 0., 0.]
    if command == 511 and STREAM_INTERVALS.get(message_id) == interval:
        return [float(message_id), float(interval), 0., 0., 0., 0., 0.]
    raise ValueError("not an allowed telemetry request")


def finite(values):
    return all(type(v) in (int, float) and math.isfinite(v) for v in values)


def estimator_reason(flags):
    if type(flags) is not int or not 0 <= flags <= 65535:
        return "invalid_estimator_flags"
    if flags & 128:
        return "estimator_constant_position_mode"
    if flags & (1024 | 2048):
        return "estimator_gps_or_accelerometer_fault"
    if not flags & 1:
        return "estimator_attitude_invalid"
    if not flags & (8 | 16):
        return "estimator_horizontal_position_invalid"
    # LOCAL_POSITION_NED.z is a local origin height, not terrain-relative AGL.
    if not flags & 32:
        return "estimator_vertical_position_invalid"
    if flags & 6 != 6:
        return "estimator_velocity_invalid"
    return ""


class TelemetryState:
    def __init__(self, system_id=1, component_id=1, *, timing=None):
        self.source = (system_id, component_id)
        self.timing = timing if timing is not None else ObserveTimesync()
        if self.timing.enabled and self.source != (1, 1):
            raise ValueError("TIMESYNC observe requires fixed physical FC identity 1/1")
        self.timing_sequence = 0
        self.timing_sequences = {}
        self.samples = {}
        self.received = {}
        self.relative_time = RelativeSourceTimePolicy()
        # Preserve the existing attributes used by diagnostics and tests while
        # sharing the policy implementation with route control.
        self.stamps = self.relative_time.stamps
        self.clock_offsets = self.relative_time.minimum_offsets
        self.counts = {}
        self.fault = ""
        # Diagnostic-only state; these fields never participate in acceptance,
        # freshness, estimator validity, source-clock mapping or request output.
        # Unknown MAVLink names share OTHER so remote input cannot grow keys.
        self.rx_counts = {}
        self.rejected_counts = {}
        self.last_rx = {}
        self.last_rejected = {}

    def _record_rx(self, kind, value, system_id, component_id, now):
        key = kind if kind in DIAGNOSTIC_MESSAGE_KINDS else "OTHER"
        self.rx_counts[key] = self.rx_counts.get(key, 0) + 1
        metadata = {"received_monotonic_s": diagnostic_number(now),
                    "expected_source": (system_id, component_id) == self.source}
        if kind in {"ATTITUDE_QUATERNION", "LOCAL_POSITION_NED", "ESTIMATOR_STATUS"}:
            field = "time_usec" if kind == "ESTIMATOR_STATUS" else "time_boot_ms"
            stamp = value.get(field)
            source_us = stamp if field == "time_usec" else stamp * 1000 if type(stamp) is int else None
            safe_source_us = diagnostic_number(source_us)
            offset = (now - safe_source_us / 1e6
                      if metadata["received_monotonic_s"] is not None and safe_source_us is not None else None)
            previous_min = self.clock_offsets.get(kind)
            metadata.update(source_timestamp_field=field,
                            source_timestamp=diagnostic_number(stamp),
                            source_time_us=safe_source_us,
                            offset_s=diagnostic_number(offset),
                            previous_minimum_offset_s=diagnostic_number(previous_min),
                            minimum_offset_s=diagnostic_number(previous_min),
                            offset_delta_s=diagnostic_number(offset-previous_min)
                            if offset is not None and previous_min is not None else None,
                            policy_source_time_us_before=diagnostic_number(self.stamps.get(kind)))
        self.last_rx[key] = metadata
        return key

    def _record_decision(self, key, reason=""):
        metadata = self.last_rx[key]
        metadata.update(decision="rejected" if reason else "accepted", reason=reason)
        if "source_time_us" in metadata:
            minimum = self.clock_offsets.get(key)
            metadata["minimum_offset_s"] = diagnostic_number(minimum)
            offset = metadata["offset_s"]
            metadata["offset_delta_s"] = (diagnostic_number(offset-minimum)
                                           if offset is not None and minimum is not None else None)
            metadata["policy_source_time_us_after"] = diagnostic_number(self.stamps.get(key))
        if reason:
            counts = self.rejected_counts.setdefault(key, {})
            counts[reason] = counts.get(reason, 0) + 1
            self.last_rejected[key] = dict(metadata)
        return not reason

    def accept(self, kind, value, system_id, component_id, now, *, received_ns=None):
        diagnostic_key = self._record_rx(kind, value, system_id, component_id, now)
        if self.timing.enabled and not valid_identity((system_id, component_id)):
            return self._record_decision(diagnostic_key, "unexpected_source")
        if (system_id, component_id) != self.source or not math.isfinite(now):
            return self._record_decision(diagnostic_key, "unexpected_source"
                                         if (system_id, component_id) != self.source else "invalid_receipt_time")
        if kind == "HEARTBEAT":
            if value.get("autopilot") != 12 or value.get("type") != 2:
                self.fault = "unexpected_autopilot_identity"
                return self._record_decision(diagnostic_key, "unexpected_autopilot_identity")
        elif not self.fresh("HEARTBEAT", now, 2.5):
            if self.timing.enabled and self.timing.mapper.anchor_remote_ns is not None:
                self.timing.mapper._fail("heartbeat_lost_restart_required")
                self.fault = self.timing.mapper.fault
            return self._record_decision(diagnostic_key, "heartbeat_missing_or_stale")
        if self.fault:
            return self._record_decision(diagnostic_key, "fault_latched")
        if kind == "TIMESYNC" and self.timing.enabled:
            now_ns = round(now * NS)
            decision = self.timing.accept_reply(value, self.source,
                now_ns if received_ns is None else received_ns, now_ns=now_ns)
            self.fault = self.timing.mapper.fault
            return self._record_decision(diagnostic_key, "" if decision.accepted else decision.reason)
        if kind not in {"HEARTBEAT", "ATTITUDE_QUATERNION", "LOCAL_POSITION_NED",
                        "ESTIMATOR_STATUS", "AUTOPILOT_VERSION", "GPS_RAW_INT"}:
            return self._record_decision(diagnostic_key, "unsupported_message_type")
        if kind in {"ATTITUDE_QUATERNION", "LOCAL_POSITION_NED", "ESTIMATOR_STATUS"}:
            stamp = value.get("time_usec") if kind == "ESTIMATOR_STATUS" else value.get("time_boot_ms")
            if type(stamp) is not int or stamp <= 0:
                return self._record_decision(diagnostic_key, "invalid_source_timestamp")
            stamp_us = stamp if kind == "ESTIMATOR_STATUS" else stamp * 1000
            if not self.timing.enabled:
                accepted, reason = self.relative_time.assess(kind, stamp_us, now)
                self.fault = self.relative_time.fault
                if not accepted:
                    self.received.pop(kind, None)
                    return self._record_decision(diagnostic_key, reason)
        if kind == "ATTITUDE_QUATERNION":
            q = [value.get(k) for k in ("q1", "q2", "q3", "q4")]
            rates = [value.get(k) for k in ("rollspeed", "pitchspeed", "yawspeed")]
            if not finite(q + rates) or abs(sum(v*v for v in q) - 1.) > .02:
                self.received.pop(kind, None)
                return self._record_decision(diagnostic_key, "invalid_attitude")
        if kind == "LOCAL_POSITION_NED" and not finite([value.get(k) for k in ("x", "y", "z", "vx", "vy", "vz")]):
            self.received.pop(kind, None)
            return self._record_decision(diagnostic_key, "invalid_position")
        if kind == "ESTIMATOR_STATUS" and type(value.get("flags")) is not int:
            self.received.pop(kind, None)
            return self._record_decision(diagnostic_key, "invalid_estimator_flags")
        if self.timing.enabled and kind in KINDS:
            now_ns = round(now * NS)
            decision = self.timing.evaluate(kind=kind, source_ns=stamp_us * 1000,
                sender=self.source, received_ns=now_ns if received_ns is None else received_ns,
                now_ns=now_ns)
            self.fault = self.timing.mapper.fault
            if not decision.accepted:
                self.received.pop(kind, None)
                return self._record_decision(diagnostic_key, decision.reason)
            self.stamps[kind] = stamp_us
            self.timing_sequence += 1
            self.timing_sequences[kind] = self.timing_sequence
        if kind == "AUTOPILOT_VERSION":
            self.timing.observe_version(value)
            self.fault = self.timing.mapper.fault
            if self.fault:
                return self._record_decision(diagnostic_key, self.fault)
        self.samples[kind] = dict(value)
        self.received[kind] = now
        self.counts[kind] = self.counts.get(kind, 0) + 1
        return self._record_decision(diagnostic_key)

    def fresh(self, kind, now, lifetime=.25):
        receipt_fresh = not self.fault and 0 <= now - self.received.get(kind, float("-inf")) < lifetime
        if self.timing.enabled and kind in KINDS:
            valid = self.timing.mapper.evidence_fresh(self.timing.mapper.latest.get(kind), round(now * NS))
            self.fault = self.timing.mapper.fault
            return receipt_fresh and valid
        return receipt_fresh

    def timing_evidence(self, kind, now, evidence_clock_id):
        """An immutable PUBLICATION bound, never an exposure or IMU rate bound."""
        if (not self.timing.enabled or type(evidence_clock_id) is not str
                or not evidence_clock_id or len(evidence_clock_id) > 256
                or kind not in KINDS or not self.fresh(kind, now)):
            return None
        evidence = self.timing.mapper.latest[kind]
        targeted = self.timing.protocol == "targeted_v2"
        return {"schema_version": 1, "mode": "OBSERVE_ONLY",
                "policy": "timesync_v2" if targeted else "legacy_correlated",
                "selected_protocol": self.timing.protocol,
                "response_target_checked": targeted,
                "legacy_exclusive_link_attested": self.timing.legacy_exclusive_link_attested,
                "link_identity": self.timing.link_identity,
                "source": "physical_px4_usb_mavlink", "system_id": self.source[0],
                "component_id": self.source[1], "session_id": self.timing.session_id,
                "evidence_clock_id": evidence_clock_id, "sequence": self.timing_sequences[kind],
                "kind": kind, "source_timestamp_us": evidence.source_ns // 1000,
                "received_monotonic_ns": evidence.received_ns,
                "evaluated_monotonic_ns": evidence.evaluated_ns,
                "source_local_earliest_ns": evidence.source_local_earliest_ns,
                "source_local_latest_ns": evidence.source_local_latest_ns,
                "evidence_expires_monotonic_ns": evidence.expires_ns,
                "sync_expires_monotonic_ns": evidence.sync_expires_ns,
                "offset_midpoint_ns": evidence.offset_midpoint_ns,
                "offset_uncertainty_ns": evidence.offset_uncertainty_ns,
                "acquisition_time_proven": False,
                "timing_basis": "TIMESYNC_PUBLICATION_BOUND_NOT_EXPOSURE"}

    def position_reason(self, now):
        if self.fault:
            return self.fault
        for kind, name, age in (("HEARTBEAT", "heartbeat", 2.5),
                                ("ATTITUDE_QUATERNION", "attitude", .25),
                                ("LOCAL_POSITION_NED", "position", .25),
                                ("ESTIMATOR_STATUS", "estimator_status", .25)):
            if not self.fresh(kind, now, age):
                return name + "_missing_or_stale"
        reason = estimator_reason(self.samples["ESTIMATOR_STATUS"]["flags"])
        if reason:
            return reason
        stamps = [self.stamps[k] for k in ("ATTITUDE_QUATERNION", "LOCAL_POSITION_NED", "ESTIMATOR_STATUS")]
        if max(stamps) - min(stamps) > 75000:
            return "fc_sample_time_skew"
        return ""

    def attitude(self, now):
        if not self.fresh("HEARTBEAT", now, 2.5) or not self.fresh("ATTITUDE_QUATERNION", now):
            return None
        v = self.samples["ATTITUDE_QUATERNION"]
        return {"timestamp": self.stamps["ATTITUDE_QUATERNION"],
                "q": [v[k] for k in ("q1", "q2", "q3", "q4")],
                "xyz": [v[k] for k in ("rollspeed", "pitchspeed", "yawspeed")]}

    def position(self, now):
        if not self.fresh("LOCAL_POSITION_NED", now):
            return None
        v = self.samples["LOCAL_POSITION_NED"]
        result = {k: v[k] for k in ("x", "y", "z", "vx", "vy", "vz")}
        reason = self.position_reason(now)
        result.update(timestamp=self.stamps["LOCAL_POSITION_NED"], valid=not reason, reason=reason)
        att = self.attitude(now)
        result["heading"] = 0.
        if att:
            w, x, y, z = att["q"]
            result["heading"] = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        return result

    def diagnostic(self, now):
        att = self.attitude(now)
        position = self.position(now)
        version = self.samples.get("AUTOPILOT_VERSION", {}).get("flight_sw_version")
        angles = None
        if att:
            w, x, y, z = att["q"]
            angles = [math.degrees(math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))),
                      math.degrees(math.asin(max(-1., min(1., 2*(w*y-z*x))))),
                      math.degrees(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))]
        return {"mode": "OBSERVE_ONLY", "flight_commands_enabled": False,
                "source": "physical_px4_usb_mavlink", "system_id": self.source[0],
                "component_id": self.source[1], "heartbeat_received": self.fresh("HEARTBEAT", now, 2.5),
                "attitude_received": att is not None, "position_received": position is not None,
                "position_valid": not self.position_reason(now), "reason": self.position_reason(now),
                "attitude_rpy_deg": angles, "angular_velocity_rad_s": att["xyz"] if att else None,
                "position_ned_m": [position[k] for k in ("x", "y", "z")] if position else None,
                "estimator_flags": self.samples.get("ESTIMATOR_STATUS", {}).get("flags") if self.fresh("ESTIMATOR_STATUS", now) else None,
                "gps_fix_type": self.samples.get("GPS_RAW_INT", {}).get("fix_type") if self.fresh("GPS_RAW_INT", now, 2.) else None,
                "firmware": f'{version >> 24}.{(version >> 16) & 255}.{(version >> 8) & 255}' if type(version) is int else None,
                "timing": self.timing.diagnostic(round(now * NS)),
                "accepted_counts": dict(self.counts), "valid_for_s": .25,
                "rx_diagnostics": {
                    "scope": "Parsed messages presented to TelemetryState.accept; not USB bytes. Metadata does not grant freshness or control authority.",
                    "observed_monotonic_s": diagnostic_number(now),
                    "raw_counts": dict(self.rx_counts),
                    "rejected_counts": {kind: dict(counts) for kind, counts in self.rejected_counts.items()},
                    "last_received": {kind: dict(value) for kind, value in self.last_rx.items()},
                    "last_rejected": {kind: dict(value) for kind, value in self.last_rejected.items()},
                    "accepted_receipt_age_s": {kind: diagnostic_number(now-receipt)
                                               for kind, receipt in self.received.items()},
                    "limits_unchanged": {"source_offset_excess_s": .20,
                                         "source_clock_jump_s": .25,
                                         "telemetry_receipt_lease_s": .25,
                                         "heartbeat_receipt_lease_s": 2.5}}}

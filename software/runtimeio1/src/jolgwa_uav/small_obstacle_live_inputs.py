"""Read-only live-input audit for the offline small-obstacle guard.

Missing flight state / ground reference / full policy evidence remain missing.
This first input audit intentionally cannot authorize or enter an active demo.
The guard receives missing inputs, NOT the matched telemetry shown in the audit.
"""
from collections import Counter, OrderedDict
from dataclasses import asdict
import json
import math

from .small_obstacle_demo_guard import SmallObstacleDemoGuard
from .observer_fc_telemetry import estimator_reason

NS = 1_000_000_000
POSE = "LOCAL_POSITION_NED"
ATT = "ATTITUDE_QUATERNION"
EST = "ESTIMATOR_STATUS"
KINDS = (POSE, ATT, EST)
MAX_MESSAGE_BYTES = 65536


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def natural(value):
    return type(value) is int and 0 < value < 2**64


def decode(value):
    if type(value) is str:
        if len(value.encode("utf-8")) > MAX_MESSAGE_BYTES:
            raise ValueError("message_too_large")
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("object_required")
    encoded = json.dumps(value, allow_nan=False)
    if len(encoded.encode("utf-8")) > MAX_MESSAGE_BYTES:
        raise ValueError("message_too_large")
    return json.loads(encoded)  # detach caller-owned mutable storage


class LiveInputs:
    def __init__(self, clock_id):
        if not isinstance(clock_id, str) or not clock_id.startswith("linux-monotonic:"):
            raise ValueError("same_host_linux_clock_identity_required")
        self.clock_id = clock_id
        self.guard = SmallObstacleDemoGuard()
        self.epoch = None
        self.report_session = None
        self.fatal_reason = ""
        self.counts = Counter()
        self.errors = {}
        self.evidence = {kind: OrderedDict() for kind in KINDS}
        self.highest_stamp = {kind: 0 for kind in KINDS}
        self.poses = {POSE: OrderedDict(), ATT: OrderedDict()}
        self.status = None
        self.report = None
        self.last_tick_ns = None

    def _bind_epoch(self, session, link):
        if not isinstance(session, str) or not session or not isinstance(link, str) or not link:
            raise ValueError("source_epoch_missing")
        epoch = (session, link)
        if self.epoch is not None and epoch != self.epoch:
            self.fatal_reason = "source_epoch_changed_restart_shadow_required"
            raise ValueError(self.fatal_reason)
        self.epoch = epoch

    @staticmethod
    def _put(cache, key, value):
        cache[key] = value
        while len(cache) > 32:
            cache.popitem(last=False)

    def receive(self, channel, value, now_ns):
        """Callbacks store only detached, bounded data; never refresh a lease."""
        if channel not in ("position", "attitude", "timing", "status", "report"):
            raise ValueError("unsupported_channel")
        self.counts["received_" + channel] += 1
        if self.fatal_reason:
            return False
        try:
            if not natural(now_ns):
                raise ValueError("receipt_clock_invalid")
            data = decode(value)
            if channel in ("position", "attitude"):
                kind = POSE if channel == "position" else ATT
                stamp = data.get("timestamp_us")
                if not natural(stamp):
                    raise ValueError("pose_source_stamp_invalid")
                fields = ("x", "y", "z", "vx", "vy", "vz", "heading") if kind == POSE else ("qw", "qx", "qy", "qz")
                if not all(finite(data.get(k)) for k in fields):
                    raise ValueError("pose_nonfinite")
                if kind == POSE and not all(type(data.get(k)) is bool for k in ("xy_valid", "z_valid", "v_xy_valid", "v_z_valid")):
                    raise ValueError("pose_validity_missing")
                if kind == ATT and abs(sum(data[k]**2 for k in fields)-1.) > .02:
                    raise ValueError("attitude_norm_invalid")
                previous = self.poses[kind].get(stamp)
                if previous is not None:
                    if previous != data:
                        raise ValueError("same_stamp_changed_pose")
                    self.counts["duplicate_" + channel] += 1
                    return False
                if self.poses[kind] and stamp < max(self.poses[kind]):
                    raise ValueError("reordered_pose")
                self._put(self.poses[kind], stamp, data)
            elif channel == "timing":
                kind = data.get("kind")
                if kind not in KINDS or data.get("evidence_clock_id") != self.clock_id:
                    raise ValueError("timing_kind_or_clock_mismatch")
                if (data.get("mode") != "OBSERVE_ONLY" or data.get("source") != "physical_px4_usb_mavlink"
                        or data.get("system_id") != 1 or data.get("component_id") != 1
                        or data.get("selected_protocol") != "legacy_correlated"
                        or data.get("response_target_checked") is not False
                        or data.get("acquisition_time_proven") is not False):
                    raise ValueError("timing_identity_or_scope_invalid")
                self._bind_epoch(data.get("session_id"), data.get("link_identity"))
                keys = ("sequence", "source_timestamp_us", "received_monotonic_ns", "evaluated_monotonic_ns",
                        "source_local_earliest_ns", "source_local_latest_ns", "evidence_expires_monotonic_ns", "sync_expires_monotonic_ns")
                if not all(natural(data.get(k)) for k in keys):
                    raise ValueError("timing_bounds_missing")
                earliest, latest = data["source_local_earliest_ns"], data["source_local_latest_ns"]
                received, evaluated = data["received_monotonic_ns"], data["evaluated_monotonic_ns"]
                expiry = data["evidence_expires_monotonic_ns"]
                if not (earliest <= latest and earliest <= received <= evaluated <= now_ns < expiry
                        <= min(earliest+250_000_000, received+250_000_000, data["sync_expires_monotonic_ns"])
                        and evaluated-earliest <= 200_000_000):
                    raise ValueError("original_timing_bound_invalid_or_expired")
                stamp = data["source_timestamp_us"]
                previous = self.evidence[kind].get(stamp)
                if previous is not None:
                    if previous != data:
                        raise ValueError("same_sample_evidence_changed")
                    self.counts["duplicate_timing"] += 1
                    return False
                if stamp <= self.highest_stamp[kind]:
                    raise ValueError("reordered_timing_sample")
                self.highest_stamp[kind] = stamp
                self._put(self.evidence[kind], stamp, data)
            elif channel == "status":
                timing = data.get("timing", {})
                if (data.get("mode") != "OBSERVE_ONLY" or data.get("flight_commands_enabled") is not False
                        or data.get("source") != "physical_px4_usb_mavlink" or data.get("system_id") != 1
                        or data.get("component_id") != 1 or not isinstance(timing, dict)):
                    raise ValueError("status_scope_invalid")
                self._bind_epoch(timing.get("session_id"), timing.get("link_identity"))
                observed = data.get("rx_diagnostics", {}).get("observed_monotonic_s")
                expiry = data.get("publication_expires_monotonic_ns")
                if not finite(observed) or not natural(expiry) or not 0 <= observed*NS <= now_ns < expiry <= observed*NS+250_000_001:
                    raise ValueError("original_status_expired_or_invalid")
                if self.status is not None and observed <= self.status["rx_diagnostics"]["observed_monotonic_s"]:
                    raise ValueError("duplicate_or_reordered_status")
                self.status = data
            else:
                if data.get("mode") != "OBSERVE_ONLY" or data.get("flight_commands_enabled") is not False:
                    raise ValueError("report_scope_invalid")
                session, sequence = data.get("session"), data.get("sequence")
                if not isinstance(session, str) or not session or not natural(sequence):
                    raise ValueError("report_identity_missing")
                if self.report_session is not None and session != self.report_session:
                    self.fatal_reason = "observer_session_changed_restart_shadow_required"
                    raise ValueError(self.fatal_reason)
                self.report_session = session
                published, lifetime = data.get("published_monotonic_s"), data.get("valid_for_s")
                if not finite(published) or not finite(lifetime) or not 0 <= published*NS <= now_ns or not 0 <= lifetime <= .5:
                    raise ValueError("report_clock_or_lease_invalid")
                if self.report is not None and sequence <= self.report["sequence"]:
                    raise ValueError("duplicate_or_reordered_report")
                self.report = data
            self.errors.pop(channel, None)
            self.counts["accepted_" + channel] += 1
            return True
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
            self.errors[channel] = str(exc)[:160]
            self.counts["rejected_" + channel] += 1
            # Never continue displaying a previously good stream after bad input.
            if channel in ("position", "attitude"):
                self.poses[POSE if channel == "position" else ATT].clear()
            elif channel == "status":
                self.status = None
            elif channel == "report":
                self.report = None
            elif channel == "timing":
                for cache in self.evidence.values():
                    cache.clear()
            return False

    def _matched(self, kind, now_ns):
        cache = self.poses[kind]
        if not cache:
            return None
        stamp = max(cache)
        proof = self.evidence[kind].get(stamp)
        if proof is None or not proof["evaluated_monotonic_ns"] <= now_ns < proof["evidence_expires_monotonic_ns"]:
            return None
        return {"value": cache[stamp], "original_evidence": proof}

    def snapshot(self, now_ns):
        if not natural(now_ns):
            raise ValueError("snapshot_clock_invalid")
        if self.last_tick_ns is not None and now_ns < self.last_tick_ns:
            self.fatal_reason = "local_monotonic_clock_regressed"
        self.last_tick_ns = now_ns
        pos, att = self._matched(POSE, now_ns), self._matched(ATT, now_ns)
        status = self.status if self.status is not None and now_ns < self.status["publication_expires_monotonic_ns"] else None
        est = next(reversed(self.evidence[EST].values()), None) if self.evidence[EST] else None
        if est is not None and now_ns >= est["evidence_expires_monotonic_ns"]:
            est = None
        missing = ["actual_armed_state", "actual_airborne_or_landed_state", "pilot_presence_lease",
                   "same_origin_measured_ground_reference", "fc_local_origin_reset_epoch",
                   "full_policy_velocity_and_configuration_binding", "decision_original_input_binding_and_deadline"]
        if pos is None: missing.append("fresh_position_exact_source_evidence")
        if att is None: missing.append("fresh_attitude_exact_source_evidence")
        if est is None: missing.append("fresh_estimator_evidence")
        if status is None: missing.append("fresh_fc_status_original_lease")
        valid = bool(not self.fatal_reason and pos and att and est and status and status.get("position_valid") is True
                     and status.get("heartbeat_received") is True and status.get("timing", {}).get("ready") is True
                     and status.get("reason") == "" and status.get("connection_error") == ""
                     and estimator_reason(status.get("estimator_flags")) == ""
                     and all(pos["value"][key] is True for key in ("xy_valid", "z_valid", "v_xy_valid", "v_z_valid")))
        if pos and att and est:
            stamps = (pos["value"]["timestamp_us"], att["value"]["timestamp_us"], est["source_timestamp_us"])
            valid = valid and max(stamps)-min(stamps) <= 75000
        if not valid: missing.append("valid_matched_navigation_dependencies")
        report = self.report
        report_current = bool(not self.fatal_reason and report and report.get("assessment") != "UNKNOWN"
                              and now_ns < (report["published_monotonic_s"]+report["valid_for_s"])*NS)
        if not report_current: missing.append("current_positive_observer_report")
        # Deliberately no Telemetry/DecisionEvidence fabricated from partial data.
        intent = self.guard.tick(now_s=now_ns/NS, telemetry=None, evidence=None)
        return {"schema_version": 1, "mode": "SHADOW_READONLY", "flight_commands_enabled": False,
                "automatic_ready": False, "guard_inputs_complete": False,
                "guard_evaluation_scope": "MISSING_INPUT_REJECTION_ONLY",
                "actual_guard_telemetry_supplied": False, "actual_guard_decision_supplied": False,
                "observed_monotonic_ns": now_ns, "evidence_clock_id": self.clock_id,
                "fixed_fc_epoch": list(self.epoch) if self.epoch else None,
                "fixed_observer_session": self.report_session, "terminal_reason": self.fatal_reason,
                "guard": asdict(intent), "missing_inputs": missing, "input_errors": dict(self.errors),
                "matched_position": pos, "matched_attitude": att, "navigation_dependencies_valid": valid,
                "estimator_evidence": est, "fc_status": None if status is None else {
                    "reason": status.get("reason"), "position_valid": status.get("position_valid"),
                    "estimator_flags": status.get("estimator_flags"),
                    "observed_monotonic_s": status["rx_diagnostics"]["observed_monotonic_s"],
                    "publication_expires_monotonic_ns": status["publication_expires_monotonic_ns"]},
                "observer_report": None if report is None else {k: report.get(k) for k in
                    ("assessment", "reason", "session", "sequence", "published_monotonic_s", "valid_for_s",
                     "metrics", "geometry", "input", "time_basis")},
                "observer_report_current_receipt_approximation": report_current,
                "counts": dict(self.counts), "scope": "REAL_INPUT_AUDIT_NOT_AUTOMATIC_FLIGHT_VALIDATION"}

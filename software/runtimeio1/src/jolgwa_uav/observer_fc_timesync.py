"""Observe-only, explicitly selected TIMESYNC source-time bounds. No I/O.

The mapper never sets an OS/FC clock. A separate request-only transport is
permitted only after an operator-proven exact firmware build and selected local
dialect evidence match. Legacy correlation additionally requires an exclusive
USB-link attestation; it is never a fallback and never invents target fields.
The supported relative drift bound is 100 ppm; this is not exposure-time proof.
"""
from dataclasses import dataclass
from copy import deepcopy
import re
import uuid


NS = 1_000_000_000
KINDS = frozenset({"ATTITUDE_QUATERNION", "LOCAL_POSITION_NED", "ESTIMATOR_STATUS"})
PROTOCOLS = frozenset({"targeted_v2", "legacy_correlated"})


def valid_ns(value):
    return type(value) is int and 0 < value < (1 << 63)


def valid_identity(value):
    return (isinstance(value, (tuple, list)) and len(value) == 2
            and all(type(v) is int and 1 <= v <= 255 for v in value))


@dataclass(frozen=True)
class Request:
    tc1: int
    ts1: int
    target_system: int | None
    target_component: int | None


@dataclass(frozen=True)
class Evidence:
    epoch: str
    kind: str
    source_ns: int
    received_ns: int
    evaluated_ns: int
    source_local_earliest_ns: int
    source_local_latest_ns: int
    age_upper_ns: int
    expires_ns: int
    sync_expires_ns: int
    offset_midpoint_ns: int
    offset_uncertainty_ns: int


@dataclass(frozen=True)
class Decision:
    accepted: bool
    reason: str
    evidence: Evidence | None = None


class ClockMapper:
    """Rejected epochs require a NEW explicit observe session.

    Sync-specific constants are fixed, not caller-expandable. The
    existing 200 ms source/backlog gate and 250 ms pose lease are not increased.
    THREE replies must be consistent over >=0.5 s before any source is accepted.
    Default strict protocol-v2 addressing; legacy must be explicitly selected.
    """
    RTT_MAX_NS = 50_000_000
    SYNC_LEASE_NS = 2 * NS
    SOURCE_AGE_MAX_NS = 200_000_000
    POSE_LEASE_NS = 250_000_000
    DRIFT_PPM = 100  # relative error, not 100 ppm for EACH oscillator
    REMOTE_QUANTUM_NS = 1_000  # PX4 hrt_absolute_time() has microsecond units
    MIN_REPLIES = 3
    MIN_WARMUP_NS = 500_000_000

    def __init__(self, *, epoch: str, remote=(1, 1), local=(245, 191), protocol="targeted_v2"):
        if type(epoch) is not str or not epoch.strip():
            raise ValueError("explicit clock/link session epoch required")
        if not valid_identity(remote) or not valid_identity(local):
            raise ValueError("non-broadcast fixed component identities required")
        if tuple(remote) == tuple(local):
            raise ValueError("distinct system/component pairs required")
        if type(protocol) is not str or protocol not in PROTOCOLS:
            raise ValueError("explicit supported TIMESYNC protocol required")
        if protocol == "legacy_correlated" and (tuple(remote) != (1, 1) or tuple(local) != (245, 191)):
            raise ValueError("legacy correlation requires fixed physical USB component identities")
        self._protocol = protocol
        self.epoch, self.remote, self.local = epoch, tuple(remote), tuple(local)
        self.fault = ""
        self.last_now_ns = 0
        self.last_nonce = 0
        self.pending = None
        self.anchor_remote_ns = None
        self.offset_low_ns = self.offset_high_ns = 0
        self.sync_expires_ns = 0
        self.first_sync_sent_ns = None
        self.sync_count = 0
        self.ready = False
        self.stamps = {}
        self.latest = {}

    @property
    def protocol(self):
        return self._protocol

    def _fail(self, reason):
        if not self.fault:
            self.fault = reason
        self.ready = False
        self.pending = None
        self.latest.clear()
        return False

    def _clock(self, now_ns):
        if self.fault:
            return False
        if not valid_ns(now_ns):
            return self._fail("invalid_local_clock")
        if now_ns < self.last_now_ns:
            return self._fail("local_clock_regressed")
        self.last_now_ns = now_ns
        # Check OLD lease before a late reply could make it appear fresh again.
        if self.anchor_remote_ns is not None and now_ns >= self.sync_expires_ns:
            return self._fail("sync_lease_expired_restart_required")
        return True

    @classmethod
    def _drift_margin(cls, remote_delta_ns):
        # If dR/dL is in [1-d, 1+d], |delta(L-R)| <= |delta R|*d/(1-d).
        denominator = 1_000_000 - cls.DRIFT_PPM
        return (abs(remote_delta_ns) * cls.DRIFT_PPM + denominator - 1) // denominator

    def issue_request(self, send_ns):
        """Return request DATA; this method transmits nothing.

        ts1 is the actual send-boundary monotonic timestamp and unique echoed
        request token. Integration must sample before serial write, not after.
        """
        if not self._clock(send_ns):
            return None
        if send_ns <= self.last_nonce:
            return None
        if self.pending is not None and send_ns - self.pending < self.RTT_MAX_NS:
            return None
        self.pending = self.last_nonce = send_ns
        targets = self.remote if self.protocol == "targeted_v2" else (None, None)
        return Request(0, send_ns, *targets)

    def accept_reply(self, *, tc1, ts1, sender, target=None, receive_ns, now_ns=None):
        now_ns = receive_ns if now_ns is None else now_ns
        if not self._clock(now_ns):
            return Decision(False, self.fault)
        sender_valid = valid_identity(sender) and tuple(sender) == self.remote
        target_valid = (valid_identity(target) and tuple(target) == self.local
                        if self.protocol == "targeted_v2" else target is None)
        if not sender_valid or not target_valid:
            return Decision(False, "unexpected_source_or_target")
        if not valid_ns(tc1) or not valid_ns(ts1):
            return Decision(False, "invalid_reply_timestamp")
        if not valid_ns(receive_ns) or receive_ns > now_ns:
            return Decision(False, "invalid_reply_receipt")
        if self.pending is None or ts1 != self.pending:
            return Decision(False, "unmatched_or_consumed_nonce")
        self.pending = None  # a correlated reply is SINGLE USE, including failure
        rtt = receive_ns - ts1
        if not 0 < rtt <= self.RTT_MAX_NS or now_ns - ts1 > self.RTT_MAX_NS:
            return Decision(False, "reply_rtt_excess_or_invalid")
        if self.anchor_remote_ns is not None and tc1 <= self.anchor_remote_ns:
            self._fail("remote_clock_regressed_or_duplicate_restart_required")
            return Decision(False, self.fault)

        # Reply timestamp was sampled somewhere between local send and receive.
        low = ts1 - tc1 - self.REMOTE_QUANTUM_NS
        high = receive_ns - tc1 + self.REMOTE_QUANTUM_NS
        if self.anchor_remote_ns is not None:
            margin = self._drift_margin(tc1 - self.anchor_remote_ns)
            low = max(low, self.offset_low_ns - margin)
            high = min(high, self.offset_high_ns + margin)
            if low > high:
                self._fail("offset_interval_discontinuous_restart_required")
                return Decision(False, self.fault)
        self.anchor_remote_ns = tc1
        self.offset_low_ns, self.offset_high_ns = low, high
        # Do not add a fresh 2 s at callback/processing time: anchor is SEND time.
        self.sync_expires_ns = ts1 + self.SYNC_LEASE_NS
        if self.first_sync_sent_ns is None:
            self.first_sync_sent_ns = ts1
        self.sync_count = min(self.MIN_REPLIES, self.sync_count + 1)
        self.ready = (self.sync_count >= self.MIN_REPLIES
                      and ts1 - self.first_sync_sent_ns >= self.MIN_WARMUP_NS)
        return Decision(True, "synchronized" if self.ready else "warming_up")

    def evaluate(self, *, kind, source_ns, sender, received_ns, now_ns):
        """Map a PUBLICATION timestamp. Does not validate estimator/pose content.

        An accepted evidence's absolute deadline is immutable. Consumers must
        carry it end-to-end instead of assigning new freshness at ROS receipt.
        """
        if not self._clock(now_ns):
            return Decision(False, self.fault)
        if (type(kind) is not str or kind not in KINDS or not valid_identity(sender)
                or tuple(sender) != self.remote):
            return Decision(False, "unexpected_telemetry_source_or_kind")
        if not valid_ns(source_ns) or not valid_ns(received_ns) or received_ns > now_ns:
            return Decision(False, "invalid_telemetry_timestamp")
        if not self.ready:
            return Decision(False, "sync_not_ready")
        previous = self.stamps.get(kind, 0)
        if source_ns < previous:
            self._fail("telemetry_clock_regressed_restart_required")
            return Decision(False, self.fault)
        if source_ns == previous:
            return Decision(False, "duplicate_source_timestamp")
        self.stamps[kind] = source_ns
        self.latest.pop(kind, None)
        quantum = 1_000 if kind == "ESTIMATOR_STATUS" else 1_000_000
        # Source msg is truncated to milliseconds/microseconds; include its full
        # quantization interval. Bounds use floor/ceil conservatively throughout.
        distance = max(abs(source_ns - self.anchor_remote_ns),
                       abs(source_ns + quantum - self.anchor_remote_ns))
        drift = self._drift_margin(distance)
        earliest = source_ns + self.offset_low_ns - drift
        latest = source_ns + quantum + self.offset_high_ns + drift
        if earliest > received_ns:
            self._fail("source_clock_future_restart_required")
            return Decision(False, self.fault)
        age_upper = now_ns - earliest
        if age_upper > self.SOURCE_AGE_MAX_NS:
            return Decision(False, "source_age_upper_excess")
        if now_ns - received_ns >= self.POSE_LEASE_NS:
            return Decision(False, "receipt_lease_expired")
        expires = min(earliest + self.POSE_LEASE_NS,
                      received_ns + self.POSE_LEASE_NS, self.sync_expires_ns)
        if now_ns >= expires:
            return Decision(False, "source_deadline_expired")
        evidence = Evidence(
            self.epoch, kind, source_ns, received_ns, now_ns, earliest, latest,
            age_upper, expires, self.sync_expires_ns,
            (self.offset_low_ns + self.offset_high_ns) // 2,
            (self.offset_high_ns - self.offset_low_ns + 1) // 2)
        self.latest[kind] = evidence
        return Decision(True, "fresh_bounded_publication", evidence)

    def synchronized(self, now_ns):
        """Runtime readiness query; raw `ready` is only a last-update snapshot."""
        return self._clock(now_ns) and self.ready

    def evidence_fresh(self, evidence, now_ns):
        """An extra sync reply never extends an older sample's expiration."""
        return (self._clock(now_ns) and isinstance(evidence, Evidence)
                and evidence.epoch == self.epoch
                and self.latest.get(evidence.kind) == evidence
                and evidence.evaluated_ns <= now_ns < evidence.expires_ns)


VERSION_UINT_FIELDS = {
    "capabilities": 64, "flight_sw_version": 32, "middleware_sw_version": 32,
    "os_sw_version": 32, "board_version": 32, "vendor_id": 16,
    "product_id": 16, "uid": 64,
}
VERSION_BYTE_FIELDS = {
    "flight_custom_version": 8, "middleware_custom_version": 8,
    "os_custom_version": 8, "uid2": 18,
}


def bounded_autopilot_version(value):
    """Fixed AUTOPILOT_VERSION fields only; absent/invalid values stay unknown."""
    result, validity = {}, {}
    for name, bits in VERSION_UINT_FIELDS.items():
        item = value.get(name)
        valid = type(item) is int and 0 <= item < (1 << bits)
        result[name] = item if valid else None
        validity[name] = valid
    for name, size in VERSION_BYTE_FIELDS.items():
        item = value.get(name)
        valid = (isinstance(item, (bytes, bytearray, list, tuple)) and len(item) == size
                 and all(type(n) is int and 0 <= n <= 255 for n in item))
        result[name + "_hex"] = bytes(item).hex() if valid else None
        validity[name] = valid
    result["field_validity"] = validity
    result["all_fields_valid"] = all(validity.values())
    return result


def inspect_timesync_dialect(common, *, package_version=None, file_sha256=None, protocol="targeted_v2"):
    """Inspect the installed encoder/decoder without sending any packet."""
    if type(protocol) is not str or protocol not in PROTOCOLS:
        raise ValueError("explicit supported TIMESYNC protocol required")
    cls = getattr(common, "MAVLink_timesync_message", None)
    names = list(getattr(cls, "fieldnames", ()))
    types = list(getattr(cls, "fieldtypes", ()))
    targeted = protocol == "targeted_v2"
    expected_names = ["tc1", "ts1"] + (["target_system", "target_component"] if targeted else [])
    expected_types = ["int64_t", "int64_t"] + (["uint8_t", "uint8_t"] if targeted else [])
    reason_prefix = "timesync_v2" if targeted else "legacy_correlated"
    wire = getattr(common, "WIRE_PROTOCOL_VERSION", None)
    module = getattr(common, "__name__", None)
    result = {
        "supported": False, "reason": reason_prefix + "_dialect_unavailable",
        "selected_protocol": protocol,
        "wire_protocol_version": wire if wire in ("1.0", "2.0") else None,
        "module": module if type(module) is str and re.fullmatch(r"[A-Za-z0-9_.]{1,128}", module) else None,
        "pymavlink_version": package_version if type(package_version) is str
        and re.fullmatch(r"[A-Za-z0-9_.+\-]{1,64}", package_version) else None,
        "module_sha256": file_sha256 if type(file_sha256) is str
        and re.fullmatch(r"[0-9a-f]{64}", file_sha256) else None,
        "timesync_message_id": 111 if getattr(common, "MAVLINK_MSG_ID_TIMESYNC", None) == 111 else None,
        "fieldnames": names if names == expected_names else [],
        "fieldtypes": types if types == expected_types else [],
        "target_fields_present": "target_system" in names and "target_component" in names,
        "selected_fields_match": names == expected_names and types == expected_types,
        "local_encode_decode_verified": False,
    }
    if wire != "2.0" or not result["selected_fields_match"] or result["timesync_message_id"] != 111:
        return result
    try:
        encoder = common.MAVLink(None, srcSystem=245, srcComponent=191)
        packet = (cls(0, 13, 1, 1) if targeted else cls(0, 13)).pack(encoder)
        messages = common.MAVLink(None).parse_buffer(packet)
        decoded = messages[0] if messages and len(messages) == 1 else None
        valid = (packet[0] == 0xFD and decoded is not None
                 and decoded.get_type() == "TIMESYNC"
                 and decoded.get_srcSystem() == 245 and decoded.get_srcComponent() == 191
                 and decoded.tc1 == 0 and decoded.ts1 == 13)
        valid = valid and (decoded.target_system == 1 and decoded.target_component == 1
                          if targeted else not hasattr(decoded, "target_system")
                          and not hasattr(decoded, "target_component"))
    except Exception:
        # This startup-only codec probe grants no authority on any failure.
        result["reason"] = reason_prefix + "_dialect_codec_probe_failed"
        return result
    result.update(supported=bool(valid), local_encode_decode_verified=bool(valid),
                  reason=reason_prefix + ("_dialect_verified" if valid else "_dialect_roundtrip_invalid"))
    return result


class ObserveTimesync:
    """Opt-in, exact-build-gated correlation. Never silently falls back to v1.

    Expected build values are OPERATOR ATTESTATION, not autonomous capability
    detection. Three actual replies of the explicitly selected protocol are
    required. Legacy correlation makes no targeted-response or authentication claim.
    """
    REQUEST_INTERVAL_NS = NS

    def __init__(self, *, enabled=False, expected_flight_sw_version=0,
                 expected_custom_version_hex="", session_id=None, protocol="targeted_v2",
                 legacy_exclusive_link_attested=False):
        if type(enabled) is not bool:
            raise ValueError("observe_timesync_enabled must be boolean")
        if type(protocol) is not str or protocol not in PROTOCOLS:
            raise ValueError("explicit supported TIMESYNC protocol required")
        if type(legacy_exclusive_link_attested) is not bool:
            raise ValueError("exclusive link attestation must be boolean")
        self._protocol = protocol
        self.legacy_exclusive_link_attested = legacy_exclusive_link_attested
        self.link_identity = None
        if type(expected_flight_sw_version) is not int or not 0 <= expected_flight_sw_version < (1 << 32):
            raise ValueError("expected flight_sw_version must be uint32")
        if (type(expected_custom_version_hex) is not str or
                (expected_custom_version_hex and not re.fullmatch(r"[0-9a-f]{16}", expected_custom_version_hex))):
            raise ValueError("expected custom version must be 8 wire bytes as lowercase hex")
        self.enabled = enabled
        self.expected_version = expected_flight_sw_version
        self.expected_custom = expected_custom_version_hex
        self.session_id = str(uuid.UUID(session_id)) if session_id is not None else str(uuid.uuid4())
        self.mapper = ClockMapper(epoch=self.session_id, protocol=protocol)
        self.version = None
        self._bound_build = None
        self.dialect = {"supported": False, "reason": "dialect_not_inspected"}
        self.last_request_ns = 0
        self.last_reply_reason = "not_requested"
        self.last_rtt_ns = None
        self.accepted_replies = 0
        self.rejected_replies = 0

    @property
    def protocol(self):
        return self._protocol

    def bind_link(self, identity, *, exclusive):
        """Bind one already-opened USB instance, not a remote authentication claim."""
        if (type(identity) is not str or not re.fullmatch(r"[A-Za-z0-9_./:\-]{1,256}", identity)
                or type(exclusive) is not bool or not exclusive):
            self.mapper._fail("exclusive_link_identity_invalid_restart_required")
            return False
        if self.link_identity is not None and identity != self.link_identity:
            self.mapper._fail("link_identity_changed_restart_required")
            return False
        self.link_identity = identity
        self._bind_build_if_allowed()
        return not self.mapper.fault

    def configure_dialect(self, common, **metadata):
        evidence = inspect_timesync_dialect(common, protocol=self.protocol, **metadata)
        if self.enabled and self.dialect.get("supported") and evidence != self.dialect:
            self.mapper._fail("dialect_changed_restart_required")
        self.dialect = evidence
        self._bind_build_if_allowed()

    def _bind_build_if_allowed(self):
        if self.enabled and not self.gate_reason() and self._bound_build is None:
            self._bound_build = tuple(self.version[k] for k in (
                "flight_sw_version", "flight_custom_version_hex", "vendor_id",
                "product_id", "uid", "uid2_hex"))

    def observe_version(self, value):
        evidence = bounded_autopilot_version(value)
        fingerprint = (evidence["flight_sw_version"], evidence["flight_custom_version_hex"],
                       evidence["vendor_id"], evidence["product_id"], evidence["uid"], evidence["uid2_hex"])
        if self.enabled and self._bound_build is not None and fingerprint != self._bound_build:
            self.mapper._fail("firmware_identity_changed_restart_required")
        self.version = evidence
        self._bind_build_if_allowed()

    def gate_reason(self):
        if not self.enabled:
            return "legacy_timing_selected"
        if self.mapper.fault:
            return self.mapper.fault
        if self.protocol == "legacy_correlated":
            if not self.legacy_exclusive_link_attested:
                return "legacy_exclusive_link_not_attested"
            if self.link_identity is None:
                return "legacy_exclusive_link_not_bound"
        if (self.expected_version == 0 or not self.expected_custom
                or self.expected_custom == "0000000000000000"):
            return ("firmware_v2_proof_not_configured" if self.protocol == "targeted_v2"
                    else "firmware_legacy_proof_not_configured")
        if not self.dialect.get("supported"):
            return self.dialect.get("reason", "timesync_v2_dialect_unavailable")
        if self.version is None:
            return "autopilot_version_missing"
        if (self.version["flight_sw_version"] != self.expected_version
                or self.version["flight_custom_version_hex"] != self.expected_custom):
            return ("firmware_v2_proof_mismatch" if self.protocol == "targeted_v2"
                    else "firmware_legacy_proof_mismatch")
        return ""

    def issue_request(self, now_ns):
        if self.gate_reason():
            return None
        if not self.mapper._clock(now_ns):
            return None
        if self.last_request_ns and now_ns - self.last_request_ns < self.REQUEST_INTERVAL_NS:
            return None
        request = self.mapper.issue_request(now_ns)
        if request is not None:
            self.last_request_ns = now_ns
        return request

    def accept_reply(self, value, sender, receive_ns, now_ns=None):
        reason = self.gate_reason()
        if reason:
            decision = Decision(False, reason)
        else:
            # Legacy must really lack target fields, not receive fabricated
            # local IDs through the targeted mapper or silently accept zeros.
            target = ((value.get("target_system"), value.get("target_component"))
                      if self.protocol == "targeted_v2" or "target_system" in value
                      or "target_component" in value else None)
            decision = self.mapper.accept_reply(
                tc1=value.get("tc1"), ts1=value.get("ts1"), sender=sender,
                target=target,
                receive_ns=receive_ns, now_ns=now_ns)
        self.last_reply_reason = decision.reason
        if decision.accepted:
            self.accepted_replies += 1
            self.last_rtt_ns = receive_ns - value["ts1"]
        else:
            self.rejected_replies += 1
        return decision

    def evaluate(self, **kwargs):
        reason = self.gate_reason()
        return Decision(False, reason) if reason else self.mapper.evaluate(**kwargs)

    def diagnostic(self, now_ns):
        reason = self.gate_reason()
        if not reason and not self.mapper.synchronized(now_ns):
            reason = self.mapper.fault or "timesync_warming_up"
        return {
            "enabled": self.enabled,
            "policy": ("timesync_v2" if self.protocol == "targeted_v2" else "legacy_correlated")
                      if self.enabled else "legacy_lifetime_min",
            "selected_protocol": self.protocol,
            "response_target_checked": self.protocol == "targeted_v2",
            "expected_responder_system": 1, "expected_responder_component": 1,
            "legacy_exclusive_link_attested": self.legacy_exclusive_link_attested,
            "link_identity": self.link_identity,
            "ready": self.enabled and not reason, "reason": reason,
            "session_id": self.session_id,
            "expected_build_is_operator_attestation": bool(self.expected_version and self.expected_custom),
            "expected_flight_sw_version": self.expected_version or None,
            "expected_flight_custom_version_hex": self.expected_custom or None,
            "actual_strict_correlated_replies": self.accepted_replies,
            "rejected_replies": self.rejected_replies, "last_reply_reason": self.last_reply_reason,
            "last_rtt_ns": self.last_rtt_ns,
            "sync_expires_monotonic_ns": self.mapper.sync_expires_ns or None,
            "offset_midpoint_ns": ((self.mapper.offset_low_ns+self.mapper.offset_high_ns)//2
                                   if self.mapper.anchor_remote_ns is not None else None),
            "offset_uncertainty_ns": ((self.mapper.offset_high_ns-self.mapper.offset_low_ns+1)//2
                                      if self.mapper.anchor_remote_ns is not None else None),
            "bounds": {"source_age_upper_ns": ClockMapper.SOURCE_AGE_MAX_NS,
                       "pose_lease_ns": ClockMapper.POSE_LEASE_NS,
                       "maximum_rtt_ns": ClockMapper.RTT_MAX_NS,
                       "sync_lease_ns": ClockMapper.SYNC_LEASE_NS,
                       "relative_drift_ppm": ClockMapper.DRIFT_PPM},
            "firmware": deepcopy(self.version), "dialect": deepcopy(self.dialect),
            "acquisition_time_proven": False,
            "timing_basis": "TIMESYNC_PUBLICATION_BOUND_NOT_EXPOSURE",
        }

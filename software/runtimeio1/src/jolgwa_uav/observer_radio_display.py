"""Receive-only radio display: source claims are NOT end-to-end live evidence."""
from copy import deepcopy
import math

from .observer_radio_protocol import decode_status, EXPECTED_SOURCE, TARGET, PAYLOAD_TYPE

WARNING = "수신한 Jetson 판단 / 전송 지연 미검증"


class RadioDisplay:
    def __init__(self, *, retired_session_limit=64):
        if type(retired_session_limit) is not int or not 1 <= retired_session_limit <= 256:
            raise ValueError("invalid_session_limit")
        self._retired_limit = retired_session_limit
        self._retired = set()
        self._session = None
        self._sequence = -1
        self._generated = float("-inf")
        self._now = float("-inf")
        self._received = None
        self._report = None
        self._historical = None
        self._fault = ""
        self._reason = "waiting_for_radio_status"
        self.accepted = self.rejected = 0

    def _check_time(self, now):
        if type(now) not in (int, float) or not math.isfinite(now) or now < 0 or now < self._now:
            self._fault = "receiver_clock_invalid_restart_required"
            self._report = None
            return False
        self._now = now
        return not self._fault

    def reject(self, reason):
        self.rejected += 1
        self._report = None
        self._reason = str(reason)[:256]
        return False

    def receive(self, payload, *, source, target, payload_type, received_s, mavlink2=True):
        if not self._check_time(received_s):
            return self.reject(self._fault)
        if (mavlink2 is not True or type(payload_type) is not int or payload_type != PAYLOAD_TYPE
                or type(source) is not tuple or len(source) != 2
                or type(target) is not tuple or len(target) != 2
                or any(type(v) is not int for v in (*source, *target))
                or source != EXPECTED_SOURCE or target != TARGET):
            return self.reject("unexpected_radio_envelope")
        try:
            report = decode_status(payload)
        except (ValueError, TypeError, OverflowError) as exc:
            return self.reject(str(exc))
        session = report["session"]
        if session in self._retired:
            return self.reject("retired_session_replay")
        if session == self._session:
            if report["sequence"] <= self._sequence or report["generated_s"] <= self._generated:
                return self.reject("duplicate_or_reordered_status")
        elif self._session is not None:
            if len(self._retired) >= self._retired_limit:
                self._fault = "session_history_full_restart_required"
                return self.reject(self._fault)
            self._retired.add(self._session)
            self._report = None
        self._session, self._sequence, self._generated = session, report["sequence"], report["generated_s"]
        report["session"] = session.hex()
        self._received = received_s
        self._report = report
        self._historical = dict(deepcopy(report), received_monotonic_s=received_s,
                                end_to_end_latency_verified=False, control_authorized=False)
        self._reason = "received_unverified_source_claim"
        self.accepted += 1
        return True

    def receive_message(self, message, *, received_s):
        """Consume a parsed common-v20 TUNNEL only; no encoding or send API."""
        if message.get_type() != "TUNNEL":
            return None  # QGC forwards unrelated telemetry too; it grants no status.
        try:
            size = message.payload_length
            if type(size) is not int or not 1 <= size <= 128:
                raise ValueError("invalid_tunnel_payload_length")
            data = bytes(message.payload)
            if len(data) != 128 or any(data[size:]):
                raise ValueError("invalid_tunnel_padding")
            return self.receive(data[:size], source=(message.get_srcSystem(), message.get_srcComponent()),
                target=(message.target_system, message.target_component), payload_type=message.payload_type,
                received_s=received_s, mavlink2=bytes(message.get_msgbuf())[:1] == b"\xfd")
        except (ValueError, TypeError, AttributeError, OverflowError) as exc:
            return self.reject(str(exc))

    def snapshot(self, now_s):
        self._check_time(now_s)
        valid_now = type(now_s) in (int, float) and math.isfinite(now_s)
        delta = None if self._received is None or not valid_now else now_s - self._received
        active = dict(assessment="UNKNOWN", reason=self._fault or self._reason, sensor={},
                      camera_ok=False, px4_pose_received=False)
        report = self._report
        if report is not None and not self._fault and delta is not None and 0 <= delta < 1.:
            active = deepcopy(report)
            # This only SHORTENS local display time. It cannot prove one-way age.
            if delta >= report["judgment_remaining_ms"] / 1000.:
                active.update(assessment="UNKNOWN", reason="source_reported_display_window_expired")
            if delta >= report["sensor_remaining_ms"] / 1000.:
                active.update(sensor={}, camera_ok=False)
            if delta >= report["judgment_remaining_ms"] / 1000.:
                active["px4_pose_received"] = False
        elif report is not None:
            active["reason"] = self._fault or "radio_receive_expired"
        return dict(schema_version=1, transport="qgc_loopback_udp_receive_only", warning=WARNING,
                    mode="DISPLAY_ONLY", control_authorized=False, flight_commands_enabled=False,
                    end_to_end_latency_verified=False, source_age_upper_bound_s=None,
                    source_identity_authenticated=False,
                    source_reported=active, historical_last_received=deepcopy(self._historical),
                    last_receipt_delta_s=delta if delta is not None and delta >= 0 else None,
                    accepted=self.accepted, rejected=self.rejected,
                    display_state="RECEIVED_UNVERIFIED" if report and not self._fault and delta is not None and 0 <= delta < 1 else "UNKNOWN")

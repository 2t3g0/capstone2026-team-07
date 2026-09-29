"""Physical USB owner: whitelisted telemetry requests and opt-in display TX."""
import json
import hashlib
import errno
from importlib import metadata
import time
import math
import threading
import uuid
from pathlib import Path
from .observer_fc_telemetry import request_parameters
from .observer_fc_timesync import ObserveTimesync, NS


def _allow_verified_pixhawk_acm_modem_control_epipe(port):
    """Tolerate only unsupported CDC modem-control requests on Pixhawk USB.

    Jetson's xHCI/cdc_acm stack can return EPIPE for TIOCMBIC/TIOCMBIS even
    while the ACM bulk endpoints remain healthy. MAVLink does not use DTR or
    RTS. The caller invokes this helper only after the ACM node's Pixhawk 6C
    USB VID/PID has been verified. All other exceptions still fail closed.
    """
    ignored = []
    for method_name in ("_update_dtr_state", "_update_rts_state"):
        original = getattr(port, method_name)

        def guarded(original=original, method_name=method_name):
            try:
                original()
            except OSError as exc:
                if exc.errno != errno.EPIPE:
                    raise
                ignored.append(method_name)

        setattr(port, method_name, guarded)
    port.jolgwa_ignored_modem_control_epipe = ignored


class UsbTelemetryTransport:
    def __init__(self, journal, device="/dev/jolgwa-pixhawk6c", *, timing=None):
        import serial
        from pymavlink.dialects.v20 import common
        self.timing = timing if timing is not None else ObserveTimesync()
        try:
            package_version = metadata.version("pymavlink")
        except metadata.PackageNotFoundError:
            package_version = None
        self.timing.configure_dialect(common, package_version=package_version,
            file_sha256=hashlib.sha256(Path(common.__file__).read_bytes()).hexdigest())
        # Prevent accidentally opening a network connection, SITL, or another USB model.
        path = Path(device).resolve(strict=True)
        if not path.name.startswith("ttyACM"):
            raise ValueError("not a USB ACM device")
        node = (Path('/sys/class/tty') / path.name / 'device').resolve(strict=True)
        for parent in (node, *node.parents):
            if (parent/'idVendor').exists():
                if ((parent/'idVendor').read_text().strip(), (parent/'idProduct').read_text().strip()) != ('3185', '0038'):
                    raise ValueError("not the expected Pixhawk 6C USB device")
                break
        else:
            raise ValueError("USB identity unavailable")
        self.port = serial.Serial(port=None, baudrate=921600, timeout=.01, write_timeout=.2, exclusive=True)
        _allow_verified_pixhawk_acm_modem_control_epipe(self.port)
        self.port.dtr = False
        self.port.rts = False
        self.port.port = str(path)
        try:
            self.port.open()
            self.port.reset_input_buffer()
            if self.timing.enabled and self.timing.protocol == "legacy_correlated":
                # A fresh open gets a NEW connection identity even at the same tty.
                # The caller separately attests no competing synchronizer/routing.
                identity = 'usb-exclusive:' + path.name + ':' + uuid.uuid4().hex
                if not self.timing.bind_link(identity, exclusive=True):
                    raise ValueError(self.timing.mapper.fault)
            self.parser = common.MAVLink(None)
            self.parser.robust_parsing = True
            self.encoder = common.MAVLink(None, srcSystem=245, srcComponent=191)
            self.common = common
            self.journal = journal
            ignored_controls = tuple(dict.fromkeys(
                self.port.jolgwa_ignored_modem_control_epipe))
            if ignored_controls:
                self.journal.write(json.dumps({
                    "event": "pixhawk_acm_modem_control_epipe_ignored",
                    "device": str(path),
                    "operations": list(ignored_controls),
                }, allow_nan=False) + "\n")
                self.journal.flush()
            self.tx_count = 0
            self.tx_radio_count = 0
            self._owner_thread = threading.get_ident()
            self._radio_encoder = None
            self.bad_data = 0
            # reset_input_buffer only flushes the host. The FC USB queue can still
            # contain many seconds of old telemetry from before the previous close.
            self.discard_until = time.monotonic()+1.
            self.startup_discarded_messages = 0
        except BaseException:
            # __init__ has not returned, so callers still hold transport=None.
            # Release this exclusively opened descriptor before propagating.
            if self.port.is_open:
                self.port.close()
            raise

    def read(self):
        strict_legacy = (getattr(self, 'timing', None) is not None
                         and self.timing.enabled and self.timing.protocol == 'legacy_correlated')
        previous_read_ns = getattr(self, 'last_read_monotonic_ns', None)
        try:
            data = self.port.read(min(max(self.port.in_waiting, 1), 8192))
        except Exception:
            if strict_legacy:
                self.timing.mapper._fail('usb_read_failed_restart_required')
            raise
        # One original receipt boundary for this read, before parsing/queue work.
        self.last_read_monotonic_ns = round(time.monotonic() * NS)
        messages = self.parser.parse_buffer(data) or []
        self.bad_data += sum(m.get_type() == 'BAD_DATA' for m in messages)
        backlog = len(data) == 8192 and self.port.in_waiting >= 8192
        if backlog:
            self.discard_until = max(self.discard_until,time.monotonic()+.2)
        if strict_legacy:
            gap_ns = (self.last_read_monotonic_ns - previous_read_ns
                      if previous_read_ns is not None else 0)
            if gap_ns < 0:
                self.timing.mapper._fail('local_clock_regressed')
            elif self.timing.mapper.anchor_remote_ns is not None and (
                    backlog or gap_ns >= self.timing.mapper.POSE_LEASE_NS):
                self.timing.mapper._fail('usb_backlog_or_read_gap_restart_required')
        if strict_legacy and self.timing.mapper.fault:
            self.startup_discarded_messages += len(messages)
            return []
        if time.monotonic() < self.discard_until:
            self.startup_discarded_messages += len(messages)
            return []
        return messages

    def request(self, command, message_id, interval=0, system_id=1, component_id=1):
        params = request_parameters(command, message_id, interval)
        if (system_id, component_id) != (1, 1):
            raise ValueError("unexpected FC target")
        message = self.common.MAVLink_command_long_message(system_id, component_id, command, 0, *params)
        packet = message.pack(self.encoder)
        # Record before transmission; a broken log must not allow unrecorded requests.
        self.journal.write(json.dumps({"kind": "tx_telemetry_request", "monotonic_s": time.monotonic(),
                                      "command": command, "message_id": message_id, "interval_us": interval,
                                      "packet_hex": packet.hex()})+'\n')
        self.journal.flush()
        count = self.port.write(packet)
        if count != len(packet):
            raise OSError("partial telemetry request")
        self.encoder.seq = (self.encoder.seq+1) % 256
        self.tx_count += 1

    def request_timesync(self):
        """Only tc1=0 requests of the explicit protocol; never clock replies.

        Exact build attestation and locally verified v2 codec are necessary but
        do not make the map ready. Three real correlated replies are required.
        Encoding/journal/write latency belongs to the ORIGINAL request budget.
        """
        timing = self.timing
        sent_ns = round(time.monotonic() * NS)
        if (timing.protocol == 'legacy_correlated'
                and sent_ns / NS < getattr(self, 'discard_until', 0.)):
            return False
        request = timing.issue_request(sent_ns)
        if request is None:
            return False
        try:
            targeted = timing.protocol == 'targeted_v2'
            message = (self.common.MAVLink_timesync_message(0, request.ts1, 1, 1) if targeted
                       else self.common.MAVLink_timesync_message(0, request.ts1))
            packet = message.pack(self.encoder)
            self.journal.write(json.dumps({"kind": "tx_timesync_request_only",
                "sent_monotonic_ns": sent_ns, "session_id": timing.session_id,
                "protocol": timing.protocol,
                "target_system": request.target_system, "target_component": request.target_component,
                "response_target_checked": targeted,
                "expected_responder_system": 1, "expected_responder_component": 1,
                "link_identity": timing.link_identity,
                "tc1": 0, "ts1": request.ts1, "packet_hex": packet.hex()})+'\n')
            self.journal.flush()
            before_write_ns = round(time.monotonic() * NS)
            if before_write_ns < sent_ns or before_write_ns - sent_ns >= timing.mapper.RTT_MAX_NS:
                raise OSError("TIMESYNC original request deadline expired before write")
            count = self.port.write(packet)
            if count != len(packet):
                raise OSError("partial TIMESYNC telemetry request")
        except Exception:
            # A transport/log failure must retire this evidence epoch before the
            # supervisor restarts; exception is never hidden or retried here.
            timing.mapper._fail("timesync_transport_failed_restart_required")
            raise
        self.encoder.seq = (self.encoder.seq+1) % 256
        self.tx_count += 1
        return True

    def send_observer_status(self, payload, *, expires_s, status_text=None):
        """Display-only TUNNEL/STATUSTEXT, written by the existing USB owner.

        No arbitrary MAVLink message, flight command or sensor input is accepted.
        Radio journal/encoding failures disable that optional feature upstream;
        partial/failed serial writes remain terminal to avoid an ambiguous stream.
        """
        from .observer_radio_protocol import decode_status
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError('observer radio write attempted outside USB owner')
        if not isinstance(payload, bytes) or not 1 <= len(payload) <= 128:
            raise ValueError('invalid observer radio payload')
        decode_status(payload)  # closed display schema, not an arbitrary tunnel
        if (type(expires_s) not in (float, int) or not math.isfinite(expires_s)):
            raise ValueError('invalid observer radio deadline')
        now = time.monotonic()
        if now < getattr(self, 'discard_until', 0.) or now >= expires_s:
            return False
        if getattr(self.port, 'out_waiting', 0):
            return False  # low priority: never add display backlog to flight IO
        if status_text is not None and (not isinstance(status_text, str)
                or not status_text.startswith('OBS source:')
                or len(status_text.encode('ascii', errors='strict')) > 50):
            raise ValueError('invalid observer status text')
        if self._radio_encoder is None:
            self._radio_encoder = self.common.MAVLink(None, srcSystem=1, srcComponent=191)
        encoder = self._radio_encoder
        message = self.common.MAVLink_tunnel_message(0, 0, 32768, len(payload),
                                                    list(payload.ljust(128, b'\0')))
        messages = [message]
        if status_text is not None:
            messages.append(self.common.MAVLink_statustext_message(6, status_text.encode('ascii')))
        packets = []
        sequence = encoder.seq
        for message in messages:
            if message.get_type() not in ('TUNNEL', 'STATUSTEXT'):
                raise ValueError('not a display-only MAVLink message')
            packets.append(message.pack(encoder, force_mavlink1=False))
            encoder.seq = (encoder.seq + 1) % 256
        encoder.seq = sequence
        # A failed log cannot produce unrecorded output. Do not renew the source
        # lease after encoding, filesystem latency, or an earlier serial write.
        self.journal.write(json.dumps({'kind': 'tx_observer_display_attempt',
            'monotonic_s': now, 'expires_monotonic_s': expires_s,
            'source_system': 1, 'source_component': 191,
            'message_types': [m.get_type() for m in messages],
            'packets_hex': [p.hex() for p in packets]}) + '\n')
        self.journal.flush()
        sent = False
        for packet in packets:
            before_write = time.monotonic()
            if before_write < now or before_write >= expires_s:
                return sent
            old_timeout = self.port.write_timeout
            try:
                # Bounded optional output; never retain the port's longer
                # telemetry-request timeout for a display-only message.
                self.port.write_timeout = min(.02, old_timeout) if old_timeout is not None else .02
                try:
                    count = self.port.write(packet)
                except Exception as exc:
                    raise ObserverRadioSerialError('observer radio serial write failed') from exc
                if count != len(packet):
                    raise ObserverRadioSerialError('partial observer radio serial write')
            finally:
                self.port.write_timeout = old_timeout
            encoder.seq = (encoder.seq + 1) % 256
            self.tx_radio_count += 1
            sent = True
        return sent

    def close(self):
        self.port.close()


class ObserverRadioSerialError(OSError):
    """Possible partial packet; do not continue a potentially corrupt USB stream."""

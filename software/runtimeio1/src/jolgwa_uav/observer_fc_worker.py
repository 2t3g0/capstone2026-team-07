"""Single-owner USB/TIMESYNC worker; immutable, latest-only ROS handoff.

No ROS imports, flight commands or added leases. Only this thread touches the
transport, TelemetryState, mapper and journal. A slow ROS publisher may drop
outputs but cannot stall serial reads by holding a shared state/IO lock.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading
import time
import uuid

from .observer_fc_telemetry import TelemetryState, STREAM_INTERVALS
from .observer_fc_timesync import KINDS, NS
from .observer_fc_transport import UsbTelemetryTransport, ObserverRadioSerialError


CHANNELS = ("ATTITUDE_QUATERNION", "LOCAL_POSITION_NED", "ESTIMATOR_STATUS", "status")


@dataclass(frozen=True)
class RadioStatusSnapshot:
    generated_s: float
    status_json: str


class ObserverRadioSender:
    """One immutable latest-only slot; ROS submits, only USB owner transmits.

    This is an opt-in display output, never a new source of flight authority.
    Queue and report leases remain original. Stale/malformed input becomes an
    explicitly UNKNOWN display record; it cannot keep an older green result.
    """
    def __init__(self, clock_id, *, clock=time.monotonic):
        if not isinstance(clock_id, str) or not clock_id:
            raise ValueError('observer radio requires local clock identity')
        self._clock_id, self._clock = clock_id, clock
        self._lock = threading.Lock()
        self._slot = None
        self._reason = 'waiting_for_observer_status'
        self._stopped = False
        self._disabled = False
        self._next_send = 0.
        self._last_generated = float('-inf')
        self._session = uuid.uuid4().bytes
        self._sequence = 0
        self._last_text_state = None
        self._next_text = 0.
        self.error = ''

    def submit(self, raw):
        from .observer_radio_protocol import encode_status
        try:
            if not isinstance(raw, str) or len(raw) > 16384 or len(raw.encode('utf-8')) > 16384:
                raise ValueError('observer_radio_input_size')
            envelope = json.loads(raw)
            if (not isinstance(envelope, dict) or type(envelope.get('schema_version')) is not int
                    or envelope['schema_version'] != 1 or envelope.get('clock_id') != self._clock_id):
                raise ValueError('observer_radio_clock_or_schema')
            generated = envelope.get('generated_s')
            now = self._clock()
            if (type(generated) not in (int, float) or not math.isfinite(generated)
                    or not 0 <= now - generated < .5):
                raise ValueError('observer_radio_source_expired')
            status = envelope.get('status')
            encode_status(status, session=self._session, sequence=0, generated_s=generated, now_s=now)
            snapshot = RadioStatusSnapshot(float(generated), json.dumps(status, allow_nan=False,
                                                                        separators=(',', ':')))
            with self._lock:
                if self._stopped or self._disabled:
                    return False
                if generated <= self._last_generated:
                    raise ValueError('observer_radio_source_not_advancing')
                self._last_generated = generated
                self._slot = snapshot
                self._reason = ''
            return True
        except (ValueError, TypeError, OverflowError, RecursionError, UnicodeError):
            with self._lock:
                self._slot = None
                self._reason = 'observer_radio_input_invalid'
            return False

    def stop(self):
        with self._lock:
            self._stopped = True
            self._slot = None

    def diagnostic(self, transport=None):
        with self._lock:
            return {'configured': True, 'enabled': not self._stopped and not self._disabled,
                    'error': self.error, 'tx_display_messages':
                        getattr(transport, 'tx_radio_count', 0) if transport is not None else 0,
                    'display_only': True, 'flight_commands_enabled': False}

    @staticmethod
    def _unknown(reason):
        return {'schema_version': 1, 'mode': 'OBSERVE_ONLY', 'flight_commands_enabled': False,
                'report': {'schema_version': 1, 'mode': 'OBSERVE_ONLY',
                    'flight_commands_enabled': False, 'assessment': 'UNKNOWN',
                    'reason': reason, 'valid_for_s': 0.},
                'camera_ok': False, 'px4_pose_received': False,
                'sensor': {}, 'sensor_valid_for_s': 0.}

    def poll(self, transport, *, heartbeat_fresh):
        from .observer_radio_protocol import encode_status, decode_status
        now = self._clock()
        with self._lock:
            if self._stopped or self._disabled:
                return False
            slot, reason = self._slot, self._reason
        if (not heartbeat_fresh or now < self._next_send
                or now < getattr(transport, 'discard_until', 0.)):
            return False
        self._next_send = now + .5
        if slot is None:
            status, generated = self._unknown(reason), now
        else:
            status, generated = json.loads(slot.status_json), slot.generated_s
            if not 0 <= now - generated < .5:
                status, generated = self._unknown('observer_radio_source_expired'), now
        try:
            payload = encode_status(status, session=self._session, sequence=self._sequence,
                                    generated_s=generated, now_s=now)
            decoded = decode_status(payload)
            deadlines = [now + decoded[key] / 1000. for key in
                         ('judgment_remaining_ms', 'sensor_remaining_ms') if decoded[key] > 0]
            expires = min(deadlines) if deadlines else now + .05
            state = decoded['assessment']
            text = None
            if state != self._last_text_state and now >= self._next_text:
                distance = decoded['sensor'].get('front_near_m')
                detail = (f'{distance:.2f}m' if distance is not None
                          else decoded['reason'].encode('ascii', 'replace').decode('ascii'))
                text = f'OBS source:{state} {detail}'[:50]
            with self._lock:
                if self._stopped or self._disabled or self._slot is not slot:
                    return False
            sent = transport.send_observer_status(payload, expires_s=expires, status_text=text)
            if sent:
                self._sequence = (self._sequence + 1) % (1 << 32)
                if text is not None:
                    self._last_text_state, self._next_text = state, now + 5.
            return sent
        except ObserverRadioSerialError:
            raise  # possible partial packet: established-link failure, not a retry
        except Exception as exc:
            # Optional formatting/logging feature failure must not stop RX.
            self.error = 'observer_radio_disabled:' + type(exc).__name__
            with self._lock:
                self._disabled = True
                self._slot = None
            return False


@dataclass(frozen=True)
class PublicationSnapshot:
    epoch: str
    channel: str
    sequence: int
    generated_ns: int
    expires_ns: int
    payload_json: str
    evidence_json: str = ""
    revocation: int = 0


class LatestSnapshots:
    """At most four immutable slots. No lock is held across ROS or disk IO."""
    def __init__(self, epoch):
        self.epoch = epoch
        self._lock = threading.Lock()
        self._slots = {}
        self._taken = {}
        self._inflight = {}
        self._generation = dict.fromkeys(CHANNELS, 0)
        self._publications = {}
        self._fault = ""
        self._stopped = False

    def put(self, snapshot):
        if (snapshot.epoch != self.epoch or snapshot.channel not in CHANNELS
                or snapshot.expires_ns <= snapshot.generated_ns):
            raise ValueError("invalid immutable publication snapshot")
        with self._lock:
            if self._fault or self._stopped or snapshot.revocation != self._generation[snapshot.channel]:
                return
            previous = self._slots.get(snapshot.channel)
            if previous and snapshot.sequence <= previous.sequence:
                raise ValueError("snapshot sequence regressed")
            self._slots[snapshot.channel] = snapshot

    def invalidate(self, channel):
        with self._lock:
            revoked = {channel, "status"}
            if channel in ("ATTITUDE_QUATERNION", "ESTIMATOR_STATUS"):
                revoked.add("LOCAL_POSITION_NED")
            for kind in revoked:
                self._generation[kind] += 1
                self._slots.pop(kind, None)
                self._inflight.pop(kind, None)

    def generation(self, channel):
        with self._lock:
            return self._generation[channel]

    def record_publication(self, channel, started_ns, completed_ns):
        """Bounded diagnostic counters only, never timing authority."""
        if channel not in ("status", "evidence", "attitude", "rates", "position"):
            raise ValueError("unknown ROS publication counter")
        with self._lock:
            item = self._publications.setdefault(channel, dict(count=0, last_started_ns=None,
                max_start_gap_ns=0, max_call_duration_ns=0))
            if item["last_started_ns"] is not None:
                item["max_start_gap_ns"] = max(item["max_start_gap_ns"], started_ns-item["last_started_ns"])
            item["count"] += 1
            item["last_started_ns"] = started_ns
            item["max_call_duration_ns"] = max(item["max_call_duration_ns"], completed_ns-started_ns)

    def publication_diagnostics(self):
        with self._lock:
            return {kind: dict(value) for kind, value in self._publications.items()}

    def fail(self, reason):
        with self._lock:
            if not self._fault:
                self._fault = str(reason) or "observer_worker_failed"
            self._slots.clear()
            self._inflight.clear()

    def stop(self):
        with self._lock:
            self._stopped = True
            self._slots.clear()
            self._inflight.clear()

    def raise_if_failed(self):
        with self._lock:
            reason = self._fault
        if reason:
            raise RuntimeError(reason)

    def take(self, channel):
        with self._lock:
            if self._fault or self._stopped:
                return None
            snapshot = self._slots.get(channel)
            if snapshot is None or self._taken.get(channel, 0) >= snapshot.sequence:
                return None
            self._taken[channel] = snapshot.sequence
            self._inflight[channel] = snapshot
            return snapshot

    def is_current(self, snapshot, now_ns):
        with self._lock:
            return (not self._fault and not self._stopped
                    and (self._slots.get(snapshot.channel) is snapshot
                         or self._inflight.get(snapshot.channel) is snapshot)
                    and snapshot.revocation == self._generation[snapshot.channel]
                    and snapshot.epoch == self.epoch
                    and type(now_ns) is int
                    and snapshot.generated_ns <= now_ns < snapshot.expires_ns)


class ObserverFcWorker:
    """Enabled timing path only. Ordinary missing USB retains the 2 s retry.

    An established-link or permanent timing fault is terminal for this worker;
    the ROS owner raises it so the original appliance supervisor can restart.
    Constructor performs no IO. start() transfers exclusive state ownership.
    """
    def __init__(self, *, timing, journal_path, evidence_clock_id,
                 transport_factory=UsbTelemetryTransport, clock=time.monotonic,
                 radio_sender=None):
        if not timing.enabled or not evidence_clock_id:
            raise ValueError("worker requires explicit timing and local clock identity")
        self.mailbox = LatestSnapshots(timing.session_id)
        self._state = TelemetryState(timing=timing)
        self._clock_id = evidence_clock_id
        self._journal_path = Path(journal_path)
        self._factory = transport_factory
        self._clock = clock
        self._transport = None
        self._journal = None
        self._requested = False
        self._next_open = 0.
        self._next_report = 0.
        self._connection_error = "waiting_for_usb"
        self._sequence = 0
        self._last_read_ns = None
        self._max_read_gap_ns = 0
        self._read_calls = 0
        self._quality = None
        self._stop = threading.Event()
        self._thread = None
        self.radio_sender = radio_sender

    def start(self):
        if self._thread is not None:
            raise RuntimeError("worker cannot restart a previous epoch")
        self._thread = threading.Thread(target=self._run, name="observer-fc-usb", daemon=True)
        self._thread.start()

    def stop(self):
        if self.radio_sender is not None:
            self.radio_sender.stop()
        self.mailbox.stop()  # in-flight snapshots lose authority before joining
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.)
            if self._thread.is_alive():
                raise RuntimeError("observer USB worker shutdown not confirmed")

    def _check_fault(self):
        reason = self._state.fault or self._state.timing.mapper.fault
        if reason:
            raise RuntimeError(reason)

    def _put(self, channel, now_ns, expires_ns, payload, evidence=None):
        self._sequence += 1
        self.mailbox.put(PublicationSnapshot(
            self.mailbox.epoch, channel, self._sequence, now_ns, expires_ns,
            json.dumps(payload, allow_nan=False),
            json.dumps(evidence, allow_nan=False) if evidence is not None else "",
            self.mailbox.generation(channel)))

    def _reconcile_dependencies(self):
        """New valid samples are not revocations; semantic invalidity is.

        Called after EACH accepted/rejected input, so an invalid->valid change
        inside one read batch still retires any earlier positive snapshot.
        """
        now = self._clock()
        state = self._state
        current = dict(heartbeat=state.fresh("HEARTBEAT", now, 2.5),
            attitude=state.fresh("ATTITUDE_QUATERNION", now),
            position=state.fresh("LOCAL_POSITION_NED", now),
            estimator=state.fresh("ESTIMATOR_STATUS", now),
            reason=state.position_reason(now))
        current["estimator_flags"] = (state.samples["ESTIMATOR_STATUS"]["flags"]
                                      if current["estimator"] else None)
        self._check_fault()
        previous = self._quality
        self._quality = current
        if previous is None or current == previous:
            return previous is None
        self.mailbox.invalidate("status")
        for field, kind in (("attitude", "ATTITUDE_QUATERNION"),
                            ("position", "LOCAL_POSITION_NED"),
                            ("estimator", "ESTIMATOR_STATUS")):
            if previous[field] and not current[field]:
                self.mailbox.invalidate(kind)
        if previous["reason"] != current["reason"]:
            self.mailbox.invalidate("LOCAL_POSITION_NED")
        if previous["heartbeat"] and not current["heartbeat"]:
            for kind in KINDS:
                self.mailbox.invalidate(kind)
        return True

    def _export_pose(self, kind):
        now = self._clock()
        now_ns = round(now * NS)
        state = self._state
        evidence = state.timing_evidence(kind, now, self._clock_id)
        self._check_fault()
        if evidence is None:
            self.mailbox.invalidate(kind)
            return
        # The original evidence is copied unchanged. Dependency deadlines may
        # only SHORTEN the local output permission, never extend that evidence.
        expiry = min(evidence["evidence_expires_monotonic_ns"],
                     round((state.received["HEARTBEAT"] + 2.5) * NS))
        if kind == "ATTITUDE_QUATERNION":
            value = state.attitude(now)
        elif kind == "LOCAL_POSITION_NED":
            value = state.position(now)
            for dependency in ("ATTITUDE_QUATERNION", "ESTIMATOR_STATUS"):
                dep = state.timing.mapper.latest.get(dependency)
                if dep is not None and state.fresh(dependency, now):
                    expiry = min(expiry, dep.expires_ns)
        else:
            value = {}  # ESTIMATOR_STATUS has evidence only, no new ROS pose type
        self._check_fault()
        if value is None or now_ns >= expiry:
            self.mailbox.invalidate(kind)
            return
        self._put(kind, now_ns, expiry, value, evidence)

    def _report(self, *, force=False):
        now = self._clock()
        journal_due = now >= self._next_report
        if not force and not journal_due:
            return
        if journal_due:
            self._next_report = now + .1
        data = self._state.diagnostic(now)
        self._check_fault()
        transport = self._transport
        data.update(connection_error=self._connection_error,
                    tx_telemetry_requests=transport.tx_count if transport else 0,
                    rx_bad_frames=transport.bad_data if transport else 0,
                    startup_discarded_messages=transport.startup_discarded_messages if transport else 0,
                    io_worker={"single_owner": True, "read_calls": self._read_calls,
                               "last_read_monotonic_ns": self._last_read_ns,
                               "max_read_gap_ns": self._max_read_gap_ns,
                               "latest_only_channels": len(CHANNELS),
                               "ros_publications": self.mailbox.publication_diagnostics()})
        if self.radio_sender is not None:
            data['observer_radio'] = self.radio_sender.diagnostic(transport)
        now_ns = round(now * NS)
        expiry = now_ns + 250_000_000
        if data["heartbeat_received"]:
            expiry = min(expiry, round((self._state.received["HEARTBEAT"] + 2.5) * NS))
        if data["gps_fix_type"] is not None:
            expiry = min(expiry, round((self._state.received["GPS_RAW_INT"] + 2.) * NS))
        if data["timing"]["ready"]:
            expiry = min(expiry, data["timing"]["sync_expires_monotonic_ns"])
        for kind in KINDS:
            evidence = self._state.timing.mapper.latest.get(kind)
            if evidence is not None and self._state.fresh(kind, now):
                expiry = min(expiry, evidence.expires_ns)
        data["publication_expires_monotonic_ns"] = expiry
        self._check_fault()
        if journal_due:
            self._journal.write(json.dumps(data, allow_nan=False) + "\n")
        # Original diagnostic timestamp/expiry survive disk IO and ROS queues.
        self._put("status", now_ns, expiry, data)

    def _step(self):
        now = self._clock()
        if self._transport is None:
            if now < self._next_open:
                self._report()
                return
            self._next_open = now + 2.
            try:
                self._transport = self._factory(self._journal, timing=self._state.timing)
                self._connection_error = ""
            except (OSError, ValueError) as exc:
                self._connection_error = str(exc)
                self._check_fault()
                self._report()
                return
        messages = self._transport.read()
        received_ns = self._transport.last_read_monotonic_ns
        self._read_calls += 1
        if self._last_read_ns is not None:
            self._max_read_gap_ns = max(self._max_read_gap_ns, received_ns - self._last_read_ns)
        self._last_read_ns = received_ns
        self._check_fault()
        changed = set()
        refresh_status = self._reconcile_dependencies()
        for message in messages:
            now = self._clock()
            kind = message.get_type()
            sender = (message.get_srcSystem(), message.get_srcComponent())
            value = message.to_dict()
            if kind == "COMMAND_ACK" and sender == (1, 1):
                self._journal.write(json.dumps({"kind": "command_ack", "data": value}) + "\n")
            accepted = self._state.accept(kind, value, *sender, now, received_ns=received_ns)
            self._check_fault()
            refresh_status = self._reconcile_dependencies() or refresh_status
            if not accepted:
                if kind in KINDS and sender == (1, 1):
                    self.mailbox.invalidate(kind)
                    changed.discard(kind)
                    refresh_status = True
                continue
            if kind == "HEARTBEAT" and not self._requested:
                for message_id, interval in STREAM_INTERVALS.items():
                    self._transport.request(511, message_id, interval)
                self._transport.request(512, 148)
                self._requested = True
            if kind in KINDS:
                changed.add(kind)
                refresh_status = True
                if kind in ("ATTITUDE_QUATERNION", "ESTIMATOR_STATUS"):
                    # Recompute dependencies using the SAME source/evidence
                    # deadline, not a new receipt or synthetic position sample.
                    changed.add("LOCAL_POSITION_NED")
        if self._state.fresh("HEARTBEAT", self._clock(), 2.5):
            self._transport.request_timesync()
        self._check_fault()
        for kind in CHANNELS[:-1]:
            if kind in changed:
                self._export_pose(kind)
        self._report(force=refresh_status)
        if self.radio_sender is not None and not self._stop.is_set():
            self.radio_sender.poll(self._transport,
                heartbeat_fresh=self._state.fresh('HEARTBEAT', self._clock(), 2.5))

    def _run(self):
        try:
            self._journal_path.parent.mkdir(parents=True, exist_ok=True)
            self._journal = self._journal_path.open("x", buffering=1)
            while not self._stop.is_set():
                self._step()
                # Serial read already waits <=10 ms; yield even when USB is
                # continuously buffered or missing. No busy-spin ROS timer.
                self._stop.wait(.001 if self._transport is not None else .01)
        except BaseException as exc:
            self.mailbox.fail(str(exc) or type(exc).__name__)
            # Best effort diagnostic only, after revocation; a broken journal
            # must not prevent closing the exclusively owned serial handle.
            if self._journal is not None:
                try:
                    self._journal.write(json.dumps({"kind": "worker_fault", "reason": str(exc),
                        "observed_monotonic_s": self._clock(), "read_calls": self._read_calls,
                        "last_read_monotonic_ns": self._last_read_ns,
                        "max_read_gap_ns": self._max_read_gap_ns}) + "\n")
                except Exception:
                    pass
        finally:
            if self.radio_sender is not None:
                self.radio_sender.stop()
            self.mailbox.stop()
            try:
                if self._transport is not None:
                    self._transport.close()
            except BaseException as exc:
                self.mailbox.fail("worker_serial_close_failed: " + str(exc))
            finally:
                if self._journal is not None:
                    try:
                        self._journal.close()
                    except BaseException as exc:
                        self.mailbox.fail("worker_journal_close_failed: " + str(exc))

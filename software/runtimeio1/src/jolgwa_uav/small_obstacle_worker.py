"""One physical USB owner; bounded callback mailboxes, explicit demo requests.

Constructors/callbacks never open or write serial. Only _step's worker thread
owns transport, telemetry, TIMESYNC, executor and journal. Existing observer
publication conversion/dependency revocation is inherited without modification.
"""
from dataclasses import asdict
import json
import threading
import time
import uuid

from .observer_fc_worker import ObserverFcWorker, CHANNELS
from .observer_fc_telemetry import TelemetryState, STREAM_INTERVALS
from .observer_fc_timesync import KINDS, NS, valid_ns
from .small_obstacle_fc_link import SmallObstacleFcTransport, SmallObstacleFcTelemetry, EXTRA_INTERVALS
from .small_obstacle_execution import ExplicitDemoExecutor
from .small_obstacle_perception import read_decision_evidence

REQUEST_NS = 200_000_000
MAX_ENVELOPE_BYTES = 65536


class _DemoTelemetryState(TelemetryState):
    runtime_mode = "observe"

    def diagnostic(self, now):
        data = super().diagnostic(now)
        active = self.runtime_mode == "auto-demo"
        data.update(mode="AUTO_DEMO" if active else "OBSERVE_ONLY", flight_commands_enabled=active,
                    telemetry_scope_only=True, runtime_mode=self.runtime_mode,
                    runtime_flight_output_authorized=active)
        return data


class SmallObstacleWorker(ObserverFcWorker):
    def __init__(self, *, timing, journal_path, evidence_clock_id,
                 transport_factory=SmallObstacleFcTransport, clock=time.monotonic,
                 executor_factory=ExplicitDemoExecutor, mode="observe"):
        if mode != "observe":
            raise ValueError("runtime_boot_must_be_observe_explicit_later_start_required")
        super().__init__(timing=timing, journal_path=journal_path,
            evidence_clock_id=evidence_clock_id, transport_factory=transport_factory, clock=clock)
        self._state = _DemoTelemetryState(timing=timing)
        self._executor_factory = executor_factory
        self._fc = self._executor = None
        self._demo_lock = threading.Lock()
        self._request = None
        self._perception_json = None
        self._perception_identity = None
        self._perception_sequence = 0
        self._perception_error = "perception_not_received"
        self._perception_retired = False
        self._next_demo_ns = 0
        self._last_request = None
        self._demo_json = json.dumps(dict(mode="observe", phase="STARTING",
            reason="waiting_for_usb", flight_output_authorized=False, last_request=None))
        self._shutdown_requested = threading.Event()
        self._shutdown_complete = threading.Event()

    def _now_ns(self):
        return round(self._clock()*NS)

    def submit_request(self, request):
        """Caller supplies ORIGINAL same-clock deadline; returns queue ID only."""
        if not isinstance(request, dict):
            raise ValueError("request_object_required")
        allowed = {"kind", "token", "observe_passed", "submitted_monotonic_ns", "expires_monotonic_ns"}
        if set(request)-allowed or request.get("kind") not in ("start", "observe"):
            raise ValueError("unsupported_demo_request")
        now = self._now_ns()
        submitted, expires = request.get("submitted_monotonic_ns"), request.get("expires_monotonic_ns")
        if not valid_ns(submitted) or not valid_ns(expires) or not submitted <= now < expires <= submitted+REQUEST_NS:
            raise ValueError("original_request_expired_or_invalid")
        if request["kind"] == "start" and (
                not isinstance(request.get("token"), str) or not 1 <= len(request["token"]) <= 128
                or request.get("observe_passed") is not True):
            raise ValueError("explicit_token_and_observe_result_required")
        if request["kind"] == "observe" and set(request) & {"token", "observe_passed"}:
            raise ValueError("observe_request_cannot_carry_start_authority")
        detached = dict(request, request_id=uuid.uuid4().hex)
        with self._demo_lock:
            if self._shutdown_requested.is_set() or self._stop.is_set():
                raise RuntimeError("worker_stopping")
            if self._request is not None and request["kind"] != "observe":
                raise ValueError("one_pending_request_only")
            # Observe supersedes, never queues behind, a pending start.
            self._request = detached
        return detached["request_id"]

    def submit_perception(self, envelope):
        """Callback validates/detaches only. Invalid input revokes the slot."""
        now = self._now_ns()
        try:
            if isinstance(envelope, str):
                if len(envelope.encode("utf-8")) > MAX_ENVELOPE_BYTES:
                    raise ValueError("perception_too_large")
                value = json.loads(envelope)
            elif isinstance(envelope, dict):
                value = envelope
            else:
                raise ValueError("perception_object_required")
            wire = json.dumps(value, allow_nan=False, separators=(",", ":"))
            if len(wire.encode("utf-8")) > MAX_ENVELOPE_BYTES:
                raise ValueError("perception_too_large")
            value = json.loads(wire)
            read_decision_evidence(value, self._clock_id, now)
            binding = value["input_binding"]
            identity = (value["backend_epoch"], value["producer_session"],
                        binding["fc_session_id"], binding["fc_link_identity"])
            if any(not isinstance(v, str) or not v or len(v) > 256 for v in identity):
                raise ValueError("perception_identity_invalid")
            with self._demo_lock:
                if self._perception_retired:
                    raise ValueError("perception_epoch_retired")
                if self._perception_identity is not None and identity != self._perception_identity:
                    self._perception_retired = True
                    raise ValueError("perception_epoch_changed")
                if value["sequence"] <= self._perception_sequence:
                    raise ValueError("perception_replayed_or_reordered")
                self._perception_identity = identity
                self._perception_sequence = value["sequence"]
                self._perception_json = wire
                self._perception_error = ""
            return True
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
            with self._demo_lock:
                self._perception_json = None
                self._perception_error = str(exc)[:160]
            return False

    def last_demo_status(self):
        with self._demo_lock:
            wire = self._demo_json
        return json.loads(wire)  # caller cannot mutate worker authority/state

    def _put(self, channel, now_ns, expires_ns, payload, evidence=None):
        if channel == "status":
            active = bool(self._executor and self._executor.mode == "auto-demo")
            payload = dict(payload, telemetry_scope_only=True,
                mode="AUTO_DEMO" if active else "OBSERVE_ONLY", flight_commands_enabled=active,
                runtime_mode=self._executor.mode if self._executor else "observe",
                runtime_flight_output_authorized=active)
        return super()._put(channel, now_ns, expires_ns, payload, evidence)

    def _check_fault(self):
        super()._check_fault()
        if self._fc is not None and self._fc.fault:
            raise RuntimeError(self._fc.fault)

    def _report(self, *, force=False):
        # Both journal and private publication tell the truth; timing_evidence
        # itself retains its original observation-only evidence schema.
        self._state.runtime_mode = self._executor.mode if self._executor else "observe"
        return super()._report(force=force)

    def _step(self):
        now = self._clock()
        if self._transport is None:
            if now < self._next_open:
                self._demo_tick()
                self._report()
                return
            self._next_open = now+2.
            try:
                self._transport = self._factory(self._journal, timing=self._state.timing, mode="observe")
                self._fc = SmallObstacleFcTelemetry(self._state, self._clock_id)
                self._transport.bind_telemetry(self._fc)
                self._executor = self._executor_factory(self._transport)
                self._connection_error = ""
            except (OSError, ValueError) as exc:
                # If construction succeeded, never retain a half-bound descriptor.
                if self._transport is not None:
                    self._transport.close()
                    self._transport = None
                    raise
                self._connection_error = str(exc)
                self._check_fault()
                self._demo_tick()
                self._report()
                return
        messages = self._transport.read()
        received_ns = self._transport.last_read_monotonic_ns
        self._read_calls += 1
        if self._last_read_ns is not None:
            self._max_read_gap_ns = max(self._max_read_gap_ns, received_ns-self._last_read_ns)
        self._last_read_ns = received_ns
        self._check_fault()
        changed = set()
        refresh_status = self._reconcile_dependencies()
        for message in messages:
            now_ns = self._now_ns()
            kind = message.get_type()
            sender = message.get_srcSystem(), message.get_srcComponent()
            value = message.to_dict()
            accepted = self._fc.accept(kind, value, *sender, received_ns, now_ns)
            if self._fc.fault:
                # Save the actual reset/rejection row before _check_fault exits.
                self._journal.write(json.dumps(dict(kind="demo_fc_rx_fault",
                    observed_monotonic_ns=now_ns, fc_epoch=self._fc.fc_epoch,
                    reason=self._fc.fault, diagnostics=self._fc.diagnostic()), allow_nan=False)+"\n")
            self._check_fault()
            refresh_status = self._reconcile_dependencies() or refresh_status
            if not accepted:
                if kind in KINDS and sender == (1, 1):
                    self.mailbox.invalidate(kind)
                    changed.discard(kind)
                    refresh_status = True
                continue
            if kind == "COMMAND_ACK":
                self._journal.write(json.dumps(dict(kind="demo_command_ack", data=value,
                    received_monotonic_ns=received_ns))+"\n")
            if kind == "HEARTBEAT" and not self._requested:
                for message_id, interval in STREAM_INTERVALS.items():
                    self._transport.request(511, message_id, interval)
                self._transport.request(512, 148)
                for message_id in EXTRA_INTERVALS:
                    self._transport.request_demo_stream(message_id)
                self._requested = True
            if kind in KINDS:
                changed.add(kind)
                refresh_status = True
                if kind in ("ATTITUDE_QUATERNION", "ESTIMATOR_STATUS"):
                    changed.add("LOCAL_POSITION_NED")
        if self._state.fresh("HEARTBEAT", self._clock(), 2.5):
            if not self._shutdown_requested.is_set() and not self._stop.is_set():
                self._transport.request_heartbeat()
            self._transport.request_timesync()
        self._check_fault()
        for kind in CHANNELS[:-1]:
            if kind in changed:
                self._export_pose(kind)
        self._demo_tick()
        self._report(force=refresh_status)

    def _demo_tick(self):
        now = self._now_ns()
        with self._demo_lock:
            request = self._request
            due = request is not None or self._shutdown_requested.is_set() or now >= self._next_demo_ns
            if not due:
                return
            self._request = None
            wire, input_error = self._perception_json, self._perception_error
        self._next_demo_ns = now+50_000_000
        stopping = self._shutdown_requested.is_set()
        expired = request is not None and not request["submitted_monotonic_ns"] <= now < request["expires_monotonic_ns"]
        evidence = None
        if wire is not None:
            try:
                envelope = json.loads(wire)
                evidence = read_decision_evidence(envelope, self._clock_id, now)
                binding = envelope["input_binding"]
                if self._fc is None or (binding["fc_session_id"], binding["fc_link_identity"]) != self._fc.identity:
                    raise ValueError("perception_actual_fc_epoch_mismatch")
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                input_error, evidence = str(exc)[:160], None
        snapshot = self._fc.snapshot(now) if self._fc else None
        kwargs = {}
        if stopping or (request and not expired and request["kind"] == "observe"):
            kwargs["stop"] = True
        elif request and not expired:
            kwargs.update(start_token=request["token"], observe_passed=request["observe_passed"])
        if self._executor is None:
            status = dict(mode="observe", phase="STARTING", reason="waiting_for_usb",
                          flight_output_authorized=False, updated_monotonic_ns=now)
        else:
            # Queue parsing/snapshot work cannot renew a start request's lease.
            checked = self._now_ns()
            if request is not None and not request["submitted_monotonic_ns"] <= checked < request["expires_monotonic_ns"]:
                expired = True
                kwargs = {"stop": True} if stopping else {}
            status = self._executor.tick(snapshot, evidence, checked, **kwargs)
            # The initial authorization tick emits NO setpoint. If scheduling
            # stalls inside that tick, retire it before a following tick can TX.
            completed = self._now_ns()
            if (request is not None and request["kind"] == "start"
                    and not request["submitted_monotonic_ns"] <= completed < request["expires_monotonic_ns"]):
                expired = True
                if status.get("mode") == "auto-demo":
                    status = self._executor.tick(snapshot, None, completed, stop=True)
        if request is not None:
            applied = (request["kind"] == "observe" and status["mode"] == "observe")
            accepted = (request["kind"] == "start" and status.get("mode") == "auto-demo"
                        and status.get("phase") == "PRESTREAM")
            self._last_request = dict(request_id=request["request_id"], kind=request["kind"],
                result="expired" if expired else "applied" if applied else "accepted" if accepted else "rejected",
                reason=status["reason"],
                submitted_monotonic_ns=request["submitted_monotonic_ns"],
                expires_monotonic_ns=request["expires_monotonic_ns"], processed_monotonic_ns=self._now_ns())
        status.update(last_request=self._last_request, perception_error=input_error,
            evidence_clock_id=self._clock_id, fc=asdict(snapshot) if snapshot is not None else None,
            fc_rx_diagnostics=self._fc.diagnostic() if self._fc else None,
            shutdown_requested=stopping)
        serialized = json.dumps(status, allow_nan=False)
        if self._journal is not None:
            self._journal.write(json.dumps(dict(kind="demo_execution_status", data=status),allow_nan=False)+"\n")
        with self._demo_lock:
            self._demo_json = serialized
        if stopping:
            self._shutdown_complete.set()

    def stop(self):
        # Never touch transport/executor from the ROS/socket caller.
        self._shutdown_requested.set()
        if self._thread is not None and self._thread.is_alive():
            self._shutdown_complete.wait(timeout=.5)
        super().stop()

    def _run(self):
        """Final handover attempt and close both remain on the USB owner thread."""
        fault = ""
        try:
            self._journal_path.parent.mkdir(parents=True, exist_ok=True)
            self._journal = self._journal_path.open("x", buffering=1)
            while not self._stop.is_set():
                self._step()
                self._stop.wait(.001 if self._transport is not None else .01)
        except BaseException as exc:
            fault = str(exc) or type(exc).__name__
            self.mailbox.fail(fault)
        finally:
            self.mailbox.stop()
            if self._transport is not None:
                try:
                    self._transport.stop()  # internally fresh+owned only; otherwise TX0
                except BaseException as exc:
                    fault = fault or "final_handover_not_sent:"+str(exc)
                try:
                    self._transport.close()
                except BaseException as exc:
                    fault = fault or "serial_close_failed:"+str(exc)
            if fault:
                self.mailbox.fail(fault)
            with self._demo_lock:
                previous = json.loads(self._demo_json)
                previous.update(mode="observe", phase="FAULT" if fault else "STOPPED",
                    reason=fault or "worker_stopped", flight_output_authorized=False,
                    updated_monotonic_ns=self._now_ns())
                self._demo_json = json.dumps(previous, allow_nan=False)
            if self._journal is not None:
                try:
                    self._journal.write(json.dumps(dict(kind="demo_worker_stopped", reason=fault,
                        observed_monotonic_ns=self._now_ns()))+"\n")
                    self._journal.close()
                except BaseException as exc:
                    self.mailbox.fail("journal_close_failed:"+str(exc))
            self._shutdown_complete.set()

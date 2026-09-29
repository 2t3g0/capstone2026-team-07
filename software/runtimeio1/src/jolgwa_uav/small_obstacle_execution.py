"""Explicit observe-to-single-demo execution boundary (no automatic takeoff).

The owner calls tick on the same thread that owns the MAVLink transport. Neither
ROS callbacks nor a user socket may write to that transport. Real authorization
is separate from the existing offline guard's purely descriptive intents.
"""
from dataclasses import asdict
import math

from .small_obstacle_demo_guard import DemoState, SmallObstacleDemoGuard, Telemetry
from .small_obstacle_fc_link import FcSnapshot

NS = 1_000_000_000


class ExplicitDemoExecutor:
    def __init__(self, transport):
        self.transport = transport
        self.guard = SmallObstacleDemoGuard()
        self.mode = "observe"
        self.phase = "OBSERVE"
        self.reason = "explicit_start_required"
        self.started_ns = self.requested_ns = 0
        self.last_tick_ns = 0
        self.authorized_until_ns = 0
        self.used = False
        self.last_intent = None
        self.first_zero_ns = None
        self.last_zero_ns = None
        self.start_tokens = set()

    def _finish(self, reason):
        # stop() itself rechecks fresh actual mode and whether this process
        # requested it. A stale or pilot-owned link must receive no commands.
        try:
            self.transport.stop()
        except (OSError, ValueError, PermissionError, TimeoutError) as exc:
            reason += ";handover_not_sent:" + str(exc)
        finally:
            self.mode = "observe"
            self.phase = "FINISHED"
            self.reason = reason
            try:
                self.transport.set_mode("observe")
            except (OSError, ValueError, PermissionError, TimeoutError) as exc:
                self.reason += ";transport_observe_failed:" + str(exc)

    @staticmethod
    def _snapshot_error(snapshot, now_ns):
        # The guard sees telemetry, not the link's earliest original expiry
        # (which also includes ODOM and dependency proofs). Never replace it
        # with a fresh callback receipt, including on the authorization tick.
        if (not isinstance(snapshot, FcSnapshot) or snapshot.reason
                or not isinstance(snapshot.telemetry, Telemetry)):
            return "actual_fc_snapshot_unavailable"
        if type(snapshot.expires_ns) is not int or now_ns >= snapshot.expires_ns:
            return "original_fc_snapshot_deadline_expired"
        if (snapshot.fc_epoch != snapshot.telemetry.fc_epoch
                or not all(value is True for value in (snapshot.armed, snapshot.airborne,
                    snapshot.receiver_healthy, snapshot.ground_reference_ready))
                or type(snapshot.main_mode) is not int
                or type(snapshot.actual_offboard) is not bool
                or snapshot.actual_offboard != (snapshot.main_mode == 6)):
            return "actual_fc_snapshot_inconsistent"
        return ""

    def _status(self, now_ns):
        return {"schema_version": 1, "mode": self.mode, "phase": self.phase,
                "reason": self.reason, "one_demo_used": self.used,
                "flight_output_authorized": self.mode == "auto-demo",
                "ready_for_explicit_start": (not self.used and self.phase == "OBSERVE"
                    and self.last_intent is not None
                    and self.last_intent.reason == "pilot_start_required"),
                "updated_monotonic_ns": now_ns,
                "intent": asdict(self.last_intent) if self.last_intent else None,
                "scope": "ONE_OBSTACLE_NO_ARM_NO_TAKEOFF_NO_DESCENT"}

    def tick(self, snapshot, evidence, now_ns, *, start_token=None,
             observe_passed=False, stop=False):
        if type(now_ns) is not int or now_ns <= 0 or now_ns < self.last_tick_ns:
            if self.mode == "auto-demo":
                self._finish("executor_clock_invalid")
            else:
                self.phase, self.reason = "FINISHED", "executor_clock_invalid"
            return self._status(max(0, self.last_tick_ns))
        previous_tick = self.last_tick_ns
        self.last_tick_ns = now_ns
        if stop:
            if self.mode == "auto-demo":
                self._finish("operator_requested_observe")
            else:
                self.phase, self.reason = "FINISHED", "operator_requested_observe"
            return self._status(now_ns)
        if self.phase == "FINISHED":
            return self._status(now_ns)
        snapshot_error = self._snapshot_error(snapshot, now_ns)
        telemetry = snapshot.telemetry if not snapshot_error else None
        if self.mode == "observe":
            # Rejected requests are not queued for a later airborne condition.
            if start_token is not None:
                if (not isinstance(start_token, str) or not 1 <= len(start_token) <= 128
                        or start_token in self.start_tokens or len(self.start_tokens) >= 64):
                    self.phase, self.reason = "FINISHED", "invalid_or_reused_start_token"
                    return self._status(now_ns)
                self.start_tokens.add(start_token)
            accepted_request = (start_token is not None and observe_passed is True
                                and not self.used and not snapshot_error and snapshot.main_mode == 3)
            self.last_intent = self.guard.tick(now_s=now_ns/NS, telemetry=telemetry,
                evidence=evidence, pilot_start_token=start_token if accepted_request else None)
            self.reason = self.last_intent.reason
            if start_token is not None and not accepted_request:
                self.reason = "start_rejected_observe_result_and_actual_posctl_required"
            if self.guard.state in (DemoState.STOP, DemoState.COMPLETE):
                self.phase = "FINISHED"
            elif accepted_request and self.guard.state is not DemoState.WAIT_PILOT:
                self.used = True
                self.started_ns, self.authorized_until_ns = now_ns, now_ns + 60*NS
                try:
                    self.transport.set_mode("auto-demo")
                    self.transport.authorize(start_token, snapshot.fc_epoch,
                                             self.authorized_until_ns)
                    self.mode, self.phase = "auto-demo", "PRESTREAM"
                    self.reason = "zero_velocity_prestream_before_offboard"
                except (OSError, ValueError, PermissionError, TimeoutError) as exc:
                    self._finish("authorization_rejected:"+str(exc))
            # The first authorized tick still emits nothing; no start request
            # can both change ownership and move the aircraft in one callback.
            return self._status(now_ns)
        if start_token is not None:
            self._finish("another_start_during_demo")
            return self._status(now_ns)
        if previous_tick and now_ns-previous_tick >= 250_000_000:
            self._finish("executor_tick_gap_exceeded")
            return self._status(now_ns)
        if snapshot_error:
            self._finish(snapshot_error)
            return self._status(now_ns)
        if now_ns >= self.authorized_until_ns:
            self._finish("one_demo_authorization_expired")
            return self._status(now_ns)
        if ((self.phase == "PRESTREAM" and snapshot.main_mode != 3)
                or (self.phase == "AWAIT_OFFBOARD" and snapshot.main_mode not in (3, 6))
                or (self.phase == "ACTIVE" and snapshot.main_mode != 6)):
            self._finish("native_mode_changed_no_automatic_reentry")
            return self._status(now_ns)
        self.last_intent = self.guard.tick(now_s=now_ns/NS, telemetry=telemetry, evidence=evidence)
        intent = self.last_intent
        if intent.pilot_handover_requested:
            self._finish(intent.reason)
            return self._status(now_ns)
        expires = min(snapshot.expires_ns, int(intent.expires_at_s*NS), self.authorized_until_ns)
        if expires <= now_ns:
            self._finish("original_guard_or_fc_deadline_expired")
            return self._status(now_ns)
        yaw = math.remainder(self.guard.heading, 2*math.pi)
        try:
            if self.phase in ("PRESTREAM", "AWAIT_OFFBOARD"):
                fresh_ack = False
                if self.phase == "AWAIT_OFFBOARD":
                    # An ACK cannot renew the original two-second attempt or
                    # claim receipt in the future. Timeout wins at the boundary.
                    if now_ns-self.requested_ns >= 2*NS:
                        self._finish("offboard_mode_or_ack_timeout")
                        return self._status(now_ns)
                    ack_at = snapshot.ack_received_ns
                    if type(ack_at) is not int or ack_at < 0 or ack_at > now_ns:
                        self._finish("offboard_ack_receipt_invalid")
                        return self._status(now_ns)
                    fresh_ack = snapshot.last_ack_command == 176 and ack_at >= self.requested_ns
                    if fresh_ack and snapshot.last_ack_result not in (0, 5):
                        self._finish("offboard_request_rejected_by_fc")
                        return self._status(now_ns)
                # Don't start a timed approval while drifting during prestream.
                if (math.dist(telemetry.position_ned_m, self.guard.origin) > .25
                        or math.hypot(*telemetry.velocity_ned_mps) > .1
                        or self.guard.state is not DemoState.APPROACH
                        or intent.intent != "ADVANCE"):
                    self._finish("hover_or_clear_lost_during_offboard_entry")
                    return self._status(now_ns)
                if self.transport.send_velocity(snapshot, 0., 0., 0., yaw, expires_ns=expires):
                    if self.last_zero_ns is None or now_ns-self.last_zero_ns >= 250_000_000:
                        self.first_zero_ns = now_ns
                    self.last_zero_ns = now_ns
                if self.phase == "PRESTREAM" and self.first_zero_ns is not None and now_ns-self.first_zero_ns >= NS:
                    if self.transport.request_offboard(snapshot, expires_ns=expires):
                        self.requested_ns = now_ns
                        self.phase, self.reason = "AWAIT_OFFBOARD", "waiting_for_actual_mode_and_ack"
                elif self.phase == "AWAIT_OFFBOARD":
                    if snapshot.actual_offboard and fresh_ack and snapshot.last_ack_result == 0:
                        self.phase, self.reason = "ACTIVE", "actual_offboard_confirmed"
                return self._status(now_ns)
            vx = vy = vz = 0.
            if intent.intent == "ADVANCE":
                dx = intent.target_ned_m[0]-telemetry.position_ned_m[0]
                dy = intent.target_ned_m[1]-telemetry.position_ned_m[1]
                distance = math.hypot(dx, dy)
                speed = min(.5, intent.speed_cap_mps, distance)
                if distance > 0:
                    vx, vy = speed*dx/distance, speed*dy/distance
                # No automatic descent. Large height drift is a handover, not
                # a hidden altitude recovery that conflicts with pilot intent.
                if abs(telemetry.position_ned_m[2]-intent.target_ned_m[2]) > .25:
                    self._finish("advance_height_drift")
                    return self._status(now_ns)
            elif intent.intent == "CLIMB_STEP":
                vz = -min(.6, intent.speed_cap_mps)
            elif intent.intent != "HOLD":
                self._finish("unsupported_guard_intent")
                return self._status(now_ns)
            self.transport.send_velocity(snapshot, vx, vy, vz, yaw, expires_ns=expires)
            self.reason = intent.reason
        except (OSError, ValueError, PermissionError, TimeoutError) as exc:
            self._finish("flight_output_rejected:"+str(exc))
        return self._status(now_ns)

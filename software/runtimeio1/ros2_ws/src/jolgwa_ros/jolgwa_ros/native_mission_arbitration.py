"""Transport-free safety contract for assisting an existing PX4 mission.

This module sends no commands, uploads no plan, sets no mission index, and grants
no arming/takeoff permission. It is deliberately NOT wired to the controller.
An integration adapter must correlate mode acknowledgements to these one-use
tokens and independently enforce the existing command-owner gate.

All times are evidence times on one monotonic clock, not the time a cached value
was republished. Vehicle/navigation epochs must change after reboot, estimator
origin reset, or an equivalent loss of coordinate continuity. A fingerprint
must identify the complete approved mission, not merely its waypoint count.
These are adapter requirements, not assertions that a particular transport
currently supplies them. Supported mission item semantics remain an integration
decision (PX4 camera-trigger missions, for example, can rewind on resume).
"""

from dataclasses import dataclass
from enum import Enum
import math


MAX_EVIDENCE_AGE_S = 0.5


class NativeMode(str, Enum):
    MISSION = "MISSION"
    HOLD = "HOLD"
    OFFBOARD = "OFFBOARD"
    OTHER = "OTHER"


class AssistPhase(str, Enum):
    DISABLED = "DISABLED"
    MONITORING = "MONITORING"
    PAUSE_PENDING = "PAUSE_PENDING"
    HOLD_CONFIRMED = "HOLD_CONFIRMED"
    TAKEOVER_PENDING = "TAKEOVER_PENDING"
    AVOIDING = "AVOIDING"
    RESUME_PENDING = "RESUME_PENDING"
    RETIRED = "RETIRED"
    TERMINAL = "TERMINAL"


class AssistIntent(str, Enum):
    NONE = "NONE"
    REQUEST_HOLD = "REQUEST_HOLD"
    REQUEST_OFFBOARD = "REQUEST_OFFBOARD"
    REQUEST_MISSION = "REQUEST_MISSION"
    REQUEST_RTL = "REQUEST_RTL"


@dataclass(frozen=True)
class MissionSnapshot:
    fingerprint: str
    current_seq: int
    total_items: int
    valid: bool
    observed_at: float


@dataclass(frozen=True)
class NativeTelemetry:
    mode: NativeMode
    armed: bool
    airborne: bool
    manual_override: bool
    failsafe: bool
    navigation_valid: bool
    vehicle_epoch: str
    navigation_epoch: str
    status_observed_at: float
    navigation_observed_at: float
    mission: MissionSnapshot


@dataclass(frozen=True)
class AssistApproval:
    approval_id: str
    mission_fingerprint: str
    vehicle_epoch: str


@dataclass(frozen=True)
class TransitionToken:
    """Correlation token, not a command or a transport authentication secret."""

    approval_id: str
    episode: int
    serial: int
    target_mode: NativeMode
    mission_fingerprint: str
    mission_seq: int
    issued_at: float
    expires_at: float


@dataclass(frozen=True)
class TransitionAcknowledgement:
    token: TransitionToken
    accepted: bool
    observed_at: float


@dataclass(frozen=True)
class SafetyProof:
    confirmed: bool
    observed_at: float


@dataclass(frozen=True)
class RecoveryEvidence:
    """Upstream geometry proofs, not a replacement for the geometry algorithm.

    The producer must keep all existing distance/uncertainty requirements. In
    particular, absence of a depth hit is not a positive descent-volume proof.
    """

    approval_id: str
    episode: int
    mission_fingerprint: str
    mission_seq: int
    obstacle_clear: SafetyProof
    descent_volume_clear: SafetyProof
    route_rejoined: SafetyProof


@dataclass(frozen=True)
class ArbitrationResult:
    phase: AssistPhase
    intent: AssistIntent
    reason: str
    token: TransitionToken | None = None


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite_time(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0.0


def _fresh(observed_at: object, now: float) -> bool:
    return _finite_time(observed_at) and 0.0 <= now - observed_at <= MAX_EVIDENCE_AGE_S


class NativeMissionArbitrator:
    """One explicit approval lifetime; retirement cannot be reset or resumed.

    Each healthy obstacle episode preserves the exact paused mission sequence.
    Progress is allowed only in MONITORING, after native mission ownership has
    been confirmed. A fresh instance AND a fresh explicit approval are required
    after terminal/retired states; this class never requests them automatically.
    """

    def __init__(self) -> None:
        self.phase = AssistPhase.DISABLED
        self._approval: AssistApproval | None = None
        self._vehicle_epoch = ""
        self._navigation_epoch = ""
        self._mission_fingerprint = ""
        self._mission_count = 0
        self._paused_seq: int | None = None
        self._episode = 0
        self._serial = 0
        self._pending: TransitionToken | None = None
        self._last_now: float | None = None
        self._avoidance_started_at: float | None = None
        self._reason = "assist is disabled"

    @property
    def episode(self) -> int:
        return self._episode

    @property
    def pending_transition(self) -> TransitionToken | None:
        return self._pending

    def _result(self, reason: str | None = None, intent: AssistIntent = AssistIntent.NONE) -> ArbitrationResult:
        return ArbitrationResult(self.phase, intent, reason or self._reason, self._pending)

    def _retire(self, reason: str) -> ArbitrationResult:
        self.phase = AssistPhase.RETIRED
        self._pending = None
        self._paused_seq = None
        self._avoidance_started_at = None
        self._reason = reason
        return self._result()

    def withdraw_approval(self, reason: str) -> ArbitrationResult:
        """Explicit operator/integration cancellation, never a mode command."""
        if self.phase in (AssistPhase.RETIRED, AssistPhase.TERMINAL):
            return self._result()
        detail = reason.strip() if _nonempty(reason) else "approval was withdrawn"
        return self._retire(detail)

    def _telemetry_failure(self, view: NativeTelemetry, now: float) -> str | None:
        if not _finite_time(now):
            return "invalid monotonic time"
        if self._last_now is not None and now < self._last_now:
            return "monotonic clock moved backwards"
        if self._last_now is not None and now - self._last_now > MAX_EVIDENCE_AGE_S:
            return "arbitration watchdog missed its evidence deadline"
        if not isinstance(view, NativeTelemetry) or not isinstance(view.mission, MissionSnapshot):
            return "invalid telemetry schema"
        flags = (view.armed, view.airborne, view.manual_override, view.failsafe, view.navigation_valid)
        if any(type(flag) is not bool for flag in flags):
            return "telemetry flags must be explicit booleans"
        if view.manual_override:
            return "manual control has priority"
        if view.failsafe:
            return "native failsafe has priority"
        if not view.armed or not view.airborne:
            return "vehicle is not armed and airborne"
        if not view.navigation_valid:
            return "local navigation is invalid"
        if not isinstance(view.mode, NativeMode):
            return "unknown navigation mode"
        if not _nonempty(view.vehicle_epoch) or not _nonempty(view.navigation_epoch):
            return "vehicle/navigation epoch is missing"
        if not _fresh(view.status_observed_at, now) or not _fresh(view.navigation_observed_at, now):
            return "vehicle status or local navigation is stale"
        mission = view.mission
        if mission.valid is not True or not _nonempty(mission.fingerprint):
            return "mission validity or fingerprint is missing"
        if type(mission.current_seq) is not int or type(mission.total_items) is not int:
            return "mission sequence/count must be integers"
        if not 0 <= mission.current_seq < mission.total_items:
            return "mission sequence is out of range"
        if not _fresh(mission.observed_at, now):
            return "mission snapshot is stale"
        return None

    def approve(self, approval: AssistApproval, view: NativeTelemetry, *, now: float) -> ArbitrationResult:
        if self.phase is not AssistPhase.DISABLED:
            return self._result("an existing approval lifetime cannot be replaced")
        failure = self._telemetry_failure(view, now)
        if failure:
            return self._result(failure)
        if not isinstance(approval, AssistApproval) or not _nonempty(approval.approval_id):
            return self._result("explicit assist approval is required")
        if approval.vehicle_epoch != view.vehicle_epoch or approval.mission_fingerprint != view.mission.fingerprint:
            return self._result("approval does not identify this vehicle and mission")
        if view.mode is not NativeMode.MISSION:
            return self._result("approval requires an already running native mission")
        self._approval = approval
        self._vehicle_epoch = view.vehicle_epoch
        self._navigation_epoch = view.navigation_epoch
        self._mission_fingerprint = view.mission.fingerprint
        self._mission_count = view.mission.total_items
        self._last_now = now
        self.phase = AssistPhase.MONITORING
        self._reason = "native mission is monitored without external setpoints"
        return self._result()

    def observe(self, view: NativeTelemetry, *, now: float) -> ArbitrationResult:
        if self.phase in (AssistPhase.DISABLED, AssistPhase.RETIRED, AssistPhase.TERMINAL):
            return self._result()
        failure = self._telemetry_failure(view, now)
        if failure:
            return self._retire(failure)
        self._last_now = now
        if (view.vehicle_epoch, view.navigation_epoch) != (self._vehicle_epoch, self._navigation_epoch):
            return self._retire("vehicle or navigation coordinate epoch changed")
        if (view.mission.fingerprint, view.mission.total_items) != (self._mission_fingerprint, self._mission_count):
            return self._retire("approved mission was revised")
        if self._paused_seq is not None and view.mission.current_seq != self._paused_seq:
            return self._retire("paused mission sequence changed")
        if self._pending is not None and now > self._pending.expires_at:
            return self._retire("expected mode transition expired")
        allowed = {
            AssistPhase.MONITORING: (NativeMode.MISSION,),
            AssistPhase.PAUSE_PENDING: (NativeMode.MISSION, NativeMode.HOLD),
            AssistPhase.HOLD_CONFIRMED: (NativeMode.HOLD,),
            AssistPhase.TAKEOVER_PENDING: (NativeMode.HOLD, NativeMode.OFFBOARD),
            AssistPhase.AVOIDING: (NativeMode.OFFBOARD,),
            AssistPhase.RESUME_PENDING: (NativeMode.OFFBOARD, NativeMode.MISSION),
        }[self.phase]
        if view.mode not in allowed:
            return self._retire("unrequested native/manual mode transition")
        return self._result()

    def _request(self, phase: AssistPhase, target: NativeMode, intent: AssistIntent, now: float) -> ArbitrationResult:
        assert self._approval is not None and self._paused_seq is not None
        self._serial += 1
        self._pending = TransitionToken(
            self._approval.approval_id, self._episode, self._serial, target,
            self._mission_fingerprint, self._paused_seq, now, now + MAX_EVIDENCE_AGE_S,
        )
        self.phase = phase
        self._reason = "mode acknowledgement and matching observed mode are required"
        return self._result(intent=intent)

    def pause_for_obstacle(self, view: NativeTelemetry, *, now: float) -> ArbitrationResult:
        self.observe(view, now=now)
        if self.phase is not AssistPhase.MONITORING:
            return self._result("pause is not available in the current phase")
        self._episode += 1
        self._paused_seq = view.mission.current_seq
        return self._request(AssistPhase.PAUSE_PENDING, NativeMode.HOLD, AssistIntent.REQUEST_HOLD, now)

    def request_takeover(self, view: NativeTelemetry, *, now: float) -> ArbitrationResult:
        self.observe(view, now=now)
        if self.phase is not AssistPhase.HOLD_CONFIRMED:
            return self._result("takeover requires an acknowledged native HOLD")
        return self._request(AssistPhase.TAKEOVER_PENDING, NativeMode.OFFBOARD, AssistIntent.REQUEST_OFFBOARD, now)

    def confirm_transition(
        self, acknowledgement: TransitionAcknowledgement, view: NativeTelemetry, *, now: float
    ) -> ArbitrationResult:
        self.observe(view, now=now)
        token = self._pending
        if token is None or not isinstance(acknowledgement, TransitionAcknowledgement) or acknowledgement.token != token:
            return self._result("unmatched or already consumed transition acknowledgement")
        if acknowledgement.accepted is not True:
            return self._retire("mode transition was not explicitly accepted")
        if not _fresh(acknowledgement.observed_at, now) or acknowledgement.observed_at < token.issued_at:
            return self._retire("mode acknowledgement is stale or predates its request")
        if view.mode is not token.target_mode:
            return self._result("accepted mode command has not been observed yet")
        if view.status_observed_at < token.issued_at:
            return self._result("matching navigation mode predates its request")
        self._pending = None
        if self.phase is AssistPhase.PAUSE_PENDING:
            self.phase = AssistPhase.HOLD_CONFIRMED
            self._reason = "native HOLD is confirmed; no takeoff or arming is authorized"
        elif self.phase is AssistPhase.TAKEOVER_PENDING:
            self.phase = AssistPhase.AVOIDING
            self._avoidance_started_at = now
            self._reason = "bounded avoidance epoch is active"
        elif self.phase is AssistPhase.RESUME_PENDING:
            self.phase = AssistPhase.MONITORING
            self._paused_seq = None
            self._avoidance_started_at = None
            self._reason = "same native mission and sequence resumed; native owns all setpoints"
        return self._result()

    def avoidance_authorized(self, view: NativeTelemetry, *, now: float) -> bool:
        """One-shot check; callers must also enforce their existing safety gate.

        False throughout native ownership, takeover priming and resume pending.
        In particular, a requested handback stops external trajectory ownership
        immediately; a delayed native-mode acknowledgement does not extend it.
        """
        self.observe(view, now=now)
        return self.phase is AssistPhase.AVOIDING

    def request_resume(self, evidence: RecoveryEvidence, view: NativeTelemetry, *, now: float) -> ArbitrationResult:
        self.observe(view, now=now)
        if self.phase is not AssistPhase.AVOIDING:
            return self._result("resume requires the currently owned avoidance epoch")
        assert self._approval is not None and self._avoidance_started_at is not None
        if not isinstance(evidence, RecoveryEvidence) or (
            evidence.approval_id, evidence.episode, evidence.mission_fingerprint, evidence.mission_seq
        ) != (self._approval.approval_id, self._episode, self._mission_fingerprint, self._paused_seq):
            return self._result("recovery evidence does not identify this paused mission episode")
        if type(evidence.episode) is not int or type(evidence.mission_seq) is not int:
            return self._result("recovery episode and mission sequence must be integers")
        proofs = (evidence.obstacle_clear, evidence.descent_volume_clear, evidence.route_rejoined)
        if any(
            not isinstance(proof, SafetyProof) or proof.confirmed is not True
            or not _fresh(proof.observed_at, now) or proof.observed_at < self._avoidance_started_at
            for proof in proofs
        ):
            return self._result("fresh positive obstacle-clear, descent-volume and rejoin proofs are all required")
        return self._request(AssistPhase.RESUME_PENDING, NativeMode.MISSION, AssistIntent.REQUEST_MISSION, now)

    def terminal_event(self, view: NativeTelemetry, *, now: float) -> ArbitrationResult:
        """Finish the event flow with one terminal RTL intent, never a resume.

        Only an owned and currently healthy OFFBOARD episode can request RTL.
        An operator/native takeover, stale data, or previous retirement instead
        produces NONE. Either way, late ACKs and fresh telemetry cannot revive
        this approval. Recording success/failure is the event owner's policy.
        """
        if self.phase in (AssistPhase.RETIRED, AssistPhase.TERMINAL):
            return self._result()
        self.observe(view, now=now)
        if self.phase is AssistPhase.RETIRED:
            return self._result()
        owned = self.phase in (AssistPhase.AVOIDING, AssistPhase.RESUME_PENDING) and view.mode is NativeMode.OFFBOARD
        self.phase = AssistPhase.TERMINAL
        self._pending = None
        self._paused_seq = None
        self._avoidance_started_at = None
        self._reason = "event is terminal; no resumable mission epoch remains"
        return self._result(intent=AssistIntent.REQUEST_RTL if owned else AssistIntent.NONE)

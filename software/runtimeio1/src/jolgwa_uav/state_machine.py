from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from time import monotonic

from .models import AfterResponse, EventName, MissionPlan, PlanStatus, ResponseAction
from .zones import Route, Waypoint


class MissionPhase(StrEnum):
    IDLE = "IDLE"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    TAKEOFF = "TAKEOFF"
    PATROL = "PATROL"
    RESPONDING = "RESPONDING"
    RETURNING_HOME = "RETURNING_HOME"
    PAUSED_MANUAL = "PAUSED_MANUAL"
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"


@dataclass(frozen=True)
class EventResponse:
    event: EventName
    actions: tuple[ResponseAction, ...]
    resume_waypoint_index: int


@dataclass(frozen=True)
class MissionProgress:
    phase: MissionPhase
    waypoint_index: int
    waypoint_count: int
    completed_laps: int
    patrol_elapsed_s: float
    active_event: EventName | None


class MissionRuntime:
    """Deterministic mission state machine; it never calls an LLM."""

    def __init__(self) -> None:
        self.phase = MissionPhase.IDLE
        self.plan: MissionPlan | None = None
        self.route: Route | None = None
        self.waypoint_index = 0
        self.completed_laps = 0
        self.patrol_elapsed_s = 0.0
        self._patrol_started_at: float | None = None
        self.active_response: EventResponse | None = None
        self._phase_before_manual: MissionPhase | None = None

    def submit(self, plan: MissionPlan, route: Route) -> None:
        if self.phase not in {
            MissionPhase.IDLE,
            MissionPhase.COMPLETED,
            MissionPhase.ABORTED,
        }:
            raise RuntimeError(f"cannot submit a mission while phase is {self.phase}")
        if plan.status is not PlanStatus.OK:
            raise ValueError("only an OK plan can be submitted")
        if not route.waypoints_per_lap:
            raise ValueError("route has no waypoints")
        self.plan = plan
        self.route = route
        self.waypoint_index = 0
        self.completed_laps = 0
        self.patrol_elapsed_s = 0.0
        self._patrol_started_at = None
        self.active_response = None
        self.phase = MissionPhase.AWAITING_APPROVAL

    def approve(self) -> None:
        self._require(MissionPhase.AWAITING_APPROVAL)
        self.phase = MissionPhase.TAKEOFF

    def airborne(self, now: float | None = None) -> None:
        self._require(MissionPhase.TAKEOFF)
        self.phase = MissionPhase.PATROL
        self._patrol_started_at = monotonic() if now is None else now

    def current_waypoint(self) -> Waypoint | None:
        if self.route is None or self.phase not in {
            MissionPhase.PATROL,
            MissionPhase.RESPONDING,
        }:
            return None
        return self.route.waypoints_per_lap[self.waypoint_index]

    def waypoint_reached(self, now: float | None = None) -> None:
        self._require(MissionPhase.PATROL)
        self._update_patrol_clock(now)
        assert self.route is not None
        self.waypoint_index += 1
        if self.waypoint_index >= len(self.route.waypoints_per_lap):
            self.waypoint_index = 0
            self.completed_laps += 1
            if (
                self.route.lap_target is not None
                and self.completed_laps >= self.route.lap_target
            ):
                self.phase = MissionPhase.RETURNING_HOME

    def tick(self, now: float | None = None) -> None:
        if self.phase is not MissionPhase.PATROL or self.route is None:
            return
        self._update_patrol_clock(now)
        if (
            self.route.duration_s is not None
            and self.patrol_elapsed_s >= self.route.duration_s
        ):
            self.phase = MissionPhase.RETURNING_HOME

    def observe_event(
        self,
        event: EventName,
        now: float | None = None,
    ) -> EventResponse | None:
        if self.phase is not MissionPhase.PATROL or self.plan is None:
            return None
        if event not in self.plan.expanded_events():
            return None
        actions = self.plan.actions_for(event)
        if not actions:
            return None
        self._pause_patrol_clock(now)
        response = EventResponse(event, actions, self.waypoint_index)
        self.active_response = response
        self.phase = MissionPhase.RESPONDING
        return response

    def response_completed(self, now: float | None = None) -> None:
        self._require(MissionPhase.RESPONDING)
        assert self.plan is not None
        self.active_response = None
        if self.plan.after_response is AfterResponse.RETURN_HOME:
            self.phase = MissionPhase.RETURNING_HOME
        else:
            self.phase = MissionPhase.PATROL
            self._patrol_started_at = monotonic() if now is None else now

    def manual_override(self, now: float | None = None) -> None:
        if self.phase in {
            MissionPhase.IDLE,
            MissionPhase.COMPLETED,
            MissionPhase.ABORTED,
        }:
            return
        self._pause_patrol_clock(now)
        self._phase_before_manual = self.phase
        self.phase = MissionPhase.PAUSED_MANUAL

    def resume_autonomy(self, now: float | None = None) -> None:
        self._require(MissionPhase.PAUSED_MANUAL)
        target = self._phase_before_manual or MissionPhase.PATROL
        if target not in {MissionPhase.PATROL, MissionPhase.RESPONDING}:
            target = MissionPhase.PATROL
        self.phase = target
        if target is MissionPhase.PATROL:
            self._patrol_started_at = monotonic() if now is None else now
        self._phase_before_manual = None

    def landed(self) -> None:
        self._require(MissionPhase.RETURNING_HOME)
        self.phase = MissionPhase.COMPLETED

    def abort(self, now: float | None = None) -> None:
        self._pause_patrol_clock(now)
        self.phase = MissionPhase.ABORTED

    def progress(self) -> MissionProgress:
        waypoint_count = len(self.route.waypoints_per_lap) if self.route else 0
        event = self.active_response.event if self.active_response else None
        return MissionProgress(
            phase=self.phase,
            waypoint_index=self.waypoint_index,
            waypoint_count=waypoint_count,
            completed_laps=self.completed_laps,
            patrol_elapsed_s=self.patrol_elapsed_s,
            active_event=event,
        )

    def _update_patrol_clock(self, now: float | None) -> None:
        if self._patrol_started_at is None:
            return
        current = monotonic() if now is None else now
        self.patrol_elapsed_s += max(0.0, current - self._patrol_started_at)
        self._patrol_started_at = current

    def _pause_patrol_clock(self, now: float | None = None) -> None:
        if self.phase is MissionPhase.PATROL:
            self._update_patrol_clock(now)
        self._patrol_started_at = None

    def _require(self, expected: MissionPhase) -> None:
        if self.phase is not expected:
            raise RuntimeError(f"expected phase {expected}, got {self.phase}")

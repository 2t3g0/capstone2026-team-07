import math
import uuid
from dataclasses import dataclass, replace
from enum import Enum
from typing import Sequence


PHASE1_EVENT_TYPES = frozenset(
    {
        "FIRE_SMOKE",
        "HUMAN_VIOLENCE",
        "LITTERING",
        "INTRUSION_ATTEMPT",
        "CALL_FOR_HELP",
        "VEHICLE_ACCIDENT",
    }
)
PHASE1_TO_MISSION_EVENT = {
    "FIRE_SMOKE": "FIRE",
    "HUMAN_VIOLENCE": "VIOLENCE",
    "LITTERING": "LITTERING",
    "INTRUSION_ATTEMPT": "RESTRICTED_ENTRY",
    "CALL_FOR_HELP": "CALL_FOR_HELP",
    "VEHICLE_ACCIDENT": "TRAFFIC_ACCIDENT",
}


def event_terminal_intent(
    *, policy: str, capture_succeeded: bool, autonomy_available: bool
) -> str:
    """Select an intent, never a command or permission to reacquire control.

    The command owner must independently check its current flight epoch and
    manual/native-failsafe ownership immediately before acting on this intent.
    """
    if policy not in {"legacy_rejoin", "capture_then_rtl"}:
        raise ValueError("unknown event completion policy")
    if type(capture_succeeded) is not bool or type(autonomy_available) is not bool:
        raise ValueError("capture and autonomy states must be explicit booleans")
    if not autonomy_available:
        return "NO_AUTONOMY_REACQUIRE"
    if policy == "capture_then_rtl":
        return "RTL"
    return "REJOIN_ROUTE" if capture_succeeded else "REQUIRE_RESUME"


class EventLeaseState(str, Enum):
    REQUESTED = "REQUESTED"
    ACTIVE = "ACTIVE"
    RELEASE_REQUESTED = "RELEASE_REQUESTED"
    REJOINING = "REJOINING"


@dataclass
class EventControlLease:
    lease_id: str
    mission_id: str
    event_id: str
    event_type: str
    track_id: str
    confidence: float
    source: str
    requested_at: float
    expires_at: float
    state: EventLeaseState = EventLeaseState.REQUESTED
    capture_succeeded: bool = False
    release_reason: str = ""


class EventControlCoordinator:
    """Own one bounded Jetson event-capture lease at a time."""

    def __init__(
        self,
        max_duration_s: float = 5.0,
        allowed_event_types: frozenset[str] = PHASE1_EVENT_TYPES,
    ) -> None:
        if not math.isfinite(max_duration_s) or max_duration_s <= 0.0:
            raise ValueError("max_duration_s must be finite and positive")
        self.max_duration_s = float(max_duration_s)
        self.allowed_event_types = frozenset(
            str(value).strip() for value in allowed_event_types if str(value).strip()
        )
        if not self.allowed_event_types:
            raise ValueError("allowed_event_types must not be empty")
        self._lease: EventControlLease | None = None

    def snapshot(self) -> EventControlLease | None:
        return replace(self._lease) if self._lease is not None else None

    def request(
        self,
        *,
        mission_id: str,
        event_id: str,
        event_type: str,
        track_id: str,
        confidence: float,
        source: str,
        now: float,
    ) -> EventControlLease:
        if self._lease is not None:
            raise ValueError("an event control lease is already active")
        values = {
            "mission_id": mission_id,
            "event_id": event_id,
            "event_type": event_type,
            "track_id": track_id,
            "source": source,
        }
        for name, value in values.items():
            if not str(value).strip():
                raise ValueError(f"{name} must not be empty")
        event_type = str(event_type).strip()
        if event_type not in self.allowed_event_types:
            raise ValueError(f"unsupported Phase 1 event type: {event_type}")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not math.isfinite(now):
            raise ValueError("now must be finite")
        self._lease = EventControlLease(
            lease_id=str(uuid.uuid4()),
            mission_id=mission_id,
            event_id=event_id,
            event_type=event_type,
            track_id=track_id,
            confidence=float(confidence),
            source=source,
            requested_at=float(now),
            expires_at=float(now) + self.max_duration_s,
        )
        return self.snapshot()

    def activate(self, lease_id: str, *, now: float | None = None) -> EventControlLease:
        lease = self._require(lease_id)
        if lease.state is not EventLeaseState.REQUESTED:
            raise ValueError("event lease is not waiting for activation")
        activation_time = lease.requested_at if now is None else float(now)
        if not math.isfinite(activation_time):
            raise ValueError("now must be finite")
        # The capture budget begins only after mission_manager has issued HOLD
        # and confirmed that the aircraft has settled. Service/queue latency must
        # never shorten the requested recording duration.
        lease.expires_at = activation_time + self.max_duration_s
        lease.state = EventLeaseState.ACTIVE
        return self.snapshot()

    def request_release(
        self,
        *,
        lease_id: str,
        source: str,
        capture_succeeded: bool,
        reason: str,
    ) -> EventControlLease:
        lease = self._require(lease_id)
        if source != lease.source:
            raise ValueError("event lease source does not match")
        if lease.state is EventLeaseState.RELEASE_REQUESTED:
            # Release is intentionally idempotent. The event recorder retries
            # when a ROS service reply is lost, while the first outcome remains
            # authoritative.
            return self.snapshot()
        if lease.state not in (
            EventLeaseState.REQUESTED,
            EventLeaseState.ACTIVE,
        ):
            raise ValueError("event lease is already being released")
        lease.capture_succeeded = bool(capture_succeeded)
        lease.release_reason = reason.strip() or "capture complete"
        lease.state = EventLeaseState.RELEASE_REQUESTED
        return self.snapshot()

    def expire(self, now: float) -> EventControlLease | None:
        lease = self._lease
        if lease is None or lease.state in (
            EventLeaseState.RELEASE_REQUESTED,
            EventLeaseState.REJOINING,
        ):
            return self.snapshot()
        if now >= lease.expires_at:
            lease.capture_succeeded = False
            lease.release_reason = (
                "event control lease reached the "
                f"{self.max_duration_s:g} second capture limit"
            )
            lease.state = EventLeaseState.RELEASE_REQUESTED
        return self.snapshot()

    def begin_rejoin(self, lease_id: str) -> EventControlLease:
        lease = self._require(lease_id)
        if lease.state is not EventLeaseState.RELEASE_REQUESTED:
            raise ValueError("event lease has not been released")
        lease.state = EventLeaseState.REJOINING
        return self.snapshot()

    def clear(self, lease_id: str) -> None:
        self._require(lease_id)
        self._lease = None

    def clear_for_mission(self, mission_id: str) -> None:
        """Drop a lease when its owning mission has finished or aborted."""

        if self._lease is not None and self._lease.mission_id == mission_id:
            self._lease = None

    def _require(self, lease_id: str) -> EventControlLease:
        if self._lease is None or self._lease.lease_id != lease_id:
            raise ValueError("unknown event control lease")
        return self._lease


@dataclass(frozen=True)
class RouteRejoinTarget:
    segment_index: int
    fraction: float
    position_ned_m: tuple[float, float, float]
    distance_m: float


def nearest_route_segment(
    position_ned_m: Sequence[float],
    route_ned_m: Sequence[Sequence[float]],
) -> RouteRejoinTarget:
    """Project a 3-D NED position onto the nearest open route segment."""

    if len(position_ned_m) != 3:
        raise ValueError("position must have three NED coordinates")
    if len(route_ned_m) < 2:
        raise ValueError("route must contain at least two points")
    position = tuple(float(value) for value in position_ned_m)
    if not all(math.isfinite(value) for value in position):
        raise ValueError("position must be finite")

    best: RouteRejoinTarget | None = None
    for index, (raw_start, raw_end) in enumerate(
        zip(route_ned_m, route_ned_m[1:])
    ):
        if len(raw_start) != 3 or len(raw_end) != 3:
            raise ValueError("route points must have three NED coordinates")
        start = tuple(float(value) for value in raw_start)
        end = tuple(float(value) for value in raw_end)
        if not all(math.isfinite(value) for value in (*start, *end)):
            raise ValueError("route points must be finite")
        delta = tuple(end[axis] - start[axis] for axis in range(3))
        length_sq = sum(value * value for value in delta)
        if length_sq <= 1e-12:
            fraction = 0.0
        else:
            fraction = sum(
                (position[axis] - start[axis]) * delta[axis]
                for axis in range(3)
            ) / length_sq
            fraction = min(1.0, max(0.0, fraction))
        projected = tuple(
            start[axis] + fraction * delta[axis] for axis in range(3)
        )
        distance = math.sqrt(
            sum((position[axis] - projected[axis]) ** 2 for axis in range(3))
        )
        candidate = RouteRejoinTarget(index, fraction, projected, distance)
        if best is None or candidate.distance_m < best.distance_m:
            best = candidate
    assert best is not None
    return best


def route_rejoin_with_lookahead(
    position_ned_m: Sequence[float],
    route_ned_m: Sequence[Sequence[float]],
    lookahead_m: float,
) -> RouteRejoinTarget:
    """Project onto the route, then advance a bounded distance along it.

    Rejoining the exact projection can deadlock with an active obstacle pass:
    the avoidance layer may need a small amount of forward progress before it
    is safe to descend, while the event layer waits for descent at the current
    projection.  A forward target gives both layers the same progress goal.
    """

    lookahead = float(lookahead_m)
    if not math.isfinite(lookahead) or lookahead < 0.0:
        raise ValueError("route rejoin lookahead must be finite and non-negative")
    nearest = nearest_route_segment(position_ned_m, route_ned_m)
    if lookahead <= 1e-9:
        return nearest

    current = tuple(float(value) for value in nearest.position_ned_m)
    remaining = lookahead
    segment_index = nearest.segment_index
    fraction = nearest.fraction
    while segment_index < len(route_ned_m) - 1:
        end = tuple(float(value) for value in route_ned_m[segment_index + 1])
        delta = tuple(end[axis] - current[axis] for axis in range(3))
        distance = math.sqrt(sum(value * value for value in delta))
        if distance > 1e-9 and remaining <= distance:
            ratio = remaining / distance
            target = tuple(
                current[axis] + ratio * delta[axis] for axis in range(3)
            )
            original_start = tuple(
                float(value) for value in route_ned_m[segment_index]
            )
            original_end = tuple(
                float(value) for value in route_ned_m[segment_index + 1]
            )
            original_delta = tuple(
                original_end[axis] - original_start[axis] for axis in range(3)
            )
            original_length_sq = sum(value * value for value in original_delta)
            target_fraction = (
                0.0
                if original_length_sq <= 1e-12
                else sum(
                    (target[axis] - original_start[axis]) * original_delta[axis]
                    for axis in range(3)
                )
                / original_length_sq
            )
            return RouteRejoinTarget(
                segment_index,
                min(1.0, max(0.0, target_fraction)),
                target,
                math.sqrt(
                    sum(
                        (float(position_ned_m[axis]) - target[axis]) ** 2
                        for axis in range(3)
                    )
                ),
            )
        remaining -= distance
        current = end
        segment_index += 1
        fraction = 0.0

    endpoint = tuple(float(value) for value in route_ned_m[-1])
    return RouteRejoinTarget(
        len(route_ned_m) - 2,
        1.0,
        endpoint,
        math.sqrt(
            sum(
                (float(position_ned_m[axis]) - endpoint[axis]) ** 2
                for axis in range(3)
            )
        ),
    )

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math


class AvoidanceState(StrEnum):
    CLEAR = "CLEAR"
    SLOW = "SLOW"
    HOLD = "HOLD"
    EVADE = "EVADE"
    STALE = "STALE"


class AvoidanceDirection(StrEnum):
    FORWARD = "FORWARD"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    STOP = "STOP"


@dataclass(frozen=True, slots=True)
class RangeReading:
    """One range measurement and its age at decision time."""

    distance_m: float
    age_s: float

    @property
    def valid(self) -> bool:
        return (
            math.isfinite(self.distance_m)
            and self.distance_m >= 0.0
            and math.isfinite(self.age_s)
            and self.age_s >= 0.0
        )


@dataclass(frozen=True, slots=True)
class ObstacleSnapshot:
    front: RangeReading | None
    left: RangeReading | None
    right: RangeReading | None
    down: RangeReading | None
    current_speed_mps: float


@dataclass(frozen=True, slots=True)
class AvoidanceDecision:
    state: AvoidanceState
    direction: AvoidanceDirection
    forward_speed_limit_mps: float
    lateral_speed_mps: float = 0.0
    vertical_speed_mps: float = 0.0
    reasons: tuple[str, ...] = ()

    @property
    def must_hold(self) -> bool:
        return self.state in {AvoidanceState.HOLD, AvoidanceState.STALE}


@dataclass(frozen=True, slots=True)
class SafetyConfig:
    stale_timeout_s: float = 0.30
    stale_recovery_samples: int = 2
    reaction_time_s: float = 0.25
    max_deceleration_mps2: float = 3.0
    max_cruise_speed_mps: float = 4.0
    slow_speed_mps: float = 1.5
    evade_forward_speed_mps: float = 0.6
    lateral_evade_speed_mps: float = 0.8
    vertical_evade_speed_mps: float = 0.6
    front_hold_m: float = 0.8
    front_evade_m: float = 2.0
    front_slow_m: float = 4.0
    side_hold_m: float = 0.55
    side_evade_m: float = 1.5
    side_escape_m: float = 2.2
    down_hold_m: float = 0.35
    down_slow_m: float = 0.75
    hysteresis_m: float = 0.4
    direction_tie_m: float = 0.25

    def __post_init__(self) -> None:
        positive = {
            "stale_timeout_s": self.stale_timeout_s,
            "reaction_time_s": self.reaction_time_s,
            "max_deceleration_mps2": self.max_deceleration_mps2,
            "max_cruise_speed_mps": self.max_cruise_speed_mps,
            "slow_speed_mps": self.slow_speed_mps,
            "lateral_evade_speed_mps": self.lateral_evade_speed_mps,
            "vertical_evade_speed_mps": self.vertical_evade_speed_mps,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError(f"configuration values must be positive: {invalid}")
        if self.stale_recovery_samples < 1:
            raise ValueError("stale_recovery_samples must be at least 1")
        if not 0.0 <= self.evade_forward_speed_mps <= self.slow_speed_mps:
            raise ValueError("evade speed must be between zero and slow speed")
        if self.slow_speed_mps > self.max_cruise_speed_mps:
            raise ValueError("slow speed cannot exceed cruise speed")
        if not self.front_hold_m < self.front_evade_m < self.front_slow_m:
            raise ValueError("front thresholds must increase from hold to slow")
        if not self.side_hold_m < self.side_evade_m <= self.side_escape_m:
            raise ValueError("side thresholds must increase from hold to escape")
        if not self.down_hold_m < self.down_slow_m:
            raise ValueError("down_hold_m must be below down_slow_m")
        if self.hysteresis_m < 0.0 or self.direction_tie_m < 0.0:
            raise ValueError("hysteresis values cannot be negative")


class ObstacleAvoidanceCore:
    """Deterministic local safety policy with no network or model dependency.

    Positive lateral velocity means left and positive vertical velocity means up.
    The caller remains responsible for combining these limits with its flight
    controller and enforcing the vehicle's absolute velocity limits.
    """

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()
        self._previous_state = AvoidanceState.CLEAR
        self._previous_direction = AvoidanceDirection.FORWARD
        self._fresh_samples = self.config.stale_recovery_samples

    @property
    def previous_state(self) -> AvoidanceState:
        return self._previous_state

    def reset(self) -> None:
        self._previous_state = AvoidanceState.CLEAR
        self._previous_direction = AvoidanceDirection.FORWARD
        self._fresh_samples = self.config.stale_recovery_samples

    def evaluate(self, snapshot: ObstacleSnapshot) -> AvoidanceDecision:
        stale_reasons = self._stale_reasons(snapshot)
        if stale_reasons:
            self._fresh_samples = 0
            return self._remember(self._stale(stale_reasons))

        if self._previous_state is AvoidanceState.STALE:
            self._fresh_samples += 1
            if self._fresh_samples < self.config.stale_recovery_samples:
                return self._remember(
                    self._stale(("waiting_for_consecutive_fresh_samples",))
                )

        front = snapshot.front.distance_m  # type: ignore[union-attr]
        left = snapshot.left.distance_m  # type: ignore[union-attr]
        right = snapshot.right.distance_m  # type: ignore[union-attr]
        down = snapshot.down.distance_m  # type: ignore[union-attr]
        speed = abs(snapshot.current_speed_mps)
        braking_margin = (
            speed * self.config.reaction_time_s
            + speed * speed / (2.0 * self.config.max_deceleration_mps2)
        )
        hysteresis = (
            self.config.hysteresis_m
            if self._previous_state not in {AvoidanceState.CLEAR, AvoidanceState.STALE}
            else 0.0
        )

        front_hold = self.config.front_hold_m + braking_margin
        front_evade = self.config.front_evade_m + braking_margin
        front_slow = self.config.front_slow_m + braking_margin

        if down <= self.config.down_hold_m + hysteresis:
            return self._remember(
                self._evade(
                    AvoidanceDirection.UP,
                    forward_limit=0.0,
                    reason="down_clearance_critical",
                )
            )

        left_critical = left <= self.config.side_hold_m + hysteresis
        right_critical = right <= self.config.side_hold_m + hysteresis
        if left_critical and right_critical:
            return self._remember(self._hold("both_sides_critical"))
        if left_critical:
            return self._remember(self._side_evade(AvoidanceDirection.RIGHT, "left_critical"))
        if right_critical:
            return self._remember(self._side_evade(AvoidanceDirection.LEFT, "right_critical"))

        if front <= front_hold + hysteresis:
            direction = self._escape_direction(left, right, hysteresis)
            if direction is None:
                return self._remember(self._hold("front_critical_no_escape"))
            return self._remember(self._side_evade(direction, "front_critical"))

        if front <= front_evade + hysteresis:
            direction = self._escape_direction(left, right, hysteresis)
            if direction is None:
                return self._remember(self._hold("front_blocked_no_escape"))
            return self._remember(self._side_evade(direction, "front_obstacle"))

        left_near = left <= self.config.side_evade_m + hysteresis
        right_near = right <= self.config.side_evade_m + hysteresis
        if left_near and not right_near:
            return self._remember(self._side_evade(AvoidanceDirection.RIGHT, "left_obstacle"))
        if right_near and not left_near:
            return self._remember(self._side_evade(AvoidanceDirection.LEFT, "right_obstacle"))

        slow_reasons: list[str] = []
        if front <= front_slow + hysteresis:
            slow_reasons.append("front_within_slow_envelope")
        if left_near and right_near:
            slow_reasons.append("narrow_side_clearance")
        if down <= self.config.down_slow_m + hysteresis:
            slow_reasons.append("low_down_clearance")
        if slow_reasons:
            return self._remember(
                AvoidanceDecision(
                    state=AvoidanceState.SLOW,
                    direction=AvoidanceDirection.FORWARD,
                    forward_speed_limit_mps=self.config.slow_speed_mps,
                    reasons=tuple(slow_reasons),
                )
            )

        return self._remember(
            AvoidanceDecision(
                state=AvoidanceState.CLEAR,
                direction=AvoidanceDirection.FORWARD,
                forward_speed_limit_mps=self.config.max_cruise_speed_mps,
            )
        )

    def _stale_reasons(self, snapshot: ObstacleSnapshot) -> tuple[str, ...]:
        reasons: list[str] = []
        for name in ("front", "left", "right", "down"):
            reading = getattr(snapshot, name)
            if reading is None:
                reasons.append(f"{name}_missing")
            elif not reading.valid:
                reasons.append(f"{name}_invalid")
            elif reading.age_s > self.config.stale_timeout_s:
                reasons.append(f"{name}_stale")
        if not math.isfinite(snapshot.current_speed_mps):
            reasons.append("speed_invalid")
        return tuple(reasons)

    def _escape_direction(
        self, left: float, right: float, hysteresis: float
    ) -> AvoidanceDirection | None:
        required = self.config.side_escape_m + hysteresis
        left_open = left >= required
        right_open = right >= required
        if not left_open and not right_open:
            return None
        if left_open and not right_open:
            return AvoidanceDirection.LEFT
        if right_open and not left_open:
            return AvoidanceDirection.RIGHT
        if (
            self._previous_state is AvoidanceState.EVADE
            and self._previous_direction in {AvoidanceDirection.LEFT, AvoidanceDirection.RIGHT}
            and abs(left - right) <= self.config.direction_tie_m
        ):
            return self._previous_direction
        return AvoidanceDirection.LEFT if left >= right else AvoidanceDirection.RIGHT

    def _side_evade(
        self, direction: AvoidanceDirection, reason: str
    ) -> AvoidanceDecision:
        sign = 1.0 if direction is AvoidanceDirection.LEFT else -1.0
        return self._evade(
            direction,
            forward_limit=self.config.evade_forward_speed_mps,
            lateral_speed=sign * self.config.lateral_evade_speed_mps,
            reason=reason,
        )

    def _evade(
        self,
        direction: AvoidanceDirection,
        *,
        forward_limit: float,
        lateral_speed: float = 0.0,
        reason: str,
    ) -> AvoidanceDecision:
        vertical_speed = (
            self.config.vertical_evade_speed_mps
            if direction is AvoidanceDirection.UP
            else 0.0
        )
        return AvoidanceDecision(
            state=AvoidanceState.EVADE,
            direction=direction,
            forward_speed_limit_mps=forward_limit,
            lateral_speed_mps=lateral_speed,
            vertical_speed_mps=vertical_speed,
            reasons=(reason,),
        )

    @staticmethod
    def _hold(reason: str) -> AvoidanceDecision:
        return AvoidanceDecision(
            state=AvoidanceState.HOLD,
            direction=AvoidanceDirection.STOP,
            forward_speed_limit_mps=0.0,
            reasons=(reason,),
        )

    @staticmethod
    def _stale(reasons: tuple[str, ...]) -> AvoidanceDecision:
        return AvoidanceDecision(
            state=AvoidanceState.STALE,
            direction=AvoidanceDirection.STOP,
            forward_speed_limit_mps=0.0,
            reasons=reasons,
        )

    def _remember(self, decision: AvoidanceDecision) -> AvoidanceDecision:
        self._previous_state = decision.state
        self._previous_direction = decision.direction
        return decision

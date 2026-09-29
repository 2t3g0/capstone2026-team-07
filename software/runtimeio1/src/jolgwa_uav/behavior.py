from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable


class PoseClass(IntEnum):
    """Pose values defined by ``Gazebo JSON 설명.md``."""

    UNKNOWN = 0
    STANDING = 1
    HAND_RAISE = 2
    FALLEN = 3
    AIMING = 4


class BehaviorEvent(IntEnum):
    """Behavior values sent in the perception JSON.

    ``HELP_SIGNAL`` is kept as a wire-compatible alias for the document name.
    The mission-facing canonical name is ``CALL_FOR_HELP``.
    """

    NONE = 0
    CALL_FOR_HELP = 1
    HELP_SIGNAL = 1
    FALL = 2


POSE_TO_EVENT: dict[PoseClass, BehaviorEvent] = {
    PoseClass.HAND_RAISE: BehaviorEvent.CALL_FOR_HELP,
    PoseClass.FALLEN: BehaviorEvent.FALL,
}


@dataclass(frozen=True, slots=True)
class BehaviorConfig:
    hand_raise_seconds: float = 1.0
    fallen_seconds: float = 2.0
    confidence_threshold: float = 0.6
    missing_tolerance_seconds: float = 0.35
    event_cooldown_seconds: float = 5.0
    track_ttl_seconds: float = 3.0

    def __post_init__(self) -> None:
        finite_nonnegative = {
            "hand_raise_seconds": self.hand_raise_seconds,
            "fallen_seconds": self.fallen_seconds,
            "missing_tolerance_seconds": self.missing_tolerance_seconds,
            "event_cooldown_seconds": self.event_cooldown_seconds,
            "track_ttl_seconds": self.track_ttl_seconds,
        }
        for name, value in finite_nonnegative.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.confidence_threshold) or not (
            0.0 <= self.confidence_threshold <= 1.0
        ):
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.track_ttl_seconds <= self.missing_tolerance_seconds:
            raise ValueError("track_ttl_seconds must exceed missing_tolerance_seconds")

    def confirmation_seconds(self, event: BehaviorEvent) -> float:
        if event is BehaviorEvent.CALL_FOR_HELP:
            return self.hand_raise_seconds
        if event is BehaviorEvent.FALL:
            return self.fallen_seconds
        raise ValueError(f"event has no confirmation duration: {event!r}")


@dataclass(frozen=True, slots=True)
class BehaviorObservation:
    track_id: int
    pose_class: PoseClass | int
    confidence: float

    def __post_init__(self) -> None:
        if self.track_id < 0:
            raise ValueError("track_id must be non-negative")
        try:
            pose = PoseClass(self.pose_class)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid pose_class: {self.pose_class!r}") from exc
        if not math.isfinite(self.confidence) or not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "pose_class", pose)


@dataclass(frozen=True, slots=True)
class BehaviorDecision:
    track_id: int
    event: BehaviorEvent
    confidence: float
    event_duration_s: float
    confirmed: bool
    emitted: bool = False
    cleared_events: tuple[BehaviorEvent, ...] = ()
    track_expired: bool = False

    @property
    def behavior_event(self) -> int:
        return int(self.event)

    @property
    def behavior_confidence(self) -> float:
        return self.confidence

    @property
    def cleared(self) -> bool:
        return bool(self.cleared_events)


@dataclass(slots=True)
class _TrackState:
    last_seen_at: float
    candidate: BehaviorEvent = BehaviorEvent.NONE
    candidate_started_at: float | None = None
    candidate_min_confidence: float = 0.0
    confirmed: BehaviorEvent = BehaviorEvent.NONE
    last_emitted_at: dict[BehaviorEvent, float] = field(default_factory=dict)


class BehaviorEventCore:
    """Deterministic temporal behavior classifier keyed by ByteTrack ID.

    Call :meth:`process_frame` once for every perception frame, including frames
    with no people. An observation below the confidence threshold is treated as
    missing, not as evidence that a different pose was seen.
    """

    def __init__(self, config: BehaviorConfig | None = None) -> None:
        self.config = config or BehaviorConfig()
        self._tracks: dict[int, _TrackState] = {}
        self._last_timestamp: float | None = None

    @property
    def track_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._tracks))

    def process_frame(
        self,
        timestamp: float,
        observations: Iterable[BehaviorObservation],
    ) -> tuple[BehaviorDecision, ...]:
        """Process one complete frame and return decisions sorted by track ID."""

        self._validate_timestamp(timestamp)
        by_track: dict[int, BehaviorObservation] = {}
        for observation in observations:
            if not isinstance(observation, BehaviorObservation):
                raise TypeError("observations must contain BehaviorObservation values")
            if observation.track_id in by_track:
                raise ValueError(f"duplicate track_id in frame: {observation.track_id}")
            by_track[observation.track_id] = observation

        # Validate the whole frame before mutating state.
        self._last_timestamp = timestamp
        decisions: list[BehaviorDecision] = []
        known_track_ids = set(self._tracks)

        for track_id in sorted(by_track):
            observation = by_track[track_id]
            state = self._tracks.get(track_id)
            if state is None:
                state = _TrackState(last_seen_at=timestamp)
                self._tracks[track_id] = state
            if observation.confidence >= self.config.confidence_threshold:
                decisions.append(self._observe(timestamp, observation, state))
            else:
                decisions.append(self._mark_missing(timestamp, track_id, state))

        for track_id in sorted(known_track_ids - by_track.keys()):
            state = self._tracks[track_id]
            decisions.append(self._mark_missing(timestamp, track_id, state))

        return tuple(sorted(decisions, key=lambda decision: decision.track_id))

    def snapshot(self, track_id: int, timestamp: float | None = None) -> BehaviorDecision | None:
        state = self._tracks.get(track_id)
        if state is None:
            return None
        now = self._last_timestamp if timestamp is None else timestamp
        if now is None:
            now = state.last_seen_at
        if not math.isfinite(now) or now < state.last_seen_at:
            raise ValueError("snapshot timestamp cannot precede the track's last observation")
        return self._decision(track_id, state, now)

    def reset(self) -> None:
        self._tracks.clear()
        self._last_timestamp = None

    def _validate_timestamp(self, timestamp: float) -> None:
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("timestamp must not move backwards")

    def _observe(
        self,
        timestamp: float,
        observation: BehaviorObservation,
        state: _TrackState,
    ) -> BehaviorDecision:
        state.last_seen_at = timestamp
        event = POSE_TO_EVENT.get(observation.pose_class, BehaviorEvent.NONE)
        cleared: tuple[BehaviorEvent, ...] = ()

        if event is BehaviorEvent.NONE:
            if state.confirmed is not BehaviorEvent.NONE:
                cleared = (state.confirmed,)
            self._clear_active(state)
            return self._decision(observation.track_id, state, timestamp, cleared=cleared)

        if event is not state.candidate:
            if state.confirmed is not BehaviorEvent.NONE:
                cleared = (state.confirmed,)
            state.candidate = event
            state.candidate_started_at = timestamp
            state.candidate_min_confidence = observation.confidence
            state.confirmed = BehaviorEvent.NONE
        else:
            state.candidate_min_confidence = min(
                state.candidate_min_confidence, observation.confidence
            )

        started_at = state.candidate_started_at
        assert started_at is not None
        duration = timestamp - started_at
        emitted = False
        if (
            state.confirmed is BehaviorEvent.NONE
            and duration >= self.config.confirmation_seconds(event)
        ):
            state.confirmed = event
            last_emitted = state.last_emitted_at.get(event)
            if (
                last_emitted is None
                or timestamp - last_emitted >= self.config.event_cooldown_seconds
            ):
                state.last_emitted_at[event] = timestamp
                emitted = True

        return self._decision(
            observation.track_id,
            state,
            timestamp,
            emitted=emitted,
            cleared=cleared,
        )

    def _mark_missing(
        self,
        timestamp: float,
        track_id: int,
        state: _TrackState,
    ) -> BehaviorDecision:
        missing_for = timestamp - state.last_seen_at
        if missing_for > self.config.track_ttl_seconds:
            cleared = (
                (state.confirmed,) if state.confirmed is not BehaviorEvent.NONE else ()
            )
            del self._tracks[track_id]
            return BehaviorDecision(
                track_id=track_id,
                event=BehaviorEvent.NONE,
                confidence=0.0,
                event_duration_s=0.0,
                confirmed=False,
                cleared_events=cleared,
                track_expired=True,
            )

        if missing_for > self.config.missing_tolerance_seconds:
            cleared = (
                (state.confirmed,) if state.confirmed is not BehaviorEvent.NONE else ()
            )
            self._clear_active(state)
            return self._decision(track_id, state, timestamp, cleared=cleared)

        return self._decision(track_id, state, timestamp)

    @staticmethod
    def _clear_active(state: _TrackState) -> None:
        state.candidate = BehaviorEvent.NONE
        state.candidate_started_at = None
        state.candidate_min_confidence = 0.0
        state.confirmed = BehaviorEvent.NONE

    @staticmethod
    def _decision(
        track_id: int,
        state: _TrackState,
        timestamp: float,
        *,
        emitted: bool = False,
        cleared: tuple[BehaviorEvent, ...] = (),
    ) -> BehaviorDecision:
        started_at = state.candidate_started_at
        duration = 0.0 if started_at is None else max(0.0, timestamp - started_at)
        event = state.confirmed
        return BehaviorDecision(
            track_id=track_id,
            event=event,
            confidence=(state.candidate_min_confidence if event is not BehaviorEvent.NONE else 0.0),
            event_duration_s=duration,
            confirmed=event is not BehaviorEvent.NONE,
            emitted=emitted,
            cleared_events=cleared,
        )

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class GimbalTrackingPolicy:
    yaw_min_deg: float = -90.0
    yaw_max_deg: float = 90.0
    pitch_min_deg: float = -80.0
    pitch_max_deg: float = 20.0
    max_yaw_rate_deg_s: float = 30.0
    max_pitch_rate_deg_s: float = 20.0
    target_lost_grace_s: float = 1.0
    horizontal_fov_deg: float = 90.0
    vertical_fov_deg: float = 58.7
    proportional_gain: float = 0.70
    center_deadband: float = 0.04
    nominal_update_hz: float = 10.0

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(float(value)) for value in vars(self).values()
        ):
            raise ValueError("all virtual gimbal policy values must be finite")
        if not -180.0 <= self.yaw_min_deg < self.yaw_max_deg <= 180.0:
            raise ValueError("virtual yaw limits must fit inside [-180, 180]")
        if not -90.0 <= self.pitch_min_deg < self.pitch_max_deg <= 90.0:
            raise ValueError("virtual pitch limits must fit inside [-90, 90]")
        if self.max_yaw_rate_deg_s <= 0 or self.max_pitch_rate_deg_s <= 0:
            raise ValueError("virtual gimbal slew rates must be positive")
        if self.target_lost_grace_s < 0:
            raise ValueError("target_lost_grace_s must not be negative")
        if self.horizontal_fov_deg <= 0 or self.vertical_fov_deg <= 0:
            raise ValueError("camera field of view must be positive")
        if not 0 < self.proportional_gain <= 1:
            raise ValueError("proportional_gain must be in (0, 1]")
        if not 0 <= self.center_deadband < 0.5:
            raise ValueError("center_deadband must be in [0, 0.5)")
        if self.nominal_update_hz <= 0:
            raise ValueError("nominal_update_hz must be positive")


@dataclass(frozen=True)
class GimbalCommand:
    yaw_deg: float
    pitch_deg: float
    reason: str


class GimbalTargetTracker:
    """Convert normalized boxes to AirSim NED camera yaw/pitch setpoints."""

    def __init__(self, policy: GimbalTrackingPolicy | None = None) -> None:
        self.policy = policy or GimbalTrackingPolicy()
        self.yaw_deg = 0.0
        self.pitch_deg = 0.0
        self.last_seen_at: float | None = None
        self._last_motion_at: float | None = None

    def observe_bbox(
        self, bbox: Sequence[float], *, now: float
    ) -> GimbalCommand:
        timestamp = _finite("now", now)
        if len(bbox) != 4:
            raise ValueError("bbox must contain four normalized coordinates")
        x1, y1, x2, y2 = (float(value) for value in bbox)
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            raise ValueError("bbox coordinates must be finite")
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError("bbox must be ordered and normalized to [0, 1]")
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        error_x = (
            0.0
            if abs(center_x - 0.5) <= self.policy.center_deadband
            else center_x - 0.5
        )
        error_y = (
            0.0
            if abs(center_y - 0.5) <= self.policy.center_deadband
            else center_y - 0.5
        )
        target_yaw = self.yaw_deg + (
            error_x
            * self.policy.horizontal_fov_deg
            * self.policy.proportional_gain
        )
        target_pitch = self.pitch_deg - (
            error_y
            * self.policy.vertical_fov_deg
            * self.policy.proportional_gain
        )
        self.last_seen_at = timestamp
        return self._slew(
            target_yaw, target_pitch, timestamp, "TRACK_TARGET"
        )

    def begin_return(self, *, now: float) -> None:
        timestamp = _finite("now", now)
        self.last_seen_at = timestamp - self.policy.target_lost_grace_s

    def tick(self, *, now: float) -> GimbalCommand | None:
        timestamp = _finite("now", now)
        if self.last_seen_at is not None:
            age = max(0.0, timestamp - self.last_seen_at)
            if age < self.policy.target_lost_grace_s:
                return None
        if abs(self.yaw_deg) < 0.05 and abs(self.pitch_deg) < 0.05:
            self.yaw_deg = 0.0
            self.pitch_deg = 0.0
            return None
        return self._slew(0.0, 0.0, timestamp, "RETURN_FORWARD")

    def _slew(
        self,
        target_yaw: float,
        target_pitch: float,
        now: float,
        reason: str,
    ) -> GimbalCommand:
        if self._last_motion_at is None or now <= self._last_motion_at:
            dt = 1.0 / self.policy.nominal_update_hz
        else:
            dt = min(now - self._last_motion_at, 0.5)
        self._last_motion_at = now
        yaw_delta = _clamp(
            target_yaw - self.yaw_deg,
            -self.policy.max_yaw_rate_deg_s * dt,
            self.policy.max_yaw_rate_deg_s * dt,
        )
        pitch_delta = _clamp(
            target_pitch - self.pitch_deg,
            -self.policy.max_pitch_rate_deg_s * dt,
            self.policy.max_pitch_rate_deg_s * dt,
        )
        self.yaw_deg = _clamp(
            self.yaw_deg + yaw_delta,
            self.policy.yaw_min_deg,
            self.policy.yaw_max_deg,
        )
        self.pitch_deg = _clamp(
            self.pitch_deg + pitch_delta,
            self.policy.pitch_min_deg,
            self.policy.pitch_max_deg,
        )
        return GimbalCommand(self.yaw_deg, self.pitch_deg, reason)


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))

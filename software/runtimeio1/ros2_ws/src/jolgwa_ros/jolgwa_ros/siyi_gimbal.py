from __future__ import annotations

import math
import socket
import struct
import threading
from dataclasses import dataclass
from typing import Sequence


SIYI_STX = 0x6655
SIYI_CONTROL_PORT = 37260
CMD_GIMBAL_ROTATION = 0x07
CMD_CENTER_GIMBAL = 0x08
CMD_SET_GIMBAL_ATTITUDE = 0x0E

A8_MECHANICAL_YAW_MIN_DEG = -135.0
A8_MECHANICAL_YAW_MAX_DEG = 135.0
A8_MECHANICAL_PITCH_MIN_DEG = -90.0
A8_MECHANICAL_PITCH_MAX_DEG = 25.0


def crc16_ccitt(data: bytes, initial: int = 0) -> int:
    """Return the SIYI SDK CRC16 (CCITT/XMODEM polynomial 0x1021)."""

    crc = initial & 0xFFFF
    for value in data:
        crc ^= value << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def encode_siyi_packet(
    command_id: int,
    payload: bytes = b"",
    *,
    sequence: int = 0,
    need_ack: bool = True,
) -> bytes:
    if not 0 <= command_id <= 0xFF:
        raise ValueError("command_id must fit in one byte")
    if not 0 <= sequence <= 0xFFFF:
        raise ValueError("sequence must fit in two bytes")
    if len(payload) > 0xFFFF:
        raise ValueError("payload is too large")
    control = 0 if need_ack else 1
    frame = struct.pack(
        "<HBHHB",
        SIYI_STX,
        control,
        len(payload),
        sequence,
        command_id,
    ) + payload
    return frame + struct.pack("<H", crc16_ccitt(frame))


class SiyiUdpClient:
    """Minimal, bounded SIYI UDP command sender.

    The client is inert unless ``enabled`` is explicitly true. This mirrors the
    repository's hardware safety gates and lets the complete node be exercised
    without moving a real gimbal.
    """

    def __init__(
        self,
        host: str = "192.168.144.25",
        port: int = SIYI_CONTROL_PORT,
        *,
        enabled: bool = False,
    ) -> None:
        if not str(host).strip():
            raise ValueError("host must not be empty")
        if not 1 <= int(port) <= 65535:
            raise ValueError("port must be between 1 and 65535")
        self.host = str(host)
        self.port = int(port)
        self.enabled = bool(enabled)
        self._sequence = 0
        self._lock = threading.Lock()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if enabled else None

    def set_attitude(self, yaw_deg: float, pitch_deg: float) -> None:
        yaw = _finite_angle("yaw_deg", yaw_deg)
        pitch = _finite_angle("pitch_deg", pitch_deg)
        if not A8_MECHANICAL_YAW_MIN_DEG <= yaw <= A8_MECHANICAL_YAW_MAX_DEG:
            raise ValueError("yaw_deg exceeds the A8 Mini mechanical range")
        if not A8_MECHANICAL_PITCH_MIN_DEG <= pitch <= A8_MECHANICAL_PITCH_MAX_DEG:
            raise ValueError("pitch_deg exceeds the A8 Mini mechanical range")
        payload = struct.pack("<hh", round(yaw * 10.0), round(pitch * 10.0))
        self._send(CMD_SET_GIMBAL_ATTITUDE, payload)

    def center(self) -> None:
        self._send(CMD_CENTER_GIMBAL, b"\x01")

    def stop(self) -> None:
        self._send(CMD_GIMBAL_ROTATION, b"\x00\x00")

    def close(self) -> None:
        with self._lock:
            if self._socket is not None:
                self._socket.close()
                self._socket = None

    def _send(self, command_id: int, payload: bytes) -> None:
        with self._lock:
            packet = encode_siyi_packet(
                command_id,
                payload,
                sequence=self._sequence,
                need_ack=True,
            )
            self._sequence = (self._sequence + 1) & 0xFFFF
            if not self.enabled:
                return
            if self._socket is None:
                raise RuntimeError("SIYI UDP client is closed")
            sent = self._socket.sendto(packet, (self.host, self.port))
            if sent != len(packet):
                raise OSError(f"partial SIYI UDP write: {sent}/{len(packet)} bytes")


@dataclass(frozen=True)
class GimbalTrackingPolicy:
    yaw_min_deg: float = -90.0
    yaw_max_deg: float = 90.0
    pitch_min_deg: float = -80.0
    pitch_max_deg: float = 20.0
    max_yaw_rate_deg_s: float = 30.0
    max_pitch_rate_deg_s: float = 20.0
    target_lost_grace_s: float = 1.0
    horizontal_fov_deg: float = 81.0
    vertical_fov_deg: float = 65.0
    proportional_gain: float = 0.70
    center_deadband: float = 0.04
    nominal_update_hz: float = 10.0

    def __post_init__(self) -> None:
        values = vars(self)
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError("all gimbal policy values must be finite")
        if not (
            A8_MECHANICAL_YAW_MIN_DEG
            <= self.yaw_min_deg
            < self.yaw_max_deg
            <= A8_MECHANICAL_YAW_MAX_DEG
        ):
            raise ValueError("yaw operating limits exceed the A8 Mini range")
        if not (
            A8_MECHANICAL_PITCH_MIN_DEG
            <= self.pitch_min_deg
            < self.pitch_max_deg
            <= A8_MECHANICAL_PITCH_MAX_DEG
        ):
            raise ValueError("pitch operating limits exceed the A8 Mini range")
        if self.max_yaw_rate_deg_s <= 0 or self.max_pitch_rate_deg_s <= 0:
            raise ValueError("gimbal slew rates must be positive")
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
    """Convert normalized target boxes to rate- and angle-limited setpoints."""

    def __init__(self, policy: GimbalTrackingPolicy | None = None) -> None:
        self.policy = policy or GimbalTrackingPolicy()
        self.yaw_deg = 0.0
        self.pitch_deg = 0.0
        self.last_seen_at: float | None = None
        self._last_motion_at: float | None = None

    def observe_bbox(
        self,
        bbox: Sequence[float],
        *,
        now: float,
    ) -> GimbalCommand:
        timestamp = _finite_angle("now", now)
        if len(bbox) != 4:
            raise ValueError("bbox must contain four normalized coordinates")
        x1, y1, x2, y2 = (float(value) for value in bbox)
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            raise ValueError("bbox coordinates must be finite")
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError("bbox must be ordered and normalized to [0, 1]")

        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        error_x = 0.0 if abs(center_x - 0.5) <= self.policy.center_deadband else center_x - 0.5
        error_y = 0.0 if abs(center_y - 0.5) <= self.policy.center_deadband else center_y - 0.5

        # Historical SIYI mounting convention: image-right is negative yaw.
        target_yaw = self.yaw_deg - (
            error_x * self.policy.horizontal_fov_deg * self.policy.proportional_gain
        )
        target_pitch = self.pitch_deg - (
            error_y * self.policy.vertical_fov_deg * self.policy.proportional_gain
        )
        self.last_seen_at = timestamp
        return self._slew(target_yaw, target_pitch, timestamp, "TRACK_TARGET")

    def mark_target_lost(self) -> None:
        """Keep the last seen timestamp so the grace interval can elapse."""

    def begin_return(self, *, now: float) -> None:
        timestamp = _finite_angle("now", now)
        self.last_seen_at = timestamp - self.policy.target_lost_grace_s

    def tick(self, *, now: float) -> GimbalCommand | None:
        timestamp = _finite_angle("now", now)
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


def _finite_angle(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))

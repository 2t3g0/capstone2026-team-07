"""JOR1 display-only status, carried in MAVLink2 TUNNEL (not a command).

Remaining times are SOURCE-REPORTED at encoding. A one-way radio receiver
cannot prove transit delay or that these leases are still valid on receipt.
"""
import math
import struct

MAGIC = b"JOR1"
EXPECTED_SOURCE = (1, 191)
TARGET = (0, 0)
PAYLOAD_TYPE = 32768
MAX_PAYLOAD = 128
HEADER = struct.Struct("<4s16sIdHHBBffffB")
MAX_REASON_BYTES = MAX_PAYLOAD - HEADER.size
ASSESSMENTS = {"UNKNOWN": 0, "CLEAR": 1, "CLIMB_REQUIRED": 2, "BLOCKED": 3}
METRICS = ("front_near_m", "front_median_m", "upper_roi_near_m", "valid_fraction")


def _number(value, name, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid_" + name)
    if maximum is not None and value > maximum:
        raise ValueError("invalid_" + name)
    return float(value)


def _reason(text):
    if not isinstance(text, str):
        raise ValueError("invalid_reason")
    text = "".join(" " if ord(c) < 32 or ord(c) == 127 else c for c in text)
    return text.encode("utf-8")[:MAX_REASON_BYTES].decode("utf-8", errors="ignore").encode("utf-8")


def encode_status(status: dict, *, session: bytes, sequence: int,
                  generated_s: float, now_s: float) -> bytes:
    """Deduct queue/encoding-entry delay from the ORIGINAL snapshot lifetime.

    Caller must also enforce the original deadlines after any later I/O, before
    writing the encoded bytes. This function does not authorize any transport.
    """
    if type(session) is not bytes or len(session) != 16 or not any(session):
        raise ValueError("invalid_session")
    if type(sequence) is not int or not 0 <= sequence <= 0xFFFFFFFF:
        raise ValueError("invalid_sequence")
    generated_s = _number(generated_s, "generated_time")
    now_s = _number(now_s, "encode_time")
    if now_s < generated_s:
        raise ValueError("source_clock_regressed")
    if (not isinstance(status, dict) or status.get("mode") != "OBSERVE_ONLY"
            or status.get("flight_commands_enabled") is not False):
        raise ValueError("not_observe_only")
    report = status.get("report")
    if (not isinstance(report, dict) or report.get("mode") != "OBSERVE_ONLY"
            or report.get("flight_commands_enabled") is not False
            or report.get("assessment") not in ASSESSMENTS):
        raise ValueError("invalid_observer_report")
    for flag in ("camera_ok", "px4_pose_received"):
        if type(status.get(flag)) is not bool:
            raise ValueError("invalid_" + flag)
    elapsed = now_s - generated_s
    judgment_ms = math.floor(max(0., _number(report.get("valid_for_s"), "judgment_lifetime", .5) - elapsed) * 1000)
    sensor_ms = math.floor(max(0., _number(status.get("sensor_valid_for_s"), "sensor_lifetime", .5) - elapsed) * 1000)
    assessment, reason = report["assessment"], _reason(report.get("reason"))
    if not judgment_ms or assessment == "UNKNOWN":
        if assessment != "UNKNOWN":
            reason = b"observation_expired"
        assessment, judgment_ms = "UNKNOWN", 0
    sensor = status.get("sensor")
    if not isinstance(sensor, dict):
        raise ValueError("invalid_sensor")
    metrics = []
    for key in METRICS:
        value = sensor.get(key)
        if value is None:
            metrics.append(-1.)
        else:
            # Validate even expired input; malformed source is not silently repaired.
            value = _number(value, key, 1. if key == "valid_fraction" else 3.4028234e38)
            metrics.append(value if sensor_ms else -1.)
    flags = int(status["camera_ok"] and sensor_ms > 0) | (int(status["px4_pose_received"]) << 1)
    return HEADER.pack(MAGIC, session, sequence, generated_s, judgment_ms, sensor_ms,
                       ASSESSMENTS[assessment], flags, *metrics, len(reason)) + reason


def decode_status(payload: bytes) -> dict:
    """Strict decoder. Never interprets receipt time as source-time proof."""
    if type(payload) is not bytes or not HEADER.size <= len(payload) <= MAX_PAYLOAD:
        raise ValueError("invalid_payload_length")
    magic, session, sequence, generated, judgment_ms, sensor_ms, state, flags, *tail = HEADER.unpack_from(payload)
    metrics, reason_len = tail[:4], tail[4]
    if magic != MAGIC or not any(session):
        raise ValueError("invalid_magic_or_session")
    _number(generated, "generated_time")
    if (judgment_ms > 500 or sensor_ms > 500 or state not in ASSESSMENTS.values()
            or flags & ~3 or len(payload) != HEADER.size + reason_len):
        raise ValueError("invalid_header")
    assessment = next(name for name, value in ASSESSMENTS.items() if value == state)
    if (assessment == "UNKNOWN") != (judgment_ms == 0):
        raise ValueError("inconsistent_judgment_lifetime")
    sensor = {}
    for key, value in zip(METRICS, metrics):
        if value == -1.:
            sensor[key] = None
        else:
            sensor[key] = _number(value, key, 1. if key == "valid_fraction" else None)
    if sensor_ms == 0 and (flags & 1 or any(v is not None for v in sensor.values())):
        raise ValueError("expired_sensor_has_data")
    try:
        reason = payload[HEADER.size:].decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("invalid_reason_utf8") from exc
    if _reason(reason) != payload[HEADER.size:]:
        raise ValueError("invalid_reason_text")
    return dict(session=session, sequence=sequence, generated_s=generated,
                judgment_remaining_ms=judgment_ms, sensor_remaining_ms=sensor_ms,
                assessment=assessment, camera_ok=bool(flags & 1),
                px4_pose_received=bool(flags & 2), sensor=sensor, reason=reason)

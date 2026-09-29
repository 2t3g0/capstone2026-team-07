import json
import math
from dataclasses import dataclass
from typing import Any


SUPPORTED_BROWSER_MESSAGES = {
    "command.text",
    "mission.propose",
    "mission.approve",
    "mission.execute",
    "mission.resume",
    "mission.emergency_land",
    "mission.forward_test_prepare",
    "manual.override",
    "manual.velocity",
    "heartbeat",
}

PROPOSAL_STATUSES = {
    "OK": 0,
    "NEED_CLARIFICATION": 1,
    "UNSUPPORTED": 2,
    "ERROR": 3,
}


class ProtocolError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ManualVelocity:
    sequence: int
    forward_m_s: float
    right_m_s: float
    down_m_s: float
    yaw_rad_s: float
    is_release: bool = False


def parse_browser_message(raw: str) -> dict[str, Any]:
    try:
        message = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ProtocolError(
            "invalid_json", "message must be valid JSON"
        ) from exc
    if not isinstance(message, dict):
        raise ProtocolError(
            "invalid_envelope", "message must be a JSON object"
        )
    message_type = message.get("type")
    if (
        not isinstance(message_type, str)
        or message_type not in SUPPORTED_BROWSER_MESSAGES
    ):
        raise ProtocolError(
            "unsupported_type", "unsupported or missing message type"
        )
    return validate_browser_message(message)


def validate_browser_message(message: dict[str, Any]) -> dict[str, Any]:
    message_type = message["type"]
    if message_type == "command.text":
        return {
            "type": message_type,
            "command": _required_text(message, "command"),
        }
    if message_type == "mission.propose":
        raw_command = _required_text(message, "raw_command")
        plan = message.get("plan")
        if not isinstance(plan, dict):
            raise ProtocolError("invalid_plan", "plan must be a JSON object")
        try:
            json.dumps(plan, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ProtocolError(
                "invalid_plan", "plan must contain finite JSON values"
            ) from exc
        status = plan.get("status")
        if status not in PROPOSAL_STATUSES:
            raise ProtocolError(
                "invalid_plan_status", "plan.status is not supported"
            )
        return {"type": message_type, "raw_command": raw_command, "plan": plan}
    if message_type == "mission.approve":
        approved = message.get("approved")
        if not isinstance(approved, bool):
            raise ProtocolError("invalid_approval", "approved must be boolean")
        return {
            "type": message_type,
            "proposal_id": _required_text(message, "proposal_id"),
            "approved": approved,
            "operator_id": _required_text(message, "operator_id"),
        }
    if message_type == "mission.execute":
        plan_json = _required_text(message, "plan_json")
        try:
            plan = json.loads(plan_json)
        except json.JSONDecodeError as exc:
            raise ProtocolError(
                "invalid_plan_json", "plan_json must be valid JSON"
            ) from exc
        if not isinstance(plan, dict):
            raise ProtocolError(
                "invalid_plan_json", "plan_json must encode an object"
            )
        try:
            json.dumps(plan, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ProtocolError(
                "invalid_plan_json",
                "plan_json must contain finite JSON values",
            ) from exc
        return {
            "type": message_type,
            "proposal_id": _required_text(message, "proposal_id"),
            "mission_id": _required_text(message, "mission_id"),
            "plan_json": plan_json,
            "route_waypoints_enu": _route_waypoints(plan),
        }
    if message_type == "mission.resume":
        return {
            "type": message_type,
            "mission_id": _required_text(message, "mission_id"),
            "operator_id": _required_text(message, "operator_id"),
            "reason": _string(message, "reason"),
        }
    if message_type == "mission.emergency_land":
        return {
            "type": message_type,
            "mission_id": _required_text(message, "mission_id"),
            "operator_id": _required_text(message, "operator_id"),
            "reason": _string(message, "reason"),
        }
    if message_type == "mission.forward_test_prepare":
        target_altitude = _finite_number(message, "target_altitude_home_m")
        if target_altitude not in (1.0, 2.0):
            raise ProtocolError(
                "invalid_low_speed_altitude",
                "target_altitude_home_m must be exactly 1 or 2",
            )
        return {
            "type": message_type,
            "operator_id": _required_text(message, "operator_id"),
            "target_altitude_home_m": target_altitude,
        }
    if message_type == "manual.override":
        active = message.get("active")
        if not isinstance(active, bool):
            raise ProtocolError("invalid_override", "active must be boolean")
        return {
            "type": message_type,
            "active": active,
            "source": _required_text(message, "source"),
            "reason": _string(message, "reason"),
        }
    if message_type == "manual.velocity":
        normalized = validate_manual_input(message)
        return {"type": message_type, **normalized}
    if message_type == "heartbeat":
        timestamp = _finite_number(message, "timestamp")
        return {"type": message_type, "timestamp": timestamp}
    raise ProtocolError("unsupported_type", "unsupported message type")


def _route_waypoints(plan: dict[str, Any]) -> list[tuple[float, float, float]]:
    source = plan.get("route_waypoints_enu")
    if source is None:
        return []
    if not isinstance(source, list) or not 2 <= len(source) <= 300:
        raise ProtocolError(
            "invalid_route", "route_waypoints_enu must contain 2 to 300 points"
        )
    output = []
    for index, point in enumerate(source):
        if not isinstance(point, list) or len(point) not in (3, 4):
            raise ProtocolError(
                "invalid_route", "route waypoint %d must contain x, y, z and optional yaw" % index
            )
        values = []
        for value in point:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ProtocolError("invalid_route", "route coordinates must be numbers")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ProtocolError("invalid_route", "route coordinates must be finite")
            values.append(numeric)
        if not 1.0 <= values[2] <= 120.0:
            raise ProtocolError(
                "invalid_route", "route altitude must be between 1 and 120 m"
            )
        if len(values) == 4 and not -180.0 <= values[3] <= 180.0:
            raise ProtocolError(
                "invalid_route", "route yaw must be between -180 and 180 degrees"
            )
        output.append((values[0], values[1], values[2]))
    return output


def validate_manual_input(message: dict[str, Any]) -> dict[str, Any]:
    sequence = message.get("seq")
    if (
        isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence < 0
    ):
        raise ProtocolError(
            "invalid_sequence", "seq must be a non-negative integer"
        )
    deadman = message.get("deadman")
    if not isinstance(deadman, bool):
        raise ProtocolError("invalid_deadman", "deadman must be boolean")
    values = {
        name: _finite_number(message, name)
        for name in ("forward", "right", "up", "yaw")
    }
    if any(abs(value) > 1.0 for value in values.values()):
        raise ProtocolError(
            "control_out_of_range", "manual axes must be within [-1, 1]"
        )
    return {"seq": sequence, **values, "deadman": deadman}


class Mode2ControlMapper:
    def __init__(
        self,
        max_horizontal_m_s: float = 5.0,
        max_vertical_m_s: float = 2.0,
        max_yaw_rad_s: float = 1.0,
    ) -> None:
        limits = (max_horizontal_m_s, max_vertical_m_s, max_yaw_rad_s)
        if any(not math.isfinite(value) or value <= 0.0 for value in limits):
            raise ValueError(
                "manual control limits must be finite and positive"
            )
        self._max_horizontal = max_horizontal_m_s
        self._max_vertical = max_vertical_m_s
        self._max_yaw = max_yaw_rad_s
        self._last_sequence: int | None = None
        self._deadman_active = False

    def map(self, message: dict[str, Any]) -> ManualVelocity | None:
        values = validate_manual_input(message)
        sequence = values["seq"]

        # A release is always safe and starts a new sequence epoch. This lets a
        # freshly restarted hub recover while the long-running ROS process is
        # still holding the previous hub's larger sequence number.
        if not values["deadman"]:
            self._last_sequence = sequence
            if not self._deadman_active:
                return None
            self._deadman_active = False
            return self._zero(sequence, is_release=True)

        if self._last_sequence is not None and sequence <= self._last_sequence:
            raise ProtocolError(
                "stale_sequence", "manual velocity sequence is stale"
            )
        self._last_sequence = sequence

        self._deadman_active = True
        return ManualVelocity(
            sequence=sequence,
            forward_m_s=values["forward"] * self._max_horizontal,
            right_m_s=values["right"] * self._max_horizontal,
            down_m_s=-values["up"] * self._max_vertical,
            yaw_rad_s=values["yaw"] * self._max_yaw,
        )

    def reset_connection(self) -> ManualVelocity | None:
        release = (
            self._zero(self._last_sequence or 0, is_release=True)
            if self._deadman_active
            else None
        )
        self._last_sequence = None
        self._deadman_active = False
        return release

    @staticmethod
    def _zero(sequence: int, is_release: bool) -> ManualVelocity:
        return ManualVelocity(sequence, 0.0, 0.0, 0.0, 0.0, is_release)


def _required_text(message: dict[str, Any], field: str) -> str:
    value = _string(message, field).strip()
    if not value:
        raise ProtocolError(
            "missing_field", "%s must be a non-empty string" % field
        )
    return value


def _string(message: dict[str, Any], field: str) -> str:
    value = message.get(field)
    if not isinstance(value, str):
        raise ProtocolError("invalid_field", "%s must be a string" % field)
    return value


def _finite_number(message: dict[str, Any], field: str) -> float:
    value = message.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError("invalid_number", "%s must be a number" % field)
    value = float(value)
    if not math.isfinite(value):
        raise ProtocolError("non_finite_number", "%s must be finite" % field)
    return value

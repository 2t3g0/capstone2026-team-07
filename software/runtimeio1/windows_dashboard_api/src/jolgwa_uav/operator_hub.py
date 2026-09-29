from __future__ import annotations

import logging
import asyncio
import json
import math
import time
import uuid
from typing import Annotated, Any, Literal

from fastapi import WebSocket
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
)
from starlette.websockets import WebSocketDisconnect
from .websocket_delivery import SocketWriter

from .operations_observer import OperationsObserver


MAX_MESSAGE_BYTES = 64 * 1024
MAX_NESTING_DEPTH = 16
MAX_COLLECTION_ITEMS = 2_000
MAX_GENERIC_STRING_LENGTH = 48_000
MANUAL_RELAY_PERIOD_S = 0.05
CONTROL_LEASE_TIMEOUT_S = 120.0
MANUAL_CONTROL_LEASE_TIMEOUT_S = 30.0

Identifier = Annotated[StrictStr, Field(min_length=1, max_length=256)]
ShortText = Annotated[StrictStr, Field(max_length=1_024)]
CommandText = Annotated[StrictStr, Field(min_length=1, max_length=4_000)]
PlanJson = Annotated[StrictStr, Field(min_length=2, max_length=48_000)]
Axis = StrictFloat | StrictInt


class ProtocolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CommandTextMessage(ProtocolModel):
    type: Literal["command.text"]
    command: CommandText


class MissionProposeMessage(ProtocolModel):
    type: Literal["mission.propose"]
    raw_command: CommandText
    plan: dict[str, Any]


class MissionApproveMessage(ProtocolModel):
    type: Literal["mission.approve"]
    proposal_id: Identifier
    approved: StrictBool
    operator_id: Identifier


class MissionExecuteMessage(ProtocolModel):
    type: Literal["mission.execute"]
    proposal_id: Identifier
    mission_id: Identifier
    plan_json: PlanJson

    @field_validator("plan_json")
    @classmethod
    def require_json_object(cls, value: str) -> str:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("plan_json must contain valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("plan_json must contain a JSON object")
        return value


class MissionResumeMessage(ProtocolModel):
    type: Literal["mission.resume"]
    mission_id: Identifier
    operator_id: Identifier
    reason: ShortText


class MissionEmergencyLandMessage(ProtocolModel):
    type: Literal["mission.emergency_land"]
    mission_id: Identifier
    operator_id: Identifier
    reason: ShortText


class ForwardTestPrepareMessage(ProtocolModel):
    type: Literal["mission.forward_test_prepare"]
    operator_id: Identifier
    target_altitude_home_m: Literal[1, 2]


class ManualOverrideMessage(ProtocolModel):
    type: Literal["manual.override"]
    active: StrictBool
    source: Identifier
    reason: ShortText


class ControlClaimMessage(ProtocolModel):
    type: Literal["control.claim"]


class ManualVelocityMessage(ProtocolModel):
    type: Literal["manual.velocity"]
    seq: Annotated[StrictInt, Field(ge=0)]
    forward: Axis
    right: Axis
    up: Axis
    yaw: Axis
    deadman: StrictBool

    @field_validator("forward", "right", "up", "yaw")
    @classmethod
    def require_finite_axis(cls, value: float | int) -> float:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("manual axes must be finite")
        if not -1.0 <= numeric <= 1.0:
            raise ValueError("manual axes must be between -1.0 and 1.0")
        return numeric


class HeartbeatMessage(ProtocolModel):
    type: Literal["heartbeat"]
    timestamp: StrictFloat | StrictInt

    @field_validator("timestamp")
    @classmethod
    def require_finite_timestamp(cls, value: float | int) -> float | int:
        if not math.isfinite(value):
            raise ValueError("timestamp must be finite")
        return value


class MissionApprovalResultMessage(ProtocolModel):
    type: Literal["mission.approval_result"]
    accepted: StrictBool
    mission_id: Annotated[StrictStr, Field(max_length=256)]
    message: ShortText
    proposal_id: Identifier


class MissionExecutionResultMessage(ProtocolModel):
    type: Literal["mission.execution_result"]
    success: StrictBool
    final_phase: Identifier | StrictInt
    message: ShortText
    proposal_id: Identifier | None = None
    mission_id: Identifier | None = None


class MissionExecutionHealthMessage(ProtocolModel):
    type: Literal["mission.execution_health"]
    mission_id: Identifier
    stale: StrictBool
    feedback_age_ms: Annotated[StrictInt, Field(ge=0)]
    detail: ShortText


class MissionResumeResultMessage(ProtocolModel):
    type: Literal["mission.resume_result"]
    accepted: StrictBool
    mission_id: Identifier
    message: ShortText


class MissionEmergencyResultMessage(ProtocolModel):
    type: Literal["mission.emergency_result"]
    action: Literal["LAND"]
    accepted: StrictBool
    mission_id: Identifier
    message: ShortText


class GatewayStatusMessage(ProtocolModel):
    type: Literal["gateway.status"]
    ros_connected: StrictBool
    detail: ShortText


class ErrorMessage(ProtocolModel):
    type: Literal["error"]
    code: Identifier
    message: ShortText


OPERATOR_MODELS: dict[str, type[ProtocolModel]] = {
    "command.text": CommandTextMessage,
    "mission.propose": MissionProposeMessage,
    "mission.approve": MissionApproveMessage,
    "mission.execute": MissionExecuteMessage,
    "mission.resume": MissionResumeMessage,
    "mission.emergency_land": MissionEmergencyLandMessage,
    "mission.forward_test_prepare": ForwardTestPrepareMessage,
    "control.claim": ControlClaimMessage,
    "manual.override": ManualOverrideMessage,
    "manual.velocity": ManualVelocityMessage,
    "heartbeat": HeartbeatMessage,
}

CONTROL_MESSAGE_TYPES = frozenset(
    {
        "mission.approve",
        "mission.execute",
        "mission.resume",
        "mission.emergency_land",
        "mission.forward_test_prepare",
        "manual.override",
        "manual.velocity",
    }
)

ROS_MESSAGE_TYPES = frozenset(
    {
        "mission.proposal",
        "mission.status",
        "vehicle.state",
        "mission.approval_result",
        "mission.execution_result",
        "mission.execution_health",
        "mission.execution_started",
        "mission.route_frame_status",
        "gateway.capabilities",
        "mission.resume_result",
        "mission.emergency_result",
        "mission.forward_test_prepare_result",
        "gateway.status",
        "error",
    }
)

ROS_MODELS: dict[str, type[ProtocolModel]] = {
    "mission.approval_result": MissionApprovalResultMessage,
    "mission.execution_result": MissionExecutionResultMessage,
    "mission.execution_health": MissionExecutionHealthMessage,
    "mission.resume_result": MissionResumeResultMessage,
    "mission.emergency_result": MissionEmergencyResultMessage,
    "gateway.status": GatewayStatusMessage,
    "error": ErrorMessage,
}


class ProtocolError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def error_envelope(code: str, message: str) -> dict[str, str]:
    return {"type": "error", "code": code, "message": message}


def decode_message(raw: str) -> dict[str, Any]:
    if len(raw.encode("utf-8")) > MAX_MESSAGE_BYTES:
        raise ProtocolError("payload_too_large", "message exceeds 65536 bytes")
    try:
        value = json.loads(raw, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ProtocolError("invalid_json", "message must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ProtocolError("invalid_envelope", "message must be a JSON object")
    _validate_structure(value)
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON number: {value}")


def validate_operator_message(
    value: dict[str, Any], last_manual_seq: int | None
) -> tuple[dict[str, Any], int | None]:
    message_type = value.get("type")
    if not isinstance(message_type, str):
        raise ProtocolError("invalid_type", "type must be a string")
    model_type = OPERATOR_MODELS.get(message_type)
    if model_type is None:
        raise ProtocolError("unsupported_type", f"unsupported operator type: {message_type}")
    try:
        message = model_type.model_validate(value)
    except ValidationError as exc:
        raise ProtocolError("validation_error", _validation_message(exc)) from exc

    output = message.model_dump(mode="json")
    if isinstance(message, ManualVelocityMessage):
        if last_manual_seq is not None and message.seq <= last_manual_seq:
            raise ProtocolError(
                "stale_sequence", "manual.velocity seq must increase monotonically"
            )
        if not message.deadman:
            output.update(forward=0.0, right=0.0, up=0.0, yaw=0.0)
        return output, message.seq
    return output, last_manual_seq


def validate_ros_message(value: dict[str, Any]) -> dict[str, Any]:
    message_type = value.get("type")
    if not isinstance(message_type, str):
        raise ProtocolError("invalid_type", "type must be a string")
    if message_type not in ROS_MESSAGE_TYPES:
        raise ProtocolError("unsupported_type", f"unsupported ROS type: {message_type}")
    model_type = ROS_MODELS.get(message_type)
    if model_type is not None:
        try:
            return model_type.model_validate(value).model_dump(
                mode="json", exclude_none=True
            )
        except ValidationError as exc:
            raise ProtocolError("validation_error", _validation_message(exc)) from exc
    return value


def _validate_structure(value: Any, depth: int = 0) -> None:
    if depth > MAX_NESTING_DEPTH:
        raise ProtocolError("invalid_payload", "message nesting is too deep")
    if isinstance(value, str):
        if len(value) > MAX_GENERIC_STRING_LENGTH:
            raise ProtocolError("invalid_payload", "message contains an oversized string")
        return
    if isinstance(value, dict):
        if len(value) > MAX_COLLECTION_ITEMS:
            raise ProtocolError("invalid_payload", "message object has too many fields")
        for key, child in value.items():
            if not isinstance(key, str) or len(key) > 256:
                raise ProtocolError("invalid_payload", "message contains an invalid field name")
            _validate_structure(child, depth + 1)
        return
    if isinstance(value, list):
        if len(value) > MAX_COLLECTION_ITEMS:
            raise ProtocolError("invalid_payload", "message array has too many items")
        for child in value:
            _validate_structure(child, depth + 1)


def _validation_message(exc: ValidationError) -> str:
    error = exc.errors(include_url=False)[0]
    location = ".".join(str(part) for part in error["loc"])
    return f"{location}: {error['msg']}" if location else str(error["msg"])


class OperatorHub:
    """Per-application WebSocket relay between operators and one ROS gateway."""

    def __init__(self, operations: OperationsObserver | None = None) -> None:
        self._operations = operations
        self._writers = {}
        self._operators: set[WebSocket] = set()
        self._operator_session_ids: dict[WebSocket, str] = {}
        self._send_locks: dict[WebSocket, asyncio.Lock] = {}
        self._ros: WebSocket | None = None
        self._latest_gateway_capabilities: dict[str, Any] | None = None
        self._control_owner: WebSocket | None = None
        self._control_generation = 0
        self._control_last_seen_at = 0.0
        self._latest_manual: dict[str, Any] | None = None
        self._manual_sequence = 0
        self._manual_received_count = 0
        self._manual_relayed_count = 0
        self._last_manual_received_at = 0.0
        self._last_manual_relayed_at = 0.0
        self._lock = asyncio.Lock()
        self._ros_send_lock = asyncio.Lock()
        self._manual_relay_task: asyncio.Task[None] | None = None

    async def operator_session(self, websocket: WebSocket) -> None:
        await websocket.accept()
        session_id = str(uuid.uuid4())
        async with self._lock:
            self._operators.add(websocket)
            self._operator_session_ids[websocket] = session_id
            self._send_locks[websocket] = asyncio.Lock()
            ros_connected = self._ros is not None
            latest_gateway_capabilities = (
                dict(self._latest_gateway_capabilities)
                if self._latest_gateway_capabilities is not None
                else None
            )
        if self._operations is not None:
            self._operations.record(
                "operator.connected",
                source="operator_hub",
                session_id=session_id,
                ros_connected=ros_connected,
            )
        await self._send_json(
            websocket,
            self._gateway_status(
                ros_connected,
                "ROS gateway connected"
                if ros_connected
                else "ROS gateway disconnected",
            ),
        )
        if latest_gateway_capabilities is not None:
            await self._send_json(websocket, latest_gateway_capabilities)

        last_manual_seq: int | None = None
        try:
            while True:
                raw = await self._receive_text(websocket)
                if raw is None:
                    continue
                try:
                    value = decode_message(raw)
                    message, last_manual_seq = validate_operator_message(
                        value, last_manual_seq
                    )
                    if self._operations is not None:
                        self._operations.observe_operator(
                            message, session_id=session_id
                        )
                    if message["type"] == "control.claim":
                        await self._claim_control(websocket)
                        continue
                    if (
                        message["type"] in CONTROL_MESSAGE_TYPES
                        and not await self._owns_control(websocket)
                    ):
                        await self._send_json(
                            websocket,
                            error_envelope(
                                "control_not_owned",
                                "이 세션이 제어권을 보유하고 있지 않습니다",
                            ),
                        )
                        continue
                    if message["type"] == "manual.velocity":
                        if not await self._accept_manual_velocity(websocket, message):
                            await self._send_json(
                                websocket,
                                error_envelope(
                                    "ros_unavailable", "ROS gateway is not connected"
                                ),
                            )
                        continue
                    if message["type"] == "heartbeat":
                        await self._touch_control_lease(websocket)
                    if not await self._send_to_ros(message):
                        await self._send_json(
                            websocket,
                            error_envelope(
                                "ros_unavailable", "ROS gateway is not connected"
                            ),
                        )
                except ProtocolError as exc:
                    await self._send_json(
                        websocket, error_envelope(exc.code, exc.message)
                    )
        except WebSocketDisconnect as exc:
            logging.getLogger(__name__).info("websocket_receive_closed generation=%s code=%s type=%s", id(websocket), exc.code, type(exc).__name__)
        finally:
            await self._close_writer(websocket)
            was_owner = False
            async with self._lock:
                self._operators.discard(websocket)
                self._operator_session_ids.pop(websocket, None)
                if self._control_owner is websocket:
                    self._control_owner = None
                    self._control_generation += 1
                    self._latest_manual = None
                    was_owner = True
            if was_owner:
                if self._operations is not None:
                    self._operations.operator_control_changed(
                        False,
                        reason="owner_disconnected",
                        previous_session_id=session_id,
                    )
                await self._send_manual_stop()
                asyncio.create_task(
                    self._broadcast_control_status(),
                    name="control-status-after-disconnect",
                )
            if self._operations is not None:
                self._operations.record(
                    "operator.disconnected",
                    source="operator_hub",
                    session_id=session_id,
                    was_owner=was_owner,
                )

    async def ros_session(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            if self._ros is not None:
                accepted = False
            else:
                self._ros = websocket
                self._send_locks[websocket] = asyncio.Lock()
                accepted = True
        if not accepted:
            await self._send_json(
                websocket,
                error_envelope(
                    "ros_gateway_already_connected",
                    "another ROS gateway is already connected",
                ),
            )
            await websocket.close(code=1008)
            return

        if self._operations is not None:
            self._operations.gateway_changed(True)
        await self.broadcast(self._gateway_status(True, "ROS gateway connected"))
        try:
            while True:
                raw = await self._receive_text(websocket)
                if raw is None:
                    continue
                try:
                    value = validate_ros_message(decode_message(raw))
                    if value["type"] == "gateway.capabilities":
                        async with self._lock:
                            if self._ros is websocket:
                                self._latest_gateway_capabilities = dict(value)
                    if self._operations is not None:
                        self._operations.observe_ros(value)
                    await self.broadcast(value)
                except ProtocolError as exc:
                    await self._send_json(
                        websocket, error_envelope(exc.code, exc.message)
                    )
        except WebSocketDisconnect as exc:
            logging.getLogger(__name__).info("websocket_receive_closed generation=%s code=%s type=%s", id(websocket), exc.code, type(exc).__name__)
        finally:
            await self._close_writer(websocket)
            async with self._lock:
                if self._ros is websocket:
                    self._ros = None
                    self._latest_gateway_capabilities = None
                    self._latest_manual = None
                    disconnected = True
                else:
                    disconnected = False
            if disconnected:
                if self._operations is not None:
                    self._operations.gateway_changed(False)
                await self.broadcast(
                    self._gateway_status(False, "ROS gateway disconnected")
                )

    async def broadcast(self, message: dict[str, Any]) -> None:
        async with self._lock:
            operators = tuple(self._operators)
        for operator in operators:
            writer = self._writer(operator)
            if message.get("type") == "gateway.status" and not message.get("ros_connected"):
                writer.discard_telemetry()
            writer.enqueue(message)

    def _writer(self, websocket):
        writer = self._writers.get(websocket)
        if writer is None:
            writer = SocketWriter(websocket.send_json, lambda: self._delivery_failed(websocket))
            self._writers[websocket] = writer
        return writer

    async def _close_writer(self, websocket):
        writer = self._writers.pop(websocket, None)
        if writer is not None:
            await writer.close()
        if hasattr(self, "_send_locks"):
            self._send_locks.pop(websocket, None)

    async def _delivery_failed(self, websocket):
        was_owner = False
        disconnected = False
        async with self._lock:
            self._operators.discard(websocket)
            if self._control_owner is websocket:
                self._control_owner = None
                self._control_generation += 1
                self._latest_manual = None
                was_owner = True
            if self._ros is websocket:
                self._ros = None
                self._latest_gateway_capabilities = None
                self._latest_manual = None
                disconnected = True
        try:
            await asyncio.wait_for(websocket.close(code=1013), 0.250)
        except Exception as exc:
            logging.getLogger(__name__).warning("websocket_close_failed generation=%s type=%s detail=%r", id(websocket), type(exc).__name__, exc)
        if was_owner:
            await self._send_manual_stop()
            await self._broadcast_control_status()
        if disconnected:
            operations = getattr(self, "_operations", None)
            if operations is not None:
                operations.gateway_changed(False)
            await self.broadcast(self._gateway_status(False, "ROS gateway send failed"))

    async def _send_to_ros(self, message: dict[str, Any], *, control_generation: int | None = None) -> bool:
        try:
            return await asyncio.wait_for(self._send_to_ros_serialized(
                message, control_generation=control_generation), 0.250)
        except asyncio.TimeoutError:
            async with self._lock:
                ros = self._ros
            if ros is not None:
                await self._delivery_failed(ros)
                await self._close_writer(ros)
            return False

    async def _send_to_ros_serialized(
        self,
        message: dict[str, Any],
        *,
        control_generation: int | None = None,
    ) -> bool:
        disconnected = False
        async with self._ros_send_lock:
            async with self._lock:
                if (
                    control_generation is not None
                    and control_generation != self._control_generation
                ):
                    return True
                ros = self._ros
            if ros is None:
                return False
            if await self._send_json(ros, message):
                return True
            async with self._lock:
                if self._ros is ros:
                    self._ros = None
                    self._latest_manual = None
                    disconnected = True
        if disconnected:
            await self.broadcast(
                self._gateway_status(False, "ROS gateway disconnected")
            )
        return False

    async def _claim_control(self, websocket: WebSocket) -> None:
        async with self._lock:
            previous_owner = self._control_owner
            previous_session_id = self._operator_session_ids.get(previous_owner)
            session_id = self._operator_session_ids.get(websocket)
            self._control_owner = websocket
            self._control_generation += 1
            self._control_last_seen_at = time.monotonic()
            self._latest_manual = None
        if self._operations is not None:
            self._operations.operator_control_changed(
                True,
                reason=(
                    "control_transferred"
                    if previous_owner is not None and previous_owner is not websocket
                    else "control_claimed"
                ),
                session_id=session_id,
                previous_session_id=previous_session_id,
            )
        await self._send_manual_stop()
        await self._broadcast_control_status()
        await self._ensure_manual_relay_task()

    async def _owns_control(self, websocket: WebSocket) -> bool:
        async with self._lock:
            return self._control_owner is websocket

    async def _send_manual_stop(self) -> None:
        message = await self._serialize_manual_sequence(
            {
                "type": "manual.velocity",
                "seq": 0,
                "forward": 0.0,
                "right": 0.0,
                "up": 0.0,
                "yaw": 0.0,
                "deadman": False,
            }
        )
        await self._send_to_ros(message)

    async def _accept_manual_velocity(
        self, websocket: WebSocket, message: dict[str, Any]
    ) -> bool:
        async with self._lock:
            if self._control_owner is not websocket:
                return True
            now = time.monotonic()
            self._control_last_seen_at = now
            self._last_manual_received_at = now
            self._manual_received_count += 1
            self._latest_manual = dict(message) if message["deadman"] else None
            generation = self._control_generation
        serialized = await self._serialize_manual_sequence(message)
        return await self._send_to_ros(
            serialized, control_generation=generation
        )

    async def _touch_control_lease(self, websocket: WebSocket) -> None:
        async with self._lock:
            if self._control_owner is websocket:
                self._control_last_seen_at = time.monotonic()

    async def _ensure_manual_relay_task(self) -> None:
        async with self._lock:
            if self._manual_relay_task is None or self._manual_relay_task.done():
                self._manual_relay_task = asyncio.create_task(
                    self._manual_relay_loop(), name="manual-velocity-relay"
                )

    async def _manual_relay_loop(self) -> None:
        while True:
            await asyncio.sleep(MANUAL_RELAY_PERIOD_S)
            expired = False
            async with self._lock:
                owner = self._control_owner
                if owner is None:
                    continue
                if (
                    time.monotonic() - self._control_last_seen_at
                    > (MANUAL_CONTROL_LEASE_TIMEOUT_S if self._latest_manual is not None
                       else CONTROL_LEASE_TIMEOUT_S)
                ):
                    self._control_owner = None
                    self._control_generation += 1
                    self._latest_manual = None
                    expired = True
                    message = None
                    generation = self._control_generation
                else:
                    message = (
                        dict(self._latest_manual)
                        if self._latest_manual is not None
                        else None
                    )
                    generation = self._control_generation
            if expired:
                if self._operations is not None:
                    self._operations.operator_control_changed(
                        False,
                        reason="control_lease_expired",
                        previous_session_id=self._operator_session_ids.get(owner),
                    )
                await self._send_manual_stop()
                await self._broadcast_control_status()
                continue
            if message is not None:
                serialized = await self._serialize_manual_sequence(message)
                if await self._send_to_ros(
                    serialized, control_generation=generation
                ):
                    async with self._lock:
                        self._manual_relayed_count += 1
                        self._last_manual_relayed_at = time.monotonic()

    async def control_snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        async with self._lock:
            return {
                "owner_connected": self._control_owner is not None,
                "ros_connected": self._ros is not None,
                "latest_manual": (
                    dict(self._latest_manual)
                    if self._latest_manual is not None
                    else None
                ),
                "control_age_s": (
                    round(now - self._control_last_seen_at, 3)
                    if self._control_last_seen_at
                    else None
                ),
                "manual_received_count": self._manual_received_count,
                "manual_relayed_count": self._manual_relayed_count,
                "last_manual_received_age_s": (
                    round(now - self._last_manual_received_at, 3)
                    if self._last_manual_received_at
                    else None
                ),
                "last_manual_relayed_age_s": (
                    round(now - self._last_manual_relayed_at, 3)
                    if self._last_manual_relayed_at
                    else None
                ),
                "relay_task_running": bool(
                    self._manual_relay_task is not None
                    and not self._manual_relay_task.done()
                ),
            }

    async def _broadcast_control_status(self) -> None:
        async with self._lock:
            operators = tuple(self._operators)
            owner = self._control_owner
        failed: list[WebSocket] = []
        for operator in operators:
            if not await self._send_json(
                operator,
                {
                    "type": "control.status",
                    "owned": operator is owner,
                    "detail": (
                        "이 세션이 제어권을 보유하고 있습니다"
                        if operator is owner
                        else (
                            "제어권 보유 세션이 없습니다"
                            if owner is None
                            else "다른 세션이 제어권을 보유하고 있습니다"
                        )
                    ),
                },
            ):
                failed.append(operator)
        if failed:
            async with self._lock:
                self._operators.difference_update(failed)

    async def _serialize_manual_sequence(
        self, message: dict[str, Any]
    ) -> dict[str, Any]:
        async with self._lock:
            self._manual_sequence = max(
                self._manual_sequence + 1, int(message["seq"])
            )
            sequence = self._manual_sequence
        return {**message, "seq": sequence}

    @staticmethod
    async def _receive_text(websocket: WebSocket) -> str | None:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(message.get("code", 1000))
        text = message.get("text")
        if text is None:
            await websocket.send_json(
                error_envelope("invalid_frame", "binary WebSocket frames are not supported")
            )
            return None
        return text

    async def _send_json(self, websocket: WebSocket, message: dict[str, Any]) -> bool:
        return await self._writer(websocket).send(message)

    @staticmethod
    def _gateway_status(connected: bool, detail: str) -> dict[str, Any]:
        return {
            "type": "gateway.status",
            "ros_connected": connected,
            "detail": detail,
        }

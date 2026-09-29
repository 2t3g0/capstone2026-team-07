import asyncio
import json
import math
import queue
import threading
import time
import uuid
from typing import Any
from jolgwa_uav.websocket_delivery import with_residence_age, SEND_TIMEOUT_S

import rclpy
from geometry_msgs.msg import TwistStamped
from jolgwa_interfaces.action import ExecuteMission
from jolgwa_interfaces.msg import (
    LowSpeedSafetyState,
    MissionProposal,
    MissionStatus,
    RouteFrameStatus,
    VehicleControlState,
    VehicleGeoState,
)
from jolgwa_interfaces.srv import (
    ApproveMission, EmergencyLand, PrepareForwardTest, ResumeMission,
    SetManualOverride,
)
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String

from .approval_execution import ApprovalContractError, ApprovalExecutionLedger
from .integrated_event_policy import POLICY_SUMMARY_KO, validate_integrated_event_plan
from .low_speed import is_low_speed_profile, profile_for_plan, validate_low_speed_plan
from .operator_protocol import (
    Mode2ControlMapper,
    PROPOSAL_STATUSES,
    ProtocolError,
)
from .operator_protocol import parse_browser_message
from .topic_names import (
    APPROVE_MISSION_SERVICE,
    EMERGENCY_LAND_SERVICE,
    EXECUTE_MISSION_ACTION,
    MANUAL_OVERRIDE_SERVICE,
    MANUAL_VELOCITY,
    MISSION_PROPOSAL,
    MISSION_STATUS,
    ROUTE_FRAME_STATUS,
    RESUME_MISSION_SERVICE,
    TEXT_INPUT,
    VEHICLE_CONTROL_STATE,
    VEHICLE_GEO_STATE,
    LOW_SPEED_SAFETY_STATE,
    PREPARE_FORWARD_TEST_SERVICE,
)

BUILD_ID = "low-speed-runtimeio1-20260929"
PROTOCOL_VERSION = 13
ROUTE_FRAME_VERSION = 1


class RosWebSocketClient:
    def __init__(
        self,
        url: str,
        incoming: queue.Queue,
        reconnect_initial_s: float,
        reconnect_max_s: float,
        fatal_callback=None,
    ) -> None:
        self._url = url
        self._incoming = incoming
        self._critical_outgoing: queue.Queue[tuple[int, dict[str, Any], float]] = queue.Queue(maxsize=256)
        self._latest_telemetry: dict[str, tuple[int, dict[str, Any], float]] = {}
        self._outgoing_lock = threading.Lock()
        self._unhealthy = False
        self._generation = 0
        self._fatal_callback = fatal_callback
        self._reconnect_initial_s = reconnect_initial_s
        self._reconnect_max_s = reconnect_max_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._thread_main, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def send(self, message: dict[str, Any]) -> None:
        if self._unhealthy:
            return
        safe = _json_safe(message)
        message_type = str(safe.get("type", ""))
        if message_type in {"vehicle.state", "mission.status", "mission.route_frame_status"}:
            key = message_type
            if message_type == "mission.route_frame_status":
                key += ":" + str(safe.get("proposal_id", ""))
            with self._outgoing_lock:
                self._latest_telemetry[key] = (self._generation, safe, time.monotonic())
                while len(self._latest_telemetry) > 64:
                    self._latest_telemetry.pop(next(iter(self._latest_telemetry)))
            return
        try:
            self._critical_outgoing.put_nowait((self._generation, safe, time.monotonic()))
        except queue.Full:
            self._unhealthy = True
            self._put_incoming({
                "type": "_gateway.error",
                "detail": "critical WebSocket output queue overflow; gateway restart required",
            })
            self._stop.set()
            if self._fatal_callback is not None:
                self._fatal_callback(
                    "critical WebSocket output queue overflow; gateway restart required")

    def _put_incoming(self, message: dict[str, Any]) -> bool:
        try:
            self._incoming.put_nowait(message)
            return True
        except queue.Full:
            self._unhealthy = True
            self._stop.set()
            return False

    def clear_telemetry(self) -> None:
        with self._outgoing_lock:
            self._latest_telemetry.clear()

    def begin_session(self) -> int:
        """Start a clean output generation before exposing a new socket."""
        with self._outgoing_lock:
            self._generation += 1
            generation = self._generation
            self._latest_telemetry.clear()
            while True:
                try:
                    self._critical_outgoing.get_nowait()
                except queue.Empty:
                    break
        return generation

    def invalidate_session(self) -> None:
        self.begin_session()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def _thread_main(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        try:
            import websockets
        except ImportError:
            self._put_incoming(
                {
                    "type": "_gateway.error",
                    "detail": "Python websockets package is unavailable",
                }
            )
            return

        delay = self._reconnect_initial_s
        while not self._stop.is_set():
            try:
                self._io_phase = 'connect'
                async with websockets.connect(
                    self._url,
                    open_timeout=5.0,
                    ping_interval=10.0,
                    ping_timeout=10.0,
                    max_size=1024 * 1024,
                ) as websocket:
                    generation = self.begin_session()
                    self._put_incoming({
                        "type": "_gateway.connected", "generation": generation})
                    delay = self._reconnect_initial_s
                    await self._run_session(websocket, generation)
            except Exception as exc:
                if not self._stop.is_set():
                    diagnostic = dict(error_type=type(exc).__name__, detail=str(exc),
                        close_code=getattr(exc, "code", None), generation=self._generation,
                        phase=getattr(self, '_io_phase', 'connect'),
                        queue_critical=self._critical_outgoing.qsize(), queue_telemetry=len(self._latest_telemetry),
                        queue_age_s=getattr(self, '_io_age', None),
                        send_s=(time.monotonic()-self._io_started if getattr(self, '_io_phase', '') == 'send'
                                else getattr(self, '_last_send_s', None)))
                    self.invalidate_session()
                    self._put_incoming({"type": "_gateway.disconnected", "detail": json.dumps(diagnostic)})
                    await self._wait_for_stop(delay)
                    delay = min(delay * 2.0, self._reconnect_max_s)

    async def _run_session(self, websocket, generation: int) -> None:
        receive_task = asyncio.create_task(websocket.recv())
        try:
            while not self._stop.is_set():
                if receive_task.done():
                    self._io_phase = 'receive'
                    raw = receive_task.result()
                    if not isinstance(raw, str):
                        self.send_error(
                            "binary_message",
                            "binary WebSocket messages are unsupported",
                        )
                    else:
                        try:
                            self._put_incoming(parse_browser_message(raw))
                        except ProtocolError as exc:
                            self.send_error(exc.code, str(exc))
                    receive_task = asyncio.create_task(websocket.recv())

                for _ in range(64):
                    try:
                        item_generation, message, queued_at = self._critical_outgoing.get_nowait()
                    except queue.Empty:
                        break
                    if item_generation != generation:
                        continue
                    self._io_phase = 'send'
                    self._io_started = time.monotonic()
                    self._io_age = self._io_started-queued_at
                    await asyncio.wait_for(websocket.send(json.dumps(
                        with_residence_age(message, queued_at), ensure_ascii=False,
                        separators=(",", ":"), allow_nan=False)), SEND_TIMEOUT_S)
                    self._last_send_s = time.monotonic()-self._io_started
                    self._io_phase = "receive"
                with self._outgoing_lock:
                    telemetry = tuple(self._latest_telemetry.values())
                    self._latest_telemetry.clear()
                for item_generation, message, queued_at in telemetry:
                    if item_generation != generation:
                        continue
                    self._io_phase = 'send'
                    self._io_started = time.monotonic()
                    self._io_age = self._io_started-queued_at
                    await asyncio.wait_for(websocket.send(json.dumps(
                        with_residence_age(message, queued_at), ensure_ascii=False,
                        separators=(",", ":"), allow_nan=False)), SEND_TIMEOUT_S)
                    self._last_send_s = time.monotonic()-self._io_started
                    self._io_phase = "receive"
                await asyncio.sleep(0.02)
        finally:
            receive_task.cancel()
            await asyncio.gather(receive_task, return_exceptions=True)

    async def _wait_for_stop(self, delay: float) -> None:
        remaining = delay
        while remaining > 0.0 and not self._stop.is_set():
            interval = min(remaining, 0.1)
            await asyncio.sleep(interval)
            remaining -= interval

    def send_error(self, code: str, message: str) -> None:
        self.send({"type": "error", "code": code, "message": message})


class OperatorGatewayNode(Node):
    def __init__(self) -> None:
        super().__init__("operator_gateway")
        self.declare_parameter("test_scenario_enabled", False)
        self._scenario_enabled = bool(self.get_parameter("test_scenario_enabled").value)
        self.declare_parameter("gateway_url", "ws://127.0.0.1:9293/ws/ros")
        self.declare_parameter("reconnect_initial_s", 0.5)
        self.declare_parameter("reconnect_max_s", 10.0)
        self.declare_parameter("max_horizontal_m_s", 5.0)
        self.declare_parameter("max_vertical_m_s", 2.0)
        self.declare_parameter("max_yaw_rad_s", 1.0)
        self.declare_parameter("enforce_integrated_event_policy", False)
        self._enforce_integrated_event_policy = bool(
            self.get_parameter("enforce_integrated_event_policy").value)

        callback_group = ReentrantCallbackGroup()
        self._incoming: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1024)
        self._manual_mapper = Mode2ControlMapper(
            float(self.get_parameter("max_horizontal_m_s").value),
            float(self.get_parameter("max_vertical_m_s").value),
            float(self.get_parameter("max_yaw_rad_s").value),
        )
        self._last_vehicle_state: VehicleControlState | None = None
        self._last_vehicle_state_at = 0.0
        self._last_vehicle_geo: VehicleGeoState | None = None
        self._last_vehicle_geo_at = 0.0
        self._last_low_speed_safety: LowSpeedSafetyState | None = None
        self._last_low_speed_safety_at = 0.0
        self._approval_ledger = ApprovalExecutionLedger(timeout_s=5.0)
        self._route_frame_status = {}
        self._pending_goal_responses = {}
        self._service_request_lock = threading.Lock()
        self._pending_service_requests = {}
        self._service_generation = 0
        self._active_execution_feedback = {}
        self._last_mission_payload = None
        self._last_execution_result = None
        self._last_execution_health = None
        self._fatal_gateway_error = ""

        self._text_publisher = self.create_publisher(String, TEXT_INPUT, 10)
        self._proposal_publisher = self.create_publisher(
            MissionProposal, MISSION_PROPOSAL, 10
        )
        self._manual_velocity_publisher = self.create_publisher(
            TwistStamped, MANUAL_VELOCITY, 10
        )
        self.create_subscription(
            MissionProposal,
            MISSION_PROPOSAL,
            self._on_proposal,
            10,
            callback_group=callback_group,
        )
        self.create_subscription(
            RouteFrameStatus, ROUTE_FRAME_STATUS, self._on_route_frame_status, 10,
            callback_group=callback_group)
        self.create_subscription(
            MissionStatus,
            MISSION_STATUS,
            self._on_mission_status,
            10,
            callback_group=callback_group,
        )
        self.create_subscription(
            VehicleControlState,
            VEHICLE_CONTROL_STATE,
            self._on_vehicle_state,
            qos_profile_sensor_data,
            callback_group=callback_group,
        )
        self.create_subscription(
            VehicleGeoState, VEHICLE_GEO_STATE, self._on_vehicle_geo,
            qos_profile_sensor_data, callback_group=callback_group)
        self.create_subscription(
            LowSpeedSafetyState, LOW_SPEED_SAFETY_STATE, self._on_low_speed_safety,
            qos_profile_sensor_data, callback_group=callback_group)
        self._approval_client = self.create_client(
            ApproveMission,
            APPROVE_MISSION_SERVICE,
            callback_group=callback_group,
        )
        self._manual_override_client = self.create_client(
            SetManualOverride,
            MANUAL_OVERRIDE_SERVICE,
            callback_group=callback_group,
        )
        self._resume_client = self.create_client(
            ResumeMission,
            RESUME_MISSION_SERVICE,
            callback_group=callback_group,
        )
        self._forward_test_client = self.create_client(
            PrepareForwardTest, PREPARE_FORWARD_TEST_SERVICE,
            callback_group=callback_group)
        self._emergency_land_client = self.create_client(
            EmergencyLand, EMERGENCY_LAND_SERVICE,
            callback_group=callback_group)
        self._execute_client = ActionClient(
            self,
            ExecuteMission,
            EXECUTE_MISSION_ACTION,
            callback_group=callback_group,
        )

        self._websocket = RosWebSocketClient(
            str(self.get_parameter("gateway_url").value),
            self._incoming,
            float(self.get_parameter("reconnect_initial_s").value),
            float(self.get_parameter("reconnect_max_s").value),
            fatal_callback=self._on_gateway_fatal,
        )
        # Operator messages remain serviceable if Gazebo pauses or loses /clock.
        self._service_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(
            0.02, self._drain_incoming, callback_group=callback_group,
            clock=self._service_clock
        )
        self.create_timer(0.2, self._expire_goal_responses,
                          callback_group=callback_group, clock=self._service_clock)
        self.create_timer(0.2, self._expire_service_requests,
                          callback_group=callback_group, clock=self._service_clock)
        self._websocket.start()

    def _drain_incoming(self) -> None:
        if self._fatal_gateway_error:
            raise RuntimeError(self._fatal_gateway_error)
        for _ in range(64):
            try:
                message = self._incoming.get_nowait()
            except queue.Empty:
                return
            try:
                self._dispatch(message)
            except ProtocolError as exc:
                self._send_error(exc.code, str(exc))
            except Exception as exc:
                self.get_logger().error(
                    "operator gateway dispatch failed: %s" % exc
                )
                self._send_error("ros_dispatch_failed", str(exc))

    def _dispatch(self, message: dict[str, Any]) -> None:
        message_type = message["type"]
        if message_type == "_gateway.connected":
            self._invalidate_service_requests()
            self._approval_ledger.invalidate(connected=True)
            self._release_manual_for_connection_change()
            self._send_gateway_status(True, "ROS gateway connected")
            self._send({
                "type": "gateway.capabilities", "build_id": BUILD_ID,
                "protocol_version": PROTOCOL_VERSION, "auto_execute": True,
                "route_frame_version": ROUTE_FRAME_VERSION,
                "live_geo_position": True,
                "low_speed_profile_version": 2,
                "flight_output_handshake_version": 8,
                "altitude_reference_version": 6,
                "home_correction_version": 3,
                "status_freshness_version": 3,
                "emergency_control_version": 2,
                "low_speed_target_altitudes_m": [1.0, 2.0],
                "forward_test": True,
                "forward_test_camera_switch": True,
                "test_scenario_v1": bool(self.get_parameter("test_scenario_enabled").value),
            })
            if self._last_vehicle_state is not None:
                self._on_vehicle_state(self._last_vehicle_state, replay=True)
            for payload in (
                    self._last_mission_payload,
                    self._last_execution_result,
                    self._last_execution_health):
                if payload is not None:
                    self._send(payload)
        elif message_type == "_gateway.disconnected":
            self._websocket.clear_telemetry()
            self._invalidate_service_requests()
            self._approval_ledger.invalidate(connected=False)
            self._release_manual_for_connection_change()
            self.get_logger().warning(
                "operator WebSocket disconnected: %s" % message["detail"]
            )
        elif message_type == "_gateway.error":
            self.get_logger().error(message["detail"])
            self._fatal_gateway_error = message["detail"]
        elif message_type == "command.text":
            output = String()
            output.data = message["command"]
            self._text_publisher.publish(output)
        elif message_type == "mission.propose":
            self._publish_inbound_proposal(message)
        elif message_type == "mission.approve":
            self._request_approval(message)
        elif message_type == "mission.execute":
            self._execute_mission(message)
        elif message_type == "mission.resume":
            self._resume_mission(message)
        elif message_type == "mission.forward_test_prepare":
            self._prepare_forward_test(message)
        elif message_type == "mission.emergency_land":
            self._request_emergency_land(message)
        elif message_type == "manual.override":
            self._set_manual_override(message)
        elif message_type == "manual.velocity":
            velocity = self._manual_mapper.map(message)
            if velocity is not None:
                self._publish_manual_velocity(velocity)
        elif message_type == "heartbeat":
            return

    def _publish_inbound_proposal(self, source: dict[str, Any]) -> None:
        plan = source["plan"]
        try:
            validate_low_speed_plan(plan)
        except ValueError as exc:
            self._send_error("invalid_flight_profile", str(exc))
            return
        message = MissionProposal()
        message.stamp = self.get_clock().now().to_msg()
        message.proposal_id = str(uuid.uuid4())
        message.command_id = str(uuid.uuid4())
        message.raw_command = source["raw_command"]
        message.status = PROPOSAL_STATUSES[plan["status"]]
        message.plan_json = json.dumps(
            plan, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        message.message = str(plan.get("message") or "")
        message.requires_approval = (plan["status"] == "OK"
                                     and plan.get("request_purpose", "EXECUTE_MISSION") == "EXECUTE_MISSION")
        if (self._enforce_integrated_event_policy and message.requires_approval
                and not is_low_speed_profile(profile_for_plan(plan))):
            try:
                validate_integrated_event_plan(plan)
            except ValueError as exc:
                message.status = PROPOSAL_STATUSES["UNSUPPORTED"]
                message.requires_approval = False
                message.message = str(exc)
                self._send_error("integrated_policy_conflict", str(exc))
        self._proposal_publisher.publish(message)

    def _request_approval(self, source: dict[str, Any]) -> None:
        if not self._approval_client.service_is_ready():
            self._send_error(
                "approval_service_unavailable",
                "ApproveMission service is unavailable",
            )
            return
        frame = self._route_frame_status.get(source["proposal_id"])
        if source["approved"] and (not frame or frame.get("valid") is not True):
            self._send_error("route_frame_invalid", (frame or {}).get(
                "reason", "matching route frame status is unavailable"))
            return
        try:
            pending = self._approval_ledger.begin(source["proposal_id"], source["approved"])
        except ApprovalContractError as exc:
            self._send_error(exc.code, str(exc))
            return
        request = ApproveMission.Request()
        request.proposal_id = source["proposal_id"]
        request.approved = source["approved"]
        request.operator_id = source["operator_id"]
        try:
            future = self._approval_client.call_async(request)
            self._track_service_request(
                future, "approval", pending, self._on_approval_result)
        except Exception as exc:
            self._approval_ledger.cancel(pending)
            self._send_error("approval_failed", str(exc))

    def _on_approval_result(self, future, pending) -> None:
        result = None
        try:
            result = future.result()
            execution = self._approval_ledger.finish(
                pending, accepted=bool(result.accepted), mission_id=result.mission_id
            )
            self._send(
                {
                    "type": "mission.approval_result",
                    "accepted": bool(result.accepted),
                    "mission_id": result.mission_id,
                    "message": result.message,
                    "proposal_id": pending.proposal_id,
                }
            )
            if execution is not None:
                self._execute_mission(execution)
        except ApprovalContractError as exc:
            if pending.approved and result is not None and result.accepted and result.mission_id:
                self._revoke_execution_approval(
                    {"proposal_id": pending.proposal_id, "mission_id": result.mission_id},
                    "approval superseded before execution")
            self._send_error(exc.code, str(exc))
        except Exception as exc:
            self._approval_ledger.cancel(pending)
            self._send_error("approval_failed", str(exc))

    def _set_manual_override(self, source: dict[str, Any]) -> None:
        if source["active"]:
            self._approval_ledger.invalidate()
        if not self._manual_override_client.service_is_ready():
            self._send_error(
                "manual_override_service_unavailable",
                "SetManualOverride service is unavailable",
            )
            return
        request = SetManualOverride.Request()
        request.active = source["active"]
        request.source = source["source"]
        request.reason = source["reason"]
        future = self._manual_override_client.call_async(request)
        self._track_service_request(
            future, "manual_override", None,
            lambda completed, _context: self._on_manual_override_result(completed))

    def _on_manual_override_result(self, future) -> None:
        try:
            result = future.result()
            if not result.accepted:
                self._send_error("manual_override_rejected", result.message)
        except Exception as exc:
            self._send_error("manual_override_failed", str(exc))

    def _resume_mission(self, source: dict[str, Any]) -> None:
        if not self._resume_client.service_is_ready():
            self._send_error(
                "resume_service_unavailable",
                "ResumeMission service is unavailable",
            )
            return
        request = ResumeMission.Request()
        request.mission_id = source["mission_id"]
        request.operator_id = source["operator_id"]
        request.reason = source["reason"]
        future = self._resume_client.call_async(request)
        self._track_service_request(
            future, "resume", source["mission_id"], self._on_resume_result)

    def _prepare_forward_test(self, source: dict[str, Any]) -> None:
        if not self._forward_test_client.service_is_ready():
            self._send_error("forward_test_unavailable", "PrepareForwardTest service is unavailable")
            return
        request = PrepareForwardTest.Request()
        request.operator_id = source["operator_id"]
        request.target_altitude_home_m = source["target_altitude_home_m"]
        future = self._forward_test_client.call_async(request)
        self._track_service_request(
            future, "forward_test_prepare", None,
            lambda completed, _context: self._on_forward_test_prepared(completed))

    def _track_service_request(self, future, kind, context, callback):
        token = uuid.uuid4().hex
        with self._service_request_lock:
            self._pending_service_requests[token] = {
                "deadline": time.monotonic()+3.0,
                "generation": self._service_generation,
                "kind": kind,
                "context": context,
                "callback": callback,
            }
        future.add_done_callback(
            lambda completed, request_token=token: self._complete_service_request(
                request_token, completed))

    def _complete_service_request(self, token, future):
        with self._service_request_lock:
            pending = self._pending_service_requests.pop(token, None)
            generation = self._service_generation
        if pending is None or pending["generation"] != generation:
            return  # Timed out; a late response cannot revive the request.
        pending["callback"](future, pending["context"])

    def _invalidate_service_requests(self):
        with self._service_request_lock:
            pending = tuple(self._pending_service_requests.values())
            self._pending_service_requests.clear()
            self._service_generation += 1
        for request in pending:
            if request["kind"] == "approval" and request["context"] is not None:
                self._approval_ledger.cancel(request["context"])

    def _expire_service_requests(self):
        now = time.monotonic()
        expired = []
        with self._service_request_lock:
            for token, pending in tuple(self._pending_service_requests.items()):
                if now >= pending["deadline"]:
                    self._pending_service_requests.pop(token, None)
                    expired.append(pending)
        for pending in expired:
            if pending["kind"] == "approval" and pending["context"] is not None:
                self._approval_ledger.cancel(pending["context"])
            self._send_error(
                pending["kind"]+"_timeout",
                pending["kind"]+" service did not respond within 3 seconds",
            )

    def _on_forward_test_prepared(self, future) -> None:
        try:
            result = future.result()
            self._send({"type": "mission.forward_test_prepare_result",
                        "accepted": bool(result.accepted),
                        "proposal_id": result.proposal_id,
                        "message": result.message})
        except Exception as exc:
            self._send_error("forward_test_prepare_failed", str(exc))

    def _request_emergency_land(self, source: dict[str, Any]) -> None:
        if not self._emergency_land_client.service_is_ready():
            self._send_error(
                "emergency_land_service_unavailable",
                "EmergencyLand service is unavailable",
            )
            return
        request = EmergencyLand.Request()
        request.mission_id = source["mission_id"]
        request.operator_id = source["operator_id"]
        request.reason = source["reason"]
        try:
            future = self._emergency_land_client.call_async(request)
            self._track_service_request(
                future, "emergency_land", source["mission_id"],
                self._on_emergency_land_result,
            )
        except Exception as exc:
            self._send_error("emergency_land_failed", str(exc))

    def _on_emergency_land_result(self, future, mission_id: str) -> None:
        try:
            result = future.result()
            if result.accepted:
                active = self._active_execution_feedback.get(mission_id)
                goal_handle = active.get("goal_handle") if active is not None else None
                if goal_handle is not None:
                    # The service has already latched terminal LAND.  Also
                    # cancelling the Action wakes any executor wait promptly;
                    # the emergency LAND latch remains the authoritative end
                    # state instead of becoming a plain abort.
                    goal_handle.cancel_goal_async()
            self._send({
                "type": "mission.emergency_result",
                "action": "LAND",
                "accepted": bool(result.accepted),
                "mission_id": result.mission_id or mission_id,
                "message": result.message,
            })
        except Exception as exc:
            self._send_error("emergency_land_failed", str(exc))

    def _on_resume_result(self, future, mission_id: str) -> None:
        try:
            result = future.result()
            self._send(
                {
                    "type": "mission.resume_result",
                    "accepted": bool(result.accepted),
                    "mission_id": mission_id,
                    "message": result.message,
                }
            )
        except Exception as exc:
            self._send_error("resume_failed", str(exc))

    def _execute_mission(self, source: dict[str, Any]) -> None:
        if not self._execute_client.server_is_ready():
            self._send_error(
                "execute_action_unavailable",
                "ExecuteMission action is unavailable",
            )
            self._send_execution_result(
                False, MissionStatus.PHASE_ERROR,
                "ExecuteMission action is unavailable", source)
            self._revoke_execution_approval(source, "ExecuteMission action is unavailable")
            return
        context = None
        try:
            # Final epoch/expiry check and nonblocking dispatch are one state
            # transaction. No reconnect/manual/proposal callback can slip
            # between consuming this permit and publishing the action request.
            with self._approval_ledger.lock:
                source = self._approval_ledger.claim_execution(source)
                context = {
                    "proposal_id": source["proposal_id"],
                    "mission_id": source["mission_id"],
                }
                goal = ExecuteMission.Goal()
                goal.proposal_id = source["proposal_id"]
                goal.mission_id = source["mission_id"]
                goal.plan_json = source["plan_json"]
                goal.takeoff_altitude_m = 0.0
                future = self._execute_client.send_goal_async(
                    goal, feedback_callback=lambda feedback: self._on_execution_feedback(feedback, context))
                self._pending_goal_responses[goal.mission_id] = (
                    time.monotonic()+5.0, context)
        except ApprovalContractError as exc:
            self._send_error(exc.code, str(exc))
            return
        except Exception as exc:
            self._send_error("execution_dispatch_failed", str(exc))
            if context is not None:
                self._send_execution_result(
                    False, MissionStatus.PHASE_ERROR,
                    "execution dispatch failed: " + str(exc), context)
                self._revoke_execution_approval(context, str(exc))
            return
        future.add_done_callback(
            lambda completed, ctx=context: self._on_goal_response(
                completed, ctx
            )
        )

    def _on_goal_response(self, future, context: dict[str, str]) -> None:
        if self._pending_goal_responses.pop(context["mission_id"], None) is None:
            return
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self._send_execution_result(
                    False, MissionStatus.PHASE_ERROR, "mission goal rejected", context
                )
                self._revoke_execution_approval(context, "mission goal rejected")
                return
            self._send({"type": "mission.execution_started", **context,
                        "message": "ExecuteMission goal accepted"})
            self._active_execution_feedback[context["mission_id"]] = {
                "context": context,
                "goal_handle": goal_handle,
                "last_at": time.monotonic(),
                "stale_reported": False,
            }
            self._send_execution_health(context, False, 0, "feedback active")
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(
                lambda completed, ctx=context: self._on_execution_result(completed, ctx)
            )
        except Exception as exc:
            self._send_execution_result(
                False, MissionStatus.PHASE_ERROR, str(exc), context
            )
            self._revoke_execution_approval(context, str(exc))

    def _expire_goal_responses(self):
        now = time.monotonic()
        expired = []
        for mission_id, (deadline, context) in tuple(self._pending_goal_responses.items()):
            if now >= deadline:
                self._pending_goal_responses.pop(mission_id, None)
                expired.append(context)
        for context in expired:
            self._send_error("execute_action_timeout",
                             "ExecuteMission goal was not accepted within 5 seconds")
            self._revoke_execution_approval(context, "ExecuteMission goal timeout")
        for active in tuple(self._active_execution_feedback.values()):
            if now-active["last_at"] >= 3.0 and not active["stale_reported"]:
                active["stale_reported"] = True
                self._send_execution_health(
                    active["context"], True,
                    int((now-active["last_at"])*1000),
                    "ExecuteMission feedback stale; no automatic cancel was sent",
                )

    def _revoke_execution_approval(self, context, reason):
        if not self._approval_client.service_is_ready():
            self._send_error("approval_service_unavailable", reason)
            return
        request = ApproveMission.Request()
        request.proposal_id = context["proposal_id"]
        request.approved = False
        request.operator_id = "operator-gateway"
        try:
            future = self._approval_client.call_async(request)
            self._track_service_request(
                future, "approval_revoke", None,
                lambda completed, _context: completed.result(),
            )
        except Exception as exc:
            self._send_error("approval_revoke_failed", str(exc))

    def _on_route_frame_status(self, message: RouteFrameStatus) -> None:
        payload = {
            "type": "mission.route_frame_status",
            "proposal_id": message.proposal_id,
            "valid": bool(message.valid),
            "site_id": message.site_id,
            "home_wgs84": [message.home_latitude_deg, message.home_longitude_deg],
            "home_ned_m": [float(v) for v in message.home_ned_m],
            "first_target_ned_m": [float(v) for v in message.first_target_ned_m],
            "first_leg_m": float(message.first_leg_m),
            "maximum_leg_m": float(message.maximum_leg_m),
            "reason": message.reason,
            "preview_state": message.preview_state,
            "altitude_reference_error_valid": message.altitude_reference_error_valid,
            "altitude_reference_valid": bool(
                message.altitude_reference_valid),
            "altitude_reference_diagnostics_available": bool(
                message.altitude_reference_diagnostics_available),
            "aligned_home_z_ned_m": float(
                message.aligned_home_z_ned_m),
            "fc_altitude_home_relative_m": float(
                message.fc_altitude_home_relative_m),
            "normalized_fc_altitude_home_relative_m": float(
                message.normalized_fc_altitude_home_relative_m),
            "frame_altitude_home_relative_m": float(
                message.frame_altitude_home_relative_m),
            "altitude_reference_error_m": float(
                message.altitude_reference_error_m),
            "altitude_reference_detail": message.altitude_reference_detail,
            "home_correction_state": int(message.home_correction_state),
            "home_correction_valid": bool(message.home_correction_valid),
            "home_correction_revision": int(message.home_correction_revision),
            "home_altitude_correction_m": float(
                message.home_altitude_correction_m),
            "home_correction_opposition_error_m": float(
                message.home_correction_opposition_error_m),
            "home_correction_detail": message.home_correction_detail,
            "home_phase": int(message.home_phase),
            "execution_home_lock_valid": bool(
                message.execution_home_lock_valid),
            "execution_home_mission_id": message.execution_home_mission_id,
            "execution_home_lock_revision": int(
                message.execution_home_lock_revision),
            "provisional_home_revision": int(
                message.provisional_home_revision),
            "provisional_px4_home_altitude_amsl_m": float(
                message.provisional_px4_home_altitude_amsl_m),
            "provisional_px4_home_z_ned_m": float(
                message.provisional_px4_home_z_ned_m),
            "frozen_px4_home_altitude_amsl_m": float(
                message.frozen_px4_home_altitude_amsl_m),
            "frozen_px4_home_z_ned_m": float(
                message.frozen_px4_home_z_ned_m),
            "current_px4_home_altitude_amsl_m": float(
                message.current_px4_home_altitude_amsl_m),
            "current_px4_home_z_ned_m": float(
                message.current_px4_home_z_ned_m),
            "home_z_correction_m": float(message.home_z_correction_m),
            "altitude_epoch_failure_latched": bool(
                message.altitude_epoch_failure_latched),
        }
        self._route_frame_status[message.proposal_id] = payload
        while len(self._route_frame_status) > 64:
            self._route_frame_status.pop(next(iter(self._route_frame_status)))
        self._send(payload)

    def _on_execution_feedback(
        self, wrapped_feedback, context: dict[str, str]
    ) -> None:
        feedback = wrapped_feedback.feedback
        active = self._active_execution_feedback.get(context["mission_id"])
        if active is not None:
            was_stale = active["stale_reported"]
            active["last_at"] = time.monotonic()
            active["stale_reported"] = False
            if was_stale:
                self._send_execution_health(
                    context, False, 0, "ExecuteMission feedback resumed")
        vehicle = self._last_vehicle_state
        payload = {
                "type": "mission.status",
                "stamp": _stamp_dict(self.get_clock().now().to_msg()),
                "mission_id": context["mission_id"],
                "proposal_id": context["proposal_id"],
                "phase": int(feedback.phase),
                "approved": True,
                "manual_override": (
                    bool(vehicle.manual_override) if vehicle else False
                ),
                "armed": bool(vehicle.armed) if vehicle else False,
                "offboard": bool(vehicle.offboard) if vehicle else False,
                "current_waypoint": int(feedback.current_waypoint),
                "total_waypoints": int(feedback.total_waypoints),
                "detail": feedback.detail,
            }
        self._last_mission_payload = payload
        self._send(payload)

    def _on_execution_result(self, future, context: dict[str, str]) -> None:
        self._active_execution_feedback.pop(context["mission_id"], None)
        self._send_execution_health(context, False, 0, "execution completed")
        try:
            result = future.result().result
            self._send_execution_result(
                bool(result.success), int(result.final_phase), result.message, context
            )
        except Exception as exc:
            self._send_execution_result(
                False, MissionStatus.PHASE_ERROR, str(exc), context
            )

    def _send_execution_result(
        self, success: bool, final_phase: int, message: str,
        context: dict[str, str] | None = None,
    ) -> None:
        payload = {
                "type": "mission.execution_result",
                "success": success,
                "final_phase": final_phase,
                "message": message,
            }
        if context is not None:
            payload.update(proposal_id=context["proposal_id"], mission_id=context["mission_id"])
        self._last_execution_result = payload
        self._send(payload)

    def _send_execution_health(self, context, stale, feedback_age_ms, detail):
        payload = {
            "type": "mission.execution_health",
            "mission_id": context["mission_id"],
            "stale": bool(stale),
            "feedback_age_ms": max(0, int(feedback_age_ms)),
            "detail": str(detail),
        }
        self._last_execution_health = payload
        self._send(payload)

    def _on_proposal(self, message: MissionProposal) -> None:
        policy_error = ""
        policy_enabled = self._enforce_integrated_event_policy
        policy_applies = False
        try:
            plan = json.loads(message.plan_json)
            executable = (message.status == PROPOSAL_STATUSES["OK"]
                          and message.requires_approval and isinstance(plan, dict)
                          and plan.get("status") == "OK"
                          and plan.get("request_purpose", "EXECUTE_MISSION") == "EXECUTE_MISSION")
            json.dumps(plan, allow_nan=False)
            if executable and getattr(self, "_scenario_enabled", False):
                from .scenario_contract import validate_spec
                try:
                    validate_spec(plan)
                except ValueError as exc:
                    policy_error = "scenario candidate mismatch: " + str(exc)
                    executable = False
            policy_applies = (policy_enabled
                              and not is_low_speed_profile(profile_for_plan(plan)))
            if executable and policy_applies:
                try:
                    validate_integrated_event_plan(plan)
                except ValueError as exc:
                    policy_error = str(exc)
                    executable = False
        except (ValueError, TypeError):
            executable = False
        self._approval_ledger.remember(message.proposal_id, message.plan_json,
                                       executable=executable)
        self._send(
            {
                "type": "mission.proposal",
                "stamp": _stamp_dict(message.stamp),
                "proposal_id": message.proposal_id,
                "command_id": message.command_id,
                "raw_command": message.raw_command,
                "status": PROPOSAL_STATUSES["UNSUPPORTED"] if policy_error else int(message.status),
                "plan_json": message.plan_json,
                "message": policy_error or (message.message + "\n" + POLICY_SUMMARY_KO
                                             if policy_applies and executable else message.message),
                "requires_approval": bool(executable),
            }
        )
        if policy_error:
            # The existing ground UI renders error notices, but does not yet
            # render proposal.message. Do not leave a disabled approval button
            # with its policy conflict hidden in an unused JSON field.
            self._send_error("integrated_policy_conflict", policy_error)

    def _on_mission_status(self, message: MissionStatus) -> None:
        payload = {
                "type": "mission.status",
                "stamp": _stamp_dict(message.stamp),
                "mission_id": message.mission_id,
                "proposal_id": message.proposal_id,
                "phase": int(message.phase),
                "approved": bool(message.approved),
                "manual_override": bool(message.manual_override),
                "armed": bool(message.armed),
                "offboard": bool(message.offboard),
                "current_waypoint": int(message.current_waypoint),
                "total_waypoints": int(message.total_waypoints),
                "control_owner": message.control_owner,
                "event_id": message.event_id,
                "event_type": message.event_type,
                "event_lease_id": message.event_lease_id,
                "event_lease_remaining_s": float(message.event_lease_remaining_s),
                "detail": message.detail,
            }
        self._last_mission_payload = payload
        self._send(payload)

    def _on_vehicle_state(self, message: VehicleControlState, *, replay=False) -> None:
        prior = self._last_vehicle_state
        self._last_vehicle_state = message
        if not replay:
            self._last_vehicle_state_at = time.monotonic()
        age_ms = max(0, int((time.monotonic()-self._last_vehicle_state_at)*1000))
        if ((message.manual_override and not getattr(prior, "manual_override", False))
                or (getattr(message, "flight_epoch_retired", False)
                    and not getattr(prior, "flight_epoch_retired", False))):
            self._approval_ledger.invalidate()
        geo = self._last_vehicle_geo
        geo_age_ms = (min(2**32-1, int(getattr(geo, "age_ms", 0)
                                      + max(0.0, time.monotonic()-self._last_vehicle_geo_at)*1000))
                      if geo is not None and self._last_vehicle_geo_at > 0 else 2**32-1)
        low_speed_safety = self._last_low_speed_safety
        safety_fresh = bool(
            low_speed_safety is not None
            and self._last_low_speed_safety_at > 0.0
            and 0.0 <= time.monotonic()-self._last_low_speed_safety_at <= 0.5
        )
        payload = {
                "type": "vehicle.state",
                "gateway_vehicle_age_ms": age_ms,
                "active_execution_mission_id": message.active_execution_mission_id,
                "terminal_in_progress": message.terminal_in_progress,
                "emergency_land_available": age_ms < 500 and message.emergency_land_available,
                "emergency_land_detail": message.emergency_land_detail,
                "frame_altitude_valid": message.frame_altitude_valid,
                "altitude_reference_error_valid": message.altitude_reference_error_valid,
                "first_fault": message.first_fault,
                "terminal_detail": message.terminal_detail,
                "stamp": _stamp_dict(message.stamp),
                "mission_id": message.mission_id,
                "command_output_enabled": bool(message.command_output_enabled),
                "approved": bool(message.approved),
                "manual_override": bool(message.manual_override),
                "connected": bool(message.connected),
                "preflight_checks_pass": bool(message.preflight_checks_pass),
                "position_valid": bool(message.position_valid),
                "armed": bool(message.armed),
                "offboard": bool(message.offboard),
                "landed": bool(message.landed),
                "position_ned_m": [
                    float(value) for value in message.position_ned_m
                ],
                "velocity_ned_m_s": [
                    float(value) for value in getattr(message, "velocity_ned_m_s", (math.nan,)*3)
                ],
                "active_authority": message.active_authority,
                "jetson_safety_state": message.jetson_safety_state,
                "jetson_safety_fresh": bool(message.jetson_safety_fresh),
                "last_error": message.last_error,
                "avoidance_active": bool(getattr(message, "avoidance_active", False)),
                "flight_epoch_retired": bool(getattr(message, "flight_epoch_retired", False)),
                "heading_rad": float(getattr(message, "heading_rad", math.nan)),
                "heading_fresh": bool(getattr(message, "heading_fresh", False)),
                "route_heading_gate_enabled": bool(getattr(message, "route_heading_gate_enabled", False)),
                "terminal_owned_handoff_enabled": bool(getattr(message, "terminal_owned_handoff_enabled", False)),
                "low_speed_obstacle_guard_enabled": bool(getattr(message, "low_speed_obstacle_guard_enabled", False)),
                "forward_test_camera_bypass_enabled": bool(getattr(
                    message, "forward_test_camera_bypass_enabled", False
                )),
                "vehicle_status_fresh": bool(getattr(
                    message, "vehicle_status_fresh", False)),
                "arming_state_valid": bool(getattr(
                    message, "arming_state_valid", False)),
                "landed_state_valid": bool(getattr(
                    message, "landed_state_valid", False)),
                "terminal_state": int(getattr(
                    message, "terminal_state", getattr(
                        VehicleControlState, "TERMINAL_NONE", 0))),
                "flight_output_ready": bool(getattr(
                    message, "flight_output_ready", False)),
                "flight_output_detail": str(getattr(
                    message, "flight_output_detail", "")),
                "altitude_reference_valid": bool(getattr(
                    message, "altitude_reference_valid", False)),
                "fc_altitude_home_relative_m": float(getattr(
                    message, "fc_altitude_home_relative_m", 0.0)),
                "normalized_fc_altitude_home_relative_m": float(getattr(
                    message, "normalized_fc_altitude_home_relative_m", 0.0)),
                "frame_altitude_home_relative_m": float(getattr(
                    message, "frame_altitude_home_relative_m", 0.0)),
                "altitude_reference_error_m": float(getattr(
                    message, "altitude_reference_error_m", 0.0)),
                "altitude_reference_detail": str(getattr(
                    message, "altitude_reference_detail", "unavailable")),
                "home_correction_state": int(getattr(
                    message, "home_correction_state", 0)),
                "home_correction_valid": bool(getattr(
                    message, "home_correction_valid", False)),
                "home_correction_revision": int(getattr(
                    message, "home_correction_revision", 0)),
                "home_altitude_correction_m": float(getattr(
                    message, "home_altitude_correction_m", 0.0)),
                "home_correction_opposition_error_m": float(getattr(
                    message, "home_correction_opposition_error_m", 0.0)),
                "home_correction_detail": str(getattr(
                    message, "home_correction_detail", "unavailable")),
                "home_phase": int(getattr(message, "home_phase", 0)),
                "execution_home_lock_valid": bool(getattr(
                    message, "execution_home_lock_valid", False)),
                "execution_home_mission_id": str(getattr(
                    message, "execution_home_mission_id", "")),
                "execution_home_lock_revision": int(getattr(
                    message, "execution_home_lock_revision", 0)),
                "provisional_home_revision": int(getattr(
                    message, "provisional_home_revision", 0)),
                "provisional_px4_home_altitude_amsl_m": float(getattr(
                    message, "provisional_px4_home_altitude_amsl_m", 0.0)),
                "provisional_px4_home_z_ned_m": float(getattr(
                    message, "provisional_px4_home_z_ned_m", 0.0)),
                "current_px4_home_altitude_amsl_m": float(getattr(
                    message, "current_px4_home_altitude_amsl_m", 0.0)),
                "current_px4_home_z_ned_m": float(getattr(
                    message, "current_px4_home_z_ned_m", 0.0)),
                "home_z_correction_m": float(getattr(
                    message, "home_z_correction_m", 0.0)),
                "altitude_epoch_failure_latched": bool(getattr(
                    message, "altitude_epoch_failure_latched", False)),
                "altitude_alignment_ready": bool(getattr(
                    message, "altitude_alignment_ready", False)),
                "altitude_alignment_state": int(getattr(
                    message, "altitude_alignment_state", 2)),
                "altitude_alignment_epoch": int(getattr(
                    message, "altitude_alignment_epoch", 0)),
                "altitude_alignment_sample_count": int(getattr(
                    message, "altitude_alignment_sample_count", 0)),
                "altitude_alignment_window_ms": int(getattr(
                    message, "altitude_alignment_window_ms", 0)),
                "altitude_alignment_candidate_span_m": float(getattr(
                    message, "altitude_alignment_candidate_span_m", 0.0)),
                "altitude_alignment_source_skew_ms": int(getattr(
                    message, "altitude_alignment_source_skew_ms", 2**32-1)),
                "altitude_alignment_detail": str(getattr(
                    message, "altitude_alignment_detail",
                    "altitude_reference_state_missing")),
                "geo_valid": bool(geo and geo.valid and geo_age_ms <= 750),
                "latitude_deg": float(geo.latitude_deg) if geo else math.nan,
                "longitude_deg": float(geo.longitude_deg) if geo else math.nan,
                "altitude_home_relative_m": float(geo.altitude_home_relative_m) if geo else math.nan,
                "altitude_amsl_m": float(geo.altitude_amsl_m) if geo else math.nan,
                "geo_age_ms": geo_age_ms,
                "low_speed_safety_valid": bool(
                    safety_fresh and low_speed_safety.valid),
                "offboard_loss_action_land": bool(low_speed_safety and low_speed_safety.offboard_loss_action_land),
                "mpc_land_speed_m_s": (float(low_speed_safety.mpc_land_speed_m_s)
                                         if low_speed_safety else math.nan),
                "low_speed_safety_detail": (low_speed_safety.detail if low_speed_safety else
                                             "PX4 low-speed parameter state unavailable"),
            }
        self._send(payload)

    def _on_vehicle_geo(self, message: VehicleGeoState) -> None:
        self._last_vehicle_geo = message
        self._last_vehicle_geo_at = time.monotonic()

    def _on_low_speed_safety(self, message: LowSpeedSafetyState) -> None:
        self._last_low_speed_safety = message
        self._last_low_speed_safety_at = time.monotonic()

    def _release_manual_for_connection_change(self) -> None:
        velocity = self._manual_mapper.reset_connection()
        if velocity is not None:
            self._publish_manual_velocity(velocity)

    def _publish_manual_velocity(self, velocity) -> None:
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "map_ned"
        message.twist.linear.x = velocity.forward_m_s
        message.twist.linear.y = velocity.right_m_s
        message.twist.linear.z = velocity.down_m_s
        message.twist.angular.z = velocity.yaw_rad_s
        self._manual_velocity_publisher.publish(message)

    def _send_gateway_status(self, connected: bool, detail: str) -> None:
        self._send(
            {
                "type": "gateway.status",
                "ros_connected": connected,
                "detail": detail,
            }
        )

    def _send_error(self, code: str, message: str) -> None:
        self._send({"type": "error", "code": code, "message": message})

    def _on_gateway_fatal(self, detail: str) -> None:
        # The steady timer raises this on the ROS executor thread. main() does
        # not swallow it, so the process exits non-zero instead of running with
        # an incomplete critical-message stream.
        self._fatal_gateway_error = str(detail)

    def _send(self, message: dict[str, Any]) -> None:
        self._websocket.send(message)

    def destroy_node(self):
        self._release_manual_for_connection_change()
        self._websocket.close()
        self._execute_client.destroy()
        return super().destroy_node()


def _stamp_dict(stamp) -> dict[str, int]:
    return {"sec": int(stamp.sec), "nanosec": int(stamp.nanosec)}


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OperatorGatewayNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

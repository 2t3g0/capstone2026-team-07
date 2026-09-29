import asyncio

from starlette.websockets import WebSocketDisconnect

from jolgwa_uav.operator_hub import (
    OperatorHub, validate_operator_message, validate_ros_message,
)


class _FakeOperatorSocket:
    def __init__(self):
        self.accepted = False
        self.sent = []

    async def accept(self):
        self.accepted = True

    async def send_json(self, message):
        self.sent.append(message)

    async def receive(self):
        raise WebSocketDisconnect(1000)


def test_late_operator_receives_cached_gateway_capabilities():
    hub = OperatorHub()
    hub._ros = object()
    hub._latest_gateway_capabilities = {
        "type": "gateway.capabilities",
        "build_id": "low-speed-homecorr2-homelock2-safetycontract1-20260922",
        "protocol_version": 13,
        "route_frame_version": 1,
        "low_speed_profile_version": 2,
        "flight_output_handshake_version": 8,
        "altitude_reference_version": 6,
        "home_correction_version": 3,
        "status_freshness_version": 3,
        "emergency_control_version": 2,
        "auto_execute": True,
    }
    operator = _FakeOperatorSocket()

    asyncio.run(hub.operator_session(operator))

    assert operator.accepted
    assert operator.sent[0]["type"] == "gateway.status"
    assert operator.sent[0]["ros_connected"] is True
    assert operator.sent[1] == hub._latest_gateway_capabilities


def test_execution_result_preserves_mission_and_proposal_identity():
    result = validate_ros_message({
        "type": "mission.execution_result",
        "success": False,
        "final_phase": 8,
        "message": "takeoff failed",
        "proposal_id": "proposal-1",
        "mission_id": "mission-1",
    })
    assert result["proposal_id"] == "proposal-1"
    assert result["mission_id"] == "mission-1"


def test_execution_health_is_strictly_validated():
    result = validate_ros_message({
        "type": "mission.execution_health",
        "mission_id": "mission-1",
        "stale": True,
        "feedback_age_ms": 3100,
        "detail": "feedback stale",
    })
    assert result["stale"] is True


def test_emergency_land_and_result_are_strictly_validated():
    request, sequence = validate_operator_message({
        "type": "mission.emergency_land",
        "mission_id": "mission-1",
        "operator_id": "operator",
        "reason": "unexpected motion",
    }, None)
    assert request["mission_id"] == "mission-1"
    assert sequence is None

    result = validate_ros_message({
        "type": "mission.emergency_result",
        "action": "LAND",
        "accepted": True,
        "mission_id": "mission-1",
        "message": "accepted",
    })
    assert result["accepted"] is True

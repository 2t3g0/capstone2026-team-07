import queue
import asyncio
import json
import time
from types import MethodType, SimpleNamespace

from jolgwa_ros.operator_gateway_node import OperatorGatewayNode, RosWebSocketClient


def test_gateway_accounts_for_telemetry_queue_residence():
    async def run():
        client = _client()
        generation = client.begin_session()
        client.send(dict(type='vehicle.state', gateway_vehicle_age_ms=300, geo_age_ms=300))
        await asyncio.sleep(.06)
        sent = []
        class Socket:
            async def recv(self): await asyncio.Event().wait()
            async def send(self, raw):
                sent.append(json.loads(raw)); client._stop.set()
        await client._run_session(Socket(), generation)
        assert sent[0]['gateway_vehicle_age_ms'] >= 350
        assert sent[0]['geo_age_ms'] >= 350
    asyncio.run(run())


def test_gateway_stalled_send_is_bounded_and_session_queue_can_be_invalidated():
    async def run():
        client = _client(); generation=client.begin_session()
        client.send(dict(type='vehicle.state', gateway_vehicle_age_ms=0))
        class Socket:
            async def recv(self): await asyncio.Event().wait()
            async def send(self, raw): await asyncio.Event().wait()
        import pytest
        started=time.monotonic()
        with pytest.raises(asyncio.TimeoutError):
            await client._run_session(Socket(), generation)
        assert time.monotonic()-started < .6
        client.invalidate_session()
        assert client._latest_telemetry=={} and client._critical_outgoing.empty()
    asyncio.run(run())


def _client(fatal_callback=None):
    return RosWebSocketClient(
        "ws://127.0.0.1:1/ws/ros", queue.Queue(maxsize=16),
        0.1, 1.0, fatal_callback=fatal_callback,
    )


def test_new_websocket_generation_discards_old_critical_and_telemetry():
    client = _client()
    client.send({"type": "mission.execution_result", "message": "old"})
    client.send({"type": "vehicle.state", "mission_id": "old"})

    generation = client.begin_session()

    assert generation == 1
    assert client._critical_outgoing.empty()
    assert client._latest_telemetry == {}
    client.send({"type": "mission.execution_result", "message": "new"})
    item_generation, message, queued_at = client._critical_outgoing.get_nowait()
    assert item_generation == generation
    assert message["message"] == "new"


def test_critical_queue_overflow_is_fail_stop():
    failures = []
    client = _client(failures.append)
    client._critical_outgoing = queue.Queue(maxsize=1)

    client.send({"type": "mission.execution_result", "message": "one"})
    client.send({"type": "mission.execution_result", "message": "two"})

    assert client._unhealthy
    assert client._stop.is_set()
    assert failures == [
        "critical WebSocket output queue overflow; gateway restart required"]
    error = client._incoming.get_nowait()
    assert error["type"] == "_gateway.error"


def test_accepted_emergency_land_also_wakes_action_with_cancel():
    class _GoalHandle:
        def __init__(self):
            self.cancel_count = 0

        def cancel_goal_async(self):
            self.cancel_count += 1

    goal_handle = _GoalHandle()
    sent = []
    gateway = SimpleNamespace(
        _active_execution_feedback={
            "mission-1": {"goal_handle": goal_handle}},
        _send=sent.append,
        _send_error=lambda *args: (_ for _ in ()).throw(AssertionError(args)),
    )
    gateway._on_emergency_land_result = MethodType(
        OperatorGatewayNode._on_emergency_land_result, gateway)
    future = SimpleNamespace(result=lambda: SimpleNamespace(
        accepted=True, mission_id="mission-1", message="accepted"))

    gateway._on_emergency_land_result(future, "mission-1")

    assert goal_handle.cancel_count == 1
    assert sent == [{
        "type": "mission.emergency_result",
        "action": "LAND",
        "accepted": True,
        "mission_id": "mission-1",
        "message": "accepted",
    }]

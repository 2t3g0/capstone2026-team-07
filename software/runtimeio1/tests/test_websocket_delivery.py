import asyncio
import json
import time

from jolgwa_uav.operator_hub import OperatorHub
from jolgwa_uav.websocket_delivery import SocketWriter, with_residence_age


class Socket:
    def __init__(self, blocked=False):
        self.blocked = blocked
        self.messages = []
        self.closed = False

    async def send_json(self, value):
        if self.blocked:
            await asyncio.Event().wait()
        self.messages.append(value)

    async def close(self, **kwargs):
        self.closed = True


def test_slow_owner_does_not_block_observer_and_releases_control():
    async def run():
        hub = OperatorHub()
        slow, fast = Socket(True), Socket()
        hub._operators.update((slow, fast))
        hub._control_owner = slow
        stops = []
        async def stop(): stops.append(True)
        hub._send_manual_stop = stop
        source = dict(type='vehicle.state', gateway_vehicle_age_ms=300, geo_age_ms=300)
        await hub.broadcast(source)
        await asyncio.sleep(.03)
        assert fast.messages and not slow.messages
        assert source['gateway_vehicle_age_ms'] == 300
        await asyncio.sleep(.28)
        assert slow.closed and hub._control_owner is None and stops == [True]
        for peer in (slow, fast): await hub._close_writer(peer)
    asyncio.run(run())


def test_queues_are_bounded_and_critical_overflow_disconnects():
    async def run():
        peer = Socket(True)
        failures = []
        async def failed(): failures.append(True)
        writer = SocketWriter(peer.send_json, failed)
        for i in range(100):
            writer.enqueue(dict(type='vehicle.state', gateway_vehicle_age_ms=i))
        assert len(writer._telemetry) == 1
        for i in range(80):
            writer.enqueue(dict(type='mission.route_frame_status', proposal_id=str(i)))
        assert len(writer._telemetry) == 64
        futures = [writer.enqueue(dict(type='error', message=str(i))) for i in range(257)]
        assert len(writer._critical) == 256
        await asyncio.sleep(0)
        assert failures == [True]
        assert not any(await asyncio.gather(*futures))
        await writer.close()
    asyncio.run(run())


def test_residence_age_is_additive_without_mutating_source():
    original = dict(type='vehicle.state', gateway_vehicle_age_ms=300, geo_age_ms=300,
                    low_speed_safety_valid=True, altitude_alignment_ready=True)
    gateway = with_residence_age(original, 10., 10.2)
    hub = with_residence_age(gateway, 20., 20.1)
    assert 599 <= hub['gateway_vehicle_age_ms'] <= 601
    assert original['gateway_vehicle_age_ms'] == 300
    assert not hub['low_speed_safety_valid'] and not hub['altitude_alignment_ready']
    assert with_residence_age(original, 20., 19.)['gateway_vehicle_age_ms'] == 2**32-1


def test_fifo_and_reconnect_telemetry_invalidation():
    async def run():
        hub = OperatorHub(); peer = Socket()
        hub._operators.add(peer)
        await hub.broadcast(dict(type='vehicle.state', gateway_vehicle_age_ms=0))
        await hub.broadcast(dict(type='gateway.status', ros_connected=False))
        await asyncio.sleep(.01)
        assert [m['type'] for m in peer.messages] == ['gateway.status']
        writer = hub._writer(peer)
        results = [writer.enqueue(dict(type='error', message=str(i))) for i in range(10)]
        assert all(await asyncio.gather(*results))
        assert [m['message'] for m in peer.messages[1:]] == list(map(str, range(10)))
        await hub._close_writer(peer)
        assert peer not in hub._writers
    asyncio.run(run())


def test_cancelled_telemetry_waiter_can_be_replaced():
    async def run():
        peer=Socket()
        async def failed(): pass
        writer=SocketWriter(peer.send_json, failed)
        first=writer.enqueue(dict(type='vehicle.state',gateway_vehicle_age_ms=0))
        first.cancel()
        second=writer.enqueue(dict(type='vehicle.state',gateway_vehicle_age_ms=100))
        assert await second
        assert len(peer.messages)==1 and peer.messages[0]['gateway_vehicle_age_ms']>=100
        await writer.close()
    asyncio.run(run())


def test_actual_loopback_websocket_delivery():
    """Real WebSocket transport, loopback only; no operator/ROS process is contacted."""
    import websockets
    async def run():
        writers = []
        async def handler(socket, path=None):
            async def send_json(payload): await socket.send(json.dumps(payload))
            async def failed(): await socket.close()
            writer = SocketWriter(send_json, failed); writers.append(writer)
            try:
                for i in range(3):
                    assert await writer.send(dict(type='vehicle.state', gateway_vehicle_age_ms=100+i))
                await socket.wait_closed()
            finally:
                await writer.close()
        async with websockets.serve(handler, '127.0.0.1', 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with websockets.connect(f'ws://127.0.0.1:{port}') as client:
                samples = [json.loads(await asyncio.wait_for(client.recv(), 1.)) for _ in range(3)]
                assert all(100+i <= value['gateway_vehicle_age_ms'] < 500 for i, value in enumerate(samples))
        assert len(writers) == 1 and writers[0]._task.done()
    asyncio.run(run())


def test_ground_relay_freshness_matches_browser_without_reviving_invalid_flags():
    source = dict(type='vehicle.state',gateway_vehicle_age_ms=0,armed=False,landed=True,
        arming_state_valid=True,landed_state_valid=True,terminal_in_progress=False,
        active_execution_mission_id='',low_speed_safety_valid=True,altitude_alignment_ready=True,
        emergency_land_available=True)
    for age in (499,500,999,1000):
        payload=with_residence_age(dict(source,gateway_vehicle_age_ms=age),10.,10.)
        assert payload['low_speed_safety_valid'] == (age<1000)
        assert payload['emergency_land_available'] == (age<500)
        flight=with_residence_age(dict(source,gateway_vehicle_age_ms=age,armed=True),10.,10.)
        assert flight['low_speed_safety_valid'] == (age<500)
    assert not with_residence_age(dict(source,altitude_alignment_ready=False),10.,10.)['altitude_alignment_ready']


def test_real_server_close_reports_code_and_new_connection_has_no_old_queue(caplog):
    import websockets
    async def run():
        seen=[]; failed=[]
        async def handler(socket, path=None):
            async for data in socket:
                seen.append(json.loads(data))
        async with websockets.serve(handler,'127.0.0.1',0) as server:
            port=server.sockets[0].getsockname()[1]
            client=await websockets.connect(f'ws://127.0.0.1:{port}')
            async def send(payload): await client.send(json.dumps(payload))
            async def failure(): failed.append(True)
            writer=SocketWriter(send,failure)
            await client.close(code=1001,reason='server shutdown simulation')
            assert not await writer.send(dict(type='error',message='old'))
            await writer.close()
            async with websockets.connect(f'ws://127.0.0.1:{port}') as fresh:
                async def send_fresh(payload): await fresh.send(json.dumps(payload))
                new=SocketWriter(send_fresh,failure)
                assert await new.send(dict(type='error',message='new'))
                await new.close()
        assert seen==[dict(type='error',message='new')]
        assert failed==[True]
    asyncio.run(run())
    assert 'socket_writer_failed phase=send' in caplog.text
    assert 'close_code=1001' in caplog.text
    assert 'queue_critical=0' in caplog.text and 'send_s=' in caplog.text

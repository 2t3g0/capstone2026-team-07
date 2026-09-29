import asyncio
from types import SimpleNamespace

import jolgwa_uav.operator_hub as hub_module


def test_owner_survives_old_timeout_and_expires_after_120_seconds(monkeypatch):
    async def run():
        hub = hub_module.OperatorHub()
        owner = object()
        hub._control_owner = owner
        hub._control_last_seen_at = 10.0
        clock = [10.0]
        observations = []
        stops = []
        broadcasts = []
        ages = iter([30.001, 119.999, 120.0, 120.001])

        class Finished(Exception):
            pass

        async def tick(_):
            observations.append(hub._control_owner is owner)
            age = next(ages, None)
            if age is None:
                raise Finished()
            clock[0] = 10.0 + age

        async def stop():
            stops.append(True)

        async def broadcast():
            broadcasts.append(True)

        hub._send_manual_stop = stop
        hub._broadcast_control_status = broadcast
        monkeypatch.setattr(hub_module, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
        monkeypatch.setattr(hub_module, 'asyncio', SimpleNamespace(sleep=tick))
        try:
            await hub._manual_relay_loop()
        except Finished:
            pass
        assert observations == [True, True, True, True, False]
        assert stops == broadcasts == [True]
        assert hub._control_generation == 1

    asyncio.run(run())


def test_only_current_owners_heartbeat_renews_the_lease(monkeypatch):
    async def run():
        hub = hub_module.OperatorHub()
        owner = object()
        hub._control_owner = owner
        hub._control_last_seen_at = 10.0
        monkeypatch.setattr(hub_module, 'time', SimpleNamespace(monotonic=lambda: 129.0))
        await hub._touch_control_lease(object())
        assert hub._control_last_seen_at == 10.0
        await hub._touch_control_lease(owner)
        assert hub._control_last_seen_at == 129.0

    asyncio.run(run())


def test_active_manual_relay_still_stops_at_the_original_timeout(monkeypatch):
    async def run():
        hub = hub_module.OperatorHub()
        hub._control_owner = object()
        hub._control_last_seen_at = 10.0
        hub._latest_manual = {'deadman': True, 'forward': 0.5}
        stops = []

        class Finished(Exception):
            pass

        async def tick(_):
            pass

        async def stop():
            stops.append(True)

        async def broadcast():
            raise Finished()

        hub._send_manual_stop = stop
        hub._broadcast_control_status = broadcast
        monkeypatch.setattr(hub_module, 'time', SimpleNamespace(monotonic=lambda: 40.001))
        monkeypatch.setattr(hub_module, 'asyncio', SimpleNamespace(sleep=tick))
        try:
            await hub._manual_relay_loop()
        except Finished:
            pass
        assert hub._control_owner is None
        assert hub._latest_manual is None
        assert stops == [True]

    asyncio.run(run())

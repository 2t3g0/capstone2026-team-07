"""Regression for the field log: one-shot command expiry released active Home.

Exercise real bridge callbacks and RouteFcState without an FC or ROS executor.
In particular this is not a test of a duplicated lifecycle predicate.
"""
from collections import deque
from types import MethodType, SimpleNamespace as NS

import pytest
from jolgwa_interfaces.msg import MissionApproval, OffboardCommand, FlightControlContract, FlightCommandRequest
from jolgwa_ros.mavlink_usb_bridge_node import MavlinkUsbBridgeNode
from jolgwa_uav.mavlink_route_fc_link import RouteFcState


START = 10_000_000_000


def bridge_fixture():
    route = RouteFcState()
    route.home_position = dict(latitude_deg=35.0, longitude_deg=129.0,
                               altitude_m=40.0, ned=(0.0, 0.0, 0.0))
    route.latest_px4_home_position = dict(route.home_position)
    route.home_candidate_since_ns = START - 4_000_000_000
    route.global_position = dict(latitude_deg=35.0, longitude_deg=129.0, altitude_m=40.0)
    route.local_position = dict(position=(0.0, 0.0, 0.0))
    assert route.lock_execution_home("active-mission")[0]
    bridge = NS(
        _battery_landings={}, _last_health_diagnostic=None, _restart_required=False,
        _journal=__import__('io').StringIO(),
        _route_state=route, _ended_mission_ids=deque(maxlen=64),
        _command_mission_id="active-mission", _command_received_ns=START,
        _command_code=OffboardCommand.COMMAND_TAKEOFF,
        _command_profile=OffboardCommand.FLIGHT_PROFILE_NORMAL,
        _mode=object(), _setpoint=object(), _envelope=object(),
        _output_pairs=NS(clear=lambda: None),
        _commands=deque(), _last_setpoint_tx_ns=START,
        _consecutive_setpoint_tx=1, _last_envelope_valid=True,
        _contract=None, _intent=None, _approved_missions={"active-mission"},
        _intent_history={},
        _tx_run_started_ns=START, _tx_run_id=1, _arm_transmitted=False,
        _pending_acks={}, _last_tx_terminal_brake=False,
        _altitude_reference_sync=NS(
            release_execution_home=lambda: None,
            set_home_reference=lambda **kw: None,
            set_provisional_home=lambda **kw: None),
        get_logger=lambda: NS(info=lambda msg: None),
    )
    for name in ("_synchronize_home_lifecycle", "_on_mission_approval",
                 "_on_offboard_command", "_stream_setpoint", "_drain_command_queue",
                 "_reset_tx_run", "_contract_valid", "_load_output_command", "_on_output_contract", "_commit_output_contract",
                 "_observe_battery_health"):
        setattr(bridge, name, MethodType(getattr(MavlinkUsbBridgeNode, name), bridge))
    return bridge


def status(connected=True, armed=False, landed=True):
    # RouteFcState sets connected=False/landed=None when the source is stale.
    return NS(connected=connected, armed=armed, landed=landed)


def complete(bridge, mission_id="active-mission"):
    bridge._on_mission_approval(MissionApproval(mission_id=mission_id, approved=False))


@pytest.mark.parametrize("command_age_ms", [249, 250, 251, 1000, 1200, 3000, 10000])
def test_one_shot_takeoff_keeps_home_through_prearm_tx(command_age_ms):
    bridge = bridge_fixture()
    now = START + command_age_ms * 1_000_000
    bridge._setpoint_received_ns = bridge._envelope_received_ns = now
    bridge._last_setpoint_tx_ns = now
    for kind in ("HOME_POSITION", "LOCAL_POSITION_NED", "GLOBAL_POSITION_INT"):
        bridge._route_state.received_ns[kind] = now
    original_home = dict(bridge._route_state.home_position)
    original_stability_start = bridge._route_state.home_candidate_since_ns
    bridge._synchronize_home_lifecycle(status(), now)
    assert bridge._route_state.home_locked
    assert bridge._route_state.home_candidate_since_ns == original_stability_start
    assert bridge._route_state.home_position == original_home
    assert bridge._route_state.route_home(now)[0] is not None


def test_tx_offboard_arm_landing_sequence_keeps_lock_until_explicit_completion():
    bridge = bridge_fixture()
    # One command, many TX cycles, then Offboard confirmation/ARM/flight/LAND.
    for ms in range(0, 1250, 50):
        bridge._synchronize_home_lifecycle(status(), START + ms * 1_000_000)
        assert bridge._route_state.home_locked
    for ms, state in ((1300, status(armed=True)),
                      (2000, status(armed=True, landed=False)),
                      (6000, status(armed=True)), (7000, status())):
        bridge._synchronize_home_lifecycle(state, START + ms * 1_000_000)
        assert bridge._route_state.home_locked
    complete(bridge)
    bridge._synchronize_home_lifecycle(status(), START + 8_000_000_000)
    assert not bridge._route_state.home_locked


@pytest.mark.parametrize("state", [status(connected=False), status(landed=None),
                                  status(armed=True), status(landed=False)])
def test_completion_does_not_release_home_without_fresh_ground_confirmation(state):
    bridge = bridge_fixture()
    complete(bridge)
    bridge._synchronize_home_lifecycle(state, START + 5_000_000_000)
    assert bridge._route_state.home_locked
    bridge._synchronize_home_lifecycle(status(), START + 6_000_000_000)
    assert not bridge._route_state.home_locked


def test_completion_for_another_mission_and_link_loss_do_not_release_home():
    bridge = bridge_fixture()
    complete(bridge, "other-mission")
    bridge._synchronize_home_lifecycle(status(connected=False), START + 90_000_000_000)
    bridge._synchronize_home_lifecycle(status(), START + 91_000_000_000)
    assert bridge._route_state.home_locked


def test_release_clears_output_and_does_not_repeat_or_accept_delayed_takeoff():
    bridge = bridge_fixture()
    bridge._commands.append((176, (), START))
    complete(bridge)
    bridge._synchronize_home_lifecycle(status(), START + 5_000_000_000)
    assert not bridge._route_state.home_locked
    assert bridge._mode is bridge._setpoint is bridge._envelope is None
    assert not bridge._commands
    assert bridge._consecutive_setpoint_tx == 0
    stability_start = bridge._route_state.home_candidate_since_ns
    complete(bridge)
    bridge._synchronize_home_lifecycle(status(), START + 6_000_000_000)
    assert bridge._route_state.home_candidate_since_ns == stability_start
    bridge._on_offboard_command(OffboardCommand(
        mission_id="active-mission", sequence=100, command=OffboardCommand.COMMAND_TAKEOFF))
    assert bridge._command_received_ns == START
    bridge._stream_setpoint(status(), START + 6_000_000_000)
    assert bridge._last_output_detail == "flight_control_contract_missing_or_mismatched"


@pytest.mark.parametrize("command", [176, 400])
def test_completed_mission_cannot_dispatch_queued_mode_or_arm(command):
    bridge = bridge_fixture()
    complete(bridge)
    bridge._transport = NS(send_vehicle_command=lambda *a: pytest.fail("unexpected FC write"))
    bridge._private_command_graph_ready = lambda now: True
    bridge._commands.append(FlightCommandRequest(command=command,
        params=[1.0]*7, issued_monotonic_ns=START))
    acks = []
    bridge._publish_local_ack = lambda *args: acks.append(args)
    bridge._drain_command_queue(status(), START+1)
    assert not bridge._commands
    assert acks == [(command, "VEHICLE_CMD_RESULT_DENIED")]


def test_terminal_land_remains_allowed_while_waiting_for_ground():
    bridge = bridge_fixture()
    land = OffboardCommand(mission_id="active-mission", sequence=2,
                           flight_output_handshake_version=8,
                           command=OffboardCommand.COMMAND_LAND, approved=True)
    bridge._on_offboard_command(land)
    bridge._on_output_contract(FlightControlContract(
        command=land, output_epoch="epoch", output_sequence=1, handshake_version=8))
    assert bridge._command_code == OffboardCommand.COMMAND_LAND
    bridge._synchronize_home_lifecycle(status(armed=True, landed=False), START+5_000_000_000)
    assert bridge._route_state.home_locked
    sent = []
    bridge._transport = NS(send_vehicle_command=lambda *a: sent.append(a))
    bridge._private_command_graph_ready = lambda now: True
    bridge._journal = __import__('io').StringIO()
    bridge._commands.append(FlightCommandRequest(mission_id="active-mission",
        source_sequence=2, output_epoch="epoch", output_sequence=1,
        command=21, params=[0.0]*7, issued_monotonic_ns=START))
    state = status(armed=True, landed=False)
    state.offboard = True
    bridge._drain_command_queue(state, START+1)
    assert sent[0][0] == 21

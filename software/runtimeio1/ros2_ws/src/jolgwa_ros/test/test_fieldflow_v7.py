"""Field failures reproduced through production policy and controller callbacks."""
import copy
import json
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
from jolgwa_ros import scenario_altitude_recovery as recovery
from jolgwa_ros.scenario_contract import ScenarioRegistry, make_spec, within
from jolgwa_ros.scenario_runtime import navigation_error
from jolgwa_ros.scenario_sequence import ScenarioSequence
from jolgwa_ros.px4_offboard_controller import Px4OffboardController as Controller
from jolgwa_ros.flight_contract import command_matches_intent
from test_battery_landing import controller
from test_safety_contract import proof, bridge
from test_scenario import plan
from test_front_climb import evidence


def approved(c):
    c._scenario_enabled = True
    c._scenario_registry = ScenarioRegistry()
    c._scenario_registry.proposal(NS(proposal_id='p', plan_json=json.dumps(plan())))
    c._scenario_registry.approval(NS(mission_id='m', proposal_id='p', approved=True))
    c.get_logger = lambda: NS(info=lambda _: None, warning=lambda _: None, error=lambda _: None)
    return c


def recovering(age=0):
    c = approved(controller())
    c._last_px4_message_at = c._last_landed_message_at = 100.
    c._heading_fresh = lambda now: True
    c._altitude_reference_status = Controller._altitude_reference_status.__get__(c)
    c._altitude_reference_state_received_at = 100.
    c._altitude_reference_state = NS(valid=True, stable=True, state=1,
        transport_epoch=1, sequence=21, local_age_ms=age, global_age_ms=age,
        source_skew_ms=0, local_z_ned_m=-1., fc_altitude_home_relative_m=1.,
        normalized_fc_altitude_home_relative_m=1., execution_home_mission_id='m',
        execution_home_lock_revision=1)
    c.velocities = []
    c._publish_velocity_setpoint = lambda v, yaw: c.velocities.append(v)
    c.retired = []
    c._retire_autonomy_epoch = lambda reason: c.retired.append(reason)
    fault = proof(c, 100_000_000_000)
    fault.detail = 'altitude_pose_time_skew'
    old = copy.deepcopy(c._output_contract)
    with patch('time.monotonic', return_value=100.), patch('time.monotonic_ns', return_value=100_000_000_000):
        c._on_flight_output_state(fault)
    assert c._altitude_recovery and c._active_command.command == 0
    assert c._output_contract.output_sequence == old.output_sequence+1
    assert command_matches_intent(c._active_command, old.command, allow_recovery_hold=True)
    assert not command_matches_intent(c._active_command, old.command)
    return c, old


def test_new_hold_tx_then_new_sample_resumes_same_target():
    c, old = recovering()
    assert recovery.service(c, 100.1)
    assert c.velocities == [[0., 0., 0.]]
    stale = proof(c, 100_100_000_000); stale.detail = recovery.READY
    stale.output_sequence = old.output_sequence
    c._flight_output_state = stale
    c._altitude_reference_state.sequence += 1
    with patch('time.monotonic_ns', return_value=100_100_000_000):
        assert recovery.service(c, 100.1) and c._altitude_recovery
        c._flight_output_state = proof(c, 100_100_000_000)
        c._flight_output_state.detail = recovery.READY
        assert recovery.service(c, 100.1)
    assert c._altitude_recovery is None
    assert list(c._active_command.position_ned_m) == list(old.command.position_ned_m)
    assert c._active_command.command == 1
    assert c._output_contract.output_sequence == old.output_sequence+2


@pytest.mark.parametrize('elapsed,expired', [(.499,False),(.5,True),(.501,True)])
def test_recovery_deadline_no_repeat_extension(elapsed,expired):
    c, _ = recovering()
    start = c._altitude_recovery['deadline']
    c._last_px4_message_at = 100.+elapsed
    assert recovery.service(c, 100.+elapsed)
    assert (c._terminal_started_at is not None) == expired
    if not expired:
        assert c._altitude_recovery['deadline'] == start
    else:
        assert c._active_command.command == 3
        assert not c._terminal_brake_transmitted


def test_original_sample_age_ends_recovery_before_half_second():
    c, _ = recovering(300)
    c._last_px4_message_at = 100.201
    recovery.service(c, 100.201)
    assert c._terminal_started_at == 100.201


@pytest.mark.parametrize('fault', ['rc','approval','heading','epoch','home','new_command'])
def test_recovery_never_restores_over_hard_fault_or_new_command(fault):
    c, _ = recovering()
    if fault == 'rc': c._gate.manual_override = True
    if fault == 'approval': c._gate.is_approved = lambda _: False
    if fault == 'heading': c._heading_fresh = lambda _: False
    if fault == 'epoch': c._altitude_reference_state.transport_epoch = 2
    if fault == 'home': c._altitude_reference_state.execution_home_lock_revision = 2
    if fault == 'new_command': c._active_command.sequence += 1
    recovery.service(c, 100.1)
    assert c._altitude_recovery is None
    assert c._active_command.command != 1


@pytest.mark.parametrize('radius,allowed',[(.499,True),(.5,True),(.501,False)])
def test_takeoff_radius_and_delayed_takeoff_fence(radius,allowed):
    c = approved(controller())
    first = copy.deepcopy(c._active_command)
    assert (navigation_error(c, first, [-radius,0.,-.5]) == '') == allowed
    move = copy.deepcopy(first); move.sequence += 1; move.command = 2
    assert navigation_error(c, move, [0.,0.,-1.]) == ''
    assert navigation_error(c, first, [0.,0.,-1.]) == 'scenario_takeoff_phase_retired'


def test_fresh_clear_after_stop_resumes_search_without_completing_obstacle():
    s = ScenarioSequence(make_spec([0.,0.,0.],0.,0.,1,1.),100.)
    s.tick((1.,0.,-1.),(0.,)*3,evidence(100.,2),100.)
    s.tick(s.target,(0.,)*3,evidence(100.1,2),100.1)
    assert s.tick(s.target,(0.,)*3,evidence(100.7,0),100.7)[0] == 'SEARCH_OBSTACLE'
    assert s.target == (10.,0.,-1.)
    assert s.brake_elapsed == pytest.approx(.7)


def test_brake_chatter_cannot_renew_sixty_second_total_budget():
    s=ScenarioSequence(make_spec([0.,0.,0.],0.,0.,1,1.),100.)
    for i in range(100):
        t=100.+i
        s.tick((1.,0.,-1.),(0.,)*3,evidence(t,2),t)
        s.tick(s.target,(0.,)*3,evidence(t+.1,2),t+.1)
        s.tick(s.target,(0.,)*3,evidence(t+.7,0),t+.7)
        if s.phase=='FAILED': break
    assert s.reason=='scenario_stage_timeout:BRAKE'
    assert 59.3 <= s.brake_elapsed <= 60.


def test_takeoff_settle_requires_half_second_in_ordinary_corridor():
    from jolgwa_ros.mission_manager_node import MissionManagerNode as Manager
    c=approved(controller()); c._low_settle_since=None
    state=NS(mission_id='m',approved=True,manual_override=False,active_authority='LLM_ROUTE',
        velocity_ned_m_s=[0.,0.,0.],position_ned_m=[-.21,0.,-1.],last_error='')
    with patch('time.monotonic',return_value=100.):
        assert not Manager._low_target_reached(c,state,'m',[0.,0.,-1.])
    state.position_ned_m=[-.199,0.,-1.]
    with patch('time.monotonic',return_value=101.):
        assert not Manager._low_target_reached(c,state,'m',[0.,0.,-1.])
    with patch('time.monotonic',return_value=101.499):
        assert not Manager._low_target_reached(c,state,'m',[0.,0.,-1.])
    with patch('time.monotonic',return_value=101.5):
        assert Manager._low_target_reached(c,state,'m',[0.,0.,-1.])


@pytest.mark.parametrize('state,stamp', [(0,100.),(4,100.7)])
def test_old_clear_unknown_do_not_resume(state,stamp):
    s = ScenarioSequence(make_spec([0.,0.,0.],0.,0.,1,1.),100.)
    s.tick((1.,0.,-1.),(0.,)*3,evidence(100.,2),100.)
    s.tick(s.target,(0.,)*3,evidence(100.1,2),100.1)
    assert s.tick(s.target,(0.,)*3,evidence(stamp,state),100.7)[0] == 'BRAKE'


def test_bridge_recovery_requires_zero_velocity_and_original_contract(bridge):
    from test_battery_home_v6 import setup_v6
    from test_vehicle_command_transport import fc_snapshot
    from jolgwa_interfaces.msg import FlightEnvelope
    b = bridge; c = setup_v6(b)
    original = copy.deepcopy(b._contract)
    hold = copy.deepcopy(original); hold.output_sequence += 1; hold.command.command = 0
    b._on_output_contract(hold)
    assert b._contract_valid()
    e = FlightEnvelope(setpoint_kind=FlightEnvelope.SETPOINT_VELOCITY)
    snap = fc_snapshot(armed=True, offboard=True, landed=False)
    assert recovery.bridge_hold(b,snap,100_000_000_000,e,[0.,0.,0.])
    with pytest.raises(ValueError,match='zero_velocity'):
        recovery.bridge_hold(b,snap,100_001_000_000,e,[.1,0.,0.])
    with pytest.raises(ValueError,match='expired'):
        recovery.bridge_hold(b,snap,100_500_000_000,e,[0.,0.,0.])


@pytest.mark.parametrize('skew,blocked', [(149,False),(150,False),(151,True),(159,True)])
def test_production_bridge_skew_to_new_zero_tx_then_land(bridge,skew,blocked):
    import time
    from dataclasses import replace
    from test_battery_home_v6 import setup_v6
    from test_battery_landing import brake_pair, output
    from test_vehicle_command_transport import fc_snapshot
    b = bridge; c = setup_v6(b)
    b._route_state.reset_counter = 0
    b._route_state.home_correction_valid = True
    b._altitude_reference_sync.set_home_reference(**b._route_state.home_correction_snapshot())
    base = time.monotonic_ns()-2_000_000_000
    for i in range(21):
        ns = base+i*100_000_000
        b._altitude_reference_sync.add_local(time_boot_ms=1000+i*100,z_ned_m=0.,received_ns=ns)
        sample=b._altitude_reference_sync.add_global(time_boot_ms=1000+i*100,
            fc_altitude_home_relative_m=0.,global_altitude_amsl_m=40.,received_ns=ns,now_ns=ns)
    snap=fc_snapshot(armed=True,offboard=True,landed=False,position=(0.,0.,0.),
        position_source='LOCAL_POSITION_NED',position_received_ns=ns,time_boot_ms=3000+skew,
        transport_epoch=sample.transport_epoch)
    original=copy.deepcopy(b._contract)
    with patch('time.monotonic_ns',return_value=ns),patch('time.monotonic',return_value=ns/1e9):
        brake_pair(b,ns); b._stream_setpoint(snap,ns)
        assert bool(b._transport.packets) != blocked, b._last_output_detail
        if not blocked: return
        assert b._last_output_detail == 'altitude_pose_time_skew'
        hold=copy.deepcopy(original); hold.command.command=0; hold.output_sequence+=1
        b._on_output_contract(hold)
        brake_pair(b,ns+10_000_000); b._stream_setpoint(snap,ns+10_000_000)
        assert b._last_output_detail == recovery.WAIT
        assert len(b._transport.packets)==1
        # The skew disappears without moving the prepared target or Home.
        snap=replace(snap,time_boot_ms=3000)
        brake_pair(b,ns+70_000_000); b._stream_setpoint(snap,ns+70_000_000)
        assert b._last_output_detail == recovery.READY
        assert len(b._transport.packets)==2
        # A real loss of altitude evidence still admits only the terminal brake.
        b._altitude_reference_sync.reject_home_correction('test_reference_failure')
        land=copy.deepcopy(original); land.command.command=3; land.output_sequence+=2
        b._on_output_contract(land)
        brake_pair(b,ns+130_000_000); b._stream_setpoint(snap,ns+130_000_000)
        assert len(b._transport.packets)==3, b._last_output_detail
        assert b._last_tx_terminal_brake

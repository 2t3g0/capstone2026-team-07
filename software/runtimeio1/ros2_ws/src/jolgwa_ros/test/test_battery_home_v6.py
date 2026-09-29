"""Production callback tests with an approved v6 contract and no flight hardware."""
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import pytest
from jolgwa_ros import scenario_battery_home as policy
from jolgwa_ros.scenario_contract import ScenarioRegistry
from jolgwa_ros.low_speed import AltitudeReferenceSynchronizer
from test_safety_contract import bridge
from test_battery_landing import controller, install, battery_snapshot, output
from test_scenario import plan


def setup_v6(b):
    c=controller(); install(b,c)
    for node in (b,c):
        node._scenario_enabled=True; node._scenario_registry=ScenarioRegistry()
        node._scenario_registry.proposal(NS(proposal_id='p',plan_json=json.dumps(plan())))
        node._scenario_registry.approval(NS(mission_id='m',proposal_id='p',approved=True))
    return c


def snapshot(ns,state=2,**kw):
    return battery_snapshot(battery_navigation_healthy=True,transport_epoch=1,
        battery_sample=dict(valid=True,id=0,charge_state=state,fault_bitmask=0,received_ns=ns,epoch=1,voltage_v=14.),**kw)


@pytest.mark.parametrize('order',['intent','contract'])
@pytest.mark.parametrize('delay_ms,expired',[(10,False),(499,False),(500,True),(501,True)])
def test_pair_order_does_not_poison_battery_handoff(bridge,order,delay_ms,expired):
    b=bridge; c=setup_v6(b); base=time.monotonic_ns()
    cmd=copy.deepcopy(c._active_command); cmd.command=0; cmd.sequence+=1
    contract=copy.deepcopy(c._output_contract); contract.command=cmd; contract.output_sequence+=1
    old=b._contract.command.sequence
    with patch('time.monotonic_ns',return_value=base):
        if order=='intent': b._on_offboard_command(cmd)
        else: b._on_output_contract(contract)
        assert b._contract_valid()
        assert b._contract.command.sequence==old
        assert not b._observe_battery_health(snapshot(base),base)
        assert b._battery_navigation_allowed and not b._battery_landings
    now=base+delay_ms*10**6
    with patch('time.monotonic_ns',return_value=now):
        if order=='intent': b._on_output_contract(contract)
        else: b._on_offboard_command(cmd)
    assert policy.pair_expired(b,now)==expired
    assert b._contract.command.sequence==(old if expired else cmd.sequence)
    assert b._contract_valid() and not b._battery_landings


def test_duplicate_wait_deadline_and_revoke(bridge):
    b=bridge; setup_v6(b); base=time.monotonic_ns()
    cmd=copy.deepcopy(b._intent); cmd.sequence+=1
    with patch('time.monotonic_ns',return_value=base): b._on_offboard_command(cmd)
    with patch('time.monotonic_ns',return_value=base+400_000_000): b._on_offboard_command(cmd)
    assert b._scenario_pair_started_ns==base
    b._on_mission_approval(NS(mission_id='m',approved=False))
    assert not b._observe_battery_health(snapshot(base),base)
    assert not b._battery_navigation_allowed and not b._contract_valid()
    assert b._scenario_pending_contract is None and b._scenario_pair_started_ns is None


def test_changed_output_epoch_cannot_reacquire_current_mission(bridge):
    b=bridge; c=setup_v6(b)
    old=b._contract
    new=copy.deepcopy(old); new.output_epoch='restarted'
    b._on_output_contract(new)
    assert b._restart_required and b._contract is old
    assert b._last_output_detail=='scenario_output_epoch_changed'


def test_land_contract_before_intent_preserves_battery_authority(bridge):
    b=bridge; c=setup_v6(b); base=time.monotonic_ns()
    assert b._observe_battery_health(snapshot(base,3),base)
    cmd=copy.deepcopy(c._active_command); cmd.command=3; cmd.sequence+=1
    new=copy.deepcopy(c._output_contract); new.command=cmd; new.output_sequence+=1
    with patch('time.monotonic_ns',return_value=base):
        b._on_output_contract(new)
        assert b._observe_battery_health(snapshot(base,3),base)
    with patch('time.monotonic_ns',return_value=base+10_000_000): b._on_offboard_command(cmd)
    assert b._contract.command.sequence==cmd.sequence
    assert b._contract_valid() and not b._battery_landings['m']['failed']


@pytest.mark.parametrize('battery',[False,True])
def test_lost_land_intent_repairs_context_without_extending_deadline(bridge,battery):
    from test_vehicle_command_transport import fc_snapshot
    b=bridge; c=setup_v6(b); base=time.monotonic_ns(); old=copy.deepcopy(c._output_contract)
    if battery: assert b._observe_battery_health(snapshot(base,3),base)
    land=copy.deepcopy(c._active_command); land.command=3; land.sequence+=1
    with patch('time.monotonic_ns',return_value=base),patch('time.monotonic',return_value=base/1e9):
        c._terminal_handoff_valid=True  # Set by the owned Manager LAND acceptance path.
        c._begin_terminal(land,base/1e9)  # Contract delivered; intent deliberately lost.
    deadline=c._terminal_started_at; brake=c._terminal_brake_started_at
    now=base+500_000_000
    snap=snapshot(now,3) if battery else fc_snapshot(armed=True,offboard=True,landed=False)
    with patch('time.monotonic_ns',return_value=now),patch('time.monotonic',return_value=now/1e9):
        b._stream_setpoint(snap,now)
        c._last_px4_message_at=c._last_landed_message_at=now/1e9
        prior_output,_=output(b,snap,c)
    assert c._terminal_started_at==deadline and c._terminal_brake_started_at==brake
    assert c._terminal_pair_repaired and not c._terminal_brake_transmitted
    assert b._contract_valid() and b._contract.command.sequence==old.command.sequence
    assert b._contract.command.command==3 and b._contract.output_sequence>old.output_sequence
    c._terminal_command_sent=True; c._offboard=False
    state=c._terminal_state
    with patch('time.monotonic_ns',return_value=now+10_000_000),patch('time.monotonic',return_value=(now+10_000_000)/1e9):
        c._on_flight_output_state(prior_output)
    assert c._terminal_state==state  # A delayed wait diagnostic cannot undo native LAND.


def test_low_then_new_terminal_brake_context_not_old_tx(bridge):
    b=bridge; c=setup_v6(b); base=time.monotonic_ns()
    for ms in range(0,5000,200):
        now=base+ms*10**6
        with patch('time.monotonic_ns',return_value=now),patch('time.monotonic',return_value=now/1e9):
            state,status=output(b,snapshot(now))
            assert state.detail=='battery_low_observing'
            assert not status.failsafe and not status.pre_flight_checks_pass
            assert not b._battery_landings
    now=base+5_000_000_000
    with patch('time.monotonic_ns',return_value=now),patch('time.monotonic',return_value=now/1e9):
        c._last_px4_message_at=c._last_landed_message_at=now/1e9
        state,_=output(b,snapshot(now),c)
        c._try_battery_terminal(now/1e9)
    assert 'm' in b._battery_landings and not b._battery_landings['m']['failed']
    assert c._terminal_started_at is not None
    assert b._contract.command.command==3
    assert b._last_setpoint_tx_ns < now  # Fresh terminal brake is still required.


@pytest.mark.parametrize('ms,failed',[(1999,False),(2000,False),(2001,True)])
def test_home_evidence_before_watchdog_boundary(ms,failed):
    s=AltitudeReferenceSynchronizer(); s.confirmation_ms=2000
    meta=dict(frozen_altitude_amsl_m=40.,frozen_z_ned_m=0.,current_altitude_amsl_m=40.,
        current_z_ned_m=0.,correction_valid=True,correction_revision=0,opposition_error_m=0.,
        estimator_reset_counter_valid=True,estimator_reset_counter=0,detail='locked')
    s.set_home_reference(**meta); base=10_000_000_000
    def sample(boot,ns,raw):
        s.add_local(time_boot_ms=boot,z_ned_m=0.,received_ns=ns)
        return s.add_global(time_boot_ms=boot,fc_altitude_home_relative_m=raw,
            global_altitude_amsl_m=40.,received_ns=ns,now_ns=ns)
    for i in range(41): sample(1000+i*50,base+i*50_000_000,0.)
    start=base+2_050_000_000
    assert sample(3050,start,1.).home_correction_state==1
    for elapsed in range(50,ms,50): sample(3050+elapsed,start+elapsed*10**6,1.)
    now=start+ms*10**6
    with patch('time.monotonic_ns',return_value=now):
        s.set_home_reference(**{**meta,'current_altitude_amsl_m':39.,'current_z_ned_m':1.,'correction_revision':1})
        result=sample(3050+ms,now,1.)
    assert result.altitude_epoch_failure_latched==failed
    assert result.home_correction_state==(3 if failed else 2)
    assert s._frozen_home_z_ned_m==0.


def test_home_policy_only_approved_v6(bridge):
    b=bridge; c=setup_v6(b)
    assert policy.home_confirmation_ms(b,b._intent)==2000
    b._scenario_registry.approval(NS(mission_id='m',approved=False))
    assert policy.home_confirmation_ms(b,b._intent)==400


@pytest.mark.parametrize('confirmation_ms,failed',[(400,True),(2000,False)])
def test_flight58_observed_home_metadata_order(confirmation_ms,failed):
    data=json.loads((Path(__file__).resolve().parents[4]/'tests/fixtures/home_flight_58.json').read_text())
    pending,late=data['observed_transitions'][:2]
    start=pending['observed_monotonic_ns']; received=late['published_monotonic_ns']
    assert 400_000_000 < received-start < 2_000_000_000
    s=AltitudeReferenceSynchronizer(); s.confirmation_ms=confirmation_ms
    frozen=pending['frozen_home_altitude_amsl_m']; before=pending['current_home_altitude_amsl_m']; after=late['current_home_altitude_amsl_m']
    meta=dict(frozen_altitude_amsl_m=frozen,frozen_z_ned_m=0.,current_altitude_amsl_m=before,
        current_z_ned_m=frozen-before,correction_valid=True,correction_revision=1,opposition_error_m=0.,
        estimator_reset_counter_valid=True,estimator_reset_counter=0,detail='locked')
    def sample(ns,home):
        boot=1000+round((ns-(start-3_000_000_000))/1e6)
        s.add_local(time_boot_ms=boot,z_ned_m=-1.,received_ns=ns)
        return s.add_global(time_boot_ms=boot,fc_altitude_home_relative_m=frozen+1-home,
            global_altitude_amsl_m=frozen+1,received_ns=ns,now_ns=ns)
    with patch('time.monotonic_ns',return_value=start-3_000_000_000): assert s.set_home_reference(**meta)
    for i in range(60): sample(start-3_000_000_000+i*50_000_000,before)
    assert sample(start,after).home_correction_state==1
    for ns in range(start+50_000_000,received,50_000_000): sample(ns,after)
    with patch('time.monotonic_ns',return_value=received):
        s.set_home_reference(**{**meta,'current_altitude_amsl_m':after,'current_z_ned_m':frozen-after,'correction_revision':2})
        result=sample(received,after)
    assert result.altitude_epoch_failure_latched==failed
    assert result.home_correction_state==(3 if failed else 2)
    assert s._frozen_home_z_ned_m==0. and s._frozen_home_altitude_amsl_m==frozen
    if not failed: assert result.normalized_fc_altitude_home_relative_m==pytest.approx(1.)


def test_home_request_immediate_200ms_then_default_1s():
    from test_vehicle_command_transport import memory_transport
    t=memory_transport(); t.last_home_request_ns=0
    assert t.request_home_position(1_000_000_000)
    assert t.request_home_position(1_010_000_000,force=True,period_ns=200_000_000)
    assert not t.request_home_position(1_209_000_000,period_ns=200_000_000)
    assert t.request_home_position(1_210_000_000,period_ns=200_000_000)
    assert not t.request_home_position(2_209_000_000)
    assert t.request_home_position(2_210_000_000)


@pytest.mark.parametrize('fault',['rc','epoch','position','health','stale'])
def test_grace_never_hides_control_or_navigation_loss(bridge,fault):
    b=bridge; setup_v6(b); ns=time.monotonic_ns(); snap=snapshot(ns)
    from dataclasses import replace
    if fault=='rc': snap=replace(snap,offboard=False)
    elif fault=='epoch': snap=replace(snap,transport_epoch=2)
    elif fault=='position': snap=replace(snap,position_valid=False,battery_navigation_healthy=False,battery_health_terminal_only=False)
    elif fault=='health': snap=replace(snap,battery_navigation_healthy=False,battery_health_terminal_only=False)
    elif fault=='stale': snap=replace(snap,battery_sample={**snap.battery_sample,'received_ns':ns-751_000_000})
    b._observe_battery_health(snap,ns)
    assert not b._battery_navigation_allowed


@pytest.mark.parametrize('intent_lost',[True,False])
def test_pair_timeout_uses_last_committed_source_with_new_terminal_contract(bridge,intent_lost):
    from test_battery_landing import brake_pair
    from test_vehicle_command_transport import fc_snapshot
    b=bridge; c=setup_v6(b); base=time.monotonic_ns()
    old=copy.deepcopy(c._output_contract)
    cmd=copy.deepcopy(c._active_command); cmd.sequence+=1; cmd.command=0
    c._active_command=cmd
    with patch('time.monotonic_ns',return_value=base):
        if not intent_lost: b._on_offboard_command(cmd)
        c._contract_publisher=NS(publish=b._on_output_contract if intent_lost else lambda _:None)
        c._activate_output_contract(cmd)
    now=base+500_000_000
    snap=fc_snapshot(armed=True,offboard=True,landed=False)
    with patch('time.monotonic_ns',return_value=now),patch('time.monotonic',return_value=now/1e9):
        b._stream_setpoint(snap,now)
        assert b._last_output_detail==policy.PAIR_TIMEOUT
        c._last_px4_message_at=c._last_landed_message_at=now/1e9
        c._contract_publisher=NS(publish=b._on_output_contract)
        output(b,snap,c)
        assert c._terminal_started_at is not None
        assert b._contract.command.sequence==old.command.sequence
        assert b._contract.output_sequence > old.output_sequence
        assert b._contract_valid()
        brake_pair(b,now); b._stream_setpoint(snap,now)
        assert b._last_tx_terminal_brake and b._last_setpoint_tx_ns==now


@pytest.mark.parametrize('state',[1,2,3,4,5,6])
@pytest.mark.parametrize('command',[176,400])
@pytest.mark.parametrize('pair_pending',[False,True])
def test_mode_arm_cannot_use_low_grace(bridge,state,command,pair_pending):
    from test_vehicle_command_transport import fc_snapshot
    b=bridge; c=setup_v6(b); now=time.monotonic_ns()
    sample=dict(valid=True,received_ns=now,charge_state=state,fault_bitmask=0)
    snap=fc_snapshot(armed=False,offboard=command==400,battery_sample=sample)
    # Other write gates independently have production regression coverage.
    # Satisfy them here so this assertion isolates the severity gate.
    b._stream_setpoint=lambda *a,**k:True
    b._last_envelope_valid=True; b._last_setpoint_tx_ns=now
    b._tx_run_started_ns=now-1_000_000_000
    b._altitude_reference_sync.snapshot=lambda _:NS(state=1,transport_epoch=1)
    if pair_pending: b._scenario_pair_started_ns=now-10_000_000
    with patch('time.monotonic_ns',return_value=now):
        c._send_vehicle_command(command,param1=1.,param2=6. if command==176 else 0.)
        b._drain_command_queue(snap,now)
    assert bool(b._transport.packets)==(state==1 and not pair_pending)


@pytest.mark.parametrize('order',['intent','contract'])
def test_real_dds_pair_join(bridge,order):
    import uuid
    from rclpy.executors import SingleThreadedExecutor
    from jolgwa_interfaces.msg import OffboardCommand, FlightControlContract
    b=bridge; c=setup_v6(b); old=b._contract.command.sequence
    for timer in b.timers: timer.cancel()  # This test spins only the two DDS callbacks.
    cmd=copy.deepcopy(c._active_command); cmd.command=0; cmd.sequence+=1
    contract=copy.deepcopy(c._output_contract); contract.command=cmd; contract.output_sequence+=1
    suffix=uuid.uuid4().hex; seen=[]
    callbacks={'intent':(OffboardCommand,b._on_offboard_command,cmd),
               'contract':(FlightControlContract,b._on_output_contract,contract)}
    pubs={}
    for name,(typ,callback,_) in callbacks.items():
        topic='/batteryhome_pair_'+name+'_'+suffix
        b.create_subscription(typ,topic,lambda m,cb=callback,n=name:(cb(m),seen.append(n)),10)
        pubs[name]=b.create_publisher(typ,topic,10)
    ex=SingleThreadedExecutor(); ex.add_node(b)
    try:
        for name in (order,'contract' if order=='intent' else 'intent'):
            deadline=time.monotonic()+2.
            while pubs[name].get_subscription_count()==0 and time.monotonic()<deadline: ex.spin_once(timeout_sec=.01)
            assert pubs[name].get_subscription_count()>0
            pubs[name].publish(callbacks[name][2])
            while name not in seen and time.monotonic()<deadline: ex.spin_once(timeout_sec=.01)
            assert name in seen
            assert b._contract_valid()
            if len(seen)==1:
                assert b._contract.command.sequence==old
                now=time.monotonic_ns(); assert not b._observe_battery_health(snapshot(now),now)
                assert not b._battery_landings
        assert b._contract.command.sequence==cmd.sequence
        assert not b._battery_landings
    finally:
        ex.remove_node(b); ex.shutdown()

import copy
import json
import threading
import time
import uuid
from types import SimpleNamespace as NS
import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from px4_msgs.msg import VehicleLocalPosition
from jolgwa_interfaces.msg import AltitudeReferenceState, MissionApproval, OffboardCommand, FlightOutputState
from jolgwa_ros.px4_offboard_controller import Px4OffboardController as Controller
from jolgwa_ros.scenario_contract import ScenarioRegistry
from jolgwa_ros.scenario_yaw_reset import observe_reset_metadata
from test_scenario import plan
from test_safety_contract import bridge
from test_front_climb import begin_climb, evidence


def configured(monkeypatch):
    monkeypatch.setattr(Controller,'_heartbeat_loop',lambda self:None)
    c=Controller()
    c._scenario_enabled=True; c._scenario_registry=ScenarioRegistry()
    c._scenario_registry.proposal(NS(proposal_id='p',plan_json=json.dumps(plan())))
    approval=MissionApproval(mission_id='m',proposal_id='p',approved=True)
    c._scenario_registry.approval(approval); c._on_approval(approval)
    c._active_command=OffboardCommand(mission_id='m',approved=True,command=2,
        position_ned_m=[0.,0.,-3.],home_z_ned_m=0.,altitude_reference_epoch=1,flight_profile=2)
    c._gate.armed=True; c._gate.connected=True
    c._offboard=True; c._px4_failsafe_active=False
    c._navigation_reset_counters=(12,)*5
    c._route_heading_gate_enabled=False
    sample=AltitudeReferenceState(valid=True,stable=True,state=1,transport_epoch=1,sequence=10,
        estimator_reset_counter_valid=True,estimator_reset_counter=12,
        execution_home_lock_valid=True,execution_home_mission_id='m',home_phase=1)
    c._on_altitude_reference_state(sample)
    return c,sample


def local(heading=13, other=12, stamp=1100000):
    msg=VehicleLocalPosition(timestamp=stamp,timestamp_sample=stamp,x=0.,y=0.,z=-1.5,
        vx=0.,vy=0.,vz=0.,xy_valid=True,z_valid=True,v_xy_valid=True,v_z_valid=True,heading=.164,
        xy_reset_counter=other,z_reset_counter=other,vxy_reset_counter=other,vz_reset_counter=other,
        heading_reset_counter=heading)
    return msg


def test_late_metadata_invalidates_old_clear_without_extending_ascent():
    seq=begin_climb();deadline=seq.phase_started+seq.spec['stage_timeout_s']
    pos=(1.,0.,-1.8)
    seq.tick(pos,(0.,)*3,evidence(101.),101.)
    seq.tick(pos,(0.,)*3,evidence(101.1),101.1)
    assert seq.tick(pos,(0.,)*3,evidence(101.7),101.7)[0]=='PASS'
    seq.invalidate_pre_reset_clear(101.8,pos)
    assert seq.phase=='CLIMB' and seq.target==pos and seq.climb_clear_hold
    assert seq.phase_started+seq.spec['stage_timeout_s']==deadline
    seq.tick(pos,(0.,)*3,None,101.81)
    assert seq.depth_waiting
    seq.tick(pos,(0.,)*3,evidence(102.),102.)
    assert seq.tick(pos,(0.,)*3,evidence(102.6),102.6)[0]=='PASS'
    assert seq.target==(4.,0.,-1.8)


def test_same_altitude_sequence_reset_notice_never_extends_depth_barrier():
    previous=AltitudeReferenceState(transport_epoch=1,sequence=10,estimator_reset_counter_valid=True,estimator_reset_counter=12)
    incoming=copy.deepcopy(previous);incoming.estimator_reset_counter=13
    incoming.execution_home_lock_valid=True;incoming.execution_home_mission_id='m'
    manager=NS(_scenario_enabled=True,_active_mission_id='m',_scenario_inputs=NS(evidence={'state':0}))
    observe_reset_metadata(manager,previous,incoming)
    started=manager._scenario_yaw_reset_at
    assert manager._scenario_inputs.evidence is None
    observe_reset_metadata(manager,previous,incoming)
    assert manager._scenario_yaw_reset_at==started
    incoming.sequence=9
    observe_reset_metadata(manager,previous,incoming)
    assert manager._scenario_yaw_reset_at==started


@pytest.mark.parametrize('fault',['none','position','multi_heading','version','build','epoch','pending','stale','rc','external','revoked'])
def test_controller_only_accepts_approved_heading_generation(bridge,monkeypatch,fault):
    c,sample=configured(monkeypatch)
    try:
        msg=local()
        if fault=='position':msg.xy_reset_counter=13
        if fault=='multi_heading':msg.heading_reset_counter=14
        if fault in ('version','build'):
            p=plan();p['test_scenario']['version' if fault=='version' else 'build_id']=4 if fault=='version' else 'old'
            c._scenario_registry.proposal(NS(proposal_id='p',plan_json=json.dumps(p)))
        if fault=='epoch':sample.transport_epoch=2
        if fault=='pending':sample.home_phase=2
        if fault=='stale':c._altitude_reference_state_received_at-=.501
        if fault=='rc':c._gate.manual_override=True
        if fault=='external':c._external_mode_fenced=True
        if fault=='revoked':c._scenario_registry.approval(MissionApproval(mission_id='m',proposal_id='p',approved=False))
        c._on_local_position(msg)
        assert c._autonomy_reentry_required is (fault!='none')
        if fault=='none':
            assert c._active_command.yaw_rad==0.
            assert list(c._active_command.position_ned_m)==[0.,0.,-3.]
            assert c._gate.is_approved('m')
            assert c._jetson_observation_at==float('-inf')
            # A delayed prior-generation position cannot trigger a second reset.
            c._on_local_position(local(12,stamp=1099999))
            assert c._navigation_reset_counters==(12,12,12,12,13)
            assert not c._autonomy_reentry_required
            c._usb_output_contract=True
            c._flight_output_state=FlightOutputState(last_tx_monotonic_ns=int((c._scenario_yaw_reset_at-.01)*1e9))
            assert not c._flight_output_ready(c._active_command,time.monotonic())
    finally:
        c.destroy_node()


@pytest.mark.parametrize('order',['metadata_first','position_first'])
def test_isolated_dds_reset_metadata_and_position_reordering(bridge,monkeypatch,order):
    c,old=configured(monkeypatch)
    wire=rclpy.create_node('yaw_reset_order_'+uuid.uuid4().hex)
    executor=MultiThreadedExecutor(num_threads=2); executor.add_node(wire)
    event=threading.Event(); mid=uuid.uuid4().hex
    manager=NS(_scenario_enabled=True,_active_mission_id='m',
               _scenario_inputs=NS(evidence={'state':0}),previous=old)
    def metadata(msg):
        observe_reset_metadata(manager,manager.previous,msg);manager.previous=msg
        c._on_altitude_reference_state(msg);event.set()
    wire.create_subscription(AltitudeReferenceState,'/yaw_test/metadata_'+mid,metadata,10)
    wire.create_subscription(VehicleLocalPosition,'/yaw_test/position_'+mid,lambda msg:(c._on_local_position(msg),event.set()),10)
    pp=wire.create_publisher(VehicleLocalPosition,'/yaw_test/position_'+mid,10)
    ap=wire.create_publisher(AltitudeReferenceState,'/yaw_test/metadata_'+mid,10)
    worker=threading.Thread(target=executor.spin,daemon=True);worker.start()
    try:
        for pub in (pp,ap):
            deadline=time.monotonic()+3.
            while pub.get_subscription_count()==0 and time.monotonic()<deadline: time.sleep(.01)
            assert pub.get_subscription_count()>0
        new=copy.deepcopy(old);new.sequence=11;new.estimator_reset_counter=13
        pairs=[(ap,new),(pp,local())]
        if order=='position_first':pairs.reverse()
        for pub,msg in pairs:
            event.clear();pub.publish(msg);assert event.wait(2.)
        assert not c._autonomy_reentry_required
        assert manager._scenario_inputs.evidence is None
        assert c._navigation_reset_counters==(12,12,12,12,13)
        # Old metadata is fenced by sequence; old local position by source stamp.
        event.clear();ap.publish(old);assert event.wait(2.)
        event.clear();pp.publish(local(12,stamp=1000000));assert event.wait(2.)
        assert c._altitude_reference_state.sequence==11
        assert not c._autonomy_reentry_required
    finally:
        executor.shutdown(timeout_sec=2.);worker.join(2.);wire.destroy_node();c.destroy_node()

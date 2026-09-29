"""Regression expectations for the corrected scenario2 demo.

Assertions require corrected behavior.
No hardware, network flight output, or product edits.
"""
import json
import threading
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import pytest
from jolgwa_ros.scenario_contract import make_spec, point, ScenarioRegistry
from jolgwa_ros.scenario_sequence import ScenarioSequence
from jolgwa_ros.scenario_runtime import Inputs
from jolgwa_uav.depth_geometry import (
    DepthGeometryTracker, DepthIntrinsics, CameraExtrinsics, VehiclePose)
from jolgwa_uav.vertical_avoidance import VerticalObstacleAvoidanceCore, VerticalAvoidanceConfig


def spec(): return make_spec((0.,0.,0.),0.,0.,1,1.)
def evidence(now,**kw):
    data=dict(observed=now,sequence=1,state=0,geometry_valid=True,
        far=(2.,0.,0.),roof_clear=True,roof_gap=1.1,passed=True,descent_clear=True)
    data.update(kw)
    return data


def test_R1_real_geometry_and_policy_first_obstacle_holds_for_boundary():
    depth=np.full((48,64),2.5,dtype=np.float32)
    observation=DepthGeometryTracker().observe(depth,
        intrinsics=DepthIntrinsics(64,48,60.,60.,31.5,23.5),
        extrinsics=CameraExtrinsics.from_mount_quaternion((1.,0.,0.,0.),(0.,0.,0.)),
        pose=VehiclePose((0.,0.,-1.),(1.,0.,0.,0.),100.),
        depth_timestamp_s=100.,now_s=100.,route_direction_ned=(1.,0.,0.),
        avoidance_active=False,nominal_altitude_m=1.)
    geometry=observation.as_payload()
    policy=VerticalObstacleAvoidanceCore(VerticalAvoidanceConfig(require_geometry=True))
    decision=policy.evaluate(depth,frame_age_s=0.,current_altitude_m=1.,geometry=geometry)
    assert geometry['geometry_valid'] and not geometry['obstacle_extent_valid']
    assert decision.state.value in ('HOLD','EVADE')
    state={'HOLD':2,'EVADE':3}[decision.state.value]
    sequence=ScenarioSequence(spec(),100.)
    phase,_=sequence.tick((0.,0.,-1.),(0.,0.,0.),evidence(100.,state=state,far=None),100.)
    print('R1',decision.state.value,geometry['obstacle_extent_valid'],phase,sequence.reason)
    assert phase=='BRAKE' and not sequence.reason


@pytest.mark.parametrize('far',[8.,10.])
def test_descent_boundary_is_diagnostic_and_cannot_change_fixed_pass(far):
    sequence=ScenarioSequence(spec(),100.)
    sequence.pass_progress=7.
    sequence.transition('DESCEND',point(spec(),7.),100.)
    phase,target=sequence.tick((7.,0.,-1.8),(0.,0.,.2),
        evidence(100.1,far=(far,0.,0.)),100.1)
    assert phase=='DESCEND' and target==(7.,0.,-1.)


def test_R3_photo_already_late_when_polled_cannot_resume_route():
    sequence=ScenarioSequence(spec(),100.)
    sequence.transition('PHOTO',point(spec(),7.),100.)
    phase,_=sequence.tick(sequence.target,(0.,0.,0.),evidence(110.001),110.001,photo='saved')
    print('R3 photo at 10.001s:',phase)
    assert phase=='FAILED' and sequence.reason=='photo_save_timeout'


def test_R3_event_after_20_seconds_is_rejected():
    sequence=ScenarioSequence(spec(),100.)
    sequence.event_opened=100.
    sequence.transition('WAIT_EVENT',point(spec(),10.),100.)
    phase,_=sequence.tick(sequence.target,(0.,0.,0.),evidence(120.001),120.001,
        event=dict(type='LITTERING',input_received=120.001,event_id='late'))
    print('R3 incident at 20.001s:',phase)
    assert phase=='FAILED' and sequence.reason=='littering_not_detected'


def test_R4_confirmed_incident_with_capture_error_retained_for_stop_and_failure():
    node=NS(_lock=threading.RLock(),_active_goal=True,_active_mission_id='current')
    inputs=Inputs(node)
    item=dict(clock_id='clock',event_type='LITTERING',original_model_event={'state':'CONFIRMED'},
        capture_status='ERROR',model_sha256='cddc20536cd76db52e340746a2c807e5f53cf1fe2a60edeace53973029256935',
        event_id='a77ba582-de6e-4494-bacd-b4ef1a970b91',camera_received_monotonic_s=100.,error='disk full')
    with patch('jolgwa_ros.evidence_clock.local_evidence_clock_id',return_value='clock'),patch('time.monotonic',return_value=100.):
        inputs.incident(NS(data=json.dumps(item)))
    print('R4 confirmed capture error, manager event:',inputs.event)
    assert inputs.event['capture_status']=='ERROR'
    inputs.pool.shutdown()


@pytest.mark.parametrize('elapsed,expired',[(4.999,False),(5.,True),(5.001,True)])
def test_depth_recovery_fixed_deadline(elapsed,expired):
    s=ScenarioSequence(spec(),100.)
    assert s.tick((0.,0.,-1.),(0.,)*3,None,100.)[0]=='SEARCH_OBSTACLE'
    assert s.depth_waiting and s.target==(0.,0.,-1.)
    phase,target=s.tick((0.,0.,-1.),(0.,)*3,evidence(100.+elapsed),100.+elapsed)
    assert (phase=='FAILED')==expired
    if not expired: assert target==(10.,0.,-1.) and not s.depth_waiting


@pytest.mark.parametrize('elapsed,expired',[(59.999,False),(60.,True),(60.001,True)])
def test_brake_uses_stage_deadline_and_never_requires_extent(elapsed,expired):
    s=ScenarioSequence(spec(),100.)
    s.tick((0.,0.,-1.),(0.,)*3,evidence(100.,state=2,far=None),100.)
    phase,_=s.tick((0.,0.,-1.),(0.,)*3,evidence(100.+elapsed,state=3,far=None),100.+elapsed)
    assert (phase=='FAILED')==expired


@pytest.mark.parametrize('phase,limit,reason',[
    ('PHOTO',10.,'photo_save_timeout'),('WAIT_EVENT',20.,'littering_not_detected'),
    ('REJOIN',60.,'scenario_stage_timeout:REJOIN')])
@pytest.mark.parametrize('delta',[-.001,0.,.001])
def test_deadlines_precede_new_evidence(phase,limit,reason,delta):
    s=ScenarioSequence(spec(),100.); s.transition(phase,point(spec(),7.),100.)
    s.event_opened=100.
    now=100.+limit+delta
    result,_=s.tick(s.target,(0.,)*3,evidence(now),now,photo='saved',
        event=dict(type='LITTERING',input_received=now))
    if delta>=0: assert result=='FAILED' and s.reason==reason
    else: assert result!='FAILED'


def pending_owner():
    from jolgwa_ros.scenario_pending import defer
    registry=ScenarioRegistry(); delivered=[]
    n=NS(_scenario_enabled=True,_scenario_registry=registry,_active_command=None,
         _gate=NS(manual_override=False,is_approved=lambda _:True),_last_error='')
    n._on_command=delivered.append
    command=NS(command=1,mission_id='m',approved=True,altitude_reference_epoch=1)
    assert defer(n,command,100.)
    return n,command,delivered


def add_metadata(n):
    from test_scenario import plan
    n._scenario_registry.proposal(NS(proposal_id='p',plan_json=json.dumps(plan())))
    n._scenario_registry.approval(NS(proposal_id='p',mission_id='m',approved=True))


@pytest.mark.parametrize('delay,allowed',[(1.999,True),(2.,False),(2.001,False)])
def test_R5_metadata_order_buffer_has_fixed_deadline(delay,allowed):
    from jolgwa_ros.scenario_pending import defer,drain
    n,c,out=pending_owner()
    assert defer(n,c,101.)
    assert n._scenario_pending_takeoff[1]==102.
    add_metadata(n); drain(n,100.+delay)
    assert bool(out)==allowed
    drain(n,104.)
    assert len(out)==int(allowed)


@pytest.mark.parametrize('cause',['rc','epoch','revoke','cancel','replacement'])
def test_pending_takeoff_cannot_survive_authority_or_identity_change(cause):
    from jolgwa_ros.scenario_pending import defer,drain
    n,c,out=pending_owner(); add_metadata(n)
    if cause=='rc': n._external_mode_fenced=True
    if cause=='epoch': n._altitude_reference_state=NS(transport_epoch=2)
    if cause=='revoke': n._scenario_registry.approval(NS(mission_id='m',approved=False))
    if cause=='cancel': defer(n,NS(command=3,mission_id='m'),101.)
    if cause=='replacement': defer(n,NS(**{**vars(c),'altitude_reference_epoch':2}),101.)
    drain(n,101.)
    assert not out


def test_previous_mission_terminal_does_not_cancel_current_pending_takeoff():
    from jolgwa_ros.scenario_pending import defer,drain
    n,c,out=pending_owner()
    assert not defer(n,NS(command=3,mission_id='old'),101.)
    add_metadata(n); drain(n,101.)
    assert out==[c]


def test_ready_metadata_uses_original_start_validation_without_new_interlocks():
    from jolgwa_ros.scenario_pending import defer
    n,c,_=pending_owner(); n._scenario_pending_takeoff=None
    add_metadata(n); n._autonomy_resume_required=True
    assert not defer(n,c,101.)  # Controller still checks the original handoff gates.


def test_revocation_supersedes_delayed_positive_approval():
    from jolgwa_ros.approval_execution import ApprovalExecutionLedger,ApprovalContractError
    ledger=ApprovalExecutionLedger(clock=lambda:100.)
    ledger.invalidate(connected=True); ledger.remember('p','{}',executable=True)
    positive=ledger.begin('p',True); negative=ledger.begin('p',False)
    with pytest.raises(ApprovalContractError): ledger.finish(positive,accepted=True,mission_id='m')
    assert ledger.finish(negative,accepted=True,mission_id='m') is None
    with pytest.raises(ApprovalContractError): ledger.begin('p',True)


def test_demo_producer_can_release_without_roof_certificate_but_ordinary_cannot():
    from dataclasses import replace
    config=VerticalAvoidanceConfig(require_geometry=True,release_samples=1,
        min_evade_climb_m=0.,post_clear_climb_m=0.,scenario_demo=True)
    depth=np.full((48,64),8.,dtype=np.float32)
    geometry=dict(geometry_valid=True,roof_clearance_verified=False,
        roof_passage_verified=False,roof_vertical_gap_m=None)
    for demo in (True,False):
        core=VerticalObstacleAvoidanceCore(replace(config,scenario_demo=demo))
        core._active=core._roof_guard_active=True
        decision=core.evaluate(depth,frame_age_s=0.,current_altitude_m=2.,geometry=geometry)
        assert (decision.state.value=='CLEAR')==demo
        assert not geometry['roof_clearance_verified']
        if demo: assert not core._roof_guard_active



def test_pass_and_descent_keep_existing_arrival_tolerance_without_extra_precision_gate():
    s=ScenarioSequence(spec(),100.); s.pass_progress=3.
    s.transition('PASS',point(s.spec,3.,z=-2.),100.)
    position=(2.9,0.,-2.)
    s.tick(position,(0.,)*3,evidence(100.),100.)
    assert s.tick(position,(0.,)*3,evidence(100.5),100.5)[0]=='DESCEND'
    assert s.tick(position,(0.,)*3,evidence(100.6),100.6)[0]=='DESCEND'


def test_diagnostic_boundary_updates_do_not_restart_descent_deadline():
    s=ScenarioSequence(spec(),100.)
    s.transition('DESCEND',point(s.spec,7.),100.)
    assert s.tick((7.,0.,-1.8),(0.,)*3,evidence(101.,far=(8.,0.,0.)),101.)[0]=='DESCEND'
    assert s.phase_started == 100.
    assert s.tick((7.,0.,-1.8),(0.,)*3,evidence(160.,far=(8.,0.,0.)),160.)[0]=='FAILED'
    assert s.reason=='scenario_stage_timeout:DESCEND'


@pytest.mark.parametrize('elapsed,expired',[(239.999,False),(240.,True),(240.001,True)])
def test_total_deadline_is_not_extended_by_fresh_input(elapsed,expired):
    s=ScenarioSequence(spec(),100.)
    result,_=s.tick((0.,0.,-1.),(0.,)*3,evidence(100.+elapsed),100.+elapsed)
    assert (result=='FAILED')==expired

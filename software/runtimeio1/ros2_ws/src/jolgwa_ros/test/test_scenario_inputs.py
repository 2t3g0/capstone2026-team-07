import json
import math
import threading
from types import SimpleNamespace as NS
from unittest.mock import patch
import pytest
from jolgwa_ros.scenario_runtime import Inputs
from jolgwa_ros.scenario_sequence import ScenarioSequence
from jolgwa_ros.scenario_incident_node import ScenarioIncidentNode
from test_scenario import plan

HASH='cddc20536cd76db52e340746a2c807e5f53cf1fe2a60edeace53973029256935'


def owner():
    return NS(_lock=threading.RLock(),_active_goal=True,_active_mission_id='current',
              get_clock=lambda:NS(now=lambda:NS(nanoseconds=100_000_000_000)))


@pytest.mark.parametrize('defect',['old','future','clock','wrong_model','person','candidate','saved_error','inactive','good'])
def test_only_current_confirmed_v2_incident_is_bound_to_active_mission(defect):
    node=owner(); inputs=Inputs(node)
    event=dict(event_id='07e6c9b9-230c-402c-9229-82898088dfe6',clock_id='clock',event_type='LITTERING',
               capture_status='SAVED',model_sha256=HASH,camera_received_monotonic_s=99.5,
               original_model_event={'state':'CONFIRMED'})
    if defect=='old': event['camera_received_monotonic_s']=94.999
    if defect=='future': event['camera_received_monotonic_s']=100.01
    if defect=='clock': event['clock_id']='other_host'
    if defect=='wrong_model': event['model_sha256']='other'
    if defect=='person': event['event_type']='person'
    if defect=='candidate': event['original_model_event']['state']='CANDIDATE'
    if defect=='saved_error': event['capture_status']='ERROR'
    if defect=='inactive': node._active_goal=False
    with patch('time.monotonic',return_value=100.),patch('jolgwa_ros.evidence_clock.local_evidence_clock_id',return_value='clock'):
        inputs.incident(NS(data=json.dumps(event)))
        if defect in ('good','saved_error'):
            assert inputs.event['mission_id']=='current'
            node._active_mission_id='next'
            inputs.incident(NS(data=json.dumps(event)))
            assert inputs.event['mission_id']=='current' # Duplicate never crosses mission boundary.
        else: assert inputs.event is None
    inputs.pool.shutdown()


def test_perception_source_age_plus_transport_delay_not_rejuvenated():
    node=owner(); inputs=Inputs(node)
    sample=NS(source='jetson-realsense-d435i',sequence=1,observation_age_s=.3,
        stamp=NS(sec=99,nanosec=700_000_000),obstacle_extent_valid=False,
        geometry_valid=False,state=0,roof_clearance_verified=False,
        roof_vertical_gap_m=math.nan,roof_passage_verified=False,descent_corridor_clear=False)
    with patch('time.monotonic',return_value=100.): inputs.safety(sample)
    assert 100.-inputs.evidence['observed']==pytest.approx(.65)
    seq=ScenarioSequence(plan()['test_scenario'],100.)
    assert seq.tick((0,0,-1),(0,0,0),inputs.evidence,100.)[0]=='SEARCH_OBSTACLE'
    assert seq.depth_waiting
    assert seq.tick((0,0,-1),(0,0,0),inputs.evidence,105.)[0]=='FAILED'
    sample.observation_age_s=0; sample.stamp=NS(sec=100,nanosec=0)
    with patch('time.monotonic',return_value=100.): inputs.safety(sample)
    assert 100.-inputs.evidence['observed']==pytest.approx(.65) # Same sequence ignored.
    inputs.pool.shutdown()


def test_incident_bridge_uses_actual_loaded_model_hash_and_deduplicates():
    published=[]
    node=NS(bench=NS(lock=threading.RLock(),events=[dict(event_id='event',event_type='LITTERING',
        source_timestamp_ns=99_000_000_000,camera_received_monotonic_s=99.7)],
        inference={'model_hashes':{'abnormal_behavior':HASH}}),sent=set(),
        get_clock=lambda:NS(now=lambda:NS(nanoseconds=100_000_000_000)),
        publisher=NS(publish=lambda m:published.append(json.loads(m.data))))
    with patch('time.monotonic',return_value=100.):
        ScenarioIncidentNode.publish_new(node); ScenarioIncidentNode.publish_new(node)
    assert len(published)==1
    assert published[0]['model_sha256']==HASH
    assert published[0]['camera_received_monotonic_s']==99.


@pytest.mark.parametrize('age',[.1,.5,.501,-.01])
def test_incident_camera_callback_discards_delayed_frames(age):
    received=[]
    node=NS(last_observe_rgb_at=float('-inf'), observation=NS(rgb=lambda *args:None),
            get_clock=lambda:NS(now=lambda:NS(nanoseconds=100_000_000_000)),
            bench=NS(submit=lambda *args:received.append(args)))
    stamp=int((100-age)*1e9)
    message=NS(header=NS(stamp=NS(sec=stamp//10**9,nanosec=stamp%10**9)),data=b'jpeg')
    ScenarioIncidentNode.receive(node,message)
    assert bool(received)==(0 <= age <= .5)


@pytest.mark.parametrize('version',['current','missing','previous_build'])
def test_gateway_cannot_approve_old_manager_plan_in_scenario_launch(version):
    from jolgwa_ros.operator_gateway_node import OperatorGatewayNode
    from jolgwa_interfaces.msg import MissionProposal
    value=plan(); value['status']='OK'
    if version=='missing': value.pop('test_scenario')
    if version=='previous_build': value['test_scenario']['build_id']='old'
    received=[]; remembered=[]
    node=NS(_scenario_enabled=True,_enforce_integrated_event_policy=False,
        _approval_ledger=NS(remember=lambda *a,**kw:remembered.append(kw)),
        _send=received.append,_send_error=lambda *a:None)
    message=MissionProposal(proposal_id='p',plan_json=json.dumps(value),
        status=MissionProposal.STATUS_OK,requires_approval=True)
    OperatorGatewayNode._on_proposal(node,message)
    assert received[-1]['requires_approval']==(version=='current')
    assert remembered[-1]['executable']==(version=='current')

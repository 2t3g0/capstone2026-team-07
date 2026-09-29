import copy
import json
import math
from types import SimpleNamespace as NS
import pytest
from jolgwa_ros.scenario_contract import (make_spec, point, validate_spec,
    ScenarioRegistry, within, ceiling)
from jolgwa_ros.scenario_sequence import ScenarioSequence
from jolgwa_ros.scenario_runtime import navigation_error, Inputs


def plan(height=1.):
    spec = make_spec([0., 0., 0.], 0., 0., 1, height)
    return dict(test_scenario=spec, mission_kind="FORWARD_TEST_1M",
        preview_start_ned_m=[0.,0.,0.], preview_end_ned_m=[10.,0.,0.],
        preview_heading_rad=0., low_speed_limits={"target_altitude_home_m":height})


def command(height=1.):
    return NS(mission_id="m", approved=True, command=2, position_ned_m=[10.,0.,-height],
              home_z_ned_m=0., altitude_reference_epoch=1, flight_profile=2 if height==1 else 1)


@pytest.mark.parametrize("height", [1.,2.])
def test_registry_reordered_approval_and_retirement(height):
    registry=ScenarioRegistry()
    approval=NS(mission_id="m",proposal_id="p",approved=True)
    registry.approval(approval)
    assert registry.command_spec(command(height)) is None
    registry.proposal(NS(proposal_id="p",plan_json=json.dumps(plan(height))))
    assert ceiling(registry.command_spec(command(height)))==height+2.5
    approval.approved=False; registry.approval(approval)
    approval.approved=True; registry.approval(approval)
    assert registry.command_spec(command(height)) is None


@pytest.mark.parametrize("field,value",[("search_m",11.),("climb_m",3.),("version",1),
    ("build_id","older"),("home_z_ned_m",float("nan")),("transport_epoch",0)])
def test_spec_cannot_expand_or_claim_invalid_epoch(field,value):
    p=plan(); p["test_scenario"][field]=value
    with pytest.raises(ValueError): validate_spec(p)


def test_registry_collision_and_epoch_reject():
    registry=ScenarioRegistry(); p=plan()
    registry.proposal(NS(proposal_id="p",plan_json=json.dumps(p)))
    registry.approval(NS(mission_id="m",proposal_id="p",approved=True))
    c=command(); c.altitude_reference_epoch=2
    assert registry.command_spec(c) is None
    p["test_scenario"]["home_z_ned_m"]=-1
    registry.proposal(NS(proposal_id="p",plan_json=json.dumps(p)))
    assert registry.command_spec(command()) is None


def evidence(now, state=0, seq=1, **kw):
    return dict(observed=now, sequence=seq, state=state, geometry_valid=True,
                far=(2.,0.,0.), roof_clear=True, roof_gap=1.1,
                passed=True, descent_clear=True, **kw)


@pytest.mark.parametrize("height",[1.,2.])
@pytest.mark.parametrize("fault",[None,"missing_obstacle","stale","second_obstacle","no_event","photo_failed","no_descent","climb_blocked"])
def test_sequence_realistic_motion(height, fault):
    seq=ScenarioSequence(plan(height)["test_scenario"],100.)
    pos=[0.,0.,-height]; phases=[]; now=100.
    for step in range(4000):
        now+=.05
        phases.append(seq.phase)
        vel=[max(-.5,min(.5,2*(t-p))) for t,p in zip(seq.target,pos)]
        pos=[p+v*.05 for p,v in zip(pos,vel)]
        state=3 if 1<=pos[0]<2 and pos[2]>-height-.85 and fault!="missing_obstacle" else 0
        ev=evidence(now,state,step)
        if fault=="stale" and seq.phase=="CLIMB": ev["observed"]=now-.501
        if fault=="second_obstacle" and seq.phase=="SEARCH_EVENT": ev["state"]=3
        if fault=="no_descent": ev["descent_clear"]=False
        if fault=="climb_blocked" and seq.phase=="CLIMB": ev["state"]=3
        event=None
        if seq.phase=="SEARCH_EVENT" and pos[0]>7.2 and fault!="no_event":
            event=dict(type="LITTERING",input_received=now,event_id="event")
        photo="failed" if fault=="photo_failed" else "saved"
        phase,_=seq.tick(pos,vel,ev,now,event=event,photo=photo if seq.phase=="PHOTO" else None)
        if phase in ("FAILED","LAND"): break
    if fault in (None,"no_descent"):
        assert seq.phase=="LAND",seq.reason
        assert all(p in phases for p in ("BRAKE","CLIMB","PASS","DESCEND","SEARCH_EVENT","EVENT_HOLD","PHOTO","REJOIN","FINAL"))
        assert 8<=seq.target[0]<=11
        assert seq.target[2]==-height
    else:
        assert seq.phase=="FAILED"
        assert seq.reason


def test_no_replay_or_person_event_and_exact_wait_boundary():
    seq=ScenarioSequence(plan()["test_scenario"],100.)
    seq.transition("WAIT_EVENT",point(seq.spec,10),100.)
    seq.event_opened=100.
    for event in (dict(type="person",input_received=100.), dict(type="LITTERING",input_received=99.9)):
        assert seq.tick(seq.target,[0]*3,evidence(100.1),100.1,event=event)[0]=="WAIT_EVENT"
    assert seq.tick(seq.target,[0]*3,evidence(119.999),119.999)[0]=="WAIT_EVENT"
    assert seq.tick(seq.target,[0]*3,evidence(120.),120.)[0]=="FAILED"
    assert seq.reason=="littering_not_detected"


def test_target_checks_never_block_terminal():
    node=NS(_scenario_enabled=True,_scenario_registry=ScenarioRegistry())
    c=command()
    assert navigation_error(node,c)
    c.command=3
    assert navigation_error(node,c)==""
    assert not within(plan()["test_scenario"],(11.01,0,-1),target=True)


def test_photo_saves_original_and_stopped_frame_without_overwrite(tmp_path):
    import hashlib
    import numpy as np
    raw=b"detection payload"; stopped=b"stopped payload"
    source=tmp_path/"original.jpg"; source.write_bytes(raw)
    event=dict(event_id="e",photo_path=str(source),photo_sha256=hashlib.sha256(raw).hexdigest())
    inputs=Inputs(NS())
    assert inputs.save(tmp_path/"saved","m",event,(1,101.,stopped),np.array([1,2,3],dtype=np.float32),102.)=="saved"
    target=tmp_path/"saved/m/e"
    assert (target/"detection.jpg").read_bytes()==raw
    assert (target/"stopped.jpg").read_bytes()==stopped
    assert json.loads((target/"evidence.json").read_text())["mission_id"]=="m"
    with pytest.raises(FileExistsError): inputs.save(tmp_path/"saved","m",event,(1,101.,stopped),(1,2,3),102.)
    inputs.pool.shutdown()


def test_late_photo_completion_cannot_restart_expired_stage():
    seq=ScenarioSequence(plan()['test_scenario'],100.)
    seq.transition('PHOTO',point(seq.spec,7.),100.)
    assert seq.tick(seq.target,[0]*3,evidence(110.),110.)[0]=='FAILED'
    assert seq.reason=='photo_save_timeout'
    assert seq.tick(seq.target,[0]*3,evidence(110.1),110.1,photo='saved')[0]=='FAILED'


@pytest.mark.parametrize('invalid',[None,[],{'low_speed_limits':None},'old'])
def test_malformed_plan_never_creates_envelope(invalid):
    registry=ScenarioRegistry()
    registry.proposal(NS(proposal_id='p',plan_json=json.dumps(invalid)))
    registry.approval(NS(mission_id='m',proposal_id='p',approved=True))
    assert registry.command_spec(command()) is None

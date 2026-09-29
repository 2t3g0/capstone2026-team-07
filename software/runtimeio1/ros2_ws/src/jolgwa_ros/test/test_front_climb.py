"""Approved bounded climb, actual stop +2m, and outdoor ROI dropout regressions."""
import json
import math
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import pytest
from jolgwa_ros.scenario_contract import make_spec, within, ceiling, validate_spec
from jolgwa_ros.scenario_sequence import ScenarioSequence
from jolgwa_ros.scenario_runtime import Inputs
from jolgwa_uav.vertical_avoidance import VerticalAvoidanceConfig, VerticalObstacleAvoidanceCore


def evidence(now, state=0, **kw):
    return dict(observed=now, state=state, sequence=1, geometry_valid=False, far=None, **kw)


def begin_climb(height=1., yaw=0.):
    spec=make_spec([0.,0.,0.],yaw,0.,1,height)
    seq=ScenarioSequence(spec,100.)
    seq.tick((1.,0.,-height),(0.,)*3,evidence(100.,2),100.)
    seq.tick(seq.target,(0.,)*3,evidence(100.1,3),100.1)
    assert seq.tick(seq.target,(0.,)*3,evidence(100.7,3),100.7)[0]=='CLIMB'
    return seq


@pytest.mark.parametrize('height',[1.,2.])
def test_brake_without_boundary_climbs_to_approved_target(height):
    seq=begin_climb(height)
    assert seq.target==(1.,0.,-height-2.)
    assert ceiling(seq.spec)==height+2.5
    assert within(seq.spec,(1.,0.,-height-2.4))
    assert not within(seq.spec,(1.,0.,-height-2.4),target=True)


@pytest.mark.parametrize('height',[1.,2.])
@pytest.mark.parametrize('yaw',[0.,.15])
def test_early_clear_holds_then_exact_three_meter_frozen_heading_target(height,yaw):
    seq=begin_climb(height,yaw)
    pos=(1.02,.02,-height-.8)
    phase,target=seq.tick(pos,(0.,0.,-.2),evidence(101.),101.)
    assert phase=='CLIMB' and target==pos and seq.climb_clear_hold
    seq.tick(pos,(0.,)*3,evidence(101.1),101.1)
    phase,target=seq.tick(pos,(0.,)*3,evidence(101.7),101.7)
    assert phase=='PASS'
    assert target==pytest.approx((pos[0]+3*math.cos(yaw),pos[1]+3*math.sin(yaw),pos[2]))
    assert math.dist(pos,target)==pytest.approx(3.)


def test_clear_reblocked_hold_resumes_same_climb_ceiling_and_deadline():
    seq=begin_climb(); started=seq.phase_started
    seq.tick((1.,0.,-1.7),(0.,)*3,evidence(101.),101.)
    assert seq.climb_clear_hold
    assert seq.tick((1.,0.,-1.7),(0.,)*3,evidence(101.2,2),101.2)==('CLIMB',(1.,0.,-3.))
    assert not seq.climb_clear_hold and seq.phase_started==started
    assert seq.tick((1.,0.,-1.8),(0.,)*3,evidence(started+60.),started+60.)[0]=='FAILED'


def test_clear_captured_before_climb_cannot_start_forward_leg():
    seq=begin_climb()
    phase,target=seq.tick((1.,0.,-1.4),(0.,)*3,evidence(100.69),100.8)
    assert phase=='CLIMB' and target==(1.,0.,-3.) and not seq.climb_clear_hold


def test_maximum_climb_blocked_aborts_without_another_ascent():
    seq=begin_climb()
    seq.tick(seq.target,(0.,)*3,evidence(102.,3),102.)
    assert seq.tick(seq.target,(0.,)*3,evidence(102.6,3),102.6)[0]=='FAILED'
    assert seq.reason=='maximum_climb_insufficient'


def test_clear_stop_target_does_not_consume_altitude_overshoot_margin():
    seq=begin_climb()
    phase,target=seq.tick((1.,0.,-3.1),(0.,)*3,evidence(102.),102.)
    assert phase=='CLIMB' and target[2]==-3.


@pytest.mark.parametrize('phase',['PASS','DESCEND'])
def test_reblocked_after_forward_begins_does_not_reclimb(phase):
    seq=begin_climb(); seq.transition(phase,(3.,0.,-2.),101.)
    assert seq.tick((2.5,0.,-2.),(0.,)*3,evidence(101.1,2),101.1)[0]=='FAILED'


def test_fixed_pass_target_outside_ten_meter_search_is_rejected():
    seq=begin_climb(); seq.target=(8.1,0.,-2.); seq.climb_clear_hold=True
    seq.tick(seq.target,(0.,)*3,evidence(101.),101.)
    assert seq.tick(seq.target,(0.,)*3,evidence(101.6),101.6)[0]=='FAILED'
    assert seq.reason=='obstacle_pass_outside_search'


@pytest.mark.parametrize('elapsed,expired',[(4.999,False),(5.,True),(5.001,True)])
def test_climb_depth_loss_hold_uses_original_recovery_deadline(elapsed,expired):
    seq=begin_climb()
    seq.tick((1.,0.,-1.5),(0.,)*3,None,101.)
    assert seq.depth_waiting
    phase,_=seq.tick((1.,0.,-1.5),(0.,)*3,evidence(101.+elapsed,3),101.+elapsed)
    assert (phase=='FAILED')==expired
    if expired: assert seq.reason=='scenario_depth_recovery_timeout'


@pytest.mark.parametrize('front_valid',[True,False])
@pytest.mark.parametrize('blocked',[True,False])
def test_outdoor_upper_lower_invalid_only_new_profile_uses_valid_front(front_valid,blocked):
    depth=np.zeros((48,64),dtype=np.float32)
    if front_valid: depth[17:31,:]=2. if blocked else 8.
    for new in (False,True):
        core=VerticalObstacleAvoidanceCore(VerticalAvoidanceConfig(
            scenario_demo=True,front_climb_demo=new,require_geometry=True,trigger_samples=1))
        d=core.evaluate(depth,frame_age_s=0.,current_altitude_m=1.,
                        geometry={'geometry_valid':True,'roof_clearance_verified':False})
        if new and front_valid:
            assert d.state.value==('EVADE' if blocked else 'CLEAR')
            if blocked: assert d.direction.value=='UP'
        else: assert d.state.value=='STALE'


def test_repeated_and_old_profile_messages_do_not_refresh_front_evidence():
    import threading
    node=NS(_lock=threading.RLock(),get_clock=lambda:NS(now=lambda:NS(nanoseconds=100000000000)))
    inputs=Inputs(node)
    m=NS(source='jetson-realsense-d435i',sequence=1,observation_age_s=.02,
         stamp=NS(sec=100,nanosec=0),obstacle_extent_valid=False,
         geometry_valid=True,roof_clearance_verified=False,roof_vertical_gap_m=float('nan'),
         roof_passage_verified=False,descent_corridor_clear=False,state=0,
         reason='scenario_front_detect_2m_pass_3m_v1:front_corridor_clear')
    with patch('time.monotonic',return_value=100.): inputs.safety(m)
    original=inputs.evidence.copy()
    with patch('time.monotonic',return_value=101.): inputs.safety(m)
    assert inputs.evidence==original
    m.sequence=2; m.reason='scenario_demo_minimal_v1:front_corridor_clear'
    with patch('time.monotonic',return_value=100.): inputs.safety(m)
    assert inputs.evidence['state']==4
    inputs.pool.shutdown()

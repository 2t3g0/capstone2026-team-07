import copy
import json
import math
from pathlib import Path
import pytest
from jolgwa_uav.mavlink_route_fc_link import RouteFcState, HOME_PHASE_CORRECTION_PENDING


def quaternion(yaw=9.4, roll=0., pitch=0.):
    y, r, p = (math.radians(v)/2 for v in (yaw, roll, pitch))
    cy, sy, cr, sr, cp, sp = math.cos(y), math.sin(y), math.cos(r), math.sin(r), math.cos(p), math.sin(p)
    return [cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
            cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]


def feed(state, stamp, received, *, position=(0., 0., 0.), velocity=(0., 0., 0.), q=None, rates=(0., 0., 0.), global_position=None):
    q = q or quaternion(0)
    messages = dict(
        HEARTBEAT=dict(autopilot=12, type=2, base_mode=129, custom_mode=6<<16, system_status=4),
        SYS_STATUS=dict(onboard_control_sensors_present=7, onboard_control_sensors_enabled=7, onboard_control_sensors_health=7),
        ESTIMATOR_STATUS=dict(time_usec=stamp, flags=47),
        ATTITUDE_QUATERNION=dict(time_boot_ms=stamp//1000, q1=q[0], q2=q[1], q3=q[2], q4=q[3],
                                 rollspeed=rates[0], pitchspeed=rates[1], yawspeed=rates[2]),
        LOCAL_POSITION_NED=dict(time_boot_ms=stamp//1000, **dict(zip(('x','y','z','vx','vy','vz'), (*position,*velocity)))),
        GLOBAL_POSITION_INT=dict(time_boot_ms=stamp//1000, lat=350000000, lon=1290000000, alt=40000, relative_alt=0),
        EXTENDED_SYS_STATE=dict(landed_state=2))
    if global_position is not None:
        messages['GLOBAL_POSITION_INT'].update(lat=round(global_position[0]*1e7),
            lon=round(global_position[1]*1e7),alt=round(global_position[2]*1000))
    for kind, data in messages.items():
        assert state.accept(kind, data, 1, 1, received), (kind, state.last_rejection)


def odom(counter, stamp=1100000, q=None, position=(0.,0.,0.), velocity=(0.,0.,0.), child=1):
    return dict(frame_id=1, child_frame_id=child, reset_counter=counter, time_usec=stamp,
                q=q or quaternion(), **dict(zip(('x','y','z','vx','vy','vz'), (*position,*velocity))))


def prepared(counter=12, *, q=None, position=(0.,0.,0.), velocity=(0.,0.,0.), stamp=1000000, home=None, global_position=None):
    r = RouteFcState(transport_epoch=7, home_stability_ns=0)
    now = stamp*1000
    assert r.accept('HOME_POSITION', home or dict(latitude=350000000, longitude=1290000000, altitude=40000,
                                         x=0.,y=0.,z=0.),1,1,now)
    feed(r,stamp,now,position=position,velocity=velocity,q=q,global_position=global_position)
    assert r.accept('ODOMETRY',odom(counter,stamp,q=q or quaternion(0),position=position,velocity=velocity),1,1,now)
    assert r.lock_execution_home('m')[0]
    r.configure_yaw_reset_context(('m',7,r.execution_home_lock_revision))
    return r


@pytest.mark.parametrize('yaw,accepted', [(2.99999,False),(3.,True),(3.00001,True),
                                         (14.99999,True),(15.,True),(15.00001,False)])
def test_yaw_bounds(yaw, accepted):
    r=prepared();feed(r,1100000,1100000000,q=quaternion(yaw))
    assert r.accept('ODOMETRY',odom(13,q=quaternion(yaw)),1,1,1100000000) is accepted


@pytest.mark.parametrize('dt,accepted', [(149999,True),(150000,True),(150001,False)])
def test_interval_bounds(dt,accepted):
    r=prepared(); stamp=1000000+dt; now=stamp*1000
    feed(r,stamp,now); assert r.accept('ODOMETRY',odom(13,stamp),1,1,now) is accepted


@pytest.mark.parametrize('field,value,accepted', [('position',.099999,True),('position',.1,True),('position',.100001,False),
    ('velocity',.199999,True),('velocity',.2,True),('velocity',.200001,False)])
def test_continuity_bounds(field,value,accepted):
    r=prepared();feed(r,1100000,1100000000)
    data=odom(13,**{field:(value,0.,0.)})
    assert r.accept('ODOMETRY',data,1,1,1100000000) is accepted


@pytest.mark.parametrize('axis,value,accepted', [('roll',3.,True),('pitch',3.,True),('roll',3.001,False),('pitch',3.001,False)])
def test_roll_pitch_bounds(axis,value,accepted):
    r=prepared();feed(r,1100000,1100000000)
    assert r.accept('ODOMETRY',odom(13,q=quaternion(**{axis:value})),1,1,1100000000) is accepted


@pytest.mark.parametrize('rate,accepted', [(.199999,True),(.2,True),(.200001,False)])
def test_angular_rate_bounds(rate,accepted):
    r=prepared();feed(r,1100000,1100000000,rates=(0.,0.,rate))
    assert r.accept('ODOMETRY',odom(13),1,1,1100000000) is accepted


@pytest.mark.parametrize('fault', ['approval','epoch','mission','generation','pending','failed','multiple','reverse',
    'attitude_stale','local_stale','global_stale','rates_missing','frame_jump','position_jump','altitude_jump','velocity_jump'])
def test_missing_or_conflicting_proof_rejects(fault):
    r=prepared();feed(r,1100000,1100000000);data=odom(13)
    if fault=='approval':r.configure_yaw_reset_context(None)
    if fault=='epoch':r.transport_epoch+=1
    if fault=='mission':r.execution_home_mission_id='old'
    if fault=='generation':r.execution_home_lock_revision+=1
    if fault=='pending':r.home_phase=HOME_PHASE_CORRECTION_PENDING
    if fault=='failed':r.home_epoch_failure_latched=True
    if fault=='multiple':data['reset_counter']=14
    if fault=='reverse':data['reset_counter']=11
    if fault.endswith('_stale'):
        kind={'attitude_stale':'ATTITUDE_QUATERNION','local_stale':'LOCAL_POSITION_NED','global_stale':'GLOBAL_POSITION_INT'}[fault]
        r.received_ns[kind]=1
    if fault=='rates_missing':r.attitude['angular_rates']=()
    if fault=='frame_jump':r.home_frame_residual_vector=(2.,0.,0.)
    if fault=='position_jump':data['x']=1.
    if fault=='altitude_jump':data['z']=-1.
    if fault=='velocity_jump':data['vx']=1.
    assert not r.accept('ODOMETRY',data,1,1,1100000000)
    assert r.fault=='local_odometry_reset_changed'
    assert r.home_epoch_failure_latched


def test_wrap_duplicate_reorder_and_lifetime():
    r=prepared(255);before=copy.deepcopy(r.home_correction_snapshot());feed(r,1100000,1100000000)
    assert r.accept('ODOMETRY',odom(0),1,1,1100000000)
    assert r.navigation_reset_generations==(255,255,255,255,0)
    assert r.home_lock_reset_counter==0 and not r.fault
    assert before['correction_revision']==r.home_correction_snapshot()['correction_revision']
    assert not r.accept('ODOMETRY',odom(0),1,1,1100000001)
    assert not r.accept('ODOMETRY',odom(255,1099000),1,1,1100000002)
    assert r.navigation_reset_generations==(255,255,255,255,0)
    assert r.release_execution_home('m') and r.yaw_reset_context is None


def test_body_frd_velocity_is_compared_in_ned():
    r=prepared(velocity=(.4,0.,0.));q=quaternion(); yaw=math.radians(9.4)
    feed(r,1100000,1100000000,position=(.04,0.,0.),velocity=(.4,0.,0.),q=q)
    assert r.accept('ODOMETRY',odom(13,q=q,position=(.04,0.,0.),
        velocity=(.4*math.cos(yaw),-.4*math.sin(yaw),0.),child=12),1,1,1100000000)
    assert r.yaw_reset_events[-1]['velocity_delta_m_s']<1e-9


def test_actual_ulog_yaw_correction_keeps_home_and_navigation():
    fixture=json.loads((Path(__file__).parent/'fixtures/yaw_reset_1752.json').read_text())
    old,new=fixture['samples'];r=prepared(q=old['q'],position=old['position'],velocity=old['velocity'],stamp=old['time_usec'],home=fixture['home_message'],global_position=old['global_position'])
    home=copy.deepcopy(r.home_position); revision=r.execution_home_lock_revision
    now=new['time_usec']*1000
    feed(r,new['time_usec'],now,q=new['q'],position=new['position'],velocity=new['velocity'],rates=new['angular_rates'],global_position=new['global_position'])
    # Actual logged Home and global/local continuity are checked without rebasing.
    assert r.accept('ODOMETRY',odom(13,new['time_usec'],q=new['q'],position=new['position'],velocity=new['velocity']),1,1,now)
    assert fixture['yaw_correction_deg']==pytest.approx(9.4062223362)
    assert old['local_counters'][:4]==new['local_counters'][:4]
    assert r.home_position==home and r.execution_home_lock_revision==revision
    assert r.snapshot(now).position_valid and not r.snapshot(now).failsafe
    assert r.navigation_reset_generations==(12,12,12,12,13)
    assert r.home_correction_snapshot()['epoch_failure_latched'] is False

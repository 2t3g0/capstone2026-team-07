"""Regression expectations for the safetycontract2 fixes.

No device/network client or ROS executor is started. Production methods are
called with in-memory publishers, FC state injection and a virtual clock.
"""
import copy, json, math, threading, time
from types import SimpleNamespace as NS, MethodType
from unittest.mock import patch
import pytest
from jolgwa_interfaces.msg import OffboardCommand, ManualOverride, VehicleControlState, FlightCommandAck
from jolgwa_ros.px4_offboard_controller import Px4OffboardController as Controller
from jolgwa_ros.mission_manager_node import MissionManagerNode as Manager
from jolgwa_ros.low_speed import AltitudeReferenceSynchronizer
from test_safety_contract import controller_fixture, terminal_fixture, proof
from test_provisional_home_flow import manager_fixture, approve, goal

def emit(name, **data): print('AUDIT '+json.dumps(dict(case=name,**data),ensure_ascii=False))

def bind(obj, cls, *names):
    for name in names:
        descriptor=cls.__dict__[name]
        setattr(obj,name,descriptor.__get__(obj,cls))

def test_reflight_home_correction_revision_repro():
    s=AltitudeReferenceSynchronizer()
    s.set_provisional_home(altitude_amsl_m=40.,z_ned_m=0.,revision=1)
    meta=dict(frozen_altitude_amsl_m=40.,frozen_z_ned_m=0.,current_altitude_amsl_m=40.,
        current_z_ned_m=0.,correction_valid=True,correction_revision=0,opposition_error_m=0.,
        estimator_reset_counter_valid=True,estimator_reset_counter=0,detail='locked',provisional_home_revision=1)
    assert s.set_home_reference(**meta)
    assert s.set_home_reference(**{**meta,'current_altitude_amsl_m':39.,'current_z_ned_m':1.,'correction_revision':1})
    ns=time.monotonic_ns()
    s.add_local(time_boot_ms=1000,z_ned_m=0.,received_ns=ns)
    first=s.add_global(time_boot_ms=1000,fc_altitude_home_relative_m=1.,
        global_altitude_amsl_m=40.,received_ns=ns,now_ns=ns)
    assert first.home_correction_state==2  # Confirmed correction before explicit release.
    s.release_execution_home()
    assert s.set_provisional_home(altitude_amsl_m=39.,z_ned_m=1.,revision=2)
    accepted=s.set_home_reference(**{**meta,'frozen_altitude_amsl_m':39.,'frozen_z_ned_m':1.,
        'current_altitude_amsl_m':39.,'current_z_ned_m':1.,'provisional_home_revision':2,
        'execution_home_mission_id':'second','execution_home_lock_revision':2})
    emit('second_mission_home',accepted=accepted,revision=s._home_correction_revision,lock_valid=s._execution_home_lock_valid)
    assert accepted and s._execution_home_lock_valid

@pytest.mark.parametrize('guard',[False,True])
def test_stop_release_land_handoff_repro(guard):
    c=controller_fixture()
    c._active_command.flight_profile=2
    c._gate=NS(armed=True,manual_override=False,connected=True,command_output_enabled=True,
        preflight_checks_pass=True,is_approved=lambda mission:mission=='m')
    c._offboard=True; c._landed=False; c._px4_failsafe_active=False
    c._terminal_requires_owned_offboard=guard; c._external_mode_fenced=False
    c._autonomy_reentry_required=False; c._autonomy_resume_required=False
    c._last_px4_message_at=c._last_landed_message_at=time.monotonic()
    c._status_timeout_s=.5; c._last_sequence={'m':7}; c._retired_mission_ids=set()
    c._sim_clock_allows_output=lambda:True; c._local_navigation_fresh=lambda now:True
    c._receipt_age_usable=lambda *args:True; c._reset_avoidance_recovery=lambda:None
    c.get_logger=lambda:NS(warning=lambda *a:None,error=lambda *a:None)
    bind(c,Controller,'_low_speed_profile','_terminal_home_position_valid','_owns_terminal_handoff','_on_command','_on_manual_override')
    c._on_manual_override(ManualOverride(active=True,source='operator'))
    c._on_manual_override(ManualOverride(active=False,source='operator',reason='dashboard emergency LAND'))
    land=copy.deepcopy(c._active_command); land.command=3; land.sequence=8
    c._on_command(land)
    emit('stop_release_land',guard=guard,command=c._active_command.command,error=c._last_error)
    assert c._active_command.command == 3

def test_in_progress_then_failure_ack_repro():
    c=terminal_fixture(); c._terminal_started_at = time.monotonic(); c._sent_vehicle_commands={21}; c._terminal_attempts=1
    c.get_logger=lambda:NS(error=lambda *a:None)
    Controller._on_command_ack(c,NS(command=21,result=5))
    in_progress_state=c._terminal_state
    Controller._on_command_ack(c,NS(command=21,result=4))
    emit('ack_progress_failure',after_progress=in_progress_state,after_failure=c._terminal_state,
         pending_commands=list(c._sent_vehicle_commands))
    assert in_progress_state == VehicleControlState.TERMINAL_PENDING
    assert c._terminal_state==VehicleControlState.TERMINAL_FALLBACK

def test_bridge_progress_consumes_final_ack_context_repro():
    from jolgwa_ros.mavlink_usb_bridge_node import MavlinkUsbBridgeNode as Bridge
    c=terminal_fixture(); ns=time.monotonic_ns(); callbacks=[]
    c._sent_vehicle_commands={21}; c._pending_request_ids={21:'land-1'}
    c._terminal_started_at = time.monotonic(); c._terminal_attempts=1; c.get_logger=lambda:NS(error=lambda *a:None)
    bind(c,Controller,'_on_context_ack','_on_command_ack')
    req=NS(mission_id='m',output_epoch=c._output_contract.output_epoch,
        output_sequence=c._output_contract.output_sequence,request_id='land-1')
    def receive(message): callbacks.append(message.result); c._on_context_ack(message)
    import io
    b=NS(_contract=c._output_contract,_pending_acks={21:(req,ns)},
        _context_ack_publisher=NS(publish=receive),_ack_publisher=NS(publish=lambda _:None),
        _journal=io.StringIO(),_timestamp_us=lambda:1)
    ack=dict(command=21,result=5,progress=50,result_param2=0,target_system=245,target_component=191)
    with patch('time.monotonic_ns',return_value=ns+1): Bridge._publish_mavlink_ack(b,ack,1,1)
    with patch('time.monotonic_ns',return_value=ns+2): Bridge._publish_mavlink_ack(b,{**ack,'result':4},1,1)
    emit('bridge_ack_progress_failure',forwarded_results=callbacks,controller_state=c._terminal_state)
    assert callbacks==[5,4] and c._terminal_state==VehicleControlState.TERMINAL_FALLBACK

@pytest.mark.parametrize('age_ns,accepted',[(149999999,True),(150000000,True),(150000001,False)])
def test_proof_150ms_boundary(age_ns,accepted):
    from jolgwa_ros.flight_contract import proof_fresh
    c=controller_fixture(); ns=time.monotonic_ns(); p=proof(c,ns,25)
    assert proof_fresh(p,c._output_contract,ns+age_ns)==accepted

@pytest.mark.parametrize('age_ms,rejected',[(399,False),(400,False),(401,True)])
def test_home_deadline_boundary(age_ms,rejected):
    s=AltitudeReferenceSynchronizer()
    s._start_correction_event(10_000_000_000,-1.)
    s._home_correction_state=1
    observed=s.snapshot(10_000_000_000+age_ms*1_000_000)
    assert observed.altitude_epoch_failure_latched==rejected

def test_late_home_evidence_before_watchdog_repro():
    s=AltitudeReferenceSynchronizer()
    meta=dict(frozen_altitude_amsl_m=40.,frozen_z_ned_m=0.,current_altitude_amsl_m=40.,
        current_z_ned_m=0.,correction_valid=True,correction_revision=0,opposition_error_m=0.,
        estimator_reset_counter_valid=True,estimator_reset_counter=0,detail='locked')
    s.set_home_reference(**meta)
    base=10_000_000_000
    def sample(boot,ns,raw):
        s.add_local(time_boot_ms=boot,z_ned_m=0.,received_ns=ns)
        return s.add_global(time_boot_ms=boot,fc_altitude_home_relative_m=raw,
            global_altitude_amsl_m=40.,received_ns=ns,now_ns=ns)
    for i in range(41): sample(1000+i*50,base+i*50_000_000,0.)
    pending_ns=base+2_050_000_000
    first=sample(3050,pending_ns,1.)
    assert first.home_correction_state==1
    late_ns=pending_ns+600_000_000
    with patch('time.monotonic_ns',return_value=late_ns):
        s.set_home_reference(**{**meta,'current_altitude_amsl_m':39.,'current_z_ned_m':1.,'correction_revision':1})
        observed=sample(3650,late_ns,1.)
    emit('late_home_evidence',delay_ms=600,correction_state=observed.home_correction_state,
        epoch_failure=observed.altitude_epoch_failure_latched,pending_age_ms=observed.home_correction_pending_age_ms)
    assert observed.home_correction_state==3 and observed.altitude_epoch_failure_latched

def test_emergency_while_paused_with_cancel_delivery_loss_repro():
    clock=Clock()
    m=NS(_lock=threading.RLock(),_manual_active=True,_resume_requested=False,_resume_required=True,
        _active_goal=True,_active_mission_id='m',_active_proposal_id='p',_active_waypoint=0,
        _active_total_waypoints=1,_emergency_land_requested=False,_active_cancel_requested=False,
        _vehicle_state_received_at=100.,_vehicle_state=NS(armed=True,connected=True,
            vehicle_status_fresh=True,arming_state_valid=True,command_output_enabled=True,
            emergency_land_available=True,active_execution_mission_id='m'),
        _publish_status=lambda *a:None,_publish_feedback=lambda *a:None,
        _require_explicit_resume=lambda **kw:None,_manual_publisher=NS(publish=lambda _:None),
        get_clock=lambda:NS(now=lambda:NS(to_msg=lambda:__import__('builtin_interfaces.msg',fromlist=['Time']).Time())))
    bind(m,Manager,'_can_issue_terminal_land','_on_emergency_land')
    gh=NS(request=NS(mission_id='m'),is_cancel_requested=False)
    accepted=[]
    class BoundedProbe(Exception): pass
    def advance(dt):
        if not accepted:
            response=m._on_emergency_land(NS(mission_id='m',operator_id='audit',reason='LAND'),NS())
            accepted.append(response.accepted)
        if clock.now>102.: raise BoundedProbe()
    clock.advance=advance
    with patch('time.monotonic',side_effect=lambda:clock.now),patch('time.sleep',side_effect=clock.sleep):
        assert Manager._wait_for_explicit_resume(m,gh,return_phase=2,current_waypoint=0,total_waypoints=1,
                reason='manual',publish_hold=False) == "cancel"
    emit('emergency_pause_cancel_loss',accepted=accepted[0],cancel_latched=m._active_cancel_requested,
        emergency_latched=m._emergency_land_requested,wait_exited=True)
    assert accepted==[True] and m._active_cancel_requested

@pytest.mark.parametrize('seconds,expected',[(.999,False),(1.,True),(1.001,True)])
def test_continuous_output_warmup_boundary(seconds,expected):
    from jolgwa_ros.low_speed import flight_output_confirmation_ready
    ready=flight_output_confirmation_ready(receipt_age_s=0.,continuous_age_s=seconds,
        setpoint_age_ms=0,consecutive_transmissions=30,exact_match=True,transport_connected=True,
        command_graph_ready=True,envelope_valid=True,setpoint_transmitted=True,warmup_s=1.)
    assert ready==expected

@pytest.mark.parametrize('seconds,failed',[(89.999,False),(90.,True),(90.001,True)])
def test_terminal_timeout_boundary(seconds,failed):
    c=terminal_fixture(); c._external_mode_fenced=True; c._native_land=True
    c._supervise_terminal(10.+seconds)
    assert (c._terminal_state==VehicleControlState.TERMINAL_FAILED)==failed

def test_normal_route_cannot_inherit_camera_off_exception():
    c=controller_fixture(); c._forward_test_camera_required=False
    c._active_command.flight_profile=2
    c._active_command.requested_authority=OffboardCommand.AUTHORITY_LLM_ROUTE
    assert not Controller._forward_test_camera_bypass(c,c._active_command)
    c._active_command.requested_authority=OffboardCommand.AUTHORITY_FORWARD_TEST_1M
    assert Controller._forward_test_camera_bypass(c,c._active_command)

def test_altitude_age_is_max_instead_of_sum_repro():
    m=manager_fixture()
    sample=m._altitude_reference_state
    sample.local_age_ms=sample.global_age_ms=300
    m._altitude_reference_state_received_at=10.
    reference=dict(home_generation=m._route_home_generation,transport_epoch=sample.transport_epoch,aligned_home_z_ned_m=0.)
    reason,*_=m._altitude_reference_status_locked(reference,10.3)
    c=controller_fixture(); c._active_command.flight_profile=2
    c._active_command.altitude_reference=2; c._active_command.altitude_reference_max_error_m=1.5
    c._active_command.altitude_reference_epoch=sample.transport_epoch
    c._altitude_reference_state=sample; c._altitude_reference_state_received_at=10.
    valid,detail,*_=Controller._altitude_reference_status(c,c._active_command,10.3)
    emit('altitude_age_underestimate',source_age_ms=300,elapsed_since_receipt_ms=300,
        minimum_actual_age_ms=600,manager_reason=reason,controller_valid=valid,controller_detail=detail)
    assert reason == "altitude_reference_stale" and not valid

class Clock:
    def __init__(self): self.now=100.; self.advance=lambda dt:None
    def sleep(self,dt): self.now+=dt; self.advance(dt)

@pytest.mark.parametrize('height,camera',[(1.,True),(1.,False),(2.,True),(2.,False)])
@pytest.mark.parametrize('scenario',['normal','position_loss','cancel','terminal_stale','cancel_position_loss','epoch_loss','ack_delay_state_loss','battery','battery_cancel','arm_home_reorder'])
def test_full_action_loop(height,camera,scenario):
    clock=Clock()
    with patch('time.monotonic',side_effect=lambda:clock.now), patch('time.monotonic_ns',side_effect=lambda:int(clock.now*1e9)),patch('time.sleep',side_effect=clock.sleep):
        m=manager_fixture(); m._forward_test_camera_required=camera
        s=m._vehicle_state
        s.jetson_safety_state='CLEAR'; s.jetson_safety_fresh=True; s.low_speed_obstacle_guard_enabled=True
        s.approved=True; s.mission_id=''; s.active_execution_mission_id=''; s.active_authority='LLM_ROUTE'
        s.velocity_ned_m_s=[0.,0.,0.]; s.avoidance_active=False; s.last_error=''; s.terminal_state=0
        s.prearm_terminated=False; s.arm_request_transmitted=False
        response=NS(accepted=False,proposal_id='',message='')
        m._on_prepare_forward_test(NS(operator_id='audit',target_altitude_home_m=height),response)
        assert response.accepted,response.message
        proposal=m._proposal_publisher.messages[-1]; m._proposals[proposal.proposal_id]=proposal
        m._update_route_frame_status(proposal.proposal_id)
        approval=approve(m,proposal.proposal_id); assert approval.accepted
        request=goal(m,proposal.proposal_id,approval.mission_id)
        from rclpy.action import GoalResponse
        assert m._on_goal(request)==GoalResponse.ACCEPT
        s.mission_id=s.active_execution_mission_id=approval.mission_id
        m._max_laps=10; m._max_duration_minutes=10; m._ready_timeout=2.; m._takeoff_timeout=5.
        m._waypoint_timeout=5.; m._landing_timeout=2.; m._sequence=0
        m._manual_active=False; m._resume_required=False; m._resume_requested=False
        m._emergency_land_requested=False; m._terminal_deadlines={}
        m._active_waypoint=m._active_total_waypoints=0
        m._event_control=NS(clear_for_mission=lambda _:None)
        m._active_flight_profile='NORMAL'; m._active_mission_kind='ROUTE'
        m._active_altitude_reference=None; m._low_settle_since=None
        m.get_logger=lambda:NS(warning=lambda *a:None,error=lambda *a:None)
        m._publish_feedback=lambda *args:None
        bind(m,Manager,'_execute','_execute_impl','_cleanup_execution','_snapshot_state',
            '_vehicle_ready','_evaluate_vehicle_readiness','_wait_for','_active_low_speed_error_locked',
            '_low_target_reached','_publish_command','_complete_low_speed_terminal','_low_speed_terminal_outcome',
            '_confirmed_landed_disarmed','_land_current','_can_issue_terminal_land','_abort_result',
            '_finish_wait_failure','_execute_exception_result')
        commands=[]; command_messages=[]; last_command=[None]; goal_state=[]; arm_pending_ticks=[0]
        gh=NS(request=request,is_cancel_requested=False,is_active=True)
        gh.succeed=lambda:goal_state.append('succeed'); gh.abort=lambda:goal_state.append('abort')
        gh.canceled=lambda:goal_state.append('canceled')
        def dispatch(command):
            commands.append(command.command); command_messages.append(command); last_command[0]=command
        m._command_publisher=NS(publish=dispatch)
        def advance(dt):
            cmd=last_command[0]
            if scenario in ('terminal_stale', 'ack_delay_state_loss') and cmd is not None and cmd.command==3:
                if scenario == 'ack_delay_state_loss':
                    s.terminal_state=VehicleControlState.TERMINAL_PENDING
                    s.terminal_started_monotonic_ns=int((clock.now-.1)*1e9)
                return
            m._vehicle_state_received_at=clock.now; m._route_home_received_at=clock.now
            m._altitude_reference_state_received_at=clock.now
            m._vehicle_geo_received_at=m._low_speed_safety_received_at=clock.now
            if cmd is not None:
                if cmd.command in (1,2):
                    if scenario == 'arm_home_reorder' and cmd.command == 1 and arm_pending_ticks[0] < 2:
                        arm_pending_ticks[0] += 1
                        s.arm_request_transmitted=True; m._active_ever_armed=True
                        s.armed=False; s.landed=True; s.offboard=True
                        m._altitude_reference_state.state=4
                        m._altitude_reference_state.home_correction_pending_age_ms=arm_pending_ticks[0]*50
                        return
                    m._altitude_reference_state.state=1
                    s.armed=True; s.landed=False; s.offboard=True; m._active_ever_armed=True
                    s.position_ned_m=list(cmd.position_ned_m)
                    m._altitude_reference_state.local_z_ned_m=s.position_ned_m[2]
                    m._altitude_reference_state.fc_altitude_home_relative_m=-s.position_ned_m[2]
                    m._altitude_reference_state.normalized_fc_altitude_home_relative_m=-s.position_ned_m[2]
                    if scenario in ('position_loss','cancel_position_loss'): s.position_valid=False
                    if scenario in ('cancel','cancel_position_loss'): gh.is_cancel_requested=True
                    if scenario=='epoch_loss': m._altitude_reference_state.transport_epoch+=1
                    if scenario in ('battery', 'battery_cancel'):
                        s.preflight_checks_pass=False
                        s.last_error='LOW_SPEED_PROFILE battery health warning; braking then LAND'
                        if scenario == 'battery_cancel': gh.is_cancel_requested=True
                elif cmd.command==3:
                    s.armed=False; s.landed=True; s.terminal_state=VehicleControlState.TERMINAL_ACCEPTED
        clock.advance=advance
        result=m._execute(gh)
        emit('full_action',height=height,camera=camera,scenario=scenario,commands=commands,
            success=result.success,message=result.message,elapsed=round(clock.now-100,2),goal=goal_state)
        assert not m._active_goal and m._goal_reservation is None
        assert approval.mission_id not in m._approved
        if scenario in ('normal','arm_home_reorder'):
            assert result.success and commands[:3]==[1,2,3] and set(commands[3:]) <= {3},result.message
            goto = next(message for message in command_messages if message.command == 2)
            start = json.loads(proposal.plan_json)['preview_start_ned_m']
            assert math.hypot(goto.position_ned_m[0]-start[0],
                              goto.position_ned_m[1]-start[1]) == pytest.approx(2.0)
        else: assert not result.success and 3 in commands,result.message
        if scenario in ('battery', 'battery_cancel'):
            assert 'battery health warning' in result.message

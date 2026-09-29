"""Full production Action loop with a clocked motion/transport stand-in."""
import hashlib
import math
import threading
import time
import uuid
from concurrent.futures import Future
from types import SimpleNamespace as NS
from unittest.mock import patch
import pytest
from jolgwa_interfaces.msg import VehicleControlState
from rclpy.action import GoalResponse
from test_provisional_home_flow import manager_fixture, approve, goal
from test_safetycontract2 import bind, Clock, Manager
from test_scenario import evidence
from jolgwa_ros.scenario_runtime import Inputs
from jolgwa_ros.scenario_contract import ScenarioRegistry


@pytest.fixture(params=['in_process','isolated_dds'])
def action_transport(request):
    if request.param=='in_process':
        yield lambda callback: callback
        return
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from jolgwa_interfaces.msg import OffboardCommand
    rclpy.init(args=[])
    suffix=uuid.uuid4().hex
    node=rclpy.create_node('scenario_action_wire_test_'+suffix)
    topic='/scenario_test/action_output_'+suffix
    event=threading.Event(); dispatch=[None]
    node.create_subscription(OffboardCommand,topic,
        lambda msg:(dispatch[0](msg),event.set()),10)
    publisher=node.create_publisher(OffboardCommand,topic,10)
    executor=MultiThreadedExecutor(num_threads=2); executor.add_node(node)
    worker=threading.Thread(target=executor.spin,daemon=True); worker.start()
    deadline=time.monotonic()+3
    while publisher.get_subscription_count()==0 and time.monotonic()<deadline: time.sleep(.01)
    assert publisher.get_subscription_count()>0
    def connect(callback):
        dispatch[0]=callback
        def send(message):
            event.clear(); publisher.publish(message)
            assert event.wait(2.),'Action output DDS delivery timed out'
        return send
    try:
        yield connect
    finally:
        executor.shutdown(timeout_sec=2); worker.join(2)
        node.destroy_node(); rclpy.shutdown()


class InlinePool:
    def submit(self, fn, *args):
        future=Future()
        try: future.set_result(fn(*args))
        except Exception as exc: future.set_exception(exc)
        return future


@pytest.mark.parametrize("height",[1.,2.])
@pytest.mark.parametrize("fault,interrupt_phase",
    [(kind,None) for kind in ("normal","position","photo","stale_landing")]+
    [(kind,phase) for kind in ("cancel","emergency","rc") for phase in
     ("SEARCH_OBSTACLE","BRAKE","CLIMB","PASS","DESCEND","SEARCH_EVENT","WAIT_EVENT","EVENT_HOLD","PHOTO","REJOIN","FINAL")])
def test_entire_action_prepare_approve_takeoff_scenario_terminal(height,fault,interrupt_phase,tmp_path,action_transport):
    clock=Clock()
    with patch('time.monotonic',side_effect=lambda:clock.now), patch('time.monotonic_ns',side_effect=lambda:int(clock.now*1e9)), patch('time.sleep',side_effect=clock.sleep):
        m=manager_fixture(); m._scenario_enabled=True; m._scenario_registry=ScenarioRegistry()
        # No camera connection check at preparation, even with no perception.
        m._forward_test_camera_required=True
        s=m._vehicle_state
        s.jetson_safety_state=''; s.jetson_safety_fresh=False; s.low_speed_obstacle_guard_enabled=True
        s.approved=True; s.mission_id=''; s.active_execution_mission_id=''; s.active_authority='LLM_ROUTE'
        s.velocity_ned_m_s=[0.,0.,0.]; s.avoidance_active=False; s.last_error=''; s.terminal_state=0
        s.prearm_terminated=False; s.arm_request_transmitted=False; s.terminal_in_progress=False; s.flight_epoch_retired=False
        response=NS(accepted=False,proposal_id='',message='')
        m._on_prepare_forward_test(NS(operator_id='test',target_altitude_home_m=height),response)
        assert response.accepted,response.message
        proposal=m._proposal_publisher.messages[-1]; m._proposals[proposal.proposal_id]=proposal
        m._update_route_frame_status(proposal.proposal_id)
        approval=approve(m,proposal.proposal_id); assert approval.accepted,approval.message
        m._scenario_registry.proposal(proposal)
        m._scenario_registry.approval(NS(mission_id=approval.mission_id,proposal_id=proposal.proposal_id,approved=True))
        request=goal(m,proposal.proposal_id,approval.mission_id)
        assert m._on_goal(request)==GoalResponse.ACCEPT
        s.mission_id=s.active_execution_mission_id=approval.mission_id
        m._max_laps=10; m._max_duration_minutes=10; m._ready_timeout=2.; m._takeoff_timeout=15.
        m._waypoint_timeout=120.; m._landing_timeout=2.; m._sequence=0
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
            '_finish_wait_failure','_execute_exception_result','_execution_interrupted')
        commands=[]; last=[None]; statuses=m.statuses
        gh=NS(request=request,is_cancel_requested=False,is_active=True)
        gh.succeed=lambda:None; gh.abort=lambda:None; gh.canceled=lambda:None
        def dispatch(c): commands.append(c); last[0]=c
        m._command_publisher=NS(publish=action_transport(dispatch))
        inputs=Inputs(m); inputs.pool.shutdown(); inputs.pool=InlinePool()
        m._scenario_inputs=inputs; m._scenario_photo_root=tmp_path/'photos'
        original=tmp_path/'original.jpg'; original.write_bytes(b'original detector input')
        ticks=[0]; interrupted_at=[None]
        def advance(dt):
            ticks[0]+=1; cmd=last[0]
            if fault=='stale_landing' and cmd is not None and cmd.command==3: return
            m._vehicle_state_received_at=m._route_home_received_at=clock.now
            m._altitude_reference_state_received_at=clock.now
            m._vehicle_geo_received_at=m._low_speed_safety_received_at=clock.now
            if cmd is not None:
                if cmd.command in (0,1,2):
                    s.armed=True; s.landed=False; s.offboard=True; m._active_ever_armed=True
                    velocity=[max(-.5,min(.5,2*(t-p))) for t,p in zip(cmd.position_ned_m,s.position_ned_m)]
                    s.position_ned_m=[p+v*dt for p,v in zip(s.position_ned_m,velocity)]
                    s.velocity_ned_m_s=velocity
                    a=m._altitude_reference_state
                    a.local_z_ned_m=s.position_ned_m[2]
                    a.fc_altitude_home_relative_m=a.normalized_fc_altitude_home_relative_m=-s.position_ned_m[2]
                elif cmd.command==3:
                    s.armed=False; s.landed=True; s.terminal_state=VehicleControlState.TERMINAL_ACCEPTED
            x,y,z=s.position_ned_m
            state=3 if 1<=x<2 and z>-height-.85 else 0
            inputs.evidence=evidence(clock.now,state,ticks[0])
            inputs.frame=(ticks[0],clock.now,b'stopped frame')
            if x>7.2 and interrupt_phase!='WAIT_EVENT':
                inputs.event=dict(type='LITTERING',input_received=clock.now,event_id='event',
                    mission_id=request.mission_id,photo_path=str(original),
                    photo_sha256=hashlib.sha256(original.read_bytes()).hexdigest())
                if fault=='photo': inputs.event['photo_sha256']='wrong'
            if interrupt_phase and any(row[3]=='SCENARIO_V7_FRONT_DETECT '+interrupt_phase for row in statuses):
                if interrupted_at[0] is None: interrupted_at[0]=len(commands)
                if fault=='cancel': gh.is_cancel_requested=True
                if fault=='emergency': m._emergency_land_requested=True
                if fault=='rc': m._manual_active=True; s.manual_override=True
            if x>1 and fault=='position': s.position_valid=False
        clock.advance=advance
        result=m._execute(gh)
        assert inputs.event is None and inputs.frame is None and inputs.evidence is None
        assert not m._active_goal and m._goal_reservation is None
        assert request.mission_id not in m._approved
        if fault=='normal':
            assert result.success,result.message
            assert 'SCENARIO_V7_FRONT_DETECT' in result.message
            final=next(c for c in reversed(commands) if c.command==2)
            assert 8<=final.position_ned_m[0]<=11
            assert min(c.position_ned_m[2] for c in commands)>=-height-2.00001
            assert (tmp_path/'photos'/request.mission_id/'event/stopped.jpg').is_file()
            for phase in ('BRAKE','CLIMB','PASS','DESCEND','EVENT_HOLD','PHOTO','REJOIN','FINAL'):
                assert any(phase in row[3] for row in statuses),phase
        else:
            assert not result.success,result.message
        if interrupt_phase:
            assert interrupted_at[0] is not None,(interrupt_phase,result.message)
            assert all(c.command in (3,4,5) for c in commands[interrupted_at[0]:])
        assert all(math.hypot(c.position_ned_m[0],c.position_ned_m[1])<=11.01 for c in commands)

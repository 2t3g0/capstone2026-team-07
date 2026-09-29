"""Actual Manager Action -> Controller -> Bridge -> memory FC, with depth DDS.

No serial port, network FC, GPU, or physical dynamics. The plant integrates
only velocity setpoints that passed the production final transport gate.
"""
import copy
import hashlib
import json
import math
import threading
import time
import uuid
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from jolgwa_interfaces.msg import MissionApproval, SafetyDecision, ManualOverride
from jolgwa_ros.px4_offboard_controller import Px4OffboardController as Controller
from jolgwa_ros.scenario_contract import ScenarioRegistry
from jolgwa_ros.scenario_runtime import Inputs
from jolgwa_ros.low_speed import AltitudeReferenceSynchronizer
from jolgwa_ros import scenario_battery_home as battery_home
from jolgwa_uav.mavlink_route_fc_link import RouteFcState
from jolgwa_uav.depth_geometry import DepthGeometryTracker, DepthIntrinsics, CameraExtrinsics, VehiclePose
from jolgwa_uav.vertical_avoidance import VerticalObstacleAvoidanceCore, VerticalAvoidanceConfig
from test_safety_contract import bridge
from test_safetycontract2 import bind, Clock, Manager
from test_provisional_home_flow import manager_fixture, approve, goal
from test_vehicle_command_transport import memory_transport
from test_scenario_action import InlinePool


class Chain:
    def __init__(self,b,c,m,clock,height,fault='none'):
        self.b,self.c,self.m,self.clock,self.height=b,c,m,clock,height
        self.position=[0.,0.,0.]; self.velocity=[0.,0.,0.]
        self.armed=False; self.offboard=False; self.landing=False; self.count=0
        self.command_ids=[]; self.targets=[]; self.initial_extent_missing=False
        self.offset=0; self.tx_velocity=[0.,0.,0.]
        self.fault=fault; self.fault_started=None; self.delayed_metadata=None
        b._scenario_enabled=c._scenario_enabled=True
        b._scenario_registry=ScenarioRegistry(); c._scenario_registry=ScenarioRegistry()
        m._scenario_registry=ScenarioRegistry()
        b._altitude_reference_sync=AltitudeReferenceSynchronizer()
        b._route_state=RouteFcState(transport_epoch=b._altitude_reference_sync.transport_epoch,home_stability_ns=0)
        b._transport=memory_transport(); b._transport.bad_data=0
        b._command_graph_ready=lambda:True
        c._gate.enable_px4_commands=True; c._usb_output_contract=True
        c._require_jetson_safety=True; c._forward_test_camera_required=False
        c._terminal_requires_owned_offboard=True
        def publish(callback): return NS(publish=callback)
        c._contract_publisher=publish(b._on_output_contract)
        c._command_request_publisher=publish(b._on_command_request)
        c._offboard_mode_publisher=publish(b._on_offboard_control_mode)
        c._setpoint_publisher=publish(b._on_trajectory_setpoint)
        c._flight_envelope_publisher=publish(b._on_flight_envelope)
        b._context_ack_publisher=publish(c._on_context_ack)
        b._vehicle_status_publisher=publish(c._on_vehicle_status)
        b._local_position_publisher=publish(c._on_local_position)
        b._land_publisher=publish(c._on_land_detected)
        b._flight_output_state_publisher=publish(c._on_flight_output_state)
        b._altitude_reference_publisher=publish(lambda msg:(c._on_altitude_reference_state(msg),m._on_altitude_reference_state(msg)))
        b._geo_publisher=publish(lambda msg:(c._on_vehicle_geo_state(msg),m._on_vehicle_geo_state(msg)))
        b._low_speed_safety_publisher=publish(m._on_low_speed_safety_state)
        c._state_publisher=publish(m._on_vehicle_state)
        m._command_publisher=publish(self.command)
        self.geometry=DepthGeometryTracker()
        self.policy=VerticalObstacleAvoidanceCore(VerticalAvoidanceConfig(
            require_geometry=True,scenario_demo=True,front_climb_demo=True,trigger_distance_m=2.,release_distance_m=4.5, max_dynamic_trigger_distance_m=2.,
            minimum_standoff_m=1.0,emergency_margin_m=.5,trigger_samples=3,release_samples=1,
            min_valid_fraction=.25,near_percentile=2.,min_obstacle_fraction=.02,max_depth_m=20.))
        self.wire=rclpy.create_node('scenario_chain_'+uuid.uuid4().hex)
        self.topic='/scenario_chain/depth_'+uuid.uuid4().hex
        self.received=threading.Event()
        self.wire.create_subscription(SafetyDecision,self.topic,self.receive,10)
        self.publisher=self.wire.create_publisher(SafetyDecision,self.topic,10)
        self.executor=MultiThreadedExecutor(num_threads=2); self.executor.add_node(self.wire)
        self.worker=threading.Thread(target=self.executor.spin,daemon=True); self.worker.start()
        assert self.publisher.get_subscription_count()>0

    def receive(self,msg):
        self.c._on_jetson_safety_decision(msg)
        self.m._scenario_inputs.safety(msg)
        self.received.set()

    def command(self,msg):
        self.targets.append(copy.deepcopy(msg))
        self.b._on_offboard_command(msg)
        self.c._on_command(msg)

    def feed_fc(self):
        b=self.b; ns=int(self.clock.now*1e9); boot=round(self.clock.now*1000)
        if self.fault=='yaw_reset' and not getattr(self,'yaw_reset_injected',False):
            cmd=self.c._active_command
            if cmd is not None and cmd.command==2 and self.position[2]<-self.height-.3:
                self.yaw_reset_injected=True
                self.home_before_yaw=copy.deepcopy(b._route_state.home_position)
        angle=0.1641695499420166 if getattr(self,'yaw_reset_injected',False) else 0.
        quaternion=[math.cos(angle/2),0.,0.,math.sin(angle/2)]
        x,y,z=self.position; vx,vy,vz=self.velocity
        data={
            'HEARTBEAT':dict(autopilot=12,type=2,base_mode=129 if self.armed else 1,
                custom_mode=(6<<16 if self.offboard else (4<<16|6<<24) if self.landing else 3<<16),
                system_status=4 if self.armed else 3),
            'SYS_STATUS':dict(onboard_control_sensors_present=7,onboard_control_sensors_enabled=7,onboard_control_sensors_health=7),
            'ESTIMATOR_STATUS':dict(time_usec=boot*1000,flags=47),
            'ATTITUDE_QUATERNION':dict(time_boot_ms=boot,q1=quaternion[0],q2=0.,q3=0.,q4=quaternion[3],rollspeed=0.,pitchspeed=0.,yawspeed=.02),
            'LOCAL_POSITION_NED':dict(time_boot_ms=boot,x=x,y=y,z=z,vx=vx,vy=vy,vz=vz),
            'GLOBAL_POSITION_INT':dict(time_boot_ms=boot,lat=350000000+round(x/111320*1e7),lon=1290000000,alt=round((40-z)*1000),relative_alt=round(-z*1000)),
            'ODOMETRY':dict(time_usec=boot*1000,frame_id=1,child_frame_id=1,reset_counter=int(getattr(self,'yaw_reset_injected',False)),x=x,y=y,z=z,vx=vx,vy=vy,vz=vz,q=quaternion),
            'EXTENDED_SYS_STATE':dict(landed_state=1 if not self.armed else 2),
        }
        if self.fault.startswith('battery'):
            if x>.75 and self.fault_started is None:
                self.fault_started=self.clock.now
            elapsed=self.clock.now-self.fault_started if self.fault_started is not None else -1.
            low=elapsed>=0 and (self.fault=='battery_sustained' or elapsed<1.6)
            data['BATTERY_STATUS']=dict(id=0,charge_state=2 if low else 1,
                fault_bitmask=0,voltages=[14500]+[65535]*9)
            data['SYS_STATUS']=dict(onboard_control_sensors_present=7|(1<<25),
                onboard_control_sensors_enabled=7|(1<<25),
                onboard_control_sensors_health=7 if elapsed>=0 else 7|(1<<25))
        if self.fault=='home_delayed' and x>.75 and self.fault_started is None:
            self.fault_started=self.clock.now
            self.prepared_home=copy.deepcopy(b._route_state.home_position)
        correction=.6 if self.fault=='home_delayed' and self.fault_started is not None else 0.
        data['GLOBAL_POSITION_INT']['relative_alt']=round((-z+correction)*1000)
        if not b._route_state.home_position or self.count % 10 == 0:
            home_correction=correction if self.clock.now-(self.fault_started or self.clock.now) >= .6 else 0.
            assert b._route_state.accept('HOME_POSITION',dict(latitude=350000000,longitude=1290000000,
                altitude=round((40-home_correction)*1000),x=0.,y=0.,z=home_correction),1,1,ns)
        for kind,value in data.items():
            if kind=='ODOMETRY':
                b._refresh_yaw_reset_context(ns)
            assert b._route_state.accept(kind,value,1,1,ns),(kind,b._route_state.last_rejection,boot)
        b._journal_yaw_resets()
        if getattr(self,'yaw_reset_injected',False):
            assert b._route_state.home_position==self.home_before_yaw
            assert not b._route_state.home_epoch_failure_latched
            assert not self.c._autonomy_reentry_required
        b._publish_geo(ns)
        sync=b._altitude_reference_sync
        sync.confirmation_ms=battery_home.home_confirmation_ms(b,b._contract.command if b._contract else None)
        sync.add_local(time_boot_ms=boot,z_ned_m=z,received_ns=ns)
        sync.add_global(time_boot_ms=boot,fc_altitude_home_relative_m=-z+correction,global_altitude_amsl_m=40-z,received_ns=ns,now_ns=ns)
        b._route_state.transport_epoch=sync.transport_epoch  # Same as Bridge._poll_once.
        b._px4_parameters={'COM_OBL_RC_ACT':(4.,ns),'MPC_LAND_SPEED':(.5,ns)}
        b._publish_state()

    def depth(self):
        x,y,z=self.position; now=self.clock.now
        if self.fault=='takeoff_camera_gap' and (self.c._active_command is None or self.c._active_command.command==1):
            return
        if x>.75 and self.fault in ('short_depth_gap','long_depth_gap') and self.fault_started is None:
            self.fault_started=now
        if (self.fault_started is not None and self.fault in ('short_depth_gap','long_depth_gap')
                and now-self.fault_started < (1.5 if self.fault=='short_depth_gap' else 7.)):
            return
        depth=np.full((48,64),20.,dtype=np.float32)
        # A finite wall in the center ROI; open upper corridor. No roof or
        # floor certificate is inserted. Geometric extent comes from pixels.
        # Success layout: the finite obstacle enters view at x=3.2m.
        # Static visibility from launch is separately tested as a blocked descent.
        if (x>=3.2 or self.fault=='static_obstacle') and x<5. and z>-self.height-.65:
            depth[17:31,:]=max(.2,5.-x)
        depth[:16,:]=0.; depth[33:,:]=0.
        geo=self.geometry.observe(depth,intrinsics=DepthIntrinsics(64,48,60.,60.,31.5,23.5),
            extrinsics=CameraExtrinsics.from_mount_quaternion(),
            pose=VehiclePose(tuple(self.position),(1.,0.,0.,0.),now),depth_timestamp_s=now,now_s=now,
            route_direction_ned=(1.,0.,0.),avoidance_active=self.policy.active or self.policy._roof_guard_active,
            nominal_altitude_m=self.height).as_payload()
        if hasattr(self,'native_runtime'):
            decision,_=self.native_runtime.evaluate(depth,current_z_ned_m=z,frame_age_s=0.,
                gimbal_forward=True,geometry_metadata=dict(calibrated=True,
                    attitude_age_s=0.,camera_info_age_s=0.,width=64,height=48,
                    fx=60.,fy=60.,cx=31.5,cy=23.5,mount_rpy=(0.,0.,0.),
                    mount_translation=(0.,0.,0.),position_ned_m=tuple(self.position),
                    quaternion=(1.,0.,0.,0.),heading_rad=0.))
            self.policy=self.native_runtime.policy
            geo=decision.geometry
            assert decision.reason.startswith('scenario_front_detect_2m_pass_3m_v1:')
        else:
            decision=self.policy.evaluate(depth,frame_age_s=0.,current_altitude_m=-z,geometry=geo)
        if decision.state.value!='CLEAR' and not geo['obstacle_extent_valid']:
            self.initial_extent_missing=True
        msg=SafetyDecision(source='jetson-realsense-d435i',sequence=self.count,confidence=1.,
            state={'CLEAR':0,'HOLD':2,'EVADE':3,'STALE':4}[decision.state.value],
            reason=(decision.reason if hasattr(self,'native_runtime') else 'scenario_front_detect_2m_pass_3m_v1:'+decision.reason),observation_age_s=0.,max_speed_mps=.5)
        msg.stamp=self.wire.get_clock().now().to_msg()
        for key in ('geometry_valid','roof_clearance_verified','obstacle_extent_valid','roof_passage_verified'):
            setattr(msg,key,geo[key])
        for key in ('roof_vertical_gap_m','roof_height_m','obstacle_far_north_m','obstacle_far_east_m'):
            setattr(msg,key,float(geo[key]) if geo[key] is not None else math.nan)
        self.received.clear(); self.publisher.publish(msg)
        assert self.received.wait(2.),'depth DDS delivery'

    def advance(self,dt):
        self.count+=1
        if self.delayed_metadata is not None and getattr(self.c, '_scenario_pending_takeoff', None) is not None:
            proposal,approval=self.delayed_metadata
            self.c._scenario_registry.proposal(proposal); self.c._scenario_registry.approval(approval)
            self.delayed_metadata=None
        if self.fault=='rc' and self.position[0]>.75 and self.fault_started is None:
            self.fault_started=self.clock.now
            self.c._on_manual_override(ManualOverride(active=True,source='RC-test'))
        if self.armed:
            self.velocity=[0.,0.,.5] if self.landing else list(self.tx_velocity)
            self.position=[p+v*dt for p,v in zip(self.position,self.velocity)]
            if self.landing and self.position[2]>=0.:
                self.position[2]=0.; self.armed=False; self.velocity=[0.,0.,0.]
        self.feed_fc()
        self.depth()
        self.c._on_timer()
        ns=int(self.clock.now*1e9); snap=self.b._route_state.snapshot(ns)
        if self.fault == 'altitude_skew' and self.position[2] < -self.height-.15 and self.fault_started is None:
            self.fault_started = self.clock.now
        if self.fault == 'altitude_skew' and self.fault_started is not None and self.clock.now-self.fault_started < .1:
            snap = replace(snap, time_boot_ms=snap.time_boot_ms+159)
        self.b._stream_setpoint(snap,ns); self.b._drain_command_queue(snap,ns)
        if self.b._last_tx_terminal_brake and not hasattr(self,'first_terminal_brake_ns'):
            self.first_terminal_brake_ns=self.b._last_setpoint_tx_ns
        lines=self.b._transport.journal.getvalue().splitlines()
        for line in lines[self.offset:]:
            item=json.loads(line)
            if item['kind']=='tx_route_setpoint_local_ned': self.tx_velocity=item['velocity_ned_mps']
            if item['kind']=='tx_route_vehicle_command':
                code=item['command']; self.command_ids.append(code)
                if code==176:self.offboard=True
                if code==400:self.armed=True
                if code==21:self.landing=True; self.offboard=False
                self.b._publish_mavlink_ack(dict(command=code,result=0,progress=100,result_param2=0,target_system=245,target_component=191),1,1)
        self.offset=len(lines)
        # Delivery of fresh output proof after this tick's actual write.
        self.b._publish_state(); self.c._publish_control_state()
        self.m._route_home_received_at=self.clock.now

    def close(self):
        self.executor.shutdown(timeout_sec=2.); self.worker.join(2.)
        self.wire.destroy_node()


@pytest.mark.parametrize('height,fault',[(1.,'static_obstacle'),(1.,'none'),(2.,'none'),(1.,'yaw_reset'),(2.,'yaw_reset'),(1.,'short_depth_gap'),
    (1.,'long_depth_gap'),(1.,'metadata_reorder'),(1.,'rc'),(1.,'takeoff_camera_gap'),
    (1.,'battery_transient'),(2.,'battery_transient'),(1.,'battery_sustained'),(2.,'battery_sustained'),
    (1.,'home_delayed'),(2.,'home_delayed'),(1.,'altitude_skew'),(2.,'altitude_skew')])
def test_real_depth_dds_action_controller_bridge_memory_fc(bridge,monkeypatch,tmp_path,height,fault):
    monkeypatch.setattr(Controller,'_heartbeat_loop',lambda self:None)
    c=Controller(); clock=Clock(); chain=None
    try:
        with patch('time.monotonic',side_effect=lambda:clock.now),patch('time.monotonic_ns',side_effect=lambda:int(clock.now*1e9)),patch('time.sleep',side_effect=clock.sleep):
            m=manager_fixture(); m._scenario_enabled=True; m._max_laps=10; m._max_duration_minutes=10
            m.get_clock=c.get_clock
            m._ready_timeout=5.; m._takeoff_timeout=20.; m._waypoint_timeout=120.; m._landing_timeout=15.; m._sequence=0
            m._manual_active=m._resume_required=m._resume_requested=m._emergency_land_requested=False
            m._terminal_deadlines={}; m._active_waypoint=m._active_total_waypoints=0
            m._event_control=NS(clear_for_mission=lambda _:None)
            m._active_flight_profile='NORMAL'; m._active_mission_kind='ROUTE'
            m._active_altitude_reference=None; m._low_settle_since=None
            m.get_logger=lambda:NS(warning=lambda *a:None,error=lambda *a:print('MANAGER',*a,flush=True))
            m._publish_feedback=lambda *a:None
            bind(m,Manager,'_execute','_execute_impl','_cleanup_execution','_snapshot_state',
                '_vehicle_ready','_evaluate_vehicle_readiness','_wait_for','_active_low_speed_error_locked',
                '_low_target_reached','_publish_command','_complete_low_speed_terminal','_low_speed_terminal_outcome',
                '_confirmed_landed_disarmed','_land_current','_can_issue_terminal_land','_abort_result',
                '_finish_wait_failure','_execute_exception_result','_execution_interrupted',
                '_on_vehicle_state','_on_vehicle_geo_state','_on_altitude_reference_state','_on_low_speed_safety_state')
            inputs=m._scenario_inputs=Inputs(m); inputs.pool.shutdown(); inputs.pool=InlinePool()
            m._scenario_photo_root=tmp_path/'photos'
            chain=Chain(bridge,c,m,clock,height,fault)
            clock.advance=chain.advance
            for _ in range(65): clock.sleep(.05)
            response=NS(accepted=False,proposal_id='',message='')
            m._on_prepare_forward_test(NS(operator_id='test',target_altitude_home_m=height),response)
            assert response.accepted,response.message
            proposal=m._proposal_publisher.messages[-1]; m._proposals[proposal.proposal_id]=proposal
            m._update_route_frame_status(proposal.proposal_id)
            a=approve(m,proposal.proposal_id); assert a.accepted,a.message
            approval=MissionApproval(proposal_id=proposal.proposal_id,mission_id=a.mission_id,approved=True)
            m._scenario_registry.proposal(proposal); m._scenario_registry.approval(approval)
            for node in (bridge,c):
                if node is c and fault=='metadata_reorder':
                    chain.delayed_metadata=(proposal,approval)
                else:
                    node._scenario_registry.proposal(proposal); node._scenario_registry.approval(approval)
                (node._on_mission_approval if node is bridge else node._on_approval)(approval)
            req=goal(m,proposal.proposal_id,a.mission_id)
            m._on_goal(req)
            gh=NS(request=req,is_cancel_requested=False,is_active=True,succeed=lambda:None,abort=lambda:None,canceled=lambda:None)
            original=tmp_path/'original.jpg'; original.write_bytes(b'\xff\xd8detector fixture\xff\xd9')
            def step(dt):
                chain.advance(dt)
                inputs.frame=(chain.count,clock.now,b'\xff\xd8stopped fixture\xff\xd9')
                if chain.position[0]>8.:
                    inputs.event=dict(type='LITTERING',input_received=clock.now,event_id='event',capture_status='SAVED',
                        mission_id=req.mission_id,photo_path=str(original),photo_sha256=hashlib.sha256(original.read_bytes()).hexdigest())
            clock.advance=step
            result=m._execute(gh)
            if fault=='battery_sustained':
                assert not result.success
                assert 21 in chain.command_ids and not chain.armed,(result.message,c._last_error,bridge._last_output_detail)
                bridge._journal.flush()
                # Only this offline harness waits for diagnostic persistence.
                # The flight callbacks themselves must never wait for disk.
                if hasattr(bridge._journal, 'drain'):
                    bridge._journal.drain(timeout_s=1.)
                records=[json.loads(line) for line in Path(bridge._journal.name).read_text().splitlines()]
                latch=next(x for x in records if x.get('event')=='battery_terminal_latched')
                first=next(x for x in records if x.get('event')=='battery_policy_transition' and x['state']=='observe')
                assert 5. <= (latch['monotonic_ns']-first['monotonic_ns'])/1e9 <=5.1
                assert 0 <= chain.first_terminal_brake_ns-latch['monotonic_ns'] <=150_000_000
                assert c._terminal_completion_confirmed
                assert c._terminal_state != 4  # Successful fresh completion, not fallback.
                return
            if fault in ('long_depth_gap','rc'):
                assert not result.success
                assert chain.fault_started is not None
                if fault=='long_depth_gap': assert 21 in chain.command_ids and not chain.armed
                else: assert c._autonomy_resume_required
                return
            assert result.success,(result.message,c._last_error,bridge._last_output_detail,chain.position,chain.command_ids)
            assert chain.command_ids==[176,400,21]
            if fault=='home_delayed':
                assert chain.fault_started is not None
                assert not bridge._route_state.home_epoch_failure_latched
            if fault=='yaw_reset':
                assert chain.yaw_reset_injected
                assert hasattr(c,'_scenario_yaw_reset_at')
                assert bridge._route_state.reset_counter==1
                # Frozen heading remains zero for every prepared route target.
                assert all(abs(target.yaw_rad)<1e-6 for target in chain.targets if target.command in (0,1,2))
            assert chain.initial_extent_missing
            assert chain.position[0]>8.5 and not chain.armed
            assert (tmp_path/'photos'/req.mission_id/'event/stopped.jpg').is_file()
            assert all(abs(v)<=.50001 for v in chain.tx_velocity)
    finally:
        if chain:chain.close()
        c.destroy_node()

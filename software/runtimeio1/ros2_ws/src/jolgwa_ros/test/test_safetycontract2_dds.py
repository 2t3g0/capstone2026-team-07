"""Actual localhost DDS service/topic delivery through a multithreaded executor.

Only isolated Nodes and production callbacks are used. No bridge transport,
PX4 topics, physical services or flight process is started.
"""
import copy
import math
import threading
import time
from types import SimpleNamespace as NS

import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from jolgwa_interfaces.msg import (AltitudeReferenceState, FlightEnvelope,
                                   ManualOverride, OffboardCommand)
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint
from jolgwa_interfaces.srv import EmergencyLand
from jolgwa_ros.mission_manager_node import MissionManagerNode as Manager
from jolgwa_ros.px4_offboard_controller import Px4OffboardController as Controller
from jolgwa_ros.mavlink_usb_bridge_node import _OutputPairBuffer
from test_safety_contract import controller_fixture
from test_safetycontract2 import bind


@pytest.fixture
def dds():
    rclpy.init(args=[])
    node=rclpy.create_node('safetycontract2_isolated_test')
    executor=MultiThreadedExecutor(num_threads=2); executor.add_node(node)
    worker=threading.Thread(target=executor.spin,daemon=True); worker.start()
    yield node
    executor.shutdown(timeout_sec=2.)
    worker.join(2.)
    node.destroy_node(); rclpy.shutdown()


def deliver(node, publisher, message, event):
    deadline=time.monotonic()+3.
    while publisher.get_subscription_count()==0 and time.monotonic()<deadline:
        time.sleep(.01)
    assert publisher.get_subscription_count()>0
    event.clear(); publisher.publish(message)
    assert event.wait(2.), 'DDS callback did not complete'


def test_dds_reordered_output_topics_select_same_publication(dds):
    buffer = _OutputPairBuffer()
    event = threading.Event()
    for topic, message_type, receive in (
        ('/safetycontract2/pair/envelope', FlightEnvelope, buffer.add_envelope),
        ('/safetycontract2/pair/mode', OffboardControlMode, buffer.add_mode),
        ('/safetycontract2/pair/setpoint', TrajectorySetpoint, buffer.add_setpoint),
    ):
        dds.create_subscription(message_type, topic,
            lambda message, receive=receive: (receive(message, time.monotonic_ns()), event.set()), 10)
    ep = dds.create_publisher(FlightEnvelope, '/safetycontract2/pair/envelope', 10)
    mp = dds.create_publisher(OffboardControlMode, '/safetycontract2/pair/mode', 10)
    sp = dds.create_publisher(TrajectorySetpoint, '/safetycontract2/pair/setpoint', 10)
    base = time.time_ns()
    contract = NS(command=NS(mission_id='m', sequence=1),
                  output_epoch='e', output_sequence=1)

    def messages(source_ns, velocity):
        envelope = FlightEnvelope(mission_id='m', sequence=1, output_epoch='e',
            output_sequence=1, setpoint_kind=FlightEnvelope.SETPOINT_VELOCITY,
            expected_position_ned_m=[math.nan]*3,
            expected_velocity_ned_m_s=[velocity, 0., 0.], expected_yaw_rad=0.)
        envelope.stamp.sec = source_ns // 1_000_000_000
        envelope.stamp.nanosec = source_ns % 1_000_000_000
        mode = OffboardControlMode(timestamp=(source_ns + 1_000_000)//1000,
                                   velocity=True)
        setpoint = TrajectorySetpoint(timestamp=(source_ns + 2_000_000)//1000,
            position=[math.nan]*3, velocity=[velocity, 0., 0.], yaw=0.)
        return envelope, mode, setpoint

    e1, m1, s1 = messages(base, .1)
    e2, m2, s2 = messages(base + 50_000_000, .2)
    for publisher, message in ((ep,e1), (mp,m1), (ep,e2), (mp,m2), (sp,s1)):
        deliver(dds, publisher, message, event)
    selected = buffer.select(time.monotonic_ns(), contract)
    assert selected is not None and selected[1]['velocity'][0] == pytest.approx(.1)
    deliver(dds, sp, s2, event)
    selected = buffer.select(time.monotonic_ns(), contract)
    assert selected is not None and selected[1]['velocity'][0] == pytest.approx(.2)


def test_dds_old_sample_delivery_does_not_renew_received_time(dds):
    m=NS(_lock=threading.RLock(),_altitude_reference_state=None,_altitude_reference_state_received_at=0.)
    event=threading.Event()
    def receive(message):
        Manager._on_altitude_reference_state(m,message); event.set()
    dds.create_subscription(AltitudeReferenceState,'/safetycontract2/altitude',receive,10)
    pub=dds.create_publisher(AltitudeReferenceState,'/safetycontract2/altitude',10)
    deliver(dds,pub,AltitudeReferenceState(transport_epoch=1,sequence=10,local_age_ms=300,global_age_ms=300),event)
    received=m._altitude_reference_state_received_at
    for seq in (10,9):
        deliver(dds,pub,AltitudeReferenceState(transport_epoch=1,sequence=seq,local_age_ms=0),event)
        assert m._altitude_reference_state_received_at==received


def test_emergency_service_wakes_pause_without_action_cancel(dds):
    m=NS(_lock=threading.RLock(),_manual_active=True,_resume_requested=False,_resume_required=True,
        _active_goal=True,_active_mission_id='m',_active_proposal_id='p',_active_waypoint=0,
        _active_total_waypoints=1,_emergency_land_requested=False,_active_cancel_requested=False,
        _active_flight_profile='LOW_SPEED_1M_V1',_vehicle_state_received_at=time.monotonic(),
        _vehicle_state=NS(armed=True,connected=True,vehicle_status_fresh=True,arming_state_valid=True,
            command_output_enabled=True,emergency_land_available=True,active_execution_mission_id='m'),
        _publish_status=lambda *a:None,_publish_feedback=lambda *a:None,
        _require_explicit_resume=lambda **kw:None,_manual_publisher=NS(publish=lambda _:None),get_clock=dds.get_clock)
    bind(m,Manager,'_can_issue_terminal_land','_on_emergency_land')
    dds.create_service(EmergencyLand,'/safetycontract2/emergency',m._on_emergency_land)
    client=dds.create_client(EmergencyLand,'/safetycontract2/emergency')
    assert client.wait_for_service(timeout_sec=2.)
    m._vehicle_state_received_at=time.monotonic()
    gh=NS(request=NS(mission_id='m'),is_cancel_requested=False)
    outcomes=[]
    def pause(): outcomes.append(Manager._wait_for_explicit_resume(m,gh,return_phase=2,
        current_waypoint=0,total_waypoints=1,reason='manual',publish_hold=False))
    thread=threading.Thread(target=pause,daemon=True); thread.start()
    try:
        done=threading.Event()
        future=client.call_async(EmergencyLand.Request(mission_id='m',operator_id='test',reason='emergency'))
        future.add_done_callback(lambda _:done.set())
        assert done.wait(2.) and future.result().accepted
        thread.join(.5)
        assert outcomes==['cancel'] and not gh.is_cancel_requested
    finally:
        gh.is_cancel_requested=True; thread.join(.5)


def test_dds_land_before_release_retries_without_new_terminal_identity(dds):
    c=controller_fixture(); c._active_command.flight_profile=2
    c._gate=NS(armed=True,manual_override=False,connected=True,command_output_enabled=True,
        preflight_checks_pass=True,is_approved=lambda mission:mission=='m')
    c._offboard=True; c._landed=False; c._px4_failsafe_active=False
    c._terminal_requires_owned_offboard=True; c._external_mode_fenced=False
    c._autonomy_reentry_required=False; c._autonomy_resume_required=False
    c._last_px4_message_at=c._last_landed_message_at=time.monotonic()
    c._status_timeout_s=.5; c._last_sequence={'m':7}; c._retired_mission_ids=set()
    c._sim_clock_allows_output=lambda:True; c._local_navigation_fresh=lambda _:True
    c._receipt_age_usable=lambda *a:True; c._reset_avoidance_recovery=lambda:None
    c.get_logger=lambda:NS(warning=lambda *a:None,error=lambda *a:None)
    bind(c,Controller,'_low_speed_profile','_terminal_home_position_valid','_owns_terminal_handoff','_on_command','_on_manual_override')
    event=threading.Event()
    def manual(message): c._on_manual_override(message); event.set()
    def command(message): c._on_command(message); event.set()
    dds.create_subscription(ManualOverride,'/safetycontract2/manual',manual,10)
    dds.create_subscription(OffboardCommand,'/safetycontract2/command',command,10)
    mp=dds.create_publisher(ManualOverride,'/safetycontract2/manual',10)
    cp=dds.create_publisher(OffboardCommand,'/safetycontract2/command',10)
    land=copy.deepcopy(c._active_command); land.command=3; land.sequence=8
    deliver(dds,mp,ManualOverride(active=True,source='test'),event)
    deliver(dds,cp,land,event)
    assert c._terminal_started_at is None
    deliver(dds,mp,ManualOverride(active=False,source='test'),event)
    deliver(dds,cp,land,event)
    identity=(c._terminal_started_at,c._output_contract.output_sequence)
    assert c._active_command.command==3
    deliver(dds,cp,land,event)
    assert identity==(c._terminal_started_at,c._output_contract.output_sequence)

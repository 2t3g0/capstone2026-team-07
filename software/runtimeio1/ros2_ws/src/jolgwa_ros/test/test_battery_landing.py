"""Production callback/transport regression; memory serial only, no flight hardware."""
import copy
import os
import time
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
from rclpy.executors import SingleThreadedExecutor
from jolgwa_interfaces.msg import OffboardCommand, FlightOutputState, VehicleControlState
from px4_msgs.msg import VehicleCommandAck
from jolgwa_ros.px4_offboard_controller import Px4OffboardController as Controller
from jolgwa_ros.flight_contract import BATTERY_TERMINAL_ONLY, BATTERY_TERMINAL_UNAVAILABLE
from jolgwa_ros.low_speed import ALTITUDE_REFERENCE_MAX_ERROR_M
from test_safety_contract import bridge, pair, controller_fixture, proof
from test_safetycontract2 import bind
from test_vehicle_command_transport import memory_transport, fc_snapshot


def controller(height=1, strict=True):
    c = controller_fixture()
    cmd = c._active_command
    cmd.flight_profile = (OffboardCommand.FLIGHT_PROFILE_LOW_SPEED_1M_V1 if height == 1
                          else OffboardCommand.FLIGHT_PROFILE_LOW_SPEED_2M_V1)
    cmd.position_ned_m = [0., 0., -float(height)]
    cmd.altitude_reference = OffboardCommand.ALTITUDE_REFERENCE_FC_HOME_ALIGNED_V2
    cmd.altitude_reference_epoch = cmd.altitude_reference_sequence = 1
    cmd.altitude_reference_max_error_m = ALTITUDE_REFERENCE_MAX_ERROR_M
    c._activate_output_contract(cmd)
    c._gate = NS(armed=True, connected=True, manual_override=False,
        command_output_enabled=True, preflight_checks_pass=False, is_approved=lambda m: m == 'm')
    c._offboard = True
    c._landed = c._native_land = c._px4_failsafe_active = c._external_mode_fenced = False
    c._autonomy_reentry_required = c._autonomy_resume_required = False
    c._terminal_requires_owned_offboard = strict
    c._local_navigation_fresh = lambda now: True
    c._status_timeout_s = .5
    c._last_px4_message_at = c._last_landed_message_at = time.monotonic()
    c._retired_mission_ids = set()
    c._sent_vehicle_commands = set()
    c.get_logger = lambda: NS(error=lambda _: None)
    c.commands = []
    c._send_vehicle_command = lambda cmd: c.commands.append(cmd)
    bind(c, Controller, '_try_battery_terminal', '_battery_terminal_fallback',
        '_owns_terminal_handoff', '_terminal_home_position_valid', '_receipt_age_usable',
        '_supervise_terminal', '_prearm_terminal_confirmed', '_latch_terminal_restart_required',
        '_on_context_ack', '_on_command_ack', '_send_vehicle_command')
    return c


def install(b, c):
    b._approved_missions.add('m')
    b._on_offboard_command(c._active_command)
    b._on_output_contract(c._output_contract)
    c._contract_publisher = NS(publish=b._on_output_contract)
    b._transport = memory_transport()
    b._transport.bad_data = 0
    c._command_request_publisher = NS(publish=b._on_command_request)
    b._context_ack_publisher = NS(publish=c._on_context_ack)
    b._command_graph_ready = lambda: True


def output(b, snapshot, c=None):
    messages, statuses = [], []
    b._route_state.snapshot = lambda ns: snapshot
    b._flight_output_state_publisher = NS(publish=messages.append)
    b._vehicle_status_publisher = NS(publish=statuses.append)
    b._publish_state()
    if c: c._on_flight_output_state(messages[-1])
    return messages[-1], statuses[-1]


def brake_pair(b, now):
    pair(b, now)
    e, cmd = b._envelope, b._contract.command
    e.output_epoch = b._contract.output_epoch
    e.profile = cmd.flight_profile
    for field in ('home_z_ned_m', 'altitude_reference', 'altitude_reference_epoch',
        'altitude_reference_sequence', 'altitude_reference_max_error_m',
        'altitude_reference_local_time_boot_ms', 'altitude_reference_global_time_boot_ms'):
        setattr(e, field, getattr(cmd, field))


def battery_snapshot(**changes):
    return fc_snapshot(armed=True, offboard=True, landed=False, preflight_checks_pass=False,
        failsafe=True, battery_health_terminal_only=True,
        sensors_present=7|(1<<25), sensors_enabled=7|(1<<25), sensors_health=7,
        sensors_failed=1<<25,
        sensor_health_reason=BATTERY_TERMINAL_ONLY, **changes)


@pytest.mark.parametrize('height,camera', [(1,True),(1,False),(2,True),(2,False)])
@pytest.mark.parametrize('strict', [True, False])
def test_battery_to_new_brake_land_ack_and_fresh_completion(bridge, height, camera, strict):
    b, c = bridge, controller(height, strict)
    c._forward_test_camera_required = camera
    c._activate_output_contract(c._active_command)
    install(b, c)
    base = time.monotonic_ns()
    snap = battery_snapshot()
    with patch('time.monotonic_ns', return_value=base), patch('time.monotonic', return_value=base/1e9):
        old = proof(c, base, brake=True)
        b._stream_setpoint(snap, base)
        assert not b._transport.packets
        diagnostic, status = output(b, snap, c)
        assert diagnostic.detail == BATTERY_TERMINAL_ONLY
        assert not status.failsafe and not status.pre_flight_checks_pass
        assert not c._try_battery_terminal(base/1e9)
        first_deadline = c._terminal_started_at
        identity = c._output_contract.output_sequence
        c._on_flight_output_state(old)
        assert not c._terminal_brake_transmitted
        assert not c._begin_terminal(copy.deepcopy(c._active_command), base/1e9+.1)
        assert c._terminal_started_at == first_deadline
    now = base + 50_000_000
    with patch('time.monotonic_ns', return_value=now), patch('time.monotonic', return_value=now/1e9):
        brake_pair(b, now)
        b._stream_setpoint(snap, now)
        assert len(b._transport.packets) == 1, b._last_output_detail
        if os.environ.get('JOLGWA_TEST_PYMAVLINK') == '1':
            decoded=b._transport.common.MAVLink(None).parse_buffer(b._transport.packets[-1])[0]
            assert decoded.get_type()=='SET_POSITION_TARGET_LOCAL_NED'
            assert decoded.type_mask==2503 and (decoded.vx,decoded.vy,decoded.vz)==(0.,0.,0.)
        assert now - base <= 150_000_000
        diagnostic, _ = output(b, snap, c)
        assert diagnostic.detail == BATTERY_TERMINAL_ONLY and diagnostic.terminal_brake
        assert c._terminal_brake_transmitted and c._output_contract.output_sequence == identity
        c._last_px4_message_at = c._last_landed_message_at = now/1e9
        c._supervise_terminal(now/1e9)
        b._drain_command_queue(snap, now)
        assert 21 in b._pending_acks
        if os.environ.get('JOLGWA_TEST_PYMAVLINK') == '1':
            decoded=b._transport.common.MAVLink(None).parse_buffer(b._transport.packets[-1])[0]
            assert decoded.get_type()=='COMMAND_LONG' and decoded.command==21
        # Battery recovery keeps the same terminal contract and first cause.
        recovered = replace(snap, preflight_checks_pass=True, failsafe=False,
                            battery_health_terminal_only=False, sensor_health_reason='ready')
        assert output(b, recovered)[0].detail == BATTERY_TERMINAL_ONLY
        ack = dict(command=21, result=5, progress=50, result_param2=0, target_system=245, target_component=191)
        b._publish_mavlink_ack(ack, 1, 1)
        assert c._terminal_state == VehicleControlState.TERMINAL_PENDING
        assert c._terminal_ack_in_progress
        ack['result'] = 0
        b._publish_mavlink_ack(ack, 1, 1)
        assert c._terminal_state == VehicleControlState.TERMINAL_ACCEPTED
        assert not c._terminal_completion_confirmed
        c._landed = True; c._gate.armed = False
        c._last_px4_message_at = c._last_landed_message_at = now/1e9
        c._supervise_terminal(now/1e9)
        assert c._terminal_completion_confirmed
        assert 'battery' in c._first_fault


@pytest.mark.parametrize('fault', ['old_mission','old_epoch','old_sequence','old_source','stale','future','unknown'])
def test_diagnostic_cannot_grant_authority_without_current_identity(fault):
    c = controller()
    now = time.monotonic_ns()
    m = proof(c, now); m.detail = BATTERY_TERMINAL_ONLY
    if fault == 'old_mission': m.mission_id = 'old'
    if fault == 'old_epoch': m.output_epoch = 'old'
    if fault == 'old_sequence': m.output_sequence -= 1
    if fault == 'old_source': m.sequence -= 1
    if fault == 'stale': m.published_monotonic_ns -= 151_000_000
    if fault == 'future': m.published_monotonic_ns += 1_000_000_000
    if fault == 'unknown': m.detail += ':unknown'
    c._on_flight_output_state(m)
    c._try_battery_terminal(now/1e9)
    assert c._terminal_started_at is None


@pytest.mark.parametrize('fault', ['rc','position','system','stale_sys','epoch','approval','battery_missing','battery_disabled'])
def test_hard_fault_revokes_exception_permanently(bridge, fault):
    b, c = bridge, controller()
    install(b,c); now = time.monotonic_ns(); snap = battery_snapshot()
    assert b._observe_battery_health(snap, now)
    changes = dict(battery_health_terminal_only=False)
    if fault == 'rc': changes['offboard'] = False
    if fault == 'position': changes['position_valid'] = False
    if fault == 'epoch': changes['transport_epoch'] = 2
    if fault == 'approval': b._approved_missions.clear()
    if fault in ('battery_missing','battery_disabled'):
        changes.update(preflight_checks_pass=True,failsafe=False)
        changes['sensors_present' if fault=='battery_missing' else 'sensors_enabled']=7
    bad = replace(snap, **changes)
    assert not b._observe_battery_health(bad, now+1)
    b._stream_setpoint(bad, now+1)
    assert not b._transport.packets
    assert not b._observe_battery_health(snap, now+2)
    assert b._last_output_detail == BATTERY_TERMINAL_UNAVAILABLE


def test_dds_battery_diagnostic_preempts_navigation(bridge):
    b, c = bridge, controller()
    install(b,c)
    snap=battery_snapshot()
    for timer in b.timers: timer.cancel()
    executor = SingleThreadedExecutor(); executor.add_node(b)
    topic = '/battery_land_test/diagnostic'
    sub = b.create_subscription(FlightOutputState, topic, c._on_flight_output_state, 10)
    pub = b.create_publisher(FlightOutputState, topic, 10)
    try:
        deadline = time.monotonic()+3
        while not pub.get_subscription_count() and time.monotonic()<deadline:
            executor.spin_once(timeout_sec=.01)
        assert pub.get_subscription_count()
        started = time.monotonic()
        m, _ = output(b,snap)
        pub.publish(m)
        while getattr(c, '_battery_terminal_pending', None) is None and time.monotonic()-started<.15:
            executor.spin_once(timeout_sec=.005)
        assert c._battery_terminal_pending is not None
        c._try_battery_terminal(time.monotonic())
        assert c._active_command.command == OffboardCommand.COMMAND_LAND
        assert time.monotonic()-started<.15
        assert not c._terminal_brake_transmitted
        brake_pair(b,time.monotonic_ns())
        b._stream_setpoint(snap,time.monotonic_ns())
        elapsed_ms=(b._last_setpoint_tx_ns/1e9-started)*1000
        assert b._last_tx_terminal_brake and len(b._transport.packets)==1, b._last_output_detail
        assert 0<=elapsed_ms<=150
        print('BATTERY_DDS_FIRST_BRAKE_MS=%.3f' % elapsed_ms)
    finally:
        b.destroy_subscription(sub); b.destroy_publisher(pub)
        executor.remove_node(b); executor.shutdown()


@pytest.mark.parametrize('fault', ['rc','position','failsafe','stale_status','approval'])
@pytest.mark.parametrize('strict', [True, False])
def test_controller_refuses_handoff_and_never_reacquires(fault, strict):
    c = controller(strict=strict)
    now = time.monotonic_ns()
    m = proof(c, now); m.detail = BATTERY_TERMINAL_ONLY
    with patch('time.monotonic_ns', return_value=now), patch('time.monotonic', return_value=now/1e9):
        c._on_flight_output_state(m)
        if fault == 'rc': c._external_mode_fenced = True
        if fault == 'position': c._local_navigation_fresh = lambda _: False
        if fault == 'failsafe': c._px4_failsafe_active = True
        if fault == 'stale_status': c._last_px4_message_at = now/1e9-1
        if fault == 'approval': c._gate.is_approved = lambda _: False
        assert c._try_battery_terminal(now/1e9)
        assert c._terminal_started_at is None
    with patch('time.monotonic_ns', return_value=now+151_000_000):
        c._try_battery_terminal(now/1e9+.151)
    assert c._terminal_state == VehicleControlState.TERMINAL_FALLBACK
    assert c._active_command is None and c._autonomy_reentry_required


@pytest.mark.parametrize('case', ['prearm','normal'])
def test_battery_does_not_relax_other_modes_or_prearm(bridge, case):
    b, c = bridge, controller()
    if case == 'normal':
        c._active_command.flight_profile = OffboardCommand.FLIGHT_PROFILE_NORMAL
        c._activate_output_contract(c._active_command)
    install(b,c); now=time.monotonic_ns()
    snap = battery_snapshot()
    if case == 'prearm': snap=replace(snap, armed=False, offboard=False, landed=True)
    diagnostic, status = output(b,snap)
    assert status.failsafe and not status.pre_flight_checks_pass
    assert diagnostic.detail != BATTERY_TERMINAL_ONLY
    for cmd in (176,400):
        c._send_vehicle_command(cmd, param1=1., **({'param2':6.} if cmd==176 else {}))
        b._drain_command_queue(snap,now)
    assert not b._transport.packets and not b._pending_acks
    c._low_speed_land('test terminal', now/1e9)
    brake_pair(b, now)
    b._stream_setpoint(snap, now)
    c._send_vehicle_command(21)
    b._drain_command_queue(snap,now)
    assert not b._transport.packets and not b._pending_acks


def test_mixed_battery_fault_cannot_enter_unlatched_terminal(bridge):
    b,c=bridge,controller(); install(b,c); now=time.monotonic_ns()
    c._low_speed_land('mixed fault',now/1e9)
    snap=replace(battery_snapshot(),battery_health_terminal_only=False,sensors_failed=(1<<25)|1)
    brake_pair(b,now); b._stream_setpoint(snap,now)
    c._send_vehicle_command(21); b._drain_command_queue(snap,now)
    assert not b._transport.packets


def test_failure_diagnostic_before_warning_cannot_be_reversed():
    c=controller(); now=time.monotonic_ns()
    failed=proof(c,now); failed.detail=BATTERY_TERMINAL_UNAVAILABLE
    older=proof(c,now-1); older.detail=BATTERY_TERMINAL_ONLY
    c._on_flight_output_state(failed)
    c._on_flight_output_state(older)
    assert c._terminal_state == VehicleControlState.TERMINAL_FALLBACK
    assert c._active_command is None


def test_warning_recovery_cannot_resume_old_navigation_contract(bridge):
    b,c=bridge,controller(); install(b,c); now=time.monotonic_ns()
    snap=battery_snapshot()
    diagnostic,_=output(b,snap,c)
    recovered=replace(snap,preflight_checks_pass=True,failsafe=False,battery_health_terminal_only=False,
                      sensors_health=7|(1<<25),sensors_failed=0)
    b._stream_setpoint(recovered,now)
    assert not b._transport.packets
    moving=copy.deepcopy(c._active_command); moving.command=OffboardCommand.COMMAND_GOTO
    Controller._on_command(c,moving)
    assert c._active_command.command==OffboardCommand.COMMAND_TAKEOFF
    assert 'navigation rejected' in c._last_error


@pytest.mark.parametrize('scenario', ['stale_brake','brake_timeout','lost_ack','progress_timeout','failed','cancelled','lost_state'])
@patch('time.monotonic', new=lambda: 100.)
@patch('time.monotonic_ns', new=lambda: 100_000_000_000)
def test_battery_terminal_keeps_existing_failure_contracts(scenario):
    # A fixed clock makes the exact 90-second boundary independent of float
    # rounding in the host uptime. Production timeout behavior is unchanged.
    c=controller(); base=time.monotonic_ns(); now=base/1e9
    with patch('time.monotonic_ns',return_value=base), patch('time.monotonic',return_value=now):
        m=proof(c,base); m.detail=BATTERY_TERMINAL_ONLY
        c._on_flight_output_state(m); c._try_battery_terminal(now)
    requests=[]; c._command_request_publisher=NS(publish=requests.append)
    if scenario=='stale_brake':
        with patch('time.monotonic_ns',return_value=base+151_000_000):
            c._on_flight_output_state(proof(c,base,brake=True))
        assert not c._terminal_brake_transmitted
        return
    if scenario!='brake_timeout':
        with patch('time.monotonic_ns',return_value=base): c._on_flight_output_state(proof(c,base,brake=True))
        c._supervise_terminal(now)
    if scenario in ('failed','cancelled','progress_timeout'):
        c._on_command_ack(VehicleCommandAck(command=21,result=5))
        assert c._terminal_ack_in_progress and c._terminal_state==VehicleControlState.TERMINAL_PENDING
        if scenario in ('failed','cancelled'):
            c._on_command_ack(VehicleCommandAck(command=21,result=4 if scenario=='failed' else 6))
            assert c._terminal_state==VehicleControlState.TERMINAL_FALLBACK
            return
    elapsed=90. if scenario=='progress_timeout' else 1.1 if scenario=='brake_timeout' else 0.6 if scenario=='lost_state' else 3.1
    for dt in ([1.,2.,3.1] if scenario=='lost_ack' else [elapsed]):
        if scenario!='lost_state': c._last_px4_message_at=c._last_landed_message_at=now+dt
        c._supervise_terminal(now+dt)
    assert c._terminal_state in (VehicleControlState.TERMINAL_FALLBACK,VehicleControlState.TERMINAL_FAILED)
    if scenario=='progress_timeout': assert len(requests)==1

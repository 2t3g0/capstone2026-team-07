"""Use production bridge/controller callbacks with an in-memory transport.

No USB is opened, no executor spins and no simulation handshake is bypassed.
This is a deterministic state harness, not a claim of a PX4 SITL flight.
"""
import copy
import io
import math
import threading
import time
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import rclpy
from jolgwa_interfaces.msg import (FlightControlContract, FlightCommandRequest,
    FlightEnvelope, FlightOutputState, OffboardCommand, VehicleControlState)
from jolgwa_ros.mavlink_usb_bridge_node import MavlinkUsbBridgeNode as Bridge
from jolgwa_ros.px4_offboard_controller import Px4OffboardController as Controller
from jolgwa_ros.mission_manager_node import MissionManagerNode as Manager
from jolgwa_ros.operator_gateway_node import OperatorGatewayNode as Gateway
from jolgwa_ros.low_speed import AltitudeReferenceSynchronizer


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rclpy.init(args=[])
    b = Bridge()
    b._flight_output_requested = True
    writes = []
    b._transport = NS(flight_output_enabled=True, close=lambda: None,
        send_vehicle_command=lambda *args: writes.append(('command', args)),
        send_setpoint=lambda *args, **kw: writes.append(('setpoint', args, kw)) or True)
    b._private_command_graph_ready = lambda now: True
    b.writes = writes
    b._publish_local_ack = lambda *args: None
    route = b._route_state
    route.home_position = dict(latitude_deg=35., longitude_deg=129.,
        altitude_m=40., ned=(0., 0., 0.))
    route.latest_px4_home_position = dict(route.home_position)
    route.global_position = dict(latitude_deg=35., longitude_deg=129., altitude_m=40.)
    route.local_position = dict(position=(0., 0., 0.))
    assert route.lock_execution_home('m')[0]
    yield b
    b.destroy_node()
    rclpy.shutdown()


def contract(b, sequence=1, code=1):
    cmd = OffboardCommand(mission_id='m', sequence=sequence, command=code,
        flight_output_handshake_version=8,
        approved=True, position_ned_m=[0., 0., -1.], yaw_rad=0.)
    b._approved_missions.add('m')
    b._on_offboard_command(cmd)
    c = FlightControlContract(output_epoch='e', output_sequence=sequence, command=cmd, handshake_version=8)
    b._on_output_contract(c)
    return c


def pair(b, now):
    b._mode = dict(position=False, velocity=True, acceleration=False,
        attitude=False, body_rate=False, thrust_and_torque=False, direct_actuator=False)
    b._setpoint = dict(position=(math.nan,)*3, velocity=(0., 0., 0.), yaw=0.,
        acceleration=(math.nan,)*3, jerk=(math.nan,)*3, yawspeed=math.nan)
    b._mode_received_ns = b._setpoint_received_ns = b._envelope_received_ns = now
    b._envelope = FlightEnvelope(mission_id='m', sequence=b._command_sequence,
        output_epoch='e', output_sequence=b._contract.output_sequence,
        setpoint_kind=FlightEnvelope.SETPOINT_VELOCITY,
        expected_position_ned_m=[math.nan]*3,
        expected_velocity_ned_m_s=[0., 0., 0.], expected_yaw_rad=0.)
    b._envelope.stamp.sec = now // 1_000_000_000
    b._envelope.stamp.nanosec = now % 1_000_000_000
    b._output_pairs.modes.append((now, now + 1_000_000, b._mode))
    b._output_pairs.setpoints.append((now, now + 2_000_000, b._setpoint))
    b._output_pairs.envelopes.append((now, now, b._envelope))


def state():
    return NS(connected=True, armed=False, offboard=False, landed=True,
        position_valid=True, position=(0., 0., 0.), px4_custom_mode=0, native_land=False)


def request(b, command, now):
    c = b._contract
    return FlightCommandRequest(mission_id='m', source_sequence=c.command.sequence,
        output_epoch=c.output_epoch, output_sequence=c.output_sequence,
        request_id=f'{command}-{now}', issued_monotonic_ns=now, command=command,
        params=[1., 6. if command == 176 else 0., 0., 0., 0., 0., 0.])


@pytest.mark.parametrize('defect', ['missing', 'expired', 'epoch', 'sequence', 'value', 'graph'])
@pytest.mark.parametrize('code', [176, 400])
def test_final_write_revalidates_contract(bridge, defect, code):
    b = bridge
    contract(b)
    now = time.monotonic_ns()
    pair(b, now)
    b._tx_run_started_ns = now-2_000_000_000
    b._last_setpoint_tx_ns = now
    b._last_envelope_valid = True  # Previous success must not authorize this write.
    if defect == 'missing':
        b._envelope = None
        b._output_pairs.envelopes.clear()
    if defect == 'expired':
        b._envelope_received_ns = now-300_000_000
        b._output_pairs.envelopes[-1] = (now-300_000_000, now, b._envelope)
    if defect == 'epoch': b._envelope.output_epoch = 'old'
    if defect == 'sequence': b._envelope.output_sequence = 88
    if defect == 'value': b._envelope.expected_velocity_ned_m_s = [1., 0., 0.]
    if defect == 'graph': b._private_command_graph_ready = lambda now: False
    b._commands.append(request(b, code, now))
    b._drain_command_queue(state(), now)
    assert not b.writes


def test_actual_tx_warmup_then_offboard_before_arm(bridge):
    b = bridge
    contract(b)
    base = time.monotonic_ns()
    s = state()
    for i in range(21):
        now = base+i*50_000_000
        pair(b, now)
        b._stream_setpoint(s, now)
        b._commands.append(request(b, 176, now))
        b._drain_command_queue(s, now)
        if i < 20:
            assert not [w for w in b.writes if w[0] == 'command']
    assert [w[1][0] for w in b.writes if w[0] == 'command'] == [176]
    b._commands.append(request(b, 400, now))
    b._drain_command_queue(s, now)
    assert [w[1][0] for w in b.writes if w[0] == 'command'] == [176]
    s.offboard = True
    b._commands.append(request(b, 400, now+1))
    b._drain_command_queue(s, now+1)
    assert [w[1][0] for w in b.writes if w[0] == 'command'] == [176, 400]
    # A gap starts a new evidence interval, regardless of old successful output.
    now += 200_000_000
    pair(b, now)
    b._stream_setpoint(s, now)
    b._commands.append(request(b, 176, now))
    b._drain_command_queue(s, now)
    assert [w[1][0] for w in b.writes if w[0] == 'command'] == [176, 400]


def test_delayed_setpoint_uses_its_own_envelope_not_latest_tick(bridge):
    b = bridge
    contract(b)
    base = time.monotonic_ns()
    pair(b, base)
    first_setpoint = b._setpoint
    first_envelope = b._envelope
    first_setpoint['velocity'] = (0.10, 0., 0.)
    b._envelope.expected_velocity_ned_m_s = [0.10, 0., 0.]
    b._output_pairs.setpoints.clear()  # First setpoint is delayed in DDS.

    pair(b, base + 50_000_000)
    second_setpoint = b._setpoint
    second_setpoint['velocity'] = (0.20, 0., 0.)
    b._envelope.expected_velocity_ned_m_s = [0.20, 0., 0.]
    b._output_pairs.setpoints.clear()  # New envelope arrives before either setpoint.

    b._setpoint = first_setpoint
    b._setpoint_received_ns = base + 60_000_000
    b._output_pairs.setpoints.append((base + 60_000_000, base + 2_000_000,
                                       first_setpoint))
    b._stream_setpoint(state(), base + 70_000_000)
    assert b._last_output_detail == 'ready'
    assert len(b.writes) == 1
    assert b.writes[-1][1][0].velocity[0] == pytest.approx(0.10)

    b._setpoint = second_setpoint
    b._setpoint_received_ns = base + 80_000_000
    b._output_pairs.setpoints.append((base + 80_000_000, base + 52_000_000,
                                       second_setpoint))
    b._stream_setpoint(state(), base + 90_000_000)
    assert b._last_output_detail == 'ready'
    assert len(b.writes) == 2
    assert b.writes[-1][1][0].velocity[0] == pytest.approx(0.20)
    assert b._consecutive_setpoint_tx == 2

    # A duplicate of the older complete pair arrives after the newer write.
    b._setpoint = first_setpoint
    b._setpoint_received_ns = base + 100_000_000
    b._output_pairs.setpoints.append((base + 100_000_000, base + 2_000_000,
                                       first_setpoint))
    b._stream_setpoint(state(), base + 110_000_000)
    assert b.writes[-1][1][0].velocity[0] == pytest.approx(0.20)
    assert len(b.writes) == 3

    # Once the newer pair is lost, the old source cannot renew output proof.
    b._output_pairs.setpoints.clear()
    b._output_pairs.envelopes.clear()
    b._output_pairs.setpoints.append((base + 120_000_000, base + 2_000_000,
                                       first_setpoint))
    b._output_pairs.envelopes.append((base + 120_000_000, base,
                                       first_envelope))
    b._stream_setpoint(state(), base + 130_000_000)
    assert len(b.writes) == 3
    assert not b._last_envelope_valid


def test_output_pair_is_fenced_by_contract_transition_and_lease(bridge):
    b = bridge
    previous = contract(b)
    base = time.monotonic_ns()
    pair(b, base)
    assert b._output_pairs.select(base + 249_999_999, previous) is not None
    assert b._output_pairs.select(base + 250_000_000, previous) is None
    old_envelope = b._envelope
    old_setpoint = b._setpoint
    old_mode = b._mode

    current = FlightControlContract(output_epoch='e', output_sequence=2,
                                    command=previous.command, handshake_version=8)
    b._on_output_contract(current)
    assert b._output_pairs.select(base + 1, current) is None
    b._output_pairs.modes.append((base + 10, base + 1_000_000, old_mode))
    b._output_pairs.setpoints.append((base + 10, base + 2_000_000,
                                       old_setpoint))
    b._output_pairs.envelopes.append((base + 10, base, old_envelope))
    b._stream_setpoint(state(), base + 20)
    assert not b.writes
    assert not b._last_envelope_valid


@pytest.mark.parametrize('source_offset_ns,allowed', [
    (40_000_000, True), (40_000_001, False), (-1, False),
])
def test_output_pair_source_span_boundary(bridge, source_offset_ns, allowed):
    b = bridge
    current = contract(b)
    now = time.monotonic_ns()
    pair(b, now)
    b._output_pairs.setpoints[-1] = (
        now, now + source_offset_ns, b._setpoint)
    assert (b._output_pairs.select(now, current) is not None) == allowed


def controller_fixture():
    c = NS(_state_lock=threading.RLock(), _output_contract=None,
        _output_epoch='controller', _output_sequence=0, _pending_request_ids={},
        _usb_output_contract=True, _flight_output_state=None,
        _flight_output_received_at=0., _flight_output_ready_since=None,
        _first_fault='', _arm_transmitted=False, _low_speed_fault_since=None,
        _terminal_started_at=None, _terminal_brake_started_at=None,
        _terminal_command_sent=False, _terminal_state=0,
        _yaw_rad=0., _gate=NS(armed=True, simulation_only=True), _warmup_s=1.,
        _publish_velocity_setpoint=lambda *a: None,
        _contract_publisher=NS(publish=lambda m: None))
    for name in ('_activate_output_contract', '_begin_terminal', '_low_speed_land', '_on_flight_output_state',
                 '_flight_output_ready'):
        setattr(c, name, getattr(Controller, name).__get__(c))
    c._is_terminal_command = Controller._is_terminal_command
    c._is_low_speed_command = Controller._is_low_speed_command
    c._active_command = OffboardCommand(mission_id='m', sequence=7, command=1, approved=True,
        flight_output_handshake_version=8)
    c._activate_output_contract(c._active_command)
    return c


def proof(c, ns, count=1, run=1, brake=False):
    return FlightOutputState(mission_id='m', sequence=7,
        output_epoch=c._output_contract.output_epoch,
        output_sequence=c._output_contract.output_sequence,
        transport_connected=True, command_graph_ready=True, envelope_valid=True,
        setpoint_transmitted=True, consecutive_transmissions=count,
        tx_sequence=count, tx_run_id=run, tx_run_duration_ms=(count-1)*50,
        last_tx_monotonic_ns=ns, published_monotonic_ns=ns,
        setpoint_kind=FlightEnvelope.SETPOINT_VELOCITY, terminal_brake=brake)


def test_low_speed_terminal_ownership_does_not_depend_on_preflight_or_home_target():
    c = controller_fixture()
    c._active_command.flight_profile = OffboardCommand.FLIGHT_PROFILE_LOW_SPEED_1M_V1
    c._gate = NS(armed=True, manual_override=False, connected=True,
        preflight_checks_pass=False, is_approved=lambda mission: mission == 'm')
    c._offboard = True
    c._px4_failsafe_active = False
    c._position_ned = [100., 100., 100.]
    c._last_px4_message_at = time.monotonic()
    c._status_timeout_s = 0.5
    c._local_navigation_fresh = lambda now: True
    c._receipt_age_usable = lambda *args: True
    c._terminal_home_position_valid = Controller._terminal_home_position_valid.__get__(c)
    assert Controller._owns_terminal_handoff(c, time.monotonic(), 'm')
    landing = copy.deepcopy(c._active_command)
    landing.command = OffboardCommand.COMMAND_LAND
    assert c._terminal_home_position_valid(landing)
    c._gate.manual_override = True
    assert not Controller._owns_terminal_handoff(c, time.monotonic(), 'm')
    c._gate.manual_override = False
    c._active_command.flight_profile = OffboardCommand.FLIGHT_PROFILE_NORMAL
    assert not Controller._owns_terminal_handoff(c, time.monotonic(), 'm')


def test_internal_terminal_has_new_identity_and_rejects_old_brake_proof():
    c = controller_fixture()
    now = time.monotonic_ns()
    old = proof(c, now, 25)
    c._low_speed_land('original fault', now/1e9)
    identity = c._output_contract.output_sequence
    assert identity == 2 and c._active_command.sequence == 7
    c._on_flight_output_state(old)
    assert not c._terminal_brake_transmitted
    c._low_speed_land('repeated fault', now/1e9+.5)
    assert c._output_contract.output_sequence == identity
    assert c._terminal_started_at == now/1e9
    assert c._first_fault == 'original fault'
    c._on_flight_output_state(proof(c, time.monotonic_ns(), brake=True))
    assert c._terminal_brake_transmitted


def test_success_response_gap_requires_new_second_even_in_simulation():
    c = controller_fixture()
    base = time.monotonic_ns()
    for i in range(22):
        ns = base+i*50_000_000
        with patch('time.monotonic', return_value=ns/1e9), patch('time.monotonic_ns', return_value=ns):
            c._on_flight_output_state(proof(c, ns, i+1))
    with patch('time.monotonic_ns', return_value=ns):
        assert c._flight_output_ready(c._active_command, ns/1e9)
    ns += 200_000_000
    with patch('time.monotonic', return_value=ns/1e9), patch('time.monotonic_ns', return_value=ns):
        c._on_flight_output_state(proof(c, ns, 30))
    with patch('time.monotonic_ns', return_value=ns):
        assert not c._flight_output_ready(c._active_command, ns/1e9)


def test_unchanged_home_at_20hz_cannot_renew_confirmation_deadline():
    sync = AltitudeReferenceSynchronizer()
    start = time.monotonic_ns()
    meta = dict(frozen_altitude_amsl_m=40., frozen_z_ned_m=0.,
        current_altitude_amsl_m=40., current_z_ned_m=0., correction_valid=True,
        correction_revision=0, opposition_error_m=0., estimator_reset_counter_valid=True,
        estimator_reset_counter=0, detail='locked')
    sync.set_home_reference(**meta)
    epoch = sync.transport_epoch
    for i in range(61):
        ns = start+i*50_000_000
        sync.add_local(time_boot_ms=1000+i*50, z_ned_m=0., received_ns=ns)
        sync.add_global(time_boot_ms=1000+i*50, fc_altitude_home_relative_m=0. if i<42 else 1.08,
            global_altitude_amsl_m=40., received_ns=ns, now_ns=ns)
        sync.set_home_reference(**meta)
        observed = sync.snapshot(ns)
    assert sync.transport_epoch == epoch+1
    assert observed.altitude_epoch_failure_latched
    assert observed.home_correction_state == 3


@pytest.mark.parametrize('home_first', [True, False])
def test_home_global_order_and_duplicate_metadata_keep_normalized_altitude(home_first):
    sync = AltitudeReferenceSynchronizer()
    base = time.monotonic_ns()
    meta = dict(frozen_altitude_amsl_m=40., frozen_z_ned_m=0.,
        current_altitude_amsl_m=40., current_z_ned_m=0., correction_valid=True,
        correction_revision=0, opposition_error_m=0., estimator_reset_counter_valid=True,
        estimator_reset_counter=0, detail='locked')
    sync.set_home_reference(**meta)
    def sample(i, raw):
        ns = base+i*50_000_000
        sync.add_local(time_boot_ms=1000+i*50, z_ned_m=0., received_ns=ns)
        return sync.add_global(time_boot_ms=1000+i*50, fc_altitude_home_relative_m=raw,
            global_altitude_amsl_m=40., received_ns=ns, now_ns=ns)
    for i in range(41): observed = sample(i, 0.)
    epoch = sync.transport_epoch
    corrected = {**meta, 'current_altitude_amsl_m':38.919678,
        'current_z_ned_m':1.080322, 'correction_revision':1}
    with patch('time.monotonic_ns', return_value=base+41*50_000_000):
        if home_first: sync.set_home_reference(**corrected)
        observed = sample(41, 1.080322)
        if not home_first: sync.set_home_reference(**corrected)
    for i in range(42, 49):
        with patch("time.monotonic_ns", return_value=base+i*50_000_000):
            sync.set_home_reference(**corrected)
            observed = sample(i, 1.080322)
    assert sync.transport_epoch == epoch
    assert observed.state == 1
    assert observed.home_correction_state == 2
    assert observed.normalized_fc_altitude_home_relative_m == pytest.approx(0., abs=1e-5)
    assert not sync.set_home_reference(**meta)
    assert sync._home_correction_revision == 1


def test_gateway_reconnect_preserves_old_receipt_age():
    out = []
    g = NS(_last_vehicle_state=None, _last_vehicle_state_at=0.,
        _last_vehicle_geo=None, _last_vehicle_geo_at=0., _last_low_speed_safety=None,
        _last_low_speed_safety_at=0., _approval_ledger=NS(invalidate=lambda: None), _send=out.append)
    message = VehicleControlState(connected=True, arming_state_valid=True)
    with patch('time.monotonic', return_value=10.): Gateway._on_vehicle_state(g, message)
    with patch('time.monotonic', return_value=40.): Gateway._on_vehicle_state(g, message, replay=True)
    assert out[-1]['gateway_vehicle_age_ms'] == 30000
    assert not out[-1]['emergency_land_available']


def test_manager_does_not_use_stale_ground_state():
    m = NS(_lock=threading.RLock(), _vehicle_state=VehicleControlState(landed=True),
        _vehicle_state_received_at=time.monotonic()-30.)
    assert Manager._snapshot_state(m) is None


@pytest.mark.parametrize('manager_first', [True, False])
def test_manager_and_controller_terminal_race_preserves_original_source(bridge, manager_first):
    b = bridge
    original = contract(b).command
    internal_land = copy.deepcopy(original)
    internal_land.command = 3
    later_manager_land = copy.deepcopy(internal_land)
    later_manager_land.sequence = 2
    terminal = FlightControlContract(output_epoch='e', output_sequence=2,
        handshake_version=8, command=internal_land)
    if manager_first: b._on_offboard_command(later_manager_land)
    b._on_output_contract(terminal)
    if not manager_first: b._on_offboard_command(later_manager_land)
    assert b._contract_valid()
    assert b._command_sequence == 1
    assert b._contract.output_sequence == 2


def test_old_handshake_cannot_install_output_contract(bridge):
    legacy = FlightControlContract(output_epoch='old', output_sequence=1,
        command=OffboardCommand(mission_id='m', approved=True))
    bridge._on_output_contract(legacy)
    assert bridge._contract is None


def test_controller_starts_and_heartbeats_without_missing_initialization():
    import os
    import subprocess
    import sys
    env = dict(os.environ, ROS_DOMAIN_ID='232', ROS_LOCALHOST_ONLY='1')
    result = subprocess.run([sys.executable, '-c',
        'import rclpy,time; from jolgwa_ros.px4_offboard_controller import Px4OffboardController; '
        'rclpy.init(args=[]); n=Px4OffboardController(); time.sleep(0.3); '
        'n.destroy_node(); rclpy.shutdown(); print("HEARTBEAT_OK")'],
        env=env, capture_output=True, text=True, timeout=12)
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'HEARTBEAT_OK' in result.stdout


def terminal_fixture():
    c = controller_fixture()
    c._low_speed_land('fault', 10.)
    c._landed = False
    c._last_px4_message_at = c._last_landed_message_at = 10.
    c._external_mode_fenced = c._native_land = False
    c._offboard = True
    c._gate.connected = c._gate.command_output_enabled = True
    c._gate.manual_override = False
    c._gate.is_approved = lambda mission: mission == 'm'
    c._local_navigation_fresh = lambda now: True
    c._retired_mission_ids = set()
    c.commands = []
    c._send_vehicle_command = lambda command: c.commands.append(command)
    for name in ('_supervise_terminal', '_prearm_terminal_confirmed', '_latch_terminal_restart_required'):
        setattr(c, name, getattr(Controller, name).__get__(c))
    return c


def terminal_tick(c, now):
    c._last_px4_message_at = c._last_landed_message_at = now
    assert c._supervise_terminal(now)


def test_delayed_heartbeat_does_not_skip_brake():
    c = terminal_fixture()
    terminal_tick(c, 11.1)
    assert c.commands == []
    assert c._terminal_state == VehicleControlState.TERMINAL_FALLBACK
    assert c._restart_required_fault


def test_land_retry_budget_is_exactly_three_and_late_ack_cannot_restore_success():
    c = terminal_fixture()
    c._terminal_brake_transmitted = True
    for now in (10., 10.5, 11., 11.5, 12., 12.5, 13.): terminal_tick(c, now)
    assert c.commands == [21, 21, 21]
    assert c._terminal_state == VehicleControlState.TERMINAL_FALLBACK
    Controller._on_command_ack(c, NS(command=21, result=0))
    assert c._terminal_state == VehicleControlState.TERMINAL_FALLBACK


@pytest.mark.parametrize('native', [True, False])
def test_native_land_or_rc_edge_does_not_disable_timeout_supervisor(native):
    c = terminal_fixture()
    c._external_mode_fenced = True
    c._native_land = native
    terminal_tick(c, 10.5)
    assert not c.commands
    if not native: assert c._terminal_state == VehicleControlState.TERMINAL_FALLBACK
    terminal_tick(c, 100.)
    assert c._terminal_state == VehicleControlState.TERMINAL_FAILED


def test_landed_before_ack_is_failed_after_grace():
    c = terminal_fixture()
    c._arm_transmitted = True
    c._gate.armed = False
    c._landed = True
    terminal_tick(c, 10.1)
    assert c._terminal_state == VehicleControlState.TERMINAL_PENDING
    terminal_tick(c, 11.2)
    assert c._terminal_state == VehicleControlState.TERMINAL_FALLBACK
    assert not c.commands


@pytest.mark.parametrize('profile,height', [(2, 1.), (1, 2.)])
def test_low_speed_brake_survives_epoch_failure_but_movement_does_not(bridge, profile, height):
    b = bridge
    base = time.monotonic_ns()
    for i in range(21):
        ns = base+i*100_000_000
        b._altitude_reference_sync.add_local(time_boot_ms=1000+i*100, z_ned_m=0., received_ns=ns)
        altitude = b._altitude_reference_sync.add_global(time_boot_ms=1000+i*100,
            fc_altitude_home_relative_m=0., global_altitude_amsl_m=40., received_ns=ns, now_ns=ns)
    cmd = OffboardCommand(mission_id='m', sequence=1, command=1, approved=True,
        flight_output_handshake_version=8,
        flight_profile=profile, position_ned_m=[0., 0., -height], yaw_rad=0.,
        altitude_reference=2, altitude_reference_max_error_m=1.5,
        altitude_reference_epoch=altitude.transport_epoch, altitude_reference_sequence=altitude.sequence,
        altitude_reference_local_time_boot_ms=3000, altitude_reference_global_time_boot_ms=3000)
    b._approved_missions.add('m')
    b._on_offboard_command(cmd)
    b._on_output_contract(FlightControlContract(output_epoch='e', output_sequence=1, command=cmd, handshake_version=8))
    s = state()
    s.position_source = 'LOCAL_POSITION_NED'
    s.position_received_ns, s.time_boot_ms = ns, 3000
    s.transport_epoch = altitude.transport_epoch
    def low_pair():
        pair(b, ns)
        b._envelope.profile = profile
        b._envelope.altitude_reference = 2
        b._envelope.altitude_reference_max_error_m = 1.5
        b._envelope.altitude_reference_epoch = altitude.transport_epoch
        b._envelope.altitude_reference_sequence = altitude.sequence
        b._envelope.altitude_reference_local_time_boot_ms = 3000
        b._envelope.altitude_reference_global_time_boot_ms = 3000
    low_pair()
    b._stream_setpoint(s, ns)
    assert len(b.writes) == 1, b._last_output_detail
    b._altitude_reference_sync.reject_home_correction('estimator_reset')
    b._stream_setpoint(s, ns)
    assert len(b.writes) == 1
    # New terminal identity preserves the frozen epoch, not the failed current epoch.
    terminal = copy.deepcopy(cmd)
    terminal.command = 3
    b._on_output_contract(FlightControlContract(output_epoch='e', output_sequence=2, command=terminal, handshake_version=8))
    s.armed = s.offboard = True
    s.landed = False
    low_pair()
    b._stream_setpoint(s, ns)
    assert len(b.writes) == 2, b._last_output_detail
    assert b.writes[-1][2]['terminal_brake']


@pytest.mark.parametrize('height,camera', [(1., True), (1., False), (2., True), (2., False)])
def test_preparation_to_terminal_and_explicit_home_release_harness(bridge, height, camera):
    """Production callbacks + contract validator; FC observations are injected.

    Separate from PX4 SITL: this checks orchestration, not flight dynamics or
    the complete blocking Action execute loop.
    """
    from test_provisional_home_flow import manager_fixture, approve, goal, _Publisher
    from rclpy.action import GoalResponse
    from jolgwa_interfaces.msg import MissionApproval
    b = bridge
    m = manager_fixture()
    m._forward_test_camera_required = camera
    m._vehicle_state.jetson_safety_state = 'CLEAR'
    m._vehicle_state.jetson_safety_fresh = True
    m._vehicle_state.low_speed_obstacle_guard_enabled = True
    prepared = NS(accepted=False, proposal_id='', message='')
    m._on_prepare_forward_test(NS(operator_id='test', target_altitude_home_m=height), prepared)
    assert prepared.accepted, prepared.message
    proposal = m._proposal_publisher.messages[-1]
    m._proposals[proposal.proposal_id] = proposal
    m._update_route_frame_status(proposal.proposal_id)
    approved = approve(m, proposal.proposal_id)
    assert approved.accepted, approved.message
    req = goal(m, proposal.proposal_id, approved.mission_id)
    assert m._on_goal(req) == GoalResponse.ACCEPT
    snapshot = m._goal_reservation['snapshot']
    mission = approved.mission_id
    b._route_state.execution_home_mission_id = mission
    b._on_mission_approval(MissionApproval(mission_id=mission, approved=True))
    c = controller_fixture()
    c._active_command = None
    c._gate.armed = False
    c._gate.connected = c._gate.command_output_enabled = True
    c._gate.manual_override = False
    c._gate.is_approved = lambda value: value == mission
    c._last_sequence = {}
    c._sim_clock_allows_output = lambda: True
    c._reset_avoidance_recovery = lambda: None
    c._autonomy_reentry_required = c._autonomy_resume_required = False
    c._terminal_requires_owned_offboard = False
    c._external_mode_fenced = False
    c._forward_test_bypass_anchor = None
    c._altitude_reference_state = m._altitude_reference_state
    c._low_speed_profile = Controller._low_speed_profile
    c._contract_publisher = NS(publish=b._on_output_contract)
    c._command_request_publisher = NS(publish=b._on_command_request)
    c._sent_vehicle_commands = set()
    c._on_command = Controller._on_command.__get__(c)
    c._send_vehicle_command = Controller._send_vehicle_command.__get__(c)
    m._sequence = 0
    m._active_flight_profile = snapshot.flight_profile
    m._active_mission_kind = snapshot.mission_kind
    m._active_home_z_ned_m = snapshot.aligned_home_z_ned_m
    m._active_altitude_reference = dict(transport_epoch=snapshot.altitude_reference_epoch,
        sequence=snapshot.altitude_reference_sequence,
        local_time_boot_ms=snapshot.altitude_reference_local_time_boot_ms,
        global_time_boot_ms=snapshot.altitude_reference_global_time_boot_ms)
    def dispatch(cmd):
        b._on_offboard_command(cmd)
        c._on_command(cmd)
    m._command_publisher = NS(publish=dispatch)
    Manager._publish_command(m, mission, 1, (0., 0., -height), 0.)
    assert c._active_command is not None
    s = state()
    s.position_source = 'LOCAL_POSITION_NED'
    s.transport_epoch = snapshot.altitude_reference_epoch
    base = time.monotonic_ns()
    # A synchronized authority sample at each test tick. Never bypass the
    # production envelope/source/epoch validator with simulation_only.
    alignment = NS(**vars(m._altitude_reference_state))
    alignment.home_correction_pending_age_ms = 0
    alignment.home_altitude_correction_m = 0.
    alignment.home_correction_state = 0
    b._altitude_reference_sync.snapshot = lambda now: alignment
    def transmit(ns):
        s.position_received_ns = ns
        s.time_boot_ms = alignment.local_time_boot_ms
        pair(b, ns)
        e = b._envelope
        cmd = c._active_command
        e.mission_id = mission
        e.sequence = cmd.sequence
        e.output_epoch = c._output_contract.output_epoch
        e.output_sequence = c._output_contract.output_sequence
        e.profile = cmd.flight_profile
        for field in ('altitude_reference', 'altitude_reference_max_error_m', 'altitude_reference_epoch',
                      'altitude_reference_sequence', 'altitude_reference_local_time_boot_ms',
                      'altitude_reference_global_time_boot_ms', 'home_z_ned_m'):
            setattr(e, field, getattr(cmd, field))
        b._stream_setpoint(s, ns)
        assert b._last_envelope_valid, b._last_output_detail
        p = FlightOutputState(mission_id=mission, sequence=cmd.sequence,
            output_epoch=e.output_epoch, output_sequence=e.output_sequence,
            transport_connected=True, command_graph_ready=True, envelope_valid=True,
            setpoint_transmitted=True, consecutive_transmissions=b._consecutive_setpoint_tx,
            tx_sequence=b._tx_sequence, tx_run_id=b._tx_run_id,
            tx_run_duration_ms=(ns-b._tx_run_started_ns)//1_000_000,
            last_tx_monotonic_ns=b._last_setpoint_tx_ns, published_monotonic_ns=ns,
            setpoint_kind=2, terminal_brake=b._last_tx_terminal_brake)
        with patch('time.monotonic', return_value=ns/1e9), patch('time.monotonic_ns', return_value=ns):
            c._on_flight_output_state(p)
    for i in range(22): transmit(base+i*50_000_000)
    now = base+21*50_000_000
    with patch('time.monotonic_ns', return_value=now):
        assert c._flight_output_ready(c._active_command, now/1e9)
    with patch('time.monotonic_ns', return_value=now):
        c._send_vehicle_command(176, param1=1., param2=6.)
    b._drain_command_queue(s, now)
    assert [w[1][0] for w in b.writes if w[0] == 'command'] == [176]
    s.offboard = True
    now += 50_000_000
    transmit(now)
    with patch('time.monotonic_ns', return_value=now): c._send_vehicle_command(400, param1=1.)
    b._drain_command_queue(s, now)
    assert [w[1][0] for w in b.writes if w[0] == 'command'] == [176, 400]
    s.armed = c._gate.armed = True
    s.landed = False
    Manager._publish_command(m, mission, 2, snapshot.route_ned[0], 0.)
    transmit(now)
    c._begin_terminal(copy.copy(OffboardCommand(**{
        field:getattr(c._active_command, field) for field in c._active_command.get_fields_and_field_types()
        if field != 'command'}, command=3)), now/1e9)
    transmit(now)
    assert c._terminal_brake_transmitted
    with patch('time.monotonic_ns', return_value=now): c._send_vehicle_command(21)
    b._drain_command_queue(s, now)
    assert [w[1][0] for w in b.writes if w[0] == 'command'] == [176, 400, 21]
    # Actual contextual ACK -> terminal observer -> Manager terminal outcome.
    c._on_command_ack = Controller._on_command_ack.__get__(c)
    c.get_logger = lambda: NS(error=lambda _: None)
    b._context_ack_publisher = NS(publish=Controller._on_context_ack.__get__(c))
    with patch('time.monotonic_ns', return_value=now+1), patch('time.monotonic', return_value=(now+1)/1e9):
        b._publish_mavlink_ack(dict(command=21, result=0, target_system=245, target_component=191), 1, 1)
    assert c._terminal_state == VehicleControlState.TERMINAL_ACCEPTED
    observed = VehicleControlState(connected=True, vehicle_status_fresh=True, arming_state_valid=True,
        landed_state_valid=True, landed=True, armed=False, terminal_state=c._terminal_state)
    assert Manager._low_speed_terminal_outcome(observed, ever_armed=True) == 'ACCEPTED'
    s.armed, s.landed = False, True
    b._synchronize_home_lifecycle(s, now)
    assert b._route_state.home_locked
    m._terminal_deadlines = {}
    m._event_control = NS(clear_for_mission=lambda _: None)
    m._publish_approval = lambda proposal, mid, approved, operator: b._on_mission_approval(
        MissionApproval(proposal_id=proposal, mission_id=mid, approved=approved))
    Manager._cleanup_execution(m, req)
    b._synchronize_home_lifecycle(s, now)
    assert not b._route_state.home_locked

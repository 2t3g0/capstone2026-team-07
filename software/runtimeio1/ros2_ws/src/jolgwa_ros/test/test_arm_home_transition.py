"""Production ARM write, Home synchronization and controller altitude gates.

Only the serial endpoint is in memory. No physical ports or flight services.
"""
import time
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
from jolgwa_interfaces.msg import AltitudeReferenceState
from jolgwa_ros.px4_offboard_controller import Px4OffboardController as Controller
from jolgwa_ros.low_speed import AltitudeReferenceSynchronizer
from jolgwa_uav.mavlink_route_fc_link import RouteFcState
from test_safety_contract import bridge
from test_battery_landing import controller, install, brake_pair


def feed(route, sync, ns, boot, *, armed=False, correction=0.):
    messages = {
        'HEARTBEAT': dict(autopilot=12, type=2, base_mode=129 if armed else 1,
                          custom_mode=6<<16, system_status=4 if armed else 3),
        'SYS_STATUS': dict(onboard_control_sensors_present=7,
            onboard_control_sensors_enabled=7, onboard_control_sensors_health=7),
        'ESTIMATOR_STATUS': dict(time_usec=boot*1000, flags=47),
        'ATTITUDE_QUATERNION': dict(time_boot_ms=boot, q1=1., q2=0., q3=0., q4=0.),
        'LOCAL_POSITION_NED': dict(time_boot_ms=boot, x=0., y=0., z=0., vx=0., vy=0., vz=0.),
        'GLOBAL_POSITION_INT': dict(time_boot_ms=boot, lat=350000000, lon=1290000000,
                                   alt=40000, relative_alt=round(correction*1000)),
        'ODOMETRY': dict(time_usec=boot*1000, frame_id=1, child_frame_id=12, reset_counter=0,
            x=0., y=0., z=0., vx=0., vy=0., vz=0., q=[1., 0., 0., 0.]),
        'EXTENDED_SYS_STATE': dict(landed_state=1),
    }
    for kind, data in messages.items():
        assert route.accept(kind, data, 1, 1, ns), route.last_rejection
    sync.add_local(time_boot_ms=boot, z_ned_m=0., received_ns=ns)
    sync.add_global(time_boot_ms=boot, fc_altitude_home_relative_m=correction,
                    global_altitude_amsl_m=40., received_ns=ns, now_ns=ns)


def test_horizontal_home_metadata_refinement_confirms_without_altitude_shift():
    sync = AltitudeReferenceSynchronizer()
    route = RouteFcState(transport_epoch=sync.transport_epoch, home_stability_ns=0)
    base = time.monotonic_ns()
    home = dict(latitude=350000000, longitude=1290000000, altitude=40000,
                x=0., y=0., z=0.)
    assert route.accept('HOME_POSITION', home, 1, 1, base)
    for index in range(45):
        now = base+index*50_000_000
        with patch('time.monotonic_ns', return_value=now):
            feed(route, sync, now, 1000+index*50, armed=index > 0)
            if index == 1:
                assert route.lock_execution_home('horizontal-mission')[0]
                assert sync.set_home_reference(**route.home_correction_snapshot())
    now += 50_000_000
    refined = {**home, 'latitude': home['latitude']+90, 'x': 1.0019}
    with patch('time.monotonic_ns', return_value=now):
        assert route.accept('HOME_POSITION', refined, 1, 1, now), route.last_rejection
        assert sync.set_home_reference(**route.home_correction_snapshot())
        assert sync.snapshot(now).home_correction_state == 1  # Await a new GLOBAL sample.
    now += 50_000_000
    with patch('time.monotonic_ns', return_value=now):
        feed(route, sync, now, 3300, armed=True)
        observed = sync.snapshot(now)
    assert not route.home_epoch_failure_latched
    assert not observed.altitude_epoch_failure_latched
    assert observed.home_correction_state != 1
    assert route.home_position['ned'] == (0., 0., 0.)


@pytest.mark.parametrize('height,camera', [(1, True), (1, False), (2, True), (2, False)])
@pytest.mark.parametrize('order', ['ack-home', 'home-ack', 'global-home-ack'])
@pytest.mark.parametrize('correction', [.065, 1.08])
def test_real_arm_write_home_before_heartbeat_without_false_land(bridge, height, camera, order, correction):
    b, c = bridge, controller(height)
    c._forward_test_camera_required = camera
    c._gate.armed = False
    c._gate.preflight_checks_pass = True
    c._arm_request_started_at = None
    c._terminal_command_id = 21
    c._on_command_ack = Controller._on_command_ack.__get__(c)
    c._altitude_reference_state_received_at = 0.
    install(b, c)
    sync = b._altitude_reference_sync = AltitudeReferenceSynchronizer()
    r = b._route_state = RouteFcState(transport_epoch=sync.transport_epoch, home_stability_ns=0)
    c._active_command.altitude_reference_epoch = sync.transport_epoch
    # Install the corrected epoch in the contract as well as the intent.
    c._activate_output_contract(c._active_command)
    b._on_offboard_command(c._active_command)
    b._on_output_contract(c._output_contract)
    base = time.monotonic_ns()
    home = dict(latitude=350000000, longitude=1290000000, altitude=40000,
                x=0., y=0., z=0.)
    assert r.accept('HOME_POSITION', home, 1, 1, base)
    for i in range(62):
        now = base+i*50_000_000
        with patch('time.monotonic_ns', return_value=now), patch('time.monotonic', return_value=now/1e9):
            feed(r, sync, now, 1000+i*50)
            if i == 1:
                assert r.lock_execution_home('m')[0]
                assert sync.set_home_reference(**r.home_correction_snapshot())
            if i >= 41:
                brake_pair(b, now)
                before = len(b._transport.packets)
                b._stream_setpoint(r.snapshot(now), now)
                assert len(b._transport.packets) == before+1, b._last_output_detail
    with patch('time.monotonic_ns', return_value=now), patch('time.monotonic', return_value=now/1e9):
        c._arm_request_started_at = now/1e9
        c._send_vehicle_command(400, param1=1.)
        b._drain_command_queue(r.snapshot(now), now)
        assert r._arm_home_transition is not None
        assert not r.snapshot(now).armed
        request = b._pending_acks[400][0]
    start = now
    for i, event in enumerate(order.split('-')):
        now = start+(i+1)*60_000_000
        with patch('time.monotonic_ns', return_value=now), patch('time.monotonic', return_value=now/1e9):
            if event == 'ack':
                b._publish_mavlink_ack(dict(command=400, result=0, progress=100,
                    result_param2=0, target_system=245, target_component=191), 1, 1)
            elif event == 'home':
                assert r.accept('HOME_POSITION', {**home, 'altitude':40000-correction*1000,
                    'z':correction}, 1, 1, now)
                b._synchronize_home_lifecycle(r.snapshot(now), now)
            else:
                feed(r, sync, now, 4110, correction=correction)
            observed = sync.snapshot(now)
            sample = AltitudeReferenceState(state=observed.state, valid=observed.valid,
                stable=observed.stable, transport_epoch=observed.transport_epoch,
                sequence=observed.sequence, local_age_ms=observed.local_age_ms,
                global_age_ms=observed.global_age_ms, source_skew_ms=observed.source_skew_ms,
                local_z_ned_m=observed.local_z_ned_m,
                fc_altitude_home_relative_m=observed.fc_altitude_home_relative_m,
                normalized_fc_altitude_home_relative_m=observed.normalized_fc_altitude_home_relative_m,
                home_correction_pending_age_ms=observed.home_correction_pending_age_ms)
            c._altitude_reference_state = sample
            c._altitude_reference_state_received_at = now/1e9
            valid, detail, *_ = Controller._altitude_reference_status(c, c._active_command, now/1e9)
            assert valid, detail
            brake_pair(b, now)
            before = len(b._transport.packets)
            b._stream_setpoint(r.snapshot(now), now)
            assert len(b._transport.packets) == before+1, b._last_output_detail
            assert not r.snapshot(now).armed
    now = start+240_000_000
    with patch('time.monotonic_ns', return_value=now), patch('time.monotonic', return_value=now/1e9):
        feed(r, sync, now, 4290, armed=True, correction=correction)
        b._synchronize_home_lifecycle(r.snapshot(now), now)
        assert r.snapshot(now).armed and not r.home_epoch_failure_latched
        assert sync.snapshot(now).stable
        assert sync.snapshot(now).normalized_fc_altitude_home_relative_m == pytest.approx(0., abs=1e-6)
        assert r.home_position['ned'][2] == 0.
    assert request.request_id not in c._pending_request_ids.values()
    assert c._first_fault == '' and c._terminal_started_at is None
    commands = [row for row in b._transport.journal.getvalue().splitlines() if 'tx_route_vehicle_command' in row]
    assert len(commands) == 1 and '"command": 400' in commands[0]


@pytest.mark.parametrize('defect', ['none', 'expired', 'denied', 'wrong-contract', 'rc', 'no-request', 'non-takeoff'])
def test_controller_pending_arm_correction_remains_bounded(defect):
    c = controller()
    c._gate.armed = False
    c._arm_request_started_at = 10.
    c._pending_request_ids = {}
    contract = c._output_contract
    c._arm_home_ack = (contract.output_epoch, contract.output_sequence, 0)
    c._altitude_reference_state = AltitudeReferenceState(
        state=AltitudeReferenceState.STATE_HOME_CORRECTION_PENDING,
        valid=True, stable=True, transport_epoch=c._active_command.altitude_reference_epoch,
        local_age_ms=0, global_age_ms=0, source_skew_ms=0,
        local_z_ned_m=0., fc_altitude_home_relative_m=0.,
        normalized_fc_altitude_home_relative_m=0., home_correction_pending_age_ms=50)
    now = 10.401 if defect == 'expired' else 10.1
    c._altitude_reference_state_received_at = now
    if defect == 'denied': c._arm_home_ack = (contract.output_epoch, contract.output_sequence, 2)
    if defect == 'wrong-contract': c._arm_home_ack = ('retired', contract.output_sequence, 0)
    if defect == 'rc': c._offboard = False
    if defect == 'no-request': c._arm_home_ack = None
    if defect == 'non-takeoff': c._active_command.command = 2
    valid, detail, *_ = Controller._altitude_reference_status(c, c._active_command, now)
    assert valid is (defect == 'none'), detail
    assert not c._gate.armed

"""Battery classification uses production MAVLink ingestion, never a PX4 flag guess."""
import pytest
import math
from dataclasses import replace
from test_mavlink_route_fc_link import (_healthy_state, _transport, _snapshot,
                                       _velocity_mode, _position_mode)
from jolgwa_uav.mavlink_route_fc_link import make_route_setpoint

BATTERY = 1 << 25

@pytest.mark.parametrize('fault', ['battery', 'battery_missing', 'disabled', 'mixed',
    'stale', 'missing', 'heartbeat', 'position', 'estimator', 'epoch', 'invalid_bits'])
def test_isolated_battery_requires_fresh_complete_health_and_navigation(fault):
    now = 2_000_000_000
    s = _healthy_state(now, armed=True, offboard=True)
    present = enabled = 7 | BATTERY
    health = 7
    if fault == 'battery_missing': present &= ~BATTERY
    if fault == 'disabled': enabled &= ~BATTERY
    if fault == 'mixed': health &= ~1
    if fault == 'invalid_bits': enabled |= 16
    assert s.accept('SYS_STATUS', dict(onboard_control_sensors_present=present,
        onboard_control_sensors_enabled=enabled, onboard_control_sensors_health=health), 1, 1, now)
    if fault == 'stale': s.received_ns['SYS_STATUS'] = now - 2_000_000_000
    if fault == 'missing': s.sys_status = None
    if fault == 'heartbeat': s.heartbeat['system_status'] = 5
    if fault == 'position':
        s.received_ns['LOCAL_POSITION_NED'] = 0
    if fault == 'estimator': s.estimator_status['flags'] = 0
    if fault == 'epoch': s.fault = 'transport_epoch_changed'
    snap = s.snapshot(now)
    assert snap.battery_health_terminal_only == (fault == 'battery')
    if fault != 'disabled':
        assert not snap.preflight_checks_pass and snap.failsafe
    if fault == 'battery':
        assert snap.sensor_health_reason == 'battery_health_terminal_only'
        assert snap.sensors_failed == BATTERY
        assert snap.sensor_health_age_ns == 0


@pytest.mark.parametrize('case', ['valid','position_mode','moving','nav_lost','offboard_lost','disarmed'])
def test_terminal_transport_checks_wire_mask_not_ros_nan(case):
    t = _transport()
    s = replace(_snapshot(), offboard=True, preflight_checks_pass=False, failsafe=True)
    mode = _position_mode() if case == 'position_mode' else _velocity_mode()
    p = make_route_setpoint(mode, dict(position=(0.,0.,0.) if case=='position_mode' else (math.nan,)*3,
        velocity=(.1,0.,0.) if case=='moving' else (0.,0.,0.), yaw=0., yawspeed=math.nan))
    assert p.position == (0.,0.,0.)
    if case=='nav_lost': s=replace(s,position_valid=False)
    if case=='offboard_lost': s=replace(s,offboard=False)
    if case=='disarmed': s=replace(s,armed=False)
    if case=='valid':
        assert t.send_setpoint(p,s,terminal_brake=True)
        assert len(t.port.writes)==1
    else:
        with pytest.raises(PermissionError): t.send_setpoint(p,s,terminal_brake=True)
        assert not t.port.writes

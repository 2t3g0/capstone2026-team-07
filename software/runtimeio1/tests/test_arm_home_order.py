"""ARM/Home reordering regression; no serial port or physical output."""
import pytest

from test_mavlink_route_fc_link import _healthy_state

BASE = 1_000_000_000
MISSION = 'arm-home-order'


def locked_state():
    s = _healthy_state(BASE, offboard=True, home_stability_ns=0)
    home = dict(latitude=352350126, longitude=1290748631,
                altitude=42800, x=1., y=2., z=-1.)
    assert s.accept('HOME_POSITION', home, 1, 1, BASE)
    for i in range(2):
        ns = BASE + i*10_000_000
        assert s.accept('LOCAL_POSITION_NED', dict(time_boot_ms=801+i,
            x=1., y=2., z=-3., vx=0., vy=0., vz=0.), 1, 1, ns)
        assert s.accept('GLOBAL_POSITION_INT', dict(time_boot_ms=801+i,
            lat=home['latitude'], lon=home['longitude'], alt=44800,
            relative_alt=2000), 1, 1, ns)
        assert s.accept('ODOMETRY', dict(time_usec=801000+i*1000,
            frame_id=1, child_frame_id=12, reset_counter=0,
            x=1., y=2., z=-3., vx=0., vy=0., vz=0.,
            q=[1., 0., 0., 0.]), 1, 1, ns)
    assert s.lock_execution_home(MISSION)[0]
    return s, {**home, 'altitude': 42735, 'z': -.935}


def heartbeat(s, ns, *, armed=True, offboard=True):
    return s.accept('HEARTBEAT', dict(autopilot=12, type=2,
        base_mode=129 if armed else 1, custom_mode=(6 if offboard else 3)<<16,
        system_status=4 if armed else 3), 1, 1, ns)


def test_arm_home_before_armed_heartbeat_keeps_frozen_frame():
    s, changed = locked_state()
    assert s.note_arm_transmitted(MISSION, 'request-1', BASE+20_000_000)
    assert s.accept('HOME_POSITION', changed, 1, 1, BASE+40_000_000), s.last_rejection
    assert not s.snapshot(BASE+40_000_001).armed  # ACK/intent is never armed evidence.
    assert s.home_position['ned'][2] == -1.
    assert s.home_correction_revision == 1
    assert heartbeat(s, BASE+80_000_000)
    assert s.snapshot(BASE+80_000_001).armed
    assert not s.home_epoch_failure_latched


@pytest.mark.parametrize('order', ['home-ack-heartbeat', 'ack-home-heartbeat', 'heartbeat-home-ack'])
def test_arm_home_message_orders(order):
    s, changed = locked_state()
    assert s.note_arm_transmitted(MISSION, 'request-1', BASE+20_000_000)
    for i, event in enumerate(order.split('-')):
        ns = BASE+(40+i*20)*1_000_000
        if event == 'home':
            assert s.accept('HOME_POSITION', changed, 1, 1, ns)
        elif event == 'ack':
            s.note_arm_ack('request-1', 0)
        else:
            assert heartbeat(s, ns)
    assert s.home_correction_revision == 1
    assert s.home_position['ned'][2] == -1.
    assert not s.home_epoch_failure_latched


@pytest.mark.parametrize('elapsed_ms,allowed', [(399, True), (400, True), (401, False)])
def test_armed_confirmation_boundary_and_late_heartbeat(elapsed_ms, allowed):
    s, changed = locked_state()
    start = BASE+20_000_000
    s.note_arm_transmitted(MISSION, 'request-1', start)
    assert s.accept('HOME_POSITION', changed, 1, 1, start+20_000_000)
    heartbeat(s, start+elapsed_ms*1_000_000)
    assert s.home_epoch_failure_latched is not allowed
    if not allowed:
        assert s.home_epoch_failure_detail == 'arm_home_transition_confirmation_timeout'
        s.note_arm_ack('request-1', 0)
        assert s.home_epoch_failure_latched


@pytest.mark.parametrize('result', [1, 2, 3, 4, 6])
def test_rejected_arm_cannot_leave_provisional_correction_authorized(result):
    s, changed = locked_state()
    s.note_arm_transmitted(MISSION, 'request-1', BASE+20_000_000)
    assert s.accept('HOME_POSITION', changed, 1, 1, BASE+40_000_000)
    s.note_arm_ack('unrelated', result)
    assert not s.home_epoch_failure_latched
    s.note_arm_ack('request-1', result)
    assert s.home_epoch_failure_latched
    assert not s.arm_home_transition_valid(BASE+60_000_000)


def test_repeated_arm_and_progress_do_not_extend_deadline():
    s, changed = locked_state()
    s.note_arm_transmitted(MISSION, 'request-1', BASE+20_000_000)
    assert s.accept('HOME_POSITION', changed, 1, 1, BASE+40_000_000)
    for ms in (50, 200, 399):
        assert not s.note_arm_transmitted(MISSION, 'request-2', BASE+ms*1_000_000)
        s.note_arm_ack('request-1', 5)
    s.snapshot(BASE+421_000_000)
    assert s.home_epoch_failure_latched


@pytest.mark.parametrize('defect', ['no-write', 'wrong-mission', 'epoch', 'rc', 'health', 'no-reset', 'reset', 'xy', 'z'])
def test_transition_does_not_bypass_existing_home_or_ownership_checks(defect):
    s, changed = locked_state()
    if defect != 'no-write':
        s.note_arm_transmitted('old' if defect == 'wrong-mission' else MISSION,
                               'request-1', BASE+20_000_000)
    if defect == 'epoch': s.transport_epoch += 1
    if defect == 'rc': heartbeat(s, BASE+30_000_000, armed=False, offboard=False)
    if defect == 'health': s.heartbeat['system_status'] = 5
    if defect == 'no-reset': s.reset_counter = None
    if defect == 'reset': s.reset_counter = 1
    if defect == 'xy': changed['x'] += 1.
    if defect == 'z': s.local_position['position'] = (1., 2., -4.)
    for i in range(3):
        s.accept('HOME_POSITION', changed, 1, 1, BASE+40_000_000+i)
    assert s.home_epoch_failure_latched
    assert s.home_correction_revision == 0
    assert s.home_position['ned'][2] == -1.


def test_release_and_disarm_cannot_reuse_transition():
    s, changed = locked_state()
    s.note_arm_transmitted(MISSION, 'request-1', BASE+20_000_000)
    heartbeat(s, BASE+30_000_000)
    heartbeat(s, BASE+40_000_000, armed=False)
    assert not s.accept('HOME_POSITION', changed, 1, 1, BASE+50_000_000)
    assert s.release_execution_home(MISSION)
    assert not s.arm_home_transition_valid(BASE+60_000_000)


def test_consistent_horizontal_home_metadata_refinement_keeps_frozen_frame():
    s, _ = locked_state()
    heartbeat(s, BASE+20_000_000)
    frozen = dict(s.home_position)
    changed = dict(latitude=352350216, longitude=1290748631,
                   altitude=42800, x=2.0019, y=2., z=-1.)
    assert s.accept('HOME_POSITION', changed, 1, 1, BASE+40_000_000), s.last_rejection
    assert s.home_position == frozen
    assert s.home_correction_revision == 1
    assert s.home_horizontal_global_delta_m > 0.9
    assert s.home_horizontal_consistency_error_m < 0.02
    assert not s.home_epoch_failure_latched


@pytest.mark.parametrize('defect', ['mismatched_xy', 'too_far', 'local_jump', 'global_jump'])
def test_horizontal_home_refinement_requires_independent_continuity(defect):
    s, _ = locked_state()
    heartbeat(s, BASE+20_000_000)
    changed = dict(latitude=352350216, longitude=1290748631,
                   altitude=42800, x=2.0019, y=2., z=-1.)
    if defect == 'mismatched_xy':
        changed['x'] = 1.
    elif defect == 'too_far':
        changed['latitude'] += 200
        changed['x'] += 2.225
    elif defect == 'local_jump':
        s.local_position['position'] = (2., 2., -3.)
    else:
        s.global_position['latitude_deg'] += 1.0/111_320.0
    for index in range(3):
        s.accept('HOME_POSITION', changed, 1, 1, BASE+40_000_000+index)
    assert s.home_epoch_failure_latched
    assert s.home_correction_revision == 0

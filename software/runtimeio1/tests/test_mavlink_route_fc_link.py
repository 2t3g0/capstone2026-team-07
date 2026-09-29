import io
import math
import struct
import threading

import pytest

from jolgwa_uav.mavlink_route_fc_link import (
    MAV_CMD_COMPONENT_ARM_DISARM,
    MAV_CMD_DO_SET_MODE,
    MAV_CMD_NAV_LAND,
    MAV_PARAM_TYPE_INT32,
    MAV_PARAM_TYPE_REAL32,
    POSITION_YAW_MASK,
    VELOCITY_YAW_MASK,
    VELOCITY_YAW_RATE_MASK,
    MavlinkRouteTransport,
    RouteFcSnapshot,
    RouteFcState,
    UsbTelemetryTransport,
    decode_px4_parameter_value,
    make_route_setpoint,
    normalize_parameter_id,
    route_estimator_reason,
    validate_route_command_ack,
)


def test_parameter_id_normalization_accepts_pymavlink_string_or_bytes():
    assert normalize_parameter_id("COM_OBL_RC_ACT\x00") == "COM_OBL_RC_ACT"
    assert normalize_parameter_id(b"MPC_LAND_SPEED\x00") == "MPC_LAND_SPEED"
    assert normalize_parameter_id(b"\xff") == ""


def test_px4_int32_parameter_decodes_bytewise_land_enum():
    encoded_float = struct.unpack("<f", struct.pack("<i", 4))[0]
    assert encoded_float == pytest.approx(5.605193857299268e-45)
    assert decode_px4_parameter_value(encoded_float, MAV_PARAM_TYPE_INT32) == 4


def test_px4_real32_parameter_preserves_land_speed():
    assert decode_px4_parameter_value(
        0.6000000238418579, MAV_PARAM_TYPE_REAL32) == pytest.approx(0.6)


@pytest.mark.parametrize(("value", "param_type"), [
    (0.6, 5),
    (0.6, None),
    (float("nan"), 9),
    ("0.6", 9),
])
def test_px4_parameter_decode_fails_closed(value, param_type):
    with pytest.raises(ValueError):
        decode_px4_parameter_value(value, param_type)


def _position_mode():
    return {
        "position": True,
        "velocity": False,
        "acceleration": False,
        "attitude": False,
        "body_rate": False,
        "thrust_and_torque": False,
        "direct_actuator": False,
    }


def _velocity_mode():
    result = _position_mode()
    result["position"] = False
    result["velocity"] = True
    return result


def test_position_and_velocity_type_masks_preserve_ned_and_yaw():
    position = make_route_setpoint(_position_mode(), {
        "position": (4.0, -3.0, -7.5),
        "velocity": (math.nan, math.nan, math.nan),
        "yaw": 0.4,
        "yawspeed": math.nan,
    })
    assert position.type_mask == POSITION_YAW_MASK == 2552
    assert position.position == (4.0, -3.0, -7.5)
    assert position.yaw == 0.4

    velocity_yaw = make_route_setpoint(_velocity_mode(), {
        "position": (math.nan, math.nan, math.nan),
        "velocity": (0.4, -0.2, 0.1),
        "yaw": -0.3,
        "yawspeed": math.nan,
    })
    assert velocity_yaw.type_mask == VELOCITY_YAW_MASK == 2503
    assert velocity_yaw.velocity == (0.4, -0.2, 0.1)

    velocity_rate = make_route_setpoint(_velocity_mode(), {
        "position": (math.nan, math.nan, math.nan),
        "velocity": (0.1, 0.2, -0.1),
        "yaw": math.nan,
        "yawspeed": 0.2,
    })
    assert velocity_rate.type_mask == VELOCITY_YAW_RATE_MASK == 1479
    assert velocity_rate.yaw_rate == 0.2


def test_unsupported_or_out_of_bounds_setpoints_fail_closed():
    hybrid = _position_mode()
    hybrid["velocity"] = True
    with pytest.raises(ValueError, match="pure_position_or_velocity"):
        make_route_setpoint(hybrid, {
            "position": (1.0, 2.0, -3.0), "velocity": (0.0, 0.0, 0.0),
            "yaw": 0.0, "yawspeed": math.nan,
        })
    with pytest.raises(ValueError, match="velocity_exceeds"):
        make_route_setpoint(_velocity_mode(), {
            "position": (math.nan,) * 3, "velocity": (2.0, 0.0, 0.0),
            "yaw": 0.0, "yawspeed": math.nan,
        })


def _healthy_state(
    now_ns=1_000_000_000, *, offboard=False, armed=False, **state_kwargs
):
    state = RouteFcState(**state_kwargs)
    base_mode = 1 | (128 if armed else 0)
    main_mode = 6 if offboard else 3
    system_status = 4 if armed else 3
    assert state.accept("HEARTBEAT", {
        "autopilot": 12, "type": 2, "base_mode": base_mode,
        "custom_mode": main_mode << 16, "system_status": system_status,
    }, 1, 1, now_ns)
    assert state.accept("SYS_STATUS", {
        "onboard_control_sensors_present": 7,
        "onboard_control_sensors_enabled": 7,
        "onboard_control_sensors_health": 7,
    }, 1, 1, now_ns)
    assert state.accept("ESTIMATOR_STATUS", {
        "time_usec": 800_000, "flags": 1 | 2 | 4 | 8 | 32,
    }, 1, 1, now_ns)
    assert state.accept("ATTITUDE_QUATERNION", {
        "time_boot_ms": 800, "q1": 1.0, "q2": 0.0, "q3": 0.0, "q4": 0.0,
    }, 1, 1, now_ns)
    assert state.accept("LOCAL_POSITION_NED", {
        "time_boot_ms": 800, "x": 1.0, "y": 2.0, "z": -3.0,
        "vx": 0.0, "vy": 0.0, "vz": 0.0,
    }, 1, 1, now_ns)
    assert state.accept("EXTENDED_SYS_STATE", {"landed_state": 1}, 1, 1, now_ns)
    return state


def test_fresh_health_state_and_rc_mode_departure_are_reported():
    now_ns = 1_000_000_000
    state = _healthy_state(now_ns, offboard=True, armed=True)
    snapshot = state.snapshot(now_ns + 10_000_000)
    assert snapshot.connected
    assert snapshot.preflight_checks_pass
    assert snapshot.position_valid
    assert snapshot.armed and snapshot.offboard and snapshot.landed
    assert not snapshot.failsafe
    assert snapshot.position == (1.0, 2.0, -3.0)
    assert snapshot.position_source == "LOCAL_POSITION_NED"
    assert snapshot.position_received_ns == now_ns

    assert state.accept("HEARTBEAT", {
        "autopilot": 12, "type": 2, "base_mode": 129,
        "custom_mode": 3 << 16, "system_status": 4,
    }, 1, 1, now_ns + 20_000_000)
    assert not state.snapshot(now_ns + 30_000_000).offboard


def test_snapshot_labels_odometry_fallback_for_normal_consumers():
    now_ns = 1_000_000_000
    state = _healthy_state(now_ns)
    later_ns = now_ns+800_000_000
    assert state.accept("ESTIMATOR_STATUS", {
        "time_usec": 1_600_000, "flags": 1 | 2 | 4 | 8 | 32,
    }, 1, 1, later_ns)
    assert state.accept("ODOMETRY", {
        "time_usec": 1_600_000, "frame_id": 1, "child_frame_id": 12,
        "reset_counter": 0,
        "x": 4.0, "y": 5.0, "z": -6.0,
        "vx": 0.0, "vy": 0.0, "vz": 0.0,
        "q": [1.0, 0.0, 0.0, 0.0],
    }, 1, 1, later_ns)
    snapshot = state.snapshot(later_ns+1)
    assert snapshot.position_valid
    assert snapshot.position == (4.0, 5.0, -6.0)
    assert snapshot.position_source == "ODOMETRY"
    assert snapshot.position_received_ns == later_ns


def test_px4_v117_vehicle_at_rest_flag_does_not_deadlock_valid_absolute_aiding():
    # PX4 v1.17 reports 959 while landed with GNSS position/velocity active:
    # all required solution bits, CONST_POS_MODE, and predicted absolute aiding.
    assert route_estimator_reason(959) == ""

    # A genuinely unaided constant-position solution remains fail-closed.
    unaided = 1 | 2 | 4 | 8 | 32 | 128 | 256
    assert route_estimator_reason(unaided) == "estimator_constant_position_mode"

    # Absolute position without proof of active absolute aiding also remains
    # blocked, even if the current estimate is marked absolute.
    unproven_absolute = unaided | 16
    assert route_estimator_reason(unproven_absolute) == "estimator_constant_position_mode"


def test_wrong_source_and_stale_state_never_authorize_output():
    state = RouteFcState()
    assert not state.accept("HEARTBEAT", {
        "autopilot": 12, "type": 2, "base_mode": 1,
        "custom_mode": 3 << 16, "system_status": 3,
    }, 2, 1, 1_000_000_000)
    assert not state.snapshot(1_000_000_001).connected

    state = _healthy_state()
    stale = state.snapshot(3_600_000_000)
    assert not stale.connected
    assert not stale.preflight_checks_pass
    assert not stale.position_valid
    assert stale.failsafe


def test_normal_one_hz_heartbeat_does_not_flap_connected():
    now = 1_000_000_000
    state = _healthy_state(now)
    between_normal_heartbeats = state.snapshot(now + 1_200_000_000)
    assert between_normal_heartbeats.connected
    assert not state.snapshot(now + 2_600_000_000).connected


def test_pose_source_skew_is_not_accepted_for_control():
    state = _healthy_state()
    assert state.accept("ATTITUDE_QUATERNION", {
        "time_boot_ms": 1_000, "q1": 1.0, "q2": 0.0, "q3": 0.0, "q4": 0.0,
    }, 1, 1, 1_010_000_000)
    snapshot = state.snapshot(1_020_000_000)
    assert not snapshot.position_valid
    assert not snapshot.preflight_checks_pass


def test_home_global_and_local_must_describe_same_frame():
    now = 1_000_000_000
    state = _healthy_state(now, home_stability_ns=0)
    assert state.accept("HOME_POSITION", {
        "latitude": 352350126, "longitude": 1290748631, "altitude": 42800,
        "x": 1.0, "y": 2.0, "z": -1.0,
    }, 1, 1, now)
    # Current local (1,2,-3) is exactly 2m above Home.
    assert state.accept("GLOBAL_POSITION_INT", {
        "time_boot_ms": 801, "lat": 352350126, "lon": 1290748631,
        "alt": 44800, "relative_alt": 2000,
    }, 1, 1, now)
    home, reason, residual = state.route_home(now+10_000_000)
    assert reason == ""
    assert residual == pytest.approx(0.0)
    assert home["ned"] == (1.0, 2.0, -1.0)

    # PX4 relative_alt follows PX4's current Home and may jump when it refines
    # Home during arming. The frozen mission frame must use absolute AMSL.
    state.global_position["relative_altitude_m"] = 0.25
    home, reason, residual = state.route_home(now+10_000_000)
    assert reason == ""
    assert residual == pytest.approx(0.0)

    state.local_position["position"] = (20.0, 2.0, -3.0)
    home, reason, residual = state.route_home(now+10_000_000)
    assert home is None and "residual" in reason and residual > 5.0


def test_real_local_frame_change_and_wrong_fc_identity_latch_frame_invalid():
    now = 1_000_000_000
    state = _healthy_state(
        now, armed=True, home_stability_ns=0, home_change_confirmations=3
    )
    first = {"latitude": 352350126, "longitude": 1290748631,
             "altitude": 42800, "x": 1.0, "y": 2.0, "z": -1.0}
    assert not state.accept("HOME_POSITION", first, 2, 1, now)
    assert state.accept("HOME_POSITION", first, 1, 1, now)
    assert state.accept("GLOBAL_POSITION_INT", {
        "time_boot_ms": 801, "lat": 352350126, "lon": 1290748631,
        "alt": 44800, "relative_alt": 2000,
    }, 1, 1, now)
    assert state.route_home(now+1)[0] is not None
    assert state.lock_execution_home("mission-local-reset")[0]

    # A real local-NED jump changes the frozen transform residual and must not
    # be mistaken for harmless HOME_POSITION metadata refinement.
    state.local_position["position"] = (2.0, 2.0, -3.0)
    changed = {**first, "x": 2.0}
    assert state.accept("HOME_POSITION", changed, 1, 1, now+1)
    assert state.accept("HOME_POSITION", changed, 1, 1, now+2)
    assert not state.accept("HOME_POSITION", changed, 1, 1, now+3)
    assert state.fault == ""
    assert state.home_epoch_failure_latched
    assert state.home_correction_detail == "home_correction_horizontal_change"


def test_locked_mission_home_accepts_vertical_px4_home_only_correction():
    now = 1_000_000_000
    state = _healthy_state(
        now, armed=True, home_stability_ns=0, home_change_confirmations=3
    )
    assert state.accept("ODOMETRY", {
        "time_usec": 800_000, "frame_id": 1, "child_frame_id": 12,
        "reset_counter": 0,
        "x": 1.0, "y": 2.0, "z": -3.0,
        "vx": 0.0, "vy": 0.0, "vz": 0.0,
        "q": [1.0, 0.0, 0.0, 0.0],
    }, 1, 1, now)
    first = {"latitude": 352350126, "longitude": 1290748631,
             "altitude": 42800, "x": 1.0, "y": 2.0, "z": -1.0}
    assert state.accept("HOME_POSITION", first, 1, 1, now)
    assert state.accept("GLOBAL_POSITION_INT", {
        "time_boot_ms": 801, "lat": 352350126, "lon": 1290748631,
        "alt": 44800, "relative_alt": 2000,
    }, 1, 1, now)
    locked, reason, _ = state.route_home(now+1)
    assert reason == "" and locked is not None
    assert state.lock_execution_home("mission-home-correction")[0]

    # Reproduce the physical ULog: PX4 lowers Home AMSL by 1.080322 m and
    # raises Home NED Z by the same amount without moving the estimator frame.
    assert state.accept("LOCAL_POSITION_NED", {
        "time_boot_ms": 850, "x": 1.0, "y": 2.0, "z": -3.0,
        "vx": 0.0, "vy": 0.0, "vz": 0.0,
    }, 1, 1, now+50_000_000)
    assert state.accept("GLOBAL_POSITION_INT", {
        "time_boot_ms": 851, "lat": 352350126, "lon": 1290748631,
        "alt": 44800, "relative_alt": 3080,
    }, 1, 1, now+50_000_000)
    changed = {
        **first,
        "altitude": 41719.678,
        "z": 0.080322,
    }
    assert state.accept("HOME_POSITION", changed, 1, 1, now+60_000_000)
    assert state.accept("HOME_POSITION", changed, 1, 1, now+60_000_001)
    assert state.fault == ""
    assert state.home_position["ned"] == (1.0, 2.0, -1.0)
    assert state.home_locked
    assert state.ignored_home_refinements == 1
    correction = state.home_correction_snapshot()
    assert correction["correction_valid"]
    assert correction["correction_revision"] == 1
    assert correction["current_altitude_amsl_m"] == pytest.approx(41.719678)
    assert correction["current_z_ned_m"] == pytest.approx(0.080322)
    assert correction["opposition_error_m"] == pytest.approx(0.0)
    home, reason, _ = state.route_home(now+60_000_002)
    assert reason == "" and home == locked
    snapshot = state.snapshot(now+4)
    assert snapshot.connected
    assert snapshot.position_valid


def test_ground_home_refinement_without_fresh_landed_evidence_stays_fail_closed():
    now = 1_000_000_000
    state = _healthy_state(
        now, home_stability_ns=0, home_change_confirmations=3
    )
    first = {"latitude": 352350126, "longitude": 1290748631,
             "altitude": 42800, "x": 0.0, "y": 0.0, "z": 0.0}
    assert state.accept("HOME_POSITION", first, 1, 1, now)
    assert state.accept("GLOBAL_POSITION_INT", {
        "time_boot_ms": 801, "lat": 352350126, "lon": 1290748631,
        "alt": 45800, "relative_alt": 3000,
    }, 1, 1, now)
    state.home_locked = True
    changed = {**first, "x": 1.0}
    # Heartbeat remains within its 2.5 s lease, while the 0.75 s landed
    # evidence has expired.
    assert not state.accept("HOME_POSITION", changed, 1, 1, now+800_000_000)
    assert state.fault == ""
    assert state.home_epoch_failure_latched
    assert state.home_phase == "REJECTED"
    assert state.home_correction_detail == "frozen_home_changed_before_arm"


def test_route_home_query_keeps_home_provisional_until_exact_contract_lock():
    now = 1_000_000_000
    state = _healthy_state(now, home_stability_ns=0)
    first = {"latitude": 352350126, "longitude": 1290748631,
             "altitude": 42800, "x": 0.0, "y": 0.0, "z": 0.0}
    assert state.accept("HOME_POSITION", first, 1, 1, now)
    assert state.accept("GLOBAL_POSITION_INT", {
        "time_boot_ms": 801, "lat": 352350126, "lon": 1290748631,
        "alt": 45800, "relative_alt": 3000,
    }, 1, 1, now)
    assert state.route_home(now+1)[0] is not None
    assert not state.home_locked
    assert state.home_phase == "PROVISIONAL"

    refined = {**first, "altitude": 41800, "z": 1.0}
    assert state.accept("HOME_POSITION", refined, 1, 1, now+2)
    assert state.fault == ""
    assert not state.home_epoch_failure_latched
    assert state.home_position["ned"][2] == 1.0

    assert state.lock_execution_home("mission-1") == (
        True, "execution_home_locked")
    assert state.home_locked
    assert state.execution_home_mission_id == "mission-1"
    assert state.home_phase == "EXECUTION_LOCKED"


def test_home_route_fault_does_not_hide_fresh_arming_state():
    now = 1_000_000_000
    state = _healthy_state(now, armed=True, home_stability_ns=0)
    state.fault = "route_frame_invalid"
    snapshot = state.snapshot(now+1)
    assert snapshot.connected
    assert snapshot.armed
    assert not snapshot.preflight_checks_pass
    assert snapshot.reason == "route_frame_invalid"


def test_home_refinement_stabilizes_before_frame_is_exposed():
    now = 1_000_000_000
    state = _healthy_state(now, home_stability_ns=3_000_000_000)
    first = {"latitude": 352350126, "longitude": 1290748631,
             "altitude": 42800, "x": 0.0, "y": 0.0, "z": 0.0}
    refined = {**first, "x": 1.0}
    assert state.accept("HOME_POSITION", first, 1, 1, now)
    assert state.accept("HOME_POSITION", refined, 1, 1, now+1_000_000_000)
    assert state.fault == ""
    home, reason, _ = state.route_home(now+1_100_000_000)
    assert home is None and reason == "home_position_stabilizing"


def test_single_locked_home_outlier_recovers_without_latching_fault():
    now = 1_000_000_000
    state = _healthy_state(now, armed=True, home_stability_ns=0)
    first = {"latitude": 352350126, "longitude": 1290748631,
             "altitude": 42800, "x": 0.0, "y": 0.0, "z": 0.0}
    assert state.accept("HOME_POSITION", first, 1, 1, now)
    state.home_locked = True
    assert state.accept("HOME_POSITION", {**first, "x": 1.0}, 1, 1, now+1)
    assert state.home_change_candidate is not None
    assert state.accept("HOME_POSITION", first, 1, 1, now+2)
    assert state.home_change_candidate is None
    assert state.fault == ""


def test_only_addressed_allowlisted_command_ack_is_accepted():
    ack = {
        "command": MAV_CMD_DO_SET_MODE,
        "result": 0,
        "progress": 100,
        "result_param2": 0,
        "target_system": 245,
        "target_component": 191,
    }
    assert validate_route_command_ack(ack, 1, 1) == (
        MAV_CMD_DO_SET_MODE, 0, 100, 0
    )
    with pytest.raises(ValueError, match="target_mismatch"):
        validate_route_command_ack({**ack, "target_component": 0}, 1, 1)
    with pytest.raises(ValueError, match="not_allowlisted"):
        validate_route_command_ack({**ack, "command": 511}, 1, 1)


class _FakeMessage:
    def __init__(self, args):
        self.args = args

    def pack(self, _encoder, force_mavlink1=False):
        assert force_mavlink1 is False
        return b"packet"


class _FakeCommon:
    @staticmethod
    def MAVLink_command_long_message(*args):
        return _FakeMessage(args)

    @staticmethod
    def MAVLink_set_position_target_local_ned_message(*args):
        return _FakeMessage(args)

    @staticmethod
    def MAVLink_param_request_read_message(*args):
        return _FakeMessage(args)


class _FakePort:
    def __init__(self):
        self.writes = []

    def write(self, packet):
        self.writes.append(packet)
        return len(packet)


class _FakeEncoder:
    seq = 0


def test_transport_records_owner_thread_with_legacy_base(monkeypatch):
    def legacy_init(self, journal, device, *, timing=None):
        self.journal = journal

    monkeypatch.setattr(UsbTelemetryTransport, "__init__", legacy_init)
    transport = MavlinkRouteTransport(io.StringIO(), "/dev/legacy")
    assert transport._owner_thread == threading.get_ident()


def _transport(enabled=True):
    transport = object.__new__(MavlinkRouteTransport)
    transport.flight_output_enabled = enabled
    transport.last_setpoint_ns = 0
    transport.last_heartbeat_request_ns = 0
    transport.last_parameter_request_ns = {}
    transport.common = _FakeCommon()
    transport.encoder = _FakeEncoder()
    transport.port = _FakePort()
    transport.journal = io.StringIO()
    transport.tx_count = 0
    transport._owner_thread = threading.get_ident()
    return transport


def test_low_speed_parameter_reads_are_allowlisted_read_only_and_throttled():
    transport = _transport(enabled=False)
    assert transport.request_parameter("COM_OBL_RC_ACT", now_ns=3_000_000_000)
    assert not transport.request_parameter("COM_OBL_RC_ACT", now_ns=4_000_000_000)
    assert transport.request_parameter("MPC_LAND_SPEED", now_ns=4_000_000_000)
    with pytest.raises(ValueError, match="not allowlisted"):
        transport.request_parameter("MPC_XY_VEL_MAX", now_ns=5_000_000_000)
    assert len(transport.port.writes) == 2


def _snapshot(*, armed=True):
    return RouteFcSnapshot(
        connected=True,
        preflight_checks_pass=True,
        position_valid=True,
        armed=armed,
        offboard=False,
        landed=not armed,
        failsafe=False,
        position=(0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0),
        yaw=0.0,
        time_boot_ms=123,
        position_source="LOCAL_POSITION_NED",
        position_received_ns=1_000_000_000,
        transport_epoch=1,
        reason="",
    )


def test_last_mile_gate_and_command_allowlist():
    disabled = _transport(enabled=False)
    with pytest.raises(PermissionError, match="flight_output_disabled"):
        disabled.send_vehicle_command(
            MAV_CMD_DO_SET_MODE, (1.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0), _snapshot()
        )
    assert disabled.port.writes == []

    enabled = _transport(enabled=True)
    with pytest.raises(ValueError, match="not_allowlisted"):
        enabled.send_vehicle_command(999, (0.0,) * 7, _snapshot())
    assert enabled.send_vehicle_command(
        MAV_CMD_DO_SET_MODE, (1.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0), _snapshot()
    )
    assert enabled.send_vehicle_command(
        MAV_CMD_COMPONENT_ARM_DISARM, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), _snapshot(armed=False)
    )
    assert enabled.send_vehicle_command(MAV_CMD_NAV_LAND, (0.0,) * 7, _snapshot())
    assert len(enabled.port.writes) == 3


def test_stale_snapshot_blocks_setpoint_before_serial_write():
    transport = _transport(enabled=True)
    setpoint = make_route_setpoint(_position_mode(), {
        "position": (1.0, 2.0, -3.0), "velocity": (math.nan,) * 3,
        "yaw": 0.0, "yawspeed": math.nan,
    })
    invalid = RouteFcSnapshot(
        connected=False, preflight_checks_pass=False, position_valid=False,
        armed=False, offboard=False, landed=None, failsafe=True,
        position=(math.nan,) * 3, velocity=(math.nan,) * 3, yaw=math.nan,
        time_boot_ms=0, position_source="NONE", position_received_ns=0,
        transport_epoch=1, reason="stale",
    )
    with pytest.raises(PermissionError, match="fresh_fc_connection"):
        transport.send_setpoint(setpoint, invalid, now_ns=1_000_000_000)
    assert transport.port.writes == []

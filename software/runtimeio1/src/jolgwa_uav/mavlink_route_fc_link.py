"""Guarded MAVLink v2 transport for the route controller's single USB link.

This module deliberately does not publish ROS ``/fmu/*`` topics.  It provides
the small, testable conversion and last-mile write boundary used by the ROS
bridge.  ``UsbTelemetryTransport`` remains the sole serial descriptor owner.
"""
from collections import deque
from dataclasses import dataclass
from itertools import product
import json
import math
import struct
import threading
import time

from .observer_fc_telemetry import RelativeSourceTimePolicy, estimator_reason, finite
from .observer_fc_transport import UsbTelemetryTransport
from .battery_policy import decode_battery


TARGET_SYSTEM = 1
TARGET_COMPONENT = 1
SOURCE_SYSTEM = 245
SOURCE_COMPONENT = 191

STATUS_LEASE_NS = 750_000_000
# PX4 HEARTBEAT is normally emitted at 1 Hz. Its lease must span more than
# one normal publication interval or a healthy link oscillates to disconnected.
HEARTBEAT_LEASE_NS = 2_500_000_000
POSE_LEASE_NS = 250_000_000
SETPOINT_LEASE_NS = 250_000_000
SETPOINT_PERIOD_NS = 50_000_000
HEARTBEAT_REQUEST_PERIOD_NS = 200_000_000
ARM_HOME_TRANSITION_NS = 400_000_000
POSE_PAIR_SKEW_NS = 75_000_000

MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1
MAV_MODE_FLAG_SAFETY_ARMED = 128
PX4_MAIN_MODE_OFFBOARD = 6
MAV_STATE_STANDBY = 3
MAV_STATE_ACTIVE = 4
MAV_LANDED_STATE_ON_GROUND = 1
MAV_PARAM_TYPE_INT32 = 6
MAV_PARAM_TYPE_REAL32 = 9

# MAVLink ESTIMATOR_STATUS_FLAGS. PX4 v1.17's legacy solution bitmask sets
# CONST_POS_MODE while ``vehicle_at_rest`` is true, even when GNSS/aux global
# aiding is active.  The observer keeps reporting that legacy bit verbatim,
# but route control may ignore it only when the same bitmask independently
# proves a usable absolute horizontal solution and active absolute aiding.
ESTIMATOR_POS_HORIZ_ABS = 16
ESTIMATOR_CONST_POS_MODE = 128
ESTIMATOR_PRED_POS_HORIZ_ABS = 512

MAV_CMD_DO_SET_MODE = 176
MAV_CMD_NAV_RETURN_TO_LAUNCH = 20
MAV_CMD_NAV_LAND = 21
MAV_CMD_COMPONENT_ARM_DISARM = 400
ALLOWED_VEHICLE_COMMANDS = frozenset({
    MAV_CMD_DO_SET_MODE,
    MAV_CMD_COMPONENT_ARM_DISARM,
    MAV_CMD_NAV_RETURN_TO_LAUNCH,
    MAV_CMD_NAV_LAND,
})

# PX4 rejects SET_MESSAGE_INTERVAL for HEARTBEAT.  A bounded REQUEST_MESSAGE
# is sent separately at 5 Hz after the first normal heartbeat is observed.
ROUTE_STREAM_INTERVALS_US = {
    1: 200_000,    # SYS_STATUS
    147: 200_000,  # BATTERY_STATUS: severity, never a voltage cutoff
    31: 50_000,    # ATTITUDE_QUATERNION
    32: 50_000,    # LOCAL_POSITION_NED
    33: 100_000,   # GLOBAL_POSITION_INT
    230: 50_000,   # ESTIMATOR_STATUS
    245: 200_000,  # EXTENDED_SYS_STATE
    331: 50_000,   # ODOMETRY (diagnostic/fallback)
}


def normalize_parameter_id(value):
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            return ""
    return value.rstrip("\x00") if isinstance(value, str) else ""


def decode_px4_parameter_value(param_value, param_type):
    """Decode PX4 PARAM_VALUE using its byte-wise MAVLink representation.

    PX4 parameters used by this bridge are either INT32 or REAL32.  Both are
    carried in PARAM_VALUE.param_value, whose wire field is a float.  INT32
    values must therefore be reconstructed from the float field's four raw
    bytes rather than numerically casting the apparent float value.
    """
    if type(param_value) not in (int, float) or type(param_type) is not int:
        raise ValueError("parameter value and type must be numeric")
    value = float(param_value)
    if not math.isfinite(value):
        raise ValueError("parameter value must be finite")
    if param_type == MAV_PARAM_TYPE_REAL32:
        return value
    if param_type == MAV_PARAM_TYPE_INT32:
        return struct.unpack("<i", struct.pack("<f", value))[0]
    raise ValueError("unsupported PX4 parameter type")


ROUTE_MESSAGE_KINDS = frozenset({
    "HEARTBEAT", "SYS_STATUS", "BATTERY_STATUS", "ATTITUDE_QUATERNION", "LOCAL_POSITION_NED",
    "GLOBAL_POSITION_INT", "ESTIMATOR_STATUS", "EXTENDED_SYS_STATE",
    "HOME_POSITION", "ODOMETRY",
})
POSE_MESSAGE_KINDS = (
    "LOCAL_POSITION_NED", "ATTITUDE_QUATERNION", "ESTIMATOR_STATUS"
)
HOME_POSITION_MESSAGE_ID = 242
HOME_LEASE_NS = 2_500_000_000
GLOBAL_LEASE_NS = 750_000_000
HOME_STABILITY_NS = 3_000_000_000
HOME_CHANGE_THRESHOLD_M = 0.5
HOME_CHANGE_CONFIRMATIONS = 3
HOME_FRAME_CONTINUITY_THRESHOLD_M = 0.5
HOME_CORRECTION_HORIZONTAL_METADATA_MAX_M = 2.0
HOME_CORRECTION_HORIZONTAL_CONSISTENCY_MAX_M = 0.20
HOME_CORRECTION_OPPOSITION_MAX_M = 0.10
HOME_CORRECTION_MAX_M = 2.0
HOME_CORRECTION_SAMPLE_CONTINUITY_MAX_M = 0.25
HOME_CORRECTION_EVIDENCE_LEASE_NS = 500_000_000
HOME_PHASE_PROVISIONAL = "PROVISIONAL"
HOME_PHASE_EXECUTION_LOCKED = "EXECUTION_LOCKED"
HOME_PHASE_CORRECTION_PENDING = "CORRECTION_PENDING"
HOME_PHASE_REJECTED = "REJECTED"

# POSITION_TARGET_TYPEMASK bits.
IGNORE_POSITION = 1 | 2 | 4
IGNORE_VELOCITY = 8 | 16 | 32
IGNORE_ACCELERATION = 64 | 128 | 256
IGNORE_YAW = 1024
IGNORE_YAW_RATE = 2048
POSITION_YAW_MASK = IGNORE_VELOCITY | IGNORE_ACCELERATION | IGNORE_YAW_RATE
VELOCITY_YAW_MASK = IGNORE_POSITION | IGNORE_ACCELERATION | IGNORE_YAW_RATE
VELOCITY_YAW_RATE_MASK = IGNORE_POSITION | IGNORE_ACCELERATION | IGNORE_YAW


def _fresh(now_ns, received_ns, lease_ns):
    return (
        type(now_ns) is int
        and type(received_ns) is int
        and 0 < received_ns <= now_ns
        and now_ns - received_ns < lease_ns
    )


def _quaternion_yaw(quaternion):
    if not isinstance(quaternion, (tuple, list)) or len(quaternion) != 4:
        return math.nan
    values = [float(value) for value in quaternion]
    if not finite(values) or abs(sum(value * value for value in values) - 1.0) > 0.02:
        return math.nan
    w, x, y, z = values
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wgs84_delta_m(latitude_a, longitude_a, latitude_b, longitude_b):
    mean_latitude = math.radians((latitude_a+latitude_b)/2.0)
    north = math.radians(latitude_b-latitude_a)*6378137.0
    east = math.radians(longitude_b-longitude_a)*6378137.0*math.cos(mean_latitude)
    return east, north


def _wgs84_distance_m(latitude_a, longitude_a, latitude_b, longitude_b):
    east, north = _wgs84_delta_m(latitude_a, longitude_a, latitude_b, longitude_b)
    return math.hypot(east, north)


def route_estimator_reason(flags):
    """Apply PX4-v1.17-aware validity to its legacy MAVLink bitmask.

    PX4 v1.17 maps ``vehicle_at_rest`` into MAVLink CONST_POS_MODE.  Treating
    that bit alone as loss of navigation deadlocks preflight on the ground.
    It is safe to discount only when absolute position is currently valid and
    the estimator also reports that absolute aiding is active.  All other
    legacy faults and validity requirements remain fail-closed.
    """
    checked_flags = flags
    if (
        type(flags) is int
        and flags & ESTIMATOR_CONST_POS_MODE
        and flags & ESTIMATOR_POS_HORIZ_ABS
        and flags & ESTIMATOR_PRED_POS_HORIZ_ABS
    ):
        checked_flags = flags & ~ESTIMATOR_CONST_POS_MODE
    return estimator_reason(checked_flags)


@dataclass(frozen=True)
class RouteSetpoint:
    """Validated MAVLink LOCAL_NED setpoint without a dialect dependency."""

    type_mask: int
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    yaw: float
    yaw_rate: float


def make_route_setpoint(mode, setpoint):
    """Validate the supported ROS offboard subset and select its MAVLink mask."""
    if not isinstance(mode, dict) or not isinstance(setpoint, dict):
        raise ValueError("setpoint_mode_and_value_required")
    position_mode = mode.get("position") is True and mode.get("velocity") is False
    velocity_mode = mode.get("velocity") is True and mode.get("position") is False
    if not (position_mode or velocity_mode):
        raise ValueError("only_pure_position_or_velocity_mode_is_supported")
    if any(mode.get(name) is True for name in (
        "acceleration", "attitude", "body_rate", "thrust_and_torque", "direct_actuator"
    )):
        raise ValueError("unsupported_offboard_control_mode")

    position = tuple(float(value) for value in setpoint.get("position", ()))
    velocity = tuple(float(value) for value in setpoint.get("velocity", ()))
    if len(position) != 3 or len(velocity) != 3:
        raise ValueError("three_axis_setpoint_required")
    yaw = float(setpoint.get("yaw", math.nan))
    yaw_rate = float(setpoint.get("yawspeed", math.nan))

    if position_mode:
        if not finite(position) or not math.isfinite(yaw):
            raise ValueError("finite_position_and_yaw_required")
        if not -math.pi <= yaw <= math.pi:
            raise ValueError("yaw_out_of_range")
        return RouteSetpoint(
            POSITION_YAW_MASK,
            position,
            (0.0, 0.0, 0.0),
            yaw,
            0.0,
        )

    if not finite(velocity):
        raise ValueError("finite_velocity_required")
    horizontal_speed = math.hypot(velocity[0], velocity[1])
    if horizontal_speed > 1.5 or abs(velocity[2]) > 1.0:
        raise ValueError("velocity_exceeds_route_bridge_limit")
    yaw_finite = math.isfinite(yaw)
    yaw_rate_finite = math.isfinite(yaw_rate)
    if yaw_finite == yaw_rate_finite:
        raise ValueError("exactly_one_of_yaw_or_yaw_rate_required")
    if yaw_finite:
        if not -math.pi <= yaw <= math.pi:
            raise ValueError("yaw_out_of_range")
        mask = VELOCITY_YAW_MASK
        yaw_rate = 0.0
    else:
        if abs(yaw_rate) > 1.0:
            raise ValueError("yaw_rate_exceeds_route_bridge_limit")
        mask = VELOCITY_YAW_RATE_MASK
        yaw = 0.0
    return RouteSetpoint(mask, (0.0, 0.0, 0.0), velocity, yaw, yaw_rate)


def validate_route_command_ack(value, system_id, component_id):
    """Return the bounded flight-command ACK fields or reject the packet."""
    if (system_id, component_id) != (TARGET_SYSTEM, TARGET_COMPONENT):
        raise ValueError("unexpected_ack_source")
    if not isinstance(value, dict):
        raise ValueError("ack_dictionary_required")
    if (value.get("target_system"), value.get("target_component")) != (
        SOURCE_SYSTEM, SOURCE_COMPONENT
    ):
        raise ValueError("ack_target_mismatch")
    command, result = value.get("command"), value.get("result")
    if type(command) is not int or command not in ALLOWED_VEHICLE_COMMANDS:
        raise ValueError("ack_command_not_allowlisted")
    if type(result) is not int or not 0 <= result <= 6:
        raise ValueError("invalid_ack_result")
    progress = value.get("progress", 0)
    result_param2 = value.get("result_param2", 0)
    if type(progress) is not int or type(result_param2) is not int:
        raise ValueError("invalid_ack_parameters")
    return command, result, progress, result_param2


@dataclass(frozen=True)
class RouteFcSnapshot:
    connected: bool
    preflight_checks_pass: bool
    position_valid: bool
    armed: bool
    offboard: bool
    landed: bool | None
    failsafe: bool
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    yaw: float
    time_boot_ms: int
    position_source: str
    position_received_ns: int
    transport_epoch: int
    estimator_reset_counter_valid: bool = False
    estimator_reset_counter: int = 0
    # Inferred generations, not measured per-axis PX4 counters.
    navigation_reset_generations: tuple = ()
    post_yaw_attitude_ready: bool = True
    reason: str = ""
    px4_custom_mode: int = 0
    native_land: bool = False
    # Internal evidence; never infer a native FC failsafe from this exception.
    battery_health_terminal_only: bool = False
    sensor_health_reason: str = ""
    sensor_health_age_ns: int = -1
    sensors_present: int = 0
    sensors_enabled: int = 0
    sensors_health: int = 0
    sensors_failed: int = 0
    battery_sample: dict | None = None
    battery_navigation_healthy: bool = False


class RouteFcState:
    """Conservative state derived only from fresh messages from FC 1/1."""

    def __init__(
        self,
        *,
        heartbeat_lease_ns=HEARTBEAT_LEASE_NS,
        allow_odometry_fallback=True,
        home_stability_ns=HOME_STABILITY_NS,
        home_change_confirmations=HOME_CHANGE_CONFIRMATIONS,
        transport_epoch=0,
    ):
        self.heartbeat_lease_ns = int(heartbeat_lease_ns)
        self.allow_odometry_fallback = bool(allow_odometry_fallback)
        self.home_stability_ns = max(0, int(home_stability_ns))
        self.home_change_confirmations_required = max(
            1, int(home_change_confirmations)
        )
        self.transport_epoch = max(0, int(transport_epoch))
        self.heartbeat = None
        self.sys_status = None
        self.battery_samples = {}
        self.estimator_status = None
        self.extended_state = None
        self.local_position = None
        self.previous_local_position = None
        self.global_position = None
        self.previous_global_position = None
        self.home_position = None
        self.latest_px4_home_position = None
        self.provisional_home_revision = 0
        self.execution_home_mission_id = ""
        self.execution_home_lock_revision = 0
        self.home_phase = HOME_PHASE_PROVISIONAL
        self.home_epoch_failure_latched = False
        self.home_epoch_failure_detail = ""
        self.home_correction_valid = False
        self.home_correction_revision = 0
        self.home_correction_detail = "home_not_locked"
        self.home_correction_opposition_error_m = math.inf
        self.home_lock_reset_counter = None
        self.home_candidate_since_ns = 0
        self.home_locked = False
        self._arm_home_transition = None
        self.home_change_candidate = None
        self.home_change_count = 0
        self.home_frame_residual_vector = None
        self.home_frame_continuity_delta_m = math.inf
        self.home_horizontal_global_delta_m = 0.0
        self.home_horizontal_local_delta_m = 0.0
        self.home_horizontal_consistency_error_m = 0.0
        self.last_home_message_delta_m = 0.0
        self.ignored_home_refinements = 0
        self.attitude = None
        self.odometry = None
        self.received_ns = {}
        self.relative_time = RelativeSourceTimePolicy()
        self.source_stamps = self.relative_time.stamps
        self.reset_counter = None
        self.yaw_reset_context = None
        self.navigation_reset_generations = None
        self.yaw_reset_events = deque(maxlen=128)
        self.yaw_reset_source_stamp_us = 0
        self.fault = ""
        self.last_rejection = ""
        self.rx_counts = {}
        self.accepted_counts = {}
        self.rejected_counts = {}
        self.last_decisions = {}
        self.accepted_arrivals = {kind: deque(maxlen=512) for kind in ROUTE_MESSAGE_KINDS}
        self.pose_rows = {kind: deque(maxlen=8) for kind in POSE_MESSAGE_KINDS}
        self.pose_bundle = None

    def _reject(self, reason):
        self.last_rejection = reason
        return False

    def configure_yaw_reset_context(self, context):
        """Called only with the current approved v5 proposal by the bridge."""
        self.yaw_reset_context = context

    def _accept_bounded_yaw_reset(self, candidate, reset_counter, received_ns):
        from .yaw_reset import classify
        context = self.yaw_reset_context
        reason = ''
        evidence = {}
        identity = (self.execution_home_mission_id, self.transport_epoch,
                    self.execution_home_lock_revision)
        if (context != identity or not self.home_locked or self.home_epoch_failure_latched
                or self.home_phase != HOME_PHASE_EXECUTION_LOCKED
                or not self.home_correction_snapshot().get('correction_valid')):
            reason = 'yaw_reset_approval_or_home_unconfirmed'
        elif (reset_counter-self.reset_counter) % 256 != 1:
            reason = 'yaw_reset_counter_not_single_increment'
        elif self.odometry is None:
            reason = 'yaw_reset_previous_odometry_missing'
        else:
            ages = {name: received_ns-self.received_ns.get(name, 0) for name in
                    ('LOCAL_POSITION_NED', 'GLOBAL_POSITION_INT', 'ATTITUDE_QUATERNION')}
            evidence['evidence_age_ms'] = {name: age/1e6 for name, age in ages.items()}
            fresh = all(0 <= age <= (GLOBAL_LEASE_NS if name == 'GLOBAL_POSITION_INT'
                                    else POSE_LEASE_NS) for name, age in ages.items())
            rates = (self.attitude or {}).get('angular_rates', ())
            attitude_stamp = self.source_stamps.get('ATTITUDE_QUATERNION', 0)
            if not fresh or abs(attitude_stamp-candidate['time_usec']) > POSE_PAIR_SKEW_NS//1000:
                reason = 'yaw_reset_evidence_stale'
            elif len(rates) != 3 or not finite(rates):
                reason = 'yaw_reset_angular_rates_missing'
            elif not self._frozen_home_frame_is_continuous(received_ns):
                reason = 'yaw_reset_home_frame_discontinuity'
            else:
                numbers, reason = classify(self.odometry, candidate,
                    received_ns-self.received_ns.get('ODOMETRY', 0), rates)
                evidence.update(numbers)
        self.yaw_reset_events.append(dict(event='yaw_reset_classification',
            mission_id=self.execution_home_mission_id, transport_epoch=self.transport_epoch,
            home_lock_revision=self.execution_home_lock_revision,
            raw_counter_before=self.reset_counter, raw_counter_after=reset_counter,
            source_time_usec=candidate['time_usec'], received_monotonic_ns=received_ns,
            classification='bounded_yaw_inferred' if not reason else 'unclassified_reset',
            accepted=not bool(reason), reason=reason or 'bounded_yaw_accepted', **evidence))
        if reason:
            return False
        generations = list(self.navigation_reset_generations or (self.reset_counter,)*5)
        generations[4] = (generations[4]+1) % 256
        self.navigation_reset_generations = tuple(generations)
        self.home_lock_reset_counter = reset_counter
        self.yaw_reset_source_stamp_us = candidate['time_usec']
        # Pose bundles used for physical output must contain post-reset attitude.
        # Snapshot link/navigation truth remains fresh while those topics arrive.
        self.pose_bundle = None
        for name, rows in self.pose_rows.items():
            keep = [row for row in rows if row['source_stamp_us'] >= candidate['time_usec']]
            rows.clear()
            rows.extend(keep)
        return True

    def _accept_stamp(self, kind, stamp, received_ns):
        accepted, reason = self.relative_time.assess(kind, stamp, received_ns / 1e9)
        if self.relative_time.fault:
            self.fault = self.relative_time.fault
            if self.home_locked and not self.home_epoch_failure_latched:
                self.home_epoch_failure_latched = True
                self.home_epoch_failure_detail = self.fault
                self.home_correction_detail = self.fault
                self.home_phase = HOME_PHASE_REJECTED
        if not accepted:
            return self._reject(reason.replace("source_timestamp", kind.lower() + "_timestamp"))
        return True

    @staticmethod
    def _home_difference(first, second):
        horizontal = _wgs84_distance_m(
            first["latitude_deg"], first["longitude_deg"],
            second["latitude_deg"], second["longitude_deg"],
        )
        local = math.sqrt(sum(
            (a-b) ** 2 for a, b in zip(first["ned"], second["ned"])
        ))
        return horizontal, local

    @staticmethod
    def _home_frame_residual(home, global_position, local_position):
        """Return the frozen-Home transform error in local NED metres.

        Vertical consistency deliberately uses AMSL altitude, not PX4's
        ``relative_alt``.  PX4 may refine its own Home during arming, which
        changes relative altitude without changing the EKF local-NED frame.
        """
        east, north = _wgs84_delta_m(
            home["latitude_deg"], home["longitude_deg"],
            global_position["latitude_deg"], global_position["longitude_deg"],
        )
        altitude_above_frozen_home = (
            global_position["altitude_m"] - home["altitude_m"]
        )
        expected = (
            home["ned"][0] + north,
            home["ned"][1] + east,
            home["ned"][2] - altitude_above_frozen_home,
        )
        return tuple(
            expected_axis - local_axis
            for expected_axis, local_axis in zip(
                expected, local_position["position"]
            )
        )

    def _frozen_home_frame_is_continuous(self, received_ns):
        """Prove that a new PX4 Home message did not move local NED.

        A Home metadata refinement is harmless only when fresh global and
        local positions still produce the same transform error as when the
        mission Home was first locked.  Missing evidence remains fail-closed.
        """
        if self.home_frame_residual_vector is None:
            return False
        if not _fresh(
            received_ns,
            self.received_ns.get("GLOBAL_POSITION_INT", 0),
            GLOBAL_LEASE_NS,
        ):
            return False
        if not _fresh(
            received_ns,
            self.received_ns.get("LOCAL_POSITION_NED", 0),
            POSE_LEASE_NS,
        ):
            return False
        if not self.home_position or not self.global_position or not self.local_position:
            return False
        current = self._home_frame_residual(
            self.home_position, self.global_position, self.local_position
        )
        if not finite(current):
            return False
        delta = math.sqrt(sum(
            (current_axis - locked_axis) ** 2
            for current_axis, locked_axis in zip(
                current, self.home_frame_residual_vector
            )
        ))
        self.home_frame_continuity_delta_m = delta
        return delta <= HOME_FRAME_CONTINUITY_THRESHOLD_M

    def note_arm_transmitted(self, mission_id, request_id, now_ns):
        """Record a validated physical low-speed ARM write, never an armed state.

        The bridge calls this only after its exact contract and transport gates.
        Retransmission must not extend the first write's confirmation window.
        """
        if (not self.home_locked or mission_id != self.execution_home_mission_id
                or not request_id or self.home_epoch_failure_latched
                or self._arm_home_transition is not None):
            return False
        self._arm_home_transition = dict(
            mission_id=mission_id, request_id=request_id, started_ns=int(now_ns),
            epoch=self.transport_epoch, revision=self.execution_home_lock_revision,
            used=False, confirmed=False, failed=False)
        return True

    def _fail_arm_home_transition(self, reason):
        transition = self._arm_home_transition
        if transition is None:
            return
        transition['failed'] = True
        if transition['used'] and not self.home_epoch_failure_latched:
            self.home_epoch_failure_latched = True
            self.home_epoch_failure_detail = reason
            self.home_correction_detail = reason
            self.home_correction_valid = False
            self.home_phase = HOME_PHASE_REJECTED

    def note_arm_ack(self, request_id, result):
        transition = self._arm_home_transition
        if (transition is not None and transition['request_id'] == request_id
                and result not in (0, 5)):  # ACCEPTED / IN_PROGRESS are not armed proof.
            self._fail_arm_home_transition('arm_home_transition_arm_rejected')

    def arm_home_transition_valid(self, now_ns):
        transition = self._arm_home_transition
        if transition is None or transition['failed'] or transition['confirmed']:
            return False
        if not 0 <= now_ns-transition['started_ns'] <= ARM_HOME_TRANSITION_NS:
            self._fail_arm_home_transition('arm_home_transition_confirmation_timeout')
            return False
        heartbeat = self.heartbeat or {}
        if (not self.home_locked or self.home_epoch_failure_latched or self.fault
                or transition['mission_id'] != self.execution_home_mission_id
                or transition['epoch'] != self.transport_epoch
                or transition['revision'] != self.execution_home_lock_revision
                or not _fresh(now_ns, self.received_ns.get('HEARTBEAT', 0), self.heartbeat_lease_ns)
                or ((heartbeat.get('custom_mode', 0) >> 16) & 0xFF) != PX4_MAIN_MODE_OFFBOARD
                or heartbeat.get('system_status') not in (MAV_STATE_STANDBY, MAV_STATE_ACTIVE)):
            self._fail_arm_home_transition('arm_home_transition_ownership_or_epoch_lost')
            return False
        return True

    def _accept_home_candidate(self, candidate, received_ns):
        if self.home_position is None:
            self.home_position = candidate
            self.latest_px4_home_position = dict(candidate)
            self.provisional_home_revision += 1
            self.home_candidate_since_ns = received_ns
            return True

        if self.latest_px4_home_position is not None:
            latest_horizontal, latest_local = self._home_difference(
                self.latest_px4_home_position, candidate)
            if latest_horizontal <= 1e-3 and latest_local <= 1e-3:
                self.home_change_candidate = None
                self.home_change_count = 0
                return True

        horizontal, local = self._home_difference(self.home_position, candidate)
        self.last_home_message_delta_m = max(horizontal, local)
        changed = (
            horizontal > HOME_CHANGE_THRESHOLD_M
            or local > HOME_CHANGE_THRESHOLD_M
        )
        if not changed and not self.home_locked:
            self.latest_px4_home_position = dict(candidate)
            self.home_position = dict(candidate)
            self.provisional_home_revision += 1
            self.home_candidate_since_ns = received_ns
            self.home_correction_detail = "provisional_home_refined"
            self.home_change_candidate = None
            self.home_change_count = 0
            return True

        # PX4 can refine Home for a few seconds after boot. Before a validated
        # route frame is exposed, follow that candidate and restart stability.
        if not self.home_locked:
            self.home_position = candidate
            self.latest_px4_home_position = dict(candidate)
            self.provisional_home_revision += 1
            self.home_candidate_since_ns = received_ns
            self.home_phase = HOME_PHASE_PROVISIONAL
            self.home_epoch_failure_latched = False
            self.home_epoch_failure_detail = ""
            self.home_correction_detail = "provisional_home_refined"
            self.home_change_candidate = None
            self.home_change_count = 0
            return True

        heartbeat = self.heartbeat or {}
        armed = bool(heartbeat.get("base_mode", 0) & MAV_MODE_FLAG_SAFETY_ARMED)
        arm_transition = not armed and self.arm_home_transition_valid(received_ns)
        if not armed and not arm_transition:
            self.home_correction_valid = False
            self.home_correction_detail = "frozen_home_changed_before_arm"
            self.home_phase = HOME_PHASE_REJECTED
            if not self.home_epoch_failure_latched:
                self.home_epoch_failure_latched = True
                self.home_epoch_failure_detail = self.home_correction_detail
            return self._reject(self.home_correction_detail)

        # Once exposed to planning/control, mission Home is immutable. PX4 may
        # still publish a refined HOME_POSITION while arming. Ignore that
        # metadata-only change immediately when the independently observed
        # global-to-local transform remains continuous. This prevents a benign
        # arming-time Home refinement from stopping the stream while retaining
        # fail-closed handling for a real EKF/local-NED reset.
        if arm_transition:
            self._arm_home_transition['used'] = True
        correction_reason = self._home_only_correction_reason(
            candidate, received_ns)
        if not correction_reason:
            previous = self.latest_px4_home_position or self.home_position
            delta_altitude = candidate["altitude_m"]-previous["altitude_m"]
            delta_z = candidate["ned"][2]-previous["ned"][2]
            self.latest_px4_home_position = dict(candidate)
            self.home_correction_valid = True
            self.home_correction_revision += 1
            self.home_correction_opposition_error_m = abs(
                delta_altitude+delta_z)
            self.home_correction_detail = "px4_home_only_correction_applied"
            self.home_phase = HOME_PHASE_EXECUTION_LOCKED
            self.home_change_candidate = None
            self.home_change_count = 0
            self.ignored_home_refinements += 1
            return True

        self.home_correction_valid = False
        self.home_correction_detail = correction_reason
        self.home_phase = HOME_PHASE_CORRECTION_PENDING

        # After route geometry is exposed, one noisy sample invalidates the
        # frame temporarily but does not permanently poison the epoch. A real
        # change is latched only after repeated, mutually consistent samples.
        pending = self.home_change_candidate
        if pending is None:
            self.home_change_candidate = candidate
            self.home_change_count = 1
        else:
            pending_horizontal, pending_local = self._home_difference(
                pending, candidate
            )
            if (
                pending_horizontal <= HOME_CHANGE_THRESHOLD_M
                and pending_local <= HOME_CHANGE_THRESHOLD_M
            ):
                self.home_change_count += 1
            else:
                self.home_change_candidate = candidate
                self.home_change_count = 1
        if self.home_change_count >= self.home_change_confirmations_required:
            if not self.home_epoch_failure_latched:
                self.home_epoch_failure_latched = True
                self.home_epoch_failure_detail = correction_reason
            self.home_phase = HOME_PHASE_REJECTED
            return self._reject(correction_reason)
        return True

    def lock_execution_home(self, mission_id):
        """Freeze the current provisional Home for one validated flight contract.

        Merely reading route state must never lock Home.  The MAVLink bridge
        calls this only after command, envelope and setpoint have passed the
        exact contract checks, immediately before the first FC write.
        """
        mission = str(mission_id or "")
        if not mission or self.home_position is None:
            return False, "execution_home_lock_missing_candidate"
        if self.home_locked:
            if self.execution_home_mission_id != mission:
                return False, "execution_home_locked_by_other_mission"
            return True, "execution_home_already_locked"
        current = self.latest_px4_home_position or self.home_position
        self.home_position = dict(current)
        self.latest_px4_home_position = dict(current)
        self.home_locked = True
        self._arm_home_transition = None
        self.execution_home_mission_id = mission
        self.execution_home_lock_revision += 1
        self.home_phase = HOME_PHASE_EXECUTION_LOCKED
        self.home_lock_reset_counter = self.reset_counter
        self.home_correction_revision = 0
        self.home_horizontal_global_delta_m = 0.0
        self.home_horizontal_local_delta_m = 0.0
        self.home_horizontal_consistency_error_m = 0.0
        self.home_correction_valid = self.reset_counter is not None
        self.home_correction_detail = (
            "execution_home_locked"
            if self.home_correction_valid
            else "execution_home_locked_without_reset_counter")
        self.home_epoch_failure_latched = False
        self.home_epoch_failure_detail = ""
        if self.global_position and self.local_position:
            residual = self._home_frame_residual(
                self.home_position, self.global_position, self.local_position)
            if finite(residual):
                self.home_frame_residual_vector = residual
                self.home_frame_continuity_delta_m = 0.0
        return True, "execution_home_locked"

    def release_execution_home(self, mission_id=""):
        """Release a completed contract; callers must prove landed/disarmed."""
        mission = str(mission_id or "")
        if not self.home_locked:
            return True
        if mission and mission != self.execution_home_mission_id:
            return False
        current = self.latest_px4_home_position or self.home_position
        self.home_position = dict(current) if current is not None else None
        self.home_locked = False
        self._arm_home_transition = None
        self.provisional_home_revision += 1
        self.yaw_reset_context = None
        self.execution_home_mission_id = ""
        self.home_phase = HOME_PHASE_PROVISIONAL
        self.home_candidate_since_ns = time.monotonic_ns()
        self.home_frame_residual_vector = None
        self.home_lock_reset_counter = None
        self.home_correction_valid = False
        self.home_horizontal_global_delta_m = 0.0
        self.home_horizontal_local_delta_m = 0.0
        self.home_horizontal_consistency_error_m = 0.0
        self.home_correction_detail = "execution_home_released"
        self.home_change_candidate = None
        self.home_change_count = 0
        self.home_epoch_failure_latched = False
        self.home_epoch_failure_detail = ""
        return True

    def provisional_home_snapshot(self):
        current = self.latest_px4_home_position or self.home_position
        if current is None:
            return None
        return {
            "altitude_amsl_m": float(current["altitude_m"]),
            "z_ned_m": float(current["ned"][2]),
            "revision": int(self.provisional_home_revision),
            "detail": str(self.home_correction_detail),
        }

    def _home_only_correction_reason(self, candidate, received_ns):
        previous = self.latest_px4_home_position or self.home_position
        east_delta, north_delta = _wgs84_delta_m(
            previous["latitude_deg"], previous["longitude_deg"],
            candidate["latitude_deg"], candidate["longitude_deg"],
        )
        horizontal = math.hypot(east_delta, north_delta)
        local_x_delta = candidate["ned"][0]-previous["ned"][0]
        local_y_delta = candidate["ned"][1]-previous["ned"][1]
        xy_delta = math.hypot(local_x_delta, local_y_delta)
        horizontal_error = math.hypot(
            north_delta-local_x_delta, east_delta-local_y_delta)
        self.home_horizontal_global_delta_m = horizontal
        self.home_horizontal_local_delta_m = xy_delta
        self.home_horizontal_consistency_error_m = horizontal_error
        delta_altitude = candidate["altitude_m"]-previous["altitude_m"]
        delta_z = candidate["ned"][2]-previous["ned"][2]
        opposition_error = abs(delta_altitude+delta_z)
        self.home_correction_opposition_error_m = opposition_error
        # HOME_POSITION carries the same physical point in WGS84 and local
        # NED. A bounded metadata refinement may move both representations;
        # neither can move alone or disagree with the frozen position frame.
        if (max(horizontal, xy_delta) > HOME_CORRECTION_HORIZONTAL_METADATA_MAX_M
                or horizontal_error > HOME_CORRECTION_HORIZONTAL_CONSISTENCY_MAX_M):
            return "home_correction_horizontal_change"
        if max(abs(delta_altitude), abs(delta_z)) > HOME_CORRECTION_MAX_M:
            return "home_correction_magnitude_exceeded"
        if opposition_error > HOME_CORRECTION_OPPOSITION_MAX_M:
            return "home_correction_altitude_z_not_opposed"
        if not self._frozen_home_frame_is_continuous(received_ns):
            return "home_correction_frozen_frame_not_continuous"
        for kind in ("LOCAL_POSITION_NED", "GLOBAL_POSITION_INT", "ODOMETRY"):
            if not _fresh(
                received_ns, self.received_ns.get(kind, 0),
                HOME_CORRECTION_EVIDENCE_LEASE_NS,
            ):
                return "home_correction_evidence_stale"
        if self.reset_counter is None:
            return "home_correction_reset_counter_missing"
        if (self.home_lock_reset_counter is not None
                and self.reset_counter != self.home_lock_reset_counter):
            return "home_correction_estimator_reset_changed"
        if self.previous_local_position is None or self.local_position is None:
            return "home_correction_local_continuity_missing"
        if abs(
            self.local_position["position"][2]
            - self.previous_local_position["position"][2]
        ) > HOME_CORRECTION_SAMPLE_CONTINUITY_MAX_M:
            return "home_correction_local_z_discontinuity"
        if self.previous_global_position is None or self.global_position is None:
            return "home_correction_amsl_continuity_missing"
        if abs(
            self.global_position["altitude_m"]
            - self.previous_global_position["altitude_m"]
        ) > HOME_CORRECTION_SAMPLE_CONTINUITY_MAX_M:
            return "home_correction_amsl_discontinuity"
        if max(horizontal, xy_delta) > HOME_CORRECTION_HORIZONTAL_CONSISTENCY_MAX_M:
            local_xy = math.hypot(
                self.local_position["position"][0]-self.previous_local_position["position"][0],
                self.local_position["position"][1]-self.previous_local_position["position"][1],
            )
            if local_xy > HOME_CORRECTION_SAMPLE_CONTINUITY_MAX_M:
                return "home_correction_local_xy_discontinuity"
            if _wgs84_distance_m(
                    self.previous_global_position["latitude_deg"],
                    self.previous_global_position["longitude_deg"],
                    self.global_position["latitude_deg"],
                    self.global_position["longitude_deg"],
            ) > HOME_CORRECTION_SAMPLE_CONTINUITY_MAX_M:
                return "home_correction_global_xy_discontinuity"
        return ""

    def home_correction_snapshot(self):
        frozen = self.home_position
        current = self.latest_px4_home_position or frozen
        if frozen is None or current is None:
            return None
        return {
            "frozen_altitude_amsl_m": float(frozen["altitude_m"]),
            "frozen_z_ned_m": float(frozen["ned"][2]),
            "current_altitude_amsl_m": float(current["altitude_m"]),
            "current_z_ned_m": float(current["ned"][2]),
            "correction_valid": bool(
                self.home_locked and self.reset_counter is not None and (
                    self.home_correction_valid
                    or self.home_correction_revision == 0)),
            "correction_revision": int(self.home_correction_revision),
            "opposition_error_m": float(
                self.home_correction_opposition_error_m),
            "estimator_reset_counter_valid": self.reset_counter is not None,
            "estimator_reset_counter": int(self.reset_counter or 0),
            "detail": str(self.home_correction_detail),
            "home_phase": str(self.home_phase),
            "execution_home_lock_valid": bool(self.home_locked),
            "execution_home_mission_id": str(self.execution_home_mission_id),
            "execution_home_lock_revision": int(
                self.execution_home_lock_revision),
            "provisional_home_revision": int(self.provisional_home_revision),
            "provisional_altitude_amsl_m": float(
                current["altitude_m"]),
            "provisional_z_ned_m": float(current["ned"][2]),
            "epoch_failure_latched": bool(self.home_epoch_failure_latched),
        }

    def accept(self, kind, value, system_id, component_id, received_ns):
        key = kind if kind in ROUTE_MESSAGE_KINDS else "OTHER"
        self.rx_counts[key] = self.rx_counts.get(key, 0) + 1
        accepted = self._accept_value(kind, value, system_id, component_id, received_ns)
        reason = "" if accepted else (self.last_rejection or "rejected")
        self.last_decisions[key] = {
            "decision": "accepted" if accepted else "rejected",
            "reason": reason,
            "received_monotonic_ns": received_ns if type(received_ns) is int else None,
        }
        if accepted:
            self.accepted_counts[key] = self.accepted_counts.get(key, 0) + 1
            self.accepted_arrivals[key].append(received_ns)
        else:
            reasons = self.rejected_counts.setdefault(key, {})
            reasons[reason] = reasons.get(reason, 0) + 1
        return accepted

    def _accept_value(self, kind, value, system_id, component_id, received_ns):
        self.last_rejection = ""
        if (system_id, component_id) != (TARGET_SYSTEM, TARGET_COMPONENT):
            return self._reject("unexpected_fc_source")
        if type(received_ns) is not int or received_ns <= 0:
            return self._reject("invalid_receipt_time")
        if self.fault and kind not in ("HEARTBEAT", "EXTENDED_SYS_STATE"):
            return self._reject(self.fault)
        if not isinstance(value, dict):
            return self._reject("message_dictionary_required")

        if kind == "HEARTBEAT":
            required = ("autopilot", "type", "base_mode", "custom_mode", "system_status")
            if not all(type(value.get(name)) is int for name in required):
                return self._reject("invalid_heartbeat")
            if value["autopilot"] != 12 or value["type"] != 2:
                self.fault = "unexpected_autopilot_identity"
                return self._reject(self.fault)
            self.heartbeat = {name: value[name] for name in required}
            if self.arm_home_transition_valid(received_ns):
                if value['base_mode'] & MAV_MODE_FLAG_SAFETY_ARMED:
                    self._arm_home_transition['confirmed'] = True
        elif kind == "SYS_STATUS":
            names = (
                "onboard_control_sensors_present",
                "onboard_control_sensors_enabled",
                "onboard_control_sensors_health",
            )
            if not all(type(value.get(name)) is int and 0 <= value[name] < 2**32 for name in names):
                return self._reject("invalid_system_sensor_status")
            self.sys_status = {name: value[name] for name in names}
        elif kind == "BATTERY_STATUS":
            try:
                sample = decode_battery(value, received_ns, self.transport_epoch)
            except ValueError as exc:
                return self._reject(str(exc))
            prior = self.battery_samples.get(sample['id'])
            if prior is not None and received_ns <= prior['received_ns']:
                return self._reject('battery_status_receipt_reversed')
            self.battery_samples[sample['id']] = sample
        elif kind == "ESTIMATOR_STATUS":
            flags = value.get("flags")
            stamp = value.get("time_usec")
            if type(flags) is not int or not 0 <= flags <= 65535:
                return self._reject("invalid_estimator_flags")
            if not self._accept_stamp(kind, stamp, received_ns):
                return False
            self.estimator_status = {"flags": flags, "time_usec": stamp}
        elif kind == "EXTENDED_SYS_STATE":
            landed_state = value.get("landed_state")
            if type(landed_state) is not int or landed_state not in (1, 2, 3, 4):
                return self._reject("invalid_landed_state")
            self.extended_state = {"landed_state": landed_state}
        elif kind == "ATTITUDE_QUATERNION":
            stamp = value.get("time_boot_ms")
            quaternion = [value.get(name) for name in ("q1", "q2", "q3", "q4")]
            yaw = _quaternion_yaw(quaternion)
            stamp_us = stamp * 1000 if type(stamp) is int else stamp
            if not self._accept_stamp(kind, stamp_us, received_ns) or not math.isfinite(yaw):
                return self._reject(self.last_rejection or "invalid_attitude")
            self.attitude = {"yaw": yaw, "time_boot_ms": stamp,
                "q": tuple(quaternion),
                "angular_rates": tuple(value.get(name, math.nan) for name in
                                       ('rollspeed', 'pitchspeed', 'yawspeed'))}
        elif kind == "LOCAL_POSITION_NED":
            stamp = value.get("time_boot_ms")
            xyzv = [value.get(name) for name in ("x", "y", "z", "vx", "vy", "vz")]
            stamp_us = stamp * 1000 if type(stamp) is int else stamp
            if not self._accept_stamp(kind, stamp_us, received_ns) or not finite(xyzv):
                return self._reject(self.last_rejection or "invalid_local_position")
            self.previous_local_position = self.local_position
            self.local_position = {
                "position": tuple(float(item) for item in xyzv[:3]),
                "velocity": tuple(float(item) for item in xyzv[3:]),
                "time_boot_ms": stamp,
            }
        elif kind == "GLOBAL_POSITION_INT":
            stamp = value.get("time_boot_ms")
            fields = [value.get(name) for name in ("lat", "lon", "alt", "relative_alt")]
            stamp_us = stamp * 1000 if type(stamp) is int else stamp
            if (not self._accept_stamp(kind, stamp_us, received_ns)
                    or not all(type(item) is int for item in fields)):
                return self._reject(self.last_rejection or "invalid_global_position")
            latitude, longitude = fields[0]/1e7, fields[1]/1e7
            if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                return self._reject("global_position_out_of_range")
            self.previous_global_position = self.global_position
            self.global_position = {
                "latitude_deg": latitude, "longitude_deg": longitude,
                "altitude_m": fields[2]/1000.0,
                "relative_altitude_m": fields[3]/1000.0,
                "time_boot_ms": stamp,
            }
        elif kind == "HOME_POSITION":
            fields = [value.get(name) for name in (
                "latitude", "longitude", "altitude", "x", "y", "z")]
            if (not all(type(item) in (int, float) for item in fields)
                    or not finite([float(item) for item in fields])):
                return self._reject("invalid_home_position")
            candidate = {
                "latitude_deg": fields[0]/1e7,
                "longitude_deg": fields[1]/1e7,
                "altitude_m": fields[2]/1000.0,
                "ned": tuple(float(item) for item in fields[3:]),
            }
            if not (-90 <= candidate["latitude_deg"] <= 90
                    and -180 <= candidate["longitude_deg"] <= 180):
                return self._reject("home_position_out_of_range")
            if not self._accept_home_candidate(candidate, received_ns):
                return False
        elif kind == "ODOMETRY":
            stamp_us = value.get("time_usec")
            reset_counter = value.get("reset_counter")
            frame_id = value.get("frame_id")
            child_frame_id = value.get("child_frame_id")
            xyzv = [value.get(name) for name in ("x", "y", "z", "vx", "vy", "vz")]
            yaw = _quaternion_yaw(value.get("q"))
            if (
                frame_id != 1
                or child_frame_id not in (1, 12)
                or type(reset_counter) is not int
                or not 0 <= reset_counter <= 255
                or not self._accept_stamp(kind, stamp_us, received_ns)
                or not finite(xyzv)
                or not math.isfinite(yaw)
            ):
                return self._reject(self.last_rejection or "invalid_local_ned_odometry")
            from .yaw_reset import ned_velocity
            candidate = {
                'position': tuple(float(item) for item in xyzv[:3]),
                'velocity': tuple(float(item) for item in xyzv[3:]),
                'proof_velocity': ned_velocity(value['q'], xyzv[3:], child_frame_id),
                'q': tuple(value['q']), 'time_usec': stamp_us,
                'yaw': yaw, 'time_boot_ms': (stamp_us // 1000) & 0xFFFFFFFF}
            if (self.reset_counter is not None and reset_counter != self.reset_counter
                    and not self._accept_bounded_yaw_reset(candidate, reset_counter, received_ns)):
                self.fault = "local_odometry_reset_changed"
                if self.home_locked and not self.home_epoch_failure_latched:
                    self.home_epoch_failure_latched = True
                    self.home_epoch_failure_detail = self.fault
                    self.home_correction_detail = self.fault
                    self.home_phase = HOME_PHASE_REJECTED
                return self._reject(self.fault)
            self.reset_counter = reset_counter
            if self.navigation_reset_generations is None:
                self.navigation_reset_generations = (reset_counter,)*5
            if self.home_locked and self.home_lock_reset_counter is None:
                self.home_lock_reset_counter = reset_counter
            self.odometry = candidate
        else:
            return self._reject("unsupported_route_message")
        self.received_ns[kind] = received_ns
        if kind in POSE_MESSAGE_KINDS:
            self._remember_pose_row(kind, received_ns)
        return True

    def _remember_pose_row(self, kind, received_ns):
        if self.source_stamps[kind] < self.yaw_reset_source_stamp_us:
            return
        values = {
            "LOCAL_POSITION_NED": self.local_position,
            "ATTITUDE_QUATERNION": self.attitude,
            "ESTIMATOR_STATUS": self.estimator_status,
        }
        self.pose_rows[kind].append({
            "value": dict(values[kind]),
            "source_stamp_us": self.source_stamps[kind],
            "received_ns": received_ns,
        })
        candidates = []
        if all(self.pose_rows[name] for name in POSE_MESSAGE_KINDS):
            for rows in product(*(self.pose_rows[name] for name in POSE_MESSAGE_KINDS)):
                source_stamps = [row["source_stamp_us"] for row in rows]
                receipts = [row["received_ns"] for row in rows]
                if (max(source_stamps) - min(source_stamps) <= POSE_PAIR_SKEW_NS // 1000
                        and max(receipts) - min(receipts) <= POSE_PAIR_SKEW_NS):
                    candidates.append((max(receipts), rows, source_stamps, receipts))
        if not candidates:
            return
        _, rows, source_stamps, receipts = max(candidates, key=lambda item: item[0])
        self.pose_bundle = {
            name: dict(row["value"])
            for name, row in zip(POSE_MESSAGE_KINDS, rows)
        }
        self.pose_bundle.update({
            "source_stamps_us": tuple(source_stamps),
            "received_ns": tuple(receipts),
            "source_skew_us": max(source_stamps) - min(source_stamps),
            "receipt_skew_ns": max(receipts) - min(receipts),
            "completed_ns": max(receipts),
        })

    def route_home(self, now_ns, residual_limit_m=5.0):
        """Return Home only while fresh Global and Local positions agree."""
        if self.fault:
            return None, self.fault, math.inf
        if self.home_change_candidate is not None:
            return None, "home_position_change_unconfirmed", math.inf
        if not _fresh(now_ns, self.received_ns.get("HOME_POSITION", 0), HOME_LEASE_NS):
            return None, "home_position_missing_or_stale", math.inf
        if (
            self.home_position is None
            or self.home_candidate_since_ns <= 0
            or now_ns - self.home_candidate_since_ns < self.home_stability_ns
        ):
            return None, "home_position_stabilizing", math.inf
        if not _fresh(now_ns, self.received_ns.get("GLOBAL_POSITION_INT", 0), GLOBAL_LEASE_NS):
            return None, "global_position_missing_or_stale", math.inf
        if not _fresh(now_ns, self.received_ns.get("LOCAL_POSITION_NED", 0), POSE_LEASE_NS):
            return None, "local_position_missing_or_stale", math.inf
        home = self.home_position
        residual_vector = self._home_frame_residual(
            home, self.global_position, self.local_position
        )
        residual = math.sqrt(sum(value ** 2 for value in residual_vector))
        if not math.isfinite(residual) or residual > float(residual_limit_m):
            return None, "home_global_local_residual_exceeds_limit", residual
        # This method is read-only with respect to the Home lifecycle.  A
        # planner/status query may expose a provisional frame, but only the
        # first exact flight-output contract may freeze it.
        return dict(home), "", residual

    def snapshot(self, now_ns):
        self.arm_home_transition_valid(now_ns)
        heartbeat_fresh = _fresh(
            now_ns,
            self.received_ns.get("HEARTBEAT", 0),
            self.heartbeat_lease_ns,
        )
        # Link/arming truth is derived from the transport heartbeat only.
        # Route-frame faults fence navigation output but must not turn a fresh
        # DISARMED/ARMED report into an unknown state on the dashboard.
        connected = heartbeat_fresh
        heartbeat = self.heartbeat or {}
        base_mode = heartbeat.get("base_mode", 0)
        custom_mode = heartbeat.get("custom_mode", 0)
        main_mode = (
            (custom_mode >> 16) & 0xFF
            if base_mode & MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
            else None
        )
        armed = bool(base_mode & MAV_MODE_FLAG_SAFETY_ARMED)
        system_state_ok = heartbeat.get("system_status") == (
            MAV_STATE_ACTIVE if armed else MAV_STATE_STANDBY
        )

        sys_fresh = _fresh(now_ns, self.received_ns.get("SYS_STATUS", 0), STATUS_LEASE_NS)
        sensors_ok = False
        present = enabled = health = 0
        health_reason = "sys_status_missing_or_stale"
        battery_only = False
        if sys_fresh and self.sys_status:
            present = self.sys_status["onboard_control_sensors_present"]
            enabled = self.sys_status["onboard_control_sensors_enabled"]
            health = self.sys_status["onboard_control_sensors_health"]
            sensors_ok = enabled != 0 and (enabled & ~present) == 0 and (health & enabled) == enabled
            failed = enabled & ~health
            battery = 1 << 25  # MAV_SYS_STATUS_SENSOR_BATTERY
            battery_only = bool(enabled and not (enabled & ~present)
                                and present & enabled & battery and failed == battery)
            health_reason = ("ready" if sensors_ok else
                "battery_health_terminal_only" if battery_only else
                "sensor_presence_invalid" if not enabled or enabled & ~present else
                "non_battery_or_multiple_sensor_health_failure")

        estimator_fresh = _fresh(
            now_ns, self.received_ns.get("ESTIMATOR_STATUS", 0), POSE_LEASE_NS
        )
        estimator_ok = bool(
            estimator_fresh
            and self.estimator_status
            and not route_estimator_reason(self.estimator_status["flags"])
        )

        local_fresh = _fresh(
            now_ns, self.received_ns.get("LOCAL_POSITION_NED", 0), POSE_LEASE_NS
        )
        attitude_fresh = _fresh(
            now_ns, self.received_ns.get("ATTITUDE_QUATERNION", 0), POSE_LEASE_NS
        )
        odometry_fresh = _fresh(now_ns, self.received_ns.get("ODOMETRY", 0), POSE_LEASE_NS)
        local_pair_receipts = [
            self.received_ns.get(kind, 0)
            for kind in ("LOCAL_POSITION_NED", "ATTITUDE_QUATERNION", "ESTIMATOR_STATUS")
        ]
        local_pair_stamps_us = [
            self.source_stamps.get(kind, 0)
            for kind in ("LOCAL_POSITION_NED", "ATTITUDE_QUATERNION", "ESTIMATOR_STATUS")
        ]
        local_pair_coherent = (
            all(local_pair_receipts)
            and max(local_pair_receipts) - min(local_pair_receipts) <= POSE_PAIR_SKEW_NS
            and all(local_pair_stamps_us)
            and max(local_pair_stamps_us) - min(local_pair_stamps_us) <= POSE_PAIR_SKEW_NS // 1000
        )
        odometry_pair_coherent = (
            self.received_ns.get("ODOMETRY", 0) > 0
            and self.received_ns.get("ESTIMATOR_STATUS", 0) > 0
            and abs(
                self.received_ns["ODOMETRY"] - self.received_ns["ESTIMATOR_STATUS"]
            ) <= POSE_PAIR_SKEW_NS
            and self.source_stamps.get("ODOMETRY", 0) > 0
            and self.source_stamps.get("ESTIMATOR_STATUS", 0) > 0
            and abs(
                self.source_stamps["ODOMETRY"] - self.source_stamps["ESTIMATOR_STATUS"]
            ) <= POSE_PAIR_SKEW_NS // 1000
        )
        if (
            local_fresh
            and attitude_fresh
            and local_pair_coherent
            and self.local_position
            and self.attitude
        ):
            position = self.local_position["position"]
            velocity = self.local_position["velocity"]
            yaw = self.attitude["yaw"]
            if (self.yaw_reset_source_stamp_us and self.odometry
                    and self.source_stamps.get('ATTITUDE_QUATERNION', 0) < self.yaw_reset_source_stamp_us):
                yaw = self.odometry['yaw']
            time_boot_ms = self.local_position["time_boot_ms"]
            position_source = "LOCAL_POSITION_NED"
            position_received_ns = self.received_ns.get(
                "LOCAL_POSITION_NED", 0)
            pose_fresh = True
        elif odometry_fresh and odometry_pair_coherent and self.odometry:
            position = self.odometry["position"]
            velocity = self.odometry["velocity"]
            yaw = self.odometry["yaw"]
            time_boot_ms = self.odometry["time_boot_ms"]
            position_source = "ODOMETRY"
            position_received_ns = self.received_ns.get("ODOMETRY", 0)
            pose_fresh = True
        else:
            position = (math.nan, math.nan, math.nan)
            velocity = (math.nan, math.nan, math.nan)
            yaw = math.nan
            time_boot_ms = 0
            position_source = "NONE"
            position_received_ns = 0
            pose_fresh = False

        route_ok = not bool(self.fault)
        position_valid = bool(connected and pose_fresh and estimator_ok)
        preflight = bool(
            connected and route_ok and system_state_ok and sensors_ok
            and position_valid)
        failsafe = not bool(
            connected and route_ok and system_state_ok and sensors_ok
            and estimator_ok)
        extended_fresh = _fresh(
            now_ns, self.received_ns.get("EXTENDED_SYS_STATE", 0), STATUS_LEASE_NS
        )
        landed = (
            self.extended_state["landed_state"] == MAV_LANDED_STATE_ON_GROUND
            if extended_fresh and self.extended_state
            else None
        )

        reason = self.fault
        if not reason and not connected:
            reason = "heartbeat_missing_or_stale"
        elif not reason and not system_state_ok:
            reason = "px4_not_active_or_standby"
        elif not reason and not sensors_ok:
            reason = "system_sensor_health_missing_or_invalid"
        elif not reason and not estimator_ok:
            reason = "estimator_missing_stale_or_invalid"
        elif not reason and not pose_fresh:
            reason = "local_position_missing_or_stale"
        batteries = [s for s in self.battery_samples.values() if s['epoch'] == self.transport_epoch]
        battery_sample = batteries[0] if len(batteries) == 1 and batteries[0]['id'] == 0 else None
        return RouteFcSnapshot(
            battery_sample=battery_sample,
            battery_navigation_healthy=bool(connected and route_ok and system_state_ok
                and position_valid and estimator_ok and sys_fresh
                and bool(present & enabled & (1 << 25))
                and (sensors_ok or battery_only)),
            battery_health_terminal_only=bool(battery_only and connected and route_ok
                and system_state_ok and position_valid and estimator_ok),
            sensor_health_reason=health_reason,
            sensor_health_age_ns=(now_ns-self.received_ns["SYS_STATUS"]
                if "SYS_STATUS" in self.received_ns else -1),
            sensors_present=present, sensors_enabled=enabled, sensors_health=health,
            sensors_failed=enabled & ~health,
            connected=connected,
            preflight_checks_pass=preflight,
            position_valid=position_valid,
            armed=armed,
            offboard=connected and main_mode == PX4_MAIN_MODE_OFFBOARD,
            landed=landed,
            failsafe=failsafe,
            position=position,
            velocity=velocity,
            yaw=yaw,
            time_boot_ms=time_boot_ms,
            position_source=position_source,
            position_received_ns=position_received_ns,
            transport_epoch=self.transport_epoch,
            estimator_reset_counter_valid=self.reset_counter is not None,
            estimator_reset_counter=int(self.reset_counter or 0),
            navigation_reset_generations=(self.navigation_reset_generations or ()),
            post_yaw_attitude_ready=(self.source_stamps.get('ATTITUDE_QUATERNION', 0)
                                     >= self.yaw_reset_source_stamp_us),
            reason=reason,
            px4_custom_mode=int(custom_mode),
            native_land=bool(main_mode == 4 and ((custom_mode >> 24) & 0xFF) == 6),
        )


class MavlinkRouteTransport(UsbTelemetryTransport):
    """MAVLink encoder with a closed command surface and last-mile gates."""

    def __init__(
        self,
        journal,
        device="/dev/jolgwa-pixhawk6c",
        *,
        timing=None,
        enable_px4_commands=False,
        enable_mavlink_commands=False,
        allow_real_hardware=False,
        simulation_only=True,
    ):
        super().__init__(journal, device, timing=timing)
        # The route bridge must enforce single-threaded serial writes itself.
        # Older deployed observer transports did not expose this marker, so do
        # not rely on the base-class version for the last-mile safety check.
        self._owner_thread = threading.get_ident()
        self.flight_output_enabled = bool(
            enable_px4_commands
            and enable_mavlink_commands
            and allow_real_hardware
            and not simulation_only
        )
        self.last_setpoint_ns = 0
        self.last_heartbeat_request_ns = 0
        self.last_home_request_ns = 0
        self.last_parameter_request_ns = {}

    def _write_packet(self, message, kind, metadata, *, flight_output):
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("mavlink_write_outside_usb_owner")
        if flight_output and not self.flight_output_enabled:
            raise PermissionError("mavlink_flight_output_disabled")
        packet = message.pack(self.encoder, force_mavlink1=False)
        record = {
            "kind": kind,
            "monotonic_ns": time.monotonic_ns(),
            "flight_output": bool(flight_output),
            "packet_hex": packet.hex(),
            **metadata,
        }
        self.journal.write(json.dumps(record, allow_nan=False) + "\n")
        self.journal.flush()
        count = self.port.write(packet)
        if count != len(packet):
            raise OSError("partial_mavlink_packet_write")
        self.encoder.seq = (self.encoder.seq + 1) % 256
        self.tx_count += 1
        return True

    def request_route_stream(self, message_id):
        if type(message_id) is not int or message_id not in ROUTE_STREAM_INTERVALS_US:
            raise ValueError("not_a_fixed_route_telemetry_stream")
        interval = ROUTE_STREAM_INTERVALS_US[message_id]
        message = self.common.MAVLink_command_long_message(
            TARGET_SYSTEM, TARGET_COMPONENT, 511, 0,
            float(message_id), float(interval), 0.0, 0.0, 0.0, 0.0, 0.0,
        )
        return self._write_packet(
            message,
            "tx_route_telemetry_request",
            {"message_id": message_id, "interval_us": interval},
            flight_output=False,
        )

    def request_autopilot_version(self):
        message = self.common.MAVLink_command_long_message(
            TARGET_SYSTEM, TARGET_COMPONENT, 512, 0,
            148.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )
        return self._write_packet(
            message,
            "tx_route_autopilot_version_request",
            {"message_id": 148},
            flight_output=False,
        )

    def request_home_position(self, now_ns=None, *, period_ns=1_000_000_000, force=False):
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        if not force and self.last_home_request_ns and 0 <= now_ns-self.last_home_request_ns < period_ns:
            return False
        message = self.common.MAVLink_command_long_message(
            TARGET_SYSTEM, TARGET_COMPONENT, 512, 0,
            float(HOME_POSITION_MESSAGE_ID), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        result = self._write_packet(message, "tx_route_home_position_request",
                                    {"message_id": HOME_POSITION_MESSAGE_ID},
                                    flight_output=False)
        self.last_home_request_ns = now_ns
        return result

    def request_parameter(self, parameter_id, now_ns=None):
        """Read a fixed PX4 parameter without changing FC state."""
        if parameter_id not in ("COM_OBL_RC_ACT", "MPC_LAND_SPEED"):
            raise ValueError("parameter is not allowlisted for low-speed preflight")
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        last = self.last_parameter_request_ns.get(parameter_id, 0)
        if last and 0 <= now_ns-last < 2_000_000_000:
            return False
        message = self.common.MAVLink_param_request_read_message(
            TARGET_SYSTEM, TARGET_COMPONENT, parameter_id.encode("ascii"), -1)
        result = self._write_packet(message, "tx_route_parameter_read", {
            "parameter_id": parameter_id}, flight_output=False)
        self.last_parameter_request_ns[parameter_id] = now_ns
        return result

    def request_heartbeat(self, now_ns=None):
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        if self.last_heartbeat_request_ns and (
            now_ns < self.last_heartbeat_request_ns
            or now_ns - self.last_heartbeat_request_ns < HEARTBEAT_REQUEST_PERIOD_NS
        ):
            return False
        message = self.common.MAVLink_command_long_message(
            TARGET_SYSTEM, TARGET_COMPONENT, 512, 0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )
        result = self._write_packet(
            message,
            "tx_route_heartbeat_request",
            {"message_id": 0},
            flight_output=False,
        )
        self.last_heartbeat_request_ns = time.monotonic_ns()
        return result

    @staticmethod
    def _require_snapshot(snapshot, *, terminal=False):
        if not isinstance(snapshot, RouteFcSnapshot) or not snapshot.connected:
            raise PermissionError("fresh_fc_connection_required")
        if terminal:
            if not snapshot.armed:
                raise PermissionError("armed_fc_required_for_terminal_command")
        elif snapshot.failsafe or not snapshot.preflight_checks_pass or not snapshot.position_valid:
            raise PermissionError("healthy_preflight_and_position_required")

    def send_setpoint(self, setpoint, snapshot, *, now_ns=None, terminal_brake=False):
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        if not isinstance(setpoint, RouteSetpoint):
            raise ValueError("validated_route_setpoint_required")
        if terminal_brake:
            self._require_snapshot(snapshot, terminal=True)
            if (not snapshot.position_valid or not snapshot.offboard
                    or setpoint.type_mask not in (VELOCITY_YAW_MASK, VELOCITY_YAW_RATE_MASK)
                    or any(abs(v) > 1e-6 or not math.isfinite(v) for v in setpoint.velocity)
                    # make_route_setpoint normalizes ignored MAVLink fields
                    # to zero. The mask, not a ROS NaN, proves velocity mode.
                    or setpoint.position != (0.0, 0.0, 0.0)):
                raise PermissionError("terminal_brake_navigation_or_contract_invalid")
        else:
            self._require_snapshot(snapshot)
        if self.last_setpoint_ns and (
            now_ns < self.last_setpoint_ns or now_ns - self.last_setpoint_ns < SETPOINT_PERIOD_NS
        ):
            return False
        message = self.common.MAVLink_set_position_target_local_ned_message(
            snapshot.time_boot_ms & 0xFFFFFFFF,
            TARGET_SYSTEM,
            TARGET_COMPONENT,
            1,  # MAV_FRAME_LOCAL_NED
            setpoint.type_mask,
            *setpoint.position,
            *setpoint.velocity,
            0.0,
            0.0,
            0.0,
            setpoint.yaw,
            setpoint.yaw_rate,
        )
        result = self._write_packet(
            message,
            "tx_route_setpoint_local_ned",
            {
                "type_mask": setpoint.type_mask,
                "position_ned_m": list(setpoint.position),
                "velocity_ned_mps": list(setpoint.velocity),
                "yaw_rad": setpoint.yaw,
                "yaw_rate_rad_s": setpoint.yaw_rate,
            },
            flight_output=True,
        )
        self.last_setpoint_ns = time.monotonic_ns()
        return result

    def send_vehicle_command(self, command, params, snapshot):
        if type(command) is not int or command not in ALLOWED_VEHICLE_COMMANDS:
            raise ValueError("vehicle_command_not_allowlisted")
        if not isinstance(params, (tuple, list)) or len(params) != 7 or not finite(params):
            raise ValueError("seven_finite_vehicle_command_parameters_required")
        values = tuple(float(value) for value in params)
        terminal = command in (MAV_CMD_NAV_LAND, MAV_CMD_NAV_RETURN_TO_LAUNCH)
        self._require_snapshot(snapshot, terminal=terminal)
        if command == MAV_CMD_DO_SET_MODE and values[:2] != (1.0, 6.0):
            raise ValueError("only_px4_offboard_mode_request_is_allowed")
        if command == MAV_CMD_COMPONENT_ARM_DISARM and values[0] not in (0.0, 1.0):
            raise ValueError("invalid_arm_disarm_value")
        if command == MAV_CMD_COMPONENT_ARM_DISARM and any(values[1:]):
            raise ValueError("forced_arm_disarm_is_not_allowed")
        if command in (MAV_CMD_NAV_LAND, MAV_CMD_NAV_RETURN_TO_LAUNCH) and any(values):
            raise ValueError("terminal_command_parameters_must_be_zero")
        message = self.common.MAVLink_command_long_message(
            TARGET_SYSTEM,
            TARGET_COMPONENT,
            command,
            0,
            *values,
        )
        return self._write_packet(
            message,
            "tx_route_vehicle_command",
            {"command": command, "params": list(values)},
            flight_output=True,
        )

from . import scenario_altitude_recovery as altitude_recovery
from . import scenario_battery_home as battery_home
import json
import math
import os
import threading
from .field_diagnostics import record as field_record
import time
import copy
import uuid
from functools import wraps

import rclpy
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.clock import Clock as RosClock, ClockType
from rosgraph_msgs.msg import Clock as ClockMessage
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import String
from jolgwa_interfaces.msg import (
    AltitudeReferenceState,
    FlightEnvelope,
    FlightOutputState,
    FlightControlContract, FlightCommandRequest, FlightCommandAck,
    ManualOverride,
    MissionApproval,
    OffboardCommand,
    SafetyDecision,
    VehicleControlState,
    VehicleGeoState,
)
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleCommandAck,
    VehicleLandDetected,
    VehicleLocalPosition,
    VehicleStatus,
)
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data

from .authority import ControlAuthority, JetsonSafetyState, resolve_authority
from .safety import SafetyGate
from .scenario_runtime import init_contract, spec_for, altitude_limit, navigation_error
from . import scenario_pending
from .flight_contract import (proof_fresh, PROOF_LEASE_NS, output_identity_fresh,
    BATTERY_TERMINAL_ONLY, BATTERY_TERMINAL_UNAVAILABLE)
from .route_heading_gate import RouteHeadingGate, angle_error
from .event_home_return import near_home_for_landing
from .low_speed import (
    accumulated_sample_age_s, altitude_sample_advances,
    ALTITUDE_REFERENCE_MAX_ERROR_M, LOW_SPEED_1M_V1, LOW_SPEED_2M_V1,
    MAX_HORIZONTAL_SPEED_M_S,
    MAX_VERTICAL_SPEED_M_S, forward_test_target_within_camera_bypass,
    altitude_reference_sample_error,
    flight_output_confirmation_ready, max_altitude_for_profile,
    takeoff_handshake_action,
    target_altitude_for_profile,
)
from .sim_clock_lease import (
    SimClockLease, clock_nanoseconds, validate_sim_clock_profile,
    validate_relaxed_timing_profile,
)
from .safety_stamp_diagnostics import (
    SafetyStampDiagnostics, TOPIC as SAFETY_STAMP_DIAGNOSTIC_TOPIC,
    validate_diagnostic_profile,
)
from .topic_names import (
    ALTITUDE_REFERENCE_STATE,
    JETSON_SAFETY_DECISION,
    MANUAL_OVERRIDE,
    MANUAL_VELOCITY,
    MISSION_APPROVAL,
    OFFBOARD_COMMAND,
    PX4_COMMAND_ACK,
    PX4_LAND_DETECTED,
    PX4_LOCAL_POSITION,
    PX4_OFFBOARD_CONTROL_MODE,
    PX4_TRAJECTORY_SETPOINT,
    PX4_VEHICLE_COMMAND,
    PX4_VEHICLE_STATUS,
    VEHICLE_CONTROL_STATE,
    VEHICLE_GEO_STATE,
    FLIGHT_ENVELOPE,
    FLIGHT_OUTPUT_STATE,
    FLIGHT_CONTROL_CONTRACT, FLIGHT_COMMAND_REQUEST, FLIGHT_COMMAND_ACK,
)


def px4_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


def serialized_state(method):
    """Serialize ROS callbacks with the independent steady heartbeat thread.

    Each callback, including the final non-blocking ROS publication, observes
    one state transaction. No HTTP, sleep, or acknowledgement waits run here.
    """
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._state_lock:
            return method(self, *args, **kwargs)
    return wrapped


class Px4OffboardController(Node):
    """The only node allowed to publish PX4 /fmu/in control messages."""

    HEARTBEAT_PERIOD_S = 0.05

    def __init__(self) -> None:
        super().__init__("px4_offboard_controller")
        init_contract(self)
        self.declare_parameter("usb_output_contract", False)
        self._usb_output_contract = bool(self.get_parameter("usb_output_contract").value)
        self._output_epoch = uuid.uuid4().hex
        self._pending_request_ids = {}
        self._output_sequence = 0
        self._output_contract = None
        self._first_fault = ""
        self._native_land = False
        self._arm_transmitted = False
        self._state_lock = threading.RLock()
        self.declare_parameter("simulation_only", True)
        self.declare_parameter("enable_px4_commands", False)
        # A non-DDS adapter has its own independent last-mile output gate.
        # DDS keeps the default True, so the existing behavior is unchanged.
        self.declare_parameter("enable_transport_commands", True,
                               ParameterDescriptor(read_only=True))
        self.declare_parameter("allow_real_hardware", False)
        self.declare_parameter("experiment_stage", 0, ParameterDescriptor(read_only=True))
        self.declare_parameter("require_sim_clock", False,
                               ParameterDescriptor(read_only=True))
        self._require_sim_clock = self.get_parameter("require_sim_clock").value
        validate_sim_clock_profile(
            required=self._require_sim_clock,
            simulation_only=self.get_parameter("simulation_only").value,
            allow_real_hardware=self.get_parameter("allow_real_hardware").value,
            use_sim_time=self.get_parameter("use_sim_time").value,
        )
        self.declare_parameter("diagnostic_relaxed_timing", False,
                               ParameterDescriptor(read_only=True))
        self._diagnostic_relaxed_timing = self.get_parameter("diagnostic_relaxed_timing").value
        validate_relaxed_timing_profile(
            enabled=self._diagnostic_relaxed_timing, required=self._require_sim_clock,
            simulation_only=self.get_parameter("simulation_only").value,
            allow_real_hardware=self.get_parameter("allow_real_hardware").value,
            use_sim_time=self.get_parameter("use_sim_time").value,
        )
        self._sim_clock_lease = (SimClockLease(
            diagnostic_relaxed_timing=self._diagnostic_relaxed_timing)
            if self._require_sim_clock else None)
        if self._diagnostic_relaxed_timing:
            self.get_logger().warning(
                "DIAGNOSTIC_RELAXED_TIMING: SIM ONLY; clock graph/advance AGE "
                "expiry is warning-only; invalid/epoch/sensor/approval/RC guards remain active")
        if self._require_sim_clock:
            self.add_on_set_parameters_callback(self._validate_sim_clock_parameters)
        self.declare_parameter("safety_stamp_diagnostics", False,
                               ParameterDescriptor(read_only=True))
        diagnostics_enabled = self.get_parameter("safety_stamp_diagnostics").value
        validate_diagnostic_profile(
            enabled=diagnostics_enabled, require_sim_clock=self._require_sim_clock,
            simulation_only=self.get_parameter("simulation_only").value,
            allow_real_hardware=self.get_parameter("allow_real_hardware").value,
            use_sim_time=self.get_parameter("use_sim_time").value,
        )
        self._safety_stamp_diagnostics = SafetyStampDiagnostics() if diagnostics_enabled else None
        self._safety_stamp_diagnostic_publisher = None
        if diagnostics_enabled:
            self._safety_stamp_diagnostic_publisher = self.create_publisher(
                String, SAFETY_STAMP_DIAGNOSTIC_TOPIC, px4_qos())
            self._safety_stamp_diagnostic_timer = self.create_timer(
                0.1, self._publish_safety_stamp_diagnostics,
                clock=RosClock(clock_type=ClockType.STEADY_TIME))
        self.declare_parameter("offboard_warmup_s", 1.0)
        self.declare_parameter("manual_command_timeout_s", 0.5)
        self.declare_parameter("px4_status_timeout_s", 1.0)
        self.declare_parameter("require_jetson_safety", False)
        self.declare_parameter(
            "forward_test_camera_required", True,
            ParameterDescriptor(read_only=True),
        )
        self.declare_parameter("jetson_safety_timeout_s", 0.5)
        self.declare_parameter("jetson_min_confidence", 0.5)
        self.declare_parameter(
            "expected_jetson_safety_source", "jetson-realsense-d435i"
        )
        self.declare_parameter("max_jetson_velocity_mps", 1.0)
        self.declare_parameter("max_route_velocity_mps", 1.5)
        self.declare_parameter("avoidance_clear_confirmations", 3)
        # Project each trusted front-depth hit onto the active route and retain
        # the farthest observed obstacle position. Descent is allowed only
        # after this clearance beyond that position. If no position could be
        # tracked, the same value is used beyond the stable-CLEAR anchor.
        self.declare_parameter("avoidance_obstacle_clearance_m", 5.0)
        self.declare_parameter("avoidance_rejoin_lookahead_m", 0.0)
        self.declare_parameter("avoidance_rejoin_tolerance_m", 0.5)
        self.declare_parameter("require_descent_corridor_clear", True)
        self.declare_parameter("avoidance_return_speed_mps", 0.5)
        self.declare_parameter("avoidance_altitude_tolerance_m", 0.25)
        self.declare_parameter("avoidance_step_climb_enabled", True)
        self.declare_parameter("avoidance_climb_step_m", 1.0)
        self.declare_parameter("avoidance_step_tolerance_m", 0.08)
        self.declare_parameter("avoidance_step_deceleration_mps2", 0.6)
        self.declare_parameter("avoidance_step_reaction_time_s", 0.15)
        self.declare_parameter("avoidance_step_settle_speed_mps", 0.1)
        self.declare_parameter("avoidance_step_settle_s", 0.2)
        self.declare_parameter("require_roof_geometry", True)
        self.declare_parameter("minimum_roof_gap_m", 1.0)
        # Enabled explicitly by the integrated fixed-camera mission profile.
        self.declare_parameter("route_heading_gate_enabled", False)
        self.declare_parameter("terminal_requires_owned_offboard", False)

        self._gate = SafetyGate(
            simulation_only=bool(self.get_parameter("simulation_only").value),
            enable_px4_commands=bool(
                self.get_parameter("enable_px4_commands").value
            ) and bool(self.get_parameter("enable_transport_commands").value),
            allow_real_hardware=bool(
                self.get_parameter("allow_real_hardware").value
            ),
        )
        self._route_heading_gate_enabled = bool(self.get_parameter("route_heading_gate_enabled").value)
        self._terminal_requires_owned_offboard = bool(self.get_parameter("terminal_requires_owned_offboard").value)
        self._route_heading_gate = RouteHeadingGate()
        self._last_heading_at = float("-inf")
        self._local_velocity_ned = [math.nan] * 3
        self._navigation_reset_counters = None
        self._route_descent_sequence = -1
        self._route_descent_clear_count = 0
        self._warmup_s = float(self.get_parameter("offboard_warmup_s").value)
        self._manual_timeout_s = float(
            self.get_parameter("manual_command_timeout_s").value
        )
        self._status_timeout_s = float(
            self.get_parameter("px4_status_timeout_s").value
        )
        self._require_jetson_safety = bool(
            self.get_parameter("require_jetson_safety").value
        )
        if (self._gate.command_output_enabled and not self._gate.simulation_only
                and not self._require_jetson_safety):
            raise ValueError(
                "physical PX4 command output requires require_jetson_safety=true"
            )
        self._forward_test_camera_required = bool(
            self.get_parameter("forward_test_camera_required").value
        )
        self._forward_test_bypass_anchor = None
        self._jetson_safety_timeout_s = float(
            self.get_parameter("jetson_safety_timeout_s").value
        )
        self._jetson_min_confidence = float(
            self.get_parameter("jetson_min_confidence").value
        )
        self._expected_jetson_safety_source = str(
            self.get_parameter("expected_jetson_safety_source").value
        ).strip()
        self._max_jetson_velocity_mps = float(
            self.get_parameter("max_jetson_velocity_mps").value
        )
        self._max_route_velocity_mps = float(
            self.get_parameter("max_route_velocity_mps").value
        )
        self._avoidance_clear_confirmations = int(
            self.get_parameter("avoidance_clear_confirmations").value
        )
        self._avoidance_obstacle_clearance_m = float(
            self.get_parameter("avoidance_obstacle_clearance_m").value
        )
        self._avoidance_rejoin_lookahead_m = float(
            self.get_parameter("avoidance_rejoin_lookahead_m").value
        )
        self._avoidance_rejoin_tolerance_m = float(
            self.get_parameter("avoidance_rejoin_tolerance_m").value
        )
        self._require_descent_corridor_clear = bool(
            self.get_parameter("require_descent_corridor_clear").value
        )
        self._avoidance_return_speed_mps = float(
            self.get_parameter("avoidance_return_speed_mps").value
        )
        self._avoidance_altitude_tolerance_m = float(
            self.get_parameter("avoidance_altitude_tolerance_m").value
        )
        self._avoidance_step_climb_enabled = bool(
            self.get_parameter("avoidance_step_climb_enabled").value
        )
        self._require_roof_geometry = bool(
            self.get_parameter("require_roof_geometry").value
        )
        self._experiment_policy = None
        if self.get_parameter("experiment_stage").value != 0:
            from jolgwa_uav.experiment_stage_policy import validate_controller_profile
            self._experiment_policy = validate_controller_profile(
                self.get_parameter("experiment_stage").value,
                diagnostic_relaxed_timing=self._diagnostic_relaxed_timing,
                require_jetson_safety=self._require_jetson_safety,
                route_heading_gate_enabled=self._route_heading_gate_enabled,
                require_roof_geometry=self._require_roof_geometry,
                require_descent_corridor_clear=self._require_descent_corridor_clear,
                simulation_only=self.get_parameter("simulation_only").value,
                allow_real_hardware=self.get_parameter("allow_real_hardware").value)
        for name in (
            "avoidance_climb_step_m", "avoidance_step_tolerance_m",
            "avoidance_step_deceleration_mps2", "avoidance_step_reaction_time_s",
            "avoidance_step_settle_speed_mps", "avoidance_step_settle_s",
            "minimum_roof_gap_m",
        ):
            value = float(self.get_parameter(name).value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(name + " must be finite and positive")
            setattr(self, "_" + name, value)
        if self._avoidance_climb_step_m > 1.0:
            raise ValueError("avoidance_climb_step_m cannot exceed 1 m")
        if self._avoidance_step_tolerance_m >= self._avoidance_climb_step_m:
            raise ValueError("step tolerance must be smaller than climb step")
        if self._minimum_roof_gap_m < 1.0:
            raise ValueError("minimum_roof_gap_m cannot be below 1 m")
        if self._avoidance_clear_confirmations < 1:
            raise ValueError("avoidance_clear_confirmations must be positive")
        if not math.isfinite(self._avoidance_obstacle_clearance_m) or (
            self._avoidance_obstacle_clearance_m <= 0.0
        ):
            raise ValueError(
                "avoidance_obstacle_clearance_m must be positive"
            )
        if not math.isfinite(self._avoidance_rejoin_lookahead_m) or (
            self._avoidance_rejoin_lookahead_m < 0.0
        ):
            raise ValueError(
                "avoidance_rejoin_lookahead_m cannot be negative"
            )
        if not math.isfinite(self._avoidance_rejoin_tolerance_m) or (
            self._avoidance_rejoin_tolerance_m <= 0.0
        ):
            raise ValueError("avoidance_rejoin_tolerance_m must be positive")
        if not math.isfinite(self._avoidance_return_speed_mps) or (
            self._avoidance_return_speed_mps <= 0.0
        ):
            raise ValueError("avoidance_return_speed_mps must be positive")
        if not math.isfinite(self._avoidance_altitude_tolerance_m) or (
            self._avoidance_altitude_tolerance_m <= 0.0
        ):
            raise ValueError("avoidance_altitude_tolerance_m must be positive")
        self._active_command: OffboardCommand | None = None
        self._last_sequence: dict[str, int] = {}
        self._warmup_started_at: float | None = None
        self._mode_arm_sent = False
        self._last_mode_arm_attempt_at = float("-inf")
        self._terminal_command_sent = False
        self._terminal_state = VehicleControlState.TERMINAL_NONE
        self._terminal_command_id = None
        self._terminal_attempts = 0
        self._last_terminal_attempt_at = float("-inf")
        self._terminal_brake_started_at: float | None = None
        self._terminal_brake_transmitted = False
        self._terminal_landed_before_ack_at: float | None = None
        self._terminal_started_at: float | None = None
        self._low_speed_fault_since = None
        self._terminal_handoff_valid = False
        self._last_manual_velocity: TwistStamped | None = None
        self._last_manual_velocity_at = 0.0
        self._jetson_decision: SafetyDecision | None = None
        self._jetson_decision_received_at = 0.0
        self._jetson_observation_at = float("-inf")
        self._last_jetson_sequence = -1
        self._active_authority = ControlAuthority.NONE
        self._active_jetson_state = ""
        self._jetson_safety_fresh = False
        self._avoidance_altitude_z_ned: float | None = None
        self._avoidance_clear_count = 0
        self._avoidance_descent_clear_count = 0
        self._avoidance_return_active = False
        self._avoidance_start_xy_ned: tuple[float, float] | None = None
        self._avoidance_route_direction_xy: tuple[float, float] | None = None
        self._avoidance_route_length_m: float | None = None
        self._avoidance_farthest_obstacle_progress_m: float | None = None
        self._avoidance_clear_anchor_xy_ned: tuple[float, float] | None = None
        self._avoidance_rejoin_target_xy_ned: tuple[float, float] | None = None
        self._step_anchor_xy_ned: tuple[float, float] | None = None
        self._step_target_z_ned: float | None = None
        self._step_hold_z_ned: float | None = None
        self._step_waiting = False
        self._step_settled_since: float | None = None
        self._step_wait_sequence: int | None = None
        self._roof_geometry_clear = False
        self._roof_passage_verified = False
        self._manual_warmup_started_at: float | None = None
        self._manual_mode_sent = False
        self._autonomy_resume_required = False
        self._autonomy_reentry_required = False
        self._restart_required_fault = False
        self._external_mode_fenced = False
        self._external_reentry_approval_mission_id = ""
        self._retired_mission_ids: set[str] = set()
        self._battery_landing_missions = set()
        self._battery_terminal_pending = None
        # Status and local navigation have independent receipt leases. A live
        # pose stream must not preserve stale armed/preflight/mode state.
        self._last_px4_message_at = 0.0  # VehicleStatus ONLY
        self._last_landed_message_at = 0.0
        self._px4_failsafe_active = True  # Unknown until a real status sample.
        self._position_ned = [0.0, 0.0, 0.0]
        self._vertical_velocity_mps = math.nan
        self._local_velocity_valid = False
        self._last_local_position_at = 0.0
        self._yaw_rad = math.nan
        self._offboard = False
        self._landed = True
        self._last_error = ""
        self._sent_vehicle_commands: set[int] = set()
        self._flight_output_state: FlightOutputState | None = None
        self._flight_output_received_at = float("-inf")
        self._flight_output_ready_since: float | None = None
        self._low_speed_output_was_ready = False
        self._flight_output_detail = "flight output confirmation unavailable"
        self._offboard_request_started_at: float | None = None
        self._last_arm_attempt_at = float("-inf")
        self._arm_request_started_at: float | None = None
        self._vehicle_geo_state: VehicleGeoState | None = None
        self._vehicle_geo_received_at = float("-inf")
        self._altitude_reference_state: AltitudeReferenceState | None = None
        self._altitude_reference_state_received_at = float("-inf")
        self._heartbeat_stop = threading.Event()

        qos = px4_qos()
        self._contract_publisher = self.create_publisher(
            FlightControlContract, FLIGHT_CONTROL_CONTRACT, 10)
        self._command_request_publisher = self.create_publisher(
            FlightCommandRequest, FLIGHT_COMMAND_REQUEST, 10)
        self._offboard_mode_publisher = self.create_publisher(
            OffboardControlMode, PX4_OFFBOARD_CONTROL_MODE, qos
        )
        self._setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, PX4_TRAJECTORY_SETPOINT, qos
        )
        self._vehicle_command_publisher = self.create_publisher(
            VehicleCommand, PX4_VEHICLE_COMMAND, qos
        )
        self._state_publisher = self.create_publisher(
            VehicleControlState, VEHICLE_CONTROL_STATE, qos_profile_sensor_data
        )
        self._flight_envelope_publisher = self.create_publisher(
            FlightEnvelope, FLIGHT_ENVELOPE, qos_profile_sensor_data)

        self.create_subscription(
            MissionApproval, MISSION_APPROVAL, self._on_approval, 10
        )
        self.create_subscription(
            OffboardCommand, OFFBOARD_COMMAND, self._on_command, 10
        )
        self.create_subscription(
            ManualOverride, MANUAL_OVERRIDE, self._on_manual_override, 10
        )
        self.create_subscription(
            TwistStamped, MANUAL_VELOCITY, self._on_manual_velocity, 10
        )
        self.create_subscription(
            SafetyDecision,
            JETSON_SAFETY_DECISION,
            self._on_jetson_safety_decision,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            VehicleStatus, PX4_VEHICLE_STATUS, self._on_vehicle_status, qos
        )
        self.create_subscription(
            VehicleLocalPosition, PX4_LOCAL_POSITION, self._on_local_position, qos
        )
        self.create_subscription(
            VehicleLandDetected, PX4_LAND_DETECTED, self._on_land_detected, qos
        )
        if self._usb_output_contract:
            self.create_subscription(FlightCommandAck, FLIGHT_COMMAND_ACK, self._on_context_ack, 10)
        else:
            self.create_subscription(VehicleCommandAck, PX4_COMMAND_ACK, self._on_command_ack, qos)
        self.create_subscription(
            FlightOutputState, FLIGHT_OUTPUT_STATE,
            self._on_flight_output_state, qos_profile_sensor_data,
        )
        self.create_subscription(
            VehicleGeoState, VEHICLE_GEO_STATE,
            self._on_vehicle_geo_state, qos_profile_sensor_data,
        )
        self.create_subscription(
            AltitudeReferenceState, ALTITUDE_REFERENCE_STATE,
            self._on_altitude_reference_state, qos_profile_sensor_data,
        )
        if self._require_sim_clock:
            self.create_subscription(ClockMessage, "/clock", self._on_sim_clock,
                                     qos)
            # Graph work runs on a steady executor timer, never in the PX4
            # heartbeat or under its state lock. Delays expire the graph lease.
            self._sim_clock_graph_timer = self.create_timer(
                0.1, self._on_sim_clock_graph,
                clock=RosClock(clock_type=ClockType.STEADY_TIME),
            )
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="px4-offboard-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

        if not self._gate.command_output_enabled:
            self.get_logger().warning(
                "PX4 command output is disabled; set enable_px4_commands=true "
                "only for an intentional run"
            )

    def _timestamp_us(self) -> int:
        return self.get_clock().now().nanoseconds // 1000

    def _record_safety_stamp_diagnostic(self, message, *, outcome, reason,
                                        received_at, callback_ros_s=None,
                                        published_at_ros=None, transport_age_s=None):
        field_record(self, 'controller_safety_receive', sequence=int(message.sequence),
            input_age_s=float(message.observation_age_s), state=int(message.state),
            callback_delay_s=transport_age_s, outcome=outcome, reason=reason,
            valid=outcome == 'accepted')
        recorder = getattr(self, "_safety_stamp_diagnostics", None)
        if recorder is None:
            return
        # Metadata only. No wall clock, disk IO, graph query or flight-state
        # update occurs here. Caller already owns the controller state lock.
        stamp = getattr(message, "stamp", None)
        recorder.record(outcome=outcome, reason=reason,
            callback_monotonic_s=received_at, callback_ros_s=callback_ros_s,
            publication_sec=getattr(stamp, "sec", None),
            publication_nanosec=getattr(stamp, "nanosec", None),
            publication_ros_s=published_at_ros, transport_delta_s=transport_age_s,
            original_observation_age_s=getattr(message, "observation_age_s", None),
            sequence=getattr(message, "sequence", None), state=getattr(message, "state", None),
            source=getattr(message, "source", ""),
            active_command_present=getattr(self, "_active_command", None) is not None)

    def _publish_safety_stamp_diagnostics(self):
        # The optional diagnostic publisher is deliberately outside the flight
        # transaction lock. Slow/failed diagnostics never refresh sensor data.
        with self._state_lock:
            recorder = getattr(self, "_safety_stamp_diagnostics", None)
            if recorder is None:
                return
            batch = recorder.drain()
        if batch is None:
            return
        try:
            message = String()
            message.data = json.dumps(batch, allow_nan=False, separators=(",", ":"))
            self._safety_stamp_diagnostic_publisher.publish(message)
        except Exception:
            # Only the optional diagnostic serialization/publication is caught;
            # no control or safety callback is inside this exception boundary.
            with self._state_lock:
                recorder.publish_failed(len(batch["records"]))

    def _validate_sim_clock_parameters(self, parameters):
        expected = {"require_sim_clock": True, "use_sim_time": True,
                    "simulation_only": True, "allow_real_hardware": False,
                    "diagnostic_relaxed_timing": getattr(self, "_diagnostic_relaxed_timing", False)}
        for parameter in parameters:
            if parameter.name in expected and parameter.value is not expected[parameter.name]:
                return SetParametersResult(successful=False,
                    reason="active sim clock profile is startup-fixed")
        return SetParametersResult(successful=True)

    def _sim_clock_error(self, now):
        if not getattr(self, "_require_sim_clock", False):
            return None
        lease = getattr(self, "_sim_clock_lease", None)
        if lease is None:
            return "sim clock lease unavailable"
        error = lease.error(now)
        if error is not None:
            return error
        if self.get_clock().now().nanoseconds <= 0:
            return "ROS sim clock has not initialized"
        return None

    def _control_state_error(self, now):
        if not getattr(self, "_diagnostic_relaxed_timing", False):
            return self._last_error
        # Existing status text exposes this run as diagnostic, even while its
        # clock is fresh. Never label a diagnostic result a normal guard pass.
        notice = "DIAGNOSTIC_RELAXED_TIMING (SIM ONLY; clock age warnings only)"
        lease = getattr(self, "_sim_clock_lease", None)
        warnings = lease.timing_warnings(now) if lease is not None else ()
        return "; ".join((notice, *warnings, *([self._last_error] if self._last_error else [])))

    def _stop_for_sim_clock(self, reason):
        self._retire_autonomy_epoch(reason, retire_unsent_terminal=True)
        self._last_manual_velocity = None
        self._last_manual_velocity_at = 0.0
        self._manual_mode_sent = False
        self._manual_warmup_started_at = None
        self._active_authority = ControlAuthority.NONE
        self._active_jetson_state = ""
        self._jetson_safety_fresh = False
        self._last_error = reason + "; PX4 output stopped (sim clock lease)"

    def _sim_clock_allows_output(self):
        error = self._sim_clock_error(time.monotonic())
        if error is not None:
            self._stop_for_sim_clock(error)
            return False
        return True

    @serialized_state
    def _on_sim_clock(self, message):
        now = time.monotonic()
        try:
            source_ns = clock_nanoseconds(message.clock.sec, message.clock.nanosec)
        except (ValueError, AttributeError):
            source_ns = None
        error = self._sim_clock_lease.observe_clock(source_ns, now)
        if error is not None:
            # Preserve a just-expired epoch even when the arriving sample has
            # already made the current lease fresh again before the next tick.
            self._stop_for_sim_clock(error)

    def _on_sim_clock_graph(self):
        try:
            publishers = self.get_publishers_info_by_topic("/clock")
            gids = [list(p.endpoint_gid) if p.topic_type == "rosgraph_msgs/msg/Clock"
                    else [] for p in publishers]
        except Exception:
            # A graph-query failure is missing evidence, never permission.
            gids = []
        with self._state_lock:
            error = self._sim_clock_lease.observe_graph(gids, time.monotonic())
            if error is not None:
                self._stop_for_sim_clock(error)

    def _publish_px4_message(self, publisher, message):
        # Last-boundary lease check also covers expiry between two publications
        # in one timer tick. It never publishes a replacement HOLD on failure.
        if getattr(self, "_external_mode_fenced", False) or not self._sim_clock_allows_output():
            return False
        publisher.publish(message)
        return True

    def _heartbeat_loop(self) -> None:
        next_tick = time.monotonic() + self.HEARTBEAT_PERIOD_S
        while not self._heartbeat_stop.is_set():
            delay = max(0.0, next_tick - time.monotonic())
            if self._heartbeat_stop.wait(delay):
                return
            try:
                self._on_timer()
            except Exception as exc:
                self.get_logger().error("PX4 heartbeat thread stopped: %s" % exc)
                # A living ROS node with a dead control heartbeat is more
                # dangerous than a fail-stop process. PX4's configured
                # Offboard-loss action owns the airborne fallback.
                os._exit(70)
            next_tick += self.HEARTBEAT_PERIOD_S
            if next_tick < time.monotonic():
                next_tick = time.monotonic() + self.HEARTBEAT_PERIOD_S

    def destroy_node(self):
        self._heartbeat_stop.set()
        self._heartbeat_thread.join(timeout=1.0)
        return super().destroy_node()

    @serialized_state
    def _on_approval(self, message: MissionApproval) -> None:
        self._gate.approve(message.mission_id, message.approved)
        if getattr(self, "_external_mode_fenced", False):
            # Only a new approval received AFTER the external mode edge can
            # nominate a fresh handoff. Approval itself never releases output.
            if message.approved and message.mission_id not in self._retired_mission_ids:
                self._external_reentry_approval_mission_id = message.mission_id
            elif message.mission_id == self._external_reentry_approval_mission_id:
                self._external_reentry_approval_mission_id = ""
        if not message.approved and self._active_command is not None:
            if self._active_command.mission_id == message.mission_id:
                now = time.monotonic()
                completed = bool(self._prearm_terminal_confirmed(now) or (
                    self._terminal_state == VehicleControlState.TERMINAL_ACCEPTED
                    and self._landed and not self._gate.armed
                    and 0 <= now-self._last_px4_message_at < 0.5
                    and 0 <= now-self._last_landed_message_at < 0.5))
                if completed:
                    self._terminal_started_at = None
                    self._output_contract = None
                    self._pending_request_ids.clear()
                self._active_command = None
                if not completed:
                    self._last_error = "mission approval was revoked"

    @serialized_state
    def _on_command(self, message: OffboardCommand) -> None:
        if scenario_pending.defer(self, message, time.monotonic()):
            return
        scenario_error = navigation_error(self, message)
        if scenario_error:
            self._last_error = scenario_error
            return
        if (message.mission_id in getattr(self, "_battery_landing_missions", set())
                and not self._is_terminal_command(message)):
            self._last_error = "LOW_SPEED_PROFILE battery landing latched; navigation rejected"
            return
        if self._usb_output_contract and message.flight_output_handshake_version != 8:
            self._last_error = "flight output handshake version mismatch"
            return
        if not message.approved or not self._gate.is_approved(message.mission_id):
            self._last_error = "rejected offboard command without matching approval"
            self.get_logger().error(self._last_error)
            return
        if (self._terminal_state != VehicleControlState.TERMINAL_NONE
                and self._active_command is not None
                and self._active_command.mission_id == message.mission_id):
            return  # Duplicate terminal/navigation requests cannot restart termination.
        if self._is_low_speed_command(message) and message.command in (
                OffboardCommand.COMMAND_HOLD, OffboardCommand.COMMAND_TAKEOFF,
                OffboardCommand.COMMAND_GOTO):
            home_z = float(message.home_z_ned_m)
            target = tuple(float(v) for v in message.position_ned_m)
            profile = self._low_speed_profile(message)
            altitude_reference = int(getattr(
                message, "altitude_reference",
                OffboardCommand.ALTITUDE_REFERENCE_NONE))
            altitude_reference_max_error = float(getattr(
                message, "altitude_reference_max_error_m", 0.0))
            if (altitude_reference
                    != OffboardCommand.ALTITUDE_REFERENCE_FC_HOME_ALIGNED_V2
                    or abs(altitude_reference_max_error
                           - ALTITUDE_REFERENCE_MAX_ERROR_M) > 1e-6):
                self._last_error = "rejected low-speed command without aligned altitude reference"
                return
            altitude_state = self._altitude_reference_state
            if (altitude_state is None
                    or not bool(altitude_state.valid)
                    or not bool(altitude_state.stable)
                    or int(altitude_state.state) != AltitudeReferenceState.STATE_READY
                    or int(getattr(message, "altitude_reference_epoch", 0)) == 0
                    or int(altitude_state.transport_epoch)
                       != int(message.altitude_reference_epoch)
                    or int(getattr(message, "altitude_reference_sequence", 0)) == 0):
                self._last_error = (
                    "rejected low-speed command without matching stable altitude epoch")
                return
            if int(altitude_state.sequence) < int(
                    message.altitude_reference_sequence):
                self._last_error = (
                    "rejected low-speed command with future altitude sequence")
                return
            if (not math.isfinite(home_z) or not all(math.isfinite(v) for v in target)
                    or home_z-target[2] > altitude_limit(self, message, max_altitude_for_profile(profile))+1e-6
                    or float(message.acceptance_radius_m) > 0.25+1e-6):
                self._last_error = "rejected command outside selected low-speed envelope"
                return
        forward_test_marker = (
            int(getattr(message, "requested_authority", -1))
            == OffboardCommand.AUTHORITY_FORWARD_TEST_1M
        )
        if forward_test_marker and spec_for(self, message) is None:
            if not self._is_low_speed_command(message) or message.command not in (
                OffboardCommand.COMMAND_HOLD,
                OffboardCommand.COMMAND_TAKEOFF,
                OffboardCommand.COMMAND_GOTO,
                OffboardCommand.COMMAND_LAND,
                OffboardCommand.COMMAND_ABORT,
            ):
                self._last_error = "rejected invalid camera-bypass forward-test command"
                return
            if message.command == OffboardCommand.COMMAND_TAKEOFF:
                self._forward_test_bypass_anchor = (
                    message.mission_id,
                    float(message.position_ned_m[0]),
                    float(message.position_ned_m[1]),
                )
            elif message.command == OffboardCommand.COMMAND_GOTO:
                anchor = self._forward_test_bypass_anchor
                if (
                    anchor is None
                    or anchor[0] != message.mission_id
                    or not forward_test_target_within_camera_bypass(
                        (anchor[1], anchor[2]),
                        (message.position_ned_m[0], message.position_ned_m[1]),
                    )
                ):
                    self._last_error = (
                        "rejected camera-bypass target outside 2 m forward-test envelope"
                    )
                    return
        policy = getattr(self, "_experiment_policy", None)
        if (policy is not None and not policy.events and
                getattr(message, "requested_authority", OffboardCommand.AUTHORITY_LLM_ROUTE)
                == OffboardCommand.AUTHORITY_JETSON_EVENT_CAPTURE):
            self._last_error = "experiment stage rejects event command; approved route unchanged"
            return
        if (getattr(self, "_require_sim_clock", False)
                and message.mission_id in self._retired_mission_ids):
            # Even legacy terminal retries cannot reuse an epoch interrupted
            # by the opt-in clock contract. External native/RC control is not
            # mediated by this controller and remains available to the pilot.
            self._last_error = "sim clock flight epoch retired; new mission required"
            return
        if not self._sim_clock_allows_output():
            self._retired_mission_ids.add(message.mission_id)
            self._autonomy_reentry_required = True
            self._autonomy_resume_required = True
            return
        external_reentry = getattr(self, "_external_mode_fenced", False)
        if external_reentry and not self._external_handoff_allowed(message, time.monotonic()):
            self._last_error = (
                "external PX4 mode owns control: new post-departure approval and "
                "safe TAKEOFF handoff required; app manual must be released"
            )
            return
        terminal = self._is_terminal_command(message)
        if getattr(self, "_restart_required_fault", False):
            self._last_error = (
                "controller restart required after terminal fallback/failure"
            )
            return
        if (message.command == OffboardCommand.COMMAND_TAKEOFF
                and self._terminal_state != VehicleControlState.TERMINAL_NONE
                and self._landed and not self._gate.armed):
            self._terminal_state = VehicleControlState.TERMINAL_NONE
            self._terminal_command_id = None
            self._terminal_attempts = 0
            self._terminal_started_at = None
            self._terminal_brake_started_at = None
            self._terminal_brake_transmitted = False
            self._terminal_landed_before_ack_at = None
        if (self._terminal_state != VehicleControlState.TERMINAL_NONE
                and not terminal):
            self._last_error = "terminal landing is latched; navigation command rejected"
            return
        if (getattr(self, "_terminal_requires_owned_offboard", False)
                and self._is_terminal_command(self._active_command)
                and self._active_command.mission_id == message.mission_id):
            self._last_error = "terminal request already latched for this mission"
            return
        prearm_terminal = bool(terminal and self._active_command is not None
            and self._active_command.mission_id == message.mission_id
            and not self._gate.armed and self._landed
            and 0 <= time.monotonic()-self._last_px4_message_at < 0.5
            and 0 <= time.monotonic()-self._last_landed_message_at < 0.5)
        if terminal and not prearm_terminal and getattr(self, "_terminal_requires_owned_offboard", False):
            if not self._owns_terminal_handoff(time.monotonic(), message.mission_id,
                    allow_app_resume=message.command == OffboardCommand.COMMAND_LAND):
                self._last_error = "terminal handoff rejected: autonomous Offboard ownership unavailable"
                return
            if not self._terminal_home_position_valid(message):
                self._last_error = "LAND rejected: not within approved Home vicinity/elevation"
                return
        if not terminal and (
            message.mission_id in getattr(self, "_retired_mission_ids", set())
            or (getattr(self, "_autonomy_reentry_required", False)
                and message.command != OffboardCommand.COMMAND_TAKEOFF)
        ):
            self._last_error = (
                "flight epoch retired: navigation requires a new approved "
                "mission and TAKEOFF handoff"
            )
            return
        last = self._last_sequence.get(message.mission_id, -1)
        if int(message.sequence) <= last:
            return
        self._last_sequence[message.mission_id] = int(message.sequence)
        self._terminal_handoff_valid = bool(terminal and getattr(self, "_terminal_requires_owned_offboard", False))
        self._active_command = message
        self._low_speed_fault_since = None
        if message.command == OffboardCommand.COMMAND_TAKEOFF or terminal:
            self._reset_avoidance_recovery()
            if getattr(self, "_route_heading_gate_enabled", False):
                self._route_heading_gate.reset()
        if message.command == OffboardCommand.COMMAND_TAKEOFF:
            self._autonomy_reentry_required = False
            if external_reentry:
                self._external_mode_fenced = False
                self._external_reentry_approval_mission_id = ""
        if not self._gate.manual_override:
            # A fresh command sequence from mission_manager is the only signal
            # that can clear the post-manual autonomy interlock.
            self._autonomy_resume_required = False
        self._terminal_command_sent = False
        if terminal:
            self._begin_terminal(message, time.monotonic())
        if message.command == OffboardCommand.COMMAND_TAKEOFF:
            self._warmup_started_at = time.monotonic()
            self._mode_arm_sent = False
            self._last_mode_arm_attempt_at = float("-inf")
            self._last_arm_attempt_at = float("-inf")
            self._offboard_request_started_at = None
            self._arm_request_started_at = None
            self._flight_output_ready_since = None
            self._low_speed_output_was_ready = False
        self._last_error = ("LOW_SPEED_PROFILE battery health warning; braking then LAND"
            if message.mission_id in getattr(self, "_battery_landing_missions", set()) else "")
        if not terminal:
            self._activate_output_contract(message)

    def _activate_output_contract(self, command):
        self._pending_request_ids.clear()
        self._output_sequence += 1
        contract = FlightControlContract()
        contract.output_epoch = self._output_epoch
        contract.handshake_version = 8
        contract.output_sequence = self._output_sequence
        contract.command = copy.deepcopy(command)
        self._output_contract = contract
        history = getattr(self, '_output_contract_history', {})
        history[(contract.output_epoch, contract.output_sequence)] = copy.deepcopy(contract)
        while len(history) > 64:
            history.pop(next(iter(history)))
        self._output_contract_history = history
        self._flight_output_ready_since = None
        self._flight_output_state = None
        self._flight_output_received_at = float("-inf")
        if command.command == OffboardCommand.COMMAND_TAKEOFF:
            self._first_fault = ""
            self._arm_transmitted = False
        if self._usb_output_contract:
            self._contract_publisher.publish(contract)

    @serialized_state
    def _on_manual_override(self, message: ManualOverride) -> None:
        if message.active:
            scenario_pending.discard(self, "scenario pending TAKEOFF cancelled by manual override")
        if getattr(self, "_external_mode_fenced", False):
            # Release is permitted for the explicit new-mission workflow;
            # activating/toggling the app cannot retake physical PX4 control.
            if not message.active:
                self._gate.manual_override = False
            self._last_manual_velocity = None
            self._last_manual_velocity_at = 0.0
            self._manual_warmup_started_at = None
            self._manual_mode_sent = False
            self._last_error = "app manual blocked by external PX4 mode fence"
            return
        if getattr(self, "_route_heading_gate_enabled", False):
            self._route_heading_gate.reset()
            self._route_descent_clear_count = 0
        if bool(message.active):
            self._autonomy_resume_required = True
        self._gate.manual_override = bool(message.active)
        self._last_manual_velocity = None
        self._last_manual_velocity_at = 0.0
        self._manual_warmup_started_at = (
            time.monotonic() if message.active else None
        )
        self._manual_mode_sent = bool(message.active and self._offboard)
        state = "active" if message.active else "released"
        self.get_logger().warning("manual override %s by %s" % (state, message.source))

    @serialized_state
    def _on_manual_velocity(self, message: TwistStamped) -> None:
        if getattr(self, "_external_mode_fenced", False) or not self._gate.manual_override:
            return
        self._last_manual_velocity = message
        self._last_manual_velocity_at = time.monotonic()

    @serialized_state
    def _on_jetson_safety_decision(self, message: SafetyDecision) -> None:
        if (getattr(self, "_autonomy_reentry_required", False)
                or self._is_terminal_command(getattr(self, "_active_command", None))):
            # Sensor recovery is not permission to resume a retired flight or
            # to recreate climb targets while native LAND/RTL owns the vehicle.
            return
        received_at = time.monotonic()
        sequence = int(message.sequence)
        if sequence <= self._last_jetson_sequence:
            self._record_safety_stamp_diagnostic(message, outcome="ignored",
                reason="sequence_not_new", received_at=received_at)
            return
        valid_states = {
            SafetyDecision.STATE_CLEAR,
            SafetyDecision.STATE_SLOW,
            SafetyDecision.STATE_HOLD,
            SafetyDecision.STATE_EVADE,
            SafetyDecision.STATE_STALE,
        }
        values = [
            float(message.confidence),
            float(message.max_speed_mps),
            *(float(value) for value in message.velocity_ned_mps),
        ]
        observation_age_s = float(getattr(message, "observation_age_s", math.nan))
        if (
            not message.source.strip()
            or (
                getattr(self, "_expected_jetson_safety_source", "")
                and message.source.strip()
                != self._expected_jetson_safety_source
            )
            or int(message.state) not in valid_states
            or any(not math.isfinite(value) for value in values)
            or not 0.0 <= float(message.confidence) <= 1.0
            or float(message.max_speed_mps) < 0.0
            or not math.isfinite(observation_age_s)
            or observation_age_s < 0.0
        ):
            self._last_error = "rejected invalid Jetson safety decision"
            self._record_safety_stamp_diagnostic(message, outcome="rejected",
                reason="invalid_payload", received_at=received_at)
            return
        # The bridge and controller must share one ROS clock (same PC in this
        # stack; cross-PC deployment requires synchronized clocks). Account for
        # transport after publication, otherwise an old in-flight frame can
        # incorrectly look as if it was captured after the step settled.
        stamp_seconds = int(message.stamp.sec)
        stamp_nanoseconds = int(message.stamp.nanosec)
        published_at_ros = stamp_seconds + stamp_nanoseconds * 1e-9
        callback_ros_s = self._ros_time_now_s()
        transport_age_s = callback_ros_s - published_at_ros
        clock_tolerance_s = 0.005
        if (
            stamp_seconds < 0 or not 0 <= stamp_nanoseconds < 1_000_000_000
            or published_at_ros <= 0.0
            or not math.isfinite(transport_age_s)
            or transport_age_s < -clock_tolerance_s
        ):
            self._last_error = "rejected Jetson decision: invalid or future publication stamp"
            self._pause_climb_step()
            self._roof_geometry_clear = False
            self._roof_passage_verified = False
            self._update_avoidance_recovery(SafetyDecision.STATE_STALE)
            self._last_jetson_sequence = sequence
            self._jetson_decision = message
            self._jetson_decision_received_at = received_at
            self._jetson_observation_at = float("-inf")
            self._record_safety_stamp_diagnostic(message, outcome="rejected",
                reason="invalid_or_future_publication_stamp", received_at=received_at,
                callback_ros_s=callback_ros_s, published_at_ros=published_at_ros,
                transport_age_s=transport_age_s)
            return
        observation_age_s += max(0.0, transport_age_s) + clock_tolerance_s
        if observation_age_s > self._jetson_safety_timeout_s:
            self._last_error = "Jetson observation expired before controller receipt"
            self._pause_climb_step()
            self._roof_geometry_clear = False
            self._roof_passage_verified = False
            self._update_avoidance_recovery(SafetyDecision.STATE_STALE)
            self._last_jetson_sequence = sequence
            self._jetson_decision = message
            self._jetson_decision_received_at = received_at
            self._jetson_observation_at = received_at - observation_age_s
            self._record_safety_stamp_diagnostic(message, outcome="rejected",
                reason="observation_expired_before_receipt", received_at=received_at,
                callback_ros_s=callback_ros_s, published_at_ros=published_at_ros,
                transport_age_s=transport_age_s)
            return
        descent_corridor_clear: bool | None = None
        front_distance_m: float | None = None
        if message.source.strip() == "jetson-realsense-d435i":
            descent_corridor_clear = bool(message.descent_corridor_clear)
            measured_front_distance_m = float(message.front_distance_m)
            effective_trigger_distance_m = float(
                message.effective_trigger_distance_m
            )
            if (
                math.isfinite(measured_front_distance_m)
                and measured_front_distance_m > 0.0
                and math.isfinite(effective_trigger_distance_m)
                and measured_front_distance_m
                <= effective_trigger_distance_m
            ):
                front_distance_m = measured_front_distance_m
        geometry_valid = (bool(getattr(message, "geometry_valid", False))
                          and int(message.state) != SafetyDecision.STATE_STALE)
        extent_north = float(getattr(message, "obstacle_far_north_m", math.nan))
        extent_east = float(getattr(message, "obstacle_far_east_m", math.nan))
        extent_valid = (geometry_valid and bool(getattr(message, "obstacle_extent_valid", False))
                        and math.isfinite(extent_north) and math.isfinite(extent_east))
        roof_gap = float(getattr(message, "roof_vertical_gap_m", math.nan))
        self._roof_geometry_clear = (
            geometry_valid
            and bool(getattr(message, "roof_clearance_verified", False))
            and math.isfinite(roof_gap)
            and roof_gap > getattr(self, "_minimum_roof_gap_m", 1.0)
        )
        self._roof_passage_verified = (
            geometry_valid
            and extent_valid
            and bool(getattr(message, "roof_passage_verified", False))
        )
        recovery_state = int(message.state)
        if (
            recovery_state == SafetyDecision.STATE_CLEAR
            and self._avoidance_altitude_z_ned is not None
            and getattr(self, "_require_roof_geometry", False)
            and not (self._roof_geometry_clear or self._roof_passage_verified)
        ):
            recovery_state = SafetyDecision.STATE_HOLD
        if recovery_state in (SafetyDecision.STATE_HOLD, SafetyDecision.STATE_STALE):
            self._pause_climb_step()
        self._update_avoidance_recovery(
            recovery_state,
            descent_corridor_clear=descent_corridor_clear,
            front_distance_m=front_distance_m,
        )
        # A lower ROI optical-axis range can be ground, not an obstacle's far
        # edge. Only validated 3-D extent may extend the original front hit.
        if extent_valid:
            self._track_geometry_obstacle(extent_north,extent_east)
        self._last_jetson_sequence = sequence
        self._jetson_decision = message
        self._jetson_decision_received_at = received_at
        self._jetson_observation_at = received_at - observation_age_s
        self._record_safety_stamp_diagnostic(message, outcome="accepted",
            reason="callback_validation_accepted", received_at=received_at,
            callback_ros_s=callback_ros_s, published_at_ros=published_at_ros,
            transport_age_s=transport_age_s)

    def _ros_time_now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _track_geometry_obstacle(self, north_m: float, east_m: float) -> None:
        """Project calibrated depth geometry, never Gazebo ground truth."""
        if self._avoidance_route_direction_xy is None or not all(
            math.isfinite(value) for value in (north_m, east_m)
        ):
            return
        progress = self._route_progress_m((north_m, east_m))
        if progress < 0.0:
            return
        if self._avoidance_farthest_obstacle_progress_m is None:
            self._avoidance_farthest_obstacle_progress_m = progress
        else:
            self._avoidance_farthest_obstacle_progress_m = max(
                progress, self._avoidance_farthest_obstacle_progress_m
            )
        self._revalidate_avoidance_return_margin()

    def _required_avoidance_pass_progress(self) -> float | None:
        if self._avoidance_altitude_z_ned is None or self._avoidance_route_direction_xy is None:
            return None
        origins = []
        if self._avoidance_clear_anchor_xy_ned is not None:
            origins.append(self._route_progress_m(self._avoidance_clear_anchor_xy_ned))
        if self._avoidance_farthest_obstacle_progress_m is not None:
            origins.append(self._avoidance_farthest_obstacle_progress_m)
        return max(origins)+self._avoidance_obstacle_clearance_m if origins else None

    def _revalidate_avoidance_return_margin(self) -> None:
        """A growing observed bound can revoke an already active descent.

        Positive footprint/passage evidence and the extra route margin are
        independent requirements. Keep the attained altitude, not the old
        pre-descent peak, and discard the obsolete fixed rejoin XY target.
        """
        if not self._avoidance_return_active and self._avoidance_rejoin_target_xy_ned is None:
            return
        required = self._required_avoidance_pass_progress()
        current = self._route_progress_m(tuple(self._position_ned[:2]))
        route_short = (required is not None and self._avoidance_route_length_m is not None
                       and required > self._avoidance_route_length_m)
        if required is None or not math.isfinite(current) or current < required or route_short:
            self._pause_avoidance_descent(float(self._position_ned[2]))
            self._avoidance_rejoin_target_xy_ned = None
            self._last_error = ('avoidance recovery blocked: route ends before obstacle clearance'
                if route_short else 'avoidance descent paused: updated obstacle clearance margin not reached')

    def _update_avoidance_recovery(
        self,
        state: int,
        *,
        descent_corridor_clear: bool | None = None,
        front_distance_m: float | None = None,
    ) -> None:
        """Keep a climb temporary and release it only after stable CLEAR.

        Route commands retain the nominal cruise altitude.  During an obstacle
        climb we hold the highest achieved altitude across transient HOLD or
        STALE samples.  After stable CLEAR and sufficient forward progress, the
        vehicle first rejoins a lookahead point on the captured route, then
        descends to the unchanged route altitude.  A re-detection cancels that
        recovery immediately and requires a new clear sequence.
        """

        if getattr(self, "_scenario_enabled", False):
            # The bounded scenario owns these targets. Do not also seed the
            # general-route recovery state from the same perception sample.
            return
        policy = getattr(self, "_experiment_policy", None)
        if policy is not None and not policy.avoidance:
            # Stage 3 only observes; stage 5 may HOLD but may never seed a
            # climb/recovery that would later alter the approved route target.
            return
        current_z = float(self._position_ned[2])
        if state in (
            SafetyDecision.STATE_SLOW,
            SafetyDecision.STATE_HOLD,
            SafetyDecision.STATE_EVADE,
        ):
            if self._avoidance_start_xy_ned is None:
                self._capture_avoidance_route_progress()
            self._track_avoidance_obstacle(front_distance_m)

        if state == SafetyDecision.STATE_EVADE:
            self._avoidance_clear_count = 0
            self._avoidance_descent_clear_count = 0
            self._avoidance_return_active = False
            self._avoidance_clear_anchor_xy_ned = None
            self._avoidance_rejoin_target_xy_ned = None
            if math.isfinite(current_z):
                if self._avoidance_altitude_z_ned is None:
                    self._avoidance_altitude_z_ned = current_z
                else:
                    self._avoidance_altitude_z_ned = min(
                        self._avoidance_altitude_z_ned, current_z
                    )
            return

        if self._avoidance_altitude_z_ned is None:
            self._avoidance_clear_count = 0
            self._avoidance_descent_clear_count = 0
            self._avoidance_return_active = False
            if state == SafetyDecision.STATE_CLEAR:
                self._reset_avoidance_recovery()
            return

        if state == SafetyDecision.STATE_STALE:
            # Fail closed at the authority layer, but retain geometric route
            # progress.  A short transport gap must pause the vehicle rather
            # than discard the tracked obstacle position (which can otherwise
            # prevent descent forever on a loaded Jetson).  The next fresh
            # safety sample still controls whether motion may resume.
            self._avoidance_clear_count = 0
            self._avoidance_descent_clear_count = 0
            self._pause_avoidance_descent(current_z)
            return

        if state == SafetyDecision.STATE_CLEAR:
            require_descent_clear = getattr(
                self, "_require_descent_corridor_clear", False
            )
            descent_clear = (
                not require_descent_clear or descent_corridor_clear is True
            )
            if getattr(self, "_require_roof_geometry", False):
                descent_clear = descent_clear and self._roof_passage_verified
            # Lower-corridor evidence must remain current throughout the pass
            # and descent, not just at the instant the pass anchor is captured.
            self._avoidance_clear_count = min(
                self._avoidance_clear_confirmations,
                self._avoidance_clear_count + 1,
            )
            if descent_clear:
                self._avoidance_descent_clear_count = min(
                    self._avoidance_clear_confirmations,
                    self._avoidance_descent_clear_count + 1,
                )
            else:
                self._avoidance_descent_clear_count = 0
                self._pause_avoidance_descent(current_z)
                # Holding a fixed descent/rejoin XY above a long roof prevents
                # the camera from ever seeing its far edge. Resume the already
                # CLEAR forward route at the held altitude until fresh lower
                # evidence and the updated obstacle clearance permit descent.
                self._avoidance_rejoin_target_xy_ned = None
            if not self._avoidance_return_active:
                # The vehicle still has upward momentum when the depth policy
                # changes from EVADE to CLEAR.  Preserve the highest altitude
                # actually reached during that settling period; otherwise the
                # route controller immediately targets the lower, last-EVADE
                # altitude and can clip the far roof edge before tracked
                # clearance is satisfied.
                if math.isfinite(current_z):
                    self._avoidance_altitude_z_ned = min(
                        self._avoidance_altitude_z_ned, current_z
                    )
                if (
                    self._avoidance_clear_count
                    >= self._avoidance_clear_confirmations
                    and self._avoidance_descent_clear_count
                    >= self._avoidance_clear_confirmations
                    and self._avoidance_clear_anchor_xy_ned is None
                ):
                    self._avoidance_clear_anchor_xy_ned = (
                        float(self._position_ned[0]),
                        float(self._position_ned[1]),
                    )
            return

        # HOLD or SLOW during recovery is treated as a re-detection.  Safety
        # authority stops or slows the vehicle, and cruise-altitude descent
        # cannot resume until a fresh CLEAR streak is observed.
        self._avoidance_clear_count = 0
        self._avoidance_descent_clear_count = 0
        self._avoidance_return_active = False
        self._avoidance_clear_anchor_xy_ned = None
        self._avoidance_rejoin_target_xy_ned = None

    def _pause_avoidance_descent(self, current_z: float) -> None:
        if self._avoidance_return_active and math.isfinite(current_z):
            # Stop at the altitude reached so far, rather than continue to the
            # cruise target or command a jump back to the pre-descent altitude.
            self._avoidance_altitude_z_ned = current_z
        self._avoidance_return_active = False

    def _capture_avoidance_route_progress(self) -> None:
        start_x = float(self._position_ned[0])
        start_y = float(self._position_ned[1])
        self._avoidance_start_xy_ned = (start_x, start_y)
        command = self._active_command
        if command is None:
            self._avoidance_route_direction_xy = None
            self._avoidance_route_length_m = None
            return
        delta_x = float(command.position_ned_m[0]) - start_x
        delta_y = float(command.position_ned_m[1]) - start_y
        length = math.hypot(delta_x, delta_y)
        if length > 1e-6:
            self._avoidance_route_direction_xy = (
                delta_x / length,
                delta_y / length,
            )
            self._avoidance_route_length_m = length
            return
        if math.isfinite(self._yaw_rad):
            self._avoidance_route_direction_xy = (
                math.cos(self._yaw_rad),
                math.sin(self._yaw_rad),
            )
        else:
            self._avoidance_route_direction_xy = None
        self._avoidance_route_length_m = None

    def _track_avoidance_obstacle(
        self, front_distance_m: float | None
    ) -> None:
        """Project a trusted forward depth hit onto the captured route.

        D435 depth is measured along the fixed forward camera axis. Vehicle yaw
        supplies the camera direction, while the active route supplies the
        one-dimensional progress axis used by recovery. Keeping the farthest
        projection lets later roof/facade observations extend the pass target
        without allowing a nearer, noisy frame to move it backward.
        """

        if (
            front_distance_m is None
            or not math.isfinite(front_distance_m)
            or front_distance_m <= 0.0
            or self._avoidance_start_xy_ned is None
            or self._avoidance_route_direction_xy is None
        ):
            return
        camera_alignment = 1.0
        if math.isfinite(self._yaw_rad):
            camera_alignment = (
                math.cos(self._yaw_rad)
                * self._avoidance_route_direction_xy[0]
                + math.sin(self._yaw_rad)
                * self._avoidance_route_direction_xy[1]
            )
        if camera_alignment <= 0.0:
            return
        current_progress_m = self._route_progress_m(
            (float(self._position_ned[0]), float(self._position_ned[1]))
        )
        obstacle_progress_m = (
            current_progress_m + front_distance_m * camera_alignment
        )
        if self._avoidance_route_length_m is not None:
            obstacle_progress_m = min(
                self._avoidance_route_length_m,
                max(0.0, obstacle_progress_m),
            )
        if self._avoidance_farthest_obstacle_progress_m is None:
            self._avoidance_farthest_obstacle_progress_m = obstacle_progress_m
        else:
            self._avoidance_farthest_obstacle_progress_m = max(
                self._avoidance_farthest_obstacle_progress_m,
                obstacle_progress_m,
            )

    def _maybe_authorize_avoidance_return(self) -> None:
        self._revalidate_avoidance_return_margin()
        if (
            self._avoidance_altitude_z_ned is None
            or self._avoidance_return_active
            or self._avoidance_clear_count
            < self._avoidance_clear_confirmations
            or self._avoidance_descent_clear_count
            < self._avoidance_clear_confirmations
            or self._avoidance_clear_anchor_xy_ned is None
            or self._avoidance_route_direction_xy is None
            or (
                getattr(self, "_require_roof_geometry", False)
                and not self._roof_passage_verified
            )
        ):
            return
        if self._avoidance_rejoin_target_xy_ned is None:
            anchor_progress_m = self._route_progress_m(
                self._avoidance_clear_anchor_xy_ned
            )
            current_progress_m = self._route_progress_m(
                (float(self._position_ned[0]), float(self._position_ned[1]))
            )
            pass_origin_progress_m = anchor_progress_m
            if self._avoidance_farthest_obstacle_progress_m is not None:
                pass_origin_progress_m = max(
                    pass_origin_progress_m,
                    self._avoidance_farthest_obstacle_progress_m,
                )
            required_progress_m = (
                pass_origin_progress_m
                + self._avoidance_obstacle_clearance_m
            )
            if (
                self._avoidance_route_length_m is not None
                and required_progress_m > self._avoidance_route_length_m
            ):
                # A short route is not evidence of obstacle clearance. Keep
                # the elevated route target, which stops at the route endpoint,
                # and require a new safe route or an operator command.
                self._last_error = (
                    "avoidance recovery blocked: route ends before "
                    "obstacle clearance"
                )
                return
            if current_progress_m < required_progress_m:
                return
            rejoin_progress_m = max(
                0.0, current_progress_m, required_progress_m
            )
            rejoin_progress_m += self._avoidance_rejoin_lookahead_m
            if self._avoidance_route_length_m is not None:
                rejoin_progress_m = min(
                    rejoin_progress_m, self._avoidance_route_length_m
                )
            self._avoidance_rejoin_target_xy_ned = (
                self._avoidance_start_xy_ned[0]
                + rejoin_progress_m * self._avoidance_route_direction_xy[0],
                self._avoidance_start_xy_ned[1]
                + rejoin_progress_m * self._avoidance_route_direction_xy[1],
            )

        current_progress_m = self._route_progress_m(
            (float(self._position_ned[0]), float(self._position_ned[1]))
        )
        rejoin_progress_m = self._route_progress_m(
            self._avoidance_rejoin_target_xy_ned
        )
        cross_track_m = self._route_cross_track_m(
            (float(self._position_ned[0]), float(self._position_ned[1]))
        )
        if (
            current_progress_m
            >= rejoin_progress_m - self._avoidance_rejoin_tolerance_m
            and cross_track_m <= self._avoidance_rejoin_tolerance_m
        ):
            self._avoidance_return_active = True

    def _route_progress_m(self, point_xy: tuple[float, float]) -> float:
        if (
            self._avoidance_start_xy_ned is None
            or self._avoidance_route_direction_xy is None
        ):
            return 0.0
        delta_x = point_xy[0] - self._avoidance_start_xy_ned[0]
        delta_y = point_xy[1] - self._avoidance_start_xy_ned[1]
        return (
            delta_x * self._avoidance_route_direction_xy[0]
            + delta_y * self._avoidance_route_direction_xy[1]
        )

    def _route_cross_track_m(self, point_xy: tuple[float, float]) -> float:
        if (
            self._avoidance_start_xy_ned is None
            or self._avoidance_route_direction_xy is None
        ):
            return math.inf
        delta_x = point_xy[0] - self._avoidance_start_xy_ned[0]
        delta_y = point_xy[1] - self._avoidance_start_xy_ned[1]
        return abs(
            delta_x * self._avoidance_route_direction_xy[1]
            - delta_y * self._avoidance_route_direction_xy[0]
        )

    def _reset_avoidance_recovery(self) -> None:
        self._reset_climb_step()
        self._avoidance_altitude_z_ned = None
        self._avoidance_clear_count = 0
        self._avoidance_descent_clear_count = 0
        self._avoidance_return_active = False
        self._avoidance_start_xy_ned = None
        self._avoidance_route_direction_xy = None
        self._avoidance_route_length_m = None
        self._avoidance_farthest_obstacle_progress_m = None
        self._avoidance_clear_anchor_xy_ned = None
        self._avoidance_rejoin_target_xy_ned = None
        self._roof_geometry_clear = False
        self._roof_passage_verified = False

    @serialized_state
    def _on_vehicle_status(self, message: VehicleStatus) -> None:
        was_armed = self._gate.armed
        was_offboard = self._offboard
        self._last_px4_message_at = time.monotonic()
        self._gate.connected = True
        self._gate.preflight_checks_pass = bool(message.pre_flight_checks_pass)
        self._gate.armed = message.arming_state == VehicleStatus.ARMING_STATE_ARMED
        self._offboard = message.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        self._native_land = message.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LAND
        self._px4_failsafe_active = getattr(message, "failsafe", None) is not False
        if was_offboard and not self._offboard:
            # A physical/native mode departure outranks app ManualOverride.
            # This edge is absent during the initial app Offboard warmup.
            if self._native_land and self._terminal_command_sent:
                self._external_mode_fenced = True
            else:
                self._fence_external_mode_departure()
        if self._guarded_return_enabled() and self._px4_failsafe_active:
            self._retire_autonomy_epoch("PX4 failsafe active or unconfirmed")
        elif was_armed and not self._gate.armed:
            self._retire_autonomy_epoch("PX4 disarmed")

    @serialized_state
    def _on_flight_output_state(self, message: FlightOutputState) -> None:
        now = time.monotonic()
        contract = self._output_contract
        if (message.detail in battery_home.GRACE_DETAILS
                and not battery_home.terminal_reason_fresh(self, message, contract, time.monotonic_ns())):
            return  # An old/foreign observation cannot exempt preflight health.
        if (getattr(self, "_usb_output_contract", False)
                and self._is_low_speed_command(self._active_command)
                and (output_identity_fresh(message, contract, time.monotonic_ns())
                     or battery_home.terminal_reason_fresh(self, message, contract, time.monotonic_ns()))
                and message.published_monotonic_ns > getattr(self, "_last_battery_evidence_ns", -1)):
            if message.detail == BATTERY_TERMINAL_ONLY:
                self._last_battery_evidence_ns = message.published_monotonic_ns
                missions = getattr(self, "_battery_landing_missions", set())
                missions.add(message.mission_id)
                self._battery_landing_missions = missions
                if not self._first_fault:
                    self._first_fault = "LOW_SPEED_PROFILE battery health warning; braking then LAND"
                self._last_error = self._first_fault
                if self._terminal_started_at is None:
                    if getattr(self, "_battery_terminal_pending", None) is None:
                        self._battery_terminal_first_at = now
                    self._battery_terminal_pending = message
            elif (message.detail == BATTERY_TERMINAL_UNAVAILABLE
                    and not message.native_land and not getattr(self, "_native_land", False)):
                self._last_battery_evidence_ns = message.published_monotonic_ns
                missions = getattr(self, "_battery_landing_missions", set())
                missions.add(message.mission_id)
                self._battery_landing_missions = missions
                self._battery_terminal_fallback("battery terminal health/ownership unavailable")
        if (message.detail == battery_home.PAIR_TIMEOUT
                and not self._terminal_command_sent
                and battery_home.terminal_reason_fresh(self, message, contract, time.monotonic_ns())):
            if self._owns_terminal_handoff(now, message.mission_id, allow_app_resume=True):
                if self._terminal_started_at is None:
                    battery_home.prepare_terminal_context(self, message)
                    self._low_speed_land('LOW_SPEED_PROFILE contract pair timeout', now)
                elif not self._terminal_command_sent and not getattr(self, '_terminal_pair_repaired', False):
                    # Repair a lost Manager LAND intent without starting the
                    # terminal/brake clocks again or reusing its old TX proof.
                    battery_home.prepare_terminal_context(self, message)
                    self._active_command.command = OffboardCommand.COMMAND_LAND
                    self._activate_output_contract(self._active_command)
                    self._terminal_pair_repaired = True
            else:
                self._battery_terminal_fallback('contract pair timeout; ownership unavailable')
        previous = self._flight_output_state
        gap = now-self._flight_output_received_at
        if (previous is None or gap > 0.15
                or message.tx_run_id != previous.tx_run_id
                or message.tx_sequence < previous.tx_sequence
                or message.consecutive_transmissions < previous.consecutive_transmissions):
            self._flight_output_ready_since = None
        self._flight_output_state = message
        self._flight_output_received_at = now
        if (self._output_contract is not None
                and message.output_epoch == self._output_contract.output_epoch
                and message.output_sequence == self._output_contract.output_sequence):
            self._arm_transmitted = bool(self._arm_transmitted or message.arm_transmitted)
        command = self._active_command
        matching = proof_fresh(message, self._output_contract, time.monotonic_ns())
        if matching:
            self._arm_transmitted = bool(self._arm_transmitted or message.arm_transmitted)
        self._flight_output_detail = str(message.detail)
        if matching:
            if self._flight_output_ready_since is None:
                self._flight_output_ready_since = now
            if (self._terminal_brake_started_at is not None
                    and self._terminal_state == VehicleControlState.TERMINAL_PENDING
                    and not self._terminal_command_sent
                    and message.terminal_brake
                    and message.setpoint_kind == FlightEnvelope.SETPOINT_VELOCITY
                    and message.last_tx_monotonic_ns >= int(self._terminal_brake_started_at*1e9)):
                # Unlike takeoff warmup, one matching write is sufficient proof
                # that the requested zero-velocity brake reached the FC link.
                self._terminal_brake_transmitted = True
        else:
            self._flight_output_ready_since = None

        altitude_recovery.start(self, message, now)

    def _battery_terminal_fallback(self, detail):
        self._battery_terminal_pending = None
        if not self._first_fault:
            self._first_fault = "LOW_SPEED_PROFILE " + detail
        self._terminal_state = VehicleControlState.TERMINAL_FALLBACK
        self._latch_terminal_restart_required("LOW_SPEED_PROFILE " + detail)

    def _try_battery_terminal(self, now):
        message = getattr(self, "_battery_terminal_pending", None)
        if message is None:
            return False
        if self._terminal_started_at is not None:
            self._battery_terminal_pending = None
            return False
        if not (output_identity_fresh(message, self._output_contract, time.monotonic_ns())
                or battery_home.terminal_reason_fresh(self, message, self._output_contract, time.monotonic_ns())):
            self._battery_terminal_fallback("battery handoff evidence expired")
            return True
        if not self._owns_terminal_handoff(now, message.mission_id, allow_app_resume=True):
            # A fresh status callback can follow this callback on another DDS
            # topic. Block navigation while awaiting it; never extend evidence.
            if now-getattr(self, "_battery_terminal_first_at", now) >= PROOF_LEASE_NS/1e9:
                self._battery_terminal_fallback("battery handoff ownership unavailable")
            return True
        self._battery_terminal_pending = None
        battery_home.prepare_terminal_context(self, message)
        self._low_speed_land("LOW_SPEED_PROFILE battery health warning; braking then LAND", now)
        return False

    @serialized_state
    def _on_vehicle_geo_state(self, message: VehicleGeoState) -> None:
        self._vehicle_geo_state = message
        self._vehicle_geo_received_at = time.monotonic()

    @serialized_state
    def _on_altitude_reference_state(
        self, message: AltitudeReferenceState
    ) -> None:
        if not altitude_sample_advances(self._altitude_reference_state, message):
            return
        self._altitude_reference_state = message
        self._altitude_reference_state_received_at = time.monotonic()

    def _altitude_reference_status(self, command, now):
        if not self._is_low_speed_command(command):
            return True, "not_required", 0.0, 0.0, 0.0
        if (int(getattr(command, "altitude_reference", 0))
                != OffboardCommand.ALTITUDE_REFERENCE_FC_HOME_ALIGNED_V2):
            return False, "altitude_reference_mismatch", math.inf, 0.0, 0.0
        maximum = float(getattr(command, "altitude_reference_max_error_m", 0.0))
        sample = self._altitude_reference_state
        if sample is None:
            return False, "altitude_reference_stale", math.inf, 0.0, 0.0
        try:
            fc_altitude = float(getattr(
                sample, "normalized_fc_altitude_home_relative_m",
                sample.fc_altitude_home_relative_m))
            local_z = float(sample.local_z_ned_m)
            frame_altitude = float(command.home_z_ned_m)-local_z
            local_age = accumulated_sample_age_s(sample.local_age_ms, self._altitude_reference_state_received_at, now)
            global_age = accumulated_sample_age_s(sample.global_age_ms, self._altitude_reference_state_received_at, now)
            source_skew = float(sample.source_skew_ms)/1000.0
        except (TypeError, ValueError, OverflowError):
            return False, "altitude_reference_stale", math.inf, 0.0, 0.0
        if int(sample.transport_epoch) != int(getattr(
                command, "altitude_reference_epoch", 0)):
            return False, "altitude_reference_epoch_changed", math.inf, fc_altitude, frame_altitude
        if (not bool(sample.valid)
                or int(sample.state) == AltitudeReferenceState.STATE_STALE):
            return False, "altitude_reference_stale", math.inf, fc_altitude, frame_altitude
        # ARM ACK and HOME_POSITION can precede the armed HEARTBEAT. Permit
        # only the existing bounded correction observation for this TAKEOFF;
        # the bridge independently requires an actual, correlated ARM write.
        arm_started = getattr(self, '_arm_request_started_at', None)
        arm_ack = getattr(self, '_arm_home_ack', None)
        output_contract = getattr(self, '_output_contract', None)
        arm_accepted = bool(output_contract is not None and arm_ack == (
            output_contract.output_epoch, output_contract.output_sequence,
            VehicleCommandAck.VEHICLE_CMD_RESULT_ACCEPTED))
        arm_transition = bool(
            arm_started is not None and 0 <= now-arm_started <= 0.4
            and command.command == OffboardCommand.COMMAND_TAKEOFF
            and self._usb_output_contract and self._offboard
            and not getattr(self, '_external_mode_fenced', False)
            and (getattr(self, '_pending_request_ids', {}).get(400) or arm_accepted))
        if (int(sample.state)
                == AltitudeReferenceState.STATE_HOME_CORRECTION_PENDING
                and (not (self._gate.armed or arm_transition)
                     or accumulated_sample_age_s(getattr(
                         sample, "home_correction_pending_age_ms", 2**32-1),
                         self._altitude_reference_state_received_at, now) > battery_home.home_confirmation_ms(self, command)/1000.0)):
            return False, "altitude_home_correction_pending", math.inf, fc_altitude, frame_altitude
        if not bool(sample.stable):
            return False, "altitude_reference_unstable", math.inf, fc_altitude, frame_altitude
        reason, error = altitude_reference_sample_error(
            aligned_home_z=command.home_z_ned_m,
            local_z_ned=local_z,
            fc_altitude_home_relative_m=fc_altitude,
            local_age_s=local_age,
            geo_age_s=global_age,
            sample_skew_s=source_skew,
            max_error_m=maximum,
        )
        return not reason, reason or "ready", error, fc_altitude, frame_altitude

    def _flight_output_ready(self, command, now):
        if self._gate.simulation_only and not self._usb_output_contract:
            return bool(
                self._warmup_started_at is not None
                and now-self._warmup_started_at >= self._warmup_s
            )
        message = self._flight_output_state
        reset_at = getattr(self, '_scenario_yaw_reset_at', None)
        if (reset_at is not None and self._usb_output_contract
                and (message is None or message.last_tx_monotonic_ns <= int(reset_at*1e9))):
            return False
        if self._usb_output_contract and not (
                proof_fresh(message, self._output_contract, time.monotonic_ns())
                and message.tx_run_duration_ms >= 1000):
            return False
        return bool(message is not None and flight_output_confirmation_ready(
            receipt_age_s=now-self._flight_output_received_at,
            continuous_age_s=(
                now-self._flight_output_ready_since
                if self._flight_output_ready_since is not None else -1.0
            ),
            setpoint_age_ms=getattr(message, "setpoint_age_ms", 2**32-1),
            consecutive_transmissions=getattr(
                message, "consecutive_transmissions", 0),
            exact_match=(
                message.mission_id == command.mission_id
                and int(message.sequence) == int(command.sequence)
            ),
            transport_connected=message.transport_connected,
            command_graph_ready=message.command_graph_ready,
            envelope_valid=message.envelope_valid,
            setpoint_transmitted=message.setpoint_transmitted,
            warmup_s=self._warmup_s,
        ))

    def _fence_external_mode_departure(self) -> None:
        reason = "PX4 left Offboard for a native/external mode"
        self._external_mode_fenced = True
        self._external_reentry_approval_mission_id = ""
        retired = getattr(self, "_retired_mission_ids", set())
        for mission_id in (self._gate.approved_mission_id,
                           getattr(self._active_command, "mission_id", "")):
            if mission_id:
                retired.add(mission_id)
        self._retired_mission_ids = retired
        self._retire_autonomy_epoch(reason, retire_unsent_terminal=True)
        # App manual may have no autonomous command at all. Fence it anyway;
        # keep an already-dispatched native LAND/RTL free to finish on PX4.
        self._autonomy_reentry_required = True
        self._autonomy_resume_required = True
        self._last_manual_velocity = None
        self._last_manual_velocity_at = 0.0
        self._manual_warmup_started_at = None
        self._manual_mode_sent = False
        self._reset_avoidance_recovery()
        self._jetson_decision = None
        self._jetson_decision_received_at = 0.0
        self._jetson_observation_at = float("-inf")
        self._active_authority = ControlAuthority.NONE
        self._active_jetson_state = ""
        self._jetson_safety_fresh = False
        self._last_error = reason + "; all external output fenced; new approved handoff required"

    def _external_handoff_allowed(self, command, now: float) -> bool:
        return (
            command.command == OffboardCommand.COMMAND_TAKEOFF
            and bool(command.mission_id)
            and command.mission_id == self._external_reentry_approval_mission_id
            and command.mission_id not in self._retired_mission_ids
            and self._gate.may_start_autonomy(command.mission_id)
            and not getattr(self, "_px4_failsafe_active", True)
            and self._local_navigation_fresh(now)
            and 0 <= now - self._last_px4_message_at <= self._status_timeout_s
        )

    @serialized_state
    def _on_local_position(self, message: VehicleLocalPosition) -> None:
        now = time.monotonic()
        if getattr(self, '_scenario_enabled', False):
            stamp = int(message.timestamp_sample)
            if stamp <= getattr(self, '_scenario_local_stamp', -1):
                return
            self._scenario_local_stamp = stamp
        counters = tuple(getattr(message, name, None) for name in
                         ("xy_reset_counter", "z_reset_counter", "vxy_reset_counter",
                          "vz_reset_counter", "heading_reset_counter"))
        prior = self._navigation_reset_counters
        if (prior is not None and counters != prior and self._gate.armed
                and (getattr(self, "_route_heading_gate_enabled", False)
                     or self._is_low_speed_command(self._active_command))):
            from .scenario_yaw_reset import allow_heading_reset
            if allow_heading_reset(self, prior, counters, now):
                self._scenario_yaw_reset_at = now
                self._jetson_observation_at = float('-inf')
                self.get_logger().info("Accepted inferred bounded yaw generation; frozen NED route unchanged")
            else:
                self._retire_autonomy_epoch("PX4 local estimator reset invalidated route/Home geometry")
                self._route_heading_gate.reset()
        self._navigation_reset_counters = counters
        self._last_local_position_at = now
        measured_position = [float(message.x), float(message.y), float(message.z)]
        measured_velocity = [float(getattr(message, name, math.nan))
                             for name in ("vx", "vy", "vz")]
        self._local_velocity_ned = measured_velocity
        self._local_velocity_valid = (
            bool(getattr(message, "v_xy_valid", False))
            and bool(getattr(message, "v_z_valid", False))
            and all(math.isfinite(value) for value in measured_velocity)
        )
        self._vertical_velocity_mps = (
            measured_velocity[2]
            if self._local_velocity_valid
            else math.nan
        )
        self._gate.position_valid = (
            bool(message.xy_valid and message.z_valid)
            and all(math.isfinite(value) for value in measured_position)
            and self._local_velocity_valid
        )
        if self._gate.position_valid:
            self._position_ned = measured_position
            heading = float(message.heading)
            if math.isfinite(heading):
                self._yaw_rad = heading
            if (math.isfinite(heading) and bool(getattr(message, "heading_good_for_control", False))):
                self._last_heading_at = now
            else:
                self._last_heading_at = float("-inf")

    @serialized_state
    def _on_land_detected(self, message: VehicleLandDetected) -> None:
        was_landed = getattr(self, "_landed", True)
        self._landed = bool(message.landed)
        self._last_landed_message_at = time.monotonic()
        if not was_landed and self._landed:
            self._retire_autonomy_epoch("PX4 reported landing")

    @staticmethod
    def _is_terminal_command(command) -> bool:
        return command is not None and command.command in (
            OffboardCommand.COMMAND_LAND,
            OffboardCommand.COMMAND_RTL,
            OffboardCommand.COMMAND_ABORT,
        )

    @staticmethod
    def _low_speed_profile(command):
        value = int(getattr(command, "flight_profile", 0)) if command is not None else 0
        if value == OffboardCommand.FLIGHT_PROFILE_LOW_SPEED_1M_V1:
            return LOW_SPEED_1M_V1
        if value == OffboardCommand.FLIGHT_PROFILE_LOW_SPEED_2M_V1:
            return LOW_SPEED_2M_V1
        return None

    @classmethod
    def _is_low_speed_command(cls, command) -> bool:
        return cls._low_speed_profile(command) is not None

    def _forward_test_camera_bypass(self, command) -> bool:
        return (
            command is not None
            and not getattr(self, "_scenario_enabled", False)
            and not self._forward_test_camera_required
            and self._is_low_speed_command(command)
            and int(getattr(command, "requested_authority", -1))
            == OffboardCommand.AUTHORITY_FORWARD_TEST_1M
        )

    def _begin_terminal(self, command, now):
        """One identity/deadline for every normal, cancelled or error terminal."""
        if self._terminal_started_at is not None:
            return False
        self._active_command = copy.deepcopy(command)
        self._activate_output_contract(self._active_command)
        self._terminal_state = VehicleControlState.TERMINAL_PENDING
        self._terminal_command_id = (
            VehicleCommand.VEHICLE_CMD_NAV_RETURN_TO_LAUNCH
            if command.command == OffboardCommand.COMMAND_RTL
                or (command.command == OffboardCommand.COMMAND_ABORT
                    and not self._is_low_speed_command(command))
            else VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self._terminal_attempts = 0
        self._last_terminal_attempt_at = float("-inf")
        self._terminal_started_at = now
        self._terminal_pair_repaired = False
        self._terminal_ack_in_progress = False
        self._terminal_completion_confirmed = False
        self._terminal_command_sent = False
        self._terminal_brake_started_at = now
        self._terminal_brake_transmitted = not (
            self._is_low_speed_command(command) or (
                self._usb_output_contract
                and self._terminal_command_id == VehicleCommand.VEHICLE_CMD_NAV_LAND))
        self._terminal_landed_before_ack_at = None
        return True

    def _low_speed_land(self, reason: str, now: float) -> None:
        self._last_error = reason
        if not self._first_fault:
            self._first_fault = reason
        if self._terminal_started_at is not None or self._active_command is None:
            return
        command = copy.deepcopy(self._active_command)
        command.command = OffboardCommand.COMMAND_LAND
        self._low_speed_fault_since = now
        self._begin_terminal(command, now)

    def _publish_low_speed_setpoint(self, command) -> None:
        target = [float(value) for value in command.position_ned_m]
        dx = target[0]-self._position_ned[0]
        dy = target[1]-self._position_ned[1]
        horizontal_distance = math.hypot(dx, dy)
        horizontal_speed = min(MAX_HORIZONTAL_SPEED_M_S, horizontal_distance)
        if horizontal_distance > 1e-6:
            vx, vy = dx*horizontal_speed/horizontal_distance, dy*horizontal_speed/horizontal_distance
        else:
            vx = vy = 0.0
        vz = max(-MAX_VERTICAL_SPEED_M_S,
                 min(MAX_VERTICAL_SPEED_M_S, target[2]-self._position_ned[2]))
        current_altitude = float(command.home_z_ned_m)-float(self._position_ned[2])
        ceiling = altitude_limit(self, command, max_altitude_for_profile(self._low_speed_profile(command)))
        if current_altitude >= ceiling and vz < 0.0:
            vz = 0.0
        if getattr(self, "_scenario_depth_hold", False):
            vx = vy = vz = 0.0
        self._publish_velocity_setpoint([vx, vy, vz], float(command.yaw_rad))

    def _retire_autonomy_epoch(self, reason: str, *, retire_unsent_terminal=False) -> None:
        """Drop abandoned goals; fresh telemetry cannot authorize their reuse.

        Native terminal commands must be allowed to finish. A normal manual
        override still uses its existing explicit-resume contract. Following
        an actual flight/navigation epoch loss, the message contract cannot
        distinguish an automatic next waypoint from explicit resume, so only
        a newly approved mission's TAKEOFF may begin another epoch.
        """
        scenario_pending.discard(self, reason)
        command = self._active_command
        if command is None:
            return
        if self._is_terminal_command(command) and (
                self._terminal_command_sent
                or (not retire_unsent_terminal
                    and not getattr(self, "_terminal_requires_owned_offboard", False))):
            # A dispatched native landing can finish. An unsent request must
            # still be retired and must never revive when telemetry recovers.
            return
        retired = getattr(self, "_retired_mission_ids", set())
        retired.add(command.mission_id)
        self._retired_mission_ids = retired
        self._autonomy_reentry_required = True
        self._autonomy_resume_required = True
        self._active_command = None
        self._terminal_handoff_valid = False
        self._reset_avoidance_recovery()
        self._jetson_decision = None
        self._jetson_decision_received_at = 0.0
        self._jetson_observation_at = float("-inf")
        self._active_authority = ControlAuthority.NONE
        self._active_jetson_state = ""
        self._jetson_safety_fresh = False
        self._last_error = reason + "; flight epoch retired; new mission required"

    def _latch_terminal_restart_required(self, reason: str) -> None:
        """Fail-stop automatic output after an unconfirmed terminal handoff."""
        command = self._active_command
        if command is not None and command.mission_id:
            self._retired_mission_ids.add(command.mission_id)
        self._restart_required_fault = True
        self._autonomy_reentry_required = True
        self._autonomy_resume_required = True
        self._active_command = None
        self._terminal_handoff_valid = False
        self._active_authority = ControlAuthority.NONE
        self._flight_output_ready_since = None
        self._last_error = reason

    @serialized_state
    def _on_context_ack(self, message):
        contract = self._output_contract
        if (contract is None or not message.transmitted
                or message.output_epoch != contract.output_epoch
                or message.output_sequence != contract.output_sequence
                or message.mission_id != contract.command.mission_id
                or message.request_id != self._pending_request_ids.get(message.command)
                or not 0 <= time.monotonic_ns()-message.received_monotonic_ns <= 500_000_000):
            return
        if message.result != VehicleCommandAck.VEHICLE_CMD_RESULT_IN_PROGRESS:
            self._pending_request_ids.pop(message.command, None)
        if message.command == 400:
            self._arm_home_ack = (contract.output_epoch, contract.output_sequence, message.result)
        self._on_command_ack(message)

    @serialized_state
    def _on_command_ack(self, message: VehicleCommandAck) -> None:
        if self._terminal_state in (VehicleControlState.TERMINAL_FALLBACK,
                                    VehicleControlState.TERMINAL_FAILED):
            return
        if int(message.command) not in self._sent_vehicle_commands:
            return
        if int(message.command) == self._terminal_command_id:
            started = self._terminal_started_at
            if started is None or not 0 <= time.monotonic()-started < 90.0:
                return
        if message.result == VehicleCommandAck.VEHICLE_CMD_RESULT_IN_PROGRESS:
            if int(message.command) == self._terminal_command_id:
                self._terminal_ack_in_progress = True
            return
        self._sent_vehicle_commands.discard(int(message.command))
        accepted = message.result == VehicleCommandAck.VEHICLE_CMD_RESULT_ACCEPTED
        if int(message.command) == self._terminal_command_id:
            self._terminal_ack_in_progress = False
            if accepted:
                self._terminal_state = VehicleControlState.TERMINAL_ACCEPTED
            elif message.result != VehicleCommandAck.VEHICLE_CMD_RESULT_TEMPORARILY_REJECTED or self._terminal_attempts >= 3:
                self._terminal_state = VehicleControlState.TERMINAL_FALLBACK
                self._latch_terminal_restart_required("terminal command rejected: result=%d" % message.result)
        if not accepted:
            self._last_error = "PX4 rejected command {} with result {}".format(
                message.command, message.result
            )
            self.get_logger().error(self._last_error)

    def _receipt_age_usable(self, now, received_at, maximum, label):
        age = now - received_at
        if not math.isfinite(age) or received_at <= 0.0 or age < 0.0:
            return False
        if age <= maximum:
            return True
        if not getattr(self, '_diagnostic_relaxed_timing', False):
            return False
        warned = getattr(self, '_diagnostic_receipt_warnings', set())
        if label not in warned:
            self.get_logger().warning('SIM diagnostic: %s receipt age %.3fs exceeds %.3fs; original values retained' % (label,age,maximum))
            warned.add(label)
            self._diagnostic_receipt_warnings = warned
        return True

    def _local_navigation_fresh(self, now: float) -> bool:
        return (
            self._gate.position_valid
            and self._local_velocity_valid
            and all(math.isfinite(value) for value in self._position_ned)
            and math.isfinite(self._vertical_velocity_mps)
            and self._receipt_age_usable(now,self._last_local_position_at,
                min(self._status_timeout_s,self._jetson_safety_timeout_s),'local_navigation')
        )

    def _heading_fresh(self, now: float) -> bool:
        return bool(
            math.isfinite(self._yaw_rad)
            and self._receipt_age_usable(
                now, self._last_heading_at,
                min(self._status_timeout_s, self._jetson_safety_timeout_s),
                "heading",
            )
        )

    def _guarded_return_enabled(self) -> bool:
        return (getattr(self, "_route_heading_gate_enabled", False)
                or getattr(self, "_terminal_requires_owned_offboard", False))

    def _terminal_home_position_valid(self, command) -> bool:
        # Low-speed LAND is a native landing at the current position, not a
        # navigation target. A broken altitude reference must not block brake.
        return (self._is_low_speed_command(command)
                or command.command != OffboardCommand.COMMAND_LAND
                or near_home_for_landing(self._position_ned, command.position_ned_m,
                                        radius_m=1.0, maximum_height_m=5.0))

    def _owns_terminal_handoff(self, now: float, mission_id: str, *, allow_app_resume=False) -> bool:
        active = self._active_command
        return (active is not None and active.mission_id == mission_id
                and self._gate.is_approved(mission_id)
                and (not self._is_terminal_command(active)
                     or getattr(self, "_terminal_handoff_valid", False))
                and self._terminal_home_position_valid(active)
                and self._gate.armed and self._offboard and not self._gate.manual_override
                and self._gate.connected
                and (self._is_low_speed_command(active) or self._gate.preflight_checks_pass)
                and not getattr(self, "_px4_failsafe_active", True)
                and not getattr(self, "_autonomy_reentry_required", False)
                and not getattr(self, "_external_mode_fenced", False)
                and (allow_app_resume or not getattr(self, "_autonomy_resume_required", False))
                and self._local_navigation_fresh(now)
                and self._receipt_age_usable(now,self._last_px4_message_at,self._status_timeout_s,'PX4 status'))

    @serialized_state
    def _on_timer(self) -> None:
        now = time.monotonic()
        scenario_pending.drain(self, now)
        if Px4OffboardController._try_battery_terminal(self, now):
            self._publish_control_state()
            return
        if self._supervise_terminal(now):
            self._publish_control_state()
            return
        if altitude_recovery.service(self, now):
            self._publish_control_state()
            return
        if not self._sim_clock_allows_output():
            self._publish_control_state()
            return
        if self._guarded_return_enabled() and getattr(self, "_px4_failsafe_active", True):
            self._retire_autonomy_epoch("PX4 failsafe active or unconfirmed")
            # Do not wait for nav_state to catch up or stream a new Offboard
            # command over a native failsafe. Physical RC/QGC remains external.
            self._publish_control_state()
            return
        if not self._receipt_age_usable(now,self._last_px4_message_at,self._status_timeout_s,'PX4 status'):
            self._retire_autonomy_epoch("PX4 VehicleStatus lease expired")
            self._gate.connected = False
        if not self._local_navigation_fresh(now):
            self._retire_autonomy_epoch("PX4 local navigation invalid or stale")
            # A live status stream cannot renew the independent position lease.
            # Do not issue a position HOLD using an old/invalid coordinate, or
            # keep Offboard alive without a usable navigation estimate. PX4's
            # configured native Offboard-loss failsafe remains responsible.
            self._gate.position_valid = False
            self._pause_climb_step("local_navigation_unavailable")
            self._update_avoidance_recovery(SafetyDecision.STATE_STALE)
            self._active_authority = ControlAuthority.NONE
            self._active_jetson_state = ""
            self._jetson_safety_fresh = False
            self._last_error = "PX4 local navigation invalid or stale: Offboard stream stopped"

        if getattr(self, "_external_mode_fenced", False):
            self._active_authority = ControlAuthority.NONE
            self._active_jetson_state = ""
            self._jetson_safety_fresh = False
            self._publish_control_state()
            return

        if self._gate.manual_override:
            self._active_authority = ControlAuthority.HUMAN
            self._active_jetson_state = ""
            self._jetson_safety_fresh = False
            if self._gate.may_stream_manual():
                self._publish_manual_setpoint(now)
                if (
                    not self._offboard
                    and not self._manual_mode_sent
                    and self._manual_warmup_started_at is not None
                    and now - self._manual_warmup_started_at >= self._warmup_s
                ):
                    self._set_offboard_mode()
                    self._manual_mode_sent = True
            self._publish_control_state()
            return

        if self._autonomy_resume_required:
            self._active_authority = ControlAuthority.NONE
            self._active_jetson_state = ""
            self._jetson_safety_fresh = False
            if (not getattr(self, "_autonomy_reentry_required", False)
                    and self._gate.connected and self._gate.armed
                    and self._gate.position_valid and self._offboard):
                self._publish_hold_setpoint()
            self._publish_control_state()
            return

        command = self._active_command
        if command is None:
            self._active_authority = ControlAuthority.NONE
            # Publish perception readiness while idle so the mission manager
            # can reject a low-speed proposal before any arm/offboard command.
            idle_state, idle_fresh = self._jetson_safety_status(now)
            self._active_jetson_state = (
                idle_state.value if idle_state is not None else ""
            )
            self._jetson_safety_fresh = idle_fresh
            self._publish_control_state()
            return

        if not self._gate.may_stream_autonomy(command.mission_id):
            self._active_authority = ControlAuthority.NONE
            self._publish_control_state()
            return

        jetson_state, jetson_fresh = self._jetson_safety_status(now)
        camera_bypass = self._forward_test_camera_bypass(command)
        if self._is_low_speed_command(command):
            low_profile = self._low_speed_profile(command)
            low_ceiling = altitude_limit(self, command, max_altitude_for_profile(low_profile))
            if self._low_speed_fault_since is not None:
                self._low_speed_land(
                    self._last_error or "LOW_SPEED_PROFILE safety fault latched; landing",
                    now)
                self._publish_control_state()
                return
            (altitude_reference_valid, altitude_reference_detail,
             _altitude_reference_error, fc_altitude,
             _frame_altitude) = self._altitude_reference_status(command, now)
            if not altitude_reference_valid:
                reason = (
                    "LOW_SPEED_ALTITUDE_REFERENCE_DIVERGED"
                    if altitude_reference_detail in {
                        "altitude_reference_mismatch",
                        "altitude_reference_unstable",
                        "altitude_reference_epoch_changed",
                    }
                    else "LOW_SPEED_ALTITUDE_REFERENCE_STALE"
                )
                self._low_speed_land(reason, now)
                self._publish_control_state()
                return
            if not self._heading_fresh(now):
                self._low_speed_land("LOW_SPEED_PROFILE heading invalid or stale; landing", now)
                self._publish_control_state()
                return
            if fc_altitude > low_ceiling:
                self._low_speed_land(
                    "LOW_SPEED_PROFILE fc_home_altitude_limit_exceeded; landing", now)
                self._publish_control_state()
                return
            if (
                self._require_jetson_safety
                and spec_for(self, command) is None
                and not camera_bypass
                and (
                    not jetson_fresh
                    or jetson_state is not JetsonSafetyState.CLEAR
                )
            ):
                self._low_speed_land("LOW_SPEED_PROFILE obstacle/safety input not CLEAR; landing", now)
                self._publish_control_state()
                return
        policy = getattr(self, "_experiment_policy", None)
        observe_only = policy is not None and not policy.perception_controls_route
        scenario = spec_for(self, command)
        self._scenario_depth_hold = False
        if getattr(self, "_scenario_enabled", False):
            error = navigation_error(self, command, self._position_ned)
            if error:
                self._low_speed_land("LOW_SPEED_PROFILE "+(error or "scenario depth stale"), now)
                self._publish_control_state()
                return
            # Manager sequences the bounded ascent. A newly observed obstacle
            # cannot let its previously issued horizontal GOTO keep moving.
            depth_required = self._gate.armed and command.command != OffboardCommand.COMMAND_TAKEOFF
            if depth_required:
                jetson_fresh = (jetson_fresh and jetson_state is not JetsonSafetyState.STALE
                    and getattr(self._jetson_decision, "reason", "").startswith("scenario_front_detect_2m_pass_3m_v1:")
                    and self._jetson_observation_at > getattr(self, '_scenario_yaw_reset_at', float('-inf')))
            depth_wait = getattr(self, "_scenario_depth_missing_since", None)
            if depth_required and depth_wait is not None and now-depth_wait >= 5.0:
                self._low_speed_land("LOW_SPEED_PROFILE scenario depth recovery timeout", now)
                self._publish_control_state()
                return
            if depth_required and not jetson_fresh:
                if depth_wait is None:
                    self._scenario_depth_missing_since = now
                self._scenario_depth_hold = True
            else:
                self._scenario_depth_missing_since = None
            if depth_required and jetson_fresh and jetson_state is not JetsonSafetyState.CLEAR:
                ascending = (command.position_ned_m[2] < self._position_ned[2]-.05
                    and math.hypot(command.position_ned_m[0]-self._position_ned[0],
                                   command.position_ned_m[1]-self._position_ned[1]) <= .25
                    and jetson_state is JetsonSafetyState.EVADE)
                self._scenario_depth_hold = not ascending
        safety_ignored = observe_only or camera_bypass or scenario is not None
        arbitration_jetson_state = None if safety_ignored else jetson_state
        arbitration_jetson_fresh = False if safety_ignored else jetson_fresh
        authority = resolve_authority(
            human_active=False,
            route_active=True,
            jetson_event_active=(
                getattr(
                    command,
                    "requested_authority",
                    OffboardCommand.AUTHORITY_LLM_ROUTE,
                )
                == OffboardCommand.AUTHORITY_JETSON_EVENT_CAPTURE
            ),
            jetson_state=arbitration_jetson_state,
            jetson_fresh=arbitration_jetson_fresh,
            require_jetson=(
                False if safety_ignored else self._require_jetson_safety
            ),
        )
        self._active_authority = authority.authority
        # Arbitration omits a safety override for CLEAR route/capture ownership.
        # Report the actual fresh input without changing the selected authority.
        reported_jetson_state = authority.safety_state
        if reported_jetson_state is None and jetson_fresh:
            reported_jetson_state = jetson_state
        self._active_jetson_state = (
            "BYPASSED"
            if camera_bypass
            else (
                reported_jetson_state.value
                if reported_jetson_state is not None
                else ""
            )
        )
        self._jetson_safety_fresh = False if camera_bypass else jetson_fresh

        if self._hold_for_route_heading(command, arbitration_jetson_state, now):
            self._publish_control_state()
            return

        if authority.authority is ControlAuthority.JETSON_SAFETY:
            self._apply_jetson_safety(command, authority.safety_state)
            self._publish_control_state()
            return

        if command.command in (
            OffboardCommand.COMMAND_HOLD,
            OffboardCommand.COMMAND_GOTO,
            OffboardCommand.COMMAND_TAKEOFF,
        ):
            if command.command != OffboardCommand.COMMAND_TAKEOFF and not self._gate.armed:
                self._last_error = "ignored navigation command while disarmed"
                self._publish_control_state()
                return
            if self._is_low_speed_command(command):
                self._publish_low_speed_setpoint(command)
            elif command.command == OffboardCommand.COMMAND_GOTO:
                # Route output owns its mode: velocity normally, position if
                # recovery must HOLD. Emit only the mode matching this tick.
                self._publish_capped_route_setpoint(command)
            else:
                self._publish_position_mode()
                self._publish_position_setpoint(
                    command.position_ned_m, command.yaw_rad
                )
            output_ready = self._flight_output_ready(command, now)
            if output_ready and self._is_low_speed_command(command):
                self._low_speed_output_was_ready = True
            actionable_output_faults = {
                        "altitude_reference_stale",
                        "altitude_reference_mismatch",
                        "altitude_reference_unstable",
                        "altitude_reference_epoch_changed",
                        "altitude_pose_source_mismatch",
                        "altitude_pose_time_skew",
                        "fc_home_altitude_limit_exceeded",
                        "local_home_altitude_limit_exceeded",
                        "command_target_altitude_limit_exceeded",
                        "setpoint_altitude_limit_exceeded",
                    }
            if (self._is_low_speed_command(command)
                    and not output_ready
                    and self._flight_output_detail in actionable_output_faults):
                fault = (
                    "LOW_SPEED_ALTITUDE_REFERENCE_DIVERGED"
                    if self._flight_output_detail in {
                        "altitude_reference_mismatch",
                        "altitude_reference_unstable",
                        "altitude_reference_epoch_changed",
                    }
                    else "LOW_SPEED_ALTITUDE_REFERENCE_STALE"
                    if self._flight_output_detail in {
                        "altitude_reference_stale",
                        "altitude_pose_time_skew",
                    }
                    else "LOW_SPEED_PROFILE " + self._flight_output_detail
                )
                if self._low_speed_output_was_ready or self._gate.armed:
                    self._low_speed_land(fault, now)
                else:
                    # A precise pre-arm contract failure must not be hidden by
                    # the generic ten-second output-confirmation timeout.
                    self._last_error = fault
                self._publish_control_state()
                return
            if (command.command == OffboardCommand.COMMAND_TAKEOFF
                    and self._warmup_started_at is not None
                    and now-self._warmup_started_at >= 10.0
                    and not output_ready
                    and not (self._is_low_speed_command(command)
                             and self._low_speed_output_was_ready)):
                self._last_error = (
                    "LOW_SPEED_PROFILE flight output confirmation timeout"
                    if self._is_low_speed_command(command)
                    else "flight output confirmation timeout"
                )
                self._publish_control_state()
                return
            if (
                command.command == OffboardCommand.COMMAND_TAKEOFF
                and not (self._gate.armed and self._offboard)
                and self._gate.may_start_autonomy(command.mission_id)
                and output_ready
            ):
                mode_timed_out = bool(
                    self._offboard_request_started_at is not None
                    and now-self._offboard_request_started_at >= 10.0
                )
                arm_timed_out = bool(
                    self._arm_request_started_at is not None
                    and now-self._arm_request_started_at >= 10.0
                )
                handshake_action = takeoff_handshake_action(
                    output_ready=output_ready,
                    offboard=self._offboard,
                    armed=self._gate.armed,
                    mode_timed_out=mode_timed_out,
                    arm_timed_out=arm_timed_out,
                )
                if (handshake_action == "MODE"
                        and now-self._last_mode_arm_attempt_at >= 1.0):
                    self._set_offboard_mode()
                    self._last_mode_arm_attempt_at = now
                    if self._offboard_request_started_at is None:
                        self._offboard_request_started_at = now
                elif (handshake_action == "ARM"
                        and now-self._last_arm_attempt_at >= 1.0):
                    self._arm()
                    self._last_arm_attempt_at = now
                    if self._arm_request_started_at is None:
                        self._arm_request_started_at = now
                self._mode_arm_sent = True
            if (command.command == OffboardCommand.COMMAND_TAKEOFF
                    and self._offboard_request_started_at is not None
                    and not self._offboard
                    and now-self._offboard_request_started_at >= 10.0):
                self._last_error = "LOW_SPEED_PROFILE Offboard mode confirmation timeout"
            if (command.command == OffboardCommand.COMMAND_TAKEOFF
                    and self._arm_request_started_at is not None
                    and not self._gate.armed
                    and now-self._arm_request_started_at >= 10.0):
                self._last_error = (
                    "LOW_SPEED_PROFILE ARM confirmation timeout"
                    if self._is_low_speed_command(command)
                    else "ARM confirmation timeout"
                )
            self._publish_control_state()
            return

        self._publish_control_state()

    def _supervise_terminal(self, now):
        """Observe native completion even when external output is fenced."""
        if self._terminal_started_at is None:
            return False
        if getattr(self, "_terminal_completion_confirmed", False):
            return True
        if now-self._terminal_started_at >= 90.0:
            self._terminal_state = VehicleControlState.TERMINAL_FAILED
            self._latch_terminal_restart_required("terminal landing timeout; restart required")
            return True
        fresh = bool(self._last_px4_message_at > 0 and self._last_landed_message_at > 0
            and 0 <= now-self._last_px4_message_at < 0.5
            and 0 <= now-self._last_landed_message_at < 0.5)
        ground = fresh and self._landed and not self._gate.armed
        if ground and self._prearm_terminal_confirmed(now):
            self._terminal_completion_confirmed = True
            return True
        if ground and self._terminal_state == VehicleControlState.TERMINAL_ACCEPTED:
            self._terminal_completion_confirmed = True
            return True
        if ground and self._terminal_state == VehicleControlState.TERMINAL_PENDING:
            if getattr(self, "_terminal_ack_in_progress", False):
                return True
            if self._terminal_landed_before_ack_at is None:
                self._terminal_landed_before_ack_at = now
            elif now-self._terminal_landed_before_ack_at >= 1.0:
                self._terminal_state = VehicleControlState.TERMINAL_FALLBACK
                self._latch_terminal_restart_required("FALLBACK_LANDED: LAND ACK not confirmed")
            return True
        if self._external_mode_fenced:
            if not self._native_land and self._terminal_state != VehicleControlState.TERMINAL_FALLBACK:
                self._terminal_state = VehicleControlState.TERMINAL_FALLBACK
                self._latch_terminal_restart_required("external control takeover; automatic LAND not confirmed")
            return True
        if self._terminal_state in (VehicleControlState.TERMINAL_FALLBACK,
                                    VehicleControlState.TERMINAL_FAILED):
            return True
        command = self._active_command
        battery_terminal = bool(getattr(self, "_battery_landing_missions", set())
            and self._output_contract is not None
            and self._output_contract.command.mission_id in self._battery_landing_missions)
        if battery_terminal and (command is None or self._px4_failsafe_active
                or self._autonomy_reentry_required):
            self._battery_terminal_fallback("battery terminal interrupted by FC/control fault")
            return True
        if command is None:
            return True
        owned = bool(fresh and self._gate.connected and self._gate.armed
            and self._offboard and not self._gate.manual_override
            and self._gate.command_output_enabled
            and self._gate.is_approved(command.mission_id)
            and self._local_navigation_fresh(now))
        if not owned:
            self._terminal_state = VehicleControlState.TERMINAL_FALLBACK
            self._latch_terminal_restart_required("terminal navigation/ownership unavailable; native failsafe required")
            return True
        if not self._terminal_brake_transmitted:
            if now-self._terminal_brake_started_at >= 1.0:
                self._terminal_state = VehicleControlState.TERMINAL_FALLBACK
                self._latch_terminal_restart_required("terminal brake TX unconfirmed; native failsafe required")
            else:
                self._publish_velocity_setpoint([0.0, 0.0, 0.0], self._yaw_rad)
            return True
        self._publish_velocity_setpoint([0.0, 0.0, 0.0], self._yaw_rad)
        if (self._terminal_state != VehicleControlState.TERMINAL_ACCEPTED
                and not getattr(self, "_terminal_ack_in_progress", False)):
            if now-self._last_terminal_attempt_at >= 1.0:
                if self._terminal_attempts >= 3:
                    self._terminal_state = VehicleControlState.TERMINAL_FALLBACK
                    self._latch_terminal_restart_required("LAND ACK missing after three attempts")
                else:
                    self._send_vehicle_command(self._terminal_command_id)
                    self._terminal_command_sent = True
                    self._terminal_attempts += 1
                    self._last_terminal_attempt_at = now
        return True

    def _prearm_terminal_confirmed(self, now):
        c, proof = self._output_contract, self._flight_output_state
        return bool(self._terminal_started_at is not None
            and not self._arm_transmitted and not self._gate.armed and self._landed
            and self._last_px4_message_at > 0 and self._last_landed_message_at > 0
            and 0 <= now-self._last_px4_message_at < 0.5
            and 0 <= now-self._last_landed_message_at < 0.5
            and (not self._usb_output_contract or (
                c is not None and proof is not None
                and proof.output_epoch == c.output_epoch
                and proof.output_sequence == c.output_sequence
                and not proof.arm_transmitted
                and 0 <= now-self._flight_output_received_at < 0.15
                and 0 <= int(now*1e9)-proof.published_monotonic_ns < PROOF_LEASE_NS)))

    def _jetson_safety_status(
        self, now: float
    ) -> tuple[JetsonSafetyState | None, bool]:
        message = self._jetson_decision
        if message is None:
            return None, False
        observation_at = getattr(self, "_jetson_observation_at", self._jetson_decision_received_at)
        fresh = (
            0.0 <= now - self._jetson_decision_received_at <= self._jetson_safety_timeout_s
            and 0.0 <= now - observation_at <= self._jetson_safety_timeout_s
        )
        if not fresh or float(message.confidence) < self._jetson_min_confidence:
            return JetsonSafetyState.STALE, False
        mapping = {
            SafetyDecision.STATE_CLEAR: JetsonSafetyState.CLEAR,
            SafetyDecision.STATE_SLOW: JetsonSafetyState.SLOW,
            SafetyDecision.STATE_HOLD: JetsonSafetyState.HOLD,
            SafetyDecision.STATE_EVADE: JetsonSafetyState.EVADE,
            SafetyDecision.STATE_STALE: JetsonSafetyState.STALE,
        }
        state = mapping[int(message.state)]
        if state is JetsonSafetyState.SLOW and getattr(self, "_step_anchor_xy_ned", None) is not None:
            return JetsonSafetyState.HOLD, True
        if state is JetsonSafetyState.CLEAR and self._avoidance_altitude_z_ned is not None:
            if (
                getattr(self, "_require_roof_geometry", False)
                and not (self._roof_geometry_clear or self._roof_passage_verified)
            ):
                self._last_error = "post-climb route blocked: roof geometry unverified"
                return JetsonSafetyState.HOLD, True
            if getattr(self, "_step_anchor_xy_ned", None) is not None:
                # CLEAR can stop a step before its 1 m target, but forward motion
                # waits for braking, settled telemetry and a post-settle frame.
                self._pause_climb_step("clear_braking")
                if not self._step_settled_with_new_decision(now):
                    return JetsonSafetyState.HOLD, True
                self._record_climb_step_event("clear_settled")
                self._reset_climb_step()
        return state, True

    def _apply_jetson_safety(
        self,
        command: OffboardCommand,
        state: JetsonSafetyState | None,
    ) -> None:
        if not self._gate.armed:
            return
        policy = getattr(self, "_experiment_policy", None)
        if policy is not None and not policy.avoidance:
            if not policy.perception_controls_route:
                return  # Defence in depth: judgement-only input has no TX.
            self._last_error = "stage 5 obstacle/stale safety HOLD; automatic avoidance disabled"
            self._publish_hold_setpoint()
            return
        if state not in (JetsonSafetyState.CLEAR, JetsonSafetyState.SLOW):
            self._route_descent_clear_count = 0
        if state in (JetsonSafetyState.HOLD, JetsonSafetyState.STALE, None):
            self._pause_climb_step()
            if state in (JetsonSafetyState.STALE, None):
                # A receive timeout has no callback to invalidate an already
                # authorized descent. Apply the same interlock as explicit STALE.
                self._update_avoidance_recovery(SafetyDecision.STATE_STALE)
            if getattr(self, "_step_anchor_xy_ned", None) is not None:
                self._publish_climb_step_hold()
            else:
                self._publish_hold_setpoint()
            return
        if state is JetsonSafetyState.EVADE:
            event_capture = (
                getattr(
                    command,
                    "requested_authority",
                    OffboardCommand.AUTHORITY_LLM_ROUTE,
                )
                == OffboardCommand.AUTHORITY_JETSON_EVENT_CAPTURE
            )
            self._publish_jetson_velocity_setpoint(vertical_only=event_capture)
            return
        if state is JetsonSafetyState.SLOW and command.command in (
            OffboardCommand.COMMAND_HOLD,
            OffboardCommand.COMMAND_GOTO,
            OffboardCommand.COMMAND_TAKEOFF,
        ):
            self._publish_slow_route_setpoint(command)
            return
        self._publish_hold_setpoint()

    def _publish_hold_setpoint(self) -> None:
        self._publish_position_mode()
        self._publish_position_setpoint(self._position_ned, self._yaw_rad)

    def _avoidance_active(self) -> bool:
        return any(getattr(self, name, None) is not None for name in
                   ("_avoidance_altitude_z_ned", "_step_anchor_xy_ned", "_avoidance_start_xy_ned"))

    def _hold_for_route_heading(self, command, state, now) -> bool:
        if (not getattr(self, "_route_heading_gate_enabled", False)
                or command.command != OffboardCommand.COMMAND_GOTO):
            return False
        if state not in (JetsonSafetyState.CLEAR, JetsonSafetyState.SLOW):
            # This gate may return before _apply_jetson_safety. A stale/negative
            # observation must still break the sequence of descent evidence.
            self._route_descent_clear_count = 0
        gate = self._route_heading_gate
        if (getattr(self, "_step_anchor_xy_ned", None) is not None
                and state in (JetsonSafetyState.HOLD, JetsonSafetyState.STALE, None)):
            # The climb state machine already owns a fixed braking anchor.
            gate.reset()
            return False
        if not self._require_jetson_safety:
            self._last_error = "fixed-camera route requires Jetson depth safety"
            self._publish_hold_setpoint()
            return True
        if state is JetsonSafetyState.EVADE:
            # Collision prevention can climb vertically while turning is paused.
            gate.reset()
            return False
        nominal_dx = float(command.position_ned_m[0])-self._position_ned[0]
        nominal_dy = float(command.position_ned_m[1])-self._position_ned[1]
        direction = getattr(self, "_avoidance_route_direction_xy", None)
        if self._avoidance_active() and direction is not None and math.hypot(nominal_dx, nominal_dy) > .25:
            if abs(angle_error(math.atan2(nominal_dy, nominal_dx), math.atan2(direction[1], direction[0]))) > math.radians(30):
                self._last_error = "route reversal blocked until existing avoidance recovery completes"
                self._publish_hold_setpoint()
                return True
        target = self._route_target_with_avoidance(command)
        dx, dy = target[0]-self._position_ned[0], target[1]-self._position_ned[1]
        yaw = math.atan2(dy, dx) if math.hypot(dx, dy) > .25 else float(command.yaw_rad)
        if not math.isfinite(yaw):
            yaw = self._yaw_rad
        key = (command.mission_id, *[float(v) for v in command.position_ned_m])
        if gate.key != key:
            self._route_descent_sequence = self._last_jetson_sequence
            self._route_descent_clear_count = 0
        ready, reason = gate.assess(
            key=key,
            position=self._position_ned, target_yaw=yaw, heading=self._yaw_rad,
            heading_at=getattr(self, "_last_heading_at", float("-inf")),
            speed_mps=math.sqrt(sum(v*v for v in self._local_velocity_ned)),
            observation_at=self._jetson_observation_at, now=now)
        if ready:
            if self._last_error.startswith("route_heading_"):
                self._last_error = ""
            return False
        self._last_error = reason
        self._publish_position_mode()
        # Fixed XYZ brake/turn only. A fresh frame after settling is required
        # before the route may translate; CLEAR from the old heading cannot do it.
        self._publish_position_setpoint(gate.anchor, yaw if reason != "route_heading_invalid_or_stale" else self._yaw_rad)
        return True

    def _camera_route_yaw(self, command, target) -> float:
        if getattr(self, "_route_heading_gate_enabled", False):
            dx, dy = target[0]-self._position_ned[0], target[1]-self._position_ned[1]
            if math.hypot(dx, dy) > .25:
                return math.atan2(dy, dx)
        return float(command.yaw_rad)

    def _cap_unverified_route_descent(self, velocity_z, now):
        if (getattr(self, '_diagnostic_relaxed_timing', False)
                and not getattr(self, '_require_descent_corridor_clear', True)):
            return velocity_z
        if not getattr(self, "_route_heading_gate_enabled", False) or velocity_z <= 0:
            return velocity_z
        decision = self._jetson_decision
        state, fresh = self._jetson_safety_status(now)
        lower = getattr(decision, "lower_clearance_m", math.nan)
        clear = (fresh and state in (JetsonSafetyState.CLEAR, JetsonSafetyState.SLOW)
                 and decision is not None
                 and isinstance(lower, (int, float)) and math.isfinite(lower) and lower > 0
                 and bool(getattr(decision, "geometry_valid", False))
                 and bool(getattr(decision, "descent_corridor_clear", False)))
        if not clear:
            self._route_descent_clear_count = 0
        elif int(decision.sequence) != self._route_descent_sequence:
            self._route_descent_clear_count = min(3, self._route_descent_clear_count+1)
        if decision is not None:
            self._route_descent_sequence = int(decision.sequence)
        if self._route_descent_clear_count < 3:
            self._last_error = "route descent blocked: fresh lower corridor proof required"
            return 0.0
        return min(velocity_z, self._avoidance_return_speed_mps)

    def _publish_slow_route_setpoint(self, command: OffboardCommand) -> None:
        target = self._route_target_with_avoidance(command)
        delta = [target[index] - self._position_ned[index] for index in range(3)]
        distance = math.sqrt(sum(value * value for value in delta))
        speed_limit = self._jetson_speed_limit()
        if distance <= 1e-6 or speed_limit <= 0.0:
            self._publish_hold_setpoint()
            return
        speed = min(speed_limit, distance)
        velocity = [value * speed / distance for value in delta]
        velocity[2] = self._cap_unverified_route_descent(velocity[2], time.monotonic())
        self._publish_velocity_setpoint(velocity, self._camera_route_yaw(command, target))

    def _publish_capped_route_setpoint(self, command: OffboardCommand) -> None:
        self._maybe_authorize_avoidance_return()
        required = self._required_avoidance_pass_progress()
        if (required is not None and self._avoidance_route_length_m is not None
                and required > self._avoidance_route_length_m):
            # No approved forward route can satisfy the unchanged +5m margin.
            # Hold now; do not silently waive it or continue to a short endpoint.
            self._last_error = 'avoidance recovery blocked: route ends before obstacle clearance'
            self._publish_hold_setpoint()
            return
        target = self._route_target_with_avoidance(command)
        nominal_z = float(command.position_ned_m[2])
        if self._avoidance_return_active and (
            abs(float(self._position_ned[2]) - nominal_z)
            <= self._avoidance_altitude_tolerance_m
        ):
            self._reset_avoidance_recovery()
            target[2] = nominal_z
        delta_x = target[0] - self._position_ned[0]
        delta_y = target[1] - self._position_ned[1]
        horizontal_distance = math.hypot(delta_x, delta_y)
        speed_limit = max(0.1, self._max_route_velocity_mps)
        horizontal_speed = min(speed_limit, horizontal_distance)
        if horizontal_distance > 1e-6:
            velocity_x = delta_x * horizontal_speed / horizontal_distance
            velocity_y = delta_y * horizontal_speed / horizontal_distance
        else:
            velocity_x = 0.0
            velocity_y = 0.0
        # Horizontal route speed and vertical recovery speed are independent
        # limits.  Capping descent by the (deliberately slow) patrol speed can
        # leave the vehicle above the next obstacle in a repeated course.
        vertical_speed_limit = (
            self._avoidance_return_speed_mps
            if self._avoidance_return_active
            else speed_limit
        )
        velocity_z = max(
            -vertical_speed_limit,
            min(
                vertical_speed_limit,
                target[2] - self._position_ned[2],
            ),
        )
        self._publish_velocity_setpoint(
            [velocity_x, velocity_y, self._cap_unverified_route_descent(velocity_z, time.monotonic())],
            self._camera_route_yaw(command, target)
        )

    def _route_target_with_avoidance(
        self, command: OffboardCommand
    ) -> list[float]:
        target = [float(value) for value in command.position_ned_m]
        if self._avoidance_rejoin_target_xy_ned is not None:
            target[0], target[1] = self._avoidance_rejoin_target_xy_ned
        if (
            self._avoidance_altitude_z_ned is not None
            and not self._avoidance_return_active
        ):
            target[2] = min(target[2], self._avoidance_altitude_z_ned)
        return target

    def _publish_jetson_velocity_setpoint(self, *, vertical_only: bool = False) -> None:
        message = self._jetson_decision
        if message is None:
            self._publish_hold_setpoint()
            return
        velocity = [float(value) for value in message.velocity_ned_mps]
        if self._guarded_return_enabled() and (
                any(not math.isfinite(value) for value in velocity)
                or math.hypot(velocity[0], velocity[1]) > 1e-6
                or velocity[2] > 1e-6):
            # This integrated fixed-front-camera profile authorizes only the
            # verified upward step. Do not silently reinterpret a lateral or
            # downward request as a safe escape while heading is unverified.
            self._route_descent_clear_count = 0
            self._pause_climb_step("unsupported_evade_vector")
            self._update_avoidance_recovery(SafetyDecision.STATE_STALE)
            self._last_error = "integrated EVADE rejected: upward-only vector required"
            if getattr(self, "_step_anchor_xy_ned", None) is not None:
                self._publish_climb_step_hold()
            else:
                self._publish_hold_setpoint()
            return
        if (
            getattr(self, "_avoidance_step_climb_enabled", False)
            and velocity[2] < -1e-6
            and (vertical_only or math.hypot(velocity[0], velocity[1]) <= 1e-6)
        ):
            self._publish_climb_step(-velocity[2])
            return
        if vertical_only:
            velocity[0] = 0.0
            velocity[1] = 0.0
        magnitude = math.sqrt(sum(value * value for value in velocity))
        speed_limit = self._jetson_speed_limit()
        if magnitude <= 1e-6 or speed_limit <= 0.0:
            self._publish_hold_setpoint()
            return
        if magnitude > speed_limit:
            velocity = [value * speed_limit / magnitude for value in velocity]
        self._publish_velocity_setpoint(velocity, self._yaw_rad)

    def _reset_climb_step(self) -> None:
        self._step_anchor_xy_ned = None
        self._step_target_z_ned = None
        self._step_hold_z_ned = None
        self._step_waiting = False
        self._step_settled_since = None
        self._step_wait_sequence = None

    def _pause_climb_step(self, reason: str = "safety_pause") -> None:
        if getattr(self, "_step_anchor_xy_ned", None) is None or self._step_waiting:
            return
        self._step_waiting = True
        self._step_hold_z_ned = float(self._position_ned[2])
        self._step_settled_since = None
        self._step_wait_sequence = None
        self._record_climb_step_event(reason)

    def _record_climb_step_event(self, event: str) -> None:
        self.get_logger().info(
            "step_climb event=%s target_z_ned=%.3f position_ned=%s vz=%.3f sequence=%d"
            % (event, self._step_target_z_ned, self._position_ned,
               self._vertical_velocity_mps, self._last_jetson_sequence)
        )

    def _step_vertical_telemetry_fresh(self, now: float) -> bool:
        return (
            math.isfinite(self._vertical_velocity_mps)
            and 0.0 <= now - self._last_local_position_at
            <= self._jetson_safety_timeout_s
        )

    def _step_settled_with_new_decision(self, now: float) -> bool:
        if (
            not self._step_vertical_telemetry_fresh(now)
            or abs(self._vertical_velocity_mps) > self._avoidance_step_settle_speed_mps
        ):
            self._step_settled_since = None
            self._step_wait_sequence = None
            return False
        if self._step_settled_since is None:
            self._step_settled_since = now
            return False
        if now - self._step_settled_since < self._avoidance_step_settle_s:
            return False
        if self._step_wait_sequence is None:
            # A new sequence received while still moving is not sufficient.
            # Start the frame barrier only AFTER the settled interval.
            self._step_wait_sequence = self._last_jetson_sequence
            return False
        return (
            self._last_jetson_sequence > self._step_wait_sequence
            and self._jetson_observation_at
            >= self._step_settled_since + self._avoidance_step_settle_s
        )

    def _publish_climb_step_hold(self) -> bool:
        if self._step_anchor_xy_ned is None or self._step_hold_z_ned is None:
            return False
        # The final publication lease can retire this epoch and clear its
        # anchor synchronously. Do not read or resume that retired step.
        if not self._publish_position_mode():
            return False
        return self._publish_position_setpoint(
            [*self._step_anchor_xy_ned, self._step_hold_z_ned], self._yaw_rad
        )

    def _publish_climb_step(self, requested_up_speed_mps: float) -> None:
        now = time.monotonic()
        current_z = float(self._position_ned[2])
        if self._step_anchor_xy_ned is None:
            self._step_anchor_xy_ned = tuple(float(v) for v in self._position_ned[:2])
            self._step_target_z_ned = current_z - self._avoidance_climb_step_m
            self._record_climb_step_event("start")
        if not self._step_vertical_telemetry_fresh(now):
            self._pause_climb_step("vertical_telemetry_invalid")
            self._last_error = "step climb paused: vertical telemetry stale or invalid"
            self._publish_climb_step_hold()
            return
        remaining_m = current_z - self._step_target_z_ned
        if not self._step_waiting and remaining_m <= self._avoidance_step_tolerance_m:
            self._pause_climb_step("target_reached")
        if self._step_waiting:
            if not self._publish_climb_step_hold():
                return
            if self._step_settled_with_new_decision(now):
                # Resume a partially completed step after HOLD/STALE. Only a
                # completed step can receive another <=1 m target.
                if remaining_m <= self._avoidance_step_tolerance_m:
                    self._step_target_z_ned = current_z - self._avoidance_climb_step_m
                    self._record_climb_step_event("next_target")
                else:
                    self._record_climb_step_event("resume_partial_target")
                self._step_waiting = False
                self._step_hold_z_ned = None
                self._step_settled_since = None
                self._step_wait_sequence = None
            return
        # Aim inside the acceptance band; aiming at its edge asymptotically
        # could otherwise prevent a discrete controller from ever settling.
        usable_remaining_m = max(0.0, remaining_m - 0.5 * self._avoidance_step_tolerance_m)
        deceleration = self._avoidance_step_deceleration_mps2
        reaction = self._avoidance_step_reaction_time_s
        upward_speed = max(0.0, -self._vertical_velocity_mps)
        stopping_distance_m = (
            upward_speed * reaction + upward_speed * upward_speed / (2.0 * deceleration)
        )
        # Invert v*t + v^2/(2*a) <= remaining, and also soften the approach.
        braking_speed = max(
            0.0,
            math.sqrt((deceleration * reaction) ** 2 + 2.0 * deceleration * usable_remaining_m)
            - deceleration * reaction,
        )
        speed = min(
            requested_up_speed_mps, self._jetson_speed_limit(),
            braking_speed, 2.0 * usable_remaining_m,
        )
        if stopping_distance_m >= usable_remaining_m:
            speed = 0.0
        self._publish_climb_step_velocity(-speed)

    def _publish_climb_step_velocity(self, velocity_z: float) -> bool:
        # PX4 permits a position-held XY pair with a velocity-controlled Z axis.
        # A fixed XY anchor prevents forward drift accumulating at each step.
        if self._step_anchor_xy_ned is None:
            return False
        position = [*self._step_anchor_xy_ned, math.nan]
        velocity = [math.nan, math.nan, float(velocity_z)]
        # Mixed-axis avoidance is encoded by PX4 as position mode. Preserve
        # that exact NaN mask in the contract rather than guessing downstream.
        self._publish_flight_envelope(
            FlightEnvelope.SETPOINT_POSITION, position, velocity, self._yaw_rad)
        if not self._publish_position_mode():
            return False
        setpoint = TrajectorySetpoint()
        setpoint.timestamp = self._timestamp_us()
        setpoint.position = position
        setpoint.velocity = velocity
        setpoint.acceleration = [math.nan, math.nan, math.nan]
        setpoint.jerk = [math.nan, math.nan, math.nan]
        setpoint.yaw = float(self._yaw_rad)
        setpoint.yawspeed = math.nan
        return self._publish_px4_message(self._setpoint_publisher, setpoint)

    def _jetson_speed_limit(self) -> float:
        message_limit = (
            float(self._jetson_decision.max_speed_mps)
            if self._jetson_decision is not None
            else 0.0
        )
        return max(0.0, min(message_limit, self._max_jetson_velocity_mps))

    def _publish_velocity_setpoint(self, velocity, yaw: float) -> None:
        position = [math.nan, math.nan, math.nan]
        values = [float(value) for value in velocity]
        self._publish_flight_envelope(
            FlightEnvelope.SETPOINT_VELOCITY, position, values, yaw)
        mode_sent = self._publish_velocity_mode()
        setpoint = TrajectorySetpoint()
        setpoint.timestamp = self._timestamp_us()
        setpoint.position = position
        setpoint.velocity = values
        setpoint.acceleration = [math.nan, math.nan, math.nan]
        setpoint.jerk = [math.nan, math.nan, math.nan]
        setpoint.yaw = float(yaw)
        setpoint.yawspeed = math.nan
        sent = self._publish_px4_message(self._setpoint_publisher, setpoint)
        if (not self._usb_output_contract and self._terminal_started_at is not None
                and mode_sent and sent and all(value == 0.0 for value in values)):
            self._terminal_brake_transmitted = True

    def _publish_position_mode(self) -> bool:
        message = OffboardControlMode()
        message.timestamp = self._timestamp_us()
        message.position = True
        message.velocity = False
        message.acceleration = False
        message.attitude = False
        message.body_rate = False
        message.thrust_and_torque = False
        message.direct_actuator = False
        return self._publish_px4_message(self._offboard_mode_publisher, message)

    def _publish_velocity_mode(self) -> None:
        message = OffboardControlMode()
        message.timestamp = self._timestamp_us()
        message.position = False
        message.velocity = True
        message.acceleration = False
        message.attitude = False
        message.body_rate = False
        message.thrust_and_torque = False
        message.direct_actuator = False
        self._publish_px4_message(self._offboard_mode_publisher, message)

    def _publish_position_setpoint(self, position, yaw: float, *, contract=True) -> bool:
        values = [float(value) for value in position]
        velocity = [math.nan, math.nan, math.nan]
        if contract:
            self._publish_flight_envelope(
                FlightEnvelope.SETPOINT_POSITION, values, velocity, yaw)
        message = TrajectorySetpoint()
        message.timestamp = self._timestamp_us()
        message.position = values
        message.velocity = velocity
        message.acceleration = [math.nan, math.nan, math.nan]
        message.jerk = [math.nan, math.nan, math.nan]
        message.yaw = float(yaw)
        message.yawspeed = math.nan
        return self._publish_px4_message(self._setpoint_publisher, message)

    def _publish_flight_envelope(self, setpoint_kind, position, velocity, yaw) -> None:
        command = self._active_command
        if command is None:
            return
        message = FlightEnvelope()
        message.stamp = self.get_clock().now().to_msg()
        message.mission_id = command.mission_id
        message.sequence = int(command.sequence)
        if self._output_contract is not None:
            message.output_epoch = self._output_contract.output_epoch
            message.output_sequence = self._output_contract.output_sequence
        low = self._is_low_speed_command(command)
        low_profile = self._low_speed_profile(command)
        message.profile = (
            FlightEnvelope.PROFILE_LOW_SPEED_1M_V1
            if low_profile == LOW_SPEED_1M_V1 else
            FlightEnvelope.PROFILE_LOW_SPEED_2M_V1
            if low_profile == LOW_SPEED_2M_V1 else
            FlightEnvelope.PROFILE_NORMAL)
        message.max_horizontal_speed_m_s = (MAX_HORIZONTAL_SPEED_M_S if low
                                             else max(0.5, self._max_route_velocity_mps))
        message.max_vertical_speed_m_s = (MAX_VERTICAL_SPEED_M_S if low
                                           else max(0.5, self._max_route_velocity_mps))
        message.target_altitude_home_m = (
            target_altitude_for_profile(low_profile) if low else 0.0)
        message.max_altitude_home_m = (
            altitude_limit(self, command, max_altitude_for_profile(low_profile)) if low else 120.0)
        message.home_z_ned_m = float(getattr(command, "home_z_ned_m", 0.0))
        message.altitude_reference = int(getattr(
            command, "altitude_reference",
            OffboardCommand.ALTITUDE_REFERENCE_NONE))
        message.altitude_reference_max_error_m = float(getattr(
            command, "altitude_reference_max_error_m", 0.0))
        message.altitude_reference_epoch = int(getattr(
            command, "altitude_reference_epoch", 0))
        message.altitude_reference_sequence = int(getattr(
            command, "altitude_reference_sequence", 0))
        message.altitude_reference_local_time_boot_ms = int(getattr(
            command, "altitude_reference_local_time_boot_ms", 0))
        message.altitude_reference_global_time_boot_ms = int(getattr(
            command, "altitude_reference_global_time_boot_ms", 0))
        message.setpoint_kind = int(setpoint_kind)
        message.expected_position_ned_m = [float(value) for value in position]
        message.expected_velocity_ned_m_s = [float(value) for value in velocity]
        message.expected_yaw_rad = float(yaw)
        self._flight_envelope_publisher.publish(message)

    def _publish_manual_setpoint(self, now: float) -> None:
        message = self._last_manual_velocity
        if message is None or now - self._last_manual_velocity_at > self._manual_timeout_s:
            self._publish_position_mode()
            self._publish_position_setpoint(
                self._position_ned, self._yaw_rad)
            return
        position = [math.nan, math.nan, math.nan]
        velocity = [
            float(message.twist.linear.x),
            float(message.twist.linear.y),
            float(message.twist.linear.z),
        ]
        yaw = math.nan
        self._publish_flight_envelope(
            FlightEnvelope.SETPOINT_VELOCITY, position, velocity, yaw)
        self._publish_velocity_mode()
        setpoint = TrajectorySetpoint()
        setpoint.timestamp = self._timestamp_us()
        setpoint.position = position
        setpoint.velocity = velocity
        setpoint.acceleration = [math.nan, math.nan, math.nan]
        setpoint.jerk = [math.nan, math.nan, math.nan]
        setpoint.yaw = yaw
        setpoint.yawspeed = float(message.twist.angular.z)
        self._publish_px4_message(self._setpoint_publisher, setpoint)

    def _set_offboard_mode(self) -> None:
        self._send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,
            param2=6.0,
        )

    def _arm(self) -> None:
        self._send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0,
        )

    def _send_vehicle_command(self, command: int, **params) -> None:
        if self._usb_output_contract:
            contract = self._output_contract
            if contract is None or self._external_mode_fenced:
                return
            request = FlightCommandRequest()
            request.mission_id = contract.command.mission_id
            request.source_sequence = contract.command.sequence
            request.output_epoch = contract.output_epoch
            request.output_sequence = contract.output_sequence
            request.request_id = uuid.uuid4().hex
            request.issued_monotonic_ns = time.monotonic_ns()
            request.command = int(command)
            request.params = [float(params.get("param"+str(i), 0.)) for i in range(1, 8)]
            self._command_request_publisher.publish(request)
            self._pending_request_ids[int(command)] = request.request_id
            if int(command) == 400 and request.params[0] == 1.0:
                self._arm_transmitted = True
            self._sent_vehicle_commands.add(int(command))
            return
        message = VehicleCommand()
        message.timestamp = self._timestamp_us()
        message.command = int(command)
        message.param1 = float(params.get("param1", 0.0))
        message.param2 = float(params.get("param2", 0.0))
        message.param3 = float(params.get("param3", 0.0))
        message.param4 = float(params.get("param4", 0.0))
        message.param5 = float(params.get("param5", 0.0))
        message.param6 = float(params.get("param6", 0.0))
        message.param7 = float(params.get("param7", 0.0))
        message.target_system = 1
        message.target_component = 1
        message.source_system = 1
        message.source_component = 1
        message.confirmation = 0
        message.from_external = True
        if self._publish_px4_message(self._vehicle_command_publisher, message):
            self._sent_vehicle_commands.add(int(command))

    def _publish_control_state(self) -> None:
        message = VehicleControlState()
        message.stamp = self.get_clock().now().to_msg()
        message.mission_id = self._gate.approved_mission_id
        message.prearm_terminated = self._prearm_terminal_confirmed(time.monotonic())
        message.terminal_started_monotonic_ns = (int(self._terminal_started_at*1e9)
            if self._terminal_started_at is not None else 0)
        message.command_output_enabled = (self._gate.command_output_enabled
            and self._sim_clock_error(time.monotonic()) is None)
        message.approved = bool(self._gate.approved_mission_id)
        message.manual_override = self._gate.manual_override
        message.connected = self._gate.connected
        message.preflight_checks_pass = self._gate.preflight_checks_pass
        message.position_valid = self._gate.position_valid
        message.armed = self._gate.armed
        message.offboard = self._offboard
        message.landed = self._landed
        message.position_ned_m = [float(value) for value in self._position_ned]
        message.velocity_ned_m_s = [float(value) for value in self._local_velocity_ned]
        message.active_authority = self._active_authority.value
        message.jetson_safety_state = self._active_jetson_state
        message.jetson_safety_fresh = self._jetson_safety_fresh
        message.last_error = self._first_fault or self._control_state_error(time.monotonic())
        active = self._active_command
        message.active_execution_mission_id = (active.mission_id if active else
            self._output_contract.command.mission_id if self._output_contract else "")
        message.terminal_in_progress = bool(self._terminal_started_at is not None
            and self._terminal_state in (VehicleControlState.TERMINAL_PENDING,
                                        VehicleControlState.TERMINAL_ACCEPTED))
        message.emergency_land_available = bool(active and self._gate.armed
            and self._gate.connected and self._gate.command_output_enabled
            and 0 <= time.monotonic()-self._last_px4_message_at < 0.5
            and (not self._external_mode_fenced or self._native_land))
        message.emergency_land_detail = ("landing in progress" if message.terminal_in_progress
            else "ready" if message.emergency_land_available else "no fresh owned armed execution")
        message.arm_request_transmitted = bool(self._arm_transmitted or self._gate.armed)
        message.first_fault = self._first_fault
        message.terminal_detail = self._last_error if self._terminal_started_at is not None else ""
        # During a diagnostic running on the old generated overlay, do not
        # advertise capabilities which that wire schema cannot represent.
        if hasattr(message, "avoidance_active"):
            message.avoidance_active = self._avoidance_active()
            message.flight_epoch_retired = bool(getattr(self, "_autonomy_reentry_required", False))
            message.heading_rad = float(self._yaw_rad)
            message.heading_fresh = bool(math.isfinite(self._yaw_rad)
                and 0 <= time.monotonic()-getattr(self, "_last_heading_at", float("-inf")) <= .25)
            message.route_heading_gate_enabled = bool(getattr(self, "_route_heading_gate_enabled", False))
            message.terminal_owned_handoff_enabled = bool(getattr(self, "_terminal_requires_owned_offboard", False))
            message.low_speed_obstacle_guard_enabled = bool(self._require_jetson_safety)
        message.forward_test_camera_bypass_enabled = bool(
                not self._forward_test_camera_required
        )
        status_age = time.monotonic()-self._last_px4_message_at
        landed_age = time.monotonic()-self._last_landed_message_at
        message.vehicle_status_fresh = bool(
            self._last_px4_message_at > 0.0 and 0.0 <= status_age <= self._status_timeout_s
        )
        message.arming_state_valid = message.vehicle_status_fresh
        message.landed_state_valid = bool(
            self._last_landed_message_at > 0.0 and 0.0 <= landed_age <= self._status_timeout_s
        )
        message.terminal_state = int(self._terminal_state)
        command = self._active_command
        message.flight_output_ready = bool(
            command is not None and self._flight_output_ready(command, time.monotonic())
        )
        message.flight_output_detail = self._flight_output_detail
        (altitude_valid, altitude_detail, altitude_error, fc_altitude,
         frame_altitude) = self._altitude_reference_status(
             command, time.monotonic())
        message.altitude_reference_valid = bool(
            altitude_valid and self._is_low_speed_command(command))
        alignment = self._altitude_reference_state
        raw_fc_altitude = float(getattr(
            alignment, "fc_altitude_home_relative_m", math.nan
        )) if alignment is not None else math.nan
        message.fc_altitude_home_relative_m = float(
            raw_fc_altitude if math.isfinite(raw_fc_altitude) else 0.0)
        message.normalized_fc_altitude_home_relative_m = float(
            getattr(alignment, "normalized_fc_altitude_home_relative_m", math.nan))
        message.frame_altitude_valid = bool(self._is_low_speed_command(command)
                                             and math.isfinite(frame_altitude))
        message.altitude_reference_error_valid = bool(self._is_low_speed_command(command)
                                                       and math.isfinite(altitude_error))
        message.frame_altitude_home_relative_m = float(
            frame_altitude if message.frame_altitude_valid else math.nan)
        message.altitude_reference_error_m = float(
            altitude_error if message.altitude_reference_error_valid else math.nan)
        message.altitude_reference_detail = altitude_detail
        message.home_correction_state = int(getattr(
            alignment, "home_correction_state", 0)) if alignment else 0
        message.home_correction_valid = bool(getattr(
            alignment, "home_correction_valid", False)) if alignment else False
        message.home_correction_revision = int(getattr(
            alignment, "home_correction_revision", 0)) if alignment else 0
        message.home_altitude_correction_m = float(getattr(
            alignment, "home_altitude_correction_m", 0.0)) if alignment else 0.0
        message.home_correction_opposition_error_m = float(getattr(
            alignment, "home_correction_opposition_error_m", 0.0)) if alignment else 0.0
        message.home_correction_detail = str(getattr(
            alignment, "home_correction_detail", "unavailable")) if alignment else "unavailable"
        message.home_phase = int(getattr(
            alignment, "home_phase", 0)) if alignment else 0
        message.execution_home_lock_valid = bool(getattr(
            alignment, "execution_home_lock_valid", False)) if alignment else False
        message.execution_home_mission_id = str(getattr(
            alignment, "execution_home_mission_id", "")) if alignment else ""
        message.execution_home_lock_revision = int(getattr(
            alignment, "execution_home_lock_revision", 0)) if alignment else 0
        message.provisional_home_revision = int(getattr(
            alignment, "provisional_home_revision", 0)) if alignment else 0
        message.provisional_px4_home_altitude_amsl_m = float(getattr(
            alignment, "provisional_px4_home_altitude_amsl_m", 0.0)) if alignment else 0.0
        message.provisional_px4_home_z_ned_m = float(getattr(
            alignment, "provisional_px4_home_z_ned_m", 0.0)) if alignment else 0.0
        message.current_px4_home_altitude_amsl_m = float(getattr(
            alignment, "current_px4_home_altitude_amsl_m", 0.0)) if alignment else 0.0
        message.current_px4_home_z_ned_m = float(getattr(
            alignment, "current_px4_home_z_ned_m", 0.0)) if alignment else 0.0
        message.home_z_correction_m = float(getattr(
            alignment, "home_z_correction_m", 0.0)) if alignment else 0.0
        message.altitude_epoch_failure_latched = bool(getattr(
            alignment, "altitude_epoch_failure_latched", False)) if alignment else False
        alignment_fresh = bool(
            alignment is not None
            and 0.0 <= time.monotonic()-self._altitude_reference_state_received_at <= 0.5
            and bool(alignment.valid))
        message.altitude_alignment_ready = bool(
            alignment_fresh and alignment.stable
            and int(alignment.state) == AltitudeReferenceState.STATE_READY)
        message.altitude_alignment_state = int(
            alignment.state if alignment is not None
            else AltitudeReferenceState.STATE_STALE)
        message.altitude_alignment_epoch = int(
            alignment.transport_epoch if alignment is not None else 0)
        message.altitude_alignment_sample_count = int(
            alignment.sample_count if alignment is not None else 0)
        message.altitude_alignment_window_ms = int(
            alignment.window_duration_ms if alignment is not None else 0)
        message.altitude_alignment_candidate_span_m = float(
            alignment.candidate_span_m
            if alignment is not None
            and math.isfinite(float(alignment.candidate_span_m)) else 0.0)
        message.altitude_alignment_source_skew_ms = int(
            alignment.source_skew_ms if alignment is not None else 2**32-1)
        message.altitude_alignment_detail = str(
            alignment.detail if alignment is not None
            else "altitude_reference_state_missing")
        self._state_publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Px4OffboardController()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

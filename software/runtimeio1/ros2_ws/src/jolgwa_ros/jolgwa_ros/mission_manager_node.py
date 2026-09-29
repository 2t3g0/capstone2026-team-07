from . import scenario_battery_home as battery_home
import json
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from ament_index_python.packages import get_package_share_directory
from jolgwa_interfaces.action import ExecuteMission
from jolgwa_interfaces.msg import (
    AltitudeReferenceState,
    LowSpeedSafetyState,
    ManualOverride,
    MissionApproval,
    MissionProposal,
    MissionStatus,
    OffboardCommand,
    RouteFrameStatus,
    VehicleControlState,
    VehicleGeoState,
)
from jolgwa_interfaces.srv import (
    ApproveMission,
    EmergencyLand,
    ReleaseEventControl,
    ResumeMission,
    RequestEventControl,
    SetManualOverride,
    PrepareForwardTest,
)
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.utilities import get_rmw_implementation_identifier
from px4_msgs.msg import HomePosition, VehicleStatus

from .coordinates import distance_ned, enu_to_ned, yaw_enu_deg_to_ned_rad
from .event_control import (
    EventControlCoordinator,
    EventLeaseState,
    route_rejoin_with_lookahead,
)
from .mission_sequence import patrol_target_count, patrol_waypoint_indices
from .event_home_return import (
    ReturnTracePolicy, VisitedReturnTrace, home_from_px4, near_home_for_landing,
    owned_navigation_error, ready_to_reverse, return_yaw, merge_collinear_return_targets,
)
from .routes import RouteCatalog
from .route_frame import (
    RouteFrameResult, RouteHome, px4_home_geodetic,
    site_enu_from_wgs84, transform_route,
)
from .low_speed import (
    accumulated_sample_age_s, altitude_sample_advances,
    ACCEPTANCE_RADIUS_M, ALTITUDE_REFERENCE_MAX_ERROR_M, FORWARD_TEST_1M,
    LOW_SPEED_1M_V1, LOW_SPEED_2M_V1,
    PREVIEW_TTL_S, ROUTE, SETTLE_ALTITUDE_ERROR_M, SETTLE_SPEED_M_S,
    ALTITUDE_ALIGNMENT_READY, SETTLE_TIME_S, altitude_reference_sample_error,
    capture_then_home_for_profile,
    fixed_limits, forward_endpoint_ned,
    is_low_speed_profile, low_speed_readiness_error, mission_kind_for_plan,
    obstacle_guard_required_for_mission, preview_valid, profile_for_plan,
    profile_for_target_altitude,
    target_altitude_for_profile, validate_low_speed_plan,
)
from .scenario_contract import make_spec, point as scenario_point, validate_spec
from .scenario_runtime import init_manager, execute_scenario, navigation_error
from .static_home_reference import StaticHomeReference, single_fastdds_prefix
from .evidence_clock import evidence_deadline_error
from .integrated_event_policy import validate_integrated_event_plan
from .topic_names import (
    APPROVE_MISSION_SERVICE,
    ALTITUDE_REFERENCE_STATE,
    EMERGENCY_LAND_SERVICE,
    EXECUTE_MISSION_ACTION,
    MANUAL_OVERRIDE,
    MANUAL_OVERRIDE_SERVICE,
    MISSION_APPROVAL,
    MISSION_PROPOSAL,
    MISSION_STATUS,
    ROUTE_FRAME_STATUS,
    OFFBOARD_COMMAND,
    RELEASE_EVENT_CONTROL_SERVICE,
    RESUME_MISSION_SERVICE,
    REQUEST_EVENT_CONTROL_SERVICE,
    VEHICLE_CONTROL_STATE,
    VEHICLE_GEO_STATE,
    LOW_SPEED_SAFETY_STATE,
    PREPARE_FORWARD_TEST_SERVICE,
    PX4_HOME_POSITION,
    PX4_VEHICLE_STATUS,
    require_px4_message_topic,
)


@dataclass(frozen=True)
class ExecutionSnapshot:
    proposal_id: str
    plan_json: str
    canonical_plan_json: str
    mission_kind: str
    flight_profile: str
    frame_result: RouteFrameResult
    points: tuple
    route_ned: tuple
    waypoint_yaws_ned: tuple
    limit_type: str
    limit_value: int
    raw_route_home: RouteHome | None = None
    aligned_home_z_ned_m: float | None = None
    capture_fc_altitude_home_relative_m: float | None = None
    altitude_reference_epoch: int = 0
    altitude_reference_sequence: int = 0
    altitude_reference_local_time_boot_ms: int = 0
    altitude_reference_global_time_boot_ms: int = 0
    preview_start_ned: tuple | None = None
    preview_heading_rad: float | None = None
    preview_end_ned: tuple | None = None
    home_generation: int = 0


class MissionManagerNode(Node):
    CACHE_LIMIT = 64
    def __init__(self) -> None:
        require_px4_message_topic(PX4_HOME_POSITION, "home_position", HomePosition)
        require_px4_message_topic(PX4_VEHICLE_STATUS, "vehicle_status", VehicleStatus)
        super().__init__("mission_manager")
        default_zones = (
            get_package_share_directory("jolgwa_ros") + "/config/zones.yaml"
        )
        self.declare_parameter("zones_file", default_zones)
        self.declare_parameter("acceptance_radius_m", 2.0)
        self.declare_parameter("takeoff_timeout_s", 60.0)
        self.declare_parameter("default_takeoff_altitude_m", 5.0,
                               ParameterDescriptor(read_only=True))
        self.declare_parameter("waypoint_timeout_s", 120.0)
        self.declare_parameter("landing_timeout_s", 90.0)
        self.declare_parameter("vehicle_ready_timeout_s", 10.0)
        self.declare_parameter("readiness_only_checkpoint", False,
                               ParameterDescriptor(read_only=True))
        self.declare_parameter("owned_sim_preparation_config", "", ParameterDescriptor(read_only=True))
        self.declare_parameter("diagnostic_relaxed_timing", False,
                               ParameterDescriptor(read_only=True))
        self.declare_parameter("simple_obstacle_demo", False,
                              ParameterDescriptor(read_only=True))
        self.declare_parameter("simple_demo_phase1_events", False,
                              ParameterDescriptor(read_only=True))
        self.declare_parameter("simple_demo_endpoint_observation_s", 10.0,
                              ParameterDescriptor(read_only=True))
        self.declare_parameter("simulation_only", True, ParameterDescriptor(read_only=True))
        self.declare_parameter("allow_real_hardware", False, ParameterDescriptor(read_only=True))
        self.declare_parameter(
            "forward_test_camera_required", True,
            ParameterDescriptor(read_only=True),
        )
        self.declare_parameter("max_first_leg_m", 50.0, ParameterDescriptor(read_only=True))
        self.declare_parameter("route_home_timeout_s", 2.5, ParameterDescriptor(read_only=True))
        self.declare_parameter("approval_execution_lease_s", 5.0, ParameterDescriptor(read_only=True))
        self.declare_parameter("experiment_stage", 0, ParameterDescriptor(read_only=True))
        self.declare_parameter("max_laps", 100)
        self.declare_parameter("max_duration_minutes", 120)
        self.declare_parameter("allow_goal_waypoints", False)
        self.declare_parameter("event_control_max_duration_s", 5.0)
        self.declare_parameter("event_hold_settle_s", 0.5)
        self.declare_parameter("event_hold_timeout_s", 3.0)
        self.declare_parameter("event_hold_drift_tolerance_m", 0.15)
        self.declare_parameter("event_rejoin_lookahead_m", 5.0)
        self.declare_parameter("event_completion_policy", "capture_then_rtl")
        # Humble graph GID association is deliberately restricted to owned SITL,
        # not offered as per-message source proof for physical automatic flight.
        self.declare_parameter("event_return_source_mode", "disabled")
        self.declare_parameter("event_return_clearance_timeout_s", 30.0)
        self.declare_parameter("event_return_timeout_s", 300.0)
        self.declare_parameter("event_return_acceptance_radius_m", 0.4)
        self.declare_parameter("event_return_home_radius_m", 1.0)
        self.declare_parameter("event_return_max_landing_height_m", 5.0)
        self.declare_parameter("event_return_trace_max_points", 4096)
        # The message/service contract is installed now; runtime activation follows
        # Jetson integration and SITL fault-injection verification.
        self.declare_parameter("enable_event_control", False)

        self._catalog = RouteCatalog.load(
            str(self.get_parameter("zones_file").value)
        )
        self._physical_route = bool(not self.get_parameter("simulation_only").value)
        self._forward_test_camera_required = bool(
            self.get_parameter("forward_test_camera_required").value
        )
        self._max_first_leg_m = float(self.get_parameter("max_first_leg_m").value)
        self._route_home_timeout_s = float(self.get_parameter("route_home_timeout_s").value)
        self._approval_execution_lease_s = float(
            self.get_parameter("approval_execution_lease_s").value)
        if not (0 < self._max_first_leg_m <= 500 and 0 < self._route_home_timeout_s <= 10
                and 0 < self._approval_execution_lease_s <= 30):
            raise ValueError("invalid route frame or approval lease bounds")
        self._acceptance_radius = float(
            self.get_parameter("acceptance_radius_m").value
        )
        self._takeoff_timeout = float(
            self.get_parameter("takeoff_timeout_s").value
        )
        self._waypoint_timeout = float(
            self.get_parameter("waypoint_timeout_s").value
        )
        self._landing_timeout = float(
            self.get_parameter("landing_timeout_s").value
        )
        self._ready_timeout = float(
            self.get_parameter("vehicle_ready_timeout_s").value
        )
        self._max_laps = int(self.get_parameter("max_laps").value)
        self._max_duration_minutes = int(
            self.get_parameter("max_duration_minutes").value
        )
        self._allow_goal_waypoints = bool(
            self.get_parameter("allow_goal_waypoints").value
        )
        self._event_control = EventControlCoordinator(
            float(self.get_parameter("event_control_max_duration_s").value)
        )
        self._event_control_enabled = bool(
            self.get_parameter("enable_event_control").value
        )
        self._event_hold_settle_s = float(
            self.get_parameter("event_hold_settle_s").value
        )
        self._event_hold_timeout_s = float(
            self.get_parameter("event_hold_timeout_s").value
        )
        self._event_hold_drift_tolerance_m = float(
            self.get_parameter("event_hold_drift_tolerance_m").value
        )
        self._event_rejoin_lookahead_m = float(
            self.get_parameter("event_rejoin_lookahead_m").value
        )
        self._event_completion_policy = str(self.get_parameter("event_completion_policy").value)
        if self._event_completion_policy not in {"capture_then_rtl", "legacy_rejoin"}:
            raise ValueError("event_completion_policy must be capture_then_rtl or legacy_rejoin")
        self._event_return_source_mode = str(self.get_parameter("event_return_source_mode").value)
        if self._event_return_source_mode not in {"disabled", "single_publisher_sitl"}:
            raise ValueError("event_return_source_mode must be disabled or single_publisher_sitl")
        self._readiness_only_checkpoint = self.get_parameter("readiness_only_checkpoint").value
        self._validate_readiness_checkpoint_profile(
            enabled=self._readiness_only_checkpoint,
            simulation_only=self.get_parameter("simulation_only").value,
            allow_real_hardware=self.get_parameter("allow_real_hardware").value,
            use_sim_time=self.get_parameter("use_sim_time").value,
            source_mode=self._event_return_source_mode,
            completion_policy=self._event_completion_policy,
        )
        self._diagnostic_relaxed_timing = self.get_parameter("diagnostic_relaxed_timing").value
        self._experiment_policy = None
        # This value is consulted by _capture_then_home() during startup profile
        # validation below.  Keep NORMAL as the fail-safe/default profile until
        # an approved mission explicitly selects another profile.
        self._active_flight_profile = "NORMAL"
        self._active_mission_kind = ROUTE
        if self.get_parameter("experiment_stage").value != 0:
            from jolgwa_uav.experiment_stage_policy import validate_manager_profile
            self._experiment_policy = validate_manager_profile(
                self.get_parameter("experiment_stage").value,
                enable_event_control=self._event_control_enabled,
                completion_policy=self._event_completion_policy,
                diagnostic_relaxed_timing=self._diagnostic_relaxed_timing,
                simulation_only=self.get_parameter("simulation_only").value,
                allow_real_hardware=self.get_parameter("allow_real_hardware").value)
        from .dashboard_sim_preparation import OwnedSimPreparation, validate_profile
        preparation_config = self.get_parameter("owned_sim_preparation_config").value
        validate_profile(enabled=bool(preparation_config),
            simulation_only=self.get_parameter("simulation_only").value,
            allow_real_hardware=self.get_parameter("allow_real_hardware").value,
            use_sim_time=self.get_parameter("use_sim_time").value,
            diagnostic_relaxed_timing=self._diagnostic_relaxed_timing,
            capture_then_home=self._capture_then_home())
        if preparation_config and (self._readiness_only_checkpoint
                or self._event_return_source_mode != "single_publisher_sitl"):
            raise ValueError("owned preparation requires native Home and a non-checkpoint action")
        self._owned_sim_preparation = OwnedSimPreparation(preparation_config) if preparation_config else None
        self._owned_sim_preparing = False
        self._owned_sim_handoff_complete = False
        self._owned_sim_failure_reason = ""
        self._validate_diagnostic_relaxed_timing_profile(
            enabled=self._diagnostic_relaxed_timing,
            simulation_only=self.get_parameter("simulation_only").value,
            allow_real_hardware=self.get_parameter("allow_real_hardware").value,
            use_sim_time=self.get_parameter("use_sim_time").value,
            source_mode=self._event_return_source_mode,
        )
        self._diagnostic_timing_warned = set()
        self._simple_demo_endpoint_observation_s = float(
            self.get_parameter("simple_demo_endpoint_observation_s").value)
        if not math.isfinite(self._simple_demo_endpoint_observation_s) or not (
                0.0 <= self._simple_demo_endpoint_observation_s <= 10.0):
            raise ValueError("simple_demo_endpoint_observation_s must be between 0 and 10")
        self._default_takeoff_altitude_m = self.get_parameter("default_takeoff_altitude_m").value
        self._validate_default_takeoff_altitude(
            self._default_takeoff_altitude_m,
            simulation_only=self.get_parameter("simulation_only").value,
            allow_real_hardware=self.get_parameter("allow_real_hardware").value,
            use_sim_time=self.get_parameter("use_sim_time").value,
            diagnostic_relaxed_timing=self._diagnostic_relaxed_timing,
        )
        self._simple_obstacle_demo = self.get_parameter("simple_obstacle_demo").value
        self._simple_demo_phase1_events = self.get_parameter("simple_demo_phase1_events").value
        if self._simple_demo_phase1_events and not self._simple_obstacle_demo:
            raise ValueError("simple_demo_phase1_events requires simple_obstacle_demo")
        if self._simple_obstacle_demo:
            if not self._diagnostic_relaxed_timing or self._event_completion_policy != "capture_then_rtl":
                raise ValueError("simple_obstacle_demo requires the explicit SIM diagnostic capture/Home profile")
            if not self._simple_demo_phase1_events:
                self._event_control.allowed_event_types = frozenset({"DEMO_PERSON"})
        if self._diagnostic_relaxed_timing:
            self.get_logger().warning("DIAGNOSTIC_RELAXED_TIMING: SIM Home timing expiry is warning-only; NOT safety validation")
        self._event_return_clearance_timeout = float(self.get_parameter("event_return_clearance_timeout_s").value)
        self._event_return_timeout = float(self.get_parameter("event_return_timeout_s").value)
        self._event_return_acceptance_radius = float(self.get_parameter("event_return_acceptance_radius_m").value)
        self._event_return_home_radius = float(self.get_parameter("event_return_home_radius_m").value)
        self._event_return_max_landing_height = float(self.get_parameter("event_return_max_landing_height_m").value)
        self._return_trace_policy = ReturnTracePolicy(
            maximum_points=int(self.get_parameter("event_return_trace_max_points").value),
            home_radius_m=self._event_return_home_radius)
        if not (math.isfinite(self._event_return_clearance_timeout)
                and 0 < self._event_return_clearance_timeout <= 120
                and self._event_return_clearance_timeout <= self._event_return_timeout <= 1800
                and 0 < self._event_return_acceptance_radius < self._return_trace_policy.spacing_m
                and 0 < self._event_return_home_radius <= 2
                and 0 < self._event_return_max_landing_height <= 10):
            raise ValueError("invalid bounded event return policy")
        if (
            self._event_hold_settle_s <= 0.0
            or self._event_hold_timeout_s < self._event_hold_settle_s
            or self._event_hold_drift_tolerance_m <= 0.0
            or not math.isfinite(self._event_rejoin_lookahead_m)
            or self._event_rejoin_lookahead_m < 0.0
        ):
            raise ValueError("invalid event HOLD confirmation policy")

        self._lock = threading.Lock()
        self._proposals: dict[str, MissionProposal] = {}
        self._approved: dict[str, str] = {}
        self._approved_at: dict[str, float] = {}
        self._claimed_approvals: set[str] = set()
        self._goal_reservation = None
        self._route_frame_results = {}
        self._route_frame_messages = {}
        self._consumed_preview_states = {}
        self._altitude_references = {}
        self._route_home = None
        self._route_home_received_at = float("-inf")
        self._route_home_fault = ""
        self._route_home_generation = 0
        self._vehicle_state: VehicleControlState | None = None
        self._vehicle_state_received_at = float("-inf")
        self._last_vehicle_ready_reason = "vehicle_state_missing"
        self._low_speed_failure_reason = ""
        self._home_candidate = None
        self._home_received_at = float("-inf")
        self._home_source_timestamp = 0
        self._home_reference = StaticHomeReference(
            diagnostic_relaxed_timing=self._diagnostic_relaxed_timing)
        self._home_graph_identity = None
        self._home_graph_received_at = float("-inf")
        self._home_prefix = None
        self._status_prefix = None
        self._home_pending = None
        self._home_replay_needed = False
        self._home_replay_attempts = 0
        self._home_graph_stable_samples = 0
        self._rmw_identifier = get_rmw_implementation_identifier()
        self._mission_home = None
        self._home_reference_invalidated = False
        self._return_trace = None
        self._trace_min_height_m = 1.0
        self._event_terminal_requested = False
        self._event_return_in_progress = False
        self._event_terminal_failure_reason = ""
        self._event_capture_succeeded = False
        self._event_outbound_command_spec = None
        self._active_goal = False
        self._manual_active = False
        self._sequence = 0
        self._active_mission_id = ""
        self._active_proposal_id = ""
        self._active_phase = MissionStatus.PHASE_IDLE
        self._active_route_ned: tuple[tuple[float, float, float], ...] = ()
        self._active_waypoint = 0
        self._active_total_waypoints = 0
        self._active_home_z_ned_m = 0.0
        self._active_altitude_reference = None
        self._active_route_home = None
        self._active_frame_continuity_error = ""
        self._low_settle_since = None
        self._vehicle_geo_state = None
        self._vehicle_geo_received_at = 0.0
        self._altitude_reference_state = None
        self._altitude_reference_state_received_at = 0.0
        self._low_speed_safety_state = None
        self._low_speed_safety_received_at = float("-inf")
        self._forward_previews = {}
        self._last_command_spec: tuple | None = None
        self._pending_resume_command: tuple | None = None
        self._resume_required = False
        self._resume_requested = False
        self._resume_reason = ""
        self._resume_return_phase = MissionStatus.PHASE_IDLE
        self._active_cancel_requested = False
        self._emergency_land_requested = False
        self._active_ever_armed = False
        self._terminal_deadlines = {}

        callback_group = ReentrantCallbackGroup()
        self._approval_publisher = self.create_publisher(
            MissionApproval, MISSION_APPROVAL, 10
        )
        self._status_publisher = self.create_publisher(
            MissionStatus, MISSION_STATUS, 10
        )
        self._route_frame_publisher = self.create_publisher(
            RouteFrameStatus, ROUTE_FRAME_STATUS, 10)
        self._command_publisher = self.create_publisher(
            OffboardCommand, OFFBOARD_COMMAND, 10
        )
        self._manual_publisher = self.create_publisher(
            ManualOverride, MANUAL_OVERRIDE, 10
        )
        self._proposal_publisher = self.create_publisher(
            MissionProposal, MISSION_PROPOSAL, 10)
        self.create_subscription(
            MissionProposal,
            MISSION_PROPOSAL,
            self._on_proposal,
            10,
            callback_group=callback_group,
        )
        self.create_subscription(
            VehicleControlState,
            VEHICLE_CONTROL_STATE,
            self._on_vehicle_state,
            qos_profile_sensor_data,
            callback_group=callback_group,
        )
        self.create_subscription(
            VehicleGeoState, VEHICLE_GEO_STATE, self._on_vehicle_geo_state,
            qos_profile_sensor_data, callback_group=callback_group)
        self.create_subscription(
            AltitudeReferenceState, ALTITUDE_REFERENCE_STATE,
            self._on_altitude_reference_state,
            qos_profile_sensor_data, callback_group=callback_group)
        self.create_subscription(
            LowSpeedSafetyState, LOW_SPEED_SAFETY_STATE,
            self._on_low_speed_safety_state, qos_profile_sensor_data,
            callback_group=callback_group)
        self._home_callback_group = callback_group
        self._home_qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._home_subscription = self.create_subscription(HomePosition, PX4_HOME_POSITION,
            self._on_home_position, self._home_qos, callback_group=callback_group)
        self.create_subscription(VehicleStatus, PX4_VEHICLE_STATUS, self._on_home_vehicle_status,
                                 qos_profile_sensor_data, callback_group=callback_group)
        # Identity/freshness checks must continue during a paused ROS sim clock.
        self._home_watchdog_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(.2, self._refresh_home_sources, callback_group=callback_group,
                          clock=self._home_watchdog_clock)
        self.create_timer(.2, self._approval_execution_watchdog,
                          callback_group=callback_group, clock=self._home_watchdog_clock)
        self.create_service(
            ApproveMission,
            APPROVE_MISSION_SERVICE,
            self._on_approve,
            callback_group=callback_group,
        )
        self.create_service(
            SetManualOverride,
            MANUAL_OVERRIDE_SERVICE,
            self._on_manual_override,
            callback_group=callback_group,
        )
        self.create_service(
            RequestEventControl,
            REQUEST_EVENT_CONTROL_SERVICE,
            self._on_request_event_control,
            callback_group=callback_group,
        )
        self.create_service(
            ReleaseEventControl,
            RELEASE_EVENT_CONTROL_SERVICE,
            self._on_release_event_control,
            callback_group=callback_group,
        )
        self.create_service(
            ResumeMission,
            RESUME_MISSION_SERVICE,
            self._on_resume_mission,
            callback_group=callback_group,
        )
        self.create_service(
            PrepareForwardTest, PREPARE_FORWARD_TEST_SERVICE,
            self._on_prepare_forward_test, callback_group=callback_group)
        self.create_service(
            EmergencyLand, EMERGENCY_LAND_SERVICE,
            self._on_emergency_land, callback_group=callback_group)
        self._action_server = ActionServer(
            self,
            ExecuteMission,
            EXECUTE_MISSION_ACTION,
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=self._on_cancel,
            callback_group=callback_group,
        )

        init_manager(self)

    def _on_proposal(self, message: MissionProposal) -> None:
        conflict = False
        with self._lock:
            existing = self._proposals.get(message.proposal_id)
            if existing is not None:
                try:
                    same = self._same_json(existing.plan_json, message.plan_json)
                except Exception:
                    same = existing.plan_json == message.plan_json
                if not same:
                    conflict = True
                else:
                    # Idempotent transport replay: retain the originally
                    # reviewed object and its route-frame/approval identity.
                    return
            else:
                self._proposals[message.proposal_id] = message
            self._prune_mission_caches_locked()
        if conflict:
            self.get_logger().error(
                "proposal_id collision rejected: " + message.proposal_id)
            self._publish_status(
                "", message.proposal_id, MissionStatus.PHASE_ERROR,
                "proposal_id collision: content differs from frozen proposal",
            )
            return
        self._update_route_frame_status(message.proposal_id)
        phase = (
            MissionStatus.PHASE_AWAITING_APPROVAL
            if message.status == MissionProposal.STATUS_OK
            else MissionStatus.PHASE_ERROR
        )
        self._publish_status(
            "",
            message.proposal_id,
            phase,
            message.message or "mission proposal received",
        )

    def _prune_mission_caches_locked(self):
        protected = set(self._approved.values())
        if self._active_proposal_id:
            protected.add(self._active_proposal_id)
        reservation = self._goal_reservation
        if reservation is not None:
            protected.add(reservation.get("proposal_id", ""))
        while len(self._proposals) > self.CACHE_LIMIT:
            candidate = next((key for key in self._proposals if key not in protected), None)
            if candidate is None:
                break
            self._proposals.pop(candidate, None)
            self._route_frame_results.pop(candidate, None)
            self._route_frame_messages.pop(candidate, None)
            self._consumed_preview_states.pop(candidate, None)
            self._forward_previews.pop(candidate, None)
            self._altitude_references.pop(candidate, None)
        for candidate in tuple(self._forward_previews):
            preview = self._forward_previews[candidate]
            if (candidate not in protected
                    and time.monotonic()-preview.get("created_at", 0.0) > PREVIEW_TTL_S):
                self._forward_previews.pop(candidate, None)
                self._route_frame_results.pop(candidate, None)
                self._altitude_references.pop(candidate, None)

    def _capture_altitude_reference_locked(self, raw_home, now=None):
        """Capture the bridge-proven stable FC-time-paired vertical datum."""
        now = time.monotonic() if now is None else float(now)
        sample = self._altitude_reference_state
        if raw_home is None or sample is None:
            raise ValueError("altitude_reference_stale")
        if hasattr(sample, "provisional_px4_home_z_ned_m"):
            # Home and altitude diagnostics arrive on independent ROS topics.
            # Do not combine Home B with a retained READY sample from Home A.
            locked = bool(sample.execution_home_lock_valid)
            sample_z = (sample.frozen_px4_home_z_ned_m if locked
                        else sample.provisional_px4_home_z_ned_m)
            sample_alt = (sample.frozen_px4_home_altitude_amsl_m if locked
                          else sample.provisional_px4_home_altitude_amsl_m)
            if (not all(math.isfinite(float(v)) for v in (sample_z, sample_alt))
                    or abs(raw_home.ned[2]-sample_z) > 0.01
                    or abs(raw_home.altitude_m-sample_alt) > 0.01):
                raise ValueError("provisional_home_alignment_pending")
        try:
            local_age = accumulated_sample_age_s(sample.local_age_ms, self._altitude_reference_state_received_at, now)
            global_age = accumulated_sample_age_s(sample.global_age_ms, self._altitude_reference_state_received_at, now)
            source_skew = float(sample.source_skew_ms)/1000.0
            local_z = float(sample.local_z_ned_m)
            fc_altitude = float(getattr(
                sample, "normalized_fc_altitude_home_relative_m",
                sample.fc_altitude_home_relative_m))
            aligned_z = float(sample.stable_aligned_home_z_ned_m)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("altitude_reference_stale") from exc
        reason, _ = altitude_reference_sample_error(
            aligned_home_z=aligned_z,
            local_z_ned=local_z,
            fc_altitude_home_relative_m=fc_altitude,
            local_age_s=local_age,
            geo_age_s=global_age,
            sample_skew_s=source_skew,
        )
        if (not bool(getattr(sample, "valid", False))
                or not bool(getattr(sample, "stable", False))
                or int(getattr(sample, "state", -1)) != ALTITUDE_ALIGNMENT_READY):
            detail = str(getattr(sample, "detail", ""))
            raise ValueError(
                "altitude_reference_stale"
                if detail == "altitude_reference_state_stale"
                else "altitude_reference_unstable")
        if reason:
            raise ValueError(reason or "altitude_reference_stale")
        return {
            "home_generation": getattr(self, "_route_home_generation", 0),
            "aligned_home_z_ned_m": aligned_z,
            "capture_local_z_ned_m": local_z,
            "capture_fc_altitude_home_relative_m": fc_altitude,
            "capture_raw_fc_altitude_home_relative_m": float(
                sample.fc_altitude_home_relative_m),
            "home_correction_revision": int(
                getattr(sample, "home_correction_revision", 0)),
            "transport_epoch": int(sample.transport_epoch),
            "sequence": int(sample.sequence),
            "local_time_boot_ms": int(sample.local_time_boot_ms),
            "global_time_boot_ms": int(sample.global_time_boot_ms),
            "captured_at": now,
            "raw_home": raw_home,
        }

    def _altitude_reference_status_locked(self, reference, now=None):
        now = time.monotonic() if now is None else float(now)
        sample = self._altitude_reference_state
        if reference is None or sample is None:
            return "altitude_reference_stale", math.inf, math.nan, math.nan
        if reference.get("home_generation", 0) != getattr(self, "_route_home_generation", 0):
            return "provisional_home_changed_reprepare_required", math.inf, math.nan, math.nan
        try:
            local_z = float(sample.local_z_ned_m)
            fc_altitude = float(getattr(
                sample, "normalized_fc_altitude_home_relative_m",
                sample.fc_altitude_home_relative_m))
            frame_altitude = float(reference["aligned_home_z_ned_m"])-local_z
            local_age = accumulated_sample_age_s(sample.local_age_ms, self._altitude_reference_state_received_at, now)
            global_age = accumulated_sample_age_s(sample.global_age_ms, self._altitude_reference_state_received_at, now)
            source_skew = float(sample.source_skew_ms)/1000.0
        except (TypeError, ValueError, OverflowError, IndexError, KeyError):
            return "altitude_reference_stale", math.inf, math.nan, math.nan
        if int(sample.transport_epoch) != int(reference.get("transport_epoch", -1)):
            return "altitude_reference_epoch_changed", math.inf, fc_altitude, frame_altitude
        pending_allowed = bool(
            int(getattr(sample, "state", -1))
            == AltitudeReferenceState.STATE_HOME_CORRECTION_PENDING
            and self._active_ever_armed
            and self._active_mission_id
            and accumulated_sample_age_s(getattr(sample, "home_correction_pending_age_ms", 2**32-1),
                self._altitude_reference_state_received_at, now) <= battery_home.home_confirmation_ms(self, mission_id=self._active_mission_id)/1000.0)
        if (not bool(getattr(sample, "valid", False))
                or int(getattr(sample, "state", -1)) == AltitudeReferenceState.STATE_STALE):
            return "altitude_reference_stale", math.inf, fc_altitude, frame_altitude
        if (not bool(getattr(sample, "stable", False))
                or (int(getattr(sample, "state", -1))
                    == AltitudeReferenceState.STATE_HOME_CORRECTION_PENDING
                    and not pending_allowed)):
            return "altitude_reference_unstable", math.inf, fc_altitude, frame_altitude
        reason, error = altitude_reference_sample_error(
            aligned_home_z=reference["aligned_home_z_ned_m"],
            local_z_ned=local_z,
            fc_altitude_home_relative_m=fc_altitude,
            local_age_s=local_age,
            geo_age_s=global_age,
            sample_skew_s=source_skew,
            max_error_m=ALTITUDE_REFERENCE_MAX_ERROR_M,
        )
        return reason, error, fc_altitude, frame_altitude

    @staticmethod
    def _home_with_aligned_z(raw_home, aligned_z):
        return RouteHome(
            raw_home.latitude_deg,
            raw_home.longitude_deg,
            raw_home.altitude_m,
            (float(raw_home.ned[0]), float(raw_home.ned[1]), float(aligned_z)),
            raw_home.source_timestamp_us,
        )

    def _update_all_route_frame_statuses(self):
        with self._lock:
            # Approval freezes both route geometry and the aligned low-speed
            # vertical datum.  Periodic telemetry must never rewrite it.
            protected = set(self._approved.values())
            if self._active_proposal_id:
                protected.add(self._active_proposal_id)
            proposal_ids = tuple(
                proposal_id for proposal_id in self._proposals
                if proposal_id not in protected
            )
        for proposal_id in proposal_ids[-4:]:
            self._update_route_frame_status(proposal_id)

    def _update_route_frame_status(self, proposal_id):
        with self._lock:
            consumed = self._consumed_preview_states.get(proposal_id)
            cached = self._route_frame_messages.get(proposal_id)
        if consumed and cached is not None:
            cached.preview_state = consumed
            cached.reason = ""
            self._route_frame_publisher.publish(cached)
            return
        now = time.monotonic()
        with self._lock:
            proposal = self._proposals.get(proposal_id)
            state = self._vehicle_state
            home = (self._route_home if 0 <= now-self._route_home_received_at
                    <= self._route_home_timeout_s else None)
            frozen_altitude_reference = self._altitude_references.get(proposal_id)
            preview = self._forward_previews.get(proposal_id)
        result, reason = None, ""
        altitude_reference = frozen_altitude_reference
        altitude_detail = "not_required"
        altitude_error = math.nan
        fc_altitude = math.nan
        frame_altitude = math.nan
        raw_home = home
        try:
            if proposal is None:
                raise ValueError("proposal unavailable")
            plan = json.loads(proposal.plan_json)
            validate_low_speed_plan(plan)
            flight_profile = profile_for_plan(plan)
            if state is None or not state.connected or not state.position_valid:
                raise ValueError("fresh valid PX4 local position is required")
            frame_home = home
            if is_low_speed_profile(flight_profile):
                if home is None:
                    raise ValueError("fresh validated PX4 Home is required")
                if altitude_reference is None:
                    with self._lock:
                        altitude_reference = self._capture_altitude_reference_locked(
                            home, now)
                with self._lock:
                    altitude_detail, altitude_error, fc_altitude, frame_altitude = (
                        self._altitude_reference_status_locked(
                            altitude_reference, now))
                if altitude_detail:
                    raise ValueError(altitude_detail)
                frame_home = self._home_with_aligned_z(
                    home, altitude_reference["aligned_home_z_ned_m"])
                altitude_detail = "ready"
            if mission_kind_for_plan(plan) == FORWARD_TEST_1M:
                if preview is None or preview["token"] != plan.get("preview_token"):
                    raise ValueError("forward-test preview is unavailable")
                valid, preview_reason = preview_valid(
                    preview, state.position_ned_m, state.heading_rad, now)
                if not valid:
                    raise ValueError(preview_reason)
                target_altitude = target_altitude_for_profile(flight_profile)
                endpoint = (preview["end_ned"][0], preview["end_ned"][1],
                            frame_home.ned[2]-target_altitude)
                first_leg = distance_ned(state.position_ned_m, endpoint)
                result = RouteFrameResult((endpoint,), frame_home, "FORWARD_TEST_LOCAL_NED",
                                          first_leg, first_leg)
            else:
                points = self._catalog.points_for_plan(proposal.plan_json)
                result = transform_route(
                    points, proposal.plan_json, home=frame_home,
                    current_ned=tuple(state.position_ned_m), physical=self._physical_route,
                    max_first_leg_m=self._max_first_leg_m)
        except Exception as exc:
            reason = str(exc)
        with self._lock:
            self._route_frame_results[proposal_id] = (result, reason, now)
            revoked = []
            if self._physical_route and result is None:
                for mission_id, approved_proposal in tuple(self._approved.items()):
                    if approved_proposal == proposal_id:
                        if (mission_id in self._claimed_approvals
                                or mission_id == self._active_mission_id):
                            # Preflight geometry is consumed at goal claim. It
                            # must never revoke an executing vehicle merely
                            # because that vehicle moved away from its preview.
                            continue
                        self._approved.pop(mission_id, None)
                        self._approved_at.pop(mission_id, None)
                        self._claimed_approvals.discard(mission_id)
                        revoked.append(mission_id)
        message = RouteFrameStatus()
        message.stamp = self.get_clock().now().to_msg()
        message.proposal_id = proposal_id
        message.valid = result is not None
        message.site_id = result.site_id if result else ""
        if result is not None:
            display_home = raw_home if raw_home is not None else result.home
            if display_home is not None:
                message.home_latitude_deg = display_home.latitude_deg
                message.home_longitude_deg = display_home.longitude_deg
                message.home_ned_m = list(display_home.ned)
            message.first_target_ned_m = list(result.route_ned[0])
            message.first_leg_m = result.first_leg_m
            message.maximum_leg_m = result.maximum_leg_m
        message.reason = reason
        message.preview_state = "AVAILABLE" if preview is not None else "NONE"
        message.altitude_reference_error_valid = math.isfinite(altitude_error)
        message.altitude_reference_valid = bool(
            altitude_reference is not None and altitude_detail == "ready")
        message.altitude_reference_diagnostics_available = bool(
            altitude_reference is not None
            and math.isfinite(fc_altitude)
            and math.isfinite(frame_altitude))
        message.aligned_home_z_ned_m = float(
            altitude_reference["aligned_home_z_ned_m"]
            if altitude_reference is not None else 0.0)
        message.fc_altitude_home_relative_m = float(
            getattr(self._altitude_reference_state,
                    "fc_altitude_home_relative_m", 0.0)
            if self._altitude_reference_state is not None else 0.0)
        message.normalized_fc_altitude_home_relative_m = float(
            fc_altitude if math.isfinite(fc_altitude) else 0.0)
        message.frame_altitude_home_relative_m = float(
            frame_altitude if math.isfinite(frame_altitude) else 0.0)
        message.altitude_reference_error_m = float(
            altitude_error if math.isfinite(altitude_error) else math.nan)
        message.altitude_reference_detail = altitude_detail
        altitude_sample = self._altitude_reference_state
        message.home_correction_state = int(getattr(
            altitude_sample, "home_correction_state", 0))
        message.home_correction_valid = bool(getattr(
            altitude_sample, "home_correction_valid", False))
        message.home_correction_revision = int(getattr(
            altitude_sample, "home_correction_revision", 0))
        message.home_altitude_correction_m = float(getattr(
            altitude_sample, "home_altitude_correction_m", 0.0))
        message.home_correction_opposition_error_m = float(getattr(
            altitude_sample, "home_correction_opposition_error_m", 0.0))
        message.home_correction_detail = str(getattr(
            altitude_sample, "home_correction_detail", "unavailable"))
        message.home_phase = int(getattr(altitude_sample, "home_phase", 0))
        message.execution_home_lock_valid = bool(getattr(
            altitude_sample, "execution_home_lock_valid", False))
        message.execution_home_mission_id = str(getattr(
            altitude_sample, "execution_home_mission_id", ""))
        message.execution_home_lock_revision = int(getattr(
            altitude_sample, "execution_home_lock_revision", 0))
        message.provisional_home_revision = int(getattr(
            altitude_sample, "provisional_home_revision", 0))
        message.provisional_px4_home_altitude_amsl_m = float(getattr(
            altitude_sample, "provisional_px4_home_altitude_amsl_m", 0.0))
        message.provisional_px4_home_z_ned_m = float(getattr(
            altitude_sample, "provisional_px4_home_z_ned_m", 0.0))
        message.frozen_px4_home_altitude_amsl_m = float(getattr(
            altitude_sample, "frozen_px4_home_altitude_amsl_m", 0.0))
        message.frozen_px4_home_z_ned_m = float(getattr(
            altitude_sample, "frozen_px4_home_z_ned_m", 0.0))
        message.current_px4_home_altitude_amsl_m = float(getattr(
            altitude_sample, "current_px4_home_altitude_amsl_m", 0.0))
        message.current_px4_home_z_ned_m = float(getattr(
            altitude_sample, "current_px4_home_z_ned_m", 0.0))
        message.home_z_correction_m = float(getattr(
            altitude_sample, "home_z_correction_m", 0.0))
        message.altitude_epoch_failure_latched = bool(getattr(
            altitude_sample, "altitude_epoch_failure_latched", False))
        self._route_frame_messages[proposal_id] = message
        self._route_frame_publisher.publish(message)
        for mission_id in revoked:
            self._publish_approval(proposal_id, mission_id, False, "mission-manager")

    def _approval_execution_watchdog(self):
        now = time.monotonic()
        expired = []
        with self._lock:
            reservation = self._goal_reservation
            if (reservation is not None and not self._active_goal
                    and now-reservation["created_at"] >= self._approval_execution_lease_s):
                mission_id = reservation["mission_id"]
                proposal_id = self._approved.pop(mission_id, None)
                self._approved_at.pop(mission_id, None)
                self._claimed_approvals.discard(mission_id)
                self._goal_reservation = None
                self._active_cancel_requested = False
                self._active_ever_armed = False
                if proposal_id:
                    expired.append((mission_id, proposal_id))
            for mission_id, approved_at in tuple(self._approved_at.items()):
                if (mission_id not in self._claimed_approvals
                        and now-approved_at >= self._approval_execution_lease_s):
                    proposal_id = self._approved.pop(mission_id, None)
                    self._approved_at.pop(mission_id, None)
                    if proposal_id:
                        expired.append((mission_id, proposal_id))
        for mission_id, proposal_id in expired:
            self._publish_approval(proposal_id, mission_id, False, "mission-manager")
            self._publish_status(mission_id, proposal_id, MissionStatus.PHASE_ERROR,
                                 "ExecuteMission goal was not accepted within 5 seconds; approval revoked")

    def _on_vehicle_state(self, message: VehicleControlState) -> None:
        now = time.monotonic()
        with self._lock:
            self._vehicle_state = message
            self._vehicle_state_received_at = now
            if self._active_goal and getattr(message, "arm_request_transmitted", False):
                self._active_ever_armed = True
            self._manual_active = bool(message.manual_override)
            if (self._active_goal and bool(getattr(message, "arming_state_valid", False))
                    and bool(message.armed)):
                self._active_ever_armed = True
            trace = getattr(self, "_return_trace", None)
            if (trace is not None and not self._event_return_in_progress
                    and not owned_navigation_error(message, mission_id=self._active_mission_id,
                        received_at=now, now=now)
                    # Height authorizes the first airborne sample only. Once
                    # started, retain fresh owned telemetry through nominal
                    # cruise-height oscillation; observe keeps real gap guards.
                    and (trace.points or self._mission_home[2]-float(message.position_ned_m[2])
                         >= self._trace_min_height_m)):
                trace.observe(message.position_ned_m, now)
        self._update_all_route_frame_statuses()

    def _on_vehicle_geo_state(self, message: VehicleGeoState) -> None:
        with self._lock:
            self._vehicle_geo_state = message
            self._vehicle_geo_received_at = time.monotonic()

    def _on_altitude_reference_state(self, message: AltitudeReferenceState) -> None:
        with self._lock:
            from .scenario_yaw_reset import observe_reset_metadata
            observe_reset_metadata(self, self._altitude_reference_state, message)
            if not altitude_sample_advances(self._altitude_reference_state, message):
                return
            self._altitude_reference_state = message
            self._altitude_reference_state_received_at = time.monotonic()

    def _on_low_speed_safety_state(self, message: LowSpeedSafetyState) -> None:
        with self._lock:
            self._low_speed_safety_state = message
            self._low_speed_safety_received_at = time.monotonic()

    def _low_speed_readiness_error_locked(
        self, *, require_ground=True, allow_active=False,
        mission_kind=ROUTE,
    ):
        if getattr(self, "_cleanup_fault", ""):
            return self._cleanup_fault
        now = time.monotonic()
        state = self._vehicle_state
        geo = (self._vehicle_geo_state
               if 0 <= now-self._vehicle_geo_received_at <= 0.75 else None)
        safety = (self._low_speed_safety_state
                  if 0 <= now-self._low_speed_safety_received_at <= 0.5
                  else None)
        home_fresh = (self._route_home is not None
                      and 0 <= now-self._route_home_received_at <= self._route_home_timeout_s)
        return low_speed_readiness_error(
            state=state, geo=geo, safety=safety, now=now,
            vehicle_state_received_at=self._vehicle_state_received_at,
            active_mission=bool(self._active_goal), allow_active=allow_active,
            require_ground=require_ground, physical=self._physical_route,
            home_fresh=home_fresh,
            require_obstacle_guard=obstacle_guard_required_for_mission(
                mission_kind, self._forward_test_camera_required and not getattr(self, "_scenario_enabled", False)
            ),
        )

    def _on_prepare_forward_test(self, request, response):
        with self._lock:
            error = self._low_speed_readiness_error_locked(
                require_ground=True, mission_kind=FORWARD_TEST_1M
            )
            state = self._vehicle_state
            home = self._route_home
            try:
                altitude_reference = (
                    None if error else
                    self._capture_altitude_reference_locked(home)
                )
            except ValueError as exc:
                altitude_reference = None
                error = str(exc)
        if error:
            response.message = error
            return response
        try:
            profile = profile_for_target_altitude(request.target_altitude_home_m)
            target_altitude = target_altitude_for_profile(profile)
            start = tuple(float(v) for v in state.position_ned_m)
            heading = float(state.heading_rad)
            end = forward_endpoint_ned(start, heading)
            proposal_id = str(uuid.uuid4())
            token = uuid.uuid4().hex
            created = time.monotonic()
            plan = {
                "request_purpose": "EXECUTE_MISSION", "status": "OK",
                "mission_kind": FORWARD_TEST_1M, "flight_profile": profile,
                "low_speed_limits": fixed_limits(profile),
                "completion_policy": "LAND_AT_FINAL_WAYPOINT",
                "after_response": "LAND_AT_FINAL_WAYPOINT",
                "camera_obstacle_guard": (
                    "REQUIRED"
                    if self._forward_test_camera_required
                    else "DISABLED_BY_OPERATOR_CONFIG"
                ),
                "preview_token": token,
                "preview_start_ned_m": list(start), "preview_end_ned_m": list(end),
                "preview_heading_rad": heading, "preview_ttl_s": PREVIEW_TTL_S,
                "preview_created_unix_ms": int(time.time()*1000),
                "message": (
                    "현재 위치에서 HOME 기준 %.1f m로 저속 이륙 후 "
                    "기수 방향 2 m 이동 및 착륙%s"
                    % (
                        target_altitude,
                        (
                            ""
                            if self._forward_test_camera_required
                            else " (카메라 장애물 보호 OFF)"
                        ),
                    )
                ),
            }
            if getattr(self, "_scenario_enabled", False):
                plan["test_scenario"] = make_spec(start, heading,
                    altitude_reference["aligned_home_z_ned_m"],
                    altitude_reference["transport_epoch"], target_altitude)
                end = scenario_point(plan["test_scenario"], 10.0, z=start[2])
                plan["preview_end_ned_m"] = list(end)
                plan["camera_obstacle_guard"] = "HUMAN_CHECK_RUNTIME_PERCEPTION"
                plan["message"] = ("통합 시험: HOME %.1fm / 상승 최대 %.1fm / 탐색 10m + 마지막 1m / "
                                   "장애물 통과·원고도 복귀 → 투기 정지·사진 → 1m 전진·착륙 / 미검출 대기 20초"
                                   % (target_altitude, target_altitude+2.0))
                validate_spec(plan)
            validate_low_speed_plan(plan)
        except (ValueError, TypeError) as exc:
            response.message = str(exc)
            return response
        preview = {"token": token, "created_at": created, "start_ned": start,
                   "end_ned": end, "heading_rad": heading, "home": home,
                   "altitude_reference": altitude_reference}
        with self._lock:
            if altitude_reference["home_generation"] != getattr(self, "_route_home_generation", 0):
                response.message = "provisional_home_changed_reprepare_required"
                return response
            self._forward_previews[proposal_id] = preview
            self._altitude_references[proposal_id] = altitude_reference
            self._prune_mission_caches_locked()
        proposal = MissionProposal()
        proposal.stamp = self.get_clock().now().to_msg()
        proposal.proposal_id = proposal_id
        proposal.command_id = str(uuid.uuid4())
        proposal.raw_command = "현재 위치 %.1f m 고도·2 m 저속 전진 시험" % target_altitude
        if "test_scenario" in plan:
            proposal.raw_command = plan["message"]
        proposal.status = MissionProposal.STATUS_OK
        proposal.plan_json = json.dumps(plan, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        proposal.message = plan["message"]
        proposal.requires_approval = True
        self._proposal_publisher.publish(proposal)
        response.accepted = True
        response.proposal_id = proposal_id
        response.message = "forward-test preview prepared"
        return response

    def _on_home_position(self, message):
        now = time.monotonic()
        revoked = []
        try:
            home_from_px4(message)
            latitude, longitude, altitude = px4_home_geodetic(message)
            candidate = RouteHome(
                latitude, longitude, altitude,
                (float(message.x), float(message.y), float(message.z)), int(message.timestamp))
        except (ValueError, TypeError, AttributeError, OverflowError):
            candidate = None
        with self._lock:
            if candidate is not None:
                frozen = self._active_route_home
                if self._route_home_fault:
                    pass
                elif (frozen is not None
                        and self._route_home_changed(frozen, candidate)):
                    self._active_frame_continuity_error = (
                        "PX4 Home/local-NED epoch changed during active mission"
                    )
                    self._route_home_fault = self._active_frame_continuity_error
                elif (self._route_home is not None and frozen is None
                        and self._route_home_changed(self._route_home, candidate, 0.001)):
                    state = self._vehicle_state
                    ground_fresh = bool(
                        state is not None
                        and 0 <= now-self._vehicle_state_received_at <= 0.5
                        and state.connected and state.vehicle_status_fresh
                        and state.arming_state_valid and not state.armed
                        and state.landed_state_valid and state.landed)
                    if (self._active_goal or self._goal_reservation is not None
                            or self._claimed_approvals):
                        # An accepted/being-validated goal owns its snapshot.
                        # Leave its approval for the terminal supervisor.
                        self._active_frame_continuity_error = (
                            "PX4 Home/local-NED epoch changed during reserved mission")
                        self._route_home_fault = self._active_frame_continuity_error
                    elif ground_fresh:
                        self._route_home_generation += 1
                        self._route_home = candidate
                        self._route_home_received_at = now
                        self._forward_previews.clear()
                        # Retain captured references with their OLD generation:
                        # an old proposal must not silently capture a new Home.
                        for mission_id, proposal_id in tuple(self._approved.items()):
                            revoked.append((mission_id, proposal_id))
                            self._approved.pop(mission_id, None)
                            self._approved_at.pop(mission_id, None)
                    # Unknown/airborne state never authorizes a new origin.
                    # Leave receipt age unchanged; retry on the next Home.
                else:
                    self._route_home = candidate
                    self._route_home_received_at = now
        for mission_id, proposal_id in revoked:
            self._publish_approval(proposal_id, mission_id, False, "mission-manager")
            self._publish_status(mission_id, proposal_id, MissionStatus.PHASE_ERROR,
                                 "provisional_home_changed_reprepare_required")
        self._update_all_route_frame_statuses()
        if self._event_return_source_mode != "single_publisher_sitl":
            return
        with self._lock:
            self._home_pending = message
            if self._home_prefix is None or self._home_graph_stable_samples < 2:
                # Do not relabel a previously unassociated message. Recreate
                # the transient-local subscription once graph identity settles.
                self._home_replay_needed = True
                return
            self._home_reference.observe_home(message, self._home_prefix, time.monotonic())
            self._home_candidate = self._home_reference.home
            self._home_received_at = self._home_reference.home_received_at
            self._home_source_timestamp = self._home_reference.home_stamp
            if self._home_reference.failure:
                self._home_reference_invalidated = True
            if (self._mission_home is not None and self._home_candidate is not None
                    and distance_ned(self._home_candidate, self._mission_home) > .5):
                self._home_reference.invalidate("Home drifted more than 0.5m from frozen mission reference")
                self._home_reference_invalidated = True
            # Read-only evidence after the original source/validity/drift checks.
            # A raw callback or pre-graph pending sample is never called accepted.
            reference = self._home_reference
            if (reference.home is not None and not reference.failure
                    and reference.home_stamp == message.timestamp
                    and reference.home_prefix == self._home_prefix
                    and getattr(self, "_home_accepted_logged_stamp", None) != reference.home_stamp):
                self._home_accepted_logged_stamp = reference.home_stamp
                self.get_logger().info("HOME_REFERENCE_ACCEPTED " + json.dumps({
                    "pid": os.getpid(), "run": os.environ.get("JOLGWA_SIM_RUN_DIR", ""),
                    "source_timestamp_us": reference.home_stamp,
                    "ned_m": list(reference.home),
                    "source_prefix": list(reference.home_prefix),
                    "received_monotonic_s": reference.home_received_at,
                    "graph_stable_samples": self._home_graph_stable_samples,
                    "failure": reference.failure,
                }, sort_keys=True))

    @staticmethod
    def _route_home_changed(first: RouteHome, second: RouteHome, threshold=0.5) -> bool:
        east, north = site_enu_from_wgs84(
            second.latitude_deg, second.longitude_deg,
            first.latitude_deg, first.longitude_deg,
        )
        return bool(
            math.hypot(east, north) > threshold
            or abs(second.altitude_m-first.altitude_m) > threshold
            or distance_ned(first.ned, second.ned) > threshold
        )

    def _on_home_vehicle_status(self, message):
        with self._lock:
            if (self._event_return_source_mode != "single_publisher_sitl"
                    or self._status_prefix is None or self._home_graph_stable_samples < 2):
                return
            self._home_reference.observe_status(message, self._status_prefix, time.monotonic())
            self._warn_diagnostic_home_timing_locked()
            if self._home_reference.failure:
                self._home_reference_invalidated = True

    def _refresh_home_sources(self):
        if self._event_return_source_mode != "single_publisher_sitl":
            return
        homes = self.get_publishers_info_by_topic(PX4_HOME_POSITION)
        statuses = self.get_publishers_info_by_topic(PX4_VEHICLE_STATUS)
        home_prefix = single_fastdds_prefix(homes, self._rmw_identifier)
        status_prefix = single_fastdds_prefix(statuses, self._rmw_identifier)
        identity = ((tuple(homes[0].endpoint_gid), tuple(statuses[0].endpoint_gid))
                    if home_prefix is not None and status_prefix is not None else None)
        replay = False
        with self._lock:
            if identity is None or home_prefix != status_prefix:
                self._home_graph_stable_samples = 0
                self._home_prefix = self._status_prefix = None
                if self._home_reference.home is not None or self._home_reference.status_samples:
                    self._home_reference.invalidate("unique matching PX4 DDS endpoints disappeared")
                    self._home_reference_invalidated = True
                return
            if self._home_graph_identity is not None and identity != self._home_graph_identity:
                self._home_reference.invalidate("PX4 DDS endpoint GID changed")
                self._home_reference_invalidated = True
                return
            self._home_graph_identity = identity
            self._home_graph_received_at = time.monotonic()
            self._home_prefix, self._status_prefix = home_prefix, status_prefix
            self._home_graph_stable_samples += 1
            if (self._home_replay_needed and self._home_graph_stable_samples >= 2
                    and self._home_replay_attempts < 3 and not self._home_reference.failure):
                self._home_replay_attempts += 1
                self._home_replay_needed = False
                replay = True
        if replay:
            self.destroy_subscription(self._home_subscription)
            self._home_subscription = self.create_subscription(HomePosition, PX4_HOME_POSITION,
                self._on_home_position, self._home_qos, callback_group=self._home_callback_group)

    def _current_home_reference_locked(self, now):
        if (self._event_return_source_mode != "single_publisher_sitl"
                or not self._home_graph_timing_valid_locked(now)):
            return None
        home = self._home_reference.current_home(now, home_prefix=self._home_prefix,
                                                status_prefix=self._status_prefix)
        self._warn_diagnostic_home_timing_locked()
        return home

    def _home_graph_timing_valid_locked(self, now):
        age = now-self._home_graph_received_at
        if not getattr(self, "_diagnostic_relaxed_timing", False):
            return 0 <= age <= .5
        # Expiry may be diagnosed, but never invent a graph association or
        # accept a clock reversal. Actual endpoint changes still latch above.
        if (self._home_graph_identity is None or not math.isfinite(age) or age < 0
                or self._home_prefix is None or self._status_prefix is None):
            return False
        if age > .5:
            self._warn_diagnostic_home_timing_locked("Home graph age exceeded")
        return True

    def _warn_diagnostic_home_timing_locked(self, extra=None):
        if not getattr(self, "_diagnostic_relaxed_timing", False):
            return
        reasons = self._home_reference.timing_warnings | ({extra} if extra else set())
        for reason in sorted(reasons-self._diagnostic_timing_warned):
            self._diagnostic_timing_warned.add(reason)
            self.get_logger().warning("DIAGNOSTIC_RELAXED_TIMING: " + reason
                                      + "; SIM ONLY, NOT safety validation")

    def _capture_then_home(self):
        return capture_then_home_for_profile(
            getattr(self, "_active_flight_profile", "NORMAL"),
            getattr(self, "_event_completion_policy", "legacy_rejoin"),
        )

    def _on_approve(self, request, response):
        with self._lock:
            proposal = self._proposals.get(request.proposal_id)
            home_generation = getattr(self, "_route_home_generation", 0)
        if proposal is None:
            response.message = "unknown proposal_id"
            return response
        if proposal.status != MissionProposal.STATUS_OK:
            response.message = "only an OK proposal can be approved"
            return response
        try:
            plan = json.loads(proposal.plan_json)
            validate_low_speed_plan(plan)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            response.message = "invalid flight profile: " + str(exc)
            return response
        captured_altitude_reference = False
        if request.approved and is_low_speed_profile(profile_for_plan(plan)):
            mission_kind = mission_kind_for_plan(plan)
            with self._lock:
                error = self._low_speed_readiness_error_locked(
                    require_ground=True,
                    mission_kind=mission_kind,
                )
                if not error:
                    reference = self._altitude_references.get(request.proposal_id)
                    try:
                        if reference is None:
                            reference = self._capture_altitude_reference_locked(
                                self._route_home)
                            self._altitude_references[request.proposal_id] = reference
                            captured_altitude_reference = True
                        reference_error, _, _, _ = (
                            self._altitude_reference_status_locked(reference))
                        if reference_error:
                            error = reference_error
                    except ValueError as exc:
                        error = str(exc)
            if error:
                response.message = "low-speed readiness failed: " + error
                return response

        self._update_route_frame_status(request.proposal_id)
        with self._lock:
            frame_result = self._route_frame_results.get(request.proposal_id)
        if request.approved and (not frame_result or frame_result[0] is None):
            if captured_altitude_reference:
                with self._lock:
                    self._altitude_references.pop(request.proposal_id, None)
            response.message = "route frame invalid: " + (
                frame_result[1] if frame_result else "status unavailable")
            return response

        if not request.approved:
            revoked = []
            with self._lock:
                for mission_id, proposal_id in list(self._approved.items()):
                    if proposal_id == request.proposal_id:
                        if (mission_id == self._active_mission_id
                                or mission_id in self._claimed_approvals):
                            self._active_cancel_requested = True
                            continue
                        revoked.append(mission_id)
                        del self._approved[mission_id]
                        self._approved_at.pop(mission_id, None)
                        self._claimed_approvals.discard(mission_id)
            for mission_id in revoked:
                self._publish_approval(
                    request.proposal_id, mission_id, False, request.operator_id
                )
            response.accepted = True
            response.message = (
                "active mission cancellation requested; terminal handling retained"
                if self._active_cancel_requested
                else "proposal approval revoked"
            )
            return response

        if not is_low_speed_profile(profile_for_plan(plan)) and self._capture_then_home():
            try:
                validate_integrated_event_plan(json.loads(proposal.plan_json))
            except (ValueError, TypeError) as exc:
                response.message = "integrated incident policy conflict: " + str(exc)
                return response

        mission_id = str(uuid.uuid4())
        with self._lock:
            if (home_generation != getattr(self, "_route_home_generation", 0)
                    or getattr(self, "_route_home_fault", "")):
                response.message = "provisional_home_changed_reprepare_required"
                return response
            self._approved[mission_id] = request.proposal_id
            self._approved_at[mission_id] = time.monotonic()
        self._publish_approval(
            request.proposal_id, mission_id, True, request.operator_id
        )
        self._publish_status(
            mission_id,
            request.proposal_id,
            MissionStatus.PHASE_AWAITING_APPROVAL,
            "approved; waiting for explicit ExecuteMission action goal",
        )
        response.accepted = True
        response.mission_id = mission_id
        response.message = "mission approved"
        return response

    def _on_manual_override(self, request, response):
        message = ManualOverride()
        message.stamp = self.get_clock().now().to_msg()
        message.active = bool(request.active)
        message.source = request.source or "operator"
        message.reason = request.reason
        self._manual_publisher.publish(message)
        with self._lock:
            self._manual_active = message.active
            active_goal = self._active_goal
            mission_id = self._active_mission_id
            proposal_id = self._active_proposal_id
            waypoint = self._active_waypoint
            total = self._active_total_waypoints
            if message.active and active_goal:
                self._mark_resume_required_locked("manual override")
        response.accepted = True
        response.message = (
            "manual override released; explicit mission.resume is required"
            if not message.active and active_goal and self._resume_required
            else "manual override updated"
        )
        if active_goal and self._resume_required:
            self._publish_status(
                mission_id,
                proposal_id,
                MissionStatus.PHASE_PAUSED_MANUAL,
                "manual control has priority; release it, then explicitly resume",
                waypoint,
                total,
            )
        return response

    def _on_emergency_land(self, request, response):
        """Atomically convert an active dashboard mission to terminal LAND.

        The service deliberately does not accept a free-standing mission id or
        create a LAND command from stale state.  Releasing app manual control
        and marking the active execution for cancellation happens under the
        same lock, before the action loop can observe the request.
        """
        with self._lock:
            state = self._vehicle_state
            active = bool(self._active_goal)
            mission_id = self._active_mission_id
            valid = bool(
                active
                and request.mission_id
                and request.mission_id == mission_id
                and 0 <= time.monotonic()-self._vehicle_state_received_at <= 0.5
                and self._can_issue_terminal_land(state)
                and bool(getattr(state, "armed", False))
                and bool(getattr(state, "emergency_land_available", False))
                and getattr(state, "active_execution_mission_id", "") == mission_id
            )
            if valid:
                if self._emergency_land_requested:
                    response.accepted = True
                    response.mission_id = mission_id
                    response.message = "emergency LAND already in progress"
                    return response
                self._emergency_land_requested = True
                self._active_cancel_requested = True
                self._manual_active = False
                self._resume_required = False
                self._resume_requested = False
        response.mission_id = mission_id
        if not valid:
            response.accepted = False
            response.message = (
                "emergency LAND rejected: active mission and fresh armed FC state required"
            )
            return response

        release = ManualOverride()
        release.stamp = self.get_clock().now().to_msg()
        release.active = False
        release.source = request.operator_id or "operator"
        release.reason = request.reason or "dashboard emergency LAND"
        self._manual_publisher.publish(release)
        self._publish_status(
            mission_id,
            self._active_proposal_id,
            MissionStatus.PHASE_LANDING,
            "operator emergency LAND requested; terminal supervisor engaged",
            self._active_waypoint,
            self._active_total_waypoints,
        )
        response.accepted = True
        response.message = "emergency LAND accepted"
        return response

    def _on_resume_mission(self, request, response):
        with self._lock:
            state = self._vehicle_state
            if getattr(self, "_event_terminal_requested", False):
                response.message = "incident ended patrol; automatic patrol resume is prohibited"
                return response
            if not self._active_goal:
                response.message = "there is no active mission to resume"
                return response
            if request.mission_id != self._active_mission_id:
                response.message = "mission_id does not match the active mission"
                return response
            if not str(request.operator_id).strip():
                response.message = "operator_id must not be empty"
                return response
            if self._manual_active:
                response.message = "manual control is still active"
                return response
            if not self._resume_required:
                response.message = "the active mission is not waiting for resume"
                return response
            if state is None or not (
                state.command_output_enabled
                and state.approved
                and state.connected
                and state.preflight_checks_pass
                and state.position_valid
                and state.armed
                and state.offboard
            ):
                response.message = "vehicle is not ready for a safe autonomous resume"
                return response
            command_spec = self._pending_resume_command
            if command_spec is None:
                response.message = "there is no saved autonomous command to resume"
                return response
            mission_id = self._active_mission_id
            proposal_id = self._active_proposal_id
            return_phase = self._resume_return_phase
            waypoint = self._active_waypoint
            total = self._active_total_waypoints

        self._republish_command_spec(command_spec)
        with self._lock:
            if not self._resume_required or mission_id != self._active_mission_id:
                response.message = "resume state changed before the command was applied"
                return response
            self._resume_requested = True
        self._publish_status(
            mission_id,
            proposal_id,
            return_phase,
            "explicit resume accepted from %s" % request.operator_id,
            waypoint,
            total,
        )
        response.accepted = True
        response.message = "mission resume accepted"
        return response

    def _on_request_event_control(self, request, response):
        if is_low_speed_profile(self._active_flight_profile):
            response.accepted = False
            response.message = "low-speed profiles disable event diversion and automatic climb"
            return response
        policy = getattr(self, "_experiment_policy", None)
        if policy is not None and not policy.events:
            response.accepted = False
            response.message = "experiment stage observes events without route interruption"
            return response
        if request.event_type == "DEMO_PERSON" and (
            not getattr(self, "_simple_obstacle_demo", False) or request.source != "demo_person"
        ):
            response.message = "DEMO_PERSON is restricted to the explicit SIM person demo"
            return response
        if not self._event_control_enabled:
            response.message = (
                "event control runtime is disabled; the ROS2 interface is ready "
                "for Jetson/SITL integration"
            )
            return response
        now = time.monotonic()
        with self._lock:
            state = self._vehicle_state
            if getattr(self, "_event_terminal_requested", False):
                response.message = "incident terminal return already requested"
                return response
            if not self._active_goal or self._active_phase != MissionStatus.PHASE_PATROL:
                response.message = "event control is only available during an active patrol"
                return response
            if self._manual_active:
                response.message = "manual control has priority"
                return response
            if state is None or not (
                state.command_output_enabled
                and state.approved
                and state.connected
                and state.preflight_checks_pass
                and state.position_valid
                and state.armed
                and state.offboard
            ):
                response.message = "vehicle is not ready for event control"
                return response
            if self._capture_then_home():
                if getattr(request, "mission_id", "") != self._active_mission_id:
                    response.message = "event request mission_id does not match the approved active mission"
                    return response
                evidence_error = evidence_deadline_error(getattr(request, "evidence_clock_id", ""),
                    getattr(request, "evidence_expires_monotonic_ns", 0))
                if evidence_error:
                    response.message = "event request evidence unavailable: " + evidence_error
                    return response
                error = self._return_navigation_error_locked(now)
                if error:
                    response.message = error
                    return response
            try:
                lease = self._event_control.request(
                    mission_id=self._active_mission_id,
                    event_id=request.event_id,
                    event_type=request.event_type,
                    track_id=request.track_id,
                    confidence=float(request.confidence),
                    source=request.source,
                    now=now,
                )
            except ValueError as exc:
                response.message = str(exc)
                return response
            mission_id = self._active_mission_id
            proposal_id = self._active_proposal_id
            waypoint = self._active_waypoint
            total = self._active_total_waypoints
            if self._capture_then_home():
                self._event_terminal_requested = True
                self._event_outbound_command_spec = self._last_command_spec
                self._pending_resume_command = None
                self._resume_required = False
                self._resume_requested = False

        self._hold_current(
            mission_id,
            requested_authority=OffboardCommand.AUTHORITY_JETSON_EVENT_CAPTURE,
        )
        self._publish_status(
            mission_id,
            proposal_id,
            MissionStatus.PHASE_PAUSED_EVENT,
            "event control requested; holding position before capture",
            waypoint,
            total,
        )
        response.accepted = True
        response.lease_id = lease.lease_id
        response.mission_id = mission_id
        response.expires_at = (
            self.get_clock().now()
            + Duration(seconds=self._event_control.max_duration_s)
        ).to_msg()
        response.message = "event control accepted; wait for EVENT_CAPTURE status"
        return response

    def _on_release_event_control(self, request, response):
        with self._lock:
            try:
                # A late successful writer callback cannot turn an expired
                # lease into a successful capture.
                self._event_control.expire(time.monotonic())
                lease = self._event_control.request_release(
                    lease_id=request.lease_id,
                    source=request.source,
                    capture_succeeded=bool(request.capture_succeeded),
                    reason=request.detail,
                )
            except ValueError as exc:
                response.message = str(exc)
                return response
        response.accepted = True
        response.message = (
            "event control release accepted; patrol is terminal; return over visited path"
            if self._capture_then_home() else
            "event control release accepted; route rejoin will begin from the nearest route segment"
        )
        self.get_logger().info("event lease %s release requested" % lease.lease_id)
        return response

    def _on_goal(self, goal_request) -> GoalResponse:
        if getattr(self, "_cleanup_fault", ""):
            return GoalResponse.REJECT
        reservation_created = False
        with self._lock:
            approved_proposal = self._approved.get(goal_request.mission_id)
            busy = self._active_goal or self._goal_reservation is not None
            proposal = self._proposals.get(goal_request.proposal_id)
            home_generation = getattr(self, "_route_home_generation", 0)
        if busy or approved_proposal != goal_request.proposal_id:
            return GoalResponse.REJECT
        with self._lock:
            # Reserve before any callback-visible validation so two Reentrant
            # Action goal callbacks cannot both pass the busy check.
            if self._active_goal or self._goal_reservation is not None:
                return GoalResponse.REJECT
            if (self._approved.get(goal_request.mission_id) != goal_request.proposal_id
                    or home_generation != getattr(self, "_route_home_generation", 0)
                    or getattr(self, "_route_home_fault", "")):
                return GoalResponse.REJECT
            self._goal_reservation = {
                "mission_id": goal_request.mission_id,
                "proposal_id": goal_request.proposal_id,
                "created_at": time.monotonic(),
                "frame_result": None,
            }
            self._active_cancel_requested = False
            self._emergency_land_requested = False
            reservation_created = True
        try:
            with self._lock:
                frame_result = self._route_frame_results.get(goal_request.proposal_id)
            if not frame_result or frame_result[0] is None:
                raise ValueError("route frame unavailable")
            goal_plan = json.loads(proposal.plan_json) if proposal is not None else None
            if not isinstance(goal_plan, dict):
                raise ValueError("approved proposal is unavailable")
            if not str(getattr(goal_request, "plan_json", "")).strip():
                raise ValueError("action plan_json is required for every profile")
            if not self._same_json(goal_request.plan_json, proposal.plan_json):
                raise ValueError("action plan differs from approved proposal")
            validate_low_speed_plan(goal_plan)
            if getattr(self, "_scenario_enabled", False):
                validate_spec(goal_plan)
            elif "test_scenario" in goal_plan:
                raise ValueError("scenario requires dedicated launch configuration")
            if ((goal_plan is None
                 or not is_low_speed_profile(profile_for_plan(goal_plan)))
                    and self._capture_then_home()):
                if proposal is None or goal_plan is None:
                    raise ValueError("approved proposal is unavailable")
                validate_integrated_event_plan(goal_plan)
            flight_profile = profile_for_plan(goal_plan)
            mission_kind = mission_kind_for_plan(goal_plan)
            points = (() if mission_kind == FORWARD_TEST_1M else
                      self._catalog.points_for_plan(
                          proposal.plan_json,
                          goal_request.waypoints_enu
                          if self._allow_goal_waypoints else (),
                      ))
            limit_type, limit_value = (
                ("LAPS", 1) if mission_kind == FORWARD_TEST_1M
                else self._catalog.patrol_limit(proposal.plan_json)
            )
            preview = None
            altitude_reference = None
            if mission_kind == FORWARD_TEST_1M:
                with self._lock:
                    preview = self._forward_previews.get(goal_request.proposal_id)
                if preview is None:
                    raise ValueError("forward-test preview is unavailable")
                with self._lock:
                    state = self._vehicle_state
                valid, preview_reason = preview_valid(
                    preview, state.position_ned_m, state.heading_rad)
                if not valid:
                    raise ValueError(preview_reason)
            if is_low_speed_profile(flight_profile):
                with self._lock:
                    altitude_reference = self._altitude_references.get(
                        goal_request.proposal_id)
                    reference_error, _, _, _ = (
                        self._altitude_reference_status_locked(
                            altitude_reference))
                if reference_error:
                    raise ValueError(reference_error)
            frame = frame_result[0]
            snapshot = ExecutionSnapshot(
                home_generation=home_generation,
                proposal_id=goal_request.proposal_id,
                plan_json=proposal.plan_json,
                canonical_plan_json=json.dumps(
                    goal_plan, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False, allow_nan=False),
                mission_kind=mission_kind,
                flight_profile=flight_profile,
                frame_result=frame,
                points=tuple(points),
                route_ned=tuple(tuple(float(v) for v in point)
                                for point in frame.route_ned),
                waypoint_yaws_ned=tuple(
                    yaw_enu_deg_to_ned_rad(point.yaw_enu_deg) for point in points),
                limit_type=limit_type,
                limit_value=int(limit_value),
                raw_route_home=(altitude_reference["raw_home"]
                                if altitude_reference is not None
                                else frame.home),
                aligned_home_z_ned_m=(
                    float(altitude_reference["aligned_home_z_ned_m"])
                    if altitude_reference is not None else None),
                capture_fc_altitude_home_relative_m=(
                    float(altitude_reference[
                        "capture_fc_altitude_home_relative_m"])
                    if altitude_reference is not None else None),
                altitude_reference_epoch=(
                    int(altitude_reference["transport_epoch"])
                    if altitude_reference is not None else 0),
                altitude_reference_sequence=(
                    int(altitude_reference["sequence"])
                    if altitude_reference is not None else 0),
                altitude_reference_local_time_boot_ms=(
                    int(altitude_reference["local_time_boot_ms"])
                    if altitude_reference is not None else 0),
                altitude_reference_global_time_boot_ms=(
                    int(altitude_reference["global_time_boot_ms"])
                    if altitude_reference is not None else 0),
                preview_start_ned=(tuple(preview["start_ned"])
                                   if preview is not None else None),
                preview_heading_rad=(float(preview["heading_rad"])
                                     if preview is not None else None),
                preview_end_ned=(tuple(preview["end_ned"])
                                 if preview is not None else None),
            )
            with self._lock:
                if (home_generation != getattr(self, "_route_home_generation", 0)
                        or getattr(self, "_route_home_fault", "")):
                    raise ValueError("Home changed during goal validation")
                self._claimed_approvals.add(goal_request.mission_id)
                reservation = self._goal_reservation
                if (reservation is None
                        or reservation.get("mission_id") != goal_request.mission_id
                        or reservation.get("proposal_id") != goal_request.proposal_id):
                    raise ValueError("execution reservation expired during validation")
                reservation["frame_result"] = frame
                reservation["plan_json"] = proposal.plan_json
                reservation["snapshot"] = snapshot
                if mission_kind == FORWARD_TEST_1M:
                    self._consumed_preview_states[goal_request.proposal_id] = "CONSUMED"
                    self._forward_previews.pop(goal_request.proposal_id, None)
            return GoalResponse.ACCEPT
        except Exception as exc:
            self.get_logger().warning("ExecuteMission goal rejected: " + str(exc))
            with self._lock:
                if (reservation_created and self._goal_reservation is not None
                        and self._goal_reservation.get("mission_id") == goal_request.mission_id):
                    self._goal_reservation = None
                self._claimed_approvals.discard(goal_request.mission_id)
            return GoalResponse.REJECT

    def _on_cancel(self, _goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle):
        """Catch every ordinary execution failure before any mutable access."""
        result = ExecuteMission.Result()
        try:
            return self._execute_impl(goal_handle, result)
        except Exception as exc:
            request = goal_handle.request
            try:
                with self._lock:
                    reservation = self._goal_reservation
                    snapshot = (reservation.get("snapshot")
                                if reservation is not None else None)
                    if snapshot is not None:
                        self._active_flight_profile = snapshot.flight_profile
                        self._active_mission_kind = snapshot.mission_kind
            except Exception:
                pass
            try:
                return self._execute_exception_result(goal_handle, result, exc)
            finally:
                self._cleanup_execution(request)

    def _execute_impl(self, goal_handle, result):
        request = goal_handle.request
        with self._lock:
            reservation = self._goal_reservation
            if (reservation is None
                    or reservation.get("mission_id") != request.mission_id
                    or reservation.get("proposal_id") != request.proposal_id):
                reservation = None
            snapshot = (reservation.get("snapshot")
                        if reservation is not None else None)
            self._active_goal = True
            self._active_ever_armed = False
        try:
            if reservation is None or snapshot is None:
                return self._abort_result(goal_handle, result, "execution reservation unavailable")
            if (not request.plan_json
                    or json.dumps(json.loads(request.plan_json), sort_keys=True,
                                  separators=(",", ":"), ensure_ascii=False,
                                  allow_nan=False) != snapshot.canonical_plan_json):
                return self._abort_result(
                    goal_handle, result, "action plan differs from approved proposal"
                )
            if request.waypoints_enu and not self._allow_goal_waypoints:
                return self._abort_result(
                    goal_handle,
                    result,
                    "goal-supplied waypoints are disabled; use the approved zone catalog",
                )

            plan_json = snapshot.plan_json
            plan = json.loads(plan_json)
            flight_profile = snapshot.flight_profile
            mission_kind = snapshot.mission_kind
            points = snapshot.points
            with self._lock:
                state_for_frame = self._vehicle_state
            if state_for_frame is None:
                raise ValueError("vehicle state unavailable for route conversion")
            if is_low_speed_profile(flight_profile):
                with self._lock:
                    readiness_error = self._low_speed_readiness_error_locked(
                        require_ground=True, allow_active=True,
                        mission_kind=mission_kind,
                    )
                if readiness_error:
                    raise ValueError("low-speed readiness failed: " + readiness_error)
            # Geometry was validated and frozen atomically at goal claim.
            # Never reinterpret a preview from the vehicle's moving position.
            frame_result = snapshot.frame_result
            route_ned = snapshot.route_ned
            with self._lock:
                if (self._route_home_fault
                        or snapshot.home_generation != self._route_home_generation):
                    raise ValueError(self._route_home_fault or
                                     "provisional_home_changed_reprepare_required")
                self._active_mission_id = request.mission_id
                self._active_proposal_id = request.proposal_id
                self._active_route_ned = route_ned
                self._active_flight_profile = flight_profile
                self._active_mission_kind = mission_kind
                self._active_home_z_ned_m = float(
                    snapshot.aligned_home_z_ned_m
                    if snapshot.aligned_home_z_ned_m is not None
                    else frame_result.home.ned[2] if frame_result.home else 0.0)
                self._active_altitude_reference = (
                    {
                        "home_generation": snapshot.home_generation,
                        "aligned_home_z_ned_m": snapshot.aligned_home_z_ned_m,
                        "capture_fc_altitude_home_relative_m": (
                            snapshot.capture_fc_altitude_home_relative_m),
                        "transport_epoch": snapshot.altitude_reference_epoch,
                        "sequence": snapshot.altitude_reference_sequence,
                        "local_time_boot_ms": (
                            snapshot.altitude_reference_local_time_boot_ms),
                        "global_time_boot_ms": (
                            snapshot.altitude_reference_global_time_boot_ms),
                        "raw_home": snapshot.raw_route_home,
                    }
                    if snapshot.aligned_home_z_ned_m is not None else None)
                self._active_route_home = snapshot.raw_route_home
                self._active_frame_continuity_error = ""
            limit_type, limit_value = snapshot.limit_type, snapshot.limit_value
            if limit_type == "LAPS" and limit_value > self._max_laps:
                return self._abort_result(goal_handle, result, "lap limit exceeds safety cap")
            if (
                limit_type == "DURATION_MINUTES"
                and limit_value > self._max_duration_minutes
            ):
                return self._abort_result(
                    goal_handle, result, "duration exceeds safety cap"
                )

            if getattr(self, "_owned_sim_preparation", None) is not None:
                preparation = self._run_owned_sim_preparation(goal_handle, len(points))
                if preparation != "ok":
                    return self._finish_owned_sim_failure(goal_handle, result, preparation)

            self._last_vehicle_ready_reason = "vehicle_state_missing"
            ready = self._wait_for(
                goal_handle,
                self._vehicle_ready,
                self._ready_timeout,
                MissionStatus.PHASE_TAKEOFF,
                0,
                len(points),
                "waiting for enabled PX4 SITL and preflight readiness",
                readiness_diagnostics=True,
            )
            if ready != "ok":
                detail = "vehicle not ready"
                if ready == "timeout":
                    detail += "; readiness=" + self._last_vehicle_ready_reason
                return self._finish_wait_failure(goal_handle, result, ready, detail)

            if getattr(self, "_readiness_only_checkpoint", False):
                return self._finish_readiness_checkpoint(goal_handle, result, True, "ready")

            state = self._snapshot_state()
            takeoff_altitude = (target_altitude_for_profile(flight_profile)
                                if is_low_speed_profile(flight_profile)
                                else self._takeoff_altitude(request.takeoff_altitude_m))
            if not 1.0 <= takeoff_altitude <= 30.0:
                return self._abort_result(
                    goal_handle, result, "takeoff altitude must be between 1 and 30 m"
                )
            if self._capture_then_home():
                with self._lock:
                    home = self._current_home_reference_locked(time.monotonic())
                    if home is None:
                        raise ValueError("QGC-armed same-epoch static PX4 Home reference is required")
                    self._mission_home = tuple(home)
                    self._home_reference_invalidated = False
                    self._trace_min_height_m = max(1., takeoff_altitude-.5)
                    self._return_trace = VisitedReturnTrace(self._mission_home, self._return_trace_policy)
            route_takeoff_home = (
                (frame_result.home.ned[0], frame_result.home.ned[1],
                 self._active_home_z_ned_m)
                if frame_result.home else (0.0, 0.0, 0.0))
            takeoff_target = (
                float(state.position_ned_m[0]),
                float(state.position_ned_m[1]),
                (self._mission_home[2]-takeoff_altitude if self._capture_then_home()
                 else route_takeoff_home[2]-takeoff_altitude),
            )
            self._publish_command(
                request.mission_id,
                OffboardCommand.COMMAND_TAKEOFF,
                takeoff_target,
                (float(state.heading_rad) if is_low_speed_profile(flight_profile) else 0.0),
            )
            self._publish_status(
                request.mission_id,
                request.proposal_id,
                MissionStatus.PHASE_TAKEOFF,
                "approved takeoff in progress",
                0,
                len(points),
            )
            self._low_settle_since = None
            airborne = self._wait_for(
                goal_handle,
                lambda value: (self._low_target_reached(value, request.mission_id, takeoff_target)
                               if is_low_speed_profile(flight_profile)
                               else self._takeoff_target_reached(value, takeoff_target)),
                self._takeoff_timeout,
                MissionStatus.PHASE_TAKEOFF,
                0,
                len(points),
                "taking off",
            )
            if airborne != "ok":
                return self._finish_wait_failure(goal_handle, result, airborne, "takeoff failed")

            if "test_scenario" in json.loads(snapshot.plan_json):
                return execute_scenario(self, goal_handle, result, snapshot)

            if mission_kind == FORWARD_TEST_1M:
                target_ned = route_ned[0]
                self._low_settle_since = None
                self._publish_command(request.mission_id, OffboardCommand.COMMAND_GOTO,
                                      target_ned, snapshot.preview_heading_rad)
                moved = self._wait_for(
                    goal_handle,
                    lambda value: self._low_target_reached(value, request.mission_id, target_ned),
                    self._waypoint_timeout, MissionStatus.PHASE_FORWARD_TEST, 0, 1,
                    "moving 2 m in the frozen heading")
                if moved != "ok":
                    return self._finish_wait_failure(goal_handle, result, moved,
                                                     "forward-test movement failed")
                self._publish_command(request.mission_id, OffboardCommand.COMMAND_LAND,
                                      target_ned, snapshot.preview_heading_rad)
                self._publish_status(request.mission_id, request.proposal_id,
                                     MissionStatus.PHASE_LANDING,
                                     "2 m target reached; landing at endpoint", 1, 1)
                terminal_outcome = self._complete_low_speed_terminal(
                    goal_handle, command_already_sent=True)
                if terminal_outcome != "ACCEPTED":
                    return self._finish_wait_failure(
                        goal_handle, result, "timeout",
                        "forward-test landing failed: " + terminal_outcome)
                goal_handle.succeed()
                result.success = True
                result.final_phase = MissionStatus.PHASE_COMPLETED
                result.message = "2 m forward low-speed test completed and disarmed"
                self._publish_status(request.mission_id, request.proposal_id,
                                     MissionStatus.PHASE_COMPLETED, result.message, 1, 1)
                return result

            start = points[0]
            start_target_ned = route_ned[0]
            self._publish_command(
                request.mission_id,
                OffboardCommand.COMMAND_GOTO,
                start_target_ned,
                snapshot.waypoint_yaws_ned[0],
            )
            at_start = self._wait_for(
                goal_handle,
                lambda value: (self._low_target_reached(value, request.mission_id, start_target_ned)
                               if is_low_speed_profile(flight_profile) else
                               self._route_target_reached(value, request.mission_id, start_target_ned)),
                self._waypoint_timeout,
                MissionStatus.PHASE_TRANSIT_TO_START,
                0,
                0,
                "moving to route start before patrol",
            )
            if at_start != "ok":
                return self._finish_wait_failure(
                    goal_handle, result, at_start, "route start timeout"
                )

            total_targets = (
                patrol_target_count(len(points), limit_value)
                if limit_type == "LAPS"
                else len(points)
            )
            self._publish_status(
                request.mission_id,
                request.proposal_id,
                MissionStatus.PHASE_PATROL,
                "route start reached; patrol started",
                0,
                total_targets,
            )
            patrol_started = time.monotonic()
            waypoint_indices = patrol_waypoint_indices(len(points))
            patrol_step = 0
            lap = 0
            while True:
                if limit_type == "LAPS" and lap >= limit_value:
                    break
                if (
                    limit_type == "DURATION_MINUTES"
                    and time.monotonic() - patrol_started >= limit_value * 60.0
                ):
                    break
                waypoint_index = next(waypoint_indices)
                point = points[waypoint_index]
                target_ned = route_ned[waypoint_index]
                yaw_ned = snapshot.waypoint_yaws_ned[waypoint_index]
                self._publish_command(
                    request.mission_id,
                    OffboardCommand.COMMAND_GOTO,
                    target_ned,
                    yaw_ned,
                )
                reached = self._wait_for(
                    goal_handle,
                    lambda value, target=target_ned: (
                        self._low_target_reached(value, request.mission_id, target)
                        if is_low_speed_profile(flight_profile) else
                        self._route_target_reached(value, request.mission_id, target)),
                    self._waypoint_timeout,
                    MissionStatus.PHASE_PATROL,
                    patrol_step,
                    total_targets,
                    "flying to patrol waypoint",
                    route_ned=route_ned,
                    target_yaw_rad=yaw_ned,
                )
                if reached != "ok":
                    if reached in {"event_home_completed", "event_home_capture_failed"}:
                        return self._event_home_result(goal_handle, result, reached)
                    return self._finish_wait_failure(
                        goal_handle, result, reached, "waypoint timeout"
                    )
                patrol_step += 1
                self._publish_status(
                    request.mission_id,
                    request.proposal_id,
                    MissionStatus.PHASE_PATROL,
                    "patrol waypoint reached",
                    patrol_step,
                    total_targets,
                )
                if waypoint_index == len(points) - 1:
                    lap += 1

            observed = self._observe_simple_demo_endpoint(
                goal_handle, route_ned, yaw_ned, patrol_step, total_targets)
            if observed != "ok":
                if observed in {"event_home_completed", "event_home_capture_failed"}:
                    return self._event_home_result(goal_handle, result, observed)
                return self._finish_wait_failure(
                    goal_handle, result, observed, "incident endpoint observation interrupted")

            if self._capture_then_home():
                with self._lock:
                    self._event_terminal_requested = True
                    self._event_outbound_command_spec = self._last_command_spec
                    self._event_capture_succeeded = True  # no incident recording was requested
                returned = self._return_over_visited_path(goal_handle)
                if returned == "event_home_completed":
                    return self._event_home_result(goal_handle, result, returned,
                                                   detail="patrol completed; visited-path Home landing confirmed")
                return self._finish_wait_failure(goal_handle, result, returned, "visited-path return failed")
            terminal_command = (OffboardCommand.COMMAND_LAND
                                if is_low_speed_profile(flight_profile)
                                else OffboardCommand.COMMAND_RTL)
            terminal_target = route_ned[-1] if is_low_speed_profile(flight_profile) else (0.0, 0.0, 0.0)
            self._publish_command(request.mission_id, terminal_command,
                                  terminal_target, math.nan)
            self._publish_status(
                request.mission_id,
                request.proposal_id,
                (MissionStatus.PHASE_LANDING if is_low_speed_profile(flight_profile)
                 else MissionStatus.PHASE_RETURNING_HOME),
                ("low-speed route complete; landing at final waypoint"
                 if is_low_speed_profile(flight_profile) else "patrol complete; returning home"),
            )
            if is_low_speed_profile(flight_profile):
                terminal_outcome = self._complete_low_speed_terminal(
                    goal_handle, command_already_sent=True)
                if terminal_outcome != "ACCEPTED":
                    return self._finish_wait_failure(
                        goal_handle, result, "timeout",
                        "landing failed: " + terminal_outcome)
            else:
                landed = self._wait_for(
                    goal_handle, lambda value: value.landed or not value.armed,
                    self._landing_timeout, MissionStatus.PHASE_RETURNING_HOME,
                    0, 0, "returning home")
                if landed != "ok":
                    return self._finish_wait_failure(
                        goal_handle, result, landed, "landing timeout")

            if getattr(self, "_owned_sim_preparation", None) is not None:
                cleanup = self._close_owned_sim_preparation(request)
                if cleanup.get('stage') != 'closed' or cleanup.get('cleanup_confirmed') is not True:
                    self._owned_sim_failure_reason = 'landing complete but owned helper cleanup unconfirmed'
                    return self._finish_owned_sim_failure(goal_handle, result, 'failed')
            goal_handle.succeed()
            result.success = True
            result.final_phase = MissionStatus.PHASE_COMPLETED
            result.message = "mission completed"
            self._publish_status(
                request.mission_id,
                request.proposal_id,
                MissionStatus.PHASE_COMPLETED,
                result.message,
            )
            return result
        except Exception as exc:
            if getattr(self, "_owned_sim_preparation", None) is not None:
                self._owned_sim_failure_reason = str(exc)
                try:
                    return self._finish_owned_sim_failure(goal_handle, result, 'failed')
                except Exception as cleanup_exc:
                    exc = RuntimeError("%s; owned cleanup failed: %s" % (exc, cleanup_exc))
            return self._execute_exception_result(goal_handle, result, exc)
        finally:
            self._cleanup_execution(request)

    def _cleanup_execution(self, request):
        """Idempotently release an execution only after terminal handling."""
        if getattr(self, "_owned_sim_preparation", None) is not None:
            try:
                self._close_owned_sim_preparation(request)
            except Exception as exc:
                self.get_logger().error("owned simulation cleanup failed: %s" % exc)
        with self._lock:
            self._active_goal = False
            self._terminal_deadlines.pop(request.mission_id, None)
            self._active_cancel_requested = False
            self._emergency_land_requested = False
            self._active_ever_armed = False
            self._event_control.clear_for_mission(request.mission_id)
            if hasattr(self, "_scenario_inputs"):
                self._scenario_inputs.event = None
                self._scenario_inputs.frame = None
                self._scenario_inputs.evidence = None
            self._active_mission_id = ""
            self._active_proposal_id = ""
            self._active_phase = MissionStatus.PHASE_IDLE
            self._active_route_ned = ()
            self._active_waypoint = 0
            self._active_total_waypoints = 0
            self._active_flight_profile = "NORMAL"
            self._active_mission_kind = ROUTE
            self._active_home_z_ned_m = 0.0
            self._active_altitude_reference = None
            self._active_route_home = None
            self._active_frame_continuity_error = ""
            self._low_settle_since = None
            self._last_command_spec = None
            self._pending_resume_command = None
            self._resume_required = False
            self._resume_requested = False
            self._resume_reason = ""
            self._resume_return_phase = MissionStatus.PHASE_IDLE
            self._mission_home = None
            self._return_trace = None
            self._event_terminal_requested = False
            self._event_return_in_progress = False
            self._event_terminal_failure_reason = ""
            self._event_capture_succeeded = False
            self._event_outbound_command_spec = None
            consumed_proposal = self._approved.pop(request.mission_id, None)
            self._approved_at.pop(request.mission_id, None)
            self._claimed_approvals.discard(request.mission_id)
            if (self._goal_reservation is not None
                    and self._goal_reservation.get("mission_id") == request.mission_id):
                self._goal_reservation = None
            self._forward_previews.pop(request.proposal_id, None)
            if request.proposal_id in self._consumed_preview_states:
                self._consumed_preview_states[request.proposal_id] = "COMPLETED"
        if consumed_proposal is not None:
            try:
                self._publish_approval(
                    consumed_proposal, request.mission_id, False, "mission-manager")
            except Exception as exc:
                self._cleanup_fault = "mission completion publication failed; restart required: " + str(exc)
                self.get_logger().error(
                    "final approval revocation publication failed: %s" % exc)

    def _run_owned_sim_preparation(self, goal_handle, total_waypoints):
        """Claimed action -> native baseline -> original Home -> route handoff.

        No mission command is issued while this hook owns preparation. The
        10-second UI ledger was already consumed by action acceptance; it is
        not extended, renewed, or replaced by a synthetic approval here.
        """
        adapter = self._owned_sim_preparation
        request = goal_handle.request
        self._owned_sim_preparing = True
        self._owned_sim_handoff_complete = False
        self._owned_sim_failure_reason = ""
        try:
            with self._lock:
                if self._approved.get(request.mission_id) != request.proposal_id:
                    raise ValueError('actual accepted mission approval is missing')
            if goal_handle.is_cancel_requested:
                return 'cancel'
            adapter.start(request.mission_id, request.proposal_id)
            deadline = time.monotonic() + 180.0
            last_stage = None
            while time.monotonic() < deadline:
                if goal_handle.is_cancel_requested:
                    return 'cancel'
                with self._lock:
                    if self._approved.get(request.mission_id) != request.proposal_id or self._manual_active:
                        raise ValueError('approval revoked or manual priority during preparation')
                    state = self._vehicle_state
                status = adapter.status()
                if status.get('error'):
                    raise ValueError(status['error'])
                stage = status.get('stage', 'starting')
                if stage != last_stage:
                    detail = 'SIM preparation: ' + stage + '; no route output before handoff'
                    self._publish_status(request.mission_id, request.proposal_id,
                        MissionStatus.PHASE_TAKEOFF, detail, 0, total_waypoints)
                    self._publish_feedback(goal_handle, MissionStatus.PHASE_TAKEOFF, 0,
                        total_waypoints, detail)
                    last_stage = stage
                if (stage == 'perception_ready' and self._vehicle_ready(state)
                        and 0 <= time.monotonic()-status.get('monotonic_s', float('-inf')) <= .5):
                    with self._lock:
                        home = self._current_home_reference_locked(time.monotonic())
                        if home is None:
                            raise ValueError('native same-epoch Home is unavailable at handoff')
                    adapter.handoff()
                    self._owned_sim_handoff_complete = True
                    self._owned_sim_preparing = False
                    return 'ok'
                time.sleep(.05)
            raise ValueError('owned SIM preparation exceeded its 180-second bound')
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            self._owned_sim_failure_reason = str(exc)
            return 'failed'

    def _close_owned_sim_preparation(self, request):
        adapter = getattr(self, '_owned_sim_preparation', None)
        if adapter is None:
            return {'stage': 'not_started', 'cleanup_confirmed': True}
        # Revoke before touching the heartbeat helper. Native RTL cleanup must
        # not compete with this mission's controller stream.
        self._owned_sim_preparing = True
        with self._lock:
            approved = self._approved.pop(request.mission_id, None)
        if approved is not None:
            self._publish_approval(request.proposal_id, request.mission_id, False, 'mission-manager')
        try:
            return adapter.close()
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            return {'stage': 'cleanup_unconfirmed', 'error': str(exc), 'cleanup_confirmed': False}

    def _finish_owned_sim_failure(self, goal_handle, result, reason):
        # Dedicated failure path: never falls into legacy HOLD/GOTO/RTL while
        # the canonical helper owns baseline or cleanup.
        self._publish_status(goal_handle.request.mission_id, goal_handle.request.proposal_id,
            MissionStatus.PHASE_HOLD_MANUAL_REQUIRED,
            'SIM preparation/flight ' + reason + ': ' + self._owned_sim_failure_reason
            + '; native cleanup in progress; no route output')
        cleanup = self._close_owned_sim_preparation(goal_handle.request)
        confirmed = cleanup.get('cleanup_confirmed') is True or cleanup.get('stage') == 'not_started'
        result.success = False
        result.final_phase = MissionStatus.PHASE_ABORTED if reason == 'cancel' else MissionStatus.PHASE_HOLD_MANUAL_REQUIRED
        result.message = ('SIM preparation/flight ' + reason + ': ' + self._owned_sim_failure_reason
            + ('; native cleanup confirmed' if confirmed else '; native cleanup NOT confirmed'))
        if goal_handle.is_active:
            if reason == 'cancel':
                goal_handle.canceled()
            else:
                goal_handle.abort()
        self._publish_status(goal_handle.request.mission_id, goal_handle.request.proposal_id,
            result.final_phase, result.message)
        return result

    @staticmethod
    def _validate_default_takeoff_altitude(value, *, simulation_only,
            allow_real_hardware, use_sim_time, diagnostic_relaxed_timing):
        if type(value) not in (int, float) or not math.isfinite(value) or not 1.0 <= value <= 30.0:
            raise ValueError("default_takeoff_altitude_m must be finite and between 1 and 30 m")
        if value != 5.0 and not (simulation_only is True and allow_real_hardware is False
                and use_sim_time is True and diagnostic_relaxed_timing is True):
            raise ValueError("non-default takeoff altitude requires the explicit SIM diagnostic profile")

    def _takeoff_altitude(self, requested_altitude):
        default = getattr(self, "_default_takeoff_altitude_m", 5.0)
        # Preserve the old small-obstacle demo's 1 m default unless overridden.
        if default == 5.0 and getattr(self, "_simple_obstacle_demo", False):
            default = 1.0
        return float(requested_altitude or default)

    def _takeoff_target_reached(self, state: VehicleControlState, target_ned) -> bool:
        if not (state.armed and state.offboard):
            return False
        if getattr(self, "_default_takeoff_altitude_m", 5.0) < 5.0:
            # Explicit low-altitude SIM: 3 m hover -> 1 m target is DESCENT.
            # The legacy one-sided test and 2 m waypoint radius pass too early.
            # This is altitude proximity only, not a velocity/settling proof.
            return abs(state.position_ned_m[2] - target_ned[2]) <= 0.25
        return state.position_ned_m[2] <= target_ned[2] + 1.0

    def _route_target_reached(self, state: VehicleControlState, mission_id: str, target_ned) -> bool:
        # Proximity during safety takeover/recovery cannot consume a route leg.
        # The controller retains the original GOTO; resume this same target only
        # after it reports route ownership again. No new distance/time policy.
        return (
            state.mission_id == mission_id
            and state.approved
            and not state.manual_override
            and getattr(state, "avoidance_active", None) is False
            and state.active_authority == "LLM_ROUTE"
            and distance_ned(state.position_ned_m, target_ned) <= self._acceptance_radius
        )

    def _low_target_reached(self, state: VehicleControlState, mission_id: str, target_ned) -> bool:
        owned = (state.mission_id == mission_id and state.approved
                 and not state.manual_override and state.active_authority == "LLM_ROUTE")
        speed_values = tuple(float(v) for v in getattr(state, "velocity_ned_m_s", (math.nan,)*3))
        speed = math.sqrt(sum(v*v for v in speed_values)) if all(math.isfinite(v) for v in speed_values) else math.inf
        position_error = distance_ned(state.position_ned_m, target_ned)
        altitude_error = abs(float(state.position_ned_m[2])-float(target_ned[2]))
        stable = (owned and getattr(state, "last_error", "") != "altitude_recovery_wait"
                  and getattr(state, "avoidance_active", False) is False
                  and position_error <= ACCEPTANCE_RADIUS_M
                  and altitude_error <= SETTLE_ALTITUDE_ERROR_M
                  and speed <= SETTLE_SPEED_M_S)
        if getattr(self, '_scenario_enabled', False):
            from .scenario_contract import within
            spec = self._scenario_registry.get(mission_id)
            if spec is not None:
                stable = stable and within(spec, state.position_ned_m)
        now = time.monotonic()
        if not stable:
            self._low_settle_since = None
            return False
        if self._low_settle_since is None:
            self._low_settle_since = now
            return False
        return now-self._low_settle_since >= SETTLE_TIME_S

    def _vehicle_ready(self, state: VehicleControlState) -> bool:
        ready, reason = self._evaluate_vehicle_readiness(state)
        self._last_vehicle_ready_reason = reason
        return ready

    @staticmethod
    def _validate_diagnostic_relaxed_timing_profile(*, enabled, simulation_only,
            allow_real_hardware, use_sim_time, source_mode):
        if type(enabled) is not bool:
            raise ValueError("diagnostic_relaxed_timing must be a bool")
        if enabled and not (simulation_only is True and allow_real_hardware is False
                and use_sim_time is True and source_mode == "single_publisher_sitl"):
            raise ValueError("diagnostic_relaxed_timing requires the explicit owned SIM clock/Home profile")

    @staticmethod
    def _validate_readiness_checkpoint_profile(*, enabled, simulation_only,
            allow_real_hardware, use_sim_time, source_mode, completion_policy):
        if type(enabled) is not bool:
            raise ValueError("readiness_only_checkpoint must be a bool")
        if enabled and not (simulation_only is True and allow_real_hardware is False
                and use_sim_time is True and source_mode == "single_publisher_sitl"
                and completion_policy == "capture_then_rtl"):
            raise ValueError("readiness_only_checkpoint requires the explicit owned SIM clock/Home profile")

    def _finish_readiness_checkpoint(self, goal_handle, result, passed: bool, detail: str):
        if goal_handle.is_active:
            if passed:
                goal_handle.succeed()
            else:
                goal_handle.abort()
        # The action transaction can finish, but this is never a mission/flight
        # success. Consumers must match the explicit checkpoint marker.
        result.success = False
        result.final_phase = MissionStatus.PHASE_IDLE if passed else MissionStatus.PHASE_ERROR
        marker = "READINESS_ONLY_PASSED" if passed else "READINESS_ONLY_FAILED"
        bounded_detail = str(detail).replace("\r", " ").replace("\n", " ")[:320]
        result.message = marker + ": " + bounded_detail + "; no Offboard commands; NOT flight validation"
        self._publish_status(goal_handle.request.mission_id, goal_handle.request.proposal_id,
                             result.final_phase, result.message)
        return result

    def _evaluate_vehicle_readiness(self, state: VehicleControlState) -> tuple[bool, str]:
        """The original readiness predicate plus bounded, read-only diagnostics."""
        if state is None:
            return False, "vehicle_state_missing"
        if self._capture_then_home():
            with self._lock:
                now = time.monotonic()
                if self._current_home_reference_locked(now) is None:
                    return False, self._home_not_ready_reason_locked(now)
                if not 0 <= time.monotonic()-self._vehicle_state_received_at <= .5:
                    return False, "vehicle_state_missing_or_stale"
                if getattr(state, "route_heading_gate_enabled", None) is not True:
                    return False, "route_heading_gate_disabled_or_unknown"
                if getattr(state, "terminal_owned_handoff_enabled", None) is not True:
                    return False, "terminal_owned_handoff_disabled_or_unknown"
        for name, reason in (
            ("command_output_enabled", "command_output_disabled"),
            ("approved", "approval_missing"),
            ("connected", "vehicle_disconnected"),
            ("preflight_checks_pass", "preflight_checks_not_passed"),
            ("position_valid", "position_invalid"),
        ):
            if not getattr(state, name):
                return False, reason
        if state.manual_override:
            return False, "manual_override_active"
        return True, "ready"

    def _home_not_ready_reason_locked(self, now: float) -> str:
        # Explanation only: _current_home_reference_locked remains the sole
        # Home readiness authority. Never reset/rebaseline its latched fault.
        reference = self._home_reference
        if reference.failure:
            fault = str(reference.failure).replace("\r", " ").replace("\n", " ")[:160]
            return "home_latched: " + fault
        if self._event_return_source_mode != "single_publisher_sitl":
            return "home_source_mode_disabled"
        if not self._home_graph_timing_valid_locked(now):
            return "home_publisher_graph_missing_or_stale"
        if reference.home is None:
            return "home_sample_missing"
        if reference.status_samples < 2:
            return "home_status_initialization_incomplete"
        if not reference.armed:
            return "home_armed_epoch_unavailable"
        if not math.isfinite(now) or not reference.status_age_valid(now):
            return "home_status_missing_or_stale"
        if (self._home_prefix is None or self._status_prefix is None
                or self._home_prefix != reference.home_prefix
                or self._status_prefix != reference.status_prefix
                or self._home_prefix != self._status_prefix):
            return "home_publisher_identity_unavailable_or_mismatched"
        if reference.home_stamp > reference.status_stamp:
            return "home_newer_than_current_status"
        return "home_unavailable"

    def _execution_interrupted(self, goal_handle):
        with self._lock:
            return bool(goal_handle.is_cancel_requested
                or getattr(self, "_active_cancel_requested", False)
                or getattr(self, "_emergency_land_requested", False))

    def _wait_for(
        self,
        goal_handle,
        predicate,
        timeout_s: float,
        phase: int,
        current_waypoint: int,
        total_waypoints: int,
        detail: str,
        *,
        route_ned: tuple[tuple[float, float, float], ...] = (),
        target_yaw_rad: float = math.nan,
        readiness_diagnostics: bool = False,
    ) -> str:
        active_elapsed = 0.0
        previous = time.monotonic()
        last_feedback_at = 0.0
        while active_elapsed < timeout_s:
            owned = getattr(self, '_owned_sim_preparation', None)
            if owned is not None:
                try:
                    status = owned.status()
                    if status.get('error') or status.get('stage') in ('cleanup', 'closed'):
                        self._owned_sim_failure_reason = status.get('error') or 'owned session ended before mission completion'
                        return 'owned_sim_failed'
                except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
                    self._owned_sim_failure_reason = str(exc)
                    return 'owned_sim_failed'
            if MissionManagerNode._execution_interrupted(self, goal_handle):
                if owned is not None:
                    return 'cancel'
                if not is_low_speed_profile(self._active_flight_profile):
                    self._hold_current(goal_handle.request.mission_id)
                return "cancel"
            now = time.monotonic()
            with self._lock:
                state = self._vehicle_state
                manual = self._manual_active
                resume_required = self._resume_required
                low_speed_runtime_error = self._active_low_speed_error_locked(now)
            if low_speed_runtime_error:
                self._low_speed_failure_reason = low_speed_runtime_error
                return "low_speed_safety_land"
            if (is_low_speed_profile(self._active_flight_profile) and state is not None
                    and str(getattr(state, "last_error", "")).startswith("LOW_SPEED_")):
                self._low_speed_failure_reason = str(state.last_error)
                return "low_speed_safety_land"
            if manual or resume_required:
                if getattr(self, "_event_terminal_requested", False):
                    self._event_terminal_failure_reason = "manual or resume interlock interrupted terminal incident flow"
                    return "event_terminal_failed"
                resumed = self._wait_for_explicit_resume(
                    goal_handle,
                    return_phase=phase,
                    current_waypoint=current_waypoint,
                    total_waypoints=total_waypoints,
                    reason="manual override" if manual else self._resume_reason,
                    publish_hold=False,
                )
                if resumed != "ok":
                    return resumed
                previous = time.monotonic()
                continue
            # Charge ordinary route/predicate wait before excluding time spent
            # in the event handler. Resetting previous first would subtract a
            # newer timestamp from now on every route poll and prevent the
            # waypoint timeout from ever expiring, even with no event lease.
            active_elapsed += now - previous
            previous = now
            if route_ned:
                event_result = self._handle_event_control(
                    goal_handle,
                    route_ned,
                    current_waypoint,
                    total_waypoints,
                    target_yaw_rad,
                )
                if event_result != "ok":
                    return event_result
                # Event HOLD/capture/rejoin has its own bounded waits and is
                # intentionally paused out of this waypoint's active budget.
                previous = time.monotonic()
            if state is not None and predicate(state):
                return "ok"
            if readiness_diagnostics and state is None:
                self._last_vehicle_ready_reason = "vehicle_state_missing"
            if now - last_feedback_at >= 0.5:
                feedback_detail = detail
                if readiness_diagnostics:
                    feedback_detail += "; readiness=" + self._last_vehicle_ready_reason
                self._publish_feedback(
                    goal_handle, phase, current_waypoint, total_waypoints, feedback_detail
                )
                if self._capture_then_home():
                    self._publish_status(goal_handle.request.mission_id,
                        goal_handle.request.proposal_id, phase, feedback_detail,
                        current_waypoint, total_waypoints)
                last_feedback_at = now
            time.sleep(0.1)
        return "timeout"

    def _active_low_speed_error_locked(self, now: float) -> str:
        if not is_low_speed_profile(self._active_flight_profile):
            return ""
        state = self._vehicle_state
        if self._active_frame_continuity_error:
            return self._active_frame_continuity_error
        if state is None or not 0.0 <= now-self._vehicle_state_received_at <= 0.5:
            return "LOW_SPEED_PROFILE vehicle state missing or stale"
        command = getattr(self, '_scenario_last_command', None)
        if command is not None and command.mission_id == self._active_mission_id:
            envelope_error = navigation_error(self, command, state.position_ned_m)
            if envelope_error:
                return 'LOW_SPEED_PROFILE '+envelope_error
        controller_error = str(getattr(state, "last_error", ""))
        for name, detail in (
            ("connected", "FC disconnected"),
            ("vehicle_status_fresh", "vehicle status stale"),
            ("arming_state_valid", "arming state unknown"),
            ("landed_state_valid", "landed state unknown"),
            ("position_valid", "local position invalid"),
            ("heading_fresh", "heading invalid or stale"),
            ("preflight_checks_pass", "PX4 preflight/failsafe state invalid"),
        ):
            if not bool(getattr(state, name, False)):
                if (name == 'preflight_checks_pass' and state.armed and state.offboard
                        and battery_home.home_confirmation_ms(self, mission_id=self._active_mission_id) == 2000
                        and state.flight_output_detail in battery_home.GRACE_DETAILS):
                    continue
                if name == "preflight_checks_pass" and controller_error.startswith("LOW_SPEED_PROFILE battery"):
                    return controller_error
                return "LOW_SPEED_PROFILE " + detail
        if (self._active_route_home is None
                or not 0.0 <= now-self._route_home_received_at <= self._route_home_timeout_s):
            return "LOW_SPEED_PROFILE frozen Home is missing or stale"
        reference_reason, _, _, _ = self._altitude_reference_status_locked(
            self._active_altitude_reference, now)
        if reference_reason == "altitude_reference_stale":
            return "LOW_SPEED_ALTITUDE_REFERENCE_STALE"
        if reference_reason in {
                "altitude_reference_mismatch",
                "altitude_reference_unstable",
                "altitude_reference_epoch_changed",
        }:
            return "LOW_SPEED_ALTITUDE_REFERENCE_DIVERGED"
        controller_error = str(getattr(state, "last_error", ""))
        fatal_markers = (
            "LOW_SPEED_PROFILE", "LOW_SPEED_ALTITUDE_REFERENCE",
            "estimator reset", "flight epoch retired",
            "Offboard mode confirmation timeout", "ARM confirmation timeout",
        )
        if controller_error and any(marker in controller_error for marker in fatal_markers):
            return controller_error
        return ""

    def _handle_event_control(
        self,
        goal_handle,
        route_ned: tuple[tuple[float, float, float], ...],
        current_waypoint: int,
        total_waypoints: int,
        target_yaw_rad: float,
    ) -> str:
        """Run capture and the configured, explicit completion policy."""

        if self._capture_then_home():
            return self._handle_terminal_event_capture(goal_handle)

        request = goal_handle.request
        with self._lock:
            lease = self._event_control.expire(time.monotonic())
            if lease is None:
                return "ok"

        hold_position_ned = self._hold_current(
            request.mission_id,
            requested_authority=OffboardCommand.AUTHORITY_JETSON_EVENT_CAPTURE,
        )
        if lease.state is EventLeaseState.REQUESTED:
            hold_result = self._wait_for_event_hold(
                goal_handle, hold_position_ned
            )
            if hold_result != "ok":
                failure_reason = (
                    "manual override interrupted event HOLD"
                    if hold_result == "manual"
                    else (
                        "mission canceled during event HOLD"
                        if hold_result == "cancel"
                        else "event HOLD confirmation timed out"
                    )
                )
                with self._lock:
                    lease = self._event_control.request_release(
                        lease_id=lease.lease_id,
                        source=lease.source,
                        capture_succeeded=False,
                        reason=failure_reason,
                    )
                if hold_result == "cancel":
                    return "cancel"
            else:
                with self._lock:
                    lease = self._event_control.activate(
                        lease.lease_id, now=time.monotonic()
                    )

        if lease.state is EventLeaseState.RELEASE_REQUESTED:
            # HOLD failed before capture. The explicit resume above authorized
            # recovery; skip EVENT_CAPTURE and proceed to route rejoin.
            pass
        else:
            self._publish_status(
                request.mission_id,
                request.proposal_id,
                MissionStatus.PHASE_EVENT_CAPTURE,
                "event capture active after confirmed HOLD (%.1f s maximum)"
                % self._event_control.max_duration_s,
                current_waypoint,
                total_waypoints,
            )
        last_feedback_at = 0.0
        while lease.state is not EventLeaseState.RELEASE_REQUESTED:
            if goal_handle.is_cancel_requested:
                if is_low_speed_profile(self._active_flight_profile):
                    self._land_current(request.mission_id)
                else:
                    self._hold_current(request.mission_id)
                goal_handle.canceled()
                return "cancel"
            now = time.monotonic()
            with self._lock:
                manual = self._manual_active
                lease = self._event_control.expire(now)
            if manual or self._resume_required:
                resumed = self._wait_for_explicit_resume(
                    goal_handle,
                    return_phase=MissionStatus.PHASE_EVENT_CAPTURE,
                    current_waypoint=current_waypoint,
                    total_waypoints=total_waypoints,
                    reason="manual override" if manual else self._resume_reason,
                    publish_hold=False,
                )
                if resumed != "ok":
                    return resumed
                continue
            if lease is None:
                return "hold_manual_required"
            if lease.state is EventLeaseState.RELEASE_REQUESTED:
                break
            if now - last_feedback_at >= 0.5:
                remaining = max(0.0, lease.expires_at - now)
                self._publish_feedback(
                    goal_handle,
                    MissionStatus.PHASE_EVENT_CAPTURE,
                    current_waypoint,
                    total_waypoints,
                    "event capture holding; %.0f s remaining" % remaining,
                )
                last_feedback_at = now
            time.sleep(0.1)

        if not lease.capture_succeeded:
            resumed = self._wait_for_explicit_resume(
                goal_handle,
                return_phase=MissionStatus.PHASE_REJOINING_ROUTE,
                current_waypoint=current_waypoint,
                total_waypoints=total_waypoints,
                reason=(
                    "event capture failed: %s"
                    % (lease.release_reason or "unknown recorder failure")
                ),
                publish_hold=True,
            )
            if resumed != "ok":
                return resumed

        with self._lock:
            lease = self._event_control.begin_rejoin(lease.lease_id)
        while True:
            state = self._snapshot_state()
            try:
                rejoin = (
                    route_rejoin_with_lookahead(
                        state.position_ned_m,
                        route_ned,
                        self._event_rejoin_lookahead_m,
                    )
                    if state is not None and state.position_valid
                    else None
                )
            except ValueError:
                rejoin = None
            if rejoin is None:
                resumed = self._wait_for_explicit_resume(
                    goal_handle,
                    return_phase=MissionStatus.PHASE_REJOINING_ROUTE,
                    current_waypoint=current_waypoint,
                    total_waypoints=total_waypoints,
                    reason="event route rejoin needs a valid position and route",
                    publish_hold=True,
                )
                if resumed != "ok":
                    return resumed
                continue
            self._publish_command(
                request.mission_id,
                OffboardCommand.COMMAND_GOTO,
                rejoin.position_ned_m,
                target_yaw_rad,
            )
            self._publish_status(
                request.mission_id,
                request.proposal_id,
                MissionStatus.PHASE_REJOINING_ROUTE,
                "event capture released; rejoining route segment %d"
                % rejoin.segment_index,
                current_waypoint,
                total_waypoints,
            )
            rejoined = self._wait_for(
                goal_handle,
                lambda value: distance_ned(value.position_ned_m, rejoin.position_ned_m)
                <= self._acceptance_radius,
                self._waypoint_timeout,
                MissionStatus.PHASE_REJOINING_ROUTE,
                current_waypoint,
                total_waypoints,
                "rejoining nearest route segment",
            )
            if rejoined == "ok":
                break
            if rejoined == "cancel":
                return rejoined
            resumed = self._wait_for_explicit_resume(
                goal_handle,
                return_phase=MissionStatus.PHASE_REJOINING_ROUTE,
                current_waypoint=current_waypoint,
                total_waypoints=total_waypoints,
                reason="event route rejoin timed out",
                publish_hold=True,
            )
            if resumed != "ok":
                return resumed
        with self._lock:
            self._event_control.clear(lease.lease_id)
        self._publish_status(
            request.mission_id,
            request.proposal_id,
            MissionStatus.PHASE_PATROL,
            "route rejoined; resuming patrol waypoint",
            current_waypoint,
            total_waypoints,
        )
        return "ok"

    def _observe_simple_demo_endpoint(self, goal_handle, route_ned, yaw_ned,
                                      current_waypoint, total_waypoints):
        if not (getattr(self, "_simple_obstacle_demo", False)
                and getattr(self, "_simple_demo_phase1_events", False)
                and getattr(self, "_diagnostic_relaxed_timing", False)):
            return "ok"
        duration = getattr(self, '_simple_demo_endpoint_observation_s', 10.0)
        if duration == 0.0:
            return "ok"  # Moving-event profile: no endpoint observation dwell.
        # Explicit SIM photo-demo observation dwell, separate from the 5 s clip.
        # Keep the final approved GOTO and PATROL alive for actual inference;
        # the existing wait handles event capture, cancellation and manual stop.
        until = time.monotonic() + duration
        return self._wait_for(
            goal_handle,
            lambda state: time.monotonic() >= until and self._route_target_reached(
                state, goal_handle.request.mission_id, route_ned[-1]),
            self._waypoint_timeout, MissionStatus.PHASE_PATROL,
            current_waypoint, total_waypoints,
            f"SIM photo endpoint observation: {duration:g} s before no-incident Home return",
            route_ned=route_ned, target_yaw_rad=yaw_ned)

    def _return_navigation_error_locked(self, now):
        if self._manual_active:
            return "manual control owns the vehicle"
        if self._mission_home is None or self._home_reference_invalidated:
            return "confirmed immutable Home reference unavailable or changed"
        if self._current_home_reference_locked(now) is None:
            return "Home reference current PX4 epoch association unavailable"
        return owned_navigation_error(self._vehicle_state, mission_id=self._active_mission_id,
                                      received_at=self._vehicle_state_received_at, now=now)

    def _event_fail(self, reason):
        self._event_terminal_failure_reason = str(reason)
        return "event_terminal_failed"

    def _publish_return_command(self, command, target, yaw, *, authority=None):
        # The controller repeats these authority checks at actual publication;
        # this manager check cannot replace the flight-owner interlock.
        with self._lock:
            error = self._return_navigation_error_locked(time.monotonic())
            mission_id = self._active_mission_id
        if error:
            self._event_terminal_failure_reason = error
            return False
        self._publish_command(mission_id, command, target, yaw, save_for_resume=False,
                              requested_authority=(OffboardCommand.AUTHORITY_LLM_ROUTE
                                                   if authority is None else authority),
                              acceptance_radius_m=self._event_return_acceptance_radius)
        return True

    def _handle_terminal_event_capture(self, goal_handle):
        request = goal_handle.request
        with self._lock:
            lease = self._event_control.expire(time.monotonic())
            if lease is None:
                return "ok"
            error = self._return_navigation_error_locked(time.monotonic())
            state = self._vehicle_state
        if error:
            return self._event_fail(error)
        hold_position = tuple(float(v) for v in state.position_ned_m)
        if lease.state is EventLeaseState.REQUESTED:
            if not self._publish_return_command(OffboardCommand.COMMAND_HOLD, hold_position,
                    float(state.heading_rad), authority=OffboardCommand.AUTHORITY_JETSON_EVENT_CAPTURE):
                return "event_terminal_failed"
            held = self._wait_for_event_hold(goal_handle, hold_position)
            with self._lock:
                lease = self._event_control.expire(time.monotonic())
                if held == "ok" and lease is not None and lease.state is EventLeaseState.REQUESTED:
                    lease = self._event_control.activate(lease.lease_id, now=time.monotonic())
                elif lease is not None:
                    lease = self._event_control.request_release(lease_id=lease.lease_id,
                        source=lease.source, capture_succeeded=False,
                        reason="event HOLD not confirmed: " + held)
            if held in {"manual", "authority", "cancel"}:
                return self._event_fail("event HOLD interrupted by " + held)
        if lease is None:
            return self._event_fail("event lease disappeared before capture")
        if lease.state is EventLeaseState.ACTIVE:
            self._publish_status(request.mission_id, request.proposal_id,
                MissionStatus.PHASE_EVENT_CAPTURE, "capture active; patrol will not resume after recording")
        while lease.state is not EventLeaseState.RELEASE_REQUESTED:
            if goal_handle.is_cancel_requested:
                return self._event_fail("operator canceled during incident capture")
            with self._lock:
                error = self._return_navigation_error_locked(time.monotonic())
                lease = self._event_control.expire(time.monotonic())
            if error:
                return self._event_fail("incident capture ownership lost: " + error)
            if lease is None:
                return self._event_fail("event lease disappeared during recording")
            time.sleep(.05)
        with self._lock:
            self._event_capture_succeeded = lease.capture_succeeded is True
            self._pending_resume_command = None
            self._resume_required = False
            self._resume_requested = False
        return self._return_over_visited_path(goal_handle)

    def _clearance_before_reverse(self, goal_handle, total_deadline):
        """Finish only the current authorized avoidance, never another patrol leg."""
        deadline = min(total_deadline, time.monotonic()+self._event_return_clearance_timeout)
        replayed = False
        while time.monotonic() < deadline:
            if goal_handle.is_cancel_requested:
                return self._event_fail("operator canceled clearance before reverse")
            with self._lock:
                state = self._vehicle_state
                error = self._return_navigation_error_locked(time.monotonic())
                command_spec = self._event_outbound_command_spec
            if error:
                return self._event_fail(error)
            if ready_to_reverse(state):
                return "ok"
            if getattr(state, "avoidance_active", None) is True and not replayed:
                if (command_spec is None or command_spec[0] != self._active_mission_id
                        or command_spec[1] != OffboardCommand.COMMAND_GOTO
                        or command_spec[4] != OffboardCommand.AUTHORITY_LLM_ROUTE):
                    return self._event_fail("no original approved GOTO available to finish avoidance")
                if not self._publish_return_command(command_spec[1], command_spec[2], command_spec[3]):
                    return "event_terminal_failed"
                replayed = True
                self._publish_status(self._active_mission_id, self._active_proposal_id,
                    MissionStatus.PHASE_RETURNING_HOME,
                    "clearance_before_reverse: finish current approved avoidance only; patrol ended")
            time.sleep(.05)
        return self._event_fail("clearance_before_reverse timed out; no blind reverse or native RTL")

    def _return_over_visited_path(self, goal_handle):
        request = goal_handle.request
        total_deadline = time.monotonic()+self._event_return_timeout
        self._publish_status(request.mission_id, request.proposal_id,
            MissionStatus.PHASE_RETURNING_HOME,
            "incident/patrol terminal; preparing depth-guarded return over visited path")
        cleared = self._clearance_before_reverse(goal_handle, total_deadline)
        if cleared != "ok":
            return cleared
        with self._lock:
            error = self._return_navigation_error_locked(time.monotonic())
            if error:
                return self._event_fail(error)
            state = self._vehicle_state
            home = self._mission_home
            try:
                if self._return_trace is None:
                    raise ValueError("no visited return trace")
                targets = self._return_trace.reverse_targets(state.position_ned_m,
                                                             self._event_return_acceptance_radius)
                raw_target_count = len(targets)
                targets = merge_collinear_return_targets(state.position_ned_m, targets)
            except ValueError as exc:
                return self._event_fail(str(exc))
            self._event_return_in_progress = True
        self._publish_status(request.mission_id, request.proposal_id,
            MissionStatus.PHASE_RETURNING_HOME,
            f"visited trace {raw_target_count} samples -> {len(targets)} conservative straight legs; "
            f"overall budget {self._event_return_timeout:g}s includes yaw/avoidance/landing")
        for index, target in enumerate(targets):
            with self._lock:
                error = self._return_navigation_error_locked(time.monotonic())
                state = self._vehicle_state
            if error:
                return self._event_fail(error)
            if not getattr(state, "heading_fresh", False):
                return self._event_fail("fresh heading unavailable for forward-facing return")
            try:
                yaw = return_yaw(state.position_ned_m, target, state.heading_rad)
            except (ValueError, TypeError) as exc:
                return self._event_fail(str(exc))
            if not self._publish_return_command(OffboardCommand.COMMAND_GOTO, target, yaw):
                return "event_terminal_failed"
            self._publish_status(request.mission_id, request.proposal_id,
                MissionStatus.PHASE_RETURNING_HOME,
                "visited-path return: heading/depth gates retain priority; no patrol resume",
                index, len(targets))
            deadline = min(total_deadline, time.monotonic()+self._waypoint_timeout)
            reached = False
            while time.monotonic() < deadline:
                if goal_handle.is_cancel_requested:
                    return self._event_fail("operator canceled visited-path return")
                with self._lock:
                    state = self._vehicle_state
                    error = self._return_navigation_error_locked(time.monotonic())
                if error:
                    return self._event_fail(error)
                if distance_ned(state.position_ned_m, target) <= self._event_return_acceptance_radius:
                    reached = True
                    break
                time.sleep(.05)
            if not reached:
                return self._event_fail("visited-path return waypoint timed out; no native RTL fallback")
        with self._lock:
            state = self._vehicle_state
            error = self._return_navigation_error_locked(time.monotonic())
        if error:
            return self._event_fail(error)
        if not ready_to_reverse(state):
            return self._event_fail("Home approach still has active avoidance or unverified depth")
        if not near_home_for_landing(state.position_ned_m, home,
                radius_m=self._event_return_home_radius,
                maximum_height_m=self._event_return_max_landing_height):
            return self._event_fail("LAND prohibited outside verified Home vicinity/elevation")
        if not self._publish_return_command(OffboardCommand.COMMAND_LAND, home, float(state.heading_rad)):
            return "event_terminal_failed"
        self._publish_status(request.mission_id, request.proposal_id,
            MissionStatus.PHASE_RETURNING_HOME, "verified Home vicinity; native LAND requested once")
        deadline = min(total_deadline, time.monotonic()+self._landing_timeout)
        while time.monotonic() < deadline:
            with self._lock:
                state = self._vehicle_state
                age = time.monotonic()-self._vehicle_state_received_at
                manual = self._manual_active
            if goal_handle.is_cancel_requested or manual:
                return self._event_fail("landing interrupted by operator; no command reacquisition")
            if (state is None or not 0 <= age <= .5 or state.connected is not True
                    or state.mission_id != request.mission_id):
                return self._event_fail("landing confirmation connection is not current")
            # After our one LAND request, native mode/retired epoch is expected.
            # Only observe; do not stream offboard, re-arm or resend LAND.
            if state.landed is True and state.armed is False:
                if (state.position_valid is not True
                        or not near_home_for_landing(state.position_ned_m, home,
                            radius_m=self._event_return_home_radius, maximum_height_m=1.)):
                    return self._event_fail("landing detected but Home position/elevation unconfirmed")
                return "event_home_completed" if self._event_capture_succeeded else "event_home_capture_failed"
            time.sleep(.05)
        return self._event_fail("Home landing/disarm not confirmed before bounded deadline")

    def _event_home_result(self, goal_handle, result, reason, detail=None):
        if getattr(self, '_owned_sim_preparation', None) is not None:
            cleanup = self._close_owned_sim_preparation(goal_handle.request)
            if cleanup.get('stage') != 'closed' or cleanup.get('cleanup_confirmed') is not True:
                self._owned_sim_failure_reason = 'Home landing observed but owned helper cleanup unconfirmed'
                return self._finish_owned_sim_failure(goal_handle, result, 'failed')
        result.success = reason == "event_home_completed"
        result.final_phase = MissionStatus.PHASE_COMPLETED
        result.message = detail or ("incident recorded; visited-path Home landing confirmed" if result.success
                                    else "Home landing confirmed, but incident recording failed or expired")
        with self._lock:
            completed_capture = self._event_control.snapshot()
        if (completed_capture is not None
                and completed_capture.mission_id == goal_handle.request.mission_id
                and completed_capture.release_reason):
            result.message += "; capture_result=" + completed_capture.release_reason
        if result.success:
            goal_handle.succeed()
        elif goal_handle.is_active:
            goal_handle.abort()
        self._publish_status(goal_handle.request.mission_id, goal_handle.request.proposal_id,
                             result.final_phase, result.message)
        return result

    def _event_terminal_failure_result(self, goal_handle, result, reason):
        if goal_handle.is_active:
            goal_handle.abort()
        with self._lock:
            error = self._return_navigation_error_locked(time.monotonic())
            state = self._vehicle_state
        if not error:
            self._publish_return_command(OffboardCommand.COMMAND_HOLD,
                tuple(state.position_ned_m), float(state.heading_rad))
        result.success = False
        result.final_phase = MissionStatus.PHASE_HOLD_MANUAL_REQUIRED
        result.message = "terminal return not completed: " + str(reason) + "; no automatic patrol resume"
        self._publish_status(goal_handle.request.mission_id, goal_handle.request.proposal_id,
                             result.final_phase, result.message)
        return result

    def _wait_for_event_hold(
        self,
        goal_handle,
        hold_position_ned: tuple[float, float, float] | None = None,
    ) -> str:
        """Confirm command ownership and a stable position before recording."""

        deadline = time.monotonic() + self._event_hold_timeout_s
        stable_since = None
        stable_position = None
        while time.monotonic() < deadline:
            if goal_handle.is_cancel_requested:
                self._hold_current(goal_handle.request.mission_id)
                goal_handle.canceled()
                return "cancel"
            with self._lock:
                state = self._vehicle_state
                manual = self._manual_active
                if self._capture_then_home() and self._return_navigation_error_locked(time.monotonic()):
                    return "authority"
            if manual:
                return "manual"
            ready = state is not None and (
                state.command_output_enabled
                and state.approved
                and state.connected
                and state.preflight_checks_pass
                and state.position_valid
                and state.armed
                and state.offboard
                and state.active_authority == "JETSON_EVENT_CAPTURE"
            )
            if not ready:
                stable_since = None
                stable_position = None
                time.sleep(0.05)
                continue
            position = tuple(float(value) for value in state.position_ned_m)
            if (
                hold_position_ned is not None
                and distance_ned(position, hold_position_ned)
                > self._event_hold_drift_tolerance_m
            ):
                # Low instantaneous movement is not enough: after a fast
                # approach the aircraft can creep toward the fixed HOLD target
                # for several seconds.  Do not start the five-second evidence
                # clip until it has actually converged to that target.
                stable_since = None
                stable_position = None
                time.sleep(0.05)
                continue
            if (
                stable_position is None
                or distance_ned(position, stable_position)
                > self._event_hold_drift_tolerance_m
            ):
                stable_position = position
                stable_since = time.monotonic()
            elif (
                stable_since is not None
                and time.monotonic() - stable_since >= self._event_hold_settle_s
            ):
                return "ok"
            time.sleep(0.05)
        return "timeout"

    def _wait_for_explicit_resume(
        self,
        goal_handle,
        *,
        return_phase: int,
        current_waypoint: int,
        total_waypoints: int,
        reason: str,
        publish_hold: bool,
    ) -> str:
        request = goal_handle.request
        if MissionManagerNode._execution_interrupted(self, goal_handle):
            return "cancel"
        self._require_explicit_resume(
            return_phase=return_phase,
            reason=reason,
            publish_hold=publish_hold,
        )
        last_feedback_at = 0.0
        while True:
            if MissionManagerNode._execution_interrupted(self, goal_handle):
                # The outer execution path owns terminal landing and Action completion.
                if not is_low_speed_profile(getattr(self, "_active_flight_profile", "")) and not getattr(self, "_emergency_land_requested", False):
                    self._hold_current(request.mission_id)
                return "cancel"
            now = time.monotonic()
            with self._lock:
                manual = self._manual_active
                requested = self._resume_requested
            if not manual and requested:
                with self._lock:
                    if (getattr(self, "_active_cancel_requested", False)
                            or getattr(self, "_emergency_land_requested", False)
                            or goal_handle.is_cancel_requested):
                        return "cancel"
                    self._resume_required = False
                    self._resume_requested = False
                    self._resume_reason = ""
                    self._pending_resume_command = None
                return "ok"
            if now - last_feedback_at >= 0.5:
                phase = (
                    MissionStatus.PHASE_PAUSED_MANUAL
                    if manual
                    else MissionStatus.PHASE_HOLD_MANUAL_REQUIRED
                )
                detail = (
                    "manual control has priority"
                    if manual
                    else "holding; an operator must send mission.resume"
                )
                self._publish_feedback(
                    goal_handle,
                    phase,
                    current_waypoint,
                    total_waypoints,
                    detail,
                )
                last_feedback_at = now
            time.sleep(0.1)

    def _require_explicit_resume(
        self,
        *,
        return_phase: int,
        reason: str,
        publish_hold: bool,
    ) -> None:
        with self._lock:
            if not self._resume_required:
                self._pending_resume_command = self._last_command_spec
                self._resume_return_phase = return_phase
            self._resume_required = True
            self._resume_requested = False
            self._resume_reason = reason
            mission_id = self._active_mission_id
            proposal_id = self._active_proposal_id
            waypoint = self._active_waypoint
            total = self._active_total_waypoints
            manual = self._manual_active
        if publish_hold:
            self._hold_current(mission_id)
        self._publish_status(
            mission_id,
            proposal_id,
            (
                MissionStatus.PHASE_PAUSED_MANUAL
                if manual
                else MissionStatus.PHASE_HOLD_MANUAL_REQUIRED
            ),
            "%s; holding until an explicit mission.resume" % reason,
            waypoint,
            total,
        )

    def _mark_resume_required_locked(self, reason: str) -> None:
        if not self._resume_required:
            self._pending_resume_command = self._last_command_spec
            self._resume_return_phase = self._active_phase
        self._resume_required = True
        self._resume_requested = False
        self._resume_reason = reason

    def _republish_command_spec(self, command_spec: tuple) -> None:
        mission_id, command, position_ned, yaw_rad, requested_authority = command_spec
        self._publish_command(
            mission_id,
            command,
            position_ned,
            yaw_rad,
            requested_authority=requested_authority,
            save_for_resume=False,
        )

    def _finish_wait_failure(self, goal_handle, result, reason: str, detail: str):
        if getattr(self, '_owned_sim_preparation', None) is not None:
            terminal_reason = (self._event_terminal_failure_reason
                               if reason == 'event_terminal_failed' else '')
            self._owned_sim_failure_reason = self._owned_sim_failure_reason or terminal_reason or detail
            return self._finish_owned_sim_failure(goal_handle, result, 'cancel' if reason == 'cancel' else 'failed')
        if getattr(self, "_readiness_only_checkpoint", False):
            return self._finish_readiness_checkpoint(goal_handle, result, False, reason + "; " + detail)
        if reason == "event_terminal_failed":
            return self._event_terminal_failure_result(goal_handle, result,
                self._event_terminal_failure_reason or detail)
        if reason == "low_speed_safety_land":
            return self._abort_result(
                goal_handle,
                result,
                self._low_speed_failure_reason or detail,
            )
        if reason == "cancel":
            terminal_outcome = "NOT_REQUIRED"
            battery_reason = ""
            state = self._snapshot_state()
            if (state is not None and state.active_execution_mission_id == goal_handle.request.mission_id):
                cause = str(getattr(state, "first_fault", "") or getattr(state, "last_error", ""))
                if cause.startswith("LOW_SPEED_PROFILE battery"):
                    battery_reason = cause
            emergency_land = bool(self._emergency_land_requested)
            if emergency_land or is_low_speed_profile(self._active_flight_profile):
                terminal_outcome = self._complete_low_speed_terminal(goal_handle)
            landed = terminal_outcome in {
                "NOT_REQUIRED", "ACCEPTED", "PREARM_TERMINATED"}
            if goal_handle.is_active:
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                else:
                    goal_handle.abort()
            result.success = False
            result.final_phase = MissionStatus.PHASE_ABORTED
            result.message = (
                ("emergency LAND completed; vehicle landed and disarmed"
                 if landed else
                 "emergency LAND failed to obtain ordered landing confirmation")
                if emergency_land else
                ("low-speed mission canceled; landed and disarmed"
                 if landed else
                 "low-speed mission canceled; landing confirmation failed")
                if is_low_speed_profile(self._active_flight_profile)
                else "mission canceled; vehicle holding position"
            )
            if battery_reason:
                result.message = battery_reason + "; " + result.message
            return result
        if reason == "hold_manual_required":
            return self._hold_manual_required_result(goal_handle, result)
        return self._abort_result(goal_handle, result, detail)

    def _execute_exception_result(self, goal_handle, result, exc: Exception):
        """Last-resort Action result path; it must not recursively fail."""
        message = "unhandled mission execution error: " + str(exc)
        result.success = False
        result.final_phase = MissionStatus.PHASE_ERROR
        result.message = message
        request = goal_handle.request
        try:
            self._publish_status(
                request.mission_id, request.proposal_id,
                MissionStatus.PHASE_ERROR, message,
            )
        except Exception as publish_exc:
            self.get_logger().error("mission error status publication failed: %s" % publish_exc)
        try:
            state = self._snapshot_state()
            if self._can_issue_terminal_land(state):
                terminal = (OffboardCommand.COMMAND_LAND
                            if is_low_speed_profile(self._active_flight_profile)
                            else OffboardCommand.COMMAND_RTL)
                target = (tuple(state.position_ned_m)
                          if terminal == OffboardCommand.COMMAND_LAND
                          else (0.0, 0.0, 0.0))
                self._publish_command(
                    request.mission_id, terminal, target, math.nan,
                    save_for_resume=False,
                )
            if is_low_speed_profile(self._active_flight_profile):
                self._complete_low_speed_terminal(
                    goal_handle, command_already_sent=self._can_issue_terminal_land(state),
                )
        except Exception as terminal_exc:
            self.get_logger().error("mission exception terminal handling failed: %s" % terminal_exc)
        try:
            if goal_handle.is_active:
                goal_handle.abort()
        except Exception as action_exc:
            self.get_logger().error("mission Action abort transition failed: %s" % action_exc)
        return result

    @staticmethod
    def _confirmed_landed_disarmed(state) -> bool:
        """Accept terminal completion only from fresh, explicitly valid PX4 states."""
        return bool(
            state is not None
            and getattr(state, "vehicle_status_fresh", False)
            and getattr(state, "arming_state_valid", False)
            and getattr(state, "landed_state_valid", False)
            and bool(state.landed)
            and not bool(state.armed)
        )

    @classmethod
    def _low_speed_terminal_outcome(cls, state, *, ever_armed):
        landed = cls._confirmed_landed_disarmed(state)
        terminal = int(getattr(
            state, "terminal_state", VehicleControlState.TERMINAL_NONE
        )) if state is not None else VehicleControlState.TERMINAL_NONE
        if (landed and not ever_armed and getattr(state, "prearm_terminated", False)
                and not getattr(state, "arm_request_transmitted", True)):
            return "PREARM_TERMINATED"
        if landed and terminal == VehicleControlState.TERMINAL_ACCEPTED:
            return "ACCEPTED"
        if landed and terminal == VehicleControlState.TERMINAL_FALLBACK:
            return "FALLBACK_LANDED"
        if terminal == VehicleControlState.TERMINAL_FAILED:
            return "FAILED"
        return None

    def _hold_manual_required_result(self, goal_handle, result):
        if goal_handle.is_active:
            goal_handle.abort()
        request = goal_handle.request
        self._hold_current(request.mission_id)
        result.success = False
        result.final_phase = MissionStatus.PHASE_HOLD_MANUAL_REQUIRED
        result.message = "event recovery requires manual operator intervention; holding"
        self._publish_status(
            request.mission_id,
            request.proposal_id,
            MissionStatus.PHASE_HOLD_MANUAL_REQUIRED,
            result.message,
        )
        return result

    def _abort_result(self, goal_handle, result, message: str):
        if getattr(self, "_readiness_only_checkpoint", False):
            return self._finish_readiness_checkpoint(goal_handle, result, False, message)
        if self._capture_then_home():
            return self._event_terminal_failure_result(goal_handle, result, message)
        result.success = False
        result.final_phase = MissionStatus.PHASE_ERROR
        result.message = message
        request = goal_handle.request
        self._publish_status(
            request.mission_id,
            request.proposal_id,
            MissionStatus.PHASE_ERROR,
            message,
        )
        state = self._snapshot_state()
        if self._can_issue_terminal_land(state):
            self._publish_command(
                request.mission_id,
                (OffboardCommand.COMMAND_LAND
                 if is_low_speed_profile(self._active_flight_profile)
                 else OffboardCommand.COMMAND_RTL),
                (tuple(state.position_ned_m)
                 if is_low_speed_profile(self._active_flight_profile)
                 else (0.0, 0.0, 0.0)),
                math.nan,
            )
        if is_low_speed_profile(self._active_flight_profile):
            self._complete_low_speed_terminal(
                goal_handle, command_already_sent=self._can_issue_terminal_land(state))
        if goal_handle.is_active:
            goal_handle.abort()
        return result

    def _complete_low_speed_terminal(self, goal_handle, *, command_already_sent=False):
        """Hold approval until the ordered native-LAND contract terminates."""
        request = goal_handle.request
        initial = self._snapshot_state()
        if getattr(initial, "active_execution_mission_id", "") != request.mission_id:
            initial = None
        if self._low_speed_terminal_outcome(
                initial, ever_armed=self._active_ever_armed
        ) == "PREARM_TERMINATED":
            return "PREARM_TERMINATED"
        if not command_already_sent:
            self._land_current(request.mission_id)
        deadline = self._terminal_deadlines.setdefault(
            request.mission_id, time.monotonic()+self._landing_timeout)
        last_feedback_at = 0.0
        terminal_observed = False
        while time.monotonic() < deadline:
            now = time.monotonic()
            with self._lock:
                state = self._vehicle_state
                age = now-self._vehicle_state_received_at
            if (state is not None and 0.0 <= age <= 0.5
                    and state.active_execution_mission_id == request.mission_id):
                terminal_started_ns = getattr(state, "terminal_started_monotonic_ns", 0)
                if terminal_started_ns and state.active_execution_mission_id == request.mission_id:
                    terminal_observed = True
                    deadline = min(deadline, terminal_started_ns/1e9+self._landing_timeout)
                    self._terminal_deadlines[request.mission_id] = deadline
                outcome = self._low_speed_terminal_outcome(
                    state, ever_armed=self._active_ever_armed)
                if outcome is not None:
                    return outcome
            if not terminal_observed:
                self._land_current(request.mission_id)
            if now-last_feedback_at >= 0.5:
                terminal = int(getattr(
                    state, "terminal_state", VehicleControlState.TERMINAL_NONE
                )) if state is not None else VehicleControlState.TERMINAL_NONE
                try:
                    self._publish_feedback(
                        goal_handle, MissionStatus.PHASE_LANDING,
                        self._active_waypoint, self._active_total_waypoints,
                        "terminal LAND pending; state=%d" % terminal,
                    )
                except Exception as exc:
                    self.get_logger().error(
                        "terminal feedback publication failed: %s" % exc)
                last_feedback_at = now
            time.sleep(0.1)
        return "TIMEOUT"

    def _land_current(self, mission_id: str) -> None:
        state = self._snapshot_state()
        with self._lock:
            authorized = mission_id in self._approved
            fresh = 0 <= time.monotonic()-self._vehicle_state_received_at <= 0.5
        if authorized and fresh and self._can_issue_terminal_land(state):
            now = time.monotonic()
            if (getattr(self, "_terminal_land_last_mission", "") == mission_id
                    and now-getattr(self, "_terminal_land_last_sent_at", 0.0) < 0.1):
                return
            requests = getattr(self, "_terminal_land_requests", None)
            if requests is None:
                self._terminal_land_requests = requests = {}
            if mission_id in requests:
                self._command_publisher.publish(requests[mission_id])
                self._terminal_land_last_sent_at = now
            else:
                message = self._publish_command(mission_id, OffboardCommand.COMMAND_LAND,
                                      tuple(state.position_ned_m), math.nan,
                                      save_for_resume=False)
                if message is not None:
                    requests.clear()
                    requests[mission_id] = message
                    self._terminal_land_last_mission = mission_id
                    self._terminal_land_last_sent_at = now

    @staticmethod
    def _can_issue_terminal_land(state) -> bool:
        """Never create a new LAND command from stale/ambiguous FC state."""
        return bool(
            state is not None
            and getattr(state, "vehicle_status_fresh", False)
            and getattr(state, "arming_state_valid", False)
            and bool(getattr(state, "connected", False))
            and (bool(getattr(state, "armed", False))
                 or (bool(getattr(state, "landed_state_valid", False))
                     and bool(getattr(state, "landed", False))))
            and bool(getattr(state, "command_output_enabled", False))
        )

    def _hold_current(
        self,
        mission_id: str,
        *,
        requested_authority: int = OffboardCommand.AUTHORITY_LLM_ROUTE,
    ) -> tuple[float, float, float] | None:
        if self._capture_then_home():
            with self._lock:
                if self._return_navigation_error_locked(time.monotonic()):
                    return None
        state = self._snapshot_state()
        if state is None or not state.position_valid:
            return None
        hold_position_ned = tuple(
            float(value) for value in state.position_ned_m
        )
        self._publish_command(
            mission_id,
            OffboardCommand.COMMAND_HOLD,
            hold_position_ned,
            math.nan,
            requested_authority=requested_authority,
            save_for_resume=False,
        )
        return hold_position_ned

    def _snapshot_state(self):
        with self._lock:
            return (self._vehicle_state
                    if 0 <= time.monotonic()-self._vehicle_state_received_at <= 0.5
                    else None)

    def _publish_approval(
        self,
        proposal_id: str,
        mission_id: str,
        approved: bool,
        operator_id: str,
    ) -> None:
        message = MissionApproval()
        message.stamp = self.get_clock().now().to_msg()
        message.proposal_id = proposal_id
        message.mission_id = mission_id
        message.approved = approved
        message.operator_id = operator_id or "operator"
        self._approval_publisher.publish(message)

    def _publish_command(
        self,
        mission_id: str,
        command: int,
        position_ned,
        yaw_rad: float,
        *,
        requested_authority: int = OffboardCommand.AUTHORITY_LLM_ROUTE,
        save_for_resume: bool = True,
        acceptance_radius_m: float | None = None,
    ) -> OffboardCommand | None:
        if getattr(self, '_owned_sim_preparing', False):
            return  # Native baseline/cleanup owns movement until an explicit handoff.
        if getattr(self, "_readiness_only_checkpoint", False):
            return  # Includes cancel/error HOLD/RTL paths: diagnostic scope has no flight output.
        self._sequence += 1
        message = OffboardCommand()
        message.stamp = self.get_clock().now().to_msg()
        message.mission_id = mission_id
        message.sequence = self._sequence
        message.flight_output_handshake_version = 8
        message.command = command
        message.requested_authority = (
            OffboardCommand.AUTHORITY_FORWARD_TEST_1M
            if (
                self._active_mission_kind == FORWARD_TEST_1M
                and requested_authority == OffboardCommand.AUTHORITY_LLM_ROUTE
            )
            else requested_authority
        )
        message.approved = True
        message.position_ned_m = [float(value) for value in position_ned]
        message.yaw_rad = float(yaw_rad)
        low = is_low_speed_profile(self._active_flight_profile)
        message.acceptance_radius_m = ((ACCEPTANCE_RADIUS_M if low else self._acceptance_radius)
                                       if acceptance_radius_m is None
                                       else float(acceptance_radius_m))
        message.flight_profile = (
            OffboardCommand.FLIGHT_PROFILE_LOW_SPEED_1M_V1
            if self._active_flight_profile == LOW_SPEED_1M_V1 else
            OffboardCommand.FLIGHT_PROFILE_LOW_SPEED_2M_V1
            if self._active_flight_profile == LOW_SPEED_2M_V1 else
            OffboardCommand.FLIGHT_PROFILE_NORMAL)
        message.home_z_ned_m = float(self._active_home_z_ned_m)
        message.altitude_reference = (
            OffboardCommand.ALTITUDE_REFERENCE_FC_HOME_ALIGNED_V2
            if low else OffboardCommand.ALTITUDE_REFERENCE_NONE)
        message.altitude_reference_max_error_m = (
            ALTITUDE_REFERENCE_MAX_ERROR_M if low else 0.0)
        reference = self._active_altitude_reference or {}
        message.altitude_reference_epoch = int(
            reference.get("transport_epoch", 0) if low else 0)
        message.altitude_reference_sequence = int(
            reference.get("sequence", 0) if low else 0)
        message.altitude_reference_local_time_boot_ms = int(
            reference.get("local_time_boot_ms", 0) if low else 0)
        message.altitude_reference_global_time_boot_ms = int(
            reference.get("global_time_boot_ms", 0) if low else 0)
        self._scenario_last_command = message
        self._command_publisher.publish(message)
        if save_for_resume:
            with self._lock:
                self._last_command_spec = (
                    mission_id,
                    command,
                    tuple(float(value) for value in position_ned),
                    float(yaw_rad),
                    requested_authority,
                )
        return message

    def _publish_status(
        self,
        mission_id: str,
        proposal_id: str,
        phase: int,
        detail: str,
        current_waypoint: int = 0,
        total_waypoints: int = 0,
    ) -> None:
        state = self._snapshot_state()
        with self._lock:
            if mission_id and mission_id == self._active_mission_id:
                self._active_phase = phase
                self._active_waypoint = current_waypoint
                self._active_total_waypoints = total_waypoints
            lease = self._event_control.snapshot()
            active_goal = self._active_goal
            manual = self._manual_active
        if manual:
            control_owner = "HUMAN"
        elif lease is not None and lease.state in (
            EventLeaseState.REQUESTED,
            EventLeaseState.ACTIVE,
        ):
            control_owner = "JETSON_EVENT_CAPTURE"
        elif active_goal and phase not in (
            MissionStatus.PHASE_COMPLETED,
            MissionStatus.PHASE_ABORTED,
            MissionStatus.PHASE_ERROR,
            MissionStatus.PHASE_HOLD_MANUAL_REQUIRED,
        ):
            control_owner = "LLM_ROUTE"
        else:
            control_owner = "NONE"
        message = MissionStatus()
        message.stamp = self.get_clock().now().to_msg()
        message.mission_id = mission_id
        message.proposal_id = proposal_id
        message.phase = phase
        message.approved = mission_id in self._approved
        message.manual_override = bool(state.manual_override) if state else False
        message.armed = bool(state.armed) if state else False
        message.offboard = bool(state.offboard) if state else False
        message.current_waypoint = current_waypoint
        message.total_waypoints = total_waypoints
        message.control_owner = control_owner
        if lease is not None:
            message.event_id = lease.event_id
            message.event_type = lease.event_type
            message.event_lease_id = lease.lease_id
            message.event_lease_remaining_s = float(
                max(0.0, lease.expires_at - time.monotonic())
            )
        message.detail = detail
        self._status_publisher.publish(message)

    @staticmethod
    def _publish_feedback(
        goal_handle,
        phase: int,
        current_waypoint: int,
        total_waypoints: int,
        detail: str,
    ) -> None:
        feedback = ExecuteMission.Feedback()
        feedback.phase = phase
        feedback.current_waypoint = current_waypoint
        feedback.total_waypoints = total_waypoints
        feedback.detail = detail
        goal_handle.publish_feedback(feedback)

    @staticmethod
    def _same_json(left: str, right: str) -> bool:
        return json.loads(left) == json.loads(right)

    def destroy_node(self):
        self._action_server.destroy()
        if hasattr(self, "_scenario_inputs"):
            self._scenario_inputs.pool.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionManagerNode()
    executor = MultiThreadedExecutor(num_threads=4)
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

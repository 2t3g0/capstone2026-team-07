import json
import sys
import threading
import time
import types
from types import MethodType, SimpleNamespace

import pytest
from builtin_interfaces.msg import Time
from jolgwa_interfaces.msg import MissionStatus, VehicleControlState
from rclpy.action import GoalResponse

from jolgwa_ros.low_speed import FORWARD_TEST_1M, ROUTE

# The callback state test does not construct a ROS node or use PX4 messages.
# Stock Humble CI does not ship px4_msgs, so provide only the two import names
# required to load MissionManagerNode. Jetson/SITL builds use the real package.
try:
    import px4_msgs.msg  # noqa: F401
except ModuleNotFoundError:
    px4_messages = types.ModuleType("px4_msgs.msg")
    for message_name in (
        "HomePosition", "OffboardControlMode", "TrajectorySetpoint",
        "VehicleCommand", "VehicleCommandAck", "VehicleLandDetected",
        "VehicleLocalPosition", "VehicleStatus",
    ):
        setattr(px4_messages, message_name, type(message_name, (), {}))
    px4_package = types.ModuleType("px4_msgs")
    px4_package.msg = px4_messages
    sys.modules["px4_msgs"] = px4_package
    sys.modules["px4_msgs.msg"] = px4_messages

from jolgwa_ros.mission_manager_node import MissionManagerNode
from jolgwa_ros.px4_offboard_controller import Px4OffboardController
from jolgwa_ros.route_frame import RouteFrameResult, RouteHome


class _ClockNow:
    @staticmethod
    def to_msg():
        return Time()


class _Clock:
    @staticmethod
    def now():
        return _ClockNow()


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def test_camera_off_forward_prepare_approve_execute_goal_keeps_mission_kind():
    """Exercise the real callbacks so approval cannot silently default to ROUTE."""
    manager = SimpleNamespace()
    manager._lock = threading.RLock()
    manager._forward_test_camera_required = False
    manager._vehicle_state = SimpleNamespace(
        position_ned_m=[10.0, 20.0, 0.0],
        heading_rad=0.0,
    )
    manager._vehicle_geo_state = SimpleNamespace(
        valid=True,
        age_ms=0,
        altitude_home_relative_m=0.03,
    )
    manager._altitude_reference_state = SimpleNamespace(
        state=1,
        valid=True,
        stable=True,
        transport_epoch=7,
        sequence=20,
        local_time_boot_ms=1000,
        global_time_boot_ms=1000,
        local_age_ms=0,
        global_age_ms=0,
        source_skew_ms=0,
        local_z_ned_m=0.0,
        fc_altitude_home_relative_m=0.03,
        stable_aligned_home_z_ned_m=0.03,
        detail="ready",
    )
    received_at = time.monotonic()
    manager._vehicle_state_received_at = received_at
    manager._vehicle_geo_received_at = received_at
    manager._altitude_reference_state_received_at = received_at
    manager._route_home = RouteHome(
        35.0, 129.0, 40.0, (0.0, 0.0, 1.14), 1
    )
    manager._forward_previews = {}
    manager._altitude_references = {}
    manager._proposal_publisher = _Publisher()
    manager._proposals = {}
    manager._route_frame_results = {}
    manager._route_frame_messages = {}
    manager._consumed_preview_states = {}
    manager._approved = {}
    manager._approved_at = {}
    manager._claimed_approvals = set()
    manager._active_goal = False
    manager._active_proposal_id = ""
    manager._goal_reservation = None
    manager.get_clock = lambda: _Clock()
    manager._publish_approval = lambda *args: None
    manager._publish_status = lambda *args: None
    manager._capture_then_home = lambda: False
    manager._same_json = MissionManagerNode._same_json
    manager._capture_altitude_reference_locked = MethodType(
        MissionManagerNode._capture_altitude_reference_locked, manager
    )
    manager._altitude_reference_status_locked = MethodType(
        MissionManagerNode._altitude_reference_status_locked, manager
    )
    manager._allow_goal_waypoints = False
    manager._update_route_frame_status = lambda proposal_id: None
    manager._prune_mission_caches_locked = lambda: None

    readiness_mission_kinds = []

    def readiness(*, require_ground=True, allow_active=False,
                  mission_kind=ROUTE):
        del require_ground, allow_active
        readiness_mission_kinds.append(mission_kind)
        return ("" if mission_kind == FORWARD_TEST_1M
                else "Jetson safety input is not CLEAR")

    manager._low_speed_readiness_error_locked = readiness
    manager._on_prepare_forward_test = MethodType(
        MissionManagerNode._on_prepare_forward_test, manager
    )
    manager._on_approve = MethodType(MissionManagerNode._on_approve, manager)
    manager._on_goal = MethodType(MissionManagerNode._on_goal, manager)

    prepare_response = SimpleNamespace(
        accepted=False, proposal_id="", message=""
    )
    manager._on_prepare_forward_test(
        SimpleNamespace(operator_id="operator", target_altitude_home_m=1.0),
        prepare_response,
    )
    assert prepare_response.accepted
    assert readiness_mission_kinds == [FORWARD_TEST_1M]

    proposal = manager._proposal_publisher.messages[-1]
    plan = json.loads(proposal.plan_json)
    assert plan["mission_kind"] == FORWARD_TEST_1M
    assert plan["preview_start_ned_m"] == [10.0, 20.0, 0.0]
    assert plan["preview_end_ned_m"] == [12.0, 20.0, 0.0]
    manager._proposals[proposal.proposal_id] = proposal
    frame = RouteFrameResult(
        route_ned=((12.0, 20.0, -1.0),), home=None,
        site_id="FORWARD_TEST_LOCAL_NED", first_leg_m=2.0,
        maximum_leg_m=2.0)
    manager._route_frame_results[proposal.proposal_id] = (frame, "", 0.0)

    approve_response = SimpleNamespace(
        accepted=False, mission_id="", message=""
    )
    manager._on_approve(
        SimpleNamespace(
            proposal_id=proposal.proposal_id,
            approved=True,
            operator_id="operator",
        ),
        approve_response,
    )
    assert approve_response.accepted
    assert readiness_mission_kinds == [FORWARD_TEST_1M, FORWARD_TEST_1M]

    goal = SimpleNamespace(
        mission_id=approve_response.mission_id,
        proposal_id=proposal.proposal_id,
        plan_json=proposal.plan_json,
        waypoints_enu=(),
    )
    assert manager._on_goal(goal) == GoalResponse.ACCEPT

    # Preview drift/TTL are pre-goal acceptance checks only. The accepted
    # execution owns the immutable frame snapshot and approval claim.
    frozen_snapshot = manager._goal_reservation["snapshot"]
    assert proposal.proposal_id not in manager._forward_previews
    manager._vehicle_state.position_ned_m = [12.0, 20.0, 0.0]
    manager._vehicle_state.heading_rad = 1.0
    assert manager._goal_reservation["snapshot"] is frozen_snapshot
    assert frozen_snapshot.route_ned == ((12.0, 20.0, -1.0),)
    assert frozen_snapshot.preview_heading_rad == 0.0
    assert manager._approved[goal.mission_id] == goal.proposal_id
    assert goal.mission_id in manager._claimed_approvals

    # A reservation is acquired atomically; another goal cannot slip through.
    second = SimpleNamespace(
        mission_id=goal.mission_id,
        proposal_id=goal.proposal_id,
        plan_json=goal.plan_json,
        waypoints_enu=(),
    )
    assert manager._on_goal(second) == GoalResponse.REJECT


def test_terminal_confirmation_rejects_stale_legacy_booleans():
    valid = dict(
        vehicle_status_fresh=True,
        arming_state_valid=True,
        landed_state_valid=True,
        landed=True,
        armed=False,
    )
    assert MissionManagerNode._confirmed_landed_disarmed(SimpleNamespace(**valid))
    for field in ("vehicle_status_fresh", "arming_state_valid", "landed_state_valid"):
        state = dict(valid)
        state[field] = False
        assert not MissionManagerNode._confirmed_landed_disarmed(
            SimpleNamespace(**state)
        )


def test_controller_accepts_a_fresh_atomic_altitude_reference_state():
    controller = Px4OffboardController.__new__(Px4OffboardController)
    controller._altitude_reference_state_received_at = 100.0
    controller._altitude_reference_state = SimpleNamespace(
        state=1,
        valid=True,
        stable=True,
        transport_epoch=7,
        sequence=21,
        local_age_ms=20,
        global_age_ms=20,
        source_skew_ms=10,
        local_z_ned_m=-1.0,
        fc_altitude_home_relative_m=0.03,
    )
    command = SimpleNamespace(
        flight_profile=1,
        altitude_reference=2,
        altitude_reference_max_error_m=1.5,
        altitude_reference_epoch=7,
        altitude_reference_sequence=20,
        home_z_ned_m=-0.97,
    )
    ready, reason, error, fc_altitude, frame_altitude = (
        controller._altitude_reference_status(command, 100.05)
    )
    assert ready
    assert reason == "ready"
    assert error == pytest.approx(0.0)
    assert fc_altitude == pytest.approx(0.03)
    assert frame_altitude == pytest.approx(0.03)


def _emergency_manager(state):
    state.emergency_land_available = True
    state.active_execution_mission_id = "mission-active"
    manager = SimpleNamespace()
    manager._lock = threading.RLock()
    manager._vehicle_state = state
    manager._vehicle_state_received_at = time.monotonic()
    manager._active_goal = True
    manager._active_mission_id = "mission-active"
    manager._active_proposal_id = "proposal-active"
    manager._active_waypoint = 1
    manager._active_total_waypoints = 2
    manager._emergency_land_requested = False
    manager._active_cancel_requested = False
    manager._manual_active = True
    manager._resume_required = True
    manager._resume_requested = True
    manager._manual_publisher = _Publisher()
    manager.statuses = []
    manager.get_clock = lambda: _Clock()
    manager._publish_status = lambda *args: manager.statuses.append(args)
    manager._can_issue_terminal_land = MissionManagerNode._can_issue_terminal_land
    manager._on_emergency_land = MethodType(
        MissionManagerNode._on_emergency_land, manager)
    return manager


def test_emergency_land_latches_cancel_and_releases_manual_hold():
    manager = _emergency_manager(SimpleNamespace(
        vehicle_status_fresh=True,
        arming_state_valid=True,
        connected=True,
        armed=True,
        command_output_enabled=True,
    ))
    response = SimpleNamespace(accepted=False, message="", mission_id="")

    manager._on_emergency_land(SimpleNamespace(
        mission_id="mission-active", operator_id="operator", reason="hazard"),
        response,
    )

    assert response.accepted
    assert response.mission_id == "mission-active"
    assert manager._emergency_land_requested
    assert manager._active_cancel_requested
    assert not manager._manual_active
    assert not manager._resume_required
    assert len(manager._manual_publisher.messages) == 1
    assert not manager._manual_publisher.messages[0].active
    assert manager.statuses[-1][2] == MissionStatus.PHASE_LANDING


def test_emergency_land_rejects_wrong_mission_or_stale_arming_state():
    state = SimpleNamespace(
        vehicle_status_fresh=True,
        arming_state_valid=True,
        connected=True,
        armed=True,
        command_output_enabled=True,
    )
    manager = _emergency_manager(state)
    response = SimpleNamespace(accepted=False, message="", mission_id="")
    manager._on_emergency_land(SimpleNamespace(
        mission_id="wrong", operator_id="operator", reason="hazard"), response)
    assert not response.accepted
    assert not manager._emergency_land_requested

    manager._vehicle_state.arming_state_valid = False
    response = SimpleNamespace(accepted=False, message="", mission_id="")
    manager._on_emergency_land(SimpleNamespace(
        mission_id="mission-active", operator_id="operator", reason="hazard"),
        response,
    )
    assert not response.accepted
    assert not manager._active_cancel_requested


def test_terminal_outcome_requires_land_ack_after_any_arm_history():
    base = dict(
        vehicle_status_fresh=True,
        arming_state_valid=True,
        landed_state_valid=True,
        landed=True,
        armed=False,
    )
    pending = SimpleNamespace(
        **base, terminal_state=VehicleControlState.TERMINAL_PENDING)
    assert MissionManagerNode._low_speed_terminal_outcome(
        pending, ever_armed=True) is None
    assert MissionManagerNode._low_speed_terminal_outcome(
        pending, ever_armed=False) is None

    no_terminal = SimpleNamespace(
        **base, terminal_state=VehicleControlState.TERMINAL_NONE)
    assert MissionManagerNode._low_speed_terminal_outcome(
        no_terminal, ever_armed=True) is None
    assert MissionManagerNode._low_speed_terminal_outcome(
        no_terminal, ever_armed=False) is None
    no_terminal.prearm_terminated = True
    no_terminal.arm_request_transmitted = False
    assert MissionManagerNode._low_speed_terminal_outcome(
        no_terminal, ever_armed=False) == "PREARM_TERMINATED"

    accepted = SimpleNamespace(
        **base, terminal_state=VehicleControlState.TERMINAL_ACCEPTED)
    assert MissionManagerNode._low_speed_terminal_outcome(
        accepted, ever_armed=True) == "ACCEPTED"

    fallback = SimpleNamespace(
        **base, terminal_state=VehicleControlState.TERMINAL_FALLBACK)
    assert MissionManagerNode._low_speed_terminal_outcome(
        fallback, ever_armed=True) == "FALLBACK_LANDED"

    failed = SimpleNamespace(
        **{**base, "landed": False, "armed": True},
        terminal_state=VehicleControlState.TERMINAL_FAILED)
    assert MissionManagerNode._low_speed_terminal_outcome(
        failed, ever_armed=True) == "FAILED"


def test_proposal_id_collision_cannot_replace_frozen_content():
    manager = SimpleNamespace()
    manager._lock = threading.RLock()
    manager._proposals = {}
    manager._approved = {}
    manager._active_proposal_id = ""
    manager._goal_reservation = None
    manager._route_frame_results = {}
    manager._route_frame_messages = {}
    manager._consumed_preview_states = {}
    manager._forward_previews = {}
    manager.CACHE_LIMIT = 64
    manager._same_json = MissionManagerNode._same_json
    manager._prune_mission_caches_locked = MethodType(
        MissionManagerNode._prune_mission_caches_locked, manager)
    manager._update_route_frame_status = lambda proposal_id: None
    manager.statuses = []
    manager._publish_status = lambda *args: manager.statuses.append(args)
    manager.get_logger = lambda: SimpleNamespace(error=lambda *args: None)
    manager._on_proposal = MethodType(MissionManagerNode._on_proposal, manager)

    original = SimpleNamespace(
        proposal_id="proposal-fixed", plan_json='{"a":1,"b":2}',
        status=0, message="")
    identical = SimpleNamespace(
        proposal_id="proposal-fixed", plan_json='{"b":2,"a":1}',
        status=0, message="")
    conflicting = SimpleNamespace(
        proposal_id="proposal-fixed", plan_json='{"a":999,"b":2}',
        status=0, message="")

    manager._on_proposal(original)
    manager._on_proposal(identical)
    manager._on_proposal(conflicting)

    assert manager._proposals["proposal-fixed"] is original
    assert manager._proposals["proposal-fixed"].plan_json == '{"a":1,"b":2}'
    assert manager.statuses[-1][2] == MissionStatus.PHASE_ERROR
    assert "collision" in manager.statuses[-1][3]


def test_home_continuity_checks_wgs84_altitude_and_local_ned():
    home = RouteHome(35.0, 129.0, 40.0, (0.0, 0.0, 0.0), 1)
    assert not MissionManagerNode._route_home_changed(home, home)
    assert MissionManagerNode._route_home_changed(
        home, RouteHome(35.00001, 129.0, 40.0, (0.0, 0.0, 0.0), 2)
    )
    assert MissionManagerNode._route_home_changed(
        home, RouteHome(35.0, 129.0, 40.6, (0.0, 0.0, 0.0), 2)
    )
    assert MissionManagerNode._route_home_changed(
        home, RouteHome(35.0, 129.0, 40.0, (0.6, 0.0, 0.0), 2)
    )

"""Real Mission Manager callbacks across provisional Home refinement."""
import threading
import time
from types import MethodType, SimpleNamespace as NS

import pytest
from rclpy.action import GoalResponse

from test_forward_test_flow import MissionManagerNode, _Clock, _Publisher


def home(z=0.0):
    return NS(timestamp=100, lat=35.0, lon=129.0, alt=40.0-z,
              x=0.0, y=0.0, z=z, valid_lpos=True,
              valid_hpos=True, valid_alt=True)


def ready(manager, z=0.0, epoch=1):
    now = time.monotonic()
    manager._vehicle_state_received_at = now
    manager._vehicle_geo_received_at = now
    manager._low_speed_safety_received_at = now
    manager._altitude_reference_state_received_at = now
    manager._vehicle_state.position_ned_m = [0.0, 0.0, z]
    manager._altitude_reference_state = NS(
        valid=True, stable=True, state=1, transport_epoch=epoch, sequence=21,
        local_time_boot_ms=2000, global_time_boot_ms=2000,
        local_age_ms=0, global_age_ms=0, source_skew_ms=0,
        local_z_ned_m=z, fc_altitude_home_relative_m=0.0,
        normalized_fc_altitude_home_relative_m=0.0,
        stable_aligned_home_z_ned_m=z, detail="ready",
        execution_home_lock_valid=False,
        provisional_px4_home_z_ned_m=z,
        provisional_px4_home_altitude_amsl_m=40.0-z)


def manager_fixture():
    manager = NS(
        _lock=threading.RLock(), _route_home=None, _route_home_received_at=-float("inf"),
        _route_home_fault="", _route_home_generation=0, _route_home_timeout_s=2.5,
        _active_route_home=None, _active_frame_continuity_error="",
        _active_goal=False, _goal_reservation=None, _claimed_approvals=set(),
        _active_proposal_id="", _active_mission_id="", _active_ever_armed=False,
        _event_return_source_mode="disabled", _physical_route=True,
        _forward_test_camera_required=False, _max_first_leg_m=100.0,
        _allow_goal_waypoints=False, _active_cancel_requested=False,
        _forward_previews={}, _altitude_references={}, _proposals={},
        _route_frame_results={}, _route_frame_messages={}, _consumed_preview_states={},
        _approved={}, _approved_at={}, CACHE_LIMIT=64,
        _proposal_publisher=_Publisher(), _route_frame_publisher=_Publisher(),
        _vehicle_state=NS(
            connected=True, vehicle_status_fresh=True, arming_state_valid=True,
            armed=False, landed_state_valid=True, landed=True,
            command_output_enabled=True, preflight_checks_pass=True,
            position_valid=True, heading_fresh=True, heading_rad=0.0,
            manual_override=False),
        _vehicle_geo_state=NS(valid=True, age_ms=0),
        _low_speed_safety_state=NS(valid=True, age_ms=0),
        get_clock=lambda: _Clock(), get_logger=lambda: NS(warning=lambda msg: None),
        _capture_then_home=lambda: False,
    )
    manager.approvals = []
    manager.statuses = []
    manager._publish_approval = lambda *args: manager.approvals.append(args)
    manager._publish_status = lambda *args: manager.statuses.append(args)
    for name in (
        "_on_home_position", "_on_prepare_forward_test", "_on_approve", "_on_goal",
        "_low_speed_readiness_error_locked", "_capture_altitude_reference_locked",
        "_altitude_reference_status_locked", "_update_all_route_frame_statuses",
        "_update_route_frame_status", "_prune_mission_caches_locked",
    ):
        setattr(manager, name, MethodType(getattr(MissionManagerNode, name), manager))
    manager._route_home_changed = MissionManagerNode._route_home_changed
    manager._home_with_aligned_z = MissionManagerNode._home_with_aligned_z
    manager._same_json = MissionManagerNode._same_json
    ready(manager)
    manager._on_home_position(home())
    return manager


def prepare(manager):
    response = NS(accepted=False, proposal_id="", message="")
    manager._on_prepare_forward_test(
        NS(operator_id="test", target_altitude_home_m=1.0), response)
    if response.accepted:
        proposal = manager._proposal_publisher.messages[-1]
        manager._proposals[proposal.proposal_id] = proposal
        manager._update_route_frame_status(proposal.proposal_id)
    return response


def approve(manager, proposal_id):
    response = NS(accepted=False, mission_id="", message="")
    manager._on_approve(NS(proposal_id=proposal_id, approved=True, operator_id="test"), response)
    return response


def goal(manager, proposal_id, mission_id):
    return NS(proposal_id=proposal_id, mission_id=mission_id, waypoints_enu=(),
              plan_json=manager._proposals[proposal_id].plan_json)


def test_home_refinement_then_real_prepare_approval_and_goal():
    manager = manager_fixture()
    # Reproduce >0.5m startup refinement; repeat it as the bridge does at 10Hz.
    for z in (1.593, 1.593, 1.7, 1.7):
        ready(manager, z, epoch=2 if z == 1.593 else 3)
        manager._on_home_position(home(z))
    assert manager._route_home_generation == 2
    assert manager._route_home_fault == ""
    prepared = prepare(manager)
    assert prepared.accepted, prepared.message
    approved = approve(manager, prepared.proposal_id)
    assert approved.accepted, approved.message
    assert manager._on_goal(goal(manager, prepared.proposal_id, approved.mission_id)) == GoalResponse.ACCEPT
    snapshot = manager._goal_reservation["snapshot"]
    assert snapshot.home_generation == manager._route_home_generation == 2
    assert snapshot.raw_route_home.ned[2] == pytest.approx(1.7)
    assert snapshot.route_ned[0][2] == pytest.approx(0.7)


def test_changed_home_revokes_old_approval_but_new_prepare_recovers():
    manager = manager_fixture()
    prepared = prepare(manager)
    approved = approve(manager, prepared.proposal_id)
    assert approved.accepted
    ready(manager, 1.593, epoch=2)
    manager._on_home_position(home(1.593))
    assert prepared.proposal_id not in manager._forward_previews
    assert approved.mission_id not in manager._approved
    assert any(row[1] == approved.mission_id and row[2] is False for row in manager.approvals)
    assert manager._on_goal(goal(manager, prepared.proposal_id, approved.mission_id)) == GoalResponse.REJECT
    assert not approve(manager, prepared.proposal_id).accepted
    assert prepare(manager).accepted
    assert manager._route_home_fault == ""


@pytest.mark.parametrize("stale_field", ["arming_state_valid", "vehicle_status_fresh", "landed_state_valid"])
def test_unknown_ground_state_does_not_rebase_home(stale_field):
    manager = manager_fixture()
    setattr(manager._vehicle_state, stale_field, False)
    manager._on_home_position(home(1.593))
    assert manager._route_home.ned[2] == 0.0
    assert manager._route_home_generation == 0
    assert manager._route_home_fault == ""
    setattr(manager._vehicle_state, stale_field, True)
    ready(manager, 1.593, 2)
    manager._on_home_position(home(1.593))
    assert prepare(manager).accepted


def test_new_home_cannot_capture_old_ready_altitude_topic():
    manager = manager_fixture()
    manager._on_home_position(home(1.593))
    response = prepare(manager)
    assert not response.accepted
    assert response.message == "provisional_home_alignment_pending"
    ready(manager, 1.593, 2)
    assert prepare(manager).accepted


def test_reserved_or_active_home_change_retains_approval_for_terminal():
    manager = manager_fixture()
    prepared = prepare(manager)
    approved = approve(manager, prepared.proposal_id)
    assert manager._on_goal(goal(manager, prepared.proposal_id, approved.mission_id)) == GoalResponse.ACCEPT
    snapshot = manager._goal_reservation["snapshot"]
    ready(manager, 1.593, 2)
    manager._on_home_position(home(1.593))
    assert manager._route_home_fault
    assert manager._active_frame_continuity_error
    assert manager._approved[approved.mission_id] == prepared.proposal_id
    assert manager._goal_reservation["snapshot"] is snapshot
    assert manager._route_home.ned[2] == 0.0


def test_home_change_during_approval_does_not_commit_permission():
    manager = manager_fixture()
    prepared = prepare(manager)
    original = manager._update_route_frame_status
    def interleave(proposal_id):
        original(proposal_id)
        ready(manager, 1.593, 2)
        manager._on_home_position(home(1.593))
    manager._update_all_route_frame_statuses = lambda: None
    manager._update_route_frame_status = interleave
    response = approve(manager, prepared.proposal_id)
    assert not response.accepted
    assert manager._approved == {}


def test_active_frame_change_still_latches_without_rebasing():
    manager = manager_fixture()
    manager._active_route_home = manager._route_home
    manager._active_goal = True
    manager._vehicle_state.armed = True
    manager._vehicle_state.landed = False
    original = manager._active_route_home
    manager._on_home_position(home(1.593))
    assert manager._route_home_fault
    assert manager._active_route_home is original
    assert manager._route_home is original


def test_home_change_during_goal_validation_rolls_back_claim():
    manager = manager_fixture()
    prepared = prepare(manager)
    approved = approve(manager, prepared.proposal_id)
    compare = manager._same_json
    def interleave(first, second):
        ready(manager, 1.593, 2)
        manager._on_home_position(home(1.593))
        return compare(first, second)
    manager._same_json = interleave
    assert manager._on_goal(goal(manager, prepared.proposal_id, approved.mission_id)) == GoalResponse.REJECT
    assert manager._goal_reservation is None
    assert not manager._claimed_approvals


def test_small_provisional_change_invalidates_preview_without_permanent_fault():
    manager = manager_fixture()
    prepared = prepare(manager)
    ready(manager, 0.1, 2)
    manager._on_home_position(home(0.1))
    assert prepared.proposal_id not in manager._forward_previews
    assert not approve(manager, prepared.proposal_id).accepted
    assert prepare(manager).accepted
    assert manager._route_home_fault == ""

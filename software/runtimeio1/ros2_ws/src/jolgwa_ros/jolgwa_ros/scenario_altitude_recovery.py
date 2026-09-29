"""Bounded, approval-derived zero-velocity recovery; no new ROS wire fields."""
import copy
import json
from .scenario_runtime import spec_for
from .flight_contract import output_identity_fresh, proof_fresh
from .low_speed import accumulated_sample_age_s

CAUSES = frozenset(('altitude_pose_time_skew', 'yaw_reset_post_attitude_pending'))
WAIT = 'altitude_recovery_wait'
READY = 'altitude_recovery_ready'


def diagnostic(node, event, **values):
    if hasattr(node, 'get_logger'):
        node.get_logger().info(json.dumps(dict(event=event, **values)))


def home_key(sample):
    return (sample.transport_epoch, sample.execution_home_mission_id,
            sample.execution_home_lock_revision)


def start(node, message, now):
    command = node._active_command
    if (getattr(node, '_altitude_recovery', None) is not None
            or node._terminal_started_at is not None or command is None
            or command.command not in (1, 2) or spec_for(node, command) is None
            or message.detail not in CAUSES
            or not output_identity_fresh(message, node._output_contract, int(now*1e9))
            or not node._owns_terminal_handoff(now, command.mission_id)
            or not node._heading_fresh(now)):
        return False
    valid, _, _, _, _ = node._altitude_reference_status(command, now)
    if not valid:
        return False
    sample = node._altitude_reference_state
    age = max(accumulated_sample_age_s(sample.local_age_ms, node._altitude_reference_state_received_at, now),
              accumulated_sample_age_s(sample.global_age_ms, node._altitude_reference_state_received_at, now))
    if age >= .5:
        return False
    node._altitude_recovery = dict(command=copy.deepcopy(command), started=now,
        deadline=min(now+.5, now+.5-age), home=home_key(sample), sequence=sample.sequence)
    node._active_command = copy.deepcopy(command)
    node._active_command.command = 0
    node._activate_output_contract(node._active_command)
    node._last_error = WAIT
    diagnostic(node, WAIT, mission_id=command.mission_id, sequence=command.sequence,
               reason=message.detail, deadline=node._altitude_recovery['deadline'], sample_age=age)
    return True


def service(node, now):
    recovery = getattr(node, '_altitude_recovery', None)
    if recovery is None:
        return False
    original = recovery['command']
    if node._terminal_started_at is not None:
        node._altitude_recovery = None
        return False
    # A newer Manager command/cancel always wins; never restore over it.
    active = node._active_command
    if (active is None or active.mission_id != original.mission_id
            or active.sequence != original.sequence or active.command != 0):
        node._altitude_recovery = None
        return False
    sample = node._altitude_reference_state
    owned = (spec_for(node, original) is not None
             and node._owns_terminal_handoff(now, original.mission_id)
             and node._heading_fresh(now))
    if not owned:
        node._altitude_recovery = None
        node._retire_autonomy_epoch('altitude recovery ownership/position lost')
        from jolgwa_interfaces.msg import VehicleControlState
        node._terminal_state = VehicleControlState.TERMINAL_FALLBACK
        return True
    valid, reason, _, _, _ = node._altitude_reference_status(original, now)
    if (now < recovery['started'] or now >= recovery['deadline'] or not valid
            or sample is None or home_key(sample) != recovery['home']):
        node._altitude_recovery = None
        node._low_speed_land('LOW_SPEED_ALTITUDE_REFERENCE_STALE recovery expired: '+str(reason), now)
        return True
    proof = node._flight_output_state
    if (sample.sequence > recovery['sequence'] and proof is not None and proof.detail == READY
            and proof_fresh(proof, node._output_contract, int(now*1e9))):
        node._active_command = copy.deepcopy(original)
        armed_tx, fault = node._arm_transmitted, node._first_fault
        node._activate_output_contract(node._active_command)
        node._arm_transmitted, node._first_fault = armed_tx, fault
        node._altitude_recovery = None
        node._last_error = ''
        diagnostic(node, 'altitude_recovery_resumed', mission_id=original.mission_id,
                   elapsed=now-recovery['started'])
        return True  # Next timer emits the original target under its NEW contract.
    node._last_error = WAIT
    node._publish_velocity_setpoint([0., 0., 0.], float(original.yaw_rad))
    return True


def bridge_hold(node, snapshot, now_ns, envelope, velocity):
    """Recognise only an approved derived HOLD; never grant moving output."""
    from .flight_contract import command_matches_intent
    contract = node._contract
    source = node._intent
    if not (contract and source and spec_for(node, contract.command)
            and contract.command.command == 0 and source.command in (1, 2)
            and command_matches_intent(contract.command, source, allow_recovery_hold=True)):
        return False
    if (envelope.setpoint_kind != envelope.SETPOINT_VELOCITY
            or not all(abs(float(v)) <= 1e-6 for v in velocity)):
        raise ValueError('altitude_recovery_requires_zero_velocity')
    if not (snapshot.armed and snapshot.offboard and snapshot.connected
            and snapshot.position_valid and not snapshot.failsafe
            and not node._restart_required):
        raise ValueError('altitude_recovery_ownership_unavailable')
    key = (contract.command.mission_id, contract.output_epoch, contract.command.sequence)
    previous = getattr(node, '_altitude_recovery_bridge', None)
    if previous is None or previous[0] != key:
        previous = node._altitude_recovery_bridge = (key, now_ns)
    if now_ns < previous[1] or now_ns-previous[1] >= 500_000_000:
        raise ValueError('altitude_recovery_expired')
    return True

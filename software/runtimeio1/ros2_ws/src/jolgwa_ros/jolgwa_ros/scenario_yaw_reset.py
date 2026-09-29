"""Consume inferred heading generations only under the approved v5 contract."""
import time
from .scenario_runtime import spec_for
from .low_speed import boot_time_forward_delta_ms


def allow_heading_reset(node, prior, counters, now):
    command = node._active_command
    spec = spec_for(node, command)
    sample = node._altitude_reference_state
    return bool(spec is not None and spec['yaw_reset_policy'] == 'BOUNDED_YAW_V1'
        and all(type(v) is int for v in (*prior, *counters))
        and counters[:4] == prior[:4] and (counters[4]-prior[4]) % 256 == 1
        and node._gate.is_approved(command.mission_id) and not node._gate.manual_override
        and node._offboard and not node._px4_failsafe_active
        and not getattr(node, '_autonomy_reentry_required', False)
        and not getattr(node, '_external_mode_fenced', False)
        and command.command in (0, 1, 2)
        and sample is not None and sample.valid and sample.stable
        and sample.transport_epoch == spec['transport_epoch']
        and sample.execution_home_lock_valid
        and sample.execution_home_mission_id == command.mission_id
        and sample.home_phase == sample.HOME_PHASE_EXECUTION_LOCKED
        and not sample.altitude_epoch_failure_latched
        and 0 <= now-node._altitude_reference_state_received_at <= .5)


def observe_reset_metadata(node, previous, message):
    """Manager observes the raw counter; classification stays in the bridge.

    Reset-related metadata can precede or follow local-position DDS delivery.
    Neither order may allow a pre-reset CLEAR to finish an ascent stage.
    """
    if (getattr(node, '_scenario_enabled', False) and previous is not None
            and getattr(previous, 'estimator_reset_counter_valid', False) and message.estimator_reset_counter_valid
            and previous.transport_epoch == message.transport_epoch
            and (message.sequence == previous.sequence
                 or boot_time_forward_delta_ms(message.sequence, previous.sequence) is not None)
            and (message.estimator_reset_counter-previous.estimator_reset_counter) % 256 == 1
            and message.execution_home_lock_valid
            and message.execution_home_mission_id == getattr(node, '_active_mission_id', '')):
        key = (message.execution_home_mission_id, message.transport_epoch, message.estimator_reset_counter)
        if key == getattr(node, '_scenario_reset_metadata_key', None):
            return
        node._scenario_reset_metadata_key = key
        node._scenario_yaw_reset_at = time.monotonic()
        inputs = getattr(node, '_scenario_inputs', None)
        if inputs is not None:
            inputs.evidence = None

"""Bounded DDS reorder buffer, never an approval or authority source."""
from copy import deepcopy


def discard(node, reason):
    pending = getattr(node, '_scenario_pending_takeoff', None)
    if pending is not None:
        retired = getattr(node, '_scenario_pending_retired', set())
        retired.add(pending[0].mission_id)
        node._scenario_pending_retired = retired
        node._scenario_pending_takeoff = None
        node._last_error = 'LOW_SPEED_PROFILE '+reason


def invalid(node, command):
    registry = node._scenario_registry
    mid = command.mission_id
    pid = registry.approvals.get(mid)
    altitude = getattr(node, '_altitude_reference_state', None)
    return (not command.approved or mid in registry.retired or pid in registry.poisoned
            or mid in getattr(node, '_scenario_pending_retired', set())
            or mid in getattr(node, '_retired_mission_ids', set())
            or getattr(node, '_external_mode_fenced', False)
            or getattr(node, '_autonomy_reentry_required', False)
            or getattr(node, '_autonomy_resume_required', False)
            or getattr(node._gate, 'manual_override', False)
            or getattr(node, '_restart_required_fault', False)
            or (altitude is not None and altitude.transport_epoch != command.altitude_reference_epoch))


def defer(node, command, now):
    if not getattr(node, '_scenario_enabled', False):
        return False
    pending = getattr(node, '_scenario_pending_takeoff', None)
    if command.command != 1:
        if pending is not None and command.mission_id == pending[0].mission_id:
            discard(node, 'scenario pending TAKEOFF superseded')
        return False
    if pending is not None and command.mission_id != pending[0].mission_id:
        return True  # A delayed command from another mission cannot retire it.
    registry = node._scenario_registry
    if command.mission_id in getattr(node, '_scenario_pending_retired', set()):
        node._last_error = 'LOW_SPEED_PROFILE scenario pending TAKEOFF already retired'
        return True
    if pending is None and registry.get(command.mission_id) is not None and node._gate.is_approved(command.mission_id):
        return False  # No reordering: preserve the original start/handoff gates.
    if invalid(node, command):
        discard(node, 'scenario pending TAKEOFF invalidated')
        node._last_error = 'LOW_SPEED_PROFILE scenario TAKEOFF retired or invalidated'
        return True
    if pending is not None:
        if now >= pending[1] or command != pending[0]:
            discard(node, 'scenario metadata timeout or command replacement')
            return True
    spec = registry.get(command.mission_id)
    if spec is not None and node._gate.is_approved(command.mission_id):
        node._scenario_pending_takeoff = None
        return False  # All original command validation still runs.
    if getattr(node, '_active_command', None) is not None:
        return False
    if pending is None:
        node._scenario_pending_takeoff = (deepcopy(command), now+2.0)
    node._last_error = 'scenario waiting for matching preparation metadata'
    return True


def drain(node, now):
    pending = getattr(node, '_scenario_pending_takeoff', None)
    if pending is None:
        return
    command, deadline = pending
    if now >= deadline or invalid(node, command):
        discard(node, 'scenario metadata timeout or authority/epoch invalidated')
        return
    if (node._scenario_registry.get(command.mission_id) is not None
            and node._gate.is_approved(command.mission_id)):
        node._scenario_pending_takeoff = None
        node._on_command(command)

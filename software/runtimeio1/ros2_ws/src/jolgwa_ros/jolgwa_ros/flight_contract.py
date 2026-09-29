"""USB output identities and proof predicates, independent of ROS executors."""
import math

PROOF_LEASE_NS = 150_000_000
BATTERY_TERMINAL_ONLY = "battery_health_terminal_only"
BATTERY_TERMINAL_UNAVAILABLE = "battery_terminal_handoff_unavailable"


def output_identity_fresh(message, contract, now_ns):
    """State evidence, not TX proof: blocked output can still request landing."""
    return bool(message is not None and contract is not None
        and message.mission_id == contract.command.mission_id
        and message.sequence == contract.command.sequence
        and message.output_epoch == contract.output_epoch
        and message.output_sequence == contract.output_sequence
        and message.transport_connected and message.command_graph_ready
        and 0 <= now_ns-message.published_monotonic_ns <= PROOF_LEASE_NS)


def same_value(a, b):
    if hasattr(a, "__len__") and not isinstance(a, (str, bytes)):
        return len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return (math.isnan(a) and math.isnan(float(b))) or a == b
    return a == b


def command_matches_intent(command, intent, *, allow_recovery_hold=False):
    if intent is None or not command.approved or not intent.approved:
        return False
    internal_land = command.command == 3 and intent.command in (0, 1, 2)
    internal_hold = allow_recovery_hold and command.command == 0 and intent.command in (1, 2)
    for field in intent.get_fields_and_field_types():
        if field == "stamp" or (internal_land and field in ("command", "position_ned_m", "yaw_rad")):
            continue
        if internal_hold and field == 'command':
            continue
        if not same_value(getattr(command, field), getattr(intent, field)):
            return False
    return True


def proof_fresh(proof, contract, now_ns):
    if proof is None or contract is None:
        return False
    return bool(
        proof.mission_id == contract.command.mission_id
        and proof.sequence == contract.command.sequence
        and proof.output_epoch == contract.output_epoch
        and proof.output_sequence == contract.output_sequence
        and proof.transport_connected and proof.command_graph_ready
        and proof.envelope_valid and proof.setpoint_transmitted
        and 0 <= now_ns-proof.last_tx_monotonic_ns <= PROOF_LEASE_NS
        and 0 <= now_ns-proof.published_monotonic_ns <= PROOF_LEASE_NS
        and proof.tx_sequence > 0)

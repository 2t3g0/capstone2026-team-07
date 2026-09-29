"""Approval-bound v6 policies; public ROS message shapes remain unchanged."""
import copy
import json
from jolgwa_uav.battery_policy import LowBatteryObservation
from .flight_contract import command_matches_intent
from .scenario_runtime import spec_for

GRACE_DETAILS = frozenset(('battery_low_observing', 'battery_low_recovery_observing',
                           'battery_low_recovered'))
PAIR_TIMEOUT = 'scenario_contract_pair_timeout'
PAIR_WAIT_NS = 500_000_000


def home_confirmation_ms(node, command=None, mission_id=None):
    spec = (spec_for(node, command) if command is not None else
            node._scenario_registry.get(mission_id)
            if getattr(node, '_scenario_enabled', False) and hasattr(node, '_scenario_registry') else None)
    return 2000 if spec and spec.get('version') == 7 else 400


def observe(node, snapshot, now_ns):
    node._battery_navigation_allowed = False
    command = node._contract.command if node._contract else None
    spec = spec_for(node, command)
    if not spec:
        return None
    owned = bool(node._contract_valid() and not node._restart_required
        and snapshot.armed and snapshot.offboard and snapshot.battery_navigation_healthy
        and snapshot.transport_epoch == spec['transport_epoch'])
    if not owned:
        return None
    key = (command.mission_id, snapshot.transport_epoch, node._contract.output_epoch)
    if not hasattr(node, '_battery_observations'):
        node._battery_observations = {}
    policy = node._battery_observations.setdefault(key, LowBatteryObservation())
    fault = bool(snapshot.sensors_failed & (1 << 25))
    result = policy.update(snapshot.battery_sample, now_ns, battery_fault=fault)
    if not policy.terminal and command.command in (3, 4, 5) and (fault or result in ('observe', 'recovered')):
        policy.terminal = True
        policy.detail = 'battery_terminal_requested'
        result = 'terminal'
    node._battery_policy_detail = policy.detail
    node._battery_navigation_allowed = bool(result in ('observe', 'recovered')
        and command.mission_id not in node._battery_landings
        and not getattr(node, '_scenario_pair_expired', False))
    diagnostic = (key, result, policy.detail)
    if diagnostic != getattr(node, '_battery_policy_diagnostic', None):
        node._battery_policy_diagnostic = diagnostic
        node._journal.write(json.dumps(dict(event='battery_policy_transition',
            monotonic_ns=now_ns, mission_id=key[0], transport_epoch=key[1], output_epoch=key[2],
            source_sequence=command.sequence, output_sequence=node._contract.output_sequence,
            state=result, detail=policy.detail, first_low_ns=policy.first_ns,
            recovery_since_ns=policy.ok_since_ns, sample=snapshot.battery_sample))+'\n')
    return result


def join_intent(node, message, now_ns):
    """Stage only a newer source intent; keep the last committed pair intact."""
    prior = node._contract
    if not spec_for(node, message):
        return False
    pending = getattr(node, '_scenario_pending_contract', None)
    if pending and command_matches_intent(pending.command, message, allow_recovery_hold=True):
        if not pair_expired(node, now_ns):
            node._intent = message
            node._commit_output_contract(pending)
            clear_join(node)
        return True
    if prior is None or prior.command.mission_id != message.mission_id:
        return False
    if message.sequence <= prior.command.sequence:
        return True
    start_join(node, now_ns, message.sequence)
    return True


def join_contract(node, contract, now_ns):
    if not spec_for(node, contract.command):
        return False
    prior = node._contract
    if (prior is not None and prior.command.mission_id == contract.command.mission_id
            and prior.output_epoch != contract.output_epoch):
        clear_join(node)
        node._restart_required = True
        node._reset_tx_run()
        node._last_output_detail = 'scenario_output_epoch_changed'
        return True
    source = node._intent_history.get((contract.command.mission_id, contract.command.sequence))
    if command_matches_intent(contract.command, source, allow_recovery_hold=True):
        if pair_expired(node, now_ns) and contract.command.command not in (3, 4, 5):
            return True
        node._intent = source
        clear_join(node)
        return False
    if not getattr(node, '_scenario_pair_expired', False):
        node._scenario_pending_contract = copy.deepcopy(contract)
        start_join(node, now_ns, contract.command.sequence)
    return True


def start_join(node, now_ns, incoming_sequence):
    if getattr(node, '_scenario_pair_started_ns', None) is None:
        node._scenario_pair_started_ns = now_ns
        node._journal.write(json.dumps(dict(event='contract_pair_wait', monotonic_ns=now_ns,
            mission_id=node._contract.command.mission_id if node._contract else '',
            incoming_source_sequence=incoming_sequence,
            committed_sequence=node._contract.command.sequence if node._contract else 0,
            output_epoch=node._contract.output_epoch if node._contract else ''))+'\n')


def clear_join(node):
    node._scenario_pair_started_ns = None
    node._scenario_pending_contract = None
    node._scenario_pair_expired = False


def pair_expired(node, now_ns):
    start = getattr(node, '_scenario_pair_started_ns', None)
    if start is not None and (now_ns < start or now_ns-start >= PAIR_WAIT_NS):
        if not getattr(node, '_scenario_pair_expired', False):
            node._journal.write(json.dumps(dict(event='contract_pair_expired',
                monotonic_ns=now_ns, started_ns=start))+'\n')
        node._scenario_pair_expired = True
    return getattr(node, '_scenario_pair_expired', False)


def terminal_reason_fresh(node, message, contract, now_ns):
    """An older committed pair may request stopping, but is never new TX proof."""
    known = (contract if message is not None and contract is not None
             and message.output_sequence == contract.output_sequence else
             getattr(node, '_output_contract_history', {}).get((message.output_epoch, message.output_sequence))
             if message is not None else None)
    return bool(message is not None and contract is not None
        and known is not None and known.command.sequence == message.sequence
        and spec_for(node, contract.command)
        and message.mission_id == contract.command.mission_id
        and message.output_epoch == contract.output_epoch
        and 0 < message.output_sequence <= contract.output_sequence
        and 0 < message.sequence <= contract.command.sequence
        and message.transport_connected and message.command_graph_ready
        and 0 <= now_ns-message.published_monotonic_ns <= 150_000_000)


def prepare_terminal_context(node, message):
    """Use the Bridge-committed source for a NEW brake contract, never old TX."""
    prior = getattr(node, '_output_contract_history', {}).get((message.output_epoch, message.output_sequence))
    if (prior is not None and prior.command.sequence == message.sequence
            and prior.command.mission_id == message.mission_id and spec_for(node, prior.command)):
        node._active_command = copy.deepcopy(prior.command)


def record_sample(node, snapshot, now_ns):
    """Separate warning-time measurements from fresh post-landing recovery."""
    if not getattr(node, '_scenario_enabled', False):
        return
    sample = snapshot.battery_sample
    if sample is None:
        return
    key = (sample.get('charge_state'), snapshot.armed, snapshot.landed)
    previous = getattr(node, '_battery_sample_log', (None, 0))
    if key == previous[0] and now_ns-previous[1] < 1_000_000_000:
        return
    node._battery_sample_log = (key, now_ns)
    node._journal.write(json.dumps(dict(event='battery_sample_observation',
        monotonic_ns=now_ns, mission_id=node._command_mission_id,
        transport_epoch=snapshot.transport_epoch, armed=snapshot.armed, landed=snapshot.landed,
        sample_age_ms=(now_ns-sample['received_ns'])/1e6, sample=sample))+'\n')

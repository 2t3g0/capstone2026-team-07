"""Bounded, optional SITL diagnostics. Never authorizes or renews flight data.

The receipt / ROS / publication values are copied from the actual decision
callback. Batching adds no accepted timestamp, lease or replacement evidence.
"""
from collections import deque
import math

TOPIC = "/jolgwa/debug/safety_stamp"
CAPACITY = 64
BATCH_SIZE = 16


def validate_diagnostic_profile(*, enabled, require_sim_clock, simulation_only,
                                allow_real_hardware, use_sim_time):
    if type(enabled) is not bool:
        raise ValueError("safety_stamp_diagnostics must be boolean")
    if enabled and (require_sim_clock is not True or simulation_only is not True
                    or allow_real_hardware is not False or use_sim_time is not True):
        raise ValueError("safety_stamp_diagnostics requires the explicit guarded SIM-only clock profile")


def finite_or_none(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


class SafetyStampDiagnostics:
    """Caller serializes access. Fixed queue; overflow is explicit, not success.

    Publishing/draining is independent of the flight-state machine. No record
    is ever consulted when accepting a safety decision or producing a command.
    """
    def __init__(self):
        self._records = deque()
        self._sequence = 0
        self._counts = dict(accepted=0, rejected=0, ignored=0)
        self.dropped_total = 0
        self.publish_failed_records_total = 0

    def record(self, *, outcome, reason, callback_monotonic_s, callback_ros_s,
               publication_sec, publication_nanosec, publication_ros_s,
               transport_delta_s, original_observation_age_s, sequence,
               state, source, active_command_present):
        if outcome not in self._counts:
            raise ValueError("unsupported diagnostic outcome")
        self._sequence += 1
        self._counts[outcome] += 1
        record = dict(
            diagnostic_sequence=self._sequence, outcome=outcome, reason=str(reason)[:240],
            callback_monotonic_s=finite_or_none(callback_monotonic_s),
            callback_ros_s=finite_or_none(callback_ros_s),
            publication_sec=finite_or_none(publication_sec),
            publication_nanosec=finite_or_none(publication_nanosec),
            publication_ros_s=finite_or_none(publication_ros_s),
            transport_delta_s=finite_or_none(transport_delta_s),
            original_observation_age_s=finite_or_none(original_observation_age_s),
            sequence=finite_or_none(sequence), state=finite_or_none(state),
            source=str(source)[:120], active_command_present=active_command_present is True,
        )
        if len(self._records) == CAPACITY:
            self._records.popleft()
            self.dropped_total += 1
        self._records.append(record)

    def drain(self):
        if not self._records:
            return None
        records = [self._records.popleft() for _ in range(min(BATCH_SIZE, len(self._records)))]
        return dict(schema=1, diagnostic_only=True, controller_mode="SIM_ONLY",
            clock_domain="ros_sim_and_host_monotonic_do_not_subtract_domains",
            records=records, totals=dict(self._counts),
            queue=dict(capacity=CAPACITY, batch_limit=BATCH_SIZE, remaining=len(self._records),
                dropped_total=self.dropped_total,
                publish_failed_records_total=self.publish_failed_records_total),
            interpretation="Callback acceptance is separate from active-command authority; "
                           "VehicleControlState.last_error may be historical and idle safety_fresh is false.")

    def publish_failed(self, count):
        if type(count) is not int or not 0 <= count <= BATCH_SIZE:
            raise ValueError("invalid diagnostic batch loss count")
        self.publish_failed_records_total += count

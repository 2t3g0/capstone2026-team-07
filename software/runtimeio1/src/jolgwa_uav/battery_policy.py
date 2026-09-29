"""Internal, monotonic battery evidence policy for approval-bound field tests."""
from dataclasses import dataclass
import math

BATTERY_LEASE_NS = 750_000_000
LOW_NS = 5_000_000_000
RECOVERY_NS = 1_000_000_000


@dataclass
class LowBatteryObservation:
    first_ns: int | None = None
    ok_since_ns: int | None = None
    last_sample_ns: int | None = None
    terminal: bool = False
    detail: str = 'battery_ready'

    def update(self, sample, now_ns, *, battery_fault=False):
        """Return ready/observe/recovered/terminal; never reset a terminal latch."""
        if self.terminal:
            return 'terminal'
        # A late OK cannot undo an expired episode.
        if self.first_ns is not None and (now_ns < self.first_ns or now_ns-self.first_ns >= LOW_NS):
            self.terminal = True
            self.detail = 'battery_low_persistent'
            return 'terminal'
        valid = bool(sample and sample.get('valid') and
                     0 <= now_ns-sample['received_ns'] <= BATTERY_LEASE_NS)
        if not valid:
            self.ok_since_ns = None
            if battery_fault or self.first_ns is not None:
                self.terminal = True
                self.detail = 'battery_severity_missing_or_stale'
                return 'terminal'
            return 'ready'
        state = sample['charge_state']
        if state not in (1, 2, 3, 4, 5, 6) and not sample['fault_bitmask']:
            self.ok_since_ns = None
            if not battery_fault and self.first_ns is None:
                self.detail = 'battery_severity_unsupported'
                return 'ready'
        if sample['fault_bitmask'] or state not in (1, 2):
            self.terminal = True
            self.detail = 'battery_severe_or_unsupported'
            return 'terminal'
        receipt = sample['received_ns']
        if self.last_sample_ns is not None and receipt < self.last_sample_ns:
            self.terminal = True
            self.detail = 'battery_evidence_reversed'
            return 'terminal'
        if self.last_sample_ns is not None and receipt-self.last_sample_ns > BATTERY_LEASE_NS:
            self.ok_since_ns = None
        if state == 2:
            if self.first_ns is None:
                self.first_ns = receipt
            self.ok_since_ns = None
            self.detail = 'battery_low_observing'
            result = 'observe'
        elif self.first_ns is not None:
            if self.ok_since_ns is None:
                self.ok_since_ns = receipt
            if receipt-self.ok_since_ns >= RECOVERY_NS:
                self.first_ns = self.ok_since_ns = None
                self.detail = 'battery_low_recovered'
                result = 'recovered'
            else:
                self.detail = 'battery_low_recovery_observing'
                result = 'observe'
        else:
            self.detail = 'battery_low_recovered' if battery_fault else 'battery_ready'
            result = 'recovered' if battery_fault else 'ready'
        self.last_sample_ns = receipt
        return result


def decode_battery(value, received_ns, epoch):
    fields = ('id', 'charge_state', 'fault_bitmask')
    if any(type(value.get(k)) is not int for k in fields):
        raise ValueError('battery_status_fields_invalid')
    if not 0 <= value['id'] <= 255 or not 0 <= value['charge_state'] <= 255 or not 0 <= value['fault_bitmask'] < 2**32:
        raise ValueError('battery_status_range_invalid')
    cells = value.get('voltages')
    if not isinstance(cells, (list, tuple)) or len(cells) != 10:
        raise ValueError('battery_status_voltages_invalid')
    if any(type(v) is not int or not 0 <= v <= 65535 for v in cells):
        raise ValueError('battery_status_voltage_range_invalid')
    # PX4 may report pack voltage in the first slot when cells are not measured.
    measured = [v for v in cells if v not in (0, 65535)]
    voltage = sum(measured)/1000.0 if measured else math.nan
    remaining = value.get('battery_remaining')
    remaining = remaining if type(remaining) is int and 0 <= remaining <= 100 else None
    return dict(id=value['id'], charge_state=value['charge_state'], remaining_percent=remaining,
                fault_bitmask=value['fault_bitmask'], voltage_v=voltage,
                valid=True, received_ns=received_ns, epoch=epoch)

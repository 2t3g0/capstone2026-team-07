"""Static PX4 Home lifetime, separate from current telemetry freshness.

Humble rclpy does not supply the publisher GID to subscription callbacks. The
manager therefore provides a prefix only after observing exactly one publisher
per topic in the current FastDDS graph. That is an endpoint association, NOT a
cryptographic or per-sample origin proof. Missing/ambiguous identity fails closed.
"""
import math

from .event_home_return import distance, home_from_px4


def single_fastdds_prefix(endpoints, rmw_identifier):
    if rmw_identifier != "rmw_fastrtps_cpp" or len(endpoints) != 1:
        return None
    try:
        gid = tuple(endpoints[0].endpoint_gid)
        if (len(gid) < 16 or any(type(v) is not int or not 0 <= v <= 255 for v in gid)
                or not any(gid[:12])):
            return None
        return bytes(gid[:12])
    except (AttributeError, TypeError, ValueError):
        return None


class StaticHomeReference:
    """One already-armed PX4/DDS epoch; errors latch until manager restart."""
    # Commander publishes unchanged VehicleStatus at nominal 2 Hz. This status-
    # only lease adds a bounded 250 ms delivery margin; it does not change pose,
    # avoidance-decision, graph identity, or VehicleControlState freshness.
    STATUS_LEASE_S = .75
    STATUS_SOURCE_GAP_US = 750_000

    def __init__(self, *, diagnostic_relaxed_timing=False):
        if type(diagnostic_relaxed_timing) is not bool:
            raise ValueError("diagnostic_relaxed_timing must be a bool")
        # The ROS caller must additionally enforce the explicit SIM-only
        # profile. This option never rebinds a failed source/arming epoch.
        self.diagnostic_relaxed_timing = diagnostic_relaxed_timing
        self.timing_warnings = set()
        self.home = None
        self.home_stamp = 0
        self.home_received_at = float("-inf")
        self.home_prefix = None
        self.status_prefix = None
        self.status_stamp = 0
        self.status_received_at = float("-inf")
        self.armed_time = None
        self.armed = False
        self.status_samples = 0
        self.failure = ""

    def invalidate(self, reason):
        self.failure = self.failure or str(reason)

    def observe_home(self, message, prefix, now):
        if self.failure:
            return
        try:
            value = home_from_px4(message)
            if prefix is None:
                raise ValueError("Home single-publisher identity unavailable")
            if not math.isfinite(now) or now < 0:
                raise ValueError("invalid Home receipt clock")
            if self.home_prefix is not None and prefix != self.home_prefix:
                raise ValueError("Home DDS participant changed")
            if message.timestamp < self.home_stamp:
                raise ValueError("Home source clock reset")
            if self.home is not None and distance(value, self.home) > .5:
                raise ValueError("Home local reference changed more than 0.5m")
            if message.timestamp == self.home_stamp:
                if value != self.home:
                    raise ValueError("conflicting Home duplicate timestamp")
                return  # Static sample does not become newly measured.
            self.home, self.home_stamp = value, message.timestamp
            self.home_received_at, self.home_prefix = now, prefix
            if self.status_prefix is not None and prefix != self.status_prefix:
                raise ValueError("Home and status DDS participants differ")
        except (ValueError, TypeError, AttributeError) as exc:
            self.invalidate(exc)

    def observe_status(self, message, prefix, now):
        if self.failure:
            return
        try:
            stamp, armed_time = message.timestamp, message.armed_time
            # DDS converts status.timestamp to its synchronized clock, whereas
            # armed_time remains a PX4 HRT epoch token. Do not compare their
            # magnitudes across clock domains.
            if (type(stamp) is not int or not 0 < stamp < 2**64
                    or type(armed_time) is not int or not 0 <= armed_time < 2**64):
                raise ValueError("invalid PX4 status source/arming timestamp")
            if prefix is None:
                raise ValueError("status single-publisher identity unavailable")
            if not math.isfinite(now) or now < 0:
                raise ValueError("invalid status receipt clock")
            if self.status_prefix is not None and prefix != self.status_prefix:
                raise ValueError("status DDS participant changed")
            if self.home_prefix is not None and prefix != self.home_prefix:
                raise ValueError("Home and status DDS participants differ")
            if self.status_stamp:
                if stamp < self.status_stamp:
                    raise ValueError("status source clock reset")
                if stamp == self.status_stamp:
                    return  # Duplicate cannot extend the current-status lease.
                receipt_gap = now-self.status_received_at
                if not 0 < receipt_gap:
                    raise ValueError("status continuity lost")
                if receipt_gap > self.STATUS_LEASE_S:
                    if not self.diagnostic_relaxed_timing:
                        raise ValueError("status continuity lost")
                    self.timing_warnings.add("status receipt continuity exceeded")
                if stamp-self.status_stamp > self.STATUS_SOURCE_GAP_US:
                    if not self.diagnostic_relaxed_timing:
                        raise ValueError("status source continuity lost")
                    self.timing_warnings.add("status source continuity exceeded")
                if self.armed_time and armed_time != self.armed_time:
                    raise ValueError("PX4 arming epoch changed")
            self.status_stamp, self.status_received_at = stamp, now
            self.status_prefix, self.armed_time = prefix, armed_time
            self.armed = getattr(message, "arming_state", None) == 2 and armed_time > 0
            self.status_samples += 1
        except (ValueError, TypeError, AttributeError) as exc:
            self.invalidate(exc)

    def current_home(self, now, *, home_prefix, status_prefix):
        if (self.failure or self.home is None or self.status_samples < 2
                or not self.armed or not math.isfinite(now)
                or not self.status_age_valid(now)
                or home_prefix is None or status_prefix is None
                or home_prefix != self.home_prefix or status_prefix != self.status_prefix
                or home_prefix != status_prefix or self.home_stamp > self.status_stamp):
            return None
        return self.home

    def status_age_valid(self, now):
        age = now-self.status_received_at
        if not self.diagnostic_relaxed_timing:
            return 0 <= age <= self.STATUS_LEASE_S
        if not math.isfinite(age) or age < 0:
            return False
        if age > self.STATUS_LEASE_S:
            self.timing_warnings.add("current status age exceeded")
        return True

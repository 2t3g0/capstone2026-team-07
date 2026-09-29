"""Opt-in SITL clock evidence; no ROS imports or wall-clock comparisons.

Publisher GID is graph association, not per-message authentication. Repeated
clock values never renew the advance lease. A recovered lease never approves a
mission: the controller separately retires any interrupted active epoch.
"""
import math


MAX_CLOCK_AGE_S = 0.25


def validate_sim_clock_profile(*, required, simulation_only, allow_real_hardware, use_sim_time):
    if type(required) is not bool:
        raise ValueError("require_sim_clock must be boolean")
    if required and (simulation_only is not True or allow_real_hardware is not False
                     or use_sim_time is not True):
        raise ValueError("require_sim_clock requires simulation_only=true, "
                         "allow_real_hardware=false and use_sim_time=true")


def validate_relaxed_timing_profile(*, enabled, required, simulation_only,
                                    allow_real_hardware, use_sim_time):
    """The age-only diagnostic is never valid in a physical clock profile."""
    if type(enabled) is not bool:
        raise ValueError("diagnostic_relaxed_timing must be boolean")
    if enabled and (required is not True or simulation_only is not True
                    or allow_real_hardware is not False or use_sim_time is not True):
        raise ValueError("diagnostic_relaxed_timing requires require_sim_clock=true, "
                         "simulation_only=true, allow_real_hardware=false and use_sim_time=true")


def clock_nanoseconds(sec, nanosec):
    if (type(sec) is not int or type(nanosec) is not int or sec < 0
            or not 0 <= nanosec < 1_000_000_000):
        raise ValueError("invalid /clock fields")
    return sec * 1_000_000_000 + nanosec


def _finite_time(value):
    return type(value) in (int, float) and math.isfinite(value)


class SimClockLease:
    """Two advancing samples plus fresh unique graph evidence are required.

    Regression/invalid data after a positive baseline or a changed bound GID
    latches the clock epoch fault until a new controller instance. A missing
    stream/graph or a stall can recover, but each observe method returns any
    detected interruption *before* accepting a replacement sample. This lets a
    caller retire an active mission even if recovery precedes its next tick.
    """
    def __init__(self, *, diagnostic_relaxed_timing=False):
        if type(diagnostic_relaxed_timing) is not bool:
            raise ValueError("diagnostic_relaxed_timing must be boolean")
        self._diagnostic_relaxed_timing = diagnostic_relaxed_timing
        self.source_ns = None
        self.last_sample_at = None
        self.last_advance_at = None
        self.has_advanced = False
        self.publisher_gid = None
        self.graph_at = None
        self.graph_valid = False
        self.fault = ""

    def error(self, now):
        if self.fault:
            return self.fault
        if not _finite_time(now):
            return "sim clock monotonic receipt unavailable"
        if any(at is not None and now < at for at in (self.last_sample_at, self.graph_at)):
            return "sim clock monotonic receipt regressed"
        if not self.graph_valid or self.graph_at is None:
            return "sim clock unique publisher unavailable"
        if (not self._diagnostic_relaxed_timing
                and now - self.graph_at > MAX_CLOCK_AGE_S):
            return "sim clock publisher graph lease expired"
        if not self.has_advanced or self.last_advance_at is None:
            return "sim clock positive advance unavailable"
        if (not self._diagnostic_relaxed_timing
                and now - self.last_advance_at > MAX_CLOCK_AGE_S):
            return "sim clock advance stalled"
        return None

    def timing_warnings(self, now):
        """Report original age breaches without refreshing either receipt.

        Only the opt-in caller may downgrade these two age failures. Missing,
        invalid, regressed or changed-epoch evidence remains an error, including
        when an earlier age failure would otherwise have hidden that error.
        """
        if not self._diagnostic_relaxed_timing or not _finite_time(now):
            return ()
        warnings = []
        if self.graph_at is not None and now - self.graph_at > MAX_CLOCK_AGE_S:
            warnings.append("sim clock publisher graph lease expired")
        if self.last_advance_at is not None and now - self.last_advance_at > MAX_CLOCK_AGE_S:
            warnings.append("sim clock advance stalled")
        return tuple(warnings)

    def observe_graph(self, gids, now):
        prior = self.error(now)
        if not _finite_time(now) or (self.graph_at is not None and now < self.graph_at):
            self.fault = "sim clock graph monotonic receipt regressed"
            return self.fault
        self.graph_at = now
        valid = (len(gids) == 1 and isinstance(gids[0], (list, tuple))
                 and len(gids[0]) == 24 and any(gids[0][:12])
                 and all(type(value) is int and 0 <= value <= 255 for value in gids[0]))
        self.graph_valid = valid
        if not valid:
            return "sim clock requires exactly one valid publisher GID"
        gid = tuple(gids[0])
        if self.publisher_gid is not None and gid != self.publisher_gid:
            self.fault = "sim clock publisher epoch changed"
            return self.fault
        self.publisher_gid = gid
        return prior

    def observe_clock(self, source_ns, now):
        prior = self.error(now)
        if not _finite_time(now) or (self.last_sample_at is not None and now < self.last_sample_at):
            self.fault = "sim clock monotonic receipt regressed"
            return self.fault
        self.last_sample_at = now
        if type(source_ns) is not int or source_ns <= 0:
            if self.source_ns is not None:
                self.fault = "sim clock became invalid or zero"
            return self.fault or "sim clock positive source unavailable"
        if self.source_ns is not None and source_ns < self.source_ns:
            self.fault = "sim clock source regressed"
            return self.fault
        if self.source_ns is None:
            self.source_ns = source_ns
            return prior or "sim clock first sample is not advance proof"
        if source_ns > self.source_ns:
            self.source_ns = source_ns
            self.last_advance_at = now
            self.has_advanced = True
        return prior or self.error(now)

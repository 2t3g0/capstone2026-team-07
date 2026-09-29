"""A fixed front camera must face travel before any route translation."""
import math


def angle_error(target, heading):
    return math.atan2(math.sin(target-heading), math.cos(target-heading))


class RouteHeadingGate:
    def __init__(self, *, tolerance_rad=math.radians(10), settle_s=.3,
                 heading_lease_s=.25, settled_speed_mps=.1):
        values = (tolerance_rad, settle_s, heading_lease_s, settled_speed_mps)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("heading gate limits must be finite and positive")
        if tolerance_rad > math.radians(15) or heading_lease_s > .25:
            raise ValueError("heading gate exceeds the fixed-camera safety limits")
        self.tolerance, self.settle_s = tolerance_rad, settle_s
        self.lease_s, self.speed_limit = heading_lease_s, settled_speed_mps
        self.reset()

    def reset(self):
        self.key = None
        self.anchor = None
        self.settled_since = None
        self.ready = False
        self.last_now = None

    def assess(self, *, key, position, target_yaw, heading, heading_at,
               speed_mps, observation_at, now):
        if self.key != key:
            self.reset()
            self.key, self.anchor = key, tuple(position)
        if self.last_now is not None and (now < self.last_now or now-self.last_now > self.lease_s):
            self.anchor = tuple(position)
            self.ready, self.settled_since = False, None
        self.last_now = now
        if (not all(math.isfinite(v) for v in (target_yaw, heading, heading_at, speed_mps, now))
                or not 0 <= now-heading_at <= self.lease_s):
            if self.ready:
                self.anchor = tuple(position)
            self.ready, self.settled_since = False, None
            return False, "route_heading_invalid_or_stale"
        if abs(angle_error(target_yaw, heading)) > self.tolerance:
            if self.ready:
                self.anchor = tuple(position)
            self.ready, self.settled_since = False, None
            return False, "route_heading_aligning"
        if self.ready:
            return True, "route_heading_ready"
        if speed_mps > self.speed_limit:
            self.settled_since = None
            return False, "route_heading_braking"
        if self.settled_since is None:
            self.settled_since = now
        deadline = self.settled_since + self.settle_s
        if now < deadline:
            return False, "route_heading_settling"
        if not math.isfinite(observation_at) or not deadline < observation_at <= now:
            return False, "route_heading_waiting_for_new_depth"
        self.ready = True
        return True, "route_heading_ready"

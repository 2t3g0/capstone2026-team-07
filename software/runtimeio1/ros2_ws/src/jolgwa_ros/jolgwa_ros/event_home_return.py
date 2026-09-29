"""Conservative, offline-testable return planning. No flight IO or native RTL.

Visited odometry points are a route record, not a present clearance certificate:
every return leg still requires the live depth/heading controller interlocks.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


def point3(value):
    if len(value) != 3:
        raise ValueError("position must contain three NED coordinates")
    result = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in result):
        raise ValueError("position must be finite")
    return result


def distance(a, b):
    return math.sqrt(sum((x-y)**2 for x, y in zip(point3(a), point3(b))))


def home_from_px4(message):
    if not all(getattr(message, name, None) is True
               for name in ("valid_lpos", "valid_hpos", "valid_alt")):
        raise ValueError("PX4 Home local/global/altitude validity is unconfirmed")
    stamp = getattr(message, "timestamp", 0)
    if type(stamp) is not int or stamp <= 0:
        raise ValueError("PX4 Home source timestamp is invalid")
    return point3((message.x, message.y, message.z))


def owned_navigation_error(state, *, mission_id, received_at, now, timeout_s=.5):
    if (not all(isinstance(v, (int, float)) and math.isfinite(v)
                for v in (received_at, now, timeout_s))
            or not 0 <= now-received_at <= timeout_s):
        return "controller_state_not_current"
    if state is None or getattr(state, "mission_id", None) != mission_id:
        return "controller_mission_identity_mismatch"
    required = ("command_output_enabled", "approved", "connected", "preflight_checks_pass",
                "position_valid", "armed", "offboard", "route_heading_gate_enabled",
                "terminal_owned_handoff_enabled")
    if any(getattr(state, key, None) is not True for key in required):
        return "owned_offboard_or_heading_gate_unavailable"
    if (getattr(state, "manual_override", None) is not False
            or getattr(state, "flight_epoch_retired", None) is not False):
        return "manual_or_retired_flight_epoch"
    if getattr(state, "active_authority", None) not in {
        "LLM_ROUTE", "JETSON_SAFETY", "JETSON_EVENT_CAPTURE"
    }:
        return "autonomous_authority_not_owned"
    try:
        point3(state.position_ned_m)
    except (ValueError, TypeError, AttributeError):
        return "invalid_navigation_position"
    return ""


def ready_to_reverse(state):
    try:
        return (state is not None and getattr(state, "avoidance_active", None) is False
                and getattr(state, "jetson_safety_fresh", None) is True
                and getattr(state, "jetson_safety_state", None) == "CLEAR"
                and getattr(state, "heading_fresh", None) is True
                and math.isfinite(float(getattr(state, "heading_rad", math.nan))))
    except (ValueError, TypeError):
        return False


def return_yaw(current, target, fallback_heading):
    current, target = point3(current), point3(target)
    dx, dy = target[0]-current[0], target[1]-current[1]
    if math.hypot(dx, dy) > .05:
        return math.atan2(dy, dx)
    if not math.isfinite(float(fallback_heading)):
        raise ValueError("vertical-only return leg requires a current heading")
    return float(fallback_heading)


def merge_collinear_return_targets(current, targets, maximum_deviation_m=.05):
    """Join only near-collinear, monotonically visited 3-D segments in O(n).

    Keeping a .75m goal for each breadcrumb prevents the existing controller
    from completing its obstacle-tail +5m pass. A longer *visited straight*
    segment provides that space without inventing a path through a corner.
    Half-tolerance against the initial ray bounds final-chord deviation by the
    full tolerance; a final exact check is still required. This geometric
    sampling bound does not certify present obstacle clearance.
    """
    if not math.isfinite(maximum_deviation_m) or not 0 <= maximum_deviation_m <= .05:
        raise ValueError("return collinearity tolerance must be in [0, .05] m")
    start = point3(current)
    values = [point3(value) for value in targets]
    result, group = [], []
    axis = None
    previous_progress = 0.

    def close_group(anchor, points):
        if not points:
            return []
        endpoint = points[-1]
        vector = tuple(endpoint[i]-anchor[i] for i in range(3))
        length = math.sqrt(sum(v*v for v in vector))
        if length <= 1e-8:
            return points
        direction = tuple(v/length for v in vector)
        last_progress = -1e-8
        for point in points:
            delta = tuple(point[i]-anchor[i] for i in range(3))
            progress = sum(delta[i]*direction[i] for i in range(3))
            residual = math.sqrt(sum((delta[i]-progress*direction[i])**2 for i in range(3)))
            if (residual > maximum_deviation_m+1e-9 or progress < last_progress-1e-9
                    or not -1e-8 <= progress <= length+1e-8):
                return points
            last_progress = progress
        return [endpoint]

    for target in values:
        delta = tuple(target[i]-start[i] for i in range(3))
        length = math.sqrt(sum(v*v for v in delta))
        if axis is None:
            if length <= 1e-8:
                continue
            axis = tuple(v/length for v in delta)
            previous_progress = length
            group = [target]
            continue
        progress = sum(delta[i]*axis[i] for i in range(3))
        residual = math.sqrt(sum((delta[i]-progress*axis[i])**2 for i in range(3)))
        if residual <= maximum_deviation_m*.5+1e-9 and progress >= previous_progress-1e-9:
            group.append(target)
            previous_progress = progress
            continue
        result.extend(close_group(start, group))
        start = group[-1]
        delta = tuple(target[i]-start[i] for i in range(3))
        length = math.sqrt(sum(v*v for v in delta))
        group = [target]
        axis = None if length <= 1e-8 else tuple(v/length for v in delta)
        previous_progress = length
    result.extend(close_group(start, group))
    return tuple(result)


@dataclass(frozen=True)
class ReturnTracePolicy:
    spacing_m: float = .75
    maximum_gap_m: float = 2.5
    maximum_receipt_gap_s: float = .5
    maximum_points: int = 4096
    home_radius_m: float = 1.

    def __post_init__(self):
        if (not all(math.isfinite(x) and x > 0 for x in
                    (self.spacing_m, self.maximum_gap_m, self.maximum_receipt_gap_s, self.home_radius_m))
                or self.spacing_m > self.maximum_gap_m or self.maximum_receipt_gap_s > .5
                or type(self.maximum_points) is not int or not 2 <= self.maximum_points <= 10000):
            raise ValueError("invalid bounded return trace policy")


class VisitedReturnTrace:
    def __init__(self, home_ned, policy=None):
        self.home_ned = point3(home_ned)
        self.policy = policy or ReturnTracePolicy()
        self.points = []
        self._last_received = None
        self._last_position = None
        self.failure = ""

    def observe(self, position, received_at):
        if self.failure:
            return False
        try:
            position = point3(position)
            if not math.isfinite(received_at) or received_at < 0:
                raise ValueError("invalid_trace_receipt_time")
            if self._last_received is not None:
                if not 0 < received_at-self._last_received <= self.policy.maximum_receipt_gap_s:
                    raise ValueError("visited_trace_receipt_gap_or_clock_reset")
                if distance(position, self._last_position) > self.policy.maximum_gap_m:
                    raise ValueError("visited_trace_position_jump")
            elif math.hypot(position[0]-self.home_ned[0], position[1]-self.home_ned[1]) > self.policy.home_radius_m:
                raise ValueError("first_airborne_position_not_near_confirmed_home")
            self._last_received, self._last_position = received_at, position
            if not self.points or distance(position, self.points[-1]) >= self.policy.spacing_m:
                if len(self.points) >= self.policy.maximum_points:
                    raise ValueError("visited_trace_capacity_exceeded")
                self.points.append(position)
            return True
        except (ValueError, TypeError) as exc:
            self.failure = str(exc)
            return False

    def reverse_targets(self, current, acceptance_radius_m):
        if self.failure or not self.points:
            raise ValueError(self.failure or "no_verified_visited_trace")
        current = point3(current)
        if (not math.isfinite(acceptance_radius_m)
                or not 0 < acceptance_radius_m < self.policy.spacing_m):
            raise ValueError("return acceptance radius must be smaller than trace spacing")
        if self._last_position is None or distance(current, self._last_position) > self.policy.maximum_gap_m:
            raise ValueError("return_start_disconnected_from_visited_trace")
        # Do not shortcut the stored route/loops. Live depth must still validate
        # the connecting legs: sampled odometry is not a corridor certificate.
        points = list(reversed(self.points))
        while len(points) > 1 and distance(current, points[0]) <= acceptance_radius_m:
            points.pop(0)
        return tuple(points)


def near_home_for_landing(position, home, *, radius_m=1., maximum_height_m=5.):
    try:
        if not (math.isfinite(radius_m) and 0 < radius_m <= 2
                and math.isfinite(maximum_height_m) and 0 < maximum_height_m <= 10):
            return False
        position, home = point3(position), point3(home)
        return (math.hypot(position[0]-home[0], position[1]-home[1]) <= radius_m
                and -.5 <= home[2]-position[2] <= maximum_height_m)
    except (ValueError, TypeError):
        return False

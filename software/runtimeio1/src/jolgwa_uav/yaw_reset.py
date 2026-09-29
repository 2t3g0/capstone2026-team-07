"""Bounded inference from aggregate MAVLink resets, never per-axis FC truth.

Only the approval-bound ROS bridge supplies the context. No route rebasing or
permission is granted here. All comparisons are inclusive at their bounds.
"""
import math

POLICY = "BOUNDED_YAW_V1"


def euler(q):
    w, x, y, z = (float(v) for v in q)
    return (math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
            math.asin(max(-1., min(1., 2*(w*y-z*x)))),
            math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


def ned_velocity(q, velocity, child_frame):
    if child_frame == 1:
        return tuple(velocity)
    w, x, y, z = q
    a, b, c = velocity
    return ((1-2*(y*y+z*z))*a+2*(x*y-w*z)*b+2*(x*z+w*y)*c,
            2*(x*y+w*z)*a+(1-2*(x*x+z*z))*b+2*(y*z-w*x)*c,
            2*(x*z-w*y)*a+2*(y*z+w*x)*b+(1-2*(x*x+y*y))*c)


def angle_delta(a, b):
    return math.atan2(math.sin(a-b), math.cos(a-b))


def classify(previous, current, received_delta_ns, angular_rates):
    """Return numerical evidence and reason, with no mutation on rejection."""
    dt = (current['time_usec']-previous['time_usec'])/1e6
    old, new = euler(previous['q']), euler(current['q'])
    angles = [abs(math.degrees(angle_delta(a, b))) for a, b in zip(new, old)]
    residual = math.dist(current['position'], tuple(
        p+v*dt for p, v in zip(previous['position'], previous['proof_velocity'])))
    velocity_delta = math.dist(current['proof_velocity'], previous['proof_velocity'])
    rate = math.sqrt(sum(v*v for v in angular_rates))
    evidence = dict(sample_interval_ms=dt*1000, receipt_interval_ms=received_delta_ns/1e6,
                    yaw_delta_deg=angles[2], roll_delta_deg=angles[0], pitch_delta_deg=angles[1],
                    angular_speed_rad_s=rate, position_residual_m=residual,
                    velocity_delta_m_s=velocity_delta)
    reason = ''
    if not all(math.isfinite(v) for v in evidence.values()):
        reason = 'nonfinite_reset_evidence'
    elif not 0 < dt <= .150 or not 0 < received_delta_ns <= 150_000_000:
        reason = 'reset_sample_interval'
    # Small floating point round-off is permitted only at angle conversions.
    elif not 3.-1e-12 <= angles[2] <= 15.+1e-12:
        reason = 'reset_yaw_outside_bounds'
    elif max(angles[:2]) > 3.+1e-12:
        reason = 'reset_roll_pitch_changed'
    elif rate > .2:
        reason = 'reset_angular_speed'
    elif residual > .1:
        reason = 'reset_position_discontinuity'
    elif velocity_delta > .2:
        reason = 'reset_velocity_discontinuity'
    return {key: (value if math.isfinite(value) else None)
            for key, value in evidence.items()}, reason

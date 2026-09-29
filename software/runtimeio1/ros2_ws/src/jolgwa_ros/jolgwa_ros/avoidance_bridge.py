from __future__ import annotations

import math


def body_frd_to_ned(
    velocity_body_mps: tuple[float, float, float],
    yaw_rad: float,
) -> tuple[float, float, float]:
    """Rotate body forward-right-down velocity into world north-east-down."""

    if len(velocity_body_mps) != 3:
        raise ValueError("velocity_body_mps must have exactly three values")
    forward, right, down = (float(value) for value in velocity_body_mps)
    values = (forward, right, down, float(yaw_rad))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("body velocity and yaw must be finite")
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    return (
        cosine * forward - sine * right,
        sine * forward + cosine * right,
        down,
    )


def avoidance_body_velocity(
    direction: str,
    *,
    lateral_velocity_body_mps: float,
    vertical_velocity_ned_mps: float,
) -> tuple[float, float, float]:
    """Validate a Jetson direction and return a body FRD escape velocity."""

    direction = str(direction).upper()
    lateral = float(lateral_velocity_body_mps)
    vertical = float(vertical_velocity_ned_mps)
    if not math.isfinite(lateral) or not math.isfinite(vertical):
        raise ValueError("avoidance velocity must be finite")
    if direction == "LEFT" and lateral < 0.0 and abs(vertical) <= 1e-6:
        return (0.0, lateral, 0.0)
    if direction == "RIGHT" and lateral > 0.0 and abs(vertical) <= 1e-6:
        return (0.0, lateral, 0.0)
    if direction == "UP" and vertical < 0.0 and abs(lateral) <= 1e-6:
        return (0.0, 0.0, vertical)
    if direction == "DOWN" and vertical > 0.0 and abs(lateral) <= 1e-6:
        return (0.0, 0.0, vertical)
    raise ValueError("direction and avoidance velocity signs disagree")

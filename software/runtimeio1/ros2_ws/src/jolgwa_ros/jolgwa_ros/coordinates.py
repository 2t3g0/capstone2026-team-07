import math
from typing import Sequence


def enu_to_ned(point: Sequence[float]) -> tuple[float, float, float]:
    if len(point) != 3:
        raise ValueError("point must have exactly three coordinates")
    east, north, up = (float(value) for value in point)
    return north, east, -up


def yaw_enu_deg_to_ned_rad(yaw_deg: float) -> float:
    yaw = math.radians(90.0 - float(yaw_deg))
    return math.atan2(math.sin(yaw), math.cos(yaw))


def distance_ned(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))

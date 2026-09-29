from collections.abc import Iterator


def patrol_waypoint_indices(point_count: int) -> Iterator[int]:
    """Yield patrol targets after the vehicle has reached waypoint zero."""
    if point_count < 2:
        raise ValueError("a patrol route requires at least two waypoints")
    yield from range(1, point_count)
    while True:
        yield from range(point_count)


def patrol_target_count(point_count: int, laps: int) -> int:
    if point_count < 2 or laps < 1:
        raise ValueError("point_count must be at least two and laps must be positive")
    return point_count * laps - 1

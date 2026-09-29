"""Mission planning and deterministic control core for the Jolgwa UAV."""

__all__ = ["MissionPlan"]


def __getattr__(name: str):
    # The Jetson native-depth service does not use mission planning. Keep its
    # imports independent from the planner's Python 3.11 StrEnum dependency.
    if name == "MissionPlan":
        from .models import MissionPlan
        return MissionPlan
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

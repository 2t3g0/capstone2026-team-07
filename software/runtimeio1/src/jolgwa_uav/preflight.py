from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .models import EventName, MissionPlan, PlanStatus, VehicleSnapshot


class IssueCode(StrEnum):
    VEHICLE_DISCONNECTED = "VEHICLE_DISCONNECTED"
    GPS_INVALID = "GPS_INVALID"
    BATTERY_LOW = "BATTERY_LOW"
    ZONE_UNAVAILABLE = "ZONE_UNAVAILABLE"
    EVENT_UNAVAILABLE = "EVENT_UNAVAILABLE"
    PLAN_NOT_EXECUTABLE = "PLAN_NOT_EXECUTABLE"


@dataclass(frozen=True, slots=True)
class PreflightIssue:
    code: IssueCode
    detail: str


@dataclass(frozen=True, slots=True)
class MissionCapabilities:
    zones: frozenset[str] = frozenset({"A", "B", "C"})
    detectable_events: frozenset[EventName] = frozenset(
        {EventName.FALL, EventName.CALL_FOR_HELP}
    )
    minimum_battery_percent: float = 30.0
    require_gps: bool = True


@dataclass(frozen=True, slots=True)
class PreflightResult:
    issues: tuple[PreflightIssue, ...]

    @property
    def can_execute(self) -> bool:
        return not self.issues


def validate_preflight(
    plan: MissionPlan,
    vehicle: VehicleSnapshot,
    capabilities: MissionCapabilities | None = None,
) -> PreflightResult:
    """Validate an LLM proposal against deterministic runtime capabilities."""

    capabilities = capabilities or MissionCapabilities()
    issues: list[PreflightIssue] = []
    if plan.status is not PlanStatus.OK:
        issues.append(
            PreflightIssue(IssueCode.PLAN_NOT_EXECUTABLE, f"plan status is {plan.status}")
        )
        return PreflightResult(tuple(issues))

    if not vehicle.connected:
        issues.append(
            PreflightIssue(IssueCode.VEHICLE_DISCONNECTED, "PX4 is not connected")
        )
    if capabilities.require_gps and not vehicle.gps_valid:
        issues.append(PreflightIssue(IssueCode.GPS_INVALID, "GPS position is invalid"))
    if (
        vehicle.battery_percent is None
        or vehicle.battery_percent < capabilities.minimum_battery_percent
    ):
        issues.append(
            PreflightIssue(
                IssueCode.BATTERY_LOW,
                "battery is unavailable or below "
                f"{capabilities.minimum_battery_percent:.0f}%",
            )
        )

    unavailable_zones = sorted(set(plan.patrol_zones) - capabilities.zones)
    if unavailable_zones:
        issues.append(
            PreflightIssue(
                IssueCode.ZONE_UNAVAILABLE,
                f"zones are not configured: {', '.join(unavailable_zones)}",
            )
        )

    unavailable_events = sorted(
        event.value
        for event in plan.expanded_events()
        if event not in capabilities.detectable_events
    )
    if unavailable_events:
        issues.append(
            PreflightIssue(
                IssueCode.EVENT_UNAVAILABLE,
                "detectors are not available: " + ", ".join(unavailable_events),
            )
        )
    return PreflightResult(tuple(issues))

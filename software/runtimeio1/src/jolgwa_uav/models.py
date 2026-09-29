from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlanStatus(StrEnum):
    OK = "OK"
    NEED_CLARIFICATION = "NEED_CLARIFICATION"
    UNSUPPORTED = "UNSUPPORTED"


class LimitType(StrEnum):
    LAPS = "LAPS"
    DURATION_MINUTES = "DURATION_MINUTES"


class EventName(StrEnum):
    FALL = "FALL"
    CALL_FOR_HELP = "CALL_FOR_HELP"
    VIOLENCE = "VIOLENCE"
    RESTRICTED_ENTRY = "RESTRICTED_ENTRY"
    FIRE = "FIRE"
    TRAFFIC_ACCIDENT = "TRAFFIC_ACCIDENT"
    ALL = "ALL"


CANONICAL_EVENTS = tuple(event for event in EventName if event is not EventName.ALL)


class ResponseAction(StrEnum):
    TRACK = "TRACK"
    RECORD = "RECORD"
    ALERT = "ALERT"


class AfterResponse(StrEnum):
    RETURN_HOME = "RETURN_HOME"
    RESUME_PATROL = "RESUME_PATROL"


class PatrolLimit(StrictModel):
    type: LimitType
    value: Annotated[int, Field(gt=0, le=1440)]


class RouteFrame(StrictModel):
    site_id: str = Field(min_length=1, max_length=80)
    horizontal: str = Field(pattern="^SITE_ENU_WGS84$")
    reference_latitude_deg: float = Field(ge=-90.0, le=90.0)
    reference_longitude_deg: float = Field(ge=-180.0, le=180.0)
    vertical: str = Field(pattern="^HOME_RELATIVE$")
    version: int = Field(default=1, ge=1, le=1)


class MissionPlan(StrictModel):
    status: PlanStatus
    patrol_zones: list[str] = Field(default_factory=list)
    patrol_limit: PatrolLimit | None = None
    route_id: str | None = Field(default=None, min_length=1, max_length=80)
    route_revision: int | None = Field(default=None, ge=1)
    route_name: str | None = Field(default=None, min_length=1, max_length=80)
    route_waypoints_enu: list[list[float]] = Field(
        default_factory=list, max_length=300
    )
    route_frame: RouteFrame | None = None
    simulation_only: bool | None = None
    monitor_events: list[EventName] = Field(default_factory=list)
    response_rules: dict[str, list[ResponseAction]] = Field(default_factory=dict)
    after_response: AfterResponse | None = None
    missing_fields: list[str] = Field(default_factory=list)
    unsupported_values: list[str] = Field(default_factory=list)
    message: str | None = None

    @model_validator(mode="before")
    @classmethod
    def default_ok_patrol_limit(cls, data):
        if isinstance(data, dict) and data.get("status") == PlanStatus.OK:
            if data.get("patrol_limit") is None:
                data = dict(data)
                data["patrol_limit"] = {"type": LimitType.LAPS, "value": 1}
        return data

    @model_validator(mode="after")
    def validate_contract(self) -> MissionPlan:
        if len(set(self.patrol_zones)) != len(self.patrol_zones):
            raise ValueError("patrol_zones must not contain duplicates")
        invalid_zones = set(self.patrol_zones) - {"A", "B", "C"}
        if invalid_zones:
            raise ValueError(f"unsupported patrol zones: {sorted(invalid_zones)}")
        if EventName.ALL in self.monitor_events and len(self.monitor_events) != 1:
            raise ValueError("ALL cannot be combined with individual monitor events")

        valid_rule_keys = {event.value for event in CANONICAL_EVENTS} | {"DEFAULT"}
        invalid_rule_keys = set(self.response_rules) - valid_rule_keys
        if invalid_rule_keys:
            raise ValueError(f"unsupported response rule keys: {sorted(invalid_rule_keys)}")
        if any(not actions for actions in self.response_rules.values()):
            raise ValueError("response rule action lists must not be empty")

        if self.status is PlanStatus.OK:
            missing: list[str] = []
            if not self.patrol_zones and self.route_id is None:
                missing.append("patrol_zones_or_route_id")
            if self.patrol_zones and self.route_id is not None:
                raise ValueError("choose patrol_zones or route_id, not both")
            if self.patrol_limit is None:
                missing.append("patrol_limit")
            if self.after_response is None:
                missing.append("after_response")
            if missing:
                raise ValueError(f"OK plan is missing required fields: {missing}")
            if self.missing_fields or self.unsupported_values:
                raise ValueError("OK plan cannot contain error detail fields")
        elif self.status is PlanStatus.NEED_CLARIFICATION:
            if not self.missing_fields:
                raise ValueError("NEED_CLARIFICATION requires missing_fields")
        elif self.status is PlanStatus.UNSUPPORTED:
            if not self.unsupported_values:
                raise ValueError("UNSUPPORTED requires unsupported_values")
        return self

    def expanded_events(self) -> tuple[EventName, ...]:
        if EventName.ALL in self.monitor_events:
            return CANONICAL_EVENTS
        return tuple(self.monitor_events)

    def actions_for(self, event: EventName) -> tuple[ResponseAction, ...]:
        actions = self.response_rules.get(event.value)
        if actions is None:
            actions = self.response_rules.get("DEFAULT", [])
        return tuple(actions)


class VehicleSnapshot(StrictModel):
    timestamp: float
    frame_id: str = "map"
    position_enu_m: tuple[float, float, float] | None = None
    battery_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    gps_valid: bool = False
    armed: bool = False
    flight_mode: str = "UNKNOWN"
    connected: bool = False


class PerceptionEvent(StrictModel):
    timestamp: float
    event: EventName
    confidence: float = Field(ge=0.0, le=1.0)
    track_id: str | None = None
    call_sign: str | None = None
    distance_m: float | None = Field(default=None, ge=0.0)
    source: str


class AvailableRoute(StrictModel):
    id: str = Field(min_length=1, max_length=80)
    revision: int = Field(ge=1)
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=300)
    waypoint_count: int = Field(ge=2, le=300)


class PlanningContext(StrictModel):
    schema_version: str = "1.0"
    source: str = "operator"
    available_zones: list[str] = Field(default_factory=lambda: ["A", "B", "C"])
    available_routes: list[AvailableRoute] = Field(default_factory=list)
    detectable_events: list[EventName] = Field(
        default_factory=lambda: [EventName.FALL, EventName.CALL_FOR_HELP]
    )
    vehicle: VehicleSnapshot | None = None

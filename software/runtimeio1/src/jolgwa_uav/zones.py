from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .models import LimitType, MissionPlan, PlanStatus


@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float
    z: float
    yaw_deg: float | None = None


@dataclass(frozen=True)
class Zone:
    zone_id: str
    name: str
    waypoints: tuple[Waypoint, ...]


@dataclass(frozen=True)
class Route:
    frame_id: str
    speed_mps: float
    acceptance_radius_m: float
    waypoints_per_lap: tuple[Waypoint, ...]
    lap_target: int | None
    duration_s: float | None


class ZoneCatalog:
    def __init__(
        self,
        frame_id: str,
        speed_mps: float,
        acceptance_radius_m: float,
        zones: dict[str, Zone],
        geofence_polygon: tuple[tuple[float, float], ...] = (),
    ) -> None:
        self.frame_id = frame_id
        self.speed_mps = speed_mps
        self.acceptance_radius_m = acceptance_radius_m
        self.zones = zones
        self.geofence_polygon = geofence_polygon

    @classmethod
    def load(cls, path: str | Path) -> ZoneCatalog:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        defaults = raw["defaults"]
        geofence_polygon = tuple(
            (float(point[0]), float(point[1]))
            for point in (raw.get("geofence") or {}).get("polygon_enu", [])
        )
        if geofence_polygon and len(geofence_polygon) < 3:
            raise ValueError("geofence polygon requires at least three points")
        zones: dict[str, Zone] = {}
        for zone_id, value in raw["zones"].items():
            points = tuple(
                Waypoint(
                    x=float(point[0]),
                    y=float(point[1]),
                    z=float(point[2]),
                    yaw_deg=float(point[3]) if len(point) > 3 else None,
                )
                for point in value["waypoints"]
            )
            if len(points) < 3:
                raise ValueError(f"zone {zone_id} requires at least three waypoints")
            if geofence_polygon:
                for point in points:
                    if not _contains_xy(geofence_polygon, point.x, point.y):
                        raise ValueError(
                            f"zone {zone_id} waypoint is outside the site geofence"
                        )
            zones[zone_id] = Zone(zone_id, value.get("name", zone_id), points)
        return cls(
            frame_id=raw.get("frame_id", "map"),
            speed_mps=float(defaults["speed_mps"]),
            acceptance_radius_m=float(defaults["acceptance_radius_m"]),
            zones=zones,
            geofence_polygon=geofence_polygon,
        )

    def build_route(self, plan: MissionPlan) -> Route:
        if plan.status is not PlanStatus.OK or plan.patrol_limit is None:
            raise ValueError("only an OK mission plan can be converted to a route")
        points: list[Waypoint] = []
        for zone_id in plan.patrol_zones:
            try:
                points.extend(self.zones[zone_id].waypoints)
            except KeyError as exc:
                raise ValueError(f"zone {zone_id} has no route definition") from exc

        if plan.patrol_limit.type is LimitType.LAPS:
            lap_target = plan.patrol_limit.value
            duration_s = None
        else:
            lap_target = None
            duration_s = float(plan.patrol_limit.value * 60)
        return Route(
            frame_id=self.frame_id,
            speed_mps=self.speed_mps,
            acceptance_radius_m=self.acceptance_radius_m,
            waypoints_per_lap=tuple(points),
            lap_target=lap_target,
            duration_s=duration_s,
        )


def _contains_xy(
    polygon: tuple[tuple[float, float], ...], x: float, y: float
) -> bool:
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        cross = (x - previous_x) * (current_y - previous_y) - (
            y - previous_y
        ) * (current_x - previous_x)
        if abs(cross) <= 1e-7 and min(previous_x, current_x) - 1e-7 <= x <= max(
            previous_x, current_x
        ) + 1e-7 and min(previous_y, current_y) - 1e-7 <= y <= max(
            previous_y, current_y
        ) + 1e-7:
            return True
        if (current_y > y) != (previous_y > y):
            intersection_x = (previous_x - current_x) * (y - current_y) / (
                previous_y - current_y
            ) + current_x
            if x < intersection_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside

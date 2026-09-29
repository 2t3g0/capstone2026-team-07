import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


@dataclass(frozen=True)
class RoutePoint:
    east: float
    north: float
    up: float
    yaw_enu_deg: float = 0.0


class RouteCatalog:
    def __init__(
        self,
        zones: dict[str, tuple[RoutePoint, ...]],
        geofence_polygon: tuple[tuple[float, float], ...] = (),
    ) -> None:
        self.zones = zones
        self.geofence_polygon = geofence_polygon

    @classmethod
    def load(cls, path: str) -> "RouteCatalog":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if raw.get("coordinate_convention") != "ENU":
            raise ValueError("zone catalog must use ENU coordinates")
        geofence_polygon = tuple(
            (float(point[0]), float(point[1]))
            for point in (raw.get("geofence") or {}).get("polygon_enu", [])
        )
        if geofence_polygon and len(geofence_polygon) < 3:
            raise ValueError("geofence polygon requires at least three points")
        zones = {}
        for zone_id, zone in raw.get("zones", {}).items():
            points = []
            for values in zone.get("waypoints", []):
                if len(values) not in (3, 4):
                    raise ValueError("waypoint must contain x, y, z and optional yaw")
                points.append(
                    RoutePoint(
                        float(values[0]),
                        float(values[1]),
                        float(values[2]),
                        float(values[3]) if len(values) == 4 else 0.0,
                    )
                )
            if not points:
                raise ValueError("zone {} has no waypoints".format(zone_id))
            if geofence_polygon:
                for point in points:
                    if not _contains_xy(
                        geofence_polygon, point.east, point.north
                    ):
                        raise ValueError(
                            "zone {} waypoint is outside the site geofence".format(
                                zone_id
                            )
                        )
            zones[str(zone_id)] = tuple(points)
        return cls(zones, geofence_polygon)

    def points_for_plan(
        self,
        plan_json: str,
        explicit_points: Iterable[object] = (),
    ) -> tuple[RoutePoint, ...]:
        plan = self._approved_plan(plan_json)
        explicit = tuple(explicit_points)
        source = plan.get("route_waypoints_enu", [])
        if not isinstance(source, list):
            raise ValueError("route_waypoints_enu must be an array")
        landmarks = plan.get("landmark_route", [])
        if not isinstance(landmarks, list):
            raise ValueError("landmark_route must be an array")
        self._validate_action_support(plan, landmarks)
        route_id = plan.get("route_id")
        if source or route_id is not None or landmarks:
            # Dashboard routes execute exactly the snapshot that was approved.
            # Never resolve its ID again or replace it with action-goal points.
            if explicit:
                raise ValueError("explicit points cannot replace an approved route snapshot")
            if not isinstance(route_id, str) or not route_id.strip() or len(route_id) > 80:
                raise ValueError("approved route snapshot requires route_id")
            revision = plan.get("route_revision")
            if type(revision) is not int or revision < 1:
                raise ValueError("approved route snapshot requires positive integer route_revision")
            if plan.get("patrol_zones"):
                raise ValueError("choose patrol_zones or an approved route snapshot, not both")
            if landmarks and not route_id.startswith("generated-"):
                raise ValueError("landmark routes require a resolved generated route_id")
            points = self._snapshot_points(source)
            self._validate_geofence(points)
            return points
        if plan.get("route_revision") is not None:
            raise ValueError("route_revision requires an approved route snapshot")
        if explicit:
            coordinates = tuple(
                (float(point.x), float(point.y), float(point.z))
                for point in explicit
            )
            headings = _route_headings_enu_deg(coordinates)
            points = tuple(
                RoutePoint(east, north, up, heading)
                for (east, north, up), heading in zip(coordinates, headings)
            )
            self._validate_geofence(points)
            return points

        zones = plan.get("patrol_zones")
        if not isinstance(zones, list) or not zones:
            raise ValueError("plan has no patrol_zones")
        points = []
        for zone_id in zones:
            if zone_id not in self.zones:
                raise ValueError("unknown patrol zone: {}".format(zone_id))
            points.extend(self.zones[zone_id])
        return tuple(points)

    @staticmethod
    def _validate_action_support(plan: dict, landmarks: list) -> None:
        actions = plan.get("route_actions", [])
        if not isinstance(actions, list):
            raise ValueError("route_actions must be an array")
        if actions:
            raise ValueError("route_actions are not executable: CAPTURE_STILL requires a camera consumer and storage ACK")
        if len(landmarks) > 12:
            raise ValueError("landmark_route may contain at most 12 steps")
        for step in landmarks:
            if not isinstance(step, dict):
                raise ValueError("landmark_route steps must be objects")
            identifier = step.get("landmark_id")
            if not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 80:
                raise ValueError("landmark_route steps require a valid landmark_id")
            if step.get("action") not in ("PASS", "ORBIT"):
                raise ValueError("landmark action is not executable: only resolved PASS/ORBIT routes are supported; CAPTURE_STILL requires storage ACK")

    @staticmethod
    def _approved_plan(plan_json: str) -> dict:
        plan = json.loads(plan_json)
        if not isinstance(plan, dict):
            raise ValueError("plan must be a JSON object")
        if plan.get("status") != "OK":
            raise ValueError("only an OK plan can be executed")
        if plan.get("request_purpose", "EXECUTE_MISSION") != "EXECUTE_MISSION":
            raise ValueError("only EXECUTE_MISSION plans can be executed; CREATE_ROUTE is draft-only")
        return plan

    @staticmethod
    def _snapshot_points(source: list) -> tuple[RoutePoint, ...]:
        if not 2 <= len(source) <= 300:
            raise ValueError("route_waypoints_enu must contain 2 to 300 resolved points")
        numeric_points = []
        for index, values in enumerate(source):
            if not isinstance(values, list) or len(values) not in (3, 4):
                raise ValueError("approved route waypoint %d requires east, north, up and optional yaw" % index)
            if any(type(value) not in (int, float) for value in values):
                raise ValueError("approved route waypoint %d values must be numbers" % index)
            try:
                numeric = tuple(float(value) for value in values)
            except OverflowError as exc:
                raise ValueError("approved route waypoint values must be finite") from exc
            if not all(math.isfinite(value) for value in numeric):
                raise ValueError("approved route waypoint %d values must be finite" % index)
            # Match the dashboard's approved resolved-route altitude contract.
            # This does not replace approval, shape, revision or geofence checks.
            if not 1.0 <= numeric[2] <= 120.0:
                raise ValueError("approved route waypoint %d altitude must be between 1 and 120 m" % index)
            if len(numeric) == 4 and not -180.0 <= numeric[3] <= 180.0:
                raise ValueError("approved route waypoint %d yaw must be between -180 and 180 degrees" % index)
            numeric_points.append(numeric)
        headings = _route_headings_enu_deg(tuple(point[:3] for point in numeric_points))
        return tuple(
            RoutePoint(*values[:3], values[3] if len(values) == 4 else heading)
            for values, heading in zip(numeric_points, headings)
        )

    def _validate_geofence(self, points: Iterable[RoutePoint]) -> None:
        if not self.geofence_polygon:
            return
        for point in points:
            if not _contains_xy(
                self.geofence_polygon, point.east, point.north
            ):
                raise ValueError("explicit waypoint is outside the site geofence")

    @staticmethod
    def patrol_limit(plan_json: str) -> tuple[str, int]:
        plan = RouteCatalog._approved_plan(plan_json)
        limit = plan.get("patrol_limit") or {}
        limit_type = limit.get("type")
        value = limit.get("value")
        if limit_type not in ("LAPS", "DURATION_MINUTES"):
            raise ValueError("unsupported patrol limit")
        if type(value) is not int or value <= 0:
            raise ValueError("patrol limit must be a positive integer")
        return limit_type, value


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


def _route_headings_enu_deg(
    coordinates: tuple[tuple[float, float, float], ...],
) -> tuple[float, ...]:
    """Point a yaw-less explicit route along its horizontal path.

    geometry_msgs/Point carries no yaw.  Treating it as ENU yaw zero points a
    forward camera east even when the aircraft is travelling north.  Each
    point therefore uses the next non-zero segment heading; the final point
    keeps the last segment heading so the camera remains route-forward.
    """
    if not coordinates:
        return ()
    headings: list[float] = []
    last_heading = 0.0
    for index, (east, north, _up) in enumerate(coordinates):
        candidates = range(index + 1, len(coordinates))
        if index == len(coordinates) - 1:
            candidates = range(index - 1, -1, -1)
        heading = None
        for other_index in candidates:
            other_east, other_north, _ = coordinates[other_index]
            if index == len(coordinates) - 1:
                delta_east = east - other_east
                delta_north = north - other_north
            else:
                delta_east = other_east - east
                delta_north = other_north - north
            if math.hypot(delta_east, delta_north) > 1e-6:
                heading = math.degrees(math.atan2(delta_north, delta_east))
                break
        if heading is None:
            heading = last_heading
        last_heading = heading
        headings.append(heading)
    return tuple(headings)

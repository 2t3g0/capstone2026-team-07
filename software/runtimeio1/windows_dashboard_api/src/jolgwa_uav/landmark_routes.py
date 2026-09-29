from __future__ import annotations

import hashlib
import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import AvailableLandmark, LandmarkAction, LandmarkRouteStep


EARTH_RADIUS_M = 6_378_137.0


@dataclass(frozen=True)
class Landmark:
    id: str
    building_number: str
    name_ko: str
    aliases: tuple[str, ...]
    search_terms: tuple[str, ...]
    departments: tuple[str, ...]
    organizations: tuple[str, ...]
    footprint_enu: tuple[tuple[float, float], ...]

    @property
    def center_enu(self) -> tuple[float, float]:
        count = len(self.footprint_enu)
        return (
            sum(point[0] for point in self.footprint_enu) / count,
            sum(point[1] for point in self.footprint_enu) / count,
        )


class LandmarkRouteResolver:
    def __init__(self, roads_path: Path, landmarks_path: Path) -> None:
        roads = json.loads(roads_path.read_text(encoding="utf-8"))
        catalog = json.loads(landmarks_path.read_text(encoding="utf-8"))
        if roads.get("site_id") != catalog.get("site_id"):
            raise ValueError("road and landmark catalogs use different sites")

        reference = roads["reference"]
        self._reference_lat = float(reference["latitude_deg"])
        self._reference_lon = float(reference["longitude_deg"])
        self._geofence = tuple(
            self._geo_to_enu(float(point[0]), float(point[1]))
            for point in roads["geofence_geo"]
        )
        self._defaults = catalog["defaults"]
        self._landmarks = self._load_landmarks(catalog["landmarks"])
        self._graph = self._build_graph(roads["segments"])
        if not self._graph:
            raise ValueError("road catalog has no connected waypoints")
        origin_node = min(self._graph, key=lambda point: math.dist(point, (0.0, 0.0)))
        self._navigable_nodes = self._connected_nodes(origin_node)

    def summaries(self) -> list[AvailableLandmark]:
        return [
            AvailableLandmark(
                id=landmark.id,
                name_ko=landmark.name_ko,
                building_number=landmark.building_number,
                aliases=list(landmark.aliases),
                search_terms=list(landmark.search_terms),
                departments=list(landmark.departments),
                organizations=list(landmark.organizations),
                routeable=True,
            )
            for landmark in self._landmarks.values()
        ]

    def generate(self, steps: list[LandmarkRouteStep]) -> dict[str, Any]:
        if not steps:
            raise ValueError("landmark route requires at least one step")

        altitude = float(self._defaults["cruise_altitude_m"])
        waypoints: list[list[float]] = []
        route_actions: list[dict[str, Any]] = []
        names: list[str] = []

        self._append_waypoint(waypoints, 0.0, 0.0, altitude)
        cursor = self._nearest_road_node((0.0, 0.0))
        self._append_waypoint(waypoints, cursor[0], cursor[1], altitude)

        for step in steps:
            landmark = self._landmarks.get(step.landmark_id)
            if landmark is None:
                raise ValueError(f"unknown landmark_id: {step.landmark_id}")
            center = landmark.center_enu
            approach = self._nearest_road_node(center)
            self._append_path(waypoints, self._shortest_path(cursor, approach), altitude)
            names.append(f"{landmark.name_ko} {self._action_label(step.action)}")

            if step.action is LandmarkAction.ORBIT:
                self._append_orbit(waypoints, landmark, approach, altitude)
            elif step.action is LandmarkAction.CAPTURE_STILL:
                capture = self._capture_point(landmark, approach)
                yaw = math.degrees(
                    math.atan2(center[1] - capture[1], center[0] - capture[0])
                )
                index = self._append_waypoint(
                    waypoints, capture[0], capture[1], altitude, yaw
                )
                route_actions.append(
                    {
                        "waypoint_index": index,
                        "action": LandmarkAction.CAPTURE_STILL.value,
                        "landmark_id": landmark.id,
                        "hold_s": float(self._defaults["capture_hold_s"]),
                        "photo_count": int(self._defaults["capture_photo_count"]),
                    }
                )
            cursor = approach

        if len(waypoints) > 300:
            raise ValueError("generated landmark route exceeds 300 waypoints")
        self._validate_geofence(waypoints)
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "steps": [step.model_dump(mode="json") for step in steps],
                    "waypoints": waypoints,
                    "actions": route_actions,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:12]
        route_name = " -> ".join(names) + " -> 복귀"
        if len(route_name) > 80:
            route_name = f"{names[0]} 외 {len(names) - 1}곳 경유 -> 복귀"
        return {
            "route_id": f"generated-{fingerprint}",
            "route_revision": 1,
            "route_name": route_name,
            "route_waypoints_enu": waypoints,
            "route_actions": route_actions,
            "message": (
                f"랜드마크 경로 생성 완료: 고도 {altitude:.0f} m, "
                f"건물 선회는 외곽 {float(self._defaults['orbit_standoff_m']):.0f} m, "
                f"촬영 HOLD {float(self._defaults['capture_hold_s']):.0f}초/"
                f"{int(self._defaults['capture_photo_count'])}장"
            ),
        }

    def _load_landmarks(self, source: list[dict[str, Any]]) -> dict[str, Landmark]:
        output: dict[str, Landmark] = {}
        for item in source:
            footprint = tuple(
                self._geo_to_enu(float(point[0]), float(point[1]))
                for point in item["footprint_geo"]
            )
            if len(footprint) < 3:
                raise ValueError(f"landmark {item['id']} has no building footprint")
            landmark = Landmark(
                id=str(item["id"]),
                building_number=str(item["building_number"]),
                name_ko=str(item["name_ko"]),
                aliases=tuple(str(alias) for alias in item.get("aliases", [])),
                search_terms=tuple(
                    str(term) for term in item.get("search_terms", [])
                ),
                departments=tuple(
                    str(department) for department in item.get("departments", [])
                ),
                organizations=tuple(
                    str(organization) for organization in item.get("organizations", [])
                ),
                footprint_enu=footprint,
            )
            if landmark.id in output:
                raise ValueError(f"duplicate landmark_id: {landmark.id}")
            output[landmark.id] = landmark
        return output

    @staticmethod
    def _build_graph(
        segments: list[dict[str, Any]],
    ) -> dict[tuple[float, float], dict[tuple[float, float], float]]:
        graph: dict[tuple[float, float], dict[tuple[float, float], float]] = {}
        for segment in segments:
            points = [
                (round(float(point[0]), 2), round(float(point[1]), 2))
                for point in segment.get("waypoints_enu", [])
            ]
            for start, end in zip(points, points[1:]):
                distance = math.dist(start, end)
                if distance <= 0.0:
                    continue
                graph.setdefault(start, {})[end] = distance
                graph.setdefault(end, {})[start] = distance
        return graph

    def _shortest_path(
        self, start: tuple[float, float], goal: tuple[float, float]
    ) -> list[tuple[float, float]]:
        distances = {start: 0.0}
        previous: dict[tuple[float, float], tuple[float, float]] = {}
        queue = [(0.0, start)]
        while queue:
            distance, current = heapq.heappop(queue)
            if distance != distances[current]:
                continue
            if current == goal:
                break
            for neighbor, edge_distance in self._graph[current].items():
                candidate = distance + edge_distance
                if candidate < distances.get(neighbor, math.inf):
                    distances[neighbor] = candidate
                    previous[neighbor] = current
                    heapq.heappush(queue, (candidate, neighbor))
        if goal not in distances:
            raise ValueError("landmark is disconnected from the campus road network")
        path = [goal]
        while path[-1] != start:
            path.append(previous[path[-1]])
        path.reverse()
        return path

    def _nearest_road_node(self, point: tuple[float, float]) -> tuple[float, float]:
        return min(
            self._navigable_nodes,
            key=lambda candidate: math.dist(point, candidate),
        )

    def _connected_nodes(
        self, start: tuple[float, float]
    ) -> frozenset[tuple[float, float]]:
        visited = {start}
        pending = [start]
        while pending:
            current = pending.pop()
            for neighbor in self._graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    pending.append(neighbor)
        return frozenset(visited)

    def _append_path(
        self,
        waypoints: list[list[float]],
        path: list[tuple[float, float]],
        altitude: float,
    ) -> None:
        for east, north in path:
            self._append_waypoint(waypoints, east, north, altitude)

    def _append_orbit(
        self,
        waypoints: list[list[float]],
        landmark: Landmark,
        approach: tuple[float, float],
        altitude: float,
    ) -> None:
        center = landmark.center_enu
        building_radius = max(math.dist(center, point) for point in landmark.footprint_enu)
        radius = building_radius + float(self._defaults["orbit_standoff_m"])
        start_angle = math.atan2(approach[1] - center[1], approach[0] - center[0])
        point_count = 20
        for index in range(point_count + 1):
            angle = start_angle - (2.0 * math.pi * index / point_count)
            east = center[0] + radius * math.cos(angle)
            north = center[1] + radius * math.sin(angle)
            yaw = math.degrees(math.atan2(center[1] - north, center[0] - east))
            self._append_waypoint(waypoints, east, north, altitude, yaw)

    def _capture_point(
        self, landmark: Landmark, approach: tuple[float, float]
    ) -> tuple[float, float]:
        center = landmark.center_enu
        direction = (approach[0] - center[0], approach[1] - center[1])
        length = math.hypot(*direction)
        if length < 1e-6:
            direction = (1.0, 0.0)
            length = 1.0
        unit = (direction[0] / length, direction[1] / length)
        edge_distance = max(
            (point[0] - center[0]) * unit[0]
            + (point[1] - center[1]) * unit[1]
            for point in landmark.footprint_enu
        )
        standoff = float(self._defaults["capture_standoff_m"])
        return (
            center[0] + unit[0] * (edge_distance + standoff),
            center[1] + unit[1] * (edge_distance + standoff),
        )

    @staticmethod
    def _append_waypoint(
        waypoints: list[list[float]],
        east: float,
        north: float,
        altitude: float,
        yaw: float | None = None,
    ) -> int:
        point = [round(east, 2), round(north, 2), round(altitude, 2)]
        if yaw is not None:
            normalized_yaw = ((yaw + 180.0) % 360.0) - 180.0
            point.append(round(normalized_yaw, 2))
        if waypoints and waypoints[-1][:3] == point[:3]:
            if yaw is not None:
                waypoints[-1] = point
            return len(waypoints) - 1
        waypoints.append(point)
        return len(waypoints) - 1

    def _validate_geofence(self, waypoints: list[list[float]]) -> None:
        for index, point in enumerate(waypoints):
            if not _contains_xy(self._geofence, point[0], point[1]):
                raise ValueError(
                    f"generated waypoint {index + 1} is outside the PNU geofence"
                )
            if not 5.0 <= point[2] <= 120.0:
                raise ValueError("generated route altitude is outside the safety range")

    def _geo_to_enu(self, latitude: float, longitude: float) -> tuple[float, float]:
        north = math.radians(latitude - self._reference_lat) * EARTH_RADIUS_M
        east = (
            math.radians(longitude - self._reference_lon)
            * EARTH_RADIUS_M
            * math.cos(math.radians(self._reference_lat))
        )
        return east, north

    @staticmethod
    def _action_label(action: LandmarkAction) -> str:
        return {
            LandmarkAction.PASS: "경유",
            LandmarkAction.ORBIT: "주변 선회",
            LandmarkAction.CAPTURE_STILL: "촬영",
        }[action]


def _contains_xy(
    polygon: tuple[tuple[float, float], ...], x: float, y: float
) -> bool:
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        if (current_y > y) != (previous_y > y):
            intersection_x = (previous_x - current_x) * (y - current_y) / (
                previous_y - current_y
            ) + current_x
            if x < intersection_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside

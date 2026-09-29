from __future__ import annotations

import json
import math
import os
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


EARTH_RADIUS_M = 6_378_137.0


class RouteStoreError(RuntimeError):
    pass


class RouteNotFound(RouteStoreError):
    pass


class RouteRevisionConflict(RouteStoreError):
    pass


class RouteWaypoint(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    latitude_deg: Annotated[float, Field(ge=-90.0, le=90.0)]
    longitude_deg: Annotated[float, Field(ge=-180.0, le=180.0)]
    altitude_m: Annotated[float, Field(ge=5.0, le=120.0)] = 15.0
    yaw_deg: Annotated[float, Field(ge=-180.0, le=180.0)] | None = None


class RouteWrite(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: Annotated[str, Field(min_length=1, max_length=80)]
    description: Annotated[str, Field(max_length=300)] = ""
    waypoints: Annotated[list[RouteWaypoint], Field(min_length=2, max_length=300)]

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("route name must not be blank")
        return normalized


class RouteUpdate(RouteWrite):
    revision: Annotated[int, Field(ge=1)]


class RouteStore:
    def __init__(self, base_catalog_path: Path, custom_routes_path: Path) -> None:
        self.base_catalog_path = base_catalog_path
        self.custom_routes_path = custom_routes_path
        self._lock = threading.Lock()
        self._base = json.loads(base_catalog_path.read_text(encoding="utf-8"))
        profile_path = os.environ.get("JOLGWA_ROUTE_SITE_PROFILE", "").strip()
        if profile_path:
            profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
            if (profile.get("schema_version") != "1.0"
                    or profile.get("survey_confirmed") is not True
                    or profile.get("simulation_only") is not False):
                raise ValueError("physical route site profile requires explicit survey confirmation")
            if profile.get("source_catalog") != base_catalog_path.name:
                raise ValueError("physical route site profile source catalog mismatch")
            self._base = {**self._base, "site_id": profile["site_id"],
                          "simulation_only": False, "reference": profile["reference"]}
        reference = self._base["reference"]
        self._reference_lat = float(reference["latitude_deg"])
        self._reference_lon = float(reference["longitude_deg"])
        self._site_id = str(self._base.get("site_id", "")).strip()
        self._simulation_only = self._base.get("simulation_only") is True
        if not self._site_id:
            raise ValueError("route catalog requires site_id")
        self._geofence_enu = tuple(
            self._geo_to_enu(float(point[0]), float(point[1]))
            for point in self._base["geofence_geo"]
        )
        if len(self._geofence_enu) < 3:
            raise ValueError("route catalog geofence requires at least three points")
        if not custom_routes_path.exists():
            custom_routes_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_document({"schema_version": "1.0", "routes": []})

    def catalog(self) -> dict[str, Any]:
        with self._lock:
            custom = self._load_document()["routes"]
        return {
            **self._base,
            "custom_routes": [self._route_payload(route) for route in custom],
        }

    def route_summaries(self) -> list[dict[str, Any]]:
        with self._lock:
            routes = self._load_document()["routes"]
        return [
            {
                "id": route["id"],
                "revision": route["revision"],
                "name": route["name"],
                "description": route.get("description", ""),
                "waypoint_count": len(route["waypoints"]),
            }
            for route in routes
        ]

    def get(self, route_id: str) -> dict[str, Any]:
        with self._lock:
            route = dict(self._find(self._load_document()["routes"], route_id))
        return self._route_payload(route)

    def create(self, request: RouteWrite) -> dict[str, Any]:
        waypoints = self._validated_waypoints(request.waypoints)
        now = _utc_now()
        route = {
            "id": f"route-{uuid.uuid4().hex[:12]}",
            "revision": 1,
            "name": request.name,
            "description": request.description,
            "created_at": now,
            "updated_at": now,
            "waypoints": waypoints,
        }
        with self._lock:
            document = self._load_document()
            document["routes"].append(route)
            self._write_document(document)
        return self._route_payload(route)

    def update(self, route_id: str, request: RouteUpdate) -> dict[str, Any]:
        waypoints = self._validated_waypoints(request.waypoints)
        with self._lock:
            document = self._load_document()
            route = self._find(document["routes"], route_id)
            if route["revision"] != request.revision:
                raise RouteRevisionConflict(
                    f"route revision changed from {request.revision} to {route['revision']}"
                )
            route.update(
                revision=route["revision"] + 1,
                name=request.name,
                description=request.description,
                updated_at=_utc_now(),
                waypoints=waypoints,
            )
            self._write_document(document)
        return self._route_payload(route)

    def delete(self, route_id: str, revision: int) -> None:
        with self._lock:
            document = self._load_document()
            route = self._find(document["routes"], route_id)
            if route["revision"] != revision:
                raise RouteRevisionConflict(
                    f"route revision changed from {revision} to {route['revision']}"
                )
            document["routes"] = [
                candidate
                for candidate in document["routes"]
                if candidate["id"] != route_id
            ]
            self._write_document(document)

    def _validated_waypoints(self, waypoints: list[RouteWaypoint]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for index, waypoint in enumerate(waypoints):
            east, north = self._geo_to_enu(
                waypoint.latitude_deg, waypoint.longitude_deg
            )
            if not _contains_xy(self._geofence_enu, east, north):
                raise ValueError(
                    f"waypoint {index + 1} is outside the PNU Busan geofence"
                )
            candidate = waypoint.model_dump(mode="json")
            if output:
                previous = output[-1]
                previous_east, previous_north = self._geo_to_enu(
                    previous["latitude_deg"], previous["longitude_deg"]
                )
                if math.hypot(east - previous_east, north - previous_north) < 0.5:
                    continue
            output.append(candidate)
        if len(output) < 2:
            raise ValueError("route requires at least two distinct waypoints")
        return output

    def _route_payload(self, route: dict[str, Any]) -> dict[str, Any]:
        waypoints_enu = []
        for point in route["waypoints"]:
            east, north = self._geo_to_enu(
                point["latitude_deg"], point["longitude_deg"]
            )
            waypoint = [round(east, 2), round(north, 2), point["altitude_m"]]
            if point.get("yaw_deg") is not None:
                waypoint.append(point["yaw_deg"])
            waypoints_enu.append(waypoint)
        return {
            **route,
            "waypoints_enu": waypoints_enu,
            "simulation_only": self._simulation_only,
            "route_frame": {
                "site_id": self._site_id,
                "horizontal": "SITE_ENU_WGS84",
                "reference_latitude_deg": self._reference_lat,
                "reference_longitude_deg": self._reference_lon,
                "vertical": "HOME_RELATIVE",
                "version": 1,
            },
        }

    def _geo_to_enu(self, latitude: float, longitude: float) -> tuple[float, float]:
        north = math.radians(latitude - self._reference_lat) * EARTH_RADIUS_M
        east = (
            math.radians(longitude - self._reference_lon)
            * EARTH_RADIUS_M
            * math.cos(math.radians(self._reference_lat))
        )
        return east, north

    def _load_document(self) -> dict[str, Any]:
        document = json.loads(self.custom_routes_path.read_text(encoding="utf-8"))
        if document.get("schema_version") != "1.0" or not isinstance(
            document.get("routes"), list
        ):
            raise ValueError("custom route store has an unsupported format")
        return document

    def _write_document(self, document: dict[str, Any]) -> None:
        temporary = self.custom_routes_path.with_suffix(
            self.custom_routes_path.suffix + ".tmp"
        )
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.custom_routes_path)

    @staticmethod
    def _find(routes: list[dict[str, Any]], route_id: str) -> dict[str, Any]:
        for route in routes:
            if route.get("id") == route_id:
                return route
        raise RouteNotFound(f"route not found: {route_id}")


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


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

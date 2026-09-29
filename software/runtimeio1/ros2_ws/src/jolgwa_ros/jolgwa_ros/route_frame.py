"""Fail-closed conversion from a surveyed site ENU route to PX4 local NED."""
from dataclasses import dataclass
import json
import math


EARTH_RADIUS_M = 6378137.0
SUPPORTED_HORIZONTAL = "SITE_ENU_WGS84"
SUPPORTED_VERTICAL = "HOME_RELATIVE"
SUPPORTED_VERSION = 1


def _px4_home_field(message, *names):
    for name in names:
        if hasattr(message, name):
            value = float(getattr(message, name))
            if not math.isfinite(value):
                raise ValueError(f"PX4 Home {name} is not finite")
            return value
    raise ValueError("PX4 Home message has no supported field: " + "/".join(names))


def px4_home_geodetic(message):
    """Read either the PX4 ``lat/lon/alt`` or long-name Home schema."""
    latitude = _px4_home_field(message, "lat", "latitude")
    longitude = _px4_home_field(message, "lon", "longitude")
    altitude = _px4_home_field(message, "alt", "altitude")
    if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
        raise ValueError("PX4 Home WGS84 coordinates are out of range")
    return latitude, longitude, altitude


def set_px4_home_geodetic(message, latitude_deg, longitude_deg, altitude_m):
    """Write Home coordinates without silently accepting an unknown schema."""
    values = tuple(float(value) for value in (
        latitude_deg, longitude_deg, altitude_m))
    if (not all(math.isfinite(value) for value in values)
            or not -90.0 <= values[0] <= 90.0
            or not -180.0 <= values[1] <= 180.0):
        raise ValueError("PX4 Home WGS84 coordinates are invalid")
    groups = (("lat", "latitude"), ("lon", "longitude"), ("alt", "altitude"))
    for aliases, value in zip(groups, values):
        matched = False
        for name in aliases:
            if hasattr(message, name):
                setattr(message, name, value)
                matched = True
        if not matched:
            raise ValueError("PX4 Home message has no supported field: " + "/".join(aliases))
    return message


@dataclass(frozen=True)
class RouteHome:
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    ned: tuple[float, float, float]
    source_timestamp_us: int = 0


@dataclass(frozen=True)
class RouteFrameResult:
    route_ned: tuple[tuple[float, float, float], ...]
    home: RouteHome | None
    site_id: str
    first_leg_m: float
    maximum_leg_m: float


def site_enu_from_wgs84(latitude_deg, longitude_deg, reference_latitude_deg,
                         reference_longitude_deg):
    values = tuple(float(v) for v in (
        latitude_deg, longitude_deg, reference_latitude_deg, reference_longitude_deg))
    if not all(math.isfinite(v) for v in values):
        raise ValueError("route_frame coordinates must be finite")
    latitude, longitude, reference_latitude, reference_longitude = values
    if not (-90 <= latitude <= 90 and -90 <= reference_latitude <= 90
            and -180 <= longitude <= 180 and -180 <= reference_longitude <= 180):
        raise ValueError("route_frame WGS84 coordinates are out of range")
    north = math.radians(latitude-reference_latitude)*EARTH_RADIUS_M
    east = (math.radians(longitude-reference_longitude)*EARTH_RADIUS_M
            * math.cos(math.radians(reference_latitude)))
    return east, north


def plan_route_frame(plan_json):
    plan = json.loads(plan_json) if isinstance(plan_json, str) else plan_json
    if not isinstance(plan, dict):
        raise ValueError("plan must be a JSON object")
    frame = plan.get("route_frame")
    if frame is None:
        return None
    if not isinstance(frame, dict):
        raise ValueError("route_frame must be an object")
    required = {
        "site_id", "horizontal", "reference_latitude_deg",
        "reference_longitude_deg", "vertical", "version",
    }
    if set(frame) != required:
        raise ValueError("route_frame has missing or unsupported fields")
    if not isinstance(frame["site_id"], str) or not frame["site_id"].strip():
        raise ValueError("route_frame site_id is required")
    if frame["horizontal"] != SUPPORTED_HORIZONTAL:
        raise ValueError("unsupported route horizontal frame")
    if frame["vertical"] != SUPPORTED_VERTICAL:
        raise ValueError("unsupported route vertical frame")
    if frame["version"] != SUPPORTED_VERSION:
        raise ValueError("unsupported route_frame version")
    site_enu_from_wgs84(frame["reference_latitude_deg"],
                        frame["reference_longitude_deg"],
                        frame["reference_latitude_deg"],
                        frame["reference_longitude_deg"])
    return frame


def transform_route(points, plan_json, *, home, current_ned,
                    physical, max_first_leg_m=50.0):
    """Freeze an approved route in PX4 NED; legacy ENU is simulation-only."""
    frame = plan_route_frame(plan_json)
    if frame is None:
        if physical:
            raise ValueError("physical mission requires immutable route_frame")
        route = tuple((float(p.north), float(p.east), -float(p.up)) for p in points)
        return _result(route, None, "LOCAL_ENU_SIM", current_ned, max_first_leg_m, False)
    plan = json.loads(plan_json) if isinstance(plan_json, str) else plan_json
    if physical and plan.get("simulation_only") is not False:
        raise ValueError("simulation-only route catalog cannot command real hardware")
    if home is None:
        raise ValueError("fresh validated PX4 Home is required")
    if not isinstance(home, RouteHome) or len(home.ned) != 3:
        raise ValueError("invalid PX4 Home sample")
    values = (*home.ned, home.latitude_deg, home.longitude_deg, home.altitude_m)
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError("PX4 Home sample is not finite")
    east_home, north_home = site_enu_from_wgs84(
        home.latitude_deg, home.longitude_deg,
        frame["reference_latitude_deg"], frame["reference_longitude_deg"])
    route = tuple((
        float(home.ned[0]) + (float(point.north)-north_home),
        float(home.ned[1]) + (float(point.east)-east_home),
        float(home.ned[2]) - float(point.up),
    ) for point in points)
    return _result(route, home, frame["site_id"], current_ned,
                   max_first_leg_m, physical)


def _result(route, home, site_id, current_ned, limit, enforce_limit):
    if not route or not all(len(p) == 3 and all(math.isfinite(v) for v in p) for p in route):
        raise ValueError("converted route is empty or non-finite")
    current = tuple(float(v) for v in current_ned)
    if len(current) != 3 or not all(math.isfinite(v) for v in current):
        raise ValueError("fresh current PX4 local position is required")
    def distance(a, b):
        return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))
    first = distance(current, route[0])
    maximum = max([first] + [distance(a, b) for a, b in zip(route, route[1:])])
    if enforce_limit and first > float(limit):
        raise ValueError("first route leg %.1fm exceeds %.1fm physical limit" % (first, limit))
    return RouteFrameResult(tuple(route), home, site_id, first, maximum)

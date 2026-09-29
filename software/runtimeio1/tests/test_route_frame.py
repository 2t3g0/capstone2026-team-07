import json
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2_ws/src/jolgwa_ros"))

from jolgwa_ros.route_frame import (
    RouteHome,
    px4_home_geodetic,
    set_px4_home_geodetic,
    site_enu_from_wgs84,
    transform_route,
)


class Px4ShortHome:
    def __init__(self):
        self.lat = 0.0
        self.lon = 0.0
        self.alt = 0.0


class Px4LongHome:
    def __init__(self):
        self.latitude = 0.0
        self.longitude = 0.0
        self.altitude = 0.0


def test_px4_home_geodetic_supports_installed_short_schema():
    message = Px4ShortHome()
    set_px4_home_geodetic(message, 35.2350126, 129.0748631, 42.5)
    assert px4_home_geodetic(message) == (35.2350126, 129.0748631, 42.5)


def test_px4_home_geodetic_supports_long_schema():
    message = Px4LongHome()
    set_px4_home_geodetic(message, 35.2350126, 129.0748631, 42.5)
    assert px4_home_geodetic(message) == (35.2350126, 129.0748631, 42.5)


class Point:
    def __init__(self, east, north, up):
        self.east, self.north, self.up = east, north, up


def plan(*, simulation_only=False):
    return json.dumps({"simulation_only": simulation_only, "route_frame": {
        "site_id": "PNU_FIELD", "horizontal": "SITE_ENU_WGS84",
        "reference_latitude_deg": 35.2350126,
        "reference_longitude_deg": 129.0748631,
        "vertical": "HOME_RELATIVE", "version": 1,
    }})


def test_pnu_site_route_becomes_nearby_home_relative_ned():
    home = RouteHome(35.231533, 129.082617, 42.8, (0.0, 0.0, 0.0))
    result = transform_route(
        [Point(733.56, -378.91, 15.0), Point(736.47, -364.29, 5.0)],
        plan(), home=home, current_ned=(0.0, 0.0, 0.0), physical=True)
    assert result.first_leg_m < 50.0
    assert math.hypot(result.route_ned[0][0], result.route_ned[0][1]) < 50.0
    assert result.route_ned[0][2] == pytest.approx(-15.0)


def test_site_enu_axis_and_home_offset():
    east, north = site_enu_from_wgs84(35.2351126, 129.0749631,
                                        35.2350126, 129.0748631)
    assert east > 8.0 and north > 11.0
    home = RouteHome(35.2350126, 129.0748631, 0.0, (4.0, 7.0, 2.0))
    result = transform_route([Point(10.0, 20.0, 5.0)], plan(), home=home,
                             current_ned=(4.0, 7.0, 2.0), physical=True)
    assert result.route_ned[0] == pytest.approx((24.0, 17.0, -3.0))


@pytest.mark.parametrize("payload,reason", [
    ({}, "requires immutable"),
    ({"simulation_only": True, "route_frame": {"site_id": "PNU", "horizontal": "SITE_ENU_WGS84",
      "reference_latitude_deg": 35.0, "reference_longitude_deg": 129.0,
      "vertical": "HOME_RELATIVE", "version": 1}},
     "simulation-only"),
])
def test_physical_rejects_legacy_and_simulation_catalog(payload, reason):
    with pytest.raises(ValueError, match=reason):
        transform_route([Point(1, 1, 5)], payload,
                        home=RouteHome(35, 129, 0, (0, 0, 0)),
                        current_ned=(0, 0, 0), physical=True)


def test_physical_first_leg_limit_blocks_output():
    with pytest.raises(ValueError, match="exceeds"):
        transform_route([Point(100, 0, 5)], plan(),
                        home=RouteHome(35.2350126, 129.0748631, 0, (0, 0, 0)),
                        current_ned=(0, 0, 0), physical=True, max_first_leg_m=50)

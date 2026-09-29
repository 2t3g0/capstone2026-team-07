import json
from pathlib import Path

import pytest

from jolgwa_uav.route_store import RouteStore


def _catalog():
    return {
        "schema_version": "1.0", "site_id": "PNU_BUSAN", "simulation_only": True,
        "reference": {"latitude_deg": 35.2350126, "longitude_deg": 129.0748631,
                      "elevation_m": 0.0},
        "geofence_geo": [[35.234, 129.073], [35.236, 129.073], [35.236, 129.076]],
        "roads": [],
    }


def test_route_snapshot_carries_immutable_simulation_frame(tmp_path):
    catalog = tmp_path/"pnu_roads.json"
    custom = tmp_path/"custom.json"
    catalog.write_text(json.dumps(_catalog()), encoding="utf-8")
    custom.write_text(json.dumps({"schema_version": "1.0", "routes": [{
        "id": "r", "revision": 1, "name": "r", "waypoints": [
            {"latitude_deg": 35.2350, "longitude_deg": 129.0740, "altitude_m": 5.0},
            {"latitude_deg": 35.2351, "longitude_deg": 129.0741, "altitude_m": 5.0}],
    }]}), encoding="utf-8")
    route = RouteStore(catalog, custom).get("r")
    frame = route["route_frame"]
    assert route["simulation_only"] is True
    assert frame == {
        "site_id": "PNU_BUSAN", "horizontal": "SITE_ENU_WGS84",
        "reference_latitude_deg": 35.2350126,
        "reference_longitude_deg": 129.0748631,
        "vertical": "HOME_RELATIVE", "version": 1}


def test_unconfirmed_physical_profile_refuses_startup(tmp_path, monkeypatch):
    catalog = tmp_path/"pnu_roads.json"
    custom = tmp_path/"custom.json"
    profile = tmp_path/"field.json"
    catalog.write_text(json.dumps(_catalog()), encoding="utf-8")
    profile.write_text(json.dumps({
        "schema_version": "1.0", "site_id": "PNU_FIELD",
        "source_catalog": "pnu_roads.json", "simulation_only": False,
        "survey_confirmed": False, "reference": _catalog()["reference"]}), encoding="utf-8")
    monkeypatch.setenv("JOLGWA_ROUTE_SITE_PROFILE", str(profile))
    with pytest.raises(ValueError, match="survey confirmation"):
        RouteStore(catalog, custom)

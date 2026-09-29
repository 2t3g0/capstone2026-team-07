"""Bounded display snapshot for the same-host USB owner, never flight output."""
from __future__ import annotations

import math


RADIO_STATUS_TOPIC = "/jolgwa/observer/radio/status"


def _remaining(age, limit):
    if type(age) not in (int, float) or not math.isfinite(age) or age < 0:
        return 0.0
    return max(0.0, limit - age)


def make_radio_envelope(snapshot, *, clock_id, generated_s):
    """Snapshot time survives DDS/journaling; consumers must deduct elapsed time.

    Do not forward large FC diagnostics, images, arbitrary message bytes or
    commands. The snapshot's existing sensor dependencies only shorten leases.
    """
    if (not isinstance(snapshot, dict) or snapshot.get("mode") != "OBSERVE_ONLY"
            or snapshot.get("flight_commands_enabled") is not False
            or not isinstance(clock_id, str) or not clock_id or len(clock_id) > 128
            or type(generated_s) not in (int, float)
            or not math.isfinite(generated_s) or generated_s < 0):
        raise ValueError("invalid_radio_snapshot_envelope")
    report, ages = snapshot.get("report"), snapshot.get("receipt_age_s")
    if not isinstance(report, dict) or not isinstance(ages, dict):
        raise ValueError("missing_radio_snapshot_dependencies")
    assessment = report.get("assessment")
    if assessment not in ("UNKNOWN", "CLEAR", "CLIMB_REQUIRED", "BLOCKED"):
        raise ValueError("invalid_radio_assessment")
    lifetime = report.get("valid_for_s")
    sensor_lifetime = snapshot.get("sensor_valid_for_s")
    for value in (lifetime, sensor_lifetime):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= .5:
            raise ValueError("invalid_radio_source_lease")
    reason = report.get("reason")
    if not isinstance(reason, str):
        raise ValueError("invalid_radio_reason")
    sensor = snapshot.get("sensor")
    if not isinstance(sensor, dict):
        raise ValueError("invalid_radio_sensor")
    depth_left = min(sensor_lifetime, _remaining(ages.get("depth"), .5))
    rgb_left = _remaining(ages.get("rgb"), .5)
    pose_left = _remaining(ages.get("px4_pose"), .25)
    camera_ok = snapshot.get("camera_ok") is True and depth_left > 0 and rgb_left > 0
    pose_ok = snapshot.get("px4_pose_received") is True and pose_left > 0
    judgment_left = min(lifetime, depth_left, rgb_left, pose_left)
    if assessment != "UNKNOWN" and (not camera_ok or not pose_ok or judgment_left <= 0):
        assessment, reason, judgment_left = "UNKNOWN", "radio_snapshot_dependency_expired", 0.0
    status = {
        "schema_version": 1, "mode": "OBSERVE_ONLY", "flight_commands_enabled": False,
        "report": {"mode": "OBSERVE_ONLY", "flight_commands_enabled": False,
                   "assessment": assessment, "reason": reason[:512],
                   "valid_for_s": judgment_left if assessment != "UNKNOWN" else 0.0},
        "camera_ok": camera_ok, "px4_pose_received": pose_ok,
        "sensor_valid_for_s": depth_left,
        "sensor": {key: sensor[key] for key in
                   ("front_near_m", "front_median_m", "upper_roi_near_m", "valid_fraction")
                   if key in sensor} if depth_left > 0 else {},
    }
    return {"schema_version": 1, "clock_id": clock_id,
            "generated_s": float(generated_s), "status": status}

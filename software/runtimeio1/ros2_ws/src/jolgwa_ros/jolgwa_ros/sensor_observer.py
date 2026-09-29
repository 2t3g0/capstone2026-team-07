"""Display-only projection of the production policy. No ROS or control API."""
from __future__ import annotations

import math
import json
from typing import Any


OBSERVER_TOPIC = "/jolgwa/observer/d435/report"


def unknown_report(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": "OBSERVE_ONLY",
        "assessment": "UNKNOWN",
        "reason": reason,
        "flight_commands_enabled": False,
        "climb_needed": None,
        "next_observation_step_m": None,
        "climb_path_verified": False,
        "valid_for_s": 0.0,
        "policy_state": "STALE",
        "policy_direction": "STOP",
        "metrics": {},
        "geometry": {},
        "corridor_quality": {},
    }


def _number(value, name: str, *, signed: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + "_not_numeric")
    if not math.isfinite(value) or (not signed and value < 0):
        raise ValueError(name + "_invalid")
    return float(value)


def make_report(payload: dict, *, timeout_s: float = 0.5) -> dict[str, Any]:
    """Never infer CLEAR from a missing, stale, or contradictory field.

    CLIMB_REQUIRED describes what the existing ROI policy would select. An
    upper image ROI is NOT a proof of a free swept volume above the aircraft.
    Consequently this module never asserts climb_path_verified or emits a
    velocity, position target, mode, or control-authority request.
    """
    try:
        timeout_s = _number(timeout_s, "timeout_s")
        if timeout_s <= 0:
            raise ValueError("timeout_s_invalid")
        state, direction = payload.get("state"), payload.get("direction")
        reason = str(payload.get("reason", "unspecified"))
        if state == "STALE":
            return unknown_report(reason)
        if (state, direction) not in {
            ("CLEAR", "FORWARD"), ("HOLD", "STOP"), ("EVADE", "UP")
        }:
            raise ValueError("unsupported_policy_state_or_direction")
        age = _number(payload.get("observation_age_s"), "observation_age_s")
        remaining = _number(payload.get("remaining_validity_s", timeout_s - age), "remaining_validity_s")
        valid_for = min(timeout_s - age, remaining)
        if valid_for <= 0:
            return unknown_report("observation_expired")
        geometry = {}
        for key in ("geometry_valid", "roof_clearance_verified", "roof_passage_verified", "obstacle_extent_valid"):
            value = payload.get(key, False)
            if not isinstance(value, bool):
                raise ValueError(key + "_not_boolean")
            geometry[key] = value
        if not geometry["geometry_valid"]:
            return unknown_report("geometry_unavailable:" + str(payload.get("geometry_reason", reason)))
        for key in ("roof_vertical_gap_m", "roof_height_m", "obstacle_far_north_m", "obstacle_far_east_m"):
            value = payload.get(key)
            geometry[key] = None if value is None else _number(value, key, signed=True)
        if geometry["roof_clearance_verified"] and not (
            geometry["roof_vertical_gap_m"] is not None and geometry["roof_vertical_gap_m"] > 1.0
            and geometry["roof_height_m"] is not None
        ):
            raise ValueError("roof_clearance_inconsistent")
        if geometry["obstacle_extent_valid"] and any(
            geometry[key] is None for key in ("obstacle_far_north_m", "obstacle_far_east_m")
        ):
            raise ValueError("obstacle_extent_inconsistent")
        if geometry["roof_passage_verified"] and not geometry["obstacle_extent_valid"]:
            raise ValueError("roof_passage_inconsistent")
        metrics = {key: _number(payload.get(key), key) for key in (
            "front_distance_m", "upper_clearance_m", "lower_clearance_m",
            "effective_trigger_distance_m", "effective_release_distance_m",
        )}
        if not 0 < metrics["effective_trigger_distance_m"] < metrics["effective_release_distance_m"]:
            raise ValueError("threshold_order_invalid")
        metrics["observation_age_s"] = age
        for key in ("inference_ms", "service_elapsed_ms", "geometry_ms"):
            if payload.get(key) is not None:
                metrics[key] = _number(payload[key], key)
        quality = {}
        for region, values in payload.get("corridor_quality", {}).items():
            if region not in ("upper", "center", "lower") or not isinstance(values, dict):
                raise ValueError("corridor_quality_invalid")
            quality[region] = {}
            for key in ("valid_fraction", "obstacle_fraction"):
                value = _number(values.get(key), key)
                if value > 1.0:
                    raise ValueError("corridor_quality_out_of_range")
                quality[region][key] = value
        report = unknown_report(reason)
        report.update({
            "assessment": {"CLEAR": "CLEAR", "EVADE": "CLIMB_REQUIRED", "HOLD": "BLOCKED"}[state],
            "climb_needed": True if state == "EVADE" else False if state == "CLEAR" else None,
            "next_observation_step_m": 1.0 if state == "EVADE" else None,
            "valid_for_s": valid_for,
            "policy_state": state, "policy_direction": direction,
            "metrics": metrics, "geometry": geometry, "corridor_quality": quality,
        })
        return report
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        return unknown_report("invalid_policy_report:" + str(exc))


class ObserverReportLease:
    """Same-host viewer watchdog; monotonic timestamps must share a host.

    Does not renew stale data on receipt or repeated publication. ROS time / a
    wall-clock adjustment cannot prolong a positive display. This class is not
    a control interlock and cannot stop a manually flown aircraft.
    """
    def __init__(self):
        self.report = unknown_report("observer_report_missing")
        self._expires_at = float("-inf")
        self._session = None
        self._sequence = -1

    def receive(self, serialized: str, *, now: float):
        try:
            report = json.loads(serialized)
            if report.get("mode") != "OBSERVE_ONLY" or report.get("flight_commands_enabled") is not False:
                raise ValueError("not_an_observer_report")
            if report.get("assessment") not in {"CLEAR", "CLIMB_REQUIRED", "BLOCKED", "UNKNOWN"}:
                raise ValueError("assessment_invalid")
            session, sequence = report.get("session"), report.get("sequence")
            if not isinstance(session, str) or not session or type(sequence) is not int or sequence < 0:
                raise ValueError("report_identity_invalid")
            if session == self._session and sequence <= self._sequence:
                raise ValueError("duplicate_or_reordered_report")
            published = _number(report.get("published_monotonic_s"), "published_monotonic_s")
            lifetime = _number(report.get("valid_for_s"), "valid_for_s")
            now = _number(now, "now")
            if published > now or lifetime > 0.5:
                raise ValueError("observer_clock_or_lifetime_invalid")
            if report["assessment"] != "UNKNOWN" and published + lifetime <= now:
                raise ValueError("observer_report_expired")
            self.report = report
            self._session, self._sequence = session, sequence
            self._expires_at = published + lifetime
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            self.report = unknown_report(str(exc))
            self._expires_at = float("-inf")

    def current(self, *, now: float):
        if self.report["assessment"] == "UNKNOWN":
            return self.report
        if not math.isfinite(now) or now >= self._expires_at:
            return unknown_report("observer_report_expired_or_process_stopped")
        return self.report

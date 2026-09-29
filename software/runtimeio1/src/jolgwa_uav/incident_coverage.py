"""Pure, event-specific Phase 1 coverage; never a flight-control permission.

HEALTHY describes a backend, not proof that its temporal model observed a live
window. NOT_TRIGGERED is conditional coverage from a fresh general detector;
it deliberately does not claim that a temporal inference ran. All thresholds,
object gating, sampling and event votes remain owned by the original runtime.
"""
from __future__ import annotations

import math


REQUIRED_INCIDENT_MODULES = (
    "general_detection", "fire_smoke", "abnormal_behavior", "vehicle_accident",
)
EVENT_MODULES = {
    "FIRE_SMOKE": "fire_smoke",
    "HUMAN_VIOLENCE": "abnormal_behavior",
    "LITTERING": "abnormal_behavior",
    "INTRUSION_ATTEMPT": "abnormal_behavior",
    "CALL_FOR_HELP": "abnormal_behavior",
    "VEHICLE_ACCIDENT": "vehicle_accident",
}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def incident_coverage(modules, detections, temporal_window_ready):
    """Return per-module evidence states and full negative-result coverage.

    Input ages are server-local durations measured by the runtime adapter, not
    client/server clock differences. A rate-limited previous temporal result is
    OBSERVED only within its declared cadence. Positive events additionally
    require inference on this very frame; a historical call is insufficient.
    """
    modules = modules if isinstance(modules, dict) else {}
    detections_valid = isinstance(detections, (tuple, list)) and all(
        isinstance(item, dict) and isinstance(item.get("class_name"), str)
        for item in detections
    )
    classes = {item["class_name"] for item in detections} if detections_valid else set()
    coverage = {}
    for name in REQUIRED_INCIDENT_MODULES:
        stats = modules.get(name)
        healthy = (isinstance(stats, dict) and stats.get("status") == "HEALTHY"
                   and stats.get("error") in (None, "")
                   and type(stats.get("calls")) is int and stats["calls"] >= 0
                   and type(stats.get("current_frame_inference")) is bool)
        entry = {"state": "UNHEALTHY", "reason": "module_missing_invalid_or_failed",
                 "current_frame_inference": False}
        if not healthy:
            coverage[name] = entry
            continue
        calls = stats["calls"]
        age, cadence = stats.get("last_inference_age_s"), stats.get("cadence_s")
        if not _finite(cadence) or not 0 < cadence <= 1.0:
            coverage[name] = entry
            continue
        if calls > 0 and (not _finite(age) or age < 0):
            coverage[name] = entry
            continue
        if stats["current_frame_inference"] and (calls == 0 or age != 0):
            coverage[name] = entry
            continue
        # General detections expire at .5 s in Phase1DemoRuntime. Other model
        # results are only reused within the model's declared sampling cadence.
        max_age = .5 if name == "general_detection" else cadence
        observed = calls > 0 and age <= max_age
        entry = {"state": "OBSERVED" if observed else "WARMING_UP",
                 "reason": "live_inference" if observed else "live_inference_not_available",
                 "calls": calls, "last_inference_age_s": age,
                 "current_frame_inference": stats["current_frame_inference"] and observed}
        if name == "general_detection" and not detections_valid:
            entry.update(state="UNHEALTHY", reason="invalid_general_detection_evidence",
                         current_frame_inference=False)
        if name in ("abnormal_behavior", "vehicle_accident"):
            general_ready = coverage["general_detection"]["state"] == "OBSERVED"
            trigger = ("person" in classes if name == "abnormal_behavior"
                       else bool(classes & {"car", "motorcycle", "bus", "truck"}))
            entry.update(trigger_present=trigger, temporal_window_ready=temporal_window_ready is True)
            if not general_ready:
                entry.update(state="WARMING_UP", reason="general_detection_not_observed",
                             current_frame_inference=False)
            elif not trigger:
                entry.update(state="NOT_TRIGGERED", reason="fresh_general_found_no_required_object",
                             current_frame_inference=False)
            elif temporal_window_ready is not True:
                entry.update(state="WARMING_UP", reason="live_temporal_window_not_ready",
                             current_frame_inference=False)
        coverage[name] = entry
    complete = all(entry["state"] in ("OBSERVED", "NOT_TRIGGERED") for entry in coverage.values())
    return coverage, complete


def incident_event_is_covered(event_type, coverage):
    """Validate a newly confirmed event against only its actual dependencies."""
    name = EVENT_MODULES.get(event_type)
    if name is None or not isinstance(coverage, dict):
        return False
    module = coverage.get(name, {})
    if module.get("state") != "OBSERVED" or module.get("current_frame_inference") is not True:
        return False
    if name == "fire_smoke":
        return True
    return (coverage.get("general_detection", {}).get("state") == "OBSERVED"
            and module.get("temporal_window_ready") is True
            and module.get("trigger_present") is True)

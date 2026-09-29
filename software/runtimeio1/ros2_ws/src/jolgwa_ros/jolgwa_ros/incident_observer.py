"""Pure incident diagnostics: data validation and terminal intent, no control IO."""
from __future__ import annotations

import math
from jolgwa_uav.incident_coverage import incident_coverage, incident_event_is_covered
from .event_control import PHASE1_EVENT_TYPES


INCIDENT_OBSERVER_TOPIC = "/jolgwa/observer/incident"


def unknown_incident(reason: str) -> dict:
    return {"schema_version": 1, "mode": "incident-observe", "assessment": "UNKNOWN",
            "reason": str(reason), "events": [], "valid_for_s": 0.0,
            "observation_only": True, "physical_fc_commands": 0,
            "terminal_intent": "NONE", "terminal_command_sent": False}


def _finite(value) -> bool:
    return type(value) in (float, int) and math.isfinite(value)


def validate_incident_result(payload, *, frame_timestamp_ns: int,
                             frame_received_at: float, requested_at: float,
                             completed_at: float, max_frame_age_s=.5,
                             max_result_age_s=2.0) -> dict:
    """Use only local receipt durations; remote source clocks are only ordered."""
    numbers = (frame_received_at, requested_at, completed_at,
               max_frame_age_s, max_result_age_s)
    if (not all(_finite(n) for n in numbers) or frame_received_at < 0
            or not 0 < max_frame_age_s <= max_result_age_s <= 5
            or not frame_received_at <= requested_at <= completed_at
            or type(frame_timestamp_ns) is not int or frame_timestamp_ns <= 0):
        return unknown_incident("invalid_local_incident_timing")
    if requested_at - frame_received_at > max_frame_age_s:
        return unknown_incident("incident_frame_stale_before_request")
    if completed_at - frame_received_at > max_result_age_s:
        return unknown_incident("incident_response_expired")
    if not isinstance(payload, dict):
        return unknown_incident("invalid_incident_response")
    if (payload.get("observation_only") is not True
            or payload.get("purpose") != "incident_observe"
            or type(payload.get("physical_fc_commands")) is not int
            or payload.get("physical_fc_commands") != 0
            or type(payload.get("frame_timestamp_ns")) is not int
            or payload["frame_timestamp_ns"] != frame_timestamp_ns):
        return unknown_incident("incident_backend_provenance_mismatch")
    partial = (payload.get("status") == "UNKNOWN"
               and payload.get("reason") == "phase1_coverage_incomplete")
    if payload.get("status") != "OBSERVED" and not partial:
        return unknown_incident("backend_" + str(payload.get("reason", "unknown")))
    if not _finite(payload.get("elapsed_s")) or payload["elapsed_s"] < 0:
        return unknown_incident("invalid_backend_elapsed_time")
    events, phase1 = payload.get("events"), payload.get("phase1")
    if not isinstance(events, list) or not isinstance(phase1, dict):
        return unknown_incident("invalid_phase1_response_shape")
    modules = phase1.get("modules")
    # Recompute, rather than trusting a remote coverage label/template.
    coverage, coverage_complete = incident_coverage(
        modules, phase1.get("detections"), phase1.get("temporal_window_ready"))
    accepted = []
    for event in events:
        if not isinstance(event, dict):
            return unknown_incident("malformed_incident_event")
        if event.get("state") != "CONFIRMED" or event.get("event_type") not in PHASE1_EVENT_TYPES:
            continue
        if (not isinstance(event.get("event_id"), str) or not event["event_id"].strip()
                or len(event["event_id"]) > 512
                or not _finite(event.get("confidence"))
                or not 0 <= event["confidence"] <= 1):
            return unknown_incident("invalid_confirmed_incident_event")
        if incident_event_is_covered(event.get("event_type"), coverage):
            accepted.append(dict(event))
    if not coverage_complete and not accepted:
        report = unknown_incident("phase1_coverage_incomplete")
        report.update(modules=modules if isinstance(modules, dict) else {},
                      coverage=coverage, coverage_complete=False)
        return report
    expires_at = min(completed_at + 1.0, frame_received_at + max_result_age_s)
    if expires_at <= completed_at:
        return unknown_incident("incident_response_expired")
    report = unknown_incident("confirmed_incident" if accepted else "no_confirmed_incident")
    report.update(assessment="INCIDENT" if accepted else "NO_CONFIRMED_INCIDENT",
                  events=accepted, modules=modules,
                  frame_timestamp_ns=frame_timestamp_ns,
                  frame_age_s=completed_at-frame_received_at,
                  response_elapsed_s=completed_at-requested_at,
                  inference_ms=phase1.get("inference_ms"),
                  coverage=coverage, coverage_complete=coverage_complete,
                  frame_received_monotonic_s=frame_received_at,
                  evidence_completed_monotonic_s=completed_at,
                  expires_monotonic_s=expires_at,
                  valid_for_s=expires_at-completed_at)
    return report


def current_incident_report(report, *, now):
    """Age one locally validated report without renewing its evidence lease.

    Absolute timestamps are local to this observer process. Published remaining
    TTL is measured at emission, never restarted by queue receipt or heartbeat.
    """
    if not isinstance(report, dict):
        return unknown_incident("invalid_incident_report")
    if report.get("assessment") == "UNKNOWN":
        return dict(report, valid_for_s=0.0, events=[])
    issued = report.get("evidence_completed_monotonic_s")
    expires = report.get("expires_monotonic_s")
    frame_at = report.get("frame_received_monotonic_s")
    if (not all(_finite(value) for value in (now, issued, expires, frame_at))
            or not 0 <= frame_at <= issued <= now < expires
            or not 0 < expires-issued <= 1.0
            or report.get("assessment") not in ("INCIDENT", "NO_CONFIRMED_INCIDENT")):
        return unknown_incident("incident_result_not_current")
    return dict(report, valid_for_s=expires-now, frame_age_s=now-frame_at)


def capture_terminal_report(result) -> dict:
    """Record what flight integration would request, without sending it."""
    return {"event_capture_succeeded": result.succeeded,
            "recording_state": "RECORDED" if result.succeeded else "FAILED",
            "detail": result.detail, "metadata_path": result.metadata_path,
            "video_path": result.video_path,
            "frames_written": result.frames_written if result.worker_finished else None,
            "source_frames_received": result.source_frames_received if result.worker_finished else None,
            "duplicated_frames": result.duplicated_frames if result.worker_finished else None,
            "worker_finished": result.worker_finished,
            "terminal_intent": "RTL", "terminal_command_sent": False,
            "terminal_suppression": "observe_only_no_control_connection"}

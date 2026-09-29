"""Bounded flight-incident HTTP evidence; no vehicle or depth-control IO.

Server and client never subtract their clocks. Source stamps are opaque ordered
identifiers; freshness comes from local receipt durations at each endpoint.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import uuid

from .incident_coverage import EVENT_MODULES, incident_coverage, incident_event_is_covered

SCHEMA = "phase1-flight-v1"
MAX_INPUT_AGE_S = .5
MAX_RESULT_AGE_S = 2.0
MAX_LOCAL_LEASE_S = 1.0
MAX_JPEG_BYTES = 8 * 1024 * 1024


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_request_identity(session_id, request_sequence, frame_timestamp_ns):
    try:
        if not isinstance(session_id, str) or str(uuid.UUID(session_id)) != session_id:
            raise ValueError
    except (ValueError, AttributeError) as exc:
        raise ValueError("phase1_session_invalid") from exc
    if type(request_sequence) is not int or request_sequence <= 0:
        raise ValueError("phase1_request_sequence_invalid")
    if type(frame_timestamp_ns) is not int or frame_timestamp_ns <= 0:
        raise ValueError("phase1_frame_timestamp_invalid")


def validate_input_age(frame_age_s):
    if not _finite(frame_age_s) or not 0 <= frame_age_s <= MAX_INPUT_AGE_S:
        raise ValueError("phase1_frame_expired_before_inference")


@dataclass
class Phase1FlightSession:
    """One temporal runtime belongs to one source session until restart.

    Caller serializes check/consume under its inference lock. A consumed failed
    request remains consumed, so retry cannot reuse a source frame as new.
    """
    session_id: str = ""
    sequence: int = 0
    stamp: int = 0

    def consume(self, session_id, request_sequence, frame_timestamp_ns):
        validate_request_identity(session_id, request_sequence, frame_timestamp_ns)
        if self.session_id and self.session_id != session_id:
            raise ValueError("phase1_session_conflict_restart_backend")
        if request_sequence <= self.sequence or frame_timestamp_ns <= self.stamp:
            raise ValueError("phase1_duplicate_or_regressing_source")
        self.session_id, self.sequence, self.stamp = session_id, request_sequence, frame_timestamp_ns


def _phase1_evidence(phase1, events):
    if not isinstance(phase1, dict) or not isinstance(events, list):
        raise ValueError("phase1_invalid_response_shape")
    coverage, complete = incident_coverage(
        phase1.get("modules"), phase1.get("detections"), phase1.get("temporal_window_ready"))
    accepted = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("phase1_malformed_event")
        if event.get("state") != "CONFIRMED" or event.get("event_type") not in EVENT_MODULES:
            continue
        if (not isinstance(event.get("event_id"), str) or not event["event_id"].strip()
                or len(event["event_id"]) > 512 or not _finite(event.get("confidence"))
                or not 0 <= event["confidence"] <= 1):
            raise ValueError("phase1_invalid_confirmed_event")
        tracks = event.get("track_ids", [])
        if not isinstance(tracks, (list, tuple)) or len(tracks) > 64:
            raise ValueError("phase1_invalid_track_ids")
        if incident_event_is_covered(event["event_type"], coverage):
            accepted.append(deepcopy(event))
    return accepted, coverage, complete


def phase1_server_response(result, *, session_id, request_sequence,
                           frame_timestamp_ns, input_frame_age_s, elapsed_s,
                           failure_reason=""):
    """Raw HTTP UNKNOWN is not inferred from a healthy backend template."""
    validate_request_identity(session_id, request_sequence, frame_timestamp_ns)
    valid_time = (_finite(input_frame_age_s) and 0 <= input_frame_age_s <= MAX_INPUT_AGE_S
                  and _finite(elapsed_s) and elapsed_s >= 0)
    age = input_frame_age_s + elapsed_s if valid_time else MAX_RESULT_AGE_S
    envelope = dict(schema=SCHEMA, purpose="phase1_flight", session_id=session_id,
                    request_sequence=request_sequence, frame_timestamp_ns=frame_timestamp_ns,
                    input_frame_age_s=input_frame_age_s, elapsed_s=elapsed_s,
                    frame_age_s=age, remaining_validity_s=0.0,
                    status="UNKNOWN", reason=failure_reason or "phase1_result_expired",
                    events=[], phase1={"enabled": result is not None})
    if failure_reason or not valid_time or age >= MAX_RESULT_AGE_S or result is None:
        return envelope
    phase1 = dict(enabled=True, inference_ms=result.inference_ms,
                  detections=list(result.detections), modules=deepcopy(result.modules),
                  temporal_window_ready=getattr(result, "temporal_window_ready", False))
    try:
        events, coverage, complete = _phase1_evidence(phase1, list(result.events))
    except (ValueError, TypeError):
        return dict(envelope, reason="phase1_invalid_runtime_evidence")
    phase1.update(coverage=coverage, coverage_complete=complete)
    return dict(envelope, status="OBSERVED" if complete else "UNKNOWN",
                reason="phase1_observed" if complete else "phase1_coverage_incomplete",
                events=events, phase1=phase1, remaining_validity_s=MAX_RESULT_AGE_S-age)


def validate_phase1_response(payload, *, session_id, request_sequence, frame_timestamp_ns,
                             frame_received_at, requested_at, completed_at):
    """Bind remote inference to its original client-local absolute deadline."""
    validate_request_identity(session_id, request_sequence, frame_timestamp_ns)
    if (not all(_finite(v) for v in (frame_received_at, requested_at, completed_at))
            or not 0 <= frame_received_at <= requested_at <= completed_at):
        raise ValueError("phase1_invalid_local_timing")
    validate_input_age(requested_at-frame_received_at)
    if completed_at-frame_received_at >= MAX_RESULT_AGE_S:
        raise ValueError("phase1_response_expired")
    if (not isinstance(payload, dict) or payload.get("schema") != SCHEMA
            or payload.get("purpose") != "phase1_flight"
            or payload.get("session_id") != session_id
            or type(payload.get("request_sequence")) is not int
            or payload["request_sequence"] != request_sequence
            or type(payload.get("frame_timestamp_ns")) is not int
            or payload["frame_timestamp_ns"] != frame_timestamp_ns):
        raise ValueError("phase1_source_echo_mismatch")
    input_age, elapsed, age, remaining = (payload.get(name) for name in (
        "input_frame_age_s", "elapsed_s", "frame_age_s", "remaining_validity_s"))
    validate_input_age(input_age)
    if (not all(_finite(v) for v in (elapsed, age, remaining)) or elapsed < 0
            or not 0 <= age < MAX_RESULT_AGE_S or not 0 < remaining <= MAX_RESULT_AGE_S
            or not math.isclose(input_age+elapsed, age, rel_tol=0, abs_tol=1e-6)
            or remaining > MAX_RESULT_AGE_S-age+1e-6):
        raise ValueError("phase1_invalid_or_expired_backend_timing")
    partial = payload.get("status") == "UNKNOWN" and payload.get("reason") == "phase1_coverage_incomplete"
    if payload.get("status") != "OBSERVED" and not partial:
        raise ValueError("phase1_backend_not_observed")
    events, coverage, complete = _phase1_evidence(payload.get("phase1"), payload.get("events"))
    report = deepcopy(payload)
    report["phase1"].update(coverage=coverage, coverage_complete=complete)
    report.update(events=events, status="OBSERVED" if complete else "UNKNOWN",
                  reason="phase1_observed" if complete else "phase1_coverage_incomplete")
    # Internal provenance replaces any same-named remote values. Queue receipt
    # and publication can only consume this lease; they cannot create one.
    report["_local_evidence"] = dict(
        frame_received_at=frame_received_at, completed_at=completed_at,
        expires_at=min(frame_received_at+MAX_RESULT_AGE_S,
                       completed_at+MAX_LOCAL_LEASE_S, completed_at+remaining))
    return current_phase1_response(report, now=completed_at)


def current_phase1_response(report, *, now):
    evidence = report.get("_local_evidence") if isinstance(report, dict) else None
    if not isinstance(evidence, dict):
        raise ValueError("phase1_missing_local_evidence")
    frame_at, completed, expires = (evidence.get(name) for name in (
        "frame_received_at", "completed_at", "expires_at"))
    if (not all(_finite(v) for v in (now, frame_at, completed, expires))
            or not 0 <= frame_at <= completed <= now < expires
            or not 0 < expires-completed <= MAX_LOCAL_LEASE_S
            or expires > frame_at+MAX_RESULT_AGE_S):
        raise ValueError("phase1_evidence_expired_or_invalid")
    return dict(report, remaining_validity_s=expires-now, frame_age_s=now-frame_at)

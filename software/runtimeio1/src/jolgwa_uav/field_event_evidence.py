"""Read-only event-flight evaluation over the existing auto04 record schemas.

No source label proves physical Jetson use. No vehicle commands or model loads.
Missing evidence is UNVERIFIED, contrary evidence BLOCKED, all checks PASS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

EVENTS = {"FIRE_SMOKE", "HUMAN_VIOLENCE", "LITTERING", "INTRUSION_ATTEMPT",
          "CALL_FOR_HELP", "VEHICLE_ACCIDENT"}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def vector(value):
    return isinstance(value, (list, tuple)) and len(value) == 3 and all(map(finite, value))


def distance(a, b, dimensions=3):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(dimensions)))


def _motion(rows):
    if len(rows) < 2:
        return None
    span = rows[-1]["monotonic_s"] - rows[0]["monotonic_s"]
    if span <= 0:
        return None
    positions = [r["payload"]["position_ned_m"] for r in rows]
    path = sum(distance(a, b, 2) for a, b in zip(positions, positions[1:]))
    return dict(span_s=span, displacement_m=distance(positions[0], positions[-1], 2),
                path_m=path, mean_speed_mps=path/span,
                max_gap_s=max(b["monotonic_s"]-a["monotonic_s"] for a, b in zip(rows, rows[1:])))


def evaluate_event_flight(observations, capture, operations, *, mission_id, stage=5,
                          obstacle_evaluation=None, route_context=None,
                          video_verification=None, provenance=None):
    if stage not in (5, 6):
        raise ValueError("event evaluation only supports stage 5 or 6")
    checks, metrics = {}, {}
    prov = provenance or {}
    observation_clock = prov.get("observation_clock_id")
    capture_clock = capture.get("extra", {}).get("evidence_clock_id") or prov.get("capture_clock_id")
    clock_present = all(isinstance(value, str) and value.strip() for value in (observation_clock, capture_clock))
    same_clock = observation_clock == capture_clock if clock_present else None
    checks["same_observation_capture_clock"] = same_clock
    synthetic = any(r.get("evidence_kind") == "synthetic_test_fixture" or r.get("synthetic_inputs") is True
                    or r.get("synthetic") is True for r in observations if isinstance(r, dict))
    declared_kind = prov.get("evidence_kind")
    evidence_kind = "synthetic_test_fixture" if synthetic else declared_kind or "unverified"
    checks["evidence_kind_declared"] = None if declared_kind is None else declared_kind in ("actual", "synthetic_test_fixture")
    checks["no_synthetic_to_actual_promotion"] = not (synthetic and declared_kind == "actual")
    rows = [r for r in observations if isinstance(r, dict) and finite(r.get("monotonic_s"))
            and isinstance(r.get("payload"), dict)]
    checks["valid_monotonic_observations"] = bool(rows) and len(rows) == len(observations) and all(
        a["monotonic_s"] <= b["monotonic_s"] for a, b in zip(rows, rows[1:]))
    checks["observation_clock_consistent"] = None if not observation_clock else all(
        row.get("clock_id", observation_clock) == observation_clock for row in rows)
    mission = [r for r in rows if r.get("kind") == "mission" and r["payload"].get("mission_id") == mission_id]
    vehicles = [r for r in rows if r.get("kind") == "vehicle" and r["payload"].get("mission_id") == mission_id
                and vector(r["payload"].get("position_ned_m")) and r["payload"].get("position_valid") is True]
    extra = capture.get("extra", {})
    event_id, kind = capture.get("event_id"), capture.get("event_type")
    lease_id = extra.get("lease_id")
    checks["capture_mission_event_lease"] = bool(event_id and lease_id and extra.get("mission_id") == mission_id)
    events = [r for r in rows if r.get("kind") == "event" and r["payload"].get("event_id") == event_id]
    event = events[0] if events else None
    event_clock = event["payload"].get("evidence_clock_id") if event else None
    checks["event_clock_consistent"] = (None if not observation_clock else
        not event_clock or event_clock == observation_clock)
    checks["confirmed_original_event"] = None if not event else (
        event["payload"].get("state") == "CONFIRMED" and kind in EVENTS
        and event["payload"].get("event_type") == kind
        and finite(event["payload"].get("confidence"))
        and 0 <= event["payload"]["confidence"] <= 1
        and event["payload"]["confidence"] == capture.get("confidence"))
    t0 = event["monotonic_s"] if event else math.inf
    before = [r for r in mission if r["monotonic_s"] < t0]
    prior = before[-1] if before else None
    checks["event_preceded_by_unfinished_patrol"] = None if not prior else (
        prior["payload"].get("phase") in (3, "PATROL")
        and prior["payload"].get("control_owner") == "LLM_ROUTE"
        and prior["payload"].get("approved") is True
        and type(prior["payload"].get("current_waypoint")) is int
        and type(prior["payload"].get("total_waypoints")) is int
        and 0 <= prior["payload"]["current_waypoint"] < prior["payload"]["total_waypoints"])
    pre = [r for r in vehicles if t0-1.5 <= r["monotonic_s"] < t0]
    movement = _motion(pre)
    metrics["pre_event_motion"] = movement
    checks["actually_moving_before_event"] = None if movement is None else (
        movement["span_s"] >= 1.0 and movement["max_gap_s"] <= .5
        and t0-pre[-1]["monotonic_s"] <= .25
        and movement["displacement_m"]/movement["span_s"] >= .05)
    checks["not_safety_or_manual_stop"] = None if not pre else all(
        r["payload"].get("active_authority") == "LLM_ROUTE"
        and r["payload"].get("jetson_safety_state") == "CLEAR"
        and r["payload"].get("avoidance_active") is False
        and r["payload"].get("manual_override") is False for r in pre)
    index = prior["payload"].get("current_waypoint") if prior else None
    target = None
    if isinstance(route_context, dict):
        target = route_context.get("patrol_targets_ned_m", {}).get(str(index))
        if target is None:
            target = route_context.get("active_waypoint_ned_m")
    context_ok = bool(route_context and prior and pre and
        route_context.get("mission_id") == mission_id and
        route_context.get("proposal_id") == prior["payload"].get("proposal_id") and
        route_context.get("current_waypoint", index) == index and vector(target) and
        finite(route_context.get("acceptance_radius_m")) and route_context["acceptance_radius_m"] > 0)
    checks["approved_waypoint_context"] = None if route_context is None else context_ok
    if context_ok:
        remaining = distance(pre[-1]["payload"]["position_ned_m"], target)
        metrics["distance_to_active_waypoint_m"] = remaining
        checks["before_arrival_not_endpoint_dwell"] = (
            remaining > route_context["acceptance_radius_m"] + .5 and route_context.get("endpoint_observation_s") == 0)
    else:
        checks["before_arrival_not_endpoint_dwell"] = None
    associated = [r for r in mission if r["payload"].get("event_id") == event_id
                  and r["payload"].get("event_type") == kind
                  and r["payload"].get("event_lease_id") == lease_id]
    holds = [r for r in associated if r["payload"].get("phase") in (10, "PAUSED_EVENT")]
    captures = [r for r in associated if r["payload"].get("phase") in (11, "EVENT_CAPTURE")]
    returns = [r for r in associated if r["payload"].get("phase") in (4, "RETURNING_HOME")]
    hold, active, returning = (items[0] if items else None for items in (holds, captures, returns))
    checks["event_owns_hold_before_capture"] = None if not hold or not active else (
        t0 <= hold["monotonic_s"] <= active["monotonic_s"]
        and hold["payload"].get("control_owner") == "JETSON_EVENT_CAPTURE"
        and active["payload"].get("control_owner") == "JETSON_EVENT_CAPTURE")
    targets = [r for r in rows if r.get("kind") == "trajectory" and hold and active
               and hold["monotonic_s"] <= r["monotonic_s"] <= active["monotonic_s"] + .5
               and vector(r["payload"].get("position"))]
    checks["fixed_position_hold_output"] = None if not targets else all(
        distance(r["payload"]["position"], targets[0]["payload"]["position"]) <= .05
        and all(v in ("nan", None) or (isinstance(v, float) and math.isnan(v)) or v == 0
                for v in r["payload"].get("velocity", ["MISSING"])) for r in targets)
    policy = capture.get("capture_policy", {})
    checks["five_second_capture_finalized"] = (
        capture.get("succeeded") is True and capture.get("worker_finished") is True
        and capture.get("retention") == "permanent" and policy.get("duration_s") == 5.0
        and policy.get("fps") == 25 and policy.get("codec") == "h264"
        and capture.get("frames_written") == 125 and capture.get("stored_duration_s") == 5.0
        and type(capture.get("unique_source_frames_used")) is int
        and type(capture.get("duplicated_frames")) is int
        and capture["unique_source_frames_used"] >= 5 and capture["duplicated_frames"] >= 0
        and capture["unique_source_frames_used"] + capture["duplicated_frames"] == 125)
    start, finalized = capture.get("record_started_monotonic_s"), capture.get("finalized_monotonic_s")
    clock_valid = same_clock is True and finite(start) and finite(finalized) and start < finalized
    checks["capture_after_event_and_before_return"] = None if not active or not returning or not clock_valid else (
        active["monotonic_s"] <= start and start+5 <= finalized <= returning["monotonic_s"])
    during = [r for r in vehicles if clock_valid and start <= r["monotonic_s"] <= start+5]
    motion = _motion(during)
    tail = _motion([r for r in during if r["monotonic_s"] >= start+4])
    metrics.update(capture_motion=motion, capture_last_second_motion=tail)
    max_hold = max((distance(r["payload"]["position_ned_m"], targets[0]["payload"]["position"])
                    for r in during), default=None) if targets else None
    metrics["max_hold_distance_m"] = max_hold
    checks["actual_deceleration_and_stable_hold"] = None if not motion or not tail or max_hold is None else (
        motion["span_s"] >= 4.5 and motion["max_gap_s"] <= .5 and tail["span_s"] >= .7
        and tail["mean_speed_mps"] <= .1 and max_hold <= .5)
    checks["video_decoded_matches_metadata"] = None if video_verification is None else (
        video_verification.get("passed") is True and video_verification.get("evaluator") == "closed_event_video"
        and video_verification.get("event_id") == event_id and video_verification.get("frames_decoded") == 125)
    completed = [r for r in associated if r["payload"].get("phase") in (6, "COMPLETED")]
    checks["visited_path_home_landing"] = None if not completed or not returning else (
        returning["monotonic_s"] < completed[-1]["monotonic_s"]
        and "visited-path Home landing confirmed" in completed[-1]["payload"].get("detail", ""))
    landed = [r for r in vehicles if completed and
              completed[-1]["monotonic_s"]-.5 <= r["monotonic_s"] <= completed[-1]["monotonic_s"]+10]
    checks["fresh_landed_and_disarmed"] = None if not landed else (
        landed[-1]["payload"].get("landed") is True and landed[-1]["payload"].get("armed") is False
        and landed[-1]["monotonic_s"] <= completed[-1]["monotonic_s"] + 10)
    checks["no_fallback_native_rtl"] = not any(r.get("kind") == "command" and r["monotonic_s"] >= t0
        and r["payload"].get("command") == 20 for r in rows)
    api = [r for r in operations if r.get("mission_id") == mission_id]
    approved = [r for r in api if r.get("event") == "mission.approval_result" and r.get("accepted") is True]
    results = [r for r in api if r.get("event") == "mission.execution_result"]
    proposal = prior["payload"].get("proposal_id") if prior else None
    checks["same_ui_approval_and_api_success"] = None if not approved or not results or not proposal else (
        all(r.get("proposal_id") == proposal for r in approved+results)
        and results[-1].get("success") is True and results[-1].get("final_phase") == 6
        and bool(approved[0].get("timestamp")) and approved[0]["timestamp"] <= results[-1].get("timestamp", ""))
    if stage == 6:
        from .field_route_evidence import STAGE4_REQUIRED_CHECKS, observation_digest
        checks["same_mission_obstacle_avoid_resume"] = None if obstacle_evaluation is None else (
            obstacle_evaluation.get("passed") is True and obstacle_evaluation.get("mission_id") == mission_id
            and obstacle_evaluation.get("schema_version") == 1
            and obstacle_evaluation.get("evaluator") == "field_route_evidence"
            and obstacle_evaluation.get("stage") == 4
            and isinstance(observation_clock, str) and bool(observation_clock)
            and obstacle_evaluation.get("clock_id") == observation_clock
            and isinstance(obstacle_evaluation.get("checks"), dict) and bool(obstacle_evaluation["checks"])
            and STAGE4_REQUIRED_CHECKS.issubset(obstacle_evaluation["checks"])
            and all(value is True for value in obstacle_evaluation["checks"].values())
            and obstacle_evaluation.get("observations_sha256") == observation_digest(observations)
            and obstacle_evaluation.get("evidence_kind") == evidence_kind
            and finite(obstacle_evaluation.get("metrics", {}).get("recovery_completed_monotonic_s"))
            and obstacle_evaluation["metrics"]["recovery_completed_monotonic_s"] < t0)
        if obstacle_evaluation and obstacle_evaluation.get("evidence_kind") == "synthetic_test_fixture":
            evidence_kind = "synthetic_test_fixture"
    missing = [name for name, value in checks.items() if value is None]
    failed = [name for name, value in checks.items() if value is False]
    sim = prov.get("simulation_only")
    backend = prov.get("backend_identity", prov)
    jetson = backend.get("actual_jetson_used")
    scope = ("synthetic_test_fixture" if evidence_kind == "synthetic_test_fixture" else
             "SIM_ONLY" if sim is True else "PHYSICAL_DECLARED_NOT_INDEPENDENTLY_ATTESTED"
             if sim is False and jetson is True else "PROVENANCE_UNVERIFIED")
    return dict(schema_version=1, evaluator="field_event_evidence", stage=stage, mission_id=mission_id, event_id=event_id,
                evidence_kind=evidence_kind, status="BLOCKED" if failed else "UNVERIFIED" if missing else "PASS",
                passed=not failed and not missing, checks=checks, metrics=metrics,
                missing_evidence=missing, errors=failed, video_verification=video_verification,
                provenance=dict(scope=scope, declared_simulation_only=sim, declared_actual_jetson_used=jetson,
                                observation_clock_id=observation_clock, capture_clock_id=capture_clock,
                                source_string_not_hardware_proof=True, real_flight_verified=False),
                physical_fc_commands_sent_by_evaluator=0)


def verify_capture_video(metadata_path, capture):
    """Decode the closed local file; metadata success alone is insufficient."""
    name = capture.get("video_file", "")
    if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
        return dict(passed=False, error="invalid video filename")
    path = Path(metadata_path).resolve().parent / name
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        return dict(passed=False, error="missing/empty/nonlocal closed video")
    try:
        import cv2
        video = cv2.VideoCapture(str(path))
        try:
            if not video.isOpened():
                return dict(passed=False, error="video decoder could not open file")
            fps = float(video.get(cv2.CAP_PROP_FPS))
            code = int(video.get(cv2.CAP_PROP_FOURCC))
            codec = "".join(chr((code >> 8*i) & 255) for i in range(4)).lower()
            count, dimensions_ok = 0, True
            while True:
                ok, image = video.read()
                if not ok:
                    break
                count += 1
                dimensions_ok &= image.shape[:2] == (capture.get("height"), capture.get("width"))
                if count > 126:
                    break
        finally:
            video.release()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024*1024), b""):
                digest.update(block)
        return dict(evaluator="closed_event_video", event_id=capture.get("event_id"),
                    passed=count == 125 and finite(fps) and abs(fps-25) <= .01 and dimensions_ok
                    and codec in ("h264", "avc1"), frames_decoded=count, fps=fps, codec=codec,
                    dimensions_match=dimensions_ok, sha256=digest.hexdigest(), file=str(path))
    except Exception as exc:
        return dict(passed=False, error=f"{type(exc).__name__}: {exc}")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        rows = [json.loads(line, parse_constant=_invalid_constant) for line in stream if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("JSONL records must be objects")
    return rows


def _invalid_constant(value):
    raise ValueError("non-standard JSON numeric constant: " + value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("observations", "capture-metadata", "api-log", "mission-id", "output"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--stage", type=int, choices=(5, 6), default=5)
    for name in ("route-context", "provenance", "obstacle-evaluation"):
        parser.add_argument("--"+name)
    args = parser.parse_args(argv)
    def read(path):
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=_invalid_constant) if path else None
        if value is not None and not isinstance(value, dict):
            raise ValueError("JSON evidence must be an object")
        return value
    try:
        capture = read(args.capture_metadata)
        report = evaluate_event_flight(read_jsonl(args.observations), capture, read_jsonl(args.api_log),
            mission_id=args.mission_id, stage=args.stage, route_context=read(args.route_context),
            provenance=read(args.provenance), obstacle_evaluation=read(args.obstacle_evaluation),
            video_verification=verify_capture_video(args.capture_metadata, capture))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        report = dict(schema_version=1, stage=args.stage, mission_id=args.mission_id, status="BLOCKED",
                      passed=False, input_error=f"{type(exc).__name__}: {exc}")
    try:
        with Path(args.output).open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    except OSError as exc:
        report.update(status="BLOCKED", passed=False, output_error=f"{type(exc).__name__}: {exc}")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

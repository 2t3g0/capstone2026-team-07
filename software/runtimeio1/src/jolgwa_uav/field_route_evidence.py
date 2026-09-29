"""Evaluate closed route observations, not commands or claimed runtime success.

No ROS/FC access. Context must contain the approved PATROL index->NED targets;
the caller must not derive a different target from the observed flight itself.
Provenance is explicit and recorded, not authenticated by this offline tool.
"""
import hashlib
import json
import math


STAGE4_REQUIRED_CHECKS = frozenset({
    "approved_route_progress", "actual_avoidance_pause_and_climb",
    "waypoint_preserved", "nominal_altitude_recovered", "approved_route_resumed",
    "no_manual_or_epoch_loss", "sample_continuity",
})


def observation_digest(observations):
    return hashlib.sha256(json.dumps(observations, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _position(value):
    return isinstance(value, (list, tuple)) and len(value) == 3 and all(map(_number, value))


def _owned(payload):
    return all(payload.get(key) is True for key in (
        "approved", "command_output_enabled", "connected", "preflight_checks_pass", "position_valid", "armed", "offboard"
    )) and payload.get("manual_override") is False and payload.get("flight_epoch_retired") is False


def _distance(a, b):
    return math.dist(a, b)


def evaluate_route_flight(observations, *, mission_id, stage=4, route_context, provenance):
    """Return stage 3 observation or stage 4 recovery evidence with common clock.

Required context: schema_version=1, mission_id, patrol_targets_ned_m mapping,
    acceptance_radius_m, altitude_tolerance_m (the actual runtime values).
Required provenance: observation_clock_id, evidence_kind actual|synthetic_test_fixture.
The first complete avoidance cycle is evaluated; this is not course-wide or
model-accuracy certification. All thresholds below are evidence criteria, not
runtime guard overrides.
"""
    result = dict(schema_version=1, evaluator="field_route_evidence", stage=stage,
        passed=False, status="UNVERIFIED", mission_id=mission_id, clock_id=None, evidence_kind=None,
        checks={}, metrics={}, errors=[],
        verification_scope="closed observation analysis, not hardware certification")
    errors, checks, metrics = result["errors"], result["checks"], result["metrics"]
    if type(stage) is not int or stage not in (3, 4) or not isinstance(mission_id, str) or not mission_id:
        errors.append("require stage 3/4 and an exact mission_id"); return result
    if not isinstance(provenance, dict) or not isinstance(route_context, dict):
        errors.append("explicit route context and clock provenance required"); return result
    clock = provenance.get("observation_clock_id")
    kind = provenance.get("evidence_kind")
    result.update(clock_id=clock, evidence_kind=kind)
    if not isinstance(clock, str) or not clock or kind not in ("actual", "synthetic_test_fixture"):
        errors.append("observation clock/evidence kind missing or invalid"); return result
    targets = route_context.get("patrol_targets_ned_m")
    radius, tolerance = route_context.get("acceptance_radius_m"), route_context.get("altitude_tolerance_m")
    if (route_context.get("schema_version") != 1 or route_context.get("mission_id") != mission_id
            or not isinstance(targets, dict) or not targets
            or any(not isinstance(key, str) or not key.isdigit() or not _position(value) for key, value in targets.items())
            or not _number(radius) or not 0 < radius <= 10
            or not _number(tolerance) or not 0 < tolerance <= .5):
        errors.append("approved patrol-index targets and actual positive tolerances required"); return result
    if not isinstance(observations, list):
        errors.append("observations must be a closed list of original rows"); return result
    result["observations_sha256"] = observation_digest(observations)
    vehicles, mission, last_t, observed_perception = [], None, -math.inf, False
    for row in observations:
        if not isinstance(row, dict) or not isinstance(row.get("payload"), dict):
            errors.append("malformed observation row"); return result
        if row.get("clock_id", clock) != clock:
            errors.append("mixed observation clocks"); return result
        if kind == "actual" and (row.get("synthetic_inputs") is True or row.get("synthetic") is True
                or row.get("evidence_kind") == "synthetic_test_fixture"):
            errors.append("synthetic observation cannot be classified actual"); return result
        t, payload = row.get("monotonic_s"), row["payload"]
        if not _number(t) or t < last_t:
            errors.append("observation monotonic order is invalid"); return result
        last_t = t
        if row.get("kind") in ("safety", "decision", "perception", "event", "phase1"):
            observed_perception = True
        if payload.get("mission_id") != mission_id:
            continue
        if row.get("kind") == "mission":
            mission = payload
        if row.get("kind") == "vehicle" and mission is not None and mission.get("phase") == 3:
            if not _position(payload.get("position_ned_m")):
                errors.append("non-finite PATROL position"); return result
            index = mission.get("current_waypoint")
            if type(index) is not int or str(index) not in targets:
                errors.append("PATROL index absent from approved route context"); return result
            vehicles.append(dict(t=t, p=payload, waypoint=index, target=targets[str(index)]))
    if len(vehicles) < 3:
        errors.append("insufficient same-mission PATROL observations"); return result
    def nominal(v):
        p = v["p"]
        return _owned(p) and p.get("active_authority") == "LLM_ROUTE" and p.get("avoidance_active") is False
    def closer(a, b):
        return a["waypoint"] == b["waypoint"] and (
            _distance(a["p"]["position_ned_m"], a["target"])
            - _distance(b["p"]["position_ned_m"], a["target"]) >= .1)
    if stage == 3:
        checks["approved_route_progress"] = any(closer(a, b) for a, b in zip(vehicles, vehicles[1:])) or (
            any(closer(vehicles[0], b) for b in vehicles[1:]))
        checks["no_jetson_auto_authority"] = all(nominal(v) for v in vehicles)
        checks["judgement_observed"] = any(v["p"].get("jetson_safety_fresh") is True and
            v["p"].get("jetson_safety_state") in ("CLEAR","SLOW","HOLD","EVADE") for v in vehicles)
        checks["no_manual_or_epoch_loss"] = all(_owned(v["p"]) for v in vehicles)
        checks["sample_continuity"] = all(0 < b["t"]-a["t"] <= .5 for a,b in zip(vehicles,vehicles[1:]))
        metrics.update(first_monotonic_s=vehicles[0]["t"], last_monotonic_s=vehicles[-1]["t"])
    else:
        start = next((i for i, v in enumerate(vehicles) if i and v["p"].get("avoidance_active") is True
            and v["p"].get("active_authority") == "JETSON_SAFETY"
            and v["p"].get("jetson_safety_fresh") is True
            and v["p"].get("jetson_safety_state") == "EVADE"), None)
        if start is None:
            errors.append("no actual approved-route avoidance cycle"); return result
        before = next((vehicles[i] for i in range(start-1,-1,-1) if nominal(vehicles[i])), None)
        if before is None:
            errors.append("no approved route before avoidance"); return result
        end = next((i for i in range(start+1,len(vehicles)) if nominal(vehicles[i])
            and abs(vehicles[i]["p"]["position_ned_m"][2]-before["target"][2]) <= tolerance), None)
        if end is None:
            errors.append("no original-altitude approved-route recovery (POSCTL is not resume)"); return result
        after = vehicles[end]
        cycle = [v for v in vehicles if before["t"] <= v["t"] <= after["t"]]
        safety = [v for v in cycle if v["p"].get("active_authority") == "JETSON_SAFETY"]
        pause = False
        for i, a in enumerate(safety):
            horizontal_path = 0.
            for j in range(i+1,len(safety)):
                b, previous = safety[j], safety[j-1]
                elapsed = b["t"]-a["t"]
                if elapsed > .6 or b["t"]-previous["t"] > .5:
                    break
                horizontal_path += _distance(previous["p"]["position_ned_m"][:2],b["p"]["position_ned_m"][:2])
                if elapsed >= .2 and horizontal_path/elapsed <= .1:
                    pause = True; break
            if pause:
                break
        climbed = before["p"]["position_ned_m"][2]-min(v["p"]["position_ned_m"][2] for v in cycle)
        checks["approved_route_progress"] = closer(before, after) and _distance(
            before["p"]["position_ned_m"],before["target"]) > radius
        checks["actual_avoidance_pause_and_climb"] = pause and climbed >= .2
        checks["waypoint_preserved"] = all(v["waypoint"] == before["waypoint"] for v in cycle)
        checks["nominal_altitude_recovered"] = abs(after["p"]["position_ned_m"][2]-before["target"][2]) <= tolerance
        checks["approved_route_resumed"] = nominal(after) and closer(before, after)
        checks["no_manual_or_epoch_loss"] = all(_owned(v["p"]) for v in cycle)
        checks["sample_continuity"] = all(0 < b["t"]-a["t"] <= .5 for a,b in zip(cycle,cycle[1:]))
        metrics.update(avoidance_started_monotonic_s=vehicles[start]["t"],
            recovery_completed_monotonic_s=after["t"], climb_m=climbed,
            original_waypoint=before["waypoint"], original_target_ned_m=list(before["target"]),
            recovered_position_ned_m=list(after["p"]["position_ned_m"]))
    result["passed"] = bool(checks) and all(value is True for value in checks.values())
    result["status"] = "PASS" if result["passed"] else "BLOCKED"
    errors.extend(name for name, passed in checks.items() if passed is not True)
    return result

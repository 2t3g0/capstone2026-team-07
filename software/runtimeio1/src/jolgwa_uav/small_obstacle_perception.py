"""Lossless native-decision adapter; perception evidence is not control authority."""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import threading

from .small_obstacle_demo_guard import DecisionEvidence, POLICY_DIGEST, demo_policy_config
from .vertical_avoidance import (DepthRoiStats, DepthCorridorStats, VerticalAvoidanceDecision,
                                VerticalAvoidanceState, VerticalDirection)

NS = 1_000_000_000
DEMO_TOPIC = "/jolgwa/demo/perception"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _wire(value):
    """Preserve absent/nonfinite measurements as null, never invented distances."""
    if isinstance(value, dict): return {key: _wire(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [_wire(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def make_native_demo_contract(config, backend_epoch, decision=None):
    """Backend uses its ACTUAL config/decision objects, not the demo defaults."""
    if not isinstance(backend_epoch, str) or not backend_epoch:
        raise ValueError("backend_epoch_required")
    policy = asdict(config)
    result = {"schema_version": 1, "backend_epoch": backend_epoch,
              "policy_config": policy, "policy_digest": hashlib.sha256(canonical(policy).encode()).hexdigest()}
    if decision is not None:
        result["decision"] = _wire(asdict(decision))
    return result


def validate_contract(contract, expected_epoch=None):
    if not isinstance(contract, dict) or contract.get("schema_version") != 1:
        raise ValueError("native_demo_contract_missing")
    epoch = contract.get("backend_epoch")
    if not isinstance(epoch, str) or not epoch or (expected_epoch is not None and epoch != expected_epoch):
        raise ValueError("backend_epoch_changed_or_missing")
    if (canonical(contract.get("policy_config")) != canonical(asdict(demo_policy_config()))
            or contract.get("policy_digest") != POLICY_DIGEST):
        raise ValueError("actual_native_policy_does_not_match_demo")
    return epoch


def _number(value, name, nonnegative=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(name+"_invalid")
    return value


def _positive_int(value, name):
    if type(value) is not int or not 0 < value < 2**64:
        raise ValueError(name+"_invalid")
    return value


def _roi(value):
    fields = ("valid_fraction", "near_distance_m", "median_distance_m", "obstacle_fraction", "valid_samples", "total_samples")
    if not isinstance(value, dict) or any(key not in value for key in fields):
        raise ValueError("full_native_corridor_stats_required")
    for name in fields[:4]: _number(value[name], name, True)
    if not 0 <= value["valid_fraction"] <= 1 or not 0 <= value["obstacle_fraction"] <= 1:
        raise ValueError("corridor_fraction_invalid")
    if (type(value["valid_samples"]) is not int or type(value["total_samples"]) is not int
            or not 0 < value["valid_samples"] <= value["total_samples"]):
        raise ValueError("corridor_sample_count_invalid")
    return DepthRoiStats(**{key: value[key] for key in fields})


def native_decision(contract, expected_epoch=None):
    validate_contract(contract, expected_epoch)
    data = contract.get("decision")
    if not isinstance(data, dict): raise ValueError("original_native_decision_missing")
    state, direction = VerticalAvoidanceState(data.get("state")), VerticalDirection(data.get("direction"))
    if (state, direction) not in ((VerticalAvoidanceState.CLEAR, VerticalDirection.FORWARD),
                                  (VerticalAvoidanceState.HOLD, VerticalDirection.STOP),
                                  (VerticalAvoidanceState.EVADE, VerticalDirection.UP)):
        raise ValueError("unsupported_or_stale_native_decision")
    velocity = data.get("velocity_ned_mps")
    if not isinstance(velocity, (list, tuple)) or len(velocity) != 3:
        raise ValueError("full_native_velocity_required")
    for value in velocity: _number(value, "native_velocity")
    if state is VerticalAvoidanceState.EVADE and not (velocity[0] == velocity[1] == 0 and -.6 <= velocity[2] < 0):
        raise ValueError("unsupported_evade_velocity")
    if state is not VerticalAvoidanceState.EVADE and any(value != 0 for value in velocity):
        raise ValueError("unexpected_clear_or_hold_velocity")
    corridors = data.get("corridors")
    if not isinstance(corridors, dict): raise ValueError("native_corridors_missing")
    stats = DepthCorridorStats(*(_roi(corridors.get(name)) for name in ("upper", "center", "lower")),
                              *(_roi(corridors[name]) if corridors.get(name) is not None else None for name in ("left", "right")))
    trigger = _number(data.get("effective_trigger_distance_m"), "trigger")
    release = _number(data.get("effective_release_distance_m"), "release")
    if not 3. <= trigger < release: raise ValueError("native_thresholds_invalid")
    geometry = data.get("geometry")
    if not isinstance(geometry, dict) or geometry.get("geometry_valid") is not True:
        raise ValueError("native_geometry_missing_or_invalid")
    for key in ("roof_clearance_verified", "roof_passage_verified", "obstacle_extent_valid"):
        if type(geometry.get(key)) is not bool: raise ValueError("geometry_flag_missing:"+key)
    if geometry["roof_clearance_verified"]:
        if _number(geometry.get("roof_vertical_gap_m"), "roof_gap") <= 1.:
            raise ValueError("roof_gap_not_above_one_meter")
        _number(geometry.get("roof_height_m"), "roof_height")
    if geometry["obstacle_extent_valid"]:
        for key in ("obstacle_far_north_m", "obstacle_far_east_m"): _number(geometry.get(key), key)
    if geometry["roof_passage_verified"] and not geometry["obstacle_extent_valid"]:
        raise ValueError("passage_without_observed_extent")
    if type(data.get("descent_corridor_clear")) is not bool or not isinstance(data.get("reason"), str):
        raise ValueError("native_decision_fields_missing")
    return VerticalAvoidanceDecision(state, direction, tuple(velocity), data["reason"], stats,
                                    trigger, release, data["descent_corridor_clear"], deepcopy(geometry))


class BackendHealthCache:
    """Nonblocking readers; a separate thread supplies bounded health responses."""
    def __init__(self):
        self.lock = threading.Lock()
        self.epoch = None
        self.expires_ns = 0
        self.error = "backend_health_not_ready"
        self.retired = False

    def update(self, health, requested_ns, completed_ns):
        with self.lock:
            try:
                if self.retired: raise ValueError("backend_epoch_retired")
                if not isinstance(health, dict) or health.get("observation_only") is not True:
                    raise ValueError("dedicated_observation_backend_required")
                epoch = validate_contract(health.get("native_demo_contract"), self.epoch)
                if not 0 <= completed_ns-requested_ns <= 500_000_000:
                    raise ValueError("backend_health_request_expired")
                self.epoch, self.expires_ns, self.error = epoch, requested_ns+2*NS, ""
            except (ValueError, TypeError) as exc:
                if self.epoch is not None and isinstance(health, dict):
                    candidate = health.get("native_demo_contract", {})
                    if isinstance(candidate, dict) and candidate.get("backend_epoch") not in (None, self.epoch):
                        self.retired = True
                self.expires_ns, self.error = 0, str(exc)

    def current(self, now_ns):
        with self.lock:
            if self.error or now_ns >= self.expires_ns:
                raise ValueError(self.error or "backend_health_lease_expired")
            return self.epoch


class OriginalPoseBindings:
    """Exact source-stamp join; independently generated DDS receipt times never renew it."""
    def __init__(self, clock_id):
        self.clock_id, self.epoch, self.error = clock_id, None, ""
        self.rows = {kind: OrderedDict() for kind in ("LOCAL_POSITION_NED", "ATTITUDE_QUATERNION", "ESTIMATOR_STATUS")}
        self.lock = threading.Lock()

    def invalidate_pending(self):
        with self.lock:
            for rows in self.rows.values(): rows.clear()

    def receive(self, value, now_ns):
        with self.lock:
            if self.error: raise ValueError(self.error)
            if (not isinstance(value, dict) or value.get("evidence_clock_id") != self.clock_id
                    or value.get("mode") != "OBSERVE_ONLY" or value.get("source") != "physical_px4_usb_mavlink"
                    or type(value.get("system_id")) is not int or value["system_id"] != 1
                    or type(value.get("component_id")) is not int or value["component_id"] != 1
                    or value.get("acquisition_time_proven") is not False
                    or value.get("selected_protocol") != "legacy_correlated"
                    or value.get("response_target_checked") is not False
                    or value.get("legacy_exclusive_link_attested") is not True):
                raise ValueError("original_pose_identity_mismatch")
            epoch = (value.get("session_id"), value.get("link_identity"))
            if any(not isinstance(item, str) or not item for item in epoch): raise ValueError("pose_epoch_missing")
            if self.epoch is not None and epoch != self.epoch:
                self.error = "pose_epoch_changed_restart_required"
                self.rows = {kind: OrderedDict() for kind in self.rows}
                raise ValueError(self.error)
            self.epoch = epoch
            kind = value.get("kind")
            if kind not in self.rows: raise ValueError("pose_kind_invalid")
            stamp = _positive_int(value.get("source_timestamp_us"), "source_timestamp")
            earliest = _positive_int(value.get("source_local_earliest_ns"), "source_earliest")
            received = _positive_int(value.get("received_monotonic_ns"), "source_received")
            evaluated = _positive_int(value.get("evaluated_monotonic_ns"), "source_evaluated")
            expires = _positive_int(value.get("evidence_expires_monotonic_ns"), "source_expiry")
            sync_expiry = _positive_int(value.get("sync_expires_monotonic_ns"), "sync_expiry")
            if not (earliest <= received <= evaluated <= now_ns < expires <= min(earliest+250_000_000, received+250_000_000, sync_expiry)
                    and evaluated-earliest <= 200_000_000):
                raise ValueError("original_pose_expired_or_invalid")
            rows = self.rows[kind]
            if stamp in rows and canonical(rows[stamp]) != canonical(value):
                raise ValueError("original_pose_evidence_changed")
            if rows and stamp < next(reversed(rows)): raise ValueError("original_pose_evidence_reordered")
            rows[stamp] = deepcopy(value)
            while len(rows) > 32: rows.popitem(last=False)

    def match(self, position_stamp, attitude_stamp, now_ns):
        with self.lock:
            if self.error: raise ValueError(self.error)
            proofs = [self.rows[kind].get(stamp) for kind, stamp in
                      (("LOCAL_POSITION_NED", position_stamp), ("ATTITUDE_QUATERNION", attitude_stamp))]
            if not all(proofs): raise ValueError("exact_pose_original_evidence_missing")
            estimators = [item for item in self.rows["ESTIMATOR_STATUS"].values()
                          if abs(item["source_timestamp_us"]-position_stamp) <= 75000]
            if not estimators: raise ValueError("estimator_original_evidence_missing")
            proofs.append(min(estimators, key=lambda item: abs(item["source_timestamp_us"]-position_stamp)))
            if any(not item["evaluated_monotonic_ns"] <= now_ns < item["evidence_expires_monotonic_ns"] for item in proofs):
                raise ValueError("matched_pose_original_deadline_expired")
            return deepcopy(proofs), self.epoch


class PerceptionEnvelope:
    def __init__(self, clock_id, producer_session):
        self.clock_id, self.producer = clock_id, producer_session
        self.sequence, self.last_depth, self.backend_epoch = 0, 0, None

    def build(self, safety, binding, now_ns, backend_epoch):
        contract = safety.get("native_demo_contract")
        decision = native_decision(contract, backend_epoch)
        if self.backend_epoch is not None and backend_epoch != self.backend_epoch:
            raise ValueError("perception_backend_epoch_changed")
        self.backend_epoch = backend_epoch
        binding = deepcopy(binding)
        validate_input_binding(binding, self.clock_id, now_ns)
        if binding["depth_timestamp_ns"] <= self.last_depth:
            raise ValueError("perception_frame_replayed")
        if not 0 <= now_ns-binding["observed_monotonic_ns"] < 500_000_000 or not now_ns < binding["expires_monotonic_ns"] <= binding["observed_monotonic_ns"]+500_000_000:
            raise ValueError("perception_original_deadline_expired")
        self.last_depth = binding["depth_timestamp_ns"]
        self.sequence += 1
        evidence = DecisionEvidence(backend_epoch+":"+self.producer, self.sequence,
                                    binding["observed_monotonic_ns"]/NS, binding["expires_monotonic_ns"]/NS,
                                    decision, POLICY_DIGEST)
        return {"schema_version": 1, "mode": "DEMO_PERCEPTION_ONLY", "flight_commands_enabled": False,
                "evidence_clock_id": self.clock_id, "producer_session": self.producer,
                "backend_epoch": backend_epoch, "sequence": self.sequence,
                "published_monotonic_ns": now_ns, "input_binding": binding,
                "decision_evidence": _wire(asdict(evidence)), "native_demo_contract": deepcopy(contract),
                "valid": True, "reason": decision.reason,
                "timing_basis": "CAMERA_ROS_RECEIPT_WITH_ORIGINAL_FC_PUBLICATION_BOUNDS_NOT_EXPOSURE"}


def read_decision_evidence(envelope, clock_id, now_ns):
    """Consumer still owns session/sequence/flight permissions; this grants none."""
    if (not isinstance(envelope, dict) or envelope.get("mode") != "DEMO_PERCEPTION_ONLY"
            or envelope.get("flight_commands_enabled") is not False or envelope.get("valid") is not True
            or envelope.get("evidence_clock_id") != clock_id):
        raise ValueError("invalid_perception_envelope")
    binding, value = envelope["input_binding"], envelope["decision_evidence"]
    validate_input_binding(binding, clock_id, now_ns)
    _positive_int(envelope.get("sequence"), "perception_sequence")
    published = _positive_int(envelope.get("published_monotonic_ns"), "perception_publication")
    if published > now_ns: raise ValueError("future_perception_publication")
    if not binding["observed_monotonic_ns"] <= now_ns < binding["expires_monotonic_ns"] <= binding["observed_monotonic_ns"]+500_000_000:
        raise ValueError("perception_envelope_expired")
    decision = native_decision(envelope["native_demo_contract"], envelope["backend_epoch"])
    expected = DecisionEvidence(envelope["backend_epoch"]+":"+envelope["producer_session"], envelope["sequence"],
                                binding["observed_monotonic_ns"]/NS, binding["expires_monotonic_ns"]/NS, decision, POLICY_DIGEST)
    if canonical(_wire(asdict(expected))) != canonical(value): raise ValueError("perception_envelope_binding_changed")
    return expected


def validate_input_binding(binding, clock_id, now_ns):
    if not isinstance(binding, dict): raise ValueError("original_input_binding_missing")
    for name in ("rgb_timestamp_ns", "depth_timestamp_ns", "position_timestamp_us", "attitude_timestamp_us",
                 "angular_velocity_timestamp_us", "observed_monotonic_ns", "expires_monotonic_ns"):
        _positive_int(binding.get(name), name)
    if (abs(binding["rgb_timestamp_ns"]-binding["depth_timestamp_ns"]) > 75_000_000
            or abs(binding["position_timestamp_us"]-binding["attitude_timestamp_us"]) > 75000
            or binding["angular_velocity_timestamp_us"] != binding["attitude_timestamp_us"]
            or binding.get("camera_acquisition_time_proven") is not False
            or not isinstance(binding.get("depth_frame_id"), str) or not binding["depth_frame_id"]
            or binding.get("depth_encoding") != "16UC1"):
        raise ValueError("original_frame_pose_binding_invalid")
    for name in ("position_ned_m", "velocity_ned_mps"):
        values = binding.get(name)
        if not isinstance(values, (tuple, list)) or len(values) != 3: raise ValueError(name+"_missing")
        for item in values: _number(item, name)
    _number(binding.get("heading_rad"), "heading")
    proofs = binding.get("original_pose_evidence")
    if not isinstance(proofs, list) or len(proofs) != 3: raise ValueError("original_pose_proofs_missing")
    verifier = OriginalPoseBindings(clock_id)
    for proof in proofs: verifier.receive(proof, now_ns)
    verifier.match(binding["position_timestamp_us"], binding["attitude_timestamp_us"], now_ns)
    if verifier.epoch != (binding.get("fc_session_id"), binding.get("fc_link_identity")):
        raise ValueError("input_fc_epoch_mismatch")
    if not binding["observed_monotonic_ns"] <= now_ns < binding["expires_monotonic_ns"] <= min(
            binding["observed_monotonic_ns"]+500_000_000, *(proof["evidence_expires_monotonic_ns"] for proof in proofs)):
        raise ValueError("input_deadline_extended_or_expired")

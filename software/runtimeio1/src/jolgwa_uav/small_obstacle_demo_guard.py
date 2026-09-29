"""Offline, single-obstacle demo guard. Returns INTENTS; contains no flight IO.

The existing VerticalObstacleAvoidanceCore remains the avoidance classifier.
This module bounds a future adapter's session, not the real aircraft's motion.
User-confirmed CAD/level mounting is the demo assumption, not a new certificate.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math

from .vertical_avoidance import (
    VerticalAvoidanceConfig, VerticalAvoidanceDecision, VerticalAvoidanceState,
    VerticalDirection, VerticalObstacleAvoidanceCore,
)


class DemoState(str, Enum):
    WAIT_PILOT = "WAIT_PILOT"
    APPROACH = "APPROACH"
    STEP = "STEP"
    SETTLE = "SETTLE"
    PASS = "PASS"
    COMPLETE = "COMPLETE_HANDOVER"
    STOP = "STOP_HANDOVER"


def _finite(*values):
    return all(type(v) in (int, float) and math.isfinite(v) for v in values)


@dataclass(frozen=True)
class DemoConfig:
    # Operational caps for this isolated demo; not a validated flight envelope.
    route_length_m: float = 15.0
    corridor_half_width_m: float = 1.0
    ceiling_above_launch_m: float = 5.0
    duration_s: float = 60.0
    forward_speed_cap_mps: float = 0.5
    hover_height_m: float = 3.0
    hover_tolerance_m: float = 0.25
    hover_settle_s: float = 0.5
    settled_speed_mps: float = 0.1
    step_m: float = 1.0
    step_tolerance_m: float = 0.08
    step_settle_s: float = 0.2
    step_deceleration_mps2: float = 0.6
    step_reaction_s: float = 0.15
    trigger_floor_m: float = 3.0
    front_standoff_m: float = 2.5
    roof_gap_m: float = 1.0
    rear_pass_m: float = 5.0
    pose_lease_s: float = 0.25
    decision_lease_s: float = 0.5
    # Dedicated nominal 2 Hz VehicleStatus lease, separate from pose/decision.
    status_lease_s: float = 0.75
    heading_tolerance_rad: float = math.radians(3.0)
    clear_samples: int = 3

    def __post_init__(self):
        values = asdict(self)
        if not _finite(*[v for k, v in values.items() if k != "clear_samples"]):
            raise ValueError("nonfinite_demo_config")
        fixed = dict(hover_height_m=3., step_m=1., trigger_floor_m=3.,
                     front_standoff_m=2.5, roof_gap_m=1., rear_pass_m=5.,
                     pose_lease_s=.25, decision_lease_s=.5, status_lease_s=.75,
                     step_tolerance_m=.08, step_settle_s=.2,
                     step_deceleration_mps2=.6, step_reaction_s=.15)
        if any(getattr(self, k) != v for k, v in fixed.items()):
            raise ValueError("fixed_existing_safety_criteria_changed")
        if (not 6. < self.route_length_m <= 30. or not 0.25 <= self.corridor_half_width_m <= 2.
                or not 4. <= self.ceiling_above_launch_m <= 6.
                or not 5. <= self.duration_s <= 120.
                or not 0. < self.forward_speed_cap_mps <= 1.
                or not 0. < self.hover_tolerance_m <= .25
                or not .5 <= self.hover_settle_s <= 2.
                or not 0. < self.settled_speed_mps <= .1
                or not 0. < self.heading_tolerance_rad <= math.radians(3.)
                or type(self.clear_samples) is not int or self.clear_samples != 3):
            raise ValueError("invalid_bounded_demo_config")


def demo_policy_config() -> VerticalAvoidanceConfig:
    """Fixed snapshot of native D435 service defaults; no environment overrides.

    Exact source bindings are regression-tested against jetson_compute_service
    and patrol_stack.yaml. This is not the generic core's different defaults.
    """
    return VerticalAvoidanceConfig(
        trigger_distance_m=3., release_distance_m=5.5, minimum_standoff_m=2.5,
        emergency_margin_m=.75, reaction_time_s=.25, max_deceleration_mps2=1.,
        distance_uncertainty_m=.3, max_dynamic_trigger_distance_m=15.,
        trigger_samples=3, release_samples=3, min_evade_climb_m=0.,
        post_clear_climb_m=0., require_geometry=True, roof_minimum_gap_m=1.,
        stale_timeout_s=.5, vertical_speed_mps=.6, prefer_lateral_escape=False,
        outdoor_climb_default=True, allow_descent=False, min_depth_m=.15,
        max_depth_m=20., min_valid_fraction=.25, near_percentile=2.,
        min_obstacle_fraction=.02, min_pixel_confidence=0., min_frame_confidence=0.,
        max_pair_skew_s=.075, max_pose_age_s=.5, max_angular_rate_rad_s=1.5)


@dataclass(frozen=True)
class Telemetry:
    """Adapter-supplied actual telemetry in ONE monotonic clock domain.

    observed_at_s is the original pose/publication bound, not callback time.
    launch_z_ned_m must be measured on the ground in the SAME FC origin epoch.
    Status and every position/velocity validity flag must be true independently.
    No conversion from arbitrary clocks or missing positions is provided here.
    """
    fc_epoch: str
    observed_at_s: float
    status_at_s: float
    position_ned_m: tuple[float, float, float]
    velocity_ned_mps: tuple[float, float, float]
    yaw_rad: float
    launch_z_ned_m: float
    launch_reference_epoch: str
    position_valid: bool
    velocity_valid: bool
    armed: bool
    airborne: bool
    pilot_present: bool


@dataclass(frozen=True)
class DecisionEvidence:
    jetson_epoch: str
    sequence: int
    observed_at_s: float
    expires_at_s: float
    decision: VerticalAvoidanceDecision
    policy_digest: str


POLICY_DIGEST = hashlib.sha256(json.dumps(
    asdict(demo_policy_config()), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ExistingPolicyAdapter:
    """Offline adapter: real existing core, no alternative obstacle algorithm.

    For a remote producer, the future IO adapter must authenticate/bind the
    producer epoch, config and original leases; a matching digest alone is NOT
    proof. This helper consumes a caller's same-clock evaluation completion time.
    """
    def __init__(self, epoch):
        if not isinstance(epoch, str) or not epoch:
            raise ValueError("missing_policy_epoch")
        self.epoch = epoch
        self.core = VerticalObstacleAvoidanceCore(demo_policy_config())
        self.sequence = 0
        self.last_observed_at = -math.inf

    def evaluate(self, depth_m, *, observed_at_s, evaluated_at_s,
                 current_altitude_m, geometry, **metadata):
        if (not _finite(observed_at_s, evaluated_at_s) or observed_at_s < 0
                or observed_at_s <= self.last_observed_at or evaluated_at_s < observed_at_s):
            raise ValueError("nonadvancing_or_invalid_original_observation")
        self.last_observed_at = observed_at_s
        decision = self.core.evaluate(
            depth_m, frame_age_s=evaluated_at_s-observed_at_s,
            current_altitude_m=current_altitude_m, geometry=geometry, **metadata)
        # The core returns the decision; geometry is the separate existing
        # tracker output, attached as the service's runtime does today.
        from dataclasses import replace
        decision = replace(decision, geometry=deepcopy(geometry))
        self.sequence += 1
        return DecisionEvidence(self.epoch, self.sequence, observed_at_s,
                                observed_at_s+.5, decision, POLICY_DIGEST)


@dataclass(frozen=True)
class DemoIntent:
    state: DemoState
    intent: str
    reason: str
    expires_at_s: float
    target_ned_m: tuple[float, float, float] | None = None
    speed_cap_mps: float = 0.0
    hold_pose_available: bool = False
    pilot_handover_requested: bool = False
    flight_commands_enabled: bool = False
    mount_basis: str = "USER_CONFIRMED_MOUNT_APPROXIMATION"
    scope: str = "OFFLINE_INTENT_NOT_FLIGHT_AUTHORIZATION"


class SmallObstacleDemoGuard:
    def __init__(self, config=DemoConfig()):
        self.config = config
        self.state = DemoState.WAIT_PILOT
        self.last_now = -math.inf
        self.hover_since = None
        self.hover_last_pose = None
        self.reference = None
        self.origin = None
        self.heading = None
        self.epochs = None
        self.started_at = None
        self.used_tokens = set()
        self.last_sequence = 0
        self.last_evidence_digest = None
        self.last_observation = -math.inf
        self.step_target = None
        self.step_anchor = None
        self.settled_since = None
        self.settle_barrier = None
        self.farthest_progress = None
        self.encountered = False
        self.clear_count = 0
        self.clear_sequence = None
        self.terminal_reason = None
        self.last_pose_at = -math.inf
        self.last_pose_value = None
        self.pass_z = None

    def _intent(self, intent, reason, now, telemetry=None, evidence=None, *, target=None, speed=0.):
        active = self.state not in (DemoState.WAIT_PILOT, DemoState.STOP, DemoState.COMPLETE)
        lease = min(telemetry.observed_at_s+self.config.pose_lease_s,
                    telemetry.status_at_s+self.config.status_lease_s,
                    evidence.expires_at_s, evidence.observed_at_s+self.config.decision_lease_s
                    ) if active and telemetry is not None and evidence is not None else now
        pose_available = (telemetry is not None and self._telemetry_error(telemetry, now) == "")
        return DemoIntent(self.state, intent, reason, lease, target, speed, pose_available,
                          self.state in (DemoState.STOP, DemoState.COMPLETE))

    def _stop(self, reason, now, telemetry=None):
        self.state = DemoState.STOP
        self.terminal_reason = reason
        self.step_target = self.step_anchor = None
        return self._intent("STOP_HANDOVER", reason, now, telemetry)

    def _telemetry_error(self, t, now):
        if not isinstance(t, Telemetry):
            return "telemetry_missing"
        if (not isinstance(t.fc_epoch, str) or not t.fc_epoch
                or t.launch_reference_epoch != t.fc_epoch):
            return "fc_or_ground_reference_epoch_invalid"
        if (len(t.position_ned_m) != 3 or len(t.velocity_ned_mps) != 3
                or not _finite(*t.position_ned_m, *t.velocity_ned_mps,
                               t.yaw_rad, t.launch_z_ned_m, t.observed_at_s, t.status_at_s)):
            return "telemetry_nonfinite"
        if not all(value is True for value in (t.position_valid, t.velocity_valid,
                                              t.armed, t.airborne, t.pilot_present)):
            return "invalid_pose_velocity_or_pilot_flight_state"
        if math.hypot(*t.velocity_ned_mps[:2]) > 1.0:
            return "measured_horizontal_speed_exceeds_existing_1mps_demo_cap"
        if not 0 <= now-t.observed_at_s < self.config.pose_lease_s:
            return "pose_lease_expired_or_clock_mismatch"
        if not 0 <= now-t.status_at_s < self.config.status_lease_s:
            return "status_lease_expired_or_clock_mismatch"
        return ""

    def _evidence_error(self, e, now):
        if not isinstance(e, DecisionEvidence) or not isinstance(e.decision, VerticalAvoidanceDecision):
            return "jetson_decision_missing"
        if (not isinstance(e.jetson_epoch, str) or not e.jetson_epoch or type(e.sequence) is not int
                or e.sequence <= 0 or e.policy_digest != POLICY_DIGEST
                or not _finite(e.observed_at_s, e.expires_at_s)
                or not 0 <= now-e.observed_at_s < self.config.decision_lease_s
                or not e.observed_at_s < e.expires_at_s <= e.observed_at_s+self.config.decision_lease_s
                or now >= e.expires_at_s):
            return "jetson_epoch_profile_or_original_lease_invalid"
        d = e.decision
        if d.state not in (VerticalAvoidanceState.CLEAR, VerticalAvoidanceState.HOLD,
                            VerticalAvoidanceState.EVADE):
            return "jetson_stale_or_unsupported_state"
        if not isinstance(d.geometry, dict) or d.geometry.get("geometry_valid") is not True:
            return "jetson_geometry_unavailable"
        for key in ("roof_clearance_verified", "roof_passage_verified", "obstacle_extent_valid"):
            if type(d.geometry.get(key, False)) is not bool:
                return "jetson_geometry_flag_invalid"
        if d.geometry.get("roof_clearance_verified") is True and not (
                _finite(d.geometry.get("roof_vertical_gap_m"), d.geometry.get("roof_height_m"))
                and d.geometry["roof_vertical_gap_m"] > self.config.roof_gap_m):
            return "jetson_roof_clearance_contradiction"
        if d.geometry.get("roof_passage_verified") is True and d.geometry.get("obstacle_extent_valid") is not True:
            return "jetson_passage_extent_contradiction"
        if (not _finite(d.effective_trigger_distance_m, d.effective_release_distance_m)
                or d.effective_trigger_distance_m < 3.
                or d.effective_release_distance_m <= d.effective_trigger_distance_m):
            return "jetson_thresholds_invalid"
        if (d.corridors is None or not _finite(d.corridors.center.near_distance_m)
                or d.corridors.center.near_distance_m <= 0):
            return "front_distance_missing"
        if d.state is VerticalAvoidanceState.EVADE:
            if (d.direction is not VerticalDirection.UP or len(d.velocity_ned_mps) != 3
                    or not _finite(*d.velocity_ned_mps)
                    or abs(d.velocity_ned_mps[0])+abs(d.velocity_ned_mps[1]) > 1e-9
                    or not -.6 <= d.velocity_ned_mps[2] < 0):
                return "only_existing_upward_evade_supported"
        elif d.direction is not (VerticalDirection.FORWARD if d.state is VerticalAvoidanceState.CLEAR else VerticalDirection.STOP):
            return "contradictory_decision_direction"
        return ""

    def _progress(self, north, east):
        dx, dy = north-self.origin[0], east-self.origin[1]
        c, s = math.cos(self.heading), math.sin(self.heading)
        return dx*c+dy*s, -dx*s+dy*c

    def _settled(self, t, e, now):
        if math.hypot(*t.velocity_ned_mps) > self.config.settled_speed_mps:
            self.settled_since = self.settle_barrier = None
            return False
        if self.settled_since is None:
            self.settled_since = now
            return False
        if now-self.settled_since < self.config.step_settle_s:
            return False
        if self.settle_barrier is None:
            self.settle_barrier = e.sequence
            return False
        return e.sequence > self.settle_barrier and e.observed_at_s >= self.settled_since+self.config.step_settle_s

    def _start_step(self, t, now):
        height = t.launch_z_ned_m-t.position_ned_m[2]
        if height+self.config.step_m > self.config.ceiling_above_launch_m:
            return self._stop("next_full_step_exceeds_demo_ceiling", now, t)
        self.step_anchor = tuple(t.position_ned_m[:2])
        self.step_target = t.position_ned_m[2]-self.config.step_m
        self.settled_since = self.settle_barrier = None
        self.state = DemoState.STEP
        return None

    def tick(self, *, now_s, telemetry=None, evidence=None, pilot_start_token=None, manual_takeover=False):
        """Advance once; terminal sessions NEVER resume, even with a new token.

        A token is accepted only on the exact ready tick, not queued. The caller
        must continue ticking a watchdog; no Python object can stop a vehicle if
        its future IO process dies. Receipt/source clocks must already be mapped.
        """
        if self.state in (DemoState.STOP, DemoState.COMPLETE):
            return self._intent(self.state.value, self.terminal_reason,
                                now_s if _finite(now_s) else max(0., self.last_now))
        if not _finite(now_s) or now_s < 0 or now_s < self.last_now:
            return self._stop("monotonic_clock_invalid_or_regressed", max(0., self.last_now))
        if now_s-self.last_now >= self.config.pose_lease_s:
            if self.state is not DemoState.WAIT_PILOT:
                return self._stop("guard_tick_gap_pose_budget_exceeded", now_s, telemetry)
            self.hover_since = self.hover_last_pose = None
        self.last_now = now_s
        if manual_takeover is not False:
            return self._stop("manual_takeover_latched", now_s, telemetry)
        try:
            return self._tick(now_s, telemetry, evidence, pilot_start_token)
        except (ValueError, TypeError, AttributeError, KeyError, OverflowError):
            return self._stop("malformed_input", now_s)

    def _tick(self, now, t, e, token):
        c = self.config
        error = self._telemetry_error(t, now) or self._evidence_error(e, now)
        if error:
            if self.state is not DemoState.WAIT_PILOT:
                return self._stop(error, now, t)
            self.hover_since = self.hover_last_pose = None
            if token is not None:
                self._consume_token(token)
            return self._intent("WAIT", error, now)
        epochs = (t.fc_epoch, e.jetson_epoch)
        ref = (epochs, t.launch_z_ned_m)
        if self.state is not DemoState.WAIT_PILOT and ref != self.reference:
            return self._stop("source_epoch_or_ground_reference_changed", now, t)
        pose_value = (tuple(t.position_ned_m), tuple(t.velocity_ned_mps), t.yaw_rad)
        if ref == self.reference and (
                t.observed_at_s < self.last_pose_at or
                (t.observed_at_s == self.last_pose_at and pose_value != self.last_pose_value)):
            return self._stop("pose_reordered_or_changed_without_new_source", now, t)
        self.last_pose_at, self.last_pose_value = t.observed_at_s, pose_value
        if self.state is DemoState.WAIT_PILOT:
            height = t.launch_z_ned_m-t.position_ned_m[2]
            stable = (abs(height-c.hover_height_m) <= c.hover_tolerance_m
                      and math.hypot(*t.velocity_ned_mps) <= c.settled_speed_mps
                      and e.decision.state is VerticalAvoidanceState.CLEAR)
            if self.reference != ref or not stable:
                self.reference = ref
                self.hover_since = self.hover_last_pose = None
            if stable and (self.hover_last_pose is None or t.observed_at_s > self.hover_last_pose):
                self.hover_since = now if self.hover_since is None else self.hover_since
                self.hover_last_pose = t.observed_at_s
            ready = stable and self.hover_since is not None and now-self.hover_since >= c.hover_settle_s
            if token is None:
                return self._intent("WAIT", "pilot_start_required" if ready else "waiting_for_actual_3m_stable_hover", now, t)
            self._consume_token(token)
            if not ready:
                return self._intent("WAIT", "pilot_start_rejected_not_ready_new_token_required", now, t)
            self.origin = tuple(t.position_ned_m)
            self.heading = t.yaw_rad
            self.epochs = epochs
            self.started_at = now
            self.state = DemoState.APPROACH
        elif token is not None:
            return self._stop("new_start_not_allowed_during_demo", now, t)
        if now-self.started_at >= c.duration_s:
            return self._stop("demo_time_budget_expired", now, t)
        progress, lateral = self._progress(*t.position_ned_m[:2])
        height = t.launch_z_ned_m-t.position_ned_m[2]
        if (progress < -.25 or progress > c.route_length_m or abs(lateral) > c.corridor_half_width_m
                or height < c.hover_height_m-c.hover_tolerance_m or height > c.ceiling_above_launch_m):
            return self._stop("demo_geofence_or_altitude_limit", now, t)
        if abs(math.remainder(t.yaw_rad-self.heading, 2*math.pi)) > c.heading_tolerance_rad:
            return self._stop("fixed_camera_heading_changed", now, t)
        digest = hashlib.sha256(json.dumps(asdict(e), sort_keys=True, allow_nan=False).encode()).hexdigest()
        if (e.sequence < self.last_sequence or e.observed_at_s < self.last_observation
                or (e.sequence == self.last_sequence and digest != self.last_evidence_digest)
                or (e.sequence > self.last_sequence and e.observed_at_s <= self.last_observation)):
            return self._stop("decision_replayed_reordered_or_renewed", now, t)
        self.last_sequence, self.last_observation, self.last_evidence_digest = e.sequence, e.observed_at_s, digest
        d, geometry = e.decision, e.decision.geometry
        front = geometry.get("observed_front_distance_m")
        if front is not None and (not _finite(front) or front <= c.front_standoff_m):
            return self._stop("front_body_standoff_not_above_2p5m", now, t)
        # Missing full-body front geometry cannot prove standoff from a camera
        # ROI alone. CLEAR with no body hit is allowed by the existing tracker.
        if d.state is not VerticalAvoidanceState.CLEAR and front is None:
            return self._stop("obstacle_body_standoff_unknown", now, t)
        if geometry.get("obstacle_extent_valid") is True:
            far = (geometry.get("obstacle_far_north_m"), geometry.get("obstacle_far_east_m"))
            if not _finite(*far):
                return self._stop("obstacle_extent_nonfinite", now, t)
            far_progress, _ = self._progress(*far)
            if far_progress+c.rear_pass_m > c.route_length_m:
                return self._stop("approved_segment_cannot_cover_rear_pass", now, t)
            self.farthest_progress = far_progress if self.farthest_progress is None else max(self.farthest_progress, far_progress)
        roof_gap = geometry.get("roof_vertical_gap_m")
        roof_clear = geometry.get("roof_clearance_verified") is True and _finite(roof_gap) and roof_gap > c.roof_gap_m
        if d.state is VerticalAvoidanceState.CLEAR:
            if e.sequence != self.clear_sequence:
                self.clear_count += 1
                self.clear_sequence = e.sequence
        else:
            self.clear_count = 0
            self.clear_sequence = e.sequence
        if self.state is DemoState.PASS and d.state is not VerticalAvoidanceState.CLEAR:
            return self._stop("second_or_reappearing_obstacle_after_pass_started", now, t)
        if self.state is DemoState.PASS:
            if not roof_clear:
                return self._stop("roof_gap_evidence_lost_during_pass", now, t)
            if abs(t.position_ned_m[2]-self.pass_z) > c.hover_tolerance_m:
                return self._stop("pass_altitude_drift", now, t)
            if (geometry.get("roof_passage_verified") is True and self.farthest_progress is not None
                    and progress >= self.farthest_progress+c.rear_pass_m):
                self.state = DemoState.COMPLETE
                self.terminal_reason = "single_observed_obstacle_passed_no_automatic_descent"
                return self._intent("COMPLETE_HANDOVER", self.terminal_reason, now, t)
        if progress >= c.route_length_m:
            return self._stop("route_end_before_complete_single_pass", now, t)
        if self.state in (DemoState.STEP, DemoState.SETTLE):
            if math.hypot(t.position_ned_m[0]-self.step_anchor[0], t.position_ned_m[1]-self.step_anchor[1]) > .25:
                return self._stop("step_xy_anchor_drift", now, t)
            remaining = t.position_ned_m[2]-self.step_target
            if remaining < -c.step_tolerance_m:
                return self._stop("step_overshoot", now, t)
            if self.state is DemoState.STEP and (remaining <= c.step_tolerance_m or d.state is not VerticalAvoidanceState.EVADE):
                self.state = DemoState.SETTLE
                self.settled_since = self.settle_barrier = None
            if self.state is DemoState.SETTLE:
                if not self._settled(t, e, now):
                    return self._intent("HOLD", "settle_and_new_original_frame_required", now, t, e)
                if d.state is VerticalAvoidanceState.CLEAR:
                    if roof_clear and self.clear_count >= c.clear_samples and self.farthest_progress is not None:
                        self.state = DemoState.PASS
                        self.pass_z = t.position_ned_m[2]
                        self.step_anchor = self.step_target = None
                    else:
                        return self._intent("HOLD", "roof_gap_extent_and_clear_confirmations_required", now, t, e)
                elif d.state is VerticalAvoidanceState.EVADE:
                    if remaining <= c.step_tolerance_m:
                        stopped = self._start_step(t, now)
                        if stopped:
                            return stopped
                    else:
                        self.state = DemoState.STEP  # same partial target, no target extension
                    self.settled_since = self.settle_barrier = None
                else:
                    return self._intent("HOLD", "existing_policy_hold", now, t, e)
        if d.state is VerticalAvoidanceState.HOLD:
            return self._intent("HOLD", "existing_policy:"+d.reason, now, t, e)
        if d.state is VerticalAvoidanceState.EVADE:
            self.encountered = True
            if self.state is DemoState.APPROACH:
                # Stop horizontal drift before an upward target is permitted.
                if math.hypot(*t.velocity_ned_mps[:2]) > c.settled_speed_mps:
                    return self._intent("HOLD", "brake_horizontal_before_step", now, t, e)
                stopped = self._start_step(t, now)
                if stopped:
                    return stopped
            remaining = t.position_ned_m[2]-self.step_target
            usable = max(0., remaining-.5*c.step_tolerance_m)
            upward = max(0., -t.velocity_ned_mps[2])
            a, reaction = c.step_deceleration_mps2, c.step_reaction_s
            braking = max(0., math.sqrt((a*reaction)**2+2*a*usable)-a*reaction)
            speed = min(-d.velocity_ned_mps[2], braking, 2*usable)
            if upward*reaction+upward*upward/(2*a) >= usable:
                speed = 0.
            target = (*self.step_anchor, self.step_target)
            return self._intent("CLIMB_STEP" if speed > 0 else "HOLD", "existing_upward_policy_with_1m_braking_target",
                                now, t, e, target=target, speed=speed)
        if self.state in (DemoState.APPROACH, DemoState.PASS):
            # Only advances along the one approved ray; no autonomous descent.
            target = (self.origin[0]+math.cos(self.heading)*c.route_length_m,
                      self.origin[1]+math.sin(self.heading)*c.route_length_m,
                      self.pass_z if self.state is DemoState.PASS else self.origin[2])
            return self._intent("ADVANCE", "existing_clear_policy", now, t, e,
                                target=target, speed=c.forward_speed_cap_mps)
        return self._intent("HOLD", "waiting_for_existing_policy", now, t, e)

    def _consume_token(self, token):
        if not isinstance(token, str) or not 1 <= len(token) <= 128 or token in self.used_tokens or len(self.used_tokens) >= 64:
            raise ValueError("invalid_reused_or_excess_pilot_token")
        self.used_tokens.add(token)

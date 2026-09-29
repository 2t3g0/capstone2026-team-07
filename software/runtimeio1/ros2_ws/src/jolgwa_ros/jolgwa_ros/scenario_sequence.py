"""Deterministic single-obstacle / single-incident sequencing, no I/O."""
import math
from .scenario_contract import point, progress, within
from .event_control import route_rejoin_with_lookahead


class ScenarioSequence:
    def __init__(self, spec, now):
        self.spec = spec
        self.phase = "SEARCH_OBSTACLE"
        self.started = self.phase_started = now
        self.target = point(spec, 10.0)
        self.reason = ""
        self.event = None
        self.event_opened = math.inf
        self.brake = None
        self.pass_progress = None
        self.settled_at = None
        self.depth_missing_since = None
        self.depth_waiting = False
        self.resume_target = None
        self.descent_started = None
        self.climb_started = None
        self.climb_clear_hold = False
        self.climb_ceiling_target = None
        self.pass_clear_observed = -math.inf
        self.brake_elapsed = 0.0

    def invalidate_pre_reset_clear(self, reset, position):
        self.settled_at = None
        if self.phase == 'PASS' and self.pass_clear_observed <= reset:
            # Metadata may follow heading DDS. Return only to the same ascent
            # HOLD/settle, preserving its original deadline and height ceiling.
            self.phase = 'CLIMB'
            self.phase_started = self.climb_started
            self.climb_clear_hold = True
            self.target = tuple(position)
            if self.depth_waiting:
                self.resume_target = self.target

    def update_boundary(self, evidence):
        far = evidence.get("far")
        if evidence.get("geometry_valid") and far is not None and len(far) == 3 and all(math.isfinite(v) for v in far):
            required = progress(self.spec, far)[0] + self.spec["clearance_m"]
            self.pass_progress = max(self.pass_progress or 0., required)
        return self.pass_progress is not None

    def fail(self, reason):
        self.reason = reason
        self.phase = "FAILED"
        return self.phase, self.target

    def transition(self, phase, target, now):
        if self.phase == "BRAKE":
            self.brake_elapsed += max(0., now-self.phase_started)
        if phase == "DESCEND":
            if self.descent_started is None:
                self.descent_started = now
            # Boundary extension may temporarily return to PASS. Reaching
            # its new target must not grant another full descent deadline.
            now = self.descent_started
        self.phase, self.target, self.phase_started = phase, tuple(target), now
        self.settled_at = None
        return phase, self.target

    def settled(self, position, velocity, now):
        stable = (math.dist(position, self.target) <= .25
                  and abs(position[2]-self.target[2]) <= .15
                  and all(math.isfinite(v) for v in velocity)
                  and math.sqrt(sum(v*v for v in velocity)) <= .1)
        if not stable:
            self.settled_at = None
            return False
        if self.settled_at is None:
            self.settled_at = now
        return now-self.settled_at >= .5

    def tick(self, position, velocity, evidence, now, *, event=None, photo=None):
        if self.phase in ("FAILED", "LAND"):
            return self.phase, self.target
        if not math.isfinite(now) or now < self.phase_started or now >= self.started+self.spec["total_timeout_s"]:
            return self.fail("scenario_deadline")
        if not within(self.spec, position):
            return self.fail("scenario_position_outside_prepared_envelope")
        # Deadlines precede evidence/completion acceptance, including after a
        # delayed executor callback. A HOLD never restarts a stage deadline.
        elapsed = now-self.phase_started
        if self.phase == "BRAKE" and self.brake_elapsed+elapsed >= self.spec["stage_timeout_s"]:
            return self.fail("scenario_stage_timeout:BRAKE")
        if self.phase == "WAIT_EVENT" and now >= self.phase_started+self.spec["incident_wait_s"]:
            return self.fail("littering_not_detected")
        if self.phase == "PHOTO" and now >= self.phase_started+self.spec["photo_timeout_s"]:
            return self.fail("photo_save_timeout")
        if self.phase not in ("SEARCH_OBSTACLE", "SEARCH_EVENT", "WAIT_EVENT") and now >= self.phase_started+self.spec["stage_timeout_s"]:
            return self.fail("scenario_stage_timeout:"+self.phase)
        if self.depth_missing_since is not None and now >= self.depth_missing_since+self.spec["depth_wait_s"]:
            return self.fail("scenario_depth_recovery_timeout")
        if (evidence is None or not math.isfinite(evidence.get("observed", math.nan))
                or not 0 <= now-evidence["observed"] <= .5
                or evidence.get("state") not in (0, 1, 2, 3)):
            if self.depth_missing_since is None:
                self.depth_missing_since = now
                self.resume_target = self.target
                self.target = tuple(position)
                self.settled_at = None
            self.depth_waiting = True
            return self.phase, self.target
        if self.depth_waiting:
            self.target = self.resume_target
            self.depth_missing_since = None
            self.depth_waiting = False
            self.settled_at = None
        state = evidence["state"]
        if self.phase == "SEARCH_OBSTACLE":
            if state != 0:
                self.brake = tuple(position)
                return self.transition("BRAKE", self.brake, now)
            if self.settled(position, velocity, now):
                return self.fail("obstacle_stage_not_performed")
        elif self.phase == "BRAKE":
            if self.settled(position, velocity, now):
                # Require a newly observed blocked front after actual stop.
                if evidence["observed"] < self.settled_at+.5:
                    return self.phase, self.target
                if state == 0:
                    return self.transition("SEARCH_OBSTACLE", point(self.spec, 10.0), now)
                self.climb_started = now
                self.climb_ceiling_target = (*self.brake[:2], point(self.spec, 0)[2]-self.spec["climb_m"])
                return self.transition("CLIMB", self.climb_ceiling_target, now)
        elif self.phase == "CLIMB":
            if state == 0 and evidence["observed"] >= self.climb_started:
                if not self.climb_clear_hold:
                    self.climb_clear_hold = True
                    self.target = (*position[:2], max(position[2], self.climb_ceiling_target[2]))
                    self.settled_at = None
                if self.settled(position, velocity, now):
                    if evidence["observed"] < self.settled_at+.5:
                        return self.phase, self.target
                    yaw = self.spec["heading_rad"]
                    target = (position[0]+self.spec["pass_m"]*math.cos(yaw),
                              position[1]+self.spec["pass_m"]*math.sin(yaw), position[2])
                    if progress(self.spec, target)[0] > self.spec["search_m"] or not within(self.spec, target, target=True):
                        return self.fail("obstacle_pass_outside_search")
                    self.pass_progress = progress(self.spec, target)[0]
                    self.pass_clear_observed = evidence['observed']
                    return self.transition("PASS", target, now)
            elif state != 0:
                if self.climb_clear_hold:
                    self.climb_clear_hold = False
                    self.target = self.climb_ceiling_target
                    self.settled_at = None
                if self.settled(position, velocity, now):
                    return self.fail("maximum_climb_insufficient")
        elif self.phase == "PASS":
            if state != 0:
                return self.fail("pass_blocked_no_second_climb")
            if self.settled(position, velocity, now):
                return self.transition("DESCEND", (*self.target[:2], point(self.spec, 0)[2]), now)
        elif self.phase == "DESCEND":
            if state != 0:
                return self.fail("descent_front_obstructed")
            if self.settled(position, velocity, now):
                self.event_opened = now
                return self.transition("SEARCH_EVENT", point(self.spec, 10), now)
        elif self.phase in ("SEARCH_EVENT", "WAIT_EVENT"):
            if state != 0:
                return self.fail("second_obstacle")
            if event is not None and self.event is None:
                if (event.get("type") == "LITTERING"
                        and self.event_opened <= event.get("input_received", -math.inf) <= now
                        and 0 <= now-event.get("input_received", -math.inf) <= self.spec["event_max_age_s"]):
                    self.event = event
                    return self.transition("EVENT_HOLD", position, now)
            if self.phase == "SEARCH_EVENT" and self.settled(position, velocity, now):
                return self.transition("WAIT_EVENT", self.target, now)
        elif self.phase == "EVENT_HOLD":
            if state != 0:
                return self.fail("event_hold_obstructed")
            if self.settled(position, velocity, now):
                if self.event is not None and self.event.get("capture_status", "SAVED") != "SAVED":
                    return self.fail("incident_photo_save_failed:"+str(self.event.get("error", "capture unavailable")))
                return self.transition("PHOTO", self.target, now)
        elif self.phase == "PHOTO":
            if state != 0:
                return self.fail("photo_hold_obstructed")
            if photo == "failed":
                return self.fail("photo_save_failed")
            if photo == "saved":
                rejoin = route_rejoin_with_lookahead(position,
                    (point(self.spec, 0), point(self.spec, 10)), 0.0)
                return self.transition("REJOIN", rejoin.position_ned_m, now)
        elif self.phase == "REJOIN":
            if state != 0:
                return self.fail("rejoin_obstructed")
            if self.settled(position, velocity, now):
                # Distance is from the actual settled rejoin position, using
                # the heading frozen by preparation (not the current yaw).
                yaw = self.spec["heading_rad"]
                target = (position[0]+math.cos(yaw), position[1]+math.sin(yaw), point(self.spec, 0)[2])
                if not within(self.spec, target, target=True):
                    return self.fail("final_target_outside_prepared_envelope")
                return self.transition("FINAL", target, now)
        elif self.phase == "FINAL":
            if state != 0:
                return self.fail("final_leg_obstructed")
            if self.settled(position, velocity, now):
                return self.transition("LAND", self.target, now)
        return self.phase, self.target

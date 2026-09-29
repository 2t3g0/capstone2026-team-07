"""Versioned, approval-bound envelope for the opt-in field scenario.

Uses existing proposal/approval messages. It never turns a missing proposal
into a legacy camera bypass and never grants authority itself.
"""
from collections import OrderedDict
from copy import deepcopy
import json
import math

BUILD_ID = "low-speed-runtimeio1-20260929"
SPEC = dict(version=7, build_id=BUILD_ID, profile="FRONT_DETECT_2M_PASS_3M_V1",
            search_m=10.0, final_m=1.0, climb_m=2.0, pass_m=3.0, detect_m=2.0, release_m=4.5, minimum_standoff_m=1.0, max_dynamic_trigger_m=2.0, clearance_m=1.0,
            incident_wait_s=20.0, depth_wait_s=5.0, extent_wait_s=5.0,
            stage_timeout_s=60.0, photo_timeout_s=10.0, event_max_age_s=5.0,
            total_timeout_s=240.0, metadata_wait_s=2.0, yaw_reset_policy="BOUNDED_YAW_V1",
            battery_low_s=5.0, battery_recovery_s=1.0, home_confirmation_ms=2000,
            contract_pair_wait_ms=500, takeoff_radius_m=0.5,
            altitude_recovery_ms=500, brake_clear_policy="RESUME_SEARCH")


def make_spec(start, yaw, home_z, epoch, altitude):
    return dict(SPEC, start_ned_m=list(start), heading_rad=float(yaw),
                home_z_ned_m=float(home_z), transport_epoch=int(epoch),
                cruise_altitude_m=float(altitude))


def validate_spec(plan):
    if not isinstance(plan, dict) or not isinstance(plan.get("low_speed_limits"), dict):
        raise ValueError("scenario plan/limits invalid")
    spec = plan.get("test_scenario")
    if not isinstance(spec, dict) or set(spec) != set(SPEC) | {
            "start_ned_m", "heading_rad", "home_z_ned_m", "transport_epoch",
            "cruise_altitude_m"}:
        raise ValueError("scenario metadata missing or unknown")
    if any(type(spec[k]) != type(v) or spec[k] != v for k, v in SPEC.items()):
        raise ValueError("scenario version/limits mismatch")
    start = spec["start_ned_m"]
    if not isinstance(start, list) or len(start) != 3:
        raise ValueError("scenario start invalid")
    values = [*start, spec["heading_rad"], spec["home_z_ned_m"], spec["cruise_altitude_m"]]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("scenario nonfinite geometry")
    if (type(spec["transport_epoch"]) is not int or spec["transport_epoch"] <= 0
            or spec["cruise_altitude_m"] not in (1.0, 2.0)
            or plan.get("mission_kind") != "FORWARD_TEST_1M"
            or start != plan.get("preview_start_ned_m")
            or spec["heading_rad"] != plan.get("preview_heading_rad")
            or plan.get("low_speed_limits", {}).get("target_altitude_home_m") != spec["cruise_altitude_m"]):
        raise ValueError("scenario preparation mismatch")
    end = point(spec, 10.0, z=start[2])
    if list(end) != plan.get("preview_end_ned_m"):
        raise ValueError("scenario endpoint mismatch")
    return deepcopy(spec)


def progress(spec, position):
    dx, dy = position[0]-spec["start_ned_m"][0], position[1]-spec["start_ned_m"][1]
    c, s = math.cos(spec["heading_rad"]), math.sin(spec["heading_rad"])
    return dx*c+dy*s, -dx*s+dy*c


def point(spec, along, *, z=None):
    return (spec["start_ned_m"][0]+along*math.cos(spec["heading_rad"]),
            spec["start_ned_m"][1]+along*math.sin(spec["heading_rad"]),
            spec["home_z_ned_m"]-spec["cruise_altitude_m"] if z is None else z)


def within(spec, position, *, target=False, takeoff=False):
    if len(position) != 3 or not all(math.isfinite(v) for v in position):
        return False
    along, cross = progress(spec, position)
    # Preserve the preparation drift and low-speed arrival tolerance. Targets
    # cannot claim a new route outside the prepared straight corridor.
    horizontal = (math.hypot(along, cross) <= spec["takeoff_radius_m"]
                  if takeoff and not target else
                  -0.2 <= along <= 11.0+(0.0 if target else 0.25)
                  and abs(cross) <= (0.25 if target else 0.5))
    return (horizontal
            and spec["home_z_ned_m"]-position[2] <= (
+                spec["cruise_altitude_m"]+spec["climb_m"] if target else ceiling(spec))+1e-6)


def ceiling(spec):
    return spec["cruise_altitude_m"]+spec["climb_m"]+0.5


class ScenarioRegistry:
    """Handle cross-topic reordering without allowing proposal replacement."""
    def __init__(self):
        self.proposals = OrderedDict()
        self.approvals = {}
        self.retired = set()
        self.poisoned = set()

    def proposal(self, message):
        key = message.proposal_id
        try:
            plan = json.loads(message.plan_json)
            spec = validate_spec(plan)
        except (ValueError, TypeError, KeyError):
            self.poisoned.add(key)
            return
        if key in self.proposals and self.proposals[key] != spec:
            self.poisoned.add(key)
        self.proposals[key] = spec
        if len(self.proposals) > 128:
            old, _ = self.proposals.popitem(last=False)
            self.poisoned.add(old)

    def approval(self, message):
        mid = message.mission_id
        if not message.approved:
            self.retired.add(mid)
            self.approvals.pop(mid, None)
            return
        if mid in self.retired:
            return
        previous = self.approvals.get(mid)
        if previous is not None and previous != message.proposal_id:
            self.retired.add(mid)
            self.approvals.pop(mid, None)
        else:
            self.approvals[mid] = message.proposal_id

    def get(self, mission_id):
        pid = self.approvals.get(mission_id)
        if mission_id in self.retired or pid in self.poisoned:
            return None
        return self.proposals.get(pid)

    def command_spec(self, command):
        spec = self.get(command.mission_id)
        if (spec is None or not command.approved
                or not math.isfinite(command.home_z_ned_m)
                or abs(command.home_z_ned_m-spec["home_z_ned_m"]) > 1e-5
                or command.altitude_reference_epoch != spec["transport_epoch"]
                or command.flight_profile != (2 if spec["cruise_altitude_m"] == 1 else 1)):
            return None
        return spec

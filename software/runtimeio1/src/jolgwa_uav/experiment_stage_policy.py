"""Immutable stage capabilities; no transport, approval, I/O or flight.

Zero preserves the existing runtime. Stage 1 is inert ground integration;
stage 2 is the integrated real-Jetson-compute/SITL rehearsal, NOT the old
single-obstacle USB demo. Stage 3 is judgement-only on the approved route,
not collision avoidance: perception must neither stop nor move that route.
"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExperimentStagePolicy:
    stage: int
    name: str
    route: bool
    perception_controls_route: bool
    avoidance: bool
    events: bool
    visited_home: bool
    physical_transport_ready: bool = False

    def as_dict(self):
        return asdict(self)


_POLICIES = {
    0: ExperimentStagePolicy(0, "legacy", True, True, True, True, True),
    1: ExperimentStagePolicy(1, "ground", False, False, False, False, False),
    2: ExperimentStagePolicy(2, "jetson-sitl", True, True, True, True, True),
    3: ExperimentStagePolicy(3, "route-observe", True, False, False, False, False),
    4: ExperimentStagePolicy(4, "route-avoid-resume", True, True, True, False, False),
    5: ExperimentStagePolicy(5, "event-only", True, True, False, True, True),
    6: ExperimentStagePolicy(6, "integrated", True, True, True, True, True),
}


def policy_for_stage(stage: int) -> ExperimentStagePolicy:
    if type(stage) is not int or stage not in _POLICIES:
        raise ValueError("experiment_stage must be an integer in 0..6")
    return _POLICIES[stage]


def validate_controller_profile(stage: int, *, diagnostic_relaxed_timing: bool,
        require_jetson_safety: bool, route_heading_gate_enabled: bool,
        require_roof_geometry: bool, require_descent_corridor_clear: bool,
        simulation_only=None, allow_real_hardware=None):
    policy = policy_for_stage(stage)
    if stage == 0:
        return policy
    if not policy.route:
        raise ValueError("stage 1 is ground-only, not a route controller mode")
    if stage == 2 and not (simulation_only is True and allow_real_hardware is False):
        raise ValueError("stage 2 requires SITL-only FC output, never physical FC control")
    if diagnostic_relaxed_timing:
        raise ValueError("field experiment stages do not inherit SIM relaxed timing")
    if stage == 3:
        if require_jetson_safety or route_heading_gate_enabled:
            raise ValueError("stage 3 is observation-only: perception/heading route intervention must be off")
    elif not all((require_jetson_safety, route_heading_gate_enabled,
                  require_roof_geometry, require_descent_corridor_clear)):
        raise ValueError("stages 4/5/6 require original depth, heading, roof and descent guards")
    return policy


def validate_manager_profile(stage: int, *, enable_event_control: bool,
        completion_policy: str, diagnostic_relaxed_timing: bool,
        simulation_only=None, allow_real_hardware=None):
    policy = policy_for_stage(stage)
    if stage == 0:
        return policy
    if not policy.route or diagnostic_relaxed_timing:
        raise ValueError("route experiment requires stage 2..6 and non-relaxed timing")
    if stage == 2 and not (simulation_only is True and allow_real_hardware is False):
        raise ValueError("stage 2 requires SITL-only FC output, never physical FC control")
    if enable_event_control is not policy.events:
        raise ValueError("event control activation contradicts the immutable experiment stage")
    expected = "capture_then_rtl" if policy.visited_home else "legacy_rejoin"
    if completion_policy != expected:
        raise ValueError("experiment stage requires event_completion_policy=" + expected)
    return policy


def ros_stage_parameters(stage: int) -> dict:
    """Role/strict-guard parameters only: no output enable, approval or Home.

Stages 5/6 still need independently verified Home provenance. The physical
adapter does not exist; single_publisher_sitl is not a physical alternative.
"""
    policy = policy_for_stage(stage)
    if stage not in (2, 3, 4, 5, 6):
        raise ValueError("ROS stage parameters exist only for stages 2..6")
    parameters = {
        "px4_offboard_controller": {
            "experiment_stage": stage, "diagnostic_relaxed_timing": False,
            "require_jetson_safety": policy.perception_controls_route,
            "route_heading_gate_enabled": policy.perception_controls_route,
            "require_roof_geometry": True, "require_descent_corridor_clear": True,
            "terminal_requires_owned_offboard": True,
        },
        "mission_manager": {
            "experiment_stage": stage, "diagnostic_relaxed_timing": False,
            "enable_event_control": policy.events,
            "event_completion_policy": "capture_then_rtl" if policy.visited_home else "legacy_rejoin",
        },
        "event_response": {"enabled": policy.events, "recording_enabled": policy.events},
    }
    if stage == 2:
        for node in ("px4_offboard_controller", "mission_manager", "event_response"):
            parameters[node].update(simulation_only=True, allow_real_hardware=False)
    return parameters


PHYSICAL_ROUTE_TRANSPORT_GAPS = (
    "single USB owner must relay original FC pose/status/land/Home with source timestamp, "
    "origin/reset/armed epoch and original expiry to the ROS route controller",
    "controller output must return to that SAME owner with matching approved mission, "
    "sequence and FC epoch; no second USB reader or parallel setpoint writer",
    "physical Home provenance and cross-process clock identity must be preserved; "
    "single_publisher_sitl and SIM baseline helper are not hardware transports",
)

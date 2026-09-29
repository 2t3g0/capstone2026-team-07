"""ROS adapters for the opt-in scenario; all flight output stays in the owner."""
import hashlib
import json
import math
from pathlib import Path
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from .scenario_contract import ScenarioRegistry, ceiling, within, validate_spec, progress
from .scenario_sequence import ScenarioSequence
from .field_diagnostics import record as field_record

INCIDENT_TOPIC = "/jolgwa/internal/scenario_incident"


def init_contract(node):
    from rcl_interfaces.msg import ParameterDescriptor
    from jolgwa_interfaces.msg import MissionApproval, MissionProposal
    from .topic_names import MISSION_APPROVAL, MISSION_PROPOSAL
    node.declare_parameter("test_scenario_enabled", False, ParameterDescriptor(read_only=True))
    node._scenario_enabled = bool(node.get_parameter("test_scenario_enabled").value)
    node._scenario_registry = ScenarioRegistry()
    if node._scenario_enabled:
        node.create_subscription(MissionProposal, MISSION_PROPOSAL, node._scenario_registry.proposal, 10)
        node.create_subscription(MissionApproval, MISSION_APPROVAL, node._scenario_registry.approval, 10)


def spec_for(node, command):
    if not getattr(node, "_scenario_enabled", False) or command is None:
        return None
    return node._scenario_registry.command_spec(command)


def altitude_limit(node, command, default):
    spec = spec_for(node, command)
    return ceiling(spec) if spec is not None else default


def navigation_error(node, command, current=None):
    if not getattr(node, "_scenario_enabled", False) or command.command in (3, 4, 5):
        return ""
    spec = spec_for(node, command)
    if spec is None:
        return "scenario_approval_metadata_missing_or_mismatched"
    if not within(spec, command.position_ned_m, target=True):
        return "scenario_target_outside_prepared_envelope"
    stages = getattr(node, '_scenario_takeoff_stages', None)
    if stages is None:
        stages = node._scenario_takeoff_stages = {}
    entry = stages.setdefault(command.mission_id, [command.sequence, command.command != 1])
    recovery_hold = (command.command == 0 and command.sequence == entry[0] and not entry[1])
    if (command.command != 1 or command.sequence != entry[0]) and not recovery_hold:
        entry[1] = True
    takeoff = command.command in (0, 1) and command.sequence == entry[0] and not entry[1]
    if command.command == 1 and not takeoff:
        return 'scenario_takeoff_phase_retired'
    if current is not None and not within(spec, current, takeoff=takeoff):
        along, cross = progress(spec, current)
        detail = dict(event='scenario_envelope_rejected', mission_id=command.mission_id,
            sequence=command.sequence, phase='TAKEOFF' if takeoff else 'ROUTE',
            start=spec['start_ned_m'], current=[float(v) for v in current], target=[float(v) for v in command.position_ned_m],
            along_m=along, cross_m=cross, radius_m=math.hypot(along,cross),
            takeoff_radius_m=spec['takeoff_radius_m'], route_rear_m=.2, route_cross_m=.5)
        if hasattr(node, 'get_logger'):
            node.get_logger().warning(json.dumps(detail))
        return "scenario_position_outside_prepared_envelope"
    return ""


def init_manager(node):
    from jolgwa_interfaces.msg import SafetyDecision
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import String
    from rclpy.qos import qos_profile_sensor_data
    from .topic_names import OBSTACLE_SAFETY_DECISION, FORWARD_CAMERA_COMPRESSED
    init_contract(node)
    if not node._scenario_enabled:
        return
    node.declare_parameter("scenario_photo_root", "~/outputs/scenario_photos")
    node.declare_parameter("scenario_camera_topic", "/camera/camera/color/image_raw/compressed")
    node._scenario_photo_root = Path(node.get_parameter("scenario_photo_root").value).expanduser()
    node._scenario_inputs = Inputs(node)
    node.create_subscription(SafetyDecision, OBSTACLE_SAFETY_DECISION, node._scenario_inputs.safety, qos_profile_sensor_data)
    node.create_subscription(CompressedImage, node.get_parameter("scenario_camera_topic").value,
                             node._scenario_inputs.image, qos_profile_sensor_data)
    node.create_subscription(String, INCIDENT_TOPIC, node._scenario_inputs.incident, 10)


class Inputs:
    def __init__(self, node):
        self.node = node
        self.evidence = self.frame = self.event = None
        self.last_sequence = -1
        self.seen = set()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scenario-photo")

    def safety(self, message):
        now = time.monotonic()
        age = float(message.observation_age_s)
        ros_now = self.node.get_clock().now().nanoseconds/1e9
        stamp = message.stamp.sec+message.stamp.nanosec/1e9
        delay = ros_now-stamp
        field_record(self.node, 'manager_safety_receive', sequence=int(message.sequence),
            input_age_s=age, callback_delay_s=delay, state=int(message.state),
            valid=bool(message.source == 'jetson-realsense-d435i' and message.sequence > self.last_sequence
                and stamp > 0 and math.isfinite(delay) and delay >= -.05 and math.isfinite(age)
                and age >= 0 and 0 <= age+max(0.,delay)+.05 <= .5 and message.state in (0,1,2,3)),
            previous_sequence=self.last_sequence)
        with self.node._lock:
            if (message.source != "jetson-realsense-d435i" or message.sequence <= self.last_sequence
                    or not math.isfinite(age) or age < 0 or not math.isfinite(delay)
                    or stamp <= 0 or delay < -.05):
                return
            self.last_sequence = message.sequence
            if now-age-max(0., delay)-.05 <= getattr(self.node, '_scenario_yaw_reset_at', -math.inf):
                return
            far = ((float(message.obstacle_far_north_m), float(message.obstacle_far_east_m), 0.)
                   if message.obstacle_extent_valid else None)
            self.evidence = dict(observed=now-age-max(0., delay)-.05, sequence=message.sequence,
                state=(message.state if getattr(message, "reason", "").startswith("scenario_front_detect_2m_pass_3m_v1:") else 4),
                geometry_valid=message.geometry_valid, far=far,
                roof_clear=message.roof_clearance_verified, roof_gap=float(message.roof_vertical_gap_m),
                passed=message.roof_passage_verified, descent_clear=message.descent_corridor_clear)

    def image(self, message):
        payload = bytes(message.data)
        stamp = message.header.stamp.sec*10**9+message.header.stamp.nanosec
        delay = (self.node.get_clock().now().nanoseconds-stamp)/1e9
        if (not 0 <= delay <= .5 or not 4 <= len(payload) <= 8*1024*1024
                or not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9")):
            return
        with self.node._lock:
            if self.frame is None or stamp > self.frame[0]:
                self.frame = (stamp, time.monotonic()-delay, payload)

    def incident(self, message):
        from .evidence_clock import local_evidence_clock_id
        try:
            item = json.loads(message.data)
            if (item["clock_id"] != local_evidence_clock_id()
                    or item["event_type"] != "LITTERING"
                    or item["original_model_event"].get("state") != "CONFIRMED"
                    or item["model_sha256"] != "cddc20536cd76db52e340746a2c807e5f53cf1fe2a60edeace53973029256935"):
                return
            uuid.UUID(item["event_id"])
            received = float(item["camera_received_monotonic_s"])
            if not 0 <= time.monotonic()-received <= 5:
                return
            with self.node._lock:
                if not self.node._active_goal or item["event_id"] in self.seen:
                    return
                self.seen.add(item["event_id"])
                if len(self.seen) > 1024:
                    return  # Session restart, never evict IDs into replay eligibility.
                if self.event is None or received > self.event["input_received"]:
                    self.event = dict(item, type="LITTERING", input_received=received,
                                      mission_id=self.node._active_mission_id)
        except (ValueError, TypeError, KeyError, AttributeError):
            return

    def save(self, root, mission_id, event, frame, position, now):
        """Called on a worker, using immutable snapshots. No flight side effects."""
        source = Path(event["photo_path"])
        evidence = source.read_bytes()
        if hashlib.sha256(evidence).hexdigest() != event["photo_sha256"]:
            raise ValueError("incident_photo_hash_mismatch")
        folder = root / mission_id / event["event_id"]
        folder.mkdir(parents=True, exist_ok=False)
        for name, payload in (("detection.jpg", evidence), ("stopped.jpg", frame[2])):
            with (folder/name).open("xb") as stream:
                stream.write(payload)
        record = dict(event, mission_id=mission_id, stopped_position_ned=[float(value) for value in position],
                      stopped_frame_stamp_ns=frame[0], stopped_frame_received=frame[1],
                      saved_monotonic_s=now, stopped_sha256=hashlib.sha256(frame[2]).hexdigest())
        with (folder/"evidence.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, allow_nan=False, indent=2)
        return "saved"


def execute_scenario(node, goal, result, snapshot):
    from jolgwa_interfaces.msg import OffboardCommand, MissionStatus
    spec = validate_spec(json.loads(snapshot.plan_json))
    sequence = ScenarioSequence(spec, time.monotonic())
    inputs = node._scenario_inputs
    last_command = None
    photo_future = None
    last_feedback = -math.inf
    last_phase = None
    last_reset = -math.inf
    hold_target = None
    hold_phase = None
    while sequence.phase not in ("FAILED", "LAND"):
        now = time.monotonic()
        if node._execution_interrupted(goal):
            return node._finish_wait_failure(goal, result, "cancel", "scenario interrupted")
        with node._lock:
            state = node._vehicle_state
            error = node._active_low_speed_error_locked(now)
            manual = node._manual_active or node._resume_required
            evidence, event, frame = inputs.evidence, inputs.event, inputs.frame
            reset = getattr(node, '_scenario_yaw_reset_at', -math.inf)
        if reset > last_reset:
            if state is not None:
                sequence.invalidate_pre_reset_clear(reset, state.position_ned_m)
            last_reset = reset
        if error or manual or state is None or not state.armed or not state.offboard:
            return node._finish_wait_failure(goal, result, "failed", error or "scenario ownership lost")
        if state.terminal_in_progress or state.flight_epoch_retired:
            return node._finish_wait_failure(goal, result, "failed", state.last_error or "scenario terminal already started")
        if (state.last_error == 'altitude_recovery_wait'
                and state.active_execution_mission_id == goal.request.mission_id):
            # Controller owns the bounded stop. Original phase/total clocks keep running.
            time.sleep(.05)
            continue
        if event is not None and event["mission_id"] != goal.request.mission_id:
            event = None
        photo = None
        if sequence.phase == "PHOTO" and now-sequence.phase_started < spec["photo_timeout_s"]:
            if photo_future is None and frame is not None and frame[1] > sequence.phase_started and now-frame[1] <= .5:
                photo_future = inputs.pool.submit(inputs.save, node._scenario_photo_root,
                    goal.request.mission_id, sequence.event, frame, tuple(state.position_ned_m), now)
            if photo_future is not None and photo_future.done():
                try:
                    photo = photo_future.result()
                except Exception as exc:
                    node.get_logger().error("scenario photo: "+str(exc))
                    photo = "failed"
        phase, target = sequence.tick(tuple(state.position_ned_m), tuple(state.velocity_ned_m_s),
                                       evidence, now, event=event, photo=photo)
        if phase in ("FAILED", "LAND"):
            break
        command = OffboardCommand.COMMAND_HOLD if sequence.depth_waiting or (phase == "CLIMB" and sequence.climb_clear_hold) or phase in ("BRAKE", "WAIT_EVENT", "EVENT_HOLD", "PHOTO") else OffboardCommand.COMMAND_GOTO
        # Repeated noisy CLEAR/lost-depth transitions in the same phase must
        # not chase measured position with a new HOLD identity every tick.
        if command == OffboardCommand.COMMAND_HOLD:
            if hold_phase != phase or last_command is None or last_command[1] != command:
                hold_target = tuple(target)
                hold_phase = phase
            target = hold_target
        else:
            hold_phase = None
        identity = (phase, command, tuple(target))
        if identity != last_command:
            node._publish_command(goal.request.mission_id, command, target, snapshot.preview_heading_rad)
            last_command = identity
        if phase != last_phase or now-last_feedback >= .5:
            detail = "SCENARIO_V7_FRONT_DETECT "+phase
            node._publish_status(goal.request.mission_id, goal.request.proposal_id,
                                 MissionStatus.PHASE_FORWARD_TEST, detail)
            node._publish_feedback(goal, MissionStatus.PHASE_FORWARD_TEST, 0, 1, detail)
            last_feedback = now
            last_phase = phase
        time.sleep(.05)
    if sequence.phase == "FAILED":
        return node._finish_wait_failure(goal, result, "failed", "SCENARIO_V7_FRONT_DETECT "+sequence.reason)
    node._publish_command(goal.request.mission_id, OffboardCommand.COMMAND_LAND,
                          sequence.target, snapshot.preview_heading_rad)
    outcome = node._complete_low_speed_terminal(goal, command_already_sent=True)
    if outcome != "ACCEPTED":
        return node._finish_wait_failure(goal, result, "failed", "SCENARIO_V7_FRONT_DETECT landing: "+outcome)
    goal.succeed()
    result.success = True
    result.final_phase = MissionStatus.PHASE_COMPLETED
    result.message = "SCENARIO_V7_FRONT_DETECT obstacle passed; littering photos saved; final 1m and disarm confirmed"
    node._publish_status(goal.request.mission_id, goal.request.proposal_id, result.final_phase, result.message)
    return result

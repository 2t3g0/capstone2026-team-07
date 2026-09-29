from . import scenario_altitude_recovery as altitude_recovery
"""Single-owner Pixhawk USB MAVLink bridge for the existing route controller.

The bridge uses private ``/jolgwa/mavlink/fmu/*`` topics.  It never publishes
the public DDS ``/fmu/*`` namespace and it does not broaden the observe-only or
small-obstacle-demo transports.
"""
from collections import deque
import json
import copy
from dataclasses import replace
import math
from pathlib import Path
import time
import uuid
from .async_journal import AsyncJournal

import rclpy
from jolgwa_interfaces.msg import (
    AltitudeReferenceState, FlightEnvelope, FlightOutputState,
    FlightControlContract, FlightCommandRequest, FlightCommandAck,
    LowSpeedSafetyState, MissionApproval, OffboardCommand, VehicleGeoState,
)
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String
from px4_msgs.msg import (
    OffboardControlMode,
    HomePosition,
    TrajectorySetpoint,
    VehicleAttitude,
    VehicleAngularVelocity,
    VehicleCommand,
    VehicleCommandAck,
    VehicleLandDetected,
    VehicleLocalPosition,
    VehicleStatus,
)

from jolgwa_uav.mavlink_route_fc_link import (
    SETPOINT_LEASE_NS,
    MavlinkRouteTransport,
    RouteFcState,
    decode_px4_parameter_value,
    make_route_setpoint,
    normalize_parameter_id,
    validate_route_command_ack,
)
from jolgwa_uav.observer_fc_telemetry import TelemetryState, TOPIC_PREFIX
from jolgwa_uav.observer_fc_timesync import ObserveTimesync

from .px4_offboard_controller import px4_qos
from .flight_contract import (command_matches_intent, same_value, PROOF_LEASE_NS,
    BATTERY_TERMINAL_ONLY, BATTERY_TERMINAL_UNAVAILABLE)
from .route_frame import set_px4_home_geodetic
from .scenario_runtime import init_contract, spec_for, navigation_error
from .scenario_contract import ceiling as scenario_ceiling
from . import scenario_battery_home as battery_home
from .low_speed import (
    ALTITUDE_REFERENCE_MAX_ERROR_M,
    FLIGHT_OUTPUT_PAIR_SKEW_NS,
    LOW_SPEED_1M_V1,
    LOW_SPEED_2M_V1,
    PX4_COM_OBL_RC_ACT_LAND,
    PX4_LAND_SPEED_MAX_M_S,
    AltitudeReferenceSynchronizer,
    boot_time_distance_ms,
    envelope_allows,
    flight_output_pair_skew_valid,
    altitude_reference_sample_error,
    geo_is_fresh,
    low_speed_envelope_profiles_match,
    low_speed_pose_coherence_reason,
    px4_land_speed_is_valid,
    px4_offboard_loss_action_is_land,
    setpoint_contract_matches,
)
from .topic_names import (
    ALTITUDE_REFERENCE_STATE, FLIGHT_ENVELOPE, FLIGHT_OUTPUT_STATE,
    LOW_SPEED_SAFETY_STATE, MISSION_APPROVAL, OFFBOARD_COMMAND, VEHICLE_GEO_STATE,
    FLIGHT_CONTROL_CONTRACT, FLIGHT_COMMAND_REQUEST, FLIGHT_COMMAND_ACK,
)


BRIDGE_PREFIX = "/jolgwa/mavlink/fmu"
BRIDGE_OFFBOARD_CONTROL_MODE = BRIDGE_PREFIX + "/in/offboard_control_mode"
BRIDGE_TRAJECTORY_SETPOINT = BRIDGE_PREFIX + "/in/trajectory_setpoint"
BRIDGE_VEHICLE_COMMAND = BRIDGE_PREFIX + "/in/vehicle_command"
BRIDGE_VEHICLE_STATUS = BRIDGE_PREFIX + "/out/vehicle_status_v1"
BRIDGE_LOCAL_POSITION = BRIDGE_PREFIX + "/out/vehicle_local_position_v1"
BRIDGE_HOME_POSITION = BRIDGE_PREFIX + "/out/home_position_v1"
BRIDGE_LAND_DETECTED = BRIDGE_PREFIX + "/out/vehicle_land_detected"
BRIDGE_COMMAND_ACK = BRIDGE_PREFIX + "/out/vehicle_command_ack"

COMMAND_LEASE_NS = 250_000_000
PAIR_SKEW_NS = FLIGHT_OUTPUT_PAIR_SKEW_NS
SOURCE_PAIR_SPAN_NS = 40_000_000  # One 50 ms controller publication, with margin.


def _set_if_present(message, name, value):
    if hasattr(message, name):
        setattr(message, name, value)


class _OutputPairBuffer:
    """Match independently delivered DDS messages from one controller tick.

    Arrival order is not a publication identity.  The source timestamps and
    exact setpoint values must agree before a cached message can authorize a
    physical write.  Bounded history lets a late member complete its own pair
    without pairing it with a newer tick's envelope.
    """

    def __init__(self):
        self.modes = deque(maxlen=8)
        self.setpoints = deque(maxlen=8)
        self.envelopes = deque(maxlen=8)

    def clear(self):
        self.modes.clear()
        self.setpoints.clear()
        self.envelopes.clear()

    def add_mode(self, message, received_ns):
        mode = {name: bool(getattr(message, name, False)) for name in (
            "position", "velocity", "acceleration", "attitude", "body_rate",
            "thrust_and_torque", "direct_actuator")}
        self.modes.append((received_ns, int(message.timestamp) * 1000, mode))
        return mode

    def add_setpoint(self, message, received_ns):
        setpoint = {
            "position": tuple(float(value) for value in message.position),
            "velocity": tuple(float(value) for value in message.velocity),
            "yaw": float(message.yaw),
            "yawspeed": float(message.yawspeed),
        }
        self.setpoints.append((received_ns, int(message.timestamp) * 1000, setpoint))
        return setpoint

    def add_envelope(self, message, received_ns):
        stamp = message.stamp
        source_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        self.envelopes.append((received_ns, source_ns, message))

    @staticmethod
    def _fresh(received_ns, now_ns):
        return 0 <= now_ns - received_ns < SETPOINT_LEASE_NS

    def select(self, now_ns, contract, *, minimum_source_ns=0):
        """Return the matching mode, setpoint, envelope and receipt times."""
        if contract is None:
            return None
        command = contract.command
        for setpoint_received, setpoint_source, setpoint in sorted(
                self.setpoints, key=lambda item: item[1], reverse=True):
            if (not self._fresh(setpoint_received, now_ns)
                    or setpoint_source <= 0
                    or setpoint_source < minimum_source_ns):
                continue
            for envelope_received, envelope_source, envelope in reversed(self.envelopes):
                if (not self._fresh(envelope_received, now_ns)
                        or envelope_source <= 0
                        or envelope.mission_id != command.mission_id
                        or int(envelope.sequence) != int(command.sequence)
                        or envelope.output_epoch != contract.output_epoch
                        or int(envelope.output_sequence) != int(contract.output_sequence)
                        or not flight_output_pair_skew_valid(
                            envelope_received, setpoint_received)
                        or not 0 <= setpoint_source - envelope_source <= SOURCE_PAIR_SPAN_NS):
                    continue
                for mode_received, mode_source, mode in reversed(self.modes):
                    if (not self._fresh(mode_received, now_ns)
                            or not flight_output_pair_skew_valid(
                                mode_received, setpoint_received)
                            or not envelope_source <= mode_source <= setpoint_source):
                        continue
                    matches, _ = setpoint_contract_matches(
                        setpoint_kind=envelope.setpoint_kind,
                        mode=mode,
                        position=setpoint["position"],
                        velocity=setpoint["velocity"],
                        yaw=setpoint["yaw"],
                        expected_position=envelope.expected_position_ned_m,
                        expected_velocity=envelope.expected_velocity_ned_m_s,
                        expected_yaw=envelope.expected_yaw_rad,
                    )
                    if matches:
                        return (mode, setpoint, envelope,
                                setpoint_received, envelope_received,
                                setpoint_source)
        return None


class MavlinkUsbBridgeNode(Node):
    """Own one USB descriptor and translate a closed ROS/PX4 message subset."""

    def __init__(self):
        super().__init__("mavlink_usb_bridge")
        init_contract(self)
        readonly = ParameterDescriptor(read_only=True)
        self.declare_parameter("mavlink_device", "/dev/jolgwa-pixhawk6c", readonly)
        self.declare_parameter("enable_px4_commands", False, readonly)
        self.declare_parameter("enable_mavlink_commands", False, readonly)
        self.declare_parameter("allow_real_hardware", False, readonly)
        self.declare_parameter("simulation_only", True, readonly)
        self.declare_parameter("target_system", 1, readonly)
        self.declare_parameter("target_component", 1, readonly)
        self.declare_parameter("source_system", 245, readonly)
        self.declare_parameter("source_component", 191, readonly)
        self.declare_parameter("journal_directory", "outputs/mavlink_route_fc", readonly)
        self.declare_parameter("home_local_residual_limit_m", 5.0, readonly)

        identity = tuple(int(self.get_parameter(name).value) for name in (
            "target_system", "target_component", "source_system", "source_component"
        ))
        if identity != (1, 1, 245, 191):
            raise ValueError("fixed FC 1/1 and Jetson 245/191 identity required")
        self._device = str(self.get_parameter("mavlink_device").value)
        self._transport_kwargs = {
            "enable_px4_commands": bool(self.get_parameter("enable_px4_commands").value),
            "enable_mavlink_commands": bool(self.get_parameter("enable_mavlink_commands").value),
            "allow_real_hardware": bool(self.get_parameter("allow_real_hardware").value),
            "simulation_only": bool(self.get_parameter("simulation_only").value),
        }
        self._flight_output_requested = all((
            self._transport_kwargs["enable_px4_commands"],
            self._transport_kwargs["enable_mavlink_commands"],
            self._transport_kwargs["allow_real_hardware"],
            not self._transport_kwargs["simulation_only"],
        ))

        directory = Path(str(self.get_parameter("journal_directory").value))
        directory.mkdir(parents=True, exist_ok=True)
        journal_path = directory / (
            time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            + "_" + uuid.uuid4().hex + ".jsonl"
        )
        self._journal = AsyncJournal.open(journal_path)
        self._journal.write(json.dumps({
            "kind": "mavlink_route_bridge_start",
            "flight_output_enabled": self._flight_output_requested,
            "device": self._device,
        }) + "\n")

        self._altitude_reference_sync = AltitudeReferenceSynchronizer()
        self._route_state = RouteFcState(
            transport_epoch=self._altitude_reference_sync.transport_epoch)
        self._observer_state = TelemetryState(timing=ObserveTimesync(enabled=False))
        self._transport = None
        self._next_open_at = 0.0
        self._connection_error = "waiting_for_usb"
        self._streams_requested = False
        self._mode = None
        self._mode_received_ns = 0
        self._setpoint = None
        self._setpoint_received_ns = 0
        self._output_pairs = _OutputPairBuffer()
        self._last_pair_source_ns = 0
        self._commands = deque(maxlen=16)
        self._intent = None
        self._intent_history = {}
        self._contract = None
        self._approved_missions = set()
        self._seen_request_ids = deque(maxlen=256)
        self._tx_run_id = 0
        self._tx_run_started_ns = 0
        self._tx_sequence = 0
        self._arm_transmitted = False
        self._last_tx_terminal_brake = False
        self._last_tx_kind = 0
        self._pending_acks = {}
        self._last_correction_audit = None
        self._last_conversion_warning_ns = 0
        self._last_graph_warning_ns = 0
        self._envelope = None
        self._envelope_received_ns = 0
        self._command_profile = OffboardCommand.FLIGHT_PROFILE_NORMAL
        self._command_mission_id = ""
        self._command_sequence = 0
        self._command_received_ns = 0
        # OffboardCommand is an event, not a periodically renewed mission lease.
        # Only explicit approval retirement ends ownership of the frozen Home.
        self._ended_mission_ids = deque(maxlen=64)
        self._command_code = OffboardCommand.COMMAND_HOLD
        self._command_position_ned_m = (math.nan, math.nan, math.nan)
        self._command_home_z_ned_m = 0.0
        self._command_altitude_reference = OffboardCommand.ALTITUDE_REFERENCE_NONE
        self._command_altitude_reference_max_error_m = 0.0
        self._command_altitude_reference_epoch = 0
        self._command_altitude_reference_sequence = 0
        self._command_altitude_reference_local_time_boot_ms = 0
        self._command_altitude_reference_global_time_boot_ms = 0
        self._px4_parameters = {}
        self._restart_required = False
        self._last_setpoint_tx_ns = 0
        self._setpoint_tx_mission_id = ""
        self._setpoint_tx_sequence = 0
        self._consecutive_setpoint_tx = 0
        self._last_output_detail = "waiting_for_setpoint"
        self._battery_landings = {}
        self._last_health_diagnostic = None
        self._last_output_warning_detail = ""
        self._last_output_warning_ns = 0
        self._last_envelope_valid = False
        self._home_residual_limit_m = float(
            self.get_parameter("home_local_residual_limit_m").value)

        qos = px4_qos()
        self._vehicle_status_publisher = self.create_publisher(
            VehicleStatus, BRIDGE_VEHICLE_STATUS, qos
        )
        self._local_position_publisher = self.create_publisher(
            VehicleLocalPosition, BRIDGE_LOCAL_POSITION, qos
        )
        home_qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._home_position_publisher = self.create_publisher(
            HomePosition, BRIDGE_HOME_POSITION, home_qos)
        self._land_publisher = self.create_publisher(
            VehicleLandDetected, BRIDGE_LAND_DETECTED, qos
        )
        self._context_ack_publisher = self.create_publisher(FlightCommandAck, FLIGHT_COMMAND_ACK, 10)
        self._ack_publisher = self.create_publisher(
            VehicleCommandAck, BRIDGE_COMMAND_ACK, qos
        )
        self._geo_publisher = self.create_publisher(
            VehicleGeoState, VEHICLE_GEO_STATE, qos_profile_sensor_data)
        self._altitude_reference_publisher = self.create_publisher(
            AltitudeReferenceState, ALTITUDE_REFERENCE_STATE,
            qos_profile_sensor_data)
        self._low_speed_safety_publisher = self.create_publisher(
            LowSpeedSafetyState, LOW_SPEED_SAFETY_STATE, qos_profile_sensor_data)
        self._flight_output_state_publisher = self.create_publisher(
            FlightOutputState, FLIGHT_OUTPUT_STATE, qos_profile_sensor_data)
        self._observer_positions = self.create_publisher(
            VehicleLocalPosition, TOPIC_PREFIX + "/vehicle_local_position", qos_profile_sensor_data
        )
        self._observer_attitudes = self.create_publisher(
            VehicleAttitude, TOPIC_PREFIX + "/vehicle_attitude", qos_profile_sensor_data
        )
        self._observer_rates = self.create_publisher(
            VehicleAngularVelocity, TOPIC_PREFIX + "/vehicle_angular_velocity", qos_profile_sensor_data
        )
        self._observer_status = self.create_publisher(String, TOPIC_PREFIX + "/status", 1)

        self.create_subscription(
            OffboardControlMode,
            BRIDGE_OFFBOARD_CONTROL_MODE,
            self._on_offboard_control_mode,
            qos,
        )
        self.create_subscription(
            TrajectorySetpoint,
            BRIDGE_TRAJECTORY_SETPOINT,
            self._on_trajectory_setpoint,
            qos,
        )
        self.create_subscription(
            FlightCommandRequest,
            FLIGHT_COMMAND_REQUEST,
            self._on_command_request,
            10,
        )
        self.create_subscription(FlightControlContract, FLIGHT_CONTROL_CONTRACT,
                                 self._on_output_contract, 10)
        self.create_subscription(
            FlightEnvelope, FLIGHT_ENVELOPE, self._on_flight_envelope,
            qos_profile_sensor_data)
        self.create_subscription(
            OffboardCommand, OFFBOARD_COMMAND, self._on_offboard_command, 10)
        self.create_subscription(
            MissionApproval, MISSION_APPROVAL, self._on_mission_approval, 10)
        self.create_timer(0.01, self._poll)
        self.create_timer(0.05, self._publish_state)

        if self._flight_output_requested:
            self.get_logger().warning(
                "PHYSICAL MAVLink flight output enabled; USB owner and PX4 failsafe checks are mandatory"
            )
        else:
            self.get_logger().warning(
                "MAVLink bridge is telemetry-only; all three command gates and simulation_only=false are required"
            )

    @staticmethod
    def _timestamp_us():
        return time.monotonic_ns() // 1000

    def _on_offboard_control_mode(self, message):
        self._mode_received_ns = time.monotonic_ns()
        self._mode = self._output_pairs.add_mode(message, self._mode_received_ns)

    def _on_trajectory_setpoint(self, message):
        self._setpoint_received_ns = time.monotonic_ns()
        self._setpoint = self._output_pairs.add_setpoint(
            message, self._setpoint_received_ns)

    def _on_flight_envelope(self, message):
        self._envelope = message
        self._envelope_received_ns = time.monotonic_ns()
        self._output_pairs.add_envelope(message, self._envelope_received_ns)

    def _on_offboard_command(self, message):
        key = (message.mission_id, int(message.sequence))
        previous = self._intent_history.get(key)
        if previous is not None and any(not same_value(getattr(previous, field), getattr(message, field))
                for field in message.get_fields_and_field_types() if field != "stamp"):
            self._reset_tx_run()
            self._last_output_detail = "source_command_identity_conflict"
            return
        self._intent_history[key] = copy.deepcopy(message)
        if previous is None and spec_for(self, message):
            self._journal.write(json.dumps(dict(event='source_intent_received',
                monotonic_ns=time.monotonic_ns(), mission_id=message.mission_id,
                source_sequence=message.sequence, command=message.command))+'\n')
        protected = ((self._contract.command.mission_id, self._contract.command.sequence)
                     if self._contract is not None else None)
        for old in list(self._intent_history):
            if len(self._intent_history) <= 64:
                break
            if old != protected and old != key:
                self._intent_history.pop(old, None)
        if (self._contract is not None and self._contract.command.command in (3, 4, 5)
                and self._contract.command.mission_id == message.mission_id):
            return  # A repeated terminal request cannot replace its proof identity.
        if battery_home.join_intent(self, message, time.monotonic_ns()):
            return
        self._intent = message
        if self._contract is not None:
            self._reset_tx_run()

    def _load_output_command(self, message):
        if (str(message.mission_id) in self._ended_mission_ids
                and not (self._route_state.home_locked
                         and self._route_state.execution_home_mission_id == str(message.mission_id)
                         and int(message.command) in (
                             OffboardCommand.COMMAND_LAND,
                             OffboardCommand.COMMAND_ABORT))):
            return  # A delayed command must not resurrect a completed mission.
        self._command_profile = int(getattr(
            message, "flight_profile", OffboardCommand.FLIGHT_PROFILE_NORMAL))
        self._command_mission_id = str(message.mission_id)
        self._command_sequence = int(message.sequence)
        self._command_received_ns = time.monotonic_ns()
        self._command_code = int(message.command)
        self._command_position_ned_m = tuple(
            float(value) for value in message.position_ned_m)
        self._command_home_z_ned_m = float(message.home_z_ned_m)
        self._command_altitude_reference = int(getattr(
            message, "altitude_reference",
            OffboardCommand.ALTITUDE_REFERENCE_NONE))
        self._command_altitude_reference_max_error_m = float(getattr(
            message, "altitude_reference_max_error_m", 0.0))
        self._command_altitude_reference_epoch = int(getattr(
            message, "altitude_reference_epoch", 0))
        self._command_altitude_reference_sequence = int(getattr(
            message, "altitude_reference_sequence", 0))
        self._command_altitude_reference_local_time_boot_ms = int(getattr(
            message, "altitude_reference_local_time_boot_ms", 0))
        self._command_altitude_reference_global_time_boot_ms = int(getattr(
            message, "altitude_reference_global_time_boot_ms", 0))

    def _on_mission_approval(self, message):
        # Mission Manager retains approval through active cancellation and
        # terminal handling, then publishes approved=False during cleanup.
        mission_id = str(message.mission_id)
        if message.approved and mission_id not in self._ended_mission_ids:
            self._approved_missions.add(mission_id)
        elif not message.approved:
            self._approved_missions.discard(mission_id)
            if self._contract is not None and self._contract.command.mission_id == mission_id:
                battery_home.clear_join(self)
        if not message.approved and mission_id and mission_id not in self._ended_mission_ids:
            self._ended_mission_ids.append(mission_id)

    def _reset_tx_run(self):
        if self._tx_run_started_ns:
            self._tx_run_id += 1
        self._tx_run_started_ns = 0
        self._consecutive_setpoint_tx = 0
        self._last_envelope_valid = False

    def _on_output_contract(self, contract):
        prior = self._contract
        if (not contract.output_epoch or contract.output_sequence == 0
                or contract.handshake_version != 8):
            return
        if spec_for(self, contract.command):
            self._journal.write(json.dumps(dict(event='output_contract_received',
                monotonic_ns=time.monotonic_ns(), mission_id=contract.command.mission_id,
                source_sequence=contract.command.sequence, output_sequence=contract.output_sequence,
                output_epoch=contract.output_epoch))+'\n')
        if prior is not None:
            if (contract.output_epoch == prior.output_epoch
                    and contract.output_sequence <= prior.output_sequence):
                return
            if (prior.command.command in (3, 4, 5)
                    and prior.command.mission_id == contract.command.mission_id):
                return
        if battery_home.join_contract(self, contract, time.monotonic_ns()):
            return
        self._commit_output_contract(contract)

    def _commit_output_contract(self, contract):
        if contract.command.command in (3, 4, 5):
            battery_home.clear_join(self)
        # Store even when the reliable intent callback has not arrived yet;
        # the final write validator must still observe both before output.
        self._contract = contract
        self._output_pairs.clear()
        self._last_pair_source_ns = 0
        self._terminal_contract_started_ns = time.monotonic_ns() if contract.command.command in (3, 4, 5) else 0
        self._load_output_command(contract.command)
        self._commands.clear()
        self._pending_acks.clear()
        self._progress_ack_requests = set()
        self._reset_tx_run()
        self._journal.write(json.dumps({"event": "flight_contract_transition",
            "mission_id": contract.command.mission_id, "source_sequence": contract.command.sequence,
            "output_epoch": contract.output_epoch, "output_sequence": contract.output_sequence,
            "command": contract.command.command}) + "\n")

    def _contract_valid(self):
        c = self._contract
        source = (self._intent_history.get((c.command.mission_id, c.command.sequence))
                  if c is not None and c.command.command in (3, 5) else self._intent)
        return bool(c is not None
                    and c.handshake_version == 8
                    and c.command.flight_output_handshake_version == 8
                    and c.command.mission_id in self._approved_missions
                    and c.command.mission_id not in self._ended_mission_ids
                    and command_matches_intent(c.command, source, allow_recovery_hold=bool(spec_for(self, c.command))))

    def _on_command_request(self, request):
        if not request.request_id or request.request_id in self._seen_request_ids:
            return
        self._seen_request_ids.append(request.request_id)
        if len(self._commands) < self._commands.maxlen:
            self._commands.append(request)

    def _observe_battery_health(self, snapshot, now_ns):
        """Latch a landing-only exception for an already owned low-speed flight.

        The aggregate transport gate remains fail-closed. Only the ROS status
        used by this owned terminal handoff excludes the isolated battery bit.
        Recovery cannot make navigation or Mode/ARM writable again.
        """
        policy_state = battery_home.observe(self, snapshot, now_ns)
        c = self._contract
        mission = c.command.mission_id if c is not None else ""
        health_key = (getattr(snapshot, "sensor_health_reason", ""),
            getattr(snapshot, "sensors_present", 0), getattr(snapshot, "sensors_enabled", 0),
            getattr(snapshot, "sensors_health", 0), mission)
        if health_key != self._last_health_diagnostic:
            self._last_health_diagnostic = health_key
            self._journal.write(json.dumps({"event": "sensor_health_transition",
                "monotonic_ns": now_ns, "mission_id": mission,
                "output_epoch": c.output_epoch if c else "",
                "transport_epoch": getattr(snapshot, "transport_epoch", 0),
                "age_ns": getattr(snapshot, "sensor_health_age_ns", -1),
                "reason": health_key[0], "present": health_key[1],
                "enabled": health_key[2], "health": health_key[3],
                "failed": getattr(snapshot, "sensors_failed", 0)}) + "\n")
        low = c is not None and c.command.flight_profile in (
            OffboardCommand.FLIGHT_PROFILE_LOW_SPEED_1M_V1,
            OffboardCommand.FLIGHT_PROFILE_LOW_SPEED_2M_V1)
        valid = bool(low and self._contract_valid() and not self._restart_required)
        record = self._battery_landings.get(mission)
        if getattr(self, "_battery_navigation_allowed", False):
            return False
        if (record is None and valid and snapshot.armed and snapshot.offboard
                and (getattr(snapshot, "battery_health_terminal_only", False) or policy_state == "terminal")):
            record = dict(epoch=snapshot.transport_epoch, output_epoch=c.output_epoch,
                          first_ns=now_ns, failed=False)
            self._battery_landings[mission] = record
            self._journal.write(json.dumps({"event": "battery_terminal_latched",
                "mission_id": mission, "output_epoch": c.output_epoch,
                "source_sequence": c.command.sequence, "monotonic_ns": now_ns}) + "\n")
        if record is None:
            return False
        available = bool(valid and snapshot.connected and snapshot.armed and snapshot.offboard
            and snapshot.position_valid and snapshot.transport_epoch == record['epoch']
            and c.output_epoch == record['output_epoch']
            and ((policy_state == "terminal" and snapshot.battery_navigation_healthy)
                 or getattr(snapshot, "battery_health_terminal_only", False)
                 or (getattr(snapshot, "preflight_checks_pass", False)
                     and not getattr(snapshot, "failsafe", True)
                     and getattr(snapshot, "sensors_present", 0)
                         & getattr(snapshot, "sensors_enabled", 0) & (1 << 25))))
        if not available:
            if not record['failed']:
                self._journal.write(json.dumps(dict(event='battery_handoff_first_failure',
                    monotonic_ns=now_ns, mission_id=mission, contract_valid=valid,
                    connected=snapshot.connected, armed=snapshot.armed, offboard=snapshot.offboard,
                    position_valid=snapshot.position_valid, transport_epoch=snapshot.transport_epoch,
                    expected_epoch=record['epoch'], sensor_health=snapshot.sensor_health_reason))+'\n')
            record['failed'] = True
        return bool(available and not record['failed'])

    def _on_vehicle_command(self, message):
        command = int(message.command)
        if (
            int(message.target_system) != 1
            or int(message.target_component) != 1
            or not bool(message.from_external)
        ):
            self.get_logger().error("Rejected VehicleCommand with unexpected target or origin")
            self._publish_local_ack(command, "VEHICLE_CMD_RESULT_DENIED")
            return
        if len(self._commands) == self._commands.maxlen:
            self.get_logger().error("VehicleCommand queue full; rejecting newest command")
            self._publish_local_ack(command, "VEHICLE_CMD_RESULT_TEMPORARILY_REJECTED")
            return
        params = tuple(float(getattr(message, "param" + str(index))) for index in range(1, 8))
        self._commands.append((command, params, time.monotonic_ns()))

    def _open(self):
        # Fail before importing the dedicated serial/MAVLink dependencies when
        # the configured target is not even an ACM device.  Full VID/PID and
        # exclusive-open validation remains in UsbTelemetryTransport.
        path = Path(self._device).resolve(strict=True)
        if not path.name.startswith("ttyACM"):
            raise ValueError("not a USB ACM device")
        self._transport = MavlinkRouteTransport(
            self._journal,
            self._device,
            timing=self._observer_state.timing,
            **self._transport_kwargs,
        )
        self._connection_error = ""
        self._streams_requested = False
        # A reopened USB descriptor starts a new frame-validation epoch.
        self._altitude_reference_sync.begin_epoch("transport_opened")
        self._route_state = RouteFcState(
            transport_epoch=self._altitude_reference_sync.transport_epoch)

    def _poll(self):
        active = self._contract.command if self._contract else self._intent
        self._altitude_reference_sync.confirmation_ms = battery_home.home_confirmation_ms(self, active)
        if self._restart_required:
            return
        try:
            self._poll_once()
        except (OSError, RuntimeError) as exc:
            # An initial exclusive-open failure is a deployment error and must
            # still terminate the process.  Runtime loss after ownership was
            # established is latched so stale commands can never revive.
            if self._transport is None:
                raise
            self._latch_transport_fault(str(exc))

    def _poll_once(self):
        now = time.monotonic()
        if self._transport is None:
            if now < self._next_open_at:
                return
            self._next_open_at = now + 2.0
            try:
                self._open()
            except (OSError, ValueError) as exc:
                self._connection_error = str(exc)
                if self._flight_output_requested:
                    raise RuntimeError(
                        "MAVLink route mode requires exclusive USB ownership; refusing to wait or steal it: "
                        + str(exc)
                    ) from exc
                return

        for message in self._transport.read():
            kind = message.get_type()
            if kind == "BAD_DATA":
                continue
            data = message.to_dict()
            source_system = message.get_srcSystem()
            source_component = message.get_srcComponent()
            received_ns = self._transport.last_read_monotonic_ns
            now_s = received_ns / 1e9
            observer_accepted = self._observer_state.accept(
                kind,
                data,
                source_system,
                source_component,
                now_s,
                received_ns=received_ns,
            )
            self._refresh_yaw_reset_context(received_ns)
            route_accepted = self._route_state.accept(
                kind, data, source_system, source_component, received_ns
            )
            self._journal_yaw_resets()
            if route_accepted and kind == 'BATTERY_STATUS':
                self._journal.write(json.dumps(dict(event='battery_status_sample',
                    monotonic_ns=received_ns, transport_epoch=self._route_state.transport_epoch,
                    mission_id=self._command_mission_id,
                    sample=self._route_state.battery_samples.get(data.get('id'))))+'\n')
            if route_accepted and kind == 'HOME_POSITION':
                self._journal.write(json.dumps(dict(event='home_position_received',
                    monotonic_ns=received_ns, transport_epoch=self._route_state.transport_epoch,
                    mission_id=self._command_mission_id, metadata=data))+'\n')
            if observer_accepted:
                self._publish_observer_sample(kind, now_s)
            if route_accepted and kind == "LOCAL_POSITION_NED":
                local = self._route_state.local_position
                if local is not None:
                    self._altitude_reference_sync.add_local(
                        time_boot_ms=local["time_boot_ms"],
                        z_ned_m=local["position"][2],
                        received_ns=received_ns,
                    )
            if route_accepted and kind == "GLOBAL_POSITION_INT":
                global_position = self._route_state.global_position
                if global_position is not None:
                    self._altitude_reference_sync.add_global(
                        time_boot_ms=global_position["time_boot_ms"],
                        fc_altitude_home_relative_m=(
                            global_position["relative_altitude_m"]),
                        global_altitude_amsl_m=(
                            global_position["altitude_m"]),
                        received_ns=received_ns,
                        now_ns=received_ns,
                    )
                self._publish_geo(received_ns)
            if route_accepted and kind == "HEARTBEAT" and not self._streams_requested:
                for message_id in sorted(self._transport_stream_ids()):
                    self._transport.request_route_stream(message_id)
                self._transport.request_autopilot_version()
                self._streams_requested = True
            if kind == "COMMAND_ACK":
                self._publish_mavlink_ack(data, source_system, source_component)
            elif kind == "PARAM_VALUE":
                parameter_id = normalize_parameter_id(data.get("param_id", ""))
                if (source_system, source_component) == (1, 1) and parameter_id in (
                        "COM_OBL_RC_ACT", "MPC_LAND_SPEED"):
                    value = data.get("param_value")
                    param_type = data.get("param_type")
                    try:
                        decoded = decode_px4_parameter_value(value, param_type)
                    except ValueError as exc:
                        self.get_logger().warning(
                            f"Ignoring invalid PX4 parameter {parameter_id}: {exc}")
                    else:
                        self._px4_parameters[parameter_id] = (
                            decoded, received_ns, int(param_type))

        # Route pose metadata follows the altitude-reference epoch.  A
        # vertical-frame discontinuity can advance it without reopening USB.
        self._route_state.transport_epoch = (
            self._altitude_reference_sync.transport_epoch)
        now_ns = time.monotonic_ns()
        snapshot = self._route_state.snapshot(now_ns)
        if self._streams_requested and snapshot.connected:
            self._transport.request_heartbeat(now_ns)
            pending = bool(self._altitude_reference_sync._home_correction_pending_since_ns)
            fast = pending and self._altitude_reference_sync.confirmation_ms == 2000
            event = self._altitude_reference_sync._correction_event_id if fast else None
            self._transport.request_home_position(now_ns,
                period_ns=200_000_000 if fast else 1_000_000_000,
                force=fast and event != getattr(self, '_fast_home_event', None))
            self._fast_home_event = event
            for parameter_id in ("COM_OBL_RC_ACT", "MPC_LAND_SPEED"):
                self._transport.request_parameter(parameter_id, now_ns)
        self._stream_setpoint(snapshot, now_ns)
        self._drain_command_queue(snapshot, time.monotonic_ns())

    def _latch_transport_fault(self, detail):
        try:
            self._transport.close()
        except Exception:
            pass
        self._transport = None
        self._connection_error = "restart_required: " + str(detail)
        self._streams_requested = False
        self._mode = None
        self._mode_received_ns = 0
        self._setpoint = None
        self._setpoint_received_ns = 0
        self._envelope = None
        self._envelope_received_ns = 0
        self._output_pairs.clear()
        self._last_pair_source_ns = 0
        self._commands.clear()
        self._px4_parameters.clear()
        self._pending_acks.clear()
        self._progress_ack_requests = set()
        self._reset_tx_run()
        self._last_setpoint_tx_ns = 0
        self._consecutive_setpoint_tx = 0
        self._last_envelope_valid = False
        self._last_output_detail = self._connection_error
        self._altitude_reference_sync.begin_epoch(
            "transport_fault_restart_required")
        self._route_state = RouteFcState(
            transport_epoch=self._altitude_reference_sync.transport_epoch)
        self._altitude_reference_sync.mark_stale(
            "transport_fault_restart_required")
        self._restart_required = bool(self._flight_output_requested)
        self.get_logger().error("MAVLink transport fenced; process restart required: " + str(detail))

    @staticmethod
    def _transport_stream_ids():
        from jolgwa_uav.mavlink_route_fc_link import ROUTE_STREAM_INTERVALS_US
        return ROUTE_STREAM_INTERVALS_US

    def _drain_command_queue(self, snapshot, now_ns):
        battery_available = self._observe_battery_health(snapshot, now_ns)
        while self._commands and not 0 <= now_ns-self._commands[0].issued_monotonic_ns < COMMAND_LEASE_NS:
            expired = self._commands.popleft()
            self._publish_local_ack(expired.command, "VEHICLE_CMD_RESULT_TEMPORARILY_REJECTED")
        if (
            not self._commands
            or self._transport is None
            or not self._private_command_graph_ready(now_ns)
        ):
            return
        request = self._commands.popleft()
        # ROS float64[7] deserializes to a NumPy array; tuple(array) retains
        # numpy.float64 elements, which the strict transport validator rejects.
        # Normalize at the ROS boundary; keep finite/value checks in transport.
        command = int(request.command)
        params = tuple(float(value) for value in request.params)
        pending = self._pending_acks.get(command)
        if (command in (20, 21) and pending is not None
                and pending[0].request_id in getattr(self, "_progress_ack_requests", set())):
            return  # A duplicate request must not restart an in-progress LAND.
        c = self._contract
        exact = bool(self._contract_valid() and c is not None
            and request.mission_id == c.command.mission_id
            and request.source_sequence == c.command.sequence
            and request.output_epoch == c.output_epoch
            and request.output_sequence == c.output_sequence)
        tx_ready = bool(self._last_envelope_valid and self._tx_run_started_ns
            and 0 <= now_ns-self._last_setpoint_tx_ns <= PROOF_LEASE_NS)
        mode_arm = command == 176 or (command == 400 and params[0] == 1.0)
        terminal = command in (21, 20)
        allowed = exact and snapshot.connected
        battery_fault = bool(getattr(snapshot, "sensors_failed", 0) & (1 << 25)
                             or getattr(snapshot, "battery_health_terminal_only", False))
        if battery_fault and not battery_available:
            allowed = False
        if self._command_mission_id in self._battery_landings:
            allowed = bool(allowed and battery_available and command == 21)
        if mode_arm:
            if getattr(self, '_scenario_pair_started_ns', None) is not None:
                allowed = False  # A newer incomplete intent cannot authorize an old ARM.
            sample = getattr(snapshot, 'battery_sample', None)
            if (c is not None and spec_for(self, c.command) and sample
                    and sample.get('valid') and 0 <= now_ns-sample['received_ns'] <= 750_000_000
                    and (sample['charge_state'] in (2, 3, 4, 5, 6) or sample['fault_bitmask'])):
                allowed = False  # LOW grace never authorizes a new Mode/ARM.
            allowed = bool(allowed and self._stream_setpoint(snapshot, now_ns, validate_only=True))
            allowed = bool(allowed and tx_ready
                and self._last_setpoint_tx_ns-self._tx_run_started_ns >= 1_000_000_000
                and c.command.command == OffboardCommand.COMMAND_TAKEOFF
                and (command != 400 or snapshot.offboard))
            if self._command_profile != OffboardCommand.FLIGHT_PROFILE_NORMAL:
                altitude = self._altitude_reference_sync.snapshot(now_ns)
                allowed = bool(allowed and altitude.state == AltitudeReferenceState.STATE_READY
                    and altitude.transport_epoch == self._command_altitude_reference_epoch)
        elif terminal:
            allowed = bool(allowed and c.command.command in (3, 4, 5)
                and snapshot.armed and snapshot.offboard
                and (self._command_profile == OffboardCommand.FLIGHT_PROFILE_NORMAL
                     or (tx_ready and self._last_tx_terminal_brake)))
        elif command == 400 and params[0] == 0.0:
            allowed = bool(allowed and snapshot.landed is True)
        else:
            allowed = False
        if not allowed:
            self._publish_local_ack(command, "VEHICLE_CMD_RESULT_DENIED")
            return
        if (self._command_mission_id in self._ended_mission_ids
                and (command == 176 or (command == 400 and params[0] == 1.0))):
            self._publish_local_ack(command, "VEHICLE_CMD_RESULT_DENIED")
            return
        try:
            self._transport.send_vehicle_command(command, params, snapshot)
        except PermissionError as exc:
            if not self._transport.flight_output_enabled:
                self.get_logger().error(str(exc))
                self._publish_local_ack(command, "VEHICLE_CMD_RESULT_DENIED")
            return
        except ValueError as exc:
            self.get_logger().error("Rejected VehicleCommand: " + str(exc))
            self._publish_local_ack(command, "VEHICLE_CMD_RESULT_UNSUPPORTED")
            return
        self._pending_acks[command] = (request, now_ns)
        if command == 400 and params[0] == 1.0:
            self._arm_transmitted = True
            if self._command_profile in (
                    OffboardCommand.FLIGHT_PROFILE_LOW_SPEED_1M_V1,
                    OffboardCommand.FLIGHT_PROFILE_LOW_SPEED_2M_V1):
                self._route_state.note_arm_transmitted(
                    request.mission_id, request.request_id, now_ns)
        self._journal.write(json.dumps({"event": "flight_command_written",
            "mission_id": request.mission_id, "output_epoch": request.output_epoch,
            "source_sequence": request.source_sequence,
            "output_sequence": request.output_sequence, "request_id": request.request_id,
            "command": command, "monotonic_ns": now_ns}) + "\n")

    def _stream_setpoint(self, snapshot, now_ns, *, validate_only=False):
        battery_available = self._observe_battery_health(snapshot, now_ns)
        battery_latched = self._command_mission_id in self._battery_landings
        battery_fault = bool(getattr(snapshot, "sensors_failed", 0) & (1 << 25)
                             or getattr(snapshot, "battery_health_terminal_only", False))
        if battery_fault and not battery_available and not battery_latched and not getattr(self, "_battery_navigation_allowed", False):
            self._reset_tx_run()
            self._last_output_detail = "battery_health_without_owned_terminal_contract"
            return False
        if battery_home.pair_expired(self, now_ns) and self._command_code not in (3, 4, 5):
            self._last_output_detail = battery_home.PAIR_TIMEOUT
            self._reset_tx_run()
            return False
        if battery_latched and (not battery_available or self._command_code != OffboardCommand.COMMAND_LAND):
            self._reset_tx_run()
            self._last_output_detail = (BATTERY_TERMINAL_ONLY if battery_available
                                        else BATTERY_TERMINAL_UNAVAILABLE)
            return False
        if self._tx_run_started_ns and now_ns-self._last_setpoint_tx_ns > PROOF_LEASE_NS:
            self._reset_tx_run()
        if not self._contract_valid():
            self._reset_tx_run()
            self._last_output_detail = "flight_control_contract_missing_or_mismatched"
            return
        if snapshot.armed and not snapshot.offboard:
            self._reset_tx_run()
            self._last_output_detail = "external_or_native_mode_owns_vehicle"
            return
        if (self._command_mission_id in self._ended_mission_ids
                and not (self._route_state.home_locked
                         and self._route_state.execution_home_mission_id == self._command_mission_id
                         and self._command_code in (
                             OffboardCommand.COMMAND_LAND,
                             OffboardCommand.COMMAND_ABORT))):
            self._reset_tx_run()
            self._last_output_detail = "mission_contract_ended"
            return
        if (
            self._transport is None
            or not self._transport.flight_output_enabled
            or not self._private_command_graph_ready(now_ns)
        ):
            self._reset_tx_run()
            self._last_output_detail = "setpoint_pair_missing_or_stale"
            return
        selected = self._output_pairs.select(
            now_ns, self._contract,
            minimum_source_ns=self._last_pair_source_ns)
        if selected is None:
            self._reset_tx_run()
            self._last_envelope_valid = False
            detail = "setpoint_pair_missing_or_stale"
            if self._mode is not None and self._setpoint is not None and self._envelope is not None:
                _, mismatch = setpoint_contract_matches(
                    setpoint_kind=self._envelope.setpoint_kind,
                    mode=self._mode,
                    position=self._setpoint["position"],
                    velocity=self._setpoint["velocity"],
                    yaw=self._setpoint["yaw"],
                    expected_position=self._envelope.expected_position_ned_m,
                    expected_velocity=self._envelope.expected_velocity_ned_m_s,
                    expected_yaw=self._envelope.expected_yaw_rad,
                )
                detail = mismatch or "setpoint_publication_pair_missing_or_stale"
            self._last_output_detail = detail
            if (detail != self._last_output_warning_detail
                    or now_ns-self._last_output_warning_ns >= 1_000_000_000):
                self.get_logger().error("Rejected offboard setpoint: " + detail)
                self._last_output_warning_detail = detail
                self._last_output_warning_ns = now_ns
            return
        (mode, setpoint_data, envelope, setpoint_received_ns,
         envelope_received_ns, setpoint_source_ns) = selected
        rejection_context = ""
        try:
            setpoint = make_route_setpoint(mode, setpoint_data)
            scenario = spec_for(self, self._contract.command)
            scenario_error = navigation_error(self, self._contract.command, snapshot.position)
            if scenario_error:
                raise ValueError(scenario_error)
            command_profile = {
                OffboardCommand.FLIGHT_PROFILE_LOW_SPEED_1M_V1: LOW_SPEED_1M_V1,
                OffboardCommand.FLIGHT_PROFILE_LOW_SPEED_2M_V1: LOW_SPEED_2M_V1,
            }.get(self._command_profile)
            envelope_profile = ({
                FlightEnvelope.PROFILE_LOW_SPEED_1M_V1: LOW_SPEED_1M_V1,
                FlightEnvelope.PROFILE_LOW_SPEED_2M_V1: LOW_SPEED_2M_V1,
            }.get(int(envelope.profile)) if envelope is not None else None)
            low_speed_expected = command_profile is not None or envelope_profile is not None
            altitude_contract_matches = False
            if envelope is not None:
                envelope_altitude_reference = int(getattr(
                    envelope, "altitude_reference",
                    FlightEnvelope.ALTITUDE_REFERENCE_NONE))
                envelope_altitude_max_error = float(getattr(
                    envelope, "altitude_reference_max_error_m", 0.0))
                envelope_altitude_epoch = int(getattr(
                    envelope, "altitude_reference_epoch", 0))
                envelope_altitude_sequence = int(getattr(
                    envelope, "altitude_reference_sequence", 0))
                envelope_local_boot_ms = int(getattr(
                    envelope, "altitude_reference_local_time_boot_ms", 0))
                envelope_global_boot_ms = int(getattr(
                    envelope, "altitude_reference_global_time_boot_ms", 0))
                altitude_contract_matches = bool(
                    envelope_altitude_reference
                    == self._command_altitude_reference
                    and abs(float(envelope.home_z_ned_m)
                            - self._command_home_z_ned_m) <= 1e-5
                    and abs(envelope_altitude_max_error
                            - self._command_altitude_reference_max_error_m) <= 1e-6
                    and envelope_altitude_epoch
                        == self._command_altitude_reference_epoch
                    and envelope_altitude_sequence
                        == self._command_altitude_reference_sequence
                    and envelope_local_boot_ms
                        == self._command_altitude_reference_local_time_boot_ms
                    and envelope_global_boot_ms
                        == self._command_altitude_reference_global_time_boot_ms
                    and (
                        not low_speed_expected
                        or (
                            envelope_altitude_reference
                            == FlightEnvelope.ALTITUDE_REFERENCE_FC_HOME_ALIGNED_V2
                            and envelope_altitude_epoch > 0
                            and envelope_altitude_sequence > 0
                            and abs(envelope_altitude_max_error
                                    - ALTITUDE_REFERENCE_MAX_ERROR_M) <= 1e-6
                        )
                    )
                )
            profile_matches = bool(
                envelope is not None
                and int(envelope.profile) == int(self._command_profile)
                and altitude_contract_matches
                and (not low_speed_expected or low_speed_envelope_profiles_match(
                    command_profile, envelope_profile))
            )
            contract_valid, contract_reason = (False, "flight_envelope_missing")
            if envelope is not None:
                contract_valid, contract_reason = setpoint_contract_matches(
                    setpoint_kind=getattr(envelope, "setpoint_kind", 0),
                    mode=mode,
                    position=setpoint_data["position"],
                    velocity=setpoint_data["velocity"],
                    yaw=setpoint_data["yaw"],
                    expected_position=getattr(
                        envelope, "expected_position_ned_m", ()),
                    expected_velocity=getattr(
                        envelope, "expected_velocity_ned_m_s", ()),
                    expected_yaw=getattr(envelope, "expected_yaw_rad", math.nan),
                )
            envelope_valid = not self._flight_output_requested or not (
                    envelope is None or not self._command_mission_id
                    or not envelope.mission_id
                    or envelope.mission_id != self._command_mission_id
                    or int(envelope.sequence) != self._command_sequence
                    or envelope.output_epoch != self._contract.output_epoch
                    or envelope.output_sequence != self._contract.output_sequence
                    or not profile_matches
                    or not contract_valid
                    or now_ns < envelope_received_ns
                    or now_ns-envelope_received_ns >= SETPOINT_LEASE_NS
                    or not flight_output_pair_skew_valid(
                        envelope_received_ns,
                        setpoint_received_ns))
            self._last_envelope_valid = bool(envelope_valid)
            if not envelope_valid:
                raise ValueError(
                    contract_reason
                    if envelope is not None and not contract_valid
                    else "altitude_reference_mismatch"
                    if envelope is not None and not altitude_contract_matches
                    else "fresh exact flight-output contract is required")
            recovery_hold = altitude_recovery.bridge_hold(self, snapshot, now_ns, envelope, setpoint.velocity)
            recovery_ready = getattr(snapshot, 'post_yaw_attitude_ready', True)
            if (not recovery_ready
                    and self._command_code not in (3, 5) and not recovery_hold):
                raise ValueError("yaw_reset_post_attitude_pending")
            if low_speed_expected:
                hold_mode = all(value != value for value in setpoint.velocity)
                checked_velocity = (0.0, 0.0, 0.0) if hold_mode else setpoint.velocity
                terminal_brake = bool(
                    self._command_code in (
                        OffboardCommand.COMMAND_LAND,
                        OffboardCommand.COMMAND_ABORT,
                    )
                    and int(envelope.setpoint_kind)
                    == FlightEnvelope.SETPOINT_VELOCITY
                    and all(abs(float(value)) <= 1e-6
                            for value in setpoint.velocity)
                )
                fc_altitude = None
                if not terminal_brake:
                    altitude_state = self._altitude_reference_sync.snapshot(now_ns)
                    if not altitude_state.valid:
                        raise ValueError("altitude_reference_stale")
                    correction_pending_allowed = bool(
                        altitude_state.state
                        == AltitudeReferenceState.STATE_HOME_CORRECTION_PENDING
                        and (snapshot.armed
                             or self._route_state.arm_home_transition_valid(now_ns))
                        and altitude_state.home_correction_pending_age_ms <= self._altitude_reference_sync.confirmation_ms)
                    if (not altitude_state.stable
                            or (altitude_state.state
                                != AltitudeReferenceState.STATE_READY
                                and not correction_pending_allowed)):
                        raise ValueError("altitude_reference_unstable")
                    if (altitude_state.transport_epoch
                            != self._command_altitude_reference_epoch):
                        raise ValueError("altitude_reference_epoch_changed")
                    if altitude_state.sequence < self._command_altitude_reference_sequence:
                        raise ValueError("altitude_reference_stale")
                    fc_altitude = float(
                        altitude_state.normalized_fc_altitude_home_relative_m)
                    try:
                        target_altitude = (
                            float(envelope.home_z_ned_m)
                            - float(self._command_position_ned_m[2]))
                        local_altitude = (
                            float(envelope.home_z_ned_m)
                            - float(altitude_state.local_z_ned_m))
                    except (TypeError, ValueError, OverflowError, IndexError):
                        target_altitude = math.nan
                        local_altitude = math.nan
                    rejection_context = (
                        f"mission={self._command_mission_id} "
                        f"sequence={self._command_sequence} "
                        f"home_z={float(envelope.home_z_ned_m):.3f} "
                        f"command_target_z={float(self._command_position_ned_m[2]):.3f} "
                        f"target_home_alt={target_altitude:.3f} "
                        f"paired_local_z={float(altitude_state.local_z_ned_m):.3f} "
                        f"paired_local_boot_ms={int(altitude_state.local_time_boot_ms)} "
                        f"snapshot_z={float(snapshot.position[2]):.3f} "
                        f"snapshot_source={snapshot.position_source} "
                        f"snapshot_boot_ms={int(snapshot.time_boot_ms)} "
                        f"pose_skew_ms={boot_time_distance_ms(snapshot.time_boot_ms, altitude_state.local_time_boot_ms)} "
                        f"raw_fc_home_alt={float(altitude_state.fc_altitude_home_relative_m):.3f} "
                        f"normalized_fc_home_alt={fc_altitude:.3f} "
                        f"home_correction={float(altitude_state.home_altitude_correction_m):.3f} "
                        f"home_correction_state={int(altitude_state.home_correction_state)} "
                        f"local_home_alt={local_altitude:.3f} "
                        f"max_home_alt={float(envelope.max_altitude_home_m):.3f} "
                        f"epoch={int(altitude_state.transport_epoch)}"
                    )
                    pose_reason = low_speed_pose_coherence_reason(
                        position_source=snapshot.position_source,
                        position_received_age_s=(
                            (now_ns-snapshot.position_received_ns)/1e9
                            if snapshot.position_received_ns else math.inf),
                        position_time_boot_ms=snapshot.time_boot_ms,
                        position_transport_epoch=snapshot.transport_epoch,
                        altitude_local_time_boot_ms=(
                            altitude_state.local_time_boot_ms),
                        altitude_transport_epoch=(
                            altitude_state.transport_epoch),
                    )
                    if pose_reason:
                        if recovery_hold and pose_reason == 'altitude_pose_time_skew':
                            recovery_ready = False
                        else:
                            raise ValueError(pose_reason)
                    reference_reason, _ = altitude_reference_sample_error(
                        aligned_home_z=float(envelope.home_z_ned_m),
                        local_z_ned=float(altitude_state.local_z_ned_m),
                        fc_altitude_home_relative_m=fc_altitude,
                        local_age_s=altitude_state.local_age_ms/1000.0,
                        geo_age_s=altitude_state.global_age_ms/1000.0,
                        sample_skew_s=altitude_state.source_skew_ms/1000.0,
                        max_error_m=float(
                            envelope.altitude_reference_max_error_m),
                    )
                    if reference_reason:
                        raise ValueError(reference_reason)
                    allowed, reason = envelope_allows(
                        profile=command_profile,
                        velocity=checked_velocity,
                        target_position=self._command_position_ned_m,
                        current_local_z_ned=float(
                            altitude_state.local_z_ned_m),
                        home_z_ned=float(envelope.home_z_ned_m),
                        envelope_age_s=(now_ns-envelope_received_ns)/1e9,
                        maximum_altitude_override=scenario_ceiling(scenario) if scenario else None,
                        fc_altitude_home_relative_m=fc_altitude,
                        validate_command_target=self._command_code in (
                            OffboardCommand.COMMAND_HOLD,
                            OffboardCommand.COMMAND_TAKEOFF,
                            OffboardCommand.COMMAND_GOTO,
                        ))
                    if not allowed:
                        raise ValueError(reason)
            # Establish the immutable execution Home only after every member
            # of the flight-output contract has passed.  Status polling and
            # planning previews intentionally leave Home provisional.
            if not self._route_state.home_locked:
                execution_home, execution_home_reason, _ = (
                    self._route_state.route_home(
                        now_ns, self._home_residual_limit_m))
                if execution_home is None:
                    raise ValueError(
                        execution_home_reason
                        or "execution_home_not_ready")
            locked, lock_detail = self._route_state.lock_execution_home(
                self._command_mission_id)
            if not locked:
                raise ValueError(lock_detail)
            home_correction = self._route_state.home_correction_snapshot()
            is_brake = bool(self._command_code in (3, 5)
                and int(envelope.setpoint_kind) == FlightEnvelope.SETPOINT_VELOCITY
                and all(abs(float(v)) <= 1e-6 for v in setpoint.velocity))
            if battery_latched and not is_brake:
                raise ValueError("battery_terminal_requires_zero_velocity_brake")
            if home_correction is not None:
                synchronized = self._altitude_reference_sync.set_home_reference(
                    **home_correction)
                if not synchronized and not is_brake:
                    raise ValueError("execution_home_synchronization_failed")
            elif not is_brake:
                raise ValueError("execution_home_metadata_missing")
            if validate_only:
                return True
            transmitted = self._transport.send_setpoint(
                setpoint, (replace(snapshot, failsafe=False, preflight_checks_pass=True)
                           if getattr(self, "_battery_navigation_allowed", False) else snapshot),
                now_ns=now_ns, terminal_brake=is_brake)
            if transmitted:
                self._last_pair_source_ns = setpoint_source_ns
                if not self._tx_run_started_ns:
                    self._tx_run_id += 1
                    self._tx_run_started_ns = now_ns
                    self._journal.write(json.dumps({"event": "flight_tx_run_started",
                        "mission_id": self._command_mission_id, "source_sequence": self._command_sequence,
                        "output_epoch": self._contract.output_epoch,
                        "output_sequence": self._contract.output_sequence, "tx_run_id": self._tx_run_id,
                        "monotonic_ns": now_ns, "terminal_brake": is_brake}) + "\n")
                self._tx_sequence += 1
                self._last_tx_terminal_brake = is_brake
                self._last_tx_kind = int(envelope.setpoint_kind)
                same_stream = (
                    self._setpoint_tx_mission_id == self._command_mission_id
                    and self._setpoint_tx_sequence == self._command_sequence
                )
                self._consecutive_setpoint_tx = (
                    self._consecutive_setpoint_tx + 1 if same_stream else 1
                )
                self._setpoint_tx_mission_id = self._command_mission_id
                self._setpoint_tx_sequence = self._command_sequence
                self._last_setpoint_tx_ns = now_ns
                self._last_output_detail = (BATTERY_TERMINAL_ONLY if battery_latched else
                    (altitude_recovery.READY if recovery_ready else altitude_recovery.WAIT)
                    if recovery_hold else "ready")
                if not recovery_hold:
                    self._altitude_recovery_bridge = None
        except PermissionError:
            self._reset_tx_run()
            self._last_output_detail = "transport_preflight_gate_rejected_setpoint"
            return
        except ValueError as exc:
            self._reset_tx_run()
            self._last_envelope_valid = False
            self._last_output_detail = str(exc)
            detail = str(exc)
            if (detail != self._last_output_warning_detail
                    or now_ns-self._last_output_warning_ns >= 1_000_000_000):
                suffix = (" " + rejection_context) if rejection_context else ""
                self.get_logger().error(
                    "Rejected offboard setpoint: " + detail + suffix)
                self._last_output_warning_detail = detail
                self._last_output_warning_ns = now_ns

    def _private_command_graph_ready(self, now_ns):
        if not self._flight_output_requested:
            return True
        counts = {
            topic: self.count_publishers(topic)
            for topic in (
                BRIDGE_OFFBOARD_CONTROL_MODE,
                BRIDGE_TRAJECTORY_SETPOINT,
                FLIGHT_COMMAND_REQUEST,
                FLIGHT_CONTROL_CONTRACT,
            )
        }
        ready = all(count == 1 for count in counts.values())
        if not ready and now_ns - self._last_graph_warning_ns >= 1_000_000_000:
            self.get_logger().error(
                "MAVLink output fenced: private command publisher counts must all equal one: "
                + str(counts)
            )
            self._last_graph_warning_ns = now_ns
        return ready

    def _command_graph_ready(self):
        return all(self.count_publishers(topic) == 1 for topic in (
            BRIDGE_OFFBOARD_CONTROL_MODE,
            BRIDGE_TRAJECTORY_SETPOINT,
            FLIGHT_COMMAND_REQUEST,
            FLIGHT_CONTROL_CONTRACT,
        ))

    def _publish_geo(self, received_ns):
        global_position = self._route_state.global_position
        now_ns = time.monotonic_ns()
        geo_age_ms = max(0, (now_ns-received_ns)//1_000_000)
        geo = VehicleGeoState()
        geo.stamp = self.get_clock().now().to_msg()
        geo.valid = geo_is_fresh(global_position is not None, geo_age_ms)
        if global_position:
            geo.latitude_deg = float(global_position["latitude_deg"])
            geo.longitude_deg = float(global_position["longitude_deg"])
            geo.altitude_home_relative_m = float(global_position["relative_altitude_m"])
            geo.altitude_amsl_m = float(global_position["altitude_m"])
        geo.age_ms = min(int(geo_age_ms), 2**32-1)
        self._geo_publisher.publish(geo)

    def _synchronize_home_lifecycle(self, snapshot, now_ns):
        # A single TAKEOFF command remains active while the controller streams
        # its matching envelope/setpoints. In particular, landed/disarmed is
        # expected throughout the >=1 second pre-arm TX handshake.
        if (
            self._route_state.home_locked
            and snapshot.connected
            and snapshot.landed is True
            and not snapshot.armed
            and self._route_state.execution_home_mission_id in self._ended_mission_ids
        ):
            completed_mission = self._route_state.execution_home_mission_id
            self._route_state.release_execution_home(
                completed_mission)
            self._altitude_reference_sync.release_execution_home()
            self._contract = None
            self._intent = None
            battery_home.clear_join(self)
            self._battery_observations = {}
            self._arm_transmitted = False
            self._pending_acks.clear()
            self._progress_ack_requests = set()
            self._reset_tx_run()
            # Fence cached output before another mission can acquire Home.
            self._mode = None
            self._mode_received_ns = 0
            self._setpoint = None
            self._setpoint_received_ns = 0
            self._envelope = None
            self._envelope_received_ns = 0
            self._output_pairs.clear()
            self._last_pair_source_ns = 0
            self._commands.clear()
            self._last_setpoint_tx_ns = 0
            self._consecutive_setpoint_tx = 0
            self._last_envelope_valid = False
            self._last_output_detail = "mission_contract_ended"
            self.get_logger().info(
                "Execution Home released after approval retirement and fresh "
                "landed/disarmed: " + completed_mission)
        if self._route_state.home_locked:
            metadata = self._route_state.home_correction_snapshot()
            if metadata is not None:
                if metadata.get("epoch_failure_latched"):
                    self._altitude_reference_sync.reject_home_correction(
                        metadata.get("detail")
                        or "home_correction_rejected")
                self._altitude_reference_sync.set_home_reference(**metadata)
            return
        provisional = self._route_state.provisional_home_snapshot()
        if provisional is not None:
            self._altitude_reference_sync.set_provisional_home(
                altitude_amsl_m=provisional["altitude_amsl_m"],
                z_ned_m=provisional["z_ned_m"],
                revision=provisional["revision"],
                detail=provisional["detail"],
            )

    def _refresh_yaw_reset_context(self, now_ns):
        context = None
        contract = getattr(self, '_contract', None)
        command = contract.command if contract is not None else None
        spec = spec_for(self, command)
        state = self._route_state
        if (spec is not None and spec.get('yaw_reset_policy') == 'BOUNDED_YAW_V1'
                and self._contract_valid()
                and command.command in (0, 1, 2)
                and command.mission_id in self._approved_missions
                and command.mission_id not in self._ended_mission_ids
                and state.transport_epoch == spec['transport_epoch']):
            snapshot = state.snapshot(now_ns)
            altitude = self._altitude_reference_sync.snapshot(now_ns)
            if (snapshot.armed and snapshot.offboard and snapshot.connected
                    and snapshot.position_valid and not snapshot.failsafe
                    and altitude.valid and altitude.stable
                    and altitude.state == AltitudeReferenceState.STATE_READY
                    and not altitude.altitude_epoch_failure_latched
                    and altitude.transport_epoch == state.transport_epoch):
                context = (command.mission_id, state.transport_epoch,
                           state.execution_home_lock_revision)
        state.configure_yaw_reset_context(context)

    def _journal_yaw_resets(self):
        while self._route_state.yaw_reset_events:
            item = self._route_state.yaw_reset_events.popleft()
            contract = getattr(self, '_contract', None)
            item.update(output_epoch=contract.output_epoch if contract else '',
                        output_sequence=contract.output_sequence if contract else 0,
                        source_sequence=contract.command.sequence if contract else 0)
            item['navigation_reset_generations_inferred'] = self._route_state.navigation_reset_generations
            self._journal.write(json.dumps(item, allow_nan=False)+'\n')

    def _publish_state(self):
        now_ns = time.monotonic_ns()
        snapshot = self._route_state.snapshot(now_ns)
        battery_home.record_sample(self, snapshot, now_ns)
        battery_available = self._observe_battery_health(snapshot, now_ns)
        route_home, route_frame_reason, route_frame_residual = self._route_state.route_home(
            now_ns, self._home_residual_limit_m)
        self._synchronize_home_lifecycle(snapshot, now_ns)
        altitude = self._altitude_reference_sync.snapshot(now_ns)
        altitude_message = AltitudeReferenceState()
        altitude_message.stamp = self.get_clock().now().to_msg()
        altitude_message.state = int(altitude.state)
        altitude_message.valid = bool(altitude.valid)
        altitude_message.stable = bool(altitude.stable)
        altitude_message.transport_epoch = int(altitude.transport_epoch)
        altitude_message.sequence = int(altitude.sequence)
        altitude_message.local_time_boot_ms = int(altitude.local_time_boot_ms)
        altitude_message.global_time_boot_ms = int(altitude.global_time_boot_ms)
        altitude_message.local_age_ms = int(altitude.local_age_ms)
        altitude_message.global_age_ms = int(altitude.global_age_ms)
        altitude_message.source_skew_ms = int(altitude.source_skew_ms)
        altitude_message.local_z_ned_m = float(altitude.local_z_ned_m)
        altitude_message.fc_altitude_home_relative_m = float(
            altitude.fc_altitude_home_relative_m)
        altitude_message.normalized_fc_altitude_home_relative_m = float(
            altitude.normalized_fc_altitude_home_relative_m)
        altitude_message.global_altitude_amsl_m = float(
            altitude.global_altitude_amsl_m)
        altitude_message.candidate_aligned_home_z_ned_m = float(
            altitude.candidate_aligned_home_z_ned_m)
        altitude_message.stable_aligned_home_z_ned_m = float(
            altitude.stable_aligned_home_z_ned_m)
        altitude_message.candidate_span_m = float(altitude.candidate_span_m)
        altitude_message.sample_count = int(altitude.sample_count)
        altitude_message.window_duration_ms = int(altitude.window_duration_ms)
        altitude_message.home_correction_state = int(
            altitude.home_correction_state)
        altitude_message.home_phase = int(altitude.home_phase)
        altitude_message.execution_home_lock_valid = bool(
            altitude.execution_home_lock_valid)
        altitude_message.execution_home_mission_id = str(
            altitude.execution_home_mission_id)
        altitude_message.execution_home_lock_revision = int(
            altitude.execution_home_lock_revision)
        altitude_message.provisional_home_revision = int(
            altitude.provisional_home_revision)
        altitude_message.provisional_px4_home_altitude_amsl_m = float(
            altitude.provisional_px4_home_altitude_amsl_m)
        altitude_message.provisional_px4_home_z_ned_m = float(
            altitude.provisional_px4_home_z_ned_m)
        altitude_message.home_correction_valid = bool(
            altitude.home_correction_valid)
        altitude_message.home_correction_revision = int(
            altitude.home_correction_revision)
        correction_key = (self._altitude_reference_sync._correction_event_id,
                          altitude.home_correction_state)
        if correction_key != self._last_correction_audit:
            self._last_correction_audit = correction_key
            self._journal.write(json.dumps({"event": "home_correction_transition",
                "mission_id": self._command_mission_id, "source_sequence": self._command_sequence,
                "output_epoch": self._contract.output_epoch if self._contract else "",
                "output_sequence": self._contract.output_sequence if self._contract else 0,
                "correction_event_id": correction_key[0], "state": correction_key[1],
                "observed_monotonic_ns": self._altitude_reference_sync._home_correction_pending_since_ns,
                "confirmation_baseline_revision": self._altitude_reference_sync._correction_event_revision,
                "revision": altitude.home_correction_revision,
                "detail": altitude.home_correction_detail,
                "route_detail": self._route_state.home_correction_detail,
                "published_monotonic_ns": now_ns,
                "armed_observed": snapshot.armed,
                "heartbeat_received_ns": self._route_state.received_ns.get("HEARTBEAT", 0),
                "arm_home_transition": getattr(self._route_state, "_arm_home_transition", None),
                "frozen_home_altitude_amsl_m": altitude.frozen_px4_home_altitude_amsl_m,
                "current_home_altitude_amsl_m": altitude.current_px4_home_altitude_amsl_m,
                "home_horizontal_global_delta_m": self._route_state.home_horizontal_global_delta_m,
                "home_horizontal_local_delta_m": self._route_state.home_horizontal_local_delta_m,
                "home_horizontal_consistency_error_m": self._route_state.home_horizontal_consistency_error_m,
                "home_frame_continuity_delta_m": self._route_state.home_frame_continuity_delta_m,
                "estimator_reset_counter_valid": altitude.estimator_reset_counter_valid}) + "\n")
        altitude_message.home_correction_pending_age_ms = int(
            altitude.home_correction_pending_age_ms)
        altitude_message.frozen_px4_home_altitude_amsl_m = float(
            altitude.frozen_px4_home_altitude_amsl_m)
        altitude_message.current_px4_home_altitude_amsl_m = float(
            altitude.current_px4_home_altitude_amsl_m)
        altitude_message.frozen_px4_home_z_ned_m = float(
            altitude.frozen_px4_home_z_ned_m)
        altitude_message.current_px4_home_z_ned_m = float(
            altitude.current_px4_home_z_ned_m)
        altitude_message.home_altitude_correction_m = float(
            altitude.home_altitude_correction_m)
        altitude_message.home_z_correction_m = float(
            altitude.home_z_correction_m)
        altitude_message.home_correction_opposition_error_m = float(
            altitude.home_correction_opposition_error_m)
        altitude_message.frozen_home_crosscheck_error_m = float(
            altitude.frozen_home_crosscheck_error_m)
        altitude_message.estimator_reset_counter_valid = bool(
            altitude.estimator_reset_counter_valid)
        altitude_message.estimator_reset_counter = int(
            altitude.estimator_reset_counter)
        altitude_message.altitude_epoch_failure_latched = bool(
            altitude.altitude_epoch_failure_latched)
        altitude_message.home_correction_detail = str(
            altitude.home_correction_detail)
        altitude_message.detail = str(altitude.detail)
        self._altitude_reference_publisher.publish(altitude_message)
        output = FlightOutputState()
        output.stamp = self.get_clock().now().to_msg()
        output.mission_id = self._setpoint_tx_mission_id
        output.sequence = int(self._setpoint_tx_sequence)
        output.transport_connected = bool(
            self._transport is not None and snapshot.connected
            and not self._restart_required
        )
        output.command_graph_ready = self._command_graph_ready()
        output.envelope_valid = bool(self._last_envelope_valid)
        tx_age_ns = now_ns-self._last_setpoint_tx_ns if self._last_setpoint_tx_ns else 2**63-1
        output.setpoint_age_ms = min(max(0, int(tx_age_ns//1_000_000)), 2**32-1)
        output.setpoint_transmitted = bool(
            output.transport_connected and output.command_graph_ready
            and output.envelope_valid and output.setpoint_age_ms <= 150
        )
        output.consecutive_transmissions = (
            int(self._consecutive_setpoint_tx) if output.setpoint_transmitted else 0
        )
        output.detail = (self._battery_policy_detail if getattr(self, "_battery_navigation_allowed", False)
                         else self._last_output_detail)
        battery_latched = self._command_mission_id in self._battery_landings
        if battery_latched and output.detail != battery_home.PAIR_TIMEOUT:
            output.detail = BATTERY_TERMINAL_ONLY if battery_available else BATTERY_TERMINAL_UNAVAILABLE
        if battery_latched or output.detail in battery_home.GRACE_DETAILS or output.detail == battery_home.PAIR_TIMEOUT:
            # The diagnostic identifies the current contract even before its
            # first TX. setpoint_transmitted remains independent proof.
            output.mission_id = self._command_mission_id
            output.sequence = self._command_sequence
        if not output.setpoint_transmitted:
            self._reset_tx_run()
        contract = self._contract
        output.output_epoch = contract.output_epoch if contract else ""
        output.output_sequence = contract.output_sequence if contract else 0
        output.tx_run_id = self._tx_run_id
        output.tx_run_duration_ms = (
            max(0, self._last_setpoint_tx_ns-self._tx_run_started_ns)//1_000_000
            if self._tx_run_started_ns else 0)
        output.tx_sequence = self._tx_sequence
        output.last_tx_monotonic_ns = self._last_setpoint_tx_ns
        output.published_monotonic_ns = now_ns
        output.setpoint_kind = self._last_tx_kind
        output.terminal_brake = self._last_tx_terminal_brake
        output.arm_transmitted = self._arm_transmitted
        output.px4_custom_mode = snapshot.px4_custom_mode
        output.native_land = snapshot.native_land
        self._flight_output_state_publisher.publish(output)
        safety = LowSpeedSafetyState()
        safety.stamp = self.get_clock().now().to_msg()
        action = self._px4_parameters.get("COM_OBL_RC_ACT")
        land_speed = self._px4_parameters.get("MPC_LAND_SPEED")
        ages = [now_ns-item[1] for item in (action, land_speed) if item]
        safety.age_ms = min(int(max(ages)/1_000_000), 2**32-1) if len(ages) == 2 else 2**32-1
        safety.offboard_loss_action_land = bool(
            action and px4_offboard_loss_action_is_land(action[0]))
        safety.mpc_land_speed_m_s = float(land_speed[0]) if land_speed else 0.0
        safety.land_speed_valid = bool(
            land_speed and px4_land_speed_is_valid(land_speed[0]))
        safety.valid = bool(safety.age_ms <= 10_000 and safety.offboard_loss_action_land
                            and safety.land_speed_valid)
        if safety.valid:
            safety.detail = "ready"
        elif len(ages) != 2 or safety.age_ms > 10_000:
            safety.detail = "PX4 저속 failsafe 파라미터가 없거나 오래되었습니다"
        elif not safety.offboard_loss_action_land:
            raw_action = "unavailable" if action is None else f"{action[0]:g}"
            safety.detail = (
                f"COM_OBL_RC_ACT raw={raw_action}; "
                f"QGC에서 Land({PX4_COM_OBL_RC_ACT_LAND})로 설정하십시오")
        else:
            raw_speed = "unavailable" if land_speed is None else f"{land_speed[0]:g}"
            safety.detail = (
                f"MPC_LAND_SPEED={raw_speed} m/s; "
                f"0보다 크고 {PX4_LAND_SPEED_MAX_M_S:.1f} 이하로 설정하십시오")
        self._low_speed_safety_publisher.publish(safety)
        if snapshot.connected:
            status = VehicleStatus()
            status.timestamp = self._timestamp_us()
            status.arming_state = (
                VehicleStatus.ARMING_STATE_ARMED
                if snapshot.armed
                else VehicleStatus.ARMING_STATE_DISARMED
            )
            status.nav_state = (
                VehicleStatus.NAVIGATION_STATE_OFFBOARD
                if snapshot.offboard
                else getattr(VehicleStatus, "NAVIGATION_STATE_AUTO_LAND", 18)
                if snapshot.native_land
                else getattr(VehicleStatus, "NAVIGATION_STATE_MAX", 255)
            )
            status.pre_flight_checks_pass = snapshot.preflight_checks_pass
            status.failsafe = bool(snapshot.failsafe and not battery_available
                                   and not getattr(self, "_battery_navigation_allowed", False))
            self._vehicle_status_publisher.publish(status)

            position = VehicleLocalPosition()
            position.timestamp = position.timestamp_sample = (
                snapshot.time_boot_ms * 1000 if snapshot.time_boot_ms else self._timestamp_us()
            )
            position.xy_valid = position.z_valid = snapshot.position_valid
            position.v_xy_valid = position.v_z_valid = snapshot.position_valid
            position.x, position.y, position.z = snapshot.position
            position.vx, position.vy, position.vz = snapshot.velocity
            position.heading = snapshot.yaw
            _set_if_present(position, "heading_good_for_control", snapshot.position_valid)
            for index, name in enumerate((
                "xy_reset_counter", "z_reset_counter", "vxy_reset_counter",
                "vz_reset_counter", "heading_reset_counter"
            )):
                _set_if_present(
                    position, name,
                    snapshot.navigation_reset_generations[index]
                    if (getattr(self, '_scenario_enabled', False)
                        and len(snapshot.navigation_reset_generations) == 5)
                    else snapshot.estimator_reset_counter
                    if snapshot.estimator_reset_counter_valid else 0)
            self._local_position_publisher.publish(position)

            if route_home is not None:
                home = HomePosition()
                home.timestamp = self._timestamp_us()
                set_px4_home_geodetic(
                    home,
                    route_home["latitude_deg"],
                    route_home["longitude_deg"],
                    route_home["altitude_m"],
                )
                home.x, home.y, home.z = route_home["ned"]
                _set_if_present(home, "valid_hpos", True)
                _set_if_present(home, "valid_alt", True)
                _set_if_present(home, "valid_lpos", True)
                self._home_position_publisher.publish(home)

            if snapshot.landed is not None:
                land = VehicleLandDetected()
                land.timestamp = self._timestamp_us()
                land.landed = snapshot.landed
                self._land_publisher.publish(land)

        observer = self._observer_state.diagnostic(now_ns / 1e9)
        observer.update({
            "mode": "USB_MAVLINK_ROUTE_BRIDGE",
            "flight_commands_enabled": bool(
                self._transport and self._transport.flight_output_enabled
            ),
            "connection_error": self._connection_error,
            "route_connected": snapshot.connected,
            "route_preflight_checks_pass": snapshot.preflight_checks_pass,
            "route_position_valid": snapshot.position_valid,
            "route_armed": snapshot.armed,
            "route_offboard": snapshot.offboard,
            "route_landed": snapshot.landed,
            "route_failsafe": snapshot.failsafe,
            "route_reason": snapshot.reason,
            "route_frame_valid": route_home is not None,
            "route_frame_reason": route_frame_reason,
            "home_global_local_residual_m": (
                route_frame_residual if route_frame_residual != float("inf") else None),
            "mission_home_frozen": self._route_state.home_locked,
            "execution_home_mission_id": self._route_state.execution_home_mission_id,
            "execution_home_end_received": (
                self._route_state.execution_home_mission_id in self._ended_mission_ids),
            "offboard_command_age_ms": (
                max(0, now_ns-self._command_received_ns)//1_000_000
                if self._command_received_ns else None),
            "ignored_px4_home_refinements": (
                self._route_state.ignored_home_refinements
            ),
            "last_px4_home_message_delta_m": (
                self._route_state.last_home_message_delta_m
            ),
            "home_frame_continuity_delta_m": (
                self._route_state.home_frame_continuity_delta_m
                if math.isfinite(
                    self._route_state.home_frame_continuity_delta_m
                ) else None
            ),
            "tx_count": self._transport.tx_count if self._transport else 0,
            "rx_bad_frames": self._transport.bad_data if self._transport else 0,
        })
        if hasattr(self._journal, 'status'):
            observer['journal_delivery'] = self._journal.status()
        serialized = json.dumps(observer, allow_nan=False)
        # Snapshot persistence must not delay FC polling or ROS state publication.
        # Tests/legacy embedders may still provide an ordinary in-memory stream.
        writer = getattr(self._journal, 'write_diagnostic', self._journal.write)
        writer(serialized + "\n")
        message = String()
        message.data = serialized
        self._observer_status.publish(message)

    def _publish_observer_sample(self, kind, now_s):
        if kind == "ATTITUDE_QUATERNION":
            value = self._observer_state.attitude(now_s)
            if value:
                attitude = VehicleAttitude()
                attitude.timestamp = attitude.timestamp_sample = value["timestamp"]
                attitude.q = value["q"]
                self._observer_attitudes.publish(attitude)
                rates = VehicleAngularVelocity()
                rates.timestamp = rates.timestamp_sample = value["timestamp"]
                rates.xyz = value["xyz"]
                self._observer_rates.publish(rates)
        elif kind == "LOCAL_POSITION_NED":
            value = self._observer_state.position(now_s)
            if value:
                position = VehicleLocalPosition()
                position.timestamp = position.timestamp_sample = value["timestamp"]
                position.xy_valid = position.z_valid = value["valid"]
                position.v_xy_valid = position.v_z_valid = value["valid"]
                for field in ("x", "y", "z", "vx", "vy", "vz", "heading"):
                    setattr(position, field, float(value[field]))
                self._observer_positions.publish(position)

    def _publish_mavlink_ack(self, data, system_id, component_id):
        try:
            command, result, progress, result_param2 = validate_route_command_ack(
                data, system_id, component_id
            )
        except ValueError:
            return
        pending = self._pending_acks.get(command)
        if pending is None:
            return
        request = pending[0]
        now_ns = time.monotonic_ns()
        progress_requests = getattr(self, "_progress_ack_requests", set())
        started_ns = getattr(self, "_terminal_contract_started_ns", pending[1])
        limit = 90_000_000_000 if command in (20, 21) and request.request_id in progress_requests else 2_000_000_000
        origin = started_ns if limit == 90_000_000_000 else pending[1]
        if not 0 <= now_ns-origin < limit:
            self._pending_acks.pop(command, None)
            progress_requests.discard(request.request_id)
            return
        if (self._contract is None or request.output_epoch != self._contract.output_epoch
                or request.output_sequence != self._contract.output_sequence):
            return
        if command == 400 and request.params[0] == 1.0:
            self._route_state.note_arm_ack(request.request_id, result)
        if result == VehicleCommandAck.VEHICLE_CMD_RESULT_IN_PROGRESS:
            progress_requests.add(request.request_id)
            self._progress_ack_requests = progress_requests
        else:
            self._pending_acks.pop(command, None)
            progress_requests.discard(request.request_id)
        correlated = FlightCommandAck(mission_id=request.mission_id,
            output_epoch=request.output_epoch, output_sequence=request.output_sequence,
            request_id=request.request_id, command=command, result=result,
            received_monotonic_ns=time.monotonic_ns(), transmitted=True)
        self._context_ack_publisher.publish(correlated)
        self._journal.write(json.dumps({"event": "flight_command_ack",
            "request_id": request.request_id, "command": command, "result": result,
            "mission_id": request.mission_id, "output_epoch": request.output_epoch,
            "output_sequence": request.output_sequence,
            "monotonic_ns": now_ns}) + "\n")
        message = VehicleCommandAck()
        message.timestamp = self._timestamp_us()
        message.command = command
        message.result = result
        _set_if_present(message, "result_param1", progress)
        _set_if_present(message, "result_param2", result_param2)
        _set_if_present(message, "target_system", 1)
        _set_if_present(message, "target_component", 1)
        _set_if_present(message, "from_external", True)
        self._ack_publisher.publish(message)

    def _publish_local_ack(self, command, result_name):
        message = VehicleCommandAck()
        message.timestamp = self._timestamp_us()
        message.command = int(command)
        message.result = int(getattr(VehicleCommandAck, result_name))
        _set_if_present(message, "from_external", True)
        self._ack_publisher.publish(message)

    def destroy_node(self):
        if self._transport is not None:
            self._transport.close()
        if self._journal is not None:
            try:
                self._journal.close()
            except OSError as exc:
                self.get_logger().error('Journal shutdown incomplete: '+str(exc))
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MavlinkUsbBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

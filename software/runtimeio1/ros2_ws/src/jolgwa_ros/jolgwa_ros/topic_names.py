TEXT_INPUT = "/jolgwa/operator/text"
TEXT_COMMAND = "/jolgwa/operator/command"
MISSION_PROPOSAL = "/jolgwa/mission/proposal"
MISSION_APPROVAL = "/jolgwa/mission/approval"
MISSION_STATUS = "/jolgwa/mission/status"
ROUTE_FRAME_STATUS = "/jolgwa/mission/route_frame_status"
OFFBOARD_COMMAND = "/jolgwa/control/offboard_command"
MANUAL_OVERRIDE = "/jolgwa/control/manual_override"
MANUAL_VELOCITY = "/jolgwa/control/manual_velocity_ned"
OBSTACLE_SAFETY_DECISION = "/jolgwa/safety/decision"
LOCAL_D435I_DEBUG_DECISION = "/jolgwa/safety/local_d435i_debug"
# Backwards-compatible name for the former DA3-only safety source.
JETSON_SAFETY_DECISION = OBSTACLE_SAFETY_DECISION
VEHICLE_CONTROL_STATE = "/jolgwa/control/vehicle_state"
VEHICLE_GEO_STATE = "/jolgwa/telemetry/vehicle_geo_state"
ALTITUDE_REFERENCE_STATE = "/jolgwa/telemetry/altitude_reference_state"
FLIGHT_ENVELOPE = "/jolgwa/control/flight_envelope"
FLIGHT_OUTPUT_STATE = "/jolgwa/control/flight_output_state"
FLIGHT_CONTROL_CONTRACT = "/jolgwa/mavlink/control/flight_contract"
FLIGHT_COMMAND_REQUEST = "/jolgwa/mavlink/control/command_request"
FLIGHT_COMMAND_ACK = "/jolgwa/mavlink/control/command_ack"
LOW_SPEED_SAFETY_STATE = "/jolgwa/telemetry/low_speed_safety_state"
EVENT_OBSERVATION = "/jolgwa/perception/event_observation"
FORWARD_CAMERA_COMPRESSED = "/camera/front/compressed"

APPROVE_MISSION_SERVICE = "/jolgwa/mission/approve"
MANUAL_OVERRIDE_SERVICE = "/jolgwa/control/set_manual_override"
REQUEST_EVENT_CONTROL_SERVICE = "/jolgwa/control/request_event"
RELEASE_EVENT_CONTROL_SERVICE = "/jolgwa/control/release_event"
RESUME_MISSION_SERVICE = "/jolgwa/mission/resume"
EXECUTE_MISSION_ACTION = "/jolgwa/mission/execute"
PREPARE_FORWARD_TEST_SERVICE = "/jolgwa/mission/prepare_forward_test"
EMERGENCY_LAND_SERVICE = "/jolgwa/mission/emergency_land"

PX4_OFFBOARD_CONTROL_MODE = "/fmu/in/offboard_control_mode"
PX4_TRAJECTORY_SETPOINT = "/fmu/in/trajectory_setpoint"
PX4_VEHICLE_COMMAND = "/fmu/in/vehicle_command"


def px4_output_topic(name, message_version):
    """PX4 DDS naming: schema zero has no suffix; versioned schemas use _vN."""
    if (not isinstance(name, str) or not name or not name.isascii()
            or not all(char.islower() or char.isdigit() or char == "_" for char in name)
            or type(message_version) is not int or not 0 <= message_version <= 2**32-1):
        raise ValueError("invalid PX4 output topic/schema version")
    suffix = f"_v{message_version}" if message_version else ""
    return "/fmu/out/"+name+suffix


def require_px4_message_topic(topic, name, message_type):
    """Fail closed on an installed-message/schema mismatch; never try aliases."""
    actual = px4_output_topic(name, getattr(message_type, "MESSAGE_VERSION", 0))
    if topic != actual:
        raise ValueError(f"PX4 {name} schema/topic mismatch: configured {topic}, installed message requires {actual}")
    return topic


# Pin the protocol this project verifies (PX4 HomePosition.MESSAGE_VERSION=1).
# Pure tools need no ROS import; manager startup and actual ROS contract tests
# check the installed generated message against this supported schema.
PX4_HOME_POSITION = px4_output_topic("home_position", 1)
PX4_VEHICLE_STATUS = "/fmu/out/vehicle_status_v1"
PX4_LOCAL_POSITION = "/fmu/out/vehicle_local_position_v1"
PX4_LAND_DETECTED = "/fmu/out/vehicle_land_detected"
PX4_COMMAND_ACK = "/fmu/out/vehicle_command_ack"

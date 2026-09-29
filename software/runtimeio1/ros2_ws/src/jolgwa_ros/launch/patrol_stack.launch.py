from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def _build_launch(context):
    overlay_path = LaunchConfiguration("experiment_parameter_file").perform(context)
    stage_overlay = [overlay_path] if overlay_path else []
    share = get_package_share_directory("jolgwa_ros")
    config = share + "/config/patrol_stack.yaml"
    zones = share + "/config/zones.yaml"
    zones_file = LaunchConfiguration("zones_file")
    experiment_stage = LaunchConfiguration("experiment_stage")

    px4_transport = LaunchConfiguration("px4_transport")
    px4_transport_value = context.launch_configurations.get("px4_transport", "dds")
    if px4_transport_value not in ("dds", "usb_mavlink"):
        raise ValueError("px4_transport must be dds or usb_mavlink")
    usb_mavlink = px4_transport_value == "usb_mavlink"
    mavlink_device = LaunchConfiguration("mavlink_device")
    mavlink_python = LaunchConfiguration("mavlink_python")
    enable_mavlink_commands = LaunchConfiguration("enable_mavlink_commands")
    ros_domain_id = LaunchConfiguration("ros_domain_id")
    rmw_implementation = LaunchConfiguration("rmw_implementation")

    simulation_only = LaunchConfiguration("simulation_only")
    enable_px4_commands = LaunchConfiguration("enable_px4_commands")
    allow_real_hardware = LaunchConfiguration("allow_real_hardware")
    require_jetson_safety = LaunchConfiguration("require_jetson_safety")
    forward_test_camera_required = LaunchConfiguration(
        "forward_test_camera_required"
    )
    enable_jetson_safety_bridge = LaunchConfiguration(
        "enable_jetson_safety_bridge"
    )
    enable_d435i_safety_bridge = LaunchConfiguration(
        "enable_d435i_safety_bridge"
    )
    authorize_local_d435i_safety = LaunchConfiguration(
        "authorize_local_d435i_safety"
    )
    local_d435i_authorized = str(context.launch_configurations.get(
        "authorize_local_d435i_safety", "false"
    )).lower() == "true"
    local_d435i_enabled = str(context.launch_configurations.get(
        "enable_d435i_safety_bridge", "false"
    )).lower() == "true"
    if local_d435i_authorized and not local_d435i_enabled:
        raise ValueError(
            "authorize_local_d435i_safety=true requires "
            "enable_d435i_safety_bridge=true"
        )
    expected_safety_source = (
        "local-realsense-d435i-v1"
        if local_d435i_authorized
        else "jetson-realsense-d435i"
    )
    enable_jetson_d435i_bridge = LaunchConfiguration(
        "enable_jetson_d435i_bridge"
    )
    jetson_url = LaunchConfiguration("jetson_url")
    forward_camera_topic = LaunchConfiguration("forward_camera_topic")
    d435i_depth_topic = LaunchConfiguration("d435i_depth_topic")
    d435_depth_camera_info_topic = LaunchConfiguration("d435_depth_camera_info_topic")
    d435_geometry_profile = LaunchConfiguration("d435_geometry_profile")
    d435_geometry_required = LaunchConfiguration("d435_geometry_required")
    d435i_sensor_timeout = LaunchConfiguration("d435i_sensor_timeout")
    d435i_position_timeout = LaunchConfiguration("d435i_position_timeout")
    d435i_max_pair_skew = LaunchConfiguration("d435i_max_pair_skew")
    jetson_evidence_dir = LaunchConfiguration("jetson_evidence_dir")
    jetson_evidence_max_pairs = LaunchConfiguration(
        "jetson_evidence_max_pairs"
    )
    planner_endpoint = LaunchConfiguration("planner_endpoint")
    enable_operator_gateway = LaunchConfiguration("enable_operator_gateway")
    operator_gateway_url = LaunchConfiguration("operator_gateway_url")
    enable_event_response_node = LaunchConfiguration(
        "enable_event_response_node"
    )
    enable_event_response = LaunchConfiguration("enable_event_response")
    enable_gimbal_commands = LaunchConfiguration("enable_gimbal_commands")
    airsim_host = LaunchConfiguration("airsim_host")
    enable_phase1_bridge = LaunchConfiguration("enable_phase1_bridge")
    phase1_run_root = LaunchConfiguration("phase1_run_root")
    event_camera_backend = LaunchConfiguration("event_camera_backend")
    event_gimbal_backend = LaunchConfiguration("event_gimbal_backend")
    event_camera_topic = LaunchConfiguration("event_camera_topic")
    allow_local_event_capture = LaunchConfiguration(
        "allow_local_event_capture"
    )
    event_request_source = LaunchConfiguration("event_request_source")
    event_storage_root = LaunchConfiguration("event_storage_root")
    enable_incident_video = LaunchConfiguration("enable_incident_video")
    incident_input_topic = LaunchConfiguration("incident_input_topic")
    incident_media_path = LaunchConfiguration("incident_media_path")
    incident_trigger_x = LaunchConfiguration("incident_trigger_x")
    incident_active_duration = LaunchConfiguration(
        "incident_active_duration"
    )
    incident_inset_margin_fraction = LaunchConfiguration(
        "incident_inset_margin_fraction"
    )

    controller_remappings = []
    if usb_mavlink:
        controller_remappings = [
            ("/fmu/in/offboard_control_mode", "/jolgwa/mavlink/fmu/in/offboard_control_mode"),
            ("/fmu/in/trajectory_setpoint", "/jolgwa/mavlink/fmu/in/trajectory_setpoint"),
            ("/fmu/in/vehicle_command", "/jolgwa/mavlink/fmu/in/vehicle_command"),
            ("/fmu/out/vehicle_status_v1", "/jolgwa/mavlink/fmu/out/vehicle_status_v1"),
            ("/fmu/out/vehicle_local_position_v1", "/jolgwa/mavlink/fmu/out/vehicle_local_position_v1"),
            ("/fmu/out/vehicle_land_detected", "/jolgwa/mavlink/fmu/out/vehicle_land_detected"),
            ("/fmu/out/vehicle_command_ack", "/jolgwa/mavlink/fmu/out/vehicle_command_ack"),
        ]
    manager_remappings = []
    safety_remappings = []
    if usb_mavlink:
        manager_remappings = [
            ("/fmu/out/home_position_v1", "/jolgwa/mavlink/fmu/out/home_position_v1"),
            ("/fmu/out/vehicle_status_v1", "/jolgwa/mavlink/fmu/out/vehicle_status_v1"),
        ]
        safety_remappings = [
            ("/fmu/out/vehicle_local_position_v1",
             "/jolgwa/mavlink/fmu/out/vehicle_local_position_v1"),
        ]
    disabled_for_usb = IfCondition("false") if usb_mavlink else None

    def optional_condition(configuration):
        return disabled_for_usb if disabled_for_usb is not None else IfCondition(configuration)

    return LaunchDescription(
        [
            DeclareLaunchArgument("simulation_only", default_value="true"),
            DeclareLaunchArgument(
                "px4_transport", default_value="dds", choices=["dds", "usb_mavlink"]
            ),
            DeclareLaunchArgument(
                "mavlink_device", default_value="/dev/jolgwa-pixhawk6c"
            ),
            DeclareLaunchArgument(
                "mavlink_python",
                default_value="/home/jetson/jolgwa/.venv-observer-fc/bin/python",
            ),
            DeclareLaunchArgument(
                "enable_mavlink_commands", default_value="false"
            ),
            DeclareLaunchArgument("ros_domain_id", default_value="42"),
            DeclareLaunchArgument(
                "rmw_implementation", default_value="rmw_fastrtps_cpp"
            ),
            DeclareLaunchArgument("experiment_stage", default_value="0", choices=["0", "2", "3", "4", "5", "6"]),
            DeclareLaunchArgument("zones_file", default_value=zones),
            DeclareLaunchArgument(
                "enable_px4_commands", default_value="false"
            ),
            DeclareLaunchArgument(
                "allow_real_hardware", default_value="false"
            ),
            DeclareLaunchArgument(
                "require_jetson_safety", default_value="false"
            ),
            DeclareLaunchArgument(
                "forward_test_camera_required",
                default_value="true",
                choices=["true", "false"],
                description=(
                    "Keep true for camera-protected flight. False permits only "
                    "the deterministic 1 m/2 m low-speed forward test to "
                    "ignore camera safety; all other low-speed routes remain guarded."
                ),
            ),
            DeclareLaunchArgument(
                "enable_jetson_safety_bridge", default_value="false"
            ),
            DeclareLaunchArgument(
                "enable_d435i_safety_bridge", default_value="false"
            ),
            DeclareLaunchArgument(
                "authorize_local_d435i_safety", default_value="false"
            ),
            DeclareLaunchArgument(
                "enable_jetson_d435i_bridge", default_value="false"
            ),
            DeclareLaunchArgument(
                "jetson_url",
                default_value="http://192.168.50.112:8765",
            ),
            DeclareLaunchArgument(
                "forward_camera_topic",
                default_value="/camera/front/compressed",
            ),
            DeclareLaunchArgument(
                "d435i_depth_topic",
                default_value="/camera/camera/depth/image_rect_raw",
            ),
            DeclareLaunchArgument(
                "d435i_sensor_timeout", default_value="0.5"
            ),
            DeclareLaunchArgument(
                "d435_depth_camera_info_topic", default_value="/camera/d435i/depth/camera_info"
            ),
            DeclareLaunchArgument(
                "d435_geometry_required", default_value="true"
            ),
            DeclareLaunchArgument(
                "d435_geometry_profile", default_value="uncalibrated",
                choices=["uncalibrated", "gazebo_x500"],
                description="Gazebo preset is accepted only with simulation_only=true; physical mount remains uncalibrated",
            ),
            DeclareLaunchArgument(
                "d435i_position_timeout", default_value="0.5"
            ),
            DeclareLaunchArgument(
                "d435i_max_pair_skew", default_value="0.075"
            ),
            DeclareLaunchArgument("jetson_evidence_dir", default_value=""),
            DeclareLaunchArgument(
                "jetson_evidence_max_pairs", default_value="100"
            ),
            DeclareLaunchArgument(
                "planner_endpoint",
                default_value="http://127.0.0.1:9293/v1/plan/text",
            ),
            DeclareLaunchArgument(
                "enable_operator_gateway", default_value="false"
            ),
            DeclareLaunchArgument(
                "operator_gateway_url",
                default_value="ws://127.0.0.1:9293/ws/ros",
            ),
            DeclareLaunchArgument(
                "enable_event_response_node", default_value="true"
            ),
            DeclareLaunchArgument(
                "enable_event_response", default_value="true"
            ),
            DeclareLaunchArgument(
                "enable_gimbal_commands", default_value="true"
            ),
            DeclareLaunchArgument("airsim_host", default_value="127.0.0.1"),
            DeclareLaunchArgument(
                "enable_phase1_bridge", default_value="false"
            ),
            DeclareLaunchArgument(
                "phase1_run_root",
                default_value=(
                    "/mnt/c/work/jolgwajetson/phase1-demo-local/"
                    "phase1-demo-local/runs/vision/phase1-demo"
                ),
            ),
            DeclareLaunchArgument(
                "event_camera_backend", default_value="airsim"
            ),
            DeclareLaunchArgument(
                "event_gimbal_backend", default_value="airsim"
            ),
            DeclareLaunchArgument(
                "event_camera_topic", default_value="/camera/front/compressed"
            ),
            DeclareLaunchArgument(
                "allow_local_event_capture", default_value="true"
            ),
            DeclareLaunchArgument(
                "event_request_source", default_value="airsim-event-node"
            ),
            DeclareLaunchArgument(
                "event_storage_root",
                default_value="/mnt/c/work/jolgwajetson/outputs/event_captures",
            ),
            DeclareLaunchArgument(
                "enable_incident_video", default_value="false"
            ),
            DeclareLaunchArgument(
                "incident_input_topic",
                default_value="/camera/front/gazebo/compressed",
            ),
            DeclareLaunchArgument("incident_media_path", default_value=""),
            DeclareLaunchArgument(
                "incident_trigger_x", default_value="24.0"
            ),
            DeclareLaunchArgument(
                "incident_active_duration", default_value="15.0"
            ),
            DeclareLaunchArgument(
                "incident_inset_margin_fraction", default_value="0.08"
            ),
            SetEnvironmentVariable("ROS_DOMAIN_ID", ros_domain_id),
            SetEnvironmentVariable("RMW_IMPLEMENTATION", rmw_implementation),
            Node(
                package="jolgwa_ros",
                executable="incident_video_compositor_node",
                name="incident_video_compositor",
                output="screen",
                condition=optional_condition(enable_incident_video),
                parameters=[
                    {
                        "enabled": True,
                        "input_topic": incident_input_topic,
                        "output_topic": forward_camera_topic,
                        "media_path": incident_media_path,
                        "trigger_x_m": ParameterValue(
                            incident_trigger_x, value_type=float
                        ),
                        "active_duration_s": ParameterValue(
                            incident_active_duration, value_type=float
                        ),
                        "inset_margin_fraction": ParameterValue(
                            incident_inset_margin_fraction, value_type=float
                        ),
                    }
                ],
            ),
            Node(
                package="jolgwa_ros",
                executable="jetson_safety_bridge_node",
                name="jetson_safety_bridge",
                output="screen",
                # Camera safety is allowed with USB MAVLink when explicitly
                # enabled.  PX4 output ownership remains in the USB bridge.
                condition=IfCondition(enable_jetson_safety_bridge),
                parameters=[
                    config,
                    {
                        "jetson_url": jetson_url,
                        "camera_topic": forward_camera_topic,
                        "evidence_dir": jetson_evidence_dir,
                        "evidence_max_pairs": ParameterValue(
                            jetson_evidence_max_pairs, value_type=int
                        ),
                    },
                ],
                remappings=safety_remappings,
            ),
            Node(
                package="jolgwa_ros",
                executable="d435i_safety_bridge_node",
                name="d435i_safety_bridge",
                output="screen",
                condition=IfCondition(enable_d435i_safety_bridge),
                parameters=[config, {
                    "depth_topic": d435i_depth_topic,
                    "authoritative_output": ParameterValue(
                        authorize_local_d435i_safety, value_type=bool
                    ),
                }],
                remappings=safety_remappings,
            ),
            Node(
                package="jolgwa_ros",
                executable="jetson_d435i_bridge_node",
                name="jetson_d435i_bridge",
                output="screen",
                condition=IfCondition(enable_jetson_d435i_bridge),
                parameters=[
                    config,
                    {
                        "jetson_url": jetson_url,
                        "color_topic": forward_camera_topic,
                        "depth_topic": d435i_depth_topic,
                        "depth_camera_info_topic": d435_depth_camera_info_topic,
                        "geometry_required": ParameterValue(d435_geometry_required, value_type=bool),
                        "geometry_calibrated": ParameterValue(PythonExpression([
                            "'", d435_geometry_profile, "' == 'gazebo_x500' and '",
                            simulation_only, "'.lower() == 'true'",
                        ]), value_type=bool),
                        # SDF camera=model(.20,0,.18), base_link=model(0,0,.24).
                        # PX4 estimator uses IMU/base_link: camera FRD=(.20,0,+.06).
                        # Gazebo-only horizontal mount keeps the post-roof
                        # ground return inside the metric depth range.
                        "camera_mount_x_m": 0.20,
                        "camera_mount_y_m": 0.0,
                        "camera_mount_z_m": 0.06,
                        "camera_mount_roll_rad": 0.0,
                        "camera_mount_pitch_rad": 0.0,
                        "camera_mount_yaw_rad": 0.0,
                        "sensor_timeout_s": ParameterValue(
                            d435i_sensor_timeout, value_type=float
                        ),
                        "position_timeout_s": ParameterValue(
                            d435i_position_timeout, value_type=float
                        ),
                        "max_pair_skew_s": ParameterValue(
                            d435i_max_pair_skew, value_type=float
                        ),
                    },
                ],
                remappings=safety_remappings,
            ),
            Node(
                package="jolgwa_ros",
                executable="phase1_event_bridge_node",
                name="phase1_event_bridge",
                output="screen",
                condition=optional_condition(enable_phase1_bridge),
                parameters=[{"run_root": phase1_run_root}],
            ),
            Node(
                package="jolgwa_ros",
                executable="operator_gateway_node",
                name="operator_gateway",
                output="screen",
                condition=IfCondition(enable_operator_gateway),
                parameters=[{"gateway_url": operator_gateway_url}] + stage_overlay,
            ),
            Node(
                package="jolgwa_ros",
                executable="event_response_node",
                name="event_response",
                output="screen",
                condition=optional_condition(enable_event_response_node),
                parameters=[
                    config,
                    {
                        "enabled": ParameterValue(
                            enable_event_response, value_type=bool
                        ),
                        "gimbal_commands_enabled": ParameterValue(
                            enable_gimbal_commands, value_type=bool
                        ),
                        "airsim_host": airsim_host,
                        "camera_backend": event_camera_backend,
                        "gimbal_backend": event_gimbal_backend,
                        "ros_camera_topic": event_camera_topic,
                        "allow_local_capture_without_control": ParameterValue(
                            allow_local_event_capture, value_type=bool
                        ),
                        "request_source": event_request_source,
                        "storage_root": event_storage_root,
                    },
                ] + stage_overlay,
            ),
            Node(
                package="jolgwa_ros",
                executable="command_input_node",
                name="command_input",
                output="screen",
            ),
            Node(
                package="jolgwa_ros",
                executable="mission_planner_node",
                name="mission_planner",
                output="screen",
                parameters=[
                    config,
                    {"planner_endpoint": planner_endpoint},
                ],
            ),
            Node(
                package="jolgwa_ros",
                executable="mission_manager_node",
                name="mission_manager",
                output="screen",
                parameters=[config, {"zones_file": zones_file,
                                     "experiment_stage": ParameterValue(experiment_stage, value_type=int),
                                     "simulation_only": ParameterValue(
                                         False if usb_mavlink else simulation_only, value_type=bool),
                                     "allow_real_hardware": ParameterValue(
                                         allow_real_hardware, value_type=bool),
                                     "forward_test_camera_required": ParameterValue(
                                         forward_test_camera_required,
                                         value_type=bool,
                                     )}] + stage_overlay,
                remappings=manager_remappings,
            ),
            Node(
                package="jolgwa_ros",
                executable="mavlink_usb_bridge_node",
                name="mavlink_usb_bridge",
                output="screen",
                prefix=mavlink_python,
                condition=IfCondition("true" if usb_mavlink else "false"),
                parameters=[{
                    "mavlink_device": mavlink_device,
                    "simulation_only": ParameterValue(
                        False, value_type=bool
                    ),
                    "enable_px4_commands": ParameterValue(
                        enable_px4_commands, value_type=bool
                    ),
                    "enable_mavlink_commands": ParameterValue(
                        enable_mavlink_commands, value_type=bool
                    ),
                    "allow_real_hardware": ParameterValue(
                        allow_real_hardware, value_type=bool
                    ),
                }] + stage_overlay,
            ),
            Node(
                package="jolgwa_ros",
                executable="px4_offboard_controller",
                name="px4_offboard_controller",
                output="screen",
                parameters=[
                    config,
                    {
                        "usb_output_contract": usb_mavlink,
                        "simulation_only": ParameterValue(
                            False if usb_mavlink else simulation_only,
                            value_type=bool,
                        ),
                        "enable_px4_commands": ParameterValue(
                            enable_px4_commands, value_type=bool
                        ),
                        "enable_transport_commands": ParameterValue(
                            enable_mavlink_commands if usb_mavlink else True,
                            value_type=bool,
                        ),
                        "allow_real_hardware": ParameterValue(
                            allow_real_hardware, value_type=bool
                        ),
                        "require_jetson_safety": ParameterValue(
                            require_jetson_safety, value_type=bool
                        ),
                        "forward_test_camera_required": ParameterValue(
                            forward_test_camera_required, value_type=bool
                        ),
                        "expected_jetson_safety_source": expected_safety_source,
                        "experiment_stage": ParameterValue(experiment_stage, value_type=int),
                    },
                ] + stage_overlay,
                remappings=controller_remappings,
            ),
        ]
    ).entities


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("experiment_parameter_file", default_value="",
                              description="Optional node-scoped stage parameter overlay; applied after existing node parameters"),
        OpaqueFunction(function=_build_launch),
    ])

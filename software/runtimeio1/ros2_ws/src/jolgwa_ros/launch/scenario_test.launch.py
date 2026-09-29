"""Dedicated scenario stack. Physical output remains opt-in at launch."""
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    share = Path(get_package_share_directory("jolgwa_ros"))
    python = LaunchConfiguration("perception_python")
    enabled = LaunchConfiguration("physical_output")
    return LaunchDescription([
        DeclareLaunchArgument("physical_output", default_value="false"),
        DeclareLaunchArgument("perception_python", default_value="/home/jetson/jolgwa/.venv-da3/bin/python"),
        DeclareLaunchArgument("mavlink_python", default_value="/home/jetson/jolgwa/.venv-observer-fc/bin/python"),
        DeclareLaunchArgument("phase1_root", default_value="/home/jetson/jolgwa-models/phase1-demo-local-v2-20260912"),
        DeclareLaunchArgument("operator_gateway_url", default_value="ws://192.168.137.1:9293/ws/ros"),
        SetEnvironmentVariable("ROS_DOMAIN_ID", "42"),
        SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "0"),
        SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
        SetEnvironmentVariable("JOLGWA_PHASE1_ENABLED", "false"),
        SetEnvironmentVariable("JOLGWA_SCENARIO_PROFILE", "FRONT_DETECT_2M_PASS_3M_V1"),
        SetEnvironmentVariable("JOLGWA_D435I_RELEASE_FRAMES", "1"),
        SetEnvironmentVariable("JOLGWA_DA3_EAGER_LOAD", "false"),
        SetEnvironmentVariable("JOLGWA_NATIVE_DEPTH_LIBRARY", "/home/jetson/jolgwa/native/build/libjolgwa_depth.so"),
        ExecuteProcess(cmd=[python, "-m", "jolgwa_uav.jetson_compute_service",
                            "--mode", "control", "--host", "127.0.0.1", "--port", "8768"], output="screen"),
        ExecuteProcess(cmd=["bash", str(share/"config/scenario_camera.sh")], output="screen"),
        ExecuteProcess(cmd=[python, "-m", "jolgwa_ros.scenario_incident_node", "--ros-args",
                            "-p", ["phase1_root:=", LaunchConfiguration("phase1_root")]], output="screen"),
        Node(package="jolgwa_ros", executable="jetson_d435i_bridge_node", name="scenario_depth",
             output="screen", parameters=[str(share/"config/d435_observer.yaml"), {
                 "jetson_url": "http://127.0.0.1:8768", "phase1_enabled": False,
                 "color_topic": "/camera/camera/color/image_raw/compressed",
                 "depth_topic": "/camera/camera/depth/image_rect_raw",
                 "depth_camera_info_topic": "/camera/camera/depth/camera_info"}],
             remappings=[("/fmu/out/vehicle_local_position_v1", "/jolgwa/observer/fc/vehicle_local_position"),
                         ("/fmu/out/vehicle_attitude", "/jolgwa/observer/fc/vehicle_attitude"),
                         ("/fmu/out/vehicle_angular_velocity", "/jolgwa/observer/fc/vehicle_angular_velocity")]),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(share/"launch/patrol_stack.launch.py")),
            launch_arguments={"px4_transport": "usb_mavlink", "simulation_only": "false",
                "allow_real_hardware": enabled, "enable_px4_commands": enabled,
                "enable_mavlink_commands": enabled,
                "mavlink_python": LaunchConfiguration("mavlink_python"),
                "mavlink_device": "/dev/jolgwa-pixhawk6c",
                "forward_test_camera_required": "false", "require_jetson_safety": "true",
                "enable_event_response_node": "false", "enable_event_response": "false",
                "enable_operator_gateway": "true", "operator_gateway_url": LaunchConfiguration("operator_gateway_url"),
                "experiment_parameter_file": str(share/"config/scenario_test.yaml"),
                "ros_domain_id": "42", "rmw_implementation": "rmw_fastrtps_cpp",
                "planner_endpoint": "http://192.168.137.1:9293/v1/plan/text"}.items()),
    ])

"""Same startup command, selectable output mode; never starts a flight controller."""
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context):
    mode = LaunchConfiguration("mode").perform(context)
    if mode not in ("observe", "control"):
        raise ValueError("mode must be observe or control")
    endpoint = LaunchConfiguration("jetson_url").perform(context)
    if not endpoint:
        endpoint = "http://127.0.0.1:" + ("8766" if mode == "observe" else "8765")
    nodes = [Node(
        package="jolgwa_ros", executable="d435_bridge_node", name="d435_" + mode,
        arguments=["--mode", mode], output="screen",
        parameters=[LaunchConfiguration("config_file"), {"jetson_url": endpoint}],
    )]
    if mode == "observe":
        nodes.append(Node(package="jolgwa_ros", executable="d435_observer_viewer", output="screen"))
    return nodes


def generate_launch_description():
    default = get_package_share_directory("jolgwa_ros") + "/config/d435_observer.yaml"
    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="observe"),
        DeclareLaunchArgument("config_file", default_value=default),
        DeclareLaunchArgument("jetson_url", default_value=""),
        OpaqueFunction(function=_nodes),
    ])

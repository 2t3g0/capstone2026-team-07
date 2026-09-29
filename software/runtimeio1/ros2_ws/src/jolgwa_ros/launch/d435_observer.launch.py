"""Observation only. Does not launch a camera driver, FC bridge or controller."""
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default = get_package_share_directory("jolgwa_ros") + "/config/d435_observer.yaml"
    return LaunchDescription([
        DeclareLaunchArgument("config_file", default_value=default),
        Node(package="jolgwa_ros", executable="d435_observer_node", name="d435_observer",
             parameters=[LaunchConfiguration("config_file")], output="screen"),
        Node(package="jolgwa_ros", executable="d435_observer_viewer", name="d435_observer_viewer",
             output="screen"),
    ])

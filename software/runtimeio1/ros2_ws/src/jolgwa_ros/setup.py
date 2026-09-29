from glob import glob
from setuptools import find_packages, setup


package_name = "jolgwa_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml") + glob("config/*.sh")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=[
        "setuptools",
        "PyYAML",
        "websockets>=9.1",
        "imageio-ffmpeg>=0.5",
    ],
    zip_safe=True,
    maintainer="Jolgwa Team",
    maintainer_email="kkangye6@gmail.com",
    description="Mission management and PX4 offboard command ownership.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "scenario_incident_node = jolgwa_ros.scenario_incident_node:main",
            "command_input_node = jolgwa_ros.command_input_node:main",
            "mission_manager_node = jolgwa_ros.mission_manager_node:main",
            "mission_planner_node = jolgwa_ros.mission_planner_node:main",
            "px4_offboard_controller = "
            "jolgwa_ros.px4_offboard_controller:main",
            "mavlink_usb_bridge_node = "
            "jolgwa_ros.mavlink_usb_bridge_node:main",
            "send_text_command = jolgwa_ros.send_text_command:main",
            "gazebo_perception_node = jolgwa_ros.gazebo_perception_node:main",
            "operator_gateway_node = jolgwa_ros.operator_gateway_node:main",
            "event_response_node = jolgwa_ros.event_response_node:main",
            "phase1_event_bridge_node = "
            "jolgwa_ros.phase1_event_bridge_node:main",
            "jetson_safety_bridge_node = "
            "jolgwa_ros.jetson_safety_bridge_node:main",
            "d435i_safety_bridge_node = "
            "jolgwa_ros.d435i_safety_bridge_node:main",
            "jetson_d435i_bridge_node = "
            "jolgwa_ros.jetson_d435i_bridge_node:main",
            "d435_observer_node = jolgwa_ros.d435_observer_node:main",
            "d435_bridge_node = jolgwa_ros.d435_mode_node:main",
            "d435_observer_viewer = jolgwa_ros.d435_observer_viewer:main",
            "d435_status_server = jolgwa_ros.d435_status_server:main",
            "incident_video_compositor_node = "
            "jolgwa_ros.incident_video_compositor_node:main",
        ],
    },
)

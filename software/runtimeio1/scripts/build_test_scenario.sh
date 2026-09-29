#!/usr/bin/env bash
set -eo pipefail
set +u
release="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "$release/scripts/verify_route_release.py" "$release" | tail -n 1
export ROS_DOMAIN_ID=232 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source /opt/ros/humble/setup.bash
source "${JOLGWA_ROS_UNDERLAY:-$HOME/jolgwa/ros2_ws/install/setup.bash}"
cd "$release/ros2_ws"
colcon build --packages-select jolgwa_interfaces jolgwa_ros
source install/setup.bash
export PYTHONPATH="$release/src:$release/ros2_ws/src/jolgwa_ros/test:$release/tests${PYTHONPATH:+:$PYTHONPATH}"
colcon test --packages-select jolgwa_interfaces jolgwa_ros
colcon test-result --verbose
python3 -m pytest -o addopts='' -q \
  "$release/tests/test_approval_execution_lease.py" \
  "$release/tests/test_mavlink_route_fc_link.py" \
  "$release/tests/test_arm_home_order.py" \
  "$release/tests/test_battery_health_classification.py" \
  "$release/tests/test_observer_fc_transport.py" \
  "$release/tests/test_route_frame.py" \
  "$release/tests/test_bounded_yaw_reset.py" \
  "$release/tests/test_battery_observation.py" \
  "$release/tests/test_fieldflow_home67.py"
export JOLGWA_VERIFY_RELEASE="$release"
python3 - <<'PY'
import os, hashlib
from pathlib import Path
import jolgwa_interfaces
from jolgwa_ros import operator_gateway_node, scenario_runtime, mavlink_usb_bridge_node, px4_offboard_controller
root=Path(os.environ['JOLGWA_VERIFY_RELEASE']).resolve()
for module in (jolgwa_interfaces, operator_gateway_node, scenario_runtime, mavlink_usb_bridge_node, px4_offboard_controller):
    path=Path(module.__file__).resolve()
    assert path.is_relative_to(root),path
    print(path)
assert operator_gateway_node.BUILD_ID=='low-speed-runtimeio1-20260929'
for source in (root/'ros2_ws/src/jolgwa_ros/jolgwa_ros').glob('*.py'):
    installed=Path(operator_gateway_node.__file__).parent/source.name
    assert hashlib.sha256(source.read_bytes()).digest()==hashlib.sha256(installed.read_bytes()).digest(),source.name
print('IMPORT_OK',operator_gateway_node.BUILD_ID)
PY
python3 "$release/scripts/verify_route_release.py" "$release" | tail -n 1

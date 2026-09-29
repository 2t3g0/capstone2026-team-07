#!/usr/bin/env bash
set -eo pipefail
set +u
release="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-false}"
case "$output" in true|false) ;; *) echo 'usage: bash run_scenario.sh [true|false]' >&2; exit 2 ;; esac
python3 "$release/scripts/verify_route_release.py" "$release" | tail -n 1
systemctl --user stop jolgwa-observer-radio.service
if systemctl --user is-active --quiet jolgwa-observer-radio.service; then
  echo 'STOP: observer service remains active' >&2; exit 1
fi
for i in $(seq 1 15); do
  if ! fuser /dev/jolgwa-pixhawk6c >/dev/null 2>&1; then break; fi
  sleep 1
done
if fuser /dev/jolgwa-pixhawk6c >/dev/null 2>&1; then
  echo 'STOP: previous Pixhawk USB owner remains' >&2; exit 1
fi
source /opt/ros/humble/setup.bash
source "$HOME/jolgwa/ros2_ws/install/setup.bash"
source "$release/ros2_ws/install/setup.bash"
export PYTHONPATH="$release/src${PYTHONPATH:+:$PYTHONPATH}"
export ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export JOLGWA_VERIFY_RELEASE="$release"
python3 - <<'PY'
import os, importlib, hashlib
from pathlib import Path
from jolgwa_ros.operator_gateway_node import BUILD_ID
root=Path(os.environ['JOLGWA_VERIFY_RELEASE']).resolve()
assert BUILD_ID == 'low-speed-runtimeio1-20260929', BUILD_ID
for name in ('operator_gateway_node','mission_manager_node','mavlink_usb_bridge_node',
             'px4_offboard_controller','scenario_contract','scenario_runtime','scenario_sequence',
             'async_journal','scenario_altitude_recovery','field_diagnostics','flight_contract','scenario_pending','scenario_battery_home','scenario_yaw_reset','approval_execution','scenario_incident_node','scenario_incident_bench','low_speed','scenario_camera_status','scenario_observe_view'):
    installed=Path(importlib.import_module('jolgwa_ros.'+name).__file__).resolve()
    source=root/'ros2_ws/src/jolgwa_ros/jolgwa_ros'/(name+'.py')
    assert installed.is_relative_to(root),installed
    assert hashlib.sha256(installed.read_bytes()).digest()==hashlib.sha256(source.read_bytes()).digest(),name
print('INSTALLED_COMPONENTS_OK', BUILD_ID, flush=True)
PY
mkdir -p "$HOME/jolgwa-flight-logs"
log="$HOME/jolgwa-flight-logs/flight-runtimeio1-output-$output-$(date +%Y%m%d-%H%M%S).log"
printf 'LOG: %s\nPHYSICAL_OUTPUT: %s\n' "$log" "$output"
ros2 launch jolgwa_ros scenario_test.launch.py physical_output:="$output" 2>&1 | tee "$log"

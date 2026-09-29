#!/usr/bin/env bash
set -eo pipefail
set +u
# Reuse the accepted private OpenCV / RealSense driver overlay. This process
# owns the camera only; no FC, arming or camera-connection start interlock.
source /home/jetson/jolgwa/observer_vision_ws/install/local_setup.bash
exec ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true enable_depth:=true enable_infra1:=false enable_infra2:=false \
  enable_gyro:=false enable_accel:=false pointcloud.enable:=false align_depth.enable:=false \
  rgb_camera.color_profile:=640x480x30 depth_module.depth_profile:=640x480x30 \
  enable_sync:=true reconnect_timeout:=2.0 initial_reset:=false

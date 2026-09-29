# Jetson–Pixhawk 단일 USB MAVLink 경로 제어

이 모드는 Pixhawk USB-C를 `/dev/jolgwa-pixhawk6c`로 열어 기존 경로
컨트롤러의 명령을 MAVLink 2로 변환합니다. Micro XRCE-DDS Agent는 사용하지
않으며, bridge는 공용 `/fmu/*` 토픽을 발행하지 않습니다.

## 중요한 제한

- Pixhawk USB 포트를 여는 프로세스는 정확히 하나여야 합니다.
- 기존 `observer_usb_fc_node` 또는 `jolgwa-observer-radio.service`가 포트를
  소유한 상태에서 route bridge는 시작을 거부합니다. `pkill`로 우회하지 말고
  해당 서비스의 정상 종료 절차를 사용합니다.
- USB 모드에서는 사건 대응, 카메라 제어 및 장애물 회피 노드를 launch에서
  강제로 비활성화합니다.
- 실제 출력은 `enable_px4_commands`, `enable_mavlink_commands`,
  `allow_real_hardware`가 모두 `true`일 때만 열립니다.
- 코드 설치만으로 비행 가능 판정을 하지 않습니다. 프로펠러 제거 시험과
  RC/PX4 failsafe 시험이 필수입니다.

## 설치 및 빌드

Jetson의 기존 pymavlink 환경을 재사용합니다.

```bash
source /opt/ros/humble/setup.bash
source /home/jetson/jolgwa/ros2_ws/install/setup.bash

cd /home/jetson/jolgwa/ros2_ws
colcon build --packages-select jolgwa_interfaces jolgwa_ros
source /home/jetson/jolgwa/ros2_ws/install/setup.bash

/home/jetson/jolgwa/.venv-observer-fc/bin/python -c \
  "import pymavlink, serial, rclpy; print('MAVLink bridge dependencies OK')"
```

## 1. 출력 없이 연결 확인

먼저 기존 USB observer 서비스를 정상 종료하고 소유자가 없는지 확인합니다.

```bash
sudo fuser -v /dev/jolgwa-pixhawk6c
```

아무 PID도 출력되지 않을 때 다음을 실행합니다.

```bash
source /opt/ros/humble/setup.bash
source /home/jetson/jolgwa/ros2_ws/install/setup.bash

export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 launch jolgwa_ros patrol_stack.launch.py \
  px4_transport:=usb_mavlink \
  mavlink_device:=/dev/jolgwa-pixhawk6c \
  enable_px4_commands:=false \
  enable_mavlink_commands:=false \
  allow_real_hardware:=false \
  enable_operator_gateway:=true \
  operator_gateway_url:=ws://WINDOWS_IP:9293/ws/ros \
  planner_endpoint:=http://WINDOWS_IP:9293/v1/plan/text
```

다른 터미널에서 확인합니다.

```bash
ros2 topic echo /jolgwa/mavlink/fmu/out/vehicle_status_v1 --once
ros2 topic echo /jolgwa/mavlink/fmu/out/vehicle_local_position_v1 --once
ros2 topic echo /jolgwa/control/vehicle_state --once

ros2 topic info /fmu/out/vehicle_status_v1 -v
ros2 topic info /jolgwa/mavlink/fmu/out/vehicle_status_v1 -v
sudo fuser -v /dev/jolgwa-pixhawk6c
```

정상 기준은 private status publisher 1개, USB 소유 PID 1개, 공용
`/fmu/out/vehicle_status_v1` publisher 0개입니다. Observer status는
`/jolgwa/observer/fc/status`에서 계속 확인할 수 있습니다.

## 2. 프로펠러 제거 지상시험에서만 출력 활성화

프로펠러 제거, 기체 고정, RC 조종자 대기, PX4의 `COM_OF_LOSS_T`와
`COM_OBL_RC_ACT` 확인 후에만 세 게이트를 동시에 켭니다.

```bash
ros2 launch jolgwa_ros patrol_stack.launch.py \
  px4_transport:=usb_mavlink \
  mavlink_device:=/dev/jolgwa-pixhawk6c \
  enable_px4_commands:=true \
  enable_mavlink_commands:=true \
  allow_real_hardware:=true \
  enable_operator_gateway:=true \
  operator_gateway_url:=ws://WINDOWS_IP:9293/ws/ros \
  planner_endpoint:=http://WINDOWS_IP:9293/v1/plan/text
```

승인한 가까운 지점과 낮은 시험 고도를 입력하고 다음을 확인합니다.

1. `/jolgwa/mavlink/fmu/in/trajectory_setpoint`의 X=북, Y=동, Z=하향
   부호가 계획과 일치하는지 확인합니다.
2. 1초 이상 setpoint pre-stream 뒤 Offboard와 Arm이 순서대로 승인되는지
   QGC와 bridge journal에서 확인합니다.
3. RC로 POSCTL 등 다른 모드로 바꾸면 즉시 `offboard=false`가 되고 이후
   setpoint가 끊기는지 확인합니다.
4. ROS launch를 정상 종료했을 때 PX4가 설정된 Hold/RTL/Land 동작으로
   전환되는지 확인합니다.

지상시험을 통과한 뒤에만 통제된 공간의 단일 근거리 지점 비행, 이후 전체
경로 시험 순서로 진행합니다.

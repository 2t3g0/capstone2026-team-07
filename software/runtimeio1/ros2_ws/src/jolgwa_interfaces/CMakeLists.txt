cmake_minimum_required(VERSION 3.8)
project(jolgwa_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AltitudeReferenceState.msg"
  "msg/EventObservation.msg"
  "msg/FlightEnvelope.msg"
  "msg/FlightControlContract.msg"
  "msg/FlightCommandRequest.msg"
  "msg/FlightCommandAck.msg"
  "msg/FlightOutputState.msg"
  "msg/LowSpeedSafetyState.msg"
  "msg/ManualOverride.msg"
  "msg/MissionApproval.msg"
  "msg/MissionProposal.msg"
  "msg/MissionStatus.msg"
  "msg/OffboardCommand.msg"
  "msg/RouteFrameStatus.msg"
  "msg/SafetyDecision.msg"
  "msg/TextCommand.msg"
  "msg/VehicleControlState.msg"
  "msg/VehicleGeoState.msg"
  "srv/ApproveMission.srv"
  "srv/RequestEventControl.srv"
  "srv/ReleaseEventControl.srv"
  "srv/ResumeMission.srv"
  "srv/SetManualOverride.srv"
  "srv/PrepareForwardTest.srv"
  "srv/EmergencyLand.srv"
  "action/ExecuteMission.action"
  DEPENDENCIES builtin_interfaces geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()

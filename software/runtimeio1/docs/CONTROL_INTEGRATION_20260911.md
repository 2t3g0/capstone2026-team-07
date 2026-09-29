# 회피 제어 통합 재개 — 2026-09-11

## 하드웨어 오버라이드 시험 인계

사용자는 하드웨어 담당자로부터 실제 조종기 오버라이드 시험이 완료되었다고 전달받았으며,
이번 개발에서 그 시험을 완료된 것으로 취급하도록 승인했다. 이를 **사용자/하드웨어 담당자의
시험 결과 인계**로 수용한다. 동일 하드웨어 시험을 다시 선행 조건으로 요구하지 않는다.
에이전트가 해당 시험을 직접 수행하거나 원본 로그를 검토했다는 뜻은 아니다. 시험 당시의
개별 파라미터 값, 스틱 임계량, 모드 스위치 배치, 인계 지연 수치는 전달되지 않았다.

이 인계는 PX4·실제 조종기의 오버라이드에 관한 것이다. 앱의 `ManualOverride` 메시지는
Offboard 안에서 속도를 지정하는 별도 경로이며, 이 경로와 Jetson 자동 제어의 소프트웨어
우선순위·재진입 차단은 이번 통합 회귀로 확인한다. 실제 자동 회피 비행 성공을 대신하지 않는다.

## 이번 구현 범위

1. 앱 수동 속도 제어 중에도 실제 PX4가 Offboard를 벗어나면 자동 제어 세션을 폐기한다.
   오래된 승인·수동 토글·센서 회복만으로 기존 비행 명령을 다시 발행하지 않는다.
2. 실제 Jetson에 이미 발행되는 private ROS 관찰 입력을 별도 읽기 전용 실행기에 연결한다.
   위치·속도·자세와 원본 TIMESYNC 유효기간을 대조하고, 작은 장애물 데모에 부족한 입력을
   구체적으로 기록한다. 없는 비행 상태나 지상 기준을 임의로 채우지 않는다.
3. 실제 ROS 생성자·직렬화·pub/sub 경로를 사용하는 격리된 제어권 인계 회귀를 추가한다.
   이 검사는 합성 PX4 메시지이며 Gazebo 물리 비행이나 실제 RC 시험이 아니다.

기존 `jolgwa-observer.service`와 3차 시간 동기 배포는 유지한다. ARM/이륙/비행 모드/목표값
명령을 실제 Pixhawk에 보내지 않는다. 이번 인계 수용은 관찰 성공 후 별도로 진행할 자동
비행 단계의 실행 승인을 대체하지 않는다.

## 기존 관찰 실행과 다른 점

`observe`는 거리·상승 필요 판단 표시와 기록이다. 새 `shadow`는 실제 관찰 메시지들을
자동 회피 입력 계약과 대조하는 **무동작 통합 점검**이다. 둘 다 기체를 움직이지 않는다.
작은 장애물 guard 자체가 생성하는 `STOP_HANDOVER`도 실제 기체의 정지·모드 변경 명령이
아니다. 실제 제어 연결은 동일 USB의 두 번째 reader를 여는 방식으로 구현하지 않는다.

현재 USB 관찰 출력에는 유효한 지상 기준, landed/airborne, 조종기 현재 연결 상태,
완전한 제어용 정책 결과가 모두 갖춰져 있지 않다. 실내 `estimator_constant_position_mode`는
실제 위치가 유효하지 않다는 뜻이다. 이 공백들을 승인·캘리브레이션 체크로 덮지 않는다.

## 결과 기록

- 제어기: 물리/native 모드 이탈을 앱 `ManualOverride`보다 우선하는 송신 차단을 추가했다.
  마지막 PX4 발행 경계에서도 차단한다. 초기 앱 Offboard 진입은 유지하고, 이탈 뒤에는
  새 미션 ID의 **이탈 이후 승인 + 명시적 handoff 명령 + 현재 상태 유효성**이 있어야 한다.
  토글, CLEAR, 이전 미션의 새 sequence만으로는 재진입할 수 없다.
- Windows의 회피 guard/승인/사건 Home/통합 하네스 관련 회귀 **436개 통과**.
  `artifacts/control_integration_20260911/contracts_02.xml`.
- WSL의 ROS 메시지 설치본을 사용한 기존 수동 우선순위 회귀 **103개 통과**.
  `artifacts/control_integration_20260911/ros_manual.xml`.
- 실제 ROS 생성자·heartbeat thread·DDS pub/sub·C 직렬화 경유 시험 **2개 통과**, 5.716초.
  `artifacts/control_integration_20260911/ros_manual_middleware_02.xml`.
  앱 수동만 있는 경우와 기존 승인 HOLD가 있던 경우를 각각 시험했다. 정상 속도 출력
  3회 이상을 먼저 관측하고, native 모드 이탈 이후 3종 PX4 출력이 모두 차단됨을 확인했다.
  domain231/localhost 및 UUID 토픽으로 격리한 합성 FC 시험이며 실제 PX4 시험이 아니다.
- 입력 감사 모듈: 새 회귀 **17개 통과**. 실제 ROS 입력을 수집·대조하는 코드이며,
  부족한 계약 때문에 현재 guard에는 완전한 telemetry/decision을 공급하지 않는다.
  `guard_evaluation_scope=MISSING_INPUT_REJECTION_ONLY`를 명시한다. **실제 센서로
  자동 회피 상태기계의 전체 동작을 검증한 결과가 아니며 실기체에 아직 배포하지 않았다.**

위 테스트 묶음은 서로 일부 중복된다. 합쳐서 서로 다른 시험 수로 보고하지 않는다.
첫 Windows 실행의 임시 폴더 접근 제한과 첫 middleware 실행의 토픽 remap 초기화 오류는
각각 새 전용 실행으로 수정·재검증했다. 시간/센서 기준을 완화한 것은 아니다.

실제 Jetson은 읽기 확인 당시 기존 관찰 서비스 `active`, 카메라 중앙값 약3.006m,
TIMESYNC ready, 실내 `estimator_constant_position_mode`였다. 어떤 비행 명령도 보내지
않았고 관찰 서비스를 재시작하지 않았다. 하드웨어 RC 오버라이드 시험은 반복하지 않았다.

### 재현

```powershell
wsl -d Ubuntu-22.04 -- bash /mnt/c/work/jolgwajetson/scripts/test_control_integration_ros.sh ros2_ws/src/jolgwa_ros/test/test_manual_offboard_exit_middleware.py
```

실제 제어 연결의 다음 작업은 USB MAVLink 관찰 경로와 기존 ROS Offboard 제어기 사이의
통신 연결이다. 입력 감사 실행기를 추가한 것만으로 이 통신 연결이 완료되지는 않는다.

기존 관찰 서비스/새 PC 사용법은 [세션 인수인계](SESSION_HANDOFF_20260911.md)와
[현장 빠른 시작](FIELD_DEMO_QUICKSTART_20260911.md)을 함께 참고한다.

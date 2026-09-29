# ARM 직후 Home 갱신 순서 오류 수정

후보: `low-speed-armhomeorder1-20260924` (protocol 13).

## 현장 근거와 재현

현장 임무 `ed7f879c-cbc5-45f6-88b7-30c84608c740`에서 Offboard ACK와 ARM ACK가
수락된 직후 `frozen_home_changed_before_arm`이 발생했다. ARM ACK 시각
2568603578061 ns에서 LAND 송신 2568894369561 ns까지 약 0.291초였다.
Controller에 남은 첫 오류는 `LOW_SPEED_ALTITUDE_REFERENCE_DIVERGED`였다.

기존 RouteFcState는 잠긴 Home이 변하면 마지막 HEARTBEAT의 armed 비트만
사용했다. ARM 명령 송신/수락 뒤 새 armed HEARTBEAT보다 Home 갱신이 먼저
오면 즉시 실패를 고정했다. 생산 코드로 이 오류를 수정 전에 재현했다.
현장 메시지 전체를 재생한 시험은 아니다. 실제 Home 갱신의 물리적 정당성은
제공된 이벤트 발췌만으로 확정하지 않는다.

## 수정 내용

- 실제 USB MAVLink ARM 쓰기가 성공한 저속 TAKEOFF에만 현재 mission,
  execution Home 세대, transport epoch, request ID와 최초 송신 시각을 기록한다.
- 최초 쓰기부터 최대 400ms 동안 아직 armed HEARTBEAT가 오지 않았다는 이유만으로
  Home 보정을 거부하지 않는다. 기존 위치/AMSL 연속성, ODOMETRY reset counter,
  수평 변화, 보정 크기, 증거 freshness 검사를 모두 통과해야 보정을 수용한다.
  실행 Home 좌표는 그대로 동결한다. ACK는 armed 상태를 만들지 않는다.
- 이 예외를 사용한 경우 실제 armed HEARTBEAT가 400ms 안에 확인되지 않으면
  실패를 고정한다. ARM 재전송과 IN_PROGRESS는 최초 기한을 연장하지 않는다.
  ARM 거부, Offboard 이탈, epoch 변경은 예외를 취소한다. 늦은 ACK/HEARTBEAT는
  이미 고정된 실패를 해제하지 않는다. Home 해제 시 기록을 폐기한다.
- Bridge와 Controller는 이 전환 구간에서 기존 Home 보정 pending 상태를
  관찰할 수 있다. Bridge는 실제 ARM 쓰기 증거를 독립 검사한다. Manager는
  기존 ARM 전송 관찰과 400ms pending 계약을 유지한다.
- 최초 ARM 확인 제한과 기존 Home 보정 이벤트의 400ms 기한은 각각 유지한다.
  150ms 출력 증거, 1초 warmup, 승인/제어권 검사, LAND 및 착륙 확인은 유지한다.
- Home 전환 journal에 마지막 HEARTBEAT 수신 시각, 관측 armed 상태,
  내부 ARM 전환 기록과 Home/estimator 진단을 추가했다.
- 공개 ROS 메시지와 enum, API는 바꾸지 않았다. Bridge/core/Controller와
  새 build ID를 기대하는 프런트엔드는 동일 후보로 적용해야 한다.
- 이전 지상 준비 유효시간 1000ms, 비행 중 500ms, 운영권 120초/수동 30초
  변경도 새 묶음에 포함했다. Windows API와 통합 API 각각의 기존 사본을 보존했다.

## 검증 범위

- 수정 전 재현: `frozen_home_changed_before_arm`으로 실패.
- 깨끗한 WSL ROS overlay 빌드 및 import 경로 확인.
- ROS 회귀 359개, core 회귀 81개, 프런트엔드 43개, API 두 사본 각각 13개.
- 생산 Bridge 계약/transport 검사 및 Controller 고도 검사 연결 시험:
  1m/2m × 카메라 ON/OFF × ACK/Home/GLOBAL 순서 × 0.065m/1.08m 보정.
- ARM 거부/유실, 399/400/401ms, 재전송, 오래된 request/epoch, RC 개입,
  reset 증거 누락/변경, 실제 좌표 불연속, 해제/재잠금 방어 시험.
- 실제 localhost DDS에서 상관관계 있는 ARM ACK 후 pending 고도 상태 전달.
- 기존 전체 Manager Action 루프에 ARM/Home 순서 역전 대기를 추가했다.
  이 루프의 FC 상태는 모의 입력이며 물리적 비행 동역학을 검증하지 않는다.
- 실제 pymavlink encoder/decoder와 메모리 serial을 이용한 별도 wire 회귀.
  상세 건수/결과는 동봉 `verification/wire-tests.log`를 확인한다.

wire 시험에서 기존 90초 경계 시험이 임의 호스트 uptime의 부동소수점 반올림에
의존하는 문제가 드러나 해당 시험의 시계를 고정했다. 제품의 90초 제한은 변경하지 않았다.

## 배포와 잔여 확인

기존 후보는 보존한다. 새 디렉터리에서 manifest 검증 후 빌드/테스트하고,
ROS overlay와 core PYTHONPATH를 함께 선택한다. Jetson과 프런트엔드의
build ID가 일치해야 한다. 명령은 `DEPLOY_ARMHOMEORDER1.md`에 있다.

이 작업에서 실기체 ARM/비행, Jetson 배포, 실행 서버 교체를 수행하지 않았다.
Jetson SSH 비대화식 인증은 실패했으므로 Jetson에서의 빌드/실행은 미검증이다.

PX4 SITL에서 실제 ARM 과정의 Home/HEARTBEAT 순서를 바꿔 재생하고,
Home/위치/ODOMETRY 증거가 정상일 때만 이륙을 계속하는지 확인한다.
400ms 초과, 실제 reset, RC 개입, 위치 유실 때는 기존 terminal/fallback으로
종료되어야 한다. 이후 현장에서는 1m 시험부터 ARM/Home/첫 TX/LAND 진단을
확인한다. reset counter 등 보정 증거가 실제로 부족하면 이 수정도 차단을 유지한다.
로컬 통과는 실제 비행 합격을 뜻하지 않는다.

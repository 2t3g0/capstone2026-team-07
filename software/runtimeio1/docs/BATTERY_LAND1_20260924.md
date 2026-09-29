# 배터리 경고 시 제동·LAND 인계 수정 결과

후보: `low-speed-batteryland1-20260924` / 기준: `low-speed-commandparams1-20260923`.
검증일: 2026-09-24. 공개 API·ROS 메시지·enum, protocol 13, handshake 8 유지.
USB MAVLink LOW_SPEED_1M/2M 경로의 배터리 중단만 수정했다. 실행 서버 및 Jetson은 교체하지 않았다.

## 확인한 원인과 수정

현장 ULog와 기존 Bridge 로그에서 배터리 경고 이후 출력이 중단되고 PX4가 Offboard loss로
착륙한 순서가 관측됐다. 보존된 ULog SHA-256은
`27b7ecb557c124c992e055129ed8a81c5fbaa1de82425cd4b484b5b4b9f474fb`이며,
현장 진단은 `outputs/field-altitude-20260923/px4/COMPLETE_ULOG_DIAGNOSIS.md`에 있다.
이 기록을 수정 후보의 실제 비행 검증 결과로 사용하지 않는다.

1. `RouteFcSnapshot` 내부 진단에 SYS_STATUS age와 present/enabled/health/failed 비트를 추가했다.
   신선한 SYS_STATUS에서 present/enabled 배터리 비트만 실패하고 항법·HEARTBEAT·epoch가
   유효할 때만 `battery_health_terminal_only`로 분류한다. 이 분류는 경고의 심각도를 뜻하지 않는다.
   원래 transport snapshot의 preflight=false/failsafe=true는 유지한다.
2. Bridge는 승인된 현재 저속 임무의 armed·Offboard·항법이 유효할 때만 착륙 전용 latch를 만든다.
   이때 ROS VehicleStatus의 배터리 단독 합성 failsafe를 제외하고 preflight=false는 유지한다.
   새 Mode/ARM, 기존 TAKEOFF/GOTO 및 일반 경로에는 예외를 주지 않는다.
   경고 회복도 이동을 재개시키지 않는다. 배터리 present/enabled 누락, 승인·epoch·제어권·항법
   상실 또는 복합 이상을 관측하면 해당 임무의 예외를 영구 폐기한다.
3. Controller는 현재 mission/source sequence/output epoch/sequence와 일치하고 150ms 이내인
   정확한 detail 값만 해석한다. 이동 출력보다 먼저 승인·Offboard 소유권·기체·위치 freshness를
   확인하고 기존 terminal 경로를 시작한다. DDS 토픽 순서 차이로 최신 상태가 뒤따를 때도
   그동안 이동을 내보내지 않는다. 시작을 반복하거나 최초 deadline을 연장하지 않는다.
4. 새 terminal 계약의 영속도 TX 후에만 LAND를 전달한다. ACK의 IN_PROGRESS와 최종 ACCEPTED,
   신선한 landed/disarmed 완료를 분리한다. Manager는 배터리 사유를 취소와 동시에 발생해도
   결과에 보존하며 기존 100ms LAND 재전달과 최초 terminal deadline을 유지한다.
5. 실제 transport 연결 시험에서 추가로 확인한 제동 차단을 수정했다.
   `make_route_setpoint`는 무시되는 MAVLink 위치 필드를 0으로 정규화하지만 제동 gate는 NaN을
   요구하고 있었다. 이제 검증된 RouteSetpoint의 속도 제어 마스크, 영속도, 무시되는 0 위치 필드를
   확인한다. 위치 제어·이동 속도·항법 유실·Offboard 이탈·disarmed는 계속 거부한다.

제동 계약 이전의 TX는 새 제동 증거가 아니다. 150ms 출력 증거, 1초 warmup/제동 확인,
LAND 재시도·진행 ACK, 90초 terminal 제한과 신선한 착륙 확인 기준을 변경하지 않았다.
수평 제어 이득, 고도 안정화 기준, Home 기준, PX4 배터리 임계값은 변경하지 않았다.

## 검증 결과

| 대상 | 이번 실행 결과 | 증거 |
|---|---|---|
| 깨끗한 ROS Humble overlay | 두 패키지 빌드, 323 tests / 0 errors / 0 failures / 0 skipped | verification/clean-build.log, clean-test-result.log |
| 핵심 Python | 58 passed | verification/clean-python-tests.log |
| 실제 pymavlink 2.4.49 인코더 + 메모리 serial | 80 passed | verification/wire-tests.log |
| 프런트엔드 | 40 passed, 새 expected build ID로 production build 완료 | verification/frontend-tests.log, frontend-build.log |
| 통합 API / Windows API | 각각 10 passed | verification/source-api-tests.log, windows-api-tests.log |

검증 당시 별도 ROS overlay에서 시험했다. 로컬 절대 경로는 공개본에서 생략한다.
ROS 패키지와 생성 인터페이스는 이 overlay에서, core는 같은 복사본의 src에서 import한 것을 확인했다.
API 검증은 Windows Python 3.11 환경에서 실시했다. 운영 API 가상환경에는 pytest가 없어 사용하지
않았으며 운영 의존성을 변경하지 않았다. 인코더 시험은 ROS 시험 중 일부를 별도로 재실행한 것으로,
위 수치를 합해 독립 시험 개수로 계산하지 않는다. Vite의 기존 500KiB 번들 크기 경고는 남아 있다.

| 필수 시나리오 | 판정 |
|---|---|
| 배터리 단독 → 새 제동 TX → LAND → 진행/최종 ACK → fresh landed/disarmed | 로컬 통과 |
| 1m/2m × 카메라 ON/OFF, strict ownership ON/OFF | 로컬 통과 |
| 경고 회복·중복 terminal·기존 제동 증거·이전 mission/epoch/sequence·미지 detail | 로컬 통과 |
| ARM 전, 일반 경로 및 복합 배터리 이상에는 착륙 예외 금지 | 로컬 통과 |
| RC/모드·위치·승인·epoch·SYS_STATUS 이상, 배터리 present/enabled 누락 | 로컬 통과 |
| 제동 1초 초과, 진행 ACK 90초 초과, 최종 실패/취소, ACK·착륙 상태 유실 | 로컬 통과 |
| 전체 Manager Action: 정상·배터리·배터리+취소 및 기존 복합 장애 | 로컬 통과 |
| 격리 DDS에서 배터리 진단 전달부터 첫 실제 transport 제동 write | 4.763ms, 150ms 이하 |

시험 경계: 전체 Manager Action은 모의 기체 상태를 사용한다. Controller→Bridge→production
transport 연결 시험은 메모리 serial을 사용하며 실제 MAVLink 2 제동/COMMAND_LONG 패킷을 해석한다.
DDS 시험은 실제 localhost executor를 사용하며 USB를 열 수 있는 Bridge timer를 취소한다.
이 세 층의 로컬 검증은 전체 PX4 폐루프 실행이나 하드웨어 실시간 보장이 아니다.
150ms 판정은 정상 스케줄링의 해당 실행 측정이며 과부하·네트워크·OS 지연 상한을 보장하지 않는다.

진단 JSONL에는 SYS_STATUS age/비트/실패 비트/mission/transport epoch/output epoch,
최초 battery_terminal_latched, 새 flight_tx_run_started(terminal_brake=true),
flight_command_written(command=21), flight_command_ack의 monotonic 시간이 남는다.
detail은 제동 성공 뒤에도 battery_health_terminal_only를 보존한다. 인계 불가는 별도
battery_terminal_handoff_unavailable 및 terminal_detail에 기록한다.

## 잔여 검증과 현장 순서

1. 새 후보 전체 해시·import 경로·프런트엔드/Gateway build ID를 먼저 확인한다.
   Bridge·Controller·Manager·core를 부분 교체하지 않는다. API 두 사본의 기존 차이를 유지한다.
2. PX4 SITL에서 배터리 건강 비트 단독 저하를 주입하고 preflight=false, 일반 이동 TX 중단,
   새 계약의 영속도 TX, COMMAND_LONG 21, ACK와 fresh 착륙 확인 순서를 확인한다.
   최초 경고 이후 Mode/ARM 또는 GOTO 재개가 없어야 한다.
3. SITL에서 배터리+RC, 배터리+항법 유실, SYS_STATUS 만료/복합 이상, ACK 지연·유실,
   native LAND 전환의 순서를 바꾼다. 예외 해제 후 자동 Offboard 재진입·재ARM이 없어야 하며,
   앱 인계가 확인되지 않으면 FALLBACK/FAILED로 남아야 한다.
4. 그 다음 프로펠러를 제거한 지상 시험에서 패킷·상태·ACK와 제어권 양도를 확인한다.
5. 현장에서는 정상 배터리로 1m/카메라 OFF부터, 카메라 ON 및 2m 순서로 확인한다.
   실제 저전압을 만들기 위해 배터리를 소모시키지 않는다. 경고 주입 검증은 SITL에서 실시한다.
   현장 측정에서도 최초 경고→첫 제동 TX ≤150ms, fresh proof→LAND, fresh landed/disarmed를
   각각 확인한다. 측정 불가 또는 timeout을 통과로 판정하지 않는다.

PX4 SITL·Jetson 실행·실기체 ARM·비행은 이번 작업에서 수행하지 않았다.
MAVLink SYS_STATUS만으로 PX4 내부 failsafe 전체와 경고 등급을 복원하지 않는다.
네트워크/OS/브라우저 및 ROS 발행→콜백 지연의 측정 한계(N7/N8)는 그대로 남는다.
적용 명령은 같은 후보의 `DEPLOY_BATTERYLAND1.md`에 있으며 tmux를 사용하지 않는다.

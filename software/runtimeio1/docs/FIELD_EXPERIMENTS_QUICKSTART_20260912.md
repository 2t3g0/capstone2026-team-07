# 6단계 실험 도구 — 상세 이벤트·기록 스키마 참고서

처음 실행할 때는 [공식 빠른 안내](FIELD_EXPERIMENTS_QUICKSTART_20260912_KO.md)를 먼저 읽는다. 이 문서는 사건 리허설과 닫힌 기록 평가의 상세 참고서다.

2026-09-12 · 소스 준비본. **배포·실모델 실행·새 비행 성공을 뜻하지 않는다.**

대시보드 실기체 경로 비행과 PX4/조종기 override 완료 사실은 유지한다. 이번 도구는 새 Jetson 연결·사건 경계를 단계별로 기록하고 평가하기 위한 것이며, 이미 성공한 전체 시험을 다시 돌리는 준비 절차가 아니다. [완료 사실·검증 재사용 정책](C:/work/jolgwajetson/docs/VALIDATION_REUSE_POLICY_20260911.md:34)

## 무엇을 실행할 수 있는가

| 단계 이름 | 목적 | 이번 프로그램의 실행 범위 |
| --- | --- | --- |
| `1` / `ground` | 지상 입력 연결·사건 리허설 | 기존 상태 파일 수집, 수신 전용 radio, 원본 Phase1 사건 판단·5초 녹화 |
| `2` / `jetson-sitl` | 실제 Jetson 연산↔SITL | 출력 잠긴 연결 계획 생성. Jetson/소유 SIM 연결·실행은 별도 |
| `3` / `route-observe` | 승인 경로+Jetson 관찰 | 단계별 정책과 닫힌 경로 관측 평가 |
| `4` / `route-avoid` | 장애물 1개 회피·재합류 | 단계별 정책과 같은 승인 목표·원고도 복귀 평가 |
| `5` / `event-only` | 이동 중 사건 정지·촬영·귀환 | 사건 증거·실제 영상 디코딩·Home 착륙 평가 |
| `6` / `integrated` | 회피→재합류→사건→귀환 | 같은 임무·원본 관측·시계의 단계4+사건 결과 결합 |

`--mode observe`가 기본이다. `simulate`도 이 runner에서 비행을 시작하지 않는다. **`control`은 차단된다.** 실제 물리 USB 단일 소유자에서 ROS로 유효한 Home/기체 상태를 공급하고 승인된 명령을 되돌리는 제어 bridge가 아직 연결되지 않았다. 단계3/4의 기존 SIM lifecycle 연결도 미완료다. 출력 잠금 `simulation_only=true`, `allow_real_hardware=false`, `enable_px4_commands=false`를 푼다고 이 공백이 해결되지 않는다.

기존 auto04의 PC/WSL SIM 사건 정지→5초 촬영→visited-path Home·착륙 성공은 재사용한다. `jetson-phase1-d435i` 같은 과거 source 문자열을 물리 Jetson/D435i 사용 증명으로 읽지 않는다. 실제 센서 기준은 **D435, RGB+depth이며 IMU가 아니다.** 아래 사건 벤치는 D435의 RGB만 사용한다. [기존 SIM 완주 증거](C:/work/jolgwajetson/artifacts/dashboard_event_20260912/MOVING_EVENT_AUTO04_EVIDENCE.md:3)

## 기본은 계획만 생성

기존 프로젝트 환경을 활성화하고 프로젝트 루트에서 실행한다. 아래 `python`은 그 환경의 Python 3.10 이상을 뜻한다. 패키지를 받았다면 먼저 파일과 SHA256을 검토하고 기존 프로젝트에 백업 후 반영한다. **이 묶음은 source overlay이며 설치된 프로그램도, 전체 프로젝트/모델/ROS 설치본도 아니다.**

```console
python scripts/run_field_experiment.py --stage ground
python scripts/run_field_experiment.py --stage jetson-sitl --mode simulate --target sim --backend-url http://ACTUAL_JETSON_IP:8775 --backend-manifest existing_backend_identity.json
```

`--execute`가 없으면 고유 폴더에 `plan.json`, `profile.json`, `stage.params.yaml`, `result.json`만 만든다. 카메라·모델·네트워크·비행은 시작하지 않는다. 단계2에서 PC-local/localhost 연산을 실제 Jetson으로 대체 표기하지 않으며 manifest도 현재 health 증명은 아니다. 생성된 launch 명령은 **출력 잠긴 참고 계획**이지 추가 ROS owner를 실행하라는 지시가 아니다.

## 단계1: 지상 수집과 실제 사건 리허설

다음 실행 명령은 사용자가 해당 지상 작업을 요청한 뒤에만 사용한다. 경로 예시는 반드시 실제 파일로 바꾼다.

기존 관측기의 상태 JSON만 수집한다. USB/카메라를 새로 열지 않으며 오래된 파일은 계속 과거 기록으로 취급한다.

```console
python scripts/run_field_experiment.py --stage ground --status-file existing_observer_status.json --duration-s 30 --execute
```

기록 영상으로 원본 Phase1 사건 모델을 실행하고, 실제 CONFIRMED 사건에만 기존 H.264/MKV recorder로 5초·25FPS 영상을 저장한다. **영상 재생은 실제 현장 카메라/비행 검증이 아니다.** 양성의 기대 라벨은 평가용 주석일 뿐 모델 입력이나 강제 사건이 아니다.

```console
python scripts/run_field_experiment.py --stage ground --rehearse-event --event-source video --video recorded_event.mp4 --phase1-root /home/jetson/jolgwa/phase1-demo-local --session-label positive --expected-event CALL_FOR_HELP --duration-s 30 --output-root /home/jetson/jolgwa/outputs/field_experiments --execute
python scripts/run_field_experiment.py --stage ground --rehearse-event --event-source video --video recorded_negative.mp4 --phase1-root /home/jetson/jolgwa/phase1-demo-local --session-label negative --duration-s 30 --output-root /home/jetson/jolgwa/outputs/field_experiments --execute
```

지원 사건은 `FIRE_SMOKE`, `HUMAN_VIOLENCE`, `LITTERING`, `INTRUSION_ATTEMPT`, `CALL_FOR_HELP`, `VEHICLE_ACCIDENT`이다. 단순 `PERSON` 검출로 대체하지 않는다. 원 체크포인트·threshold·temporal voting을 변경하지 않는다. 모델의 `healthy`와 registry 정확도 합격도 구분한다.

실제 D435 입력은 **기존 observer가 이미 RGB/depth를 소유하고 있지 않은 별도 지상 시간대에만** 허용한다. 기존 서비스를 자동 중지하지 않는다. 담당자가 지상에서 기존 소유자를 정리했다고 확인한 뒤에만 다음 옵션을 쓴다. `--camera-owner-confirmed`는 자동 소유권 해결 기능이 아니다. 장치가 바쁘거나 D435가 아니면 실패하며, D435i/웹캠으로 대체하지 않는다.

```console
python scripts/run_field_experiment.py --stage ground --rehearse-event --event-source d435 --serial ACTUAL_D435_SERIAL --camera-owner-confirmed --phase1-root /home/jetson/jolgwa/phase1-demo-local --session-label positive --expected-event CALL_FOR_HELP --duration-s 30 --output-root /home/jetson/jolgwa/outputs/field_experiments --execute
```

필요한 기존 의존성은 Phase1 전체 원본 모델/라이브러리, OpenCV, H.264 encoder(GStreamer 또는 FFmpeg), live RGB의 경우 `pyrealsense2`다. 모델/GPU 실행은 이 준비 작업에서 하지 않았다. 실행 시 부족한 의존성은 `BLOCKED`로 보고하며 자동 다운로드/설치를 하지 않는다. 저장은 지정한 영구 내부 경로에 하며 tmp/UNC를 거부한다. 경로 검사만으로 물리 디스크가 내부 저장장치임을 보증하지는 않는다.

결과는 상위 세션 폴더와 `event_rehearsal/session_<UUID>/`의 `session.json`, `diagnostics.jsonl`, `result.json`, `captures/*/metadata.json`, `clip.mkv`에 남는다. metadata는 보존 정책 `permanent`, 원본/중복 프레임 수, 실제 파일 완결성을 구분한다. 음성 세션은 유효 판정 20개 이상·5초 이상·유효 비율 80% 이상이 필요하며, 입력 없음·부족한 모델 coverage·추론 오류를 “사건 없음 성공”으로 표시하지 않는다. 벤치 PASS도 비행 PASS가 아니다.

## 단계3~6: 닫힌 비행 증거 평가

**비행이 종료되어 쓰기가 끝난 파일만** 사용한다. 다음 명령은 재비행·승인·ARM을 하지 않는다. `ACTUAL_MISSION_UUID`와 각 파일을 해당 동일 임무의 실제 기록으로 바꾼다.

```console
python scripts/run_field_experiment.py --stage route-observe --evaluate --observations closed/observations.jsonl --mission-id ACTUAL_MISSION_UUID --route-context approved-route.json --provenance clock-provenance.json --execute
python scripts/run_field_experiment.py --stage route-avoid --evaluate --observations closed/observations.jsonl --mission-id ACTUAL_MISSION_UUID --route-context approved-route.json --provenance clock-provenance.json --execute
python scripts/run_field_experiment.py --stage event-only --evaluate --observations closed/observations.jsonl --capture-metadata closed/capture/metadata.json --api-log closed/operations.jsonl --mission-id ACTUAL_MISSION_UUID --route-context approved-route.json --provenance clock-provenance.json --execute
python scripts/run_field_experiment.py --stage integrated --evaluate --observations closed/observations.jsonl --capture-metadata closed/capture/metadata.json --api-log closed/operations.jsonl --mission-id ACTUAL_MISSION_UUID --route-context approved-route.json --provenance clock-provenance.json --obstacle-evaluation previous_stage4/route_evaluation.json --execute
```

입력 계약:

- `observations.jsonl`: 기존 auto04 형식 `{kind, monotonic_s, payload}`의 원본 ROS 관측. 실제/합성 표시를 보존한다.
- `approved-route.json`: `schema_version=1`, `mission_id`, `proposal_id`, 승인 PATROL index 문자열→NED 좌표의 `patrol_targets_ned_m`, 실제 `acceptance_radius_m`, `altitude_tolerance_m`, `endpoint_observation_s`. 비행 관측 위치에서 목표를 역추정하지 않는다. 사건 평가는 도달 전 정지 구분을 위해 종점 대기 0과 목표까지 `arrival tolerance + 0.5m`보다 먼 상태를 요구한다.
- `clock-provenance.json`: `evidence_kind`는 `actual` 또는 `synthetic_test_fixture`; `observation_clock_id`는 관측 시계. 사건 평가는 **동일한 `capture_clock_id`** 또는 metadata `extra.evidence_clock_id`가 필요하다. 없는 식별자를 임의로 동일하게 채우지 않는다. Windows와 Jetson monotonic을 빼지 않는다. `simulation_only`와 `actual_jetson_used`는 출처 선언이며 하드웨어 인증을 대신하지 않는다.
- 단계5/6은 기존 recorder metadata와 옆의 `video_file`을 실제 디코딩한다. 같은 event/mission/lease, CONFIRMED, 정지 전 1초 이상 이동·미완료 목표, EVENT 소유 HOLD/감속, 5초·125 저장 프레임, 파일 완결 후 visited-path Home·landed/disarmed와 API 성공을 함께 검사한다. native RTL fallback 착륙은 정상 visited-path 성공으로 세지 않는다.
- 단계6은 실제 stage4 evaluator 형식, 모든 필수 검사, **같은 관측 SHA256·시계·임무**, 사건 이전 회피 복귀 완료 시각을 요구한다. 임의의 `passed=true` 객체로 통과하지 않는다.

`PASS`는 제공된 기록의 해당 검사만 통과했다는 뜻이다. `UNVERIFIED`는 필수 증거 부족, `BLOCKED`는 모순·실패·잘못된 입력이다. 모든 결과에 새 비행/승인 실행 0과 물리 비행 준비 미완료를 유지한다. 코드용 합성 fixture 통과를 실제 모델·기체 성공으로 승격하지 않는다.

## 다음 실행자가 알아야 할 경계

원본 카메라/USB는 한 소유자만 사용한다. 현장 지상 관측과 이미 승인된 다른 세션을 이 도구가 중단하지 않는다. 실제 제어 bridge·Home·ACK·ownership 연결이 준비되기 전 `control` 차단을 제거하지 않는다. source overlay를 자동 덮어쓰기·서비스 재시작·실기체 배포하는 설치기는 포함하지 않는다. 실패 시 새로 생긴 증거를 보존하고 실패한 새 경계만 고친다.

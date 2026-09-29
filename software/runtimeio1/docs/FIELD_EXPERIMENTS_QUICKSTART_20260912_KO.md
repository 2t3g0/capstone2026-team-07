# 6단계 실험 프로그램 — 준비본 / 2026-09-12

이 문서가 공식 시작 안내다. 사건 리허설·입력 계약·기록 스키마는 [상세 이벤트 참고서](FIELD_EXPERIMENTS_QUICKSTART_20260912.md)를 참고한다.

ZIP은 기존 프로젝트에 비교 적용할 **소스 overlay**이며 자동 설치·덮어쓰기를 하지 않는다. 적용 전 대상 PC의 변경 파일 3개를 별도 백업하고 비교한다. 특히 `patrol_stack.launch.py`는 수정 전 직접 백업이 없으며, 로컬 재구성 snapshot을 원본 복구본으로 보장하지 않는다. 패키지에는 직접 보존한 manager/controller 원본 2개만 포함한다.

이 문서는 **새 프로그램의 설치·선택·관측·증거 평가 방법**이다. 이 패키지로 새 비행을 실행하거나
현장 자동회피를 검증한 것은 아니다. 기존 대시보드를 통한 실기체 경로 비행 성공은 사용자 확인대로 유지한다.
이미 성공한 회피·검출 전체 검사를 반복하지 않고, 바뀐 연결부와 해당 단계의 새로운 기록만 확인한다.

## 가장 먼저 알아둘 점

- 기본 명령은 계획과 잠긴 ROS 설정만 생성한다. `--execute`가 없으면 카메라·모델·ROS·네트워크를 실행하지 않는다.
- 지금 실행 가능한 것은 **기존 상태 파일 기록, QGC 라디오 수신 기록, 지상 사건 리허설, 저장된 비행 증거 평가**다.
- `--mode control --execute`로 실기체 자동비행이 켜지지는 않는다. 실제 USB 단일 소유자와 ROS 경로/사건/Home 제어 연결이 아직 없다.
- `--mode simulate` 또한 시뮬레이터를 임의로 켜거나 기존 비행 소유자를 교체하지 않는다. 설정·명령을 준비하며 미연결 부분은 `BLOCKED`로 기록한다.
- 생성되는 잠긴 SIM 프로필은 사건 요청 출처를 `gazebo-d435-event-node`로 명시한다. 이 문자열은 물리 Jetson/카메라 사용 증명이 아니다.
- 카메라는 **D435**이며 자체 IMU가 없다. 실제 자세·각속도는 Pixhawk/PX4 자료를 사용한다. 기존 이름의 `d435i` 토픽/API는 호환 이름이다.

## 여섯 단계

| 선택 | 범위 | 이번 준비본에서 제공하는 것 |
|---|---|---|
| `1` / `ground` | 지상 링크·거리·사건 리허설 | 상태/라디오 기록, 원 사건 모델 + 실제 영상 또는 D435 RGB의 5초 녹화 도구 |
| `2` / `jetson-sitl` | 실제 Jetson 연산을 사용하는 SITL | 명시한 backend에 연결하는 잠긴 ROS 프로필과 경로·회피·사건 정책. 실제 Jetson/카메라 bridge/SITL 동일 소유자 연결 검증은 별도 |
| `3` / `route-observe` | 대시보드 경로 + 판단만 | 기존 경로에 Jetson 판단이 정지/상승 명령을 끼워 넣지 않는 정책, 관측 기록·경로 증거 평가 |
| `4` / `route-avoid` | 경로 + 장애물 1개 회피·재합류 | 회피 켜짐/사건 대응 꺼짐 정책, 회피·경로 복귀 증거 평가 |
| `5` / `event-only` | 사건 정지·5초 촬영·귀환 | 자동 회피 상승 꺼짐, 예상 밖 장애물은 안전 정지. 사건/촬영/귀환 증거 평가 |
| `6` / `integrated` | 장애물 1개 + 사건 1개 | 회피→원 경로 복귀→사건→촬영→실제 Home 귀환 정책, 두 증거의 같은 임무·시각 연결 평가 |

단계 3~6의 물리 제어 통합은 미완료다. 기존 단독 작은 장애물 `auto-demo`를 이 단계들의 전체 제어 연결로 대체하지 않는다.
단계 3/4는 기존 `prepare_owned_dashboard_session.py`의 사건/Home 필수 baseline helper에 단순 플래그로 연결할 수 없다.
단계 5/6의 기존 성공 SIM lifecycle도 현재 실행 소유자·native Home·실제 대시보드 승인 인계가 필요하다.

## 설치/실행 위치

Python 3.10 이상과 기존 프로젝트가 있는 PC/Jetson에서 프로젝트 루트 기준으로 실행한다.
순수 계획·상태 파일 수집은 Python 표준 라이브러리만 사용한다. 추가 기능은 기존 환경을 재사용한다.

- 라디오 수신: `pymavlink`, QGC에서 localhost UDP 14551로 전달 설정. 이 도구는 COM을 열거나 MAVLink를 송신하지 않는다.
- 사건 리허설: 기존 Phase1 모델/의존성, OpenCV, H.264 인코더, 실제 D435일 때 `pyrealsense2`.
- ROS 프로필: 기존 `jolgwa_ros`·PX4 메시지·ROS 설치와 해당 노드 코드 갱신/빌드가 필요하다.
- 파일 복사만으로 실행 중인 Jetson 서비스나 ROS 노드가 갱신되지는 않는다. 현재 작업에서는 배포·서비스 재시작을 하지 않았다.

다른 Codex에게는 다음처럼 전달한다.

> FIELD_EXPERIMENTS_QUICKSTART_20260912_KO.md와 패키지 manifest를 먼저 읽으세요. 기존 파일을 백업한 뒤 같은 상대경로의 소스 overlay만 적용하고, 현재 사용 중인 관측 서비스/카메라/FC 소유자를 유지하세요. 우선 계획 생성과 파일 기반 검사만 하세요. 실제 비행·ARM·승인·서비스 변경은 별도 지시 없이 하지 마세요. 누락된 물리 제어 연결을 이미 준비된 것으로 취급하지 마세요.

## 1. 계획 만들기 — 장비를 켜지 않아도 됨

```powershell
python scripts/run_field_experiment.py --stage ground
python scripts/run_field_experiment.py --stage route-observe --mode observe
python scripts/run_field_experiment.py --stage route-avoid --mode simulate
python scripts/run_field_experiment.py --stage event-only --mode simulate
python scripts/run_field_experiment.py --stage integrated --mode simulate
python scripts/run_field_experiment.py --stage jetson-sitl --mode simulate --backend-url http://192.168.50.112:8775
```

마지막 주소는 기존 환경의 예시다. 현재 실제 Jetson endpoint를 확인해 지정한다. 다른 endpoint로 자동 대체하지 않는다.
`--backend-manifest`에 `backend_url`/`compute_location`이 든 기록을 줄 수도 있다. CLI 인수와 manifest가 다르면 거부한다.
`pc_local` 또는 loopback을 실제 Jetson으로 표시하지 않으며, endpoint 선택만으로 `actual_jetson_used=true`를 기록하지 않는다.

매번 `outputs/field_experiments/<UTC>_<단계>_<무작위ID>/`가 새로 만들어진다.

- `plan.json`: 실제 선택, 빠진 부분, 잠긴 ROS launch 명령.
- `profile.json`: 해당 단계·기존 거리/촬영 기본 기준.
- `stage.params.yaml`: 마지막에 적용할 노드별 설정. JSON 문법의 유효한 YAML이며 `enable_px4_commands=false`다.
- `result.json`: `PASS`, `BLOCKED`, `UNVERIFIED`, 적용 범위와 기록 위치.

`UNVERIFIED`는 실패와 다르다. 계획만 만들었거나 기록이 부족해 성공을 증명하지 못했다는 뜻이다.
프로그램 종료 코드 0만으로 비행 성공을 판정하지 않는다. 기존 파일이나 다른 실행 결과는 덮어쓰지 않는다.

## 2. 기존 관측 상태 기록 / 수동 비행 관찰

현재 관측 프로세스가 갱신하는 JSON 파일 경로를 전달한다. 새 USB 소유자를 열지 않는다.

```powershell
python scripts/run_field_experiment.py --stage ground --status-file C:\실제경로\status.json --duration-s 30 --execute
python scripts/run_field_experiment.py --stage route-observe --target real --status-file C:\실제경로\status.json --duration-s 300 --execute
```

위 `C:\실제경로\status.json`은 입력 자리표시자다. 없으면 실행이 `BLOCKED`다. 노트북이 Jetson 파일을 자동으로 볼 수 있다는 뜻이 아니다.
갱신되지 않는 파일을 반복 읽어도 현재 `CLEAR`로 인증하지 않는다. 원본 payload·파일 시각·수집 시각을 같이 저장한다.
실제 수동 비행이나 대시보드 경로 시작은 기존 조종/승인 절차에서 한다.

네트워크/SSH 없이 기존 QGC 라디오 전달을 이용하려면:

```powershell
python scripts/run_field_experiment.py --stage route-observe --target real --radio --listen-port 14551 --duration-s 300 --execute
```

같은 UDP 포트를 사용하는 관측창이 이미 있으면 또 실행하지 않는다. 포트 충돌 시 다른 프로그램을 자동 종료하지 않는다.
라디오 전송 지연이 입증되지 않은 값은 수신 기록이지 현재 비행 허가가 아니다.

## 3. 지상 사건 리허설

원 모델과 실제 촬영 영상을 사용한다. 테스트 라벨은 결과 비교용이며 검출 이벤트를 강제로 생성하지 않는다.

```powershell
python scripts/run_field_experiment.py --stage ground --rehearse-event --event-source video --video C:\영상\littering_demo.mp4 --phase1-root C:\모델경로 --session-label positive --expected-event LITTERING --duration-s 30 --execute
python scripts/run_field_experiment.py --stage ground --rehearse-event --event-source video --video C:\영상\normal_scene.mp4 --phase1-root C:\모델경로 --session-label negative --duration-s 30 --execute
```

GPU가 없는 환경의 모델 지원 여부를 확인한 뒤 `--device cpu`를 선택할 수 있다. 기본은 `--device 0`이다.
실제 D435 RGB를 쓰려면 `--event-source d435 --camera-owner-confirmed`를 명시한다. 다른 RGB/Depth 소유자가 없음을 직접 확인한 경우에만 사용한다.
이 도구가 기존 observe 서비스를 자동 중단하지 않는다. 소유자를 바꾸기 어렵다면 기존에 저장한 영상으로 먼저 리허설한다.

리허설은 10~600초를 받아 원 사건 모델·이벤트 확인·5초/25fps 녹화 경로를 재사용한다. 실행 결과는 하위 `event_rehearsal/`에 남는다.
한 모델 입력/녹화 성공은 공중의 정지·귀환 성공이 아니다. 시간 초과로 중단되면 영상 finalize/자식 정리가 확인되지 않은 결과는 합격으로 표시하지 않는다.

## 4. 저장된 경로/회피/사건 증거 평가

평가는 닫힌 관측 파일을 읽을 뿐 새 실험을 실행하지 않는다. 이미 통과한 단계는 그 기록으로 유지하고, 바뀐 부분만 새 기록을 추가한다.

```powershell
python scripts/run_field_experiment.py --stage route-avoid --evaluate --observations observations.jsonl --route-context route_context.json --provenance provenance.json --mission-id 실제-임무-UUID --execute
python scripts/run_field_experiment.py --stage event-only --evaluate --observations observations.jsonl --capture-metadata metadata.json --api-log operations.jsonl --route-context route_context.json --provenance provenance.json --mission-id 실제-임무-UUID --execute
python scripts/run_field_experiment.py --stage integrated --evaluate --observations observations.jsonl --capture-metadata metadata.json --api-log operations.jsonl --route-context route_context.json --provenance provenance.json --obstacle-evaluation route_evaluation.json --mission-id 실제-임무-UUID --execute
```

`route_context`는 실제 승인 계획의 목표 NED 좌표·실제 도달 허용거리이며 추정값을 넣지 않는다.
`provenance`에는 실제 관측/촬영 시계의 동일성을 증명한 식별자가 필요하다. Windows·WSL·Jetson 시각을 문자열만 같게 만들어 통과시키지 않는다.
기존 metadata에 시계 근거가 없으면 해당 기록의 결과는 `UNVERIFIED`일 수 있다. 이것은 과거 비행 자체가 실패했다는 뜻이 아니다.
단계6은 같은 임무·시계의 단계4 회피복귀 결과와 사건 시각 순서까지 확인한다. 예제/단위시험 fixture는 실제 비행 증거로 승격시키지 않는다.

## 5. ROS 설정 연결 방식과 아직 남은 것

`patrol_stack.launch.py`는 `experiment_stage:=2..6`과 `experiment_parameter_file:=<stage.params.yaml>`을 받는다.
프로필은 mission_manager/controller/event_response의 기존 매개변수 **뒤에** 적용하므로 단순히 만들어 놓고 무시되는 파일이 아니다.
단계0/overlay 미지정은 기존 실행 동작을 유지한다. 사건 단계는 ROS 압축영상 + 고정 카메라를 쓰며 AirSim을 자동 실행하지 않는다.
프로필과 launch 명령은 실제 ROS 실행 환경에서 생성하거나 경로를 그 환경의 `/mnt/c/...` 등으로 맞춘다.

잠긴 프로필을 실행하면 관측·노드 초기화 준비이지 비행이 아니다. 기존 동작 중인 stack과 중복 실행하면 안 된다.
실제 비행에 필요한 작업은 다음 연결부다.

1. 실제 USB 단일 소유자의 FC 원자료·Home·시각/arming epoch를 ROS에 그대로 전달.
2. 승인된 임무의 제어 출력을 동일 소유자에게 반환하고, 기존 수동 우선권·출력 취소를 보존.
3. 준비/이륙/동일 controller 인계와 실제 대시보드 승인을 연결. 가짜 Home/승인/두 번째 setpoint 발행자를 만들지 않음.
4. 단계2의 실제 Jetson 연산 + 같은 native SITL의 영상/깊이/판단 왕복을 검증.

새 field 프로필은 과거 SIM의 완화 시간 설정을 상속하지 않는다. 따라서 과거 완화 프로필의 완주 기록을 새 field 프로필의 현장 안전성 증명으로 바꾸지 않는다.
관련 단위검사는 소프트웨어 정책·입출력 정합만 검증했으며 새 실비행·실제 모델 성능 검증은 아니다.

## 오류 / 복구

- `BLOCKED ... transport`: 실기체 제어 연결 미완료. 모드 인수나 가드를 바꿔 우회하지 않는다. 관측/파일 평가부터 사용한다.
- `BLOCKED ... owner`: 현재 카메라/UDP/FC 소유자를 확인한다. 임의 서비스 종료나 두 번째 USB 연결을 만들지 않는다.
- `UNVERIFIED ... evidence`: 실제 동일 임무·시계·승인 계획 기록을 추가한다. 빠진 값을 추측해 넣지 않는다.
- `adapter.log`: 외부 도구의 모델 경로·인코더·의존성 오류 확인. 수정 후 실패한 도구만 다시 실행한다.
- overlay 적용 전 백업을 복원하면 소스 변경을 되돌릴 수 있다. 본 배포본은 설치 스크립트가 기존 프로그램을 자동 덮어쓰지 않으며, 모델·키·비밀번호·원본 비행영상은 포함하지 않는다.

기준값은 전방 감지 3m, 1m 단계 상승, 지붕 여유 1m, 기본 후단 통과 5m, 사건 촬영 5초, 내부 저장·영구 보존을 유지한다.
실제 안전거리와 현재 센서/자세 유효성 판단은 기존 제어 코어의 값을 사용하며 이 runner가 덮어쓰지 않는다.

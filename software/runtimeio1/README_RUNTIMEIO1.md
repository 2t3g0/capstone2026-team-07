# runtimeio1: 실행 루프의 진단 디스크 쓰기 분리

기준 후보 fieldflow1은 보존한다. Bridge journal 및 고주기 추론 diagnostics를 제한된 FIFO writer로 분리했다. 사진/events.jsonl의 저장 완료·fsync, USB 실제 송신, ACK, freshness와 비행 한계는 변경하지 않았다.

큐는 in-flight를 포함해 8MiB/2048개로 제한하고 중요 기록을 위해 256KiB/64개를 예약한다. 일반 스냅샷은 포화 시 누락 수를 기록한다. 중요 기록의 큐 포화와 알려진 I/O 실패는 OSError로 명시해 기존 오류 경로로 전달한다. flush는 큐 전달을 뜻하며 디스크 영속화 증거가 아니다. 정상 종료는 최대 3초 drain 후, 잔여 기록이 있으면 미완료를 명시한다.

로컬 ROS 747 / 핵심 Python 156 / native·observe 38 / frontend 70 통과. 인터페이스와 두 API 사본은 기준 후보와 동일하다. 현장 2.8초 사례의 모든 원인이 확정된 것은 아니다. 장비 재검증 결과는 별도 배포 보고서를 따른다.

```bash
set +u
release="$HOME/jolgwa-releases/low-speed-runtimeio1-20260929"
bash "$release/scripts/build_test_scenario.sh"
# 실내 출력 차단 실행. Ctrl+C로 종료.
bash "$release/scripts/run_scenario.sh" false
```

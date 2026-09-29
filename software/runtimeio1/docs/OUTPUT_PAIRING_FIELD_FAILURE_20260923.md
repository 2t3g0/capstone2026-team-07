# 저속 전진 시험 출력 계약 불일치 — 2026-09-23

후보 `low-speed-outputpair1-20260923`은 `safetycontract2`의 현장 실패를 수정한다.
이전 후보는 두 차례 1m/카메라 OFF 이륙 준비에서 Bridge의
`velocity_contract_mismatch`가 반복되었고, 현재 임무의 연속 송신 증거를
확보하지 못해 `LOW_SPEED_PROFILE flight output confirmation timeout`으로 끝났다.
Windows 운영 기록의 실패 시각은 13:03:58, 13:06:09 KST다.

Controller는 한 50ms heartbeat에서 FlightEnvelope, OffboardControlMode,
TrajectorySetpoint를 차례로 발행한다. 기존 Bridge는 세 토픽의 최신값만
보관해 150ms 이내라면 서로 다른 heartbeat의 값을 비교할 수 있었다.
기체 로그에는 실제 양쪽 속도값이 없어, 이번 두 실패에서 DDS 역전이
일어난 세부 순서까지 증명되지는 않는다. 확인된 직접 원인은 Bridge의
반복적인 속도 계약 불일치이며, 이 변경은 그 불일치가 생길 수 있는
교차 발행주기 결합을 제거한다.

Bridge는 각 토픽의 최근 8개 값을 보관하고, 같은 공개 계약 ID,
원본 발행시각 순서, 40ms 이내의 발행 구간, 150ms 이내의 수신 skew,
250ms 입력 lease, 정확한 값과 NaN mask를 모두 만족하는 쌍만 송신한다.
후보 중 원본 발행시각이 가장 최신인 쌍을 쓰고, 이미 송신한 시각보다
과거인 중복·지연 메시지로 출력을 되돌리지 않는다.
계약 전환·USB fault·Home 해제 때 캐시를 폐기한다. 일치하는 쌍이
없으면 송신 및 Mode/ARM 준비 증거를 만들지 않는다. 150ms TX 증거,
1초 warmup, 10초 확인 timeout은 완화하지 않았다. 공개 API·ROS 메시지와
protocol 13 / handshake 8도 유지한다.

격리 DDS 순서 역전과 Bridge transport 모형에서 이전·다음 heartbeat의
속도가 다른 경우를 검사했다. 기존 N1–N8 회귀와 깨끗한 ROS overlay
빌드·테스트를 재실행한다. 이 결과는 PX4 SITL 또는 실기체 검증이 아니다.
다음 단계는 SITL에서 이륙 준비, 메시지 지연·유실·중복, terminal LAND를
확인한 뒤 프로펠러 제거 현장에서 TX 증거→Offboard→ARM 순서를
관찰하는 것이다. 기존 `safetycontract2` 후보로 비행을 재시도하지 않는다.

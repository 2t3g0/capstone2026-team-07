"""Validate the operator-agreed incident preset without editing approved JSON."""
from __future__ import annotations


SUPPORTED_EVENTS = frozenset({"FIRE_SMOKE", "HUMAN_VIOLENCE", "LITTERING",
                              "INTRUSION_ATTEMPT", "CALL_FOR_HELP", "VEHICLE_ACCIDENT"})
EVENT_ALIASES = {"FIRE": "FIRE_SMOKE", "VIOLENCE": "HUMAN_VIOLENCE",
                 "RESTRICTED_ENTRY": "INTRUSION_ATTEMPT",
                 "TRAFFIC_ACCIDENT": "VEHICLE_ACCIDENT",
                 "CALL_FOR_HELP": "CALL_FOR_HELP"}
POLICY_SUMMARY_KO = (
    "통합 시험 정책: Phase1 지원 6종(화재·연기, 폭행, 투기, 침입 시도, 도움 요청, 차량 사고)을 "
    "감시하고, 확인 시 정지·5초 촬영 후 PX4 Home으로 회피하며 귀환합니다. "
    "순찰 자동 재개·추적 비행·별도 FALL 검출은 지원하지 않습니다."
)


def validate_integrated_event_plan(plan):
    if not isinstance(plan, dict):
        raise ValueError("integrated incident policy requires a plan object")
    after = plan.get("after_response")
    if after not in (None, "RETURN_HOME"):
        raise ValueError("사건 뒤 동작은 RETURN_HOME이어야 합니다. 순찰 재개 계획은 실행할 수 없습니다.")
    monitor = plan.get("monitor_events", [])
    if not isinstance(monitor, list) or any(not isinstance(v, str) for v in monitor):
        raise ValueError("monitor_events must be a string array")
    if len(set(monitor)) != len(monitor):
        raise ValueError("monitor_events must not contain duplicates")
    if monitor not in ([], ["ALL"]):
        resolved = {EVENT_ALIASES.get(v, v) for v in monitor}
        if resolved != SUPPORTED_EVENTS:
            raise ValueError("통합 시험은 실제 Phase1 지원 6종 전체를 감시합니다. ALL로 요청하세요. FALL은 지원하지 않습니다.")
    rules = plan.get("response_rules", {})
    if not isinstance(rules, dict):
        raise ValueError("response_rules must be an object")
    for name, actions in rules.items():
        if name != "DEFAULT" and EVENT_ALIASES.get(name, name) not in SUPPORTED_EVENTS:
            raise ValueError("지원하지 않는 사건 대응 규칙입니다: " + str(name))
        if (not isinstance(actions, list) or not actions
                or any(not isinstance(v, str) for v in actions)
                or len(set(actions)) != len(actions)
                or not set(actions) <= {"RECORD", "ALERT"} or "RECORD" not in actions):
            raise ValueError("사건 대응은 RECORD(선택 ALERT)를 포함해야 합니다. TRACK은 지원하지 않습니다.")

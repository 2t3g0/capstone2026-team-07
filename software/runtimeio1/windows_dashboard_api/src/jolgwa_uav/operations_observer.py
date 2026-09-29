from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from collections import Counter, defaultdict, deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .feedback import SpeechFeedback


_STOP_LOG_WRITER = object()


PHASE_NAMES = {
    0: "IDLE",
    1: "AWAITING_APPROVAL",
    2: "TAKEOFF",
    3: "PATROL",
    4: "RETURNING_HOME",
    5: "PAUSED_MANUAL",
    6: "COMPLETED",
    7: "ABORTED",
    8: "ERROR",
    9: "MOVING_TO_START",
    10: "PAUSED_EVENT",
    11: "EVENT_CAPTURE",
    12: "REJOINING_ROUTE",
    13: "HOLD_RESUME_REQUIRED",
}


class Speaker(Protocol):
    def speak(self, message: str) -> bool: ...

    def close(self, *, wait: bool = True, timeout: float | None = 2.0) -> None: ...


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() not in {"0", "false", "no", "off"}


def default_log_path(project_root: Path) -> Path:
    configured = os.getenv("JOLGWA_OPERATIONS_LOG")
    if configured:
        return Path(configured).expanduser()
    stamp = datetime.now(UTC).astimezone().strftime("%Y%m%d")
    return project_root / "outputs" / "operations" / f"operations-{stamp}.jsonl"


class OperationsObserver:
    """Thread-safe flight-operations event logger and transition-driven TTS."""

    def __init__(
        self,
        log_path: Path,
        *,
        tts_enabled: bool = True,
        speaker: Speaker | None = None,
        speaker_factory: Callable[[], Speaker] = SpeechFeedback,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.log_path = Path(log_path)
        self.tts_enabled = tts_enabled
        self._speaker = speaker
        self._speaker_factory = speaker_factory
        self._clock = clock
        self._wall_clock = wall_clock or (lambda: datetime.now(UTC))
        self._lock = threading.RLock()
        self._started_at = clock()
        self._counts: Counter[str] = Counter()
        self._durations: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=1_000)
        )
        self._last_event: dict[str, Any] | None = None
        self._last_error: str | None = None
        self._last_phase_by_mission: dict[str, tuple[str, float]] = {}
        self._proposal_started_at: dict[str, float] = {}
        self._approval_started_at: dict[str, float] = {}
        self._last_approved_mission_id: str | None = None
        self._auto_started_missions: set[str] = set()
        self._mission_control_owner: str | None = None
        self._vehicle_authority: str | None = None
        self._effective_authority: str | None = None
        self._authority_entered_at = self._started_at
        self._operator_owner_present = False
        self._log_queue: queue.Queue[str | object] = queue.Queue()
        self._log_closed = False
        self._log_worker_thread = threading.Thread(
            target=self._log_worker,
            name="jolgwa-operations-jsonl",
            daemon=True,
        )
        self._log_worker_thread.start()

    @classmethod
    def from_environment(cls, project_root: Path) -> OperationsObserver:
        return cls(
            default_log_path(project_root),
            tts_enabled=_env_bool("JOLGWA_TTS_ENABLED", True),
        )

    def new_request_id(self) -> str:
        return str(uuid.uuid4())

    def record(self, event: str, *, source: str, **fields: Any) -> dict[str, Any]:
        now = self._clock()
        timestamp = self._wall_clock()
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        entry = {
            "timestamp": timestamp.astimezone(UTC).isoformat(timespec="milliseconds"),
            "monotonic_s": round(now - self._started_at, 6),
            "event": event,
            "source": source,
            **fields,
        }
        line = json.dumps(entry, ensure_ascii=False, separators=(",", ":"), default=str)
        with self._lock:
            self._counts[event] += 1
            self._last_event = entry
            duration = fields.get("duration_ms")
            if isinstance(duration, (int, float)):
                self._durations[event].append(float(duration))
            if not self._log_closed:
                self._log_queue.put_nowait(line)
        return entry

    def planning_started(
        self,
        request_id: str,
        *,
        input_mode: str,
        command: str | None = None,
        audio_bytes: int | None = None,
        mime_type: str | None = None,
    ) -> None:
        fields: dict[str, Any] = {"request_id": request_id, "input_mode": input_mode}
        if command is not None:
            fields["command"] = command[:4_000]
        if audio_bytes is not None:
            fields["audio_bytes"] = audio_bytes
        if mime_type is not None:
            fields["mime_type"] = mime_type
        self.record("planning.requested", source="planner_api", **fields)

    def planning_completed(
        self,
        request_id: str,
        plan: Any,
        *,
        llm_duration_ms: float,
        route_duration_ms: float,
        total_duration_ms: float,
    ) -> None:
        purpose = getattr(getattr(plan, "request_purpose", None), "value", None)
        status = getattr(getattr(plan, "status", None), "value", None)
        route_id = getattr(plan, "route_id", None)
        waypoints = getattr(plan, "route_waypoints_enu", ())
        common = {
            "request_id": request_id,
            "purpose": purpose,
            "status": status,
            "route_id": route_id,
            "waypoint_count": len(waypoints),
        }
        self.record(
            "llm.completed",
            source="gemini",
            duration_ms=round(llm_duration_ms, 3),
            **common,
        )
        self.record(
            "route_resolution.completed",
            source="planner_api",
            duration_ms=round(route_duration_ms, 3),
            **common,
        )
        self.record(
            "planning.completed",
            source="planner_api",
            duration_ms=round(total_duration_ms, 3),
            llm_duration_ms=round(llm_duration_ms, 3),
            route_duration_ms=round(route_duration_ms, 3),
            **common,
        )
        if purpose == "CREATE_ROUTE" and status == "OK":
            self.announce(
                "경로 생성이 완료되었습니다. 편집 화면에서 확인해 주세요.",
                reason="route_draft_completed",
            )

    def planning_failed(
        self,
        request_id: str,
        *,
        stage: str,
        duration_ms: float,
        error: Exception,
    ) -> None:
        self.record(
            "planning.failed",
            source="planner_api",
            request_id=request_id,
            stage=stage,
            duration_ms=round(duration_ms, 3),
            error_type=type(error).__name__,
            error=str(error)[:1_024],
        )
        self.announce("임무 계획을 만들지 못했습니다. 오류 내용을 확인해 주세요.", reason="planning_failed")

    def observe_operator(
        self, message: dict[str, Any], *, session_id: str | None = None
    ) -> None:
        message_type = str(message.get("type", "unknown"))
        if message_type in {"heartbeat", "manual.velocity"}:
            return
        fields: dict[str, Any] = {
            "message_type": message_type,
            "session_id": session_id,
        }
        for key in (
            "proposal_id",
            "mission_id",
            "operator_id",
            "approved",
            "active",
            "reason",
        ):
            if key in message:
                fields[key] = message[key]
        if "source" in message:
            fields["operator_source"] = message["source"]
        if message_type == "command.text":
            fields["command"] = str(message.get("command", ""))[:4_000]
        elif message_type == "mission.propose":
            fields["command"] = str(message.get("raw_command", ""))[:4_000]
            plan = message.get("plan")
            if isinstance(plan, dict):
                fields["route_id"] = plan.get("route_id")
                fields["waypoint_count"] = len(plan.get("route_waypoints_enu") or [])
        self.record("operator.message", source="operator_websocket", **fields)

    def operator_control_changed(
        self,
        owned: bool,
        *,
        reason: str,
        session_id: str | None = None,
        previous_session_id: str | None = None,
    ) -> None:
        with self._lock:
            if self._operator_owner_present == owned and reason != "control_transferred":
                return
            previous_owned = self._operator_owner_present
            self._operator_owner_present = owned
        self.record(
            "operator.control_changed",
            source="operator_hub",
            owned=owned,
            previous_owned=previous_owned,
            reason=reason,
            session_id=session_id,
            previous_session_id=previous_session_id,
        )

    def gateway_changed(self, connected: bool) -> None:
        self.record(
            "ros_gateway.connected" if connected else "ros_gateway.disconnected",
            source="operator_hub",
        )

    def observe_ros(self, message: dict[str, Any]) -> None:
        message_type = message.get("type")
        if message_type == "mission.proposal":
            self._on_proposal(message)
        elif message_type == "mission.approval_result":
            self._on_approval_result(message)
        elif message_type == "mission.execution_result":
            self._on_execution_result(message)
        elif message_type == "mission.status":
            self._on_mission_status(message)
        elif message_type == "vehicle.state":
            self._on_vehicle_state(message)
        elif message_type == "mission.resume_result":
            self.record(
                "mission.resume_result",
                source="ros_gateway",
                accepted=message.get("accepted"),
                mission_id=message.get("mission_id"),
                detail=str(message.get("message", ""))[:1_024],
            )
        elif message_type == "error":
            self.record(
                "runtime.error",
                source="ros_gateway",
                code=message.get("code"),
                detail=str(message.get("message", ""))[:1_024],
            )
            self.announce("드론 관제 오류가 발생했습니다. 화면을 확인해 주세요.", reason="runtime_error")

    def announce(self, message: str, *, reason: str) -> bool:
        if not self.tts_enabled:
            return False
        try:
            with self._lock:
                if self._speaker is None:
                    self._speaker = self._speaker_factory()
                speaker = self._speaker
            accepted = speaker.speak(message)
            self.record(
                "tts.queued" if accepted else "tts.suppressed",
                source="operations_observer",
                reason=reason,
                message=message,
            )
            return accepted
        except Exception as exc:
            with self._lock:
                self._last_error = f"TTS failed: {exc}"
                self.tts_enabled = False
            self.record(
                "tts.failed",
                source="operations_observer",
                reason=reason,
                error_type=type(exc).__name__,
                error=str(exc)[:1_024],
            )
            return False

    def summary(self) -> dict[str, Any]:
        with self._lock:
            timing = {
                name: {
                    "count": len(values),
                    "last_ms": round(values[-1], 3),
                    "average_ms": round(sum(values) / len(values), 3),
                }
                for name, values in self._durations.items()
                if values
            }
            backend = getattr(getattr(self._speaker, "backend", None), "name", None)
            return {
                "log_path": str(self.log_path.resolve()),
                "tts_enabled": self.tts_enabled,
                "tts_backend": backend,
                "event_counts": dict(self._counts),
                "timings": timing,
                "effective_authority": self._effective_authority,
                "operator_control_owned": self._operator_owner_present,
                "active_missions": {
                    mission_id: phase
                    for mission_id, (phase, _) in self._last_phase_by_mission.items()
                },
                "last_event": self._last_event,
                "last_error": self._last_error,
                "pending_log_entries": self._log_queue.unfinished_tasks,
            }

    def flush(self) -> None:
        self._log_queue.join()

    def close(self) -> None:
        with self._lock:
            speaker = self._speaker
            self._speaker = None
            if not self._log_closed:
                self._log_closed = True
                self._log_queue.put_nowait(_STOP_LOG_WRITER)
                close_log = True
            else:
                close_log = False
        if close_log:
            self._log_worker_thread.join(2.0)
        if speaker is not None:
            try:
                speaker.close()
            except Exception:
                pass

    def _log_worker(self) -> None:
        stream = None
        try:
            while True:
                item = self._log_queue.get()
                try:
                    if item is _STOP_LOG_WRITER:
                        return
                    assert isinstance(item, str)
                    if stream is None:
                        self.log_path.parent.mkdir(parents=True, exist_ok=True)
                        stream = self.log_path.open(
                            "a", encoding="utf-8", newline="\n", buffering=1
                        )
                    stream.write(item + "\n")
                    stream.flush()
                except OSError as exc:
                    with self._lock:
                        self._last_error = f"log write failed: {exc}"
                    if stream is not None:
                        stream.close()
                        stream = None
                finally:
                    self._log_queue.task_done()
        finally:
            if stream is not None:
                stream.close()

    def _on_proposal(self, message: dict[str, Any]) -> None:
        proposal_id = str(message.get("proposal_id", ""))
        now = self._clock()
        with self._lock:
            if proposal_id:
                self._proposal_started_at[proposal_id] = now
        self.record(
            "mission.proposal_ready",
            source="ros_gateway",
            proposal_id=proposal_id or None,
            command_id=message.get("command_id"),
            command=str(message.get("raw_command", ""))[:4_000],
            requires_approval=message.get("requires_approval"),
            status=message.get("status"),
        )
        if message.get("requires_approval"):
            self.announce(
                "임무 계획이 완성되었습니다. 검토 후 승인해 주세요.",
                reason="mission_proposal_ready",
            )

    def _on_approval_result(self, message: dict[str, Any]) -> None:
        now = self._clock()
        proposal_id = str(message.get("proposal_id", ""))
        mission_id = str(message.get("mission_id", ""))
        with self._lock:
            proposed_at = self._proposal_started_at.get(proposal_id)
            if message.get("accepted") and mission_id:
                self._approval_started_at[mission_id] = now
                self._last_approved_mission_id = mission_id
        duration = (now - proposed_at) * 1_000 if proposed_at is not None else None
        fields: dict[str, Any] = {
            "proposal_id": proposal_id or None,
            "mission_id": mission_id or None,
            "accepted": bool(message.get("accepted")),
            "detail": str(message.get("message", ""))[:1_024],
        }
        if duration is not None:
            fields["duration_ms"] = round(duration, 3)
        self.record("mission.approval_result", source="ros_gateway", **fields)
        if not message.get("accepted"):
            self.announce("임무 계획이 거절되었습니다.", reason="mission_rejected")

    def _on_execution_result(self, message: dict[str, Any]) -> None:
        mission_id = str(message.get("mission_id", "")) or self._last_approved_mission_id
        now = self._clock()
        duration = None
        if mission_id:
            with self._lock:
                started_at = self._approval_started_at.get(mission_id)
            if started_at is not None:
                duration = (now - started_at) * 1_000
        fields: dict[str, Any] = {
            "mission_id": mission_id,
            "proposal_id": message.get("proposal_id"),
            "success": bool(message.get("success")),
            "final_phase": message.get("final_phase"),
            "detail": str(message.get("message", ""))[:1_024],
        }
        if duration is not None:
            fields["duration_ms"] = round(duration, 3)
        self.record("mission.execution_result", source="ros_gateway", **fields)

    def _on_mission_status(self, message: dict[str, Any]) -> None:
        now = self._clock()
        mission_id = str(message.get("mission_id", "")) or "unknown"
        raw_phase = message.get("phase")
        phase = PHASE_NAMES.get(raw_phase, str(raw_phase))
        with self._lock:
            previous = self._last_phase_by_mission.get(mission_id)
            changed = previous is None or previous[0] != phase
            if changed:
                self._last_phase_by_mission[mission_id] = (phase, now)
            if "control_owner" in message:
                self._mission_control_owner = str(message.get("control_owner") or "NONE")
        if changed:
            fields: dict[str, Any] = {
                "mission_id": mission_id,
                "proposal_id": message.get("proposal_id"),
                "phase": phase,
                "previous_phase": previous[0] if previous else None,
                "current_waypoint": message.get("current_waypoint"),
                "total_waypoints": message.get("total_waypoints"),
                "detail": str(message.get("detail", ""))[:1_024],
            }
            if previous is not None:
                phase_duration_ms = round((now - previous[1]) * 1_000, 3)
                fields["duration_ms"] = phase_duration_ms
                fields["previous_phase_duration_ms"] = phase_duration_ms
            self.record("mission.phase_changed", source="ros_gateway", **fields)
            self._announce_phase(
                mission_id,
                phase,
                previous[0] if previous else None,
                str(message.get("detail", "")),
            )
        self._refresh_effective_authority(mission_id)

    def _on_vehicle_state(self, message: dict[str, Any]) -> None:
        if "active_authority" in message:
            with self._lock:
                self._vehicle_authority = str(message.get("active_authority") or "NONE")
        self._refresh_effective_authority(str(message.get("mission_id", "")) or None)

    def _refresh_effective_authority(self, mission_id: str | None) -> None:
        now = self._clock()
        with self._lock:
            candidates = {self._mission_control_owner, self._vehicle_authority}
            if "HUMAN" in candidates:
                authority = "HUMAN"
            elif "JETSON_EVENT_CAPTURE" in candidates:
                authority = "JETSON_EVENT_CAPTURE"
            elif "JETSON_SAFETY" in candidates:
                authority = "JETSON_SAFETY"
            elif "LLM_ROUTE" in candidates:
                authority = "LLM_ROUTE"
            else:
                authority = "NONE"
            previous = self._effective_authority
            if previous == authority:
                return
            previous_duration = (now - self._authority_entered_at) * 1_000
            self._effective_authority = authority
            self._authority_entered_at = now
        fields: dict[str, Any] = {
            "mission_id": mission_id,
            "previous_authority": previous,
            "authority": authority,
        }
        if previous is not None:
            authority_duration_ms = round(previous_duration, 3)
            fields["duration_ms"] = authority_duration_ms
            fields["previous_authority_duration_ms"] = authority_duration_ms
        self.record("control.authority_changed", source="ros_gateway", **fields)
        if authority == "HUMAN":
            self.announce("수동 조작으로 제어권을 전환했습니다.", reason="authority_human")
        elif authority == "JETSON_EVENT_CAPTURE":
            self.announce(
                "젯슨이 사건 대응을 위해 제어권을 인수했습니다.",
                reason="authority_jetson_event",
            )
        elif authority == "JETSON_SAFETY":
            self.announce(
                "젯슨 안전 제어가 장애물 회피를 위해 개입했습니다.",
                reason="authority_jetson_safety",
            )
        elif authority == "LLM_ROUTE" and previous in {
            "HUMAN",
            "JETSON_EVENT_CAPTURE",
            "JETSON_SAFETY",
        }:
            self.announce("자동 경로 제어로 복귀합니다.", reason="authority_llm_route")

    def _announce_phase(
        self, mission_id: str, phase: str, previous: str | None, detail: str
    ) -> None:
        if phase in {"TAKEOFF", "PATROL", "MOVING_TO_START"}:
            with self._lock:
                first_start = mission_id not in self._auto_started_missions
                self._auto_started_missions.add(mission_id)
            if first_start:
                self.announce("자동 비행을 시작합니다.", reason="mission_started")
        if phase == "RETURNING_HOME" and previous not in {None, "RETURNING_HOME"}:
            if "patrol complete" in detail.casefold():
                message = "순찰 경로를 완료했습니다. 복귀를 시작합니다."
                reason = "route_completed"
            else:
                message = "복귀를 시작합니다."
                reason = "returning_home"
            self.announce(message, reason=reason)
        elif phase == "COMPLETED":
            self.announce("임무를 완료했습니다.", reason="mission_completed")
        elif phase == "ABORTED":
            self.announce("임무가 중단되었습니다.", reason="mission_aborted")
        elif phase == "ERROR":
            self.announce("임무 오류가 발생했습니다. 화면을 확인해 주세요.", reason="mission_error")
        elif phase == "HOLD_RESUME_REQUIRED":
            self.announce(
                "드론이 제자리 비행 중입니다. 자동 비행 복귀 승인이 필요합니다.",
                reason="resume_required",
            )

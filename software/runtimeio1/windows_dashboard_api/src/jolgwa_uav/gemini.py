from __future__ import annotations

import base64
import json
import os
import time
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import ValidationError

from .models import MissionPlan, PlanningContext
from .prompt import SYSTEM_PROMPT, build_user_prompt


class PlannerError(RuntimeError):
    pass


def _response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "request_purpose": {
                "type": "string",
                "enum": ["EXECUTE_MISSION", "CREATE_ROUTE"],
            },
            "status": {
                "type": "string",
                "enum": ["OK", "NEED_CLARIFICATION", "UNSUPPORTED"],
            },
            "patrol_zones": {
                "type": "array",
                "items": {"type": "string", "enum": ["A", "B", "C"]},
            },
            "patrol_limit": {
                "anyOf": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["LAPS", "DURATION_MINUTES"],
                            },
                            "value": {"type": "integer", "minimum": 1},
                        },
                        "required": ["type", "value"],
                    },
                    {"type": "null"},
                ]
            },
            "route_id": {"type": ["string", "null"]},
            "landmark_route": {
                "type": "array",
                "maxItems": 12,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "landmark_id": {"type": "string"},
                        "action": {
                            "type": "string",
                            "enum": ["PASS", "ORBIT", "CAPTURE_STILL"],
                        },
                    },
                    "required": ["landmark_id", "action"],
                },
            },
            "monitor_events": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "FALL",
                        "CALL_FOR_HELP",
                        "VIOLENCE",
                        "RESTRICTED_ENTRY",
                        "FIRE",
                        "TRAFFIC_ACCIDENT",
                        "ALL",
                    ],
                },
            },
            "response_rules": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["TRACK", "RECORD", "ALERT"],
                    },
                },
            },
            "after_response": {
                "type": ["string", "null"],
                "enum": ["RETURN_HOME", "RESUME_PATROL", None],
            },
            "missing_fields": {"type": "array", "items": {"type": "string"}},
            "unsupported_values": {
                "type": "array",
                "items": {"type": "string"},
            },
            "message": {"type": ["string", "null"]},
        },
        "required": [
            "request_purpose",
            "status",
            "patrol_zones",
            "patrol_limit",
            "route_id",
            "landmark_route",
            "monitor_events",
            "response_rules",
            "after_response",
            "missing_fields",
            "unsupported_values",
            "message",
        ],
    }


class GeminiPlanner:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3.7-flash",
        thinking_level: str = "medium",
        timeout_s: float = 45.0,
        client_factory: Callable[[], httpx.Client] | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise PlannerError("GEMINI_API_KEY is not configured")
        if thinking_level not in {"low", "medium", "high"}:
            raise ValueError("thinking_level must be low, medium, or high")
        self.model = model
        self.thinking_level = thinking_level
        self.timeout_s = timeout_s
        self._client_factory = client_factory or (
            lambda: httpx.Client(timeout=self.timeout_s)
        )

    @property
    def endpoint(self) -> str:
        return (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )

    def plan_text(
        self,
        command: str,
        context: PlanningContext | None = None,
    ) -> MissionPlan:
        command = command.strip()
        if not command:
            raise ValueError("command must not be empty")
        parts = [{"text": build_user_prompt(command, context)}]
        return self._generate(parts)

    def plan_audio(
        self,
        audio: bytes,
        mime_type: str,
        context: PlanningContext | None = None,
    ) -> MissionPlan:
        if not audio:
            raise ValueError("audio must not be empty")
        if len(audio) > 18 * 1024 * 1024:
            raise ValueError("inline audio is limited to 18 MiB by this service")
        instruction = (
            "Transcribe the operator audio accurately, then create the mission JSON.\n"
            + build_user_prompt("<spoken command in attached audio>", context)
        )
        parts = [
            {"text": instruction},
            {
                "inlineData": {
                    "mimeType": mime_type,
                    "data": base64.b64encode(audio).decode("ascii"),
                }
            },
        ]
        return self._generate(parts)

    def _generate(self, parts: list[dict[str, Any]]) -> MissionPlan:
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": _response_schema(),
                "maxOutputTokens": 8192,
                "thinkingConfig": {"thinkingLevel": self.thinking_level},
            },
        }
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                with self._client_factory() as client:
                    response = client.post(
                        self.endpoint,
                        headers={"x-goog-api-key": self.api_key},
                        json=payload,
                    )
                response.raise_for_status()
                body = response.json()
                text = self._extract_text(body)
                candidate = json.loads(text)
                if (
                    candidate.get("status") == "OK"
                    and candidate.get("after_response") is None
                ):
                    candidate["after_response"] = "RETURN_HOME"
                return MissionPlan.model_validate(candidate)
            except (
                httpx.HTTPError,
                KeyError,
                ValueError,
                ValidationError,
                PlannerError,
            ) as exc:
                last_error = exc
                retryable_http = isinstance(exc, httpx.HTTPStatusError) and (
                    exc.response.status_code == 429
                    or exc.response.status_code >= 500
                )
                retryable_output = isinstance(
                    exc, (json.JSONDecodeError, ValidationError, KeyError, PlannerError)
                )
                retryable = retryable_http or retryable_output
                if not retryable or attempt == 1:
                    break
                if retryable_output:
                    payload["contents"][0]["parts"].append(
                        {
                            "text": (
                                "The previous response was incomplete or invalid. "
                                "Regenerate one complete JSON object that strictly "
                                "matches the response schema."
                            )
                        }
                    )
                time.sleep(0.5 * (attempt + 1))
        raise PlannerError(f"Gemini mission planning failed: {last_error}") from last_error

    @staticmethod
    def _extract_text(body: dict[str, Any]) -> str:
        parts = body["candidates"][0]["content"]["parts"]
        for part in parts:
            if "text" in part and not part.get("thought", False):
                return str(part["text"])
        raise PlannerError("Gemini returned no mission JSON")

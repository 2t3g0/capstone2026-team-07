from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import httpx


class RemoteComputeError(RuntimeError):
    """Raised when Jetson cannot provide a valid safety decision."""


@dataclass(frozen=True, slots=True)
class RemoteSafetyDecision:
    state: str
    direction: str
    vertical_velocity_ned_mps: float
    front_distance_m: float | None
    upper_clearance_m: float | None
    lower_clearance_m: float | None
    reason: str
    inference_ms: float | None = None
    lateral_velocity_body_mps: float = 0.0
    left_clearance_m: float | None = None
    right_clearance_m: float | None = None

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "RemoteSafetyDecision":
        state = str(value.get("state", "")).upper()
        direction = str(value.get("direction", "")).upper()
        if state not in {"CLEAR", "HOLD", "EVADE", "STALE"}:
            raise RemoteComputeError(
                f"invalid safety state from Jetson: {state!r}"
            )
        if direction not in {
            "FORWARD",
            "LEFT",
            "RIGHT",
            "UP",
            "DOWN",
            "STOP",
        }:
            raise RemoteComputeError(
                f"invalid avoidance direction from Jetson: {direction!r}"
            )
        try:
            vertical = float(value.get("vertical_velocity_ned_mps", 0.0))
            lateral = float(value.get("lateral_velocity_body_mps", 0.0))
        except (TypeError, ValueError) as exc:
            raise RemoteComputeError(
                "invalid avoidance velocity from Jetson"
            ) from exc
        if not math.isfinite(vertical) or not math.isfinite(lateral):
            raise RemoteComputeError(
                "avoidance velocity from Jetson must be finite"
            )
        if direction == "UP" and vertical >= 0.0:
            raise RemoteComputeError(
                "UP decision must have negative NED Z velocity"
            )
        if direction == "DOWN" and vertical <= 0.0:
            raise RemoteComputeError(
                "DOWN decision must have positive NED Z velocity"
            )
        if direction in {"FORWARD", "LEFT", "RIGHT", "STOP"} and abs(
            vertical
        ) > 1e-6:
            raise RemoteComputeError(
                f"{direction} decision must not include vertical velocity"
            )
        if direction == "LEFT" and lateral >= 0.0:
            raise RemoteComputeError(
                "LEFT decision must have negative body Y velocity"
            )
        if direction == "RIGHT" and lateral <= 0.0:
            raise RemoteComputeError(
                "RIGHT decision must have positive body Y velocity"
            )
        if direction in {"FORWARD", "UP", "DOWN", "STOP"} and abs(
            lateral
        ) > 1e-6:
            raise RemoteComputeError(
                f"{direction} decision must not include lateral velocity"
            )
        return cls(
            state=state,
            direction=direction,
            vertical_velocity_ned_mps=vertical,
            front_distance_m=_optional_float(value.get("front_distance_m")),
            upper_clearance_m=_optional_float(value.get("upper_clearance_m")),
            lower_clearance_m=_optional_float(value.get("lower_clearance_m")),
            reason=str(value.get("reason", "")),
            inference_ms=_optional_float(value.get("inference_ms")),
            lateral_velocity_body_mps=lateral,
            left_clearance_m=_optional_float(value.get("left_clearance_m")),
            right_clearance_m=_optional_float(
                value.get("right_clearance_m")
            ),
        )


@dataclass(frozen=True, slots=True)
class RemotePerceptionDecision:
    safety: RemoteSafetyDecision
    events: tuple[dict[str, Any], ...]
    phase1_inference_ms: float | None = None
    detections: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "RemotePerceptionDecision":
        safety = value.get("safety")
        events = value.get("events")
        phase1 = value.get("phase1", {})
        if not isinstance(safety, dict):
            raise RemoteComputeError(
                "Jetson perception response has no safety object"
            )
        if not isinstance(events, list) or not all(
            isinstance(event, dict) for event in events
        ):
            raise RemoteComputeError("Jetson perception events were invalid")
        if not isinstance(phase1, dict):
            raise RemoteComputeError("Jetson Phase 1 status was invalid")
        detections = phase1.get("detections", [])
        if not isinstance(detections, list) or not all(
            isinstance(item, dict) for item in detections
        ):
            raise RemoteComputeError("Jetson Phase 1 detections were invalid")
        return cls(
            safety=RemoteSafetyDecision.from_json(safety),
            events=tuple(dict(event) for event in events),
            phase1_inference_ms=_optional_float(phase1.get("inference_ms")),
            detections=tuple(dict(item) for item in detections),
        )


class JetsonComputeClient:
    """Fail-closed client for RGB inference and event policy on Jetson."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 2.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must not be empty")
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout_s)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def health(self) -> dict[str, Any]:
        try:
            response = self._client.get(f"{self.base_url}/health")
            response.raise_for_status()
            value = response.json()
        except Exception as exc:
            raise RemoteComputeError(
                f"Jetson health check failed: {exc}"
            ) from exc
        if not isinstance(value, dict) or value.get("status") != "ok":
            raise RemoteComputeError("Jetson returned an unhealthy response")
        return value

    def infer_rgb(
        self,
        jpeg: bytes,
        *,
        focal_px: float,
        current_z_ned_m: float,
        frame_timestamp_ns: int,
        gimbal_forward: bool = True,
    ) -> RemoteSafetyDecision:
        if not jpeg:
            raise ValueError("jpeg must not be empty")
        params = {
            "focal_px": focal_px,
            "current_z_ned_m": current_z_ned_m,
            "frame_timestamp_ns": frame_timestamp_ns,
            "gimbal_forward": str(gimbal_forward).lower(),
            "sent_monotonic_s": time.monotonic(),
        }
        try:
            response = self._client.post(
                f"{self.base_url}/v1/depth-decision",
                params=params,
                content=jpeg,
                headers={"content-type": "image/jpeg"},
            )
            response.raise_for_status()
            value = response.json()
        except Exception as exc:
            raise RemoteComputeError(
                f"Jetson depth request failed: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise RemoteComputeError("Jetson depth response was not an object")
        return RemoteSafetyDecision.from_json(value)

    def infer_perception(
        self,
        jpeg: bytes,
        *,
        focal_px: float,
        current_z_ned_m: float,
        frame_timestamp_ns: int,
        gimbal_forward: bool = True,
    ) -> RemotePerceptionDecision:
        """Run DA3 and all Phase 1 models on the same Jetson frame."""

        if not jpeg:
            raise ValueError("jpeg must not be empty")
        params = {
            "focal_px": focal_px,
            "current_z_ned_m": current_z_ned_m,
            "frame_timestamp_ns": frame_timestamp_ns,
            "gimbal_forward": str(gimbal_forward).lower(),
            "sent_monotonic_s": time.monotonic(),
        }
        try:
            response = self._client.post(
                f"{self.base_url}/v1/perception-decision",
                params=params,
                content=jpeg,
                headers={"content-type": "image/jpeg"},
            )
            response.raise_for_status()
            value = response.json()
        except Exception as exc:
            raise RemoteComputeError(
                f"Jetson perception request failed: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise RemoteComputeError(
                "Jetson perception response was not an object"
            )
        return RemotePerceptionDecision.from_json(value)

    def approve_event(self, event: dict[str, Any]) -> bool:
        """Ask Jetson to apply the automatic-response event gate."""

        try:
            response = self._client.post(
                f"{self.base_url}/v1/event-decision", json=event
            )
            response.raise_for_status()
            value = response.json()
        except Exception as exc:
            raise RemoteComputeError(
                f"Jetson event request failed: {exc}"
            ) from exc
        if not isinstance(value, dict) or not isinstance(
            value.get("respond"), bool
        ):
            raise RemoteComputeError("Jetson event response was invalid")
        return bool(value["respond"])


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RemoteComputeError(
            f"invalid numeric field from Jetson: {value!r}"
        ) from exc

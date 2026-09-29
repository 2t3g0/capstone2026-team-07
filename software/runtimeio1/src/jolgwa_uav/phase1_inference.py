from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import time
from typing import Any

import numpy as np


PHASE1_EVENT_TYPES = frozenset(
    {
        "FIRE_SMOKE",
        "HUMAN_VIOLENCE",
        "LITTERING",
        "INTRUSION_ATTEMPT",
        "CALL_FOR_HELP",
        "VEHICLE_ACCIDENT",
    }
)


class Phase1InferenceError(RuntimeError):
    """Raised when the Phase 1 runtime cannot provide a trustworthy result."""


@dataclass(frozen=True, slots=True)
class Phase1FrameResult:
    events: tuple[dict[str, Any], ...]
    detections: tuple[dict[str, Any], ...]
    modules: dict[str, dict[str, Any]]
    inference_ms: float
    temporal_window_ready: bool = False


class Phase1JetsonRuntime:
    """Load the Phase 1 demo models and evaluate frames on the Jetson GPU.

    The original Phase 1 runtime remains the source of truth for model loading,
    temporal sampling, voting, thresholds, and event schema. This adapter only
    supplies decoded AirSim frames and returns newly confirmed events.
    """

    REQUIRED_MODELS = (
        "general_detection",
        "fire_smoke",
        "abnormal_behavior",
        "vehicle_accident",
    )

    def __init__(
        self,
        repo_root: str | Path,
        *,
        config_path: str | Path | None = None,
        device: str = "0",
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.config_path = (
            Path(config_path).expanduser().resolve()
            if config_path is not None
            else self.repo_root / "config" / "vision" / "phase1-demo.yaml"
        )
        self.device = str(device)
        self.loaded = False
        self._runtime: Any | None = None
        self._models: dict[str, Any] = {}
        self._model_hashes: dict[str, str] = {}
        self._event_cursor = 0
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self.loaded:
                return
            service_root = self.repo_root / "services" / "vision"
            if not service_root.is_dir():
                raise Phase1InferenceError(
                    f"Phase 1 service source is missing: {service_root}"
                )
            if not self.config_path.is_file():
                raise Phase1InferenceError(
                    f"Phase 1 config is missing: {self.config_path}"
                )
            service_text = str(service_root)
            if service_text not in sys.path:
                sys.path.insert(0, service_text)
            try:
                from vectorforce_vision.demo.backends import (
                    DetectionBackend,
                    TemporalBackend,
                    load_demo_specs,
                )
                from vectorforce_vision.demo.runtime import (
                    DemoThresholds,
                    Phase1DemoRuntime,
                )
            except Exception as exc:
                raise Phase1InferenceError(
                    f"Phase 1 runtime import failed: {exc}"
                ) from exc

            try:
                specs = load_demo_specs(
                    self.config_path, repo_root=self.repo_root
                )
                missing = [name for name in self.REQUIRED_MODELS if name not in specs]
                if missing:
                    raise Phase1InferenceError(
                        f"Phase 1 config is missing models: {', '.join(missing)}"
                    )
                models: dict[str, Any] = {}
                for name in self.REQUIRED_MODELS:
                    spec = specs[name]
                    if name in {"general_detection", "fire_smoke"}:
                        model = DetectionBackend(
                            spec,
                            device=self.device,
                            track=name == "general_detection",
                        )
                    else:
                        model = TemporalBackend(spec, device=self.device)
                    if not bool(getattr(model, "healthy", False)):
                        raise Phase1InferenceError(
                            f"Phase 1 model {name} is unhealthy: "
                            f"{getattr(model, 'error', 'unknown error')}"
                        )
                    models[name] = model

                hashes = {
                    name: str(getattr(model, "actual_sha256", ""))
                    for name, model in models.items()
                }
                thresholds = DemoThresholds(
                    fire_confidence=float(specs["fire_smoke"].confidence),
                    abnormal_confidence=float(
                        specs["abnormal_behavior"].confidence
                    ),
                    accident_probability=float(
                        specs["vehicle_accident"].threshold
                    ),
                )
                self._runtime = Phase1DemoRuntime(
                    general_backend=models["general_detection"],
                    fire_backend=models["fire_smoke"],
                    abnormal_backend=models["abnormal_behavior"],
                    vehicle_backend=models["vehicle_accident"],
                    thresholds=thresholds,
                    model_hashes=hashes,
                    warning_models=tuple(
                        name
                        for name, spec in specs.items()
                        if str(spec.status).upper() != "PASS"
                    ),
                    general_hz=float(specs["general_detection"].rate_hz),
                    fire_hz=float(specs["fire_smoke"].rate_hz),
                    temporal_hz=min(
                        float(specs["abnormal_behavior"].rate_hz),
                        float(specs["vehicle_accident"].rate_hz),
                    ),
                )
            except Phase1InferenceError:
                raise
            except Exception as exc:
                raise Phase1InferenceError(
                    f"Phase 1 model load failed: {exc}"
                ) from exc

            self._models = models
            self._model_hashes = hashes
            self._event_cursor = 0
            self.loaded = True

    def warmup(self) -> None:
        self.load()
        image = np.zeros((360, 640, 3), dtype=np.uint8)
        with self._lock:
            for name in ("general_detection", "fire_smoke"):
                try:
                    self._models[name].infer(image, 0.0)
                except Exception as exc:
                    self.loaded = False
                    raise Phase1InferenceError(
                        f"Phase 1 model {name} warmup failed: {exc}"
                    ) from exc

    def evaluate_bgr(
        self, image_bgr: np.ndarray, *, timestamp_s: float | None = None
    ) -> Phase1FrameResult:
        if image_bgr.dtype != np.uint8 or image_bgr.ndim != 3:
            raise ValueError("Phase 1 input must be an HWC uint8 BGR image")
        if image_bgr.shape[2] != 3 or image_bgr.size == 0:
            raise ValueError("Phase 1 input must have three non-empty channels")
        self.load()
        started = time.monotonic()
        with self._lock:
            assert self._runtime is not None
            try:
                from vectorforce_vision.demo.inputs import FramePacket

                packet = FramePacket(
                    timestamp_s=(
                        float(timestamp_s)
                        if timestamp_s is not None
                        else time.monotonic()
                    ),
                    rgb=image_bgr,
                )
                snapshot = self._runtime.process(packet)
                # Read the actual live buffer, not warmup success or historical
                # call counts. No sampling, trigger, threshold or vote changes.
                temporal_window_ready = (self._runtime._window_ready()
                                         and len(self._runtime._sample_packets()) == 12)
            except Exception as exc:
                raise Phase1InferenceError(
                    f"Phase 1 frame inference failed: {exc}"
                ) from exc
            all_events = self._runtime.events
            new_events = tuple(
                dict(event)
                for event in all_events[self._event_cursor :]
                if event.get("state") == "CONFIRMED"
                and event.get("event_type") in PHASE1_EVENT_TYPES
            )
            self._event_cursor = len(all_events)
            detections = tuple(dict(value) for value in snapshot.detections)
            modules = {
                name: {
                    "status": str(stats.status),
                    "calls": int(stats.calls),
                    "last_inference_ms": float(stats.last_inference_ms),
                    "error": str(stats.error),
                    "last_inference_age_s": (
                        packet.timestamp_s - stats._last_call_ts
                        if stats._last_call_ts is not None else None
                    ),
                    "current_frame_inference": stats._last_call_ts == packet.timestamp_s,
                    "cadence_s": 1.0 / self._runtime.rates[name],
                }
                for name, stats in snapshot.modules.items()
                if name in self.REQUIRED_MODELS
            }
        return Phase1FrameResult(
            events=new_events,
            detections=detections,
            modules=modules,
            inference_ms=(time.monotonic() - started) * 1000.0,
            temporal_window_ready=temporal_window_ready,
        )

    def health(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "device": self.device,
            "config": str(self.config_path),
            "models": {
                name: {
                    "healthy": bool(getattr(model, "healthy", False)),
                    "sha256": self._model_hashes.get(name, ""),
                }
                for name, model in self._models.items()
            },
        }

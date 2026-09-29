from __future__ import annotations

import importlib
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


DEFAULT_MODEL_ID = "depth-anything/DA3METRIC-LARGE"
CANONICAL_FOCAL_PX = 300.0


class DepthAnythingV3Error(RuntimeError):
    """Base error raised by the Depth Anything V3 metric-depth adapter."""


class DepthAnythingV3UnavailableError(DepthAnythingV3Error):
    """Raised when the optional DA3 runtime or requested accelerator is unavailable."""


class DepthAnythingV3InferenceError(DepthAnythingV3Error):
    """Raised when DA3 rejects an input or returns an invalid Prediction."""


class _DepthAnythingModel(Protocol):
    def inference(self, image: list[np.ndarray], **kwargs: Any) -> Any: ...


ModelLoader = Callable[[str, str], _DepthAnythingModel]


@dataclass(frozen=True, slots=True)
class MetricDepthPrediction:
    """One DA3METRIC-LARGE prediction converted to real-world metres."""

    depth_m: np.ndarray
    canonical_depth: np.ndarray
    confidence: np.ndarray | None
    sky_mask: np.ndarray | None
    processed_focal_px: float
    processed_size_hw: tuple[int, int]

    @property
    def valid_mask(self) -> np.ndarray:
        """Pixels that contain positive finite depth and are not classified as sky."""

        valid = np.isfinite(self.depth_m) & (self.depth_m > 0.0)
        if self.sky_mask is not None:
            valid &= ~self.sky_mask
        return valid


class DepthAnythingV3MetricEstimator:
    """Lazy, single-image adapter around the official DA3 metric checkpoint.

    ``DA3METRIC-LARGE`` predicts depth in a canonical camera space whose focal
    length is 300 pixels. The official conversion is therefore::

        depth_m = network_depth * processed_focal_px / 300

    The focal length supplied to :meth:`estimate` belongs to the original RGB
    image. This adapter scales ``fx`` and ``fy`` independently to the actual
    depth-map shape before taking their mean, matching DA3's image processor.

    The optional runtime is imported and the checkpoint is loaded only on the
    first estimate. ``model_loader`` is injectable so unit tests and offline
    callers do not need PyTorch, Hugging Face, or model weights.
    """

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        model_loader: ModelLoader | None = None,
    ) -> None:
        if not model_id:
            raise ValueError("model_id must not be empty")
        if not device:
            raise ValueError("device must not be empty")
        if process_res <= 0:
            raise ValueError("process_res must be positive")
        if process_res_method not in {
            "upper_bound_resize",
            "lower_bound_resize",
            "upper_bound_crop",
            "lower_bound_crop",
        }:
            raise ValueError(f"unsupported process_res_method: {process_res_method}")

        self.model_id = model_id
        self.device = device
        self.process_res = int(process_res)
        self.process_res_method = process_res_method
        self._model_loader = model_loader or self._load_official_model
        self._model: _DepthAnythingModel | None = None
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the official checkpoint now instead of waiting for first inference."""

        with self._lock:
            self._get_model()

    def estimate(
        self,
        rgb: np.ndarray,
        *,
        fx_px: float,
        fy_px: float | None = None,
    ) -> MetricDepthPrediction:
        """Estimate metric depth from one HWC uint8 RGB image.

        ``fx_px`` and ``fy_px`` must describe the original input resolution.
        When ``fy_px`` is omitted, square pixels are assumed.
        """

        image = self._validate_rgb(rgb)
        fx = self._validate_focal("fx_px", fx_px)
        fy = fx if fy_px is None else self._validate_focal("fy_px", fy_px)

        with self._lock:
            model = self._get_model()
            try:
                prediction = model.inference(
                    [image],
                    process_res=self.process_res,
                    process_res_method=self.process_res_method,
                )
            except DepthAnythingV3Error:
                raise
            except Exception as exc:
                raise DepthAnythingV3InferenceError(
                    f"DA3 metric inference failed: {exc}"
                ) from exc

        canonical = self._single_map(prediction, "depth", required=True)
        assert canonical is not None
        confidence = self._single_map(prediction, "conf", required=False)
        sky = self._single_map(prediction, "sky", required=False)

        if confidence is not None and confidence.shape != canonical.shape:
            raise DepthAnythingV3InferenceError(
                "Prediction.conf shape does not match Prediction.depth"
            )
        if sky is not None and sky.shape != canonical.shape:
            raise DepthAnythingV3InferenceError(
                "Prediction.sky shape does not match Prediction.depth"
            )

        input_h, input_w = image.shape[:2]
        output_h, output_w = canonical.shape
        fx_processed = fx * output_w / input_w
        fy_processed = fy * output_h / input_h
        processed_focal = (fx_processed + fy_processed) / 2.0

        canonical_f32 = np.asarray(canonical, dtype=np.float32)
        depth_m = canonical_f32 * np.float32(
            processed_focal / CANONICAL_FOCAL_PX
        )
        confidence_f32 = (
            np.asarray(confidence, dtype=np.float32) if confidence is not None else None
        )
        sky_mask = None
        if sky is not None:
            sky_array = np.asarray(sky)
            sky_mask = (
                sky_array.copy()
                if sky_array.dtype == np.bool_
                else np.asarray(sky_array >= 0.5, dtype=bool)
            )

        return MetricDepthPrediction(
            depth_m=depth_m,
            canonical_depth=canonical_f32,
            confidence=confidence_f32,
            sky_mask=sky_mask,
            processed_focal_px=processed_focal,
            processed_size_hw=(output_h, output_w),
        )

    def _get_model(self) -> _DepthAnythingModel:
        if self._model is None:
            try:
                self._model = self._model_loader(self.model_id, self.device)
            except DepthAnythingV3Error:
                raise
            except (ImportError, ModuleNotFoundError) as exc:
                raise self._unavailable_error(exc) from exc
            except Exception as exc:
                raise DepthAnythingV3UnavailableError(
                    f"Could not load {self.model_id} on {self.device}: {exc}"
                ) from exc
            if self._model is None or not callable(
                getattr(self._model, "inference", None)
            ):
                self._model = None
                raise DepthAnythingV3UnavailableError(
                    "DA3 model loader did not return an object with inference()"
                )
        return self._model

    @staticmethod
    def _load_official_model(model_id: str, device: str) -> _DepthAnythingModel:
        try:
            torch = importlib.import_module("torch")
            api = importlib.import_module("depth_anything_3.api")
        except (ImportError, ModuleNotFoundError) as exc:
            raise DepthAnythingV3MetricEstimator._unavailable_error(exc) from exc

        if device.startswith("cuda") and not bool(torch.cuda.is_available()):
            raise DepthAnythingV3UnavailableError(
                f"CUDA device '{device}' was requested but "
                "torch.cuda.is_available() is false"
            )

        try:
            model = api.DepthAnything3.from_pretrained(model_id)
            model = model.to(device=device)
            model.eval()
            return model
        except Exception as exc:
            raise DepthAnythingV3UnavailableError(
                f"Could not load official checkpoint {model_id} on {device}: {exc}"
            ) from exc

    @staticmethod
    def _unavailable_error(exc: BaseException) -> DepthAnythingV3UnavailableError:
        return DepthAnythingV3UnavailableError(
            "Depth Anything V3 optional dependencies are unavailable. Install a "
            "CUDA-enabled PyTorch build plus xformers and torchvision, then install "
            "the official ByteDance-Seed/Depth-Anything-3 package. "
            f"Original error: {exc}"
        )

    @staticmethod
    def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
        if not isinstance(rgb, np.ndarray):
            raise TypeError("rgb must be a numpy.ndarray")
        if rgb.dtype != np.uint8:
            raise ValueError("rgb must have dtype uint8")
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must have shape (height, width, 3)")
        if rgb.shape[0] == 0 or rgb.shape[1] == 0:
            raise ValueError("rgb dimensions must be non-zero")
        return rgb

    @staticmethod
    def _validate_focal(name: str, value: float) -> float:
        try:
            focal = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive finite number") from exc
        if not np.isfinite(focal) or focal <= 0.0:
            raise ValueError(f"{name} must be a positive finite number")
        return focal

    @staticmethod
    def _single_map(
        prediction: Any, name: str, *, required: bool
    ) -> np.ndarray | None:
        value = getattr(prediction, name, None)
        if value is None:
            if required:
                raise DepthAnythingV3InferenceError(
                    f"Prediction.{name} is missing"
                )
            return None
        array = np.asarray(value)
        if array.ndim != 3 or array.shape[0] != 1:
            raise DepthAnythingV3InferenceError(
                f"Prediction.{name} must have shape (1, height, width), "
                f"got {array.shape}"
            )
        if array.shape[1] == 0 or array.shape[2] == 0:
            raise DepthAnythingV3InferenceError(
                f"Prediction.{name} dimensions must be non-zero"
            )
        return array[0]

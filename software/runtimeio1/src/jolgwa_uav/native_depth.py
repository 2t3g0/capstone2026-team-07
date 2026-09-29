"""Optional C++ corridor statistics; None means use the existing NumPy path.

This only accelerates the stateless image operation. Configuration, validity
checks, state transitions and flight commands remain in vertical_avoidance.
Build explicitly with ``python native/build.py``; never compile at runtime.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .vertical_avoidance import DepthCorridorStats, VerticalAvoidanceConfig


class _RoiStats(ctypes.Structure):
    _fields_ = [
        ("valid_fraction", ctypes.c_double),
        ("near_distance_m", ctypes.c_double),
        ("median_distance_m", ctypes.c_double),
        ("obstacle_fraction", ctypes.c_double),
        ("valid_samples", ctypes.c_uint64),
        ("total_samples", ctypes.c_uint64),
    ]


@lru_cache(maxsize=4)
def _load_library(override: str) -> tuple[ctypes.CDLL | None, str]:
    if override:
        candidates = [Path(override)]
    else:
        package = Path(__file__).resolve().parent
        build = package.parents[1] / "native" / "build"
        names = ("jolgwa_depth.dll", "libjolgwa_depth.so", "libjolgwa_depth.dylib")
        candidates = [directory / name for directory in (package, build, build / "Release") for name in names]
    errors: list[str] = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            lib = ctypes.CDLL(str(path))
            lib.jolgwa_depth_abi_version.argtypes = []
            lib.jolgwa_depth_abi_version.restype = ctypes.c_uint32
            if lib.jolgwa_depth_abi_version() != 1:
                errors.append(f"{path}: incompatible ABI")
                continue
            lib.jolgwa_depth_corridors.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int32, ctypes.c_int32,
                ctypes.c_ssize_t, ctypes.c_ssize_t,
                ctypes.c_ssize_t, ctypes.c_ssize_t,
                ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
                ctypes.c_double, ctypes.c_float, ctypes.POINTER(_RoiStats),
            ]
            lib.jolgwa_depth_corridors.restype = ctypes.c_int
            return lib, str(path)
        except (OSError, AttributeError) as exc:
            errors.append(f"{path}: {exc}")
    return None, "; ".join(errors) or "native library not built"


def native_depth_status() -> dict[str, object]:
    """Expose the selected backend without promising a performance bound."""
    if os.getenv("JOLGWA_NATIVE_DEPTH", "auto").strip().lower() in {"0", "false", "off", "numpy"}:
        return {"available": False, "backend": "numpy", "detail": "disabled by JOLGWA_NATIVE_DEPTH"}
    library, detail = _load_library(os.getenv("JOLGWA_NATIVE_DEPTH_LIBRARY", ""))
    return {"available": library is not None, "backend": "cpp" if library else "numpy", "detail": detail}


def _pixel_interval(length: int, interval: tuple[float, float]) -> tuple[int, int]:
    start = min(length - 1, max(0, int(math.floor(length * interval[0]))))
    end = min(length, max(start + 1, int(math.ceil(length * interval[1]))))
    return start, end


@lru_cache(maxsize=32)
def _regions(shape: tuple[int, int], config: VerticalAvoidanceConfig) -> np.ndarray:
    height, width = shape
    roi_width = max(1, int(round(width * config.horizontal_roi_fraction)))
    x0 = max(0, (width - roi_width) // 2)
    x1 = min(width, x0 + roi_width)
    regions = [(*_pixel_interval(height, interval), x0, x1)
               for interval in (config.upper_roi, config.center_roi, config.lower_roi)]
    regions.extend((*_pixel_interval(height, config.side_vertical_roi),
                    *_pixel_interval(width, interval))
                   for interval in (config.left_roi, config.right_roi))
    values = np.asarray(regions, dtype=np.int32)
    values.flags.writeable = False
    return values


def native_corridor_stats(
    depth: np.ndarray,
    valid: np.ndarray,
    config: VerticalAvoidanceConfig,
    obstacle_distance_m: float,
) -> DepthCorridorStats | None:
    """Return equivalent corridor statistics or None for a safe Python fallback.

    Input arrays may be non-contiguous and negative-strided. The normal caller
    provides native-endian float32 depth and bool validity without a copy.
    Invalid shapes, build/load errors and native errors all retain NumPy.
    """
    if os.getenv("JOLGWA_NATIVE_DEPTH", "auto").strip().lower() in {"0", "false", "off", "numpy"}:
        return None
    library, _ = _load_library(os.getenv("JOLGWA_NATIVE_DEPTH_LIBRARY", ""))
    if library is None:
        return None
    try:
        depth = np.asarray(depth, dtype=np.float32)
        valid = np.asarray(valid, dtype=np.bool_)
        if depth.ndim != 2 or depth.size == 0 or valid.shape != depth.shape:
            return None
        if not depth.flags.aligned:
            depth = np.require(depth, dtype=np.float32, requirements=["A"])
        if max(depth.shape) > np.iinfo(np.int32).max or not math.isfinite(obstacle_distance_m):
            return None
        regions = _regions(depth.shape, config)
        output = (_RoiStats * 5)()
        result = library.jolgwa_depth_corridors(
            depth.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            valid.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            depth.shape[0], depth.shape[1], *depth.strides, *valid.strides,
            regions.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 5,
            config.near_percentile, obstacle_distance_m, output,
        )
        if result != 0:
            return None
    except (TypeError, ValueError, OverflowError, MemoryError, ctypes.ArgumentError):
        return None
    # Local import avoids a cycle when vertical_avoidance enables this backend.
    from .vertical_avoidance import DepthCorridorStats, DepthRoiStats
    return DepthCorridorStats(*(DepthRoiStats(
        item.valid_fraction, item.near_distance_m, item.median_distance_m,
        item.obstacle_fraction, item.valid_samples, item.total_samples,
    ) for item in output))

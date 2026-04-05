"""
Optional CPU image metrics via ``sdx_image_metrics`` shared library.
Provides lightweight quality stats for uint8 HWC images.
"""

from __future__ import annotations

import ctypes
from typing import Dict, Optional

import numpy as np

from sdx_native.native_tools import image_metrics_shared_library_path


class ImageMetricsLib:
    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        p = image_metrics_shared_library_path()
        if p is None:
            return
        try:
            self._lib = ctypes.CDLL(str(p))
        except OSError:
            return
        self._lib.sdx_image_mean_luma_u8.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
        ]
        self._lib.sdx_image_mean_luma_u8.restype = ctypes.c_int
        self._lib.sdx_image_clip_ratio_u8.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_ubyte,
            ctypes.c_ubyte,
            ctypes.POINTER(ctypes.c_double),
        ]
        self._lib.sdx_image_clip_ratio_u8.restype = ctypes.c_int
        self._lib.sdx_image_laplacian_var_u8.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
        ]
        self._lib.sdx_image_laplacian_var_u8.restype = ctypes.c_int
        self._lib.sdx_image_count_components_u8.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_ubyte,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.sdx_image_count_components_u8.restype = ctypes.c_int

    @property
    def available(self) -> bool:
        return self._lib is not None

    def _prep(self, hwc: np.ndarray) -> np.ndarray:
        if hwc.ndim != 3:
            raise ValueError("expected uint8 ndarray shape (H, W, C)")
        if hwc.dtype != np.uint8:
            hwc = hwc.astype(np.uint8)
        return np.ascontiguousarray(hwc)

    def stats(self, hwc: np.ndarray, *, clip_low: int = 2, clip_high: int = 253) -> Dict[str, float]:
        if not self._lib:
            raise RuntimeError("sdx_image_metrics not built")
        arr = self._prep(hwc)
        h, w, c = arr.shape
        mean = ctypes.c_double(0.0)
        ratio = ctypes.c_double(0.0)
        lap = ctypes.c_double(0.0)
        rc = self._lib.sdx_image_mean_luma_u8(arr.ctypes.data_as(ctypes.c_void_p), h, w, c, ctypes.byref(mean))
        if rc != 0:
            raise RuntimeError("sdx_image_mean_luma_u8 failed")
        rc = self._lib.sdx_image_clip_ratio_u8(
            arr.ctypes.data_as(ctypes.c_void_p),
            h,
            w,
            c,
            ctypes.c_ubyte(int(clip_low)),
            ctypes.c_ubyte(int(clip_high)),
            ctypes.byref(ratio),
        )
        if rc != 0:
            raise RuntimeError("sdx_image_clip_ratio_u8 failed")
        rc = self._lib.sdx_image_laplacian_var_u8(arr.ctypes.data_as(ctypes.c_void_p), h, w, c, ctypes.byref(lap))
        if rc != 0:
            raise RuntimeError("sdx_image_laplacian_var_u8 failed")
        return {
            "mean_luma": float(mean.value),
            "clip_ratio": float(ratio.value),
            "laplacian_var": float(lap.value),
        }

    def count_components(
        self,
        hwc: np.ndarray,
        *,
        threshold: int = 140,
        min_area: int = 16,
        max_area: int = 0,
    ) -> int:
        if not self._lib:
            raise RuntimeError("sdx_image_metrics not built")
        arr = self._prep(hwc)
        h, w, c = arr.shape
        rc = self._lib.sdx_image_count_components_u8(
            arr.ctypes.data_as(ctypes.c_void_p),
            h,
            w,
            c,
            ctypes.c_ubyte(int(threshold)),
            int(min_area),
            int(max_area),
        )
        if rc < 0:
            raise RuntimeError("sdx_image_count_components_u8 failed")
        return int(rc)


_LIB: Optional[ImageMetricsLib] = None


def get_image_metrics_lib() -> ImageMetricsLib:
    global _LIB
    if _LIB is None:
        _LIB = ImageMetricsLib()
    return _LIB


def maybe_image_stats_native(
    hwc: np.ndarray,
    *,
    clip_low: int = 2,
    clip_high: int = 253,
) -> Optional[Dict[str, float]]:
    lib = get_image_metrics_lib()
    if not lib.available:
        return None
    return lib.stats(hwc, clip_low=clip_low, clip_high=clip_high)


def maybe_count_components_native(
    hwc: np.ndarray,
    *,
    threshold: int = 140,
    min_area: int = 16,
    max_area: int = 0,
) -> Optional[int]:
    lib = get_image_metrics_lib()
    if not lib.available:
        return None
    return lib.count_components(hwc, threshold=threshold, min_area=min_area, max_area=max_area)

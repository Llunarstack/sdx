"""
Optional CUDA image luma metrics via ``sdx_cuda_image_metrics`` shared library.
"""

from __future__ import annotations

import ctypes
from typing import Dict, Optional

import numpy as np

from sdx_native.native_tools import cuda_image_metrics_shared_library_path


class CudaImageMetricsLib:
    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        p = cuda_image_metrics_shared_library_path()
        if p is None:
            return
        try:
            self._lib = ctypes.CDLL(str(p))
        except OSError:
            return
        self._lib.sdx_cuda_luma_stats_u8_host.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_ubyte,
            ctypes.c_ubyte,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self._lib.sdx_cuda_luma_stats_u8_host.restype = ctypes.c_int

    @property
    def available(self) -> bool:
        return self._lib is not None

    def stats(self, hwc: np.ndarray, *, clip_low: int = 2, clip_high: int = 253) -> Dict[str, float]:
        if not self._lib:
            raise RuntimeError("sdx_cuda_image_metrics not built")
        if hwc.ndim != 3:
            raise ValueError("expected ndarray shape (H, W, C)")
        if hwc.dtype != np.uint8:
            hwc = hwc.astype(np.uint8)
        arr = np.ascontiguousarray(hwc)
        h, w, c = arr.shape
        mean = ctypes.c_float(0.0)
        ratio = ctypes.c_float(0.0)
        rc = self._lib.sdx_cuda_luma_stats_u8_host(
            arr.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(int(h)),
            ctypes.c_int(int(w)),
            ctypes.c_int(int(c)),
            ctypes.c_ubyte(int(clip_low)),
            ctypes.c_ubyte(int(clip_high)),
            ctypes.byref(mean),
            ctypes.byref(ratio),
        )
        if rc != 0:
            raise RuntimeError(f"sdx_cuda_luma_stats_u8_host failed (code {rc})")
        return {"mean_luma": float(mean.value), "clip_ratio": float(ratio.value)}


_LIB: Optional[CudaImageMetricsLib] = None


def get_cuda_image_metrics_lib() -> CudaImageMetricsLib:
    global _LIB
    if _LIB is None:
        _LIB = CudaImageMetricsLib()
    return _LIB


def maybe_image_luma_stats_cuda(
    hwc: np.ndarray,
    *,
    clip_low: int = 2,
    clip_high: int = 253,
) -> Optional[Dict[str, float]]:
    lib = get_cuda_image_metrics_lib()
    if not lib.available:
        return None
    return lib.stats(hwc, clip_low=clip_low, clip_high=clip_high)

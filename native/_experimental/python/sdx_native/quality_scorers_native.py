"""Optional Rust ``sdx_quality_scorers`` cdylib: highlight-clip fraction on uint8 RGB."""

from __future__ import annotations

import ctypes
import functools

import numpy as np

from sdx_native.native_tools import load_shared_library


@functools.cache
def _lib() -> ctypes.CDLL | None:
    lib = load_shared_library("sdx_quality_scorers")
    if lib is not None:
        lib.sdx_quality_highlight_frac_u8.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
        ]
        lib.sdx_quality_highlight_frac_u8.restype = ctypes.c_double
    return lib


def available() -> bool:
    return _lib() is not None


def highlight_frac(rgb: np.ndarray, thr: float = 245.0) -> float | None:
    """Fraction of pixels whose luma is >= ``thr``; ``None`` if unavailable or the image is too small."""
    lib = _lib()
    if lib is None or rgb.ndim != 3 or rgb.shape[2] < 3 or min(rgb.shape[:2]) < 3:
        return None
    arr = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
    h, w = arr.shape[:2]
    v = float(lib.sdx_quality_highlight_frac_u8(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), h, w, thr))
    return v if v >= 0.0 else None

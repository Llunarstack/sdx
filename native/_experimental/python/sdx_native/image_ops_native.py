"""
Optional Rust ``sdx_image_ops`` cdylib wrapper: connected-components blob counting for count-adherence
scoring (same contract as the NumPy reference in ``scripts/tools/eval/scorers.py``).

Build: ``cd native/rust/sdx-image-ops && cargo build --release``.
"""

from __future__ import annotations

import ctypes
import functools

import numpy as np

from sdx_native.native_tools import load_shared_library


@functools.cache
def _lib() -> ctypes.CDLL | None:
    lib = load_shared_library("sdx_image_ops")
    if lib is not None:
        lib.sdx_count_blobs_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
        ]
        lib.sdx_count_blobs_f32.restype = ctypes.c_int64
    return lib


def available() -> bool:
    return _lib() is not None


def count_blobs(gray: np.ndarray, min_area_frac: float = 0.005) -> int | None:
    """Foreground blob count of a 2-D grayscale array in [0, 1]; ``None`` if unavailable or rejected."""
    lib = _lib()
    arr = np.ascontiguousarray(gray, dtype=np.float32)
    if lib is None or arr.ndim != 2:
        return None
    h, w = arr.shape
    n = lib.sdx_count_blobs_f32(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), h, w, float(min_area_frac))
    return int(n) if n >= 0 else None

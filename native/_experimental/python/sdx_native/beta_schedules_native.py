"""Optional ctypes bridge to ``sdx_beta_schedules`` for squared-cosine v2 betas; :mod:`diffusion.schedules`
falls back to NumPy when the library is missing."""

from __future__ import annotations

import ctypes
import functools

import numpy as np

from sdx_native.native_tools import load_shared_library


@functools.cache
def _lib() -> ctypes.CDLL | None:
    lib = load_shared_library("sdx_beta_schedules")
    if lib is not None:
        lib.sdx_squared_cosine_betas_v2.argtypes = [
            ctypes.c_int,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int64,
        ]
        lib.sdx_squared_cosine_betas_v2.restype = ctypes.c_int64
    return lib


def squared_cosine_betas_v2_native(n: int, max_beta: float = 0.999) -> np.ndarray | None:
    """``(n,)`` float64 betas, or ``None`` if the native library is unavailable or returns an error."""
    lib = _lib()
    if n < 1 or lib is None:
        return None
    out = np.empty(n, dtype=np.float64)
    ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    if int(lib.sdx_squared_cosine_betas_v2(int(n), float(max_beta), ptr, int(n))) != int(n):
        return None
    return out

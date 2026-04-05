"""
Optional ctypes bridge to ``sdx_beta_schedules`` for squared-cosine v2 betas.
Falls back to pure NumPy in :mod:`diffusion.schedules` when the DLL is missing.
"""

from __future__ import annotations

import ctypes
from typing import Any, Optional

import numpy as np

from sdx_native.native_tools import beta_schedules_shared_library_path

_DLL: Any = None


def _get_dll() -> Optional[ctypes.CDLL]:
    global _DLL
    if _DLL is False:
        return None
    if _DLL is not None:
        return _DLL
    p = beta_schedules_shared_library_path()
    if p is None:
        _DLL = False
        return None
    try:
        dll = ctypes.CDLL(str(p))
        dll.sdx_squared_cosine_betas_v2.argtypes = [
            ctypes.c_int,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int64,
        ]
        dll.sdx_squared_cosine_betas_v2.restype = ctypes.c_int64
        _DLL = dll
        return dll
    except OSError:
        _DLL = False
        return None


def squared_cosine_betas_v2_native(n: int, max_beta: float = 0.999) -> Optional[np.ndarray]:
    """
    Return ``(n,)`` float64 betas or ``None`` if the native library is unavailable
    or returns an error.
    """
    if n < 1:
        return None
    dll = _get_dll()
    if dll is None:
        return None
    out = np.empty(n, dtype=np.float64)
    ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ret = int(dll.sdx_squared_cosine_betas_v2(int(n), float(max_beta), ptr, ctypes.c_int64(int(n))))
    if ret != int(n):
        return None
    return out

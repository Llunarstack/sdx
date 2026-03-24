"""
Optional ctypes bridge to ``sdx_inference_timesteps`` (C++), mirroring
``diffusion.inference_timesteps._resample_length``. If the shared library is missing
or returns an error, callers should fall back to pure NumPy.
"""

from __future__ import annotations

import ctypes
from typing import Any, Optional

import numpy as np

from sdx_native.native_tools import inference_timesteps_shared_library_path

_DLL: Any = None


def _get_dll() -> Optional[ctypes.CDLL]:
    global _DLL
    if _DLL is False:
        return None
    if _DLL is not None:
        return _DLL
    p = inference_timesteps_shared_library_path()
    if p is None:
        _DLL = False
        return None
    try:
        dll = ctypes.CDLL(str(p))
        dll.sdx_it_finalize_path.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
        ]
        dll.sdx_it_finalize_path.restype = ctypes.c_int64
        _DLL = dll
        return dll
    except OSError:
        _DLL = False
        return None


def finalize_inference_timesteps_native(
    raw: np.ndarray, target_len: int, num_train: int
) -> Optional[np.ndarray]:
    """
    Match :func:`diffusion.inference_timesteps._resample_length_numpy` when the native
    library is built and accepts the inputs. Returns ``None`` if unavailable or on error.
    """
    if target_len <= 0 or num_train < 1:
        return None
    dll = _get_dll()
    if dll is None:
        return None
    raw_i = np.asarray(raw, dtype=np.int64).ravel()
    out_cap = max(int(target_len), 2) + 32
    out = np.empty(out_cap, dtype=np.int64)
    raw_ptr = raw_i.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
    n = int(
        dll.sdx_it_finalize_path(
            raw_ptr,
            ctypes.c_int64(int(raw_i.size)),
            int(target_len),
            int(num_train),
            out_ptr,
            ctypes.c_int64(int(out_cap)),
        )
    )
    if n < 0:
        return None
    return np.ascontiguousarray(out[:n], dtype=np.int64)

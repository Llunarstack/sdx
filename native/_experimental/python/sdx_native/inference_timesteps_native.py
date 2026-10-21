"""Optional ctypes bridge to the C++ ``sdx_inference_timesteps`` library, mirroring
``diffusion.inference_timesteps._resample_length_numpy``; callers fall back to NumPy when it is missing or
returns an error.
"""

from __future__ import annotations

import ctypes
import functools

import numpy as np

from sdx_native.native_tools import load_shared_library


@functools.cache
def _lib() -> ctypes.CDLL | None:
    lib = load_shared_library("sdx_inference_timesteps")
    if lib is not None:
        lib.sdx_it_finalize_path.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
        ]
        lib.sdx_it_finalize_path.restype = ctypes.c_int64
    return lib


def finalize_inference_timesteps_native(raw: np.ndarray, target_len: int, num_train: int) -> np.ndarray | None:
    """Native :func:`diffusion.inference_timesteps._resample_length_numpy`; ``None`` if unavailable or on error."""
    lib = _lib()
    if target_len <= 0 or num_train < 1 or lib is None:
        return None
    raw_i = np.ascontiguousarray(raw, dtype=np.int64).ravel()
    out_cap = max(int(target_len), 2) + 32
    out = np.empty(out_cap, dtype=np.int64)
    n = int(
        lib.sdx_it_finalize_path(
            raw_i.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            int(raw_i.size),
            int(target_len),
            int(num_train),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            out_cap,
        )
    )
    if n < 0:
        return None
    return np.ascontiguousarray(out[:n])

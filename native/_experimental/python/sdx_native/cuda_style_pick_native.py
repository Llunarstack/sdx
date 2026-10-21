"""CUDA ``sdx_cuda_ml``: pick the best style embedding row by cosine (rows L2-normalized on the GPU first)."""

from __future__ import annotations

import ctypes
import functools

import numpy as np

from sdx_native.native_tools import load_shared_library


@functools.cache
def _lib() -> ctypes.CDLL | None:
    lib = load_shared_library("sdx_cuda_ml")
    if lib is None or not hasattr(lib, "sdx_cuda_style_pick_best_f32_host"):
        return None
    lib.sdx_cuda_l2_normalize_rows_f32_host.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float]
    lib.sdx_cuda_l2_normalize_rows_f32_host.restype = ctypes.c_int
    lib.sdx_cuda_style_pick_best_f32_host.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.sdx_cuda_style_pick_best_f32_host.restype = ctypes.c_int
    return lib


def available() -> bool:
    return _lib() is not None


def maybe_pick_best_style_embedding(
    query: np.ndarray,
    candidates: np.ndarray,
    *,
    eps: float = 1e-8,
) -> tuple[int, float] | None:
    """``query`` (dim,), ``candidates`` (n, dim) -> ``(best_index, cosine)``; ``None`` when unavailable."""
    lib = _lib()
    if lib is None or query.ndim != 1 or candidates.ndim != 2 or query.shape[0] != candidates.shape[1]:
        return None
    # Owned C-order copies: the kernel normalizes them in place and reads rows contiguously.
    q = np.array(query, dtype=np.float32, order="C").reshape(1, -1)
    c = np.array(candidates, dtype=np.float32, order="C")
    for m in (q, c):
        rc = lib.sdx_cuda_l2_normalize_rows_f32_host(m.ctypes.data_as(ctypes.c_void_p), m.shape[0], m.shape[1], eps)
        if rc != 0:
            return None
    out_i = ctypes.c_int(-1)
    out_s = ctypes.c_float(0.0)
    rc = lib.sdx_cuda_style_pick_best_f32_host(
        q.ctypes.data_as(ctypes.c_void_p),
        c.ctypes.data_as(ctypes.c_void_p),
        c.shape[0],
        c.shape[1],
        ctypes.byref(out_i),
        ctypes.byref(out_s),
    )
    if rc != 0:
        return None
    return int(out_i.value), float(out_s.value)

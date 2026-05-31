"""
CUDA ``sdx_cuda_ml``: pick best style embedding row by cosine (L2-normalized dot).
"""

from __future__ import annotations

import ctypes
from typing import Optional, Tuple

import numpy as np

from sdx_native.cuda_l2_normalize import get_cuda_ml_lib
from sdx_native.native_tools import cuda_ml_shared_library_path


class CudaStylePickLib:
    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        p = cuda_ml_shared_library_path()
        if p is None:
            return
        try:
            lib = ctypes.CDLL(str(p))
        except OSError:
            return
        if not hasattr(lib, "sdx_cuda_style_pick_best_f32_host"):
            return
        lib.sdx_cuda_style_pick_best_f32_host.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.sdx_cuda_style_pick_best_f32_host.restype = ctypes.c_int
        self._lib = lib

    @property
    def available(self) -> bool:
        return self._lib is not None


_PICK_LIB: Optional[CudaStylePickLib] = None


def get_cuda_style_pick_lib() -> CudaStylePickLib:
    global _PICK_LIB
    if _PICK_LIB is None:
        _PICK_LIB = CudaStylePickLib()
    return _PICK_LIB


def maybe_pick_best_style_embedding(
    query: np.ndarray,
    candidates: np.ndarray,
    *,
    eps: float = 1e-8,
) -> Optional[Tuple[int, float]]:
    """
    ``query`` shape (dim,), ``candidates`` (n, dim). Normalizes rows on GPU path when available.
    Returns (best_index, cosine_score) or None.
    """
    if query.ndim != 1 or candidates.ndim != 2:
        return None
    q = np.asarray(query, dtype=np.float32).reshape(1, -1).copy()
    c = np.asarray(candidates, dtype=np.float32).copy()
    if q.shape[1] != c.shape[1]:
        return None

    ml = get_cuda_ml_lib()
    pick = get_cuda_style_pick_lib()
    if ml.available:
        try:
            ml.l2_normalize_rows_inplace(q, eps=eps)
            ml.l2_normalize_rows_inplace(c, eps=eps)
        except Exception:
            pass

    if not pick.available:
        return _numpy_pick(q[0], c)

    out_i = ctypes.c_int(-1)
    out_s = ctypes.c_float(0.0)
    rc = pick._lib.sdx_cuda_style_pick_best_f32_host(
        q.ctypes.data_as(ctypes.c_void_p),
        c.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(int(c.shape[0])),
        ctypes.c_int(int(c.shape[1])),
        ctypes.byref(out_i),
        ctypes.byref(out_s),
    )
    if rc != 0:
        return _numpy_pick(q[0], c)
    return int(out_i.value), float(out_s.value)


def _numpy_pick(query: np.ndarray, candidates: np.ndarray) -> Tuple[int, float]:
    qn = query / (np.linalg.norm(query) + 1e-8)
    cn = candidates / (np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-8)
    scores = cn @ qn
    idx = int(np.argmax(scores))
    return idx, float(scores[idx])


__all__ = ["get_cuda_style_pick_lib", "maybe_pick_best_style_embedding"]

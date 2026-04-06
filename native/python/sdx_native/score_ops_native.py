"""
Optional CPU score ops via ``sdx_score_ops`` shared library.
Used by quality pick-best score normalization and weighted fusion.
"""

from __future__ import annotations

import ctypes
from typing import List, Optional

import numpy as np

from sdx_native.native_tools import score_ops_shared_library_path


class ScoreOpsLib:
    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        p = score_ops_shared_library_path()
        if p is None:
            return
        try:
            self._lib = ctypes.CDLL(str(p))
        except OSError:
            return
        self._lib.sdx_score_minmax_norm_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._lib.sdx_score_minmax_norm_f32.restype = ctypes.c_int
        self._lib.sdx_score_weighted_sum_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._lib.sdx_score_weighted_sum_f32.restype = ctypes.c_int

    @property
    def available(self) -> bool:
        return self._lib is not None

    def norm01(self, scores: List[float]) -> List[float]:
        if not self._lib:
            raise RuntimeError("sdx_score_ops not built")
        arr = np.asarray(scores, dtype=np.float32)
        out = np.zeros_like(arr)
        rc = self._lib.sdx_score_minmax_norm_f32(
            arr.ctypes.data_as(ctypes.c_void_p),
            int(arr.size),
            out.ctypes.data_as(ctypes.c_void_p),
        )
        if rc != 0:
            raise RuntimeError("sdx_score_minmax_norm_f32 failed")
        return [float(x) for x in out.tolist()]

    def weighted_sum(self, score_lists: List[List[float]], weights: List[float]) -> List[float]:
        if not self._lib:
            raise RuntimeError("sdx_score_ops not built")
        if not score_lists:
            return []
        rows = len(score_lists)
        cols = len(score_lists[0])
        if rows != len(weights):
            raise ValueError("weights length must equal number of score rows")
        matrix = np.asarray(score_lists, dtype=np.float32)
        if matrix.shape != (rows, cols):
            raise ValueError("score matrix must be rectangular")
        w = np.asarray(weights, dtype=np.float32)
        out = np.zeros((cols,), dtype=np.float32)
        rc = self._lib.sdx_score_weighted_sum_f32(
            matrix.ctypes.data_as(ctypes.c_void_p),
            int(rows),
            int(cols),
            w.ctypes.data_as(ctypes.c_void_p),
            out.ctypes.data_as(ctypes.c_void_p),
        )
        if rc != 0:
            raise RuntimeError("sdx_score_weighted_sum_f32 failed")
        return [float(x) for x in out.tolist()]


_LIB: Optional[ScoreOpsLib] = None


def get_score_ops_lib() -> ScoreOpsLib:
    global _LIB
    if _LIB is None:
        _LIB = ScoreOpsLib()
    return _LIB


def maybe_norm01_native(scores: List[float]) -> Optional[List[float]]:
    lib = get_score_ops_lib()
    if not lib.available:
        return None
    return lib.norm01(scores)


def maybe_weighted_sum_native(score_lists: List[List[float]], weights: List[float]) -> Optional[List[float]]:
    lib = get_score_ops_lib()
    if not lib.available:
        return None
    return lib.weighted_sum(score_lists, weights)

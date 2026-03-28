"""Optional ``sdx_cuda_sdpa_online`` — multi-head SDPA (head_dim=64) with online softmax tiles."""

from __future__ import annotations

import ctypes
from typing import Optional

import numpy as np

from sdx_native.native_tools import cuda_sdpa_online_shared_library_path


class CudaSdpaOnlineLib:
    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        p = cuda_sdpa_online_shared_library_path()
        if p is None:
            return
        try:
            self._lib = ctypes.CDLL(str(p))
        except OSError:
            return
        self._lib.sdx_cuda_sdpa_online_f32_host.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_float,
        ]
        self._lib.sdx_cuda_sdpa_online_f32_host.restype = ctypes.c_int

    @property
    def available(self) -> bool:
        return self._lib is not None

    def sdpa_host(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, *, scale: float) -> np.ndarray:
        if not self._lib:
            raise RuntimeError("sdx_cuda_sdpa_online not built")
        q = np.asarray(q, dtype=np.float32, order="C")
        k = np.asarray(k, dtype=np.float32, order="C")
        v = np.asarray(v, dtype=np.float32, order="C")
        if q.shape != k.shape or q.shape != v.shape or q.ndim != 4 or q.shape[0] != 1 or q.shape[3] != 64:
            raise ValueError("expected Q,K,V shape (1, H, S, 64) float32 C-contiguous")
        _, h, s, d = q.shape
        out = np.empty_like(q)
        rc = self._lib.sdx_cuda_sdpa_online_f32_host(
            q.ctypes.data_as(ctypes.c_void_p),
            k.ctypes.data_as(ctypes.c_void_p),
            v.ctypes.data_as(ctypes.c_void_p),
            out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int32(int(h)),
            ctypes.c_int32(int(s)),
            ctypes.c_float(float(scale)),
        )
        if rc != 0:
            raise RuntimeError(f"sdx_cuda_sdpa_online_f32_host failed ({rc})")
        return out


_LIB: Optional[CudaSdpaOnlineLib] = None


def get_cuda_sdpa_lib() -> CudaSdpaOnlineLib:
    global _LIB
    if _LIB is None:
        _LIB = CudaSdpaOnlineLib()
    return _LIB


def maybe_sdpa_online_cuda(q: np.ndarray, k: np.ndarray, v: np.ndarray, *, scale: float) -> Optional[np.ndarray]:
    lib = get_cuda_sdpa_lib()
    if not lib.available:
        return None
    return lib.sdpa_host(q, k, v, scale=scale)

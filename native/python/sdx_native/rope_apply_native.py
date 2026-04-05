"""Optional ``sdx_cuda_rope`` wrapper (interleaved RoPE apply on host Q/K arrays)."""

from __future__ import annotations

import ctypes
from typing import Optional, Tuple

import numpy as np

from sdx_native.native_tools import cuda_rope_shared_library_path


class CudaRopeLib:
    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        p = cuda_rope_shared_library_path()
        if p is None:
            return
        try:
            self._lib = ctypes.CDLL(str(p))
        except OSError:
            return
        self._lib.sdx_cuda_apply_rope_interleaved_f32_host.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_float,
        ]
        self._lib.sdx_cuda_apply_rope_interleaved_f32_host.restype = ctypes.c_int

    @property
    def available(self) -> bool:
        return self._lib is not None

    def apply_rope_host(self, q: np.ndarray, k: np.ndarray, *, theta_base: float = 10000.0) -> Tuple[np.ndarray, np.ndarray]:
        if not self._lib:
            raise RuntimeError("sdx_cuda_rope not built")
        q = np.asarray(q, dtype=np.float32, order="C")
        k = np.asarray(k, dtype=np.float32, order="C")
        if q.shape != k.shape or q.ndim != 2 or (q.shape[1] % 2) != 0:
            raise ValueError("expected q,k shape (n_tokens, dim) with even dim")
        qo = q.copy()
        ko = k.copy()
        rc = self._lib.sdx_cuda_apply_rope_interleaved_f32_host(
            qo.ctypes.data_as(ctypes.c_void_p),
            ko.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int32(int(qo.shape[0])),
            ctypes.c_int32(int(qo.shape[1])),
            ctypes.c_float(float(theta_base)),
        )
        if rc != 0:
            raise RuntimeError(f"sdx_cuda_apply_rope_interleaved_f32_host failed ({rc})")
        return qo, ko


_LIB: Optional[CudaRopeLib] = None


def get_cuda_rope_lib() -> CudaRopeLib:
    global _LIB
    if _LIB is None:
        _LIB = CudaRopeLib()
    return _LIB


def apply_rope_interleaved_numpy(
    q: np.ndarray,
    k: np.ndarray,
    *,
    theta_base: float = 10000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    if q.shape != k.shape or q.ndim != 2 or (q.shape[1] % 2) != 0:
        raise ValueError("expected q,k shape (n_tokens, dim) with even dim")
    n_tokens, dim = q.shape
    half = dim // 2
    qo = q.copy()
    ko = k.copy()
    pair_idx = np.arange(half, dtype=np.float32)
    inv_freq = theta_base ** (-2.0 * (pair_idx / float(dim)))
    t = np.arange(n_tokens, dtype=np.float32)[:, None]
    ang = t * inv_freq[None, :]
    c = np.cos(ang).astype(np.float32)
    s = np.sin(ang).astype(np.float32)

    q0 = qo[:, 0::2].copy()
    q1 = qo[:, 1::2].copy()
    qo[:, 0::2] = q0 * c - q1 * s
    qo[:, 1::2] = q0 * s + q1 * c

    k0 = ko[:, 0::2].copy()
    k1 = ko[:, 1::2].copy()
    ko[:, 0::2] = k0 * c - k1 * s
    ko[:, 1::2] = k0 * s + k1 * c
    return qo.astype(np.float32), ko.astype(np.float32)


def maybe_apply_rope_interleaved_cuda(
    q: np.ndarray,
    k: np.ndarray,
    *,
    theta_base: float = 10000.0,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    lib = get_cuda_rope_lib()
    if not lib.available:
        return None
    return lib.apply_rope_host(q, k, theta_base=theta_base)

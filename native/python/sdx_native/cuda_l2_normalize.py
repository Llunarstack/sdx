"""
Optional ``sdx_cuda_ml`` DLL: L2-normalize each row of a float32 matrix ``(n_rows, dim)`` (embedding-style).
"""

from __future__ import annotations

import ctypes
from typing import Optional

import numpy as np

from sdx_native.native_tools import cuda_ml_shared_library_path


class CudaMlLib:
    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        p = cuda_ml_shared_library_path()
        if p is None:
            return
        try:
            self._lib = ctypes.CDLL(str(p))
        except OSError:
            return
        self._lib.sdx_cuda_l2_normalize_rows_f32_host.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]
        self._lib.sdx_cuda_l2_normalize_rows_f32_host.restype = ctypes.c_int

    @property
    def available(self) -> bool:
        return self._lib is not None

    def l2_normalize_rows_inplace(self, x: np.ndarray, *, eps: float = 1e-8) -> None:
        if x.dtype != np.float32 or x.ndim != 2:
            raise ValueError("expected float32 ndarray shape (n_rows, dim)")
        if not self._lib:
            raise RuntimeError("sdx_cuda_ml not built")
        n, d = x.shape
        rc = self._lib.sdx_cuda_l2_normalize_rows_f32_host(
            x.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(int(n)),
            ctypes.c_int(int(d)),
            ctypes.c_float(float(eps)),
        )
        if rc != 0:
            raise RuntimeError(f"sdx_cuda_l2_normalize_rows_f32_host failed (code {rc})")


_LIB: Optional[CudaMlLib] = None


def get_cuda_ml_lib() -> CudaMlLib:
    global _LIB
    if _LIB is None:
        _LIB = CudaMlLib()
    return _LIB


def l2_normalize_rows_numpy(x: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """CPU reference."""
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return (x / n).astype(np.float32)


def maybe_l2_normalize_rows_cuda(x: np.ndarray, *, eps: float = 1e-8) -> Optional[np.ndarray]:
    """Return normalized copy via CUDA if DLL exists; else ``None`` (caller can use NumPy)."""
    lib = get_cuda_ml_lib()
    if not lib.available:
        return None
    y = np.array(x, dtype=np.float32, copy=True)
    lib.l2_normalize_rows_inplace(y, eps=eps)
    return y

"""Optional ``sdx_cuda_rmsnorm`` wrapper (row-wise RMSNorm on float32 host arrays)."""

from __future__ import annotations

import ctypes
from typing import Optional

import numpy as np

from sdx_native.native_tools import cuda_rmsnorm_shared_library_path


class CudaRmsNormLib:
    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        p = cuda_rmsnorm_shared_library_path()
        if p is None:
            return
        try:
            self._lib = ctypes.CDLL(str(p))
        except OSError:
            return
        self._lib.sdx_cuda_rmsnorm_rows_f32_host.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_float,
        ]
        self._lib.sdx_cuda_rmsnorm_rows_f32_host.restype = ctypes.c_int

    @property
    def available(self) -> bool:
        return self._lib is not None

    def rmsnorm_rows_host(self, x: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
        if not self._lib:
            raise RuntimeError("sdx_cuda_rmsnorm not built")
        x = np.asarray(x, dtype=np.float32, order="C")
        if x.ndim != 2:
            raise ValueError("expected 2D array (n_rows, dim)")
        out = x.copy()
        n_rows, dim = out.shape
        rc = self._lib.sdx_cuda_rmsnorm_rows_f32_host(
            out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int32(int(n_rows)),
            ctypes.c_int32(int(dim)),
            ctypes.c_float(float(eps)),
        )
        if rc != 0:
            raise RuntimeError(f"sdx_cuda_rmsnorm_rows_f32_host failed ({rc})")
        return out


_LIB: Optional[CudaRmsNormLib] = None


def get_cuda_rmsnorm_lib() -> CudaRmsNormLib:
    global _LIB
    if _LIB is None:
        _LIB = CudaRmsNormLib()
    return _LIB


def rmsnorm_rows_numpy(x: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    den = np.sqrt(np.maximum(np.mean(a * a, axis=1, keepdims=True), float(eps)))
    return (a / den).astype(np.float32)


def maybe_rmsnorm_rows_cuda(x: np.ndarray, *, eps: float = 1e-6) -> Optional[np.ndarray]:
    lib = get_cuda_rmsnorm_lib()
    if not lib.available:
        return None
    return lib.rmsnorm_rows_host(x, eps=eps)

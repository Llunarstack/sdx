"""
Optional ``sdx_cuda_percentile_clamp`` wrapper — per-sample percentile clamp on
float32 tensors (B, row_len).

Matches ``dynamic_percentile_clamp()`` in ``diffusion/holy_grail/latent_refiner.py``.

Falls back to pure NumPy when the native library is not built.
"""

from __future__ import annotations

import ctypes
from typing import Optional

import numpy as np

from sdx_native.native_tools import cuda_percentile_clamp_shared_library_path


class CudaPercentileClampLib:
    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        p = cuda_percentile_clamp_shared_library_path()
        if p is None:
            return
        try:
            lib = ctypes.CDLL(str(p))
        except OSError:
            return
        lib.sdx_cuda_percentile_clamp_f32.argtypes = [
            ctypes.c_void_p,  # data_host (B, row_len), modified in-place
            ctypes.c_int,     # B
            ctypes.c_int,     # row_len
            ctypes.c_float,   # quantile
            ctypes.c_float,   # floor_val
        ]
        lib.sdx_cuda_percentile_clamp_f32.restype = ctypes.c_int
        self._lib = lib

    @property
    def available(self) -> bool:
        return self._lib is not None

    def clamp(
        self,
        x: np.ndarray,
        quantile: float,
        floor_val: float,
    ) -> np.ndarray:
        """Clamp a (B, row_len) float32 array in-place and return it."""
        if self._lib is None:
            raise RuntimeError("sdx_cuda_percentile_clamp not built")
        if x.ndim != 2:
            raise ValueError("expected 2-D array (B, row_len)")
        x = np.ascontiguousarray(x, dtype=np.float32)
        B, row_len = x.shape
        rc = self._lib.sdx_cuda_percentile_clamp_f32(
            x.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(B),
            ctypes.c_int(row_len),
            ctypes.c_float(float(quantile)),
            ctypes.c_float(float(floor_val)),
        )
        if rc != 0:
            raise RuntimeError(f"sdx_cuda_percentile_clamp_f32 failed (rc={rc})")
        return x


_LIB: Optional[CudaPercentileClampLib] = None


def _get_lib() -> CudaPercentileClampLib:
    global _LIB
    if _LIB is None:
        _LIB = CudaPercentileClampLib()
    return _LIB


def percentile_clamp_numpy(
    x: np.ndarray,
    quantile: float,
    floor_val: float,
) -> np.ndarray:
    """Pure-NumPy fallback — same semantics as the CUDA kernel."""
    x = np.asarray(x, dtype=np.float32)
    B, row_len = x.shape
    out = x.copy()
    for b in range(B):
        row = out[b]
        bound = float(np.quantile(np.abs(row), quantile))
        if bound < floor_val:
            bound = floor_val
        np.clip(row, -bound, bound, out=row)
        row /= bound
    return out


def maybe_percentile_clamp_cuda(
    x: np.ndarray,
    quantile: float,
    floor_val: float,
) -> Optional[np.ndarray]:
    """Return clamped array via CUDA kernel, or None if not available."""
    lib = _get_lib()
    if not lib.available:
        return None
    return lib.clamp(x, quantile, floor_val)

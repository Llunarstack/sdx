"""
Optional CUDA layout transform: uint8 ``(H,W,3)`` HWC -> float32 ``(3,H,W)`` NCHW in ``[0,1]``.

Build C++ with ``-DSDX_BUILD_CUDA=ON`` (requires nvcc). For training, prefer **torch** tensors on GPU;
this exists for micro-benchmarks and ctypes experiments outside the main PyTorch graph.
"""

from __future__ import annotations

import ctypes
from typing import Optional

import numpy as np

from sdx_native.native_tools import cuda_hwc_to_chw_shared_library_path


class CudaHwcLib:
    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        p = cuda_hwc_to_chw_shared_library_path()
        if p is None:
            return
        try:
            self._lib = ctypes.CDLL(str(p))
        except OSError:
            return
        self._lib.sdx_cuda_u8hwc3_to_chw_f32_host.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._lib.sdx_cuda_u8hwc3_to_chw_f32_host.restype = ctypes.c_int

    @property
    def available(self) -> bool:
        return self._lib is not None

    def u8_hwc3_to_chw_f32(self, hwc: np.ndarray) -> np.ndarray:
        if hwc.dtype != np.uint8 or hwc.ndim != 3 or hwc.shape[2] != 3:
            raise ValueError("expected uint8 ndarray shape (H, W, 3)")
        if not self._lib:
            raise RuntimeError("sdx_cuda_hwc_to_chw not built")
        h, w, _ = hwc.shape
        out = np.empty((3, h, w), dtype=np.float32)
        rc = self._lib.sdx_cuda_u8hwc3_to_chw_f32_host(
            hwc.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(int(h)),
            ctypes.c_int(int(w)),
            out.ctypes.data_as(ctypes.c_void_p),
        )
        if rc != 0:
            raise RuntimeError(f"sdx_cuda_u8hwc3_to_chw_f32_host failed (code {rc})")
        return out


_LIB: Optional[CudaHwcLib] = None


def get_cuda_hwc_lib() -> CudaHwcLib:
    global _LIB
    if _LIB is None:
        _LIB = CudaHwcLib()
    return _LIB


def u8_hwc_to_chw_f32_numpy(hwc: np.ndarray) -> np.ndarray:
    """Pure NumPy reference (CPU) — same layout as CUDA path."""
    if hwc.dtype != np.uint8:
        hwc = hwc.astype(np.uint8)
    x = hwc.astype(np.float32) * (1.0 / 255.0)
    return np.transpose(x, (2, 0, 1)).copy()


def maybe_u8_hwc_to_chw_f32_cuda(hwc: np.ndarray) -> Optional[np.ndarray]:
    """Return GPU-backed result via optional DLL, or ``None`` if not built."""
    lib = get_cuda_hwc_lib()
    if not lib.available:
        return None
    return lib.u8_hwc3_to_chw_f32(hwc)

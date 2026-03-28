"""
Optional ``sdx_cuda_flow_matching`` DLL: elementwise ``v = epsilon - x0`` (float32, host buffers).

Educational kernel aligned with ``diffusion/flow_matching.py``; training stays in PyTorch.
"""

from __future__ import annotations

import ctypes
from typing import Optional

import numpy as np

from sdx_native.native_tools import cuda_flow_matching_shared_library_path


class CudaFlowVelocityLib:
    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        p = cuda_flow_matching_shared_library_path()
        if p is None:
            return
        try:
            self._lib = ctypes.CDLL(str(p))
        except OSError:
            return
        self._lib.sdx_cuda_flow_velocity_residual_f32_host.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
        ]
        self._lib.sdx_cuda_flow_velocity_residual_f32_host.restype = ctypes.c_int

    @property
    def available(self) -> bool:
        return self._lib is not None

    def velocity_residual_host(self, x0: np.ndarray, eps: np.ndarray) -> np.ndarray:
        if not self._lib:
            raise RuntimeError("sdx_cuda_flow_matching not built")
        x0 = np.asarray(x0, dtype=np.float32, order="C")
        eps = np.asarray(eps, dtype=np.float32, order="C")
        if x0.shape != eps.shape:
            raise ValueError("x0 and eps must have the same shape")
        flat = x0.size
        x0f = x0.reshape(-1)
        epsf = eps.reshape(-1)
        out = np.empty_like(x0f)
        rc = self._lib.sdx_cuda_flow_velocity_residual_f32_host(
            x0f.ctypes.data_as(ctypes.c_void_p),
            epsf.ctypes.data_as(ctypes.c_void_p),
            out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int64(int(flat)),
        )
        if rc != 0:
            raise RuntimeError(f"sdx_cuda_flow_velocity_residual_f32_host failed ({rc})")
        return out.reshape(x0.shape)


_LIB: Optional[CudaFlowVelocityLib] = None


def get_cuda_flow_velocity_lib() -> CudaFlowVelocityLib:
    global _LIB
    if _LIB is None:
        _LIB = CudaFlowVelocityLib()
    return _LIB


def flow_velocity_residual_numpy(x0: np.ndarray, eps: np.ndarray) -> np.ndarray:
    return (np.asarray(eps, dtype=np.float32) - np.asarray(x0, dtype=np.float32)).astype(np.float32)


def maybe_flow_velocity_residual_cuda(x0: np.ndarray, eps: np.ndarray) -> Optional[np.ndarray]:
    lib = get_cuda_flow_velocity_lib()
    if not lib.available:
        return None
    return lib.velocity_residual_host(x0, eps)

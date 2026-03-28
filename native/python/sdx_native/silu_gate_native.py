"""Optional ``sdx_cuda_silu_gate`` wrapper (fused SiLU(x) * gate host path)."""

from __future__ import annotations

import ctypes
from typing import Optional

import numpy as np

from sdx_native.native_tools import cuda_silu_gate_shared_library_path


class CudaSiluGateLib:
    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        p = cuda_silu_gate_shared_library_path()
        if p is None:
            return
        try:
            self._lib = ctypes.CDLL(str(p))
        except OSError:
            return
        self._lib.sdx_cuda_silu_gate_f32_host.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
        ]
        self._lib.sdx_cuda_silu_gate_f32_host.restype = ctypes.c_int

    @property
    def available(self) -> bool:
        return self._lib is not None

    def silu_gate_host(self, x: np.ndarray, gate: np.ndarray) -> np.ndarray:
        if not self._lib:
            raise RuntimeError("sdx_cuda_silu_gate not built")
        x = np.asarray(x, dtype=np.float32, order="C")
        g = np.asarray(gate, dtype=np.float32, order="C")
        if x.shape != g.shape:
            raise ValueError("x and gate must have same shape")
        xf = x.reshape(-1)
        gf = g.reshape(-1)
        out = np.empty_like(xf)
        rc = self._lib.sdx_cuda_silu_gate_f32_host(
            xf.ctypes.data_as(ctypes.c_void_p),
            gf.ctypes.data_as(ctypes.c_void_p),
            out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int64(int(xf.size)),
        )
        if rc != 0:
            raise RuntimeError(f"sdx_cuda_silu_gate_f32_host failed ({rc})")
        return out.reshape(x.shape)


_LIB: Optional[CudaSiluGateLib] = None


def get_cuda_silu_gate_lib() -> CudaSiluGateLib:
    global _LIB
    if _LIB is None:
        _LIB = CudaSiluGateLib()
    return _LIB


def silu_gate_numpy(x: np.ndarray, gate: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    g = np.asarray(gate, dtype=np.float32)
    sig = 1.0 / (1.0 + np.exp(-x))
    return (x * sig * g).astype(np.float32)


def maybe_silu_gate_cuda(x: np.ndarray, gate: np.ndarray) -> Optional[np.ndarray]:
    lib = get_cuda_silu_gate_lib()
    if not lib.available:
        return None
    return lib.silu_gate_host(x, gate)

"""Optional ``sdx_cuda_nf4`` — host-side NF4 block dequant (matches ``utils.quantization.nf4_codec``)."""

from __future__ import annotations

import ctypes
from typing import Optional

import numpy as np

from sdx_native.native_tools import cuda_nf4_shared_library_path


class CudaNf4Lib:
    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        p = cuda_nf4_shared_library_path()
        if p is None:
            return
        try:
            self._lib = ctypes.CDLL(str(p))
        except OSError:
            return
        self._lib.sdx_cuda_nf4_dequant_f32_host.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        self._lib.sdx_cuda_nf4_dequant_f32_host.restype = ctypes.c_int

    @property
    def available(self) -> bool:
        return self._lib is not None

    def dequant_host(self, packed: np.ndarray, absmax: np.ndarray, *, block_size: int, n_weights: int) -> np.ndarray:
        if not self._lib:
            raise RuntimeError("sdx_cuda_nf4 not built")
        packed = np.asarray(packed, dtype=np.uint8, order="C")
        absmax = np.asarray(absmax, dtype=np.float32, order="C")
        n_blocks = absmax.size
        out = np.empty((n_weights,), dtype=np.float32)
        rc = self._lib.sdx_cuda_nf4_dequant_f32_host(
            packed.ctypes.data_as(ctypes.c_void_p),
            absmax.ctypes.data_as(ctypes.c_void_p),
            out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int32(int(n_blocks)),
            ctypes.c_int32(int(block_size)),
            ctypes.c_int32(int(n_weights)),
        )
        if rc != 0:
            raise RuntimeError(f"sdx_cuda_nf4_dequant_f32_host failed ({rc})")
        return out


_LIB: Optional[CudaNf4Lib] = None


def get_cuda_nf4_lib() -> CudaNf4Lib:
    global _LIB
    if _LIB is None:
        _LIB = CudaNf4Lib()
    return _LIB


def maybe_nf4_dequant_cuda(packed: np.ndarray, absmax: np.ndarray, *, block_size: int, n_weights: int) -> Optional[np.ndarray]:
    lib = get_cuda_nf4_lib()
    if not lib.available:
        return None
    return lib.dequant_host(packed, absmax, block_size=block_size, n_weights=n_weights)

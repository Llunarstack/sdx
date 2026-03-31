"""
Optional ``sdx_cuda_gaussian_blur`` wrapper — depthwise Gaussian blur on float32
latent tensors (B, C, H, W).

Matches ``gaussian_blur_latent()`` in ``diffusion/sampling_utils.py`` and
``_gaussian_blur()`` in ``diffusion/holy_grail/latent_refiner.py``.

Falls back to pure PyTorch when the native library is not built.
"""

from __future__ import annotations

import ctypes
from typing import Optional

import numpy as np

from sdx_native.native_tools import cuda_gaussian_blur_shared_library_path


class CudaGaussianBlurLib:
    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        p = cuda_gaussian_blur_shared_library_path()
        if p is None:
            return
        try:
            lib = ctypes.CDLL(str(p))
        except OSError:
            return
        lib.sdx_cuda_gaussian_blur_latent_f32.argtypes = [
            ctypes.c_void_p,  # src_host
            ctypes.c_void_p,  # dst_host
            ctypes.c_int,     # B
            ctypes.c_int,     # C
            ctypes.c_int,     # H
            ctypes.c_int,     # W
            ctypes.c_float,   # sigma
        ]
        lib.sdx_cuda_gaussian_blur_latent_f32.restype = ctypes.c_int
        self._lib = lib

    @property
    def available(self) -> bool:
        return self._lib is not None

    def blur(self, x: np.ndarray, sigma: float) -> np.ndarray:
        """Blur a (B, C, H, W) float32 array in-place and return it."""
        if self._lib is None:
            raise RuntimeError("sdx_cuda_gaussian_blur not built")
        if x.ndim != 4:
            raise ValueError("expected 4-D array (B, C, H, W)")
        x = np.ascontiguousarray(x, dtype=np.float32)
        B, C, H, W = x.shape
        dst = np.empty_like(x)
        rc = self._lib.sdx_cuda_gaussian_blur_latent_f32(
            x.ctypes.data_as(ctypes.c_void_p),
            dst.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(B),
            ctypes.c_int(C),
            ctypes.c_int(H),
            ctypes.c_int(W),
            ctypes.c_float(float(sigma)),
        )
        if rc != 0:
            raise RuntimeError(f"sdx_cuda_gaussian_blur_latent_f32 failed (rc={rc})")
        return dst


_LIB: Optional[CudaGaussianBlurLib] = None


def _get_lib() -> CudaGaussianBlurLib:
    global _LIB
    if _LIB is None:
        _LIB = CudaGaussianBlurLib()
    return _LIB


def maybe_gaussian_blur_cuda(x: np.ndarray, sigma: float) -> Optional[np.ndarray]:
    """Return blurred array via CUDA kernel, or None if not available."""
    lib = _get_lib()
    if not lib.available:
        return None
    return lib.blur(x, sigma)

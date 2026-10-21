"""
Optional ``sdx_cuda_gaussian_blur`` wrapper: depthwise Gaussian blur on float32 latents (B, C, H, W),
matching ``diffusion.sampling_utils.gaussian_blur_latent``. Callers fall back to PyTorch when it is not built.
"""

from __future__ import annotations

import ctypes
import functools

import numpy as np

from sdx_native.native_tools import load_shared_library


@functools.cache
def _lib() -> ctypes.CDLL | None:
    lib = load_shared_library("sdx_cuda_gaussian_blur")
    if lib is not None:
        # src_host, dst_host, B, C, H, W, sigma
        lib.sdx_cuda_gaussian_blur_latent_f32.argtypes = (
            [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int] * 4 + [ctypes.c_float]
        )
        lib.sdx_cuda_gaussian_blur_latent_f32.restype = ctypes.c_int
    return lib


def maybe_gaussian_blur_cuda(x: np.ndarray, sigma: float) -> np.ndarray | None:
    """Blurred copy of a (B, C, H, W) array via the CUDA kernel, or ``None`` if not available."""
    lib = _lib()
    if lib is None:
        return None
    if x.ndim != 4:
        raise ValueError("expected 4-D array (B, C, H, W)")
    x = np.ascontiguousarray(x, dtype=np.float32)
    dst = np.empty_like(x)
    rc = lib.sdx_cuda_gaussian_blur_latent_f32(
        x.ctypes.data_as(ctypes.c_void_p), dst.ctypes.data_as(ctypes.c_void_p), *x.shape, float(sigma)
    )
    if rc != 0:
        raise RuntimeError(f"sdx_cuda_gaussian_blur_latent_f32 failed (rc={rc})")
    return dst

"""
Optional ``sdx_mask_ops`` wrapper: CPU mask -> patch-weight average pooling, matching
``image_mask_to_patch_weights()`` in ``utils/training/part_aware_training.py`` (runs every training step
when grounding masks are present). Callers fall back to PyTorch when it is not built.
"""

from __future__ import annotations

import ctypes
import functools

import numpy as np

from sdx_native.native_tools import load_shared_library


@functools.cache
def _lib() -> ctypes.CDLL | None:
    lib = load_shared_library("sdx_mask_ops")
    if lib is not None:
        # mask (B, 1, H, W), out (B, ph*pw), B, H, W, ph, pw
        lib.sdx_mask_to_patch_weights_f32.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int] * 5
        lib.sdx_mask_to_patch_weights_f32.restype = ctypes.c_int
    return lib


def maybe_mask_to_patch_weights_native(mask: np.ndarray, ph: int, pw: int) -> np.ndarray | None:
    """(B, 1, H, W) mask -> (B, ph*pw) cell means via the C++ kernel, or ``None`` if not built."""
    lib = _lib()
    if lib is None:
        return None
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must be (B, 1, H, W)")
    mask = np.ascontiguousarray(mask, dtype=np.float32)
    b, _, h, w = mask.shape
    out = np.empty((b, ph * pw), dtype=np.float32)
    rc = lib.sdx_mask_to_patch_weights_f32(
        mask.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p), b, h, w, ph, pw
    )
    if rc == -3:
        raise ValueError(f"mask dimensions ({h}, {w}) not divisible by patch grid ({ph}, {pw})")
    if rc != 0:
        raise RuntimeError(f"sdx_mask_to_patch_weights_f32 failed (rc={rc})")
    return out

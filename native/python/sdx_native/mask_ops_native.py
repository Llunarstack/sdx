"""
Optional ``sdx_mask_ops`` wrapper — CPU mask → patch-weight downsampling.

Matches ``image_mask_to_patch_weights()`` in
``utils/training/part_aware_training.py``.

Called every training step when grounding masks are present; the C++ loop
avoids Python overhead and PyTorch autograd bookkeeping for this pure
data-prep operation.

Falls back to NumPy when the native library is not built.
"""

from __future__ import annotations

import ctypes
from typing import Optional

import numpy as np

from sdx_native.native_tools import mask_ops_shared_library_path


class MaskOpsLib:
    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        p = mask_ops_shared_library_path()
        if p is None:
            return
        try:
            lib = ctypes.CDLL(str(p))
        except OSError:
            return
        lib.sdx_mask_to_patch_weights_f32.argtypes = [
            ctypes.c_void_p,  # mask  (B, 1, H, W)
            ctypes.c_void_p,  # out   (B, ph*pw)
            ctypes.c_int,     # B
            ctypes.c_int,     # H
            ctypes.c_int,     # W
            ctypes.c_int,     # ph
            ctypes.c_int,     # pw
        ]
        lib.sdx_mask_to_patch_weights_f32.restype = ctypes.c_int
        self._lib = lib

    @property
    def available(self) -> bool:
        return self._lib is not None

    def mask_to_patch_weights(
        self,
        mask: np.ndarray,
        ph: int,
        pw: int,
    ) -> np.ndarray:
        """
        Downsample (B, 1, H, W) mask to (B, ph*pw) via average pooling.

        H must be divisible by ph; W must be divisible by pw.
        """
        if self._lib is None:
            raise RuntimeError("sdx_mask_ops not built")
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("mask must be (B, 1, H, W)")
        mask = np.ascontiguousarray(mask, dtype=np.float32)
        B, _, H, W = mask.shape
        out = np.empty((B, ph * pw), dtype=np.float32)
        rc = self._lib.sdx_mask_to_patch_weights_f32(
            mask.ctypes.data_as(ctypes.c_void_p),
            out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(B),
            ctypes.c_int(H),
            ctypes.c_int(W),
            ctypes.c_int(ph),
            ctypes.c_int(pw),
        )
        if rc == -3:
            raise ValueError(
                f"mask dimensions ({H}, {W}) not divisible by patch grid ({ph}, {pw})"
            )
        if rc != 0:
            raise RuntimeError(f"sdx_mask_to_patch_weights_f32 failed (rc={rc})")
        return out


_LIB: Optional[MaskOpsLib] = None


def _get_lib() -> MaskOpsLib:
    global _LIB
    if _LIB is None:
        _LIB = MaskOpsLib()
    return _LIB


def mask_to_patch_weights_numpy(
    mask: np.ndarray,
    ph: int,
    pw: int,
) -> np.ndarray:
    """Pure-NumPy fallback — same semantics as the C++ kernel."""
    mask = np.asarray(mask, dtype=np.float32)
    B, _, H, W = mask.shape
    cell_h, cell_w = H // ph, W // pw
    # Reshape into patch grid and average each cell.
    m = mask[:, 0]  # (B, H, W)
    m = m.reshape(B, ph, cell_h, pw, cell_w)
    return m.mean(axis=(2, 4)).reshape(B, ph * pw)


def maybe_mask_to_patch_weights_native(
    mask: np.ndarray,
    ph: int,
    pw: int,
) -> Optional[np.ndarray]:
    """Return patch weights via C++ kernel, or None if not available."""
    lib = _get_lib()
    if not lib.available:
        return None
    return lib.mask_to_patch_weights(mask, ph, pw)

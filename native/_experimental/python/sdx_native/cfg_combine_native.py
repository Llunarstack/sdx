"""Optional C ``sdx_c_cfg_combine`` — classic CFG + rescale on float32 buffers."""

from __future__ import annotations

import ctypes

import numpy as np

from sdx_native.native_tools import c_cfg_combine_shared_library_path


class CfgCombineLib:
    def __init__(self) -> None:
        self._lib: ctypes.CDLL | None = None
        p = c_cfg_combine_shared_library_path()
        if p is None:
            return
        try:
            lib = ctypes.CDLL(str(p))
        except OSError:
            return
        try:
            lib.sdx_c_cfg_combine_f32.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.c_float,
                ctypes.c_float,
            ]
            lib.sdx_c_cfg_combine_f32.restype = ctypes.c_int
        except AttributeError:
            return
        self._lib = lib

    @property
    def available(self) -> bool:
        return self._lib is not None

    def combine(
        self,
        cond: np.ndarray,
        uncond: np.ndarray,
        *,
        scale: float = 7.5,
        rescale_phi: float = 0.0,
    ) -> np.ndarray | None:
        if self._lib is None:
            return None
        a = np.ascontiguousarray(cond, dtype=np.float32)
        b = np.ascontiguousarray(uncond, dtype=np.float32)
        if a.shape != b.shape:
            return None
        out = np.empty_like(a)
        rc = self._lib.sdx_c_cfg_combine_f32(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(a.size),
            ctypes.c_float(scale),
            ctypes.c_float(rescale_phi),
        )
        return out if rc == 0 else None


_LIB: CfgCombineLib | None = None


def _lib() -> CfgCombineLib:
    global _LIB
    if _LIB is None:
        _LIB = CfgCombineLib()
    return _LIB


def cfg_combine_f32(
    cond: np.ndarray,
    uncond: np.ndarray,
    *,
    scale: float = 7.5,
    rescale_phi: float = 0.0,
) -> np.ndarray | None:
    return _lib().combine(cond, uncond, scale=scale, rescale_phi=rescale_phi)


def cfg_combine_numpy(
    cond: np.ndarray,
    uncond: np.ndarray,
    *,
    scale: float = 7.5,
    rescale_phi: float = 0.0,
) -> np.ndarray:
    """Native if available, else pure NumPy classic CFG (+ optional rescale)."""
    hit = cfg_combine_f32(cond, uncond, scale=scale, rescale_phi=rescale_phi)
    if hit is not None:
        return hit
    a = np.asarray(cond, dtype=np.float32)
    b = np.asarray(uncond, dtype=np.float32)
    out = b + float(scale) * (a - b)
    phi = float(rescale_phi)
    if phi <= 0.0:
        return out
    std_c = float(np.std(a)) + 1e-8
    std_o = float(np.std(out)) + 1e-8
    mean_o = float(np.mean(out))
    centered = (out - mean_o) * (std_c / std_o) + mean_o
    return (phi * centered + (1.0 - phi) * out).astype(np.float32)


def available() -> bool:
    return _lib().available


__all__ = ["CfgCombineLib", "cfg_combine_f32", "cfg_combine_numpy", "available"]

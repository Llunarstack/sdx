"""Optional Rust ``sdx_quality_scorers`` — Laplacian / highlight / entropy kernels."""

from __future__ import annotations

import ctypes

import numpy as np

from sdx_native.native_tools import rust_quality_scorers_shared_library_path


class QualityScorersLib:
    def __init__(self) -> None:
        self._lib: ctypes.CDLL | None = None
        p = rust_quality_scorers_shared_library_path()
        if p is None:
            return
        try:
            lib = ctypes.CDLL(str(p))
        except OSError:
            return
        for name in (
            "sdx_quality_laplacian_var_u8",
            "sdx_quality_highlight_frac_u8",
            "sdx_quality_midtone_entropy_u8",
        ):
            getattr(lib, name).restype = ctypes.c_double
        lib.sdx_quality_laplacian_var_u8.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        lib.sdx_quality_highlight_frac_u8.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
        ]
        lib.sdx_quality_midtone_entropy_u8.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float,
        ]
        self._lib = lib

    @property
    def available(self) -> bool:
        return self._lib is not None

    def _rgb_u8(self, rgb: np.ndarray) -> tuple[np.ndarray, int, int] | None:
        arr = np.ascontiguousarray(rgb)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return None
        h, w = int(arr.shape[0]), int(arr.shape[1])
        if h < 3 or w < 3:
            return None
        return arr[..., :3].astype(np.uint8, copy=False), h, w

    def laplacian_var(self, rgb: np.ndarray) -> float | None:
        if self._lib is None:
            return None
        packed = self._rgb_u8(rgb)
        if packed is None:
            return None
        arr, h, w = packed
        v = float(
            self._lib.sdx_quality_laplacian_var_u8(
                arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.c_size_t(h),
                ctypes.c_size_t(w),
            )
        )
        return v if v >= 0.0 else None

    def highlight_frac(self, rgb: np.ndarray, thr: float = 245.0) -> float | None:
        if self._lib is None:
            return None
        packed = self._rgb_u8(rgb)
        if packed is None:
            return None
        arr, h, w = packed
        v = float(
            self._lib.sdx_quality_highlight_frac_u8(
                arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.c_size_t(h),
                ctypes.c_size_t(w),
                ctypes.c_float(thr),
            )
        )
        return v if v >= 0.0 else None

    def midtone_entropy(self, rgb: np.ndarray, lo: float = 40.0, hi: float = 220.0) -> float | None:
        if self._lib is None:
            return None
        packed = self._rgb_u8(rgb)
        if packed is None:
            return None
        arr, h, w = packed
        v = float(
            self._lib.sdx_quality_midtone_entropy_u8(
                arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.c_size_t(h),
                ctypes.c_size_t(w),
                ctypes.c_float(lo),
                ctypes.c_float(hi),
            )
        )
        return v if v >= 0.0 else None


_LIB: QualityScorersLib | None = None


def _lib() -> QualityScorersLib:
    global _LIB
    if _LIB is None:
        _LIB = QualityScorersLib()
    return _LIB


def laplacian_var(rgb: np.ndarray) -> float | None:
    return _lib().laplacian_var(rgb)


def highlight_frac(rgb: np.ndarray, thr: float = 245.0) -> float | None:
    return _lib().highlight_frac(rgb, thr=thr)


def midtone_entropy(rgb: np.ndarray, lo: float = 40.0, hi: float = 220.0) -> float | None:
    return _lib().midtone_entropy(rgb, lo=lo, hi=hi)


def available() -> bool:
    return _lib().available


__all__ = ["QualityScorersLib", "laplacian_var", "highlight_frac", "midtone_entropy", "available"]

"""Optional Rust ``sdx_canny_ops`` — Canny / soft-edge control maps."""

from __future__ import annotations

import ctypes

import numpy as np

from sdx_native.native_tools import rust_canny_ops_shared_library_path


class CannyOpsLib:
    def __init__(self) -> None:
        self._lib: ctypes.CDLL | None = None
        p = rust_canny_ops_shared_library_path()
        if p is None:
            return
        try:
            lib = ctypes.CDLL(str(p))
        except OSError:
            return
        lib.sdx_canny_u8.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
        ]
        lib.sdx_canny_u8.restype = ctypes.c_int
        self._lib = lib

    @property
    def available(self) -> bool:
        return self._lib is not None

    def canny(
        self,
        rgb: np.ndarray,
        *,
        low: float = 40.0,
        high: float = 100.0,
        soft: bool = False,
    ) -> np.ndarray | None:
        if self._lib is None:
            return None
        arr = np.ascontiguousarray(rgb)
        if arr.ndim == 2:
            h, w = arr.shape
            ch = 1
            arr = arr.astype(np.uint8, copy=False)
        elif arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
            h, w, ch = arr.shape
            arr = arr.astype(np.uint8, copy=False)
        else:
            return None
        out = np.empty((h, w), dtype=np.uint8)
        rc = self._lib.sdx_canny_u8(
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_size_t(h),
            ctypes.c_size_t(w),
            ctypes.c_size_t(ch),
            ctypes.c_float(low),
            ctypes.c_float(high),
            ctypes.c_int(1 if soft else 0),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
        return out if rc == 0 else None


_LIB: CannyOpsLib | None = None


def _lib() -> CannyOpsLib:
    global _LIB
    if _LIB is None:
        _LIB = CannyOpsLib()
    return _LIB


def canny_u8(rgb: np.ndarray, *, low: float = 40.0, high: float = 100.0, soft: bool = False) -> np.ndarray | None:
    return _lib().canny(rgb, low=low, high=high, soft=soft)


def available() -> bool:
    return _lib().available


__all__ = ["CannyOpsLib", "canny_u8", "available"]

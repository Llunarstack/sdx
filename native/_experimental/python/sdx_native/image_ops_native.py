"""
Optional Rust ``sdx_image_ops`` cdylib wrapper — deterministic object/blob
counting for count-adherence scoring.

Matches the NumPy reference in ``scripts/tools/eval/scorers.py`` but runs the
connected-components pass in Rust (GIL-free, no per-pixel Python overhead).

Build:
    cd native/rust/sdx-image-ops
    cargo build --release

Falls back to ``None`` when the library is not built, so callers degrade to the
NumPy implementation.
"""

from __future__ import annotations

import ctypes

import numpy as np

from sdx_native.native_tools import rust_image_ops_shared_library_path


class ImageOpsLib:
    def __init__(self) -> None:
        self._lib: ctypes.CDLL | None = None
        p = rust_image_ops_shared_library_path()
        if p is None:
            return
        try:
            lib = ctypes.CDLL(str(p))
        except OSError:
            return
        lib.sdx_count_blobs_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
        ]
        lib.sdx_count_blobs_f32.restype = ctypes.c_int64
        self._lib = lib

    @property
    def available(self) -> bool:
        return self._lib is not None

    def count_blobs(self, gray: np.ndarray, min_area_frac: float = 0.005) -> int | None:
        """
        Count foreground blobs in a 2-D grayscale array (values in [0, 1]).

        Returns the blob count, or None if the native library is unavailable or
        the call reported invalid arguments.
        """
        if self._lib is None:
            return None
        arr = np.ascontiguousarray(gray, dtype=np.float32)
        if arr.ndim != 2:
            return None
        h, w = arr.shape
        ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        n = self._lib.sdx_count_blobs_f32(ptr, ctypes.c_size_t(h), ctypes.c_size_t(w), ctypes.c_float(min_area_frac))
        return int(n) if n >= 0 else None


_LIB: ImageOpsLib | None = None


def _lib() -> ImageOpsLib:
    global _LIB
    if _LIB is None:
        _LIB = ImageOpsLib()
    return _LIB


def count_blobs(gray: np.ndarray, min_area_frac: float = 0.005) -> int | None:
    """Module-level convenience: count blobs via the cached native library."""
    return _lib().count_blobs(gray, min_area_frac=min_area_frac)


def available() -> bool:
    return _lib().available


__all__ = ["ImageOpsLib", "count_blobs", "available"]

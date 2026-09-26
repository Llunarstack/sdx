"""Optional Rust ``sdx_style_embed`` — InstantStyle CLIP vector math."""

from __future__ import annotations

import ctypes

import numpy as np

from sdx_native.native_tools import rust_style_embed_shared_library_path


class StyleEmbedLib:
    def __init__(self) -> None:
        self._lib: ctypes.CDLL | None = None
        p = rust_style_embed_shared_library_path()
        if p is None:
            return
        try:
            lib = ctypes.CDLL(str(p))
        except OSError:
            return
        lib.sdx_style_weighted_mean_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.sdx_style_weighted_mean_f32.restype = ctypes.c_int
        lib.sdx_style_subtract_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.sdx_style_subtract_f32.restype = ctypes.c_int
        lib.sdx_style_l2_normalize_rows_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        lib.sdx_style_l2_normalize_rows_f32.restype = ctypes.c_int
        self._lib = lib

    @property
    def available(self) -> bool:
        return self._lib is not None

    def weighted_mean(self, rows: np.ndarray, weights: np.ndarray) -> np.ndarray | None:
        if self._lib is None:
            return None
        mat = np.ascontiguousarray(rows, dtype=np.float32)
        w = np.ascontiguousarray(weights, dtype=np.float32).reshape(-1)
        if mat.ndim != 2 or w.shape[0] != mat.shape[0]:
            return None
        out = np.empty((mat.shape[1],), dtype=np.float32)
        rc = self._lib.sdx_style_weighted_mean_f32(
            mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(mat.shape[0]),
            ctypes.c_size_t(mat.shape[1]),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return out if rc == 0 else None

    def subtract(self, image: np.ndarray, content: np.ndarray, strength: float = 1.0) -> np.ndarray | None:
        if self._lib is None:
            return None
        a = np.ascontiguousarray(image, dtype=np.float32).reshape(-1)
        b = np.ascontiguousarray(content, dtype=np.float32).reshape(-1)
        if a.shape != b.shape:
            return None
        out = np.empty_like(a)
        rc = self._lib.sdx_style_subtract_f32(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(a.size),
            ctypes.c_float(strength),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return out if rc == 0 else None

    def l2_normalize_rows(self, rows: np.ndarray) -> np.ndarray | None:
        if self._lib is None:
            return None
        mat = np.ascontiguousarray(rows, dtype=np.float32)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        if mat.ndim != 2:
            return None
        out = mat.copy()
        rc = self._lib.sdx_style_l2_normalize_rows_f32(
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(out.shape[0]),
            ctypes.c_size_t(out.shape[1]),
        )
        return out if rc == 0 else None


_LIB: StyleEmbedLib | None = None


def _lib() -> StyleEmbedLib:
    global _LIB
    if _LIB is None:
        _LIB = StyleEmbedLib()
    return _LIB


def weighted_mean(rows: np.ndarray, weights: np.ndarray) -> np.ndarray | None:
    return _lib().weighted_mean(rows, weights)


def subtract(image: np.ndarray, content: np.ndarray, strength: float = 1.0) -> np.ndarray | None:
    return _lib().subtract(image, content, strength)


def l2_normalize_rows(rows: np.ndarray) -> np.ndarray | None:
    return _lib().l2_normalize_rows(rows)


def available() -> bool:
    return _lib().available


__all__ = ["StyleEmbedLib", "weighted_mean", "subtract", "l2_normalize_rows", "available"]

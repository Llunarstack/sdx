"""
Optional image metrics for uint8 HWC images: CPU ``sdx_image_metrics`` (luma, clipping, Laplacian variance,
component count) and CUDA ``sdx_cuda_image_metrics`` (luma, clipping). Every ``maybe_*`` returns ``None``
when its library is not built.
"""

from __future__ import annotations

import ctypes
import functools

import numpy as np

from sdx_native.native_tools import load_shared_library

_C_HWC = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]


@functools.cache
def _cpu() -> ctypes.CDLL | None:
    lib = load_shared_library("sdx_image_metrics")
    if lib is None:
        return None
    for fn, extra in (
        ("sdx_image_mean_luma_u8", [ctypes.POINTER(ctypes.c_double)]),
        ("sdx_image_clip_ratio_u8", [ctypes.c_ubyte, ctypes.c_ubyte, ctypes.POINTER(ctypes.c_double)]),
        ("sdx_image_laplacian_var_u8", [ctypes.POINTER(ctypes.c_double)]),
        ("sdx_image_count_components_u8", [ctypes.c_ubyte, ctypes.c_int, ctypes.c_int]),
    ):
        getattr(lib, fn).argtypes = _C_HWC + extra
        getattr(lib, fn).restype = ctypes.c_int
    return lib


@functools.cache
def _cuda() -> ctypes.CDLL | None:
    lib = load_shared_library("sdx_cuda_image_metrics")
    if lib is not None:
        lib.sdx_cuda_luma_stats_u8_host.argtypes = _C_HWC + [
            ctypes.c_ubyte,
            ctypes.c_ubyte,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.sdx_cuda_luma_stats_u8_host.restype = ctypes.c_int
    return lib


def _hwc(hwc: np.ndarray) -> tuple[np.ndarray, list]:
    """Contiguous uint8 copy plus its ctypes args; callers must keep the array alive while the pointer is used."""
    if hwc.ndim != 3:
        raise ValueError("expected uint8 ndarray shape (H, W, C)")
    arr = np.ascontiguousarray(hwc, dtype=np.uint8)
    return arr, [arr.ctypes.data_as(ctypes.c_void_p), *arr.shape]


def maybe_image_stats_native(hwc: np.ndarray, *, clip_low: int = 2, clip_high: int = 253) -> dict[str, float] | None:
    """``mean_luma`` / ``clip_ratio`` / ``laplacian_var`` via the CPU library."""
    lib = _cpu()
    if lib is None:
        return None
    _arr, args = _hwc(hwc)
    mean, ratio, lap = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
    if (
        lib.sdx_image_mean_luma_u8(*args, ctypes.byref(mean))
        or lib.sdx_image_clip_ratio_u8(*args, clip_low, clip_high, ctypes.byref(ratio))
        or lib.sdx_image_laplacian_var_u8(*args, ctypes.byref(lap))
    ):
        raise RuntimeError("sdx_image_metrics call failed")
    return {"mean_luma": mean.value, "clip_ratio": ratio.value, "laplacian_var": lap.value}


def maybe_image_luma_stats_cuda(hwc: np.ndarray, *, clip_low: int = 2, clip_high: int = 253) -> dict[str, float] | None:
    """``mean_luma`` / ``clip_ratio`` via the CUDA library."""
    lib = _cuda()
    if lib is None:
        return None
    _arr, args = _hwc(hwc)
    mean, ratio = ctypes.c_float(), ctypes.c_float()
    rc = lib.sdx_cuda_luma_stats_u8_host(*args, clip_low, clip_high, ctypes.byref(mean), ctypes.byref(ratio))
    if rc != 0:
        raise RuntimeError(f"sdx_cuda_luma_stats_u8_host failed (code {rc})")
    return {"mean_luma": float(mean.value), "clip_ratio": float(ratio.value)}


def maybe_count_components_native(
    hwc: np.ndarray,
    *,
    threshold: int = 140,
    min_area: int = 16,
    max_area: int = 0,
) -> int | None:
    """Bright connected components with ``min_area <= area <= max_area`` (0 = unbounded)."""
    lib = _cpu()
    if lib is None:
        return None
    _arr, args = _hwc(hwc)
    rc = lib.sdx_image_count_components_u8(*args, threshold, int(min_area), int(max_area))
    if rc < 0:
        raise RuntimeError("sdx_image_count_components_u8 failed")
    return int(rc)

"""
ctypes bindings for the optional native libraries: ``sdx_cpu`` / ``sdx_cuda`` (``native/cpp``) and ``sdx_rs``
(``native/rust``). Every ``maybe_*`` / kernel wrapper returns ``None`` when its library is not built, so callers
fall back to NumPy / PyTorch; C error codes raise ``RuntimeError`` / ``ValueError``.
"""

from __future__ import annotations

import ctypes
import functools

import numpy as np

from sdx_native.native_tools import load_shared_library

_P = ctypes.c_void_p
_I = ctypes.c_int
_U8 = ctypes.c_ubyte
_SZ = ctypes.c_size_t
_I64 = ctypes.c_int64
_F32 = ctypes.c_float
_F64 = ctypes.c_double
_F64P = ctypes.POINTER(ctypes.c_double)
_HWC = [_P, _I, _I, _I]

_SIGNATURES: dict[str, dict[str, tuple[list, type]]] = {
    "sdx_cpu": {
        "sdx_it_finalize_path": ([_P, _I64, _I, _I, _P, _I64], _I64),
        "sdx_squared_cosine_betas_v2": ([_I, _F64, _P, _I64], _I64),
        "sdx_image_mean_luma_u8": ([*_HWC, _F64P], _I),
        "sdx_image_clip_ratio_u8": ([*_HWC, _U8, _U8, _F64P], _I),
        "sdx_image_laplacian_var_u8": ([*_HWC, _F64P], _I),
        "sdx_image_count_components_u8": ([*_HWC, _U8, _I, _I], _I),
        "sdx_mask_to_patch_weights_f32": ([_P, _P, _I, _I, _I, _I, _I], _I),
    },
    "sdx_cuda": {
        "sdx_cuda_gaussian_blur_latent_f32": ([_P, _P, _I, _I, _I, _I, _F32], _I),
        "sdx_cuda_luma_stats_u8_host": ([*_HWC, _U8, _U8, ctypes.POINTER(_F32), ctypes.POINTER(_F32)], _I),
        "sdx_cuda_l2_normalize_rows_f32_host": ([_P, _I, _I, _F32], _I),
        "sdx_cuda_style_pick_best_f32_host": ([_P, _P, _I, _I, ctypes.POINTER(_I), ctypes.POINTER(_F32)], _I),
    },
    "sdx_rs": {
        "sdx_alpha_cumprod_f64": ([_P, _P, _SZ], ctypes.c_int32),
        "sdx_snr_from_alpha_cumprod_f64": ([_P, _P, _SZ], ctypes.c_int32),
        "sdx_linear_beta_schedule_f64": ([_P, _SZ, _F64, _F64], ctypes.c_int32),
        "sdx_squaredcos_beta_schedule_v2_f64": ([_P, _SZ, _F64], ctypes.c_int32),
        "sdx_cosine_beta_schedule_f64": ([_P, _SZ], ctypes.c_int32),
        "sdx_token_jaccard_utf8": ([_P, _SZ, _P, _SZ], ctypes.c_int32),
        "sdx_quality_highlight_frac_u8": ([_P, _SZ, _SZ, _F32], _F64),
        "sdx_count_blobs_f32": ([_P, _SZ, _SZ, _F32], _I64),
    },
}


@functools.cache
def _lib(name: str) -> ctypes.CDLL | None:
    """The loaded library with signatures set, or ``None`` if not built or missing a symbol (stale build)."""
    lib = load_shared_library(name)
    sigs = _SIGNATURES[name]
    if lib is None or not all(hasattr(lib, fn) for fn in sigs):
        return None
    for fn, (argtypes, restype) in sigs.items():
        getattr(lib, fn).argtypes = argtypes
        getattr(lib, fn).restype = restype
    return lib


def cpu_available() -> bool:
    return _lib("sdx_cpu") is not None


def cuda_available() -> bool:
    return _lib("sdx_cuda") is not None


def rust_available() -> bool:
    return _lib("sdx_rs") is not None


def _ptr(a: np.ndarray) -> int:
    return a.ctypes.data


# ------------------------------------------------------------------ diffusion schedules / timesteps


def _rust_into(fn: str, out: np.ndarray, *args) -> np.ndarray | None:
    lib = _lib("sdx_rs")
    if lib is None:
        return None
    rc = getattr(lib, fn)(*args)
    if rc != 0:
        raise RuntimeError(f"{fn} failed (rc={rc})")
    return out


def _rust_elementwise(fn: str, x: np.ndarray) -> np.ndarray | None:
    src = np.ascontiguousarray(x, dtype=np.float64)
    out = np.empty(len(src), dtype=np.float64)
    return _rust_into(fn, out, _ptr(src), _ptr(out), len(src))


def _rust_schedule(fn: str, n: int, *params: float) -> np.ndarray | None:
    out = np.empty(n, dtype=np.float64)
    return _rust_into(fn, out, _ptr(out), n, *params)


def maybe_alpha_cumprod_rust(betas: np.ndarray) -> np.ndarray | None:
    return _rust_elementwise("sdx_alpha_cumprod_f64", betas)


def maybe_snr_from_alpha_cumprod_rust(alpha_cumprod: np.ndarray) -> np.ndarray | None:
    return _rust_elementwise("sdx_snr_from_alpha_cumprod_f64", alpha_cumprod)


def maybe_linear_beta_schedule_rust(n: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> np.ndarray | None:
    return _rust_schedule("sdx_linear_beta_schedule_f64", n, beta_start, beta_end)


def maybe_squaredcos_beta_schedule_v2_rust(n: int, max_beta: float = 0.999) -> np.ndarray | None:
    return _rust_schedule("sdx_squaredcos_beta_schedule_v2_f64", n, max_beta)


def maybe_cosine_beta_schedule_rust(n: int) -> np.ndarray | None:
    return _rust_schedule("sdx_cosine_beta_schedule_f64", n)


def squared_cosine_betas_v2_native(n: int, max_beta: float = 0.999) -> np.ndarray | None:
    """``(n,)`` float64 squared-cosine v2 betas from ``sdx_cpu``, or ``None`` if unavailable / on error."""
    lib = _lib("sdx_cpu")
    if n < 1 or lib is None:
        return None
    out = np.empty(n, dtype=np.float64)
    return out if lib.sdx_squared_cosine_betas_v2(int(n), float(max_beta), _ptr(out), int(n)) == n else None


def finalize_inference_timesteps_native(raw: np.ndarray, target_len: int, num_train: int) -> np.ndarray | None:
    """Native :func:`diffusion.inference_timesteps._resample_length_numpy`; ``None`` if unavailable or on error."""
    lib = _lib("sdx_cpu")
    if target_len <= 0 or num_train < 1 or lib is None:
        return None
    raw_i = np.ascontiguousarray(raw, dtype=np.int64).ravel()
    out_cap = max(int(target_len), 2) + 32
    out = np.empty(out_cap, dtype=np.int64)
    n = lib.sdx_it_finalize_path(_ptr(raw_i), raw_i.size, int(target_len), int(num_train), _ptr(out), out_cap)
    return None if n < 0 else out[:n].copy()


# ------------------------------------------------------------------ image kernels


def _hwc(hwc: np.ndarray) -> tuple[np.ndarray, list]:
    """Contiguous uint8 copy plus its ctypes args; keep the array alive while the pointer is used."""
    if hwc.ndim != 3:
        raise ValueError("expected uint8 ndarray shape (H, W, C)")
    arr = np.ascontiguousarray(hwc, dtype=np.uint8)
    return arr, [_ptr(arr), *arr.shape]


def maybe_image_stats_native(hwc: np.ndarray, *, clip_low: int = 2, clip_high: int = 253) -> dict[str, float] | None:
    """``mean_luma`` / ``clip_ratio`` / ``laplacian_var`` of a uint8 HWC image via ``sdx_cpu``."""
    lib = _lib("sdx_cpu")
    if lib is None:
        return None
    _arr, args = _hwc(hwc)
    mean, ratio, lap = _F64(), _F64(), _F64()
    if (
        lib.sdx_image_mean_luma_u8(*args, ctypes.byref(mean))
        or lib.sdx_image_clip_ratio_u8(*args, clip_low, clip_high, ctypes.byref(ratio))
        or lib.sdx_image_laplacian_var_u8(*args, ctypes.byref(lap))
    ):
        raise RuntimeError("sdx_cpu image metrics call failed")
    return {"mean_luma": mean.value, "clip_ratio": ratio.value, "laplacian_var": lap.value}


def maybe_image_luma_stats_cuda(hwc: np.ndarray, *, clip_low: int = 2, clip_high: int = 253) -> dict[str, float] | None:
    """``mean_luma`` / ``clip_ratio`` via ``sdx_cuda``."""
    lib = _lib("sdx_cuda")
    if lib is None:
        return None
    _arr, args = _hwc(hwc)
    mean, ratio = _F32(), _F32()
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
    """Count of 4-connected components with luma <= ``threshold`` and ``min_area <= area <= max_area`` (0 = no cap)."""
    lib = _lib("sdx_cpu")
    if lib is None:
        return None
    _arr, args = _hwc(hwc)
    rc = lib.sdx_image_count_components_u8(*args, threshold, int(min_area), int(max_area))
    if rc < 0:
        raise RuntimeError("sdx_image_count_components_u8 failed")
    return int(rc)


def maybe_mask_to_patch_weights_native(mask: np.ndarray, ph: int, pw: int) -> np.ndarray | None:
    """(B, 1, H, W) mask -> (B, ph*pw) cell means (``utils.training.objectives``)."""
    lib = _lib("sdx_cpu")
    if lib is None:
        return None
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must be (B, 1, H, W)")
    mask = np.ascontiguousarray(mask, dtype=np.float32)
    b, _, h, w = mask.shape
    out = np.empty((b, ph * pw), dtype=np.float32)
    rc = lib.sdx_mask_to_patch_weights_f32(_ptr(mask), _ptr(out), b, h, w, ph, pw)
    if rc == -3:
        raise ValueError(f"mask dimensions ({h}, {w}) not divisible by patch grid ({ph}, {pw})")
    if rc != 0:
        raise RuntimeError(f"sdx_mask_to_patch_weights_f32 failed (rc={rc})")
    return out


def maybe_gaussian_blur_cuda(x: np.ndarray, sigma: float) -> np.ndarray | None:
    """Blurred copy of a (B, C, H, W) array (``diffusion.sampling_utils.gaussian_blur_latent``)."""
    lib = _lib("sdx_cuda")
    if lib is None:
        return None
    if x.ndim != 4:
        raise ValueError("expected 4-D array (B, C, H, W)")
    x = np.ascontiguousarray(x, dtype=np.float32)
    dst = np.empty_like(x)
    rc = lib.sdx_cuda_gaussian_blur_latent_f32(_ptr(x), _ptr(dst), *x.shape, float(sigma))
    if rc != 0:
        raise RuntimeError(f"sdx_cuda_gaussian_blur_latent_f32 failed (rc={rc})")
    return dst


def count_blobs(gray: np.ndarray, min_area_frac: float = 0.005) -> int | None:
    """Foreground blob count of a 2-D grayscale array in [0, 1] (``scripts/tools/evals.py`` contract)."""
    lib = _lib("sdx_rs")
    arr = np.ascontiguousarray(gray, dtype=np.float32)
    if lib is None or arr.ndim != 2:
        return None
    n = lib.sdx_count_blobs_f32(_ptr(arr), *arr.shape, float(min_area_frac))
    return int(n) if n >= 0 else None


def highlight_frac(rgb: np.ndarray, thr: float = 245.0) -> float | None:
    """Fraction of pixels whose luma is >= ``thr``; ``None`` if unavailable or the image is too small."""
    lib = _lib("sdx_rs")
    if lib is None or rgb.ndim != 3 or rgb.shape[2] < 3 or min(rgb.shape[:2]) < 3:
        return None
    arr = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
    v = float(lib.sdx_quality_highlight_frac_u8(_ptr(arr), arr.shape[0], arr.shape[1], thr))
    return v if v >= 0.0 else None


# ------------------------------------------------------------------ style genome


def maybe_token_jaccard(a: str, b: str) -> float | None:
    """Jaccard 0-1 on comma/whitespace tokens (``utils.prompt.style_native.text_overlap``)."""
    lib = _lib("sdx_rs")
    if lib is None:
        return None
    a_b, b_b = (a or "").encode("utf-8"), (b or "").encode("utf-8")
    bp = lib.sdx_token_jaccard_utf8(a_b, len(a_b), b_b, len(b_b))
    return bp / 10000.0 if bp >= 0 else None


def maybe_pick_best_style_embedding(
    query: np.ndarray,
    candidates: np.ndarray,
    *,
    eps: float = 1e-8,
) -> tuple[int, float] | None:
    """``query`` (dim,), ``candidates`` (n, dim) -> ``(best_index, cosine)`` on the GPU."""
    lib = _lib("sdx_cuda")
    if lib is None or query.ndim != 1 or candidates.ndim != 2 or query.shape[0] != candidates.shape[1]:
        return None
    # Owned C-order copies: the kernel normalizes them in place.
    q = np.array(query, dtype=np.float32, order="C").reshape(1, -1)
    c = np.array(candidates, dtype=np.float32, order="C")
    for m in (q, c):
        if lib.sdx_cuda_l2_normalize_rows_f32_host(_ptr(m), m.shape[0], m.shape[1], eps) != 0:
            return None
    out_i, out_s = _I(-1), _F32(0.0)
    rc = lib.sdx_cuda_style_pick_best_f32_host(
        _ptr(q), _ptr(c), c.shape[0], c.shape[1], ctypes.byref(out_i), ctypes.byref(out_s)
    )
    if rc != 0:
        return None
    return int(out_i.value), float(out_s.value)

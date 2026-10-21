"""
Optional Rust ``sdx_diffusion_math`` cdylib wrapper: VP-diffusion math (alpha_cumprod, SNR, beta schedules)
matching ``diffusion/schedules.py`` and ``diffusion/schedules.py``. Every ``maybe_*`` returns ``None`` when the
library is not built (``cd native/rust/sdx-diffusion-math && cargo build --release``).
"""

from __future__ import annotations

import ctypes
import functools

import numpy as np

from sdx_native.native_tools import load_shared_library

_F64P = ctypes.POINTER(ctypes.c_double)


@functools.cache
def _lib() -> ctypes.CDLL | None:
    lib = load_shared_library("sdx_diffusion_math")
    if lib is None:
        return None
    for fn, argtypes in (
        ("sdx_alpha_cumprod_f64", [_F64P, _F64P, ctypes.c_size_t]),
        ("sdx_snr_from_alpha_cumprod_f64", [_F64P, _F64P, ctypes.c_size_t]),
        ("sdx_linear_beta_schedule_f64", [_F64P, ctypes.c_size_t, ctypes.c_double, ctypes.c_double]),
        ("sdx_squaredcos_beta_schedule_v2_f64", [_F64P, ctypes.c_size_t, ctypes.c_double]),
        ("sdx_cosine_beta_schedule_f64", [_F64P, ctypes.c_size_t]),
    ):
        getattr(lib, fn).argtypes = argtypes
        getattr(lib, fn).restype = ctypes.c_int32
    return lib


def _elementwise(fn: str, x: np.ndarray) -> np.ndarray | None:
    lib = _lib()
    if lib is None:
        return None
    src = np.ascontiguousarray(x, dtype=np.float64)
    out = np.empty(len(src), dtype=np.float64)
    rc = getattr(lib, fn)(src.ctypes.data_as(_F64P), out.ctypes.data_as(_F64P), len(src))
    if rc != 0:
        raise RuntimeError(f"{fn} failed (rc={rc})")
    return out


def _schedule(fn: str, n: int, *params: float) -> np.ndarray | None:
    lib = _lib()
    if lib is None:
        return None
    out = np.empty(n, dtype=np.float64)
    rc = getattr(lib, fn)(out.ctypes.data_as(_F64P), n, *params)
    if rc != 0:
        raise RuntimeError(f"{fn} failed (rc={rc})")
    return out


def maybe_alpha_cumprod_rust(betas: np.ndarray) -> np.ndarray | None:
    return _elementwise("sdx_alpha_cumprod_f64", betas)


def maybe_snr_from_alpha_cumprod_rust(alpha_cumprod: np.ndarray) -> np.ndarray | None:
    return _elementwise("sdx_snr_from_alpha_cumprod_f64", alpha_cumprod)


def maybe_linear_beta_schedule_rust(n: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> np.ndarray | None:
    return _schedule("sdx_linear_beta_schedule_f64", n, beta_start, beta_end)


def maybe_squaredcos_beta_schedule_v2_rust(n: int, max_beta: float = 0.999) -> np.ndarray | None:
    return _schedule("sdx_squaredcos_beta_schedule_v2_f64", n, max_beta)


def maybe_cosine_beta_schedule_rust(n: int) -> np.ndarray | None:
    return _schedule("sdx_cosine_beta_schedule_f64", n)

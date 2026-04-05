"""
Optional Rust ``sdx_diffusion_math`` cdylib wrapper — VP-diffusion math
(alpha_cumprod, SNR, beta schedules) without NumPy overhead.

Matches ``diffusion/snr_utils.py`` and ``diffusion/schedules.py``.

Build:
    cd native/rust/sdx-diffusion-math
    cargo build --release

Falls back to NumPy when the library is not built.
"""

from __future__ import annotations

import ctypes
from typing import Optional

import numpy as np

from sdx_native.native_tools import rust_diffusion_math_shared_library_path


class DiffusionMathLib:
    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        p = rust_diffusion_math_shared_library_path()
        if p is None:
            return
        try:
            lib = ctypes.CDLL(str(p))
        except OSError:
            return

        _f64p = ctypes.POINTER(ctypes.c_double)

        lib.sdx_alpha_cumprod_f64.argtypes = [_f64p, _f64p, ctypes.c_size_t]
        lib.sdx_alpha_cumprod_f64.restype = ctypes.c_int32

        lib.sdx_snr_from_alpha_cumprod_f64.argtypes = [_f64p, _f64p, ctypes.c_size_t]
        lib.sdx_snr_from_alpha_cumprod_f64.restype = ctypes.c_int32

        lib.sdx_linear_beta_schedule_f64.argtypes = [
            _f64p, ctypes.c_size_t, ctypes.c_double, ctypes.c_double
        ]
        lib.sdx_linear_beta_schedule_f64.restype = ctypes.c_int32

        lib.sdx_squaredcos_beta_schedule_v2_f64.argtypes = [
            _f64p, ctypes.c_size_t, ctypes.c_double
        ]
        lib.sdx_squaredcos_beta_schedule_v2_f64.restype = ctypes.c_int32

        lib.sdx_cosine_beta_schedule_f64.argtypes = [_f64p, ctypes.c_size_t]
        lib.sdx_cosine_beta_schedule_f64.restype = ctypes.c_int32

        self._lib = lib

    @property
    def available(self) -> bool:
        return self._lib is not None

    def _out_f64(self, n: int) -> tuple[np.ndarray, ctypes.POINTER]:
        arr = np.empty(n, dtype=np.float64)
        ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return arr, ptr

    def alpha_cumprod(self, betas: np.ndarray) -> np.ndarray:
        betas = np.ascontiguousarray(betas, dtype=np.float64)
        n = len(betas)
        out, out_ptr = self._out_f64(n)
        b_ptr = betas.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        rc = self._lib.sdx_alpha_cumprod_f64(b_ptr, out_ptr, ctypes.c_size_t(n))
        if rc != 0:
            raise RuntimeError(f"sdx_alpha_cumprod_f64 failed (rc={rc})")
        return out

    def snr_from_alpha_cumprod(self, alpha_cumprod: np.ndarray) -> np.ndarray:
        ac = np.ascontiguousarray(alpha_cumprod, dtype=np.float64)
        n = len(ac)
        out, out_ptr = self._out_f64(n)
        ac_ptr = ac.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        rc = self._lib.sdx_snr_from_alpha_cumprod_f64(ac_ptr, out_ptr, ctypes.c_size_t(n))
        if rc != 0:
            raise RuntimeError(f"sdx_snr_from_alpha_cumprod_f64 failed (rc={rc})")
        return out

    def linear_beta_schedule(
        self, n: int, beta_start: float = 1e-4, beta_end: float = 2e-2
    ) -> np.ndarray:
        out, out_ptr = self._out_f64(n)
        rc = self._lib.sdx_linear_beta_schedule_f64(
            out_ptr,
            ctypes.c_size_t(n),
            ctypes.c_double(beta_start),
            ctypes.c_double(beta_end),
        )
        if rc != 0:
            raise RuntimeError(f"sdx_linear_beta_schedule_f64 failed (rc={rc})")
        return out

    def squaredcos_beta_schedule_v2(self, n: int, max_beta: float = 0.999) -> np.ndarray:
        out, out_ptr = self._out_f64(n)
        rc = self._lib.sdx_squaredcos_beta_schedule_v2_f64(
            out_ptr, ctypes.c_size_t(n), ctypes.c_double(max_beta)
        )
        if rc != 0:
            raise RuntimeError(f"sdx_squaredcos_beta_schedule_v2_f64 failed (rc={rc})")
        return out

    def cosine_beta_schedule(self, n: int) -> np.ndarray:
        out, out_ptr = self._out_f64(n)
        rc = self._lib.sdx_cosine_beta_schedule_f64(out_ptr, ctypes.c_size_t(n))
        if rc != 0:
            raise RuntimeError(f"sdx_cosine_beta_schedule_f64 failed (rc={rc})")
        return out


_LIB: Optional[DiffusionMathLib] = None


def _get_lib() -> DiffusionMathLib:
    global _LIB
    if _LIB is None:
        _LIB = DiffusionMathLib()
    return _LIB


def maybe_alpha_cumprod_rust(betas: np.ndarray) -> Optional[np.ndarray]:
    lib = _get_lib()
    return lib.alpha_cumprod(betas) if lib.available else None


def maybe_snr_from_alpha_cumprod_rust(alpha_cumprod: np.ndarray) -> Optional[np.ndarray]:
    lib = _get_lib()
    return lib.snr_from_alpha_cumprod(alpha_cumprod) if lib.available else None


def maybe_linear_beta_schedule_rust(
    n: int, beta_start: float = 1e-4, beta_end: float = 2e-2
) -> Optional[np.ndarray]:
    lib = _get_lib()
    return lib.linear_beta_schedule(n, beta_start, beta_end) if lib.available else None


def maybe_squaredcos_beta_schedule_v2_rust(
    n: int, max_beta: float = 0.999
) -> Optional[np.ndarray]:
    lib = _get_lib()
    return lib.squaredcos_beta_schedule_v2(n, max_beta) if lib.available else None


def maybe_cosine_beta_schedule_rust(n: int) -> Optional[np.ndarray]:
    lib = _get_lib()
    return lib.cosine_beta_schedule(n) if lib.available else None

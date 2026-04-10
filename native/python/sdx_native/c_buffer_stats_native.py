"""
Optional shared library ``libsdx_c_buffer_stats`` (compile ``native/c/src/sdx_c_buffer_stats.c``).

Falls back to Python byte scans when the DLL is absent.
"""

from __future__ import annotations

import ctypes
from typing import Optional, Tuple

from sdx_native.native_tools import c_buffer_stats_shared_library_path


class BufferStatsLib:
    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        p = c_buffer_stats_shared_library_path()
        if p is None:
            return
        try:
            dll = ctypes.CDLL(str(p))
            dll.sdx_c_count_newlines_u8.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
            dll.sdx_c_count_newlines_u8.restype = ctypes.c_size_t
            dll.sdx_c_sum_bytes_u8.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
            dll.sdx_c_sum_bytes_u8.restype = ctypes.c_uint64
            self._lib = dll
        except OSError:
            self._lib = None

    @property
    def available(self) -> bool:
        return self._lib is not None

    def count_newlines(self, buf: bytes) -> int:
        if not self._lib:
            raise RuntimeError("sdx_c_buffer_stats not built")
        if not buf:
            return 0
        b = ctypes.create_string_buffer(buf, len(buf))
        void_p = ctypes.cast(b, ctypes.c_void_p)
        return int(self._lib.sdx_c_count_newlines_u8(void_p, len(buf)))

    def sum_bytes(self, buf: bytes) -> int:
        if not self._lib:
            raise RuntimeError("sdx_c_buffer_stats not built")
        if not buf:
            return 0
        b = ctypes.create_string_buffer(buf, len(buf))
        void_p = ctypes.cast(b, ctypes.c_void_p)
        return int(self._lib.sdx_c_sum_bytes_u8(void_p, len(buf)))


_LIB: Optional[BufferStatsLib] = None


def get_buffer_stats_lib() -> BufferStatsLib:
    global _LIB
    if _LIB is None:
        _LIB = BufferStatsLib()
    return _LIB


def maybe_count_newlines_native(buf: bytes) -> Optional[int]:
    lib = get_buffer_stats_lib()
    if not lib.available:
        return None
    try:
        return lib.count_newlines(buf)
    except Exception:
        return None


def maybe_sum_bytes_native(buf: bytes) -> Optional[int]:
    lib = get_buffer_stats_lib()
    if not lib.available:
        return None
    try:
        return lib.sum_bytes(buf)
    except Exception:
        return None


def count_newlines_py(buf: bytes) -> int:
    return buf.count(b"\n")


def sum_bytes_py(buf: bytes) -> int:
    return int(sum(buf))


def newline_and_sum(buf: bytes) -> Tuple[int, int]:
    """Prefer C when built; else Python."""
    n = maybe_count_newlines_native(buf)
    s = maybe_sum_bytes_native(buf)
    if n is not None and s is not None:
        return n, s
    return count_newlines_py(buf), sum_bytes_py(buf)

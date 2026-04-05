"""
``sdx_line_stats`` C++ helper: fast byte + newline count for huge JSONL (no JSON parse).

Build: ``cmake -S native/cpp -B native/cpp/build && cmake --build native/cpp/build``
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Optional, Tuple

from sdx_native.native_tools import line_stats_shared_library_path


class LineStatsLib:
    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        p = line_stats_shared_library_path()
        if p is None:
            return
        try:
            self._lib = ctypes.CDLL(str(p))
        except OSError:
            return
        self._lib.sdx_count_file_bytes_newlines.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_ulonglong),
        ]
        self._lib.sdx_count_file_bytes_newlines.restype = ctypes.c_int

    @property
    def available(self) -> bool:
        return self._lib is not None

    def count_bytes_newlines(self, path: Path) -> Tuple[int, int]:
        if not self._lib:
            raise RuntimeError("sdx_line_stats not built")
        b = ctypes.c_ulonglong(0)
        n = ctypes.c_ulonglong(0)
        rc = self._lib.sdx_count_file_bytes_newlines(str(path).encode("utf-8"), ctypes.byref(b), ctypes.byref(n))
        if rc != 0:
            raise OSError(f"sdx_count_file_bytes_newlines failed for {path}")
        return int(b.value), int(n.value)


_LIB: Optional[LineStatsLib] = None


def get_line_stats_lib() -> LineStatsLib:
    global _LIB
    if _LIB is None:
        _LIB = LineStatsLib()
    return _LIB


def count_file_bytes_newlines(path: Path) -> Optional[Tuple[int, int]]:
    """Return ``(bytes, newline_count)`` if the shared library is built; else ``None``."""
    lib = get_line_stats_lib()
    if not lib.available:
        return None
    return lib.count_bytes_newlines(path)

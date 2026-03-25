"""
Optional ``sdx_fnv64_file`` DLL: same FNV-1a 64 + newline semantics as :func:`native_tools.fnv1a64_file`.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Optional, Tuple

from sdx_native.native_tools import fnv64_file_shared_library_path


class Fnv64Lib:
    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        p = fnv64_file_shared_library_path()
        if p is None:
            return
        try:
            self._lib = ctypes.CDLL(str(p))
        except OSError:
            return
        self._lib.sdx_fnv1a64_file_stream.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_ulonglong),
        ]
        self._lib.sdx_fnv1a64_file_stream.restype = ctypes.c_int

    @property
    def available(self) -> bool:
        return self._lib is not None

    def hash_bytes_newlines(self, path: Path) -> Tuple[int, int, int]:
        if not self._lib:
            raise RuntimeError("sdx_fnv64_file not built")
        h = ctypes.c_ulonglong(0)
        b = ctypes.c_ulonglong(0)
        n = ctypes.c_ulonglong(0)
        rc = self._lib.sdx_fnv1a64_file_stream(str(path).encode("utf-8"), ctypes.byref(h), ctypes.byref(b), ctypes.byref(n))
        if rc != 0:
            raise OSError(f"sdx_fnv1a64_file_stream failed for {path}")
        return int(h.value), int(b.value), int(n.value)


_LIB: Optional[Fnv64Lib] = None


def get_fnv64_lib() -> Fnv64Lib:
    global _LIB
    if _LIB is None:
        _LIB = Fnv64Lib()
    return _LIB


def maybe_fnv1a64_file_native(path: Path) -> Optional[Tuple[int, int, int]]:
    """Return ``(hash64, bytes, newlines)`` if DLL exists, else ``None``."""
    lib = get_fnv64_lib()
    if not lib.available:
        return None
    return lib.hash_bytes_newlines(path)

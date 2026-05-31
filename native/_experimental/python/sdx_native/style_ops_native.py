"""
Rust ``sdx_prompt_ops`` style helpers: Jaccard overlap, FNV fingerprint, multi-axis merge.
"""

from __future__ import annotations

import ctypes
from typing import Optional, Sequence

from sdx_native.native_tools import rust_prompt_ops_shared_library_path


class StyleOpsLib:
    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        self._has_jaccard = False
        self._has_fnv = False
        self._has_merge_axes = False
        p = rust_prompt_ops_shared_library_path()
        if p is None:
            return
        try:
            lib = ctypes.CDLL(str(p))
        except OSError:
            return
        if hasattr(lib, "sdx_token_jaccard_utf8"):
            lib.sdx_token_jaccard_utf8.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            lib.sdx_token_jaccard_utf8.restype = ctypes.c_int32
            self._has_jaccard = True
        if hasattr(lib, "sdx_fnv1a64_utf8"):
            lib.sdx_fnv1a64_utf8.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            lib.sdx_fnv1a64_utf8.restype = ctypes.c_uint64
            self._has_fnv = True
        if hasattr(lib, "sdx_merge_style_axes_utf8"):
            lib.sdx_merge_style_axes_utf8.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            lib.sdx_merge_style_axes_utf8.restype = ctypes.c_int64
            self._has_merge_axes = True
        self._lib = lib

    @property
    def available(self) -> bool:
        return self._lib is not None and (self._has_jaccard or self._has_fnv or self._has_merge_axes)


_LIB: Optional[StyleOpsLib] = None


def get_style_ops_lib() -> StyleOpsLib:
    global _LIB
    if _LIB is None:
        _LIB = StyleOpsLib()
    return _LIB


def maybe_token_jaccard(a: str, b: str) -> Optional[float]:
    """Jaccard 0–1 on comma/whitespace tokens; ``None`` if Rust not built."""
    lib = get_style_ops_lib()
    if not lib.available or not lib._has_jaccard:
        return None
    a_b = (a or "").encode("utf-8")
    b_b = (b or "").encode("utf-8")
    bp = lib._lib.sdx_token_jaccard_utf8(a_b, len(a_b), b_b, len(b_b))
    if bp < 0:
        return None
    return float(bp) / 10000.0


def maybe_fnv1a64(text: str) -> Optional[int]:
    lib = get_style_ops_lib()
    if not lib.available or not lib._has_fnv:
        return None
    b = (text or "").encode("utf-8")
    return int(lib._lib.sdx_fnv1a64_utf8(b, len(b)))


def maybe_merge_style_axes(parts: Sequence[str]) -> Optional[str]:
    lib = get_style_ops_lib()
    if not lib.available or not lib._has_merge_axes or not parts:
        return None
    chunks = [(p or "").encode("utf-8") for p in parts if (p or "").strip()][:8]
    if not chunks:
        return ""
    ptrs = (ctypes.c_void_p * len(chunks))()
    lens = (ctypes.c_size_t * len(chunks))()
    for i, c in enumerate(chunks):
        ptrs[i] = ctypes.cast(ctypes.c_char_p(c), ctypes.c_void_p)
        lens[i] = len(c)
    fn = lib._lib.sdx_merge_style_axes_utf8
    rc = fn(ptrs, lens, len(chunks), ctypes.c_void_p(), 0)
    if rc < 0:
        return None
    cap = int(rc) + 1
    buf = ctypes.create_string_buffer(cap)
    rc2 = fn(ptrs, lens, len(chunks), ctypes.cast(buf, ctypes.c_void_p), cap)
    if rc2 < 0:
        return None
    return buf.raw[: int(rc2)].decode("utf-8", errors="replace")


def merge_style_axes(parts: Sequence[str]) -> str:
    out = maybe_merge_style_axes(parts)
    if out is not None:
        return out
    from sdx_native.caption_csv_fast import merge_caption_csv

    acc = ""
    for p in parts:
        if (p or "").strip():
            acc = merge_caption_csv(acc, p)
    return acc


__all__ = [
    "StyleOpsLib",
    "get_style_ops_lib",
    "maybe_fnv1a64",
    "maybe_merge_style_axes",
    "maybe_token_jaccard",
    "merge_style_axes",
]

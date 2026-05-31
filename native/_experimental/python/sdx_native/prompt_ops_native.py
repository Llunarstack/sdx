"""
Optional Rust ``sdx_prompt_ops`` cdylib — fast caption merge and pos/neg filter.

Build::

    cd native/rust/sdx-prompt-ops && cargo build --release

Falls back to pure-Python implementations when the library is not built.
"""

from __future__ import annotations

import ctypes
from typing import Optional

from sdx_native.native_tools import rust_prompt_ops_shared_library_path


class PromptOpsLib:
    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        p = rust_prompt_ops_shared_library_path()
        if p is None:
            return
        try:
            lib = ctypes.CDLL(str(p))
        except OSError:
            return
        lib.sdx_filter_negative_utf8.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.sdx_filter_negative_utf8.restype = ctypes.c_int64
        lib.sdx_merge_caption_utf8.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.sdx_merge_caption_utf8.restype = ctypes.c_int64
        self._lib = lib

    @property
    def available(self) -> bool:
        return self._lib is not None


_LIB: Optional[PromptOpsLib] = None


def get_prompt_ops_lib() -> PromptOpsLib:
    global _LIB
    if _LIB is None:
        _LIB = PromptOpsLib()
    return _LIB


def maybe_filter_negative_by_positive(positive: str, negative: str) -> Optional[str]:
    """Rust filter when built; else ``None`` (caller uses Python)."""
    lib = get_prompt_ops_lib()
    if not lib.available:
        return None
    pos_b = (positive or "").encode("utf-8")
    neg_b = (negative or "").encode("utf-8")
    fn = lib._lib.sdx_filter_negative_utf8
    rc = fn(
        pos_b,
        len(pos_b),
        neg_b,
        len(neg_b),
        ctypes.c_void_p(),
        0,
    )
    if rc < 0:
        return None
    cap = int(rc) + 1
    buf = ctypes.create_string_buffer(cap)
    rc2 = fn(
        pos_b,
        len(pos_b),
        neg_b,
        len(neg_b),
        ctypes.cast(buf, ctypes.c_void_p),
        cap,
    )
    if rc2 < 0:
        return None
    return buf.raw[: int(rc2)].decode("utf-8", errors="replace")


def maybe_merge_caption_csv(a: str, b: str) -> Optional[str]:
    lib = get_prompt_ops_lib()
    if not lib.available:
        return None
    a_b = (a or "").encode("utf-8")
    b_b = (b or "").encode("utf-8")
    fn = lib._lib.sdx_merge_caption_utf8
    rc = fn(a_b, len(a_b), b_b, len(b_b), ctypes.c_void_p(), 0)
    if rc < 0:
        return None
    cap = int(rc) + 1
    buf = ctypes.create_string_buffer(cap)
    rc2 = fn(a_b, len(a_b), b_b, len(b_b), ctypes.cast(buf, ctypes.c_void_p), cap)
    if rc2 < 0:
        return None
    return buf.raw[: int(rc2)].decode("utf-8", errors="replace")


def merge_caption_csv(a: str, b: str) -> str:
    out = maybe_merge_caption_csv(a, b)
    if out is not None:
        return out
    from sdx_native.caption_csv_fast import merge_caption_csv as _py

    return _py(a, b)


__all__ = [
    "PromptOpsLib",
    "get_prompt_ops_lib",
    "maybe_filter_negative_by_positive",
    "maybe_merge_caption_csv",
    "merge_caption_csv",
]

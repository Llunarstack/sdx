"""Rust ``sdx_prompt_ops`` style helper: token Jaccard overlap (``utils.prompt.style_native.text_overlap``)."""

from __future__ import annotations

import ctypes
import functools

from sdx_native.native_tools import load_shared_library


@functools.cache
def _lib() -> ctypes.CDLL | None:
    lib = load_shared_library("sdx_prompt_ops")
    if lib is None or not hasattr(lib, "sdx_token_jaccard_utf8"):
        return None
    lib.sdx_token_jaccard_utf8.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    lib.sdx_token_jaccard_utf8.restype = ctypes.c_int32
    return lib


def available() -> bool:
    return _lib() is not None


def maybe_token_jaccard(a: str, b: str) -> float | None:
    """Jaccard 0-1 on comma/whitespace tokens; ``None`` if the Rust library is not built."""
    lib = _lib()
    if lib is None:
        return None
    a_b = (a or "").encode("utf-8")
    b_b = (b or "").encode("utf-8")
    bp = lib.sdx_token_jaccard_utf8(a_b, len(a_b), b_b, len(b_b))
    return float(bp) / 10000.0 if bp >= 0 else None

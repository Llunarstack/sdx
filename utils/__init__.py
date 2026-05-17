"""Utility package: image post-process helpers re-exported at package root."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["sharpen", "sharpen_pil", "contrast", "contrast_pil"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    q = import_module("utils.quality.quality")
    val = getattr(q, name)
    globals()[name] = val
    return val


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

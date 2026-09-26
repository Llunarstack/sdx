"""Stable alias for native helpers: ``from utils.nt import …``.

Delegates to ``utils.native`` (which puts ``native/_experimental/python`` on
``sys.path`` for ``sdx_native``).
"""

from __future__ import annotations

from typing import Any

from utils import native as _native

__all__ = list(getattr(_native, "__all__", []))


def __getattr__(name: str) -> Any:
    return getattr(_native, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

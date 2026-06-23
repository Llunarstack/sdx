"""
Deprecated import path — use ``diffusion.sampling``.

Re-exports lazily from the canonical package.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

_CANONICAL = "diffusion.sampling"


def __getattr__(name: str) -> Any:
    mod = import_module(_CANONICAL)
    if not hasattr(mod, name):
        raise AttributeError(f"module {_CANONICAL!r} has no attribute {name!r}")
    val = getattr(mod, name)
    globals()[name] = val
    return val


def __dir__() -> list[str]:
    return sorted(dir(import_module(_CANONICAL)))

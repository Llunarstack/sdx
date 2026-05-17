"""Checkpoint load/save and checkpoint diff utilities.

Submodules load on first ``utils.checkpoint.<name>`` access.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent
_SUBMODULE_NAMES: frozenset[str] = frozenset(
    p.stem for p in _pkg_dir.glob("*.py") if p.name != "__init__.py"
)
__all__: list[str] = sorted(_SUBMODULE_NAMES)  # pyright: ignore[reportUnsupportedDunderAll]


def __getattr__(name: str):
    if name not in _SUBMODULE_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = import_module(f".{name}", __package__)
    globals()[name] = mod
    return mod


def __dir__() -> list[str]:
    return sorted(set(globals()) | _SUBMODULE_NAMES)

"""Canonical import path; implementations in ``utils._archive.architecture``."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

_ARCHIVE = "utils._archive.architecture"
_pkg_dir = Path(__file__).resolve().parent
_SUBMODULE_NAMES: frozenset[str] = frozenset(
    p.stem for p in (Path(__file__).resolve().parent.parent / "_archive" / "architecture").glob("*.py")
    if p.name != "__init__.py"
)
__all__: list[str] = sorted(_SUBMODULE_NAMES)


def __getattr__(name: str):
    if name not in _SUBMODULE_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = import_module(f"{_ARCHIVE}.{name}")
    globals()[name] = mod
    return mod


def __dir__() -> list[str]:
    return sorted(set(globals()) | _SUBMODULE_NAMES)

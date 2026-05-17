"""Modeling helpers.

Submodules resolve on first ``utils.modeling.<name>`` access so ``import utils.modeling``
does not require Torch or load heavy inspectors until needed.

Prefer submodule imports in hot paths::

    from utils.modeling.model_paths import default_t5_path
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
    """Resolve ``utils.modeling.<submodule>`` on first access."""
    if name not in _SUBMODULE_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = import_module(f".{name}", __package__)
    globals()[name] = mod
    return mod


def __dir__() -> list[str]:
    return sorted(set(globals()) | _SUBMODULE_NAMES)

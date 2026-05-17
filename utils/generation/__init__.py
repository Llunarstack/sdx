"""Generation helpers.

Heavy submodules load lazily via ``__getattr__``; only PIL/subprocess helpers below
are imported eagerly. Torch-dependent code loads when a submodule is accessed.

For hot paths prefer direct imports::

    from utils.generation.master_integration import quick_generate
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from . import edit_masks, sample_edit_runner, segmentation_to_mask  # noqa: F401

_pkg_dir = Path(__file__).resolve().parent
_SUBMODULE_NAMES: frozenset[str] = frozenset(
    p.stem for p in _pkg_dir.glob("*.py") if p.name != "__init__.py"
)
# Subpackage module names (stable order for tab-completion / ``from pkg import *``).
__all__: list[str] = sorted(_SUBMODULE_NAMES)  # pyright: ignore[reportUnsupportedDunderAll]


def __getattr__(name: str):
    """Resolve ``utils.generation.<submodule>`` on first access."""
    if name not in _SUBMODULE_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = import_module(f".{name}", __package__)
    globals()[name] = mod
    return mod


def __dir__() -> list[str]:
    return sorted(set(globals()) | _SUBMODULE_NAMES)

"""Prompt utilities.



Submodules load on first attribute access (``utils.prompt.<name>``), similar to

``utils.generation``. Prefer explicit submodule imports for hot paths:



    from utils.prompt.neg_filter import filter_negative_by_positive



Torch-heavy modules therefore do not load when you only ``import utils.prompt``.

"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent

_SUBMODULE_NAMES: frozenset[str] = frozenset(
    p.stem for p in _pkg_dir.glob("*.py") if p.name != "__init__.py"
) | frozenset({"stack"})

__all__: list[str] = sorted(_SUBMODULE_NAMES)  # pyright: ignore[reportUnsupportedDunderAll]


def __getattr__(name: str):
    """Resolve ``utils.prompt.<submodule>`` on first access."""
    if name not in _SUBMODULE_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name == "stack":
        mod = import_module(".stack", __package__)
    else:
        mod = import_module(f".{name}", __package__)
    globals()[name] = mod
    return mod


def __dir__() -> list[str]:
    return sorted(set(globals()) | _SUBMODULE_NAMES)

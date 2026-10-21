"""Image quality critics, test-time selection and the finishing pipeline.

Submodules load on first access; any other public name resolves from ``quality.py``
(``utils.quality.naturalize`` is ``utils.quality.quality.naturalize``).
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

_SUBMODULE_NAMES: frozenset[str] = frozenset(
    p.stem for p in Path(__file__).resolve().parent.glob("*.py") if p.name != "__init__.py"
)
__all__: list[str] = sorted(_SUBMODULE_NAMES)  # pyright: ignore[reportUnsupportedDunderAll]


def __getattr__(name: str) -> Any:
    if name in _SUBMODULE_NAMES:
        val = import_module(f".{name}", __package__)
    elif not name.startswith("_") and hasattr(q := import_module(".quality", __package__), name):
        val = getattr(q, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = val
    return val


def __dir__() -> list[str]:
    return sorted(set(globals()) | _SUBMODULE_NAMES)

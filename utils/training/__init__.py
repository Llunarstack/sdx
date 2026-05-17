"""Training utilities.

Submodules load on first ``utils.training.<name>`` access so ``import utils.training``
does not pull Torch or NumPy-heavy helpers until needed.

Prefer explicit submodule imports::

    from utils.training.config_validator import validate_train_config
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
    """Resolve ``utils.training.<submodule>`` on first access."""
    if name not in _SUBMODULE_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = import_module(f".{name}", __package__)
    globals()[name] = mod
    return mod


def __dir__() -> list[str]:
    return sorted(set(globals()) | _SUBMODULE_NAMES)

"""Image quality and test-time selection helpers.

Sibling modules (e.g. ``test_time_pick``) load lazily. Names defined in ``quality.py`` are
re-exported at package scope on first use (same surface as the old star-import).
"""

from __future__ import annotations

import ast
from importlib import import_module
from pathlib import Path
from typing import Any

_pkg_dir = Path(__file__).resolve().parent


def _public_names_quality_py() -> frozenset[str]:
    tree = ast.parse((_pkg_dir / "quality.py").read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if not node.target.id.startswith("_"):
                names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and not t.id.startswith("_"):
                    names.add(t.id)
    return frozenset(names)


_SUBMODULE_NAMES: frozenset[str] = frozenset(
    p.stem for p in _pkg_dir.glob("*.py") if p.name != "__init__.py"
)
_QUALITY_PUBLIC: frozenset[str] = _public_names_quality_py()
_EXPORT_NAMES: frozenset[str] = _SUBMODULE_NAMES | _QUALITY_PUBLIC
__all__: list[str] = sorted(_EXPORT_NAMES)  # pyright: ignore[reportUnsupportedDunderAll]


def __getattr__(name: str) -> Any:
    if name in _SUBMODULE_NAMES:
        mod = import_module(f".{name}", __package__)
        globals()[name] = mod
        return mod
    if name in _QUALITY_PUBLIC:
        q = import_module(".quality", __package__)
        val = getattr(q, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _EXPORT_NAMES)

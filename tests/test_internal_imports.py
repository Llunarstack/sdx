"""Every ``from <internal module> import name`` (at any nesting level) names something that exists.

Imports inside functions only fail when that code path runs, so tests miss them; this catches a deleted or
renamed helper anywhere in the repo.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".venv", "external", "__pycache__", "node_modules", "pretrained", "outputs", "results"}
SEARCH = (ROOT, ROOT / "native" / "_experimental" / "python")


def _module_file(dotted: str) -> Path | None:
    rel = Path(*dotted.split("."))
    for base in SEARCH:
        for cand in (base / rel.with_suffix(".py"), base / rel / "__init__.py"):
            if cand.is_file():
                return cand
    return None


def _top_names(path: Path) -> tuple[set[str], bool]:
    """Names bound at module level (including under if/try), and whether a lazy ``__getattr__`` exists."""
    names: set[str] = set()
    lazy = False

    def visit(body: list[ast.stmt]) -> None:
        nonlocal lazy
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
                lazy |= node.name == "__getattr__"
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names.update(n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update(a.asname or a.name.split(".")[0] for a in node.names)
            elif isinstance(node, (ast.If, ast.Try, ast.With)):
                visit([*node.body, *getattr(node, "orelse", []), *getattr(node, "finalbody", [])])
                for h in getattr(node, "handlers", []):
                    visit(h.body)

    visit(ast.parse(path.read_text(encoding="utf-8", errors="ignore")).body)
    return names, lazy


def _py_files() -> list[Path]:
    parts = (p.relative_to(ROOT).parts for p in ROOT.rglob("*.py"))
    return [ROOT.joinpath(*ps) for ps in parts if not any(x in SKIP or x.startswith(".") for x in ps)]


def test_internal_from_imports_resolve() -> None:
    internal = {p.name for p in ROOT.iterdir() if p.is_dir() and p.name not in SKIP} | {"sdx_native"}
    internal.discard("datasets")  # data folder without __init__; `datasets` is the Hugging Face package
    problems = []
    for path in _py_files():
        rel = path.relative_to(ROOT)
        pkg = list(rel.parent.parts)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8", errors="ignore"))):
            if not isinstance(node, ast.ImportFrom):
                continue
            base = pkg[: len(pkg) - node.level + 1] if node.level else []
            mod = ".".join([*base, *([node.module] if node.module else [])])
            if not mod or mod.split(".")[0] not in internal:
                continue
            target = _module_file(mod)
            if target is None:
                if not (ROOT / Path(*mod.split("."))).is_dir():  # namespace packages are fine
                    problems.append(f"{rel}:{node.lineno} from {mod}: no such module")
                continue
            names, lazy = _top_names(target)
            missing = [a.name for a in node.names if not (lazy or a.name in names or _module_file(f"{mod}.{a.name}"))]
            problems += [f"{rel}:{node.lineno} from {mod} import {n}" for n in missing]
    assert not problems, "Unresolved internal imports:\n" + "\n".join(problems)

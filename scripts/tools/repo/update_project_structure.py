#!/usr/bin/env python3
"""
Write PROJECT_STRUCTURE.md at the repo root: an ASCII tree of the SDX codebase.

Skips cache dirs, gitignored noise, and large optional trees. Safe to re-run after moves.

Usage (from repo root):
    python -m scripts.tools update_project_structure
    python -m scripts.tools update_project_structure --max-depth 4 --out PROJECT_STRUCTURE.md
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Repo root: scripts/tools/repo/update_project_structure.py -> parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]

# Directory names to skip entirely (anywhere under root)
SKIP_DIRS: set[str] = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "env",
    ".tox",
    "node_modules",
    "htmlcov",
    ".eggs",
}

# Skip by default (large / generated / optional parallel trees)
SKIP_DIRS_DEFAULT: set[str] = {
    "model",  # downloaded weights (gitignored)
    "external",  # cloned reference repos
    "enhanced_dit",  # optional parallel package tree (main code is under models/, train.py)
}

# File suffixes to skip
SKIP_SUFFIXES: tuple[str, ...] = (".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib")

# Names to skip (files)
SKIP_FILES: set[str] = {
    ".DS_Store",
    "Thumbs.db",
}


def _should_skip_dir(name: str, *, skip_extra: set[str]) -> bool:
    if name in SKIP_DIRS or name in skip_extra:
        return True
    if name.endswith(".egg-info"):
        return True
    return False


def _should_skip_file(path: Path) -> bool:
    if path.name in SKIP_FILES:
        return True
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    return False


_DOTFILES_OK: frozenset[str] = frozenset({".editorconfig", ".gitignore", ".env.example"})


def _iter_sorted_children(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    items: list[Path] = []
    for p in path.iterdir():
        name = p.name
        if name.startswith(".") and name not in _DOTFILES_OK:
            continue
        items.append(p)
    dirs = sorted([p for p in items if p.is_dir()], key=lambda x: x.name.lower())
    files = sorted([p for p in items if p.is_file()], key=lambda x: x.name.lower())
    return dirs + files


def build_tree_lines(
    root: Path,
    *,
    max_depth: int,
    skip_extra_dirs: set[str],
    prefix: str = "",
    depth: int = 0,
) -> list[str]:
    lines: list[str] = []
    if depth >= max_depth:
        return lines

    children = _iter_sorted_children(root)
    for i, path in enumerate(children):
        is_last = i == len(children) - 1
        branch = "└── " if is_last else "├── "
        if path.is_dir():
            if _should_skip_dir(path.name, skip_extra=skip_extra_dirs):
                continue
            lines.append(f"{prefix}{branch}{path.name}/")
            extension = "    " if is_last else "│   "
            sub = build_tree_lines(
                path,
                max_depth=max_depth,
                skip_extra_dirs=skip_extra_dirs,
                prefix=prefix + extension,
                depth=depth + 1,
            )
            lines.extend(sub)
        else:
            if _should_skip_file(path):
                continue
            lines.append(f"{prefix}{branch}{path.name}")
    return lines


def main() -> int:
    p = argparse.ArgumentParser(description="Regenerate PROJECT_STRUCTURE.md")
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "PROJECT_STRUCTURE.md",
        help="Output markdown path (default: repo root PROJECT_STRUCTURE.md)",
    )
    p.add_argument("--max-depth", type=int, default=5, help="Max directory depth from repo root (default: 5)")
    p.add_argument(
        "--include-model",
        action="store_true",
        help="Include model/ in tree (usually empty gitignored downloads)",
    )
    p.add_argument(
        "--include-external",
        action="store_true",
        help="Include external/ cloned reference repos",
    )
    p.add_argument(
        "--skip-dir",
        action="append",
        default=[],
        metavar="NAME",
        help="Extra directory name to skip (repeatable), e.g. --skip-dir enhanced_dit",
    )
    p.add_argument(
        "--include-enhanced-dit",
        action="store_true",
        help="Include enhanced_dit/ (skipped by default; main DiT code lives under models/)",
    )
    args = p.parse_args()

    skip_extra: set[str] = set(SKIP_DIRS_DEFAULT)
    if args.include_model:
        skip_extra.discard("model")
    if args.include_external:
        skip_extra.discard("external")
    if args.include_enhanced_dit:
        skip_extra.discard("enhanced_dit")
    skip_extra.update(str(x).strip() for x in (args.skip_dir or []) if str(x).strip())

    tree_lines = build_tree_lines(REPO_ROOT, max_depth=int(args.max_depth), skip_extra_dirs=skip_extra)
    tree_body = "\n".join(tree_lines) if tree_lines else "(empty)"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    skipped_display = sorted(SKIP_DIRS_DEFAULT & skip_extra)
    now_line = (
        f"> Generated: **{now}** · max depth: **{args.max_depth}** · repo root: `{REPO_ROOT.name}/`"
    )
    if skipped_display:
        now_line += f"\n>\n> Skipped directories: **{', '.join(skipped_display)}** (see `--help` to include)."
    md = f"""# SDX project structure

> **Auto-generated** — do not edit by hand. Regenerate after moving files:
>
> ```bash
> python -m scripts.tools update_project_structure
> ```
>
{now_line}

## Tree

```
{REPO_ROOT.name}/
{tree_body}
```

## See also

- [docs/CODEBASE.md](docs/CODEBASE.md) — navigate the tree, `scripts/` layout, contribution rules
- [docs/FILES.md](docs/FILES.md) — full file map

"""
    out: Path = args.out
    if not out.is_absolute():
        out = (REPO_ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"Wrote {out} ({len(md)} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

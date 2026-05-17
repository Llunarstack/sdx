"""
Smoke test: import all *internal* SDX modules to catch broken links early.

This does NOT try to run training or load huge checkpoints; it only checks that imports
across the codebase succeed (excluding `external/`).

Usage:
    python -m scripts.tools smoke_imports
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path


def iter_internal_modules(root_pkg: str, root_path: Path):
    for m in pkgutil.walk_packages([str(root_path)], prefix=f"{root_pkg}."):
        yield m.name


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))

    # Only include internal packages we own.
    packages = [
        ("config", repo_root / "config"),
        ("data", repo_root / "data"),
        ("diffusion", repo_root / "diffusion"),
        ("models", repo_root / "models"),
        ("training", repo_root / "training"),
        ("utils", repo_root / "utils"),
        ("pipelines", repo_root / "pipelines"),
        ("vit_quality", repo_root / "vit_quality"),
        ("scripts.tools", repo_root / "scripts" / "tools"),
    ]

    skip_prefixes = (
        "research.agi_image.",  # experimental scaffold; optional heavy deps
    )

    failures = []
    for pkg, path in packages:
        if not path.exists():
            continue
        for mod_name in iter_internal_modules(pkg, path):
            if mod_name.endswith(".__pycache__"):
                continue
            if any(mod_name.startswith(p) for p in skip_prefixes):
                continue
            try:
                importlib.import_module(mod_name)
            except Exception as e:
                failures.append((mod_name, repr(e)))

    if failures:
        print("SMOKE IMPORTS: failures:")
        for mod_name, err in failures:
            print(f"- {mod_name}: {err}")
        raise SystemExit(1)

    print("SMOKE IMPORTS: all internal modules imported successfully.")


if __name__ == "__main__":
    main()

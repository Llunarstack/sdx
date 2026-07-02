#!/usr/bin/env python3
"""Generate utils/<pkg>/ shims that re-export utils/_archive/<pkg>/."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def write_shims(archive_pkg: str) -> int:
    archive = ROOT / "utils" / "_archive" / archive_pkg
    public_dir = ROOT / "utils" / archive_pkg
    public_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for py in sorted(archive.glob("*.py")):
        if py.name == "__init__.py":
            # Copy verbatim: relative imports and lazy __getattr__ maps resolve
            # against the shim submodules, preserving the package's laziness
            # contract (tests/test_prompt_training_pkg_lazy.py) and __all__.
            (public_dir / "__init__.py").write_text(py.read_text(encoding="utf-8"), encoding="utf-8")
            count += 1
            continue
        mod = f"utils._archive.{archive_pkg}.{py.stem}"
        text = (
            f'"""Shim → ``{mod}``."""\n\n'
            f"import {mod} as _src\n\n"
            "for _name in dir(_src):\n"
            "    if not _name.startswith('__'):\n"
            "        globals()[_name] = getattr(_src, _name)\n\n"
            "del _name, _src\n"
        )
        (public_dir / py.name).write_text(text, encoding="utf-8")
        count += 1
    return count


def main() -> None:
    for pkg in (
        "superior",
        "agentic",
        "brain",
        "modeling",
        "analysis",
        "checkpoint",
        "consistency",
        "quantization",
        "runtime",
        "visual_design",
    ):
        n = write_shims(pkg)
        print(f"{pkg}: {n} files")


if __name__ == "__main__":
    main()

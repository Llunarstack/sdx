"""
Delete local generated artifacts from the repository tree.

Examples:
    python -m scripts.tools clean_repo_artifacts --dry-run
    python -m scripts.tools clean_repo_artifacts
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

TARGET_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".hypothesis",
}

TARGET_FILE_SUFFIXES = {
    ".pyc",
}

TARGET_FILE_NAMES = {
    ".DS_Store",
}

SKIP_WALK_DIRS = {
    ".git",
    ".venv",
    "venv",
    ".pixi",
    "node_modules",
}


def _safe_candidates(repo_root: Path, *, include_external: bool = False) -> list[Path]:
    paths: list[Path] = []
    root = repo_root.resolve()

    def _on_walk_error(err: OSError) -> None:
        print(f"! skipped inaccessible: {err.filename}")

    for dirpath, dirnames, filenames in os.walk(root, topdown=True, onerror=_on_walk_error):
        keep_dirs: list[str] = []
        for d in dirnames:
            if d in SKIP_WALK_DIRS:
                continue
            if d == "external" and not include_external and Path(dirpath).resolve() == root:
                continue
            full = Path(dirpath) / d
            if d in TARGET_DIR_NAMES:
                paths.append(full)
                continue
            keep_dirs.append(d)
        dirnames[:] = keep_dirs

        for fn in filenames:
            if fn in TARGET_FILE_NAMES or Path(fn).suffix in TARGET_FILE_SUFFIXES:
                paths.append(Path(dirpath) / fn)

    uniq = sorted({p.resolve() for p in paths})
    safe: list[Path] = []
    for p in uniq:
        try:
            p.relative_to(root)
        except ValueError:
            continue
        safe.append(p)
    return safe


def main() -> int:
    ap = argparse.ArgumentParser(description="Delete generated cache artifacts in this repo.")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would be removed.")
    ap.add_argument(
        "--include-external",
        action="store_true",
        help="Also scan external/ cloned repos (off by default for speed/safety).",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    targets = _safe_candidates(repo_root, include_external=bool(args.include_external))
    if not targets:
        print("No generated artifacts found.")
        return 0

    print(f"Repo: {repo_root}")
    print(f"Targets: {len(targets)}")
    for t in targets:
        rel = t.relative_to(repo_root)
        kind = "dir " if t.is_dir() else "file"
        print(f"- [{kind}] {rel}")

    if args.dry_run:
        return 0

    removed = 0
    for t in targets:
        try:
            if t.is_dir():
                shutil.rmtree(t, ignore_errors=True)
            else:
                t.unlink(missing_ok=True)
            removed += 1
        except Exception as e:
            print(f"! failed: {t} ({e})")
    print(f"Removed {removed}/{len(targets)} targets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


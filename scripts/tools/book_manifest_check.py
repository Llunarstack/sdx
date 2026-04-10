#!/usr/bin/env python3
"""
Validate ``book_manifest.json`` from ``generate_book.py --write-book-manifest``.

Usage (repo root)::

    python -m scripts.tools book_manifest_check path/to/out/book_manifest.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    p = argparse.ArgumentParser(description="Validate book_manifest.json structure and flag weak configs.")
    p.add_argument("manifest", type=Path, help="Path to book_manifest.json")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 2 if there are warnings (default: warnings only print to stderr).",
    )
    args = p.parse_args()

    root = _repo_root()
    rs = str(root)
    if rs not in sys.path:
        sys.path.insert(0, rs)

    from pipelines.book_comic.book_manifest_utils import (
        load_book_manifest,
        manifest_summary_lines,
        validate_book_manifest,
    )

    path = args.manifest.expanduser()
    if not path.is_file():
        print(f"Not a file: {path}", file=sys.stderr)
        return 2
    manifest = load_book_manifest(path)
    errs, warns = validate_book_manifest(manifest)
    for line in manifest_summary_lines(manifest):
        print(line)
    for w in warns:
        print(f"WARNING: {w}", file=sys.stderr)
    for e in errs:
        print(f"ERROR: {e}", file=sys.stderr)
    if errs:
        return 1
    if warns and args.strict:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

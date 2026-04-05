#!/usr/bin/env python3
"""
Merge multiple manifest JSONL files with dedupe (first wins).

Prefers the Go binary ``native/go/sdx-manifest`` if built; otherwise uses
:func:`utils.native.native_tools.merge_jsonl_files` (pure Python).

Usage (repo root)::

    python -m scripts.tools jsonl_merge -o merged.jsonl a.jsonl b.jsonl
    python -m scripts.tools jsonl_merge -o merged.jsonl --dedupe-key path part1.jsonl part2.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge JSONL manifests with optional dedupe.")
    ap.add_argument("-o", "--out", required=True, help="Output JSONL path")
    ap.add_argument(
        "--dedupe-key",
        default="image_path",
        help="JSON field for dedupe (default image_path; falls back to path/image in rows)",
    )
    ap.add_argument("inputs", nargs="+", help="Input .jsonl files (order preserved for first-wins)")
    args = ap.parse_args()

    from utils.native.native_tools import merge_jsonl_files

    ins = [Path(p) for p in args.inputs]
    for p in ins:
        if not p.is_file():
            print(f"Not found: {p}", file=sys.stderr)
            return 1
    merge_jsonl_files(ins, Path(args.out), dedupe_key=args.dedupe_key, prefer_go=True)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

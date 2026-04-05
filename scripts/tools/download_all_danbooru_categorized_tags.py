#!/usr/bin/env python3
"""
Run the full pipeline: fetch tags → split general → merge one TSV.

  1) fetch_danbooru_tags.py  → data/danbooru/tags/raw/*.txt
  2) split_danbooru_general_tags.py → data/danbooru/tags/buckets/*.txt
  3) merge_danbooru_categorized_tags.py → data/danbooru/tags/all_tags_categorized.txt

Any unknown CLI flags are forwarded to ``fetch_danbooru_tags.py`` (e.g. ``--max-pages 50``).

Examples::

    python scripts/tools/download_all_danbooru_categorized_tags.py
    python scripts/tools/download_all_danbooru_categorized_tags.py --max-pages 10 --sleep 0.2
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch Danbooru tags and build categorized export.")
    ap.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/danbooru/tags/raw"),
        help="Raw tag output directory",
    )
    ap.add_argument(
        "--buckets-dir",
        type=Path,
        default=Path("data/danbooru/tags/buckets"),
        help="Bucket output directory",
    )
    ap.add_argument(
        "--rules",
        type=Path,
        default=Path("data/danbooru/general_subsplit_rules.json"),
        help="General-tag split rules",
    )
    ap.add_argument(
        "--merged-out",
        type=Path,
        default=Path("data/danbooru/tags/all_tags_categorized.txt"),
        help="Single combined TSV output",
    )
    args, fetch_extra = ap.parse_known_args()

    py = sys.executable

    def run(script: str, argv: list[str]) -> int:
        cmd = [py, str(REPO / "scripts" / "tools" / script)] + argv
        print("+", " ".join(cmd), file=sys.stderr)
        return subprocess.call(cmd, cwd=str(REPO))

    rc = run(
        "fetch_danbooru_tags.py",
        ["--out-dir", str(args.raw_dir)] + fetch_extra,
    )
    if rc != 0:
        return rc

    rc = run(
        "split_danbooru_general_tags.py",
        [
            "--general",
            str(args.raw_dir / "general.txt"),
            "--rules",
            str(args.rules),
            "--out-dir",
            str(args.buckets_dir),
        ],
    )
    if rc != 0:
        return rc

    rc = run(
        "merge_danbooru_categorized_tags.py",
        [
            "--raw-dir",
            str(args.raw_dir),
            "--buckets-dir",
            str(args.buckets_dir),
            "--out",
            str(args.merged_out),
        ],
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

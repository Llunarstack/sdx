#!/usr/bin/env python3
"""
Merge Danbooru raw category files + general bucket files into one labeled export.

Input layout (default under data/danbooru/tags/)::
  raw/{artist,copyright,character,meta,general}.txt
  buckets/{art_style,clothes,objects,...,general_other}.txt

Output line format::
  category<TAB>name<TAB>post_count<TAB>deprecated

``category`` is either a Danbooru API name (artist, copyright, character, meta) or a
bucket name (art_style, clothes, ...) for tags split from **general**.

**Important:** Bucket files are a *partition* of ``general.txt``. This merge writes
**non-general** rows from raw plus **all bucket** rows. It does **not** duplicate
``general.txt`` when buckets are present. If no bucket ``*.txt`` files exist, it falls
back to labeling ``raw/general.txt`` as category ``general``.

Usage::

    python -m scripts.tools merge_danbooru_categorized_tags \\
        --out data/danbooru/tags/all_tags_categorized.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List


def _iter_lines(path: Path) -> Iterable[str]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield line.rstrip("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge categorized Danbooru tag files into one TSV.")
    ap.add_argument("--raw-dir", type=Path, default=Path("data/danbooru/tags/raw"))
    ap.add_argument("--buckets-dir", type=Path, default=Path("data/danbooru/tags/buckets"))
    ap.add_argument("--out", type=Path, default=Path("data/danbooru/tags/all_tags_categorized.txt"))
    args = ap.parse_args()

    raw_files: List[tuple[str, Path]] = [
        ("artist", args.raw_dir / "artist.txt"),
        ("copyright", args.raw_dir / "copyright.txt"),
        ("character", args.raw_dir / "character.txt"),
        ("meta", args.raw_dir / "meta.txt"),
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    bucket_txts = sorted(args.buckets_dir.glob("*.txt")) if args.buckets_dir.is_dir() else []
    bucket_txts = [p for p in bucket_txts if not p.name.startswith(".")]

    with args.out.open("w", encoding="utf-8") as out:
        for cat, path in raw_files:
            for line in _iter_lines(path):
                out.write(f"{cat}\t{line}\n")
                n += 1

        if bucket_txts:
            for path in bucket_txts:
                bucket = path.stem
                for line in _iter_lines(path):
                    out.write(f"{bucket}\t{line}\n")
                    n += 1
        else:
            g = args.raw_dir / "general.txt"
            for line in _iter_lines(g):
                out.write(f"general\t{line}\n")
                n += 1

    print(f"Wrote {n} lines to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

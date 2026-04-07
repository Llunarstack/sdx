#!/usr/bin/env python3
"""
Split ``general.txt`` from fetch_danbooru_tags.py into training-oriented buckets.

Uses underscore-aware substring rules in ``data/danbooru/general_subsplit_rules.json``.
First matching bucket in ``bucket_order`` wins.

Input line format: name<TAB>post_count<TAB>deprecated  (only the first column is used).

Examples::

    python -m scripts.tools split_danbooru_general_tags \\
        --general data/danbooru/tags/raw/general.txt \\
        --rules data/danbooru/general_subsplit_rules.json \\
        --out-dir data/danbooru/tags/buckets
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def _wrap(s: str) -> str:
    """Normalize for token-ish matching: _foo_bar_."""
    t = s.lower().replace("-", "_").replace(" ", "_").strip("_")
    return f"_{t}_"


def matches_pattern(tag: str, pattern: str) -> bool:
    return _wrap(pattern) in _wrap(tag)


def load_rules(path: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    order = list(data["bucket_order"])
    patterns = {k: list(v) for k, v in data["patterns"].items()}
    return order, patterns


def split_general(
    general_path: Path,
    rules_path: Path,
    out_dir: Path,
) -> Dict[str, int]:
    order, patterns = load_rules(rules_path)
    # Ensure all pattern buckets exist in order
    buckets: Dict[str, List[str]] = {b: [] for b in order}
    buckets["general_other"] = []

    with general_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tag = line.split("\t", 1)[0].strip()
            if not tag:
                continue
            placed = False
            for bucket in order:
                for p in patterns.get(bucket, []):
                    if matches_pattern(tag, p):
                        buckets[bucket].append(line + "\n")
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                buckets["general_other"].append(line + "\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    for name, lines in buckets.items():
        p = out_dir / f"{name}.txt"
        p.write_text("".join(lines), encoding="utf-8")
        counts[name] = len(lines)
        print(f"{name}: {len(lines)} -> {p}", file=sys.stderr)
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description="Bucket Danbooru general tags for training.")
    ap.add_argument("--general", type=Path, required=True, help="Path to general.txt from fetch_danbooru_tags.py")
    ap.add_argument(
        "--rules",
        type=Path,
        default=Path("data/danbooru/general_subsplit_rules.json"),
        help="JSON with bucket_order and patterns",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("data/danbooru/tags/buckets"), help="Output directory")
    args = ap.parse_args()

    if not args.general.is_file():
        print(f"Missing {args.general}", file=sys.stderr)
        return 1
    if not args.rules.is_file():
        print(f"Missing {args.rules}", file=sys.stderr)
        return 2

    split_general(args.general, args.rules, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

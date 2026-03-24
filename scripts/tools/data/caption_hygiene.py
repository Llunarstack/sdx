#!/usr/bin/env python3
"""
JSONL caption QA: Unicode normalization preview, pos/neg overlap, duplicate fingerprints.

Implements Python-side checks described in ``docs/NATIVE_AND_SYSTEM_LIBS.md`` (complements
Rust ``sdx-jsonl-tools prompt-lint`` and Python ``sdx_native.jsonl_manifest_pure``).

Usage (repo root on PYTHONPATH; ``native/python`` for ``sdx_native``)::

  python scripts/tools/data/caption_hygiene.py data/manifest.jsonl --report-dups --max-overlap-show 15
  python scripts/tools/data/caption_hygiene.py data/manifest.jsonl --normalize-samples 5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

# Repo root
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_np = ROOT / "native" / "python"
if str(_np) not in sys.path:
    sys.path.insert(0, str(_np))

from sdx_native.text_hygiene import (  # noqa: E402
    caption_fingerprint,
    jsonl_row_caption_keys,
    normalize_caption_for_training,
    pos_neg_token_overlap,
)


def _caption_from_row(row: Dict[str, Any]) -> str:
    for k in jsonl_row_caption_keys():
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _negative_from_row(row: Dict[str, Any]) -> str:
    for k in ("negative_caption", "negative_prompt"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _image_key(row: Dict[str, Any]) -> str:
    return str(row.get("image_path") or row.get("path") or row.get("image") or "")


def main() -> int:
    p = argparse.ArgumentParser(description="Caption hygiene report for JSONL manifests.")
    p.add_argument("jsonl", type=Path, help="Manifest JSONL path")
    p.add_argument("--report-dups", action="store_true", help="Report duplicate normalized caption fingerprints")
    p.add_argument("--report-overlap", action="store_true", help="Report rows with pos/neg token overlap")
    p.add_argument("--overlap-min-jaccard", type=float, default=0.0, help="Min Jaccard to print overlap row")
    p.add_argument("--max-overlap-show", type=int, default=30, help="Max overlap rows to print")
    p.add_argument("--normalize-samples", type=int, default=0, help="Print first N before/after normalize")
    p.add_argument("--empty-warn", action="store_true", help="Count rows with empty caption")
    args = p.parse_args()
    path: Path = args.jsonl
    if not path.is_file():
        print(f"Not a file: {path}", file=sys.stderr)
        return 1

    dup_buckets: DefaultDict[str, List[str]] = defaultdict(list)
    overlap_rows: List[Tuple[float, str, str, str]] = []
    empty_captions = 0
    lines = 0
    samples_left = int(args.normalize_samples or 0)

    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            cap = _caption_from_row(row)
            neg = _negative_from_row(row)
            ik = _image_key(row)
            if not cap:
                empty_captions += 1
            if samples_left > 0 and cap:
                norm = normalize_caption_for_training(cap)
                print(f"--- sample {samples_left} ---")
                print(f"before: {cap[:500]}{'...' if len(cap) > 500 else ''}")
                print(f"after:  {norm[:500]}{'...' if len(norm) > 500 else ''}")
                samples_left -= 1

            if args.report_dups and cap:
                fp = caption_fingerprint(cap, algorithm="sha256")
                key = ik or f"line{lines}"
                dup_buckets[fp].append(key)

            if args.report_overlap and cap and neg:
                _i, _u, j = pos_neg_token_overlap(cap, neg)
                if j >= float(args.overlap_min_jaccard) and _i > 0:
                    overlap_rows.append((j, ik or f"line{lines}", cap[:120], neg[:120]))

    if args.empty_warn:
        print(f"rows_read={lines} empty_caption_rows={empty_captions}")

    if args.report_dups:
        multi = [(fp, paths) for fp, paths in dup_buckets.items() if len(paths) > 1]
        multi.sort(key=lambda x: -len(x[1]))
        print(f"duplicate_caption_groups={len(multi)}")
        for fp, paths in multi[:50]:
            print(f"fp={fp[:16]}… count={len(paths)} example={paths[0]}")

    if args.report_overlap:
        overlap_rows.sort(key=lambda x: -x[0])
        for i, (j, key, c, n) in enumerate(overlap_rows[: int(args.max_overlap_show)]):
            print(f"overlap[{i}] jaccard={j:.3f} key={key}")
            print(f"  pos: {c}")
            print(f"  neg: {n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

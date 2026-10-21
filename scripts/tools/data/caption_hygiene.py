#!/usr/bin/env python3
"""
JSONL caption QA: Unicode normalization preview, pos/neg overlap, duplicate fingerprints.

Implements Python-side checks described in ``docs/NATIVE_AND_SYSTEM_LIBS.md`` (complements
Rust ``sdx-jsonl-tools prompt-lint`` and ``python -m scripts.tools prompt_lint``).

Usage (repo root on PYTHONPATH; ``native/_experimental/python`` for ``sdx_native``)::

  python -m scripts.tools caption_hygiene data/manifest.jsonl --report-dups --max-overlap-show 15
  python -m scripts.tools caption_hygiene data/manifest.jsonl --normalize-samples 5
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.tools._repo_bootstrap import ensure_repo_paths  # noqa: E402

ensure_repo_paths()

from data.manifest_utils import caption_from_row, first_str, iter_manifest_rows, negative_caption_from_row  # noqa: E402

from sdx_native.text_hygiene import (  # noqa: E402
    caption_fingerprint,
    normalize_caption_for_training,
    pos_neg_token_overlap,
)


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

    dup_buckets: defaultdict[str, list[str]] = defaultdict(list)
    overlap_rows: list[tuple[float, str, str, str]] = []
    empty_captions = 0
    lines = 0
    samples_left = int(args.normalize_samples or 0)

    for row in iter_manifest_rows(path):
        if not isinstance(row, dict):
            continue
        lines += 1
        cap = caption_from_row(row)
        neg = negative_caption_from_row(row)
        ik = first_str(row, ("image_path", "path", "image"))
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

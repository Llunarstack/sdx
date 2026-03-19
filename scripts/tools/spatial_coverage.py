"""
Spatial wording coverage analysis for captions.

Counts how often captions mention spatial relations relevant to:
  behind / next to / under / left of / right of / in front of / above / below

Usage:
  python -m scripts.tools.spatial_coverage --manifest manifest.jsonl --out spatial.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _get_caption(rec: Dict[str, Any]) -> str:
    for k in ("caption", "text", "prompt"):
        v = rec.get(k)
        if v is not None:
            s = str(v).strip()
            if s:
                return s
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze spatial relation keywords in captions.")
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--min-rows", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    import sys

    sys.path.insert(0, str(repo_root))

    spatial_terms = [
        "behind",
        "in front of",
        "front of",
        "next to",
        "beside",
        "under",
        "below",
        "above",
        "left of",
        "right of",
    ]

    counts = {t: 0 for t in spatial_terms}
    total = 0

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Not found: {manifest_path}")

    for rec in _iter_jsonl(manifest_path):
        cap = _get_caption(rec)
        if not cap:
            continue
        total += 1
        cap_lower = cap.lower()
        for t in spatial_terms:
            if t in cap_lower:
                counts[t] += 1
        if args.min_rows and total >= args.min_rows:
            break

    if total == 0:
        print("No captions found.")
        return

    report = {
        "manifest": str(manifest_path),
        "rows_scanned": total,
        "term_coverage": {t: counts[t] / total for t in spatial_terms},
        "counts": counts,
    }

    print(f"Spatial coverage (rows_scanned={total})")
    for t in spatial_terms:
        cov = report["term_coverage"][t]
        print(f"- {t}: {cov:.3f} ({counts[t]}/{total})")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote report: {out_path}")


if __name__ == "__main__":
    main()


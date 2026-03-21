"""
Tag coverage analysis for training captions.

Counts how often your manifest's captions contain:
- quality tags
- hard-style tags (3d / realistic / style_mix / 2.5d etc.)
- person descriptors (subject/age/height/build/body parts)
- anatomy/body-part tags
- concept-bleed tags

Usage:
  python -m scripts.tools.tag_coverage --manifest manifest.jsonl --out report.json

Manifest format:
  JSONL with at least one of these keys per row:
    - caption
    - text
    - prompt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


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


def _contains_any(caption_lower: str, terms: List[str]) -> Tuple[bool, List[str]]:
    hits = []
    for t in terms:
        if t and t in caption_lower:
            hits.append(t)
    return (len(hits) > 0), hits


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze how well captions cover important tag groups.")
    ap.add_argument("--manifest", type=str, required=True, help="Input JSONL manifest")
    ap.add_argument("--out", type=str, default="", help="Optional JSON output report")
    ap.add_argument("--min-rows", type=int, default=0, help="Stop after scanning N rows (0 = all)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    # Local imports after repo_root resolution
    import sys

    sys.path.insert(0, str(repo_root))

    from data.caption_utils import (
        AGE_TAGS,
        ANATOMY_FRAMING_TAGS,
        BODY_PART_TAGS,
        BUILD_BODY_TAGS,
        DOMAIN_TAGS,
        HARD_STYLE_TAGS_FLAT,
        HEIGHT_TAGS,
        QUALITY_TAGS,
        SUBJECT_PREFIXES,
    )

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Not found: {manifest_path}")

    quality_terms = QUALITY_TAGS
    hard_style_terms = HARD_STYLE_TAGS_FLAT

    person_terms = list(SUBJECT_PREFIXES) + list(AGE_TAGS) + list(HEIGHT_TAGS) + list(BUILD_BODY_TAGS)
    anatomy_terms = list(ANATOMY_FRAMING_TAGS) + list(BODY_PART_TAGS)
    concept_bleed_terms = DOMAIN_TAGS.get("concept_bleed", [])

    groups = [
        ("quality_tags", quality_terms),
        ("hard_style_tags", hard_style_terms),
        ("person_descriptors", person_terms),
        ("anatomy_body_parts", anatomy_terms),
        ("concept_bleed_tags", concept_bleed_terms),
    ]

    total = 0
    group_hit_counts = {name: 0 for name, _ in groups}
    term_counts: Dict[str, int] = {}

    for rec in _iter_jsonl(manifest_path):
        cap = _get_caption(rec)
        if not cap:
            continue
        total += 1
        cap_lower = cap.lower()
        for name, terms in groups:
            hit, hits = _contains_any(cap_lower, terms)
            if hit:
                group_hit_counts[name] += 1
                for h in hits:
                    term_counts[h] = term_counts.get(h, 0) + 1
        if args.min_rows and total >= args.min_rows:
            break

    if total == 0:
        print("No captions found in manifest.")
        return

    report = {
        "manifest": str(manifest_path),
        "rows_scanned": total,
        "group_coverage": {},
        "top_terms": [],
    }
    for name, _ in groups:
        report["group_coverage"][name] = group_hit_counts[name] / max(1, total)

    # Simple top-term counts for quick inspection
    top = sorted(term_counts.items(), key=lambda kv: kv[1], reverse=True)[:30]
    report["top_terms"] = [{"term": t, "count": c} for t, c in top]

    # Human readable
    print(f"Tag coverage (rows_scanned={total})")
    for name, _ in groups:
        cov = report["group_coverage"][name]
        print(f"- {name}: {cov:.3f} ({group_hit_counts[name]}/{total})")
    print("\nTop terms:")
    for item in report["top_terms"][:15]:
        print(f"  {item['term']}: {item['count']}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote report: {out_path}")


if __name__ == "__main__":
    main()

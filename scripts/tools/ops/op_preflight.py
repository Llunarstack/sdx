"""
OP preflight: dataset QA checks for “hard styles + person/anatomy + spatial wording”.

This is a fast, non-training step that helps you detect why the model might fail
at 3D/realism/mixes, lose people descriptor ordering, or ignore spatial relations.

Usage:
  python -m scripts.tools op_preflight --manifest manifest.jsonl
  python -m scripts.tools op_preflight --manifest manifest.jsonl --min-hard-style 0.20 --min-person 0.30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _iter_jsonl(path: Path):
    import json as _json

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield _json.loads(line)
            except _json.JSONDecodeError:
                continue


def _get_caption(rec: Dict[str, Any]) -> str:
    for k in ("caption", "text", "prompt"):
        v = rec.get(k)
        if v is not None:
            s = str(v).strip()
            if s:
                return s
    return ""


def _contains_any(caption_lower: str, terms: List[str]) -> bool:
    for t in terms:
        if t and t in caption_lower:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Dataset preflight checks before training.")
    ap.add_argument("--manifest", type=str, required=True, help="Input JSONL manifest")
    ap.add_argument("--out", type=str, default="", help="Optional JSON report")
    ap.add_argument("--min-hard-style", type=float, default=0.10, help="Fail if hard-style coverage < this fraction")
    ap.add_argument("--min-person", type=float, default=0.20, help="Fail if person-descriptor coverage < this fraction")
    ap.add_argument("--min-spatial", type=float, default=0.03, help="Fail if spatial wording coverage < this fraction")
    ap.add_argument(
        "--min-anatomy", type=float, default=0.10, help="Fail if anatomy/body-parts coverage < this fraction"
    )
    ap.add_argument(
        "--min-concept-bleed", type=float, default=0.02, help="Fail if concept-bleed tag coverage < this fraction"
    )
    ap.add_argument(
        "--native-manifest-check",
        action="store_true",
        help="Before Python coverage: run Rust sdx-jsonl-tools `stats` if built (fast JSON/caption sanity).",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
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

    if args.native_manifest_check:
        try:
            from utils.native.native_tools import run_rust_jsonl_stats, rust_jsonl_tools_exe

            exe = rust_jsonl_tools_exe()
            if exe:
                r = run_rust_jsonl_stats(manifest_path)
                print("[native] sdx-jsonl-tools stats:", file=sys.stderr)
                print(r.stdout, file=sys.stderr, end="")
                if r.stderr:
                    print(r.stderr, file=sys.stderr, end="")
            else:
                print("[native] Rust sdx-jsonl-tools not built — skipping stats.", file=sys.stderr)
        except Exception as e:
            print(f"[native] stats skipped: {e}", file=sys.stderr)

    hard_terms = HARD_STYLE_TAGS_FLAT
    person_terms = list(SUBJECT_PREFIXES) + list(AGE_TAGS) + list(HEIGHT_TAGS) + list(BUILD_BODY_TAGS)
    anatomy_terms = list(ANATOMY_FRAMING_TAGS) + list(BODY_PART_TAGS)
    concept_bleed_terms = DOMAIN_TAGS.get("concept_bleed", [])

    spatial_terms = [
        "behind",
        "in front of",
        "next to",
        "beside",
        "under",
        "below",
        "above",
        "left of",
        "right of",
    ]

    total = 0
    hits_hard = hits_person = hits_anatomy = hits_concept_bleed = 0
    hits_spatial_any = 0

    # Optional: also compute quality tag presence just for visibility.
    hits_quality = 0

    for rec in _iter_jsonl(manifest_path):
        cap = _get_caption(rec)
        if not cap:
            continue
        total += 1
        cap_lower = cap.lower()

        if _contains_any(cap_lower, hard_terms):
            hits_hard += 1
        if _contains_any(cap_lower, person_terms):
            hits_person += 1
        if _contains_any(cap_lower, anatomy_terms):
            hits_anatomy += 1
        if _contains_any(cap_lower, concept_bleed_terms):
            hits_concept_bleed += 1
        if any(t in cap_lower for t in spatial_terms):
            hits_spatial_any += 1

        if _contains_any(cap_lower, QUALITY_TAGS):
            hits_quality += 1

    if total == 0:
        raise SystemExit("No captions found in manifest.")

    def frac(n: int) -> float:
        return n / total

    report: Dict[str, Any] = {
        "manifest": str(manifest_path),
        "rows_scanned": total,
        "coverage": {
            "hard_style": frac(hits_hard),
            "person_descriptors": frac(hits_person),
            "anatomy_body_parts": frac(hits_anatomy),
            "concept_bleed": frac(hits_concept_bleed),
            "spatial_wording": frac(hits_spatial_any),
            "quality_tags": frac(hits_quality),
        },
        "thresholds": {
            "min-hard-style": args.min_hard_style,
            "min-person": args.min_person,
            "min-spatial": args.min_spatial,
            "min-anatomy": args.min_anatomy,
            "min-concept-bleed": args.min_concept_bleed,
        },
        "suggestions": [],
    }

    failures: List[str] = []
    if report["coverage"]["hard_style"] < args.min_hard_style:
        failures.append("hard_style")
    if report["coverage"]["person_descriptors"] < args.min_person:
        failures.append("person_descriptors")
    if report["coverage"]["spatial_wording"] < args.min_spatial:
        failures.append("spatial_wording")
    if report["coverage"]["anatomy_body_parts"] < args.min_anatomy:
        failures.append("anatomy_body_parts")
    if report["coverage"]["concept_bleed"] < args.min_concept_bleed:
        failures.append("concept_bleed")

    # Lightweight suggestions: point at sample.py/data pipeline knobs you already have.
    if "hard_style" in failures:
        report["suggestions"].append(
            "Hard-style coverage is low: increase captions with `3d / realistic / style_mix` tags "
            "and consider generating hard-style variants with `--hard-style` for inference-based QA."
        )
    if "person_descriptors" in failures:
        report["suggestions"].append(
            "Person-descriptor coverage is low: ensure captions include subject/age/height/build tags "
            "so `normalize_tag_order` can order them correctly."
        )
    if "anatomy_body_parts" in failures:
        report["suggestions"].append(
            "Anatomy/body-part coverage is low: add tags like `correct hands`, `five fingers`, "
            "`hand focus`, and `correct anatomy` to captions."
        )
    if "spatial_wording" in failures:
        report["suggestions"].append(
            "Spatial wording coverage is low: add explicit relations like `behind`, `next to`, "
            "`under`, `left of/right of` early in captions."
        )
    if "concept_bleed" in failures:
        report["suggestions"].append(
            "Concept-bleed tags are rare: add `distinct colors`, `separate objects`, `defined edges` "
            "(or your dataset’s equivalent) so training can learn separation."
        )

    print("OP preflight coverage:")
    for k, v in report["coverage"].items():
        print(f"- {k}: {v:.3f}")

    if failures:
        print("\nFAIL:", ", ".join(failures))
        for s in report["suggestions"]:
            print("-", s)
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise SystemExit(2)

    print("\nPASS: dataset coverage looks adequate for these core domains.")
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

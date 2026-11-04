#!/usr/bin/env python3
"""Caption tools for JSONL training manifests: normalize, hygiene, lint, and coverage reports.

    python -m scripts.tools captions normalize --in manifest.jsonl --out manifest_norm.jsonl
    python -m scripts.tools captions hygiene data/manifest.jsonl --report-dups --max-overlap-show 15
    python -m scripts.tools captions lint data/manifest.jsonl --fail-on-overlap
    python -m scripts.tools captions tag-coverage --manifest manifest.jsonl --out report.json
    python -m scripts.tools captions spatial-coverage --manifest manifest.jsonl
    python -m scripts.tools captions complex-coverage --manifest manifest.jsonl --thresholds "clothes_wardrobe:0.25"
    python -m scripts.tools captions gap-scout --prompt "..." --lang en
    python -m scripts.tools captions preflight --manifest manifest.jsonl --min-hard-style 0.20

Old command names still work: ``normalize_captions``, ``caption_hygiene``, ``prompt_lint``,
``tag_coverage``, ``spatial_coverage``, ``complex_prompt_coverage``, ``prompt_gap_scout``, ``op_preflight``.
Manifest rows carry the caption in ``caption`` / ``text`` / ``prompt``.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import sys
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.tools._repo_bootstrap import ensure_repo_paths  # noqa: E402

ensure_repo_paths()

from data.manifest_utils import (  # noqa: E402
    caption_from_row,
    first_str,
    iter_manifest_rows,
    negative_caption_from_row,
)


def _prog(cmd: str) -> str:
    return f"python -m scripts.tools captions {cmd}"


def _write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _captions_lower(manifest: Path, limit: int = 0):
    """Lower-cased non-empty captions, at most ``limit`` (0 = all)."""
    n = 0
    for rec in iter_manifest_rows(manifest):
        cap = caption_from_row(rec) if isinstance(rec, dict) else ""
        if not cap:
            continue
        yield cap.lower()
        n += 1
        if limit and n >= limit:
            return


# --- normalize -------------------------------------------------------------------------------

_CHUNK_ROWS = 2048


@functools.cache
def _unicode_hygiene() -> Callable[[dict], Any] | None:
    try:
        from sdx_native.text_hygiene import normalize_jsonl_caption_fields
    except Exception:
        return None
    return normalize_jsonl_caption_fields


def _normalize_row(rec: dict, cfg: dict[str, Any]) -> dict:
    from data.caption_utils import (
        add_anti_blending_and_count,
        apply_training_guidance_to_caption_pair,
        boost_domain_tags,
        boost_hard_style_tags,
        boost_quality_tags,
        normalize_tag_order,
    )

    if cfg["unicode_normalize"] and _unicode_hygiene() is not None:
        _unicode_hygiene()(rec)
    cap = str(rec.get("caption", "") or "")
    neg = str(rec.get("negative_caption", rec.get("negative_prompt", "") or ""))
    cap = boost_domain_tags(boost_quality_tags(boost_hard_style_tags(normalize_tag_order(cap))))
    cap, neg = add_anti_blending_and_count(cap, neg)
    cap, neg = apply_training_guidance_to_caption_pair(
        cap,
        neg,
        shortcomings_mode=cfg["shortcomings_mitigation"],
        shortcomings_2d=cfg["shortcomings_2d"],
        art_guidance_mode=cfg["art_guidance_mode"],
        anatomy_guidance=cfg["anatomy_guidance"],
        style_guidance_mode=cfg["style_guidance_mode"],
        style_guidance_artists=cfg["style_guidance_artists"],
        include_art_guidance_photography=cfg["art_guidance_photography"],
    )
    rec["caption"] = cap.strip()
    if "negative_caption" in rec:
        rec["negative_caption"] = neg.strip()
    elif "negative_prompt" in rec:
        rec["negative_prompt"] = neg.strip()
    return rec


def _normalize_lines(lines: list[str], cfg: dict[str, Any]) -> list[str]:
    """ProcessPool worker: JSONL lines in, normalized JSON strings out (bad lines dropped)."""
    out: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        out.append(json.dumps(_normalize_row(rec, cfg), ensure_ascii=False))
    return out


def _normalize_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog=_prog("normalize"), description="Normalize and boost captions for SDX training.")
    ap.add_argument("--in", dest="inp", type=str, required=True, help="Input JSONL manifest")
    ap.add_argument("--out", dest="out", type=str, required=True, help="Output JSONL manifest")
    ap.add_argument(
        "--unicode-normalize",
        action="store_true",
        help="Apply NFKC + zero-width strip per comma segment before other processing (via sdx_native).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="If >0, parallelize JSONL caption normalization using ProcessPool (order not guaranteed).",
    )
    ap.add_argument(
        "--shortcomings-mitigation",
        type=str,
        default="none",
        choices=["none", "auto", "all"],
        help="Append failure-mode hints (same as train.py --train-shortcomings-mitigation)",
    )
    ap.add_argument("--shortcomings-2d", action="store_true", help="With auto|all: include stylized 2D packs")
    ap.add_argument(
        "--art-guidance-mode",
        type=str,
        default="none",
        choices=["none", "auto", "all"],
        help="Artist-first medium guidance packs (traditional/digital/photography)",
    )
    ap.set_defaults(art_guidance_photography=True)
    ap.add_argument(
        "--no-art-guidance-photography",
        action="store_false",
        dest="art_guidance_photography",
        help="Disable photography packs for --art-guidance-mode auto|all",
    )
    ap.add_argument(
        "--anatomy-guidance",
        type=str,
        default="none",
        choices=["none", "lite", "strong"],
        help="Add anatomy/proportion guidance tags",
    )
    ap.add_argument(
        "--style-guidance-mode",
        type=str,
        default="none",
        choices=["none", "auto", "all"],
        help="Style-domain guidance packs (anime/comic/concept/game/photo language)",
    )
    ap.set_defaults(style_guidance_artists=True)
    ap.add_argument(
        "--no-style-guidance-artists",
        action="store_false",
        dest="style_guidance_artists",
        help="Disable artist/game-name stabilization cues in style guidance",
    )
    args = ap.parse_args(argv)

    in_path, out_path = Path(args.inp), Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        k: getattr(args, k)
        for k in (
            "unicode_normalize",
            "shortcomings_mitigation",
            "shortcomings_2d",
            "art_guidance_mode",
            "art_guidance_photography",
            "anatomy_guidance",
            "style_guidance_mode",
            "style_guidance_artists",
        )
    }
    count = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as fw:
        if args.workers > 0:
            max_workers = min(args.workers, max(1, os.cpu_count() or 4))
            with in_path.open(encoding="utf-8", errors="replace") as f, ProcessPoolExecutor(max_workers) as ex:
                pending, chunk = [], []
                for line in f:
                    chunk.append(line)
                    if len(chunk) >= _CHUNK_ROWS:
                        pending.append(ex.submit(_normalize_lines, chunk, cfg))
                        chunk = []
                if chunk:
                    pending.append(ex.submit(_normalize_lines, chunk, cfg))
                for fut in pending:
                    for out_line in fut.result():
                        fw.write(out_line + "\n")
                        count += 1
        else:
            for rec in iter_manifest_rows(in_path):
                fw.write(json.dumps(_normalize_row(rec, cfg), ensure_ascii=False) + "\n")
                count += 1
    print(f"Normalized {count} captions -> {out_path}")
    return 0


# --- hygiene ---------------------------------------------------------------------------------


def _hygiene_main(argv: list[str]) -> int:
    from sdx_native.text_hygiene import caption_fingerprint, normalize_caption_for_training, pos_neg_token_overlap

    p = argparse.ArgumentParser(prog=_prog("hygiene"), description="Caption hygiene report for JSONL manifests.")
    p.add_argument("jsonl", type=Path, help="Manifest JSONL path")
    p.add_argument("--report-dups", action="store_true", help="Report duplicate normalized caption fingerprints")
    p.add_argument("--report-overlap", action="store_true", help="Report rows with pos/neg token overlap")
    p.add_argument("--overlap-min-jaccard", type=float, default=0.0, help="Min Jaccard to print overlap row")
    p.add_argument("--max-overlap-show", type=int, default=30, help="Max overlap rows to print")
    p.add_argument("--normalize-samples", type=int, default=0, help="Print first N before/after normalize")
    p.add_argument("--empty-warn", action="store_true", help="Count rows with empty caption")
    args = p.parse_args(argv)
    path: Path = args.jsonl
    if not path.is_file():
        print(f"Not a file: {path}", file=sys.stderr)
        return 1

    dup_buckets: defaultdict[str, list[str]] = defaultdict(list)
    overlap_rows: list[tuple[float, str, str, str]] = []
    empty_captions = lines = 0
    samples_left = int(args.normalize_samples or 0)
    for row in iter_manifest_rows(path):
        if not isinstance(row, dict):
            continue
        lines += 1
        cap = caption_from_row(row)
        neg = negative_caption_from_row(row)
        key = first_str(row, ("image_path", "path", "image")) or f"line{lines}"
        if not cap:
            empty_captions += 1
            continue
        if samples_left > 0:
            norm = normalize_caption_for_training(cap)
            print(f"--- sample {samples_left} ---")
            print(f"before: {cap[:500]}{'...' if len(cap) > 500 else ''}")
            print(f"after:  {norm[:500]}{'...' if len(norm) > 500 else ''}")
            samples_left -= 1
        if args.report_dups:
            dup_buckets[caption_fingerprint(cap, algorithm="sha256")].append(key)
        if args.report_overlap and neg:
            inter, _u, j = pos_neg_token_overlap(cap, neg)
            if j >= float(args.overlap_min_jaccard) and inter > 0:
                overlap_rows.append((j, key, cap[:120], neg[:120]))

    if args.empty_warn:
        print(f"rows_read={lines} empty_caption_rows={empty_captions}")
    if args.report_dups:
        multi = sorted(((fp, ps) for fp, ps in dup_buckets.items() if len(ps) > 1), key=lambda x: -len(x[1]))
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


# --- lint ------------------------------------------------------------------------------------


def _lint_main(argv: list[str]) -> int:
    from utils.prompt.text import PromptLintOptions, lint_jsonl_path, should_fail

    p = argparse.ArgumentParser(prog=_prog("lint"), description="Prompt lint for SDX JSONL manifests")
    p.add_argument("input", type=str, help="Path to JSONL manifest (one JSON object per line)")
    p.add_argument(
        "--min-caption-len-chars", type=int, default=0, help="Drop captions shorter than N characters (0=off)"
    )
    p.add_argument(
        "--max-caption-tokens", type=int, default=0, help="Max distinct-token-set size for positive captions (0=off)"
    )
    p.add_argument("--top-overlap-tokens", type=int, default=10, help="Print top overlapping tokens (0=off)")
    p.add_argument("--fail-on-overlap", action="store_true", help="Exit non-zero if any pos/neg overlap rows exist")
    p.add_argument("--json-report", type=str, default="", help="Optional path to write full stats JSON")
    args = p.parse_args(argv)

    inp = Path(args.input)
    if not inp.is_file():
        print(f"Not found: {inp}", file=sys.stderr)
        return 2
    opts = PromptLintOptions(
        min_caption_len_chars=int(args.min_caption_len_chars),
        max_caption_tokens=int(args.max_caption_tokens),
        top_overlap_tokens=int(args.top_overlap_tokens),
        fail_on_overlap=bool(args.fail_on_overlap),
    )
    stats = lint_jsonl_path(inp, opts)
    if args.json_report:
        _write_json(args.json_report, stats)

    print(f"promptlint: file {stats['promptlint_file']}")
    for k in (
        "lines_total",
        "json_parse_errors",
        "empty_caption_rows",
        "json_missing_pos_tokens",
        "rows_over_max_tokens",
        "rows_ok",
    ):
        print(f"{k}: {stats[k]}")
    print(
        "caption_token_count(set_size): "
        f"avg={stats['caption_token_count_set_size_avg']:.2f} "
        f"min={stats['caption_token_count_set_size_min']} max={stats['caption_token_count_set_size_max']}"
    )
    print(f"pos_neg_overlap_rows: {stats['pos_neg_overlap_rows']}")
    print(f"pos_neg_overlap_max_distinct_tokens: {stats['pos_neg_overlap_max_distinct_tokens']}")
    top = stats.get("top_overlap_tokens") or []
    if top:
        print("top_overlap_tokens: " + ", ".join(f"{tok}({cnt})" for tok, cnt in top))
    return 1 if should_fail(stats, opts) else 0


# --- tag / spatial coverage + preflight ------------------------------------------------------

SPATIAL_TERMS = [
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


def _tag_groups() -> dict[str, list[str]]:
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

    return {
        "quality_tags": list(QUALITY_TAGS),
        "hard_style_tags": list(HARD_STYLE_TAGS_FLAT),
        "person_descriptors": [*SUBJECT_PREFIXES, *AGE_TAGS, *HEIGHT_TAGS, *BUILD_BODY_TAGS],
        "anatomy_body_parts": [*ANATOMY_FRAMING_TAGS, *BODY_PART_TAGS],
        "concept_bleed_tags": list(DOMAIN_TAGS.get("concept_bleed", [])),
    }


def _tag_coverage_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog=_prog("tag-coverage"), description="Analyze how well captions cover important tag groups."
    )
    ap.add_argument("--manifest", type=str, required=True, help="Input JSONL manifest")
    ap.add_argument("--out", type=str, default="", help="Optional JSON output report")
    ap.add_argument("--min-rows", type=int, default=0, help="Stop after scanning N rows (0 = all)")
    args = ap.parse_args(argv)
    manifest = Path(args.manifest)
    if not manifest.exists():
        raise SystemExit(f"Not found: {manifest}")

    groups = _tag_groups()
    total = 0
    hits = dict.fromkeys(groups, 0)
    term_counts: dict[str, int] = defaultdict(int)
    for cap in _captions_lower(manifest, args.min_rows):
        total += 1
        for name, terms in groups.items():
            found = [t for t in terms if t and t in cap]
            if found:
                hits[name] += 1
                for t in found:
                    term_counts[t] += 1
    if total == 0:
        print("No captions found in manifest.")
        return 0

    top = sorted(term_counts.items(), key=lambda kv: kv[1], reverse=True)[:30]
    report = {
        "manifest": str(manifest),
        "rows_scanned": total,
        "group_coverage": {name: hits[name] / total for name in groups},
        "top_terms": [{"term": t, "count": c} for t, c in top],
    }
    print(f"Tag coverage (rows_scanned={total})")
    for name in groups:
        print(f"- {name}: {report['group_coverage'][name]:.3f} ({hits[name]}/{total})")
    print("\nTop terms:")
    for item in report["top_terms"][:15]:
        print(f"  {item['term']}: {item['count']}")
    if args.out:
        _write_json(args.out, report)
        print(f"\nWrote report: {args.out}")
    return 0


def _spatial_coverage_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog=_prog("spatial-coverage"), description="Analyze spatial relation keywords in captions."
    )
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--min-rows", type=int, default=0, help="0 = all")
    args = ap.parse_args(argv)
    manifest = Path(args.manifest)
    if not manifest.exists():
        raise SystemExit(f"Not found: {manifest}")

    counts = dict.fromkeys(SPATIAL_TERMS, 0)
    total = 0
    for cap in _captions_lower(manifest, args.min_rows):
        total += 1
        for t in SPATIAL_TERMS:
            if t in cap:
                counts[t] += 1
    if total == 0:
        print("No captions found.")
        return 0
    report = {
        "manifest": str(manifest),
        "rows_scanned": total,
        "term_coverage": {t: counts[t] / total for t in SPATIAL_TERMS},
        "counts": counts,
    }
    print(f"Spatial coverage (rows_scanned={total})")
    for t in SPATIAL_TERMS:
        print(f"- {t}: {report['term_coverage'][t]:.3f} ({counts[t]}/{total})")
    if args.out:
        _write_json(args.out, report)
        print(f"Wrote report: {args.out}")
    return 0


_PREFLIGHT_SUGGESTIONS = {
    "hard_style": "Hard-style coverage is low: increase captions with `3d / realistic / style_mix` tags "
    "and consider generating hard-style variants with `--hard-style` for inference-based QA.",
    "person_descriptors": "Person-descriptor coverage is low: ensure captions include subject/age/height/build tags "
    "so `normalize_tag_order` can order them correctly.",
    "spatial_wording": "Spatial wording coverage is low: add explicit relations like `behind`, `next to`, "
    "`under`, `left of/right of` early in captions.",
    "anatomy_body_parts": "Anatomy/body-part coverage is low: add tags like `correct hands`, `five fingers`, "
    "`hand focus`, and `correct anatomy` to captions.",
    "concept_bleed": "Concept-bleed tags are rare: add `distinct colors`, `separate objects`, `defined edges` "
    "(or your dataset’s equivalent) so training can learn separation.",
}


def _preflight_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog=_prog("preflight"), description="Dataset preflight checks before training.")
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
    args = ap.parse_args(argv)
    manifest = Path(args.manifest)
    if not manifest.exists():
        raise SystemExit(f"Not found: {manifest}")

    if args.native_manifest_check:
        try:
            from utils.native import run_rust_jsonl_stats, rust_jsonl_tools_exe

            if rust_jsonl_tools_exe():
                r = run_rust_jsonl_stats(manifest)
                print("[native] sdx-jsonl-tools stats:", file=sys.stderr)
                print(r.stdout + (r.stderr or ""), file=sys.stderr, end="")
            else:
                print("[native] Rust sdx-jsonl-tools not built — skipping stats.", file=sys.stderr)
        except Exception as e:
            print(f"[native] stats skipped: {e}", file=sys.stderr)

    g = _tag_groups()
    checks = {
        "hard_style": g["hard_style_tags"],
        "person_descriptors": g["person_descriptors"],
        "anatomy_body_parts": g["anatomy_body_parts"],
        "concept_bleed": g["concept_bleed_tags"],
        "spatial_wording": SPATIAL_TERMS,
        "quality_tags": g["quality_tags"],
    }
    hits = dict.fromkeys(checks, 0)
    total = 0
    for cap in _captions_lower(manifest):
        total += 1
        for name, terms in checks.items():
            if any(t and t in cap for t in terms):
                hits[name] += 1
    if total == 0:
        raise SystemExit("No captions found in manifest.")

    coverage = {k: hits[k] / total for k in checks}
    mins = {
        "hard_style": args.min_hard_style,
        "person_descriptors": args.min_person,
        "spatial_wording": args.min_spatial,
        "anatomy_body_parts": args.min_anatomy,
        "concept_bleed": args.min_concept_bleed,
    }
    failures = [k for k, lo in mins.items() if coverage[k] < lo]
    report: dict[str, Any] = {
        "manifest": str(manifest),
        "rows_scanned": total,
        "coverage": coverage,
        "thresholds": {
            "min-hard-style": args.min_hard_style,
            "min-person": args.min_person,
            "min-spatial": args.min_spatial,
            "min-anatomy": args.min_anatomy,
            "min-concept-bleed": args.min_concept_bleed,
        },
        "suggestions": [_PREFLIGHT_SUGGESTIONS[k] for k in _PREFLIGHT_SUGGESTIONS if k in failures],
    }
    print("OP preflight coverage:")
    for k, v in coverage.items():
        print(f"- {k}: {v:.3f}")
    if args.out:
        _write_json(args.out, report)
    if failures:
        print("\nFAIL:", ", ".join(failures))
        for s in report["suggestions"]:
            print("-", s)
        raise SystemExit(2)
    print("\nPASS: dataset coverage looks adequate for these core domains.")
    return 0


# --- complex prompt coverage / gap scout -----------------------------------------------------

# NSFW detection is token-based so it matches the dataset's own vocabulary; its suggestion stays generic.
CATEGORIES: dict[str, list[str]] = {
    "nsfw_descriptors": [
        "nsfw",
        "adult",
        "erotic",
        "explicit",
        "porn",
        "nude",
        "lingerie",
        "underwear",
        "swimsuit",
        "bare",
        "revealing",
    ],
    "weird_strange": [
        "surreal",
        "abstract",
        "bizarre",
        "uncanny",
        "weird",
        "strange",
        "impossible",
        "glitch",
        "cyberpunk",
        "horror",
        "creepy",
        "grotesque",
        "eldritch",
        "non-euclidean",
    ],
    "clothes_wardrobe": [
        "dress",
        "coat",
        "jacket",
        "shirt",
        "t-shirt",
        "hoodie",
        "sweater",
        "uniform",
        "armor",
        "gloves",
        "boots",
        "scarf",
        "belt",
        "cape",
        "cloak",
        "robe",
        "kimono",
        "suit",
        "skirt",
        "leggings",
        "leotard",
        "lingerie",
        "underwear",
        "swimsuit",
        "corset",
        "tactical vest",
    ],
    "anatomy_details": [
        "tattoo",
        "scar",
        "freckles",
        "piercing",
        "mole",
        "beauty mark",
        "bandage",
        "wounds",
        "bruise",
        "veins",
        "muscle",
        "body hair",
    ],
    "weapons_props": [
        "sword",
        "gun",
        "rifle",
        "pistol",
        "knife",
        "blade",
        "bow",
        "arrow",
        "staff",
        "hammer",
        "axe",
        "spear",
        "katana",
        "revolver",
        "crossbow",
        "grenade",
        "blaster",
        "laser",
        "trident",
        "wand",
    ],
    "food_drink": [
        "pizza",
        "ramen",
        "sushi",
        "burger",
        "taco",
        "steak",
        "soup",
        "salad",
        "sandwich",
        "cake",
        "pie",
        "dessert",
        "chocolate",
        "coffee",
        "tea",
        "noodles",
        "dumplings",
        "bread",
        "cheese",
        "spaghetti",
    ],
    "text_in_image": [
        "legible text",
        "clear text",
        "readable text",
        "correct spelling",
        "sign",
        "signage",
        "label",
        "lettering",
        "words",
        "caption",
        "text that says",
        "sign that says",
    ],
    "foreground_framing": [
        "foreground",
        "in the foreground",
        "foreground bokeh",
        "near the camera",
        "closest",
        "close-up foreground",
    ],
    "background_framing": [
        "background",
        "in the background",
        "distant background",
        "far background",
        "backdrop",
        "sky",
        "horizon",
    ],
    "fine_details": [
        "intricate",
        "ornate",
        "embroidery",
        "stitched",
        "seams",
        "texture",
        "textured",
        "fabric",
        "weathered",
        "rust",
        "scuffed",
        "steam",
        "smoke",
        "fog",
        "dust",
        "particles",
        "micro details",
        "depth of field",
        "bokeh",
    ],
    "hands_fingers": ["hands", "correct hands", "five fingers", "hand focus", "visible hands"],
    "pose_actions": [
        "pose",
        "stance",
        "standing",
        "walking",
        "running",
        "dancing",
        "crouching",
        "kneeling",
        "sitting",
        "lying",
        "lying down",
        "holding",
        "gripping",
        "grabbing",
        "wielding",
        "carrying",
        "aiming",
        "shooting",
        "swinging",
        "attacking",
        "casting",
        "eating",
        "drinking",
    ],
    "character_scene": [
        "group",
        "crowd",
        "team",
        "duo",
        "multiple people",
        "room full of people",
        "many people",
        "several people",
        "gathering",
        "audience",
        "multiple characters",
    ],
    "clothes_materials": [
        "leather",
        "silk",
        "latex",
        "denim",
        "wool",
        "cotton",
        "chainmail",
        "lace",
        "metallic",
        "ribbon",
        "feather",
        "fur",
        "plaid",
        "stitched",
        "sewn",
    ],
    "art_styles": [
        "anime",
        "manga",
        "manhwa",
        "illustration",
        "digital painting",
        "oil painting",
        "watercolor",
        "cel shading",
        "comic",
        "cartoon",
        "cinematic",
        "baroque",
        "gothic",
        "cyberpunk",
        "steampunk",
        "noir",
        "retro",
        "grainy film",
        "photography",
        "macro",
    ],
}

SUGGESTIONS_EN: dict[str, str] = {
    "nsfw_descriptors": "If you want NSFW behavior, keep your NSFW descriptors consistent across the dataset and include them early in captions.",
    "weird_strange": "Add more surreal/weird descriptors so the model learns non-standard visuals consistently.",
    "clothes_wardrobe": "Add wardrobe/clothing details (garments, materials, accessories) so outfits are stable.",
    "anatomy_details": "Add anatomy micro-details (tattoos/scars/freckles/piercings) for higher consistency and specificity.",
    "weapons_props": "Add explicit weapon/prop words and (ideally) action verbs like 'holding' / 'gripping'.",
    "food_drink": "Add explicit food/drink words and eating/serving context if relevant.",
    "text_in_image": "If you need text in-image, include explicit 'legible/clear readable text' wording and spelling cues.",
    "foreground_framing": "Add foreground framing cues (foreground bokeh, nearby objects) early in captions.",
    "background_framing": "Add background framing cues (detailed background, backdrop, sky/horizon) early in captions.",
    "fine_details": "Add texture/detail descriptors (intricate, stitched, weathered, particles, micro details).",
    "hands_fingers": "If hands fail, add 'correct hands', 'five fingers', and 'hand focus' consistently.",
    "pose_actions": "Add explicit pose/action wording (standing/walking + holding/gripping/carrying verbs) so scenes don’t collapse.",
    "character_scene": "If you want multi-character scenes, include crowd/group words consistently (group/crowd/team/room full of people).",
    "clothes_materials": "Add clothing material terms (leather/silk/latex/denim/wool/cotton) so outfits are stable.",
    "art_styles": "Add explicit art-style tokens (anime/manga/manhwa/oil painting/cyberpunk/noir/etc.) early in captions.",
}


def term_present(caption_lower: str, term: str) -> bool:
    """Multi-word terms match as substrings; single words need word-ish boundaries."""
    t = term.strip().lower()
    if not t:
        return False
    if " " in t:
        return t in caption_lower
    return re.search(rf"(?<![a-z0-9_]){re.escape(t)}(?![a-z0-9_])", caption_lower) is not None


def category_matcher(terms: list[str]) -> Callable[[str], bool]:
    """Precompiled ``any(term_present(caption_lower, t) for t in terms)`` (one regex per category)."""
    words = [t.strip().lower() for t in terms if t.strip()]
    phrases = [t for t in words if " " in t]
    singles = [t for t in words if " " not in t]
    rx = re.compile(r"(?<![a-z0-9_])(?:" + "|".join(map(re.escape, singles)) + r")(?![a-z0-9_])") if singles else None
    return lambda s: any(p in s for p in phrases) or (rx is not None and rx.search(s) is not None)


def _json_categories(path_str: str, what: str) -> dict:
    p = Path(path_str)
    if not p.exists():
        raise SystemExit(f"Not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    cats = data.get("categories", data)
    if not isinstance(cats, dict):
        raise SystemExit(f'{what} must be {{"categories": {{...}}}} or a category mapping.')
    return cats


def load_custom_categories(path_str: str) -> dict[str, list[str]]:
    """``{"categories": {name: {"terms": [...]}}}`` (or the bare mapping) -> ``{name: terms}``."""
    out: dict[str, list[str]] = {}
    for name, spec in _json_categories(path_str, "custom-categories-json").items():
        terms = spec.get("terms") if isinstance(spec, dict) else None
        if isinstance(terms, list) and all(isinstance(x, str) for x in terms):
            out[str(name)] = terms
    return out


def suggestion_picker(lang: str, suggestions_json: str, categories: dict[str, list[str]]) -> Callable[[str], str]:
    """Per-category hint: ``--suggestions-json`` entry for ``lang``, else built-in English, else a localized template."""
    from utils.prompt.text import generic_suggestion

    custom: dict[str, dict[str, str]] = {}
    if suggestions_json:
        for cat, spec in _json_categories(suggestions_json, "suggestions-json").items():
            if isinstance(spec, dict):
                custom[str(cat)] = {str(k).lower(): v for k, v in spec.items() if isinstance(v, str)}
    norm = (lang or "en").strip().lower()

    def pick(category: str) -> str:
        hit = custom.get(category, {}).get(norm)
        if hit:
            return hit
        if norm in ("en", "eng", "english"):
            return SUGGESTIONS_EN.get(category, "")
        return generic_suggestion(category, categories.get(category, []), norm)

    return pick


def _parse_thresholds(thresholds_str: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in thresholds_str.split(","):
        k, sep, v = part.partition(":")
        if not sep:
            continue
        try:
            out[k.strip()] = float(v.strip())
        except ValueError:
            continue
    return out


def _add_category_args(ap: argparse.ArgumentParser, suggestions_help: str, custom_help: str) -> None:
    ap.add_argument("--lang", type=str, default="en", help="Language code/name for suggestions (e.g. en, zh, es, fr)")
    ap.add_argument("--suggestions-json", type=str, default="", help=suggestions_help)
    ap.add_argument("--custom-categories-json", type=str, default="", help=custom_help)


def _categories(args: argparse.Namespace) -> dict[str, list[str]]:
    cats = dict(CATEGORIES)
    if args.custom_categories_json:
        cats.update(load_custom_categories(args.custom_categories_json))
    return cats


def _complex_coverage_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog=_prog("complex-coverage"), description="Complex prompt coverage for tricky categories."
    )
    ap.add_argument("--manifest", type=str, required=True, help="Input JSONL manifest")
    ap.add_argument("--out", type=str, default="", help="Optional JSON output report")
    ap.add_argument(
        "--thresholds",
        type=str,
        default="",
        help='Optional thresholds like "clothes_wardrobe:0.25,weapons_props:0.05"; only these categories are FAIL-checked.',
    )
    ap.add_argument("--min-rows", type=int, default=0, help="Stop after scanning N rows (0 = all)")
    _add_category_args(
        ap,
        "Optional JSON file with localized suggestion text per category. Supported formats: "
        '1) {"{category}": {"en": "...", "es": "..."}} '
        '2) {"categories": {"{category}": {"en": "...", "es": "..."}}} '
        "If not provided (or missing language), tool falls back to built-in English.",
        'Optional JSON file with extra/override categories: {"categories": {name: {"terms": [...]}}}',
    )
    args = ap.parse_args(argv)
    manifest = Path(args.manifest)
    if not manifest.exists():
        raise SystemExit(f"Not found: {manifest}")

    categories = _categories(args)
    matchers = {name: category_matcher(terms) for name, terms in categories.items()}
    thresholds = _parse_thresholds(args.thresholds)
    counts = dict.fromkeys(categories, 0)
    total = 0
    for cap in _captions_lower(manifest, args.min_rows):
        total += 1
        for name, match in matchers.items():
            if match(cap):
                counts[name] += 1
    if total == 0:
        raise SystemExit("No captions found in manifest.")

    fractions = {k: counts[k] / total for k in categories}
    failed: list[str] = []
    for name, frac in sorted(fractions.items(), key=lambda kv: kv[1]):
        mark = ""
        if name in thresholds and frac < thresholds[name]:
            failed.append(name)
            mark = "  <-- FAIL"
        print(f"- {name}: {frac:.3f} ({counts[name]}/{total}){mark}")
    report = {
        "manifest": str(manifest),
        "rows_scanned": total,
        "counts": counts,
        "fractions": fractions,
        "thresholds": thresholds,
    }
    if failed:
        print("\nFAIL categories:", ", ".join(failed))
        pick = suggestion_picker(args.lang, args.suggestions_json, categories)
        for name in failed:
            msg = pick(name)
            if msg:
                print("-", msg)
        report |= {"failed": failed, "suggestions_lang": args.lang}
    else:
        print("\nPASS: coverage computed (and thresholds passed, if provided).")
    if args.out:
        _write_json(args.out, report)
    if failed:
        raise SystemExit(2)
    return 0


def _gap_scout_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog=_prog("gap-scout"), description="Find missing tricky categories in a prompt/caption."
    )
    ap.add_argument("--prompt", type=str, default="", help="Prompt/caption text")
    ap.add_argument("--prompt-file", type=str, default="", help="File containing prompt/caption")
    _add_category_args(
        ap,
        "Optional JSON file with localized suggestion text per category: "
        '{"{category}": {"en": "...", "es": "..."}} or {"categories": {...}}. '
        "Missing languages fall back to built-in English or a localized generic template.",
        'Optional JSON file like {"categories": {"name": {"terms":[...]}}} to extend categories.',
    )
    ap.add_argument("--json", action="store_true", help="Output JSON only")
    args = ap.parse_args(argv)

    if args.prompt.strip():
        prompt = args.prompt.strip()
    elif args.prompt_file.strip():
        p = Path(args.prompt_file)
        if not p.exists():
            raise SystemExit(f"Not found: {p}")
        prompt = p.read_text(encoding="utf-8", errors="ignore").strip()
    else:
        raise SystemExit("Provide --prompt or --prompt-file.")

    categories = _categories(args)
    cap_lower = prompt.lower()
    present = sorted(name for name, terms in categories.items() if category_matcher(terms)(cap_lower))
    missing = sorted(set(categories) - set(present))
    pick = suggestion_picker(args.lang, args.suggestions_json, categories)
    out: dict[str, Any] = {"prompt": prompt, "present_categories": present, "missing_categories": missing}
    if missing:
        out["missing_suggestions"] = {name: pick(name) for name in missing}
    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0

    print("Prompt gap scout")
    print(f"- Present: {', '.join(present) if present else '(none)'}")
    print(f"- Missing: {', '.join(missing) if missing else '(none)'}")
    print()
    if missing:
        print("Suggestions:")
        for name, msg in out["missing_suggestions"].items():
            if msg:
                print(f"- {name}: {msg}")
    else:
        print("PASS: prompt covers most tricky categories (by keyword heuristic).")
    return 0


COMMANDS: dict[str, Callable[[list[str]], int]] = {
    "normalize": _normalize_main,
    "hygiene": _hygiene_main,
    "lint": _lint_main,
    "tag-coverage": _tag_coverage_main,
    "spatial-coverage": _spatial_coverage_main,
    "complex-coverage": _complex_coverage_main,
    "gap-scout": _gap_scout_main,
    "preflight": _preflight_main,
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        print("Commands: " + ", ".join(COMMANDS))
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    return int(COMMANDS[argv[0]](argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())

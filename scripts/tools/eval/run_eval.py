#!/usr/bin/env python3
"""
Run the competitor-weakness eval harness over a set of generated images.

Input maps each suite item id to an image, one of:
  * a directory containing ``<id>.png`` / ``<id>.jpg`` files, or
  * a JSONL file with rows ``{"id": "<suite_id>", "image_path": "..."}``
    (``"prompt"`` may be given instead of ``"id"`` to match by prompt text).

Output: a JSON report (per-item + per-category aggregates + coverage) and a
printed summary table. Missing backends (OCR, native counter) reduce coverage
rather than failing the run.

Examples:
    # Emit the suite as a manifest to generate against, then score a folder:
    python -m scripts.tools.eval.run_eval --dump-suite suite.jsonl
    python -m scripts.tools.eval.run_eval --images ./out_dir --out report.json
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.runtime.jsonutil import dumps as json_dumps  # noqa: E402
from utils.runtime.jsonutil import loads as json_loads  # noqa: E402

from scripts.tools.eval import scorers  # noqa: E402
from scripts.tools.eval.prompt_suite import (  # noqa: E402
    CATEGORIES,
    SUITE,
    SUITE_VERSION,
    suite_as_manifest_rows,
)

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _resolve_images(images_arg: str) -> dict[str, str]:
    """Return {suite_id: image_path}."""
    p = Path(images_arg)
    by_id: dict[str, str] = {}
    if p.is_dir():
        for item in SUITE:
            for ext in _IMG_EXTS:
                cand = p / f"{item.id}{ext}"
                if cand.exists():
                    by_id[item.id] = str(cand)
                    break
        return by_id
    # JSONL mapping.
    prompt_to_id = {s.prompt: s.id for s in SUITE}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json_loads(line)
            sid = row.get("id")
            if sid is None and "prompt" in row:
                sid = prompt_to_id.get(str(row["prompt"]))
            img = row.get("image_path") or row.get("image") or row.get("path")
            if sid and isinstance(img, str):
                by_id[str(sid)] = img
    return by_id


def evaluate(images_by_id: dict[str, str], *, clip_model: str, device: str) -> dict:
    suite_by_id = {s.id: s for s in SUITE}
    items = []
    for sid, img_path in images_by_id.items():
        item = suite_by_id.get(sid)
        if item is None:
            continue
        adh, adh_d = scorers.clip_adherence(img_path, item.prompt, model_id=clip_model, device=device)
        rec: dict = {
            "id": sid,
            "category": item.category,
            "image_path": img_path,
            "adherence": adh,
            "adherence_detail": adh_d,
        }
        if item.category == "count" and item.expected_count is not None:
            cs, cd = scorers.count_score(img_path, item.expected_count)
            rec["count_score"] = cs
            rec["count_detail"] = cd
        if item.category == "text_render" and item.expected_text is not None:
            ts, td = scorers.text_score(img_path, item.expected_text)
            rec["text_score"] = ts
            rec["text_detail"] = td
        if item.category == "occlusion":
            os_, od = scorers.occlusion_score(img_path)
            rec["occlusion_score"] = os_
            rec["occlusion_detail"] = od
        if item.category == "reflection":
            rs, rd = scorers.reflection_score(img_path)
            rec["reflection_score"] = rs
            rec["reflection_detail"] = rd
        if item.category in ("relation", "bind"):
            bs, bd = scorers.spatial_bind_score(img_path, item.prompt)
            rec["spatial_bind_score"] = bs
            rec["spatial_bind_detail"] = bd
        items.append(rec)

    # Per-category aggregates (mean over non-None scores).
    def _mean(vals: list) -> float | None:
        v = [x for x in vals if x is not None]
        return round(statistics.fmean(v), 4) if v else None

    per_cat: dict[str, dict] = {}
    for cat in CATEGORIES:
        cat_items = [r for r in items if r["category"] == cat]
        if not cat_items:
            continue
        agg = {"n": len(cat_items), "adherence": _mean([r["adherence"] for r in cat_items])}
        if cat == "count":
            agg["count_score"] = _mean([r.get("count_score") for r in cat_items])
        if cat == "text_render":
            agg["text_score"] = _mean([r.get("text_score") for r in cat_items])
        if cat == "occlusion":
            agg["occlusion_score"] = _mean([r.get("occlusion_score") for r in cat_items])
        if cat == "reflection":
            agg["reflection_score"] = _mean([r.get("reflection_score") for r in cat_items])
        if cat in ("relation", "bind"):
            agg["spatial_bind_score"] = _mean([r.get("spatial_bind_score") for r in cat_items])
        per_cat[cat] = agg

    return {
        "suite_version": SUITE_VERSION,
        "availability": scorers.availability(clip_model, device),
        "n_scored": len(items),
        "n_suite": len(SUITE),
        "overall_adherence": _mean([r["adherence"] for r in items]),
        "per_category": per_cat,
        "items": items,
    }


def _print_summary(report: dict) -> None:
    av = report["availability"]
    print(f"\n=== sdx eval harness (suite v{report['suite_version']}) ===")
    print(f"scored {report['n_scored']}/{report['n_suite']} suite items")
    print(f"backends: clip={av['clip_adherence']} count_native={av['count_native']} ocr={av['text_ocr']}")
    oa = report["overall_adherence"]
    print(f"overall adherence: {oa if oa is not None else 'n/a'}")
    print(f"{'category':<16}{'n':>4}{'adherence':>12}{'extra':>16}")
    for cat, agg in report["per_category"].items():
        extra = ""
        if "count_score" in agg:
            extra = f"count={agg['count_score']}"
        if "text_score" in agg:
            extra = f"text={agg['text_score']}"
        if "occlusion_score" in agg:
            extra = f"occ={agg['occlusion_score']}"
        if "reflection_score" in agg:
            extra = f"refl={agg['reflection_score']}"
        if "spatial_bind_score" in agg:
            extra = f"bind={agg['spatial_bind_score']}"
        adh = agg["adherence"]
        print(f"{cat:<16}{agg['n']:>4}{(adh if adh is not None else 'n/a'):>12}{extra:>16}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the sdx competitor-weakness eval harness")
    ap.add_argument("--images", help="Directory of <id>.png, or a JSONL id->image_path mapping")
    ap.add_argument("--out", default="", help="Write full JSON report here")
    ap.add_argument("--dump-suite", default="", help="Write the prompt suite as JSONL and exit")
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.dump_suite:
        rows = suite_as_manifest_rows()
        Path(args.dump_suite).write_text("\n".join(json_dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        print(f"wrote {len(rows)} suite prompts -> {args.dump_suite}")
        return 0

    if not args.images:
        ap.error("--images is required (or use --dump-suite)")
    images_by_id = _resolve_images(args.images)
    if not images_by_id:
        print("No images matched suite ids. Name files <id>.png or provide a JSONL mapping.", file=sys.stderr)
        return 2

    report = evaluate(images_by_id, clip_model=args.clip_model, device=args.device)
    if args.out:
        Path(args.out).write_text(json_dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())

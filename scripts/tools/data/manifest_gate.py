#!/usr/bin/env python3
"""
One-stop manifest gate for training.

This stitches together existing lightweight QC tools so you can fail-fast on
dataset issues that usually destroy training runs:

- prompt lint (caption empties, token overlap, token-set size heuristic)
- caption hygiene (duplicate fingerprints / pos-neg overlap)
- optional image quality QC (sharpness/contrast) when PIL is available
- optional rust stats/validate when native tools exist (via utils.native)

This script is intentionally conservative: by default it reports; you enable
fail-fast thresholds via flags.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _try_import_pil() -> bool:
    try:
        import PIL  # noqa: F401

        return True
    except Exception:
        return False


def _run_prompt_lint(manifest: Path, *, min_caption_len_chars: int, max_caption_tokens: int, fail_on_overlap: bool) -> Dict[str, Any]:
    from utils.prompt.prompt_lint import PromptLintOptions, lint_jsonl_path

    opts = PromptLintOptions(
        min_caption_len_chars=int(min_caption_len_chars),
        max_caption_tokens=int(max_caption_tokens),
        top_overlap_tokens=20,
        fail_on_overlap=bool(fail_on_overlap),
    )
    return dict(lint_jsonl_path(manifest, opts))


def _run_caption_hygiene(manifest: Path, *, report_dups: bool, report_overlap: bool) -> Dict[str, Any]:
    # Keep this as a callable report (no printing).
    import json as _json
    from collections import defaultdict

    from sdx_native.text_hygiene import caption_fingerprint, normalize_caption_for_training, pos_neg_token_overlap

    dup_buckets: Dict[str, int] = defaultdict(int)
    overlap_rows = 0
    rows = 0
    with manifest.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            try:
                row = _json.loads(t)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            cap = str(row.get("caption") or row.get("text") or "")
            neg = str(row.get("negative_caption") or row.get("negative_prompt") or row.get("negative_text") or "")
            if not cap.strip():
                continue
            rows += 1
            if report_dups:
                fp = caption_fingerprint(cap, algorithm="sha256", normalize_first=True)
                dup_buckets[fp] += 1
            if report_overlap and neg.strip():
                _i, _u, j = pos_neg_token_overlap(normalize_caption_for_training(cap), normalize_caption_for_training(neg))
                if _i > 0 and j > 0.0:
                    overlap_rows += 1

    dup_groups = 0
    max_dup = 1
    if report_dups and dup_buckets:
        vals = list(dup_buckets.values())
        dup_groups = sum(1 for v in vals if v > 1)
        max_dup = max(vals) if vals else 1

    return {
        "rows_with_caption": rows,
        "dup_groups": int(dup_groups),
        "max_dup_count": int(max_dup),
        "pos_neg_overlap_rows": int(overlap_rows) if report_overlap else 0,
    }


def _run_image_qc(
    manifest: Path,
    *,
    image_root: str,
    sample: int,
    min_sharpness: float,
    min_contrast: float,
) -> Dict[str, Any]:
    # Reuse the existing analyzer to avoid duplicating metric code.
    from PIL import Image
    from utils.image_quality_metrics import analyze_image_quality

    from scripts.tools.image_quality_qc import _extract_image_path, _resolve_path  # type: ignore

    results = 0
    fail = False
    sharp_min: Optional[float] = None
    con_min: Optional[float] = None

    import json as _json

    processed = 0
    with manifest.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            try:
                row = _json.loads(t)
            except Exception:
                continue
            processed += 1
            if sample and processed > sample:
                break
            img_path = _extract_image_path(row)
            if not img_path:
                continue
            path = _resolve_path(img_path, manifest, image_root=image_root.strip() or None)
            if not path.is_file():
                continue
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            m = analyze_image_quality(img)
            sharpness = float(m["sharpness"])
            contrast = float(m["contrast"])
            results += 1
            sharp_min = sharpness if sharp_min is None else min(sharp_min, sharpness)
            con_min = contrast if con_min is None else min(con_min, contrast)
            if min_sharpness > 0.0 and sharpness < min_sharpness:
                fail = True
            if min_contrast > 0.0 and contrast < min_contrast:
                fail = True

    return {
        "processed_rows": int(processed),
        "images_scored": int(results),
        "sharpness_min": float(sharp_min or 0.0),
        "contrast_min": float(con_min or 0.0),
        "fail": bool(fail),
    }


def _try_native_stats(manifest: Path) -> Dict[str, Any]:
    try:
        from utils.native import run_rust_jsonl_stats, run_rust_jsonl_validate
    except Exception:
        return {"available": False}
    out: Dict[str, Any] = {"available": True}
    try:
        v = run_rust_jsonl_validate(manifest, timeout=600)
        out["rust_validate_rc"] = int(v.returncode)
    except Exception as e:
        out["rust_validate_rc"] = None
        out["rust_validate_error"] = str(e)
    try:
        s = run_rust_jsonl_stats(manifest, timeout=600)
        out["rust_stats_rc"] = int(s.returncode)
    except Exception as e:
        out["rust_stats_rc"] = None
        out["rust_stats_error"] = str(e)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Manifest gate: lint + hygiene + optional image QC + native stats.")
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--json-report", type=Path, default=None)

    # Prompt lint thresholds
    ap.add_argument("--min-caption-len-chars", type=int, default=0)
    ap.add_argument("--max-caption-tokens", type=int, default=0)
    ap.add_argument("--fail-on-overlap", action="store_true")

    # Caption hygiene signals
    ap.add_argument("--report-dups", action="store_true")
    ap.add_argument("--report-overlap", action="store_true")
    ap.add_argument("--fail-on-dup-groups", type=int, default=0, help="Fail if dup_groups > N (0=off)")
    ap.add_argument("--fail-on-caption-overlap-rows", type=int, default=0, help="Fail if pos/neg overlap rows > N (0=off)")

    # Image QC (optional)
    ap.add_argument("--image-qc", action="store_true", help="Run image sharpness/contrast QC (requires PIL).")
    ap.add_argument("--image-root", type=str, default="")
    ap.add_argument("--sample", type=int, default=0, help="Image QC: only first N rows (0=all)")
    ap.add_argument("--min-sharpness", type=float, default=0.0)
    ap.add_argument("--min-contrast", type=float, default=0.0)

    # Native stats/validate
    ap.add_argument("--native-stats", action="store_true", help="Try Rust validate/stats if available.")

    args = ap.parse_args()
    manifest: Path = args.manifest
    if not manifest.is_file():
        print(f"Not a file: {manifest}", file=sys.stderr)
        return 2

    report: Dict[str, Any] = {"manifest": str(manifest)}
    report["promptlint"] = _run_prompt_lint(
        manifest,
        min_caption_len_chars=int(args.min_caption_len_chars),
        max_caption_tokens=int(args.max_caption_tokens),
        fail_on_overlap=bool(args.fail_on_overlap),
    )
    report["caption_hygiene"] = _run_caption_hygiene(
        manifest,
        report_dups=bool(args.report_dups),
        report_overlap=bool(args.report_overlap),
    )

    if bool(args.image_qc):
        if not _try_import_pil():
            report["image_qc"] = {"error": "PIL not installed"}
        else:
            report["image_qc"] = _run_image_qc(
                manifest,
                image_root=str(args.image_root or ""),
                sample=int(args.sample or 0),
                min_sharpness=float(args.min_sharpness or 0.0),
                min_contrast=float(args.min_contrast or 0.0),
            )

    if bool(args.native_stats):
        report["native"] = _try_native_stats(manifest)

    # Evaluate fail-fast rules
    fail = False
    if bool(args.fail_on_overlap) and int(report["promptlint"].get("pos_neg_overlap_rows", 0)) > 0:
        fail = True
    if int(args.fail_on_dup_groups or 0) > 0 and int(report["caption_hygiene"].get("dup_groups", 0)) > int(
        args.fail_on_dup_groups
    ):
        fail = True
    if int(args.fail_on_caption_overlap_rows or 0) > 0 and int(report["caption_hygiene"].get("pos_neg_overlap_rows", 0)) > int(
        args.fail_on_caption_overlap_rows
    ):
        fail = True
    if isinstance(report.get("image_qc"), dict) and bool(report["image_qc"].get("fail", False)):
        fail = True

    report["fail"] = bool(fail)

    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Small console summary
    pl = report["promptlint"]
    ch = report["caption_hygiene"]
    print(
        "manifest_gate:",
        f"fail={report['fail']}",
        f"rows_ok={pl.get('rows_ok')}",
        f"empty_caption_rows={pl.get('empty_caption_rows')}",
        f"pos_neg_overlap_rows={pl.get('pos_neg_overlap_rows')}",
        f"dup_groups={ch.get('dup_groups')}",
    )
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())


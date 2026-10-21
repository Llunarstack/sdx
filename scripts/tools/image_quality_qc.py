#!/usr/bin/env python3
"""
Image quality QC: compute sharpness/contrast for JSONL manifests.

This is a lightweight dataset QA tool to help avoid training on blurry/low-contrast images.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.manifest_utils import first_str, iter_manifest_rows, resolve_image_path  # noqa: E402


def main() -> int:
    import numpy as np
    from PIL import Image
    from utils.quality.quality import rgb_to_luminance
    from utils.quality.test_time_pick import score_edge_sharpness

    ap = argparse.ArgumentParser(description="Compute image sharpness/contrast for SDX JSONL manifests")
    ap.add_argument("input", type=str, help="Path to JSONL manifest")
    ap.add_argument("--image-root", type=str, default="", help="Optional base dir to join with image paths")
    ap.add_argument("--sample", type=int, default=0, help="Process only first N rows (0=all)")
    ap.add_argument("--min-sharpness", type=float, default=0.0, help="Fail if sharpness < value (0=off)")
    ap.add_argument("--min-contrast", type=float, default=0.0, help="Fail if contrast < value (0=off)")
    ap.add_argument("--json-report", type=str, default="", help="Optional path to write JSON report")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.is_file():
        print(f"Not found: {inp}", file=sys.stderr)
        return 2

    image_root = Path(args.image_root) if args.image_root.strip() else None
    results: list[dict[str, object]] = []
    fail = False
    processed = 0
    for row in iter_manifest_rows(inp):
        processed += 1
        if args.sample and processed > args.sample:
            break
        img_path = first_str(row, ("image_path", "path", "image"))
        path = resolve_image_path(img_path, image_root, inp.parent)
        if path is None:
            continue

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue

        rgb = np.asarray(img, dtype=np.uint8)
        sharpness = float(score_edge_sharpness(rgb))
        contrast = float(rgb_to_luminance(rgb).std())

        ok = True
        reasons: list[str] = []
        if args.min_sharpness > 0.0 and sharpness < args.min_sharpness:
            ok = False
            reasons.append(f"sharpness<{args.min_sharpness}")
        if args.min_contrast > 0.0 and contrast < args.min_contrast:
            ok = False
            reasons.append(f"contrast<{args.min_contrast}")

        if not ok:
            fail = True

        results.append(
            {
                "image_path": img_path,
                "resolved_path": str(path),
                "sharpness": sharpness,
                "contrast": contrast,
                "ok": ok,
                "reasons": reasons,
            }
        )

    report = {
        "input": str(inp),
        "processed_rows": processed,
        "count": len(results),
        "min_sharpness": args.min_sharpness,
        "min_contrast": args.min_contrast,
        "fail": fail,
    }

    if results:
        sharp_vals = [r["sharpness"] for r in results]
        con_vals = [r["contrast"] for r in results]
        report["sharpness_min"] = float(min(sharp_vals))
        report["sharpness_max"] = float(max(sharp_vals))
        report["contrast_min"] = float(min(con_vals))
        report["contrast_max"] = float(max(con_vals))

    if args.json_report:
        outp = Path(args.json_report)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(
            json.dumps({"summary": report, "rows": results}, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(f"imageqc: count={len(results)} fail={fail}")
    if fail:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Creative Co-Pilot — Decompose and Apply (inspiration → style → control → InstantStyle).

Example::

    python -m scripts.tools creative_copilot \\
        --prompt "a cybernetic cityscape under neon rain" \\
        --reference-images mood1.png,mood2.png \\
        --ckpt results/best.pt \\
        --web-search \\
        --out copilot_out.png

Dry-run (plan only, no downloads / sample)::

    python -m scripts.tools creative_copilot --prompt "oil portrait" --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from utils.generation.creative_copilot import CopilotConfig, format_plan_summary, run_creative_copilot


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", required=True, help="Creative goal / subject prompt.")
    ap.add_argument("--ckpt", default="", help="DiT checkpoint for fusion (sample.py).")
    ap.add_argument("--out", default="copilot_out.png")
    ap.add_argument("--work-dir", default="creative_copilot_run")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--reference-images",
        default="",
        help="Comma-separated local style references (moodboard).",
    )
    ap.add_argument(
        "--structure-image",
        default="",
        help="Optional layout/pose/depth source for ControlNet map (defaults to first moodboard).",
    )
    ap.add_argument("--web-search", action="store_true", help="Search the web for style references.")
    ap.add_argument("--no-web-search", action="store_true")
    ap.add_argument("--no-vlm", action="store_true", help="Skip VLM style captions (heuristic only).")
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument(
        "--control-prefer",
        default="canny",
        choices=["canny", "depth", "softedge", "hed"],
        help="Preferred control map (depth needs local Depth-Anything / Marigold).",
    )
    ap.add_argument(
        "--reference-style-mode",
        default="instantstyle",
        choices=["instantstyle", "style", "late", "full", "style_mid", "ip-adapter"],
    )
    ap.add_argument("--reference-strength", type=float, default=0.85)
    ap.add_argument("--content-prompt", default="a photo", help="CLIP content to subtract (InstantStyle).")
    ap.add_argument("--dry-run", action="store_true", help="Plan only; no search download / VLM / sample.")
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Run sample.py after planning (requires --ckpt).",
    )
    ap.add_argument(
        "--extra-sample-arg",
        action="append",
        default=[],
        help="Extra argv passed to sample.py (repeatable), e.g. --extra-sample-arg --steps=28",
    )
    args = ap.parse_args()

    refs = [x.strip() for x in str(args.reference_images or "").split(",") if x.strip()]
    web = bool(args.web_search) and not bool(args.no_web_search)
    cfg = CopilotConfig(
        web_search=web,
        use_vlm=not bool(args.no_vlm),
        extract_control=not bool(args.no_control),
        control_prefer=str(args.control_prefer),
        reference_style_mode=str(args.reference_style_mode),
        reference_strength=float(args.reference_strength),
        content_prompt=str(args.content_prompt),
        device=str(args.device),
    )

    plan = run_creative_copilot(
        goal=str(args.prompt),
        ckpt=str(args.ckpt or ""),
        work_dir=args.work_dir,
        out=args.out,
        reference_images=refs,
        structure_image=str(args.structure_image or ""),
        config=cfg,
        dry_run=bool(args.dry_run),
        execute=bool(args.execute),
        extra_sample_args=list(args.extra_sample_arg or []),
    )
    print(format_plan_summary(plan))
    if plan.work_dir and not args.dry_run:
        print(f"plan written: {Path(plan.work_dir) / 'copilot_plan.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

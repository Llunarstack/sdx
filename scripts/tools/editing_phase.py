#!/usr/bin/env python3
"""
Run the post-generation editing phase (diagnose → pieces → edit → gate loop).

Example:
  python scripts/tools/editing_phase.py --image out.png --prompt "a samurai at sunset" --dry-run
  python scripts/tools/editing_phase.py --image out.png --prompt "..." --ckpt results/best.pt --out-dir outputs/edit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description="SDX editing phase — iterate until natural / cohesive.")
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--prompt", required=True, help="Positive prompt used for generation")
    p.add_argument("--negative-prompt", default="", help="Negative prompt")
    p.add_argument("--ckpt", default="", help="Checkpoint for img2img/inpaint (optional with --dry-run)")
    p.add_argument("--out-dir", default="outputs/editing_phase", help="Work / output directory")
    p.add_argument("--max-iters", type=int, default=3)
    p.add_argument("--min-clip", type=float, default=0.28)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="Plan + masks + art_post only; no sample.py")
    p.add_argument("--no-ocr", action="store_true")
    p.add_argument("--no-art-post", action="store_true")
    p.add_argument("--caption", default="", help="Optional image caption for missing-token detection")
    p.add_argument("--scheduler", default="ays_dit")
    p.add_argument("--solver", default="dpmpp_2m")
    p.add_argument("--refine-steps", type=int, default=20)
    args = p.parse_args()

    from utils.generation.editing_phase import EditingPhaseConfig, run_editing_phase

    cfg = EditingPhaseConfig(
        max_iters=int(args.max_iters),
        min_clip=float(args.min_clip),
        enable_ocr=not bool(args.no_ocr),
        enable_art_post=not bool(args.no_art_post),
        dry_run=bool(args.dry_run) or not str(args.ckpt).strip(),
        device=str(args.device),
        scheduler=str(args.scheduler),
        solver=str(args.solver),
        refine_steps=int(args.refine_steps),
    )
    result = run_editing_phase(
        args.image,
        args.prompt,
        ckpt=str(args.ckpt).strip() or None,
        negative_prompt=str(args.negative_prompt),
        config=cfg,
        work_dir=args.out_dir,
        seed=args.seed,
        caption=str(args.caption or ""),
    )

    report = {
        "stopped_reason": result.stopped_reason,
        "iterations": result.iterations,
        "output_path": result.output_path,
        "piece_dir": result.piece_dir,
        "prompt": result.prompt,
        "actions": [
            {"kind": a.kind, "reason": a.reason, "region": a.region, "priority": a.priority}
            for a in result.actions_applied
        ],
        "diagnosis": [
            {
                "clip": d.clip_score,
                "sharpness": d.sharpness,
                "gate_passed": d.gate_passed,
                "failures": d.gate_failures,
                "missing_tokens": d.missing_tokens,
                "expected_text": d.expected_text,
                "pieces": d.piece_labels,
            }
            for d in result.diagnosis_history
        ],
    }
    report_path = Path(args.out_dir) / "editing_phase_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report -> {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

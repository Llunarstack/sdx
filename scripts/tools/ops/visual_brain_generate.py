#!/usr/bin/env python3
"""
**Visual Brain** — search, understand, dissect, generate, edit until complete.

Example::

    python -m scripts.tools visual_brain \\
        --ckpt results/best.pt \\
        --prompt "a neon diner sign reading OPEN at night, use vintage car references" \\
        --reference-images ref1.jpg,ref2.jpg \\
        --expected-text OPEN \\
        --web-search \\
        --out brain_out.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from utils.agentic import AgentContext, ImageGenerationAgent
from utils.brain import VisualBrain, VisualBrainConfig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", default="visual_brain_out.png")
    ap.add_argument("--work-dir", default="visual_brain_run")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--negative-prompt", default="")
    ap.add_argument(
        "--reference-images",
        default="",
        help="Comma-separated local reference image paths.",
    )
    ap.add_argument("--expected-text", default="", help="Text that must appear legibly (OCR verify).")
    ap.add_argument("--local-rag-jsonl", default="")
    ap.add_argument("--vit-ckpt", default="")
    ap.add_argument("--web-search", action="store_true", help="Search online for reference images.")
    ap.add_argument("--no-web-search", action="store_true")
    ap.add_argument("--no-creative-rag", action="store_true")
    ap.add_argument("--max-loops", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--use-agent", action="store_true", help="Run via ImageGenerationAgent.run_visual_brain.")
    args = ap.parse_args()

    refs = [x.strip() for x in str(args.reference_images or "").split(",") if x.strip()]
    web = bool(args.web_search) and not bool(args.no_web_search)

    if args.use_agent:
        ctx = AgentContext(
            ckpt=args.ckpt,
            prompt=args.prompt,
            work_dir=args.work_dir,
            out=args.out,
            device=args.device,
            negative_prompt=args.negative_prompt,
            reference_images=refs,
            local_rag_jsonl=args.local_rag_jsonl,
            vit_ckpt=args.vit_ckpt,
            expected_text=args.expected_text,
            web_search=web,
            dry_run=bool(args.dry_run),
        )
        agent = ImageGenerationAgent(repo_root=_REPO)
        res = agent.run_visual_brain(ctx)
        print(f"accepted={res.accepted} out={res.out_path}")
        return 0 if res.accepted or args.dry_run else 1

    cfg = VisualBrainConfig(
        web_search=web,
        creative_rag=not args.no_creative_rag,
        max_edit_loops=int(args.max_loops),
    )
    brain = VisualBrain(config=cfg, repo_root=_REPO)
    res = brain.run(
        ckpt=args.ckpt,
        prompt=args.prompt,
        work_dir=args.work_dir,
        out=args.out,
        device=args.device,
        negative_prompt=args.negative_prompt,
        reference_images=refs,
        local_rag_jsonl=args.local_rag_jsonl,
        expected_text=args.expected_text,
        vit_ckpt=args.vit_ckpt,
        dry_run=bool(args.dry_run),
    )
    print(f"accepted={res.accepted} out={res.out_path} coverage={res.metrics.get('coverage', 0):.3f}")
    print(f"references={len(res.reference_paths)} trace={res.trace_path}")
    print(f"brief saved with {len(res.brief.elements)} elements")
    return 0 if res.accepted or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())

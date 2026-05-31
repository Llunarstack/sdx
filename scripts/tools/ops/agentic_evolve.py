#!/usr/bin/env python3
"""
**Agentic evolve** — multi-trajectory GenEvolve-style comparison + experience memory.

Runs several tool configurations for the same prompt, distills best–worst differences
into ``experience_memory.jsonl`` for future runs.

Example::

    python -m scripts.tools agentic_evolve \\
        --ckpt results/best.pt \\
        --prompt "a red sports car in rain" \\
        --variants 3 --work-dir evolve_run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from config.defaults.agentic_stack import AgenticStackDefaults
from utils.agentic import AgentContext, ImageGenerationAgent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--work-dir", default="agentic_evolve")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--local-rag-jsonl", default="")
    ap.add_argument("--vit-ckpt", default="")
    ap.add_argument("--qwen-path", default="", help="HF path for LLM reflector (optional).")
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--flywheel", action="store_true", help="After evolve, run full flywheel align.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    defaults = AgenticStackDefaults(trajectory_variants=int(args.variants))
    ctx = AgentContext(
        ckpt=args.ckpt,
        prompt=args.prompt,
        work_dir=args.work_dir,
        device=args.device,
        local_rag_jsonl=args.local_rag_jsonl,
        vit_ckpt=args.vit_ckpt,
        qwen_path=str(args.qwen_path or ""),
        dry_run=bool(args.dry_run),
    )
    agent = ImageGenerationAgent(defaults, repo_root=_REPO)
    res = agent.run_evolve(ctx, variants=int(args.variants))
    print(f"best_out={res.out_path} trajectories={len(res.trace.trajectories)}")
    if res.experience:
        print(f"distilled: +prompt [{res.experience.prompt_delta}] tools {res.experience.tool_delta}")

    if args.flywheel and not args.dry_run:
        from utils.agentic.tools import AgentTool, ToolRegistry

        reg = ToolRegistry(ctx, repo_root=_REPO)
        fr = reg.execute(AgentTool.flywheel)
        print(f"flywheel ok={fr.ok}")
        return 0 if fr.ok else 2

    return 0 if res.out_path or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())

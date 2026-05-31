#!/usr/bin/env python3
"""
**Agentic flywheel** — autonomous curate → benchmark → agentic evolve → align loop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from config.defaults.agentic_stack import AgenticStackDefaults
from config.defaults.superior_stack import FlywheelPlan
from utils.agentic import AgentContext, ImageGenerationAgent
from utils.agentic.tools import AgentTool, ToolRegistry
from utils.superior.flywheel import run_flywheel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--prompt", default="", help="Optional seed prompt for evolve step.")
    ap.add_argument("--work-dir", default="agentic_flywheel")
    ap.add_argument("--local-rag-jsonl", default="")
    ap.add_argument("--vit-ckpt", default="")
    ap.add_argument("--skip-align", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    work = Path(args.work_dir)
    if args.prompt.strip():
        ctx = AgentContext(
            ckpt=args.base_ckpt,
            prompt=args.prompt,
            work_dir=str(work / "evolve"),
            local_rag_jsonl=args.local_rag_jsonl,
            vit_ckpt=args.vit_ckpt,
            dry_run=bool(args.dry_run),
        )
        ImageGenerationAgent(AgenticStackDefaults(), repo_root=_REPO).run_evolve(ctx)

    reg = ToolRegistry(
        AgentContext(ckpt=args.base_ckpt, prompt="", work_dir=str(work), dry_run=bool(args.dry_run)),
        repo_root=_REPO,
    )
    reg.execute(AgentTool.benchmark)

    if not args.skip_align:
        plan = FlywheelPlan(
            base_ckpt=args.base_ckpt,
            work_dir=str(work / "align"),
            local_rag_jsonl=args.local_rag_jsonl or None,
            vit_ckpt=args.vit_ckpt or None,
        )
        summary = run_flywheel(plan, repo_root=_REPO, dry_run=bool(args.dry_run))
        print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

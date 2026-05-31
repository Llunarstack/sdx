#!/usr/bin/env python3
"""
**Agentic generate** — plan → RAG → expand → generate → verify → reflect loop.

Example::

    python -m scripts.tools agentic_generate \\
        --ckpt results/best.pt \\
        --prompt "neon alley at night, cinematic" \\
        --local-rag-jsonl datasets/facts.jsonl \\
        --out agent_out.png
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
    ap.add_argument("--out", default="agent_out.png")
    ap.add_argument("--work-dir", default="agentic_run")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--negative-prompt", default="")
    ap.add_argument("--local-rag-jsonl", default="")
    ap.add_argument("--vit-ckpt", default="")
    ap.add_argument("--expected-text", default="")
    ap.add_argument("--max-reflect-loops", type=int, default=3)
    ap.add_argument("--num", type=int, default=4)
    ap.add_argument("--no-expand", action="store_true")
    ap.add_argument("--no-self-correct", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    defaults = AgenticStackDefaults(
        max_reflect_loops=int(args.max_reflect_loops),
        num_candidates=int(args.num),
        expand_prompt=not args.no_expand,
        self_correct=not args.no_self_correct,
    )
    ctx = AgentContext(
        ckpt=args.ckpt,
        prompt=args.prompt,
        work_dir=args.work_dir,
        out=args.out,
        device=args.device,
        negative_prompt=args.negative_prompt,
        local_rag_jsonl=args.local_rag_jsonl,
        vit_ckpt=args.vit_ckpt,
        expected_text=args.expected_text,
        dry_run=bool(args.dry_run),
    )
    agent = ImageGenerationAgent(defaults, repo_root=_REPO)
    res = agent.run(ctx)
    print(f"accepted={res.accepted} out={res.out_path}")
    if res.experience:
        print(f"experience prompt_delta={res.experience.prompt_delta!r}")
    return 0 if res.accepted or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())

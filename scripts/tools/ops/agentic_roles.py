#!/usr/bin/env python3
"""Run Designer / Verifier / Reasoner role pipeline (single pass, no reflect loop)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from config.defaults.agentic_stack import AgenticStackDefaults
from utils.agentic import AgentContext
from utils.agentic.roles import RolePipeline


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", default="roles_out.png")
    ap.add_argument("--work-dir", default="agentic_roles")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--local-rag-jsonl", default="")
    ap.add_argument("--vit-ckpt", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    d = AgenticStackDefaults()
    ctx = AgentContext(
        ckpt=args.ckpt,
        prompt=args.prompt,
        work_dir=args.work_dir,
        out=args.out,
        device=args.device,
        local_rag_jsonl=args.local_rag_jsonl,
        vit_ckpt=args.vit_ckpt,
        dry_run=bool(args.dry_run),
    )
    pipe = RolePipeline.from_context(
        ctx,
        repo_root=_REPO,
        num_candidates=d.num_candidates,
        pick_metric=d.pick_metric,
        expand_prompt=d.expand_prompt,
        self_correct=d.self_correct,
        rag_top_k=d.rag_top_k,
        extra_sample_args=d.extra_sample_args,
    )
    res = pipe.run()
    trace = {
        "prompt_final": res.prompt_final,
        "out_path": res.out_path,
        "metrics": res.metrics,
        "stages": [{"role": s.role, "ok": s.ok, "message": s.message} for s in res.stages],
    }
    work = Path(args.work_dir)
    if not args.dry_run:
        work.mkdir(parents=True, exist_ok=True)
        (work / "roles_trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print(json.dumps(trace, indent=2))
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

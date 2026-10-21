#!/usr/bin/env python3
"""Agentic image generation: generate, roles, evolve, flywheel.

    python -m scripts.tools agentic generate --ckpt best.pt --prompt "neon alley at night" --out agent.png
    python -m scripts.tools agentic roles    --ckpt best.pt --prompt "..."   # Designer / Verifier / Reasoner, one pass
    python -m scripts.tools agentic evolve   --ckpt best.pt --prompt "..." --variants 3
    python -m scripts.tools agentic flywheel --base-ckpt best.pt --prompt "..."

``generate`` runs plan -> RAG -> expand -> generate -> verify -> reflect. ``evolve`` compares several tool
configurations for one prompt and distils best-vs-worst into ``experience_memory.jsonl``. ``flywheel`` runs
evolve -> benchmark -> align. The old names (``agentic_generate``, ``agentic_roles``, ``agentic_evolve``,
``agentic_flywheel``) still work as aliases.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from config.defaults.agentic_stack import AgenticStackDefaults  # noqa: E402
from utils.agentic import AgentContext, ImageGenerationAgent  # noqa: E402


def _generate(args: argparse.Namespace) -> int:
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
    res = ImageGenerationAgent(defaults, repo_root=_REPO).run(ctx)
    print(f"accepted={res.accepted} out={res.out_path}")
    if res.experience:
        print(f"experience prompt_delta={res.experience.prompt_delta!r}")
    return 0 if res.accepted or args.dry_run else 1


def _roles(args: argparse.Namespace) -> int:
    from utils.agentic.roles import RolePipeline

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
    res = RolePipeline.from_context(
        ctx,
        repo_root=_REPO,
        num_candidates=d.num_candidates,
        pick_metric=d.pick_metric,
        expand_prompt=d.expand_prompt,
        self_correct=d.self_correct,
        rag_top_k=d.rag_top_k,
        extra_sample_args=d.extra_sample_args,
    ).run()
    trace = {
        "prompt_final": res.prompt_final,
        "out_path": res.out_path,
        "metrics": res.metrics,
        "stages": [{"role": s.role, "ok": s.ok, "message": s.message} for s in res.stages],
    }
    if not args.dry_run:
        work = Path(args.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        (work / "roles_trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print(json.dumps(trace, indent=2))
    return 0 if res.ok else 1


def _evolve(args: argparse.Namespace) -> int:
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
    agent = ImageGenerationAgent(AgenticStackDefaults(trajectory_variants=int(args.variants)), repo_root=_REPO)
    res = agent.run_evolve(ctx, variants=int(args.variants))
    print(f"best_out={res.out_path} trajectories={len(res.trace.trajectories)}")
    if res.experience:
        print(f"distilled: +prompt [{res.experience.prompt_delta}] tools {res.experience.tool_delta}")
    if args.flywheel and not args.dry_run:
        from utils.agentic.tools import AgentTool, ToolRegistry

        fr = ToolRegistry(ctx, repo_root=_REPO).execute(AgentTool.flywheel)
        print(f"flywheel ok={fr.ok}")
        return 0 if fr.ok else 2
    return 0 if res.out_path or args.dry_run else 1


def _flywheel(args: argparse.Namespace) -> int:
    from config.defaults.superior_stack import FlywheelPlan
    from utils.agentic.tools import AgentTool, ToolRegistry
    from utils.superior.auto_loop import run_flywheel

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
    ToolRegistry(
        AgentContext(ckpt=args.base_ckpt, prompt="", work_dir=str(work), dry_run=bool(args.dry_run)),
        repo_root=_REPO,
    ).execute(AgentTool.benchmark)
    if not args.skip_align:
        plan = FlywheelPlan(
            base_ckpt=args.base_ckpt,
            work_dir=str(work / "align"),
            local_rag_jsonl=args.local_rag_jsonl or None,
            vit_ckpt=args.vit_ckpt or None,
        )
        print(run_flywheel(plan, repo_root=_REPO, dry_run=bool(args.dry_run)))
    return 0


def _common(p: argparse.ArgumentParser, *, work_dir: str, out: str | None = None) -> None:
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", required=True)
    if out is not None:
        p.add_argument("--out", default=out)
    p.add_argument("--work-dir", default=work_dir)
    p.add_argument("--device", default="cuda")
    p.add_argument("--local-rag-jsonl", default="")
    p.add_argument("--vit-ckpt", default="")
    p.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("generate", help="plan -> RAG -> expand -> generate -> verify -> reflect")
    _common(p, work_dir="agentic_run", out="agent_out.png")
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--expected-text", default="")
    p.add_argument("--max-reflect-loops", type=int, default=3)
    p.add_argument("--num", type=int, default=4)
    p.add_argument("--no-expand", action="store_true")
    p.add_argument("--no-self-correct", action="store_true")
    p.set_defaults(fn=_generate)

    p = sub.add_parser("roles", help="Designer / Verifier / Reasoner, single pass")
    _common(p, work_dir="agentic_roles", out="roles_out.png")
    p.set_defaults(fn=_roles)

    p = sub.add_parser("evolve", help="Multi-trajectory comparison + experience memory")
    _common(p, work_dir="agentic_evolve")
    p.add_argument("--qwen-path", default="", help="HF path for LLM reflector (optional).")
    p.add_argument("--variants", type=int, default=3)
    p.add_argument("--flywheel", action="store_true", help="After evolve, run full flywheel align.")
    p.set_defaults(fn=_evolve)

    p = sub.add_parser("flywheel", help="evolve -> benchmark -> align")
    p.add_argument("--base-ckpt", required=True)
    p.add_argument("--prompt", default="", help="Optional seed prompt for the evolve step.")
    p.add_argument("--work-dir", default="agentic_flywheel")
    p.add_argument("--local-rag-jsonl", default="")
    p.add_argument("--vit-ckpt", default="")
    p.add_argument("--skip-align", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=_flywheel)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())

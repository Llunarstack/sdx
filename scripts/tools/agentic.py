#!/usr/bin/env python3
"""Agentic image generation and planning: generate, roles, evolve, flywheel, stills, perfect, visual-brain.

    python -m scripts.tools agentic generate --ckpt best.pt --prompt "neon alley at night" --out agent.png
    python -m scripts.tools agentic roles    --ckpt best.pt --prompt "..."   # Designer / Verifier / Reasoner, one pass
    python -m scripts.tools agentic evolve   --ckpt best.pt --prompt "..." --variants 3
    python -m scripts.tools agentic flywheel --base-ckpt best.pt --prompt "..."
    python -m scripts.tools agentic stills   --prompt "1girl, catgirl" --interview --print-interview
    python -m scripts.tools agentic perfect  --prompt "..." --no-web --print-argv
    python -m scripts.tools agentic visual-brain --ckpt best.pt --prompt "neon sign reading OPEN" --expected-text OPEN

``generate`` runs plan -> RAG -> expand -> generate -> verify -> reflect. ``evolve`` compares several tool
configurations for one prompt and distils best-vs-worst into ``experience_memory.jsonl``. ``flywheel`` runs
evolve -> benchmark -> align. ``stills`` plans interview / draft thumbs / taste / critique refine; ``perfect``
plans facet -> memory -> refs -> sample argv (no GPU); ``visual-brain`` searches, dissects, generates and edits
until complete. Old names (``agentic_generate``, ``agentic_roles``, ``agentic_evolve``, ``agentic_flywheel``,
``agentic_stills``, ``perfect_gen``, ``visual_brain``, ``visual_brain_generate``) still work as aliases.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _generate(args: argparse.Namespace) -> int:
    from config.presets import AgenticStackDefaults
    from utils.agentic.agent import AgentContext, ImageGenerationAgent

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
    from config.presets import AgenticStackDefaults
    from utils.agentic.agent import AgentContext, RolePipeline

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
    from config.presets import AgenticStackDefaults
    from utils.agentic.agent import AgentContext, ImageGenerationAgent

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
        from utils.agentic.agent import AgentTool, ToolRegistry

        fr = ToolRegistry(ctx, repo_root=_REPO).execute(AgentTool.flywheel)
        print(f"flywheel ok={fr.ok}")
        return 0 if fr.ok else 2
    return 0 if res.out_path or args.dry_run else 1


def _flywheel(args: argparse.Namespace) -> int:
    from config.presets import AgenticStackDefaults, FlywheelPlan
    from utils.agentic.agent import AgentContext, AgentTool, ImageGenerationAgent, ToolRegistry
    from utils.agentic.quality_loop import run_flywheel

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


def _load_json_arg(raw: str) -> dict:
    """JSON object from a file path or an inline JSON string ("" -> {})."""
    if not raw:
        return {}
    p = Path(raw)
    data = json.loads(p.read_text(encoding="utf-8") if p.is_file() else raw)
    if not isinstance(data, dict):
        raise SystemExit("JSON arg must be an object")
    return data


def _print_argv(label: str, argv: list[str]) -> None:
    print(f"{label} argv:")
    print(" ", " ".join(shlex.quote(a) for a in argv))


def _stills(args: argparse.Namespace) -> int:
    from utils.generation import planners

    if args.taste_quiz:
        print(json.dumps(planners.build_taste_quiz(), indent=2))
        return 0
    if args.taste_answers_json:
        taste = planners.apply_taste_quiz_answers(
            planners.load_user_taste(args.taste_path or None), _load_json_arg(args.taste_answers_json)
        )
        print(f"saved taste -> {planners.save_user_taste(taste, args.taste_path or None)}")
        if not args.prompt:
            return 0
    if args.progressive_next and args.prompt:
        from utils.agentic.interview import next_interview_item, start_progressive_interview

        nxt = next_interview_item(start_progressive_interview(args.prompt))
        print("interview complete" if nxt is None else json.dumps(nxt.to_dict(), indent=2))
        return 0
    if not args.prompt and not args.critique_image:
        args.parser.error("--prompt is required (unless --taste-quiz / --taste-answers-json only)")

    plan = planners.plan_agentic_stills(
        args.prompt or "refine",
        work_dir=args.work_dir,
        session_id=args.session_id,
        interview=bool(args.interview) and not args.no_interview,
        drafts=not args.no_drafts,
        num_drafts=int(args.num_drafts),
        assume_missing=not bool(args.ask),
        taste_path=args.taste_path,
        print_intent=args.print_intent,
        allow_web=not args.no_web,
        critique_image=args.critique_image,
        critique_answers=_load_json_arg(args.critique_answers_json) or None,
        interview_answers=_load_json_arg(args.interview_answers_json) or None,
        face_refs=list(args.face_ref or []),
        chosen_draft_index=args.chosen_draft,
    )
    out_path = Path(args.json_out or (Path(plan.work_dir) / "plan.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")

    print(f"session: {plan.session_id}")
    print(f"work:    {plan.work_dir}")
    print(f"plan:    {out_path}")
    if args.print_interview and plan.interview_text:
        print(plan.interview_text)
    if plan.notes:
        print("notes:")
        for n in plan.notes:
            print(f"  - {n}")
    if plan.needs_user:
        print("questions:")
        for q in plan.questions:
            print(f"  ? {q}")
        return 2
    if args.print_argv:
        if plan.sample_draft_argv:
            _print_argv("draft", plan.sample_draft_argv)
        if plan.sample_final_argv:
            _print_argv("final", plan.sample_final_argv)
        refine = (plan.critique or {}).get("refine_argv")
        if refine:
            _print_argv("refine", refine)
    return 0


def _perfect(args: argparse.Namespace) -> int:
    from utils.generation.planners import PerfectGenConfig, plan_perfect_gen

    answers = {k: v for k, v in (("pose", args.pose), ("place", args.place)) if v}
    interview_answers = _load_json_arg(args.interview_answers_json)
    interview_photos = _load_json_arg(args.interview_photos_json)
    # --interview with any answers/photos/refs supplied may proceed without assuming missing facets.
    supplied = bool(interview_answers or interview_photos or args.face_ref or args.style_ref)
    cfg = PerfectGenConfig(
        work_dir=args.work_dir,
        session_id=args.session_id,
        allow_web=not args.no_web,
        max_refs_per_facet=int(args.max_refs),
        assume_missing=not bool(args.ask or args.interview) or (args.interview and supplied),
        interview=bool(args.interview),
        clarify_answers=answers,
        interview_answers=interview_answers,
        interview_photos=interview_photos,
        run_rag=not args.no_rag,
        character_sheet=str(args.character_sheet or ""),
        user_face_refs=list(args.face_ref or []),
        user_style_refs=list(args.style_ref or []),
    )
    plan = plan_perfect_gen(args.prompt, cfg)
    out_path = Path(args.json_out or (Path(plan.memory_path).parent / "plan.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")

    print(f"enriched: {plan.enriched_prompt}")
    print(f"memory:   {plan.memory_path}")
    print(f"plan:     {out_path}")
    if args.print_interview or (plan.needs_user and plan.interview_text):
        print(plan.interview_text or "")
    if plan.notes:
        print("notes:")
        for n in plan.notes:
            print(f"  - {n}")
    if plan.needs_user:
        print("questions:")
        for q in plan.questions:
            print(f"  ? {q}")
        if plan.photo_requests:
            print("photo requests:")
            for q in plan.photo_requests:
                print(f"  📷 {q}")
        return 2
    if args.print_argv:
        _print_argv("sample", plan.sample_argv)
    return 0


def _visual_brain(args: argparse.Namespace) -> int:
    refs = [x.strip() for x in str(args.reference_images or "").split(",") if x.strip()]
    web = bool(args.web_search) and not bool(args.no_web_search)

    if args.use_agent:
        from utils.agentic.agent import AgentContext, ImageGenerationAgent

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
        res = ImageGenerationAgent(repo_root=_REPO).run_visual_brain(ctx)
        print(f"accepted={res.accepted} out={res.out_path}")
        return 0 if res.accepted or args.dry_run else 1

    from utils.agentic.agent import VisualBrain, VisualBrainConfig

    cfg = VisualBrainConfig(web_search=web, creative_rag=not args.no_creative_rag, max_edit_loops=int(args.max_loops))
    res = VisualBrain(config=cfg, repo_root=_REPO).run(
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
    ap = argparse.ArgumentParser(
        prog="python -m scripts.tools agentic",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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

    p = sub.add_parser("stills", help="Agentic stills planner: interview, draft thumbs, taste, critique refine")
    p.add_argument("--prompt", default="")
    p.add_argument("--work-dir", default="")
    p.add_argument("--session-id", default="")
    p.add_argument("--interview", action="store_true")
    p.add_argument("--no-interview", action="store_true")
    p.add_argument("--no-drafts", action="store_true")
    p.add_argument("--num-drafts", type=int, default=4)
    p.add_argument("--ask", action="store_true", help="Block on interview / missing facets")
    p.add_argument("--no-web", action="store_true")
    p.add_argument("--print-intent", default="", help="phone|poster|square|desktop|portrait|album")
    p.add_argument("--taste-path", default="")
    p.add_argument("--taste-quiz", action="store_true", help="Print taste quiz JSON and exit")
    p.add_argument("--taste-answers-json", default="", help="Apply taste quiz answers and save")
    p.add_argument("--interview-answers-json", default="")
    p.add_argument("--face-ref", action="append", default=[])
    p.add_argument("--chosen-draft", type=int, default=None)
    p.add_argument("--critique-image", default="")
    p.add_argument("--critique-answers-json", default="")
    p.add_argument("--progressive-next", action="store_true", help="Show next interview question only")
    p.add_argument("--print-argv", action="store_true")
    p.add_argument("--print-interview", action="store_true")
    p.add_argument("--json-out", default="")
    p.set_defaults(fn=_stills, parser=p)

    p = sub.add_parser("perfect", help="Perfect-gen agentic plan (no GPU sample by default)")
    p.add_argument("--prompt", required=True)
    p.add_argument("--work-dir", default="")
    p.add_argument("--session-id", default="")
    p.add_argument("--no-web", action="store_true")
    p.add_argument("--max-refs", type=int, default=4)
    p.add_argument("--ask", action="store_true", help="Do not assume pose/place; emit questions and exit 2 if missing.")
    p.add_argument(
        "--interview",
        action="store_true",
        help="Full pre-gen interview: pose/place/style + ask for face/outfit/moodboard photos.",
    )
    p.add_argument(
        "--interview-answers-json",
        default="",
        help='JSON object of interview id→answer, e.g. {"pose":"sitting","place":"bedroom"}',
    )
    p.add_argument(
        "--interview-photos-json",
        default="",
        help='JSON object of interview id→[paths], e.g. {"face_photo":["face.png"]}',
    )
    p.add_argument("--face-ref", action="append", default=[], help="Identity photo path (repeatable)")
    p.add_argument("--style-ref", action="append", default=[], help="Style moodboard path (repeatable)")
    p.add_argument("--pose", default="", help="Clarify answer: pose")
    p.add_argument("--place", default="", help="Clarify answer: place")
    p.add_argument("--character-sheet", default="")
    p.add_argument("--no-rag", action="store_true")
    p.add_argument("--print-argv", action="store_true")
    p.add_argument("--print-interview", action="store_true", help="Print interview checklist")
    p.add_argument("--json-out", default="", help="Write plan JSON path (default: work_dir/plan.json)")
    p.set_defaults(fn=_perfect)

    p = sub.add_parser("visual-brain", help="Visual Brain: search, understand, dissect, generate, edit until complete")
    _common(p, work_dir="visual_brain_run", out="visual_brain_out.png")
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--reference-images", default="", help="Comma-separated local reference image paths.")
    p.add_argument("--expected-text", default="", help="Text that must appear legibly (OCR verify).")
    p.add_argument("--web-search", action="store_true", help="Search online for reference images.")
    p.add_argument("--no-web-search", action="store_true")
    p.add_argument("--no-creative-rag", action="store_true")
    p.add_argument("--max-loops", type=int, default=3)
    p.add_argument("--use-agent", action="store_true", help="Run via ImageGenerationAgent.run_visual_brain.")
    p.set_defaults(fn=_visual_brain)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())

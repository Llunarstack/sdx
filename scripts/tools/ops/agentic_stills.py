#!/usr/bin/env python3
"""Agentic stills CLI — interview, draft thumbs, taste, critique refine plans.

Examples:
  python -m scripts.tools agentic_stills --prompt "1girl, catgirl, eipril style" --print-argv
  python -m scripts.tools agentic_stills --prompt "..." --interview --print-interview
  python -m scripts.tools agentic_stills --taste-quiz --taste-answers-json answers.json
  python -m scripts.tools agentic_stills --critique-image out.png --critique-answers-json c.json
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


def _load_json_arg(raw: str) -> dict:
    if not raw:
        return {}
    p = Path(raw)
    text = p.read_text(encoding="utf-8") if p.is_file() else raw
    data = json.loads(text)
    if not isinstance(data, dict):
        raise SystemExit("JSON arg must be an object")
    return data


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Agentic stills planner")
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
    args = p.parse_args(argv)

    if args.taste_quiz:
        from utils.generation.user_taste import build_taste_quiz

        print(json.dumps(build_taste_quiz(), indent=2))
        return 0

    if args.taste_answers_json:
        from utils.generation.user_taste import (
            apply_taste_quiz_answers,
            load_user_taste,
            save_user_taste,
        )

        taste = load_user_taste(args.taste_path or None)
        taste = apply_taste_quiz_answers(taste, _load_json_arg(args.taste_answers_json))
        path = save_user_taste(taste, args.taste_path or None)
        print(f"saved taste -> {path}")
        if not args.prompt:
            return 0

    if args.progressive_next and args.prompt:
        from utils.agentic.progressive_interview import next_interview_item, start_progressive_interview

        state = start_progressive_interview(args.prompt)
        nxt = next_interview_item(state)
        if nxt is None:
            print("interview complete")
            return 0
        print(json.dumps(nxt.to_dict(), indent=2))
        return 0

    if not args.prompt and not args.critique_image:
        p.error("--prompt is required (unless --taste-quiz / --taste-answers-json only)")

    from utils.generation.agentic_stills import plan_agentic_stills

    plan = plan_agentic_stills(
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
            print("draft argv:")
            print(" ", " ".join(shlex.quote(a) for a in plan.sample_draft_argv))
        if plan.sample_final_argv:
            print("final argv:")
            print(" ", " ".join(shlex.quote(a) for a in plan.sample_final_argv))
        crit = plan.critique or {}
        if crit.get("refine_argv"):
            print("refine argv:")
            print(" ", " ".join(shlex.quote(a) for a in crit["refine_argv"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

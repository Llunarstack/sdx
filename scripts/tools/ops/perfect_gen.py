#!/usr/bin/env python3
"""CLI for perfect-gen planning (facet → memory → refs → sample argv).

Examples:
  python -m scripts.tools perfect_gen --prompt "1girl, catgirl, black lingerie, eipril style"
  python -m scripts.tools perfect_gen --prompt "..." --no-web --print-argv
  python -m scripts.tools perfect_gen --prompt "..." --ask
  python -m scripts.tools perfect_gen --prompt "..." --interview   # questions + photo asks
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Perfect-gen agentic plan (no GPU sample by default)")
    p.add_argument("--prompt", required=True)
    p.add_argument("--work-dir", default="")
    p.add_argument("--session-id", default="")
    p.add_argument("--no-web", action="store_true")
    p.add_argument("--max-refs", type=int, default=4)
    p.add_argument(
        "--ask",
        action="store_true",
        help="Do not assume pose/place; emit questions and exit 2 if missing.",
    )
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
    p.add_argument("--reference-style-mode", default="instantstyle")
    p.add_argument("--no-rag", action="store_true")
    p.add_argument("--print-argv", action="store_true")
    p.add_argument("--print-interview", action="store_true", help="Print interview checklist")
    p.add_argument("--json-out", default="", help="Write plan JSON path (default: work_dir/plan.json)")
    args = p.parse_args(argv)

    from utils.generation.perfect_gen import PerfectGenConfig, plan_perfect_gen

    answers: dict = {}
    if args.pose:
        answers["pose"] = args.pose
    if args.place:
        answers["place"] = args.place
    interview_answers: dict = {}
    interview_photos: dict = {}
    if args.interview_answers_json:
        interview_answers = json.loads(
            Path(args.interview_answers_json).read_text(encoding="utf-8")
            if Path(args.interview_answers_json).is_file()
            else args.interview_answers_json
        )
    if args.interview_photos_json:
        interview_photos = json.loads(
            Path(args.interview_photos_json).read_text(encoding="utf-8")
            if Path(args.interview_photos_json).is_file()
            else args.interview_photos_json
        )

    cfg = PerfectGenConfig(
        work_dir=args.work_dir,
        session_id=args.session_id,
        allow_web=not args.no_web,
        max_refs_per_facet=int(args.max_refs),
        assume_missing=not bool(args.ask or args.interview),
        interview=bool(args.interview),
        clarify_answers=answers,
        interview_answers=interview_answers,
        interview_photos=interview_photos,
        reference_style_mode=str(args.reference_style_mode),
        run_rag=not args.no_rag,
        character_sheet=str(args.character_sheet or ""),
        user_face_refs=list(args.face_ref or []),
        user_style_refs=list(args.style_ref or []),
    )
    # With --interview + answers/photos provided, allow proceed without assume.
    if args.interview and (interview_answers or interview_photos or args.face_ref or args.style_ref):
        cfg.assume_missing = True

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
        print("sample argv:")
        print(" ", " ".join(shlex.quote(a) for a in plan.sample_argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

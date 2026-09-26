"""Unified agentic stills orchestrator — interview → drafts → critique → refine.

Plans only (no GPU) unless caller runs the returned argv lists via sample.py.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from utils.agentic.post_gen_critique import (
    build_post_gen_critique,
    critique_answers_to_fix_plan,
)
from utils.agentic.progressive_interview import (
    ProgressiveState,
    answer_interview_item,
    next_interview_item,
    skip_interview_item,
)
from utils.generation.draft_thumbs import DraftThumbPlan, plan_draft_thumbnails, promote_draft_to_final
from utils.generation.perfect_gen import PerfectGenConfig, PerfectGenPlan, plan_perfect_gen
from utils.generation.print_intent import resolve_print_intent
from utils.generation.user_taste import (
    apply_taste_to_prompts,
    load_user_taste,
    save_user_taste,
)

__all__ = [
    "AgenticStillsPlan",
    "plan_agentic_stills",
    "progressive_step",
]


@dataclass
class AgenticStillsPlan:
    session_id: str
    work_dir: str
    perfect: dict[str, Any] = field(default_factory=dict)
    draft: dict[str, Any] = field(default_factory=dict)
    critique: dict[str, Any] = field(default_factory=dict)
    taste: dict[str, Any] = field(default_factory=dict)
    interview_text: str = ""
    needs_user: bool = False
    questions: list[str] = field(default_factory=list)
    sample_draft_argv: list[str] = field(default_factory=list)
    sample_final_argv: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def progressive_step(
    state: ProgressiveState,
    *,
    item_id: str | None = None,
    text: str = "",
    photos: list[str] | None = None,
    skip: bool = False,
) -> tuple[ProgressiveState, dict[str, Any]]:
    """Advance one interview turn; returns (state, payload for UI)."""
    current = next_interview_item(state)
    if current is None:
        return state, {"done": True, "item": None}
    target_id = item_id or current.id
    if skip or str(text).strip().lower() in ("skip", "you decide"):
        state = skip_interview_item(state, target_id)
    else:
        state = answer_interview_item(state, target_id, text=text, photos=photos)
    nxt = next_interview_item(state)
    return state, {
        "done": nxt is None,
        "answered_id": target_id,
        "next": nxt.to_dict() if nxt else None,
        "pending": [i.id for i in state.interview.pending()],
    }


def plan_agentic_stills(
    prompt: str,
    *,
    work_dir: str = "",
    session_id: str = "",
    interview: bool = True,
    drafts: bool = True,
    num_drafts: int = 4,
    assume_missing: bool = True,
    taste_path: str = "",
    print_intent: str = "",
    allow_web: bool = True,
    critique_image: str = "",
    critique_answers: dict[str, str] | None = None,
    interview_answers: dict[str, str] | None = None,
    face_refs: list[str] | None = None,
    chosen_draft_index: int | None = None,
) -> AgenticStillsPlan:
    """End-to-end stills plan: taste → perfect_gen → draft thumbs → optional critique."""
    sid = (session_id or uuid.uuid4().hex[:12]).strip()
    work = Path(work_dir or f"outputs/agentic_stills/{sid}")
    work.mkdir(parents=True, exist_ok=True)

    taste = load_user_taste(taste_path or str(work / "user_taste.json"))
    pos, neg = apply_taste_to_prompts(prompt, "", taste)
    notes = []
    if pos != prompt:
        notes.append("applied user taste to prompt")

    intent = resolve_print_intent(print_intent or taste.aspect_intent)
    extra: list[str] = []
    if intent:
        extra.extend(intent.sample_argv())
        notes.append(f"print intent: {intent.label} {intent.width}x{intent.height}")
    if neg:
        extra.extend(["--negative-prompt", neg])

    pg_cfg = PerfectGenConfig(
        work_dir=str(work / "perfect"),
        session_id=sid,
        allow_web=allow_web,
        assume_missing=assume_missing,
        interview=interview,
        interview_answers=dict(interview_answers or {}),
        user_face_refs=list(face_refs or []),
        run_rag=True,
        extra_sample_args=list(extra),
    )
    perfect: PerfectGenPlan = plan_perfect_gen(pos, pg_cfg)

    out = AgenticStillsPlan(
        session_id=sid,
        work_dir=str(work),
        perfect=perfect.to_dict(),
        taste=taste.to_dict(),
        interview_text=perfect.interview_text,
        needs_user=perfect.needs_user,
        questions=list(perfect.questions),
        notes=notes + list(perfect.notes),
    )

    if perfect.needs_user:
        (work / "plan.json").write_text(json.dumps(out.to_dict(), indent=2), encoding="utf-8")
        return out

    enriched = perfect.enriched_prompt
    draft_plan: DraftThumbPlan | None = None
    if drafts:
        draft_plan = plan_draft_thumbnails(
            enriched,
            num_drafts=num_drafts,
            extra=list(extra),
        )
        # Merge moodboard/ref flags from perfect plan into draft/final
        # perfect.sample_argv starts with --prompt X; append the rest after draft's prompt
        ref_flags: list[str] = []
        argv = perfect.sample_argv
        skip_next = False
        i = 0
        while i < len(argv):
            if skip_next:
                skip_next = False
                i += 1
                continue
            if argv[i] == "--prompt":
                skip_next = True
                i += 1
                continue
            ref_flags.append(argv[i])
            i += 1
        draft_plan.draft_argv = [
            "--prompt",
            enriched,
            "--num",
            str(draft_plan.num_drafts),
            "--steps",
            str(draft_plan.draft_steps),
            "--width",
            str(draft_plan.draft_width),
            "--height",
            str(draft_plan.draft_height),
            "--seed",
            str(draft_plan.base_seed),
            "--pick-best",
            draft_plan.pick_metric,
        ] + ref_flags
        draft_plan.final_argv_template = [
            "--prompt",
            enriched,
            "--num",
            "1",
            "--steps",
            str(draft_plan.final_steps),
            "--width",
            str(intent.width if intent else draft_plan.final_width),
            "--height",
            str(intent.height if intent else draft_plan.final_height),
        ] + ref_flags
        out.draft = draft_plan.to_dict()
        out.sample_draft_argv = list(draft_plan.draft_argv)
        idx = 0 if chosen_draft_index is None else int(chosen_draft_index)
        out.sample_final_argv = promote_draft_to_final(
            draft_plan, chosen_index=idx, user_picked=chosen_draft_index is not None
        )
    else:
        out.sample_final_argv = list(perfect.sample_argv)

    if critique_image:
        cplan = build_post_gen_critique(critique_image, enriched)
        if critique_answers:
            w = intent.width if intent else 1024
            h = intent.height if intent else 1024
            cplan = critique_answers_to_fix_plan(cplan, critique_answers, work_dir=work / "critique", width=w, height=h)
        out.critique = cplan.to_dict()
        out.notes.extend(cplan.notes)

    save_user_taste(taste, work / "user_taste.json")
    (work / "plan.json").write_text(json.dumps(out.to_dict(), indent=2), encoding="utf-8")
    return out

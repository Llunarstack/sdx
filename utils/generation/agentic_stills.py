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
from utils.generation.perfect_gen import PerfectGenConfig, PerfectGenPlan, plan_perfect_gen
from utils.generation.user_taste import (
    apply_taste_to_prompts,
    load_user_taste,
    save_user_taste,
)

__all__ = [
    "AgenticStillsPlan",
    "DraftThumbPlan",
    "PRINT_INTENTS",
    "plan_agentic_stills",
    "plan_draft_thumbnails",
    "promote_draft_to_final",
    "resolve_print_intent",
]

# Output-size presets for ``print_intent`` / the taste profile's ``aspect_intent``.
PRINT_INTENTS: dict[str, tuple[int, int, str]] = {
    "phone": (576, 1024, "phone wallpaper"),
    "portrait": (768, 1024, "portrait"),
    "square": (1024, 1024, "square social"),
    "desktop": (1280, 720, "desktop wallpaper"),
    "poster": (768, 1152, "poster print"),
    "album": (1024, 1024, "album cover"),
}


@dataclass(slots=True)
class PrintIntent:
    key: str
    width: int
    height: int
    label: str

    def sample_argv(self) -> list[str]:
        return ["--width", str(self.width), "--height", str(self.height)]


def resolve_print_intent(name: str) -> PrintIntent | None:
    key = str(name or "").strip().lower()
    if key not in PRINT_INTENTS:
        return None
    w, h, label = PRINT_INTENTS[key]
    return PrintIntent(key=key, width=w, height=h, label=label)


@dataclass
class DraftThumbPlan:
    """Plan for a low-step multi-seed draft pass, then a locked final."""

    prompt: str
    num_drafts: int = 4
    draft_steps: int = 12
    draft_width: int = 512
    draft_height: int = 512
    final_steps: int = 40
    final_width: int = 1024
    final_height: int = 1024
    pick_metric: str = "combo"
    base_seed: int = 0
    draft_argv: list[str] = field(default_factory=list)
    final_argv_template: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_draft_thumbnails(
    prompt: str,
    *,
    num_drafts: int = 4,
    draft_steps: int = 12,
    draft_size: int = 512,
    final_steps: int = 40,
    final_width: int = 1024,
    final_height: int = 1024,
    pick_metric: str = "combo",
    base_seed: int = 0,
    extra: list[str] | None = None,
) -> DraftThumbPlan:
    """
    Build argv for ``sample.py`` draft pass (``--num N --steps low --pick-best``)
    and a final argv template (caller fills ``--seed`` after user/auto pick).
    """
    n = max(2, int(num_drafts))
    extra = list(extra or [])
    draft = [
        "--prompt",
        prompt,
        "--num",
        str(n),
        "--steps",
        str(max(4, int(draft_steps))),
        "--width",
        str(int(draft_size)),
        "--height",
        str(int(draft_size)),
        "--seed",
        str(int(base_seed)),
        "--pick-best",
        str(pick_metric),
    ] + extra
    final_tmpl = [
        "--prompt",
        prompt,
        "--num",
        "1",
        "--steps",
        str(max(draft_steps, int(final_steps))),
        "--width",
        str(int(final_width)),
        "--height",
        str(int(final_height)),
        # --seed filled by promote_draft_to_final
    ] + extra
    return DraftThumbPlan(
        prompt=prompt,
        num_drafts=n,
        draft_steps=int(draft_steps),
        draft_width=int(draft_size),
        draft_height=int(draft_size),
        final_steps=int(final_steps),
        final_width=int(final_width),
        final_height=int(final_height),
        pick_metric=str(pick_metric),
        base_seed=int(base_seed),
        draft_argv=draft,
        final_argv_template=final_tmpl,
        notes=[
            f"Run draft: sample.py {' '.join(draft)}",
            "User picks index 0..N-1 (or trust --pick-best winner), then promote with seed=base+index",
        ],
    )


def promote_draft_to_final(
    plan: DraftThumbPlan,
    *,
    chosen_index: int = 0,
    user_picked: bool = False,
    draft_paths: list[str] | None = None,
    log_feedback: bool = True,
) -> list[str]:
    """Lock final seed from draft index (seed = base_seed + index)."""
    idx = max(0, min(int(chosen_index), plan.num_drafts - 1))
    seed = int(plan.base_seed) + idx
    argv = list(plan.final_argv_template)
    if "--seed" not in argv:
        argv.extend(["--seed", str(seed)])
    else:
        i = argv.index("--seed")
        if i + 1 < len(argv):
            argv[i + 1] = str(seed)
    if user_picked and log_feedback and draft_paths and len(draft_paths) >= 2:
        try:
            from utils.training.feedback_bus import record_pick

            win = draft_paths[idx] if idx < len(draft_paths) else draft_paths[0]
            losers = [p for i, p in enumerate(draft_paths) if i != idx]
            record_pick(win, losers, prompt=plan.prompt, seed=seed)
        except Exception:
            pass
    note = "user-picked" if user_picked else "auto-pick-best index"
    plan.notes.append(f"final seed={seed} ({note}, draft[{idx}])")
    return argv


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

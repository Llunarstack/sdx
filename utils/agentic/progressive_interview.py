"""Progressive interview: one ask at a time, with skip → defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from utils.agentic.clarify_pose_place import assume_pose_place_defaults
from utils.agentic.gen_interview import GenInterview, InterviewItem, build_gen_interview
from utils.prompt.facet_decompose import decompose_prompt_facets

__all__ = [
    "ProgressiveState",
    "next_interview_item",
    "skip_interview_item",
    "answer_interview_item",
    "apply_you_decide_defaults",
]

_YOU_DECIDE = frozenset({"", "skip", "you decide", "your choice", "default", "idc", "whatever"})


@dataclass
class ProgressiveState:
    interview: GenInterview
    cursor: int = 0
    answers: dict[str, str] = field(default_factory=dict)
    photos: dict[str, list[str]] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "cursor": self.cursor,
            "answers": dict(self.answers),
            "photos": {k: list(v) for k, v in self.photos.items()},
            "skipped": list(self.skipped),
            "done": self.done,
            "interview": self.interview.to_dict(),
        }


def next_interview_item(
    state: ProgressiveState,
    *,
    include_optional: bool = False,
) -> InterviewItem | None:
    """Return the next unanswered required/recommended item (or optional if requested)."""
    pending = state.interview.pending(include_optional=include_optional)
    # Also skip items already answered via state.answers/photos
    for it in pending:
        if it.id in state.answers or it.id in state.photos or it.id in state.skipped:
            continue
        if it.answer or it.photo_paths:
            continue
        return it
    state.done = True
    return None


def answer_interview_item(
    state: ProgressiveState,
    item_id: str,
    *,
    text: str = "",
    photos: list[str] | None = None,
) -> ProgressiveState:
    raw = str(text or "").strip()
    if raw.lower() in _YOU_DECIDE and not photos:
        return skip_interview_item(state, item_id)
    if raw:
        state.answers[item_id] = raw
        for it in state.interview.items:
            if it.id == item_id:
                it.answer = raw
                break
    if photos:
        state.photos[item_id] = [str(p) for p in photos if p]
        for it in state.interview.items:
            if it.id == item_id:
                it.photo_paths = list(state.photos[item_id])
                break
    state.cursor += 1
    return state


def skip_interview_item(state: ProgressiveState, item_id: str) -> ProgressiveState:
    """Mark item skipped; pose/place get imagination defaults when applicable."""
    state.skipped.append(item_id)
    for it in state.interview.items:
        if it.id != item_id:
            continue
        if it.id == "pose":
            state.answers["pose"] = "standing, looking at viewer, full body"
            it.answer = state.answers["pose"]
        elif it.id == "place":
            state.answers["place"] = "simple studio, soft lighting, plain background"
            it.answer = state.answers["place"]
        elif it.kind == "confirm":
            state.answers[item_id] = "yes"
            it.answer = "yes"
        else:
            it.answer = "you decide"
            state.answers[item_id] = "you decide"
        break
    state.cursor += 1
    return state


def apply_you_decide_defaults(prompt: str, state: ProgressiveState) -> str:
    """Fill remaining holes with assume_pose_place_defaults + collected answers."""
    facets = decompose_prompt_facets(prompt)
    assumed = assume_pose_place_defaults(facets)
    merged = dict(assumed.assumed)
    for k, v in state.answers.items():
        if str(v).strip().lower() in _YOU_DECIDE:
            continue
        if k in ("pose", "place", "style", "attire", "mood", "camera", "identity_notes"):
            merged[k] = v
    base = assumed.enriched_prompt or prompt
    from utils.agentic.clarify_pose_place import apply_clarify_answers

    return apply_clarify_answers(base if not state.answers else prompt, merged)


def start_progressive_interview(prompt: str, **kwargs: Any) -> ProgressiveState:
    interview = build_gen_interview(prompt, **kwargs)
    return ProgressiveState(interview=interview)

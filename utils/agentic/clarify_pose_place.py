"""Interactive / default clarify for missing pose and place before generate."""

from __future__ import annotations

from dataclasses import dataclass, field

from utils.prompt.facet_decompose import PromptFacets, decompose_prompt_facets

__all__ = [
    "ClarifyResult",
    "build_clarify_questions",
    "apply_clarify_answers",
    "assume_pose_place_defaults",
]

_DEFAULT_POSE = "standing, looking at viewer, full body"
_DEFAULT_PLACE_LIGHT = "simple studio, soft lighting, plain background"
_DEFAULT_PLACE_DARK = "simple dark room, soft rim light, plain background"


@dataclass(slots=True)
class ClarifyResult:
    questions: list[str] = field(default_factory=list)
    needs_user: bool = False
    assumed: dict[str, str] = field(default_factory=dict)
    enriched_prompt: str = ""
    notes: str = ""


def build_clarify_questions(facets: PromptFacets | str) -> ClarifyResult:
    """Return 1–2 clarifying questions when pose/place are absent."""
    if isinstance(facets, str):
        facets = decompose_prompt_facets(facets)
    out = ClarifyResult()
    if "pose" in facets.missing:
        out.questions.append(
            "Pose missing — standing, sitting, or something else? "
            "(e.g. standing looking at viewer / sitting crossed legs)"
        )
    if "place" in facets.missing:
        out.questions.append(
            "Place/background missing — where is the character? (e.g. bedroom / white studio / outdoor night city)"
        )
    out.needs_user = bool(out.questions)
    return out


def assume_pose_place_defaults(
    facets: PromptFacets | str,
    *,
    prefer_dark_room: bool | None = None,
) -> ClarifyResult:
    """Fill missing pose/place with imagination defaults (handwritten diagram)."""
    if isinstance(facets, str):
        facets = decompose_prompt_facets(facets)
    out = ClarifyResult()
    dark = prefer_dark_room
    if dark is None:
        # Heuristic: dark clothing / lingerie → soft dark room; else light studio.
        blob = " ".join(facets.attire + facets.subject).lower()
        dark = any(k in blob for k in ("black", "dark", "lingerie", "night"))
    if "pose" in facets.missing:
        out.assumed["pose"] = _DEFAULT_POSE
    if "place" in facets.missing:
        out.assumed["place"] = _DEFAULT_PLACE_DARK if dark else _DEFAULT_PLACE_LIGHT
    out.enriched_prompt = apply_clarify_answers(facets.raw, out.assumed)
    out.notes = "assumed defaults for missing pose/place"
    return out


def apply_clarify_answers(prompt: str, answers: dict[str, str]) -> str:
    """Append user or assumed pose/place (and optional style) to the prompt."""
    base = str(prompt or "").strip().rstrip(",")
    bits: list[str] = []
    for key in ("pose", "place", "style"):
        val = str(answers.get(key) or "").strip()
        if val and val.lower() not in base.lower():
            bits.append(val)
    if not bits:
        return base
    sep = ", " if "," in base or not base else " "
    return f"{base}{sep}{', '.join(bits)}".strip()

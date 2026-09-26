"""Pre-generate interview: ask questions *and* request photos/refs.

Turns a thin prompt into a structured checklist the user (or UI) can answer
before sampling — pose/place text, face photos, outfit shots, style boards, etc.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from utils.prompt.facet_decompose import PromptFacets, decompose_prompt_facets

__all__ = [
    "InterviewItem",
    "GenInterview",
    "build_gen_interview",
    "apply_interview_answers",
    "format_interview_for_user",
]

Priority = Literal["required", "recommended", "optional"]
Kind = Literal["text", "photo", "choice", "confirm"]


@dataclass(slots=True)
class InterviewItem:
    id: str
    kind: Kind
    prompt: str
    priority: Priority = "recommended"
    choices: list[str] = field(default_factory=list)
    facet: str = ""  # subject | attire | style | pose | place | identity | mood
    answer: str = ""
    photo_paths: list[str] = field(default_factory=list)
    why: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenInterview:
    items: list[InterviewItem] = field(default_factory=list)
    needs_user: bool = False
    summary: str = ""

    def pending(self, *, include_optional: bool = False) -> list[InterviewItem]:
        out: list[InterviewItem] = []
        for it in self.items:
            if it.answer or it.photo_paths:
                continue
            if it.priority == "optional" and not include_optional:
                continue
            out.append(it)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "needs_user": self.needs_user,
            "items": [i.to_dict() for i in self.items],
        }


def build_gen_interview(
    prompt: str | PromptFacets,
    *,
    has_face_ref: bool = False,
    has_style_ref: bool = False,
    has_character_sheet: bool = False,
    aggressive: bool = True,
) -> GenInterview:
    """Build questions + photo asks from missing / weak facets."""
    facets = decompose_prompt_facets(prompt) if isinstance(prompt, str) else prompt
    items: list[InterviewItem] = []

    if "pose" in facets.missing:
        items.append(
            InterviewItem(
                id="pose",
                kind="choice",
                facet="pose",
                priority="required",
                prompt="What pose should they be in?",
                choices=[
                    "standing, looking at viewer, full body",
                    "sitting, relaxed, three-quarter view",
                    "dynamic action pose",
                    "portrait / upper body only",
                    "other (describe)",
                ],
                why="Pose is the #1 cause of generic / wrong anatomy framing.",
            )
        )
        items.append(
            InterviewItem(
                id="pose_photo",
                kind="photo",
                facet="pose",
                priority="optional",
                prompt="Optional: upload a pose reference photo (stick figure, photo, or sketch).",
                why="ControlNet/openpose or IP-Adapter pose lock from your image.",
            )
        )

    if "place" in facets.missing:
        items.append(
            InterviewItem(
                id="place",
                kind="choice",
                facet="place",
                priority="required",
                prompt="Where is this scene set?",
                choices=[
                    "plain white / studio background",
                    "simple dark room",
                    "bedroom",
                    "outdoor daylight",
                    "cyberpunk night city",
                    "other (describe)",
                ],
                why="Without place, the model invents cluttered backgrounds.",
            )
        )
        items.append(
            InterviewItem(
                id="place_photo",
                kind="photo",
                facet="place",
                priority="optional",
                prompt="Optional: upload a location / background photo to match lighting and space.",
                why="Depth/canny from a real room beats guessing.",
            )
        )

    # Identity / face — always valuable for “this character”
    if not has_face_ref and not has_character_sheet:
        items.append(
            InterviewItem(
                id="face_photo",
                kind="photo",
                facet="identity",
                priority="recommended" if aggressive else "optional",
                prompt=(
                    "Got a face or character photo? Upload 1–4 images for identity lock "
                    "(front face best; optional 3/4 and profile)."
                ),
                why="Reference adapter / InstantID-style consistency across gens.",
            )
        )
        items.append(
            InterviewItem(
                id="identity_notes",
                kind="text",
                facet="identity",
                priority="optional",
                prompt="Any must-keep identity details? (hair color, eye color, scars, species traits…)",
                why="Text backup when photos are soft or partial.",
            )
        )

    if facets.attire and aggressive:
        items.append(
            InterviewItem(
                id="attire_photo",
                kind="photo",
                facet="attire",
                priority="recommended",
                prompt=f"Upload a photo of the outfit you want (tags: {', '.join(facets.attire[:4])}).",
                why="Attire refs beat tag soup for fabric folds and cut.",
            )
        )
    elif not facets.attire and aggressive:
        items.append(
            InterviewItem(
                id="attire",
                kind="text",
                facet="attire",
                priority="recommended",
                prompt="What are they wearing? (or ‘nude’ / ‘casual streetwear’ / describe colors)",
                why="Missing clothing is a common prompt hole.",
            )
        )

    if facets.style or facets.artist_hint:
        if not has_style_ref:
            styl = facets.artist_hint or ", ".join(facets.style[:2])
            items.append(
                InterviewItem(
                    id="style_photos",
                    kind="photo",
                    facet="style",
                    priority="recommended",
                    prompt=(
                        f"Upload 2–6 images in the “{styl}” look (or confirm we should "
                        "scrape the web for artist/style refs)."
                    ),
                    why="InstantStyle / moodboard from *your* picks beats random search hits.",
                )
            )
            items.append(
                InterviewItem(
                    id="style_web_ok",
                    kind="confirm",
                    facet="style",
                    priority="optional",
                    prompt="OK to search the web for style/artist reference images?",
                    choices=["yes", "no — only use what I upload"],
                    why="Consent gate for scraping.",
                )
            )
    elif aggressive:
        items.append(
            InterviewItem(
                id="style",
                kind="choice",
                facet="style",
                priority="recommended",
                prompt="Any art style preference?",
                choices=[
                    "photoreal",
                    "anime / illustration",
                    "painterly",
                    "3D render",
                    "match a moodboard I’ll upload",
                    "surprise me",
                ],
                why="Style ambiguity → mushy hybrid look.",
            )
        )
        items.append(
            InterviewItem(
                id="style_photos",
                kind="photo",
                facet="style",
                priority="optional",
                prompt="Upload a moodboard (palette, lighting, brushwork — ignore subject).",
                why="Decompose-and-apply style without naming an artist.",
            )
        )

    # Composition / camera
    items.append(
        InterviewItem(
            id="camera",
            kind="choice",
            facet="place",
            priority="optional",
            prompt="Camera / framing?",
            choices=[
                "full body",
                "cowboy shot",
                "portrait",
                "wide establishing",
                "low angle",
                "default",
            ],
            why="Locks composition before denoise.",
        )
    )
    items.append(
        InterviewItem(
            id="mood",
            kind="text",
            facet="mood",
            priority="optional",
            prompt="Mood / lighting in one phrase? (e.g. soft morning window light, neon rim…)",
            why="Lighting tags change the whole image.",
        )
    )
    items.append(
        InterviewItem(
            id="must_avoid",
            kind="text",
            facet="subject",
            priority="optional",
            prompt="Anything to avoid? (extra limbs, text, watermark, specific colors…)",
            why="Feeds negative prompt + critic gates.",
        )
    )
    items.append(
        InterviewItem(
            id="deliverable",
            kind="choice",
            facet="subject",
            priority="optional",
            prompt="What do you want out of this run?",
            choices=[
                "single hero image",
                "4 variations, pick best",
                "character sheet (multi-view)",
                "then edit regions interactively",
                "turntable / 360 later",
            ],
            why="Routes to sample vs sheet vs video/multiview.",
        )
    )

    if facets.nsfw:
        items.append(
            InterviewItem(
                id="nsfw_confirm",
                kind="confirm",
                facet="subject",
                priority="required",
                prompt="Prompt looks NSFW — confirm uncensored generation and web ref search policy?",
                choices=["yes, proceed uncensored", "keep SFW only", "cancel"],
                why="Explicit consent before scrape / gen.",
            )
        )

    pending = [i for i in items if i.priority in ("required", "recommended")]
    interview = GenInterview(
        items=items,
        needs_user=bool(pending),
        summary=(f"{len(pending)} recommended asks ({sum(1 for i in pending if i.kind == 'photo')} photo requests)"),
    )
    return interview


def apply_interview_answers(
    prompt: str,
    interview: GenInterview,
    *,
    answers: dict[str, str] | None = None,
    photos: dict[str, list[str]] | None = None,
) -> tuple[str, GenInterview, dict[str, list[str]]]:
    """Merge text answers into prompt; collect photo paths by facet."""
    answers = answers or {}
    photos = photos or {}
    text_bits: list[str] = []
    by_facet: dict[str, list[str]] = {}

    for it in interview.items:
        if it.id in answers and str(answers[it.id]).strip():
            it.answer = str(answers[it.id]).strip()
        if it.id in photos and photos[it.id]:
            it.photo_paths = [str(p) for p in photos[it.id] if p]
            facet = it.facet or "misc"
            by_facet.setdefault(facet, []).extend(it.photo_paths)

        if not it.answer:
            continue
        if it.kind == "confirm":
            continue
        if it.id in ("must_avoid", "deliverable", "style_web_ok", "nsfw_confirm"):
            continue
        if it.answer.lower() in ("default", "surprise me", "other (describe)"):
            continue
        text_bits.append(it.answer)

    base = str(prompt or "").strip().rstrip(",")
    for bit in text_bits:
        if bit.lower() not in base.lower():
            sep = ", " if "," in base or not base else " "
            base = f"{base}{sep}{bit}"

    interview.needs_user = bool(interview.pending(include_optional=False))
    return base, interview, by_facet


def format_interview_for_user(interview: GenInterview, *, include_optional: bool = True) -> str:
    """Human-readable checklist for CLI / chat UI."""
    lines = [f"Pre-generate interview — {interview.summary}", ""]
    for it in interview.items:
        if it.priority == "optional" and not include_optional:
            continue
        mark = {"required": "!", "recommended": "*", "optional": "."}[it.priority]
        kind = "PHOTO" if it.kind == "photo" else it.kind.upper()
        lines.append(f"[{mark}] ({kind}) {it.id}: {it.prompt}")
        if it.choices:
            lines.append(f"      choices: {'; '.join(it.choices)}")
        if it.why:
            lines.append(f"      why: {it.why}")
        if it.answer:
            lines.append(f"      answer: {it.answer}")
        if it.photo_paths:
            lines.append(f"      photos: {', '.join(it.photo_paths)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"

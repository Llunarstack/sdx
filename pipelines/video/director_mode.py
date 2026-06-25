"""AI Director Mode — expand one prompt into an editable storyboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .storyboard import StoryboardCut, camera_prompt_fragment

__all__ = [
    "DirectorExpansion",
    "DirectorNotes",
    "expand_prompt_to_storyboard",
    "genre_director_notes",
]


@dataclass(slots=True)
class DirectorNotes:
    mood: str = ""
    lens: str = ""
    camera: str = ""
    color_grade: str = ""
    music: str = ""
    pacing: str = ""


@dataclass(slots=True)
class DirectorExpansion:
    cuts: List[StoryboardCut]
    notes: DirectorNotes
    genre: str = ""
    raw_prompt: str = ""


_GENRE_RULES: List[tuple[tuple[str, ...], str, DirectorNotes]] = [
    (
        ("horror", "scary", "abandoned", "creature", "dark"),
        "horror",
        DirectorNotes(
            mood="dread, unease",
            lens="35mm wide",
            camera="slow dolly, negative space",
            color_grade="desaturated, crushed blacks",
            music="low drones, distant metallic hits",
            pacing="slow reveal",
        ),
    ),
    (
        ("action", "chase", "explosion", "fight", "battle"),
        "action",
        DirectorNotes(
            mood="kinetic urgency",
            lens="24mm handheld",
            camera="tracking, whip pans",
            color_grade="teal orange, high contrast",
            music="driving percussion",
            pacing="fast cuts",
        ),
    ),
    (
        ("romance", "love", "kiss", "wedding"),
        "romance",
        DirectorNotes(
            mood="intimate warmth",
            lens="85mm shallow",
            camera="gentle push in",
            color_grade="warm soft highlights",
            music="strings, piano",
            pacing="lingering beats",
        ),
    ),
    (
        ("comedy", "funny", "slapstick"),
        "comedy",
        DirectorNotes(
            mood="playful",
            lens="50mm",
            camera="static wide for timing",
            color_grade="bright, even",
            music="light staccato",
            pacing="beat on gag",
        ),
    ),
    (
        ("anime", "ghibli", "magical"),
        "anime",
        DirectorNotes(
            mood="wonder",
            lens="anime dramatic angles",
            camera="push in on reaction",
            color_grade="painterly skies",
            music="orchestral swell",
            pacing="held emotional beats",
        ),
    ),
]


def genre_director_notes(prompt: str) -> tuple[str, DirectorNotes]:
    low = (prompt or "").lower()
    for keys, genre, notes in _GENRE_RULES:
        if any(k in low for k in keys):
            return genre, notes
    return "drama", DirectorNotes(
        mood="grounded drama",
        lens="50mm",
        camera="motivated movement",
        color_grade="naturalistic",
        music="subtle ambient",
        pacing="scene rhythm",
    )


def _infer_shot_sequence(prompt: str, genre: str) -> List[tuple[str, float, str, str]]:
    """Returns (prompt_fragment, duration_weight, camera, shot_type)."""
    low = prompt.lower()
    beats: List[tuple[str, float, str, str]] = []
    if genre in ("action", "horror"):
        beats.append(("establishing geography and stakes", 0.22, "establishing", "establishing"))
    elif "enter" in low or "walk" in low or "arrive" in low:
        beats.append(("establishing context", 0.20, "establishing", "establishing"))

    if any(w in low for w in ("fight", "chase", "battle", "explosion")):
        beats.append((f"main action: {prompt}", 0.45, "tracking", "medium"))
        beats.append(("reaction or consequence beat", 0.18, "close_up", "close_up"))
    elif any(w in low for w in ("close", "face", "eyes", "portrait")):
        beats.append((prompt, 0.55, "push_in", "close_up"))
    else:
        beats.append((f"primary story beat: {prompt}", 0.50, "tracking", "medium"))

    if genre == "horror" and len(beats) < 3:
        beats.append(("lingering unsettling hold", 0.15, "static", "wide"))
    elif len(beats) < 2:
        beats.append(("resolution or button", 0.25, "pull_back", "wide"))
    return beats


def expand_prompt_to_storyboard(
    prompt: str,
    *,
    duration_sec: float = 6.0,
    genre_override: str = "",
    characters: Optional[Sequence[str]] = None,
) -> DirectorExpansion:
    """
    Turn one line into Kling-style multi-shot storyboard + director notes.

    User can edit cuts in scene JSON before generation (--plan-only).
    """
    genre, notes = genre_director_notes(prompt)
    if genre_override:
        genre = genre_override
        _, notes = genre_director_notes(f"{genre} {prompt}")

    seq = _infer_shot_sequence(prompt, genre)
    total_w = sum(w for _, w, _, _ in seq) or 1.0
    cuts: List[StoryboardCut] = []
    chars = list(characters or [])
    for i, (frag, weight, cam, st) in enumerate(seq):
        dur = max(0.5, duration_sec * (weight / total_w))
        cam_frag = camera_prompt_fragment(cam)
        p = frag
        if cam_frag and cam_frag.lower() not in p.lower():
            p = f"{p}, {cam_frag}"
        cuts.append(
            StoryboardCut(
                id=f"dir_{i}",
                prompt=p.strip(),
                duration_sec=round(dur, 2),
                camera=cam,
                shot_type=st,
                characters=chars,
                transition="dissolve" if i > 0 and genre in ("romance", "drama") else "cut",
            )
        )
    return DirectorExpansion(cuts=cuts, notes=notes, genre=genre, raw_prompt=prompt)

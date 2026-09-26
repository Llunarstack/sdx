"""
Competitor-weakness prompt suite.

A fixed, versioned set of prompts chosen to probe the failure modes that the
leading text-to-image systems share (see docs/COMPETITIVE_ANALYSIS.md). Each
entry is tagged with a ``category`` and, where a check is objective, carries
structured expectations (``expected_count``, ``expected_text``) so the scorers
in ``scorers.py`` can grade it automatically.

Keep this list stable and append-only; scores are only comparable across runs if
the suite doesn't silently change. Bump ``SUITE_VERSION`` when you add prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SUITE_VERSION = "1.2.0"


@dataclass(frozen=True, slots=True)
class SuiteItem:
    id: str
    prompt: str
    category: str
    # Objective expectations (only set where a category supports auto-grading).
    expected_count: int | None = None
    expected_text: str | None = None
    # Free-form notes on what "passing" looks like (for human review).
    note: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


# Categories map 1:1 to documented cross-model weaknesses.
CATEGORIES: dict[str, str] = {
    "long_adherence": "Long, multi-clause prompts where late tokens get dropped.",
    "relation": "Spatial/relational composition (A left of B, holding C).",
    "count": "Exact object counts — no major model guarantees these.",
    "text_render": "Legible in-image text of a specific string.",
    "anatomy": "Hands, small/far faces, crowds — degrade at distance.",
    "negation": "Excluding a concept the prompt names ('without X').",
    "style_control": "Adhering to a named, specific style.",
    "contact": "Feet / objects planted on the supporting surface (anti-float).",
    "bind": "Attribute binding: colors/clothes stay on the correct person.",
    "occlusion": "Subject visible through or behind occluders, not melted into them.",
    "reflection": "Physically plausible reflections, not a second hallucinated object.",
}


SUITE: tuple[SuiteItem, ...] = (
    # --- long_adherence -----------------------------------------------------
    SuiteItem(
        "long_01",
        "A weathered lighthouse keeper in a yellow raincoat stands on wet black "
        "rocks at dusk, holding a brass lantern in his left hand, a border collie "
        "beside him, storm clouds and a distant sailboat on the horizon",
        "long_adherence",
        note="Check every clause: raincoat color, lantern hand, dog, sailboat.",
    ),
    SuiteItem(
        "long_02",
        "A cozy bookstore cafe interior: exposed brick, a spiral staircase on the "
        "right, a orange tabby cat asleep on a stack of books, steam rising from a "
        "latte, warm afternoon light through tall windows on the left",
        "long_adherence",
        note="Staircase-right and windows-left must both hold.",
    ),
    # --- relation -----------------------------------------------------------
    SuiteItem(
        "rel_01",
        "A red cube to the left of a blue sphere, with a green pyramid behind both",
        "relation",
        note="Left/right and behind must be correct simultaneously.",
    ),
    SuiteItem(
        "rel_02",
        "A cat sitting on top of a microwave that is on top of a refrigerator",
        "relation",
        note="Vertical stacking order.",
    ),
    # --- count --------------------------------------------------------------
    SuiteItem("count_01", "Exactly three red apples on a white plate", "count", expected_count=3),
    SuiteItem("count_02", "Five yellow rubber ducks in a row on a bathtub edge", "count", expected_count=5),
    SuiteItem("count_03", "A single lit candle in a dark room", "count", expected_count=1),
    # --- text_render --------------------------------------------------------
    SuiteItem(
        "text_01",
        'A vintage enamel shop sign that reads "OPEN" in bold serif letters',
        "text_render",
        expected_text="OPEN",
    ),
    SuiteItem(
        "text_02",
        'A latte with the word "HELLO" drawn in the foam',
        "text_render",
        expected_text="HELLO",
    ),
    # --- anatomy ------------------------------------------------------------
    SuiteItem(
        "anat_01",
        "Close-up of two hands carefully tying a shoelace, five fingers each",
        "anatomy",
        note="Finger count and hand structure.",
    ),
    SuiteItem(
        "anat_02",
        "A crowd of a dozen people crossing a busy street, seen from across the road",
        "anatomy",
        note="Small/far faces and bodies stay coherent.",
    ),
    # --- negation -----------------------------------------------------------
    SuiteItem(
        "neg_01",
        "A dining table set for breakfast, without any coffee or mugs",
        "negation",
        note="Coffee/mugs must be ABSENT.",
    ),
    SuiteItem(
        "neg_02",
        "An empty beach with no people and no boats",
        "negation",
        note="People/boats must be ABSENT.",
    ),
    # --- style_control ------------------------------------------------------
    SuiteItem(
        "style_01",
        "A mountain landscape in flat 2-color risograph print style, coarse grain",
        "style_control",
        note="Risograph look, limited palette.",
    ),
    SuiteItem(
        "style_02",
        "A portrait of a fox in the style of a technical blueprint, white lines on blue",
        "style_control",
        note="Blueprint palette and line quality.",
    ),
    # --- contact ------------------------------------------------------------
    SuiteItem(
        "contact_01",
        "A woman standing on wet pavement at night, shoes planted on the ground, "
        "contact shadow under both feet, no hovering",
        "contact",
        note="Both feet on pavement with contact shadows; no floating.",
    ),
    SuiteItem(
        "contact_02",
        "A coffee mug sitting on a wooden table, base flush with the tabletop, no floating",
        "contact",
        note="Mug base flush with table; no gap underneath.",
    ),
    # --- bind ---------------------------------------------------------------
    SuiteItem(
        "bind_01",
        "A woman in a red dress on the left and a man in a blue suit on the right, colors must not swap",
        "bind",
        note="Red dress stays on the woman; blue suit stays on the man.",
    ),
    SuiteItem(
        "bind_02",
        "A brown dog wearing a blue collar sitting beside a white cat with a green bell, wrong pairing forbidden",
        "bind",
        note="Blue collar on dog; green bell on cat — no attribute swap.",
    ),
    # --- occlusion ----------------------------------------------------------
    SuiteItem(
        "occlude_01",
        "A person standing behind a chain-link fence, face and jacket visible through "
        "the gaps, fence in the foreground",
        "occlusion",
        note="Person readable through fence gaps; not melted into mesh.",
    ),
    SuiteItem(
        "occlude_02",
        "A red apple partly hidden behind a wine glass, both complete objects, glass in front",
        "occlusion",
        note="Apple and glass remain distinct; glass occludes in front.",
    ),
    # --- reflection ---------------------------------------------------------
    SuiteItem(
        "reflect_01",
        "A chrome kettle on black granite, a sharp reflection of the kettle in the stone, not a second kettle",
        "reflection",
        note="Reflection matches kettle pose; no duplicate hallucinated object.",
    ),
    SuiteItem(
        "rel_03",
        "A yellow lamp between a red chair and a blue stool",
        "relation",
        note="Lamp in the middle; chair left, stool right; colors stay bound.",
    ),
    SuiteItem(
        "rel_04",
        "A sailor holding a brass lantern in his left hand, standing on wet rocks",
        "relation",
        note="Lantern is gripped, not floating; left hand occupancy.",
    ),
    # --- text_render (continued) --------------------------------------------
    SuiteItem(
        "text_03",
        'A storefront with two lines of text: top line "OPEN NOW" and bottom line "24 HOURS"',
        "text_render",
        expected_text="OPEN NOW",
        note='Top line "OPEN NOW" is auto-graded; bottom line "24 HOURS" is human review.',
    ),
)


def items_by_category(category: str | None = None) -> list[SuiteItem]:
    if category is None:
        return list(SUITE)
    return [s for s in SUITE if s.category == category]


def suite_as_manifest_rows() -> list[dict]:
    """Emit the suite as JSONL-ready rows (id, prompt, category, expectations)."""
    rows = []
    for s in SUITE:
        row = {"id": s.id, "prompt": s.prompt, "category": s.category}
        if s.expected_count is not None:
            row["expected_count"] = s.expected_count
        if s.expected_text is not None:
            row["expected_text"] = s.expected_text
        if s.note:
            row["note"] = s.note
        rows.append(row)
    return rows


__all__ = [
    "SUITE",
    "SUITE_VERSION",
    "CATEGORIES",
    "SuiteItem",
    "items_by_category",
    "suite_as_manifest_rows",
]

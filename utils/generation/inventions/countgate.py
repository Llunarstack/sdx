"""COUNTGATE — Discrete count binding prior for multi-object prompts.

Problem: "four cats" often yields 2–3 or 5+. COUNTGATE extracts counts,
injects hard count tags + separation positives, and builds a soft grid prior
hint (for box layout / regional CFG).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["CountSpec", "CountGatePlan", "plan_countgate", "apply_countgate_prompts", "count_grid_boxes"]

_WORD_COUNTS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "a pair of": 2,
    "a couple of": 2,
}

_COUNT_RE = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|a pair of|a couple of)\s+"
    r"([a-z][a-z0-9_-]{1,24})s?\b",
    re.IGNORECASE,
)
_BOORU_COUNT = re.compile(r"\b(\d+)(girl|boy|girls|boys|people|cats|dogs)\b", re.IGNORECASE)


@dataclass(slots=True)
class CountSpec:
    n: int
    noun: str


@dataclass
class CountGatePlan:
    specs: list[CountSpec] = field(default_factory=list)
    positive_addon: str = ""
    negative_addon: str = ""
    boxes: list[list[float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "specs": [asdict(s) for s in self.specs],
            "positive_addon": self.positive_addon,
            "negative_addon": self.negative_addon,
            "boxes": self.boxes,
            "notes": list(self.notes),
        }


def _parse_n(raw: str) -> int:
    r = raw.lower().strip()
    if r.isdigit():
        return int(r)
    return int(_WORD_COUNTS.get(r, 0))


def plan_countgate(prompt: str) -> CountGatePlan:
    text = str(prompt or "")
    specs: list[CountSpec] = []
    for m in _COUNT_RE.finditer(text):
        n = _parse_n(m.group(1))
        noun = m.group(2).lower().rstrip("s")
        if n >= 1:
            specs.append(CountSpec(n=n, noun=noun))
    for m in _BOORU_COUNT.finditer(text):
        n = int(m.group(1))
        noun = m.group(2).lower().rstrip("s")
        specs.append(CountSpec(n=n, noun=noun))

    # Dedup by noun keep max n
    by_noun: dict[str, int] = {}
    for s in specs:
        by_noun[s.noun] = max(by_noun.get(s.noun, 0), s.n)
    specs = [CountSpec(n=n, noun=k) for k, n in by_noun.items()]

    pos = []
    neg = [
        "wrong number of subjects",
        "extra subjects",
        "missing subjects",
        "merged duplicates",
        "cloned identical copies overlapping",
    ]
    primary_n = 0
    primary_noun = "subject"
    for s in specs:
        pos.append(f"exactly {s.n} distinct {s.noun}{'s' if s.n != 1 else ''}")
        pos.append(f"count:{s.n} {s.noun}")
        if s.n > primary_n:
            primary_n = s.n
            primary_noun = s.noun
        if s.n >= 2:
            pos.append("clearly separated instances")
            neg.append(f"{s.n + 1} {s.noun}s")
            if s.n > 1:
                neg.append(f"{s.n - 1} {s.noun}s")

    boxes = count_grid_boxes(primary_n) if primary_n >= 2 else []
    return CountGatePlan(
        specs=specs,
        positive_addon=", ".join(dict.fromkeys(pos)),
        negative_addon=", ".join(dict.fromkeys(neg)),
        boxes=boxes,
        notes=[f"primary {primary_n}×{primary_noun}"],
    )


def count_grid_boxes(n: int) -> list[list[float]]:
    """Axis-aligned non-overlap priors for regional binding."""
    n = max(0, min(int(n), 8))
    if n <= 1:
        return [[0.1, 0.1, 0.9, 0.9]] if n == 1 else []
    cols = 2 if n <= 4 else 3
    rows = (n + cols - 1) // cols
    boxes: list[list[float]] = []
    for i in range(n):
        r, c = divmod(i, cols)
        x0 = 0.05 + c * (0.9 / cols) + 0.02
        x1 = 0.05 + (c + 1) * (0.9 / cols) - 0.02
        y0 = 0.08 + r * (0.85 / rows) + 0.02
        y1 = 0.08 + (r + 1) * (0.85 / rows) - 0.02
        boxes.append([x0, y0, min(x1, 0.95), min(y1, 0.95)])
    return boxes


def apply_countgate_prompts(positive: str, negative: str = "", plan: CountGatePlan | None = None) -> tuple[str, str]:
    plan = plan or plan_countgate(positive)
    pos = str(positive or "").rstrip(",")
    neg = str(negative or "").rstrip(",")
    if plan.positive_addon:
        pos = f"{pos}, {plan.positive_addon}"
    if plan.negative_addon:
        neg = f"{neg}, {plan.negative_addon}".strip(", ")
    return pos.strip(", "), neg.strip(", ")

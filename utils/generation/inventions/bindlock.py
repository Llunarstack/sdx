"""BINDLOCK — Attribute Binding Lock from prompt parse.

Problem: "red cube left of blue sphere" often swaps colors/objects.
BINDLOCK extracts (object, attributes, spatial) triples and rewrites the
prompt with explicit ownership + anti-swap negatives (neuro-symbolic lite).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["BindTriple", "BindLockPlan", "plan_bindlock", "apply_bindlock_to_prompts"]

_COLOR = (
    r"red|blue|green|yellow|orange|purple|pink|black|white|brown|gray|grey|"
    r"cyan|magenta|gold|silver|teal|navy|beige"
)
_OBJECT = (
    r"cube|sphere|ball|box|cat|dog|girl|boy|man|woman|car|book|cup|bottle|"
    r"apple|bird|horse|chair|table|lamp|tree|flower|sword|shield|hat|dress|"
    r"hoodie|jacket|catgirl|fox|dragon"
)
_SPATIAL = (
    r"left of|right of|above|below|under|over|next to|beside|behind|in front of|"
    r"on top of|inside|outside"
)


@dataclass(slots=True)
class BindTriple:
    subject: str
    attributes: list[str] = field(default_factory=list)
    relation: str = ""
    other: str = ""

    def ownership_phrase(self) -> str:
        attrs = ", ".join(self.attributes) if self.attributes else ""
        core = f"{attrs} {self.subject}".strip() if attrs else self.subject
        if self.relation and self.other:
            return f"{core} {self.relation} {self.other}".strip()
        return core


@dataclass
class BindLockPlan:
    triples: list[BindTriple] = field(default_factory=list)
    positive_addon: str = ""
    negative_addon: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "triples": [asdict(t) for t in self.triples],
            "positive_addon": self.positive_addon,
            "negative_addon": self.negative_addon,
            "notes": list(self.notes),
        }


def plan_bindlock(prompt: str) -> BindLockPlan:
    text = str(prompt or "")
    triples: list[BindTriple] = []
    # Pattern: <color> <object> <spatial> <color> <object>
    pat = re.compile(
        rf"\b((?:{_COLOR})\s+)?({_OBJECT})\b(?:\s+({_SPATIAL})\s+((?:{_COLOR})\s+)?({_OBJECT}))?",
        re.IGNORECASE,
    )
    for m in pat.finditer(text):
        c1, o1, rel, c2, o2 = m.groups()
        attrs = [c1.strip()] if c1 else []
        other = ""
        if o2:
            other_attrs = [c2.strip()] if c2 else []
            other = f"{' '.join(other_attrs)} {o2}".strip()
        triples.append(
            BindTriple(
                subject=o1.lower(),
                attributes=[a.lower() for a in attrs if a],
                relation=(rel or "").lower(),
                other=other.lower(),
            )
        )

    # Also catch "N red apples" style
    count_pat = re.compile(rf"\b(\d+)\s+((?:{_COLOR})\s+)?({_OBJECT})s?\b", re.IGNORECASE)
    for m in count_pat.finditer(text):
        n, c, o = m.groups()
        attrs = [c.strip().lower()] if c else []
        attrs.append(f"count:{n}")
        triples.append(BindTriple(subject=o.lower(), attributes=attrs))

    # Dedup by subject+attrs
    seen: set[str] = set()
    uniq: list[BindTriple] = []
    for t in triples:
        key = f"{t.subject}|{','.join(t.attributes)}|{t.relation}|{t.other}"
        if key in seen:
            continue
        seen.add(key)
        uniq.append(t)

    pos_bits = [t.ownership_phrase() for t in uniq]
    # Explicit binding language
    pos_bits.extend(
        [
            "correct attribute binding",
            "each object keeps its own color",
            "no attribute swap between objects",
        ]
    )
    neg_bits = [
        "attribute swap",
        "color swap",
        "wrong color on wrong object",
        "swapped positions",
        "merged objects",
        "incorrect binding",
    ]
    # Cross-negatives: don't put A's color on B
    colors = []
    subjects = []
    for t in uniq:
        subjects.append(t.subject)
        for a in t.attributes:
            if re.fullmatch(_COLOR, a, re.IGNORECASE):
                colors.append((t.subject, a))
    for subj, col in colors:
        for other in subjects:
            if other != subj:
                neg_bits.append(f"{col} {other}")

    return BindLockPlan(
        triples=uniq,
        positive_addon=", ".join(dict.fromkeys(pos_bits)),
        negative_addon=", ".join(dict.fromkeys(neg_bits)),
        notes=[f"{len(uniq)} bind triples"],
    )


def apply_bindlock_to_prompts(positive: str, negative: str = "", plan: BindLockPlan | None = None) -> tuple[str, str]:
    plan = plan or plan_bindlock(positive)
    pos = str(positive or "").rstrip(",")
    neg = str(negative or "").rstrip(",")
    if plan.positive_addon and plan.positive_addon.lower() not in pos.lower():
        pos = f"{pos}, {plan.positive_addon}"
    if plan.negative_addon:
        neg = f"{neg}, {plan.negative_addon}".strip(", ")
    return pos.strip(", "), neg.strip(", ")

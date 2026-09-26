"""NEGATRON — Explicit negation operator for T2I prompts.

Problem: models ignore "no hat" / "without text" / "not smiling".
NEGATRON parses negation cues and forces them into negatives + stripped positives.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["NegatronPlan", "plan_negatron", "apply_negatron"]

_NEG_PATTERNS = [
    re.compile(r"\b(?:no|without|not|n't)\s+([a-z0-9][\w\s-]{1,40}?)(?=,|\.|$)", re.IGNORECASE),
    re.compile(r"\b(?:exclude|avoid|ban)\s+([a-z0-9][\w\s-]{1,40}?)(?=,|\.|$)", re.IGNORECASE),
    re.compile(r"\b(?:free of|absent of)\s+([a-z0-9][\w\s-]{1,40}?)(?=,|\.|$)", re.IGNORECASE),
]


@dataclass
class NegatronPlan:
    forbidden: list[str] = field(default_factory=list)
    cleaned_positive: str = ""
    negative_addon: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_negatron(prompt: str) -> NegatronPlan:
    text = str(prompt or "")
    forbidden: list[str] = []
    cleaned = text
    for pat in _NEG_PATTERNS:
        for m in pat.finditer(text):
            phrase = m.group(1).strip(" ,.")
            if len(phrase) < 2:
                continue
            forbidden.append(phrase)
            # Remove the negation clause from positive so model isn't confused
            cleaned = cleaned.replace(m.group(0), " ")
    # Cleanup commas
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r",\s*,+", ", ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,")

    # Expand common negation targets
    expanded: list[str] = []
    for f in forbidden:
        expanded.append(f)
        fl = f.lower()
        if "text" in fl or "watermark" in fl:
            expanded.extend(["watermark", "signature", "logo", "subtitle", "caption overlay"])
        if "hat" in fl:
            expanded.extend(["hat", "cap", "beanie"])
        if "smile" in fl:
            expanded.extend(["smiling", "grin", "toothy smile"])

    uniq = list(dict.fromkeys(expanded))
    neg = ", ".join(uniq)
    if uniq:
        neg = f"{neg}, presence of forbidden concepts"
    return NegatronPlan(
        forbidden=uniq,
        cleaned_positive=cleaned or text,
        negative_addon=neg,
        notes=[f"{len(uniq)} negation targets"],
    )


def apply_negatron(positive: str, negative: str = "") -> tuple[str, str, NegatronPlan]:
    plan = plan_negatron(positive)
    neg = str(negative or "").rstrip(",")
    if plan.negative_addon:
        neg = f"{neg}, {plan.negative_addon}".strip(", ")
    return plan.cleaned_positive, neg, plan

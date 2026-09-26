"""FAILURE ORACLE — Map likely T2I failure modes → repair skills.

Diagnoses prompt risk *before* generate and optional post-hoc cues, then
emits a repair plan (countgate, bindlock, inpaint regions, friction, etc.).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "FailureOracleReport",
    "diagnose_failures",
    "repair_plan_from_failures",
]


@dataclass
class FailureOracleReport:
    risks: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    repairs: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagnose_failures(prompt: str, *, negative: str = "") -> FailureOracleReport:
    p = str(prompt or "").lower()
    scores: dict[str, float] = {}

    # Counting risk
    if re.search(r"\b([2-9]|two|three|four|five|six)\s+\w+", p) or re.search(r"\b[2-9](girl|boy|cat)", p):
        scores["count"] = 0.85
    # Binding / spatial
    if re.search(r"\b(left of|right of|above|below|next to)\b", p) or re.search(
        r"\b(red|blue|green)\s+\w+\b.+\b(red|blue|green)\s+\w+", p
    ):
        scores["binding"] = 0.8
    # Negation
    if re.search(r"\b(no|without|not|n't)\s+\w+", p):
        scores["negation"] = 0.9
    # Hands / anatomy
    if re.search(r"\b(hand|hands|fingers|holding|grasp|pose)\b", p):
        scores["anatomy"] = 0.7
    # Text / glyph
    if '"' in prompt or "'" in prompt or re.search(r"\b(text|sign|logo|poster|typography)\b", p):
        scores["glyph"] = 0.75
    # Multi-character bleed
    if re.search(r"\b(2girl|3girl|two girls|multiple|both wearing)\b", p):
        scores["multi_char"] = 0.8
    # Plastic / beauty
    if re.search(r"\b(portrait|skin|beauty|face|close-?up)\b", p):
        scores["plastic"] = 0.55
    # Long prompt truncation
    if len(p) > 400:
        scores["long_prompt"] = 0.65

    risks = [k for k, v in sorted(scores.items(), key=lambda kv: -kv[1]) if v >= 0.5]
    return FailureOracleReport(risks=risks, scores=scores, notes=[f"{len(risks)} elevated risks"])


def repair_plan_from_failures(report: FailureOracleReport) -> FailureOracleReport:
    repairs: list[dict[str, Any]] = []
    for risk in report.risks:
        if risk == "count":
            repairs.append({"module": "countgate", "action": "apply_countgate_prompts"})
        elif risk == "binding":
            repairs.append({"module": "bindlock", "action": "apply_bindlock_to_prompts"})
        elif risk == "negation":
            repairs.append({"module": "negatron", "action": "apply_negatron"})
        elif risk == "anatomy":
            repairs.append({"module": "edit_skills", "action": "fix_region=hands,face", "post": True})
        elif risk == "glyph":
            repairs.append({"module": "ideogram", "action": "plan_ideogram_layout"})
        elif risk == "multi_char":
            repairs.append({"module": "multi_char_cast", "action": "compile_cast_scene"})
        elif risk == "plastic":
            repairs.append({"module": "friction", "action": "apply_friction_to_prompts"})
        elif risk == "long_prompt":
            repairs.append({"module": "prompt_breakdown", "action": "enable prompt_breakdown=auto"})
    report.repairs = repairs
    return report

"""
Detect logically incompatible prompt fragments before sampling wastes steps.

Unlike tag negation lists, this scores *pairs* of claims (lighting, time, quantity, space).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class Contradiction:
    left: str
    right: str
    severity: float  # 0..1
    category: str
    resolution: str


# (pattern_a, pattern_b, category, resolution hint)
_CONFLICT_RULES: Tuple[Tuple[str, str, str, str], ...] = (
    (r"\bnoon\b|\bmidday\b", r"\bsunset\b|\bsunrise\b|\bdusk\b|\bdawn\b", "time", "pick one time of day"),
    (r"\bnight\b|\bmidnight\b", r"\bbright\s+day\b|\bnoon\b", "time", "pick day or night"),
    (r"\bempty\b|\bdeserted\b|\bno\s+people\b|\bno\s+crowd", r"\bcrowd\b|\bpacked\b|\bbusy\b", "quantity", "choose empty or crowded"),
    (r"\bindoor\b|\binside\b|\binterior\b", r"\boutdoor\b|\boutside\b|\blandscape\b", "space", "choose interior or exterior"),
    (r"\bmacro\b|\bclose[- ]?up\b", r"\bwide\s+shot\b|\bestablishing\b|\baerial\b", "framing", "choose one framing scale"),
    (r"\bminimal\b|\bsparse\b", r"\bcluttered\b|\bbusy\s+background\b", "composition", "choose minimal or busy"),
    (r"\bphotorealistic\b|\bhyperreal", r"\bcartoon\b|\banime\b|\billustration\b", "style", "pick one render style"),
    (r"\bunderwater\b", r"\bdesert\b|\barid\b", "environment", "environments are incompatible"),
    (r"\bsnow\b|\bblizzard\b", r"\btropical\b|\bpalm\b", "climate", "pick one climate"),
    (r"\binvisible\b|\btransparent\b", r"\bvisible\b|\bclearly\b|\bdetailed\b", "visibility", "object cannot be both hidden and prominent"),
)


def _find_match(text: str, pattern: str) -> str | None:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return m.group(0) if m else None


class ContradictionScanner:
    """Scan prompt text for high-confidence logical conflicts."""

    def __init__(self, extra_rules: Sequence[Tuple[str, str, str, str]] = ()) -> None:
        self._rules = tuple(_CONFLICT_RULES) + tuple(extra_rules)

    def scan(self, prompt: str) -> List[Contradiction]:
        text = (prompt or "").strip()
        if not text:
            return []

        found: List[Contradiction] = []
        for pat_a, pat_b, category, resolution in self._rules:
            a = _find_match(text, pat_a)
            b = _find_match(text, pat_b)
            if a and b:
                # Longer / more specific phrases → higher severity
                severity = min(1.0, 0.55 + 0.1 * (len(a) + len(b)) / 20.0)
                found.append(
                    Contradiction(
                        left=a,
                        right=b,
                        severity=severity,
                        category=category,
                        resolution=resolution,
                    )
                )
        return sorted(found, key=lambda c: -c.severity)

    def max_severity(self, prompt: str) -> float:
        hits = self.scan(prompt)
        return hits[0].severity if hits else 0.0

    def suggest_rewrite(self, prompt: str, *, pick: str = "left") -> str:
        """
        Naive rewrite: drop the weaker side of each contradiction.

        ``pick`` is ``left`` | ``right`` — which fragment family to keep.
        """
        text = prompt
        for c in self.scan(prompt):
            drop = c.right if pick == "left" else c.left
            text = re.sub(re.escape(drop), "", text, count=1, flags=re.IGNORECASE)
        return re.sub(r"\s{2,}", " ", text).strip(" ,")

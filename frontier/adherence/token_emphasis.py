"""
Which prompt tokens deserve extra CFG / regional weight.

Hard words: text, hands, logos, counts — ahead of generic prompt adherence heatmaps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class TokenWeight:
    token: str
    weight: float  # 1.0 = default; >1 = emphasize
    reason: str


@dataclass(frozen=True)
class TokenEmphasisMap:
    weights: Tuple[TokenWeight, ...]
    cfg_multiplier: float

    def as_prompt_weights(self) -> Dict[str, float]:
        return {w.token: w.weight for w in self.weights}


_HARD_TOKENS: Tuple[Tuple[str, float, str], ...] = (
    (r"\btext\b|\btypography\b|\blogo\b|\bsign\b", 1.35, "text rendering"),
    (r"\bhands?\b|\bfingers?\b", 1.25, "anatomy"),
    (r"\beyes?\b|\bpupils?\b", 1.15, "face detail"),
    (r"\breflection\b|\bmirror\b", 1.2, "symmetry"),
    (r"\bthree\b|\bfour\b|\bfive\b|\btwo\b", 1.3, "counting"),
    (r"\bexactly\b|\bprecisely\b", 1.2, "precision"),
)


class TokenEmphasisPlanner:
    def plan(self, prompt: str, *, base_cfg: float = 7.5) -> TokenEmphasisMap:
        text = prompt or ""
        weights: List[TokenWeight] = []
        boost = 0.0
        for pattern, w, reason in _HARD_TOKENS:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                weights.append(TokenWeight(token=m.group(0), weight=w, reason=reason))
                boost = max(boost, (w - 1.0) * 0.5)
        cfg_mult = 1.0 + boost
        return TokenEmphasisMap(weights=tuple(weights), cfg_multiplier=cfg_mult)

    def augment_with_weights(self, prompt: str, plan: TokenEmphasisMap) -> str:
        """SD-style (word:1.2) fragments for supported pipelines."""
        if not plan.weights:
            return prompt
        parts = [prompt] if prompt else []
        for w in plan.weights[:4]:
            if w.weight > 1.05:
                parts.append(f"({w.token}:{w.weight:.2f})")
        return ", ".join(p for p in parts if p)


__all__ = ["TokenEmphasisMap", "TokenEmphasisPlanner", "TokenWeight"]

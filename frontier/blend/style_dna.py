"""
Style DNA — keyword vectors for mixing aesthetics in prompt space.

Cheap alternative to multi-LoRA: interpolate curated style profiles before sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class StyleProfile:
    id: str
    positive: str
    negative: str = ""
    weight: float = 1.0


@dataclass(frozen=True)
class StyleDNA:
    blended_positive: str
    blended_negative: str
    components: Tuple[Tuple[str, float], ...]


_BUILTIN: Dict[str, StyleProfile] = {
    "editorial": StyleProfile("editorial", "magazine cover, bold typography, clean layout", "amateur snapshot"),
    "noir": StyleProfile("noir", "film noir, high contrast, venetian blind shadows", "flat lighting, oversaturated"),
    "studio": StyleProfile("studio", "studio softbox, seamless backdrop, catalog lighting", "harsh flash, clutter"),
    "painterly": StyleProfile("painterly", "oil paint strokes, canvas texture, impasto", "digital sharpness, plastic"),
    "anime_cel": StyleProfile("anime_cel", "cel shading, clean lineart, flat color blocks", "photoreal skin pores"),
    "brutalist": StyleProfile("brutalist", "concrete geometry, stark shadows, minimal palette", "decorative, ornate"),
}


class StyleDNABlender:
    def __init__(self, profiles: Dict[str, StyleProfile] | None = None) -> None:
        self.profiles = dict(profiles or _BUILTIN)

    def blend(self, weights: Sequence[Tuple[str, float]]) -> StyleDNA:
        pos_parts: List[str] = []
        neg_parts: List[str] = []
        comps: List[Tuple[str, float]] = []
        for sid, w in weights:
            prof = self.profiles.get(sid)
            if prof is None or w <= 0:
                continue
            comps.append((sid, float(w)))
            if prof.positive:
                pos_parts.append(f"({prof.positive}:{w:.2f})")
            if prof.negative:
                neg_parts.append(prof.negative)
        return StyleDNA(
            blended_positive=", ".join(pos_parts),
            blended_negative=", ".join(dict.fromkeys(neg_parts)),
            components=tuple(comps),
        )

    def from_prompt_keywords(self, prompt: str) -> StyleDNA | None:
        text = (prompt or "").lower()
        hits: List[Tuple[str, float]] = []
        for sid, prof in self.profiles.items():
            if sid.strip().lower() in text or prof.id in text:
                hits.append((sid, 1.0))
        if not hits:
            return None
        return self.blend(hits)


__all__ = ["StyleDNA", "StyleDNABlender", "StyleProfile"]

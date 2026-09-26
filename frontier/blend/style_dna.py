"""
Style DNA — keyword vectors for mixing aesthetics in prompt space.

Cheap alternative to multi-LoRA: interpolate curated style profiles before sampling.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


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
    components: tuple[tuple[str, float], ...]


_BUILTIN: dict[str, StyleProfile] = {
    "editorial": StyleProfile("editorial", "magazine cover, bold typography, clean layout", "amateur snapshot"),
    "noir": StyleProfile("noir", "film noir, high contrast, venetian blind shadows", "flat lighting, oversaturated"),
    "studio": StyleProfile("studio", "studio softbox, seamless backdrop, catalog lighting", "harsh flash, clutter"),
    "painterly": StyleProfile("painterly", "oil paint strokes, canvas texture, impasto", "digital sharpness, plastic"),
    "anime_cel": StyleProfile("anime_cel", "cel shading, clean lineart, flat color blocks", "photoreal skin pores"),
    "brutalist": StyleProfile("brutalist", "concrete geometry, stark shadows, minimal palette", "decorative, ornate"),
}


class StyleDNABlender:
    def __init__(self, profiles: dict[str, StyleProfile] | None = None) -> None:
        self.profiles = dict(profiles or _BUILTIN)

    def blend(self, weights: Sequence[tuple[str, float]]) -> StyleDNA:
        pos_parts: list[str] = []
        neg_parts: list[str] = []
        comps: list[tuple[str, float]] = []
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
        hits: list[tuple[str, float]] = []
        for sid, prof in self.profiles.items():
            if sid.strip().lower() in text or prof.id in text:
                hits.append((sid, 1.0))
        if not hits:
            return None
        return self.blend(hits)


__all__ = ["StyleDNA", "StyleDNABlender", "StyleProfile"]

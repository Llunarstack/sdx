"""Chromatic Field — palette bias for still/keyframe diffusion (pairs with video chromatic_arc)."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ChromaticFieldPlan", "chromatic_field_for_palette"]

_PALETTE_BIAS: dict[str, tuple[str, str]] = {
    "hope": ("warm golden key, soft green fill", "muddy grey cast"),
    "dread": ("sickly yellow-green cast, violet shadows", "cheerful saturation"),
    "rage": ("crimson accent highlights, crushed blacks", "pastel softness"),
    "grief": ("desaturated blue-grey", "vibrant neon"),
    "wonder": ("prismatic spectral highlights", "flat dull lighting"),
    "nostalgia": ("lifted blacks, amber midtones", "clinical digital sharpness"),
}


@dataclass(frozen=True)
class ChromaticFieldPlan:
    palette: str
    positive: str
    negative: str


def chromatic_field_for_palette(palette_key: str) -> ChromaticFieldPlan:
    key = (palette_key or "neutral").lower()
    pos, neg = _PALETTE_BIAS.get(key, (f"{key} color grading", "wrong color temperature"))
    return ChromaticFieldPlan(palette=key, positive=pos, negative=neg)

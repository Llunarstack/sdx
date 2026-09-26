"""
Style **motion grammar** — competitors use one generic motion prior for everything.

Film, anime, product turntables, VFX, and live-action need different temporal
rules. This module maps style engines → concrete ProcessOptions + sample prompt
fragments + post filters so the same backbone doesn't make anime look like
mushy live-action (and vice versa).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "MotionGrammar",
    "grammar_for_engine",
    "grammar_for_prompt",
    "apply_motion_grammar",
]


@dataclass(frozen=True, slots=True)
class MotionGrammar:
    name: str
    # Temporal feel
    keyframe_interval: int = 6
    edit_strength: float = 0.55
    temporal_alpha: float = 0.10
    temporal_smooth: int = 2
    deflicker_strength: float = 0.75
    motion_beat_keyframes: bool = False
    depth_interpolate: bool = False
    velocity_ease: bool = False
    velocity_ease_mode: str = "smooth"
    # Sample prompt garnish
    positive: tuple[str, ...] = ()
    negative: tuple[str, ...] = ()
    # Permanence / artifact thresholds (stricter for product/film)
    min_permanence: float = 0.45
    min_artifact: float = 0.40
    permanence_repair: bool = True
    notes: str = ""


_GRAMMARS: dict[str, MotionGrammar] = {
    "realistic": MotionGrammar(
        name="realistic",
        keyframe_interval=6,
        edit_strength=0.52,
        temporal_alpha=0.11,
        deflicker_strength=0.82,
        depth_interpolate=True,
        motion_beat_keyframes=True,
        positive=("natural motion blur", "24fps cinematic shutter", "grounded physics"),
        negative=("anime smear frames", "stop-motion jitter", "floating camera"),
        min_permanence=0.55,
        min_artifact=0.50,
        notes="Live-action: moderate KF spacing, strong deflicker, permanence gate",
    ),
    "film": MotionGrammar(
        name="film",
        keyframe_interval=5,
        edit_strength=0.48,
        temporal_alpha=0.12,
        deflicker_strength=0.85,
        depth_interpolate=True,
        velocity_ease=True,
        velocity_ease_mode="ease_in_out",
        positive=("anamorphic bokeh", "film gate weave subtle", "motivated camera move", "24fps"),
        negative=("digital soap-opera effect", "hyper-smooth gimbal", "video-game camera"),
        min_permanence=0.60,
        min_artifact=0.55,
        notes="Narrative film grammar",
    ),
    "anime_2d": MotionGrammar(
        name="anime_2d",
        keyframe_interval=3,
        edit_strength=0.42,
        temporal_alpha=0.16,
        temporal_smooth=1,
        deflicker_strength=0.55,
        motion_beat_keyframes=True,
        velocity_ease=True,
        velocity_ease_mode="hold",
        positive=("limited animation holds", "cel smear frames", "snappy pose-to-pose", "on twos"),
        negative=("photoreal pores", "continuous motion blur", "live-action timing"),
        min_permanence=0.40,
        min_artifact=0.35,
        permanence_repair=False,  # holds are intentional "pops"
        notes="Anime: denser keyframes, weaker deflicker (keep chatter), allow holds",
    ),
    "product": MotionGrammar(
        name="product",
        keyframe_interval=8,
        edit_strength=0.35,
        temporal_alpha=0.08,
        deflicker_strength=0.90,
        depth_interpolate=False,
        velocity_ease=True,
        velocity_ease_mode="linear",
        positive=("turntable orbit", "locked product silhouette", "studio softbox", "SKU stable"),
        negative=("morphing logo", "warping label text", "hand model distractions"),
        min_permanence=0.75,
        min_artifact=0.65,
        notes="Product: slow orbit, ruthless permanence (logo must not melt)",
    ),
    "cartoon": MotionGrammar(
        name="cartoon",
        keyframe_interval=4,
        edit_strength=0.45,
        temporal_alpha=0.14,
        deflicker_strength=0.60,
        motion_beat_keyframes=True,
        positive=("squash and stretch", "exaggerated arcs", "cartoon timing"),
        negative=("photoreal skin", "uncanny faces"),
        min_permanence=0.35,
        permanence_repair=False,
        notes="Western cartoon exaggeration",
    ),
    "vfx": MotionGrammar(
        name="vfx",
        keyframe_interval=5,
        edit_strength=0.50,
        temporal_alpha=0.10,
        deflicker_strength=0.80,
        depth_interpolate=True,
        motion_beat_keyframes=True,
        positive=("practical + digital composite", "contact shadows", "consistent FX plate"),
        negative=("element popping", "unmotivated glow flicker", "scale drift"),
        min_permanence=0.65,
        min_artifact=0.55,
        notes="VFX plates: permanence on hero elements, allow FX birth/death via mask later",
    ),
    "pixar_3d": MotionGrammar(
        name="pixar_3d",
        keyframe_interval=5,
        edit_strength=0.50,
        temporal_alpha=0.11,
        deflicker_strength=0.78,
        depth_interpolate=True,
        positive=("stylized 3d animation", "appealing arcs", "soft contact"),
        negative=("live-action shake", "uncanny photoreal"),
        min_permanence=0.55,
        notes="Stylized 3D feature animation",
    ),
}


def grammar_for_engine(engine_id: str) -> MotionGrammar:
    key = (engine_id or "").strip().lower().replace("-", "_")
    aliases = {
        "live_action": "realistic",
        "realistic": "realistic",
        "anime": "anime_2d",
        "anime_2d": "anime_2d",
        "pixar_3d": "pixar_3d",
        "3d": "pixar_3d",
        "product": "product",
        "cartoon": "cartoon",
        "pixel_art": "cartoon",
        "spider_verse": "cartoon",
        "vfx": "vfx",
        "film": "film",
        "cinematic": "film",
    }
    name = aliases.get(key, key if key in _GRAMMARS else "realistic")
    return _GRAMMARS.get(name, _GRAMMARS["realistic"])


def grammar_for_prompt(prompt: str, *, style_hint: str = "") -> MotionGrammar:
    text = f"{prompt} {style_hint}".lower()
    rules: list[tuple[tuple[str, ...], str]] = [
        (("product shot", "turntable", "sku", "packshot", "ecommerce", "catalog"), "product"),
        (("anime", "manga", "cel shaded", "ghibli", "waifu"), "anime_2d"),
        (("cartoon", "toon", "comic book", "spider-verse"), "cartoon"),
        (("vfx", "explosion", "simulation", "houdini", "particle"), "vfx"),
        (("35mm", "imax", "anamorphic", "feature film", "cinematic film"), "film"),
        (("pixar", "dreamworks", "stylized 3d"), "pixar_3d"),
        (("photoreal", "live action", "documentary", "handheld"), "realistic"),
    ]
    for keys, name in rules:
        if any(k in text for k in keys):
            return _GRAMMARS[name]
    return _GRAMMARS["realistic"]


def apply_motion_grammar(opts: Any, grammar: MotionGrammar, *, prompt: str = "") -> Any:
    """Soft-apply grammar onto ProcessOptions + sample prompt fragments."""
    from dataclasses import replace as _r

    # Append motion positive/negative into args if we can reach plan later — stash on opts
    opts = _r(
        opts,
        keyframe_interval=min(int(getattr(opts, "keyframe_interval", 6) or 6), grammar.keyframe_interval)
        if grammar.keyframe_interval < int(getattr(opts, "keyframe_interval", 6) or 6)
        else grammar.keyframe_interval,
        edit_strength=grammar.edit_strength,
        temporal_alpha=grammar.temporal_alpha,
        temporal_smooth=grammar.temporal_smooth,
        deflicker_strength=max(float(getattr(opts, "deflicker_strength", 0.75) or 0.75), grammar.deflicker_strength)
        if grammar.name in ("realistic", "film", "product", "vfx")
        else grammar.deflicker_strength,
        motion_beat_keyframes=bool(getattr(opts, "motion_beat_keyframes", False)) or grammar.motion_beat_keyframes,
        depth_interpolate=bool(getattr(opts, "depth_interpolate", False)) or grammar.depth_interpolate,
        velocity_ease=bool(getattr(opts, "velocity_ease", False)) or grammar.velocity_ease,
        velocity_ease_mode=grammar.velocity_ease_mode
        if grammar.velocity_ease
        else getattr(opts, "velocity_ease_mode", "smooth"),
        permanence_repair=grammar.permanence_repair,
        min_permanence=grammar.min_permanence,
        min_artifact=grammar.min_artifact,
        motion_grammar=grammar.name,
        glyph_lock=bool(getattr(opts, "glyph_lock", False)) or grammar.name == "product",
        min_glyph=0.55 if grammar.name == "product" else getattr(opts, "min_glyph", 0.0),
        min_identity=max(float(getattr(opts, "min_identity", 0.50) or 0.50), 0.60)
        if grammar.name in ("realistic", "film", "product", "vfx")
        else getattr(opts, "min_identity", 0.50),
        extremity_lock=bool(getattr(opts, "extremity_lock", True)) and grammar.name not in ("product",),
        physics_gate=bool(getattr(opts, "physics_gate", True))
        and grammar.name
        in (
            "realistic",
            "film",
            "vfx",
            "product",
            "pixar_3d",
        ),
        motion_shutter=bool(getattr(opts, "motion_shutter", False)) or grammar.name in ("film", "realistic"),
        contact_ground=bool(getattr(opts, "contact_ground", True)) and grammar.name not in ("anime_2d", "cartoon"),
        hf_deshimmer=bool(getattr(opts, "hf_deshimmer", True)),
        count_bind=bool(getattr(opts, "count_bind", True)),
    )
    # Stash prompt fragments for segment_processor
    pos = ", ".join(grammar.positive)
    neg = ", ".join(grammar.negative)
    opts = _r(opts, motion_grammar_positive=pos, motion_grammar_negative=neg)
    return opts

"""
Per-medium temporal rules for the video pipeline.

Live action, anime, stop-motion, pixel art, product turntables and VFX plates need different keyframe
density, deflicker strength, temporal smoothing and quality gates. Weak deflicker keeps intended line
chatter and stop-motion jitter; pixel art must not blend across its grid; product shots need ruthless
permanence so logos do not melt. ``grammar_for_prompt`` picks the medium from prompt cues and
``apply_motion_grammar`` writes its dials onto ``ProcessOptions``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

__all__ = [
    "MotionGrammar",
    "apply_motion_grammar",
    "grammar_for_engine",
    "grammar_for_prompt",
]


@dataclass(frozen=True, slots=True)
class MotionGrammar:
    name: str
    cues: tuple[str, ...] = ()
    keyframe_interval: int = 6
    edit_strength: float = 0.55
    temporal_alpha: float = 0.10
    temporal_smooth: int = 2
    deflicker_strength: float = 0.75
    motion_beat_keyframes: bool = False
    depth_interpolate: bool = False
    velocity_ease: bool = False
    velocity_ease_mode: str = "smooth"
    post_grade: str = ""
    min_permanence: float = 0.45
    min_artifact: float = 0.40
    min_identity: float = 0.0
    min_physics: float = 0.0
    min_glyph: float = 0.0


# Cue matching walks this order: specific media before generic ones.
_GRAMMARS: dict[str, MotionGrammar] = {
    g.name: g
    for g in (
        MotionGrammar(
            name="product",
            cues=("product shot", "turntable", "sku", "packshot", "ecommerce", "catalog", "logo reveal"),
            keyframe_interval=8,
            edit_strength=0.35,
            temporal_alpha=0.08,
            deflicker_strength=0.90,
            velocity_ease=True,
            velocity_ease_mode="linear",
            min_permanence=0.75,
            min_artifact=0.65,
            min_identity=0.55,
            min_glyph=0.60,
        ),
        MotionGrammar(
            name="stop_motion",
            cues=(
                "stop motion",
                "stop-motion",
                "claymation",
                "clay",
                "wallace",
                "puppet animation",
                "lego",
                "minifig",
                "brickfilm",
                "brick film",
            ),
            keyframe_interval=4,
            edit_strength=0.45,
            temporal_alpha=0.14,
            temporal_smooth=1,
            deflicker_strength=0.35,
            velocity_ease=True,
            velocity_ease_mode="hold",
            post_grade="muted",
            min_permanence=0.45,
            min_artifact=0.35,
        ),
        MotionGrammar(
            name="pixel_art",
            cues=("pixel art", "8-bit", "16-bit", "retro game", "pixel animation", "minecraft", "voxel", "blocky"),
            keyframe_interval=4,
            edit_strength=0.40,
            temporal_alpha=0.05,
            temporal_smooth=0,
            deflicker_strength=0.25,
            velocity_ease=True,
            velocity_ease_mode="hold",
            min_permanence=0.40,
            min_artifact=0.35,
        ),
        MotionGrammar(
            name="spider_verse",
            cues=("spider-verse", "spider verse", "halftone", "comic frame rate"),
            keyframe_interval=3,
            edit_strength=0.45,
            temporal_alpha=0.16,
            temporal_smooth=1,
            deflicker_strength=0.45,
            motion_beat_keyframes=True,
            velocity_ease=True,
            velocity_ease_mode="hold",
            post_grade="vibrant",
            min_permanence=0.28,
            min_artifact=0.30,
        ),
        MotionGrammar(
            name="low_poly",
            cues=("low poly", "ps1", "n64", "playstation", "affine texture"),
            keyframe_interval=6,
            edit_strength=0.48,
            temporal_alpha=0.12,
            deflicker_strength=0.55,
            post_grade="muted",
            min_permanence=0.45,
            min_artifact=0.40,
        ),
        MotionGrammar(
            name="vector",
            cues=("vector", "motion graphics", "flat design", "after effects style"),
            keyframe_interval=6,
            edit_strength=0.40,
            temporal_alpha=0.10,
            deflicker_strength=0.70,
            velocity_ease=True,
            velocity_ease_mode="ease_in_out",
            min_permanence=0.55,
            min_artifact=0.50,
            min_glyph=0.50,
        ),
        MotionGrammar(
            name="dream_logic",
            cues=(
                "dreamlike",
                "dreamscape",
                "dream logic",
                "surreal",
                "impossible architecture",
                "salvador",
                "morphing reality",
            ),
            keyframe_interval=6,
            edit_strength=0.60,
            temporal_alpha=0.10,
            deflicker_strength=0.60,
            velocity_ease=True,
            min_permanence=0.25,
            min_artifact=0.35,
        ),
        MotionGrammar(
            name="vfx",
            cues=("vfx", "explosion", "houdini", "particle", "simulation", "compositing", "fx plate", "volumetric"),
            keyframe_interval=5,
            edit_strength=0.50,
            temporal_alpha=0.10,
            deflicker_strength=0.80,
            motion_beat_keyframes=True,
            depth_interpolate=True,
            post_grade="cinematic",
            min_permanence=0.65,
            min_artifact=0.55,
            min_physics=0.52,
        ),
        MotionGrammar(
            name="cgi",
            cues=("cgi", "lookdev", "octane", "redshift", "unreal engine", "pbr render", "production render"),
            keyframe_interval=5,
            edit_strength=0.50,
            temporal_alpha=0.10,
            deflicker_strength=0.84,
            motion_beat_keyframes=True,
            depth_interpolate=True,
            post_grade="cinematic",
            min_permanence=0.65,
            min_artifact=0.55,
            min_identity=0.58,
            min_physics=0.50,
        ),
        MotionGrammar(
            name="pixar_3d",
            cues=("pixar", "dreamworks", "stylized 3d", "3d animated film", "cgi character"),
            keyframe_interval=5,
            edit_strength=0.50,
            temporal_alpha=0.11,
            deflicker_strength=0.78,
            depth_interpolate=True,
            post_grade="cinematic",
            min_permanence=0.55,
            min_identity=0.58,
            min_physics=0.42,
        ),
        MotionGrammar(
            name="anime_2d",
            cues=("anime", "manga", "cel shaded", "ghibli", "sakuga", "key animation"),
            keyframe_interval=3,
            edit_strength=0.42,
            temporal_alpha=0.16,
            temporal_smooth=1,
            deflicker_strength=0.55,
            motion_beat_keyframes=True,
            velocity_ease=True,
            velocity_ease_mode="hold",
            post_grade="vibrant",
            min_permanence=0.40,
            min_artifact=0.35,
            min_identity=0.52,
        ),
        MotionGrammar(
            name="cartoon",
            cues=("cartoon", "toon", "looney", "rubber hose", "western animation", "comic book"),
            keyframe_interval=4,
            edit_strength=0.45,
            temporal_alpha=0.14,
            deflicker_strength=0.60,
            motion_beat_keyframes=True,
            post_grade="vibrant",
            min_permanence=0.35,
        ),
        MotionGrammar(
            name="film",
            cues=("feature film", "anamorphic", "imax", "35mm", "cinematic film", "narrative film"),
            keyframe_interval=5,
            edit_strength=0.48,
            temporal_alpha=0.12,
            deflicker_strength=0.85,
            depth_interpolate=True,
            velocity_ease=True,
            velocity_ease_mode="ease_in_out",
            post_grade="cinematic",
            min_permanence=0.60,
            min_artifact=0.55,
            min_identity=0.62,
        ),
        MotionGrammar(
            name="hybrid",
            cues=("hybrid", "mixed media", "live action anime", "rotoscope"),
            keyframe_interval=5,
            edit_strength=0.50,
            temporal_alpha=0.12,
            deflicker_strength=0.75,
            motion_beat_keyframes=True,
            depth_interpolate=True,
            post_grade="cinematic",
            min_permanence=0.50,
            min_artifact=0.45,
            min_identity=0.55,
        ),
        MotionGrammar(
            name="realistic",
            cues=("photoreal", "live action", "documentary", "handheld", "real footage"),
            keyframe_interval=6,
            edit_strength=0.52,
            temporal_alpha=0.11,
            deflicker_strength=0.82,
            motion_beat_keyframes=True,
            depth_interpolate=True,
            post_grade="cinematic",
            min_permanence=0.55,
            min_artifact=0.50,
            min_identity=0.60,
            min_physics=0.50,
        ),
    )
}

_ALIASES = {
    "live_action": "realistic",
    "cinematic": "film",
    "anime": "anime_2d",
    "3d": "pixar_3d",
    "3d_cgi": "cgi",
    "cgi_3d": "cgi",
    "claymation": "stop_motion",
    "lego": "stop_motion",
    "voxel": "pixel_art",
}


def grammar_for_engine(name: str) -> MotionGrammar:
    """Grammar by name or style-engine id; unknown names fall back to ``realistic``."""
    key = (name or "").strip().lower().replace("-", "_")
    return _GRAMMARS.get(_ALIASES.get(key, key), _GRAMMARS["realistic"])


def grammar_for_prompt(prompt: str, *, style_hint: str = "") -> MotionGrammar:
    text = f"{prompt} {style_hint}".lower()
    for grammar in _GRAMMARS.values():
        if any(c in text for c in grammar.cues):
            return grammar
    return _GRAMMARS["realistic"]


def apply_motion_grammar(opts: Any, grammar: MotionGrammar) -> Any:
    """Write the grammar's temporal dials and gate floors onto ``ProcessOptions``; a user ``post_grade`` wins."""
    return replace(
        opts,
        motion_grammar=grammar.name,
        keyframe_interval=grammar.keyframe_interval,
        edit_strength=grammar.edit_strength,
        temporal_alpha=grammar.temporal_alpha,
        temporal_smooth=grammar.temporal_smooth,
        deflicker_strength=grammar.deflicker_strength,
        motion_beat_keyframes=opts.motion_beat_keyframes or grammar.motion_beat_keyframes,
        depth_interpolate=opts.depth_interpolate or grammar.depth_interpolate,
        velocity_ease=opts.velocity_ease or grammar.velocity_ease,
        velocity_ease_mode=grammar.velocity_ease_mode if grammar.velocity_ease else opts.velocity_ease_mode,
        post_grade=opts.post_grade or grammar.post_grade,
        min_permanence=grammar.min_permanence,
        min_artifact=grammar.min_artifact,
        min_identity=grammar.min_identity,
        min_physics=grammar.min_physics,
        min_glyph=grammar.min_glyph,
    )

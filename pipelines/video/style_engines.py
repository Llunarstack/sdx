"""Style-specific generation engines (realistic, anime, 3D, voxel, etc.)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

__all__ = [
    "EnginePreset",
    "RenderEngine",
    "engine_by_id",
    "list_engines",
    "match_engine_from_prompt",
]


class RenderEngine(str, Enum):
    REALISTIC = "realistic"
    ANIME_2D = "anime_2d"
    PIXAR_3D = "pixar_3d"
    LOW_POLY = "low_poly"
    VOXEL = "voxel"
    LEGO = "lego"
    CLAYMATION = "claymation"
    VECTOR = "vector"
    PIXEL_ART = "pixel_art"
    SPIDER_VERSE = "spider_verse"
    HYBRID = "hybrid"
    DREAM_LOGIC = "dream_logic"


@dataclass(slots=True)
class EnginePreset:
    id: RenderEngine
    title: str
    positive: str
    negative: str
    post_grade: str = ""
    keyframe_interval: int = 6
    edit_strength: float = 0.55
    depth_interpolate: bool = False
    region_motion: bool = True
    motion_transfer: bool = True
    temporal_alpha: float = 0.10
    physics_hint: str = ""
    pipeline_notes: str = ""
    retrieval_tags: List[str] = field(default_factory=list)
    animation_preset: str = ""
    layers: List[str] = field(default_factory=list)


_ENGINES: Dict[RenderEngine, EnginePreset] = {
    RenderEngine.REALISTIC: EnginePreset(
        id=RenderEngine.REALISTIC,
        title="Live-action cinematic",
        positive="photorealistic, natural skin, real lighting, cinematic depth of field",
        negative="cartoon, anime linework, plastic CGI, uncanny valley",
        post_grade="cinematic",
        physics_hint="real-world gravity, cloth weight, foot contact",
        retrieval_tags=["live action", "cinematic", "realistic"],
        layers=["background", "character", "fx", "grade"],
    ),
    RenderEngine.ANIME_2D: EnginePreset(
        id=RenderEngine.ANIME_2D,
        title="2D anime / cel animation",
        positive="anime cel shading, clean lineart, expressive eyes, limited frames feel",
        negative="photoreal pores, muddy gradients, western comic shading",
        post_grade="vibrant",
        keyframe_interval=4,
        edit_strength=0.48,
        temporal_alpha=0.14,
        physics_hint="anime exaggeration, speed lines, held poses",
        animation_preset="anime_tv",
        retrieval_tags=["anime", "cel", "2d animation"],
        layers=["lineart", "flat_color", "fx", "background"],
    ),
    RenderEngine.PIXAR_3D: EnginePreset(
        id=RenderEngine.PIXAR_3D,
        title="3D stylized (Pixar/DreamWorks)",
        positive="3D animated film, subsurface skin, stylized proportions, soft global illumination",
        negative="flat 2D, photoreal news footage, low poly untextured",
        post_grade="cinematic",
        edit_strength=0.52,
        animation_preset="pixar",
        physics_hint="squash stretch, follow through, appeal",
        pipeline_notes="future: Blender rig → render → SDX beautify pass",
        layers=["geometry", "character", "lighting", "fx"],
    ),
    RenderEngine.LOW_POLY: EnginePreset(
        id=RenderEngine.LOW_POLY,
        title="Low-poly / PS1 aesthetic",
        positive="low poly 3D, affine texture warping, vertex jitter, retro game cutscene",
        negative="smooth HD, raytraced reflections, film grain realism",
        post_grade="muted",
        keyframe_interval=8,
        physics_hint="snappy transforms, limited blend shapes",
        retrieval_tags=["ps1", "n64", "low poly"],
    ),
    RenderEngine.VOXEL: EnginePreset(
        id=RenderEngine.VOXEL,
        title="Voxel / Minecraft-style",
        positive="voxel world, blocky characters, cubic clouds, minecraft aesthetic",
        negative="smooth curves, realistic anatomy, film lighting",
        physics_hint="block grid snapping, cube particles",
        retrieval_tags=["minecraft", "voxel", "blocky"],
    ),
    RenderEngine.LEGO: EnginePreset(
        id=RenderEngine.LEGO,
        title="LEGO brick stop-motion",
        positive="LEGO minifigure, studded bricks, practical stop-motion timing",
        negative="organic skin, smooth CGI humans",
        animation_preset="stop_motion",
        physics_hint="brick connection constraints, stiff joints",
    ),
    RenderEngine.CLAYMATION: EnginePreset(
        id=RenderEngine.CLAYMATION,
        title="Claymation / stop-motion",
        positive="clay fingerprints, miniature sets, uneven stop-motion cadence, practical lighting",
        negative="fluid CGI motion, perfect symmetry",
        animation_preset="stop_motion",
        temporal_alpha=0.18,
        physics_hint="imperfect steps, thumb marks, micro jitter",
    ),
    RenderEngine.VECTOR: EnginePreset(
        id=RenderEngine.VECTOR,
        title="Vector / motion graphics",
        positive="flat vector shapes, bold graphic design, clean curves, motion graphics",
        negative="photoreal texture, noisy grain",
        keyframe_interval=5,
    ),
    RenderEngine.PIXEL_ART: EnginePreset(
        id=RenderEngine.PIXEL_ART,
        title="Pixel art",
        positive="pixel art, limited palette, crisp pixels, 16-bit game aesthetic",
        negative="anti-aliased smooth gradients, HD photography",
        keyframe_interval=10,
        temporal_alpha=0.05,
    ),
    RenderEngine.SPIDER_VERSE: EnginePreset(
        id=RenderEngine.SPIDER_VERSE,
        title="Spider-Verse halftone",
        positive="comic halftone dots, misregistered CMYK, varied frame rates, ink outlines",
        negative="uniform framerate, clean corporate CGI",
        animation_preset="spider_verse",
    ),
    RenderEngine.HYBRID: EnginePreset(
        id=RenderEngine.HYBRID,
        title="Hybrid (per-layer routing)",
        positive="mixed media, layered styles",
        negative="",
        pipeline_notes="route characters vs background via scene layer_stack",
    ),
    RenderEngine.DREAM_LOGIC: EnginePreset(
        id=RenderEngine.DREAM_LOGIC,
        title="Dream / surreal logic",
        positive="surreal morphing, impossible architecture, dream logic, symbolic imagery",
        negative="documentary realism, rigid continuity",
        temporal_alpha=0.06,
        physics_hint="gravity optional, morphing allowed",
    ),
}


def list_engines() -> List[EnginePreset]:
    return list(_ENGINES.values())


def engine_by_id(name: str) -> Optional[EnginePreset]:
    key = (name or "").strip().lower().replace(" ", "_").replace("-", "_")
    for eng in RenderEngine:
        if eng.value == key or key in eng.value:
            return _ENGINES[eng]
    aliases = {
        "live_action": RenderEngine.REALISTIC,
        "live": RenderEngine.REALISTIC,
        "3d": RenderEngine.PIXAR_3D,
        "anime": RenderEngine.ANIME_2D,
        "minecraft": RenderEngine.VOXEL,
        "stop_motion": RenderEngine.CLAYMATION,
        "ps1": RenderEngine.LOW_POLY,
    }
    if key in aliases:
        return _ENGINES[aliases[key]]
    return None


def match_engine_from_prompt(prompt: str, *, style_hint: str = "") -> RenderEngine:
    text = f"{prompt} {style_hint}".lower()
    rules: List[tuple[tuple[str, ...], RenderEngine]] = [
        (("minecraft", "voxel", "blocky", "cube world"), RenderEngine.VOXEL),
        (("lego", "minifig", "brick"), RenderEngine.LEGO),
        (("claymation", "clay", "stop motion", "stop-motion", "wallace"), RenderEngine.CLAYMATION),
        (("pixel art", "8-bit", "16-bit", "retro game"), RenderEngine.PIXEL_ART),
        (("spider-verse", "spider verse", "halftone comic"), RenderEngine.SPIDER_VERSE),
        (("pixar", "dreamworks", "3d animated film", "stylized 3d"), RenderEngine.PIXAR_3D),
        (("anime", "ghibli", "cel shaded", "manga"), RenderEngine.ANIME_2D),
        (("ps1", "n64", "low poly", "playstation"), RenderEngine.LOW_POLY),
        (("vector", "motion graphics", "flat design"), RenderEngine.VECTOR),
        (("dream", "surreal", "salvador", "impossible"), RenderEngine.DREAM_LOGIC),
        (("photoreal", "live action", "cinematic film", "documentary"), RenderEngine.REALISTIC),
    ]
    for keys, eng in rules:
        if any(k in text for k in keys):
            return eng
    return RenderEngine.REALISTIC


def engine_edit_overrides(preset: EnginePreset) -> Dict[str, Any]:
    return {
        "post_grade": preset.post_grade,
        "keyframe_interval": preset.keyframe_interval,
        "edit_strength": preset.edit_strength,
        "depth_interpolate": preset.depth_interpolate,
        "region_motion": preset.region_motion,
        "motion_transfer": preset.motion_transfer,
        "temporal_alpha": preset.temporal_alpha,
    }

"""Compile studio block → routed engine, director storyboard, memory, rigs.

Also holds what used to be: animation_principles, camera_rig, character_memory, director_personalities, style_engines, generation_router, layer_stack, world_memory, storyboard, director_mode, thumbnail_rehearsal.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .motion_grammar import grammar_for_engine

# ==================================================================================================
# studio_compiler
# ==================================================================================================

# ``studio.rehearsal``: cheap animatic / low-poly passes before a full render.
_REHEARSAL_STAGES: dict[str, dict[str, Any]] = {
    "animatic": {
        "keyframe_interval": 12,
        "edit_strength": 0.35,
        "frame_enhance": False,
        "post_grade": "",
        "quality_retry": False,
    },
    "low_poly": {"keyframe_interval": 8, "edit_strength": 0.45, "frame_enhance": False, "post_grade": "muted"},
    "keyframe": {"keyframe_interval": 6, "quality_retry": True},
}


def compile_studio_block(
    data: Mapping[str, Any],
    *,
    scene_prompt: str,
    duration_sec: float,
    style_notes: str = "",
    existing_edit: Mapping[str, Any] | None = None,
) -> StudioCompileResult:
    from .motion import parse_motion_library, resolve_motion_clip

    studio = dict(data.get("studio") or {})
    edit = dict(existing_edit or {})
    prompt_suffix: list[str] = []
    neg_suffix: list[str] = []
    storyboard_cuts = None
    motion_clip = str(data.get("motion_clip") or studio.get("motion_clip") or "")

    # Engine router
    engine_name = str(studio.get("engine") or "auto")
    layer_stack = parse_layer_stack(studio.get("layers") or studio.get("layer_stack"))
    route = route_scene(
        scene_prompt,
        style_hint=style_notes,
        engine_override=engine_name,
        layer_stack=layer_stack.engines_by_layer() if layer_stack.layers else None,
    )
    for k, v in route.edit_overrides.items():
        if v and not edit.get(k):
            edit[k] = v
    prompt_suffix.append(route.style_positive)
    if route.style_negative:
        neg_suffix.append(route.style_negative)
    if route.retrieval_tags:
        retr = dict(data.get("retrieval") or {})
        retr.setdefault("tags", [])
        if isinstance(retr["tags"], list):
            retr["tags"] = list(dict.fromkeys(retr["tags"] + route.retrieval_tags))
        studio["_retrieval_boost"] = retr

    # Director mode
    director_mode = str(studio.get("director_mode") or "off").lower()
    if director_mode in ("auto", "on", "true"):
        genre = str(studio.get("genre") or "")
        exp = expand_prompt_to_storyboard(scene_prompt, duration_sec=duration_sec, genre_override=genre)
        storyboard_cuts = exp.cuts
        studio["director_notes"] = {
            "mood": exp.notes.mood,
            "lens": exp.notes.lens,
            "camera": exp.notes.camera,
            "color_grade": exp.notes.color_grade,
            "music": exp.notes.music,
            "pacing": exp.notes.pacing,
            "genre": exp.genre,
        }

    # Director personality
    d_id = str(studio.get("director") or studio.get("director_personality") or "")
    dp = director_by_id(d_id) if d_id else None
    if dp:
        prompt_suffix.append(dp.positive)
        neg_suffix.append(dp.negative)

    # World memory
    world = parse_world(data.get("world") or studio.get("world"))
    if world:
        prompt_suffix.append(world_to_prompt(world))
        neg_suffix.append(world_negative(world))

    # Character bibles
    bibles = parse_character_bibles(data.get("character_bibles") or studio.get("character_bibles") or {})
    for b in bibles.values():
        frag = bible_to_prompt(b)
        if frag:
            prompt_suffix.append(frag)
        neg_suffix.append(bible_negative(b))

    # Animation principles
    anim_raw = studio.get("animation_principles") or studio.get("animation")
    anim_preset = str(studio.get("animation_preset") or route.preset.animation_preset or "")
    principles = principles_from_dict(anim_raw if isinstance(anim_raw, Mapping) else {}, preset=anim_preset)
    anim_p = principles_prompt(principles)
    if anim_p:
        prompt_suffix.append(anim_p)

    # Camera rig
    rig = parse_camera_rig(studio.get("camera_rig") or studio.get("camera"))
    rig_p = rig_to_prompt(rig)
    if rig_p:
        prompt_suffix.append(rig_p)
    if rig.fps and rig.fps != 24.0:
        studio["_fps_override"] = rig.fps

    # Motion library
    motion_lib = parse_motion_library(data.get("motion_library") or studio.get("motion_library") or {})
    pack_ids = studio.get("motion_packs") or studio.get("motion") or []
    if isinstance(pack_ids, str):
        pack_ids = [pack_ids]
    resolved = resolve_motion_clip(motion_lib, pack_ids, fallback=motion_clip)
    if resolved:
        motion_clip = resolved

    # Layer stack prompts
    if layer_stack.layers:
        prompt_suffix.append(stack_prompt_suffix(layer_stack))

    # Rehearsal stage
    stage = str(studio.get("rehearsal") or studio.get("rehearsal_stage") or "full").lower()
    edit.update(_REHEARSAL_STAGES.get(stage, {}))

    # Thumbnail-first rehearsal

    thumb_cfg = parse_thumbnail_config(
        data.get("continuity"),
        studio=studio,
        edit=edit,
    )
    if thumb_cfg.enabled:
        edit.update(thumbnail_edit_overrides(thumb_cfg))
        studio["thumbnail_first"] = True
        studio["thumbnail_size"] = thumb_cfg.size
        studio["thumbnail_gate"] = thumb_cfg.gate

    return StudioCompileResult(
        edit=edit,
        prompt_suffix=", ".join(p for p in prompt_suffix if p),
        negative_suffix=", ".join(n for n in neg_suffix if n),
        storyboard_cuts=storyboard_cuts,
        engine=route.engine.value,
        motion_clip=motion_clip,
        studio_meta=dict(studio),
    )


class StudioCompileResult:
    __slots__ = ("edit", "prompt_suffix", "negative_suffix", "storyboard_cuts", "engine", "motion_clip", "studio_meta")

    def __init__(
        self,
        *,
        edit: dict[str, Any],
        prompt_suffix: str,
        negative_suffix: str,
        storyboard_cuts: Any,
        engine: str,
        motion_clip: str,
        studio_meta: dict[str, Any],
    ) -> None:
        self.edit = edit
        self.prompt_suffix = prompt_suffix
        self.negative_suffix = negative_suffix
        self.storyboard_cuts = storyboard_cuts
        self.engine = engine
        self.motion_clip = motion_clip
        self.studio_meta = studio_meta


# ==================================================================================================
# animation_principles: Disney 12 principles + style presets as tunable sliders.
# ==================================================================================================


@dataclass(slots=True)
class AnimationPrinciples:
    squash_stretch: float = 0.5
    anticipation: float = 0.5
    staging: float = 0.5
    straight_ahead_pose: float = 0.5
    follow_through: float = 0.5
    slow_in_slow_out: float = 0.5
    arc: float = 0.5
    secondary_action: float = 0.5
    timing: float = 0.5
    exaggeration: float = 0.5
    appeal: float = 0.5
    solid_drawing: float = 0.5


_PRESETS: dict[str, AnimationPrinciples] = {
    "pixar": AnimationPrinciples(
        squash_stretch=0.75, anticipation=0.7, follow_through=0.85, appeal=0.9, exaggeration=0.6
    ),
    "ghibli": AnimationPrinciples(
        squash_stretch=0.35, anticipation=0.5, follow_through=0.6, timing=0.4, appeal=0.85, exaggeration=0.3
    ),
    "anime_tv": AnimationPrinciples(squash_stretch=0.45, anticipation=0.55, exaggeration=0.7, timing=0.65, appeal=0.75),
    "looney": AnimationPrinciples(squash_stretch=0.95, anticipation=0.9, exaggeration=0.95, timing=0.85, appeal=0.8),
    "spider_verse": AnimationPrinciples(
        squash_stretch=0.6, exaggeration=0.8, timing=0.75, secondary_action=0.7, appeal=0.85
    ),
    "stop_motion": AnimationPrinciples(
        squash_stretch=0.25, timing=0.55, follow_through=0.35, exaggeration=0.2, solid_drawing=0.7
    ),
    "realistic": AnimationPrinciples(squash_stretch=0.15, exaggeration=0.1, appeal=0.5, follow_through=0.4),
}


def preset_principles(name: str) -> AnimationPrinciples:
    key = (name or "").strip().lower().replace(" ", "_").replace("-", "_")
    return _PRESETS.get(key, AnimationPrinciples())


def principles_from_dict(raw: Mapping[str, Any] | None, *, preset: str = "") -> AnimationPrinciples:
    base = preset_principles(preset) if preset else AnimationPrinciples()
    if not raw:
        return base
    fields = {f.name for f in AnimationPrinciples.__dataclass_fields__.values()}
    vals = {k: float(raw[k]) for k in raw if k in fields}
    return AnimationPrinciples(**{**base.__dict__, **vals})


def principles_prompt(p: AnimationPrinciples) -> str:
    parts: list[str] = []
    if p.squash_stretch > 0.6:
        parts.append("strong squash and stretch")
    elif p.squash_stretch < 0.3:
        parts.append("rigid form, minimal squash")
    if p.anticipation > 0.6:
        parts.append("clear anticipation poses")
    if p.follow_through > 0.65:
        parts.append("follow through and overlap on hair, cloth, limbs")
    if p.exaggeration > 0.7:
        parts.append("exaggerated poses and silhouettes")
    elif p.exaggeration < 0.25:
        parts.append("subtle restrained motion")
    if p.timing > 0.65:
        parts.append("snappy timing")
    elif p.timing < 0.35:
        parts.append("slow deliberate timing")
    if p.appeal > 0.75:
        parts.append("appealing readable silhouettes")
    if p.secondary_action > 0.6:
        parts.append("secondary motion on props and hair")
    return ", ".join(parts) if parts else "balanced animation timing"


# ==================================================================================================
# camera_rig: Virtual camera rig presets — lens, body, movement, fps.
# ==================================================================================================


@dataclass(slots=True)
class CameraRig:
    body: str = ""
    lens_mm: str = ""
    movement: str = ""
    focus: str = ""
    fps: float = 24.0
    aspect: str = ""
    filter: str = ""
    operator_style: str = ""


_camera_rig_PRESETS: dict[str, CameraRig] = {
    "arri_alexa": CameraRig(body="ARRI Alexa look", lens_mm="50", movement="steadicam", fps=24.0),
    "imax": CameraRig(body="IMAX large format", lens_mm="24", movement="slow crane", fps=24.0),
    "handheld_doc": CameraRig(
        body="documentary handheld", lens_mm="35", movement="handheld", operator_style="documentary"
    ),
    "drone": CameraRig(body="aerial drone", lens_mm="24", movement="orbit", fps=24.0),
    "gopro": CameraRig(body="GoPro wide", lens_mm="16", movement="body mount", fps=60.0),
    "vhs": CameraRig(body="VHS camcorder", lens_mm="wide", movement="handheld", filter="VHS", fps=29.97),
    "security": CameraRig(body="security CCTV", lens_mm="fixed wide", movement="static", fps=15.0),
    "anime_cam": CameraRig(
        body="anime dramatic camera", lens_mm="varied", movement="snap zoom", operator_style="anime"
    ),
}


def parse_camera_rig(raw: Any) -> CameraRig:
    if not raw:
        return CameraRig()
    if isinstance(raw, str):
        key = raw.strip().lower().replace(" ", "_")
        return _camera_rig_PRESETS.get(key, CameraRig(body=raw))
    if not isinstance(raw, Mapping):
        return CameraRig()
    preset = str(raw.get("preset") or raw.get("template") or "")
    base = _camera_rig_PRESETS.get(preset.lower(), CameraRig()) if preset else CameraRig()
    return CameraRig(
        body=str(raw.get("body") or base.body),
        lens_mm=str(raw.get("lens") or raw.get("lens_mm") or base.lens_mm),
        movement=str(raw.get("movement") or raw.get("camera_move") or base.movement),
        focus=str(raw.get("focus") or base.focus),
        fps=float(raw.get("fps") or base.fps or 24.0),
        aspect=str(raw.get("aspect") or base.aspect),
        filter=str(raw.get("filter") or base.filter),
        operator_style=str(raw.get("operator") or raw.get("style") or base.operator_style),
    )


def rig_to_prompt(r: CameraRig) -> str:
    parts = []
    if r.body:
        parts.append(r.body)
    if r.lens_mm:
        parts.append(f"{r.lens_mm}mm lens")
    if r.movement:
        parts.append(f"{r.movement} camera")
    if r.focus:
        parts.append(f"{r.focus} focus")
    if r.filter:
        parts.append(f"{r.filter} aesthetic")
    if r.operator_style:
        parts.append(f"{r.operator_style} camera operator")
    return ", ".join(parts)


# ==================================================================================================
# character_memory: Persistent character bible — identity across episodes and styles.
# ==================================================================================================


@dataclass(slots=True)
class CharacterBible:
    id: str
    name: str = ""
    hair: str = ""
    eyes: str = ""
    height: str = ""
    build: str = ""
    outfit: str = ""
    voice: str = ""
    walk_style: str = ""
    scars: str = ""
    age: str = ""
    emotion_baseline: str = ""
    reference_images: list[str] = field(default_factory=list)
    notes: str = ""


def parse_character_bibles(raw: Any) -> dict[str, CharacterBible]:
    out: dict[str, CharacterBible] = {}
    if not isinstance(raw, Mapping):
        return out
    for k, v in raw.items():
        cid = str(k)
        if isinstance(v, str):
            out[cid] = CharacterBible(id=cid, name=v)
            continue
        if not isinstance(v, Mapping):
            continue
        imgs = v.get("images") or v.get("refs") or []
        if isinstance(imgs, str):
            imgs = [imgs]
        out[cid] = CharacterBible(
            id=cid,
            name=str(v.get("name") or cid),
            hair=str(v.get("hair") or ""),
            eyes=str(v.get("eyes") or ""),
            height=str(v.get("height") or ""),
            build=str(v.get("build") or ""),
            outfit=str(v.get("outfit") or v.get("armor") or v.get("costume") or ""),
            voice=str(v.get("voice") or ""),
            walk_style=str(v.get("walk_style") or v.get("gait") or ""),
            scars=str(v.get("scars") or ""),
            age=str(v.get("age") or ""),
            emotion_baseline=str(v.get("emotion") or v.get("mood") or ""),
            reference_images=[str(x) for x in imgs],
            notes=str(v.get("notes") or v.get("description") or ""),
        )
    return out


def bible_to_prompt(b: CharacterBible) -> str:
    parts: list[str] = []
    if b.name:
        parts.append(f"character {b.name}")
    for label, val in (
        ("hair", b.hair),
        ("eyes", b.eyes),
        ("height", b.height),
        ("build", b.build),
        ("outfit", b.outfit),
        ("walk style", b.walk_style),
        ("age", b.age),
        ("scars", b.scars),
        ("baseline emotion", b.emotion_baseline),
    ):
        if val:
            parts.append(f"{label}: {val}")
    if b.notes:
        parts.append(b.notes)
    return ", ".join(parts)


def bible_negative(b: CharacterBible) -> str:
    base = "wrong face, duplicate head, morphing identity"
    if b.hair:
        base += f", wrong hair color, not {b.hair}"
    if b.eyes:
        base += ", wrong eye color"
    return base


# ==================================================================================================
# director_personalities: Director personality presets — Spielberg, Nolan, Miyazaki, etc.
# ==================================================================================================


@dataclass(frozen=True, slots=True)
class DirectorPersonality:
    id: str
    title: str
    positive: str
    negative: str
    pacing: str
    default_camera: str


_DIRECTORS: dict[str, DirectorPersonality] = {
    "spielberg": DirectorPersonality(
        "spielberg",
        "Spielberg",
        "wonder, backlight silhouettes, emotional blocking, wide wonder shots",
        "cynical flat staging",
        "measured wonder builds",
        "slow dolly inward",
    ),
    "nolan": DirectorPersonality(
        "nolan",
        "Nolan",
        "IMAX scale, practical weight, non-linear tension, architectural symmetry",
        "handheld chaos, cartoon physics",
        "deliberate escalation",
        "crane wide",
    ),
    "tarantino": DirectorPersonality(
        "tarantino",
        "Tarantino",
        "dialogue tension, trunk shots, bold color, long takes",
        "generic coverage",
        "dialogue-heavy holds",
        "low angle",
    ),
    "miyazaki": DirectorPersonality(
        "miyazaki",
        "Miyazaki",
        "quiet environmental beauty, gentle pacing, wind in grass, lived-in worlds",
        "aggressive shaky cam, harsh cuts",
        "contemplative",
        "pan across landscape",
    ),
    "anime_action": DirectorPersonality(
        "anime_action",
        "Anime action director",
        "speed lines, impact frames, dramatic angles, held reaction shots",
        "muddy coverage",
        "beat on impact",
        "snap zoom",
    ),
    "horror": DirectorPersonality(
        "horror",
        "Horror director",
        "negative space, slow dread, off-screen threat, underexposed corners",
        "bright flat comedy lighting",
        "slow reveal",
        "static wide",
    ),
}


def director_by_id(name: str) -> DirectorPersonality | None:
    return _DIRECTORS.get((name or "").strip().lower().replace(" ", "_"))


# ==================================================================================================
# style_engines: Style-specific generation engines (realistic, anime, 3D, voxel, etc.).
# ==================================================================================================


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
    region_motion: bool = True
    motion_transfer: bool = True
    pipeline_notes: str = ""
    retrieval_tags: list[str] = field(default_factory=list)
    animation_preset: str = ""


_ENGINES: dict[RenderEngine, EnginePreset] = {
    RenderEngine.REALISTIC: EnginePreset(
        id=RenderEngine.REALISTIC,
        title="Live-action cinematic",
        positive="photorealistic, natural skin, real lighting, cinematic depth of field",
        negative="cartoon, anime linework, plastic CGI, uncanny valley",
        retrieval_tags=["live action", "cinematic", "realistic"],
    ),
    RenderEngine.ANIME_2D: EnginePreset(
        id=RenderEngine.ANIME_2D,
        title="2D anime / cel animation",
        positive="anime cel shading, clean lineart, expressive eyes, limited frames feel",
        negative="photoreal pores, muddy gradients, western comic shading",
        animation_preset="anime_tv",
        retrieval_tags=["anime", "cel", "2d animation"],
    ),
    RenderEngine.PIXAR_3D: EnginePreset(
        id=RenderEngine.PIXAR_3D,
        title="3D stylized (Pixar/DreamWorks)",
        positive="3D animated film, subsurface skin, stylized proportions, soft global illumination",
        negative="flat 2D, photoreal news footage, low poly untextured",
        animation_preset="pixar",
    ),
    RenderEngine.LOW_POLY: EnginePreset(
        id=RenderEngine.LOW_POLY,
        title="Low-poly / PS1 aesthetic",
        positive="low poly 3D, affine texture warping, vertex jitter, retro game cutscene",
        negative="smooth HD, raytraced reflections, film grain realism",
        retrieval_tags=["ps1", "n64", "low poly"],
    ),
    RenderEngine.VOXEL: EnginePreset(
        id=RenderEngine.VOXEL,
        title="Voxel / Minecraft-style",
        positive="voxel world, blocky characters, cubic clouds, minecraft aesthetic",
        negative="smooth curves, realistic anatomy, film lighting",
        retrieval_tags=["minecraft", "voxel", "blocky"],
    ),
    RenderEngine.LEGO: EnginePreset(
        id=RenderEngine.LEGO,
        title="LEGO brick stop-motion",
        positive="LEGO minifigure, studded bricks, practical stop-motion timing",
        negative="organic skin, smooth CGI humans",
        animation_preset="stop_motion",
    ),
    RenderEngine.CLAYMATION: EnginePreset(
        id=RenderEngine.CLAYMATION,
        title="Claymation / stop-motion",
        positive="clay fingerprints, miniature sets, uneven stop-motion cadence, practical lighting",
        negative="fluid CGI motion, perfect symmetry",
        animation_preset="stop_motion",
    ),
    RenderEngine.VECTOR: EnginePreset(
        id=RenderEngine.VECTOR,
        title="Vector / motion graphics",
        positive="flat vector shapes, bold graphic design, clean curves, motion graphics",
        negative="photoreal texture, noisy grain",
    ),
    RenderEngine.PIXEL_ART: EnginePreset(
        id=RenderEngine.PIXEL_ART,
        title="Pixel art",
        positive="pixel art, limited palette, crisp pixels, 16-bit game aesthetic",
        negative="anti-aliased smooth gradients, HD photography",
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
    ),
}


def list_engines() -> list[EnginePreset]:
    return list(_ENGINES.values())


def engine_by_id(name: str) -> EnginePreset | None:
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
    rules: list[tuple[tuple[str, ...], RenderEngine]] = [
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


def engine_edit_overrides(preset: EnginePreset) -> dict[str, Any]:
    """Scene ``edit`` defaults for an engine; temporal dials come from the engine's motion grammar."""
    g = grammar_for_engine(preset.id.value)
    return {
        "post_grade": g.post_grade,
        "keyframe_interval": g.keyframe_interval,
        "edit_strength": g.edit_strength,
        "depth_interpolate": g.depth_interpolate,
        "region_motion": preset.region_motion,
        "motion_transfer": preset.motion_transfer,
        "temporal_alpha": g.temporal_alpha,
    }


# ==================================================================================================
# generation_router: Generation router — pick the right engine for each scene.
# ==================================================================================================


@dataclass(slots=True)
class RouteDecision:
    engine: RenderEngine
    preset: EnginePreset
    confidence: float
    reasons: list[str] = field(default_factory=list)
    edit_overrides: dict[str, Any] = field(default_factory=dict)
    style_positive: str = ""
    style_negative: str = ""
    retrieval_tags: list[str] = field(default_factory=list)


def route_scene(
    prompt: str,
    *,
    style_hint: str = "",
    engine_override: str = "",
    layer_stack: Mapping[str, str] | None = None,
) -> RouteDecision:
    """Analyze prompt + hints → engine + edit overrides."""
    reasons: list[str] = []
    if engine_override and engine_override.lower() not in ("auto", ""):
        preset = engine_by_id(engine_override)
        if preset:
            reasons.append(f"explicit engine={preset.id.value}")
            return RouteDecision(
                engine=preset.id,
                preset=preset,
                confidence=1.0,
                reasons=reasons,
                edit_overrides=engine_edit_overrides(preset),
                style_positive=preset.positive,
                style_negative=preset.negative,
                retrieval_tags=list(preset.retrieval_tags),
            )

    if layer_stack and len(layer_stack) > 1:
        eng = RenderEngine.HYBRID
        preset = engine_by_id("hybrid")
        reasons.append("multi-layer stack → hybrid router")
    else:
        eng = match_engine_from_prompt(prompt, style_hint=style_hint)
        preset = engine_by_id(eng.value)
        reasons.append(f"prompt/style match → {eng.value}")

    assert preset is not None
    return RouteDecision(
        engine=eng,
        preset=preset,
        confidence=0.75,
        reasons=reasons,
        edit_overrides=engine_edit_overrides(preset),
        style_positive=preset.positive,
        style_negative=preset.negative,
        retrieval_tags=list(preset.retrieval_tags),
    )


# ==================================================================================================
# layer_stack: Layer stack schema — regenerate one layer without full rerender.
# ==================================================================================================


@dataclass(slots=True)
class LayerSpec:
    name: str
    engine: str = "realistic"
    prompt: str = ""
    mask: str = ""
    z_order: int = 0


@dataclass(slots=True)
class LayerStack:
    layers: list[LayerSpec] = field(default_factory=list)

    def engines_by_layer(self) -> dict[str, str]:
        return {layer.name: layer.engine for layer in self.layers}


def parse_layer_stack(raw: Any) -> LayerStack:
    stack = LayerStack()
    if not raw:
        return stack
    rows = raw if isinstance(raw, list) else (raw.get("layers") if isinstance(raw, Mapping) else [])
    if not isinstance(rows, list):
        return stack
    for i, row in enumerate(rows):
        if isinstance(row, str):
            stack.layers.append(LayerSpec(name=row, z_order=i))
            continue
        if not isinstance(row, Mapping):
            continue
        stack.layers.append(
            LayerSpec(
                name=str(row.get("name") or row.get("id") or f"layer_{i}"),
                engine=str(row.get("engine") or row.get("style") or "realistic"),
                prompt=str(row.get("prompt") or ""),
                mask=str(row.get("mask") or ""),
                z_order=int(row.get("z") or row.get("z_order") or i),
            )
        )
    stack.layers.sort(key=lambda layer: layer.z_order)
    return stack


def stack_prompt_suffix(stack: LayerStack) -> str:
    parts = []
    for layer in stack.layers:
        if layer.prompt:
            parts.append(f"[layer {layer.name}: {layer.prompt}]")
        elif layer.engine:
            preset = engine_by_id(layer.engine)
            if preset:
                parts.append(f"[layer {layer.name}: {preset.positive}]")
    return " ".join(parts)


# ==================================================================================================
# world_memory: World bible — persistent setting rules across generations.
# ==================================================================================================


@dataclass(slots=True)
class WorldBible:
    id: str
    name: str = ""
    architecture: str = ""
    technology: str = ""
    weather: str = ""
    palette: str = ""
    culture: str = ""
    geography: str = ""
    magic_rules: str = ""
    era: str = ""
    locations: dict[str, str] = field(default_factory=dict)
    notes: str = ""


def parse_world(raw: Any) -> WorldBible | None:
    if not raw:
        return None
    if isinstance(raw, str):
        return WorldBible(id="world", name=raw)
    if not isinstance(raw, Mapping):
        return None
    locs = raw.get("locations") or {}
    if not isinstance(locs, Mapping):
        locs = {}
    wid = str(raw.get("id") or "world")
    magic = raw.get("magic")
    magic_s = ""
    if isinstance(magic, Mapping):
        magic_s = ", ".join(f"{k}: {v}" for k, v in magic.items())
    elif isinstance(magic, str):
        magic_s = magic
    return WorldBible(
        id=wid,
        name=str(raw.get("name") or raw.get("title") or wid),
        architecture=str(raw.get("architecture") or ""),
        technology=str(raw.get("technology") or raw.get("tech") or ""),
        weather=str(raw.get("weather") or raw.get("climate") or ""),
        palette=str(raw.get("palette") or raw.get("colors") or ""),
        culture=str(raw.get("culture") or ""),
        geography=str(raw.get("geography") or ""),
        magic_rules=magic_s,
        era=str(raw.get("era") or ""),
        locations={str(k): str(v) for k, v in locs.items()},
        notes=str(raw.get("notes") or raw.get("description") or ""),
    )


def world_to_prompt(w: WorldBible, *, location: str = "") -> str:
    parts: list[str] = [f"world {w.name}" if w.name else "consistent world"]
    for label, val in (
        ("architecture", w.architecture),
        ("technology", w.technology),
        ("weather", w.weather),
        ("palette", w.palette),
        ("culture", w.culture),
        ("geography", w.geography),
        ("era", w.era),
        ("magic", w.magic_rules),
    ):
        if val:
            parts.append(f"{label}: {val}")
    if location and location in w.locations:
        parts.append(f"location {location}: {w.locations[location]}")
    if w.notes:
        parts.append(w.notes)
    return ", ".join(parts)


def world_negative(w: WorldBible) -> str:
    parts = ["anachronistic technology mismatch", "wrong architectural style"]
    if w.palette:
        parts.append("off-palette colors")
    return ", ".join(parts)


# ==================================================================================================
# storyboard: Kling-style storyboard cuts with camera verbs.
# ==================================================================================================

CAMERA_VERBS: dict[str, str] = {
    "static": "locked tripod, static frame",
    "pan_left": "smooth pan left",
    "pan_right": "smooth pan right",
    "tilt_up": "tilt up",
    "tilt_down": "tilt down",
    "push_in": "slow dolly push in",
    "pull_back": "dolly pull back reveal",
    "tracking": "tracking shot following subject",
    "orbit": "orbiting camera around subject",
    "crane": "crane shot rising",
    "handheld": "handheld documentary camera",
    "whip_pan": "fast whip pan",
    "zoom_in": "slow zoom in",
    "zoom_out": "zoom out",
    "ots": "over the shoulder shot",
    "pov": "point of view shot",
    "establishing": "wide establishing shot",
    "close_up": "close-up shot",
}


@dataclass(slots=True)
class StoryboardCut:
    id: str = ""
    prompt: str = ""
    duration_sec: float = 0.0
    camera: str = ""
    shot_type: str = ""
    transition: str = "cut"
    characters: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    effects: list[str] = field(default_factory=list)
    start_image: str = ""
    end_image: str = ""
    flf2v: bool = False
    motion_brush: dict[str, Any] = field(default_factory=dict)
    elements: list[str] = field(default_factory=list)
    bindings: dict[str, dict[str, Any]] = field(default_factory=dict)


def camera_prompt_fragment(camera: str) -> str:
    key = (camera or "").strip().lower().replace(" ", "_").replace("-", "_")
    if key in CAMERA_VERBS:
        return CAMERA_VERBS[key]
    if key:
        return f"{camera} camera movement"
    return ""


def parse_storyboard(raw: Any) -> list[StoryboardCut]:
    if not raw:
        return []
    cuts_raw: Any = raw
    if isinstance(raw, Mapping):
        cuts_raw = raw.get("cuts") or raw.get("shots") or []
    if not isinstance(cuts_raw, list):
        return []
    cuts: list[StoryboardCut] = []
    for i, row in enumerate(cuts_raw):
        if isinstance(row, str):
            cuts.append(StoryboardCut(id=f"cut_{i}", prompt=row))
            continue
        if not isinstance(row, Mapping):
            continue
        cuts.append(
            StoryboardCut(
                id=str(row.get("id") or f"cut_{i}"),
                prompt=str(row.get("prompt") or row.get("description") or ""),
                duration_sec=float(row.get("duration_sec") or row.get("duration") or 0.0),
                camera=str(row.get("camera") or row.get("camera_move") or ""),
                shot_type=str(row.get("shot_type") or row.get("type") or ""),
                transition=str(row.get("transition") or "cut").lower(),
                characters=_str_list(row.get("characters") or row.get("cast")),
                objects=_str_list(row.get("objects") or row.get("props")),
                effects=_str_list(row.get("effects")),
                start_image=str(row.get("start_image") or row.get("start") or ""),
                end_image=str(row.get("end_image") or row.get("end") or ""),
                flf2v=bool(
                    row.get("flf2v") or row.get("first_last") or (row.get("start_image") and row.get("end_image"))
                ),
                motion_brush=dict(row.get("motion_brush") or {}),
                elements=_str_list(row.get("elements") or row.get("bind_elements")),
                bindings=dict(row.get("bindings") or {}),
            )
        )
    return cuts


def _str_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [x.strip() for x in re.split(r"[,;]", v) if x.strip()]
    return [str(x) for x in v if str(x).strip()]


# ==================================================================================================
# director_mode: AI Director Mode — expand one prompt into an editable storyboard.
# ==================================================================================================


@dataclass(slots=True)
class DirectorNotes:
    mood: str = ""
    lens: str = ""
    camera: str = ""
    color_grade: str = ""
    music: str = ""
    pacing: str = ""


@dataclass(slots=True)
class DirectorExpansion:
    cuts: list[StoryboardCut]
    notes: DirectorNotes
    genre: str = ""
    raw_prompt: str = ""


_GENRE_RULES: list[tuple[tuple[str, ...], str, DirectorNotes]] = [
    (
        ("horror", "scary", "abandoned", "creature", "dark"),
        "horror",
        DirectorNotes(
            mood="dread, unease",
            lens="35mm wide",
            camera="slow dolly, negative space",
            color_grade="desaturated, crushed blacks",
            music="low drones, distant metallic hits",
            pacing="slow reveal",
        ),
    ),
    (
        ("action", "chase", "explosion", "fight", "battle"),
        "action",
        DirectorNotes(
            mood="kinetic urgency",
            lens="24mm handheld",
            camera="tracking, whip pans",
            color_grade="teal orange, high contrast",
            music="driving percussion",
            pacing="fast cuts",
        ),
    ),
    (
        ("romance", "love", "kiss", "wedding"),
        "romance",
        DirectorNotes(
            mood="intimate warmth",
            lens="85mm shallow",
            camera="gentle push in",
            color_grade="warm soft highlights",
            music="strings, piano",
            pacing="lingering beats",
        ),
    ),
    (
        ("comedy", "funny", "slapstick"),
        "comedy",
        DirectorNotes(
            mood="playful",
            lens="50mm",
            camera="static wide for timing",
            color_grade="bright, even",
            music="light staccato",
            pacing="beat on gag",
        ),
    ),
    (
        ("anime", "ghibli", "magical"),
        "anime",
        DirectorNotes(
            mood="wonder",
            lens="anime dramatic angles",
            camera="push in on reaction",
            color_grade="painterly skies",
            music="orchestral swell",
            pacing="held emotional beats",
        ),
    ),
]


def genre_director_notes(prompt: str) -> tuple[str, DirectorNotes]:
    low = (prompt or "").lower()
    for keys, genre, notes in _GENRE_RULES:
        if any(k in low for k in keys):
            return genre, notes
    return "drama", DirectorNotes(
        mood="grounded drama",
        lens="50mm",
        camera="motivated movement",
        color_grade="naturalistic",
        music="subtle ambient",
        pacing="scene rhythm",
    )


def _infer_shot_sequence(prompt: str, genre: str) -> list[tuple[str, float, str, str]]:
    """Returns (prompt_fragment, duration_weight, camera, shot_type)."""
    low = prompt.lower()
    beats: list[tuple[str, float, str, str]] = []
    if genre in ("action", "horror"):
        beats.append(("establishing geography and stakes", 0.22, "establishing", "establishing"))
    elif "enter" in low or "walk" in low or "arrive" in low:
        beats.append(("establishing context", 0.20, "establishing", "establishing"))

    if any(w in low for w in ("fight", "chase", "battle", "explosion")):
        beats.append((f"main action: {prompt}", 0.45, "tracking", "medium"))
        beats.append(("reaction or consequence beat", 0.18, "close_up", "close_up"))
    elif any(w in low for w in ("close", "face", "eyes", "portrait")):
        beats.append((prompt, 0.55, "push_in", "close_up"))
    else:
        beats.append((f"primary story beat: {prompt}", 0.50, "tracking", "medium"))

    if genre == "horror" and len(beats) < 3:
        beats.append(("lingering unsettling hold", 0.15, "static", "wide"))
    elif len(beats) < 2:
        beats.append(("resolution or button", 0.25, "pull_back", "wide"))
    return beats


def expand_prompt_to_storyboard(
    prompt: str,
    *,
    duration_sec: float = 6.0,
    genre_override: str = "",
    characters: Sequence[str] | None = None,
) -> DirectorExpansion:
    """
    Turn one line into Kling-style multi-shot storyboard + director notes.

    User can edit cuts in scene JSON before generation (--plan-only).
    """
    genre, notes = genre_director_notes(prompt)
    if genre_override:
        genre = genre_override
        _, notes = genre_director_notes(f"{genre} {prompt}")

    seq = _infer_shot_sequence(prompt, genre)
    total_w = sum(w for _, w, _, _ in seq) or 1.0
    cuts: list[StoryboardCut] = []
    chars = list(characters or [])
    for i, (frag, weight, cam, st) in enumerate(seq):
        dur = max(0.5, duration_sec * (weight / total_w))
        cam_frag = camera_prompt_fragment(cam)
        p = frag
        if cam_frag and cam_frag.lower() not in p.lower():
            p = f"{p}, {cam_frag}"
        cuts.append(
            StoryboardCut(
                id=f"dir_{i}",
                prompt=p.strip(),
                duration_sec=round(dur, 2),
                camera=cam,
                shot_type=st,
                characters=chars,
                transition="dissolve" if i > 0 and genre in ("romance", "drama") else "cut",
            )
        )
    return DirectorExpansion(cuts=cuts, notes=notes, genre=genre, raw_prompt=prompt)


# ==================================================================================================
# thumbnail_rehearsal: Thumbnail-first rehearsal — cheap composition gates before full GPU render.
# ==================================================================================================


@dataclass(slots=True)
class ThumbnailConfig:
    enabled: bool = False
    size: int = 128
    frames_per_shot: int = 1
    gate: str = "off"  # off | warn | require_approval
    prompt_suffix: str = "storyboard thumbnail, strong silhouette, composition sketch"


@dataclass(slots=True)
class ThumbnailSpec:
    shot_id: str
    shot_index: int
    frame_role: str  # start | mid | end
    prompt: str
    width: int
    height: int
    approved: bool = False


@dataclass(slots=True)
class ThumbnailRehearsalPlan:
    enabled: bool
    config: ThumbnailConfig
    specs: list[ThumbnailSpec] = field(default_factory=list)
    gate_passed: bool = True
    pending_count: int = 0


def parse_thumbnail_config(
    data: Mapping[str, Any] | None,
    *,
    studio: Mapping[str, Any] | None = None,
    edit: Mapping[str, Any] | None = None,
) -> ThumbnailConfig:
    """Read thumbnail settings from continuity, studio, or edit blocks."""
    raw: dict[str, Any] = {}
    if isinstance(data, Mapping):
        thumb = data.get("thumbnail") or data.get("thumbnail_first")
        if isinstance(thumb, Mapping):
            raw.update(thumb)
        elif thumb is True:
            raw["enabled"] = True
    if isinstance(studio, Mapping):
        if studio.get("thumbnail_first") or studio.get("thumbnail"):
            raw.setdefault("enabled", True)
        for k in ("thumbnail_size", "thumbnail_gate", "thumbnail_frames"):
            if k in studio:
                raw[k.replace("thumbnail_", "")] = studio[k]
    if isinstance(edit, Mapping):
        if edit.get("thumbnail_first") or edit.get("thumbnail_pass"):
            raw.setdefault("enabled", True)

    enabled = bool(raw.get("enabled", False))
    size = int(raw.get("size") or raw.get("thumbnail_size") or 128)
    size = max(32, min(512, size))
    frames = int(raw.get("frames_per_shot") or raw.get("frames") or 1)
    frames = max(1, min(3, frames))
    gate = str(raw.get("gate") or "off").lower()
    suffix = str(
        raw.get("prompt_suffix") or "storyboard thumbnail, strong silhouette, readable composition, minimal detail"
    )
    return ThumbnailConfig(enabled=enabled, size=size, frames_per_shot=frames, gate=gate, prompt_suffix=suffix)


def _frame_roles(n: int) -> list[str]:
    if n <= 1:
        return ["start"]
    if n == 2:
        return ["start", "end"]
    return ["start", "mid", "end"]


def plan_thumbnails(
    shots: Sequence[Any],
    *,
    config: ThumbnailConfig,
    base_prompt: str = "",
    aspect_width: int = 16,
    aspect_height: int = 9,
) -> ThumbnailRehearsalPlan:
    if not config.enabled:
        return ThumbnailRehearsalPlan(enabled=False, config=config, gate_passed=True)

    size = config.size
    if aspect_width >= aspect_height:
        w, h = size, max(32, int(size * aspect_height / max(1, aspect_width)))
    else:
        h, w = size, max(32, int(size * aspect_width / max(1, aspect_height)))

    specs: list[ThumbnailSpec] = []
    roles = _frame_roles(config.frames_per_shot)
    pending = 0

    for i, sh in enumerate(shots):
        sid = str(getattr(sh, "id", None) or f"shot_{i}")
        prompt = str(getattr(sh, "prompt", "") or base_prompt)
        if config.prompt_suffix and config.prompt_suffix.lower() not in prompt.lower():
            prompt = f"{prompt}, {config.prompt_suffix}".strip(", ")
        approved = bool(getattr(sh, "thumbnail_approved", False))
        for role in roles:
            specs.append(
                ThumbnailSpec(
                    shot_id=sid,
                    shot_index=i,
                    frame_role=role,
                    prompt=prompt,
                    width=w,
                    height=h,
                    approved=approved,
                )
            )
        if not approved:
            pending += 1

    gate_passed = pending == 0 or config.gate in ("off", "warn")
    if config.gate == "require_approval" and pending > 0:
        gate_passed = False

    return ThumbnailRehearsalPlan(
        enabled=True,
        config=config,
        specs=specs,
        gate_passed=gate_passed,
        pending_count=pending,
    )


def thumbnail_edit_overrides(config: ThumbnailConfig) -> dict[str, Any]:
    """Cheap pass overrides when thumbnail_first is active."""
    return {
        "thumbnail_pass": True,
        "thumbnail_size": config.size,
        "keyframe_interval": 24,
        "edit_strength": 0.28,
        "frame_enhance": False,
        "quality_retry": False,
        "semantic_drift_repair": False,
        "flow_consistency": False,
        "deflicker": False,
        "post_grade": "",
    }


def apply_thumbnail_timeline(
    width: int,
    height: int,
    config: ThumbnailConfig,
) -> tuple[int, int]:
    """Downscale timeline for thumbnail pass."""
    if not config.enabled:
        return width, height
    size = config.size
    if width >= height:
        tw = size
        th = max(32, int(size * height / max(1, width)))
    else:
        th = size
        tw = max(32, int(size * width / max(1, height)))
    return tw, th


def thumbnail_gate_issues(plan: ThumbnailRehearsalPlan) -> list[str]:
    if not plan.enabled or plan.gate_passed:
        return []
    gate = plan.config.gate
    if gate == "require_approval":
        return [
            f"Thumbnail gate: {plan.pending_count} shot(s) lack thumbnail_approved: true "
            "(set on shots[] or run thumbnail pass first)"
        ]
    if gate == "warn" and plan.pending_count > 0:
        return [f"Thumbnail rehearsal: {plan.pending_count} shot(s) not marked approved (advisory)"]
    return []


__all__ = [
    "StudioCompileResult",
    "compile_studio_block",
    "AnimationPrinciples",
    "principles_from_dict",
    "principles_prompt",
    "preset_principles",
    "CameraRig",
    "parse_camera_rig",
    "rig_to_prompt",
    "CharacterBible",
    "parse_character_bibles",
    "bible_to_prompt",
    "bible_negative",
    "DirectorPersonality",
    "director_by_id",
    "EnginePreset",
    "RenderEngine",
    "engine_by_id",
    "list_engines",
    "match_engine_from_prompt",
    "RouteDecision",
    "route_scene",
    "LayerSpec",
    "LayerStack",
    "parse_layer_stack",
    "stack_prompt_suffix",
    "WorldBible",
    "parse_world",
    "world_to_prompt",
    "world_negative",
    "CAMERA_VERBS",
    "StoryboardCut",
    "camera_prompt_fragment",
    "parse_storyboard",
    "DirectorExpansion",
    "DirectorNotes",
    "expand_prompt_to_storyboard",
    "genre_director_notes",
    "ThumbnailConfig",
    "ThumbnailSpec",
    "ThumbnailRehearsalPlan",
    "parse_thumbnail_config",
    "plan_thumbnails",
    "thumbnail_edit_overrides",
    "thumbnail_gate_issues",
    "apply_thumbnail_timeline",
]

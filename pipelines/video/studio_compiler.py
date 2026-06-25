"""Compile studio block → routed engine, director storyboard, memory, rigs."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

__all__ = ["StudioCompileResult", "compile_studio_block"]


def compile_studio_block(
    data: Mapping[str, Any],
    *,
    scene_prompt: str,
    duration_sec: float,
    style_notes: str = "",
    existing_edit: Optional[Mapping[str, Any]] = None,
) -> "StudioCompileResult":
    from .animation_principles import principles_from_dict, principles_prompt
    from .camera_rig import parse_camera_rig, rig_to_prompt
    from .character_memory import bible_negative, bible_to_prompt, parse_character_bibles
    from .director_mode import expand_prompt_to_storyboard
    from .director_personalities import director_by_id
    from .generation_router import route_scene
    from .layer_stack import parse_layer_stack, stack_prompt_suffix
    from .motion_library import parse_motion_library, resolve_motion_clip
    from .rehearsal_pipeline import stage_edit_overrides
    from .world_memory import parse_world, world_negative, world_to_prompt

    studio = dict(data.get("studio") or {})
    edit = dict(existing_edit or {})
    prompt_suffix: List[str] = []
    neg_suffix: List[str] = []
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
    stage = str(studio.get("rehearsal") or studio.get("rehearsal_stage") or "full")
    edit.update(stage_edit_overrides(stage))

    # Thumbnail-first rehearsal
    from .thumbnail_rehearsal import parse_thumbnail_config, thumbnail_edit_overrides

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
        edit: Dict[str, Any],
        prompt_suffix: str,
        negative_suffix: str,
        storyboard_cuts: Any,
        engine: str,
        motion_clip: str,
        studio_meta: Dict[str, Any],
    ) -> None:
        self.edit = edit
        self.prompt_suffix = prompt_suffix
        self.negative_suffix = negative_suffix
        self.storyboard_cuts = storyboard_cuts
        self.engine = engine
        self.motion_clip = motion_clip
        self.studio_meta = studio_meta

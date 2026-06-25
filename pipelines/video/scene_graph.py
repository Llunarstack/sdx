"""
One scene graph to rule them all — scenes, shots, cast, props, effects, transforms.

You edit **one JSON file**. Everything else compiles from it. You never hand-wire
retrieval, keyframes, and stitch separately unless you want to.

Mental model (3 layers only):

    SCENE  → global prompt, duration, style, cast/props/effects libraries
    SHOTS  → camera beats (the only required list besides scene)
    ENGINE → retrieval → keyframe edit → interpolate → stitch (automatic)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .types import MasterTimeline, ShotSpec, TransitionType, VideoMode, VideoPlan

__all__ = [
    "CompiledScene",
    "EntityDef",
    "SceneGraph",
    "compile_scene_file",
    "compile_scene_graph",
    "load_scene_graph",
    "validate_scene_graph",
]

# Built-in effect tags → (positive, negative) fragments
_EFFECT_PRESETS: Dict[str, tuple[str, str]] = {
    "fog": ("atmospheric fog, volumetric haze", "clear crisp air"),
    "rain": ("rain streaks, wet surfaces, overcast", "dry weather"),
    "snow": ("falling snow, cold breath, soft diffusion", "summer heat"),
    "neon": ("neon reflections, saturated night colors", "flat daylight"),
    "film_grain": ("35mm film grain, subtle halation", "digital plastic smoothness"),
    "slow_motion": ("slow motion energy, motion blur trails", "frozen static pose"),
    "shake": ("handheld camera shake, documentary urgency", "locked tripod"),
    "vhs": ("VHS tracking, analog noise, retro bleed", "clean digital"),
    "glow": ("blooming highlights, soft glow", "harsh clipped highlights"),
    "desaturate": ("muted desaturated palette", "oversaturated neon"),
}


@dataclass(slots=True)
class EntityDef:
    """Character or object — image + text roles and control mode."""

    id: str
    description: str = ""
    negative: str = ""
    lock: bool = False
    reference_image: str = ""
    reference_strength: float = 0.8
    image_role: str = ""
    text_role: str = ""
    control: str = "transform"
    bind_input: str = ""
    bind_element: str = ""
    auto_rig: bool = False
    part: str = ""
    text_by_part: Dict[str, str] = field(default_factory=dict)
    mask_path: str = ""


@dataclass(slots=True)
class ShotNode:
    id: str
    prompt: str
    duration_sec: float = 0.0
    shot_type: str = "medium"
    characters: List[str] = field(default_factory=list)
    objects: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    transforms: List[str] = field(default_factory=list)
    motion_hint: str = ""
    transition: str = "cut"
    reference_clip: str = ""
    keyframe_interval: int = 0
    edit_strength: float = 0.0
    bindings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    start_image: str = ""
    end_image: str = ""
    flf2v: bool = False
    motion_brush: Dict[str, Any] = field(default_factory=dict)
    camera: str = ""
    bind_elements: List[str] = field(default_factory=list)
    gaze: str = ""
    props_state: Dict[str, str] = field(default_factory=dict)
    lighting: Dict[str, Any] = field(default_factory=dict)
    thumbnail_approved: bool = False
    kinetic: Dict[str, Any] = field(default_factory=dict)
    screen_direction: str = ""
    emotion: str = ""
    weather: str = ""
    weather_spec: Dict[str, Any] = field(default_factory=dict)
    witness: str = ""
    material_state: Dict[str, str] = field(default_factory=dict)
    offscreen: List[str] = field(default_factory=list)
    injuries: Dict[str, Any] = field(default_factory=dict)
    threads: List[str] = field(default_factory=list)
    silence: bool = False
    stinger: bool = False


@dataclass(slots=True)
class SceneGraph:
    """User-facing scene document."""

    version: int = 1
    mode: VideoMode = VideoMode.T2V
    scene_prompt: str = ""
    duration_sec: float = 6.0
    fps: float = 24.0
    width: int = 1280
    height: int = 720
    global_negative: str = ""
    style_notes: str = ""
    anchor_image: str = ""
    motion_clip: str = ""
    inputs: List[Any] = field(default_factory=list)
    cast: Dict[str, EntityDef] = field(default_factory=dict)
    props: Dict[str, EntityDef] = field(default_factory=dict)
    transforms: Dict[str, str] = field(default_factory=dict)
    effects: Dict[str, tuple[str, str]] = field(default_factory=dict)
    shots: List[ShotNode] = field(default_factory=list)
    elements: Dict[str, Any] = field(default_factory=dict)
    storyboard: Dict[str, Any] = field(default_factory=dict)
    retrieval: Dict[str, Any] = field(default_factory=dict)
    edit: Dict[str, Any] = field(default_factory=dict)
    continuity: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CompiledScene:
    """Output of compile — ready for pipeline."""

    graph: SceneGraph
    plan: VideoPlan
    segment_overrides: List[Dict[str, Any]] = field(default_factory=list)
    control_plans: List[Any] = field(default_factory=list)


def load_scene_graph(path: str | Path) -> SceneGraph:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_scene_dict(data)


def parse_scene_dict(data: Mapping[str, Any]) -> SceneGraph:
    scene = data.get("scene") or {}
    if isinstance(scene, str):
        scene = {"prompt": scene}
    mode_raw = str(data.get("mode") or scene.get("mode") or "t2v").lower()
    mode = VideoMode.I2V if mode_raw == "i2v" else VideoMode.T2V

    cast = _parse_entities(data.get("characters") or data.get("cast") or {})
    props = _parse_entities(data.get("objects") or data.get("props") or {}, prefix="prop")
    inputs = _parse_inputs(data.get("inputs") or [])

    transforms = {str(k): str(v) for k, v in (data.get("transforms") or {}).items()}
    effects = dict(_EFFECT_PRESETS)
    for k, v in (data.get("effects") or {}).items():
        if isinstance(v, dict):
            effects[str(k)] = (str(v.get("positive") or ""), str(v.get("negative") or ""))
        elif isinstance(v, str):
            effects[str(k)] = (v, "")

    shots_raw = data.get("shots") or []
    shots: List[ShotNode] = []
    for i, row in enumerate(shots_raw):
        if isinstance(row, str):
            shots.append(ShotNode(id=f"shot_{i}", prompt=row))
            continue
        if not isinstance(row, Mapping):
            continue
        shots.append(
            ShotNode(
                id=str(row.get("id") or f"shot_{i}"),
                prompt=str(row.get("prompt") or row.get("description") or ""),
                duration_sec=float(row.get("duration_sec") or row.get("duration") or 0.0),
                shot_type=str(row.get("shot_type") or row.get("type") or "medium"),
                characters=_as_str_list(row.get("characters") or row.get("cast")),
                objects=_as_str_list(row.get("objects") or row.get("props")),
                effects=_as_str_list(row.get("effects")),
                transforms=_as_str_list(row.get("transforms")),
                motion_hint=str(row.get("motion_hint") or row.get("motion") or ""),
                transition=str(row.get("transition") or "cut").lower(),
                reference_clip=str(row.get("reference_clip") or row.get("clip") or ""),
                keyframe_interval=int(row.get("keyframe_interval") or 0),
                edit_strength=float(row.get("edit_strength") or 0.0),
                bindings=dict(row.get("bindings") or {}),
                start_image=str(row.get("start_image") or row.get("start") or ""),
                end_image=str(row.get("end_image") or row.get("end") or ""),
                flf2v=bool(
                    row.get("flf2v") or row.get("first_last") or (row.get("start_image") and row.get("end_image"))
                ),
                motion_brush=dict(row.get("motion_brush") or {}),
                camera=str(row.get("camera") or row.get("camera_move") or ""),
                bind_elements=_as_str_list(row.get("elements") or row.get("bind_elements")),
                gaze=str(row.get("gaze") or ""),
                props_state={str(k): str(v) for k, v in (row.get("props_state") or row.get("prop_state") or {}).items()}
                if isinstance(row.get("props_state") or row.get("prop_state"), Mapping)
                else {},
                lighting=dict(row.get("lighting") or {}) if isinstance(row.get("lighting"), Mapping) else {},
                thumbnail_approved=bool(row.get("thumbnail_approved") or row.get("thumb_ok")),
                kinetic=dict(row.get("kinetic") or {}) if isinstance(row.get("kinetic"), Mapping) else {},
                screen_direction=str(row.get("screen_direction") or row.get("move_direction") or ""),
                emotion=str(row.get("emotion") or ""),
                weather=str(row.get("weather") or "") if not isinstance(row.get("weather"), Mapping) else "",
                weather_spec=dict(row.get("weather") or {}) if isinstance(row.get("weather"), Mapping) else {},
                witness=str(row.get("witness") or row.get("pov_witness") or ""),
                material_state={
                    str(k): str(v) for k, v in (row.get("material_state") or row.get("materials") or {}).items()
                }
                if isinstance(row.get("material_state") or row.get("materials"), Mapping)
                else {},
                offscreen=_as_str_list(row.get("offscreen") or row.get("offscreen_zones")),
                injuries=dict(row.get("injuries") or row.get("scars") or {})
                if isinstance(row.get("injuries") or row.get("scars"), Mapping)
                else {},
                threads=_as_str_list(row.get("threads") or row.get("narrative_threads")),
                silence=bool(row.get("silence")),
                stinger=bool(row.get("stinger") or row.get("impact_stinger")),
            )
        )

    from .elements import parse_elements

    return SceneGraph(
        version=int(data.get("version") or 1),
        mode=mode,
        scene_prompt=str(scene.get("prompt") or data.get("prompt") or ""),
        duration_sec=float(scene.get("duration_sec") or scene.get("duration") or data.get("duration") or 6.0),
        fps=float(scene.get("fps") or data.get("fps") or 24.0),
        width=int(scene.get("width") or data.get("width") or 1280),
        height=int(scene.get("height") or data.get("height") or 720),
        global_negative=str(scene.get("negative") or data.get("negative") or ""),
        style_notes=str(scene.get("style") or data.get("style") or ""),
        anchor_image=str(data.get("anchor_image") or scene.get("anchor_image") or ""),
        motion_clip=str(data.get("motion_clip") or scene.get("motion_clip") or ""),
        inputs=inputs,
        cast=cast,
        props=props,
        transforms=transforms,
        effects=effects,
        shots=shots,
        elements=parse_elements(data.get("elements") or {}).elements,
        storyboard=dict(data.get("storyboard") or {}),
        retrieval=dict(data.get("retrieval") or {}),
        edit=dict(data.get("edit") or {}),
        continuity=dict(data.get("continuity") or {}),
        raw=dict(data),
    )


def _parse_inputs(raw: Any) -> List[Any]:
    from .controls import ControlMode, MediaInput

    out: List[MediaInput] = []
    if not isinstance(raw, list):
        return out
    for i, row in enumerate(raw):
        if not isinstance(row, Mapping):
            continue
        iid = str(row.get("id") or f"input_{i}")
        ctrl = str(row.get("control") or "transform").lower()
        try:
            mode = ControlMode(ctrl)
        except ValueError:
            mode = ControlMode.TRANSFORM
        out.append(
            MediaInput(
                id=iid,
                image=str(row.get("image") or row.get("ref") or ""),
                video=str(row.get("video") or row.get("clip") or ""),
                provides=str(row.get("provides") or row.get("image_role") or ""),
                text_changes=str(row.get("text_changes") or row.get("text") or row.get("text_role") or ""),
                control=mode,
                reference_strength=float(row.get("reference_strength") or row.get("strength") or 0.8),
                auto_rig=bool(row.get("auto_rig") or row.get("rig")),
                part=str(row.get("part") or ""),
                negative=str(row.get("negative") or ""),
            )
        )
    return out


def _parse_entity_mapping(row: Mapping[str, Any], eid: str) -> EntityDef:
    tbp = row.get("text_by_part") or row.get("rig_text") or {}
    if not isinstance(tbp, dict):
        tbp = {}
    return EntityDef(
        id=eid,
        description=str(row.get("description") or row.get("prompt") or ""),
        negative=str(row.get("negative") or ""),
        lock=bool(row.get("lock") or row.get("preserve")),
        reference_image=str(row.get("reference_image") or row.get("image") or row.get("ref") or ""),
        reference_strength=float(row.get("reference_strength") or row.get("strength") or 0.8),
        image_role=str(row.get("image_role") or row.get("provides") or ""),
        text_role=str(row.get("text_role") or row.get("text_changes") or row.get("text") or ""),
        control=str(row.get("control") or ("lock" if row.get("lock") else "transform")),
        bind_input=str(row.get("bind_input") or row.get("input") or row.get("bind_element") or ""),
        bind_element=str(row.get("bind_element") or row.get("element") or ""),
        auto_rig=bool(row.get("auto_rig") or row.get("rig")),
        part=str(row.get("part") or ""),
        text_by_part={str(k): str(v) for k, v in tbp.items()},
        mask_path=str(row.get("mask") or row.get("mask_path") or ""),
    )


def _parse_entities(raw: Any, *, prefix: str = "char") -> Dict[str, EntityDef]:
    out: Dict[str, EntityDef] = {}
    if isinstance(raw, list):
        for i, row in enumerate(raw):
            if isinstance(row, str):
                eid = f"{prefix}_{i}"
                out[eid] = EntityDef(id=eid, description=row)
            elif isinstance(row, Mapping):
                eid = str(row.get("id") or f"{prefix}_{i}")
                out[eid] = _parse_entity_mapping(row, eid)
    elif isinstance(raw, Mapping):
        for k, v in raw.items():
            eid = str(k)
            if isinstance(v, str):
                out[eid] = EntityDef(id=eid, description=v)
            elif isinstance(v, Mapping):
                out[eid] = _parse_entity_mapping(v, eid)
    return out


def _as_str_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()]
    return [str(x) for x in v if str(x).strip()]


def validate_scene_graph(graph: SceneGraph) -> List[str]:
    """Return human-readable issues (empty = OK)."""
    issues: List[str] = []
    if not graph.scene_prompt and not graph.shots:
        issues.append("Need scene.prompt or at least one shot")
    if graph.mode == VideoMode.I2V and not graph.anchor_image and not graph.inputs:
        issues.append("i2v mode requires anchor_image or inputs[].image")
    for sh in graph.shots:
        for cid in sh.characters:
            if cid not in graph.cast:
                issues.append(f"shot {sh.id!r} references unknown character {cid!r}")
        for pid in sh.objects:
            if pid not in graph.props:
                issues.append(f"shot {sh.id!r} references unknown object {pid!r}")
        for ef in sh.effects:
            if ef not in graph.effects:
                issues.append(f"shot {sh.id!r} references unknown effect {ef!r} (use preset or define in effects{{}})")
        for eid in sh.bind_elements:
            if eid not in graph.elements:
                issues.append(f"shot {sh.id!r} references unknown element {eid!r}")
    for cid, ent in graph.cast.items():
        if ent.bind_element and ent.bind_element not in graph.elements:
            issues.append(f"character {cid!r} bind_element unknown: {ent.bind_element!r}")

    from .continuity_validators import parse_validator_config, run_continuity_validation
    from .thumbnail_rehearsal import parse_thumbnail_config, plan_thumbnails, thumbnail_gate_issues

    shots_for_val = list(graph.shots) if graph.shots else []
    if shots_for_val or graph.continuity:
        cfg = parse_validator_config(graph.continuity)
        if graph.continuity or any(
            getattr(s, "gaze", "") or getattr(s, "props_state", {}) or getattr(s, "lighting", {}) for s in shots_for_val
        ):
            report = run_continuity_validation(shots_for_val, continuity=graph.continuity, config=cfg)
            for ci in report.errors():
                issues.append(f"[{ci.code}] {ci.message}")
            if not cfg.strict:
                for ci in report.warnings():
                    issues.append(f"[warn:{ci.code}] {ci.message}")

    thumb_cfg = parse_thumbnail_config(
        graph.continuity,
        studio=(graph.raw.get("studio") if isinstance(graph.raw.get("studio"), Mapping) else None),
        edit=graph.edit,
    )
    if thumb_cfg.enabled and thumb_cfg.gate == "require_approval" and shots_for_val:
        tplan = plan_thumbnails(
            shots_for_val,
            config=thumb_cfg,
            base_prompt=graph.scene_prompt,
            aspect_width=graph.width,
            aspect_height=graph.height,
        )
        for msg in thumbnail_gate_issues(tplan):
            issues.append(msg)

    return issues


def _merge_csv(*parts: str) -> str:
    return ", ".join(p.strip() for p in parts if p and p.strip())


def _compile_shot_prompt(graph: SceneGraph, shot: ShotNode) -> tuple[str, str, List[str]]:
    pos_parts: List[str] = []
    neg_parts: List[str] = []
    preserve: List[str] = []

    if shot.prompt:
        pos_parts.append(shot.prompt)
    elif graph.scene_prompt:
        pos_parts.append(graph.scene_prompt)

    for cid in shot.characters:
        ent = graph.cast.get(cid)
        if ent:
            if ent.description:
                pos_parts.append(ent.description)
            if ent.negative:
                neg_parts.append(ent.negative)
            if ent.lock:
                preserve.append(cid)

    for pid in shot.objects:
        ent = graph.props.get(pid)
        if ent:
            if ent.description:
                pos_parts.append(ent.description)
            if ent.negative:
                neg_parts.append(ent.negative)

    for ef in shot.effects:
        ep = graph.effects.get(ef)
        if ep:
            if ep[0]:
                pos_parts.append(ep[0])
            if ep[1]:
                neg_parts.append(ep[1])

    for tr in shot.transforms:
        frag = graph.transforms.get(tr)
        if frag:
            pos_parts.append(frag)

    if graph.style_notes:
        pos_parts.append(graph.style_notes)

    return _merge_csv(*pos_parts), _merge_csv(*neg_parts), preserve


def _auto_shots_from_scene(graph: SceneGraph) -> List[ShotNode]:
    """One scene prompt → beat-split shots (simple mode)."""
    from .shot_planner import split_prompt_into_beats

    beats = split_prompt_into_beats(graph.scene_prompt)
    n = max(1, len(beats))
    dur = graph.duration_sec / n
    return [ShotNode(id=f"shot_{i}", prompt=b, duration_sec=dur) for i, b in enumerate(beats)]


def _shots_from_storyboard(graph: SceneGraph) -> List[ShotNode]:
    from .storyboard import _infer_shot_type, camera_prompt_fragment, parse_storyboard

    cuts = parse_storyboard(graph.storyboard)
    if not cuts:
        return []
    total = sum(c.duration_sec for c in cuts if c.duration_sec > 0)
    per = graph.duration_sec / max(1, len(cuts)) if total <= 0 else None
    shots: List[ShotNode] = []
    for i, c in enumerate(cuts):
        cam_frag = camera_prompt_fragment(c.camera)
        prompt = c.prompt or graph.scene_prompt
        if cam_frag and cam_frag.lower() not in prompt.lower():
            prompt = f"{prompt}, {cam_frag}".strip(", ")
        st = c.shot_type or _infer_shot_type(c.camera, prompt)
        dur = c.duration_sec if c.duration_sec > 0 else (per or graph.duration_sec / len(cuts))
        shots.append(
            ShotNode(
                id=c.id or f"cut_{i}",
                prompt=prompt,
                duration_sec=dur,
                shot_type=st,
                characters=list(c.characters),
                objects=list(c.objects),
                effects=list(c.effects),
                motion_hint=cam_frag or c.camera,
                transition=c.transition,
                bindings=dict(c.bindings),
                start_image=c.start_image,
                end_image=c.end_image,
                flf2v=c.flf2v,
                motion_brush=dict(c.motion_brush),
                camera=c.camera,
                bind_elements=list(c.elements),
            )
        )
    return shots


def _shots_from_director_cuts(graph: SceneGraph, cuts: Sequence[Any]) -> List[ShotNode]:
    from .storyboard import StoryboardCut

    out: List[ShotNode] = []
    for i, c in enumerate(cuts):
        if not isinstance(c, StoryboardCut):
            continue
        out.append(
            ShotNode(
                id=c.id or f"dir_{i}",
                prompt=c.prompt or graph.scene_prompt,
                duration_sec=c.duration_sec,
                shot_type=c.shot_type or "medium",
                characters=list(c.characters),
                objects=list(c.objects),
                effects=list(c.effects),
                motion_hint=c.camera,
                transition=c.transition,
                bindings=dict(c.bindings),
                start_image=c.start_image,
                end_image=c.end_image,
                flf2v=c.flf2v,
                motion_brush=dict(c.motion_brush),
                camera=c.camera,
                bind_elements=list(c.elements),
            )
        )
    return out


def compile_scene_graph(graph: SceneGraph) -> CompiledScene:
    """
    Scene graph → VideoPlan + per-segment overrides.

    This is the ONLY place characters/objects/effects/transforms merge into prompts.
    """
    issues = validate_scene_graph(graph)
    if issues:
        raise ValueError("Scene graph invalid:\n  - " + "\n  - ".join(issues))

    from .studio_compiler import compile_studio_block

    studio_out = compile_studio_block(
        graph.raw,
        scene_prompt=graph.scene_prompt,
        duration_sec=graph.duration_sec,
        style_notes=graph.style_notes,
        existing_edit=graph.edit,
    )
    graph.edit.update(studio_out.edit)
    if studio_out.prompt_suffix:
        graph.scene_prompt = f"{graph.scene_prompt}, {studio_out.prompt_suffix}".strip(", ")
    if studio_out.negative_suffix:
        graph.global_negative = _merge_csv(graph.global_negative, studio_out.negative_suffix)
    if studio_out.motion_clip:
        graph.motion_clip = studio_out.motion_clip
    graph.raw.setdefault("studio", {})
    graph.raw["studio"]["engine"] = studio_out.engine
    graph.raw["studio"]["_compiled"] = studio_out.studio_meta

    shots = list(graph.shots) if graph.shots else []
    if not shots and graph.storyboard:
        shots = _shots_from_storyboard(graph)
    elif not shots and studio_out.storyboard_cuts:
        shots = _shots_from_director_cuts(graph, studio_out.storyboard_cuts)
    if not shots:
        shots = _auto_shots_from_scene(graph)
    if not shots:
        shots = [ShotNode(id="shot_0", prompt=graph.scene_prompt or "cinematic scene", duration_sec=graph.duration_sec)]

    total_dur = sum(s.duration_sec for s in shots if s.duration_sec > 0)
    if total_dur <= 0:
        per = graph.duration_sec / max(1, len(shots))
        for s in shots:
            s.duration_sec = per
    else:
        scale = graph.duration_sec / total_dur
        for s in shots:
            if s.duration_sec <= 0:
                s.duration_sec = graph.duration_sec / len(shots)
            s.duration_sec *= scale

    from .continuity_validators import parse_validator_config, run_continuity_validation
    from .thumbnail_rehearsal import (
        apply_thumbnail_timeline,
        parse_thumbnail_config,
        plan_thumbnails,
        thumbnail_gate_issues,
    )

    thumb_cfg = parse_thumbnail_config(
        graph.continuity,
        studio=graph.raw.get("studio") if isinstance(graph.raw.get("studio"), Mapping) else None,
        edit=graph.edit,
    )
    thumb_plan = plan_thumbnails(
        shots,
        config=thumb_cfg,
        base_prompt=graph.scene_prompt,
        aspect_width=graph.width,
        aspect_height=graph.height,
    )
    if thumb_plan.enabled:
        graph.width, graph.height = apply_thumbnail_timeline(graph.width, graph.height, thumb_cfg)

    continuity_cfg = parse_validator_config(graph.continuity)
    continuity_report = run_continuity_validation(shots, continuity=graph.continuity, config=continuity_cfg)

    from .frontier_compiler import compile_frontier_layers

    studio_genre = ""
    if isinstance(graph.raw.get("studio"), Mapping):
        studio_genre = str(graph.raw["studio"].get("genre") or "")
    frontier_out = compile_frontier_layers(
        graph.raw,
        shots,
        scene_prompt=graph.scene_prompt,
        duration_sec=graph.duration_sec,
        genre=studio_genre,
        cast=graph.cast,
        props=graph.props,
    )
    graph.edit.update(frontier_out.global_edit)
    enrich_by_id = frontier_out.enrichments

    default_kf = int(graph.edit.get("keyframe_interval") or 6)
    default_strength = float(graph.edit.get("edit_strength") or 0.55)

    shot_specs: List[ShotSpec] = []
    overrides: List[Dict[str, Any]] = []

    for i, sh in enumerate(shots):
        if sh.camera and not sh.motion_hint:
            from .storyboard import camera_prompt_fragment

            sh.motion_hint = camera_prompt_fragment(sh.camera)
        en = enrich_by_id.get(sh.id)
        if en and en.duration_delta:
            sh.duration_sec = round(sh.duration_sec + en.duration_delta, 3)
        pos, neg, preserve = _compile_shot_prompt(graph, sh)
        if en:
            if en.prompt_suffix:
                pos = _merge_csv(pos, en.prompt_suffix)
            if en.negative_suffix:
                neg = _merge_csv(neg, en.negative_suffix)
            for ef in en.effects:
                if ef not in sh.effects:
                    sh.effects.append(ef)
        shot_specs.append(
            ShotSpec(
                index=i,
                prompt=pos,
                duration_sec=round(sh.duration_sec, 3),
                shot_type=sh.shot_type,
                negative=neg,
                motion_hint=sh.motion_hint,
                must_preserve=preserve,
            )
        )
        tr = TransitionType.CUT
        if sh.transition in ("dissolve", "fade"):
            tr = TransitionType.DISSOLVE
        elif sh.transition in ("match", "match_action"):
            tr = TransitionType.MATCH_ACTION
        elif sh.transition in ("whip", "whip_pan"):
            tr = TransitionType.WHIP
        elif sh.transition in ("flash", "white_flash"):
            tr = TransitionType.FLASH
        elif sh.transition in ("dip", "dip_to_black", "fade_black"):
            tr = TransitionType.DIP
        overrides.append(
            {
                "shot_id": sh.id,
                "reference_clip": sh.reference_clip,
                "keyframe_interval": (en.edit_overrides.get("keyframe_interval") if en else None)
                or sh.keyframe_interval
                or default_kf,
                "edit_strength": sh.edit_strength or default_strength,
                "transition": tr,
                "characters": list(sh.characters),
                "objects": list(sh.objects),
                "effects": list(sh.effects),
                "start_image": sh.start_image,
                "end_image": sh.end_image,
                "flf2v": sh.flf2v or bool(sh.start_image and sh.end_image),
                "motion_brush": dict(sh.motion_brush),
                "bind_elements": list(sh.bind_elements),
                "frontier": dict(en.metadata) if en else {},
            }
        )
        if en:
            for k, v in en.edit_overrides.items():
                if k not in ("keyframe_interval",):
                    overrides[-1][k] = v

    ar = "16:9"
    if graph.height > graph.width:
        ar = "9:16"
    timeline = MasterTimeline(
        fps=graph.fps,
        width=graph.width,
        height=graph.height,
        duration_sec=graph.duration_sec,
        aspect_ratio=ar,
    )

    global_neg = _merge_csv(graph.global_negative, "flicker, morphing faces, inconsistent identity")
    plan = VideoPlan(
        mode=graph.mode,
        user_prompt=graph.scene_prompt or (shots[0].prompt if shots else ""),
        timeline=timeline,
        shots=shot_specs,
        global_negative=global_neg,
        style_notes=graph.style_notes,
        metadata={
            "scene_graph": True,
            "cast_ids": list(graph.cast.keys()),
            "prop_ids": list(graph.props.keys()),
            "anchor_image": graph.anchor_image,
            "motion_clip": graph.motion_clip,
            "retrieval": dict(graph.retrieval),
            "edit": dict(graph.edit),
            "element_ids": list(graph.elements.keys()),
            "engine": studio_out.engine,
            "studio": studio_out.studio_meta,
            "thumbnail_plan": {
                "enabled": thumb_plan.enabled,
                "gate_passed": thumb_plan.gate_passed,
                "pending_count": thumb_plan.pending_count,
                "specs": [
                    {
                        "shot_id": t.shot_id,
                        "shot_index": t.shot_index,
                        "frame_role": t.frame_role,
                        "prompt": t.prompt,
                        "width": t.width,
                        "height": t.height,
                        "approved": t.approved,
                    }
                    for t in thumb_plan.specs
                ],
            },
            "continuity": {
                "ok": continuity_report.ok,
                "issues": [
                    {
                        "level": i.level,
                        "code": i.code,
                        "message": i.message,
                        "shot_id": i.shot_id,
                        "related_shot_id": i.related_shot_id,
                    }
                    for i in continuity_report.issues
                ],
            },
            "frontier": frontier_out.metadata,
            "frontier_issues": list(frontier_out.issues),
        },
    )
    gate_msgs = thumbnail_gate_issues(thumb_plan)
    if gate_msgs and thumb_cfg.gate == "require_approval":
        raise ValueError("Thumbnail gate blocked:\n  - " + "\n  - ".join(gate_msgs))
    return CompiledScene(
        graph=graph,
        plan=plan,
        segment_overrides=overrides,
        control_plans=_compile_all_control_plans(graph, shots, shot_specs, global_neg),
    )


def _input_by_id(graph: SceneGraph) -> Dict[str, Any]:
    return {getattr(x, "id", ""): x for x in graph.inputs}


def _resolve_bindings_for_shot(
    graph: SceneGraph,
    shot: ShotNode,
    shot_spec: ShotSpec,
) -> List[Any]:
    from .controls import ControlMode, InputBinding

    bindings: List[InputBinding] = []
    inputs = _input_by_id(graph)
    seen: set[str] = set()

    def _add(entity_id: str, ent: EntityDef, shot_bind: Optional[Dict[str, Any]] = None) -> None:
        if entity_id in seen:
            return
        seen.add(entity_id)
        shot_bind = shot_bind or {}
        inp = inputs.get(ent.bind_input) if ent.bind_input else None
        image = ent.reference_image or (getattr(inp, "image", "") if inp else "")
        video = getattr(inp, "video", "") if inp else ""
        element_id = ent.bind_element or str(shot_bind.get("bind_element") or shot_bind.get("element") or "")
        bind_subject = False
        if element_id and element_id in graph.elements:
            from .elements import ElementDef as _El

            el: _El = graph.elements[element_id]
            work = Path(graph.raw.get("_work_dir") or "runs/video")
            from .elements import resolve_element_images

            refs = resolve_element_images(el, work)
            if refs and not image:
                image = refs[0]
            if el.video_ref and not video:
                video = el.video_ref
            bind_subject = el.bind_subject
        element_text_hint = ""
        if element_id and element_id in graph.elements:
            element_text_hint = graph.elements[element_id].text_hint or ""
        for eid in shot.bind_elements:
            if eid in graph.elements and eid not in seen:
                el2 = graph.elements[eid]
                work = Path(graph.raw.get("_work_dir") or "runs/video")
                from .elements import resolve_element_images

                refs2 = resolve_element_images(el2, work)
                if refs2:
                    image = image or refs2[0]
                if el2.video_ref:
                    video = video or el2.video_ref
                bind_subject = bind_subject or el2.bind_subject
        image_role = shot_bind.get("provides") or ent.image_role or (getattr(inp, "provides", "") if inp else "")
        text_role = (
            shot_bind.get("text")
            or shot_bind.get("text_changes")
            or ent.text_role
            or element_text_hint
            or ent.description
            or (getattr(inp, "text_changes", "") if inp else "")
        )
        ctrl_raw = (
            shot_bind.get("control")
            or ent.control
            or (getattr(inp, "control", ControlMode.TRANSFORM).value if inp else "transform")
        )
        try:
            ctrl = ControlMode(str(ctrl_raw).lower())
        except ValueError:
            ctrl = ControlMode.TRANSFORM
        if ent.lock:
            ctrl = ControlMode.LOCK
        if bind_subject:
            ctrl = ControlMode.IDENTITY
        strength = float(
            shot_bind.get("reference_strength")
            or ent.reference_strength
            or (getattr(inp, "reference_strength", 0.8) if inp else 0.8)
        )
        auto_rig = bool(
            shot_bind.get("auto_rig") or ent.auto_rig or (getattr(inp, "auto_rig", False) if inp else False)
        )
        part = str(shot_bind.get("part") or ent.part or (getattr(inp, "part", "") if inp else ""))
        mask = str(shot_bind.get("mask") or shot_bind.get("mask_path") or ent.mask_path or "")
        rig_json = ""
        if auto_rig and image:
            from .auto_rig import auto_rig_character, write_rig_box_layout

            rig = auto_rig_character(
                entity_id,
                image,
                text_by_part=ent.text_by_part,
                lock_parts=[k for k, v in ent.text_by_part.items() if "lock" in v.lower() or "preserve" in v.lower()],
            )
            rig_path = Path(graph.raw.get("_work_dir") or "runs/video") / f"rig_{entity_id}.json"
            write_rig_box_layout(rig, rig_path, global_prompt=shot_spec.prompt)
            rig_json = str(rig_path)
        bindings.append(
            InputBinding(
                entity_id=entity_id,
                image=image,
                video=video,
                image_role=str(image_role),
                text_role=str(text_role),
                control=ctrl,
                reference_strength=strength,
                auto_rig=auto_rig,
                part=part,
                mask_path=mask,
                negative=ent.negative,
                rig_json=rig_json,
            )
        )

    for cid in shot.characters:
        ent = graph.cast.get(cid)
        if ent:
            _add(cid, ent, shot.bindings.get(cid) if isinstance(shot.bindings.get(cid), dict) else None)
    for pid in shot.objects:
        ent = graph.props.get(pid)
        if ent:
            _add(pid, ent, shot.bindings.get(pid) if isinstance(shot.bindings.get(pid), dict) else None)

    for iid, inp in inputs.items():
        if iid in seen:
            continue
        fake = EntityDef(
            id=iid,
            reference_image=getattr(inp, "image", ""),
            image_role=getattr(inp, "provides", ""),
            text_role=getattr(inp, "text_changes", ""),
            control=getattr(inp, "control", ControlMode.TRANSFORM).value,
            auto_rig=getattr(inp, "auto_rig", False),
            part=getattr(inp, "part", ""),
            negative=getattr(inp, "negative", ""),
        )
        _add(iid, fake)

    return bindings


def _compile_all_control_plans(
    graph: SceneGraph,
    shots: List[ShotNode],
    shot_specs: List[ShotSpec],
    global_neg: str,
) -> List[Any]:
    from .controls import compile_shot_control_plan

    plans = []
    for sh, spec in zip(shots, shot_specs):
        bindings = _resolve_bindings_for_shot(graph, sh, spec)
        cp = compile_shot_control_plan(
            shot_id=sh.id,
            shot_index=spec.index,
            base_prompt=spec.prompt,
            base_negative=_merge_csv(spec.negative, global_neg),
            bindings=bindings,
            global_init_image=graph.anchor_image,
        )
        if cp.init_strength and sh.edit_strength:
            cp.init_strength = float(sh.edit_strength)
        _apply_element_refs_to_plan(graph, sh, cp)
        plans.append(cp)
        # Replace shot prompt with fully merged control prompt
        spec.prompt = cp.positive_prompt
        spec.negative = cp.negative_prompt
    return plans


def _apply_element_refs_to_plan(graph: SceneGraph, shot: ShotNode, cp: Any) -> None:
    from .elements import ElementsLibrary, compile_element_refs

    work = graph.raw.get("_work_dir") or "runs/video"
    lib = ElementsLibrary(elements=dict(graph.elements))
    ids: List[str] = list(shot.bind_elements)
    for cid in shot.characters:
        ent = graph.cast.get(cid)
        if ent and ent.bind_element:
            ids.append(ent.bind_element)
    ids = list(dict.fromkeys(ids))
    if not ids:
        return
    imgs, weights, vids = compile_element_refs(lib, ids, work)
    if len(imgs) > 1 and "--style-ref" not in cp.sample_extra_args:
        cp.sample_extra_args.extend(["--style-ref", ",".join(weights)])
    if vids:
        cp.metadata["element_video_refs"] = vids
    if imgs:
        cp.metadata["element_images"] = imgs
    if any(graph.elements.get(i) and getattr(graph.elements[i], "bind_subject", False) for i in ids):
        cp.metadata["bind_subject"] = True


def compile_scene_file(path: str | Path) -> CompiledScene:
    return compile_scene_graph(load_scene_graph(path))

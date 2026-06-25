"""Frontier compiler — novel SDX video subsystems merged into shot enrichment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = [
    "ShotEnrichment",
    "FrontierCompileResult",
    "compile_frontier_layers",
    "list_frontier_modules",
]


@dataclass(slots=True)
class ShotEnrichment:
    shot_id: str
    shot_index: int
    prompt_suffix: str = ""
    negative_suffix: str = ""
    duration_delta: float = 0.0
    edit_overrides: Dict[str, Any] = field(default_factory=dict)
    effects: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FrontierCompileResult:
    enrichments: Dict[str, ShotEnrichment] = field(default_factory=dict)
    global_edit: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)


def _block(data: Mapping[str, Any], key: str) -> Any:
    if key in data:
        return data[key]
    frontier = data.get("frontier")
    if isinstance(frontier, Mapping) and key in frontier:
        return frontier[key]
    return None


def _merge_enrichment(store: Dict[str, ShotEnrichment], shot_id: str, shot_index: int) -> ShotEnrichment:
    if shot_id not in store:
        store[shot_id] = ShotEnrichment(shot_id=shot_id, shot_index=shot_index)
    return store[shot_id]


def _append_suffix(en: ShotEnrichment, pos: str = "", neg: str = "") -> None:
    if pos and pos.lower() not in en.prompt_suffix.lower():
        en.prompt_suffix = f"{en.prompt_suffix}, {pos}".strip(", ")
    if neg and neg.lower() not in en.negative_suffix.lower():
        en.negative_suffix = f"{en.negative_suffix}, {neg}".strip(", ")


def compile_frontier_layers(
    data: Mapping[str, Any],
    shots: Sequence[Any],
    *,
    scene_prompt: str = "",
    duration_sec: float = 6.0,
    genre: str = "",
    cast: Optional[Mapping[str, Any]] = None,
    props: Optional[Mapping[str, Any]] = None,
) -> FrontierCompileResult:
    """Apply all frontier/novel layers to shots."""
    from .anticipation_windup import parse_anticipation_config, plan_anticipation_windups
    from .breath_cadence import parse_breath_config, plan_breath_cadence
    from .causal_events import apply_causal_ripples, parse_causal_rules
    from .counterfactual_beats import build_counterfactual_plan, parse_counterfactuals
    from .diegetic_focus import parse_focus_config, plan_diegetic_focus
    from .kinetic_continuity import parse_kinetic_config, track_kinetic_ledger
    from .mise_en_scene import compose_all_shots, parse_mise_config
    from .motif_tracker import audit_motif_haunting, parse_motifs
    from .narrative_tension import parse_tension_curve, sample_tension_for_shots
    from .semantic_gravity import build_gravity_field, gravity_edit_overrides, parse_semantic_gravity
    from .temporal_echo import parse_echo_config, plan_temporal_echoes

    store: Dict[str, ShotEnrichment] = {}
    meta: Dict[str, Any] = {}
    issues: List[str] = []
    global_edit: Dict[str, Any] = {}

    cast = cast or {}
    props = props or {}

    # 1. Narrative Tension Thermostat
    tension_raw = _block(data, "tension") or _block(data, "tension_curve")
    curve = parse_tension_curve(tension_raw, genre=genre or str((data.get("studio") or {}).get("genre") or ""))
    tension_profiles: List[Any] = []
    if curve:
        tension_profiles = sample_tension_for_shots(curve, shots, total_duration=duration_sec)
        meta["tension_curve"] = [{"shot_id": p.shot_id, "tension": p.tension} for p in tension_profiles]
        for p in tension_profiles:
            en = _merge_enrichment(store, p.shot_id, p.shot_index)
            _append_suffix(en, p.prompt_suffix, p.negative_suffix)
            en.edit_overrides.update(p.edit_overrides)
            en.metadata["tension"] = p.tension

    # 2. Semantic Gravity
    grav_raw = _block(data, "semantic_gravity") or _block(data, "gravity")
    weights = parse_semantic_gravity(grav_raw)
    if weights or cast or props:
        field_g = build_gravity_field(cast, props, weights=weights, shots=shots)
        global_edit.update(gravity_edit_overrides(field_g))
        meta["semantic_gravity"] = {
            "dominant": field_g.dominant_id,
            "wells": [{"id": w.entity_id, "weight": w.weight} for w in field_g.wells],
        }

    # 3. Causal Ripple Engine
    causal_raw = _block(data, "causality") or _block(data, "causal_events")
    rules = parse_causal_rules(causal_raw or {})
    ripples = apply_causal_ripples(shots, rules, use_builtins=causal_raw is not None or bool(_block(data, "frontier")))
    if ripples:
        meta["causal_ripples"] = [
            {"shot_id": r.shot_id, "trigger": r.trigger, "effects": r.injected_effects} for r in ripples
        ]
        shot_index = {str(getattr(s, "id", "")): i for i, s in enumerate(shots)}
        for r in ripples:
            en = _merge_enrichment(store, r.shot_id, shot_index.get(r.shot_id, 0))
            _append_suffix(en, r.prompt_suffix)
            en.effects.extend(r.injected_effects)
            if r.camera_hint:
                en.metadata["causal_camera"] = r.camera_hint

    # 4. Motif Haunting
    motifs = parse_motifs(_block(data, "motifs") or _block(data, "motif_haunting") or {})
    if motifs:
        haunt = audit_motif_haunting(motifs, shots)
        meta["motif_haunting"] = {
            "ok": haunt.ok,
            "unresolved": haunt.unresolved,
            "appearances": haunt.appearances,
        }
        if haunt.unresolved:
            issues.extend([f"motif_unresolved:{m}" for m in haunt.unresolved])
        shot_index = {str(getattr(s, "id", "")): i for i, s in enumerate(shots)}
        for sid, frag in haunt.injections.items():
            en = _merge_enrichment(store, sid, shot_index.get(sid, 0))
            _append_suffix(en, frag)

    # 5. Mise-en-scène Grammar
    mise_cfg = parse_mise_config(_block(data, "mise_en_scene") or _block(data, "composition"))
    compositions = compose_all_shots(shots, mise_cfg)
    if compositions:
        meta["mise_en_scene"] = [{"shot_id": c.shot_id, "grammar": c.grammar_key} for c in compositions]
        shot_index = {str(getattr(s, "id", "")): i for i, s in enumerate(shots)}
        for c in compositions:
            en = _merge_enrichment(store, c.shot_id, shot_index.get(c.shot_id, 0))
            _append_suffix(en, c.prompt_suffix, c.negative_suffix)

    # 6. Narrative Spotlight (diegetic focus)
    focus_cfg = parse_focus_config(_block(data, "diegetic_focus") or _block(data, "narrative_spotlight"))
    focus_plans = plan_diegetic_focus(shots, focus_cfg, cast)
    if focus_plans:
        meta["diegetic_focus"] = [{"shot_id": f.shot_id, "heroes": f.in_focus} for f in focus_plans]
        shot_index = {str(getattr(s, "id", "")): i for i, s in enumerate(shots)}
        for f in focus_plans:
            en = _merge_enrichment(store, f.shot_id, shot_index.get(f.shot_id, 0))
            _append_suffix(en, f.prompt_suffix)
            en.edit_overrides.update(f.edit_overrides)

    # 7. Kinetic Ledger
    kin_cfg = parse_kinetic_config(_block(data, "kinetic") or _block(data, "kinetic_ledger"))
    ledger = track_kinetic_ledger(shots, config=kin_cfg)
    if ledger.states:
        meta["kinetic_ledger"] = [
            {"shot_index": i, "energy": s.energy, "vertical": s.vertical, "verb": s.verb}
            for i, s in enumerate(ledger.states)
        ]
        for iss in ledger.issues:
            issues.append(f"[{iss.level}] {iss.code}: {iss.message}")
        shot_index = {str(getattr(s, "id", "")): i for i, s in enumerate(shots)}
        for sid, patch in ledger.shot_prompt_patches.items():
            en = _merge_enrichment(store, sid, shot_index.get(sid, 0))
            _append_suffix(en, patch)

    # 8. Anticipation Borrow
    ant_cfg = parse_anticipation_config(_block(data, "anticipation") or _block(data, "windup"))
    ant_plan = plan_anticipation_windups(shots, config=ant_cfg)
    if ant_plan.enabled and ant_plan.windups:
        meta["anticipation_borrow"] = {
            "total_borrowed_sec": ant_plan.total_borrowed_sec,
            "windups": [{"shot_id": w.shot_id, "verb": w.verb, "borrow_sec": w.borrow_sec} for w in ant_plan.windups],
        }
        shot_index = {str(getattr(s, "id", "")): i for i, s in enumerate(shots)}
        for w in ant_plan.windups:
            en = _merge_enrichment(store, w.shot_id, shot_index.get(w.shot_id, 0))
            _append_suffix(en, w.windup_prompt)
            en.duration_delta += w.borrow_sec
            en.metadata["windup_anchor"] = w.keyframe_anchor
            en.edit_overrides.setdefault("motion_beat_keyframes", True)

    # 9. Temporal Echo
    echo_cfg = parse_echo_config(_block(data, "temporal_echo") or _block(data, "echo"))
    echo_plan = plan_temporal_echoes(shots, echo_cfg)
    if echo_plan.links:
        meta["temporal_echo"] = [
            {"shot_id": link.shot_id, "echoes": link.echoes_shot_id, "strength": link.echo_strength}
            for link in echo_plan.links
        ]
        shot_index = {str(getattr(s, "id", "")): i for i, s in enumerate(shots)}
        for link in echo_plan.links:
            en = _merge_enrichment(store, link.shot_id, shot_index.get(link.shot_id, 0))
            _append_suffix(en, link.prompt_suffix)

    # 10. Breath Cadence
    breath_cfg = parse_breath_config(_block(data, "breath_cadence") or _block(data, "breath"))
    tension_map = {p.shot_id: p.tension for p in tension_profiles} if tension_profiles else {}
    breaths = plan_breath_cadence(shots, config=breath_cfg, tension_by_shot=tension_map)
    if breaths:
        meta["breath_cadence"] = [{"shot_id": b.shot_id, "bpm": b.bpm, "amplitude": b.amplitude} for b in breaths]
        shot_index = {str(getattr(s, "id", "")): i for i, s in enumerate(shots)}
        for b in breaths:
            en = _merge_enrichment(store, b.shot_id, shot_index.get(b.shot_id, 0))
            _append_suffix(en, b.prompt_suffix)

    # 11. What-If Timeline
    cf_raw = _block(data, "counterfactuals") or _block(data, "what_if")
    branches, strategy = parse_counterfactuals(cf_raw)
    cf_plan = build_counterfactual_plan(shots, branches, merge_strategy=strategy)
    if cf_plan.branches:
        meta["counterfactuals"] = {
            "merge_strategy": cf_plan.merge_strategy,
            "branches": [
                {
                    "id": b.id,
                    "parent_shot": b.parent_shot_id,
                    "label": b.branch_label,
                    "alt_prompt": b.alt_prompt,
                    "probability": b.probability,
                }
                for b in cf_plan.branches
            ],
        }

    shot_index = {str(getattr(s, "id", "")): i for i, s in enumerate(shots)}
    tension_map = {p.shot_id: p.tension for p in tension_profiles} if tension_profiles else {}

    # 12. Screen Direction Lock
    from .screen_direction import parse_screen_direction_config, track_screen_direction

    sd_cfg = parse_screen_direction_config(_block(data, "screen_direction"))
    if sd_cfg.get("enabled"):
        sd_rep = track_screen_direction(shots, config=sd_cfg)
        meta["screen_direction"] = sd_rep.directions
        for iss in sd_rep.issues:
            issues.append(f"[{iss.level}] {iss.code}: {iss.message}")
        for sid, patch in sd_rep.prompt_patches.items():
            en = _merge_enrichment(store, sid, shot_index.get(sid, 0))
            _append_suffix(en, patch)

    # 13. Offscreen Space Map
    from .offscreen_space import parse_offscreen_map, plan_offscreen_space

    os_map = parse_offscreen_map(_block(data, "offscreen") or _block(data, "offscreen_map") or {})
    if os_map or any(getattr(s, "offscreen", None) for s in shots):
        os_plans = plan_offscreen_space(shots, os_map)
        meta["offscreen_space"] = [{"shot_id": p.shot_id, "zones": [e.zone for e in p.events]} for p in os_plans]
        for p in os_plans:
            en = _merge_enrichment(store, p.shot_id, shot_index.get(p.shot_id, 0))
            _append_suffix(en, p.prompt_suffix)

    # 14. Emotional Contagion
    from .emotional_contagion import parse_contagion_config, plan_emotional_contagion

    contagion_cfg = parse_contagion_config(_block(data, "emotional_contagion") or _block(data, "contagion"))
    contagion_plans = plan_emotional_contagion(shots, config=contagion_cfg)
    if contagion_plans:
        meta["emotional_contagion"] = [{"shot_id": c.shot_id, "emotion": c.source_emotion} for c in contagion_plans]
        for c in contagion_plans:
            en = _merge_enrichment(store, c.shot_id, shot_index.get(c.shot_id, 0))
            _append_suffix(en, c.prompt_suffix)

    # 15. Material Memory
    from .material_memory import parse_material_config, track_material_memory

    mat_cfg = parse_material_config(_block(data, "material_memory") or _block(data, "materials"))
    mat_initial = {}
    if isinstance(_block(data, "material_memory"), Mapping):
        mat_initial = _block(data, "material_memory").get("initial") or {}
    mat_rep = track_material_memory(
        shots, config=mat_cfg, initial=mat_initial if isinstance(mat_initial, Mapping) else {}
    )
    if mat_rep.timeline:
        meta["material_memory"] = mat_rep.timeline
        for iss in mat_rep.issues:
            issues.append(f"[{iss.level}] {iss.code}: {iss.message}")
        for sid, patch in mat_rep.prompt_injections.items():
            en = _merge_enrichment(store, sid, shot_index.get(sid, 0))
            _append_suffix(en, patch)

    # 16. Camera Empathy
    from .camera_empathy import parse_empathy_config, plan_camera_empathy

    empathy_cfg = parse_empathy_config(_block(data, "camera_empathy") or _block(data, "empathy"))
    for sid, move in plan_camera_empathy(shots, config=empathy_cfg, tension_by_shot=tension_map):
        en = _merge_enrichment(store, sid, shot_index.get(sid, 0))
        _append_suffix(en, move.camera_prompt, move.negative)
        en.edit_overrides.update(move.edit_overrides)
        en.metadata["camera_empathy"] = move.emotion

    # 17. Weather Inertia
    from .weather_inertia import parse_weather_config, track_weather_inertia

    wx_cfg = parse_weather_config(_block(data, "weather_inertia") or _block(data, "weather"))
    if wx_cfg.get("enabled"):
        wx_rep = track_weather_inertia(shots, config=wx_cfg)
        meta["weather_inertia"] = wx_rep.timeline
        for iss in wx_rep.issues:
            issues.append(f"[{iss.level}] {iss.code}: {iss.message}")
        for sid, patch in wx_rep.prompt_injections.items():
            en = _merge_enrichment(store, sid, shot_index.get(sid, 0))
            _append_suffix(en, patch)

    # 18. Narrative Debt
    from .narrative_debt import audit_narrative_debt, parse_narrative_threads

    threads = parse_narrative_threads(_block(data, "narrative_threads") or _block(data, "plot_threads") or {})
    if threads:
        debt = audit_narrative_debt(threads, shots)
        meta["narrative_debt"] = {"ok": debt.ok, "unpaid": debt.unpaid, "resolved": debt.resolved_threads}
        if debt.unpaid:
            issues.extend([f"narrative_debt_unpaid:{t}" for t in debt.unpaid])
        for sid, patch in debt.injections.items():
            en = _merge_enrichment(store, sid, shot_index.get(sid, 0))
            _append_suffix(en, patch)

    # 19. Chromatic Story Arc
    from .chromatic_arc import parse_chromatic_arc, sample_chromatic_for_shots

    chroma_arc = parse_chromatic_arc(_block(data, "chromatic_arc") or _block(data, "color_script"))
    if chroma_arc:
        chroma_beats = sample_chromatic_for_shots(chroma_arc, shots, total_duration=duration_sec)
        meta["chromatic_arc"] = [{"shot_id": b.shot_id, "palette": b.palette_key} for b in chroma_beats]
        for b in chroma_beats:
            en = _merge_enrichment(store, b.shot_id, shot_index.get(b.shot_id, 0))
            _append_suffix(en, b.prompt_suffix)
            if b.post_grade:
                en.edit_overrides["post_grade"] = b.post_grade

    # 20. Stinger Frames
    from .stinger_frames import parse_stinger_config, plan_stinger_frames

    stinger_cfg = parse_stinger_config(_block(data, "stinger_frames") or _block(data, "stingers"))
    for spec in plan_stinger_frames(shots, config=stinger_cfg):
        en = _merge_enrichment(store, spec.shot_id, shot_index.get(spec.shot_id, 0))
        _append_suffix(en, spec.prompt_suffix)
        en.edit_overrides.update(spec.edit_overrides)
        en.metadata["stinger_frames"] = spec.frame_count

    # 21. Witness Lens
    from .witness_lens import parse_witness_config, plan_witness_lens

    witness_cfg = parse_witness_config(_block(data, "witness_lens") or _block(data, "witness"))
    witness_plans = plan_witness_lens(shots, config=witness_cfg)
    if witness_plans:
        meta["witness_lens"] = [{"shot_id": w.shot_id, "lens": w.lens} for w in witness_plans]
        for w in witness_plans:
            en = _merge_enrichment(store, w.shot_id, shot_index.get(w.shot_id, 0))
            _append_suffix(en, w.prompt_suffix, w.negative_suffix)

    # 22. Silence Map
    from .silence_map import parse_silence_map, plan_silence_beats

    silence_raw = _block(data, "silence_map") or _block(data, "silence")
    silence_map = parse_silence_map(silence_raw if isinstance(silence_raw, Mapping) else {})
    silence_beats = plan_silence_beats(shots, silence_map)
    if silence_beats:
        meta["silence_map"] = [{"shot_id": b.shot_id, "duration_sec": b.duration_sec} for b in silence_beats]
        for b in silence_beats:
            en = _merge_enrichment(store, b.shot_id, shot_index.get(b.shot_id, 0))
            _append_suffix(en, b.prompt_suffix)
            en.metadata["silence_beat"] = b.duration_sec

    # 23. Scar Timeline
    from .scar_timeline import parse_scar_config, track_scar_timeline

    scar_cfg = parse_scar_config(_block(data, "scar_timeline") or _block(data, "injuries"))
    scar_rep = track_scar_timeline(shots, config=scar_cfg)
    if scar_rep.by_shot:
        meta["scar_timeline"] = scar_rep.by_shot
        for sid, patch in scar_rep.prompt_injections.items():
            if patch:
                en = _merge_enrichment(store, sid, shot_index.get(sid, 0))
                _append_suffix(en, patch)

    # 24. Parallax Depth Budget
    from .parallax_budget import parse_parallax_config, plan_parallax_budget

    parallax_cfg = parse_parallax_config(_block(data, "parallax") or _block(data, "parallax_budget"))
    parallax_plans = plan_parallax_budget(shots, config=parallax_cfg)
    if parallax_plans:
        meta["parallax_budget"] = [{"shot_id": p.shot_id, "layers": len(p.layers)} for p in parallax_plans]
        for p in parallax_plans:
            en = _merge_enrichment(store, p.shot_id, shot_index.get(p.shot_id, 0))
            _append_suffix(en, p.prompt_suffix)
            en.edit_overrides.update(p.edit_overrides)

    return FrontierCompileResult(enrichments=store, global_edit=global_edit, metadata=meta, issues=issues)


def list_frontier_modules() -> List[Dict[str, str]]:
    return [
        {
            "id": "tension_thermostat",
            "module": "narrative_tension.py",
            "summary": "Emotion curve drives camera/motion/grade",
        },
        {
            "id": "causal_ripple",
            "module": "causal_events.py",
            "summary": "Story events auto-trigger FX and camera reactions",
        },
        {
            "id": "anticipation_borrow",
            "module": "anticipation_windup.py",
            "summary": "Steals shot time for pre-action wind-up",
        },
        {"id": "motif_haunting", "module": "motif_tracker.py", "summary": "Symbols must appear until resolved"},
        {"id": "mise_en_scene", "module": "mise_en_scene.py", "summary": "Lead room/headroom composition grammar"},
        {"id": "narrative_spotlight", "module": "diegetic_focus.py", "summary": "Fog-of-narrative detail budget"},
        {"id": "kinetic_ledger", "module": "kinetic_continuity.py", "summary": "Cross-shot energy conservation"},
        {"id": "temporal_echo", "module": "temporal_echo.py", "summary": "Shots rhyme visually with earlier beats"},
        {
            "id": "semantic_gravity",
            "module": "semantic_gravity.py",
            "summary": "Narrative weight → drift repair priority",
        },
        {"id": "breath_cadence", "module": "breath_cadence.py", "summary": "Micro motion synced to tension"},
        {"id": "what_if_timeline", "module": "counterfactual_beats.py", "summary": "Parallel alternate shot branches"},
        {
            "id": "screen_direction_lock",
            "module": "screen_direction.py",
            "summary": "Movement vector 180° rule across cuts",
        },
        {
            "id": "offscreen_space",
            "module": "offscreen_space.py",
            "summary": "Off-frame zones inject audio/light/reaction",
        },
        {
            "id": "emotional_contagion",
            "module": "emotional_contagion.py",
            "summary": "Crowd inherits hero emotional state",
        },
        {"id": "material_memory", "module": "material_memory.py", "summary": "Wet/mud/blood persists until cleaned"},
        {
            "id": "camera_empathy",
            "module": "camera_empathy.py",
            "summary": "Camera retreats/lingers/trembles with subject",
        },
        {"id": "weather_inertia", "module": "weather_inertia.py", "summary": "Climate cannot flip between cuts"},
        {
            "id": "narrative_debt",
            "module": "narrative_debt.py",
            "summary": "Plot threads must pay off by deadline shot",
        },
        {"id": "chromatic_arc", "module": "chromatic_arc.py", "summary": "Color script evolves with story emotion"},
        {"id": "stinger_frames", "module": "stinger_frames.py", "summary": "Impact/sakuga emphasis on last N frames"},
        {
            "id": "witness_lens",
            "module": "witness_lens.py",
            "summary": "Who watches changes framing (child/CCTV/lover)",
        },
        {"id": "silence_map", "module": "silence_map.py", "summary": "Deliberate silence beats for horror/comedy"},
        {"id": "scar_timeline", "module": "scar_timeline.py", "summary": "Injuries accumulate on characters"},
        {"id": "parallax_budget", "module": "parallax_budget.py", "summary": "2.5D layer separation for compositing"},
    ]

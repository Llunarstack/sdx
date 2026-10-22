"""Frontier compiler — novel SDX video subsystems merged into shot enrichment.

Also holds what used to be: anticipation_windup, breath_cadence, camera_empathy, causal_events, chromatic_arc, counterfactual_beats, diegetic_focus, emotional_contagion, kinetic_continuity, material_memory, mise_en_scene, motif_tracker, narrative_debt, narrative_tension, offscreen_space, parallax_budget, scar_timeline, screen_direction, semantic_gravity, silence_map, stinger_frames, temporal_echo, weather_inertia, witness_lens.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from config.defaults._lexicon import merge_csv_unique

# ==================================================================================================
# frontier_compiler
# ==================================================================================================


def parse_toggle(raw: Any, *, default_on: bool = False, **defaults: Any) -> dict[str, Any]:
    """Config for one frontier block.

    ``None`` gives ``default_on``; any other non-mapping is truth-tested. A mapping is enabled unless it says
    ``"enabled": false``, and each ``defaults`` key is read from it and coerced to the default's type.
    """
    if not isinstance(raw, Mapping):
        return {"enabled": default_on if raw is None else bool(raw), **defaults}
    out: dict[str, Any] = {"enabled": bool(raw.get("enabled", True))}
    for key, default in defaults.items():
        value = raw.get(key)
        out[key] = default if value is None else type(default)(value)
    return out


@dataclass(slots=True)
class ShotEnrichment:
    shot_id: str
    shot_index: int
    prompt_suffix: str = ""
    negative_suffix: str = ""
    duration_delta: float = 0.0
    edit_overrides: dict[str, Any] = field(default_factory=dict)
    effects: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FrontierCompileResult:
    enrichments: dict[str, ShotEnrichment] = field(default_factory=dict)
    global_edit: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


def _block(data: Mapping[str, Any], key: str) -> Any:
    if key in data:
        return data[key]
    frontier = data.get("frontier")
    if isinstance(frontier, Mapping) and key in frontier:
        return frontier[key]
    return None


def _merge_enrichment(store: dict[str, ShotEnrichment], shot_id: str, shot_index: int) -> ShotEnrichment:
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
    cast: Mapping[str, Any] | None = None,
    props: Mapping[str, Any] | None = None,
) -> FrontierCompileResult:
    """Apply all frontier/novel layers to shots."""

    store: dict[str, ShotEnrichment] = {}
    meta: dict[str, Any] = {}
    issues: list[str] = []
    global_edit: dict[str, Any] = {}

    cast = cast or {}
    props = props or {}

    # 1. Narrative Tension Thermostat
    tension_raw = _block(data, "tension") or _block(data, "tension_curve")
    curve = parse_tension_curve(tension_raw, genre=genre or str((data.get("studio") or {}).get("genre") or ""))
    tension_profiles: list[Any] = []
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

    os_map = parse_offscreen_map(_block(data, "offscreen") or _block(data, "offscreen_map") or {})
    if os_map or any(getattr(s, "offscreen", None) for s in shots):
        os_plans = plan_offscreen_space(shots, os_map)
        meta["offscreen_space"] = [{"shot_id": p.shot_id, "zones": [e.zone for e in p.events]} for p in os_plans]
        for p in os_plans:
            en = _merge_enrichment(store, p.shot_id, shot_index.get(p.shot_id, 0))
            _append_suffix(en, p.prompt_suffix)

    # 14. Emotional Contagion

    contagion_cfg = parse_contagion_config(_block(data, "emotional_contagion") or _block(data, "contagion"))
    contagion_plans = plan_emotional_contagion(shots, config=contagion_cfg)
    if contagion_plans:
        meta["emotional_contagion"] = [{"shot_id": c.shot_id, "emotion": c.source_emotion} for c in contagion_plans]
        for c in contagion_plans:
            en = _merge_enrichment(store, c.shot_id, shot_index.get(c.shot_id, 0))
            _append_suffix(en, c.prompt_suffix)

    # 15. Material Memory

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

    empathy_cfg = parse_empathy_config(_block(data, "camera_empathy") or _block(data, "empathy"))
    for sid, move in plan_camera_empathy(shots, config=empathy_cfg, tension_by_shot=tension_map):
        en = _merge_enrichment(store, sid, shot_index.get(sid, 0))
        _append_suffix(en, move.camera_prompt, move.negative)
        en.edit_overrides.update(move.edit_overrides)
        en.metadata["camera_empathy"] = move.emotion

    # 17. Weather Inertia

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

    stinger_cfg = parse_stinger_config(_block(data, "stinger_frames") or _block(data, "stingers"))
    for spec in plan_stinger_frames(shots, config=stinger_cfg):
        en = _merge_enrichment(store, spec.shot_id, shot_index.get(spec.shot_id, 0))
        _append_suffix(en, spec.prompt_suffix)
        en.edit_overrides.update(spec.edit_overrides)
        en.metadata["stinger_frames"] = spec.frame_count

    # 21. Witness Lens

    witness_cfg = parse_witness_config(_block(data, "witness_lens") or _block(data, "witness"))
    witness_plans = plan_witness_lens(shots, config=witness_cfg)
    if witness_plans:
        meta["witness_lens"] = [{"shot_id": w.shot_id, "lens": w.lens} for w in witness_plans]
        for w in witness_plans:
            en = _merge_enrichment(store, w.shot_id, shot_index.get(w.shot_id, 0))
            _append_suffix(en, w.prompt_suffix, w.negative_suffix)

    # 22. Silence Map

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

    scar_cfg = parse_scar_config(_block(data, "scar_timeline") or _block(data, "injuries"))
    scar_rep = track_scar_timeline(shots, config=scar_cfg)
    if scar_rep.by_shot:
        meta["scar_timeline"] = scar_rep.by_shot
        for sid, patch in scar_rep.prompt_injections.items():
            if patch:
                en = _merge_enrichment(store, sid, shot_index.get(sid, 0))
                _append_suffix(en, patch)

    # 24. Parallax Depth Budget

    parallax_cfg = parse_parallax_config(_block(data, "parallax") or _block(data, "parallax_budget"))
    parallax_plans = plan_parallax_budget(shots, config=parallax_cfg)
    if parallax_plans:
        meta["parallax_budget"] = [{"shot_id": p.shot_id, "layers": len(p.layers)} for p in parallax_plans]
        for p in parallax_plans:
            en = _merge_enrichment(store, p.shot_id, shot_index.get(p.shot_id, 0))
            _append_suffix(en, p.prompt_suffix)
            en.edit_overrides.update(p.edit_overrides)

    return FrontierCompileResult(enrichments=store, global_edit=global_edit, metadata=meta, issues=issues)


def list_frontier_modules() -> list[dict[str, str]]:
    return [
        {
            "id": "tension_thermostat",
            "module": "frontier_compiler.py#narrative_tension",
            "summary": "Emotion curve drives camera/motion/grade",
        },
        {
            "id": "causal_ripple",
            "module": "frontier_compiler.py#causal_events",
            "summary": "Story events auto-trigger FX and camera reactions",
        },
        {
            "id": "anticipation_borrow",
            "module": "frontier_compiler.py#anticipation_windup",
            "summary": "Steals shot time for pre-action wind-up",
        },
        {
            "id": "motif_haunting",
            "module": "frontier_compiler.py#motif_tracker",
            "summary": "Symbols must appear until resolved",
        },
        {
            "id": "mise_en_scene",
            "module": "frontier_compiler.py#mise_en_scene",
            "summary": "Lead room/headroom composition grammar",
        },
        {
            "id": "narrative_spotlight",
            "module": "frontier_compiler.py#diegetic_focus",
            "summary": "Fog-of-narrative detail budget",
        },
        {
            "id": "kinetic_ledger",
            "module": "frontier_compiler.py#kinetic_continuity",
            "summary": "Cross-shot energy conservation",
        },
        {
            "id": "temporal_echo",
            "module": "frontier_compiler.py#temporal_echo",
            "summary": "Shots rhyme visually with earlier beats",
        },
        {
            "id": "semantic_gravity",
            "module": "frontier_compiler.py#semantic_gravity",
            "summary": "Narrative weight → drift repair priority",
        },
        {
            "id": "breath_cadence",
            "module": "frontier_compiler.py#breath_cadence",
            "summary": "Micro motion synced to tension",
        },
        {
            "id": "what_if_timeline",
            "module": "frontier_compiler.py#counterfactual_beats",
            "summary": "Parallel alternate shot branches",
        },
        {
            "id": "screen_direction_lock",
            "module": "frontier_compiler.py#screen_direction",
            "summary": "Movement vector 180° rule across cuts",
        },
        {
            "id": "offscreen_space",
            "module": "frontier_compiler.py#offscreen_space",
            "summary": "Off-frame zones inject audio/light/reaction",
        },
        {
            "id": "emotional_contagion",
            "module": "frontier_compiler.py#emotional_contagion",
            "summary": "Crowd inherits hero emotional state",
        },
        {
            "id": "material_memory",
            "module": "frontier_compiler.py#material_memory",
            "summary": "Wet/mud/blood persists until cleaned",
        },
        {
            "id": "camera_empathy",
            "module": "frontier_compiler.py#camera_empathy",
            "summary": "Camera retreats/lingers/trembles with subject",
        },
        {
            "id": "weather_inertia",
            "module": "frontier_compiler.py#weather_inertia",
            "summary": "Climate cannot flip between cuts",
        },
        {
            "id": "narrative_debt",
            "module": "frontier_compiler.py#narrative_debt",
            "summary": "Plot threads must pay off by deadline shot",
        },
        {
            "id": "chromatic_arc",
            "module": "frontier_compiler.py#chromatic_arc",
            "summary": "Color script evolves with story emotion",
        },
        {
            "id": "stinger_frames",
            "module": "frontier_compiler.py#stinger_frames",
            "summary": "Impact/sakuga emphasis on last N frames",
        },
        {
            "id": "witness_lens",
            "module": "frontier_compiler.py#witness_lens",
            "summary": "Who watches changes framing (child/CCTV/lover)",
        },
        {
            "id": "silence_map",
            "module": "frontier_compiler.py#silence_map",
            "summary": "Deliberate silence beats for horror/comedy",
        },
        {
            "id": "scar_timeline",
            "module": "frontier_compiler.py#scar_timeline",
            "summary": "Injuries accumulate on characters",
        },
        {
            "id": "parallax_budget",
            "module": "frontier_compiler.py#parallax_budget",
            "summary": "2.5D layer separation for compositing",
        },
    ]


# ==================================================================================================
# anticipation_windup: Anticipation Borrow — steals shot time before action verbs for Disney-style wind-up frames.
# ==================================================================================================

_ACTION_VERBS = (
    "jumps",
    "jump",
    "punches",
    "punch",
    "throws",
    "throw",
    "kicks",
    "kick",
    "runs",
    "run",
    "leaps",
    "leap",
    "swings",
    "swing",
    "slams",
    "slam",
    "explodes",
    "explode",
    "falls",
    "fall",
    "dives",
    "dive",
    "attacks",
    "attack",
    "strikes",
    "strike",
    "launches",
    "launch",
)


@dataclass(slots=True)
class WindupSpec:
    shot_id: str
    verb: str
    borrow_sec: float
    windup_prompt: str
    keyframe_anchor: str  # pre_action | action_peak


@dataclass(slots=True)
class AnticipationPlan:
    enabled: bool
    windups: list[WindupSpec] = field(default_factory=list)
    total_borrowed_sec: float = 0.0


def parse_anticipation_config(raw: Any) -> dict[str, Any]:
    return parse_toggle(raw, borrow_ratio=0.18, max_borrow_sec=0.45, min_shot_sec=1.0)


def _first_action_verb(prompt: str) -> str:
    p = (prompt or "").lower()
    for v in _ACTION_VERBS:
        if re.search(rf"\b{re.escape(v)}\b", p):
            return v
    return ""


_WINDUP_FRAGMENTS: dict[str, str] = {
    "jump": "anticipation crouch, coiled legs, wind-up before jump",
    "leap": "coiled stance, gathering energy before leap",
    "punch": "shoulder wind-up, fist pulled back, anticipation before punch",
    "kick": "planted foot, hip coil before kick",
    "throw": "arm drawn back, weight shift before throw",
    "run": "lean forward, preparatory step before sprint",
    "swing": "weapon drawn back, breath before swing",
    "slam": "raised arms, breath in before slam",
    "explode": "tension building, subtle rumble before explosion",
    "fall": "loss of balance, tipping anticipation before fall",
    "dive": "knees bend, arms ready before dive",
    "attack": "combat stance wind-up before attack",
    "strike": "coiled strike preparation",
    "launch": "charged posture before launch",
}


def plan_anticipation_windups(
    shots: Sequence[Any],
    *,
    config: Mapping[str, Any],
) -> AnticipationPlan:
    if not config.get("enabled"):
        return AnticipationPlan(enabled=False)
    ratio = float(config.get("borrow_ratio") or 0.18)
    max_borrow = float(config.get("max_borrow_sec") or 0.45)
    min_shot = float(config.get("min_shot_sec") or 1.0)

    windups: list[WindupSpec] = []
    total = 0.0
    for sh in shots:
        prompt = str(getattr(sh, "prompt", "") or "")
        verb = _first_action_verb(prompt)
        if not verb:
            continue
        dur = float(getattr(sh, "duration_sec", 0) or 0)
        if dur < min_shot:
            continue
        borrow = min(max_borrow, dur * ratio)
        frag = _WINDUP_FRAGMENTS.get(verb, f"anticipation wind-up before {verb}")
        windups.append(
            WindupSpec(
                shot_id=str(getattr(sh, "id", "")),
                verb=verb,
                borrow_sec=round(borrow, 3),
                windup_prompt=frag,
                keyframe_anchor="pre_action",
            )
        )
        total += borrow
    return AnticipationPlan(enabled=True, windups=windups, total_borrowed_sec=round(total, 3))


# ==================================================================================================
# breath_cadence: Breath Cadence — micro motion amplitude tied to tension (characters breathe with the scene).
# ==================================================================================================


@dataclass(slots=True)
class BreathCadence:
    shot_id: str
    bpm: float
    amplitude: float
    prompt_suffix: str


def parse_breath_config(raw: Any) -> dict[str, Any]:
    return parse_toggle(raw, base_bpm=14.0)


def plan_breath_cadence(
    shots: Sequence[Any],
    *,
    config: Mapping[str, Any],
    tension_by_shot: Mapping[str, float] | None = None,
) -> list[BreathCadence]:
    if not config.get("enabled"):
        return []
    base_bpm = float(config.get("base_bpm") or 14.0)
    tension_by_shot = tension_by_shot or {}
    out: list[BreathCadence] = []
    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        t = float(tension_by_shot.get(sid, 0.4))
        bpm = base_bpm + t * 18.0
        amp = 0.15 + t * 0.55
        if t < 0.35:
            frag = "subtle idle breathing, calm micro motion in chest and shoulders"
        elif t < 0.7:
            frag = "visible breathing rhythm, anxious micro sway"
        else:
            frag = "heavy rapid breathing, pronounced chest heave, trembling hands"
        out.append(BreathCadence(shot_id=sid, bpm=round(bpm, 1), amplitude=round(amp, 3), prompt_suffix=frag))
    return out


# ==================================================================================================
# camera_empathy: Camera Empathy — the camera has emotional attachment; it retreats, lingers, or trembles with subjects.
# ==================================================================================================


@dataclass(slots=True)
class EmpathyMove:
    emotion: str
    camera_prompt: str
    negative: str
    edit_overrides: dict[str, Any]


_EMPATHY_MAP = {
    "intimacy": EmpathyMove(
        "intimacy",
        "slow tender dolly inward, shallow depth, camera breathes with subject",
        "aggressive whip pans",
        {"camera_stabilize": True, "velocity_ease": True},
    ),
    "fear": EmpathyMove(
        "fear",
        "hesitant handheld retreat, slight delay following subject, negative space growing",
        "confident steady push in",
        {"camera_stabilize": False},
    ),
    "awe": EmpathyMove(
        "awe",
        "slow crane rise, camera stillness then subtle push, vast reveal",
        "chaotic shaky cam",
        {"camera_stabilize": True, "camera_stabilize_strength": 0.9},
    ),
    "rage": EmpathyMove(
        "rage",
        "aggressive handheld push, micro jitters synced to subject",
        "static detached framing",
        {"camera_stabilize": False, "velocity_ease": False},
    ),
    "loneliness": EmpathyMove(
        "loneliness",
        "slow pull back leaving subject small in frame, lingering hold",
        "tight crowded framing",
        {"velocity_ease": True, "velocity_ease_mode": "ease_out"},
    ),
    "suspense": EmpathyMove(
        "suspense",
        "almost imperceptible push in, camera holds breath",
        "fast cutting energy",
        {"keyframe_interval": 5},
    ),
}


def parse_empathy_config(raw: Any) -> dict[str, Any]:
    return parse_toggle(raw, default="")


def _infer_emotion(shot: Any, tension: float = 0.5) -> str:
    em = str(getattr(shot, "emotion", "") or "").lower()
    if em in _EMPATHY_MAP:
        return em
    p = str(getattr(shot, "prompt", "")).lower()
    for key in _EMPATHY_MAP:
        if key in p:
            return key
    if tension >= 0.75:
        return "suspense"
    if tension <= 0.25:
        return "loneliness"
    return "intimacy"


def plan_camera_empathy(
    shots: Sequence[Any],
    *,
    config: Mapping[str, Any],
    tension_by_shot: Mapping[str, float] | None = None,
) -> list[tuple[str, EmpathyMove]]:
    if not config.get("enabled"):
        return []
    tension_by_shot = tension_by_shot or {}
    out: list[tuple[str, EmpathyMove]] = []
    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        em = _infer_emotion(sh, tension_by_shot.get(sid, 0.5))
        move = _EMPATHY_MAP.get(em) or _EMPATHY_MAP.get(str(config.get("default") or "intimacy"))
        if move:
            out.append((sid, move))
    return out


# ==================================================================================================
# causal_events: Causal Ripple Engine — story events automatically trigger camera, FX, and prop reactions.
# ==================================================================================================

# Built-in physics-adjacent narrative causality (not pixel physics — story physics)
_BUILTIN_RULES: dict[str, list[str]] = {
    "explosion": ["camera_shake", "dust_cloud", "characters_flinch", "audio_rumble"],
    "gunshot": ["camera_jolt", "muzzle_flash", "echo_reverb"],
    "thunder": ["lightning_flash", "rain_intensify", "characters_glance_up"],
    "door_slam": ["camera_micro_shake", "dust_motes", "hanging_lamp_sway"],
    "car_crash": ["debris_spray", "camera_whiplash", "glass_shatter"],
    "magic_cast": ["light_pulse", "wind_gust", "particles_swirl"],
    "scream": ["birds_scatter", "silence_after", "characters_turn"],
    "rain_start": ["wet_surfaces", "fabric_darken", "footstep_splash"],
    "wind_gust": ["hair_motion", "cloth_flutter", "leaves_blow"],
    "footstep_close": ["floor_creak", "camera_subtle_push"],
}


@dataclass(slots=True)
class CausalRule:
    trigger: str
    effects: list[str]
    camera: str = ""
    props: dict[str, str] = field(default_factory=dict)
    delay_sec: float = 0.0


@dataclass(slots=True)
class CausalRipple:
    shot_id: str
    trigger: str
    injected_effects: list[str]
    prompt_suffix: str
    camera_hint: str
    props_patch: dict[str, str]


def parse_causal_rules(raw: Any) -> list[CausalRule]:
    rules: list[CausalRule] = []
    if isinstance(raw, Mapping):
        for trig, spec in raw.items():
            if isinstance(spec, list):
                rules.append(CausalRule(trigger=str(trig), effects=[str(x) for x in spec]))
            elif isinstance(spec, Mapping):
                rules.append(
                    CausalRule(
                        trigger=str(trig),
                        effects=[str(x) for x in (spec.get("then") or spec.get("effects") or [])],
                        camera=str(spec.get("camera") or ""),
                        props={str(k): str(v) for k, v in (spec.get("props") or {}).items()},
                        delay_sec=float(spec.get("delay_sec") or 0.0),
                    )
                )
    elif isinstance(raw, list):
        for row in raw:
            if not isinstance(row, Mapping):
                continue
            trig = str(row.get("when") or row.get("trigger") or "")
            effects = row.get("then") or row.get("effects") or []
            rules.append(
                CausalRule(
                    trigger=trig,
                    effects=[str(x) for x in effects] if isinstance(effects, list) else [str(effects)],
                    camera=str(row.get("camera") or ""),
                    props={str(k): str(v) for k, v in (row.get("props") or {}).items()},
                    delay_sec=float(row.get("delay_sec") or 0.0),
                )
            )
    return rules


def _trigger_patterns(trigger: str) -> list[re.Pattern[str]]:
    t = trigger.lower().replace("_", " ")
    return [re.compile(rf"\b{re.escape(t)}\b", re.I), re.compile(rf"\b{re.escape(trigger)}\b", re.I)]


def detect_triggers_in_prompt(prompt: str, rules: Sequence[CausalRule]) -> list[str]:
    found: list[str] = []
    p = prompt or ""
    for rule in rules:
        for pat in _trigger_patterns(rule.trigger):
            if pat.search(p):
                found.append(rule.trigger)
                break
    # Also scan built-ins when prompt matches common verbs
    for trig in _BUILTIN_RULES:
        if trig in found:
            continue
        for pat in _trigger_patterns(trig):
            if pat.search(p):
                found.append(trig)
                break
    return found


_EFFECT_PROMPTS: dict[str, str] = {
    "camera_shake": "handheld camera shake from impact",
    "camera_jolt": "sharp camera jolt",
    "camera_whiplash": "violent camera whiplash",
    "camera_micro_shake": "subtle camera vibration",
    "dust_cloud": "billowing dust cloud",
    "dust_motes": "dust motes in air",
    "characters_flinch": "characters flinch from blast",
    "muzzle_flash": "brief muzzle flash",
    "lightning_flash": "lightning illuminates scene",
    "rain_intensify": "rain intensifies",
    "wet_surfaces": "surfaces glistening wet",
    "hair_motion": "hair blown by wind",
    "cloth_flutter": "clothes flutter in gust",
    "particles_swirl": "swirling magical particles",
    "birds_scatter": "birds scatter from noise",
    "debris_spray": "debris spraying outward",
    "glass_shatter": "shattering glass fragments",
    "light_pulse": "pulsing magical light",
    "wind_gust": "strong wind gust through scene",
}


def apply_causal_ripples(
    shots: Sequence[Any],
    rules: Sequence[CausalRule],
    *,
    use_builtins: bool = True,
) -> list[CausalRipple]:
    merged_rules = list(rules)
    if use_builtins:
        existing = {r.trigger for r in merged_rules}
        for trig, effects in _BUILTIN_RULES.items():
            if trig not in existing:
                merged_rules.append(CausalRule(trigger=trig, effects=effects))

    ripples: list[CausalRipple] = []
    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        prompt = str(getattr(sh, "prompt", "") or "")
        triggers = detect_triggers_in_prompt(prompt, merged_rules)
        if not triggers:
            continue
        effects: set[str] = set()
        camera_hints: list[str] = []
        props_patch: dict[str, str] = {}
        for trig in triggers:
            for rule in merged_rules:
                if rule.trigger != trig:
                    continue
                effects.update(rule.effects)
                if rule.camera:
                    camera_hints.append(rule.camera)
                props_patch.update(rule.props)
            if use_builtins:
                effects.update(_BUILTIN_RULES.get(trig, []))

        prompt_bits = [_EFFECT_PROMPTS.get(e, e.replace("_", " ")) for e in sorted(effects)]
        ripples.append(
            CausalRipple(
                shot_id=sid,
                trigger=",".join(triggers),
                injected_effects=sorted(effects),
                prompt_suffix=", ".join(prompt_bits),
                camera_hint=", ".join(camera_hints),
                props_patch=props_patch,
            )
        )
    return ripples


# ==================================================================================================
# chromatic_arc: Chromatic Story Arc — color palette evolves with emotion independent of grade LUT.
# ==================================================================================================

_PALETTES: dict[str, str] = {
    "hope": "warm gold and soft green accents",
    "dread": "sickly yellow-green and deep violet shadows",
    "rage": "saturated crimson highlights, charcoal blacks",
    "grief": "desaturated blue-grey, muted skin tones",
    "wonder": "prismatic highlights, cyan-magenta split",
    "neutral": "balanced naturalistic palette",
    "nostalgia": "faded amber, lifted blacks, gentle sepia",
    "betrayal": "cold steel blue against warm false comfort",
}
# Frame grade (temporal.GRADES) applied on top of the prompt palette for the emotions that need it.
_GRADE_FOR = {"grief": "muted", "dread": "muted", "nostalgia": "cinematic"}


@dataclass(slots=True)
class ChromaticBeat:
    shot_id: str
    palette_key: str
    prompt_suffix: str
    post_grade: str


def parse_chromatic_arc(raw: Any) -> list[tuple[float, str]]:
    """Return list of (normalized_time, palette_key)."""
    pts: list[tuple[float, str]] = []
    if isinstance(raw, list):
        for i, v in enumerate(raw):
            if isinstance(v, str):
                pts.append((i / max(1, len(raw) - 1), v))
            elif isinstance(v, Mapping):
                pts.append((float(v.get("at") or 0), str(v.get("palette") or v.get("color") or "neutral")))
    elif isinstance(raw, Mapping):
        for k, v in raw.items():
            try:
                pts.append((float(k), str(v)))
            except ValueError:
                pts.append((0.0, str(v)))
    return sorted(pts, key=lambda x: x[0])


def _sample_palette(pts: list[tuple[float, str]], u: float) -> str:
    if not pts:
        return "neutral"
    u = max(0.0, min(1.0, u))
    if u <= pts[0][0]:
        return pts[0][1]
    if u >= pts[-1][0]:
        return pts[-1][1]
    for a, b in zip(pts, pts[1:]):
        if a[0] <= u <= b[0]:
            return a[1] if (u - a[0]) < (b[0] - u) else b[1]
    return pts[-1][1]


def sample_chromatic_for_shots(
    arc: list[tuple[float, str]],
    shots: Sequence[Any],
    *,
    total_duration: float,
) -> list[ChromaticBeat]:
    if not arc:
        return []
    dur_sum = sum(float(getattr(s, "duration_sec", 0) or 0) for s in shots)
    total = dur_sum if dur_sum > 0 else total_duration
    elapsed = 0.0
    out: list[ChromaticBeat] = []
    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        d = float(getattr(sh, "duration_sec", 0) or total / max(1, len(shots)))
        mid = (elapsed + d * 0.5) / max(1e-6, total)
        key = _sample_palette(arc, mid)
        frag = _PALETTES.get(key, f"{key} color palette")
        out.append(
            ChromaticBeat(
                shot_id=sid,
                palette_key=key,
                prompt_suffix=f"color script: {frag}",
                post_grade=_GRADE_FOR.get(key, ""),
            )
        )
        elapsed += d
    return out


# ==================================================================================================
# counterfactual_beats: What-If Timeline — counterfactual shot branches for parallel narrative exploration.
# ==================================================================================================


@dataclass(slots=True)
class CounterfactualBranch:
    id: str
    parent_shot_id: str
    branch_label: str
    alt_prompt: str
    alt_duration_sec: float = 0.0
    probability: float = 0.5


@dataclass(slots=True)
class CounterfactualPlan:
    branches: list[CounterfactualBranch] = field(default_factory=list)
    merge_strategy: str = "none"  # none | montage | choose_one


def parse_counterfactuals(raw: Any) -> tuple[list[CounterfactualBranch], str]:
    branches: list[CounterfactualBranch] = []
    strategy = "none"
    if isinstance(raw, Mapping):
        strategy = str(raw.get("merge_strategy") or raw.get("strategy") or "none")
        items = raw.get("branches") or raw.get("what_if") or []
    elif isinstance(raw, list):
        items = raw
    else:
        return [], strategy

    for i, row in enumerate(items):
        if not isinstance(row, Mapping):
            continue
        branches.append(
            CounterfactualBranch(
                id=str(row.get("id") or f"cf_{i}"),
                parent_shot_id=str(row.get("after_shot") or row.get("parent_shot") or row.get("shot") or ""),
                branch_label=str(row.get("label") or row.get("branch") or "alternate"),
                alt_prompt=str(row.get("prompt") or row.get("alt_prompt") or ""),
                alt_duration_sec=float(row.get("duration_sec") or row.get("duration") or 0.0),
                probability=float(row.get("probability") or row.get("weight") or 0.5),
            )
        )
    return branches, strategy


def build_counterfactual_plan(
    shots: Sequence[Any],
    branches: Sequence[CounterfactualBranch],
    *,
    merge_strategy: str = "none",
) -> CounterfactualPlan:
    if not branches:
        return CounterfactualPlan(merge_strategy=merge_strategy)
    shot_ids = {str(getattr(s, "id", "")) for s in shots}
    valid = [b for b in branches if not b.parent_shot_id or b.parent_shot_id in shot_ids]
    return CounterfactualPlan(branches=list(valid), merge_strategy=merge_strategy)


# ==================================================================================================
# diegetic_focus: Narrative Spotlight — diegetic focus budget: hero detail vs impressionistic periphery.
# ==================================================================================================


@dataclass(slots=True)
class ShotFocusPlan:
    shot_id: str
    in_focus: list[str]
    out_of_focus: list[str]
    prompt_suffix: str
    edit_overrides: dict[str, Any] = field(default_factory=dict)


_PERIPHERY_PROMPTS: dict[str, str] = {
    "soft": "shallow depth of field, soft background blur",
    "bokeh": "creamy bokeh, subject razor sharp",
    "abstract": "peripheral elements impressionistic, center narrative sharp",
    "pixel_melt": "dreamlike soft periphery, sharp story focus",
    "fog_of_narrative": "detail only where story looks, edges dissolve into atmosphere",
}


def parse_focus_config(raw: Any) -> dict[str, Any]:
    return parse_toggle(raw, periphery_mode="fog_of_narrative", periphery_strength=0.65, heroes=[], background=[])


def _entities_in_shot(shot: Any) -> list[str]:
    chars = list(getattr(shot, "characters", []) or [])
    objs = list(getattr(shot, "objects", []) or [])
    return [str(x) for x in chars + objs]


def plan_diegetic_focus(
    shots: Sequence[Any],
    config: Mapping[str, Any],
    cast: Mapping[str, Any],
) -> list[ShotFocusPlan]:
    if not config.get("enabled"):
        return []
    mode = str(config.get("periphery_mode") or "fog_of_narrative")
    strength = float(config.get("periphery_strength") or 0.65)
    global_heroes = [str(x) for x in (config.get("heroes") or [])]
    global_bg = [str(x) for x in (config.get("background") or [])]

    plans: list[ShotFocusPlan] = []
    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        ents = _entities_in_shot(sh)
        heroes = [e for e in ents if e in global_heroes] or ([ents[0]] if ents else global_heroes[:1])
        bg = [e for e in ents if e in global_bg] or [e for e in ents if e not in heroes]
        periphery = _PERIPHERY_PROMPTS.get(mode, _PERIPHERY_PROMPTS["fog_of_narrative"])
        hero_desc = []
        for h in heroes:
            ent = cast.get(h)
            if ent is not None:
                hero_desc.append(getattr(ent, "description", "") or h)
            else:
                hero_desc.append(h)
        focus_frag = f"narrative spotlight on {', '.join(hero_desc)}" if hero_desc else "narrative spotlight on subject"
        if bg:
            focus_frag += f", background {', '.join(bg)} rendered {mode}"
        plans.append(
            ShotFocusPlan(
                shot_id=sid,
                in_focus=heroes,
                out_of_focus=bg,
                prompt_suffix=f"{focus_frag}, {periphery}",
                edit_overrides={
                    "depth_interpolate": strength > 0.4,
                    "identity_lock": bool(heroes),
                    "identity_lock_strength": min(0.95, 0.75 + strength * 0.2),
                },
            )
        )
    return plans


# ==================================================================================================
# emotional_contagion: Emotional Contagion — background/crowd inherits hero emotional state.
# ==================================================================================================

_EMOTION_CROWD: dict[str, str] = {
    "fear": "background extras anxious, wide eyes, protective postures, scattered retreat",
    "joy": "background crowd smiling, relaxed shoulders, celebratory micro-gestures",
    "anger": "background onlookers tense, clenched jaws, confrontational stances",
    "grief": "background figures subdued, downcast eyes, slow movement",
    "awe": "background crowd frozen, upward gazes, open mouths",
    "panic": "background chaos, running figures, flailing arms at edges",
    "calm": "background extras neutral, unhurried movement, soft body language",
}


@dataclass(slots=True)
class ContagionPlan:
    shot_id: str
    source_emotion: str
    source_entity: str
    prompt_suffix: str


def parse_contagion_config(raw: Any) -> dict[str, Any]:
    return parse_toggle(raw, source="")


def _shot_emotion(shot: Any) -> str:
    em = str(getattr(shot, "emotion", "") or "").lower()
    if em:
        return em
    p = str(getattr(shot, "prompt", "")).lower()
    for key in _EMOTION_CROWD:
        if key in p:
            return key
    return ""


def plan_emotional_contagion(
    shots: Sequence[Any],
    *,
    config: Mapping[str, Any],
) -> list[ContagionPlan]:
    if not config.get("enabled"):
        return []
    default_source = str(config.get("source") or "")
    plans: list[ContagionPlan] = []
    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        em = _shot_emotion(sh)
        if not em:
            continue
        chars = list(getattr(sh, "characters", []) or [])
        source = default_source or (chars[0] if chars else "subject")
        frag = _EMOTION_CROWD.get(em, f"background reflects {em} mood")
        plans.append(ContagionPlan(shot_id=sid, source_emotion=em, source_entity=source, prompt_suffix=frag))
    return plans


# ==================================================================================================
# kinetic_continuity: Kinetic Ledger — cross-shot velocity and energy state must conserve narrative physics.
# ==================================================================================================

_ENERGY_VERBS: dict[str, float] = {
    "sprint": 0.95,
    "run": 0.85,
    "runs": 0.85,
    "chase": 0.9,
    "jump": 0.8,
    "leap": 0.85,
    "explode": 1.0,
    "collapse": 0.1,
    "walk": 0.35,
    "walks": 0.35,
    "stand": 0.05,
    "stands": 0.05,
    "sit": 0.02,
    "idle": 0.0,
    "sleep": 0.0,
    "land": 0.4,
    "lands": 0.4,
    "fall": 0.7,
    "falls": 0.7,
    "stop": 0.05,
    "stops": 0.05,
    "brake": 0.15,
}


@dataclass(slots=True)
class KineticState:
    energy: float  # 0..1
    vertical: float  # -1 falling, 0 grounded, 1 airborne
    verb: str = ""


@dataclass(slots=True)
class KineticIssue:
    level: str
    code: str
    message: str
    shot_id: str
    related_shot_id: str = ""


@dataclass(slots=True)
class KineticLedger:
    states: list[KineticState] = field(default_factory=list)
    issues: list[KineticIssue] = field(default_factory=list)
    shot_prompt_patches: dict[str, str] = field(default_factory=dict)


def parse_kinetic_config(raw: Any) -> dict[str, Any]:
    return parse_toggle(raw, strict=False, max_energy_jump=0.55)


def _infer_kinetic(prompt: str, explicit: Mapping[str, Any] | None = None) -> KineticState:
    if isinstance(explicit, Mapping):
        return KineticState(
            energy=float(explicit.get("energy") or 0.5),
            vertical=float(explicit.get("vertical") or 0.0),
            verb=str(explicit.get("verb") or ""),
        )
    p = (prompt or "").lower()
    energy = 0.3
    vertical = 0.0
    verb = ""
    for v, e in _ENERGY_VERBS.items():
        if re.search(rf"\b{re.escape(v)}\b", p):
            energy = e
            verb = v
            if v in ("jump", "leap", "falls", "fall"):
                vertical = 1.0 if v in ("jump", "leap") else -1.0
            elif v in ("land", "lands"):
                vertical = 0.0
            break
    if "airborne" in p or "mid-air" in p:
        vertical = 1.0
    if "exhausted" in p or "gasping" in p:
        energy = min(energy, 0.25)
    return KineticState(energy=energy, vertical=vertical, verb=verb)


def track_kinetic_ledger(
    shots: Sequence[Any],
    *,
    config: Mapping[str, Any],
) -> KineticLedger:
    if not config.get("enabled"):
        return KineticLedger()
    max_jump = float(config.get("max_energy_jump") or 0.55)
    strict = bool(config.get("strict"))
    level = "error" if strict else "warn"

    states: list[KineticState] = []
    issues: list[KineticIssue] = []
    patches: dict[str, str] = {}

    prev: KineticState | None = None
    prev_id = ""
    for i, sh in enumerate(shots):
        sid = str(getattr(sh, "id", f"shot_{i}"))
        kin_raw = getattr(sh, "kinetic", None)
        st = _infer_kinetic(str(getattr(sh, "prompt", "")), kin_raw if isinstance(kin_raw, Mapping) else None)
        states.append(st)

        if prev is not None:
            de = abs(st.energy - prev.energy)
            if de > max_jump and not _narrative_reset(str(getattr(sh, "prompt", ""))):
                issues.append(
                    KineticIssue(
                        level=level,
                        code="energy_discontinuity",
                        message=f"Energy jump {prev.energy:.2f}→{st.energy:.2f} across cut (max {max_jump})",
                        shot_id=prev_id,
                        related_shot_id=sid,
                    )
                )
                patches[sid] = _patch_for_energy(prev.energy, st.energy)
            if prev.vertical > 0.5 and st.vertical <= 0 and st.verb not in ("land", "lands", "fall", "falls"):
                issues.append(
                    KineticIssue(
                        level="info",
                        code="missing_landing_beat",
                        message=f"Airborne prior shot — consider landing beat before {sid}",
                        shot_id=sid,
                    )
                )
                patches[sid] = (patches.get(sid, "") + ", grounded landing recovery").strip(", ")
        prev = st
        prev_id = sid

    return KineticLedger(states=states, issues=issues, shot_prompt_patches=patches)


def _narrative_reset(prompt: str) -> bool:
    p = prompt.lower()
    return any(x in p for x in ("later", "meanwhile", "hours later", "new scene", "cut to", "title card"))


def _patch_for_energy(prev_e: float, curr_e: float) -> str:
    if curr_e > prev_e:
        return "acceleration from prior momentum, building speed naturally"
    return "deceleration from prior momentum, slowing naturally"


# ==================================================================================================
# material_memory: Material Memory — wet, mud, blood, soot persist on entities until explicitly cleaned.
# ==================================================================================================

_IRREVERSIBLE = {"bloody", "torn", "burned"}


@dataclass(slots=True)
class MaterialIssue:
    level: str
    code: str
    message: str
    shot_id: str


@dataclass(slots=True)
class MaterialMemoryReport:
    timeline: list[dict[str, str]]
    issues: list[MaterialIssue]
    prompt_injections: dict[str, str]


def parse_material_config(raw: Any) -> dict[str, Any]:
    return parse_toggle(raw, strict=False)


def _entity_materials(shot: Any) -> dict[str, str]:
    raw = getattr(shot, "material_state", None) or getattr(shot, "materials", None) or {}
    if isinstance(raw, Mapping):
        return {str(k): str(v).lower() for k, v in raw.items()}
    return {}


_STATE_PROMPTS: dict[str, str] = {
    "wet": "clothes and hair still wet, water dripping",
    "muddy": "mud stains on boots and hem, dirty splatter",
    "bloody": "blood stains persist on fabric and skin",
    "sooty": "soot smudges on face and clothing",
    "torn": "torn fabric, visible rips unchanged",
    "burned": "char marks and scorch patterns remain",
    "frozen": "frost on lashes and coat edges",
    "dry": "clean dry clothing",
}


def track_material_memory(
    shots: Sequence[Any],
    *,
    config: Mapping[str, Any],
    initial: Mapping[str, str] | None = None,
) -> MaterialMemoryReport:
    if not config.get("enabled"):
        return MaterialMemoryReport(timeline=[], issues=[], prompt_injections={})
    strict = bool(config.get("strict"))
    level = "error" if strict else "warn"
    ledger: dict[str, str] = {str(k): str(v).lower() for k, v in (initial or {}).items()}
    timeline: list[dict[str, str]] = []
    issues: list[MaterialIssue] = []
    injections: dict[str, str] = {}

    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        mats = _entity_materials(sh)
        snap = dict(ledger)
        for eid, st in mats.items():
            prev = ledger.get(eid, "dry")
            if prev in _IRREVERSIBLE and st == "dry":
                issues.append(
                    MaterialIssue(
                        level=level,
                        code="material_reset",
                        message=f"{eid} cannot go {prev}→dry without cleaning scene",
                        shot_id=sid,
                    )
                )
            ledger[eid] = st
        for eid, st in ledger.items():
            if st not in ("dry", "") and eid not in mats:
                frag = _STATE_PROMPTS.get(st, f"{st} material state persists")
                injections[sid] = f"{injections.get(sid, '')}, {eid} {frag}".strip(", ")
        timeline.append({"shot_id": sid, **snap, **{f"mat_{k}": v for k, v in ledger.items()}})
    return MaterialMemoryReport(timeline=timeline, issues=issues, prompt_injections=injections)


# ==================================================================================================
# mise_en_scene: Mise-en-scène Grammar — lead room, headroom, and balance rules as compositional prompts.
# ==================================================================================================

_SHOT_GRAMMAR: dict[str, dict[str, str]] = {
    "close_up": {"headroom": "tight", "lead_room": "minimal", "balance": "face dominant center-third"},
    "close-up": {"headroom": "tight", "lead_room": "minimal", "balance": "face dominant center-third"},
    "medium": {"headroom": "standard", "lead_room": "moderate", "balance": "subject on vertical third"},
    "wide": {"headroom": "generous sky", "lead_room": "wide negative space", "balance": "subject small in frame"},
    "establishing": {
        "headroom": "architectural scale",
        "lead_room": "deep staging",
        "balance": "environment tells story",
    },
    "over_shoulder": {
        "headroom": "standard",
        "lead_room": "space toward listener",
        "balance": "foreground shoulder frames subject",
    },
    "pov": {"headroom": "immersive", "lead_room": "center-weighted", "balance": "hands or weapon lower third optional"},
    "low_angle": {"headroom": "compressed", "lead_room": "upward power", "balance": "subject towers from below"},
    "high_angle": {"headroom": "vulnerable", "lead_room": "downward diminish", "balance": "subject isolated below"},
}


@dataclass(slots=True)
class CompositionSpec:
    shot_id: str
    grammar_key: str
    prompt_suffix: str
    negative_suffix: str


def parse_mise_config(raw: Any) -> dict[str, Any]:
    return parse_toggle(raw, default_on=True, style="classical", rule_of_thirds=True)


def _grammar_key(shot: Any) -> str:
    st = str(getattr(shot, "shot_type", "") or "").lower().replace("-", "_")
    cam = str(getattr(shot, "camera", "") or "").lower()
    prompt = str(getattr(shot, "prompt", "") or "").lower()
    if st in _SHOT_GRAMMAR:
        return st
    if "ots" in cam or "over the shoulder" in prompt:
        return "over_shoulder"
    if "pov" in cam:
        return "pov"
    if "low angle" in prompt or "low-angle" in prompt:
        return "low_angle"
    if "high angle" in prompt or "bird" in prompt:
        return "high_angle"
    if "establishing" in prompt or "wide" in prompt:
        return "establishing"
    if "close" in prompt:
        return "close_up"
    return "medium"


def compose_shot_framing(shot: Any, *, config: Mapping[str, Any]) -> CompositionSpec | None:
    if not config.get("enabled", True):
        return None
    key = _grammar_key(shot)
    gram = _SHOT_GRAMMAR.get(key, _SHOT_GRAMMAR["medium"])
    style = str(config.get("style") or "classical")
    thirds = ", rule of thirds composition" if config.get("rule_of_thirds") else ""
    pos = f"{style} framing, headroom {gram['headroom']}, lead room {gram['lead_room']}, {gram['balance']}{thirds}"
    neg = "centered bullseye framing, clipped head, cramped composition"
    return CompositionSpec(
        shot_id=str(getattr(shot, "id", "")),
        grammar_key=key,
        prompt_suffix=pos,
        negative_suffix=neg,
    )


def compose_all_shots(shots: Sequence[Any], config: Mapping[str, Any]) -> list[CompositionSpec]:
    out: list[CompositionSpec] = []
    for sh in shots:
        spec = compose_shot_framing(sh, config=config)
        if spec:
            out.append(spec)
    return out


# ==================================================================================================
# motif_tracker: Motif Haunting — recurring visual symbols must appear until narratively resolved.
# ==================================================================================================


@dataclass(slots=True)
class VisualMotif:
    id: str
    description: str
    must_appear_by_shot: int = -1  # index; -1 = optional
    resolved_after_shot: int = -1
    haunt_until_resolved: bool = True
    weight: float = 1.0


@dataclass(slots=True)
class MotifHauntReport:
    ok: bool
    unresolved: list[str] = field(default_factory=list)
    appearances: dict[str, list[str]] = field(default_factory=dict)
    injections: dict[str, str] = field(default_factory=dict)  # shot_id → prompt frag


def parse_motifs(raw: Any) -> dict[str, VisualMotif]:
    out: dict[str, VisualMotif] = {}
    if isinstance(raw, Mapping):
        for mid, spec in raw.items():
            if isinstance(spec, str):
                out[str(mid)] = VisualMotif(id=str(mid), description=spec)
            elif isinstance(spec, Mapping):
                out[str(mid)] = VisualMotif(
                    id=str(mid),
                    description=str(spec.get("description") or spec.get("prompt") or mid),
                    must_appear_by_shot=int(
                        spec["must_appear_by_shot"]
                        if spec.get("must_appear_by_shot") is not None
                        else (spec.get("by_shot") if spec.get("by_shot") is not None else -1)
                    ),
                    resolved_after_shot=int(
                        spec["resolved_after_shot"]
                        if spec.get("resolved_after_shot") is not None
                        else (spec.get("resolve_after") if spec.get("resolve_after") is not None else -1)
                    ),
                    haunt_until_resolved=bool(spec.get("haunt", spec.get("haunt_until_resolved", True))),
                    weight=float(spec.get("weight") or 1.0),
                )
    return out


def _motif_in_text(motif: VisualMotif, text: str) -> bool:
    t = (text or "").lower()
    desc = motif.description.lower()
    if desc and desc in t:
        return True
    tokens = [x for x in re.split(r"[\s,_-]+", motif.id.lower()) if len(x) > 2]
    return any(tok in t for tok in tokens)


def audit_motif_haunting(motifs: Mapping[str, VisualMotif], shots: Sequence[Any]) -> MotifHauntReport:
    appearances: dict[str, list[str]] = {m: [] for m in motifs}
    injections: dict[str, str] = {}
    unresolved: list[str] = []

    for i, sh in enumerate(shots):
        sid = str(getattr(sh, "id", f"shot_{i}"))
        prompt = str(getattr(sh, "prompt", "") or "")
        for mid, motif in motifs.items():
            if _motif_in_text(motif, prompt):
                appearances[mid].append(sid)
            elif motif.haunt_until_resolved:
                resolved_at = motif.resolved_after_shot
                if resolved_at < 0 or i <= resolved_at:
                    # Motif should haunt background until seen or resolved
                    if not appearances[mid] and (motif.must_appear_by_shot < 0 or i >= motif.must_appear_by_shot):
                        injections[sid] = merge_csv_unique(
                            injections.get(sid, ""),
                            f"subtle background presence of {motif.description}",
                        )
                    elif appearances[mid] and (resolved_at < 0 or i < resolved_at):
                        injections[sid] = merge_csv_unique(
                            injections.get(sid, ""),
                            f"recurring motif {motif.description} in frame",
                        )

    for mid, motif in motifs.items():
        if motif.must_appear_by_shot >= 0 and not appearances[mid]:
            # Check if any shot up to index saw it
            seen_in_time = False
            for i, sh in enumerate(shots):
                if i > motif.must_appear_by_shot:
                    break
                if _motif_in_text(motif, str(getattr(sh, "prompt", ""))):
                    seen_in_time = True
                    break
            if not seen_in_time:
                unresolved.append(mid)

    ok = len(unresolved) == 0
    return MotifHauntReport(ok=ok, unresolved=unresolved, appearances=appearances, injections=injections)


# ==================================================================================================
# narrative_debt: Narrative Debt — plot threads opened must pay off by mandated shot.
# ==================================================================================================


@dataclass(slots=True)
class PlotThread:
    id: str
    description: str
    opened_shot: str = ""
    payoff_by_shot: int = -1
    resolved: bool = False


@dataclass(slots=True)
class NarrativeDebtReport:
    ok: bool
    open_threads: list[str] = field(default_factory=list)
    resolved_threads: list[str] = field(default_factory=list)
    unpaid: list[str] = field(default_factory=list)
    injections: dict[str, str] = field(default_factory=dict)


def parse_narrative_threads(raw: Any) -> dict[str, PlotThread]:
    out: dict[str, PlotThread] = {}
    if isinstance(raw, Mapping):
        for tid, spec in raw.items():
            if isinstance(spec, str):
                out[str(tid)] = PlotThread(id=str(tid), description=spec)
            elif isinstance(spec, Mapping):
                payoff = spec.get("payoff_by_shot")
                out[str(tid)] = PlotThread(
                    id=str(tid),
                    description=str(spec.get("description") or spec.get("prompt") or tid),
                    opened_shot=str(spec.get("opened_shot") or spec.get("open") or ""),
                    payoff_by_shot=int(payoff if payoff is not None else -1),
                )
    return out


def _thread_in_text(thread: PlotThread, text: str) -> bool:
    t = (text or "").lower()
    if thread.description.lower() in t:
        return True
    tokens = [x for x in re.split(r"[\s,_-]+", thread.id.lower()) if len(x) > 2]
    return any(tok in t for tok in tokens)


def audit_narrative_debt(
    threads: Mapping[str, PlotThread],
    shots: Sequence[Any],
) -> NarrativeDebtReport:
    if not threads:
        return NarrativeDebtReport(ok=True)
    open_set: set[str] = set()
    resolved: list[str] = []
    unpaid: list[str] = []
    injections: dict[str, str] = {}
    opened_at: dict[str, int] = {}

    for i, sh in enumerate(shots):
        sid = str(getattr(sh, "id", f"shot_{i}"))
        prompt = str(getattr(sh, "prompt", ""))
        shot_threads = set(getattr(sh, "threads", []) or getattr(sh, "narrative_threads", []) or [])
        for tid, thread in threads.items():
            if tid in shot_threads or _thread_in_text(thread, prompt):
                if tid not in open_set:
                    open_set.add(tid)
                    opened_at[tid] = i
                if "resolve" in prompt.lower() or "reveal" in prompt.lower() or "payoff" in prompt.lower():
                    if tid in open_set:
                        open_set.discard(tid)
                        resolved.append(tid)
        for tid in list(open_set):
            thread = threads[tid]
            deadline = thread.payoff_by_shot
            if deadline >= 0 and i >= deadline and tid not in resolved:
                unpaid.append(tid)
            elif tid in open_set:
                injections[sid] = f"{injections.get(sid, '')}, subtle reminder of {thread.description}".strip(", ")

    ok = len(unpaid) == 0
    return NarrativeDebtReport(
        ok=ok,
        open_threads=list(open_set),
        resolved_threads=resolved,
        unpaid=unpaid,
        injections=injections,
    )


# ==================================================================================================
# narrative_tension: Narrative Tension Thermostat — emotion curve drives camera, motion, and grade per shot.
# ==================================================================================================


@dataclass(slots=True)
class TensionPoint:
    at: float  # 0..1 normalized timeline
    value: float  # 0..1 tension


@dataclass(slots=True)
class TensionCurve:
    points: list[TensionPoint] = field(default_factory=list)
    genre_bias: str = ""

    def sample(self, u: float) -> float:
        if not self.points:
            return 0.5
        pts = sorted(self.points, key=lambda p: p.at)
        u = max(0.0, min(1.0, u))
        if u <= pts[0].at:
            return pts[0].value
        if u >= pts[-1].at:
            return pts[-1].value
        for a, b in zip(pts, pts[1:]):
            if a.at <= u <= b.at:
                span = max(1e-6, b.at - a.at)
                t = (u - a.at) / span
                return a.value + t * (b.value - a.value)
        return pts[-1].value


@dataclass(slots=True)
class ShotTensionProfile:
    shot_id: str
    shot_index: int
    tension: float
    prompt_suffix: str
    negative_suffix: str
    edit_overrides: dict[str, Any] = field(default_factory=dict)


_GENRE_CURVES: dict[str, list[tuple[float, float]]] = {
    "horror": [(0.0, 0.25), (0.5, 0.55), (0.85, 0.95), (1.0, 0.4)],
    "action": [(0.0, 0.5), (0.3, 0.85), (0.7, 0.95), (1.0, 0.35)],
    "romance": [(0.0, 0.2), (0.5, 0.45), (0.8, 0.7), (1.0, 0.5)],
    "comedy": [(0.0, 0.15), (0.6, 0.35), (0.9, 0.8), (1.0, 0.2)],
    "documentary": [(0.0, 0.3), (0.5, 0.4), (1.0, 0.35)],
}


def parse_tension_curve(raw: Any, *, genre: str = "") -> TensionCurve | None:
    if raw is None:
        genre_key = (genre or "").lower()
        if genre_key in _GENRE_CURVES:
            return TensionCurve(
                points=[TensionPoint(at=a, value=v) for a, v in _GENRE_CURVES[genre_key]],
                genre_bias=genre_key,
            )
        return None
    if isinstance(raw, list):
        pts: list[TensionPoint] = []
        for i, v in enumerate(raw):
            if isinstance(v, (int, float)):
                at = i / max(1, len(raw) - 1)
                pts.append(TensionPoint(at=at, value=float(max(0.0, min(1.0, v)))))
            elif isinstance(v, Mapping):
                pts.append(
                    TensionPoint(
                        at=float(v.get("at") or v.get("t") or 0.0),
                        value=float(max(0.0, min(1.0, float(v.get("value") or v.get("tension") or 0.5)))),
                    )
                )
        return TensionCurve(points=pts, genre_bias=genre) if pts else None
    if isinstance(raw, Mapping):
        if raw.get("auto_genre") and genre:
            return parse_tension_curve(None, genre=genre)
        pts2 = (
            [
                TensionPoint(at=float(a), value=float(b))
                for a, b in (raw.get("points") or raw.get("curve") or {}).items()
            ]
            if isinstance(raw.get("points") or raw.get("curve"), Mapping)
            else []
        )
        if not pts2 and isinstance(raw.get("values"), list):
            return parse_tension_curve(raw["values"], genre=genre)
        return TensionCurve(points=pts2, genre_bias=str(raw.get("genre") or genre)) if pts2 else None
    return None


def sample_tension_for_shots(
    curve: TensionCurve,
    shots: Sequence[Any],
    *,
    total_duration: float,
) -> list[ShotTensionProfile]:
    if not shots:
        return []
    dur_sum = sum(float(getattr(s, "duration_sec", 0) or 0) for s in shots)
    total = dur_sum if dur_sum > 0 else total_duration
    elapsed = 0.0
    out: list[ShotTensionProfile] = []
    for i, sh in enumerate(shots):
        d = float(getattr(sh, "duration_sec", 0) or (total / max(1, len(shots))))
        mid = (elapsed + d * 0.5) / max(1e-6, total)
        t = curve.sample(mid)
        elapsed += d
        prompt, neg, edits = tension_shot_overrides(t)
        out.append(
            ShotTensionProfile(
                shot_id=str(getattr(sh, "id", f"shot_{i}")),
                shot_index=i,
                tension=round(t, 3),
                prompt_suffix=prompt,
                negative_suffix=neg,
                edit_overrides=edits,
            )
        )
    return out


def tension_shot_overrides(t: float) -> tuple[str, str, dict[str, Any]]:
    """High tension → handheld, fast cuts feel, contrast; low → still, soft."""
    t = max(0.0, min(1.0, t))
    if t >= 0.75:
        return (
            "high stakes intensity, urgent energy, sharp contrast",
            "flat calm, sleepy pacing",
            {
                "velocity_ease": False,
                "camera_stabilize": False,
                "keyframe_interval": max(3, int(6 - t * 3)),
                "motion_beat_keyframes": True,
            },
        )
    if t >= 0.45:
        return (
            "building suspense, controlled momentum",
            "chaotic shaky blur",
            {"velocity_ease": True, "velocity_ease_mode": "ease_in"},
        )
    return (
        "quiet atmosphere, gentle stillness, observational pacing",
        "hyperactive shake, strobe chaos",
        {"camera_stabilize": True, "camera_stabilize_strength": 0.8, "keyframe_interval": 8},
    )


# ==================================================================================================
# offscreen_space: Offscreen Space Map — what happens just outside frame affects sound, light, and reactions.
# ==================================================================================================

_OFFSCREEN_PRESETS: dict[str, dict[str, str]] = {
    "crowd_left": {
        "audio": "muffled crowd noise from offscreen left",
        "light": "occasional flash from offscreen left",
        "reaction": "characters glance offscreen left",
    },
    "monster_right": {
        "audio": "deep growl from offscreen right",
        "light": "shadow sweep from offscreen right",
        "reaction": "characters recoil looking offscreen right",
    },
    "traffic_behind": {
        "audio": "distant traffic hum behind camera",
        "light": "headlight streaks in background bokeh",
        "reaction": "none",
    },
    "storm_above": {
        "audio": "thunder rumble above frame",
        "light": "lightning flicker from above",
        "reaction": "characters flinch upward",
    },
    "river_below": {
        "audio": "rushing water below frame edge",
        "light": "cool blue bounce from below",
        "reaction": "balance cautious near edge",
    },
}


@dataclass(slots=True)
class OffscreenEvent:
    zone: str
    audio: str = ""
    light: str = ""
    reaction: str = ""


@dataclass(slots=True)
class OffscreenPlan:
    shot_id: str
    events: list[OffscreenEvent] = field(default_factory=list)
    prompt_suffix: str = ""


def parse_offscreen_map(raw: Any) -> dict[str, list[OffscreenEvent]]:
    out: dict[str, list[OffscreenEvent]] = {}
    if not isinstance(raw, Mapping):
        return out
    for shot_id, spec in raw.items():
        events: list[OffscreenEvent] = []
        zones = spec if isinstance(spec, list) else (spec.get("zones") or spec.get("offscreen") or [])
        if isinstance(zones, str):
            zones = [zones]
        for z in zones:
            zname = str(z)
            preset = _OFFSCREEN_PRESETS.get(zname, {})
            if isinstance(z, Mapping):
                events.append(
                    OffscreenEvent(
                        zone=str(z.get("zone") or zname),
                        audio=str(z.get("audio") or ""),
                        light=str(z.get("light") or ""),
                        reaction=str(z.get("reaction") or ""),
                    )
                )
            else:
                events.append(
                    OffscreenEvent(
                        zone=zname,
                        audio=preset.get("audio", ""),
                        light=preset.get("light", ""),
                        reaction=preset.get("reaction", ""),
                    )
                )
        if events:
            out[str(shot_id)] = events
    return out


def plan_offscreen_space(
    shots: Sequence[Any],
    offscreen_map: Mapping[str, list[OffscreenEvent]],
) -> list[OffscreenPlan]:
    plans: list[OffscreenPlan] = []
    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        inline = getattr(sh, "offscreen", None)
        events = list(offscreen_map.get(sid, []))
        if isinstance(inline, list):
            for z in inline:
                events.append(OffscreenEvent(zone=str(z)))
        elif isinstance(inline, str):
            events.append(OffscreenEvent(zone=inline))
        if not events:
            continue
        frags: list[str] = []
        for ev in events:
            if ev.audio:
                frags.append(ev.audio)
            if ev.light:
                frags.append(ev.light)
            if ev.reaction and ev.reaction != "none":
                frags.append(ev.reaction)
        plans.append(OffscreenPlan(shot_id=sid, events=events, prompt_suffix=", ".join(frags)))
    return plans


# ==================================================================================================
# parallax_budget: Parallax Depth Budget — 2.5D layer separation hints for compositing pipelines.
# ==================================================================================================


@dataclass(slots=True)
class ParallaxLayer:
    name: str
    depth: float  # 0 bg .. 1 fg
    motion_scale: float


@dataclass(slots=True)
class ParallaxPlan:
    shot_id: str
    layers: list[ParallaxLayer]
    prompt_suffix: str
    edit_overrides: dict[str, Any]


def parse_parallax_config(raw: Any) -> dict[str, Any]:
    return parse_toggle(raw, layers=3)


def plan_parallax_budget(
    shots: Sequence[Any],
    *,
    config: Mapping[str, Any],
) -> list[ParallaxPlan]:
    if not config.get("enabled"):
        return []
    n = max(2, min(5, int(config.get("layers") or 3)))
    plans: list[ParallaxPlan] = []
    names = ["far_background", "midground", "subject_plane", "foreground_occluder", "ultra_fg"]
    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        layers: list[ParallaxLayer] = []
        for i in range(n):
            depth = i / max(1, n - 1)
            motion = 0.15 + depth * 0.85
            layers.append(
                ParallaxLayer(name=names[min(i, len(names) - 1)], depth=round(depth, 2), motion_scale=round(motion, 2))
            )
        frag = f"parallax depth stack {n} layers, separated foreground midground background motion"
        plans.append(
            ParallaxPlan(
                shot_id=sid,
                layers=layers,
                prompt_suffix=frag,
                edit_overrides={"depth_interpolate": True, "region_motion": True},
            )
        )
    return plans


# ==================================================================================================
# scar_timeline: Scar Timeline — injuries and damage accumulate on character bibles across shots.
# ==================================================================================================


@dataclass(slots=True)
class ScarTimelineReport:
    by_shot: dict[str, dict[str, list[str]]]
    prompt_injections: dict[str, str]


_INJURY_WORDS = ("bruised", "bleeding", "cut", "scarred", "bandaged", "limping", "burnt", "wounded")


def parse_scar_config(raw: Any) -> dict[str, Any]:
    return parse_toggle(raw)


def _injuries_from_shot(shot: Any) -> dict[str, list[str]]:
    raw = getattr(shot, "injuries", None) or getattr(shot, "scars", None) or {}
    out: dict[str, list[str]] = {}
    if isinstance(raw, Mapping):
        for cid, inj in raw.items():
            if isinstance(inj, list):
                out[str(cid)] = [str(x) for x in inj]
            else:
                out[str(cid)] = [str(inj)]
    prompt = str(getattr(shot, "prompt", "")).lower()
    for w in _INJURY_WORDS:
        if w in prompt:
            for cid in getattr(shot, "characters", []) or []:
                out.setdefault(str(cid), []).append(w)
    return out


def track_scar_timeline(
    shots: Sequence[Any],
    *,
    config: Mapping[str, Any],
) -> ScarTimelineReport:
    if not config.get("enabled"):
        return ScarTimelineReport(by_shot={}, prompt_injections={})
    ledger: dict[str, list[str]] = {}
    by_shot: dict[str, dict[str, list[str]]] = {}
    injections: dict[str, str] = {}

    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        new_inj = _injuries_from_shot(sh)
        for cid, inj_list in new_inj.items():
            for inj in inj_list:
                if inj not in ledger.get(cid, []):
                    ledger.setdefault(cid, []).append(inj)
        snap = {cid: list(injs) for cid, injs in ledger.items()}
        by_shot[sid] = snap
        if ledger:
            parts = [f"{cid}: {', '.join(injs)}" for cid, injs in ledger.items() if injs]
            injections[sid] = f"visible accumulated injuries — {'; '.join(parts)}"
    return ScarTimelineReport(by_shot=by_shot, prompt_injections=injections)


# ==================================================================================================
# screen_direction: Screen Direction Lock — movement vector must stay consistent across cuts (vector 180° rule).
# ==================================================================================================

_DIR_ALIASES = {
    "l": "left",
    "r": "right",
    "left": "left",
    "right": "right",
    "frame_left": "left",
    "frame_right": "right",
}

_MOVE_LEFT = (r"\bmoves?\s+left\b", r"\bwalks?\s+left\b", r"\benters?\s+from\s+right\b", r"\bexits?\s+left\b")
_MOVE_RIGHT = (r"\bmoves?\s+right\b", r"\bwalks?\s+right\b", r"\benters?\s+from\s+left\b", r"\bexits?\s+right\b")


@dataclass(slots=True)
class ScreenDirectionIssue:
    level: str
    code: str
    message: str
    shot_id: str
    related_shot_id: str = ""


@dataclass(slots=True)
class ScreenDirectionReport:
    directions: dict[str, str]
    issues: list[ScreenDirectionIssue]
    prompt_patches: dict[str, str]


def parse_screen_direction_config(raw: Any) -> dict[str, Any]:
    return parse_toggle(raw, strict=False)


def _norm(d: str) -> str:
    return _DIR_ALIASES.get((d or "").strip().lower().replace(" ", "_"), "")


def _infer_direction(prompt: str) -> str:
    p = prompt or ""
    for pat in _MOVE_LEFT:
        if re.search(pat, p, re.I):
            return "left"
    for pat in _MOVE_RIGHT:
        if re.search(pat, p, re.I):
            return "right"
    return ""


def _shot_direction(shot: Any) -> str:
    explicit = _norm(str(getattr(shot, "screen_direction", "") or ""))
    if explicit:
        return explicit
    return _infer_direction(str(getattr(shot, "prompt", "")))


def track_screen_direction(shots: Sequence[Any], *, config: Mapping[str, Any]) -> ScreenDirectionReport:
    if not config.get("enabled"):
        return ScreenDirectionReport(directions={}, issues=[], prompt_patches={})
    strict = bool(config.get("strict"))
    level = "error" if strict else "warn"
    directions: dict[str, str] = {}
    issues: list[ScreenDirectionIssue] = []
    patches: dict[str, str] = {}

    prev_dir = ""
    prev_id = ""
    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        d = _shot_direction(sh)
        if d:
            directions[sid] = d
        if prev_dir and d and prev_dir != d:
            if str(getattr(sh, "transition", "cut") or "cut") in ("cut", ""):
                issues.append(
                    ScreenDirectionIssue(
                        level=level,
                        code="screen_direction_flip",
                        message=f"Movement {prev_dir}→{d} across cut breaks screen direction continuity",
                        shot_id=prev_id,
                        related_shot_id=sid,
                    )
                )
                patches[sid] = f"continue movement toward screen {prev_dir}, match prior trajectory"
        if d:
            prev_dir = d
            prev_id = sid
    return ScreenDirectionReport(directions=directions, issues=issues, prompt_patches=patches)


# ==================================================================================================
# semantic_gravity: Semantic Gravity — narrative weight ranks entities for identity lock and drift repair priority.
# ==================================================================================================


@dataclass(slots=True)
class GravityWell:
    entity_id: str
    weight: float  # 0..1
    lock_priority: int
    drift_resist: float


@dataclass(slots=True)
class SemanticGravityField:
    wells: list[GravityWell] = field(default_factory=list)
    dominant_id: str = ""

    def top_entities(self, n: int = 3) -> list[str]:
        ranked = sorted(self.wells, key=lambda w: (-w.weight, -w.lock_priority))
        return [w.entity_id for w in ranked[:n]]


def parse_semantic_gravity(raw: Any) -> dict[str, float]:
    weights: dict[str, float] = {}
    if isinstance(raw, Mapping):
        for eid, spec in raw.items():
            if isinstance(spec, (int, float)):
                weights[str(eid)] = float(spec)
            elif isinstance(spec, Mapping):
                weights[str(eid)] = float(spec.get("weight") or spec.get("gravity") or 0.5)
    return weights


def build_gravity_field(
    cast: Mapping[str, Any],
    props: Mapping[str, Any],
    *,
    weights: Mapping[str, float],
    shots: Sequence[Any],
) -> SemanticGravityField:
    appearance_count: dict[str, int] = {}
    for sh in shots:
        for eid in list(getattr(sh, "characters", []) or []) + list(getattr(sh, "objects", []) or []):
            appearance_count[str(eid)] = appearance_count.get(str(eid), 0) + 1

    wells: list[GravityWell] = []
    all_ids = set(cast.keys()) | set(props.keys()) | set(weights.keys())
    for eid in all_ids:
        base = float(weights.get(eid, 0.4))
        ent = cast.get(eid) or props.get(eid)
        if ent is not None and getattr(ent, "lock", False):
            base = max(base, 0.85)
        freq_boost = min(0.3, appearance_count.get(eid, 0) * 0.08)
        w = min(1.0, base + freq_boost)
        wells.append(
            GravityWell(
                entity_id=eid,
                weight=round(w, 3),
                lock_priority=int(w * 100),
                drift_resist=round(0.5 + w * 0.45, 3),
            )
        )
    ranked = sorted(wells, key=lambda x: -x.weight)
    dominant = ranked[0].entity_id if ranked else ""
    return SemanticGravityField(wells=wells, dominant_id=dominant)


def gravity_edit_overrides(field: SemanticGravityField) -> dict[str, Any]:
    if not field.wells:
        return {}
    top = field.top_entities(2)
    strength = 0.78
    for w in field.wells:
        if w.entity_id in top:
            strength = max(strength, w.drift_resist)
    return {
        "identity_lock": True,
        "identity_lock_strength": min(0.96, strength),
        "semantic_drift_repair": True,
        "drift_threshold": max(0.35, 0.6 - (field.wells[0].weight if field.wells else 0) * 0.2),
        "gravity_priority_entities": top,
    }


# ==================================================================================================
# silence_map: Silence Map — deliberate audio/visual silence beats for horror, comedy, and tension.
# ==================================================================================================


@dataclass(slots=True)
class SilenceBeat:
    shot_id: str
    duration_sec: float
    visual_mode: str
    prompt_suffix: str
    audio_note: str


def parse_silence_map(raw: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if isinstance(raw, Mapping):
        for sid, spec in raw.items():
            if isinstance(spec, (int, float)):
                out[str(sid)] = {"duration_sec": float(spec)}
            elif isinstance(spec, Mapping):
                out[str(sid)] = dict(spec)
    return out


def plan_silence_beats(
    shots: Sequence[Any],
    silence_map: Mapping[str, dict[str, Any]],
) -> list[SilenceBeat]:
    beats: list[SilenceBeat] = []
    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        spec = silence_map.get(sid, {})
        if getattr(sh, "silence", False):
            spec = {**spec, "duration_sec": spec.get("duration_sec") or 0.8}
        if not spec:
            continue
        dur = float(spec.get("duration_sec") or spec.get("duration") or 0.5)
        mode = str(spec.get("visual") or spec.get("mode") or "hold")
        if mode == "horror":
            frag = "held frozen frame, dread silence, micro-movement only"
        elif mode == "comedy":
            frag = "awkward pause beat, characters frozen mid-action"
        else:
            frag = "deliberate stillness, held breath moment"
        beats.append(
            SilenceBeat(
                shot_id=sid,
                duration_sec=dur,
                visual_mode=mode,
                prompt_suffix=frag,
                audio_note=str(spec.get("audio") or "drop all sound"),
            )
        )
    return beats


# ==================================================================================================
# stinger_frames: Stinger Frames — last frames of a shot get impact/sakuga emphasis (smear, hold, flash).
# ==================================================================================================

_ACTION_END = ("hits", "lands", "explodes", "strikes", "slams", "punch", "kick", "impact", "crash")


@dataclass(slots=True)
class StingerSpec:
    shot_id: str
    frame_count: int
    style: str
    prompt_suffix: str
    edit_overrides: dict[str, Any]


def parse_stinger_config(raw: Any) -> dict[str, Any]:
    return parse_toggle(raw, frames=3, style="impact")


def _needs_stinger(prompt: str) -> bool:
    p = (prompt or "").lower()
    return any(v in p for v in _ACTION_END)


def plan_stinger_frames(
    shots: Sequence[Any],
    *,
    config: Mapping[str, Any],
) -> list[StingerSpec]:
    if not config.get("enabled"):
        return []
    n = max(1, min(8, int(config.get("frames") or 3)))
    style = str(config.get("style") or "impact")
    specs: list[StingerSpec] = []
    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        prompt = str(getattr(sh, "prompt", ""))
        force = bool(getattr(sh, "stinger", False))
        if not force and not _needs_stinger(prompt):
            continue
        if style == "anime":
            frag = f"anime sakuga stinger, smear frames on last {n} frames, impact hold"
        elif style == "horror":
            frag = f"horror stinger, sudden micro-zoom last {n} frames, breath hold"
        else:
            frag = f"cinematic impact stinger, motion accent last {n} frames"
        specs.append(
            StingerSpec(
                shot_id=sid,
                frame_count=n,
                style=style,
                prompt_suffix=frag,
                edit_overrides={"motion_beat_keyframes": True, "keyframe_interval": max(2, 6 - n)},
            )
        )
    return specs


# ==================================================================================================
# temporal_echo: Temporal Echo — shots visually rhyme with earlier beats (motifs, palette, camera).
# ==================================================================================================


@dataclass(slots=True)
class EchoLink:
    shot_id: str
    echoes_shot_id: str
    echo_strength: float
    prompt_suffix: str


@dataclass(slots=True)
class TemporalEchoPlan:
    links: list[EchoLink] = field(default_factory=list)


def parse_echo_config(raw: Any) -> dict[str, Any]:
    return parse_toggle(raw, auto_rhyme=True, strength=0.45, links=[])


def _visual_tokens(prompt: str) -> set[str]:
    stop = {"the", "a", "an", "and", "with", "in", "on", "at", "to", "of", "shot", "camera"}
    words = re.findall(r"[a-z]{4,}", (prompt or "").lower())
    return {w for w in words if w not in stop}


def plan_temporal_echoes(
    shots: Sequence[Any],
    config: Mapping[str, Any],
) -> TemporalEchoPlan:
    if not config.get("enabled"):
        return TemporalEchoPlan()
    strength = float(config.get("strength") or 0.45)
    links: list[EchoLink] = []

    manual = config.get("links") or []
    for row in manual:
        if not isinstance(row, Mapping):
            continue
        links.append(
            EchoLink(
                shot_id=str(row.get("shot") or row.get("to") or ""),
                echoes_shot_id=str(row.get("echoes") or row.get("from") or ""),
                echo_strength=float(row.get("strength") or strength),
                prompt_suffix=str(row.get("prompt") or "visual rhyme echoing earlier beat"),
            )
        )

    if config.get("auto_rhyme"):
        tokens_per_shot = [(str(getattr(s, "id", "")), _visual_tokens(str(getattr(s, "prompt", "")))) for s in shots]
        for i, (sid, tok) in enumerate(tokens_per_shot):
            if i < 2:
                continue
            best_j = -1
            best_overlap = 0
            for j in range(i - 1):
                overlap = len(tok & tokens_per_shot[j][1])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_j = j
            if best_j >= 0 and best_overlap >= 3:
                ref_id = tokens_per_shot[best_j][0]
                shared = tok & tokens_per_shot[best_j][1]
                frag = f"visual echo rhyming with {ref_id}: {', '.join(sorted(shared)[:4])}"
                links.append(
                    EchoLink(
                        shot_id=sid,
                        echoes_shot_id=ref_id,
                        echo_strength=strength,
                        prompt_suffix=frag,
                    )
                )
    return TemporalEchoPlan(links=links)


# ==================================================================================================
# weather_inertia: Weather Inertia — climate state persists; storms don't vanish between cuts without reason.
# ==================================================================================================

_WEATHER_LEVELS = ("clear", "overcast", "rain", "storm", "snow", "fog", "sandstorm")
_ORDER = {w: i for i, w in enumerate(_WEATHER_LEVELS)}


@dataclass(slots=True)
class WeatherState:
    condition: str
    intensity: float  # 0..1


@dataclass(slots=True)
class WeatherIssue:
    level: str
    code: str
    message: str
    shot_id: str


@dataclass(slots=True)
class WeatherInertiaReport:
    timeline: list[dict[str, Any]]
    issues: list[WeatherIssue]
    prompt_injections: dict[str, str]


def parse_weather_config(raw: Any) -> dict[str, Any]:
    return parse_toggle(raw, max_jump=2, initial="")


def _infer_weather(prompt: str) -> WeatherState:
    p = (prompt or "").lower()
    if any(x in p for x in ("blizzard", "snow", "snowing")):
        return WeatherState("snow", 0.8)
    if any(x in p for x in ("thunder", "storm", "lightning")):
        return WeatherState("storm", 0.9)
    if any(x in p for x in ("rain", "downpour", "drizzle")):
        return WeatherState("rain", 0.7)
    if any(x in p for x in ("fog", "mist", "haze")):
        return WeatherState("fog", 0.6)
    if any(x in p for x in ("sandstorm", "dust storm")):
        return WeatherState("sandstorm", 0.85)
    if any(x in p for x in ("overcast", "cloudy")):
        return WeatherState("overcast", 0.4)
    if any(x in p for x in ("clear sky", "sunny", "bright day")):
        return WeatherState("clear", 0.2)
    return WeatherState("", 0.0)


def _shot_weather(shot: Any) -> WeatherState:
    w = getattr(shot, "weather_spec", None) or getattr(shot, "weather", None)
    if isinstance(w, Mapping):
        return WeatherState(str(w.get("condition") or w.get("type") or ""), float(w.get("intensity") or 0.5))
    if isinstance(w, str) and w:
        return WeatherState(w.lower(), 0.6)
    return _infer_weather(str(getattr(shot, "prompt", "")))


_WEATHER_FRAG: dict[str, str] = {
    "rain": "continued rain, wet atmosphere",
    "storm": "ongoing storm, wind and rain persistence",
    "snow": "continuing snowfall, cold air",
    "fog": "persistent fog layer",
    "sandstorm": "sand still in air, reduced visibility",
    "overcast": "overcast sky continuity",
}


def track_weather_inertia(
    shots: Sequence[Any],
    *,
    config: Mapping[str, Any],
) -> WeatherInertiaReport:
    if not config.get("enabled"):
        return WeatherInertiaReport(timeline=[], issues=[], prompt_injections={})
    max_jump = int(config.get("max_jump") or 2)
    init = str(config.get("initial") or "")
    prev = WeatherState(init, 0.5) if init else WeatherState("", 0.0)
    timeline: list[dict[str, Any]] = []
    issues: list[WeatherIssue] = []
    injections: dict[str, str] = {}

    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        cur = _shot_weather(sh)
        if not cur.condition and prev.condition:
            injections[sid] = _WEATHER_FRAG.get(prev.condition, f"continued {prev.condition}")
            cur = prev
        elif cur.condition and prev.condition:
            jump = abs(_ORDER.get(cur.condition, 0) - _ORDER.get(prev.condition, 0))
            if jump > max_jump and "interior" not in str(getattr(sh, "prompt", "")).lower():
                issues.append(
                    WeatherIssue(
                        level="warn",
                        code="weather_discontinuity",
                        message=f"Weather {prev.condition}→{cur.condition} without transition",
                        shot_id=sid,
                    )
                )
        if cur.condition:
            prev = cur
        timeline.append({"shot_id": sid, "condition": cur.condition, "intensity": cur.intensity})
    return WeatherInertiaReport(timeline=timeline, issues=issues, prompt_injections=injections)


# ==================================================================================================
# witness_lens: Witness Lens — per-shot embodied viewpoint (who is watching changes the frame).
# ==================================================================================================

_LENS_PROMPTS: dict[str, tuple[str, str]] = {
    "child": ("low camera height child POV, wide-eyed framing, wonder", "adult eye-level detached"),
    "cctv": ("CCTV surveillance angle, slight fish-eye, timestamp overlay feel", "cinematic beauty lighting"),
    "lover": ("intimate close proximity, soft bokeh, tender framing", "wide detached documentary"),
    "victim": ("dutch angle unease, tight claustrophobic framing", "heroic low angle power"),
    "stranger": ("candid off-center voyeur framing, subject unaware", "posed symmetrical portrait"),
    "archivist": ("flat documentary light, neutral observational distance", "dramatic stylized noir"),
    "deity": ("god's eye high angle, vast scale, subjects small", "ground level immersive"),
    "drone": ("aerial witness, detached overview, geometric composition", "handheld intimacy"),
}


@dataclass(slots=True)
class WitnessShotPlan:
    shot_id: str
    lens: str
    prompt_suffix: str
    negative_suffix: str


def parse_witness_config(raw: Any) -> dict[str, Any]:
    return parse_toggle(raw, default="neutral")


def _shot_witness(shot: Any, default: str) -> str:
    w = str(getattr(shot, "witness", "") or getattr(shot, "pov_witness", "") or "").lower()
    if w:
        return w
    p = str(getattr(shot, "prompt", "")).lower()
    for key in _LENS_PROMPTS:
        if key in p:
            return key
    return default if default in _LENS_PROMPTS else ""


def plan_witness_lens(
    shots: Sequence[Any],
    *,
    config: Mapping[str, Any],
) -> list[WitnessShotPlan]:
    if not config.get("enabled"):
        return []
    default = str(config.get("default") or "")
    plans: list[WitnessShotPlan] = []
    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        lens = _shot_witness(sh, default)
        if not lens:
            continue
        pos, neg = _LENS_PROMPTS.get(lens, (f"{lens} witness perspective", ""))
        plans.append(WitnessShotPlan(shot_id=sid, lens=lens, prompt_suffix=pos, negative_suffix=neg))
    return plans


__all__ = [
    "ShotEnrichment",
    "FrontierCompileResult",
    "compile_frontier_layers",
    "list_frontier_modules",
    "parse_toggle",
    "WindupSpec",
    "AnticipationPlan",
    "parse_anticipation_config",
    "plan_anticipation_windups",
    "BreathCadence",
    "plan_breath_cadence",
    "parse_breath_config",
    "EmpathyMove",
    "parse_empathy_config",
    "plan_camera_empathy",
    "CausalRule",
    "CausalRipple",
    "parse_causal_rules",
    "detect_triggers_in_prompt",
    "apply_causal_ripples",
    "ChromaticBeat",
    "parse_chromatic_arc",
    "sample_chromatic_for_shots",
    "CounterfactualBranch",
    "CounterfactualPlan",
    "parse_counterfactuals",
    "build_counterfactual_plan",
    "ShotFocusPlan",
    "parse_focus_config",
    "plan_diegetic_focus",
    "ContagionPlan",
    "parse_contagion_config",
    "plan_emotional_contagion",
    "KineticState",
    "KineticIssue",
    "KineticLedger",
    "parse_kinetic_config",
    "track_kinetic_ledger",
    "MaterialIssue",
    "MaterialMemoryReport",
    "parse_material_config",
    "track_material_memory",
    "CompositionSpec",
    "parse_mise_config",
    "compose_shot_framing",
    "VisualMotif",
    "MotifHauntReport",
    "parse_motifs",
    "audit_motif_haunting",
    "PlotThread",
    "NarrativeDebtReport",
    "parse_narrative_threads",
    "audit_narrative_debt",
    "TensionPoint",
    "TensionCurve",
    "ShotTensionProfile",
    "parse_tension_curve",
    "sample_tension_for_shots",
    "tension_shot_overrides",
    "OffscreenEvent",
    "OffscreenPlan",
    "parse_offscreen_map",
    "plan_offscreen_space",
    "ParallaxLayer",
    "ParallaxPlan",
    "parse_parallax_config",
    "plan_parallax_budget",
    "ScarTimelineReport",
    "parse_scar_config",
    "track_scar_timeline",
    "ScreenDirectionIssue",
    "ScreenDirectionReport",
    "parse_screen_direction_config",
    "track_screen_direction",
    "GravityWell",
    "SemanticGravityField",
    "parse_semantic_gravity",
    "build_gravity_field",
    "gravity_edit_overrides",
    "SilenceBeat",
    "parse_silence_map",
    "plan_silence_beats",
    "StingerSpec",
    "parse_stinger_config",
    "plan_stinger_frames",
    "EchoLink",
    "TemporalEchoPlan",
    "parse_echo_config",
    "plan_temporal_echoes",
    "WeatherState",
    "WeatherIssue",
    "WeatherInertiaReport",
    "parse_weather_config",
    "track_weather_inertia",
    "WitnessShotPlan",
    "parse_witness_config",
    "plan_witness_lens",
]

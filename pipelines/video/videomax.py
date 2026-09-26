"""VIDEOMAX — competitor-beating consistency stack for SDX video.

Wires orphaned repairs (count bind, shot chain, occlusion, physics assist,
invention stack on keyframes) into one enable path: ``videomax=True`` or
``--video-quality max`` / ``--videomax``.

Style-aware: realistic / anime / cartoon / VFX / stop-motion / … get different
repair dials via ``style_router`` so live-action rules don't crush cel holds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

__all__ = [
    "VideoMaxPlan",
    "plan_videomax",
    "apply_videomax_to_options",
    "videomax_sample_extras",
    "FAILURE_AXES",
    "union_invention_stacks",
]


def _invention_stack_from_extras(extras: Sequence[str]) -> str:
    for i, tok in enumerate(extras):
        if tok == "--invention-stack" and i + 1 < len(extras):
            return str(extras[i + 1])
    return ""


def union_invention_stacks(*parts: str) -> str:
    """Stable comma-union of invention stack ids (videowave, artwave, …)."""
    out: list[str] = []
    for p in parts:
        for bit in str(p or "").split(","):
            b = bit.strip()
            if b and b not in out:
                out.append(b)
    return ",".join(out)


def _union_invention_stacks(*parts: str) -> str:
    return union_invention_stacks(*parts)


def _strip_invention_stack_pairs(extras: Sequence[str]) -> list[str]:
    out: list[str] = []
    skip_next = False
    for tok in extras:
        if skip_next:
            skip_next = False
            continue
        if tok == "--invention-stack":
            skip_next = True
            continue
        out.append(tok)
    return out


def _merge_sample_extras(style: Sequence[str], base: Sequence[str]) -> list[str]:
    """Union invention stacks; prefer style garnish tokens; keep VIDEOMAX boosts."""
    inv = _union_invention_stacks(
        _invention_stack_from_extras(style),
        _invention_stack_from_extras(base),
    )
    merged = _strip_invention_stack_pairs(list(style))
    for tok in _strip_invention_stack_pairs(list(base)):
        if tok not in merged:
            merged.append(tok)
    if inv:
        merged = ["--invention-stack", inv, *merged]
    return merged


# Competitor pain axes → ProcessOptions / repair modules
FAILURE_AXES: dict[str, str] = {
    "identity_drift": "identity_bind + identity_lock + shot_chain",
    "object_vanish": "permanence_repair + count_bind",
    "hand_morph": "extremity_lock",
    "logo_melt": "glyph_lock",
    "floating_feet": "contact_ground",
    "physics_break": "physics_gate + permanence assist",
    "flicker": "deflicker + hf_deshimmer + flow_consistency",
    "prompt_drift": "prompt_ground + min_adherence + invention maxwave",
    "occlusion_pop": "occlusion_resolve",
    "cut_jump": "shot_chain.harmonize_cut",
    "camera_jump": "camera_stabilize + camera_path",
    "lip_desync": "lip_sync + native_audio",
    "style_mismatch": "style_router + motion_grammar + animation_principles",
}


@dataclass
class VideoMaxPlan:
    active: bool = True
    axes: list[str] = field(default_factory=list)
    option_overrides: dict[str, Any] = field(default_factory=dict)
    sample_extras: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    style: str = ""
    grammar: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_videomax(
    prompt: str = "",
    *,
    tier: str = "max",
    style_hint: str = "",
    force_style: str = "",
) -> VideoMaxPlan:
    """Build VIDEOMAX option overrides + keyframe sample extras (style-aware)."""
    axes = list(FAILURE_AXES.keys())
    notes = [f"tier={tier}", f"axes={len(axes)}"]

    # Adult-only gate before any figure / NSFW enrichment on video keyframes
    try:
        from utils.generation.inventions.gates import adult_gate

        gate = adult_gate(prompt)
        if not gate.allowed:
            notes.append(f"VIDEOADULTGATE:{gate.reason}")
            return VideoMaxPlan(
                active=True,
                axes=axes,
                option_overrides={
                    "videomax": True,
                    "invention_stack": "videowave",
                    "identity_bind": False,
                    "extremity_lock": False,
                },
                sample_extras=["--invention-stack", "videowave"],
                notes=notes + ["adult_gate_block — non-figurative fallback"],
            )
        notes.append("adult_gate=ok")
    except Exception as exc:
        notes.append(f"adult_gate_skip:{exc}")

    overrides: dict[str, Any] = {
        "videomax": True,
        "video_quality": "strong" if tier != "lite" else "standard",
        "permanence_repair": True,
        "identity_bind": True,
        "identity_lock": True,
        "extremity_lock": True,
        "physics_gate": True,
        "contact_ground": True,
        "secondary_track": True,
        "occlusion_resolve": True,
        "hf_deshimmer": True,
        "count_bind": True,
        "count_bind_repair": True,
        "physics_repair": True,
        "shot_chain": True,
        "flow_consistency": True,
        "deflicker": True,
        "semantic_drift_repair": True,
        "cross_keyframe_identity": True,
        "depth_interpolate": True,
        "motion_beat_keyframes": True,
        "quality_retry": True,
        "leaders_stack": True,
        "prompt_ground": True,
        "camera_stabilize": True,
        "camera_path": True,
        "lip_sync": True,
        "native_audio": True,
        "in_gen_cuts": True,
        "motion_intel": True,
        "use_permanent_dit": True,
        "invention_stack": "videowave,artwave",
        # Raise gates (style router may soften for anime/cartoon)
        "min_permanence": 0.58,
        "min_artifact": 0.52,
        "min_identity": 0.58,
        "min_extremity": 0.38,
        "min_physics": 0.48,
        "min_contact": 0.42,
        "min_count": 0.42,
        "min_shimmer": 0.48,
        "min_adherence": 0.40,
        "min_occlusion": 0.30,
        "max_retries": 3,
        "deflicker_strength": 0.85,
        "temporal_smooth": 3,
        "temporal_alpha": 0.14,
        "cross_keyframe_identity_strength": 0.28,
        "shot_chain_strength": 0.40,
    }

    style_id = ""
    grammar = ""
    extras: list[str] = [
        "--invention-stack",
        "videowave,artwave",
        "--boost-quality",
        "--less-ai",
        "--naturalize",
        "--anti-perspective-drift",
    ]

    try:
        from pipelines.video.style_router import route_style

        route = route_style(prompt, style_hint=style_hint, force=force_style)
        style_id = route.style
        grammar = route.grammar
        overrides["motion_grammar"] = route.grammar
        if route.post_grade:
            overrides["post_grade"] = route.post_grade
        # Per-medium dials win over aggressive live-action defaults
        overrides.update(route.videomax_profile or {})
        notes.extend(route.notes)
        # Merge style sample extras with VIDEOMAX extras — union invention stacks
        extras = _merge_sample_extras(list(route.sample_extras or []), extras)
        # Keep opts.invention_stack as the union of style + videowave
        style_inv = _invention_stack_from_extras(route.sample_extras or [])
        base_inv = str(overrides.get("invention_stack") or "videowave,artwave")
        overrides["invention_stack"] = _union_invention_stacks(base_inv, style_inv)
        if route.positive_addon:
            overrides["motion_grammar_positive"] = route.positive_addon
        if route.negative_addon:
            overrides["motion_grammar_negative"] = route.negative_addon
    except Exception as exc:
        notes.append(f"style_router_fallback:{exc}")
        p = str(prompt or "").lower()
        if any(w in p for w in ("logo", "text", "sign", "product", "brand")):
            overrides["glyph_lock"] = True
            notes.append("glyph_lock for product/text cue")
        if any(w in p for w in ("anime", "manga", "cel", "sakuga")):
            overrides["motion_grammar"] = "anime_2d"
            grammar = "anime_2d"
        elif any(w in p for w in ("cartoon", "toon", "looney")):
            overrides["motion_grammar"] = "cartoon"
            grammar = "cartoon"
        elif any(w in p for w in ("vfx", "explosion", "houdini", "particle")):
            overrides["motion_grammar"] = "vfx"
            grammar = "vfx"
        elif any(w in p for w in ("film", "cinematic", "movie", "anamorphic")):
            overrides["motion_grammar"] = "film"
            overrides["motion_shutter"] = True
            grammar = "film"
        elif any(w in p for w in ("realistic", "photoreal", "live action")):
            overrides["motion_grammar"] = "realistic"
            overrides["motion_shutter"] = True
            grammar = "realistic"

    p = str(prompt or "").lower()
    if any(w in p for w in ("logo", "text", "sign", "product", "brand", "sku")):
        overrides.setdefault("glyph_lock", True)
        notes.append("glyph_lock for product/text cue")

    # Keep sample_extras invention-stack in sync with final override union
    final_inv = str(overrides.get("invention_stack") or "videowave,artwave")
    extras = _merge_sample_extras(extras, ["--invention-stack", final_inv])

    return VideoMaxPlan(
        active=True,
        axes=axes,
        option_overrides=overrides,
        sample_extras=extras,
        notes=notes,
        style=style_id,
        grammar=grammar,
    )


def apply_videomax_to_options(
    opts: Any,
    *,
    prompt: str = "",
    tier: str = "max",
    style_hint: str = "",
    force_style: str = "",
) -> Any:
    """Return ProcessOptions with VIDEOMAX consistency stack enabled."""
    plan = plan_videomax(prompt, tier=tier, style_hint=style_hint, force_style=force_style)
    kw = dict(plan.option_overrides)
    # First apply strong quality preset for sample packs
    from pipelines.video.video_helpers import apply_video_quality_preset

    vq = str(kw.pop("video_quality", "strong") or "strong")
    opts = apply_video_quality_preset(opts, vq)
    # Filter to fields that exist on opts
    fields = getattr(type(opts), "__dataclass_fields__", {}) or {}
    safe = {k: v for k, v in kw.items() if k in fields}
    try:
        opts = replace(
            opts,
            video_quality=vq if "video_quality" in fields else getattr(opts, "video_quality", vq),
            **safe,
        )
    except TypeError:
        for k, v in safe.items():
            try:
                setattr(opts, k, v)
            except Exception:
                pass
    # Full style stamp (grammar + principles + engine garnish)
    try:
        from pipelines.video.style_router import apply_style_route_to_options, route_style

        force = force_style or str(getattr(opts, "motion_grammar", "") or "")
        if force in ("auto", "none", ""):
            force = plan.grammar or plan.style or ""
        route = route_style(prompt, style_hint=style_hint, force=force)
        opts = apply_style_route_to_options(opts, route)
    except Exception:
        pass
    plan_dict = plan.to_dict()
    fields = getattr(type(opts), "__dataclass_fields__", {}) or {}
    if "videomax_plan" in fields:
        try:
            opts = replace(opts, videomax_plan=plan_dict)
        except TypeError:
            pass
    return opts


def videomax_sample_extras(opts: Any, *, prompt: str = "") -> list[str]:
    """Keyframe sample.py extras when VIDEOMAX is on."""
    if not bool(getattr(opts, "videomax", False)):
        return []
    plan_dict = getattr(opts, "videomax_plan", None) or getattr(opts, "_videomax_plan", None)
    if isinstance(plan_dict, dict) and plan_dict.get("sample_extras"):
        base = list(plan_dict["sample_extras"])
    else:
        route = getattr(opts, "style_route", None) or getattr(opts, "_style_route", None)
        if isinstance(route, dict) and route.get("sample_extras"):
            base = list(route["sample_extras"])
        else:
            base = list(plan_videomax(prompt).sample_extras)
    inv = _union_invention_stacks(
        _invention_stack_from_extras(base),
        str(getattr(opts, "invention_stack", "") or ""),
        "videowave,artwave",
    )
    if inv:
        base = _strip_invention_stack_pairs(base)
        base = ["--invention-stack", inv, *base]
    return base

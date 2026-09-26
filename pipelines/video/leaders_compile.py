"""
Leaders compile — fold Seedance/Hailuo/Wan-class controls into ProcessOptions.

One entry point so scene graphs and CLI can opt into the 2026 feature set:
multimodal refs, motion intelligence, camera paths, in-gen cuts, native audio.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

__all__ = ["LeadersBundle", "compile_leaders_stack"]


class LeadersBundle:
    def __init__(self) -> None:
        self.opts: Any = None
        self.ref_positive: str = ""
        self.ref_negative: str = ""
        self.motion_positive: str = ""
        self.motion_negative: str = ""
        self.camera_prompt: str = ""
        self.audio_plan: Any = None
        self.cut_plan: Any = None
        self.camera_path: Any = None
        self.motion_plan: Any = None
        self.ref_pack: Any = None
        self.ops: list[str] = []


def compile_leaders_stack(
    opts: Any,
    *,
    prompt: str = "",
    duration_sec: float = 6.0,
    fps: float = 24.0,
    frame_count: int = 0,
    multimodal_refs: Any = None,
    enable_native_audio: bool = True,
    enable_scene_cuts: bool = True,
    enable_camera_path: bool = True,
    enable_motion_intel: bool = True,
) -> LeadersBundle:
    from .camera_path_solver import path_to_prompt, solve_camera_path
    from .in_gen_scene_cuts import plan_scene_cuts
    from .motion_intelligence import apply_gate_boosts, compile_motion_intelligence, plan_to_prompt_fragments
    from .multimodal_ref_bus import (
        compile_ref_prompt_garnish,
        parse_multimodal_refs,
        ref_budget_ok,
        resolve_audio_paths,
        resolve_identity_paths,
    )
    from .native_audio_track import plan_native_audio

    bundle = LeadersBundle()
    bundle.opts = opts

    pack = parse_multimodal_refs(multimodal_refs)
    bundle.ref_pack = pack
    ok, issues = ref_budget_ok(pack)
    if pack.refs:
        bundle.ref_positive, bundle.ref_negative = compile_ref_prompt_garnish(pack)
        bundle.ops.append(f"multimodal_refs:{pack.counts['total']}")
        if not ok:
            bundle.ops.append(f"ref_budget_warn:{','.join(issues)}")
        # Feed identity images into identity_bind refs
        id_paths = resolve_identity_paths(pack)
        if id_paths:
            existing = tuple(getattr(opts, "identity_refs", ()) or ())
            bundle.opts = replace(opts, identity_refs=existing + tuple(id_paths), identity_bind=True)
            opts = bundle.opts

    if enable_motion_intel:
        mplan = compile_motion_intelligence(prompt, style=str(getattr(opts, "motion_grammar", "") or ""))
        bundle.motion_plan = mplan
        bundle.motion_positive, bundle.motion_negative = plan_to_prompt_fragments(mplan)
        bundle.opts = apply_gate_boosts(opts, mplan)
        opts = bundle.opts
        bundle.ops.append(f"motion_intel:{mplan.primary_action}")

    if enable_camera_path:
        cpath = solve_camera_path(prompt, rig_movement=str(getattr(opts, "motion_grammar", "") or ""))
        bundle.camera_path = cpath
        bundle.camera_prompt = path_to_prompt(cpath)
        bundle.ops.append(f"camera_path:{cpath.preset}")

    n_frames = int(frame_count) if frame_count > 0 else max(8, int(duration_sec * fps))
    if enable_scene_cuts:
        cuts = plan_scene_cuts(prompt, frame_count=n_frames, fps=fps)
        bundle.cut_plan = cuts
        if cuts.cuts:
            bundle.ops.append(f"scene_cuts:{len(cuts.cuts)}")

    if enable_native_audio or bool(getattr(opts, "native_audio", False)):
        voice = resolve_audio_paths(pack) if pack else []
        bundle.audio_plan = plan_native_audio(prompt, duration_sec=duration_sec, voice_refs=voice)
        bundle.opts = replace(opts, native_audio=True)
        bundle.ops.append("native_audio:planned")

    # Merge prompt garnish onto motion_grammar fields
    pos = ", ".join(
        x
        for x in (
            getattr(bundle.opts, "motion_grammar_positive", "") or "",
            bundle.ref_positive,
            bundle.motion_positive,
            bundle.camera_prompt,
        )
        if x
    )
    neg = ", ".join(
        x
        for x in (
            getattr(bundle.opts, "motion_grammar_negative", "") or "",
            bundle.ref_negative,
            bundle.motion_negative,
        )
        if x
    )
    bundle.opts = replace(bundle.opts, motion_grammar_positive=pos, motion_grammar_negative=neg)
    return bundle

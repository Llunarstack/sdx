"""
Video keyframe helpers — port image-gen ``sample.py`` quality flags into the
retrieve→edit→interpolate pipeline (book_helpers parity).

Keyframe edits historically used a minimal argv (prompt, init-image, cfg, steps).
This module builds structured extras from ``SampleOptions`` / quality presets so
video inherits pick-best, quality packs, architecture lock, human-made, dual-stage,
and holy-grail without opaque CLI leftovers alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

__all__ = [
    "SampleOptions",
    "parse_sample_options",
    "apply_video_quality_preset",
    "append_sample_py_quality_flags",
    "append_sample_py_pick_flags",
    "extend_sample_py_control_cmd",
    "build_keyframe_sample_extras",
    "soft_architecture_lock_args",
    "apply_cli_overrides_to_process_options",
    "process_options_to_edit_dict",
]


@dataclass(slots=True)
class SampleOptions:
    """Structured ``edit.sample`` / CLI options forwarded to ``sample.py``."""

    num: int = 1
    pick_best: str = "none"
    quality_pack: str = "none"
    adherence_pack: str = "none"
    human_made: str = "none"
    architecture_lock: str = "off"  # off|auto|on|hall|strict
    dual_stage_layout: bool = False
    holy_grail: bool = False
    anti_perspective_drift: bool = False
    steps: int | None = None
    cfg_scale: float | None = None
    cfg_rescale: float = 0.0
    boost_quality: bool = False
    less_ai: bool = False
    naturalize: bool = False
    shortcomings_mitigation: str = "none"
    anatomy_guidance: str = "none"
    continuity_pick_weight: float = 0.35
    """Blend weight for previous-keyframe continuity in temporal pick (0=ignore)."""
    control_image: str = ""
    control_type: str = "auto"
    control_scale: float = 0.85
    control: tuple[str, ...] = ()
    reference_image: str = ""
    style_ref: str = ""


def parse_sample_options(raw: Mapping[str, Any] | None) -> SampleOptions:
    r = dict(raw or {})
    ctrl = r.get("control") or ()
    if isinstance(ctrl, str):
        ctrl = (ctrl,) if ctrl.strip() else ()
    elif isinstance(ctrl, (list, tuple)):
        ctrl = tuple(str(x) for x in ctrl if str(x).strip())
    else:
        ctrl = ()
    steps = r.get("steps", None)
    cfg = r.get("cfg_scale", None)
    return SampleOptions(
        num=max(1, int(r.get("num", 1) or 1)),
        pick_best=str(r.get("pick_best", "none") or "none").strip().lower(),
        quality_pack=str(r.get("quality_pack", "none") or "none").strip().lower(),
        adherence_pack=str(r.get("adherence_pack", "none") or "none").strip().lower(),
        human_made=str(r.get("human_made", "none") or "none").strip().lower(),
        architecture_lock=str(r.get("architecture_lock", "off") or "off").strip().lower(),
        dual_stage_layout=bool(r.get("dual_stage_layout", False)),
        holy_grail=bool(r.get("holy_grail", False)),
        anti_perspective_drift=bool(r.get("anti_perspective_drift", False)),
        steps=int(steps) if steps is not None else None,
        cfg_scale=float(cfg) if cfg is not None else None,
        cfg_rescale=float(r.get("cfg_rescale", 0.0) or 0.0),
        boost_quality=bool(r.get("boost_quality", False)),
        less_ai=bool(r.get("less_ai", False)),
        naturalize=bool(r.get("naturalize", False)),
        shortcomings_mitigation=str(r.get("shortcomings_mitigation", "none") or "none").strip().lower(),
        anatomy_guidance=str(r.get("anatomy_guidance", "none") or "none").strip().lower(),
        continuity_pick_weight=float(r.get("continuity_pick_weight", 0.35) or 0.35),
        control_image=str(r.get("control_image", "") or "").strip(),
        control_type=str(r.get("control_type", "auto") or "auto").strip().lower(),
        control_scale=float(r.get("control_scale", 0.85) or 0.85),
        control=ctrl,
        reference_image=str(r.get("reference_image", "") or "").strip(),
        style_ref=str(r.get("style_ref", "") or "").strip(),
    )


def apply_video_quality_preset(opts: Any, preset: str) -> Any:
    """
    Mutate / return ProcessOptions-like object with quality preset defaults.

    ``lite`` | ``standard`` | ``strong`` — strengthens temporal path + sample soft-fills.
    """
    name = str(preset or "standard").strip().lower()
    if name in ("", "none", "off"):
        return opts
    if name in ("max", "videomax"):
        from pipelines.video.videomax import apply_videomax_to_options

        return apply_videomax_to_options(opts, tier="max")

    sample = getattr(opts, "sample", None)
    if sample is None:
        sample = SampleOptions()
    else:
        sample = replace(sample)

    if name == "lite":
        sample = replace(
            sample,
            num=max(sample.num, 1),
            architecture_lock="auto" if sample.architecture_lock == "off" else sample.architecture_lock,
        )
        return replace(
            opts,
            sample=sample,
            motion_beat_keyframes=bool(getattr(opts, "motion_beat_keyframes", False)),
            depth_interpolate=bool(getattr(opts, "depth_interpolate", False)),
            quality_retry=True,
            max_retries=max(int(getattr(opts, "max_retries", 1) or 1), 1),
        )

    if name == "strong":
        sample = replace(
            sample,
            num=max(sample.num, 3),
            pick_best=sample.pick_best if sample.pick_best not in ("none", "") else "combo_vit",
            quality_pack=sample.quality_pack if sample.quality_pack != "none" else "top",
            adherence_pack=sample.adherence_pack if sample.adherence_pack != "none" else "standard",
            human_made=sample.human_made if sample.human_made != "none" else "standard",
            architecture_lock="auto" if sample.architecture_lock == "off" else sample.architecture_lock,
            dual_stage_layout=True,
            holy_grail=True,
            anti_perspective_drift=True,
            boost_quality=True,
            less_ai=True,
            naturalize=True,
            cfg_rescale=max(float(sample.cfg_rescale), 0.7),
            continuity_pick_weight=max(float(sample.continuity_pick_weight), 0.4),
        )
        return replace(
            opts,
            sample=sample,
            motion_beat_keyframes=True,
            depth_interpolate=True,
            flow_consistency=True,
            deflicker=True,
            deflicker_strength=max(float(getattr(opts, "deflicker_strength", 0.75) or 0.75), 0.82),
            identity_lock=True,
            semantic_drift_repair=True,
            quality_retry=True,
            max_retries=max(int(getattr(opts, "max_retries", 2) or 2), 3),
            temporal_alpha=max(float(getattr(opts, "temporal_alpha", 0.10) or 0.10), 0.12),
            temporal_smooth=max(int(getattr(opts, "temporal_smooth", 2) or 2), 3),
            permanence_repair=True,
            min_permanence=max(float(getattr(opts, "min_permanence", 0.45) or 0.45), 0.55),
            min_artifact=max(float(getattr(opts, "min_artifact", 0.40) or 0.40), 0.50),
            use_permanent_dit=True,
            identity_bind=True,
            extremity_lock=True,
            physics_gate=True,
            contact_ground=True,
            hf_deshimmer=True,
            count_bind=True,
            secondary_track=True,
            leaders_stack=True,
            native_audio=True,
            in_gen_cuts=True,
            motion_intel=True,
            camera_path=True,
            lip_sync=True,
            prompt_ground=True,
            min_adherence=max(float(getattr(opts, "min_adherence", 0.0) or 0.0), 0.35),
            min_identity=max(float(getattr(opts, "min_identity", 0.50) or 0.50), 0.55),
            min_extremity=max(float(getattr(opts, "min_extremity", 0.30) or 0.30), 0.35),
            min_physics=max(float(getattr(opts, "min_physics", 0.40) or 0.40), 0.45),
            min_contact=max(float(getattr(opts, "min_contact", 0.35) or 0.35), 0.40),
            min_shimmer=max(float(getattr(opts, "min_shimmer", 0.40) or 0.40), 0.45),
            min_count=max(float(getattr(opts, "min_count", 0.35) or 0.35), 0.40),
        )

    # standard
    sample = replace(
        sample,
        num=max(sample.num, 2),
        pick_best=sample.pick_best if sample.pick_best not in ("none", "") else "combo_vit",
        architecture_lock="auto" if sample.architecture_lock == "off" else sample.architecture_lock,
        anti_perspective_drift=True,
        boost_quality=True,
    )
    return replace(
        opts,
        sample=sample,
        motion_beat_keyframes=True,
        depth_interpolate=True,
        flow_consistency=True,
        deflicker=True,
        quality_retry=True,
        max_retries=max(int(getattr(opts, "max_retries", 2) or 2), 2),
    )


def append_sample_py_pick_flags(cmd: list[str], sample: SampleOptions) -> None:
    n = max(1, int(sample.num))
    metric = str(sample.pick_best or "none").strip().lower()
    if n > 1 or metric not in ("none", ""):
        cmd.extend(["--num", str(n)])
        if metric not in ("none", ""):
            cmd.extend(["--pick-best", metric])


def append_sample_py_quality_flags(cmd: list[str], sample: SampleOptions) -> None:
    qp = str(sample.quality_pack or "none").lower()
    if qp != "none":
        cmd.extend(["--quality-pack", qp])
    ap = str(sample.adherence_pack or "none").lower()
    if ap != "none":
        cmd.extend(["--adherence-pack", ap])
    hm = str(sample.human_made or "none").lower()
    if hm not in ("none", "off", "0", ""):
        cmd.extend(["--human-made", hm])
    al = str(sample.architecture_lock or "off").lower()
    if al not in ("off", "none", "0", "false"):
        cmd.extend(["--architecture-lock", al])
    if sample.dual_stage_layout:
        cmd.append("--dual-stage-layout")
    if sample.holy_grail:
        cmd.append("--holy-grail")
    if sample.anti_perspective_drift:
        cmd.append("--anti-perspective-drift")
    if sample.boost_quality:
        cmd.append("--boost-quality")
    if sample.less_ai:
        cmd.append("--less-ai")
    if sample.naturalize:
        cmd.append("--naturalize")
    sm = str(sample.shortcomings_mitigation or "none").lower()
    if sm in ("auto", "all"):
        cmd.extend(["--shortcomings-mitigation", sm])
    anat = str(sample.anatomy_guidance or "none").lower()
    if anat in ("auto", "lite", "strong"):
        cmd.extend(["--anatomy-guidance", anat])
    if float(sample.cfg_rescale) > 0.0:
        cmd.extend(["--cfg-rescale", str(float(sample.cfg_rescale))])


def extend_sample_py_control_cmd(cmd: list[str], sample: SampleOptions) -> None:
    cimg = str(sample.control_image or "").strip()
    if cimg:
        cmd.extend(["--control-image", cimg])
        cmd.extend(["--control-type", str(sample.control_type or "auto")])
        cmd.extend(["--control-scale", str(float(sample.control_scale))])
    if sample.control:
        cmd.extend(["--control"] + [str(x) for x in sample.control])
    ref = str(sample.reference_image or "").strip()
    if ref:
        cmd.extend(["--reference-image", ref])
    sref = str(sample.style_ref or "").strip()
    if sref:
        cmd.extend(["--style-ref", sref])


def soft_architecture_lock_args(prompt: str, sample: SampleOptions) -> SampleOptions:
    """Soft-enable architecture lock when the shot prompt needs depth structure."""
    mode = str(sample.architecture_lock or "off").lower()
    if mode in ("off", "none", "0", "false"):
        return sample
    if mode != "auto":
        return sample
    try:
        from utils.quality.architecture_consistency import prompt_needs_architecture_lock

        if prompt_needs_architecture_lock(prompt or ""):
            return replace(sample, architecture_lock="on", anti_perspective_drift=True)
    except Exception:
        pass
    return sample


def build_keyframe_sample_extras(
    sample: SampleOptions | None,
    *,
    prompt: str = "",
    extra_args: Sequence[str] | None = None,
) -> list[str]:
    """Build argv fragment (no program) for one keyframe ``sample.py`` call."""
    s = sample or SampleOptions()
    s = soft_architecture_lock_args(prompt, s)
    cmd: list[str] = []
    append_sample_py_pick_flags(cmd, s)
    append_sample_py_quality_flags(cmd, s)
    extend_sample_py_control_cmd(cmd, s)
    if extra_args:
        for tok in extra_args:
            if tok.startswith("-") and tok in cmd:
                continue
            cmd.append(str(tok))
    return cmd


def apply_cli_overrides_to_process_options(
    opts: Any,
    *,
    video_quality: str = "none",
    keyframe_pick_best: str = "",
    keyframe_num: int = 0,
    architecture_lock: str = "",
    neural_engine: bool = False,
    motion_grammar: str = "",
    permanence_repair: bool | None = None,
    use_permanent_dit: bool = False,
    identity_bind: bool | None = None,
    glyph_lock: bool | None = None,
    extremity_lock: bool | None = None,
    physics_gate: bool | None = None,
    leaders_stack: bool = False,
    native_audio: bool = False,
    videomax: bool = False,
) -> Any:
    """Merge generate_video CLI flags onto ProcessOptions."""
    from dataclasses import replace

    if videomax:
        from pipelines.video.videomax import apply_videomax_to_options

        opts = apply_videomax_to_options(opts, tier="max")
        return opts

    vq = str(video_quality or "none").strip().lower()
    if vq not in ("none", "", "off"):
        opts = apply_video_quality_preset(opts, vq)
        opts = replace(opts, video_quality=vq)

    sample = opts.sample
    kpb = str(keyframe_pick_best or "").strip().lower()
    if kpb and kpb not in ("none", "off", ""):
        sample = replace(sample, pick_best=kpb)
    kn = int(keyframe_num or 0)
    if kn > 1:
        sample = replace(sample, num=kn)
    al = str(architecture_lock or "").strip().lower()
    if al and al not in ("",):
        sample = replace(sample, architecture_lock=al)
    mg = str(motion_grammar or "").strip().lower()
    kw: dict[str, Any] = {
        "sample": sample,
        "neural_engine": bool(neural_engine) or bool(getattr(opts, "neural_engine", False)),
    }
    if mg and mg not in ("", "none"):
        kw["motion_grammar"] = mg
    if permanence_repair is not None:
        kw["permanence_repair"] = bool(permanence_repair)
    if use_permanent_dit:
        kw["use_permanent_dit"] = True
        kw["neural_engine"] = True
    if identity_bind is not None:
        kw["identity_bind"] = bool(identity_bind)
    if glyph_lock is not None:
        kw["glyph_lock"] = bool(glyph_lock)
    if extremity_lock is not None:
        kw["extremity_lock"] = bool(extremity_lock)
    if physics_gate is not None:
        kw["physics_gate"] = bool(physics_gate)
    if leaders_stack:
        kw["leaders_stack"] = True
        kw["native_audio"] = True
        kw["in_gen_cuts"] = True
        kw["motion_intel"] = True
        kw["camera_path"] = True
        kw["lip_sync"] = True
    if native_audio:
        kw["native_audio"] = True
    return replace(opts, **kw)


def process_options_to_edit_dict(opts: Any) -> dict[str, Any]:
    """Serialize ProcessOptions back into a scene ``edit`` mapping."""
    from dataclasses import asdict

    keys = (
        "motion_beat_keyframes",
        "depth_interpolate",
        "flow_consistency",
        "deflicker",
        "deflicker_strength",
        "identity_lock",
        "semantic_drift_repair",
        "quality_retry",
        "max_retries",
        "temporal_alpha",
        "temporal_smooth",
        "cross_keyframe_identity",
        "cross_keyframe_identity_strength",
        "real_depth",
        "neural_engine",
        "video_quality",
        "keyframe_interval",
        "edit_strength",
        "permanence_repair",
        "permanence_strength",
        "min_permanence",
        "min_artifact",
        "motion_grammar",
        "use_permanent_dit",
        "identity_bind",
        "identity_bind_strength",
        "extremity_lock",
        "glyph_lock",
        "physics_gate",
        "min_identity",
        "min_extremity",
        "min_glyph",
        "min_physics",
        "prompt_ground",
        "min_adherence",
        "leaders_stack",
        "native_audio",
    )
    out = {k: getattr(opts, k) for k in keys if hasattr(opts, k)}
    out["sample"] = asdict(opts.sample)
    return out


def sample_options_as_namespace(sample: SampleOptions) -> Any:
    """SimpleNamespace compatible with book_helpers extend_* if needed."""
    from types import SimpleNamespace

    return SimpleNamespace(**{f.name: getattr(sample, f.name) for f in fields(SampleOptions)})


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

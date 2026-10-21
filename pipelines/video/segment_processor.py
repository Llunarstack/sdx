"""Process one segment: extract → keyframe edit → motion transfer → interpolate → temporal → quality.

Also holds what used to be: video_helpers, process_options, provenance, editor, temporal, pose_control, videomax, segment_retry.
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import MISSING, asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
from utils.generation.sample_cli_passthrough import sample_argv
from utils.safety.minor_guard import require_allowed

from .motion import schedule_keyframes_for_segment
from .motion_grammar import MotionGrammar, apply_motion_grammar, grammar_for_engine, grammar_for_prompt
from .quality import score_segment_quality
from .types import KeyframeEditJob, ProvenanceRecord, SegmentAssignment, VideoPlan
from .video_io import (
    extract_frames,
    frame_count_for_duration,
    normalize_segment_window,
    read_frame_rgb,
    save_frame_rgb,
    write_video_from_frames,
)

# ==================================================================================================
# video_helpers: Keyframe ``sample.py`` options: the scene ``edit.sample`` block and its argv.
# ==================================================================================================


@dataclass(slots=True)
class SampleOptions:
    """``sample.py`` options for keyframe edits; field names are the sample.py argparse dests."""

    num: int = 1
    pick_best: str = "none"
    quality_pack: str = "none"
    adherence_pack: str = "none"
    human_made: str = "none"
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
    control_image: str = ""
    control_type: str = "auto"
    control_scale: float = 0.85
    control: tuple[str, ...] = ()
    reference_image: str = ""
    style_ref: str = ""


_PATH_FIELDS = ("control_image", "reference_image", "style_ref")


def parse_sample_options(raw: Mapping[str, Any] | None) -> SampleOptions:
    kw: dict[str, Any] = {}
    for f in fields(SampleOptions):
        v = (raw or {}).get(f.name)
        if v is None:
            continue
        if f.name == "control":
            v = tuple(c for c in ((v,) if isinstance(v, str) else map(str, v)) if c.strip())
        elif f.name == "steps":
            v = int(v)
        elif f.name == "cfg_scale":
            v = float(v)
        elif isinstance(f.default, str):
            v = str(v).strip() if f.name in _PATH_FIELDS else str(v).strip().lower()
        else:
            v = type(f.default)(v)
        kw[f.name] = v
    opts = SampleOptions(**kw)
    opts.num = max(1, opts.num)
    return opts


def merge_argv(base: Sequence[str], extra: Sequence[str]) -> list[str]:
    """Append ``extra`` to ``base``, skipping any flag ``base`` already sets together with that flag's values."""
    out = list(base)
    have = {tok for tok in out if tok.startswith("--")}
    skipping = False
    for tok in extra:
        if tok.startswith("--"):
            skipping = tok in have
        if not skipping:
            out.append(tok)
    return out


def build_keyframe_sample_extras(sample: SampleOptions | None) -> list[str]:
    """Keyframe argv fragment; ``steps`` / ``cfg_scale`` are set by the editor's base command."""
    values = asdict(sample or SampleOptions())
    values.pop("steps")
    values.pop("cfg_scale")
    return sample_argv(values)


# ==================================================================================================
# process_options: Pipeline processing options (from the scene ``edit`` block).
# ==================================================================================================


@dataclass(slots=True)
class ProcessOptions:
    motion_transfer: bool = True
    motion_transfer_retrieved: bool = True
    region_motion: bool = False
    identity_lock: bool = True
    identity_lock_strength: float = 0.82
    propagate_masks: bool = True
    depth_interpolate: bool = False
    camera_stabilize: bool = False
    camera_stabilize_strength: float = 0.65
    deflicker: bool = True
    deflicker_strength: float = 0.75
    motion_beat_keyframes: bool = False
    flow_consistency: bool = True
    frame_enhance: bool = False
    frame_enhance_amount: float = 0.32
    semantic_drift_repair: bool = True
    drift_threshold: float = 0.55
    drift_blend_strength: float = 0.45
    velocity_ease: bool = False
    velocity_ease_mode: str = "smooth"
    quality_retry: bool = True
    max_retries: int = 2
    temporal_alpha: float = 0.10
    temporal_smooth: int = 2
    post_grade: str = ""
    pose_control: bool = False
    keyframe_interval: int = 6
    edit_strength: float = 0.55
    audio_from_source: bool = False
    parallel_segments: bool = False
    max_segment_workers: int = 2
    sample: SampleOptions = field(default_factory=SampleOptions)
    # Presets resolved by videomax.apply_video_quality: none | lite | standard | strong | max.
    video_quality: str = "none"
    videomax: bool = False
    motion_grammar: str = "auto"
    # Segment quality gates (0 = axis not measured); see quality.score_segment_quality.
    min_permanence: float = 0.45
    min_artifact: float = 0.40
    min_identity: float = 0.0
    min_extremity: float = 0.0
    min_physics: float = 0.0
    min_glyph: float = 0.0


def parse_process_options(raw: Mapping[str, Any] | None) -> ProcessOptions:
    """Read every ``ProcessOptions`` field present in ``raw``; missing or null keys keep their defaults."""
    r = dict(raw or {})
    kw: dict[str, Any] = {}
    for f in fields(ProcessOptions):
        value = r.get(f.name)
        if value is None or f.default is MISSING:
            continue
        kw[f.name] = type(f.default)(value)
    if isinstance(r.get("sample"), Mapping):
        kw["sample"] = parse_sample_options(r["sample"])
    kw["video_quality"] = str(kw.get("video_quality", "none")).strip().lower()
    kw["videomax"] = bool(kw.get("videomax")) or kw["video_quality"] in ("max", "videomax")
    return ProcessOptions(**kw)


# ==================================================================================================
# provenance: Provenance logging for video segments.
# ==================================================================================================


class ProvenanceLog:
    def __init__(self) -> None:
        self.records: list[ProvenanceRecord] = []

    def add(self, record: ProvenanceRecord) -> None:
        self.records.append(record)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "segments": [
                {
                    "segment_index": r.segment_index,
                    "source": r.source,
                    "source_path": r.source_path,
                    "license": r.license,
                    "url": r.url,
                    "operations": list(r.operations),
                    "retrieved_at": r.retrieved_at,
                }
                for r in self.records
            ],
        }


def record_segment_provenance(
    seg: SegmentAssignment,
    *,
    operations: list[str] | None = None,
) -> ProvenanceRecord:
    clip = seg.clip
    return ProvenanceRecord(
        segment_index=seg.shot.index,
        source=str(clip.source.value if clip else "synthetic"),
        source_path=str(clip.path if clip else ""),
        license=str(clip.license if clip else "generated"),
        url=str(clip.url if clip else ""),
        operations=list(operations or []),
        retrieved_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
    )


def write_provenance_json(log: ProvenanceLog, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(log.to_dict(), indent=2), encoding="utf-8")
    return p


# ==================================================================================================
# editor: Keyframe editor: invoke sample.py on selected frames.
# ==================================================================================================


def build_sample_cmd_for_keyframe(
    job: KeyframeEditJob,
    *,
    ckpt: str,
    repo_root: Path | None = None,
    width: int = 0,
    height: int = 0,
    steps: int = 20,
    cfg_scale: float = 6.5,
    seed: int = 42,
    sample_options: SampleOptions | None = None,
) -> list[str]:
    root = repo_root or Path(__file__).resolve().parents[3]
    if sample_options is not None:
        steps = sample_options.steps if sample_options.steps is not None else steps
        cfg_scale = sample_options.cfg_scale if sample_options.cfg_scale is not None else cfg_scale
    cmd = [
        sys.executable,
        str(root / "sample.py"),
        "--ckpt",
        ckpt,
        "--prompt",
        job.prompt,
        "--negative-prompt",
        job.negative or " ",
        "--init-image",
        job.source_frame_path,
        "--out",
        job.output_path,
        "--steps",
        str(int(steps)),
        "--cfg-scale",
        str(float(cfg_scale)),
        "--seed",
        str(int(seed) + job.frame_index),
    ]
    if width > 0 and height > 0:
        cmd.extend(["--width", str(int(width)), "--height", str(int(height))])
    strength = float(job.init_strength)
    if strength > 0:
        cmd.extend(["--strength", str(strength)])
    if job.mask_path and Path(job.mask_path).is_file():
        cmd.extend(["--mask", job.mask_path, "--inpaint-mode", "mdm"])
    if sample_options is not None:
        cmd = merge_argv(cmd, build_keyframe_sample_extras(sample_options))
    return merge_argv(cmd, job.sample_extra_args or ())


def run_keyframe_edit(
    job: KeyframeEditJob,
    *,
    ckpt: str,
    dry_run: bool = False,
    **kwargs: Any,
) -> Path:
    Path(job.output_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = build_sample_cmd_for_keyframe(job, ckpt=ckpt, **kwargs)
    if dry_run:
        return Path(job.output_path)
    subprocess.run(cmd, check=True)
    return Path(job.output_path)


def run_keyframe_batch(
    jobs: Sequence[KeyframeEditJob],
    *,
    ckpt: str,
    dry_run: bool = False,
    **kwargs: Any,
) -> list[Path]:
    outs: list[Path] = []
    for job in jobs:
        outs.append(run_keyframe_edit(job, ckpt=ckpt, dry_run=dry_run, **kwargs))
    return outs


# ==================================================================================================
# temporal: In-place filters on a segment's frame files, run after interpolation.
#
# Temporal harmonize (EMA toward the previous frame plus a light window blend), deflicker (rolling-median
# exposure), velocity retiming (eased acceleration or holds on twos), unsharp clarity and color grade.
# ==================================================================================================


def _rewrite(frame_paths: list[Path], fn: Callable[[np.ndarray], np.ndarray]) -> list[Path]:
    for p in frame_paths:
        save_frame_rgb(p, fn(read_frame_rgb(p)))
    return frame_paths


def harmonize_frame_sequence(
    frame_paths: list[Path],
    *,
    alpha_prev: float = 0.12,
    smooth_window: int = 0,
) -> list[Path]:
    """Blend each frame ``alpha_prev`` toward the previous output, then 8% toward its ±``smooth_window`` mean."""
    if not frame_paths:
        return frame_paths
    a = float(max(0.0, min(0.95, alpha_prev)))
    rgbs = [read_frame_rgb(p).astype(np.float32) for p in frame_paths]
    for i in range(1, len(rgbs)):
        rgbs[i] = (1.0 - a) * rgbs[i] + a * rgbs[i - 1]
    if smooth_window > 0 and len(rgbs) > 1:
        w = int(smooth_window)
        means = [np.mean(rgbs[max(0, i - w) : i + w + 1], axis=0) for i in range(len(rgbs))]
        rgbs = [0.92 * x + 0.08 * m for x, m in zip(rgbs, means)]
    for p, rgb in zip(frame_paths, rgbs):
        save_frame_rgb(p, np.clip(rgb, 0, 255).astype(np.uint8))
    return frame_paths


def apply_deflicker(frame_paths: list[Path], *, window: int = 5, strength: float = 0.75) -> list[Path]:
    """Pull each frame's mean luma toward the rolling median (flicker from per-frame img2img edits)."""
    if not frame_paths:
        return []
    lumas = [float(read_frame_rgb(p)[..., :3].astype(np.float32).mean()) for p in frame_paths]
    half = max(1, int(window)) // 2
    s = float(max(0.0, min(1.0, strength)))
    for i, p in enumerate(frame_paths):
        if lumas[i] < 1e-3:
            continue
        target = float(np.median(lumas[max(0, i - half) : i + half + 1]))
        gain = 1.0 + (target / lumas[i] - 1.0) * s
        save_frame_rgb(p, np.clip(read_frame_rgb(p).astype(np.float32) * gain, 0, 255).astype(np.uint8))
    return frame_paths


def _smoothstep(u: float) -> float:
    return u * u * (3.0 - 2.0 * u)


_EASES: dict[str, Callable[[float], float]] = {
    "smooth": _smoothstep,
    "ease_in_out": _smoothstep,
    "ease_in": lambda u: u * u,
    "ease_out": lambda u: 1.0 - (1.0 - u) ** 2,
}


def ease_indices(count: int, *, ease: str = "smooth") -> list[int]:
    """Source frame for each output frame. ``hold`` animates on twos; ``linear`` and unknown modes keep timing."""
    if count <= 0:
        return []
    if ease == "hold":
        return [i - i % 2 for i in range(count)]
    curve = _EASES.get(ease)
    if curve is None or count == 1:
        return list(range(count))
    last = count - 1
    return [round(curve(t / last) * last) for t in range(count)]


def apply_velocity_ease(frame_paths: list[Path], *, ease: str = "smooth") -> list[Path]:
    if len(frame_paths) < 3:
        return frame_paths
    rgbs = [read_frame_rgb(p) for p in frame_paths]
    for p, j in zip(frame_paths, ease_indices(len(rgbs), ease=ease)):
        save_frame_rgb(p, rgbs[j])
    return frame_paths


def enhance_frame(rgb: np.ndarray, *, amount: float = 0.35) -> np.ndarray:
    """Unsharp mask (no-op without OpenCV)."""
    try:
        import cv2
    except ImportError:
        return rgb
    a = float(max(0.0, min(1.0, amount)))
    blur = cv2.GaussianBlur(rgb, (0, 0), sigmaX=1.2)
    return np.clip(cv2.addWeighted(rgb, 1.0 + a, blur, -a, 0), 0, 255).astype(np.uint8)


def enhance_sequence(frame_paths: list[Path], *, amount: float = 0.35) -> list[Path]:
    return _rewrite(frame_paths, lambda rgb: enhance_frame(rgb, amount=amount))


GRADES: dict[str, dict[str, float]] = {
    "cinematic": {"contrast": 1.08, "saturation": 0.92, "warmth": 8},
    "teal_orange": {"contrast": 1.12, "saturation": 1.05, "warmth": 12, "shadow_teal": 0.06},
    "muted": {"contrast": 0.95, "saturation": 0.78, "warmth": 0},
    "vibrant": {"contrast": 1.05, "saturation": 1.18, "warmth": 4},
}


def grade_frame(rgb: np.ndarray, preset: str = "cinematic") -> np.ndarray:
    p = GRADES[preset.lower()]
    x = (rgb.astype(np.float32) - 127.5) * p["contrast"] + 127.5
    gray = x.mean(axis=2, keepdims=True)
    x = gray + (x - gray) * p["saturation"]
    x[..., 0] += p["warmth"]
    x[..., 2] -= p["warmth"] * 0.35
    teal = p.get("shadow_teal", 0.0)
    if teal > 0:
        shadow = (1.0 - x / 255.0).mean(axis=2)
        x[..., 1] += shadow * teal * 40
        x[..., 2] += shadow * teal * 55
    return np.clip(x, 0, 255).astype(np.uint8)


def apply_grade_to_sequence(frame_paths: list[Path], preset: str) -> list[Path]:
    if not preset or preset.lower() in ("none", "off"):
        return frame_paths
    if preset.lower() not in GRADES:
        raise ValueError(f"unknown post_grade {preset!r}; expected one of {sorted(GRADES)}")
    return _rewrite(frame_paths, lambda rgb: grade_frame(rgb, preset))


# ==================================================================================================
# segment_processor
# ==================================================================================================


def synthetic_segment_from_color(
    work_dir: Path,
    *,
    width: int,
    height: int,
    frame_count: int,
    rgb: tuple[int, int, int] = (32, 32, 48),
) -> list[Path]:
    import numpy as np

    from .video_io import save_frame_rgb

    frames_dir = work_dir / "synthetic_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:] = rgb
    paths: list[Path] = []
    for i in range(frame_count):
        fp = frames_dir / f"frame_{i + 1:06d}.png"
        save_frame_rgb(fp, arr)
        paths.append(fp)
    return paths


def _resolve_options(plan: VideoPlan, process_options: ProcessOptions | None) -> ProcessOptions:
    if process_options is not None:
        return process_options

    edit = plan.metadata.get("edit") if plan.metadata else None
    return apply_video_quality(parse_process_options(edit if isinstance(edit, dict) else {}), prompt=plan.user_prompt)


def _collect_lock_anchor(control_plan: Any, plan: VideoPlan) -> tuple[str, str]:
    anchor = str(plan.metadata.get("anchor_image") or "")
    mask = ""
    if control_plan is None:
        return anchor, mask
    for b in getattr(control_plan, "bindings", []) or []:
        ctrl = getattr(b, "control", None)
        ctrl_name = ctrl.value if hasattr(ctrl, "value") else str(ctrl or "").lower()
        if ctrl_name in ("lock", "identity"):
            if getattr(b, "image", ""):
                anchor = anchor or str(b.image)
            if getattr(b, "mask_path", ""):
                mask = str(b.mask_path)
    if not anchor and getattr(control_plan, "init_image", ""):
        anchor = str(control_plan.init_image)
    return anchor, mask


def _rig_path_from_control(control_plan: Any) -> str:
    if control_plan is None:
        return ""
    rig_path = getattr(control_plan, "box_layout_path", "") or ""
    if rig_path and Path(rig_path).is_file():
        return rig_path
    for b in getattr(control_plan, "bindings", []) or []:
        if getattr(b, "rig_json", "") and Path(b.rig_json).is_file():
            return str(b.rig_json)
    return ""


def _should_motion_transfer(opts: ProcessOptions, assignment: SegmentAssignment, source_frames: list[Path]) -> bool:
    if not opts.motion_transfer or not source_frames:
        return False
    if assignment.use_motion_only:
        return True
    return bool(opts.motion_transfer_retrieved and assignment.clip is not None)


def process_segment(
    assignment: SegmentAssignment,
    plan: VideoPlan,
    work_dir: str | Path,
    *,
    ckpt: str = "",
    dry_run: bool = False,
    edit_strength: float | None = None,
    provenance: ProvenanceLog | None = None,
    sample_extra_args: Sequence[str] | None = None,
    control_plan: Any = None,
    process_options: ProcessOptions | None = None,
) -> tuple[list[Path], Any]:
    """Returns (final_frame_paths, quality_report)."""
    opts = _resolve_options(plan, process_options)
    wd = Path(work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    seg_i = assignment.shot.index
    tl = plan.timeline
    target_frames = frame_count_for_duration(assignment.shot.duration_sec, tl.fps)

    ops: list[str] = []
    clip = assignment.clip
    source_frames: list[Path] = []

    if control_plan is not None and getattr(control_plan, "metadata", {}).get("element_video_refs"):
        vrefs = control_plan.metadata["element_video_refs"]
        if vrefs and not (clip and Path(clip.path).is_file()):
            from .retrieval import build_clip_candidate_from_path

            vpath = str(vrefs[0])
            if Path(vpath).is_file():
                clip = build_clip_candidate_from_path(vpath)
                assignment = SegmentAssignment(
                    shot=assignment.shot,
                    clip=clip,
                    in_sec=assignment.in_sec,
                    out_sec=assignment.out_sec,
                    use_motion_only=True,
                    keyframe_interval=assignment.keyframe_interval,
                    edit_strength=assignment.edit_strength,
                    transition=assignment.transition,
                    start_image=assignment.start_image,
                    end_image=assignment.end_image,
                    flf2v=assignment.flf2v,
                    motion_brush=assignment.motion_brush,
                )
                ops.append("element_video_ref")

    if clip and Path(clip.path).is_file():
        info = probe_video_safe(clip.path)
        in_sec, out_sec, speed = normalize_segment_window(
            float(info.get("duration_sec") or clip.duration_sec or assignment.shot.duration_sec),
            assignment.shot.duration_sec,
            in_sec=assignment.in_sec,
        )
        raw_dir = wd / "source_frames"
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        extract_fps = tl.fps * speed if speed < 0.999 else tl.fps
        source_frames = extract_frames(clip.path, raw_dir, max_frames=target_frames, fps=extract_fps)
        ops.append(f"extract:{clip.source.value}")
        if assignment.use_motion_only:
            ops.append("motion_template")
        if opts.camera_stabilize and len(source_frames) >= 2:
            from .motion import stabilize_frame_paths

            stabilize_frame_paths(source_frames, strength=opts.camera_stabilize_strength)
            ops.append("camera_stabilize")
    elif clip and clip.path.startswith("http"):
        raise FileNotFoundError(f"remote clip not downloaded: {clip.path}")
    else:
        source_frames = synthetic_segment_from_color(wd, width=tl.width, height=tl.height, frame_count=target_frames)
        ops.append("synthetic_fallback")

    if not source_frames:
        source_frames = synthetic_segment_from_color(wd, width=tl.width, height=tl.height, frame_count=target_frames)

    beat_indices: list[int] | None = None
    if opts.motion_beat_keyframes and len(source_frames) >= 3:
        try:
            from .motion import detect_motion_beats

            beat_indices = detect_motion_beats(source_frames)
            ops.append("motion_beats")
        except Exception:
            beat_indices = None

    schedule = schedule_keyframes_for_segment(
        source_frames,
        segment_index=seg_i,
        duration_sec=assignment.shot.duration_sec,
        fps=tl.fps,
        interval=assignment.keyframe_interval or opts.keyframe_interval,
        beat_indices=beat_indices,
    )

    edited_dir = wd / "keyframes_edited"
    edited_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[KeyframeEditJob] = []
    strength = float(edit_strength if edit_strength is not None else assignment.edit_strength or opts.edit_strength)

    seg_extra = list(sample_extra_args or [])
    if opts.pose_control and control_plan is not None:
        rig_path = _rig_path_from_control(control_plan)
        if rig_path and Path(rig_path).is_file():
            try:
                import json

                data = json.loads(Path(rig_path).read_text(encoding="utf-8"))
                parts = [(r["name"], tuple(r["box"])) for r in data.get("regions", []) if "box" in r]
                pose_p = wd / "pose_control.png"
                write_pose_control_image(parts, pose_p, width=tl.width, height=tl.height)
                seg_extra.extend(pose_control_args(pose_p))
                ops.append("pose_control")
            except Exception:
                pass

    for k_i, (f_idx, src_path) in enumerate(zip(schedule.frame_indices, schedule.frame_paths)):
        out_p = edited_dir / f"kf_{k_i:03d}.png"
        if dry_run or not ckpt:
            shutil.copy2(src_path, out_p)
            ops.append("keyframe:passthrough")
        else:
            jobs.append(
                KeyframeEditJob(
                    segment_index=seg_i,
                    frame_index=f_idx,
                    source_frame_path=str(src_path),
                    output_path=str(out_p),
                    prompt=assignment.shot.prompt,
                    negative=f"{plan.global_negative}, {assignment.shot.negative}".strip(", "),
                    init_strength=strength,
                    sample_extra_args=list(seg_extra),
                )
            )

    if jobs and ckpt and not dry_run:
        run_keyframe_batch(
            jobs,
            ckpt=ckpt,
            width=tl.width,
            height=tl.height,
            sample_options=opts.sample,
        )
        ops.append("keyframe:sample.py")

    edited_key_paths = sorted(edited_dir.glob("kf_*.png"))
    if not edited_key_paths:
        edited_key_paths = schedule.frame_paths

    brush_spec = None
    if assignment.motion_brush:
        from .motion import parse_motion_brush

        brush_spec = parse_motion_brush(assignment.motion_brush)

    use_flf2v = bool(
        assignment.flf2v
        or (
            assignment.start_image
            and assignment.end_image
            and Path(assignment.start_image).is_file()
            and Path(assignment.end_image).is_file()
        )
    )

    if use_flf2v:
        from .interpolate import interpolate_flf2v_sequence

        flf_dir = wd / "flf2v"
        if flf_dir.exists():
            shutil.rmtree(flf_dir)
        start_p = assignment.start_image or str(edited_key_paths[0])
        end_p = assignment.end_image
        if not Path(end_p).is_file() and edited_key_paths:
            end_p = str(edited_key_paths[-1])
        mids = edited_key_paths[1:-1] if len(edited_key_paths) > 2 else None
        final_frames = interpolate_flf2v_sequence(
            start_p,
            end_p,
            target_frames,
            flf_dir,
            mid_keyframe_paths=mids,
            source_frame_paths=source_frames,
            use_depth=opts.depth_interpolate,
        )
        ops.append("flf2v")
    elif _should_motion_transfer(opts, assignment, source_frames) and edited_key_paths:
        rig_path = _rig_path_from_control(control_plan) if opts.region_motion else ""
        if opts.region_motion and rig_path:
            from .motion import apply_regional_motion

            mt_dir = wd / "region_motion"
            if mt_dir.exists():
                shutil.rmtree(mt_dir)
            final_frames = apply_regional_motion(
                edited_key_paths[0],
                source_frames,
                mt_dir,
                rig_path,
                target_count=target_frames,
                locked_only=False,
            )
            ops.append("region_motion")
        else:
            from .motion import motion_template_sequence

            mt_dir = wd / "motion_transfer"
            if mt_dir.exists():
                shutil.rmtree(mt_dir)
            anchor_kf = edited_key_paths[0]
            final_frames = motion_template_sequence(
                anchor_kf,
                source_frames,
                mt_dir,
                target_count=target_frames,
                blend_source=0.12,
                motion_brush=brush_spec,
            )
            ops.append("motion_transfer")
    else:
        interp_dir = wd / "interpolated"
        if interp_dir.exists():
            shutil.rmtree(interp_dir)
        if opts.depth_interpolate:
            from .interpolate import interpolate_sequence_depth

            final_frames = interpolate_sequence_depth(
                edited_key_paths,
                target_frames,
                interp_dir,
                use_flow=True,
            )
            ops.append("interpolate:depth")
        else:
            from .interpolate import interpolate_sequence

            final_frames = interpolate_sequence(
                edited_key_paths,
                target_frames,
                interp_dir,
                use_flow=True,
            )
            ops.append("interpolate")

    if opts.flow_consistency and len(final_frames) >= 2:
        from .motion import repair_flow_consistency

        repair_flow_consistency(final_frames)
        ops.append("flow_consistency")

    harmonize_frame_sequence(
        final_frames,
        alpha_prev=opts.temporal_alpha,
        smooth_window=opts.temporal_smooth,
    )
    ops.append("temporal_harmonize")

    if opts.deflicker:
        apply_deflicker(final_frames, strength=opts.deflicker_strength)
        ops.append("deflicker")

    if opts.identity_lock:
        anchor_img, mask_path = _collect_lock_anchor(control_plan, plan)
        if anchor_img and Path(anchor_img).is_file():
            if opts.propagate_masks:
                from .quality import apply_propagated_identity_lock

                apply_propagated_identity_lock(
                    final_frames,
                    anchor_img,
                    mask_path=mask_path or None,
                    strength=opts.identity_lock_strength,
                    work_dir=wd,
                )
                ops.append("identity_lock:propagated")
            else:
                from .quality import apply_identity_lock_to_sequence

                apply_identity_lock_to_sequence(
                    final_frames,
                    anchor_img,
                    mask_path=mask_path or None,
                    strength=opts.identity_lock_strength,
                )
                ops.append("identity_lock")

    if opts.semantic_drift_repair and len(final_frames) >= 2:
        from .quality import repair_sequence_drift

        anchor_img, _ = _collect_lock_anchor(control_plan, plan)
        report = repair_sequence_drift(
            final_frames,
            anchor_path=anchor_img or (str(edited_key_paths[0]) if edited_key_paths else None),
            threshold=opts.drift_threshold,
            blend_strength=opts.drift_blend_strength,
        )
        if report.repaired_count:
            ops.append(f"drift_repair:{report.repaired_count}")

    if opts.velocity_ease and len(final_frames) >= 3:
        apply_velocity_ease(final_frames, ease=opts.velocity_ease_mode)
        ops.append(f"velocity_ease:{opts.velocity_ease_mode}")

    if opts.frame_enhance:
        enhance_sequence(final_frames, amount=opts.frame_enhance_amount)
        ops.append("frame_enhance")

    if opts.post_grade:
        apply_grade_to_sequence(final_frames, opts.post_grade)
        ops.append(f"grade:{opts.post_grade}")

    q = score_segment_quality(
        final_frames,
        segment_index=seg_i,
        min_permanence=opts.min_permanence,
        min_artifact=opts.min_artifact,
        min_identity=opts.min_identity,
        min_extremity=opts.min_extremity,
        min_physics=opts.min_physics,
        min_glyph=opts.min_glyph,
    )

    if provenance is not None:
        provenance.add(record_segment_provenance(assignment, operations=ops))

    return final_frames, q


def probe_video_safe(path: str | Path):
    from .video_io import probe_video

    return probe_video(path)


def render_segment_video(frame_paths: list[Path], out_path: Path, *, fps: float) -> Path:
    return write_video_from_frames(frame_paths, out_path, fps=fps)


# ==================================================================================================
# pose_control: Pose / skeleton control image generation for sample.py ControlNet.
# ==================================================================================================


def pose_from_rig_boxes(
    width: int,
    height: int,
    parts: Sequence[tuple[str, tuple[float, float, float, float]]],
) -> np.ndarray:
    """Draw stick-figure style pose map from normalized rig boxes."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    centers: dict[str, tuple[int, int]] = {}
    for name, box in parts:
        x0, y0, x1, y1 = box
        cx = int((x0 + x1) * 0.5 * width)
        cy = int((y0 + y1) * 0.5 * height)
        centers[name] = (cx, cy)
        ix0, iy0 = int(x0 * width), int(y0 * height)
        ix1, iy1 = int(x1 * width), int(y1 * height)
        canvas[iy0:iy1, ix0:ix1] = (40, 40, 40)
    try:
        import cv2

        joints = [
            ("head", "torso"),
            ("torso", "left_arm"),
            ("torso", "right_arm"),
            ("torso", "legs"),
        ]
        for a, b in joints:
            if a in centers and b in centers:
                cv2.line(canvas, centers[a], centers[b], (255, 255, 255), 2, cv2.LINE_AA)
        for c in centers.values():
            cv2.circle(canvas, c, 4, (255, 255, 255), -1, cv2.LINE_AA)
    except ImportError:
        pass
    return canvas


def write_pose_control_image(
    parts: Sequence[tuple[str, tuple[float, float, float, float]]],
    out_path: str | Path,
    *,
    width: int = 512,
    height: int = 512,
) -> Path:
    from .video_io import save_frame_rgb

    rgb = pose_from_rig_boxes(width, height, parts)
    op = Path(out_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    save_frame_rgb(op, rgb)
    return op


def pose_control_args(control_image_path: str | Path) -> list[str]:
    p = str(control_image_path)
    return ["--control-image", p, "--control-type", "pose", "--control-scale", "0.65"]


# ==================================================================================================
# videomax: Video quality presets for the scene ``edit`` block (``video_quality``: lite | standard | strong | max).
#
# ``max`` (or ``videomax: true``) is ``strong`` plus the prompt's motion grammar (``motion_grammar`` forces one),
# stricter segment gates and more retries. A ``motion_grammar`` other than ``auto`` also applies on its own.
# ==================================================================================================

_VIDEOMAX_OVERRIDES: dict[str, Any] = {
    "identity_lock": True,
    "flow_consistency": True,
    "deflicker": True,
    "semantic_drift_repair": True,
    "camera_stabilize": True,
    "quality_retry": True,
    "max_retries": 3,
    "min_extremity": 0.38,
}


@dataclass(frozen=True)
class VideoMaxPlan:
    grammar: MotionGrammar
    overrides: dict[str, Any]


def plan_videomax(prompt: str = "", *, grammar: str = "auto") -> VideoMaxPlan:
    require_allowed(prompt)
    g = grammar_for_prompt(prompt) if grammar in ("", "auto") else grammar_for_engine(grammar)
    return VideoMaxPlan(grammar=g, overrides=dict(_VIDEOMAX_OVERRIDES))


def _lite(opts: ProcessOptions) -> ProcessOptions:
    return replace(opts, quality_retry=True, max_retries=max(opts.max_retries, 1))


def _standard(opts: ProcessOptions) -> ProcessOptions:
    s = opts.sample
    sample = replace(
        s,
        num=max(s.num, 2),
        pick_best=s.pick_best if s.pick_best != "none" else "combo_vit",
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
        max_retries=max(opts.max_retries, 2),
    )


def _strong(opts: ProcessOptions) -> ProcessOptions:
    s = opts.sample
    sample = replace(
        s,
        num=max(s.num, 3),
        pick_best=s.pick_best if s.pick_best != "none" else "combo_vit",
        quality_pack=s.quality_pack if s.quality_pack != "none" else "top",
        adherence_pack=s.adherence_pack if s.adherence_pack != "none" else "standard",
        human_made=s.human_made if s.human_made != "none" else "standard",
        dual_stage_layout=True,
        holy_grail=True,
        anti_perspective_drift=True,
        boost_quality=True,
        less_ai=True,
        naturalize=True,
    )
    return replace(
        opts,
        sample=sample,
        motion_beat_keyframes=True,
        depth_interpolate=True,
        flow_consistency=True,
        deflicker=True,
        deflicker_strength=max(opts.deflicker_strength, 0.82),
        identity_lock=True,
        semantic_drift_repair=True,
        quality_retry=True,
        max_retries=max(opts.max_retries, 3),
        temporal_alpha=max(opts.temporal_alpha, 0.12),
        temporal_smooth=max(opts.temporal_smooth, 3),
        min_permanence=max(opts.min_permanence, 0.55),
        min_artifact=max(opts.min_artifact, 0.50),
    )


_PRESETS = {"lite": _lite, "standard": _standard, "strong": _strong}


def apply_video_quality(opts: ProcessOptions, *, prompt: str = "") -> ProcessOptions:
    """Resolve ``video_quality`` / ``videomax`` / ``motion_grammar`` into concrete options."""
    if opts.videomax:
        plan = plan_videomax(prompt, grammar=opts.motion_grammar)
        opts = replace(_strong(opts), videomax=True, **plan.overrides)
        return apply_motion_grammar(opts, plan.grammar)
    preset = opts.video_quality
    if preset not in ("", "none", "off"):
        if preset not in _PRESETS:
            raise ValueError(f"unknown video_quality {preset!r}; expected one of none, lite, standard, strong, max")
        opts = _PRESETS[preset](opts)
    if opts.motion_grammar not in ("", "auto"):
        opts = apply_motion_grammar(opts, grammar_for_engine(opts.motion_grammar))
    return opts


# ==================================================================================================
# segment_retry: Retry failed segments with adjusted parameters.
# ==================================================================================================


@dataclass(slots=True)
class RetryPolicy:
    max_attempts: int = 2
    strength_decay: float = 0.08
    temporal_boost: float = 0.04


def process_segment_with_retry(
    assignment: SegmentAssignment,
    plan: VideoPlan,
    work_dir: str | Path,
    *,
    ckpt: str = "",
    dry_run: bool = False,
    sample_extra_args: Sequence[str] | None = None,
    control_plan: Any = None,
    policy: RetryPolicy | None = None,
    provenance: Any = None,
    process_options: Any | None = None,
) -> tuple[list[Path], Any]:
    pol = policy or RetryPolicy()
    best_frames: list[Path] = []
    best_q: Any = None
    wd = Path(work_dir)

    for attempt in range(max(1, pol.max_attempts)):
        attempt_dir = wd / f"attempt_{attempt:02d}"
        asg = assignment
        if attempt > 0:
            asg.edit_strength = max(0.35, float(asg.edit_strength) - pol.strength_decay * attempt)
        opts_adj = process_options
        if attempt > 0 and process_options is not None:
            opts_adj = replace(
                process_options,
                temporal_alpha=min(0.28, process_options.temporal_alpha + pol.temporal_boost * attempt),
                temporal_smooth=min(5, process_options.temporal_smooth + 1),
            )
        frames, q = process_segment(
            asg,
            plan,
            attempt_dir,
            ckpt=ckpt,
            dry_run=dry_run,
            provenance=provenance,
            sample_extra_args=sample_extra_args,
            control_plan=control_plan,
            process_options=opts_adj,
        )
        if q.passed:
            return frames, q
        if best_q is None or q.temporal_score > best_q.temporal_score:
            best_frames, best_q = frames, q
    return best_frames, best_q


__all__ = [
    "SampleOptions",
    "build_keyframe_sample_extras",
    "merge_argv",
    "parse_sample_options",
    "ProcessOptions",
    "parse_process_options",
    "ProvenanceLog",
    "record_segment_provenance",
    "write_provenance_json",
    "build_sample_cmd_for_keyframe",
    "run_keyframe_edit",
    "run_keyframe_batch",
    "GRADES",
    "apply_deflicker",
    "apply_grade_to_sequence",
    "apply_velocity_ease",
    "ease_indices",
    "enhance_frame",
    "enhance_sequence",
    "grade_frame",
    "harmonize_frame_sequence",
    "process_segment",
    "render_segment_video",
    "synthetic_segment_from_color",
    "pose_from_rig_boxes",
    "write_pose_control_image",
    "pose_control_args",
    "VideoMaxPlan",
    "apply_video_quality",
    "plan_videomax",
    "RetryPolicy",
    "process_segment_with_retry",
]

"""Process one segment: extract → keyframe edit → motion transfer → interpolate → temporal → quality."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, List, Optional, Sequence

from .editor import run_keyframe_batch
from .keyframes import schedule_keyframes_for_segment
from .process_options import ProcessOptions, parse_process_options
from .provenance import ProvenanceLog, record_segment_provenance
from .quality import score_segment_quality
from .temporal import harmonize_frame_sequence
from .timeline import frame_count_for_duration, normalize_segment_window
from .types import KeyframeEditJob, SegmentAssignment, VideoPlan
from .video_io import extract_frames, write_video_from_frames

__all__ = ["process_segment", "render_segment_video", "synthetic_segment_from_color"]


def synthetic_segment_from_color(
    work_dir: Path,
    *,
    width: int,
    height: int,
    frame_count: int,
    rgb: tuple[int, int, int] = (32, 32, 48),
) -> List[Path]:
    import numpy as np

    from .video_io import save_frame_rgb

    frames_dir = work_dir / "synthetic_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:] = rgb
    paths: List[Path] = []
    for i in range(frame_count):
        fp = frames_dir / f"frame_{i + 1:06d}.png"
        save_frame_rgb(fp, arr)
        paths.append(fp)
    return paths


def _resolve_options(plan: VideoPlan, process_options: Optional[ProcessOptions]) -> ProcessOptions:
    if process_options is not None:
        return process_options
    edit = plan.metadata.get("edit") if plan.metadata else None
    return parse_process_options(edit if isinstance(edit, dict) else {})


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


def _should_motion_transfer(opts: ProcessOptions, assignment: SegmentAssignment, source_frames: List[Path]) -> bool:
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
    edit_strength: Optional[float] = None,
    provenance: Optional[ProvenanceLog] = None,
    sample_extra_args: Optional[Sequence[str]] = None,
    control_plan: Any = None,
    process_options: Optional[ProcessOptions] = None,
) -> tuple[List[Path], Any]:
    """Returns (final_frame_paths, quality_report)."""
    opts = _resolve_options(plan, process_options)
    wd = Path(work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    seg_i = assignment.shot.index
    tl = plan.timeline
    target_frames = frame_count_for_duration(assignment.shot.duration_sec, tl.fps)

    ops: List[str] = []
    clip = assignment.clip
    source_frames: List[Path] = []

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
            from .camera_stabilize import stabilize_frame_paths

            stabilize_frame_paths(source_frames, strength=opts.camera_stabilize_strength)
            ops.append("camera_stabilize")
    elif clip and clip.path.startswith("http"):
        raise FileNotFoundError(f"remote clip not downloaded: {clip.path}")
    else:
        source_frames = synthetic_segment_from_color(wd, width=tl.width, height=tl.height, frame_count=target_frames)
        ops.append("synthetic_fallback")

    if not source_frames:
        source_frames = synthetic_segment_from_color(wd, width=tl.width, height=tl.height, frame_count=target_frames)

    beat_indices: Optional[List[int]] = None
    if opts.motion_beat_keyframes and len(source_frames) >= 3:
        try:
            from .motion_beats import detect_motion_beats

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
    jobs: List[KeyframeEditJob] = []
    strength = float(edit_strength if edit_strength is not None else assignment.edit_strength or opts.edit_strength)

    seg_extra = list(sample_extra_args or [])
    if opts.pose_control and control_plan is not None:
        rig_path = _rig_path_from_control(control_plan)
        if rig_path and Path(rig_path).is_file():
            try:
                import json

                from .pose_control import pose_control_args, write_pose_control_image

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
        run_keyframe_batch(jobs, ckpt=ckpt, image_size=min(tl.width, tl.height), extra_args=list(seg_extra))
        ops.append("keyframe:sample.py")

    edited_key_paths = sorted(edited_dir.glob("kf_*.png"))
    if not edited_key_paths:
        edited_key_paths = schedule.frame_paths

    brush_spec = None
    if assignment.motion_brush:
        from .motion_brush import parse_motion_brush

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
        from .flf2v import interpolate_flf2v_sequence

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
            from .region_motion import apply_regional_motion

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
            from .motion_transfer import motion_template_sequence

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
            from .depth_interpolate import interpolate_sequence_depth

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
        from .flow_consistency import repair_flow_consistency

        repair_flow_consistency(final_frames)
        ops.append("flow_consistency")

    harmonize_frame_sequence(
        final_frames,
        alpha_prev=opts.temporal_alpha,
        smooth_window=opts.temporal_smooth,
    )
    ops.append("temporal_harmonize")

    if opts.deflicker:
        from .deflicker import apply_deflicker

        apply_deflicker(final_frames, strength=opts.deflicker_strength)
        ops.append("deflicker")

    if opts.identity_lock:
        anchor_img, mask_path = _collect_lock_anchor(control_plan, plan)
        if anchor_img and Path(anchor_img).is_file():
            if opts.propagate_masks:
                from .identity_lock import apply_propagated_identity_lock

                apply_propagated_identity_lock(
                    final_frames,
                    anchor_img,
                    mask_path=mask_path or None,
                    strength=opts.identity_lock_strength,
                    work_dir=wd,
                )
                ops.append("identity_lock:propagated")
            else:
                from .identity_lock import apply_identity_lock_to_sequence

                apply_identity_lock_to_sequence(
                    final_frames,
                    anchor_img,
                    mask_path=mask_path or None,
                    strength=opts.identity_lock_strength,
                )
                ops.append("identity_lock")

    if opts.semantic_drift_repair and len(final_frames) >= 2:
        from .drift_repair import repair_sequence_drift

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
        from .velocity_curve import apply_velocity_ease

        apply_velocity_ease(final_frames, ease=opts.velocity_ease_mode)
        ops.append(f"velocity_ease:{opts.velocity_ease_mode}")

    if opts.frame_enhance:
        from .frame_enhance import enhance_sequence

        enhance_sequence(final_frames, amount=opts.frame_enhance_amount)
        ops.append("frame_enhance")

    if opts.post_grade:
        from .post_grade import apply_grade_to_sequence

        apply_grade_to_sequence(final_frames, opts.post_grade)
        ops.append(f"grade:{opts.post_grade}")

    q = score_segment_quality(final_frames, segment_index=seg_i)

    if provenance is not None:
        provenance.add(record_segment_provenance(assignment, operations=ops))

    return final_frames, q


def probe_video_safe(path: str | Path):
    from .video_io import probe_video

    return probe_video(path)


def render_segment_video(frame_paths: List[Path], out_path: Path, *, fps: float) -> Path:
    return write_video_from_frames(frame_paths, out_path, fps=fps)

"""Main retrieve → transform → compose video pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Optional, Sequence

from .i2v import build_i2v_plan, prepare_anchor_frame
from .process_options import ProcessOptions, parse_process_options
from .provenance import ProvenanceLog, write_provenance_json
from .segment_processor import process_segment, render_segment_video
from .segment_retry import RetryPolicy, process_segment_with_retry
from .stitch import stitch_frame_lists
from .t2v import assign_clips_to_plan, build_t2v_plan, gather_retrieval_candidates
from .types import SegmentAssignment, VideoMode, VideoPipelineResult, VideoPlan
from .video_io import save_frame_rgb

__all__ = [
    "run_from_scene_file",
    "run_i2v_pipeline",
    "run_t2v_pipeline",
    "run_video_pipeline",
    "save_plan_json",
]


def save_plan_json(plan: VideoPlan, path: str | Path) -> Path:
    p = Path(path)

    def _ser(obj: Any) -> Any:
        if hasattr(obj, "value"):
            return obj.value
        return obj

    payload = {
        "mode": plan.mode.value,
        "user_prompt": plan.user_prompt,
        "timeline": asdict(plan.timeline),
        "shots": [asdict(s) for s in plan.shots],
        "global_negative": plan.global_negative,
        "style_notes": plan.style_notes,
        "metadata": plan.metadata,
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def run_from_scene_file(
    scene_path: str | Path,
    out_path: str | Path,
    *,
    ckpt: str = "",
    work_dir: str | Path = "runs/video",
    dry_run: bool = False,
    use_pexels: bool = False,
    allow_download: bool = False,
    sample_extra_args: Optional[Sequence[str]] = None,
) -> VideoPipelineResult:
    """Run pipeline from a single scene graph JSON (recommended entry point)."""
    from .retrieval import build_clip_candidate_from_path, load_local_clip_library
    from .scene_graph import compile_scene_graph, load_scene_graph
    from .t2v import assign_clips_to_plan, gather_retrieval_candidates

    graph = load_scene_graph(scene_path)
    graph.raw["_work_dir"] = str(Path(work_dir).resolve())
    compiled = compile_scene_graph(graph)
    graph = compiled.graph
    plan = compiled.plan
    retr = graph.retrieval
    local_library = str(retr.get("local_library") or "")
    catalog_path = str(retr.get("catalog") or "data/video_catalog.json")
    keyframe_interval = int(graph.edit.get("keyframe_interval") or 6)

    download_dir = Path(work_dir) / "downloads" if allow_download else None
    candidates = gather_retrieval_candidates(
        plan.user_prompt,
        local_library=local_library,
        catalog_path=catalog_path,
        use_pexels=use_pexels and allow_download,
        download_dir=download_dir,
    )
    upload_dir = Path(work_dir) / "clips"
    if upload_dir.is_dir():
        candidates.extend(load_local_clip_library(upload_dir))

    assignments = assign_clips_to_plan(plan, candidates, keyframe_interval=keyframe_interval)

    # Apply per-shot overrides from scene graph
    for i, ov in enumerate(compiled.segment_overrides):
        if i >= len(assignments):
            break
        ref = str(ov.get("reference_clip") or "").strip()
        if ref and Path(ref).is_file():
            assignments[i].clip = build_clip_candidate_from_path(ref)
        if ov.get("keyframe_interval"):
            assignments[i].keyframe_interval = int(ov["keyframe_interval"])
        if ov.get("edit_strength"):
            assignments[i].edit_strength = float(ov["edit_strength"])
        if ov.get("transition"):
            assignments[i].transition = ov["transition"]
        if ov.get("start_image"):
            assignments[i].start_image = str(ov["start_image"])
        if ov.get("end_image"):
            assignments[i].end_image = str(ov["end_image"])
        if ov.get("flf2v"):
            assignments[i].flf2v = bool(ov["flf2v"])
        if ov.get("motion_brush"):
            assignments[i].motion_brush = dict(ov["motion_brush"])

    control_plans = list(compiled.control_plans or [])
    anchor = str(graph.anchor_image or "").strip() or None
    if not anchor and graph.inputs:
        for inp in graph.inputs:
            img = str(getattr(inp, "image", "") or "")
            if img:
                anchor = img
                break
    if graph.mode == VideoMode.I2V and anchor:
        from .i2v import prepare_anchor_frame

        prepare_anchor_frame(anchor, Path(work_dir) / "anchor", width=plan.timeline.width, height=plan.timeline.height)

    return _run_assignments(
        plan,
        assignments,
        out_path,
        ckpt=ckpt,
        work_dir=work_dir,
        dry_run=dry_run,
        sample_extra_args=sample_extra_args,
        anchor_frame=Path(anchor) if anchor else None,
        control_plans=control_plans,
        process_options=parse_process_options(graph.edit),
    )


def run_t2v_pipeline(
    prompt: str,
    out_path: str | Path,
    *,
    ckpt: str = "",
    work_dir: str | Path = "runs/video",
    duration_sec: float = 6.0,
    fps: float = 24.0,
    width: int = 1280,
    height: int = 720,
    local_library: str = "",
    catalog_path: str = "",
    use_pexels: bool = False,
    allow_download: bool = False,
    dry_run: bool = False,
    keyframe_interval: int = 6,
    sample_extra_args: Optional[Sequence[str]] = None,
) -> VideoPipelineResult:
    plan = build_t2v_plan(prompt, duration_sec=duration_sec, fps=fps, width=width, height=height)
    download_dir = Path(work_dir) / "downloads" if allow_download else None
    candidates = gather_retrieval_candidates(
        prompt,
        local_library=local_library,
        catalog_path=catalog_path,
        use_pexels=use_pexels and allow_download,
        download_dir=download_dir,
    )
    # User-uploaded clips in work_dir/clips
    upload_dir = Path(work_dir) / "clips"
    if upload_dir.is_dir():
        from .retrieval import load_local_clip_library

        candidates.extend(load_local_clip_library(upload_dir))
    assignments = assign_clips_to_plan(plan, candidates, keyframe_interval=keyframe_interval)
    return _run_assignments(
        plan,
        assignments,
        out_path,
        ckpt=ckpt,
        work_dir=work_dir,
        dry_run=dry_run,
        sample_extra_args=sample_extra_args,
    )


def run_i2v_pipeline(
    prompt: str,
    anchor_image: str | Path,
    out_path: str | Path,
    *,
    ckpt: str = "",
    motion_clip: str = "",
    work_dir: str | Path = "runs/video_i2v",
    duration_sec: float = 4.0,
    fps: float = 24.0,
    width: int = 1280,
    height: int = 720,
    dry_run: bool = False,
    keyframe_interval: int = 8,
    sample_extra_args: Optional[Sequence[str]] = None,
) -> VideoPipelineResult:
    ref_clips = [motion_clip] if motion_clip else None
    plan = build_i2v_plan(
        prompt,
        anchor_image,
        duration_sec=duration_sec,
        fps=fps,
        width=width,
        height=height,
        reference_clips=ref_clips,
    )
    wd = Path(work_dir)
    anchor = prepare_anchor_frame(anchor_image, wd / "anchor", width=width, height=height)
    candidates = []
    if motion_clip and Path(motion_clip).is_file():
        from .retrieval import build_clip_candidate_from_path

        candidates.append(build_clip_candidate_from_path(motion_clip))
    assignments = assign_clips_to_plan(plan, candidates, keyframe_interval=keyframe_interval, use_motion_only=True)
    # Force first keyframe to anchor for segment 0
    if assignments:
        seg0_dir = wd / "seg_00" / "anchor_override"
        seg0_dir.mkdir(parents=True, exist_ok=True)
        plan.metadata["anchor_frame"] = str(anchor)
    return _run_assignments(
        plan,
        assignments,
        out_path,
        ckpt=ckpt,
        work_dir=work_dir,
        dry_run=dry_run,
        sample_extra_args=sample_extra_args,
        anchor_frame=anchor,
    )


def run_video_pipeline(
    *,
    mode: str,
    prompt: str,
    out_path: str | Path,
    anchor_image: str = "",
    **kwargs: Any,
) -> VideoPipelineResult:
    m = VideoMode(str(mode).lower())
    if m == VideoMode.I2V:
        if not anchor_image:
            raise ValueError("i2v requires anchor_image")
        return run_i2v_pipeline(prompt, anchor_image, out_path, **kwargs)
    return run_t2v_pipeline(prompt, out_path, **kwargs)


def _process_one_segment(
    assignment: SegmentAssignment,
    plan: VideoPlan,
    seg_dir: Path,
    *,
    ckpt: str,
    dry_run: bool,
    prov: ProvenanceLog,
    seg_extra: List[str],
    cp: Any,
    opts: ProcessOptions,
    retry_policy: RetryPolicy,
    anchor_frame: Optional[Path],
) -> tuple[int, List[Path], Any]:
    def _run_once():
        if opts.quality_retry and ckpt and not dry_run:
            return process_segment_with_retry(
                assignment,
                plan,
                seg_dir,
                ckpt=ckpt,
                dry_run=dry_run,
                provenance=prov,
                sample_extra_args=seg_extra,
                control_plan=cp,
                policy=retry_policy,
                process_options=opts,
            )
        return process_segment(
            assignment,
            plan,
            seg_dir,
            ckpt=ckpt,
            dry_run=dry_run,
            provenance=prov,
            sample_extra_args=seg_extra,
            control_plan=cp,
            process_options=opts,
        )

    frames, q = _run_once()
    if anchor_frame is not None and assignment.shot.index == 0 and frames:
        from .video_io import read_frame_rgb

        save_frame_rgb(frames[0], read_frame_rgb(anchor_frame))
    return assignment.shot.index, frames, q


def _run_assignments(
    plan: VideoPlan,
    assignments: List[SegmentAssignment],
    out_path: str | Path,
    *,
    ckpt: str,
    work_dir: str | Path,
    dry_run: bool,
    sample_extra_args: Optional[Sequence[str]],
    anchor_frame: Optional[Path] = None,
    control_plans: Optional[Sequence[Any]] = None,
    process_options: Optional[ProcessOptions] = None,
) -> VideoPipelineResult:
    wd = Path(work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    opts = process_options or parse_process_options(plan.metadata.get("edit") if plan.metadata else {})
    retry_policy = RetryPolicy(max_attempts=max(1, opts.max_retries))
    prov = ProvenanceLog()
    seg_frame_lists: List[List[Path]] = []
    seg_videos: List[str] = []
    quality_reports = []
    source_clips: List[str] = []
    shot_durations: List[float] = []

    # Build work items
    work: List[tuple[SegmentAssignment, Path, List[str], Any]] = []
    for assignment in assignments:
        seg_dir = wd / f"seg_{assignment.shot.index:02d}"
        seg_extra = list(sample_extra_args or [])
        cp = None
        if control_plans and assignment.shot.index < len(control_plans):
            cp = control_plans[assignment.shot.index]
            from .controls import build_sample_args_for_plan

            seg_extra.extend(build_sample_args_for_plan(cp))
            if cp.positive_prompt:
                assignment.shot.prompt = cp.positive_prompt
            if cp.negative_prompt:
                assignment.shot.negative = cp.negative_prompt
        if cp and cp.init_strength:
            assignment.edit_strength = cp.init_strength
        work.append((assignment, seg_dir, seg_extra, cp))

    seg_results: dict[int, tuple[List[Path], Any]] = {}

    if opts.parallel_segments and len(work) > 1:
        from .parallel_segments import SegmentWorkItem, run_segments_parallel

        items = [SegmentWorkItem(i, work[i][1]) for i in range(len(work))]

        def _worker(item: SegmentWorkItem):
            asg, seg_dir, seg_extra, cp = work[item.index]
            return _process_one_segment(
                asg,
                plan,
                seg_dir,
                ckpt=ckpt,
                dry_run=dry_run,
                prov=prov,
                seg_extra=seg_extra,
                cp=cp,
                opts=opts,
                retry_policy=retry_policy,
                anchor_frame=anchor_frame,
            )[1:]

        parallel_out = run_segments_parallel(items, _worker, max_workers=opts.max_segment_workers)
        for pr in parallel_out:
            asg = work[pr.index][0]
            seg_results[asg.shot.index] = (pr.frames, pr.quality)
    else:
        for asg, seg_dir, seg_extra, cp in work:
            idx, frames, q = _process_one_segment(
                asg,
                plan,
                seg_dir,
                ckpt=ckpt,
                dry_run=dry_run,
                prov=prov,
                seg_extra=seg_extra,
                cp=cp,
                opts=opts,
                retry_policy=retry_policy,
                anchor_frame=anchor_frame,
            )
            seg_results[idx] = (frames, q)

    for assignment in assignments:
        frames, q = seg_results[assignment.shot.index]
        quality_reports.append(q)
        seg_dir = wd / f"seg_{assignment.shot.index:02d}"
        seg_frame_lists.append(frames)
        seg_mp4 = seg_dir / "segment.mp4"
        render_segment_video(frames, seg_mp4, fps=plan.timeline.fps)
        seg_videos.append(str(seg_mp4))
        source_clips.append(str(assignment.clip.path) if assignment.clip else "")
        shot_durations.append(float(assignment.shot.duration_sec))

    final = stitch_frame_lists(
        seg_frame_lists,
        out_path,
        fps=plan.timeline.fps,
        transitions=[a.transition for a in assignments],
    )

    if opts.audio_from_source and not dry_run:
        from .audio_mux import collect_segment_audio, mux_audio_onto_video

        audio = collect_segment_audio(source_clips, wd / "audio", durations=shot_durations)
        if audio:
            mux_audio_onto_video(final, audio)
    prov_path = wd / "provenance.json"
    write_provenance_json(prov, prov_path)
    save_plan_json(plan, wd / "plan.json")

    return VideoPipelineResult(
        output_path=str(final),
        plan=plan,
        provenance=list(prov.records),
        quality=quality_reports,
        segment_paths=seg_videos,
        metadata={"work_dir": str(wd), "provenance": str(prov_path), "dry_run": dry_run},
    )

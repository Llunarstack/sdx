"""Retrieve -> transform -> compose video pipeline: clip retrieval, plan building, segment orchestration."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

from pipelines.video.motion import estimate_motion_score
from pipelines.video.scene import (
    ClipCandidate,
    RetrievalSource,
    SegmentAssignment,
    ShotSpec,
    VideoMode,
    VideoPipelineResult,
    VideoPlan,
    compile_scene_graph,
    load_scene_graph,
    plan_video_from_prompt,
)
from pipelines.video.segment_processor import (
    ProcessOptions,
    ProvenanceLog,
    RetryPolicy,
    apply_video_quality,
    parse_process_options,
    process_segment,
    process_segment_with_retry,
    render_segment_video,
    write_provenance_json,
)
from pipelines.video.video_io import (
    SegmentWorkItem,
    collect_segment_audio,
    extract_frames,
    mux_audio_onto_video,
    probe_video,
    read_frame_rgb,
    run_segments_parallel,
    save_frame_rgb,
    stitch_frame_lists,
)


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2]


def build_clip_candidate_from_path(
    path: str | Path,
    *,
    source: RetrievalSource = RetrievalSource.LOCAL,
    title: str = "",
    tags: list[str] | None = None,
    license: str = "user",
    url: str = "",
) -> ClipCandidate:
    p = Path(path)
    info = probe_video(p) if p.suffix.lower() in (".mp4", ".mov", ".webm", ".mkv", ".avi") else {}
    return ClipCandidate(
        source=source,
        path=str(p.resolve()),
        title=title or p.stem,
        tags=list(tags or []),
        duration_sec=float(info.get("duration_sec") or 0.0),
        fps=float(info.get("fps") or 24.0),
        width=int(info.get("width") or 0),
        height=int(info.get("height") or 0),
        license=license,
        url=url,
    )


def load_local_clip_library(
    root: str | Path,
    *,
    extensions: Sequence[str] = (".mp4", ".mov", ".webm", ".mkv"),
) -> list[ClipCandidate]:
    root = Path(root)
    if not root.is_dir():
        return []
    out: list[ClipCandidate] = []
    for ext in extensions:
        for p in root.rglob(f"*{ext}"):
            sidecar = p.with_suffix(p.suffix + ".json")
            meta: dict[str, Any] = {}
            if sidecar.is_file():
                try:
                    meta = json.loads(sidecar.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            tags = meta.get("tags") or []
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            out.append(
                build_clip_candidate_from_path(
                    p,
                    source=RetrievalSource.LOCAL,
                    title=str(meta.get("title") or p.stem),
                    tags=list(tags),
                    license=str(meta.get("license") or "local"),
                    url=str(meta.get("url") or ""),
                )
            )
    return out


def rank_clips_for_shot(
    shot: ShotSpec,
    candidates: Sequence[ClipCandidate],
    *,
    motion_scores: dict[str, float] | None = None,
) -> list[ClipCandidate]:
    """Score clips by tag/token overlap + duration fit + optional motion score."""
    motion_scores = motion_scores or {}
    q = set(_tokenize(shot.prompt)) | set(_tokenize(shot.motion_hint)) | set(_tokenize(shot.shot_type))
    ranked: list[ClipCandidate] = []
    for c in candidates:
        hay = set(_tokenize(c.title)) | set(_tokenize(" ".join(c.tags)))
        overlap = len(q & hay) / max(1, len(q))
        dur_fit = 1.0
        if c.duration_sec > 0 and shot.duration_sec > 0:
            ratio = min(c.duration_sec, shot.duration_sec) / max(c.duration_sec, shot.duration_sec)
            dur_fit = ratio
        motion = float(motion_scores.get(c.path, 0.5))
        want_motion = 0.6 if "static" not in shot.motion_hint.lower() else 0.3
        motion_fit = 1.0 - abs(motion - want_motion)
        score = 0.55 * overlap + 0.25 * dur_fit + 0.20 * motion_fit
        ranked.append(
            ClipCandidate(
                source=c.source,
                path=c.path,
                title=c.title,
                tags=list(c.tags),
                duration_sec=c.duration_sec,
                fps=c.fps,
                width=c.width,
                height=c.height,
                license=c.license,
                url=c.url,
                score=float(score),
                motion_score=motion,
            )
        )
    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked


def search_pexels_videos(query: str, *, max_results: int = 5) -> list[ClipCandidate]:
    """
    Search Pexels video API when ``PEXELS_API_KEY`` is set.

    Returns metadata + download URL; does not download unless caller uses ``download_pexels_clip``.
    """
    key = (os.environ.get("PEXELS_API_KEY") or "").strip()
    if not key:
        return []
    try:
        import urllib.parse
        import urllib.request

        url = "https://api.pexels.com/videos/search?" + urllib.parse.urlencode(
            {"query": query, "per_page": str(max(1, max_results))}
        )
        req = urllib.request.Request(url, headers={"Authorization": key})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []
    out: list[ClipCandidate] = []
    for vid in data.get("videos") or []:
        files = vid.get("video_files") or []
        best = None
        best_area = 0
        for f in files:
            w, h = int(f.get("width") or 0), int(f.get("height") or 0)
            if w * h > best_area and f.get("link"):
                best_area = w * h
                best = f
        if not best:
            continue
        out.append(
            ClipCandidate(
                source=RetrievalSource.PEXELS,
                path=str(best.get("link") or ""),
                title=str(vid.get("url") or query)[:120],
                tags=_tokenize(query),
                duration_sec=float(vid.get("duration") or 0.0),
                fps=24.0,
                width=int(best.get("width") or 0),
                height=int(best.get("height") or 0),
                license="pexels",
                url=str(vid.get("url") or best.get("link") or ""),
                score=0.5,
            )
        )
    return out


def download_remote_clip(url: str, dest: Path) -> Path:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def search_web_catalog(
    query: str,
    *,
    catalog_path: str | Path = "",
    max_results: int = 5,
) -> list[ClipCandidate]:
    """
    Search a local JSON catalog of licensed web clips (metadata only).

    Catalog format: ``{"clips": [{"title", "tags", "path" or "url", "license"}]}``
    """
    cp = Path(catalog_path) if catalog_path else Path("data/video_catalog.json")
    if not cp.is_file():
        return []
    try:
        data = json.loads(cp.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = data.get("clips") or data.get("videos") or []
    q = set(_tokenize(query))
    scored: list[ClipCandidate] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or "")
        tags = row.get("tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]
        hay = set(_tokenize(title)) | set(_tokenize(" ".join(str(t) for t in tags)))
        overlap = len(q & hay) / max(1, len(q))
        if overlap <= 0:
            continue
        path = str(row.get("path") or row.get("url") or "")
        scored.append(
            ClipCandidate(
                source=RetrievalSource.WEB_CATALOG,
                path=path,
                title=title,
                tags=[str(t) for t in tags],
                duration_sec=float(row.get("duration_sec") or 0.0),
                license=str(row.get("license") or "catalog"),
                url=str(row.get("url") or path),
                score=float(overlap),
            )
        )
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:max_results]


# ==================================================================================================
# i2v: Image-to-video: anchor frame 0, borrow motion from reference clip.
# ==================================================================================================


def prepare_anchor_frame(
    image_path: str | Path,
    out_dir: str | Path,
    *,
    width: int,
    height: int,
) -> Path:
    """Resize/center-crop user image to timeline resolution."""
    from PIL import Image

    src = Path(image_path)
    out = Path(out_dir) / "anchor_frame.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src).convert("RGB")
    w, h = img.size
    target_ar = width / max(1, height)
    src_ar = w / max(1, h)
    if src_ar > target_ar:
        new_w = int(h * target_ar)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ar)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    save_frame_rgb(out, __import__("numpy").array(img))
    return out


def build_i2v_plan(
    prompt: str,
    anchor_image: str | Path,
    *,
    duration_sec: float = 4.0,
    fps: float = 24.0,
    width: int = 1280,
    height: int = 720,
    reference_clips: Sequence[str] | None = None,
) -> VideoPlan:
    plan = plan_video_from_prompt(
        prompt,
        mode=VideoMode.I2V,
        duration_sec=duration_sec,
        fps=fps,
        width=width,
        height=height,
        anchor_image_note="identity locked from anchor image, consistent subject",
    )
    plan.metadata["anchor_image"] = str(Path(anchor_image).resolve())
    if reference_clips:
        plan.metadata["reference_clips"] = [str(Path(p).resolve()) for p in reference_clips]
    return plan


# ==================================================================================================
# t2v: Text-to-video orchestration helpers.
# ==================================================================================================


def build_t2v_plan(
    prompt: str,
    *,
    duration_sec: float = 6.0,
    fps: float = 24.0,
    width: int = 1280,
    height: int = 720,
    max_shots: int = 4,
) -> VideoPlan:
    return plan_video_from_prompt(
        prompt,
        mode=VideoMode.T2V,
        duration_sec=duration_sec,
        fps=fps,
        width=width,
        height=height,
        max_shots=max_shots,
    )


def gather_retrieval_candidates(
    query: str,
    *,
    local_library: str = "",
    catalog_path: str = "",
    use_pexels: bool = False,
    download_dir: Path | None = None,
) -> list[ClipCandidate]:
    candidates: list[ClipCandidate] = []
    if local_library:
        candidates.extend(load_local_clip_library(local_library))
    candidates.extend(search_web_catalog(query, catalog_path=catalog_path))
    if use_pexels:
        for c in search_pexels_videos(query):
            if c.path.startswith("http") and download_dir is not None:
                dest = download_dir / f"pexels_{abs(hash(c.url)) % 10_000_000}.mp4"
                if not dest.is_file():
                    try:
                        download_remote_clip(c.path, dest)
                    except Exception:
                        continue
                c = ClipCandidate(
                    source=c.source,
                    path=str(dest),
                    title=c.title,
                    tags=c.tags,
                    duration_sec=c.duration_sec,
                    fps=c.fps,
                    width=c.width,
                    height=c.height,
                    license=c.license,
                    url=c.url,
                    score=c.score,
                )
                info = probe_video(dest)
                c.duration_sec = float(info.get("duration_sec") or c.duration_sec)
            if Path(c.path).is_file() or c.path.startswith("http"):
                candidates.append(c)
    return candidates


def assign_clips_to_plan(
    plan: VideoPlan,
    candidates: Sequence[ClipCandidate],
    *,
    keyframe_interval: int = 6,
    use_motion_only: bool = True,
) -> list[SegmentAssignment]:
    assignments: list[SegmentAssignment] = []
    used: set[str] = set()
    for shot in plan.shots:
        ranked = rank_clips_for_shot(shot, candidates)
        pick: ClipCandidate | None = None
        for c in ranked:
            if c.path not in used:
                pick = c
                used.add(c.path)
                break
        if pick is None and ranked:
            pick = ranked[0]
        in_sec = 0.0
        out_sec = shot.duration_sec
        if pick and pick.duration_sec > 0:
            out_sec = min(shot.duration_sec, pick.duration_sec)
        assignments.append(
            SegmentAssignment(
                shot=shot,
                clip=pick,
                in_sec=in_sec,
                out_sec=out_sec,
                use_motion_only=use_motion_only,
                keyframe_interval=keyframe_interval,
            )
        )
    return assignments


def preview_retrieval_rankings(
    prompt: str,
    *,
    local_library: str = "",
    catalog_path: str = "",
    top_k: int = 5,
) -> list[dict]:
    """Per-shot clip ranking for ``generate_video --preview-retrieval`` (no download, no sampling)."""
    candidates = gather_retrieval_candidates(prompt, local_library=local_library, catalog_path=catalog_path)
    return [
        {
            "shot_index": shot.index,
            "shot_prompt": shot.prompt,
            "candidates": [
                {"path": c.path, "title": c.title, "score": c.score, "source": c.source.value}
                for c in rank_clips_for_shot(shot, candidates)[:top_k]
            ],
        }
        for shot in build_t2v_plan(prompt).shots
    ]


# ==================================================================================================
# pipeline
# ==================================================================================================


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
    sample_extra_args: Sequence[str] | None = None,
) -> VideoPipelineResult:
    """Run pipeline from a single scene graph JSON (recommended entry point)."""
    graph = load_scene_graph(scene_path)
    graph.raw["_work_dir"] = str(Path(work_dir).resolve())
    compiled = compile_scene_graph(graph)
    graph = compiled.graph
    plan = compiled.plan
    keyframe_interval = int(graph.edit.get("keyframe_interval") or 6)

    if graph.mode == VideoMode.V2V:
        # Restyle the user's plate: every shot reads its own consecutive window of it.
        if not Path(graph.motion_clip).is_file():
            raise FileNotFoundError(f"v2v motion_clip not found: {graph.motion_clip!r}")
        assignments = assign_clips_to_plan(
            plan, [build_clip_candidate_from_path(graph.motion_clip)], keyframe_interval=keyframe_interval
        )
        t = 0.0
        for a in assignments:
            a.in_sec, a.out_sec = t, t + a.shot.duration_sec
            t = a.out_sec
    else:
        candidates = gather_retrieval_candidates(
            plan.user_prompt,
            local_library=str(graph.retrieval.get("local_library") or ""),
            catalog_path=str(graph.retrieval.get("catalog") or "data/video_catalog.json"),
            use_pexels=use_pexels and allow_download,
            download_dir=Path(work_dir) / "downloads" if allow_download else None,
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
        anchor = str(
            prepare_anchor_frame(
                anchor, Path(work_dir) / "anchor", width=plan.timeline.width, height=plan.timeline.height
            )
        )

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
        segment_overrides=compiled.segment_overrides,
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
    sample_extra_args: Sequence[str] | None = None,
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
    sample_extra_args: Sequence[str] | None = None,
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
        candidates.append(build_clip_candidate_from_path(motion_clip))
    assignments = assign_clips_to_plan(plan, candidates, keyframe_interval=keyframe_interval, use_motion_only=True)
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
    seg_extra: list[str],
    cp: Any,
    opts: ProcessOptions,
    retry_policy: RetryPolicy,
    anchor_frame: Path | None,
) -> tuple[int, list[Path], Any]:
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
        save_frame_rgb(frames[0], read_frame_rgb(anchor_frame))
    return assignment.shot.index, frames, q


def _run_assignments(
    plan: VideoPlan,
    assignments: list[SegmentAssignment],
    out_path: str | Path,
    *,
    ckpt: str,
    work_dir: str | Path,
    dry_run: bool,
    sample_extra_args: Sequence[str] | None,
    anchor_frame: Path | None = None,
    control_plans: Sequence[Any] | None = None,
    process_options: ProcessOptions | None = None,
    segment_overrides: Sequence[Mapping[str, Any]] | None = None,
) -> VideoPipelineResult:
    wd = Path(work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    opts = process_options or parse_process_options(plan.metadata.get("edit") if plan.metadata else {})
    opts = apply_video_quality(opts, prompt=plan.user_prompt)
    retry_policy = RetryPolicy(max_attempts=max(1, opts.max_retries))
    prov = ProvenanceLog()
    seg_frame_lists: list[list[Path]] = []
    seg_videos: list[str] = []
    quality_reports = []
    source_clips: list[str] = []
    shot_durations: list[float] = []

    option_names = {f.name for f in fields(ProcessOptions)}
    work: list[tuple[SegmentAssignment, Path, list[str], Any, ProcessOptions]] = []
    for i, assignment in enumerate(assignments):
        # Per-shot scene / frontier dials (post_grade, camera_stabilize, velocity_ease, ...) override the scene edit.
        ov = segment_overrides[i] if segment_overrides and i < len(segment_overrides) else {}
        seg_opts = replace(opts, **{k: v for k, v in ov.items() if k in option_names and v is not None})
        seg_dir = wd / f"seg_{assignment.shot.index:02d}"
        seg_extra = list(sample_extra_args or [])
        cp = None
        if control_plans and assignment.shot.index < len(control_plans):
            cp = control_plans[assignment.shot.index]
            from pipelines.video.controls import build_sample_args_for_plan

            seg_extra.extend(build_sample_args_for_plan(cp))
            if cp.positive_prompt:
                assignment.shot.prompt = cp.positive_prompt
            if cp.negative_prompt:
                assignment.shot.negative = cp.negative_prompt
        if cp and cp.init_strength:
            assignment.edit_strength = cp.init_strength
        work.append((assignment, seg_dir, seg_extra, cp, seg_opts))

    seg_results: dict[int, tuple[list[Path], Any]] = {}

    if opts.parallel_segments and len(work) > 1:
        items = [SegmentWorkItem(i, work[i][1]) for i in range(len(work))]

        def _worker(item: SegmentWorkItem):
            asg, seg_dir, seg_extra, cp, seg_opts = work[item.index]
            return _process_one_segment(
                asg,
                plan,
                seg_dir,
                ckpt=ckpt,
                dry_run=dry_run,
                prov=prov,
                seg_extra=seg_extra,
                cp=cp,
                opts=seg_opts,
                retry_policy=retry_policy,
                anchor_frame=anchor_frame,
            )[1:]

        parallel_out = run_segments_parallel(items, _worker, max_workers=opts.max_segment_workers)
        for pr in parallel_out:
            asg = work[pr.index][0]
            seg_results[asg.shot.index] = (pr.frames, pr.quality)
    else:
        for asg, seg_dir, seg_extra, cp, seg_opts in work:
            idx, frames, q = _process_one_segment(
                asg,
                plan,
                seg_dir,
                ckpt=ckpt,
                dry_run=dry_run,
                prov=prov,
                seg_extra=seg_extra,
                cp=cp,
                opts=seg_opts,
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


def score_clip_for_prompt(clip: Any, prompt: str) -> float:
    """
    Structured score: token overlap + grounded entity/tag hits + duration prior.

    Uses PromptGroundGraph so 'three red cars' prefers clips tagged red/car.
    """
    from pipelines.video.neural_sample import parse_prompt_ground

    g = parse_prompt_ground(prompt)
    q = set(_tokenize(prompt))
    for e in g.entities:
        q.add(e.name.lower())
        q.update(a.lower() for a in e.attributes)
    hay = set(_tokenize(getattr(clip, "title", "") or "")) | set(
        _tokenize(" ".join(str(t) for t in (getattr(clip, "tags", []) or [])))
    )
    if not q:
        return 0.0
    overlap = len(q & hay) / max(len(q), 1)
    # Entity hit bonus
    ent_hits = 0
    for e in g.entities:
        if e.negated:
            continue
        if e.name.lower() in hay or any(a.lower() in hay for a in e.attributes):
            ent_hits += 1
    ent = ent_hits / max(len([e for e in g.entities if not e.negated]), 1)
    return float(0.55 * overlap + 0.45 * ent)


def rank_clips_structured(
    clips: list[Any],
    prompt: str,
    *,
    top_k: int = 8,
) -> list[tuple[Any, float]]:
    scored = [(c, score_clip_for_prompt(c, prompt)) for c in clips]
    scored.sort(key=lambda t: -t[1])
    return scored[: max(1, top_k)]


def preview_shot_plan(
    prompt: str,
    *,
    mode: str = "t2v",
    duration_sec: float = 6.0,
    fps: float = 24.0,
) -> VideoPlan:
    return plan_video_from_prompt(
        prompt,
        mode=VideoMode(mode),
        duration_sec=duration_sec,
        fps=fps,
    )


def build_motion_score_cache(
    library_dir: str | Path,
    *,
    cache_path: Path | None = None,
    max_frames: int = 24,
) -> dict[str, float]:
    lib = Path(library_dir)
    cache_path = cache_path or lib / "motion_scores.json"
    scores: dict[str, float] = {}
    if cache_path.is_file():
        try:
            scores = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            scores = {}
    for clip in load_local_clip_library(lib):
        if clip.path in scores:
            continue
        tmp = lib / ".motion_cache" / Path(clip.path).stem
        frames = extract_frames(clip.path, tmp, max_frames=max_frames)
        scores[clip.path] = estimate_motion_score(frames)
    cache_path.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    return scores


def create_clip_sidecar(
    video_path: str | Path,
    *,
    title: str = "",
    tags: Sequence[str] | None = None,
    license: str = "local",
    url: str = "",
) -> Path:
    p = Path(video_path)
    sidecar = p.with_suffix(p.suffix + ".json")
    sidecar.write_text(
        json.dumps(
            {
                "title": title or p.stem,
                "tags": list(tags or []),
                "license": license,
                "url": url,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return sidecar


def log_segment_feedback(
    frame_paths: list[Path] | list[str],
    *,
    prompt: str = "",
    segment_index: int = 0,
    scores: dict[str, Any] | None = None,
    log_path: str | Path | None = None,
) -> Path | None:
    """Record a generate event for the hero (middle) frame of a segment."""
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if not paths:
        return None
    hero = paths[len(paths) // 2]
    try:
        from utils.training.preference import record_generation

        return record_generation(
            str(hero),
            prompt=prompt,
            media_type="video",
            log_path=log_path,
            extra={
                "segment_index": int(segment_index),
                "n_frames": len(paths),
                "scores": dict(scores or {}),
            },
        )
    except Exception:
        return None


def rate_segment(
    *,
    like: bool,
    hero_frame: str | Path,
    prompt: str = "",
    log_path: str | Path | None = None,
) -> Path:
    from utils.training.preference import record_dislike, record_like

    if like:
        return record_like(str(hero_frame), prompt=prompt, media_type="video", log_path=log_path)
    return record_dislike(str(hero_frame), prompt=prompt, media_type="video", log_path=log_path)

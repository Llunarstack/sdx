"""Text-to-video orchestration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from .retrieval import (
    download_remote_clip,
    load_local_clip_library,
    rank_clips_for_shot,
    search_pexels_videos,
    search_web_catalog,
)
from .shot_planner import plan_video_from_prompt
from .types import ClipCandidate, SegmentAssignment, VideoMode, VideoPlan
from .video_io import probe_video

__all__ = ["assign_clips_to_plan", "build_t2v_plan", "gather_retrieval_candidates"]


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
    download_dir: Optional[Path] = None,
) -> List[ClipCandidate]:
    candidates: List[ClipCandidate] = []
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
) -> List[SegmentAssignment]:
    assignments: List[SegmentAssignment] = []
    used: set[str] = set()
    for shot in plan.shots:
        ranked = rank_clips_for_shot(shot, candidates)
        pick: Optional[ClipCandidate] = None
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

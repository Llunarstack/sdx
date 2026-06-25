"""Extra helpers: plan preview, segment retry, motion scoring cache, clip sidecars."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .motion import estimate_motion_score
from .retrieval import load_local_clip_library, rank_clips_for_shot
from .segment_processor import process_segment
from .shot_planner import plan_video_from_prompt
from .t2v import gather_retrieval_candidates
from .types import SegmentAssignment, VideoMode, VideoPlan
from .video_io import extract_frames

__all__ = [
    "build_motion_score_cache",
    "create_clip_sidecar",
    "preview_retrieval_rankings",
    "preview_shot_plan",
    "retry_segment",
]


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


def preview_retrieval_rankings(
    prompt: str,
    *,
    local_library: str = "",
    catalog_path: str = "",
    top_k: int = 5,
) -> List[dict]:
    plan = preview_shot_plan(prompt)
    candidates = gather_retrieval_candidates(
        prompt,
        local_library=local_library,
        catalog_path=catalog_path,
    )
    rows: List[dict] = []
    for shot in plan.shots:
        ranked = rank_clips_for_shot(shot, candidates)[:top_k]
        rows.append(
            {
                "shot_index": shot.index,
                "shot_prompt": shot.prompt,
                "candidates": [
                    {"path": c.path, "title": c.title, "score": c.score, "source": c.source.value} for c in ranked
                ],
            }
        )
    return rows


def build_motion_score_cache(
    library_dir: str | Path,
    *,
    cache_path: Optional[Path] = None,
    max_frames: int = 24,
) -> Dict[str, float]:
    lib = Path(library_dir)
    cache_path = cache_path or lib / "motion_scores.json"
    scores: Dict[str, float] = {}
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
    tags: Optional[Sequence[str]] = None,
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


def retry_segment(
    assignment: SegmentAssignment,
    plan: VideoPlan,
    work_dir: str | Path,
    *,
    ckpt: str,
    **kwargs,
):
    return process_segment(assignment, plan, work_dir, ckpt=ckpt, **kwargs)

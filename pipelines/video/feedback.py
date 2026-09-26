"""Video segment feedback helpers for RSI (keyframes / hero frames)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["log_segment_feedback", "rate_segment"]


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
        from utils.training.feedback_bus import record_generation

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
    from utils.training.feedback_bus import record_dislike, record_like

    if like:
        return record_like(str(hero_frame), prompt=prompt, media_type="video", log_path=log_path)
    return record_dislike(str(hero_frame), prompt=prompt, media_type="video", log_path=log_path)

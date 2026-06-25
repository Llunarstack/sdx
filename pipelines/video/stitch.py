"""Stitch segments with transitions onto master timeline."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .transition_fx import apply_transition, transition_overlap_frames_fx
from .types import TransitionType
from .video_io import read_frame_rgb, save_frame_rgb, write_video_from_frames

__all__ = ["crossfade_segments", "stitch_segment_videos", "stitch_frame_lists"]


def crossfade_segments(
    frames_a: Sequence[np.ndarray],
    frames_b: Sequence[np.ndarray],
    overlap: int,
) -> List[np.ndarray]:
    if overlap <= 0 or not frames_a or not frames_b:
        return list(frames_a) + list(frames_b)
    overlap = min(overlap, len(frames_a), len(frames_b))
    head = list(frames_a[:-overlap])
    tail = list(frames_b[overlap:])
    blend: List[np.ndarray] = []
    for i in range(overlap):
        t = (i + 1) / (overlap + 1)
        a = frames_a[-overlap + i].astype(np.float32)
        b = frames_b[i].astype(np.float32)
        blend.append(np.clip((1 - t) * a + t * b, 0, 255).astype(np.uint8))
    return head + blend + tail


def stitch_frame_lists(
    segment_frames: List[List[Path]],
    out_path: str | Path,
    *,
    fps: float = 24.0,
    transitions: Optional[List[TransitionType]] = None,
) -> Path:
    transitions = transitions or [TransitionType.CUT] * len(segment_frames)
    merged: List[np.ndarray] = []
    for i, paths in enumerate(segment_frames):
        frames = [read_frame_rgb(p) for p in paths]
        if i == 0:
            merged = frames
            continue
        tr = transitions[i] if i < len(transitions) else TransitionType.CUT
        overlap = transition_overlap_frames_fx(tr, fps) if tr != TransitionType.CUT else 0
        merged = apply_transition(merged, frames, tr, overlap)
    out = Path(out_path)
    tmp = out.parent / f"{out.stem}_frames"
    tmp.mkdir(parents=True, exist_ok=True)
    fpsaths: List[Path] = []
    for j, rgb in enumerate(merged):
        fp = tmp / f"final_{j + 1:06d}.png"
        save_frame_rgb(fp, rgb)
        fpsaths.append(fp)
    return write_video_from_frames(fpsaths, out, fps=fps)


def stitch_segment_videos(
    segment_videos: List[str | Path],
    out_path: str | Path,
    *,
    fps: float = 24.0,
) -> Path:
    """Concat segment mp4s via ffmpeg when available."""
    from .video_io import extract_frames, ffmpeg_available

    if ffmpeg_available() and len(segment_videos) > 1:
        import subprocess

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        list_file = out.with_suffix(".concat.txt")
        lines = [f"file '{Path(v).resolve().as_posix()}'\n" for v in segment_videos]
        list_file.write_text("".join(lines), encoding="utf-8")
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(out)]
        subprocess.run(cmd, check=True, capture_output=True)
        list_file.unlink(missing_ok=True)
        return out
    all_frames: List[List[Path]] = []
    for v in segment_videos:
        tmp = Path(out_path).parent / f"_seg_{len(all_frames):02d}_frames"
        all_frames.append(extract_frames(v, tmp, fps=fps))
    return stitch_frame_lists(all_frames, out_path, fps=fps)

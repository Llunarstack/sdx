"""
Shot chain — last-frame → first-frame multi-shot continuity (production playbook).

Seed locking alone fails across clips. The industry fix is chaining: feed the
last frame of shot N as the first-frame condition for shot N+1, plus soft
palette/exposure harmonize across the cut.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .video_io import read_frame_rgb, save_frame_rgb

__all__ = [
    "ShotChainReport",
    "chain_anchor_from_segment",
    "harmonize_cut",
    "score_cut_continuity",
]


@dataclass(slots=True)
class ShotChainReport:
    score: float
    delta_luma: float = 0.0
    delta_chroma: float = 0.0
    notes: list[str] = field(default_factory=list)


def chain_anchor_from_segment(frame_paths: list[Path] | list[str], *, out_path: str | Path) -> Path:
    """Write the last frame of a segment as the next shot's start image."""
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if not paths:
        raise FileNotFoundError("chain_anchor_from_segment: empty segment")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rgb = read_frame_rgb(paths[-1])
    save_frame_rgb(out, rgb)
    return out


def score_cut_continuity(
    prev_frames: list[Path] | list[str],
    next_frames: list[Path] | list[str],
) -> ShotChainReport:
    prev = [Path(p) for p in prev_frames if Path(p).is_file()]
    nxt = [Path(p) for p in next_frames if Path(p).is_file()]
    if not prev or not nxt:
        return ShotChainReport(score=1.0)
    a = read_frame_rgb(prev[-1]).astype(np.float32)
    b = read_frame_rgb(nxt[0]).astype(np.float32)
    if a.shape != b.shape:
        from PIL import Image

        b = np.asarray(Image.fromarray(b.astype(np.uint8)).resize((a.shape[1], a.shape[0])), dtype=np.float32)
    luma_a = a.mean()
    luma_b = b.mean()
    chroma_a = a.reshape(-1, 3).mean(axis=0)
    chroma_b = b.reshape(-1, 3).mean(axis=0)
    dl = abs(float(luma_a - luma_b)) / 255.0
    dc = float(np.mean(np.abs(chroma_a - chroma_b))) / 255.0
    score = float(np.clip(1.0 - dl * 3.0 - dc * 4.0, 0.0, 1.0))
    notes = []
    if dl > 0.12:
        notes.append(f"cut_luma_jump={dl:.2f}")
    if dc > 0.10:
        notes.append(f"cut_chroma_jump={dc:.2f}")
    return ShotChainReport(score=score, delta_luma=dl, delta_chroma=dc, notes=notes)


def harmonize_cut(
    next_frames: list[Path],
    prev_anchor: str | Path,
    *,
    strength: float = 0.35,
    n_frames: int = 4,
) -> ShotChainReport:
    """
    Soft-match exposure/chroma of the first N frames of the next shot to the
    previous shot's last frame (bridge the cut without hard paste).
    """
    paths = [Path(p) for p in next_frames if Path(p).is_file()]
    anchor_p = Path(prev_anchor)
    if not paths or not anchor_p.is_file():
        return ShotChainReport(score=1.0)
    anchor = read_frame_rgb(anchor_p).astype(np.float32)
    a_mean = anchor.reshape(-1, 3).mean(axis=0) + 1e-3
    a = float(np.clip(strength, 0.0, 1.0))
    for p in paths[: max(1, n_frames)]:
        rgb = read_frame_rgb(p).astype(np.float32)
        b_mean = rgb.reshape(-1, 3).mean(axis=0) + 1e-3
        scale = a_mean / b_mean
        matched = np.clip(rgb * (1.0 - a) + rgb * scale * a, 0, 255)
        # Mild residual blend toward anchor luminance
        out = matched * (1.0 - 0.15 * a) + anchor.mean() * (0.15 * a)
        # Keep spatial content from matched
        out = matched
        save_frame_rgb(p, np.clip(out, 0, 255).astype(np.uint8))
    return score_cut_continuity([anchor_p], paths)

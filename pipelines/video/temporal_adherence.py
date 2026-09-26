"""
Temporal adherence — verify generated frames still match the grounded prompt.

Gates attribute presence (color proxies), count stability, and optional CLIP.
Drives segment retry when the model drifts from what the prompt asked for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .video_io import read_frame_rgb

__all__ = ["AdherenceReport", "score_temporal_adherence"]


@dataclass(slots=True)
class AdherenceReport:
    score: float
    structure_score: float = 1.0
    count_score: float = 1.0
    color_bind_score: float = 1.0
    clip_score: float = 1.0
    notes: list[str] = field(default_factory=list)


_COLOR_RGB = {
    "red": (200, 40, 40),
    "blue": (40, 40, 200),
    "green": (40, 180, 40),
    "yellow": (220, 200, 40),
    "orange": (230, 120, 30),
    "purple": (140, 40, 180),
    "pink": (230, 100, 160),
    "black": (20, 20, 20),
    "white": (230, 230, 230),
    "brown": (120, 70, 40),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
}


def _color_presence(rgb: np.ndarray, name: str) -> float:
    target = _COLOR_RGB.get(name.lower())
    if target is None:
        return 0.5
    t = np.asarray(target, dtype=np.float32)
    pix = rgb.astype(np.float32).reshape(-1, 3)
    # Fraction of pixels within distance of target
    dist = np.linalg.norm(pix - t[None], axis=1)
    hit = float((dist < 90.0).mean())
    return float(np.clip(hit * 8.0, 0.0, 1.0))  # small regions still score


def score_temporal_adherence(
    frame_paths: list[Path] | list[str],
    prompt: str,
    *,
    sample_every: int = 2,
    use_clip: bool = False,
) -> AdherenceReport:
    from .count_binder import score_count_stability
    from .prompt_ground_graph import parse_prompt_ground

    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    g = parse_prompt_ground(prompt)
    notes: list[str] = []
    if len(paths) < 1:
        return AdherenceReport(score=1.0)

    idxs = list(range(0, len(paths), max(1, sample_every)))
    # Color attribute binding: required colors should appear
    color_scores: list[float] = []
    required_colors = []
    for e in g.entities:
        if e.negated:
            continue
        for a in e.attributes:
            if a.lower() in _COLOR_RGB:
                required_colors.append(a.lower())
    if required_colors:
        for i in idxs:
            rgb = read_frame_rgb(paths[i])
            color_scores.append(float(np.mean([_color_presence(rgb, c) for c in required_colors])))
        color_bind = float(np.mean(color_scores)) if color_scores else 0.5
        if color_bind < 0.25:
            notes.append(f"weak_color_bind={color_bind:.2f}:{required_colors}")
    else:
        color_bind = 1.0

    # Negated colors should be rare
    for neg_c in [n for n in g.negations if n.lower() in _COLOR_RGB]:
        rgb = read_frame_rgb(paths[idxs[0]])
        if _color_presence(rgb, neg_c) > 0.35:
            notes.append(f"negation_leak:{neg_c}")
            color_bind *= 0.7

    count_rep = score_count_stability(paths, prompt=prompt, sample_every=sample_every)
    count_score = float(count_rep.score)
    notes.extend(count_rep.notes[:3])

    # Structure: entity density prior
    structure = 1.0
    if g.entities:
        structure = float(np.clip(0.6 + 0.1 * len([e for e in g.entities if not e.negated]), 0, 1))

    clip_s = 1.0
    if use_clip:
        try:
            import torch
            from utils.generation.clip_alignment import clip_image_text_cosine

            rgb = read_frame_rgb(paths[idxs[len(idxs) // 2]])
            t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
            sim = clip_image_text_cosine(t, g.rewritten or prompt, "openai/clip-vit-base-patch32", torch.device("cpu"))
            # Map typical CLIP sim ~0.2–0.35 → 0–1
            clip_s = float(np.clip((sim - 0.15) / 0.25, 0.0, 1.0))
            notes.append(f"clip_sim={sim:.3f}")
        except Exception as e:
            notes.append(f"clip_skip:{type(e).__name__}")

    score = float(np.clip(0.35 * color_bind + 0.30 * count_score + 0.15 * structure + 0.20 * clip_s, 0.0, 1.0))
    return AdherenceReport(
        score=score,
        structure_score=structure,
        count_score=count_score,
        color_bind_score=color_bind,
        clip_score=clip_s,
        notes=notes,
    )

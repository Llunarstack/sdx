"""
Seed-variation **guard** — detects mode collapse across a batch of generations.

Distilled/turbo models (Z-Image Turbo is the notorious case) can return
near-identical images for the same prompt across different seeds. This audits a
batch with cheap perceptual signatures (downsampled luma + color histograms),
reports a collapse score, and names which images to regenerate.

Pure numpy + PIL; no model inference. Intended for best-of-N loops and batch
CLI runs: audit → if collapsed, re-roll the flagged seeds with stronger
exploration (e.g. CADS / higher init-noise jitter).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "VariationReport",
    "audit_variation",
    "image_signature",
    "reseed_indices",
    "signature_distance",
]

_SIG_SIZE = 32
_HIST_BINS = 8
# Mean pairwise distance typical for a genuinely diverse same-prompt batch;
# calibrates collapse_score so 1.0 ≈ identical batch, 0.0 ≈ healthy variation.
_REFERENCE_DISTANCE = 0.30


@dataclass(slots=True)
class VariationReport:
    """Batch diversity audit result."""

    mean_distance: float
    min_distance: float
    collapse_score: float
    """0 = healthy variation, 1 = the batch collapsed to one image."""
    near_duplicate_pairs: list[tuple[int, int]] = field(default_factory=list)
    per_image_uniqueness: list[float] = field(default_factory=list)
    """Mean distance from each image to the rest (low = redundant)."""


def image_signature(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(normalized 32x32 luma, per-channel 8-bin histograms) for one RGB image."""
    from PIL import Image

    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img[..., :3])
    small = np.asarray(pil.resize((_SIG_SIZE, _SIG_SIZE), Image.BILINEAR), dtype=np.float32)
    luma = (small[..., 0] * 0.299 + small[..., 1] * 0.587 + small[..., 2] * 0.114) / 255.0
    hists = []
    for c in range(3):
        h, _ = np.histogram(img[..., c], bins=_HIST_BINS, range=(0, 256))
        hists.append(h.astype(np.float32) / max(float(h.sum()), 1.0))
    return luma, np.concatenate(hists)


def signature_distance(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> float:
    """Perceptual distance in [0, ~1]: structure (luma RMS) + palette (hist L1)."""
    luma_d = float(np.sqrt(np.mean((a[0] - b[0]) ** 2)))
    hist_d = float(np.abs(a[1] - b[1]).mean()) * _HIST_BINS / 2.0
    return 0.65 * luma_d + 0.35 * hist_d


def audit_variation(
    images: list[np.ndarray],
    *,
    near_duplicate_threshold: float = 0.05,
) -> VariationReport:
    """Audit a same-prompt batch for seed diversity."""
    n = len(images)
    if n < 2:
        return VariationReport(0.0, 0.0, 0.0, per_image_uniqueness=[0.0] * n)
    sigs = [image_signature(im) for im in images]
    dist = np.zeros((n, n), dtype=np.float64)
    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = signature_distance(sigs[i], sigs[j])
            dist[i, j] = dist[j, i] = d
            if d < near_duplicate_threshold:
                pairs.append((i, j))
    upper = dist[np.triu_indices(n, k=1)]
    mean_d = float(upper.mean())
    min_d = float(upper.min())
    collapse = float(np.clip(1.0 - mean_d / _REFERENCE_DISTANCE, 0.0, 1.0))
    uniqueness = [float(dist[i].sum() / (n - 1)) for i in range(n)]
    return VariationReport(mean_d, min_d, collapse, pairs, uniqueness)


def reseed_indices(report: VariationReport, *, max_collapse: float = 0.6) -> list[int]:
    """
    Which images to regenerate with fresh seeds / stronger exploration.

    From each near-duplicate pair, keep the more unique member and flag the
    other. Empty list when the batch is diverse enough (collapse below cap).
    """
    if report.collapse_score < max_collapse and not report.near_duplicate_pairs:
        return []
    flagged: set[int] = set()
    uniq = report.per_image_uniqueness
    for i, j in report.near_duplicate_pairs:
        keep, drop = (i, j) if uniq[i] >= uniq[j] else (j, i)
        if keep not in flagged:
            flagged.add(drop)
    if not flagged and report.collapse_score >= max_collapse and uniq:
        # Uniform collapse without discrete duplicate pairs: re-roll the most
        # redundant half of the batch.
        order = sorted(range(len(uniq)), key=lambda k: uniq[k])
        flagged.update(order[: max(1, len(order) // 2)])
    return sorted(flagged)

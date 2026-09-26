"""
Color-cast **neutralizer** — corrective counterpart to
``test_time_pick.score_color_cast_neutrality`` (which only scores).

Frontier models ship systematic global tints: GPT Image's viral warm bias
("piss filter") came from preference tuning and prompt rewriters pushing
golden-hour warmth into everything. The fix is measurement-driven: estimate the
midtone cast, then remove only the *excess* beyond a threshold so intentional
grading (sunsets, neon nights, noir) survives untouched.

Pure numpy; wired as an opt-in stage of ``human_made.apply_human_made_pipeline``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "CastEstimate",
    "estimate_color_cast",
    "neutralize_color_cast",
]


@dataclass(frozen=True, slots=True)
class CastEstimate:
    """Midtone-weighted global cast measurement."""

    gains: tuple[float, float, float]
    """Per-channel gray-world gains (>1 = channel is deficient)."""
    magnitude: float
    """Max |gain - 1| across channels; 0 = perfectly neutral."""
    warm_bias: float
    """Signed warmth: (R+G)/2 vs B on midtones, relative. >0 = warm/yellow cast."""


def _midtone_weights(luma: np.ndarray) -> np.ndarray:
    # Blown highlights and crushed shadows carry no cast information; weight a
    # smooth window over [~30, ~225] so the estimate reads the graded midtones.
    lo = np.clip((luma - 20.0) / 40.0, 0.0, 1.0)
    hi = np.clip((235.0 - luma) / 40.0, 0.0, 1.0)
    return lo * hi


def estimate_color_cast(image: np.ndarray) -> CastEstimate:
    """Estimate the global cast of an RGB image (uint8 or float 0-255)."""
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 3 or img.shape[2] < 3:
        return CastEstimate((1.0, 1.0, 1.0), 0.0, 0.0)
    luma = img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114
    w = _midtone_weights(luma)
    wsum = float(w.sum())
    if wsum < 1e-3:  # all-black / all-white image: nothing to measure
        return CastEstimate((1.0, 1.0, 1.0), 0.0, 0.0)
    means = [float((img[..., c] * w).sum() / wsum) for c in range(3)]
    target = sum(means) / 3.0
    if target < 1e-3:
        return CastEstimate((1.0, 1.0, 1.0), 0.0, 0.0)
    gains = tuple(target / max(m, 1e-3) for m in means)
    magnitude = max(abs(g - 1.0) for g in gains)
    warm = ((means[0] + means[1]) / 2.0 - means[2]) / target
    return CastEstimate(gains, float(magnitude), float(warm))


def neutralize_color_cast(
    image: np.ndarray,
    *,
    strength: float = 0.5,
    threshold: float = 0.06,
    max_gain: float = 0.25,
    warm_only: bool = False,
) -> np.ndarray:
    """
    Remove the excess global cast beyond ``threshold``.

    A cast at or below ``threshold`` is left alone; stronger casts are pulled
    back proportionally (never fully to neutral unless ``strength=1`` and the
    cast is far past threshold). ``warm_only=True`` targets the warm-tint
    failure mode specifically and never touches cool-graded images.
    """
    s = float(max(0.0, min(1.0, strength)))
    if s <= 0.0:
        return image
    est = estimate_color_cast(image)
    if est.magnitude <= threshold:
        return image
    if warm_only and est.warm_bias <= threshold:
        return image
    # Fraction of the cast that is "excess": 0 at threshold, →1 for strong casts.
    excess = 1.0 - threshold / max(est.magnitude, 1e-6)
    blend = s * excess
    was_uint = image.dtype == np.uint8
    img = np.asarray(image, dtype=np.float32)
    out = img.copy()
    for c, g in enumerate(est.gains):
        eff = float(g) ** blend  # partial gray-world correction
        eff = float(np.clip(eff, 1.0 - max_gain, 1.0 + max_gain))
        out[..., c] = img[..., c] * eff
    out = np.clip(out, 0.0, 255.0)
    return out.astype(np.uint8) if was_uint else out

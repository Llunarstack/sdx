from __future__ import annotations


def style_detail_mix_for_progress(
    progress: float,
    *,
    style_strength: float = 1.0,
    detail_strength: float = 1.0,
    style_late_boost: float = 1.2,
    detail_early_boost: float = 1.15,
) -> tuple[float, float]:
    """
    Return (style_scale, detail_scale) for normalized denoise progress.
    """
    p = max(0.0, min(1.0, float(progress)))
    style_scale = float(style_strength) * (1.0 + (float(style_late_boost) - 1.0) * p)
    detail_scale = float(detail_strength) * (1.0 + (float(detail_early_boost) - 1.0) * (1.0 - p))
    return style_scale, detail_scale


def bounded_scale(value: float, *, lo: float = 0.0, hi: float = 3.0) -> float:
    return max(float(lo), min(float(hi), float(value)))


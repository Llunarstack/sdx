"""
Cheap **perceptual proxies** for ranking samples and tuning pipelines.

These scores approximate sharpness / color / exposure balance so you can pick better seeds,
compare schedulers, or gate test-time compute. They do **not** measure “humanness” or guarantee
any detector will fail — use labels and platform rules for synthetic media where required.
"""

from __future__ import annotations

from .perceptual_proxies import (
    colorfulness_std,
    combined_quality_proxy,
    exposure_naturalness,
    laplacian_sharpness,
    parse_rgb01_bchw,
)
from .rank_and_gate import gate_by_proxy_threshold, rank_samples_by_proxy

__all__ = [
    "colorfulness_std",
    "combined_quality_proxy",
    "exposure_naturalness",
    "gate_by_proxy_threshold",
    "laplacian_sharpness",
    "parse_rgb01_bchw",
    "rank_samples_by_proxy",
]

SYNTHETIC_MEDIA_NOTE = (
    "High scores here only mean fewer obvious artifacts under these heuristics. "
    "They are not a substitute for disclosure where policy or law requires it."
)

"""
Bridge speed research modules to diffusion sampling extras.

Production paths:
  - Feature / block cache: ``sample.py`` ``--feature-cache-delta``, ``--block-cache-thresh``
  - Taylor cache, dynamic DiT: ``utils/superior/``
  - Holy grail presets: ``diffusion/sampling/``
"""

from __future__ import annotations

from typing import Any, Dict

from .engine import RealtimeGenerationEngine


def sample_loop_speed_kwargs(
    *,
    target_latency_ms: int = 100,
    enable_cache: bool = True,
    prune_ratio: float = 0.3,
) -> Dict[str, Any]:
    """
    Map research speed engine settings to ``sample_loop`` / ``sample.py`` kwargs.

    Returns keys safe to merge into ``_superior_kw`` or pass to ``diffusion.sample_loop``.
    """
    engine = RealtimeGenerationEngine()
    engine.token_prune.prune_ratio = float(prune_ratio)
    _ = engine  # reserved for future latency-aware routing
    kw: Dict[str, Any] = {}
    if enable_cache and target_latency_ms < 200:
        kw["feature_cache_delta_threshold"] = 0.02
        kw["feature_cache_max_reuse"] = 2
        kw["block_cache_threshold"] = 0.015
        kw["block_cache_recompute_every"] = 4
    return kw


def estimate_step_budget(target_latency_ms: int, *, ms_per_step: float = 45.0) -> int:
    """Rough step count for a latency target (heuristic, GPU-dependent)."""
    return max(4, int(target_latency_ms / max(ms_per_step, 1.0)))


__all__ = ["estimate_step_budget", "sample_loop_speed_kwargs"]

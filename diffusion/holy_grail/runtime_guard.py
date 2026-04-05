from __future__ import annotations

from typing import Dict


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(v)))


def sanitize_holy_grail_kwargs(kwargs: Dict[str, float | bool]) -> Dict[str, float | bool]:
    """
    Clamp and normalize holy-grail kwargs to stable ranges.
    Returns a copied dict.
    """
    out: Dict[str, float | bool] = dict(kwargs)
    out["holy_grail_cfg_early_ratio"] = _clamp(float(out.get("holy_grail_cfg_early_ratio", 0.72)), 0.4, 1.4)
    out["holy_grail_cfg_late_ratio"] = _clamp(float(out.get("holy_grail_cfg_late_ratio", 1.0)), 0.4, 1.6)
    out["holy_grail_control_mult"] = _clamp(float(out.get("holy_grail_control_mult", 1.0)), 0.0, 2.5)
    out["holy_grail_adapter_mult"] = _clamp(float(out.get("holy_grail_adapter_mult", 1.0)), 0.0, 2.5)
    out["holy_grail_late_adapter_boost"] = _clamp(float(out.get("holy_grail_late_adapter_boost", 1.15)), 0.8, 2.0)
    out["holy_grail_cads_strength"] = _clamp(float(out.get("holy_grail_cads_strength", 0.0)), 0.0, 0.2)
    out["holy_grail_cads_min_strength"] = _clamp(float(out.get("holy_grail_cads_min_strength", 0.0)), 0.0, 0.2)
    out["holy_grail_cads_power"] = _clamp(float(out.get("holy_grail_cads_power", 1.0)), 0.25, 3.0)
    out["holy_grail_unsharp_sigma"] = _clamp(float(out.get("holy_grail_unsharp_sigma", 0.0)), 0.0, 2.0)
    out["holy_grail_unsharp_amount"] = _clamp(float(out.get("holy_grail_unsharp_amount", 0.0)), 0.0, 0.6)
    out["holy_grail_clamp_quantile"] = _clamp(float(out.get("holy_grail_clamp_quantile", 0.0)), 0.0, 0.999)
    out["holy_grail_clamp_floor"] = _clamp(float(out.get("holy_grail_clamp_floor", 1.0)), 0.25, 3.0)
    out["holy_grail_frontload_control"] = bool(out.get("holy_grail_frontload_control", True))
    out["holy_grail_enable"] = bool(out.get("holy_grail_enable", False))

    # Consistency constraints.
    if out["holy_grail_cads_min_strength"] > out["holy_grail_cads_strength"]:
        out["holy_grail_cads_min_strength"] = out["holy_grail_cads_strength"]
    if out["holy_grail_unsharp_amount"] <= 0.0:
        out["holy_grail_unsharp_sigma"] = 0.0
    if out["holy_grail_clamp_quantile"] < 0.5:
        out["holy_grail_clamp_quantile"] = 0.0
    return out


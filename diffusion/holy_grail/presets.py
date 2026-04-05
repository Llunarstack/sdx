from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class HolyGrailPreset:
    name: str
    description: str
    cfg_early_ratio: float = 0.72
    cfg_late_ratio: float = 1.0
    control_mult: float = 1.0
    adapter_mult: float = 1.0
    frontload_control: bool = True
    late_adapter_boost: float = 1.15
    cads_strength: float = 0.0
    cads_min_strength: float = 0.0
    cads_power: float = 1.0
    unsharp_sigma: float = 0.0
    unsharp_amount: float = 0.0
    clamp_quantile: float = 0.0
    clamp_floor: float = 1.0


HOLY_GRAIL_PRESETS: Dict[str, HolyGrailPreset] = {
    "balanced": HolyGrailPreset(
        name="balanced",
        description="General quality/adherence balance for mixed prompts.",
        cfg_early_ratio=0.74,
        cfg_late_ratio=1.0,
        control_mult=1.05,
        adapter_mult=1.0,
        frontload_control=True,
        late_adapter_boost=1.12,
        cads_strength=0.015,
        cads_min_strength=0.0,
        cads_power=1.0,
        unsharp_sigma=0.45,
        unsharp_amount=0.10,
        clamp_quantile=0.992,
        clamp_floor=1.0,
    ),
    "photoreal": HolyGrailPreset(
        name="photoreal",
        description="Cleaner tones and detail for photo-like generations.",
        cfg_early_ratio=0.70,
        cfg_late_ratio=0.96,
        control_mult=1.1,
        adapter_mult=0.95,
        frontload_control=True,
        late_adapter_boost=1.08,
        cads_strength=0.02,
        cads_min_strength=0.003,
        cads_power=1.2,
        unsharp_sigma=0.55,
        unsharp_amount=0.13,
        clamp_quantile=0.994,
        clamp_floor=1.0,
    ),
    "anime": HolyGrailPreset(
        name="anime",
        description="Sharper linework and stronger style expression.",
        cfg_early_ratio=0.76,
        cfg_late_ratio=1.02,
        control_mult=1.0,
        adapter_mult=1.1,
        frontload_control=True,
        late_adapter_boost=1.2,
        cads_strength=0.012,
        cads_min_strength=0.0,
        cads_power=0.9,
        unsharp_sigma=0.50,
        unsharp_amount=0.16,
        clamp_quantile=0.995,
        clamp_floor=1.0,
    ),
    "illustration": HolyGrailPreset(
        name="illustration",
        description="Balanced stylization with robust composition adherence.",
        cfg_early_ratio=0.75,
        cfg_late_ratio=1.0,
        control_mult=1.08,
        adapter_mult=1.05,
        frontload_control=True,
        late_adapter_boost=1.16,
        cads_strength=0.014,
        cads_min_strength=0.0,
        cads_power=1.0,
        unsharp_sigma=0.52,
        unsharp_amount=0.14,
        clamp_quantile=0.993,
        clamp_floor=1.0,
    ),
    "aggressive": HolyGrailPreset(
        name="aggressive",
        description="High-adherence/high-style preset; can reduce diversity.",
        cfg_early_ratio=0.8,
        cfg_late_ratio=1.1,
        control_mult=1.2,
        adapter_mult=1.15,
        frontload_control=True,
        late_adapter_boost=1.25,
        cads_strength=0.03,
        cads_min_strength=0.005,
        cads_power=1.4,
        unsharp_sigma=0.60,
        unsharp_amount=0.20,
        clamp_quantile=0.996,
        clamp_floor=1.05,
    ),
}


def list_holy_grail_presets() -> List[str]:
    return sorted(HOLY_GRAIL_PRESETS.keys())


def get_holy_grail_preset(name: str) -> HolyGrailPreset:
    k = str(name or "").strip().lower()
    if not k:
        raise KeyError("Empty holy grail preset name")
    return HOLY_GRAIL_PRESETS[k]


def apply_holy_grail_preset_to_args(args: Any, preset_name: str) -> None:
    """
    Soft-apply preset values (only when args are still at defaults).
    """
    try:
        p = get_holy_grail_preset(preset_name)
    except KeyError:
        return

    def maybe_set(name: str, value, defaults: tuple) -> None:
        if not hasattr(args, name):
            return
        cur = getattr(args, name)
        if cur in defaults:
            setattr(args, name, value)

    if hasattr(args, "holy_grail") and not bool(getattr(args, "holy_grail", False)):
        setattr(args, "holy_grail", True)
    maybe_set("holy_grail_cfg_early_ratio", p.cfg_early_ratio, (0.72, None))
    maybe_set("holy_grail_cfg_late_ratio", p.cfg_late_ratio, (1.0, None))
    maybe_set("holy_grail_control_mult", p.control_mult, (1.0, None))
    maybe_set("holy_grail_adapter_mult", p.adapter_mult, (1.0, None))
    maybe_set("holy_grail_late_adapter_boost", p.late_adapter_boost, (1.15, None))
    maybe_set("holy_grail_cads_strength", p.cads_strength, (0.0, None))
    maybe_set("holy_grail_cads_min_strength", p.cads_min_strength, (0.0, None))
    maybe_set("holy_grail_cads_power", p.cads_power, (1.0, None))
    maybe_set("holy_grail_unsharp_sigma", p.unsharp_sigma, (0.0, None))
    maybe_set("holy_grail_unsharp_amount", p.unsharp_amount, (0.0, None))
    maybe_set("holy_grail_clamp_quantile", p.clamp_quantile, (0.0, None))
    maybe_set("holy_grail_clamp_floor", p.clamp_floor, (1.0, None))

    # Inverse CLI flag for frontload behavior.
    if hasattr(args, "holy_grail_no_frontload_control"):
        no_frontload = bool(getattr(args, "holy_grail_no_frontload_control", False))
        if not no_frontload and not p.frontload_control:
            setattr(args, "holy_grail_no_frontload_control", True)


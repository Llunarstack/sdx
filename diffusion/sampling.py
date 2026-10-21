"""
The "holy grail" sampling recipe (``--holy-grail*``): coordinated per-step schedules instead of one fixed CFG and
fixed control/adapter strengths — CFG ramps up, ControlNet front-loads, adapters boost late.

- ``build_holy_grail_step_plan``: the recipe's concrete CFG / control / adapter scalars for one step.
- ``cads_noise_std`` / ``apply_condition_noise``: CADS condition annealing (Sadat et al., ICLR 2024) — noise on
  the text conditioning early, when only coarse structure is decided, annealed away so details stay faithful;
  more varied outputs at high guidance without losing adherence.
- ``HOLY_GRAIL_PRESETS`` + ``recommend_holy_grail_preset``: tuned knob sets per content type, picked by prompt
  keywords; ``apply_holy_grail_preset_to_args`` only overrides args still at their defaults.
- ``sanitize_holy_grail_kwargs``: clamps every knob to a stable range before ``sample_loop`` sees it.

``GaussianDiffusion.sample_loop`` consumes the plan and CADS each step; the final unsharp / clamp live in
``diffusion.sampling_utils``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class HolyGrailRecipe:
    """Static recipe: CFG ramp ratios (of ``base_cfg``) and base control / adapter scales."""

    base_cfg: float = 7.0
    cfg_early_ratio: float = 0.72
    cfg_late_ratio: float = 1.0
    control_base_scale: float = 1.0
    adapter_base_scale: float = 1.0
    frontload_control: bool = True
    late_adapter_boost: float = 1.15


@dataclass(slots=True)
class HolyGrailStepPlan:
    cfg_scale: float
    control_scale: float
    adapter_scale: float


def build_holy_grail_step_plan(*, recipe: HolyGrailRecipe, step_index: int, total_steps: int) -> HolyGrailStepPlan:
    """Scalars for step ``step_index`` of ``total_steps`` (progress 0 = first, noisiest step)."""
    p = 1.0 if total_steps <= 1 else max(0.0, min(1.0, step_index / (total_steps - 1)))
    cfg = recipe.base_cfg * (recipe.cfg_early_ratio + (recipe.cfg_late_ratio - recipe.cfg_early_ratio) * p)
    control_mul = 1.2 - 0.35 * p if recipe.frontload_control else 0.9 + 0.2 * p
    adapter = recipe.adapter_base_scale * (0.9 + 0.25 * p) * (1.0 + (recipe.late_adapter_boost - 1.0) * p)
    return HolyGrailStepPlan(
        cfg_scale=float(cfg),
        control_scale=float(recipe.control_base_scale * control_mul),
        adapter_scale=float(adapter),
    )


def cads_noise_std(
    *, progress: float, base_strength: float = 0.0, min_strength: float = 0.0, power: float = 1.0
) -> float:
    """Condition-noise std at denoise ``progress`` in [0, 1]: ``base`` at the start decaying as ``(1-p)^power``
    to ``min`` at the end."""
    p = max(0.0, min(1.0, float(progress)))
    b, m = max(0.0, float(base_strength)), max(0.0, float(min_strength))
    return max(0.0, m + (b - m) * (1.0 - p) ** max(1e-6, float(power)))


def apply_condition_noise(cond: torch.Tensor, *, std: float) -> torch.Tensor:
    """``cond + N(0, std^2)`` (identity for ``std <= 0``)."""
    return cond + torch.randn_like(cond) * float(std) if float(std) > 0.0 else cond


@dataclass(frozen=True, slots=True)
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


HOLY_GRAIL_PRESETS: dict[str, HolyGrailPreset] = {
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


def list_holy_grail_presets() -> list[str]:
    return sorted(HOLY_GRAIL_PRESETS)


def get_holy_grail_preset(name: str) -> HolyGrailPreset:
    k = str(name or "").strip().lower()
    if not k:
        raise KeyError("Empty holy grail preset name")
    return HOLY_GRAIL_PRESETS[k]


def recommend_holy_grail_preset(
    *, prompt: str = "", style: str = "", has_control: bool = False, has_lora: bool = False
) -> str:
    """Keyword pick: anime > illustration > (control + LoRA -> balanced) > photoreal > balanced."""
    text = f"{prompt} {style}".lower()
    if any(k in text for k in ("anime", "manga", "cel", "waifu", "danbooru")):
        return "anime"
    if any(k in text for k in ("illustration", "concept art", "painterly", "comic", "storybook")):
        return "illustration"
    if has_control and has_lora:
        return "balanced"
    if any(k in text for k in ("photo", "photoreal", "cinematic", "dslr", "realistic", "portrait")):
        return "photoreal"
    return "balanced"


def apply_holy_grail_preset_to_args(args: Any, preset_name: str) -> None:
    """Enable the recipe and copy preset values onto args that are still at their CLI defaults."""
    try:
        p = get_holy_grail_preset(preset_name)
    except KeyError:
        return

    def maybe_set(name: str, value: float, default: float) -> None:
        if hasattr(args, name) and getattr(args, name) in (default, None):
            setattr(args, name, value)

    if hasattr(args, "holy_grail"):
        args.holy_grail = True
    maybe_set("holy_grail_cfg_early_ratio", p.cfg_early_ratio, 0.72)
    maybe_set("holy_grail_cfg_late_ratio", p.cfg_late_ratio, 1.0)
    maybe_set("holy_grail_control_mult", p.control_mult, 1.0)
    maybe_set("holy_grail_adapter_mult", p.adapter_mult, 1.0)
    maybe_set("holy_grail_late_adapter_boost", p.late_adapter_boost, 1.15)
    maybe_set("holy_grail_cads_strength", p.cads_strength, 0.0)
    maybe_set("holy_grail_cads_min_strength", p.cads_min_strength, 0.0)
    maybe_set("holy_grail_cads_power", p.cads_power, 1.0)
    maybe_set("holy_grail_unsharp_sigma", p.unsharp_sigma, 0.0)
    maybe_set("holy_grail_unsharp_amount", p.unsharp_amount, 0.0)
    maybe_set("holy_grail_clamp_quantile", p.clamp_quantile, 0.0)
    maybe_set("holy_grail_clamp_floor", p.clamp_floor, 1.0)
    if not p.frontload_control and hasattr(args, "holy_grail_no_frontload_control"):
        args.holy_grail_no_frontload_control = True


def sanitize_holy_grail_kwargs(kwargs: dict[str, float | bool]) -> dict[str, float | bool]:
    """Copy of ``kwargs`` with every holy-grail knob clamped to a stable range and made self-consistent."""
    out: dict[str, float | bool] = dict(kwargs)
    for key, default, lo, hi in (
        ("holy_grail_cfg_early_ratio", 0.72, 0.4, 1.4),
        ("holy_grail_cfg_late_ratio", 1.0, 0.4, 1.6),
        ("holy_grail_control_mult", 1.0, 0.0, 2.5),
        ("holy_grail_adapter_mult", 1.0, 0.0, 2.5),
        ("holy_grail_late_adapter_boost", 1.15, 0.8, 2.0),
        ("holy_grail_cads_strength", 0.0, 0.0, 0.2),
        ("holy_grail_cads_min_strength", 0.0, 0.0, 0.2),
        ("holy_grail_cads_power", 1.0, 0.25, 3.0),
        ("holy_grail_unsharp_sigma", 0.0, 0.0, 2.0),
        ("holy_grail_unsharp_amount", 0.0, 0.0, 0.6),
        ("holy_grail_clamp_quantile", 0.0, 0.0, 0.999),
        ("holy_grail_clamp_floor", 1.0, 0.25, 3.0),
    ):
        out[key] = max(lo, min(hi, float(out.get(key, default))))
    out["holy_grail_frontload_control"] = bool(out.get("holy_grail_frontload_control", True))
    out["holy_grail_enable"] = bool(out.get("holy_grail_enable", False))
    out["holy_grail_cads_min_strength"] = min(out["holy_grail_cads_min_strength"], out["holy_grail_cads_strength"])
    if out["holy_grail_unsharp_amount"] <= 0.0:
        out["holy_grail_unsharp_sigma"] = 0.0
    if out["holy_grail_clamp_quantile"] < 0.5:
        out["holy_grail_clamp_quantile"] = 0.0
    return out

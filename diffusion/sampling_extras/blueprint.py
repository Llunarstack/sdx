from __future__ import annotations

from dataclasses import dataclass

import torch

from .guidance_fusion import fuse_condition_scales


@dataclass
class HolyGrailRecipe:
    """
    Master recipe to coordinate strong prompt adherence and style/control stability.
    """

    base_cfg: float = 7.0
    cfg_early_ratio: float = 0.72
    cfg_late_ratio: float = 1.0
    control_base_scale: float = 1.0
    adapter_base_scale: float = 1.0
    frontload_control: bool = True
    late_adapter_boost: float = 1.15


@dataclass
class HolyGrailStepPlan:
    """
    Per-step runtime plan values.
    """

    cfg_scale: float
    control_scale: float
    adapter_scale: float
    refine_strength: float


def build_holy_grail_step_plan(
    *,
    recipe: HolyGrailRecipe,
    step_index: int,
    total_steps: int,
) -> HolyGrailStepPlan:
    """
    Build per-step scalar knobs for inference loops.
    """
    if total_steps <= 1:
        p = 1.0
    else:
        p = float(step_index) / float(max(total_steps - 1, 1))
        p = max(0.0, min(1.0, p))

    cfg_scale = float(recipe.base_cfg) * (
        float(recipe.cfg_early_ratio) + (float(recipe.cfg_late_ratio) - float(recipe.cfg_early_ratio)) * p
    )
    cond = fuse_condition_scales(
        base_control_scale=float(recipe.control_base_scale),
        base_adapter_scale=float(recipe.adapter_base_scale),
        progress=p,
        frontload_control=bool(recipe.frontload_control),
    )
    adapter_scale = float(cond[1]) * (1.0 + (float(recipe.late_adapter_boost) - 1.0) * p)
    refine_strength = 0.05 + 0.20 * p
    return HolyGrailStepPlan(
        cfg_scale=float(cfg_scale),
        control_scale=float(cond[0]),
        adapter_scale=float(adapter_scale),
        refine_strength=float(refine_strength),
    )


def apply_step_plan_to_kwargs(
    model_kwargs: dict,
    plan: HolyGrailStepPlan,
) -> dict:
    """
    Copy and apply the plan onto model kwargs.
    """
    mk = dict(model_kwargs)
    mk["control_scale"] = torch.tensor(plan.control_scale, dtype=torch.float32)
    mk["adapter_scale"] = torch.tensor(plan.adapter_scale, dtype=torch.float32)
    mk["cfg_scale"] = float(plan.cfg_scale)
    mk["refine_strength"] = float(plan.refine_strength)
    return mk


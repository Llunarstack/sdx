"""
Consume frontier planner outputs into real sample args / diffusion kwargs.

Frontier modules emit rich plans; without this bridge those fields are dropped
or (worse) TypeError when leaked into ``sample_loop``.
"""

from __future__ import annotations

import re
from typing import Any

# Keys that gaussian_diffusion.sample_loop / flow loops accept (subset + known extras).
_DIFFUSION_SAFE_KEYS = frozenset(
    {
        "step_noise_scales",
        "attention_layout_plan",
        "attention_layout_masks",
        "per_region_cads",
        "cads_noise_scale",
        "cads_mix_ratio",
    }
)


def apply_token_emphasis_to_args(args: Any, *, prompt: str | None = None) -> dict[str, Any]:
    """
    Soft-apply TokenEmphasisPlanner: SD-weight fragments + CFG bump for hard tokens.

    Safe to call from quality policy / frontier prompt phase. Skips if prompt
    already contains ``(word:1.`` weight tags from a prior pass.
    """
    from frontier.adherence.token_emphasis import TokenEmphasisPlanner

    text = prompt if prompt is not None else str(getattr(args, "prompt", "") or "")
    if not text.strip():
        return {"skipped": "empty"}
    if re.search(r"\([^)]+:1\.\d+", text):
        return {"skipped": "already_weighted"}

    planner = TokenEmphasisPlanner()
    plan = planner.plan(text)
    if not plan.weights:
        return {"skipped": "no_hard_tokens"}

    args.prompt = planner.augment_with_weights(text, plan)
    cfg = getattr(args, "cfg_scale", None)
    if cfg is not None and plan.cfg_multiplier > 1.0:
        args.cfg_scale = float(cfg) * float(plan.cfg_multiplier)
    return {
        "weights": plan.as_prompt_weights(),
        "cfg_multiplier": plan.cfg_multiplier,
        "prompt": args.prompt,
    }


def merge_step_emphasis_into_noise_scales(
    *,
    step_emphasis: list[float] | tuple[float, ...] | None,
    step_noise_scales: list[float] | tuple[float, ...] | None,
) -> list[float] | None:
    """
    Map narrative/mood ``step_emphasis`` into ``step_noise_scales`` (multiplicative).

    Sampler only understands ``step_noise_scales``; emphasis alone was a dead field.
    """
    if not step_emphasis:
        return list(step_noise_scales) if step_noise_scales is not None else None
    emp = [float(x) for x in step_emphasis]
    if not step_noise_scales:
        # Center around 1.0 so mean emphasis ≈ identity noise scale
        mean = sum(emp) / max(len(emp), 1)
        mean = mean if mean > 1e-6 else 1.0
        return [max(0.05, e / mean) for e in emp]
    base = [float(x) for x in step_noise_scales]
    n = max(len(emp), len(base))
    out: list[float] = []
    mean_e = sum(emp) / max(len(emp), 1) or 1.0
    for i in range(n):
        e = emp[min(i, len(emp) - 1)] / mean_e
        b = base[min(i, len(base) - 1)]
        out.append(max(0.05, b * e))
    return out


def sanitize_diffusion_feature_kwargs(extra: dict[str, Any]) -> dict[str, Any]:
    """Drop / remap planner keys that would crash ``sample_loop(**kwargs)``."""
    if not extra:
        return {}
    work = dict(extra)
    emphasis = work.pop("step_emphasis", None)
    if emphasis is not None:
        work["step_noise_scales"] = merge_step_emphasis_into_noise_scales(
            step_emphasis=emphasis,
            step_noise_scales=work.get("step_noise_scales"),
        )
    # Strip known metadata that never belongs in the sampler
    for dead in (
        "frontier_token_emphasis",
        "frontier_cfg_emphasis_mult",
        "frontier_step_emphasis",
        "frontier_recommend_best_of_n",
        "frontier_guidance_tiers",
        "frontier_compute_cost",
        "frontier_relation_hints",
        "cfg_scale_multiplier",
        "perfect_frontier",
        "safety_decision",
        "safety_tier",
        "refused",
        "refuse_reasons",
        "prompt",
        "negative_prompt",
        "serendipity_scales",
        "entropy_per_step",
    ):
        work.pop(dead, None)

    # Keep safe keys + anything that looks like an intentional sampler hook
    out: dict[str, Any] = {}
    for k, v in work.items():
        if k in _DIFFUSION_SAFE_KEYS or k.startswith(("cads_", "attention_", "step_", "region_")):
            out[k] = v
    return out


__all__ = [
    "apply_token_emphasis_to_args",
    "merge_step_emphasis_into_noise_scales",
    "sanitize_diffusion_feature_kwargs",
]

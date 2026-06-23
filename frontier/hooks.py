"""
Wire frontier plans into the production sampling stack.

Safe to import from ``sample.py`` — no heavy model weights.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .engine import FrontierEngine, FrontierPlan, analyze_prompt


def apply_frontier_to_args(
    args: Any,
    *,
    prompt: Optional[str] = None,
    engine: Optional[FrontierEngine] = None,
) -> FrontierPlan:
    """
    Mutate an argparse namespace (or similar) with frontier-derived fields.

    Sets ``prompt``, ``negative_prompt``, and attaches ``frontier_plan`` on ``args``.
    """
    eng = engine or FrontierEngine()
    p = prompt if prompt is not None else getattr(args, "prompt", "") or ""
    plan = eng.analyze(p)
    kw = eng.sample_kwargs(plan, base_negative=getattr(args, "negative_prompt", "") or "")

    if kw.get("prompt"):
        args.prompt = kw["prompt"]
    if kw.get("negative_prompt") is not None:
        args.negative_prompt = kw["negative_prompt"]
    args.frontier_plan = plan
    args.frontier_risk_score = kw.get("frontier_risk_score", 0.0)
    return plan


def frontier_diffusion_hooks(plan: FrontierPlan) -> Dict[str, Any]:
    """
    Extra kwargs for ``gaussian_diffusion`` sample loops (optional).

    Consumers can multiply per-step noise by ``serendipity_scales[step]`` and
    ``entropy_per_step[step]`` when present.
    """
    out: Dict[str, Any] = {}
    if plan.serendipity is not None:
        out["serendipity_scales"] = list(plan.serendipity.scales)
    if plan.entropy is not None:
        out["entropy_per_step"] = list(plan.entropy.per_step)
    if plan.moment is not None:
        out["step_emphasis"] = list(plan.moment.step_emphasis)
    return out


__all__ = [
    "analyze_prompt",
    "apply_frontier_to_args",
    "frontier_diffusion_hooks",
    "FrontierEngine",
    "FrontierPlan",
]

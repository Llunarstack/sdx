"""Wire frontier plans into the production sampling stack (light import: no model weights)."""

from __future__ import annotations

from typing import Any

from .engine import FrontierEngine, FrontierPlan, analyze_prompt


def apply_frontier_to_args(
    args: Any,
    *,
    prompt: str | None = None,
    engine: FrontierEngine | None = None,
) -> FrontierPlan:
    """Set ``prompt``, ``negative_prompt``, ``frontier_plan`` and ``frontier_risk_score`` on an argparse namespace."""
    eng = engine or FrontierEngine()
    plan = eng.analyze(prompt if prompt is not None else getattr(args, "prompt", "") or "")
    kw = eng.sample_kwargs(plan, base_negative=getattr(args, "negative_prompt", "") or "")
    if kw["prompt"]:
        args.prompt = kw["prompt"]
    args.negative_prompt = kw["negative_prompt"]
    args.frontier_plan = plan
    args.frontier_risk_score = kw["frontier_risk_score"]
    return plan


def frontier_diffusion_hooks(plan: FrontierPlan) -> dict[str, Any]:
    """Per-step curves for the sample loop: ``serendipity_scales``, ``entropy_per_step``, ``step_emphasis``."""
    out: dict[str, Any] = {}
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

"""
Deep frontier synthesis — compose all analyzers into one ``DeepFrontierPlan``.

Use from ``sample.py`` with ``--frontier deep`` (when wired) or directly in tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .adherence.token_emphasis import TokenEmphasisMap, TokenEmphasisPlanner
from .causality.physical_plausibility import PhysicalPlausibilityScanner, PlausibilityFlag
from .economy.compute_budget import ComputeBudget, ComputeBudgetPlanner
from .engine import FrontierPlan, analyze_prompt
from .semantics.relation_graph import SceneRelationGraph, SceneRelationParser
from .uncertainty.confidence_gate import ConfidenceGate, UncertaintyReport


@dataclass
class DeepFrontierPlan:
    base: FrontierPlan
    plausibility: List[PlausibilityFlag] = field(default_factory=list)
    uncertainty: Optional[UncertaintyReport] = None
    relations: Optional[SceneRelationGraph] = None
    token_emphasis: Optional[TokenEmphasisMap] = None
    compute: Optional[ComputeBudget] = None
    deep_augmented_prompt: str = ""


def analyze_deep(
    prompt: str,
    *,
    num_steps: int = 28,
    serendipity_dial: float = 0.25,
    layout_regions: int = 0,
    auto_resolve_contradictions: bool = False,
) -> DeepFrontierPlan:
    base = analyze_prompt(
        prompt,
        num_steps=num_steps,
        serendipity_dial=serendipity_dial,
        auto_resolve_contradictions=auto_resolve_contradictions,
    )

    phys = PhysicalPlausibilityScanner()
    flags = phys.scan(base.augmented_prompt or prompt)

    gate = ConfidenceGate()
    uncertainty = gate.analyze(prompt, contradiction_count=len(base.contradictions))

    relations = SceneRelationParser().parse(prompt)

    emphasis = TokenEmphasisPlanner().plan(base.augmented_prompt or prompt)

    compute = ComputeBudgetPlanner(num_steps=num_steps).plan(
        risk_score=max(base.risk_score, uncertainty.score),
        layout_regions=layout_regions,
    )

    augmented = base.augmented_prompt or prompt
    if flags:
        augmented = phys.augment_prompt(augmented, flags, max_add=2)
    augmented = TokenEmphasisPlanner().augment_with_weights(augmented, emphasis)

    return DeepFrontierPlan(
        base=base,
        plausibility=flags,
        uncertainty=uncertainty,
        relations=relations,
        token_emphasis=emphasis,
        compute=compute,
        deep_augmented_prompt=augmented,
    )


def deep_sample_kwargs(plan: DeepFrontierPlan, *, base_negative: str = "") -> Dict[str, Any]:
    from .engine import FrontierEngine

    eng = FrontierEngine()
    kw = eng.sample_kwargs(plan.base, base_negative=base_negative)
    kw["prompt"] = plan.deep_augmented_prompt or kw.get("prompt", "")

    if plan.uncertainty:
        mult = kw.get("cfg_scale_multiplier") or 1.0
        kw["cfg_scale_multiplier"] = mult * (1.0 + plan.uncertainty.cfg_boost)
        kw["frontier_recommend_best_of_n"] = plan.uncertainty.recommend_best_of_n

    if plan.token_emphasis:
        kw["frontier_token_emphasis"] = plan.token_emphasis.as_prompt_weights()
        kw["frontier_cfg_emphasis_mult"] = plan.token_emphasis.cfg_multiplier

    if plan.compute:
        kw["frontier_guidance_tiers"] = [t.value for t in plan.compute.tiers]
        kw["frontier_compute_cost"] = plan.compute.estimated_cost

    if plan.relations and plan.relations.edges:
        kw["frontier_relation_hints"] = plan.relations.to_regional_hints()

    if plan.plausibility:
        phys = PhysicalPlausibilityScanner()
        extra_neg = phys.negative_suffix(plan.plausibility)
        neg = kw.get("negative_prompt") or ""
        kw["negative_prompt"] = f"{neg}, {extra_neg}" if neg else extra_neg

    return kw


__all__ = ["DeepFrontierPlan", "analyze_deep", "deep_sample_kwargs"]

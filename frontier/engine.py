"""
Frontier engine: compose logic, narrative, chaos, and memory into one plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .chaos.entropy_budget import EntropyBudget, EntropyBudgetAllocator
from .chaos.serendipity import SerendipityCurve, SerendipityInjector
from .logic.absence import AbsenceConstraint, AbsenceExtractor
from .logic.contradiction import Contradiction, ContradictionScanner
from .memory.generation_echo import GenerationEchoMemory
from .narrative.moment import MomentCue, TemporalMomentAnalyzer
from .narrative.witness import WitnessFrame, WitnessPerspectiveAnalyzer


@dataclass
class FrontierPlan:
    """Full outside-the-box analysis for one prompt."""

    prompt: str
    contradictions: List[Contradiction] = field(default_factory=list)
    absence: List[AbsenceConstraint] = field(default_factory=list)
    witness: Optional[WitnessFrame] = None
    moment: Optional[MomentCue] = None
    serendipity: Optional[SerendipityCurve] = None
    entropy: Optional[EntropyBudget] = None
    echo_negative: str = ""
    augmented_prompt: str = ""
    risk_score: float = 0.0


def analyze_prompt(
    prompt: str,
    *,
    num_steps: int = 28,
    serendipity_dial: float = 0.25,
    entropy_total: float = 1.0,
    echo_memory: Optional[GenerationEchoMemory] = None,
    auto_resolve_contradictions: bool = False,
) -> FrontierPlan:
    """Run all frontier analyzers and return a unified plan."""
    scanner = ContradictionScanner()
    absence_ext = AbsenceExtractor()
    witness_an = WitnessPerspectiveAnalyzer()
    moment_an = TemporalMomentAnalyzer(num_steps=num_steps)
    serendipity_inj = SerendipityInjector(num_steps=num_steps)
    entropy_alloc = EntropyBudgetAllocator(num_steps=num_steps)

    contradictions = scanner.scan(prompt)
    absence = absence_ext.extract(prompt)
    witness = witness_an.analyze(prompt)
    moment = moment_an.analyze(prompt)
    serendipity = serendipity_inj.curve(serendipity_dial)
    entropy = entropy_alloc.allocate(entropy_total)

    working_prompt = prompt
    if auto_resolve_contradictions and contradictions:
        working_prompt = scanner.suggest_rewrite(prompt)

    augmented = working_prompt
    frags: List[str] = []
    frags.extend(witness.prompt_fragments)
    frags.extend(moment.prompt_fragments)
    if frags:
        augmented = f"{working_prompt}, {', '.join(frags)}"

    echo_neg = ""
    if echo_memory is not None:
        echo_neg = echo_memory.negative_suffix(prompt)

    risk = scanner.max_severity(prompt)
    if absence:
        risk = min(1.0, risk + 0.05 * len(absence))
    if witness.confidence > 0.5:
        risk = min(1.0, risk + 0.05)

    return FrontierPlan(
        prompt=prompt,
        contradictions=contradictions,
        absence=absence,
        witness=witness,
        moment=moment,
        serendipity=serendipity,
        entropy=entropy,
        echo_negative=echo_neg,
        augmented_prompt=augmented,
        risk_score=risk,
    )


class FrontierEngine:
    """Facade for repeated frontier analysis + sample.py kwargs export."""

    def __init__(
        self,
        num_steps: int = 28,
        serendipity_dial: float = 0.25,
        echo_memory: Optional[GenerationEchoMemory] = None,
    ) -> None:
        self.num_steps = num_steps
        self.serendipity_dial = serendipity_dial
        self.echo_memory = echo_memory or GenerationEchoMemory()
        self._absence = AbsenceExtractor()

    def analyze(self, prompt: str, **kwargs: Any) -> FrontierPlan:
        return analyze_prompt(
            prompt,
            num_steps=kwargs.get("num_steps", self.num_steps),
            serendipity_dial=kwargs.get("serendipity_dial", self.serendipity_dial),
            echo_memory=self.echo_memory,
            **{k: v for k, v in kwargs.items() if k in ("entropy_total", "auto_resolve_contradictions")},
        )

    def sample_kwargs(
        self,
        plan: FrontierPlan,
        *,
        base_negative: str = "",
    ) -> Dict[str, Any]:
        """Map a plan to kwargs compatible with ``sample.py`` / diffusion hooks."""
        neg = self._absence.merge_negative_prompt(base_negative, plan.absence)
        if plan.echo_negative:
            neg = f"{neg}, {plan.echo_negative}" if neg else plan.echo_negative

        cfg_scale = None
        if plan.witness and plan.witness.confidence > 0:
            cfg_scale = plan.witness.cfg_bias

        return {
            "prompt": plan.augmented_prompt or plan.prompt,
            "negative_prompt": neg,
            "frontier_risk_score": plan.risk_score,
            "frontier_serendipity_scales": list(plan.serendipity.scales) if plan.serendipity else None,
            "frontier_step_emphasis": list(plan.moment.step_emphasis) if plan.moment else None,
            "frontier_entropy_per_step": list(plan.entropy.per_step) if plan.entropy else None,
            "cfg_scale_multiplier": cfg_scale,
            "frontier_contradictions": [
                {"left": c.left, "right": c.right, "severity": c.severity, "category": c.category}
                for c in plan.contradictions
            ],
        }

    def record_failure(self, prompt: str, tags: List[str], **kwargs: Any) -> None:
        self.echo_memory.record_failure(prompt, tags, **kwargs)


__all__ = ["FrontierEngine", "FrontierPlan", "analyze_prompt"]

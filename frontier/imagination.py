"""
Imagination engine — creative frontier without duplicating the prompt stack.

Produces prompt fragments AND diffusion knobs (serendipity, CFG, step curves).
Use ``--frontier-creative`` instead of stacking more art-medium tags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config.defaults.art_mediums import merge_csv_unique

from .archetype import SymbolMapEngine
from .cinema import ShotGrammar
from .collective import CrowdGrammar
from .constraint import CreativeConstraintEngine
from .focal import FocalStoryteller
from .fusion import GenreCollisionEngine
from .glitch import GlitchPlanner
from .mutation import PromptMutator
from .paradox import ParadoxKeeper
from .rhythm import RhythmPlanner
from .scale import MagnitudePlanner
from .surreal import DreamLogicPlanner
from .synesthesia import SynesthesiaEngine
from .vibe import MoodPhysics
from .weathering import PatinaStoryteller


@dataclass
class ImaginationPlan:
    prompt: str
    augmented_prompt: str = ""
    merged_positive: str = ""
    merged_negative: str = ""
    serendipity_dial: float = 0.25
    cfg_multiplier: float = 1.0
    step_emphasis: Optional[Tuple[float, ...]] = None
    suppress_contradiction_resolve: bool = False
    mutations: List[str] = field(default_factory=list)
    creative_trace: List[str] = field(default_factory=list)


def analyze_imagination(
    prompt: str,
    *,
    num_steps: int = 28,
    base_serendipity: float = 0.25,
    mutate_count: int = 0,
    mutate_seed: int = 0,
    random_constraint_seed: int | None = None,
) -> ImaginationPlan:
    """Run creative-only analyzers — no art_mediums / shortcomings overlap."""
    trace: List[str] = []
    pos_parts: List[str] = []
    neg_parts: List[str] = []

    dream_pos, dream_neg, dream_ser = DreamLogicPlanner().fragments(prompt)
    if dream_pos:
        pos_parts.append(dream_pos)
        trace.append("surreal")
    if dream_neg:
        neg_parts.append(dream_neg)

    paradox_pos, paradox_neg, suppress = ParadoxKeeper().fragments(prompt)
    if paradox_pos:
        pos_parts.append(paradox_pos)
        trace.append("paradox")
    if paradox_neg:
        neg_parts.append(paradox_neg)

    constraint = CreativeConstraintEngine().pack(prompt)
    if constraint.positive:
        pos_parts.append(constraint.positive)
        trace.append("constraint")
    elif random_constraint_seed is not None:
        rnd = CreativeConstraintEngine().suggest_random(seed=random_constraint_seed)
        if rnd.positive:
            pos_parts.append(rnd.positive)
            neg_parts.append(rnd.negative)
            trace.append(f"constraint_random:{rnd.rules[0].value}")

    if constraint.negative:
        neg_parts.append(constraint.negative)

    syn = SynesthesiaEngine().map_prompt(prompt)
    ser_off, syn_cfg = SynesthesiaEngine().diffusion_knobs(prompt)
    if syn.merged_color:
        pos_parts.append(syn.merged_color)
        trace.append("synesthesia")
    if syn.merged_rhythm:
        pos_parts.append(syn.merged_rhythm)

    cinema_pos, cinema_neg = ShotGrammar().fragments(prompt)
    if cinema_pos:
        pos_parts.append(cinema_pos)
        trace.append("cinema")
    if cinema_neg:
        neg_parts.append(cinema_neg)

    mood = MoodPhysics().analyze(prompt)
    if mood.prompt_fragment:
        pos_parts.append(mood.prompt_fragment)
        trace.append("mood_physics")

    pat_pos, pat_neg = PatinaStoryteller().fragments(prompt)
    if pat_pos:
        pos_parts.append(pat_pos)
        trace.append("weathering")
    if pat_neg:
        neg_parts.append(pat_neg)

    glitch_pos, glitch_neg, glitch_ser = GlitchPlanner().fragments(prompt)
    if glitch_pos:
        pos_parts.append(glitch_pos)
        trace.append("glitch")
    if glitch_neg:
        neg_parts.append(glitch_neg)

    fusion_pos, fusion_neg = GenreCollisionEngine().fragments(prompt)
    if fusion_pos:
        pos_parts.append(fusion_pos)
        trace.append("fusion")
    if fusion_neg:
        neg_parts.append(fusion_neg)

    arch_pos, arch_neg = SymbolMapEngine().fragments(prompt)
    if arch_pos:
        pos_parts.append(arch_pos)
        trace.append("archetype")
    if arch_neg:
        neg_parts.append(arch_neg)

    rhythm_pos, rhythm_neg = RhythmPlanner().fragments(prompt)
    if rhythm_pos:
        pos_parts.append(rhythm_pos)
        trace.append("rhythm")
    if rhythm_neg:
        neg_parts.append(rhythm_neg)

    focal_cfg = 1.0
    focal_pos, focal_neg, focal_cfg = FocalStoryteller().fragments(prompt)
    if focal_pos:
        pos_parts.append(focal_pos)
        trace.append("focal")
    if focal_neg:
        neg_parts.append(focal_neg)

    crowd_pos, crowd_neg = CrowdGrammar().fragments(prompt)
    if crowd_pos:
        pos_parts.append(crowd_pos)
        trace.append("collective")
    if crowd_neg:
        neg_parts.append(crowd_neg)

    scale_pos, scale_neg = MagnitudePlanner().fragments(prompt)
    if scale_pos:
        pos_parts.append(scale_pos)
        trace.append("scale")
    if scale_neg:
        neg_parts.append(scale_neg)

    merged_pos = merge_csv_unique(*pos_parts)
    merged_neg = merge_csv_unique(*neg_parts)

    ser = float(base_serendipity) + dream_ser + glitch_ser + ser_off + mood.serendipity_offset
    ser = max(0.0, min(0.85, ser))

    cfg = mood.cfg_mult * syn_cfg * focal_cfg

    step_curve = MoodPhysics().step_emphasis_curve(mood, num_steps) if mood.vectors else None

    augmented = prompt
    if merged_pos and prompt and len(merged_pos) < 350:
        augmented = f"{prompt}, {merged_pos}"

    mutations: List[str] = []
    if mutate_count > 0:
        muts = PromptMutator().mutate_batch(prompt, seed=mutate_seed, count=mutate_count)
        mutations = [m.mutated for m in muts if m.mutated.strip()]
        trace.append(f"mutations:{len(mutations)}")

    return ImaginationPlan(
        prompt=prompt,
        augmented_prompt=augmented,
        merged_positive=merged_pos,
        merged_negative=merged_neg,
        serendipity_dial=ser,
        cfg_multiplier=cfg,
        step_emphasis=step_curve,
        suppress_contradiction_resolve=suppress,
        mutations=mutations,
        creative_trace=trace,
    )


def imagination_sample_kwargs(plan: ImaginationPlan, *, base_negative: str = "") -> Dict[str, Any]:
    neg = merge_csv_unique(base_negative, plan.merged_negative)
    kw: Dict[str, Any] = {
        "prompt": plan.augmented_prompt or plan.prompt,
        "negative_prompt": neg,
        "cfg_scale_multiplier": plan.cfg_multiplier,
        "frontier_serendipity_dial": plan.serendipity_dial,
        "frontier_creative": True,
        "creative_trace": list(plan.creative_trace),
        "suppress_contradiction_resolve": plan.suppress_contradiction_resolve,
    }
    if plan.step_emphasis:
        kw["frontier_step_emphasis"] = list(plan.step_emphasis)
    if plan.mutations:
        kw["creative_prompt_mutations"] = list(plan.mutations)
    return kw


__all__ = ["ImaginationPlan", "analyze_imagination", "imagination_sample_kwargs"]

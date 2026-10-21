"""Deep frontier synthesis: the base frontier plan plus plausibility, uncertainty, relations, emphasis and compute.

Also holds what used to be: adherence, causality, economy, semantics, uncertainty.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .engine import FrontierEngine, FrontierPlan, analyze_prompt

# ==================================================================================================
# adherence: Which prompt tokens deserve extra CFG / prompt weight: hard words such as text, hands, logos and counts.
# ==================================================================================================


@dataclass(frozen=True)
class TokenWeight:
    token: str
    weight: float  # 1.0 = default; >1 = emphasize
    reason: str


@dataclass(frozen=True)
class TokenEmphasisMap:
    weights: tuple[TokenWeight, ...]
    cfg_multiplier: float

    def as_prompt_weights(self) -> dict[str, float]:
        return {w.token: w.weight for w in self.weights}


_HARD_TOKENS: tuple[tuple[str, float, str], ...] = (
    (r"\btext\b|\btypography\b|\blogo\b|\bsign\b", 1.35, "text rendering"),
    (r"\bhands?\b|\bfingers?\b", 1.25, "anatomy"),
    (r"\beyes?\b|\bpupils?\b", 1.15, "face detail"),
    (r"\breflection\b|\bmirror\b", 1.2, "symmetry"),
    (r"\bthree\b|\bfour\b|\bfive\b|\btwo\b", 1.3, "counting"),
    (r"\bexactly\b|\bprecisely\b", 1.2, "precision"),
)


class TokenEmphasisPlanner:
    def plan(self, prompt: str) -> TokenEmphasisMap:
        text = prompt or ""
        weights: list[TokenWeight] = []
        boost = 0.0
        for pattern, w, reason in _HARD_TOKENS:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                weights.append(TokenWeight(token=m.group(0), weight=w, reason=reason))
                boost = max(boost, (w - 1.0) * 0.5)
        return TokenEmphasisMap(weights=tuple(weights), cfg_multiplier=1.0 + boost)

    def augment_with_weights(self, prompt: str, plan: TokenEmphasisMap) -> str:
        """SD-style (word:1.2) fragments for supported pipelines."""
        weighted = [f"({w.token}:{w.weight:.2f})" for w in plan.weights[:4] if w.weight > 1.05]
        return ", ".join(p for p in (prompt, *weighted) if p)


# ==================================================================================================
# causality: If the prompt names a cause (rain, fire), flag the missing implied effect (wet, smoke).
#
# Training-free guardrail: augment the prompt / negative prompt before wasting steps.
# ==================================================================================================


@dataclass(frozen=True)
class PlausibilityFlag:
    trigger: str
    missing: str
    severity: float  # 0..1
    category: str
    fix_hint: str


@dataclass(frozen=True)
class EffectImplication:
    """When ``cause`` appears, ``effect`` should appear (soft expectation)."""

    cause_pattern: str
    effect_pattern: str
    category: str
    fix_hint: str


_IMPLICATIONS: tuple[EffectImplication, ...] = (
    EffectImplication(
        r"\brain\b|\bstorm\b|\bdownpour\b",
        r"\bwet\b|\bpuddle\b|\breflect(?:ion|ive)\b|\bglisten",
        "weather",
        "add wet surfaces or reflections",
    ),
    EffectImplication(
        r"\bsnow\b|\bblizzard\b",
        r"\bfrost\b|\bicy\b|\bcold\b|\bwhite\s+cover",
        "weather",
        "add frost, ice, or snow accumulation",
    ),
    EffectImplication(
        r"\bfire\b|\bflames\b|\bburning\b",
        r"\bsmoke\b|\bsoot\b|\bchar\b|\bglow\b|\bheat\s+haze",
        "physics",
        "add smoke, glow, or heat distortion",
    ),
    EffectImplication(
        r"\bunderwater\b|\bsubmerged\b",
        r"\bbubble\b|\bcaustic\b|\brefraction\b|\bfloating\s+hair",
        "medium",
        "add underwater cues (caustics, bubbles, hair float)",
    ),
    EffectImplication(
        r"\bnight\b|\bmidnight\b",
        r"\bdark\b|\bshadow\b|\bartificial\s+light\b|\bstreetlamp\b|\bmoon",
        "lighting",
        "add night lighting sources or deep shadows",
    ),
    EffectImplication(
        r"\brunning\b|\bsprint\b|\bmotion\s+blur\b",
        r"\bblur\b|\bdynamic\b|\bwind\b|\bstrain\b",
        "motion",
        "add motion blur or dynamic pose cues",
    ),
    EffectImplication(
        r"\bbroken\b|\bshattered\b|\bcrack",
        r"\bdebris\b|\bfragment\b|\bjagged\b",
        "damage",
        "add debris or jagged edges",
    ),
)

_GENERIC_NEGATIVES = ("physically impossible", "floating without support", "wrong lighting for weather")


class PhysicalPlausibilityScanner:
    """Scan prompts for missing implied effects."""

    def __init__(self, rules: Sequence[EffectImplication] | None = None) -> None:
        self.rules = tuple(rules or _IMPLICATIONS)

    def scan(self, prompt: str) -> list[PlausibilityFlag]:
        text = (prompt or "").strip()
        flags: list[PlausibilityFlag] = []
        for rule in self.rules:
            cause = re.search(rule.cause_pattern, text, flags=re.IGNORECASE)
            if cause is None or re.search(rule.effect_pattern, text, flags=re.IGNORECASE):
                continue
            flags.append(
                PlausibilityFlag(
                    trigger=cause.group(0),
                    missing=rule.effect_pattern,
                    severity=0.55,
                    category=rule.category,
                    fix_hint=rule.fix_hint,
                )
            )
        return flags

    def augment_prompt(self, prompt: str, flags: Sequence[PlausibilityFlag], max_add: int = 2) -> str:
        """Append cheap fix hints as comma-separated fragments."""
        if not flags:
            return prompt
        hints = [f.fix_hint for f in flags[:max_add]]
        return f"{prompt}, {', '.join(hints)}" if prompt else ", ".join(hints)

    def negative_suffix(self, flags: Sequence[PlausibilityFlag]) -> str:
        """One generic physics negative per flag (at most three)."""
        return ", ".join(_GENERIC_NEGATIVES[: len(flags)])


# ==================================================================================================
# economy: Adaptive compute allocation across denoise steps.
#
# Marks each step as skip / lite / full / heavy guidance so a sampler could spend regional, attention and
# extra CFG passes only where they matter. The tiers are exported as ``frontier_guidance_tiers``; no sampler
# acts on them yet.
# ==================================================================================================


class GuidanceTier(str, Enum):
    SKIP = "skip"  # no CFG, no regional: trust momentum
    LITE = "lite"  # single CFG pass
    FULL = "full"  # regional + attention + batched CFG
    HEAVY = "heavy"  # best-of-N micro-rerank at this step


# Relative cost of one step at each tier (FULL = 1).
_COST = {GuidanceTier.SKIP: 0.0, GuidanceTier.LITE: 0.35, GuidanceTier.FULL: 1.0, GuidanceTier.HEAVY: 1.8}


@dataclass(frozen=True)
class ComputeBudget:
    num_steps: int
    tiers: tuple[GuidanceTier, ...]
    estimated_cost: float  # relative to an all-FULL schedule


class ComputeBudgetPlanner:
    """
    Default policy: FULL early while the layout locks (when there are regions), LITE once structure is
    committed, SKIP for late texture steps on low-risk prompts, and one HEAVY step at ~65% progress for
    composition rescue on risky prompts.
    """

    def __init__(self, num_steps: int = 28) -> None:
        self.num_steps = max(4, int(num_steps))

    def plan(
        self,
        *,
        risk_score: float = 0.0,
        layout_regions: int = 0,
        heavy_at_progress: float = 0.65,
    ) -> ComputeBudget:
        n = self.num_steps
        heavy_idx = int(heavy_at_progress * (n - 1))
        tiers: list[GuidanceTier] = []
        for i in range(n):
            p = i / (n - 1)
            if layout_regions > 0 and p < 0.35:
                tiers.append(GuidanceTier.FULL)
            elif i == heavy_idx and risk_score > 0.35:
                tiers.append(GuidanceTier.HEAVY)
            elif p < 0.75:
                tiers.append(GuidanceTier.LITE)
            else:
                tiers.append(GuidanceTier.SKIP if risk_score < 0.4 else GuidanceTier.LITE)
        return ComputeBudget(num_steps=n, tiers=tuple(tiers), estimated_cost=sum(_COST[t] for t in tiers) / n)


# ==================================================================================================
# semantics: Subject-relation-object graph parsed from prompts, for layout and regional prompting hints.
# ==================================================================================================


class RelationKind(str, Enum):
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    HOLDING = "holding"
    WEARING = "wearing"
    INSIDE = "inside"
    NEAR = "near"


@dataclass(frozen=True)
class RelationEdge:
    subject: str
    kind: RelationKind
    object: str
    confidence: float


@dataclass
class SceneRelationGraph:
    entities: list[str] = field(default_factory=list)
    edges: list[RelationEdge] = field(default_factory=list)

    def to_regional_hints(self) -> list[tuple[str, str]]:
        """Map spatial relations to (entity, anchor) for Omost-style layout."""
        hints: list[tuple[str, str]] = []
        for e in self.edges:
            if e.kind == RelationKind.LEFT_OF:
                hints += [(e.subject, "left"), (e.object, "right")]
            elif e.kind == RelationKind.RIGHT_OF:
                hints += [(e.subject, "right"), (e.object, "left")]
            elif e.kind == RelationKind.ABOVE:
                hints.append((e.subject, "top"))
            elif e.kind == RelationKind.BELOW:
                hints.append((e.subject, "bottom"))
        return hints


_PHRASE = r"(\w+(?:\s+\w+)?)"
_REL_PATTERNS: tuple[tuple[re.Pattern, RelationKind], ...] = tuple(
    (re.compile(rf"{_PHRASE}\s+{rel}\s+{_PHRASE}", re.I), kind)
    for rel, kind in (
        ("on the left of", RelationKind.LEFT_OF),
        ("to the left of", RelationKind.LEFT_OF),
        ("on the right of", RelationKind.RIGHT_OF),
        (r"holding(?:\s+a)?", RelationKind.HOLDING),
        (r"wearing(?:\s+a)?", RelationKind.WEARING),
        (r"inside(?:\s+a)?", RelationKind.INSIDE),
    )
)


class SceneRelationParser:
    def parse(self, prompt: str) -> SceneRelationGraph:
        text = (prompt or "").strip()
        graph = SceneRelationGraph()
        for pattern, kind in _REL_PATTERNS:
            for m in pattern.finditer(text):
                sub, obj = m.group(1).strip(), m.group(2).strip()
                graph.edges.append(RelationEdge(sub, kind, obj, confidence=0.7))
                for ent in (sub, obj):
                    if ent not in graph.entities:
                        graph.entities.append(ent)
        return graph


# ==================================================================================================
# uncertainty: Uncertainty-aware generation: score prompt ambiguity before sampling and escalate CFG / best-of-N.
# ==================================================================================================


class UncertaintySignal(str, Enum):
    VAGUE_QUANTITY = "vague_quantity"
    CONFLICTING_STYLE = "conflicting_style"
    UNGROUNDED_REFERENCE = "ungrounded_reference"
    ORPHAN_PRONOUN = "orphan_pronoun"
    OVERLOADED_SCENE = "overloaded_scene"


@dataclass(frozen=True)
class UncertaintyReport:
    score: float  # 0..1
    signals: tuple[UncertaintySignal, ...]
    clarification_questions: tuple[str, ...]
    cfg_boost: float = 0.0
    recommend_best_of_n: int = 1


class ConfidenceGate:
    """Heuristic uncertainty before sampling."""

    _VAGUE = re.compile(r"\bsome\b|\bfew\b|\bseveral\b|\bstuff\b|\bthings\b", re.I)
    _PRONOUN = re.compile(r"\b(it|they|them|that one|the other)\b", re.I)
    _STYLE_MIX = re.compile(
        r"(photoreal|hyperreal|3d render).*(cartoon|anime|illustration)|"
        r"(cartoon|anime).*(photoreal|hyperreal)",
        re.I | re.S,
    )

    def analyze(self, prompt: str, *, contradiction_count: int = 0) -> UncertaintyReport:
        text = (prompt or "").strip()
        signals: list[UncertaintySignal] = []
        questions: list[str] = []

        if self._VAGUE.search(text):
            signals.append(UncertaintySignal.VAGUE_QUANTITY)
            questions.append("How many subjects, and how large in frame?")

        if self._PRONOUN.search(text) and len(text.split()) < 25:
            signals.append(UncertaintySignal.ORPHAN_PRONOUN)
            questions.append("What does 'it/they' refer to?")

        if self._STYLE_MIX.search(text):
            signals.append(UncertaintySignal.CONFLICTING_STYLE)
            questions.append("Pick one render style: photo, illustration, or 3D?")

        clauses = [c.strip() for c in text.split(",") if c.strip()]
        if len(clauses) >= 7:
            signals.append(UncertaintySignal.OVERLOADED_SCENE)
            questions.append("Which 2–3 elements matter most?")

        if contradiction_count > 0:
            signals.append(UncertaintySignal.CONFLICTING_STYLE)
            questions.append("Resolve contradictory lighting/time/space cues.")

        score = min(1.0, 0.12 * len(signals) + 0.08 * contradiction_count)
        cfg_boost = 0.15 * score
        bon = 1 if score < 0.35 else (2 if score < 0.6 else 4)

        return UncertaintyReport(
            score=score,
            signals=tuple(signals),
            clarification_questions=tuple(questions[:3]),
            cfg_boost=cfg_boost,
            recommend_best_of_n=bon,
        )


# ==================================================================================================
# synthesis
# ==================================================================================================


@dataclass
class DeepFrontierPlan:
    base: FrontierPlan
    plausibility: list[PlausibilityFlag] = field(default_factory=list)
    uncertainty: UncertaintyReport | None = None
    relations: SceneRelationGraph | None = None
    token_emphasis: TokenEmphasisMap | None = None
    compute: ComputeBudget | None = None
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
    working = base.augmented_prompt or prompt
    phys = PhysicalPlausibilityScanner()
    flags = phys.scan(working)
    uncertainty = ConfidenceGate().analyze(prompt, contradiction_count=len(base.contradictions))
    emphasis_planner = TokenEmphasisPlanner()
    emphasis = emphasis_planner.plan(working)
    compute = ComputeBudgetPlanner(num_steps=num_steps).plan(
        risk_score=max(base.risk_score, uncertainty.score),
        layout_regions=layout_regions,
    )
    augmented = phys.augment_prompt(working, flags, max_add=2)
    return DeepFrontierPlan(
        base=base,
        plausibility=flags,
        uncertainty=uncertainty,
        relations=SceneRelationParser().parse(prompt),
        token_emphasis=emphasis,
        compute=compute,
        deep_augmented_prompt=emphasis_planner.augment_with_weights(augmented, emphasis),
    )


def deep_sample_kwargs(plan: DeepFrontierPlan, *, base_negative: str = "") -> dict[str, Any]:
    kw = FrontierEngine().sample_kwargs(plan.base, base_negative=base_negative)
    kw["prompt"] = plan.deep_augmented_prompt or kw["prompt"]

    if plan.uncertainty:
        mult = kw["cfg_scale_multiplier"] or 1.0
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
        extra_neg = PhysicalPlausibilityScanner().negative_suffix(plan.plausibility)
        neg = kw["negative_prompt"]
        kw["negative_prompt"] = f"{neg}, {extra_neg}" if neg else extra_neg

    return kw


__all__ = [
    "TokenEmphasisMap",
    "TokenEmphasisPlanner",
    "TokenWeight",
    "EffectImplication",
    "PhysicalPlausibilityScanner",
    "PlausibilityFlag",
    "ComputeBudget",
    "ComputeBudgetPlanner",
    "GuidanceTier",
    "RelationEdge",
    "RelationKind",
    "SceneRelationGraph",
    "SceneRelationParser",
    "ConfidenceGate",
    "UncertaintyReport",
    "UncertaintySignal",
    "DeepFrontierPlan",
    "analyze_deep",
    "deep_sample_kwargs",
]

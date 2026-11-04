"""Prompt-side frontier planners.

- Deep synthesis (``analyze_deep``): the base frontier plan plus plausibility, uncertainty, relations,
  emphasis and compute.
- Subject (``analyze_subject``): bodies, creatures, mature content, mediums, brushwork, realism.
- Perfect (``analyze_perfect``, ``--frontier-perfect``): deep + subject + scene-quality analyzers behind
  the content policy (the minor block in ``utils.safety.minor_guard`` always runs first).
- Video bridge: frontier image analyzers -> the video frontier compiler.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from config.prompts import merge_csv_unique
from config.styles import guidance_fragments
from utils.prompt.art_direction import ArtDirection, plan_art_direction
from utils.safety import minor_guard

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


# ==================================================================================================
# anatomy: Body-mode planner — human realism vs intentional fantasy vs creature anatomy.
#
# Models confuse *broken human* anatomy with *designed non-human* forms. This router
# picks the right positive/negative pack so dragons are not penalized for extra limbs.
# ==================================================================================================


class BodyMode(str, Enum):
    REALISTIC_HUMAN = "realistic_human"
    STYLIZED_HUMAN = "stylized_human"  # anime, cartoon people
    CREATURE = "creature"  # single-species monster/animal
    CHIMERA = "chimera"  # mixed anatomy by design
    HORROR_BODY = "horror_body"  # intentional wrongness
    MECHA = "mecha"  # robots, suits, hard-surface figures
    ABSTRACT = "abstract"  # no body guidance


@dataclass(frozen=True)
class AnatomyRisk:
    score: float  # 0..1 likelihood of accidental bad anatomy
    triggers: tuple[str, ...]
    hand_focus: bool
    full_body: bool


@dataclass(frozen=True)
class BodyPlan:
    mode: BodyMode
    risk: AnatomyRisk
    positive: str
    negative: str
    cfg_bias: float  # multiply base CFG for hard anatomy steps


_HUMAN = re.compile(
    r"\b(person|people|man|woman|girl|boy|portrait|model|dancer|athlete|"
    r"full body|half body|selfie|fashion|boudoir|nude|lingerie)\b",
    re.I,
)
_HANDS = re.compile(r"\bhands?\b|\bfingers?\b|\bholding\b|\bgrip\b", re.I)
_FULL = re.compile(r"\bfull body\b|\bstanding\b|\bwalking\b|\bpose\b", re.I)
_CREATURE = re.compile(
    r"\b(dragon|griffin|wyvern|phoenix|unicorn|demon|devil|angel|serpent|"
    r"wolf|bear|tiger|lion|eagle|hawk|owl|spider|insect|beetle|mantis|"
    r"octopus|squid|jellyfish|shark|whale|dinosaur|raptor|trex|alien|xenomorph)\b",
    re.I,
)
_CHIMERA = re.compile(
    r"\b(chimera|hybrid|half[- ]human|half[- ]animal|centaur|minotaur|mermaid|"
    r"sphinx|lamia|harpy|satyr|faun|werewolf|anthro|furry|kemono)\b",
    re.I,
)
_HORROR = re.compile(
    r"\b(eldritch|body horror|flesh horror|cronenberg|distorted body|"
    r"wrong proportions|nightmare creature|lovecraft)\b",
    re.I,
)
_MECHA = re.compile(
    r"\b(mecha|robot|cyborg|android|gundam|power armor|exosuit|automaton|"
    r"hard surface suit|mechanical body)\b",
    re.I,
)
_STYLIZED = re.compile(
    r"\b(anime|manga|cartoon|chibi|toon|cel shaded|disney|pixar style|"
    r"illustration character|stylized character)\b",
    re.I,
)
_REALISM = re.compile(
    r"\b(photoreal|hyperreal|dslr|raw photo|8k photo|lifelike|documentary)\b",
    re.I,
)

_POS: dict[BodyMode, str] = {
    BodyMode.REALISTIC_HUMAN: (
        "accurate human anatomy, natural skeletal proportions, coherent joint hierarchy, "
        "five fingers per hand, believable weight distribution, natural skin and muscle form"
    ),
    BodyMode.STYLIZED_HUMAN: (
        "consistent stylized proportions, clean silhouette, stable character design grammar, "
        "readable hands with intentional simplification, coherent pose design"
    ),
    BodyMode.CREATURE: (
        "species-consistent anatomy, believable skeletal logic for the creature type, "
        "coherent limb attachment, surface material truth (scales/fur/chitin/feathers)"
    ),
    BodyMode.CHIMERA: (
        "intentional hybrid anatomy with clear transition zones, readable combined silhouette, "
        "consistent material breakup between merged species, designed limb count not accidental"
    ),
    BodyMode.HORROR_BODY: (
        "intentional unsettling anatomy, deliberate asymmetry and distortion, "
        "coherent horror design language, not random glitch artifacts"
    ),
    BodyMode.MECHA: (
        "mechanical joint logic, panel line coherence, plausible actuator placement, "
        "consistent hard-surface scale, readable limb segmentation"
    ),
    BodyMode.ABSTRACT: "",
}

_NEG: dict[BodyMode, str] = {
    BodyMode.REALISTIC_HUMAN: (
        "bad anatomy, extra fingers, fused fingers, broken joints, dislocated limbs, "
        "noodle limbs, floating limbs, asymmetric eyes, plastic skin"
    ),
    BodyMode.STYLIZED_HUMAN: (
        "inconsistent character proportions, melted face, random extra limbs, "
        "broken line art hands, asymmetry drift across the figure"
    ),
    BodyMode.CREATURE: (
        "mammal legs on bird body, random extra heads without design intent, "
        "inconsistent scale between limbs, floating paws, incoherent species mashup"
    ),
    BodyMode.CHIMERA: (
        "accidental human hand on animal limb, seams without transition design, "
        "limb count drift, incoherent fur-to-scale boundary"
    ),
    BodyMode.HORROR_BODY: "accidental AI glitch, jpeg mush, random duplicate features",
    BodyMode.MECHA: (
        "mushy panels, impossible joint bending, greeble noise without structure, "
        "inconsistent mech scale, melted hard-surface"
    ),
    BodyMode.ABSTRACT: "",
}


class BodyPlanner:
    def detect_mode(self, prompt: str) -> BodyMode:
        text = prompt or ""
        if _HORROR.search(text):
            return BodyMode.HORROR_BODY
        if _CHIMERA.search(text):
            return BodyMode.CHIMERA
        if _MECHA.search(text):
            return BodyMode.MECHA
        if _HUMAN.search(text):
            if _REALISM.search(text) and not _STYLIZED.search(text):
                return BodyMode.REALISTIC_HUMAN
            return BodyMode.STYLIZED_HUMAN
        return BodyMode.CREATURE if _CREATURE.search(text) else BodyMode.ABSTRACT

    def assess_risk(self, prompt: str, mode: BodyMode | None = None) -> AnatomyRisk:
        text = prompt or ""
        mode = mode or self.detect_mode(text)
        triggers: list[str] = []
        score = 0.0
        hand_focus = bool(_HANDS.search(text))
        full_body = bool(_FULL.search(text))

        if mode in (BodyMode.REALISTIC_HUMAN, BodyMode.STYLIZED_HUMAN):
            score = 0.35
            if hand_focus:
                score += 0.25
                triggers.append("hands")
            if full_body:
                score += 0.2
                triggers.append("full_body")
            if re.search(r"\bmultiple\b|\bcrowd\b|\bgroup\b", text, re.I):
                score += 0.15
                triggers.append("multi_person")

        if mode == BodyMode.CHIMERA:
            score = 0.45
            triggers.append("hybrid")

        return AnatomyRisk(
            score=min(1.0, score),
            triggers=tuple(triggers),
            hand_focus=hand_focus,
            full_body=full_body,
        )

    def plan(self, prompt: str) -> BodyPlan:
        mode = self.detect_mode(prompt)
        risk = self.assess_risk(prompt, mode)
        cfg = 1.08 if risk.score > 0.5 else 1.0
        if risk.hand_focus and mode == BodyMode.REALISTIC_HUMAN:
            cfg = 1.12
        return BodyPlan(mode=mode, risk=risk, positive=_POS[mode], negative=_NEG[mode], cfg_bias=cfg)


# ==================================================================================================
# creatures: Creature taxonomy — species-aware guidance for monsters, animals, and aliens.
#
# Gives the model a *body plan* instead of human defaults that fight the prompt.
# ==================================================================================================


class CreatureFamily(str, Enum):
    DRACONIC = "draconic"
    AVIAN = "avian"
    FELINE = "feline"
    CANINE = "canine"
    INSECTOID = "insectoid"
    AQUATIC = "aquatic"
    ELDRITCH = "eldritch"
    ALIEN = "alien"
    UNDEAD = "undead"
    GENERIC = "generic"


@dataclass(frozen=True)
class CreaturePlan:
    family: CreatureFamily
    limb_hint: str
    surface_hint: str
    silhouette_hint: str
    negative: str


_FAMILY_RULES: tuple[tuple[re.Pattern, CreatureFamily], ...] = (
    (re.compile(r"\b(dragon|wyvern|drake|wyrm)\b", re.I), CreatureFamily.DRACONIC),
    (re.compile(r"\b(griffin|eagle|hawk|owl|phoenix|harpy|bird)\b", re.I), CreatureFamily.AVIAN),
    (re.compile(r"\b(cat|tiger|lion|panther|feline)\b", re.I), CreatureFamily.FELINE),
    (re.compile(r"\b(wolf|dog|fox|canine|werewolf)\b", re.I), CreatureFamily.CANINE),
    (re.compile(r"\b(insect|beetle|mantis|spider|arachnid|chitin)\b", re.I), CreatureFamily.INSECTOID),
    (re.compile(r"\b(fish|shark|whale|octopus|squid|jellyfish|mermaid)\b", re.I), CreatureFamily.AQUATIC),
    (re.compile(r"\b(eldritch|lovecraft|tentacle horror|void creature)\b", re.I), CreatureFamily.ELDRITCH),
    (re.compile(r"\b(alien|xenomorph|extraterrestrial|ufo creature)\b", re.I), CreatureFamily.ALIEN),
    (re.compile(r"\b(zombie|skeleton|lich|undead|revenant)\b", re.I), CreatureFamily.UNDEAD),
)
_GENERIC_CREATURE = re.compile(r"\b(creature|monster|beast)\b", re.I)


_PLANS: dict[CreatureFamily, CreaturePlan] = {
    CreatureFamily.DRACONIC: CreaturePlan(
        CreatureFamily.DRACONIC,
        "four legs or bipedal haunches with wings anchored at shoulders, long neck, tail counterbalance",
        "overlapping scales, horn and claw keratin, membrane wing translucency",
        "serpentine neck, wedge head, sweeping wing silhouette",
        "bat wings on mammal torso, random extra limbs, bird beak on dragon snout",
    ),
    CreatureFamily.AVIAN: CreaturePlan(
        CreatureFamily.AVIAN,
        "bipedal legs, wings from shoulder girdle, lightweight hollow-bone proportions",
        "feather barb texture, beak keratin, talon curvature",
        "triangular wing profile in flight, compact body",
        "mammal ears on bird, feather fur confusion, wrong wing attachment",
    ),
    CreatureFamily.FELINE: CreaturePlan(
        CreatureFamily.FELINE,
        "quadruped digitigrade legs, flexible spine, retractable claw logic",
        "directional fur flow, whisker pads, wet nose highlight",
        "low prowling stance or arched back read",
        "human eyes on quadruped, dog snout on cat, broken leg joints",
    ),
    CreatureFamily.CANINE: CreaturePlan(
        CreatureFamily.CANINE,
        "quadruped legs, shoulder higher than hips in many breeds, bushy tail option",
        "fur clumping, wet nose, ear cartilage variation",
        "alert stance, readable snout length",
        "cat eyes on dog, broken forelimb symmetry",
    ),
    CreatureFamily.INSECTOID: CreaturePlan(
        CreatureFamily.INSECTOID,
        "six legs on thorax, segmented abdomen, antennae optional, mandible or piercing mouthparts",
        "chitin plates, jointed exoskeleton, compound eye facets",
        "radial leg spread, clear head-thorax-abdomen read",
        "spider with six legs, mammal joints on insect, random leg count",
    ),
    CreatureFamily.AQUATIC: CreaturePlan(
        CreatureFamily.AQUATIC,
        "streamlined body, fin or flipper placement, neutral buoyancy pose",
        "wet specular, subsurface scatter in skin, scale or smooth glide surface",
        "horizontal spine in swim, caudal fin drive",
        "dry fur underwater, legs on fish body, gills in wrong place",
    ),
    CreatureFamily.ELDRITCH: CreaturePlan(
        CreatureFamily.ELDRITCH,
        "intentional impossible geometry with focal anchor, tentacles or extra appendages by design",
        "wet organic membrane, bioluminescent pores, iridescent slime",
        "silhouette dread, asymmetric but composed",
        "random glitch limbs, jpeg tentacles, incoherent copy-paste eyes",
    ),
    CreatureFamily.ALIEN: CreaturePlan(
        CreatureFamily.ALIEN,
        "consistent non-terrestrial joint logic, bilateral or radial symmetry chosen deliberately",
        "non-earth skin: exoskeleton, gelatin, crystalline, or gas membrane",
        "readable alien silhouette, one bold anatomical rule",
        "human with gray skin only, earth mammal copy without design",
    ),
    CreatureFamily.UNDEAD: CreaturePlan(
        CreatureFamily.UNDEAD,
        "bone exposure zones, partial tissue retention, gravity-sagging flesh where appropriate",
        "desiccated skin, marrow hints, torn ligament shadow",
        "collapsed posture or aggressive lurch",
        "plastic skeleton, symmetrical bone noise, living skin on skull only",
    ),
    CreatureFamily.GENERIC: CreaturePlan(
        CreatureFamily.GENERIC,
        "coherent limb count and attachment, species-internal consistency",
        "surface material matches habitat",
        "readable creature silhouette",
        "limb soup, fused anatomy, random duplicate heads",
    ),
}


class CreatureTaxonomy:
    def classify(self, prompt: str) -> CreatureFamily:
        text = prompt or ""
        return next((fam for pat, fam in _FAMILY_RULES if pat.search(text)), CreatureFamily.GENERIC)

    def plan(self, prompt: str) -> CreaturePlan | None:
        """Body plan for the detected family; ``None`` when the prompt names no creature at all."""
        fam = self.classify(prompt)
        if fam == CreatureFamily.GENERIC and not _GENERIC_CREATURE.search(prompt or ""):
            return None
        return _PLANS[fam]

    def prompt_fragments(self, prompt: str) -> tuple[str, str]:
        cp = self.plan(prompt)
        if cp is None:
            return "", ""
        pos = ", ".join(p for p in (cp.limb_hint, cp.surface_hint, cp.silhouette_hint) if p)
        return pos, cp.negative


# ==================================================================================================
# mature: Mature-content quality guidance — anatomical and lighting coherence for adult art.
#
# This module improves *render quality* for artistic nude, boudoir, and explicit prompts.
# It does not filter; every prompt already passed ``utils.safety.minor_guard``, which also
# adds the youth negative to sexual prompts.
# ==================================================================================================


class MatureClass(str, Enum):
    NONE = "none"
    ARTISTIC_NUDE = "artistic_nude"
    BOUDOIR = "boudoir"
    EXPLICIT = "explicit"
    PINUP = "pinup"


@dataclass(frozen=True)
class MaturePlan:
    content_class: MatureClass
    positive: str
    negative: str
    lighting_hint: str
    anatomy_mode: str  # none | lite | strong: suggested anatomy-guidance strength


_NUDE = re.compile(
    r"\b(nude|naked|topless|artistic nude|figure study|life drawing|nsfw)\b",
    re.I,
)
_BOUDOIR = re.compile(r"\b(boudoir|lingerie|sensual|intimate portrait|bedroom)\b", re.I)
_EXPLICIT = re.compile(
    r"\b(explicit|erotic|sex|intercourse|genital|xxx|porn|hentai|ecchi|lewd)\b",
    re.I,
)
_PINUP = re.compile(r"\b(pin[- ]?up|glamour|playboy style|cheesecake)\b", re.I)

_POS_BASE = (
    "natural adult proportions, believable skin texture with subsurface scatter, "
    "coherent anatomy without plastic smoothing, respectful lighting that models form"
)
_NEG_BASE = (
    "plastic doll skin, barbie anatomy, impossible proportions, airbrushed mannequin, "
    "wax figure, asymmetric nipples placement error, floating limbs, censored blur patch"
)
_LIGHT_SOFT = "soft window key with gentle fill, form-revealing but not flat frontal flash"
_LIGHT_DRAMATIC = "motivated rim and bounce light, cinematic falloff, preserved highlight rolloff"
_LIGHT_PINUP = "classic glamour key with controlled specular on skin, clean background separation"
_LIGHTING = {
    MatureClass.ARTISTIC_NUDE: _LIGHT_SOFT,
    MatureClass.BOUDOIR: _LIGHT_DRAMATIC,
    MatureClass.EXPLICIT: _LIGHT_DRAMATIC,
    MatureClass.PINUP: _LIGHT_PINUP,
}
_EXTRA_POS = {
    MatureClass.PINUP: "classic glamour pose readability, confident silhouette",
    MatureClass.BOUDOIR: "intimate but tasteful composition, fabric interaction truth",
}


class MatureGuidance:
    def classify(self, prompt: str) -> MatureClass:
        text = prompt or ""
        if _EXPLICIT.search(text):
            return MatureClass.EXPLICIT
        if _BOUDOIR.search(text):
            return MatureClass.BOUDOIR
        if _PINUP.search(text):
            return MatureClass.PINUP
        if _NUDE.search(text):
            return MatureClass.ARTISTIC_NUDE
        return MatureClass.NONE

    def plan(self, prompt: str) -> MaturePlan:
        cls = self.classify(prompt)
        if cls == MatureClass.NONE:
            return MaturePlan(cls, "", "", "", "none")

        anatomy = "strong" if cls in (MatureClass.EXPLICIT, MatureClass.ARTISTIC_NUDE) else "lite"
        lighting = _LIGHTING[cls]
        pos = ", ".join(p for p in (_POS_BASE, _EXTRA_POS.get(cls, ""), lighting) if p)
        return MaturePlan(cls, pos, _NEG_BASE, lighting, anatomy)

    def fragments(self, prompt: str) -> tuple[str, str]:
        p = self.plan(prompt)
        return p.positive, p.negative


# ==================================================================================================
# medium: Brush-stroke planner — direction, pressure, and edge behavior per medium.
#
# Complements ``config.styles`` with stroke-level vocabulary models understand.
# ==================================================================================================


class StrokeStyle(str, Enum):
    IMPASTO = "impasto"
    WET_ON_WET = "wet_on_wet"
    DRY_BRUSH = "dry_brush"
    HATCHING = "hatching"
    STIPPLING = "stippling"
    SPRAY = "spray"
    PALETTE_KNIFE = "palette_knife"
    CALLIGRAPHIC = "calligraphic"
    FLAT_FILL = "flat_fill"
    SOFT_BLEND = "soft_blend"


@dataclass(frozen=True)
class BrushPlan:
    primary_style: StrokeStyle
    positive: str
    negative: str


_BRUSH_RULES: tuple[tuple[re.Pattern, StrokeStyle, str, str], ...] = (
    (
        re.compile(r"\b(impasto|thick paint|palette knife|alla prima)\b", re.I),
        StrokeStyle.IMPASTO,
        "visible impasto ridges, directional brush load, thick paint peaks catching light",
        "flat digital gradient, airbrush mush",
    ),
    (
        re.compile(r"\b(watercolor|wet[- ]on[- ]wet|wash)\b", re.I),
        StrokeStyle.WET_ON_WET,
        "pigment bloom on paper tooth, controlled backrun edges, transparent glaze layers",
        "opaque plastic fill, random stain noise",
    ),
    (
        re.compile(r"\b(dry brush|scrubbed brush|broken stroke)\b", re.I),
        StrokeStyle.DRY_BRUSH,
        "broken bristle drag marks, canvas weave catching through, scratchy edge energy",
        "uniform smooth blend, textureless stroke",
    ),
    (
        re.compile(r"\b(crosshatch|hatching|engraving|etching|woodcut|linocut)\b", re.I),
        StrokeStyle.HATCHING,
        "consistent hatch angle families, line weight hierarchy, plate pressure variation",
        "random scribble noise, wobbly parallel lines",
    ),
    (
        re.compile(r"\b(stipple|pointillism|dotwork)\b", re.I),
        StrokeStyle.STIPPLING,
        "rhythmic dot clusters, optical color mixing, disciplined dot scale",
        "uniform noise spray, muddy dot soup",
    ),
    (
        re.compile(r"\b(spray paint|street art|graffiti|aerosol)\b", re.I),
        StrokeStyle.SPRAY,
        "soft overspray falloff, cap pressure variation, layered stencil edges",
        "gaussian blur smear, incoherent drip physics",
    ),
    (
        re.compile(r"\b(calligraphy|brush lettering|sumi)\b", re.I),
        StrokeStyle.CALLIGRAPHIC,
        "pressure-thick to thin stroke taper, intentional ink pooling at turns",
        "uniform vector stroke, wobbly sans-serif",
    ),
    (
        re.compile(r"\b(cel shading|flat color|vector)\b", re.I),
        StrokeStyle.FLAT_FILL,
        "decisive flat fills, clean shape breaks, minimal accidental gradient",
        "unintentional photoreal gradient leak",
    ),
    (
        re.compile(r"\b(airbrush|soft blend|painterly blend)\b", re.I),
        StrokeStyle.SOFT_BLEND,
        "controlled soft transitions, edge hierarchy preserved, no muddy midtone soup",
        "banded gradient, plastic skin airbrush",
    ),
)


class BrushPlanner:
    def plan(self, prompt: str) -> BrushPlan | None:
        text = prompt or ""
        for pat, style, pos, neg in _BRUSH_RULES:
            if pat.search(text):
                return BrushPlan(primary_style=style, positive=pos, negative=neg)
        if re.search(r"\b(oil|acrylic|gouache|pastel|charcoal|ink)\b", text, re.I):
            return BrushPlan(
                StrokeStyle.SOFT_BLEND,
                "intentional hand-made stroke variation, edge variety, medium-true surface response",
                "ai soup smear, texture spam, over-blended plastic",
            )
        return None

    def fragments(self, prompt: str) -> tuple[str, str]:
        p = self.plan(prompt)
        if p is None:
            return "", ""
        return p.positive, p.negative


# ==================================================================================================
# realism: Photorealism stack and anti-AI-slop guidance.
# ==================================================================================================


class RealismTier(str, Enum):
    NONE = "none"
    STANDARD = "standard"
    HYPERREAL = "hyperreal"
    DOCUMENTARY = "documentary"


@dataclass(frozen=True)
class AntiSlopPlan:
    tier: RealismTier
    positive: str
    negative: str
    microdetail_hint: str


_REALISM_TRIGGERS = re.compile(
    r"\b(photoreal|hyperreal|lifelike|dslr|raw photo|8k photo|ultra realistic|"
    r"real life|documentary photo|cinematic photo|portrait photo)\b",
    re.I,
)
_HYPER = re.compile(r"\b(hyperreal|8k|ultra realistic|extreme detail)\b", re.I)
_DOC = re.compile(r"\b(documentary|photojournalism|candid|street photo)\b", re.I)

_NEG_COMMON = (
    "ai generated look, plastic skin, waxy pores, over-smoothed skin, beauty filter, "
    "dead glassy eyes, oversharpened irises, symmetry perfection, uncanny valley, "
    "hdr halo, crunchy micro-contrast, ai soup texture, watermark, signature text"
)
_POS_STANDARD = (
    "natural camera response, imperfect skin with pores and freckles, "
    "asymmetric facial micro-features, realistic highlight rolloff, true lens bokeh"
)
_POS_HYPER = (
    "micro-detail without crunch, subsurface skin scatter, individual hair strand variation, "
    "material-true specular, optical depth falloff, filmic tone mapping"
)
_POS_DOC = (
    "authentic candid moment, unposed gesture, environmental context truth, "
    "natural white balance drift, available light only"
)
_POS_BY_TIER = {
    RealismTier.STANDARD: _POS_STANDARD,
    RealismTier.HYPERREAL: f"{_POS_STANDARD}, {_POS_HYPER}",
    RealismTier.DOCUMENTARY: _POS_DOC,
}
_MICRO = "preserve pore-scale detail in focal plane only; avoid global oversharpen"


class AntiSlopScanner:
    def detect_tier(self, prompt: str) -> RealismTier:
        text = prompt or ""
        if not _REALISM_TRIGGERS.search(text):
            return RealismTier.NONE
        if _DOC.search(text):
            return RealismTier.DOCUMENTARY
        if _HYPER.search(text):
            return RealismTier.HYPERREAL
        return RealismTier.STANDARD

    def plan(self, prompt: str) -> AntiSlopPlan:
        tier = self.detect_tier(prompt)
        if tier == RealismTier.NONE:
            return AntiSlopPlan(tier, "", "", "")
        return AntiSlopPlan(tier, _POS_BY_TIER[tier], _NEG_COMMON, _MICRO)

    def fragments(self, prompt: str) -> tuple[str, str]:
        p = self.plan(prompt)
        return p.positive, p.negative


@dataclass(frozen=True)
class PhotorealPlan:
    tier: RealismTier
    positive: str
    negative: str
    lens_hint: str
    sensor_hint: str


# First match wins.
_LENSES: tuple[tuple[re.Pattern, str], ...] = (
    (
        re.compile(r"\b(portrait|85mm|50mm|headshot|bokeh)\b", re.I),
        "portrait lens compression, shallow depth of field, creamy bokeh discs",
    ),
    (
        re.compile(r"\b(wide angle|24mm|16mm|landscape photo|architecture)\b", re.I),
        "wide-angle perspective, controlled edge distortion, deep focus hyperfocal",
    ),
    (
        re.compile(r"\b(macro|100mm macro|close-up|insect photo)\b", re.I),
        "macro shallow depth, diffraction softening at extreme close focus",
    ),
    (
        re.compile(r"\b(telephoto|200mm|sports|wildlife)\b", re.I),
        "telephoto compression, subject isolation, motion-aware shutter",
    ),
)
_SENSOR = "full-frame dynamic range"


class PhotorealStack:
    def __init__(self) -> None:
        self._slop = AntiSlopScanner()

    def plan(self, prompt: str) -> PhotorealPlan:
        slop = self._slop.plan(prompt)
        if slop.tier == RealismTier.NONE:
            return PhotorealPlan(RealismTier.NONE, "", "", "", "")
        # Vignette, CA and sensor noise come from the capture finish; asking the model too doubles them.
        text = prompt or ""
        lens = next((hint for pat, hint in _LENSES if pat.search(text)), "natural lens perspective")
        pos = ", ".join(p for p in (slop.positive, lens, _SENSOR, slop.microdetail_hint) if p)
        return PhotorealPlan(slop.tier, pos, slop.negative, lens, _SENSOR)

    def fragments(self, prompt: str) -> tuple[str, str]:
        p = self.plan(prompt)
        return p.positive, p.negative


# ==================================================================================================
# subject
# ==================================================================================================


@dataclass
class SubjectFrontierPlan:
    prompt: str
    body: BodyPlan | None = None
    creature_positive: str = ""
    creature_negative: str = ""
    mature: MaturePlan | None = None
    brush_positive: str = ""
    brush_negative: str = ""
    medium_positive: str = ""
    medium_negative: str = ""
    photoreal: PhotorealPlan | None = None
    augmented_prompt: str = ""
    merged_positive: str = ""
    merged_negative: str = ""


def analyze_subject(
    prompt: str,
    *,
    include_photography: bool = True,
    medium_mode: str = "auto",
) -> SubjectFrontierPlan:
    """Run body, creature, mature, medium, brush, and photoreal analyzers."""
    body = BodyPlanner().plan(prompt)
    creature_pos, creature_neg = CreatureTaxonomy().prompt_fragments(prompt)
    mature = MatureGuidance().plan(prompt)
    med_pos, med_neg = guidance_fragments(
        prompt,
        medium_mode,  # type: ignore[arg-type]
        include_photography=include_photography,
        anatomy_mode="none",  # the body planner owns anatomy
    )
    brush_pos, brush_neg = BrushPlanner().fragments(prompt)
    photo = PhotorealStack().plan(prompt)

    merged_pos = merge_csv_unique(body.positive, creature_pos, mature.positive, med_pos, brush_pos, photo.positive)
    merged_neg = merge_csv_unique(body.negative, creature_neg, mature.negative, med_neg, brush_neg, photo.negative)
    augmented = f"{prompt}, {merged_pos}" if prompt and merged_pos and len(merged_pos) < 400 else prompt

    return SubjectFrontierPlan(
        prompt=prompt,
        body=body,
        creature_positive=creature_pos,
        creature_negative=creature_neg,
        mature=mature if mature.content_class != MatureClass.NONE else None,
        brush_positive=brush_pos,
        brush_negative=brush_neg,
        medium_positive=med_pos,
        medium_negative=med_neg,
        photoreal=photo if photo.tier != RealismTier.NONE else None,
        augmented_prompt=augmented,
        merged_positive=merged_pos,
        merged_negative=merged_neg,
    )


def subject_sample_kwargs(plan: SubjectFrontierPlan, *, base_negative: str = "") -> dict[str, Any]:
    cfg_mult = 1.0
    if plan.body:
        cfg_mult = max(plan.body.cfg_bias, 1.1) if plan.body.risk.hand_focus else plan.body.cfg_bias
    return {
        "prompt": plan.augmented_prompt or plan.prompt,
        "negative_prompt": merge_csv_unique(base_negative, plan.merged_negative),
        "cfg_scale_multiplier": cfg_mult,
        "subject_body_mode": plan.body.mode.value if plan.body else "abstract",
        "subject_anatomy_risk": plan.body.risk.score if plan.body else 0.0,
        "subject_mature_class": plan.mature.content_class.value if plan.mature else "none",
        "subject_realism_tier": plan.photoreal.tier.value if plan.photoreal else "none",
        "subject_merged_positive": plan.merged_positive,
        "subject_merged_negative": plan.merged_negative,
    }


# ==================================================================================================
# atmosphere: Atmospheric depth — fog, haze, god rays, weather mood.
# ==================================================================================================


class WeatherMood(str, Enum):
    FOG = "fog"
    MIST = "mist"
    RAIN = "rain"
    SNOW = "snow"
    STORM = "storm"
    CLEAR = "clear"
    GOD_RAYS = "god_rays"
    DUST = "dust"


@dataclass(frozen=True)
class AtmospherePlan:
    mood: WeatherMood
    positive: str
    negative: str


_ATMOSPHERE_RULES: tuple[tuple[re.Pattern, WeatherMood, str, str], ...] = (
    (
        re.compile(r"\b(fog|foggy|misty city|pea soup)\b", re.I),
        WeatherMood.FOG,
        "volumetric fog depth falloff, soft aerial perspective, light scatter",
        "uniform gray overlay, no depth falloff",
    ),
    (
        re.compile(r"\b(god rays|crepuscular|light shafts|volumetric light)\b", re.I),
        WeatherMood.GOD_RAYS,
        "crepuscular god rays through particulate, motivated beam direction",
        "random lens flare streaks without volume",
    ),
    (
        re.compile(r"\b(rain|rainy|downpour|wet street)\b", re.I),
        WeatherMood.RAIN,
        "rain streaks, wet surface reflections, micro ripples, moody atmosphere",
        "dry pavement in rain, plastic rain overlay",
    ),
    (
        re.compile(r"\b(snow|blizzard|snowfall|winter storm)\b", re.I),
        WeatherMood.SNOW,
        "snow particle depth, accumulation on surfaces, cold breath optional",
        "white noise overlay, no surface accumulation",
    ),
    (
        re.compile(r"\b(dust|sandstorm|haze desert|smoke filled)\b", re.I),
        WeatherMood.DUST,
        "particulate haze, warm dust scatter, reduced distant contrast",
        "clean air in dust storm",
    ),
)


class AtmospherePlanner:
    def plan(self, prompt: str) -> AtmospherePlan:
        text = prompt or ""
        for pat, mood, pos, neg in _ATMOSPHERE_RULES:
            if pat.search(text):
                return AtmospherePlan(mood, pos, neg)
        if re.search(r"\b(landscape|exterior|outdoor|city)\b", text, re.I):
            return AtmospherePlan(
                WeatherMood.CLEAR,
                "atmospheric perspective, distant desaturation, aerial depth",
                "flat depth, everything same sharpness",
            )
        return AtmospherePlan(WeatherMood.CLEAR, "", "")

    def fragments(self, prompt: str) -> tuple[str, str]:
        p = self.plan(prompt)
        return p.positive, p.negative


# ==================================================================================================
# era: Historical era accuracy — costume, architecture, technology cues.
# ==================================================================================================


class HistoricalEra(str, Enum):
    ANCIENT = "ancient"
    MEDIEVAL = "medieval"
    RENAISSANCE = "renaissance"
    VICTORIAN = "victorian"
    EDWARDIAN = "edwardian"
    ART_DECO = "art_deco"
    MID_CENTURY = "mid_century"
    EIGHTIES = "eighties"
    NINETIES = "nineties"
    NEAR_FUTURE = "near_future"
    GENERIC = "generic"


@dataclass(frozen=True)
class EraPlan:
    era: HistoricalEra
    positive: str
    negative: str


_ERAS: tuple[tuple[re.Pattern, HistoricalEra, str, str], ...] = (
    (
        re.compile(r"\b(ancient rome|greek|egyptian|roman|spartan)\b", re.I),
        HistoricalEra.ANCIENT,
        "period-accurate drapery and armor, pre-industrial materials, patina",
        "modern zippers, wristwatches, synthetic fabric",
    ),
    (
        re.compile(r"\b(medieval|knight|castle|middle ages|viking)\b", re.I),
        HistoricalEra.MEDIEVAL,
        "medieval garment construction, hand-forged metal, timber and stone architecture",
        "modern eyeglasses, printed text, machine stitching",
    ),
    (
        re.compile(r"\b(renaissance|15th century|16th century|leonardo|medici)\b", re.I),
        HistoricalEra.RENAISSANCE,
        "renaissance textile and silhouette, oil-lamp lighting, classical proportion",
        "neon signs, plastic buttons, contemporary haircut",
    ),
    (
        re.compile(r"\b(victorian|19th century|steampunk|industrial revolution)\b", re.I),
        HistoricalEra.VICTORIAN,
        "victorian tailoring, gaslight warmth, brass and iron technology",
        "smartphones, LED lights, modern sneakers",
    ),
    (
        re.compile(r"\b(art deco|1920s|1930s|gatsby|flapper)\b", re.I),
        HistoricalEra.ART_DECO,
        "art deco geometry, jazz age fashion, geometric typography cues",
        "modern minimal UI, contemporary athleisure",
    ),
    (
        re.compile(r"\b(1950s|1960s|mid century|mid-century|mad men)\b", re.I),
        HistoricalEra.MID_CENTURY,
        "mid-century furniture lines, period hair and makeup, analog technology",
        "flat screen TVs, modern logos, contemporary streetwear",
    ),
    (
        re.compile(r"\b(1980s|80s|synthwave|retro 80)\b", re.I),
        HistoricalEra.EIGHTIES,
        "1980s fashion volume, analog grain optional, period correct electronics",
        "smartphones, 2020s streetwear, modern EV cars",
    ),
    (
        re.compile(r"\b(1990s|90s|grunge|y2k)\b", re.I),
        HistoricalEra.NINETIES,
        "1990s silhouette and tech, CRT optional, period denim and outerwear",
        "modern slim-fit only, 2020s phone design",
    ),
    (
        re.compile(r"\b(near future|2050|cyberpunk|sci-fi city)\b", re.I),
        HistoricalEra.NEAR_FUTURE,
        "coherent future design language, worn technology, plausible infrastructure",
        "random hologram spam, inconsistent tech eras mixed",
    ),
)


class EraPlanner:
    def plan(self, prompt: str) -> EraPlan:
        text = prompt or ""
        for pat, era, pos, neg in _ERAS:
            if pat.search(text):
                return EraPlan(era, pos, neg)
        return EraPlan(HistoricalEra.GENERIC, "", "")

    def fragments(self, prompt: str) -> tuple[str, str]:
        p = self.plan(prompt)
        return p.positive, p.negative


# ==================================================================================================
# materials: Surface material truth — metal, glass, fabric, skin, wood.
# ==================================================================================================


class SurfaceClass(str, Enum):
    METAL = "metal"
    GLASS = "glass"
    FABRIC = "fabric"
    SKIN = "skin"
    WOOD = "wood"
    STONE = "stone"
    LIQUID = "liquid"
    HAIR = "hair"


@dataclass(frozen=True)
class MaterialHint:
    surface: SurfaceClass
    positive: str
    negative: str


_SURFACES: tuple[tuple[re.Pattern, SurfaceClass, str, str], ...] = (
    (
        re.compile(r"\b(chrome|steel|brass|copper|metal|metallic|gold|silver)\b", re.I),
        SurfaceClass.METAL,
        "correct metal Fresnel, roughness breakup, specular highlight rolloff",
        "plastic painted as metal, mirror chrome everywhere",
    ),
    (
        re.compile(r"\b(glass|window|lens|crystal|transparent|refraction)\b", re.I),
        SurfaceClass.GLASS,
        "refraction and caustics, clean transmission, edge darkening",
        "sticker transparency, no refraction",
    ),
    (
        re.compile(r"\b(silk|velvet|linen|cotton|fabric|cloth|dress|suit)\b", re.I),
        SurfaceClass.FABRIC,
        "weave micro-texture, fold tension, subsurface on thin edges",
        "plastic wrap cloth, stiff cardboard folds",
    ),
    (
        re.compile(r"\b(skin|face|portrait|hands|body)\b", re.I),
        SurfaceClass.SKIN,
        "subsurface scatter, pore micro-detail in focal zone, natural oil sheen",
        "plastic wax skin, uniform pore stamp",
    ),
    (
        re.compile(r"\b(wood|oak|pine|mahogany|grain|timber)\b", re.I),
        SurfaceClass.WOOD,
        "directional wood grain, anisotropic highlight, edge wear",
        "repeating texture tile, plastic wood",
    ),
    (
        re.compile(r"\b(stone|marble|granite|concrete|brick)\b", re.I),
        SurfaceClass.STONE,
        "mineral grain, weathering variation, contact shadow grounding",
        "uniform noise rock, floating stones",
    ),
    (
        re.compile(r"\b(water|wine|liquid|pouring|splash|ocean)\b", re.I),
        SurfaceClass.LIQUID,
        "surface tension, caustics, refraction, believable viscosity",
        "solid blue plastic water",
    ),
    (
        re.compile(r"\b(hair|braid|ponytail|fur|mane)\b", re.I),
        SurfaceClass.HAIR,
        "individual strand groups, specular along fiber, flyaway variation",
        "solid helmet hair, plastic fur clumps",
    ),
)


class MaterialPlanner:
    def scan(self, prompt: str) -> list[MaterialHint]:
        text = prompt or ""
        return [MaterialHint(surf, pos, neg) for pat, surf, pos, neg in _SURFACES if pat.search(text)]

    def fragments(self, prompt: str, max_surfaces: int = 4) -> tuple[str, str]:
        hints = self.scan(prompt)[:max_surfaces]
        return ", ".join(h.positive for h in hints), ", ".join(h.negative for h in hints)


# ==================================================================================================
# motion: Motion blur direction and action peak moments.
# ==================================================================================================


class MotionStyle(str, Enum):
    FREEZE = "freeze"
    PAN = "pan"
    TRACK = "track"
    SPIN = "spin"
    IMPACT = "impact"
    FALL = "fall"


@dataclass(frozen=True)
class MotionPlan:
    style: MotionStyle
    positive: str
    negative: str


_motion_RULES: tuple[tuple[re.Pattern, MotionStyle, str, str], ...] = (
    (
        re.compile(r"\b(motion blur|panning shot|speed|racing|running)\b", re.I),
        MotionStyle.PAN,
        "directional motion blur on background, sharp subject core, panning streaks",
        "global blur mush, subject and background equally smeared",
    ),
    (
        re.compile(r"\b(frozen|mid[- ]air|peak action|decisive moment|splash)\b", re.I),
        MotionStyle.FREEZE,
        "frozen peak action, crisp subject edges, suspended particles",
        "motion smear on static pose, ambiguous timing",
    ),
    (
        re.compile(r"\b(punch|impact|explosion|collision|shatter)\b", re.I),
        MotionStyle.IMPACT,
        "radial debris, impact frame read, force direction clear",
        "random particles without force vector",
    ),
    (
        re.compile(r"\b(falling|jump|leap|dive)\b", re.I),
        MotionStyle.FALL,
        "gravity-readable pose, hair and cloth lag, vertical motion cue",
        "floating without weight, stiff cloth",
    ),
)


class MotionPlanner:
    def plan(self, prompt: str) -> MotionPlan:
        text = prompt or ""
        for pat, style, pos, neg in _motion_RULES:
            if pat.search(text):
                return MotionPlan(style, pos, neg)
        return MotionPlan(MotionStyle.FREEZE, "", "")

    def fragments(self, prompt: str) -> tuple[str, str]:
        p = self.plan(prompt)
        return p.positive, p.negative


# ==================================================================================================
# safety: Tiered pre-generation content policy for ``--frontier-perfect``.
#
# The minor-sexualization block runs first and ignores the tier: it is
# :mod:`utils.safety.minor_guard`. Tiers only decide how adult NSFW is steered toward
# adult-coded language: ``off`` never steers, ``moderate`` steers sexual prompts without an
# explicit adult cue, ``strict`` steers every sexual prompt.
# ==================================================================================================


class PolicyTier(str, Enum):
    OFF = "off"
    MODERATE = "moderate"
    STRICT = "strict"


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    STEER = "steer"
    REFUSE = "refuse"


@dataclass(frozen=True)
class SafetyReport:
    decision: PolicyDecision
    tier: PolicyTier
    reasons: tuple[str, ...]
    steer_positive: str = ""


_ADULT_AFFIRM = re.compile(
    r"\b(adult woman|adult man|mature woman|mature man|25[- ]year|30[- ]year|grown woman|grown man|of age)\b",
    re.I,
)
_STEER_POS = "adult subject with mature facial features, adult proportions, clearly of age"


class ContentPolicy:
    def __init__(self, tier: PolicyTier = PolicyTier.MODERATE) -> None:
        self.tier = tier

    def evaluate(self, prompt: str, *, negative: str = "") -> SafetyReport:
        blocked = minor_guard.reasons(prompt, negative)
        if blocked:
            return SafetyReport(PolicyDecision.REFUSE, self.tier, blocked)
        if self.tier == PolicyTier.OFF or not minor_guard.is_sexual(prompt):
            return SafetyReport(PolicyDecision.ALLOW, self.tier, ())
        if self.tier == PolicyTier.STRICT or not _ADULT_AFFIRM.search(prompt or ""):
            return SafetyReport(
                PolicyDecision.STEER, self.tier, ("adult_steering_recommended",), steer_positive=_STEER_POS
            )
        return SafetyReport(PolicyDecision.ALLOW, self.tier, ("adult_affirmation_present",))

    def apply_steering(self, prompt: str, report: SafetyReport) -> str:
        if report.decision != PolicyDecision.STEER or not report.steer_positive:
            return prompt
        return f"{prompt}, {report.steer_positive}" if prompt else report.steer_positive


# ==================================================================================================
# typography: Typography prompt packs for legible in-image text.
# ==================================================================================================


@dataclass(frozen=True)
class TypographyPlan:
    quoted_text: tuple[str, ...]
    positive: str
    negative: str
    cfg_boost: float


_QUOTED = re.compile(r'"([^"]{1,80})"')
_BRACKET = re.compile(r"\[text:\s*([^\]]{1,80})\]", re.I)
_SIGN = re.compile(r"\b(sign|logo|poster|banner|headline|typography|lettering|title card)\b", re.I)


class TypographyPlanner:
    def extract_quotes(self, prompt: str) -> list[str]:
        text = prompt or ""
        found = (t.strip() for t in (*_QUOTED.findall(text), *_BRACKET.findall(text)))
        return list(dict.fromkeys(t for t in found if t))

    def plan(self, prompt: str) -> TypographyPlan:
        quotes = tuple(self.extract_quotes(prompt))
        if not quotes and not _SIGN.search(prompt or ""):
            return TypographyPlan((), "", "", 0.0)

        spell = ", ".join(f'legible text reading exactly "{q}"' for q in quotes[:3])
        pos = ", ".join(
            p
            for p in (
                spell,
                "crisp typography, correct spelling, high contrast text-background separation",
                "professional kerning and baseline alignment",
            )
            if p
        )
        neg = (
            "misspelled text, garbled letters, melted typography, nonsense characters, "
            "duplicate letters, watermark, signature overlay"
        )
        return TypographyPlan(quotes, pos, neg, cfg_boost=1.15 if quotes else 1.08)

    def fragments(self, prompt: str) -> tuple[str, str]:
        p = self.plan(prompt)
        return p.positive, p.negative


# ==================================================================================================
# perfect
# ==================================================================================================


@dataclass
class PerfectFrontierPlan:
    prompt: str
    deep: DeepFrontierPlan
    subject: SubjectFrontierPlan
    safety: SafetyReport
    art_direction: ArtDirection | None = None
    atmosphere_pos: str = ""
    atmosphere_neg: str = ""
    materials_pos: str = ""
    materials_neg: str = ""
    motion_pos: str = ""
    motion_neg: str = ""
    era_pos: str = ""
    era_neg: str = ""
    typography_pos: str = ""
    typography_neg: str = ""
    final_prompt: str = ""
    final_negative: str = ""
    refused: bool = False
    refuse_reasons: list[str] = field(default_factory=list)


def analyze_perfect(
    prompt: str,
    *,
    num_steps: int = 28,
    serendipity_dial: float = 0.25,
    layout_regions: int = 0,
    auto_resolve_contradictions: bool = False,
    medium_mode: str = "auto",
    safety_tier: PolicyTier = PolicyTier.MODERATE,
) -> PerfectFrontierPlan:
    """Evaluate the content policy first; a refused prompt gets a plan with ``refused=True`` and no final prompt."""
    policy = ContentPolicy(safety_tier)
    safety = policy.evaluate(prompt)
    if safety.decision == PolicyDecision.REFUSE:
        return PerfectFrontierPlan(
            prompt=prompt,
            deep=analyze_deep(prompt, num_steps=num_steps, serendipity_dial=0.0, layout_regions=layout_regions),
            subject=analyze_subject(prompt, medium_mode=medium_mode),
            safety=safety,
            refused=True,
            refuse_reasons=list(safety.reasons),
        )

    working = policy.apply_steering(prompt, safety)

    deep = analyze_deep(
        working,
        num_steps=num_steps,
        serendipity_dial=serendipity_dial,
        layout_regions=layout_regions,
        auto_resolve_contradictions=auto_resolve_contradictions,
    )
    subject = analyze_subject(working, medium_mode=medium_mode)

    art = plan_art_direction(working)
    atm_p, atm_n = AtmospherePlanner().fragments(working)
    mat_p, mat_n = MaterialPlanner().fragments(working)
    mot_p, mot_n = MotionPlanner().fragments(working)
    era_p, era_n = EraPlanner().fragments(working)
    typ_p, typ_n = TypographyPlanner().fragments(working)

    scene_pos = merge_csv_unique(art.caption(), atm_p, mat_p, mot_p, era_p, typ_p)
    scene_neg = merge_csv_unique(atm_n, mat_n, mot_n, era_n, typ_n)

    base_prompt = deep.deep_augmented_prompt or working
    final_prompt = merge_csv_unique(base_prompt, subject.merged_positive, scene_pos)
    if len(final_prompt) > 900:
        final_prompt = merge_csv_unique(base_prompt, scene_pos[:400])

    return PerfectFrontierPlan(
        prompt=prompt,
        deep=deep,
        subject=subject,
        safety=safety,
        art_direction=art,
        atmosphere_pos=atm_p,
        atmosphere_neg=atm_n,
        materials_pos=mat_p,
        materials_neg=mat_n,
        motion_pos=mot_p,
        motion_neg=mot_n,
        era_pos=era_p,
        era_neg=era_n,
        typography_pos=typ_p,
        typography_neg=typ_n,
        final_prompt=final_prompt or working,
        final_negative=merge_csv_unique(subject.merged_negative, scene_neg),
    )


def perfect_sample_kwargs(plan: PerfectFrontierPlan, *, base_negative: str = "") -> dict[str, Any]:
    if plan.refused:
        return {"refused": True, "refuse_reasons": plan.refuse_reasons, "prompt": plan.prompt}

    deep_kw = deep_sample_kwargs(plan.deep, base_negative=base_negative)
    sub_kw = subject_sample_kwargs(plan.subject, base_negative=plan.final_negative or base_negative)

    cfg = float(deep_kw["cfg_scale_multiplier"] or 1.0) * float(sub_kw["cfg_scale_multiplier"])
    cfg *= max(1.0, TypographyPlanner().plan(plan.prompt).cfg_boost)
    return {
        **deep_kw,
        **{k: v for k, v in sub_kw.items() if k not in ("prompt", "negative_prompt")},
        "prompt": plan.final_prompt,
        "negative_prompt": merge_csv_unique(base_negative, plan.final_negative, deep_kw["negative_prompt"]),
        "cfg_scale_multiplier": cfg,
        "perfect_frontier": True,
        "safety_decision": plan.safety.decision.value,
        "safety_tier": plan.safety.tier.value,
    }


def list_video_frontier_modules() -> list[dict[str, str]]:
    from pipelines.video.frontier_compiler import list_frontier_modules

    return list_frontier_modules()


def bridge_shot_prompt(
    prompt: str,
    *,
    tension: float = 0.5,
    num_steps: int = 28,
) -> dict[str, Any]:
    """Augment a single keyframe prompt using SDX + video frontier."""
    from pipelines.video.frontier_compiler import (
        apply_causal_ripples,
        compose_shot_framing,
        parse_causal_rules,
        parse_mise_config,
    )

    from frontier.engine import TemporalMomentAnalyzer, analyze_prompt, build_tension_field

    class _Shot:
        def __init__(self, p: str) -> None:
            self.id = "bridge"
            self.prompt = p
            self.shot_type = "medium"
            self.camera = ""

    plan = analyze_prompt(prompt, num_steps=num_steps)
    tension_plan = build_tension_field(num_steps).plan(tension)
    moment = TemporalMomentAnalyzer(num_steps=num_steps).analyze(prompt)
    plaus = PhysicalPlausibilityScanner().scan(prompt)
    ripple = apply_causal_ripples([_Shot(prompt)], parse_causal_rules({}), use_builtins=True)
    mise = compose_shot_framing(_Shot(prompt), config=parse_mise_config({"enabled": True}))

    frags: list[str] = list(
        plan.augmented_prompt.replace(prompt, "").strip(", ").split(", ") if plan.augmented_prompt != prompt else []
    )
    frags.extend(tension_plan.prompt_fragments)
    frags.extend(moment.prompt_fragments)
    if mise:
        frags.append(mise.prompt_suffix)
    if ripple:
        frags.append(ripple[0].prompt_suffix)

    negatives: list[str] = []
    if plan.echo_negative:
        negatives.append(plan.echo_negative)
    for flag in plaus:
        if flag.severity >= 0.5:
            negatives.append(f"missing {flag.missing}")

    augmented = prompt
    clean_frags = [f.strip() for f in frags if f.strip()]
    if clean_frags:
        augmented = f"{prompt}, {', '.join(dict.fromkeys(clean_frags))}"

    return {
        "augmented_prompt": augmented,
        "negative_suffix": ", ".join(dict.fromkeys(negatives)),
        "tension": tension,
        "step_emphasis": list(tension_plan.step_emphasis),
        "cfg_boost": tension_plan.cfg_boost,
        "moment_phase": moment.phase.value,
        "risk_score": plan.risk_score,
        "plausibility_flags": [{"trigger": f.trigger, "missing": f.missing, "severity": f.severity} for f in plaus],
    }


def unified_frontier_augment(
    data: Mapping[str, Any],
    shots: list[Any],
    **kwargs: Any,
) -> dict[str, Any]:
    from pipelines.video.frontier_compiler import compile_frontier_layers

    result = compile_frontier_layers(data, shots, **kwargs)
    return {
        "metadata": result.metadata,
        "issues": result.issues,
        "global_edit": result.global_edit,
        "enrichments": {
            k: {
                "prompt_suffix": v.prompt_suffix,
                "negative_suffix": v.negative_suffix,
                "duration_delta": v.duration_delta,
                "metadata": v.metadata,
            }
            for k, v in result.enrichments.items()
        },
    }


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
    "AnatomyRisk",
    "BodyMode",
    "BodyPlan",
    "BodyPlanner",
    "CreatureFamily",
    "CreaturePlan",
    "CreatureTaxonomy",
    "MatureClass",
    "MatureGuidance",
    "MaturePlan",
    "BrushPlan",
    "BrushPlanner",
    "StrokeStyle",
    "AntiSlopPlan",
    "AntiSlopScanner",
    "PhotorealPlan",
    "PhotorealStack",
    "RealismTier",
    "SubjectFrontierPlan",
    "analyze_subject",
    "subject_sample_kwargs",
    "AtmospherePlan",
    "AtmospherePlanner",
    "WeatherMood",
    "EraPlan",
    "EraPlanner",
    "HistoricalEra",
    "MaterialHint",
    "MaterialPlanner",
    "SurfaceClass",
    "MotionPlan",
    "MotionPlanner",
    "MotionStyle",
    "ContentPolicy",
    "PolicyDecision",
    "PolicyTier",
    "SafetyReport",
    "TypographyPlan",
    "TypographyPlanner",
    "PerfectFrontierPlan",
    "analyze_perfect",
    "perfect_sample_kwargs",
    "bridge_shot_prompt",
    "list_video_frontier_modules",
    "unified_frontier_augment",
]

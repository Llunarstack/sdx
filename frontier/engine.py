"""Frontier engine: compose logic, narrative, chaos, and memory analyzers into one plan, and wire frontier plans
into the production sampling stack (``apply_frontier_to_args`` / ``frontier_diffusion_hooks``; no model weights).
"""

from __future__ import annotations

import math
import re
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from config.prompts import merge_csv_unique

# ==================================================================================================
# chaos: Controlled randomness for sampling.
#
# ``EntropyBudgetAllocator`` spends a fixed noise budget across steps and regions (layout early, texture late)
# instead of making every step equally noisy. ``SerendipityInjector`` turns a 0..1 dial into a per-step noise
# scale curve for happy accidents.
# ==================================================================================================

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class EntropyBudget:
    total: float
    per_step: tuple[float, ...]
    per_region: tuple[float, ...]  # (foreground, background) shares when there are two regions


class EntropyBudgetAllocator:
    """Split ``total`` entropy across steps (default 40% early / 30% mid / 30% late) and regions."""

    def __init__(self, num_steps: int = 28, num_regions: int = 2) -> None:
        self.num_steps = max(4, int(num_steps))
        self.num_regions = max(1, int(num_regions))

    def allocate(
        self,
        total: float = 1.0,
        *,
        early_weight: float = 0.4,
        mid_weight: float = 0.3,
        late_weight: float = 0.3,
        foreground_share: float = 0.35,
    ) -> EntropyBudget:
        total = max(0.0, float(total))
        third = max(1, self.num_steps // 3)
        weights = [
            early_weight if i < third else mid_weight if i < 2 * third else late_weight for i in range(self.num_steps)
        ]
        s = sum(weights) or 1.0
        fg = max(0.0, min(1.0, float(foreground_share)))
        per_region = (fg, 1.0 - fg) if self.num_regions == 2 else (1.0 / self.num_regions,) * self.num_regions
        return EntropyBudget(total=total, per_step=tuple(total * w / s for w in weights), per_region=per_region)


@dataclass(frozen=True)
class SerendipityCurve:
    """Per-step multiplicative noise scale (1.0 = no extra surprise)."""

    scales: tuple[float, ...]
    peak_step: int
    dial: float  # user creativity 0..1


class SerendipityInjector:
    """Bell-shaped serendipity curve: mid-schedule surprises explore layout; early/late steps stay faithful."""

    def __init__(self, num_steps: int = 28) -> None:
        self.num_steps = max(4, int(num_steps))

    def curve(self, dial: float = 0.3) -> SerendipityCurve:
        dial = max(0.0, min(1.0, float(dial)))
        n = self.num_steps
        mid = (n - 1) / 2.0
        scales = tuple(1.0 + dial * 0.35 * math.exp(-0.5 * ((i - mid) / max(1.0, mid)) ** 2) for i in range(n))
        peak = max(range(n), key=scales.__getitem__)
        return SerendipityCurve(scales=scales, peak_step=peak, dial=dial)

    def apply_to_noise(self, noise: torch.Tensor, step_index: int, curve: SerendipityCurve) -> torch.Tensor:
        if not 0 <= step_index < len(curve.scales):
            return noise
        s = curve.scales[step_index]
        return noise if abs(s - 1.0) < 1e-6 else noise * s


# ==================================================================================================
# logic: Prompt logic: absence and contradictions.
#
# ``AbsenceExtractor`` turns "empty bench", "no text", "clear sky" into explicit negative-space constraints.
# ``ContradictionScanner`` flags mutually exclusive asks ("night" + "midday sun") and can resolve them.
# ==================================================================================================


@dataclass(frozen=True)
class AbsenceConstraint:
    """Something the scene must lack or keep empty."""

    subject: str
    strength: float  # 0..1
    source_phrase: str
    negative_prompt_fragment: str


# (pattern, subject template for ``Match.expand``, strength)
_ABSENCE_PATTERNS: tuple[tuple[str, str, float], ...] = (
    (r"\bno\s+([a-z][\w-]{1,24}s?)\b", r"\1", 0.9),
    (r"\bwithout\s+([a-z][\w-]{1,24}s?)\b", r"\1", 0.85),
    (r"\bempty\s+([a-z][\w-]{1,24}s?)\b", r"\1", 0.75),
    (r"\bclear\s+([a-z][\w-]{1,24}s?)\b", r"\1", 0.7),
    (r"\bbare\s+([a-z][\w-]{1,24}s?)\b", r"\1", 0.65),
    (r"\bnegative\s+space\b", "clutter", 0.8),
    (r"\bminimal\s+background\b", "busy background, clutter", 0.7),
)


class AbsenceExtractor:
    """Pull absence constraints from natural-language prompts."""

    def extract(self, prompt: str) -> list[AbsenceConstraint]:
        text = (prompt or "").strip()
        out: list[AbsenceConstraint] = []
        seen: set[str] = set()
        for pattern, subject_tpl, strength in _ABSENCE_PATTERNS:
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                subject = m.expand(subject_tpl).lower()
                if subject in seen:
                    continue
                seen.add(subject)
                out.append(
                    AbsenceConstraint(
                        subject=subject,
                        strength=strength,
                        source_phrase=m.group(0),
                        negative_prompt_fragment=f"{subject}, {subject} everywhere, crowded {subject}",
                    )
                )
        return out

    def merge_negative_prompt(self, base_negative: str, constraints: Sequence[AbsenceConstraint]) -> str:
        """Append the fragments of constraints with strength >= 0.7 to ``base_negative`` (deduped)."""
        return merge_csv_unique(base_negative, *(c.negative_prompt_fragment for c in constraints if c.strength >= 0.7))


@dataclass(frozen=True)
class Contradiction:
    left: str
    right: str
    severity: float  # 0..1
    category: str
    resolution: str


# (pattern_a, pattern_b, category, resolution hint)
_CONFLICT_RULES: tuple[tuple[str, str, str, str], ...] = (
    (r"\bnoon\b|\bmidday\b", r"\bsunset\b|\bsunrise\b|\bdusk\b|\bdawn\b", "time", "pick one time of day"),
    (r"\bnight\b|\bmidnight\b", r"\bbright\s+day\b|\bnoon\b", "time", "pick day or night"),
    (
        r"\bempty\b|\bdeserted\b|\bno\s+people\b|\bno\s+crowd",
        r"\bcrowd\b|\bpacked\b|\bbusy\b",
        "quantity",
        "choose empty or crowded",
    ),
    (
        r"\bindoor\b|\binside\b|\binterior\b",
        r"\boutdoor\b|\boutside\b|\blandscape\b",
        "space",
        "choose interior or exterior",
    ),
    (
        r"\bmacro\b|\bclose[- ]?up\b",
        r"\bwide\s+shot\b|\bestablishing\b|\baerial\b",
        "framing",
        "choose one framing scale",
    ),
    (r"\bminimal\b|\bsparse\b", r"\bcluttered\b|\bbusy\s+background\b", "composition", "choose minimal or busy"),
    (r"\bphotorealistic\b|\bhyperreal", r"\bcartoon\b|\banime\b|\billustration\b", "style", "pick one render style"),
    (r"\bunderwater\b", r"\bdesert\b|\barid\b", "environment", "environments are incompatible"),
    (r"\bsnow\b|\bblizzard\b", r"\btropical\b|\bpalm\b", "climate", "pick one climate"),
    (
        r"\binvisible\b|\btransparent\b",
        r"\bvisible\b|\bclearly\b|\bdetailed\b",
        "visibility",
        "object cannot be both hidden and prominent",
    ),
)


def _find_match(text: str, pattern: str) -> str | None:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return m.group(0) if m else None


class ContradictionScanner:
    """Scan prompt text for high-confidence logical conflicts."""

    def __init__(self, extra_rules: Sequence[tuple[str, str, str, str]] = ()) -> None:
        self._rules = _CONFLICT_RULES + tuple(extra_rules)

    def scan(self, prompt: str) -> list[Contradiction]:
        """Conflicts found in ``prompt``, most severe first."""
        text = (prompt or "").strip()
        found: list[Contradiction] = []
        for pat_a, pat_b, category, resolution in self._rules:
            a = _find_match(text, pat_a)
            b = _find_match(text, pat_b)
            if a and b:
                # Longer (more specific) matched phrases read as a more deliberate conflict.
                severity = min(1.0, 0.55 + 0.1 * (len(a) + len(b)) / 20.0)
                found.append(
                    Contradiction(left=a, right=b, severity=severity, category=category, resolution=resolution)
                )
        return sorted(found, key=lambda c: c.severity, reverse=True)

    def suggest_rewrite(self, prompt: str, *, pick: str = "left") -> str:
        """Naive rewrite: delete the first occurrence of one side of each conflict (``pick`` = side to keep)."""
        text = prompt
        for c in self.scan(prompt):
            drop = c.right if pick == "left" else c.left
            text = re.sub(re.escape(drop), "", text, count=1, flags=re.IGNORECASE)
        return re.sub(r"\s{2,}", " ", text).strip(" ,")


# ==================================================================================================
# memory: Cross-run memory of generation failures: not full RL, just "don't step on the same rake."
#
# Stores artifact tags from rejected outputs and merges them into negative prompts.
# ==================================================================================================


@dataclass
class ArtifactEcho:
    prompt_fingerprint: str
    tags: list[str]
    severity: float


# Always appended to the echo negative, whether or not anything was recorded.
_BASELINE_ARTIFACT_TAGS = ("extra fingers", "melted hands", "duplicate face")


class GenerationEchoMemory:
    """Ring buffer of recent failure echoes keyed by a coarse prompt fingerprint (first N lowercase words)."""

    def __init__(self, capacity: int = 64, fingerprint_words: int = 6) -> None:
        self.capacity = max(4, int(capacity))
        self.fingerprint_words = max(2, int(fingerprint_words))
        self._echoes: deque[ArtifactEcho] = deque(maxlen=self.capacity)

    @staticmethod
    def fingerprint(prompt: str, *, words: int = 6) -> str:
        return " ".join((prompt or "").lower().split()[:words])

    def record_failure(self, prompt: str, tags: Sequence[str], *, severity: float = 0.8) -> ArtifactEcho:
        echo = ArtifactEcho(
            prompt_fingerprint=self.fingerprint(prompt, words=self.fingerprint_words),
            tags=[t.strip().lower() for t in tags if t.strip()],
            severity=float(severity),
        )
        self._echoes.append(echo)
        return echo

    def negative_suffix(self, prompt: str, *, min_severity: float = 0.5) -> str:
        """Tags of severe echoes whose fingerprint matches or half-overlaps ``prompt``'s, plus the baseline tags."""
        fp = self.fingerprint(prompt, words=self.fingerprint_words)
        tags: list[str] = []
        for echo in self._echoes:
            if echo.severity >= min_severity and (
                echo.prompt_fingerprint == fp or self._overlap(fp, echo.prompt_fingerprint)
            ):
                tags.extend(echo.tags)
        return ", ".join(dict.fromkeys([*tags, *_BASELINE_ARTIFACT_TAGS]))

    @staticmethod
    def _overlap(a: str, b: str) -> bool:
        wa, wb = set(a.split()), set(b.split())
        if not wa or not wb:
            return False
        return len(wa & wb) / min(len(wa), len(wb)) >= 0.5


# ==================================================================================================
# narrative: Narrative reading of a single frame.
#
# ``TemporalMomentAnalyzer`` treats the image as a slice of a longer event ("about to fall", "just landed") and
# emits step-emphasis hints. ``WitnessPerspectiveAnalyzer`` picks whose eyes the camera borrows (child, guard,
# lover, stranger, archivist, ...) and the framing that implies.
# ==================================================================================================


class MomentPhase(str, Enum):
    STATIC = "static"
    ANTICIPATION = "anticipation"  # before the action
    CLIMAX = "climax"  # peak action
    AFTERMATH = "aftermath"  # just after


@dataclass(frozen=True)
class MomentCue:
    phase: MomentPhase
    confidence: float
    verb: str
    prompt_fragments: list[str]
    step_emphasis: tuple[float, ...]  # one weight per step; higher = more structure


# Checked in order; the first phase with a matching cue wins.
_PHASE_CUES: tuple[tuple[MomentPhase, tuple[str, ...], float, tuple[str, ...]], ...] = (
    (
        MomentPhase.ANTICIPATION,
        (r"\babout\s+to\b", r"\bon\s+the\s+verge\b", r"\bmoment\s+before\b", r"\bpoised\s+to\b", r"\bjust\s+before\b"),
        0.85,
        ("tension in muscles", "frozen instant", "implied motion blur minimal"),
    ),
    (
        MomentPhase.CLIMAX,
        (r"\bmid[- ]?air\b", r"\bat\s+the\s+peak\b", r"\bin\s+motion\b", r"\bsplashing\b"),
        0.8,
        ("dynamic motion", "action peak", "motion blur on extremities"),
    ),
    (
        MomentPhase.AFTERMATH,
        (r"\bjust\s+after\b", r"\baftermath\b", r"\bsettling\b", r"\bstill\s+smoking\b"),
        0.75,
        ("debris settling", "trailing smoke", "relaxed poses"),
    ),
)


class TemporalMomentAnalyzer:
    """Classify temporal stance and emit a denoise-step emphasis curve."""

    def __init__(self, num_steps: int = 28) -> None:
        self.num_steps = max(4, int(num_steps))

    def analyze(self, prompt: str) -> MomentCue:
        text = (prompt or "").lower()
        for phase, patterns, confidence, frags in _PHASE_CUES:
            for pat in patterns:
                m = re.search(pat, text)
                if m:
                    return MomentCue(phase, confidence, m.group(0), list(frags), self._emphasis_curve(phase))
        return MomentCue(MomentPhase.STATIC, 0.0, "", [], self._emphasis_curve(MomentPhase.STATIC))

    def _emphasis_curve(self, phase: MomentPhase) -> tuple[float, ...]:
        n = self.num_steps
        if phase == MomentPhase.STATIC:
            return (1.0,) * n
        if phase == MomentPhase.ANTICIPATION:
            # Structure early, leave noise late for "about to happen".
            return tuple(1.0 - 0.6 * (i / (n - 1)) for i in range(n))
        if phase == MomentPhase.CLIMAX:
            mid = n // 2
            return tuple(0.7 + 0.6 * (1.0 - abs(i - mid) / mid) for i in range(n))
        # Aftermath: lock composition early, refine texture late.
        return tuple(0.5 + 0.5 * (i / (n - 1)) for i in range(n))


class WitnessPerspective(str, Enum):
    NEUTRAL = "neutral"
    CHILD = "child"  # low angle, wonder
    GUARD = "guard"  # surveillance, flat, wide
    LOVER = "lover"  # intimate close, soft bokeh
    STRANGER = "stranger"  # candid, off-center
    VICTIM = "victim"  # dutch angle, tension
    ARCHIVIST = "archivist"  # documentary, flat light


@dataclass(frozen=True)
class WitnessFrame:
    perspective: WitnessPerspective
    confidence: float
    prompt_fragments: list[str]
    cfg_bias: float  # multiply base CFG slightly


_WITNESS_TRIGGERS: dict[WitnessPerspective, tuple[str, ...]] = {
    WitnessPerspective.CHILD: (
        r"\bchild'?s?\s+(?:view|eyes|perspective)\b",
        r"\bthrough\s+a\s+child\b",
        r"\bwonder\b",
    ),
    WitnessPerspective.GUARD: (
        r"\bsecurity\s+camera\b",
        r"\bCCTV\b",
        r"\bsurveillance\b",
        r"\bmonitor\s+feed\b",
    ),
    WitnessPerspective.LOVER: (
        r"\bintimate\b",
        r"\blover'?s?\s+gaze\b",
        r"\bfrom\s+the\s+bed\b",
    ),
    WitnessPerspective.STRANGER: (
        r"\bcandid\b",
        r"\bstreet\s+photography\b",
        r"\bunposed\b",
    ),
    WitnessPerspective.VICTIM: (
        r"\bthreatening\b",
        r"\bhorror\s+POV\b",
        r"\bfirst[- ]person\s+terror\b",
    ),
    WitnessPerspective.ARCHIVIST: (
        r"\bdocumentary\b",
        r"\barchival\b",
        r"\bhistorical\s+photo\b",
    ),
}

_FRAGMENT_HINTS: dict[WitnessPerspective, tuple[str, ...]] = {
    WitnessPerspective.CHILD: ("low angle", "wide eyes in foreground optional", "soft natural light"),
    WitnessPerspective.GUARD: ("fisheye lens", "timestamp overlay", "high angle", "flat fluorescent"),
    WitnessPerspective.LOVER: ("shallow depth of field", "warm skin tones", "close framing"),
    WitnessPerspective.STRANGER: ("decisive moment", "off-center subject", "35mm lens"),
    WitnessPerspective.VICTIM: ("dutch angle", "motion blur edge", "claustrophobic framing"),
    WitnessPerspective.ARCHIVIST: ("neutral color grade", "even lighting", "medium format"),
}

_CFG_BIAS: dict[WitnessPerspective, float] = {
    WitnessPerspective.NEUTRAL: 1.0,
    WitnessPerspective.CHILD: 1.05,
    WitnessPerspective.GUARD: 0.92,
    WitnessPerspective.LOVER: 1.08,
    WitnessPerspective.STRANGER: 1.0,
    WitnessPerspective.VICTIM: 1.12,
    WitnessPerspective.ARCHIVIST: 0.95,
}


class WitnessPerspectiveAnalyzer:
    """Infer the embodied viewer from prompt cues (most matching cues wins; ties keep table order)."""

    def analyze(self, prompt: str) -> WitnessFrame:
        text = (prompt or "").strip()
        best, best_score = WitnessPerspective.NEUTRAL, 0
        for perspective, patterns in _WITNESS_TRIGGERS.items():
            score = sum(1 for pat in patterns if re.search(pat, text, flags=re.IGNORECASE))
            if score > best_score:
                best, best_score = perspective, score
        return WitnessFrame(
            perspective=best,
            confidence=min(1.0, best_score / 2.0),
            prompt_fragments=list(_FRAGMENT_HINTS.get(best, ())),
            cfg_bias=_CFG_BIAS[best],
        )


# ==================================================================================================
# engine
# ==================================================================================================


@dataclass
class FrontierPlan:
    prompt: str
    contradictions: list[Contradiction] = field(default_factory=list)
    absence: list[AbsenceConstraint] = field(default_factory=list)
    witness: WitnessFrame | None = None
    moment: MomentCue | None = None
    serendipity: SerendipityCurve | None = None
    entropy: EntropyBudget | None = None
    echo_negative: str = ""
    augmented_prompt: str = ""
    risk_score: float = 0.0


def analyze_prompt(
    prompt: str,
    *,
    num_steps: int = 28,
    serendipity_dial: float = 0.25,
    entropy_total: float = 1.0,
    echo_memory: GenerationEchoMemory | None = None,
    auto_resolve_contradictions: bool = False,
) -> FrontierPlan:
    """Run all frontier analyzers and return a unified plan."""
    scanner = ContradictionScanner()
    contradictions = scanner.scan(prompt)
    absence = AbsenceExtractor().extract(prompt)
    witness = WitnessPerspectiveAnalyzer().analyze(prompt)
    moment = TemporalMomentAnalyzer(num_steps=num_steps).analyze(prompt)

    working = scanner.suggest_rewrite(prompt) if auto_resolve_contradictions and contradictions else prompt
    frags = [*witness.prompt_fragments, *moment.prompt_fragments]
    augmented = f"{working}, {', '.join(frags)}" if frags else working

    risk = contradictions[0].severity if contradictions else 0.0
    risk = min(1.0, risk + 0.05 * len(absence) + (0.05 if witness.confidence > 0.5 else 0.0))

    return FrontierPlan(
        prompt=prompt,
        contradictions=contradictions,
        absence=absence,
        witness=witness,
        moment=moment,
        serendipity=SerendipityInjector(num_steps=num_steps).curve(serendipity_dial),
        entropy=EntropyBudgetAllocator(num_steps=num_steps).allocate(entropy_total),
        echo_negative=echo_memory.negative_suffix(prompt) if echo_memory is not None else "",
        augmented_prompt=augmented,
        risk_score=risk,
    )


class FrontierEngine:
    """Repeated frontier analysis with a shared failure memory, plus ``sample.py`` kwargs export."""

    def __init__(
        self,
        num_steps: int = 28,
        serendipity_dial: float = 0.25,
        echo_memory: GenerationEchoMemory | None = None,
    ) -> None:
        self.num_steps = num_steps
        self.serendipity_dial = serendipity_dial
        self.echo_memory = echo_memory or GenerationEchoMemory()
        self._absence = AbsenceExtractor()

    def analyze(
        self,
        prompt: str,
        *,
        num_steps: int | None = None,
        serendipity_dial: float | None = None,
        entropy_total: float = 1.0,
        auto_resolve_contradictions: bool = False,
    ) -> FrontierPlan:
        return analyze_prompt(
            prompt,
            num_steps=self.num_steps if num_steps is None else num_steps,
            serendipity_dial=self.serendipity_dial if serendipity_dial is None else serendipity_dial,
            entropy_total=entropy_total,
            echo_memory=self.echo_memory,
            auto_resolve_contradictions=auto_resolve_contradictions,
        )

    def sample_kwargs(self, plan: FrontierPlan, *, base_negative: str = "") -> dict[str, Any]:
        """Map a plan to kwargs compatible with ``sample.py`` / diffusion hooks."""
        neg = self._absence.merge_negative_prompt(base_negative, plan.absence)
        if plan.echo_negative:
            neg = f"{neg}, {plan.echo_negative}" if neg else plan.echo_negative
        return {
            "prompt": plan.augmented_prompt or plan.prompt,
            "negative_prompt": neg,
            "frontier_risk_score": plan.risk_score,
            "frontier_serendipity_scales": list(plan.serendipity.scales) if plan.serendipity else None,
            "frontier_step_emphasis": list(plan.moment.step_emphasis) if plan.moment else None,
            "frontier_entropy_per_step": list(plan.entropy.per_step) if plan.entropy else None,
            "cfg_scale_multiplier": plan.witness.cfg_bias if plan.witness and plan.witness.confidence > 0 else None,
            "frontier_contradictions": [
                {"left": c.left, "right": c.right, "severity": c.severity, "category": c.category}
                for c in plan.contradictions
            ],
        }

    def record_failure(self, prompt: str, tags: list[str], *, severity: float = 0.8) -> None:
        self.echo_memory.record_failure(prompt, tags, severity=severity)


# ==================================================================================================
# Restored from b654ae9:frontier/narrative/tension_field.py
# ==================================================================================================
@dataclass(frozen=True)
class TensionFieldPlan:
    tension: float
    prompt_fragments: tuple[str, ...]
    step_emphasis: tuple[float, ...]
    cfg_boost: float


@dataclass
class TensionField:
    num_steps: int = 28

    def plan(self, tension: float) -> TensionFieldPlan:
        t = max(0.0, min(1.0, float(tension)))
        n = max(4, self.num_steps)
        if t >= 0.75:
            frags = ("sharp micro-contrast", "urgent diagonal energy", "high-stakes focal sharpness")
            curve = tuple(0.55 + 0.9 * (1.0 - abs(i / max(1, n - 1) - 0.35)) for i in range(n))
            cfg = 1.0 + t * 0.35
        elif t >= 0.4:
            frags = ("controlled drama", "building visual pressure")
            curve = tuple(0.7 + 0.5 * (i / max(1, n - 1)) for i in range(n))
            cfg = 1.0 + t * 0.15
        else:
            frags = ("calm negative space", "gentle atmospheric falloff")
            curve = tuple(1.0 - 0.55 * (i / max(1, n - 1)) for i in range(n))
            cfg = 1.0
        return TensionFieldPlan(tension=t, prompt_fragments=frags, step_emphasis=curve, cfg_boost=cfg)


def build_tension_field(num_steps: int = 28) -> TensionField:
    return TensionField(num_steps=num_steps)


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
    "EntropyBudget",
    "EntropyBudgetAllocator",
    "SerendipityCurve",
    "SerendipityInjector",
    "AbsenceConstraint",
    "AbsenceExtractor",
    "Contradiction",
    "ContradictionScanner",
    "ArtifactEcho",
    "GenerationEchoMemory",
    "MomentCue",
    "TemporalMomentAnalyzer",
    "WitnessFrame",
    "WitnessPerspective",
    "WitnessPerspectiveAnalyzer",
    "FrontierEngine",
    "FrontierPlan",
    "analyze_prompt",
    "apply_frontier_to_args",
    "frontier_diffusion_hooks",
]

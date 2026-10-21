"""Perfect frontier (``--frontier-perfect``): deep + subject + scene-quality analyzers behind the content policy.

Also holds what used to be: atmosphere, era, materials, motion, safety, typography.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from config.defaults._lexicon import merge_csv_unique
from utils.prompt.art_direction import ArtDirection, plan_art_direction
from utils.safety import minor_guard

from .subject import SubjectFrontierPlan, analyze_subject, subject_sample_kwargs
from .synthesis import DeepFrontierPlan, analyze_deep, deep_sample_kwargs

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


_RULES: tuple[tuple[re.Pattern, WeatherMood, str, str], ...] = (
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
        for pat, mood, pos, neg in _RULES:
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


__all__ = [
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
]

"""Subject-aware frontier synthesis: bodies, creatures, mature content, mediums, brushwork, realism.

Also holds what used to be: anatomy, creatures, mature, medium, realism.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from config.defaults._lexicon import merge_csv_unique
from config.defaults.art_mediums import guidance_fragments

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
# Complements ``config.defaults.art_mediums`` with stroke-level vocabulary models understand.
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


_RULES: tuple[tuple[re.Pattern, StrokeStyle, str, str], ...] = (
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
        for pat, style, pos, neg in _RULES:
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


__all__ = [
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
]

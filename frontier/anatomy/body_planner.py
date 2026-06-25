"""
Body-mode planner — human realism vs intentional fantasy vs creature anatomy.

Models confuse *broken human* anatomy with *designed non-human* forms. This router
picks the right positive/negative pack so dragons are not penalized for extra limbs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


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
    triggers: Tuple[str, ...]
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
        if _CREATURE.search(text) and not _HUMAN.search(text):
            return BodyMode.CREATURE
        if _STYLIZED.search(text) and _HUMAN.search(text):
            return BodyMode.STYLIZED_HUMAN
        if _HUMAN.search(text):
            return BodyMode.REALISTIC_HUMAN if _REALISM.search(text) else BodyMode.STYLIZED_HUMAN
        if _CREATURE.search(text):
            return BodyMode.CREATURE
        return BodyMode.ABSTRACT

    def assess_risk(self, prompt: str, mode: BodyMode | None = None) -> AnatomyRisk:
        text = prompt or ""
        mode = mode or self.detect_mode(text)
        triggers: List[str] = []
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
        cfg = 1.0
        if risk.score > 0.5:
            cfg = 1.08
        if risk.hand_focus and mode == BodyMode.REALISTIC_HUMAN:
            cfg = max(cfg, 1.12)
        return BodyPlan(
            mode=mode,
            risk=risk,
            positive=_POS.get(mode, ""),
            negative=_NEG.get(mode, ""),
            cfg_bias=cfg,
        )


__all__ = ["AnatomyRisk", "BodyMode", "BodyPlan", "BodyPlanner"]

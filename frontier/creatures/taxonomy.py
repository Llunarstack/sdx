"""
Creature taxonomy — species-aware guidance for monsters, animals, and aliens.

Gives the model a *body plan* instead of human defaults that fight the prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple


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


_FAMILY_RULES: Tuple[Tuple[re.Pattern, CreatureFamily], ...] = (
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


_PLANS: Dict[CreatureFamily, CreaturePlan] = {
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
        for pat, fam in _FAMILY_RULES:
            if pat.search(text):
                return fam
        if re.search(r"\b(creature|monster|beast)\b", text, re.I):
            return CreatureFamily.GENERIC
        return CreatureFamily.GENERIC

    def plan(self, prompt: str) -> Optional[CreaturePlan]:
        fam = self.classify(prompt)
        if fam == CreatureFamily.GENERIC and not re.search(
            r"\b(creature|monster|beast|dragon|alien)\b", prompt or "", re.I
        ):
            return None
        return _PLANS[fam]

    def prompt_fragments(self, prompt: str) -> Tuple[str, str]:
        cp = self.plan(prompt)
        if cp is None:
            return "", ""
        pos = ", ".join(p for p in (cp.limb_hint, cp.surface_hint, cp.silhouette_hint) if p)
        return pos, cp.negative


__all__ = ["CreatureFamily", "CreaturePlan", "CreatureTaxonomy"]

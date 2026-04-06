"""Auto-original-character helpers for prompt-driven OC synthesis."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from random import Random
from typing import Dict, Optional

_OC_TRIGGER_RE = re.compile(
    r"\boriginal character\b|\bmy oc\b|\bnew oc\b|\bcreate (?:an|a) character\b|"
    r"\bcharacter design\b|\bdesign (?:an|a) oc\b|\binvent (?:an|a) character\b|\boc\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AutoOriginalCharacterProfile:
    name: str
    archetype: str
    visual_traits: str
    wardrobe: str
    silhouette: str
    color_motifs: str
    expression_sheet: str
    negative_block: str

    def to_prompt_block(self) -> str:
        bits = [
            f"original character {self.name}, consistent identity across frames",
            f"archetype {self.archetype.replace('_', ' ')}",
            f"signature traits: {self.visual_traits}",
            f"consistent wardrobe: {self.wardrobe}",
            f"silhouette lock: {self.silhouette}",
            f"color motif: {self.color_motifs}",
            f"expression sheet anchors: {self.expression_sheet}",
            "same face structure and hairstyle in every frame",
        ]
        return ", ".join([b for b in bits if b.strip()])


_FIRST_NAMES = (
    "Astra",
    "Nia",
    "Kairo",
    "Sora",
    "Riven",
    "Lyra",
    "Vega",
    "Kael",
    "Mira",
    "Noel",
    "Tali",
    "Rune",
)
_LAST_NAMES = (
    "Vale",
    "Ashen",
    "Kestrel",
    "Morrow",
    "Locke",
    "Voss",
    "Drake",
    "Rowan",
    "Quill",
    "Sterling",
)

_ARCHETYPES = (
    "shonen_lead",
    "cool_rival",
    "mentor",
    "antihero",
    "magical_girl",
    "noir_detective",
    "space_pilot",
)

_ARCHETYPE_KEYWORDS: Dict[str, str] = {
    "noir_detective": "detective noir trench mystery crime",
    "space_pilot": "space sci-fi pilot spaceship mecha cyberpunk",
    "magical_girl": "magical idol sparkle transformation wand",
    "mentor": "mentor teacher master sage veteran",
    "antihero": "antihero rogue assassin dark grim",
    "cool_rival": "rival elite duelist prodigy",
    "shonen_lead": "hero adventure battle protagonist",
}

_STYLE_ARCHETYPE_KEYWORDS: Dict[str, str] = {
    "space_pilot": "anime 3d game sci-fi cyberpunk mecha pbr unreal",
    "noir_detective": "noir ink crosshatch detective",
    "magical_girl": "idol shoujo magical pastel vtuber",
    "cool_rival": "seinen rival edgy dark",
    "shonen_lead": "shonen battle hero action",
}

_TRAITS = (
    "short asymmetric hair",
    "silver streaked bangs",
    "sharp amber eyes",
    "soft heterochromia",
    "cheek scar",
    "ear cuff and utility comms",
    "freckles across nose bridge",
    "distinctive brow shape",
    "signature glove design",
    "small beauty mark under eye",
)

_WARDROBE_BY_ARCHETYPE: Dict[str, tuple[str, ...]] = {
    "space_pilot": ("flight jacket with utility harness", "reinforced boots and tactical gloves"),
    "noir_detective": ("long trench coat and waistcoat", "fedora-inspired headwear with leather holster"),
    "magical_girl": ("iconic transformation outfit with ribbon motifs", "knee-high boots and emblem accessory"),
    "mentor": ("layered coat with ceremonial accents", "minimalist robe and signature pendant"),
    "antihero": ("high-collar asymmetrical coat", "combat boots with utility straps"),
    "cool_rival": ("tailored jacket with geometric trim", "sleek gloves and metallic accessories"),
    "shonen_lead": ("sporty jacket with bold emblem", "battle-ready belt and fingerless gloves"),
}

_SILHOUETTE_BY_ARCHETYPE: Dict[str, str] = {
    "space_pilot": "broad upper silhouette with tapered lower body",
    "noir_detective": "long vertical silhouette with coat-tail rhythm",
    "magical_girl": "clear hourglass silhouette with iconic accessory shapes",
    "mentor": "stable triangular silhouette with grounded stance",
    "antihero": "angular asymmetrical silhouette with sharp shoulder cues",
    "cool_rival": "tall narrow silhouette with precise edge language",
    "shonen_lead": "dynamic V-shape torso with athletic stance",
}

_COLOR_MOTIFS = (
    "teal and orange accents on dark neutral base",
    "crimson and charcoal with silver highlights",
    "cobalt and white with neon accent details",
    "lavender and gold with soft pastel support",
    "forest green and bronze with warm shadows",
    "black and ivory with one signature accent color",
)

_EXPRESSIONS_BY_ARCHETYPE: Dict[str, str] = {
    "space_pilot": "confident smirk, focused glare, determined command shout",
    "noir_detective": "skeptical glance, dry half-smile, intense confrontation stare",
    "magical_girl": "gentle smile, determined shout, radiant heroic expression",
    "mentor": "calm smile, stern warning, reflective concern",
    "antihero": "cold stare, restrained smirk, explosive battle shout",
    "cool_rival": "composed stare, contempt smirk, competitive grin",
    "shonen_lead": "energetic grin, battle shout, resilient determined face",
}

_DEFAULT_NEGATIVE = (
    "identity drift, inconsistent face structure, hairstyle changing between shots, "
    "outfit inconsistency, color motif drift, random extra accessories"
)

_STYLE_TRAIT_BOOSTS: Dict[str, tuple[str, ...]] = {
    "anime 3d game": ("clean anime face planes", "stylized topology-friendly hair silhouette"),
    "ink": ("ink-friendly shape breakup", "strong contour readability under monochrome"),
    "oil": ("painterly facial plane transitions", "brush-character-friendly hair masses"),
}


def prompt_requests_original_character(prompt: str) -> bool:
    p = str(prompt or "").strip()
    if not p:
        return False
    return bool(_OC_TRIGGER_RE.search(p))


def _seeded_rng(prompt: str, seed: int = 0) -> Random:
    h = hashlib.sha256(f"{prompt}|{int(seed)}".encode("utf-8", errors="ignore")).hexdigest()
    return Random(int(h[:16], 16))


def _pick_archetype(prompt: str, style_context: str, rng: Random) -> str:
    p = f"{str(prompt or '').lower()} {str(style_context or '').lower()}"
    for archetype, kw_blob in _ARCHETYPE_KEYWORDS.items():
        if any(k in p for k in kw_blob.split()):
            return archetype
    for archetype, kw_blob in _STYLE_ARCHETYPE_KEYWORDS.items():
        if any(k in p for k in kw_blob.split()):
            return archetype
    return _ARCHETYPES[rng.randrange(len(_ARCHETYPES))]


def infer_auto_original_character(
    prompt: str,
    *,
    seed: int = 0,
    style_context: str = "",
) -> Optional[AutoOriginalCharacterProfile]:
    """
    Return a synthesized OC profile only when prompt asks for OC/character design.
    """
    p = str(prompt or "").strip()
    if not prompt_requests_original_character(p):
        return None
    rng = _seeded_rng(p, seed=seed)
    archetype = _pick_archetype(p, style_context, rng)
    first = _FIRST_NAMES[rng.randrange(len(_FIRST_NAMES))]
    last = _LAST_NAMES[rng.randrange(len(_LAST_NAMES))]
    picked_traits = list(rng.sample(_TRAITS, k=3))
    sc = str(style_context or "").lower()
    for key, extra in _STYLE_TRAIT_BOOSTS.items():
        if key in sc:
            picked_traits.append(extra[rng.randrange(len(extra))])
            break
    traits = ", ".join(picked_traits)
    wardrobe = ", ".join(rng.sample(_WARDROBE_BY_ARCHETYPE.get(archetype, _WARDROBE_BY_ARCHETYPE["shonen_lead"]), k=2))
    silhouette = _SILHOUETTE_BY_ARCHETYPE.get(archetype, _SILHOUETTE_BY_ARCHETYPE["shonen_lead"])
    color_motifs = _COLOR_MOTIFS[rng.randrange(len(_COLOR_MOTIFS))]
    expression_sheet = _EXPRESSIONS_BY_ARCHETYPE.get(archetype, _EXPRESSIONS_BY_ARCHETYPE["shonen_lead"])
    return AutoOriginalCharacterProfile(
        name=f"{first} {last}",
        archetype=archetype,
        visual_traits=traits,
        wardrobe=wardrobe,
        silhouette=silhouette,
        color_motifs=color_motifs,
        expression_sheet=expression_sheet,
        negative_block=_DEFAULT_NEGATIVE,
    )


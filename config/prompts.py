"""Prompt packs and keyword helpers: tag merging, shared negatives/domains, physics, creature and failure-mode packs (stdlib only, plus the minor guard)."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from utils.safety.minor_guard import YOUTH_NEGATIVE


# Keyword matching and comma-tag merging shared by the prompt, style and preset packs (stdlib only).
def word_re(term: str) -> re.Pattern:
    return re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)


def compile_keywords(keywords: Sequence[str]) -> tuple[re.Pattern, ...]:
    return tuple(word_re(k) for k in keywords)


def matches(prompt: str, patterns: Sequence[re.Pattern]) -> bool:
    return any(p.search(prompt) for p in patterns)


def alias_patterns(aliases: Mapping[str, Sequence[str]]) -> tuple[tuple[re.Pattern, str], ...]:
    """``{canonical: (alias, ...)}`` -> ``((alias_regex, canonical), ...)``."""
    return tuple((word_re(alias), canonical) for canonical, group in aliases.items() for alias in group)


def expand_aliases(prompt: str, patterns: Sequence[tuple[re.Pattern, str]]) -> str:
    """Append the canonical term of every alias found, so keyword specs match synonyms."""
    p = str(prompt or "").strip()
    if not p:
        return p
    found = sorted({canonical for pat, canonical in patterns if pat.search(p)})
    return f"{p}, {', '.join(found)}" if found else p


def dedupe_tags(tags: Iterable[str]) -> list[str]:
    """Stripped, non-empty tags without case-insensitive repeats (first casing and order kept)."""
    seen: set[str] = set()
    out: list[str] = []
    for tag in tags:
        t = str(tag).strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def merge_csv_unique(*chunks: str) -> str:
    """Join comma-separated chunks with :func:`dedupe_tags`."""
    return ", ".join(dedupe_tags(part for chunk in chunks for part in str(chunk or "").split(",")))


# Shared prompt strings: default/text-in-image negatives, hard-style prefixes and the anti-AI-look lexicon.
# Used when the negative prompt is left empty. Kept minimal because SDXL-like models over-obey long negatives.
# "text" suppresses stray lettering; prompts that ask for legible text get TEXT_IN_IMAGE_NEGATIVE instead.
DEFAULT_NEGATIVE_PROMPT = (
    "low quality, worst quality, blurry, jpeg artifacts, watermark, signature, text, "
    "bad anatomy, bad hands, missing fingers, extra fingers, fused fingers, deformed, duplicate"
)

# Replaces DEFAULT_NEGATIVE_PROMPT when the prompt asks for text: punishes garbled text but not "text" itself.
TEXT_IN_IMAGE_NEGATIVE = (
    "low quality, worst quality, blurry, jpeg artifacts, watermark, signature, "
    "garbled text, misspelled, wrong spelling, illegible, unreadable, "
    "bad anatomy, bad hands, deformed, duplicate"
)

# Phrases that mean the prompt wants legible lettering (switches to TEXT_IN_IMAGE_NEGATIVE).
TEXT_IN_IMAGE_PHRASES = (
    "sign that says",
    "sign says",
    "text that says",
    "text reads",
    'says "',
    'reads "',
    "lettering",
    "written",
    "caption",
    "label",
    "headline",
    "title says",
    "subtitle",
    "words ",
    "phrase ",
    "spelling",
    "legible",
    "readable text",
    "clear text",
)


# ``--hard-style`` prefixes: models blur 3D vs 2D, realistic vs illustrated, and mixed styles (2.5D).
HARD_STYLE_RECOMMENDED_PROMPTS = {
    "3d": [
        "3d render, octane render, masterpiece, best quality, 8k",
        "3d illustration, blender, solid shading, clean 3d, detailed",
        "cg, 3d model, rendered, subsurface scattering, high quality",
    ],
    "realistic": [
        "photorealistic, raw photo, natural lighting, masterpiece, 8k uhd",
        "hyperrealistic, real photography, detailed skin, skin texture, dslr",
        "realistic, real life, film grain, depth of field, sharp focus",
    ],
    "3d_realistic": [
        "3d render, photorealistic, octane render, natural lighting, masterpiece",
        "realistic 3d, hyperrealistic cg, detailed skin, solid shading, 8k",
    ],
    "style_mix": [
        "2.5d, semi-realistic, masterpiece, best quality, detailed",
        "photorealistic anime, realistic anime, stylized realistic, hybrid style",
        "3d anime, anime 3d, mixed style, detailed, sharp focus",
    ],
}


# The one anti-AI-look lexicon (content_controls anti-AI packs and --naturalize read these).
# Grain and sensor noise are not requested here: the capture finish (utils/quality/finishing.py)
# adds them physically, and ARTIFACT_NEGATIVES keeps generated noise out of the sample.
ANTI_AI_LOOK_NEGATIVE = (
    "oversaturated, plastic skin, smooth skin, airbrushed, waxy, doll-like, "
    "perfect skin, flawless skin, synthetic, artificial, CGI, uncanny, "
    "AI generated look, neural render aesthetic, digital art look, overly smooth, porcelain skin, plastic look, "
    "hyperreal plastic, hdr glow, halo artifacts, oversharpened, "
    "too perfect symmetry, stock photo stiffness, candy saturation, glossy unreal skin"
)

# Stronger de-AI / de-CG pack (--anti-ai-pack strong, --naturalize-deep).
ANTI_AI_LOOK_NEGATIVE_STRONG = (
    ANTI_AI_LOOK_NEGATIVE + ", midjourney look, generic AI illustration, same-face syndrome, "
    "floating specular highlights, subsurface abuse, waxy teeth, dead glassy eyes, "
    "uniform poreless skin, beauty filter face, snapchat smooth skin, "
    "tilt-shift miniature fake, fake lens blur, computational hdr, "
    "neon edge glow, cfg burn edges, posterization, gradient banding"
)

# Positive: subtle hints for a more natural, photographic feel (prepended with --naturalize).
NATURAL_LOOK_POSITIVE = (
    "natural skin texture, subtle imperfections, raw photo, natural lighting, "
    "believable shadows, organic micro contrast, real camera response"
)

# Deeper natural / human-capture cues (--naturalize-deep, --anti-ai-pack strong).
NATURAL_LOOK_POSITIVE_DEEP = (
    NATURAL_LOOK_POSITIVE + ", authentic photograph, candid capture, imperfect but real, "
    "shot on real camera, natural white balance, photographic depth"
)

# When stacking LoRAs: negatives that reduce muddy fusion / double-style (pair with lower strengths).
LORA_STACK_NEGATIVE = (
    "double exposure style clash, conflicting adaptations, merged incompatible styles, "
    "lora artifact seams, ghosting from adaptation, oversaturated lora bloom, muddy blend"
)

# Concept bleeding (colors/objects bleed into each other, e.g. red shirt + blue pants → purple).
CONCEPT_BLEEDING_NEGATIVE = "color bleed, mixed colors, blended colors, muddy colors, merged objects, fused objects"
CONCEPT_BLEEDING_POSITIVE = "distinct colors, separate colors, clear separation, no color bleed, defined edges"

# Artifacts: white dots, speckles, spiky/pixel-stretch, grain in dark areas (Illustrious, SDXL spiky bug).
ARTIFACT_NEGATIVES = (
    "white dots, speckles, particles, noise in dark, spiky artifacts, pixel stretch, "
    "stretched pixels, compression artifacts, jpeg artifacts, dot pattern, sensor noise"
)

# Stubborn watermarks (e.g. Illustrious baked-in logos); use when negative prompting alone isn't enough.
WATERMARK_NEGATIVE_STRONG = (
    "watermark, logo, signature, text in corner, corner logo, brand, stamp, "
    "baked in text, stylized text, arknights logo, game logo"
)

# Repetitive / "default" face (Flux face, same face every time).
FLUX_FACE_DIVERSITY_NEGATIVE = "repetitive face, same face, default face, generic face, clone face, identical face"
DIVERSITY_POSITIVE = "unique face, diverse features, distinct face, varied expression, individual"


# Prompt packs for physics-heavy scenes: fluids, transparency, rigid/soft contact, lighting.
PHYSICS_MATERIAL_RECOMMENDED_PROMPTS_BY_DOMAIN = {
    "fluids": [
        "clear liquid surface, meniscus, coherent reflections on water, believable specular highlights",
        "fluid simulation quality, natural splash shape, volume conservation cues, wet surface contact line",
        "underwater caustics subtle, refractive distortion through water, air-water interface sharp",
        "pouring liquid, laminar flow then breakup, droplets with highlight and shadow",
    ],
    "transparency": [
        "transparent glass, correct refraction, visible background distortion through glass, thin-edge highlights",
        "frosted glass, diffuse transmission, soft silhouette, believable thickness",
        "clear acrylic, fresnel reflections, double reflection on curved surface, clean edges",
        "translucent material, subsurface scatter subtle, light transmission, depth-consistent opacity",
    ],
    "physics_materials": [
        "physically plausible lighting, consistent shadow direction, contact shadows, ambient occlusion",
        "material-consistent reflections, roughness matches surface, specular on wet vs dry regions",
        "rigid body contact, stable stacking, weight believable, no interpenetration",
        "single light narrative, coherent bounce light, no contradictory shadow sources",
    ],
    "soft_bodies": [
        "fabric drapes under gravity, folds follow tension, hem obeys weight, cloth collision with body",
        "soft deformation, compression where supported, natural wrinkles, no clipping through skin",
        "hair strands grouped, gravity on long hair, flyaways subtle, lighting on hair volume",
        "elastic material, stretch localized, no melting merge with background",
    ],
}


# Appended to the negative when the prompt mentions fluids / glass / physics.
PHYSICS_COMMON_NEGATIVE_ADDON = (
    "impossible physics, merged materials, contradictory lighting, floating objects, "
    "wrong refraction, flat liquid, plastic water, pasted transparency, interpenetration, "
    "muddy fluids, incoherent splash, duplicate reflections, shadow direction mismatch"
)


# Prompt packs for creatures, anthro/furry characters, robots/mechs and humanoid monsters.
# Danbooru-style tag vocabulary; multi-word phrases are fine.
HUMANOID_MONSTER_PROMPT_KEYWORDS = (
    "orc",
    "ogre",
    "troll",
    "goblin",
    "kobold",
    "demon",
    "devil",
    "imp",
    "daemon",
    "succubus",
    "incubus",
    "cambion",
    "vampire",
    "dhampir",
    "nosferatu",
    "angel",
    "archangel",
    "seraph",
    "fallen angel",
    "undead",
    "zombie",
    "lich",
    "revenant",
    "wraith",
    "banshee",
    "phantom",
    "ghost",
    "specter",
    "spectre",
    "werewolf",
    "lycanthrope",
    "lycan",
    "monster",
    "eldritch",
    "abomination",
    "tiefling",
    "aasimar",
    "drow",
    "dark elf",
    "fae",
    "faerie",
    "fairy",
    "pixie",
    "sprite",
    "dryad",
    "nymph",
    "oni",
    "yokai",
    "kitsune",
    "kumiho",
    "nine-tailed",
    "harpy",
    "lamia",
    "arachne",
    "gargoyle",
    "sphinx",
    "siren",
    "scylla",
    "charybdis",
    "medusa",
    "gorgon",
    "cyclops",
    "minotaur",
    "centaur",
    "satyr",
    "faun",
    "naga",
    "slime girl",
    "cow girl",
    "cat girl",
    "wolf girl",
    "dragon girl",
    "demon girl",
    "angel girl",
    "vampire girl",
    "body horror",
    "drider",
    "dullahan",
    "headless",
    "rakshasa",
    "djinn",
    "genie",
    "ifrit",
    "efreet",
    "wendigo",
    "skinwalker",
    "baphomet",
    "death knight",
    "skeletal dragon",
    "frost giant",
    "fire giant",
    "stone giant",
)

CREATURE_CHARACTER_RECOMMENDED_PROMPTS_BY_DOMAIN = {
    "creatures": [
        "fantasy creature, coherent anatomy, species-consistent silhouette, clear muscle or carapace read",
        "mythical beast, four legs grounded, believable weight, tail and spine alignment natural",
        "alien wildlife, non-human proportions intentional, textured skin or hide, sharp focus eyes",
        "dragon or griffin, wing attachment believable, scales or feathers distinct from mammal fur",
        "hydra or multi-headed beast, necks branch from shared torso, heads distinct, scales consistent",
        "chimera, lion-goat-snake regions visually separated, coherent hide and fur transitions",
        "undersea creature, bioluminescence optional, fins and eyes readable, pressure-adapted silhouette",
        "insectoid alien, exoskeleton plates, jointed legs symmetric, compound eyes stylized or realistic pick one",
    ],
    "anthro_furry": [
        "anthropomorphic character, humanoid body plan, animal head species-consistent, digitigrade or plantigrade feet clear",
        "furry character, fur texture directional, muzzle and ears species read, hands with pads or claws consistent",
        "kemono style, blend animal face with human posture, clothing fits snout and ears",
        "anthro wolf, thick neck fur ruff, tail pose follows spine, no human ears duplicate",
        "scalie anthro, dragon or lizard muzzle, scales on cheeks and brows, horns optional single design",
        "avian anthro, beak shape species-consistent, feathered arms or wings attach at back, no mammal nose",
        "protogen or synth furry, visor face, mechanical ears optional, fur meets tech at clean seams",
        "pooltoy or inflatable anthro (stylized), seams and valve readable, vinyl highlight consistent",
    ],
    "robots_mechs": [
        "robot character, hard-surface panels, panel lines, mechanical joints, seams and bolts readable",
        "mecha, armor plating, hydraulic detail, no organic skin on metal, emissive eyes optional",
        "android, synthetic skin vs exposed mechanism boundary clear, uncanny valley intentional if human-like face",
        "cyborg, metal limb attachment site believable, cable routing, flesh-to-chrome transition sharp",
        "power armor or exosuit, pilot limbs inside shell, joint limits believable, helmet read",
        "industrial drone or walker, sensor array, landing gear, scale vs human reference",
        "gothic or baroque robot, ornate metal filigree, still reads as machined not carved flesh",
        "nanite swarm humanoid silhouette, edge glow optional, boundary surface coherent",
    ],
    "humanoid_monsters": [
        "humanoid monster, exaggerated but coherent limbs, horror lighting, skin or chitin material consistent",
        "orc or ogre, heavy brow, tusks and jaw proportion, green or gray skin tone even, practical costume",
        "vampire or undead, pallor consistent, eye detail, no melted face merge with background",
        "eldritch humanoid, wrong geometry intentional, single focal silhouette, readable pose",
        "succubus or incubus, horns placement consistent, wings attach at back believably, tail from coccyx, demonic features readable",
        "demon humanoid, skin tone even, claws and fangs proportion clear, burning eyes optional, no random extra faces",
        "tiefling or cambion, horn shape single design, tail length consistent, fantasy costume fits horns",
        "angel or fallen angel, feather wings layered, halo optional, serene or corrupted expression matches theme",
        "dark elf or drow, pointed ears one pair, pale or ash skin even, white or silver hair coherent",
        "yokai humanoid, oni horns, kitsune ears and tail count explicit, kimono or fantasy wear fits ears",
        "dhampir or vampire lord, subtle fangs, red-eye or normal eye pick one, aristocratic or feral read",
        "lamia or naga lower body, human torso scale matches serpent coils, seamless scale transition",
        "arachne or drider, spider abdomen attachment clear, extra legs symmetric, human torso upright",
        "dullahan or headless rider, neck stump or floating head pick one, consistent myth read",
        "rakshasa or tiger-demon humanoid, reversed hands if lore-accurate, ornate fantasy dress",
        "djinn or ifrit humanoid, smoke or flame hair optional, jewelry and skin tone even",
        "wendigo or emaciated horror humanoid, elongated limbs one axis, antlers readable, gaunt face",
        "death knight or skeletal champion, armor over bone, eyes glow one color, frost or unholy aura optional",
    ],
}


CREATURE_COMMON_NEGATIVE_ADDON = (
    "wrong species mix, melted anatomy, duplicate ears, floating limbs, merged fingers, "
    "inconsistent tail, plastic fur, rubber mech, flesh-metal confusion, unreadable silhouette"
)

# SFW: general-audience character art; the negative pushes away from adult themes and minors.
CREATURE_SFW_CONTEXT_POSITIVE = (
    "sfw, general audience, character design, wholesome, adventure, expressive pose, "
    "costume appropriate, clear silhouette, safe for work, family friendly"
)

CREATURE_SFW_NEGATIVE_ADDON = (
    "nsfw, nude, naked, explicit, sexual, erotic, pornographic, suggestive, "
    "minor, child, loli, shota, underage, school uniform fetish, "
    "wardrobe malfunction, see-through clothes, sheer fabric focus"
)

# NSFW: adult-only; the negative always carries YOUTH_NEGATIVE and the same topology rules as SFW.
CREATURE_NSFW_CONTEXT_POSITIVE = (
    "adult character, mature themes, explicit anatomy coherent with species, "
    "consistent body plan, same tag vocabulary as training data, 18+, adult only"
)

CREATURE_NSFW_NEGATIVE_ADDON = (
    f"{YOUTH_NEGATIVE}, "
    "melted anatomy, fused limbs, impossible joints, duplicate torsos, merged bodies, "
    "species-anatomy mismatch, rubber skin, identity blur between characters, "
    "wrong hole topology, extra genitals, merged partners, floating hands, disembodied limbs"
)

# Per-domain positives with the rating baked in.
CREATURE_CHARACTER_SFW_RECOMMENDED_PROMPTS_BY_DOMAIN = {
    "creatures": [
        "sfw fantasy creature, family-friendly design, readable expression, no gore",
        "creature concept art, adventure rpg style, heroic or cute tone, clean anatomy",
        "cute beast companion, pet-like proportions, soft lighting, storybook illustration",
        "mount or steed design, tack and saddle readable, rider optional sfw, hooves or paws clear",
        "zoo exhibit style creature, educational plate, neutral background, labeled anatomy optional",
        "pokemon-like stylized monster, chunky silhouette, big eyes, non-threatening",
        "digimon-like partner creature, armor panels as shell, friendly expression",
        "mtg-style fantasy beast, card art composition, readable at small size",
    ],
    "anthro_furry": [
        "sfw anthro character, dressed, expressive, commission-friendly, species clear",
        "furry character art, clothed, dynamic pose, mascot or story illustration tone",
        "office worker anthro, business casual, lanyard or mug prop, slice of life",
        "adventurer anthro, cloak and pack, sword sheathed, rpg party member",
        "idol or pop-star anthro, stage costume modest, microphone, sparkles sfw",
        "sports anthro, jersey and shorts, team colors, action pose non-sexual",
        "winter anthro, coat scarf mittens, breath visible cold air, cozy",
        "beach anthro sfw, swim trunks or one-piece, towel, volleyball optional",
    ],
    "robots_mechs": [
        "sfw robot character, mecha pilot suit or full machine, sci-fi adventure, no horror gore",
        "friendly android or industrial robot, clear hard-surface read, character sheet friendly",
        "delivery drone bot, box payload, urban rooftop, soft daylight",
        "medical assistant android, clean white shell, display face, hospital corridor",
        "farm or construction mech, utilitarian paint, warning stripes, dust optional",
        "retro tin toy robot, rivets, antenna, primary colors, nostalgic",
        "steampunk automaton, brass and leather trim, pressure gauges, no body horror",
        "space station repair bot, magnetic feet, tool arms, starfield window",
    ],
    "humanoid_monsters": [
        "sfw monster design, stylized horror or fantasy villain, no extreme gore, readable silhouette",
        "humanoid creature, game enemy or npc design, menacing but not explicit",
        "sfw demon or angel character, costume modest, wings and horns clear, no suggestive pose",
        "sfw tiefling or aasimar, ttrpg hero portrait, horns or halo readable, clothed adventurer",
        "sfw vampire hunter gear, cross and cloak, pale npc ally, no blood spray",
        "sfw zombie npc, green grey skin, torn but clothed, comedy or cartoon tone",
        "sfw orc merchant, apron and scales, market stall, friendly expression",
        "sfw yokai festival mask, kimono, lantern, summer night street",
        "sfw slime companion, transparent but opaque core, no adult body emphasis",
        "sfw harpy bard, feathered wings folded, lute, modest tunic",
    ],
}

CREATURE_CHARACTER_NSFW_RECOMMENDED_PROMPTS_BY_DOMAIN = {
    "creatures": [
        "adult fantasy creature, mature explicit scene, anatomy matches species, coherent limbs",
        "non-human lover or companion design, explicit but topology-consistent, clear eyes and muzzle",
        "explicit dragon or griffin mount, rider contact points clear, scales do not melt into skin",
        "adult lamia or naga intimate, coils support weight, human torso scale consistent",
        "adult centaur explicit, equine and human junction seamless, tail and mane coherent",
        "adult alien creature explicit, non-human erogenous layout intentional, single subject focus",
        "tentacle creature mature explicit, suction cups pattern consistent, consenting adult partner",
        "slime monster adult explicit, transparent layers readable, core silhouette stable",
    ],
    "anthro_furry": [
        "adult anthro, mature explicit, fur and skin boundaries clear, muzzle species-locked, paws consistent",
        "explicit furry character, adult body proportions, tail attachment believable, no duplicate ears",
        "adult scalie explicit, ventral scales vs belly fur transition sharp, claws optional",
        "adult avian anthro explicit, beak does not replace wrong anatomy, wings fold clear",
        "adult kemono bara or slim build explicit, muscle or soft body one read, fur direction",
        "collar and leash adult anthro explicit, pet play tone, still humanoid hands if intended",
        "latex or rubber suit anthro explicit, shine vs fur boundary, zipper or seam",
        "explicit aftercare cuddling anthro, relaxed pose, fur clumping wet optional",
    ],
    "robots_mechs": [
        "adult android explicit, synthetic skin seams vs panel lines clear, mechanical parts intentional",
        "mature mecha pilot scene, suit or partial armor, human and machine boundaries readable",
        "gynoid or masculine android explicit, chassis vents and ports placed consistently",
        "partial dismantling android explicit, exposed servos, cables routed, no random flesh blobs",
        "power armor open hatch explicit, pilot skin vs interior padding boundary",
        "chrome doll android explicit, mirror highlights, panel gaps at joints only",
        "repair scene android explicit, tools, diagnostic light, intimate but readable workshop",
        "hive drone humanoid explicit, uniform chassis marks, serial subtle, same model twins",
    ],
    "humanoid_monsters": [
        "adult monster explicit, horror-mature tone, exaggerated but coherent anatomy, single focal subject",
        "mature demon or orc scene, explicit anatomy consistent with fantasy species, readable pose",
        "adult succubus or incubus explicit, wings and tail topology coherent, horns match reference design",
        "mature vampire or dhampir explicit, fangs consistent, pallor even, eyes one style",
        "adult tiefling explicit, horn and tail anchors believable, skin tint even, fantasy race readable",
        "adult angel or fallen explicit, feather mess on sheets optional, halo off or cracked",
        "adult drow or dark elf explicit, underdark glow mushrooms, skin ash even",
        "adult oni or ogre explicit, heavy muscle read, skin tone even, tusks proportion",
        "adult arachne or drider explicit, spider segment joins human waist, legs symmetric",
        "adult dullahan explicit, head held or nearby, neck stump or magic flame pick one",
        "adult werewolf mid-transformation explicit, fur growth direction, claws emerging",
        "adult eldritch humanoid explicit, wrong angles intentional, still one penetrative read",
    ],
}


# Taxonomy of common generative-image failure modes with prompt/negative fragments that counter them.
MitigationMode = Literal["none", "auto", "all"]


@dataclass(frozen=True, slots=True)
class ShortcomingSpec:
    id: str
    keywords: tuple[str, ...]
    positive_hints: str
    negative_hints: str
    is_2d_style: bool = False


# Order matters for stable merge of hints (photoreal + digital + CG, then stylized 2D-specific).
SHORTCOMING_SPECS: tuple[ShortcomingSpec, ...] = (
    ShortcomingSpec(
        id="skin_detail_tangents",
        keywords=(
            "portrait",
            "skin",
            "close-up",
            "closeup",
            "freckles",
            "complexion",
            "face",
            "beauty",
        ),
        positive_hints=(
            "natural skin texture, subtle skin detail, soft subsurface scattering, clear overlap and separation of forms"
        ),
        negative_hints=(
            "plastic skin, wax skin, airbrushed, overly smooth skin, doll-like skin, confusing tangents, merged edges"
        ),
    ),
    ShortcomingSpec(
        id="spatial_support",
        keywords=(
            "sitting",
            "leaning",
            "chair",
            "couch",
            "holding",
            "gripping",
            "standing",
            "feet",
            "ground",
            "floor",
            "table",
            "drape",
            "fabric",
        ),
        positive_hints="believable weight and contact, feet planted, fabric creases and drape, hands gripping objects naturally",
        negative_hints="floating, hovering, no ground contact, clipping, stiff cloth, floating hands",
    ),
    ShortcomingSpec(
        id="lighting_gi",
        keywords=(
            "sunlight",
            "studio",
            "neon",
            "lamp",
            "shadow",
            "night",
            "interior",
            "room",
            "window",
            "cinematic",
            "rim light",
        ),
        positive_hints="single coherent light direction, soft bounce light, subtle color bleed, ambient occlusion in contact areas",
        negative_hints="contradictory shadows, inconsistent lighting, disconnected elements, flat ambient",
    ),
    ShortcomingSpec(
        id="line_edges",
        keywords=("lineart", "line art", "ink", "comic", "manga", "outline", "crosshatch", "etching"),
        positive_hints="varied line weight, tapered strokes, lost and found edges, readable silhouette",
        negative_hints="uniform outline, deep-fried edges, over-defined contour everywhere, mechanical line weight",
    ),
    ShortcomingSpec(
        id="anatomy_deep",
        keywords=(
            "hand",
            "hands",
            "fingers",
            "pose",
            "full body",
            "limbs",
            "elbow",
            "knee",
            "shoulder",
            "muscular",
            "anatomy",
        ),
        positive_hints="coherent skeleton and joints, natural limb attachment, correct hands and fingers",
        negative_hints="noodle limbs, broken joints, twisted anatomy, impossible pose, fused fingers",
    ),
    ShortcomingSpec(
        id="composition_flow",
        keywords=("landscape", "cityscape", "wide shot", "poster", "cover", "scene", "crowd", "panorama"),
        positive_hints="rule of thirds, intentional negative space, clear focal hierarchy, visual rest",
        negative_hints="cluttered composition, horror vacui, everything centered, busy corners",
    ),
    ShortcomingSpec(
        id="materials_texture",
        keywords=("metal", "rust", "leather", "marble", "wood", "fabric", "glass", "stone", "armor", "weathered"),
        positive_hints="material-consistent texture, wear where physics and moisture collect, believable folds and highlights",
        negative_hints="random noise texture, plastic everywhere, inconsistent material reads",
    ),
    ShortcomingSpec(
        id="narrative_wear",
        keywords=("battle", "adventurer", "mud", "dust", "traveler", "ruins", "vintage", "worn", "torn", "stained"),
        positive_hints="story-consistent wear and dirt, directional weathering",
        negative_hints="generic scratches, random grime, incoherent damage",
    ),
    ShortcomingSpec(
        id="perspective_foreshortening",
        keywords=("foreshortening", "fisheye", "wide angle", "from above", "from below", "dramatic angle", "pov"),
        positive_hints="coherent perspective, consistent scale in depth, readable overlapping forms",
        negative_hints="warped limbs, broken perspective, mushy depth",
    ),
    ShortcomingSpec(
        id="facial_nuance",
        keywords=("smile", "expression", "emotive", "laughing", "crying", "portrait", "eyes", "gaze"),
        positive_hints="natural facial asymmetry, subtle micro-expression, lifelike eyes",
        negative_hints="perfect symmetry, mask-like expression, dead eyes, uncanny valley face",
    ),
    ShortcomingSpec(
        id="color_value_discipline",
        keywords=("monochrome", "muted", "film", "noir", "zorn", "limited palette", "desaturated", "earth tones"),
        positive_hints="disciplined palette, coherent value structure, mood-consistent grading",
        negative_hints="rainbow oversaturation, neon clutter, chaotic values, posterized noise colors",
    ),
    ShortcomingSpec(
        id="extra_gaps",
        keywords=("typography", "lettering", "sign", "logo", "pattern", "plaid", "stripes", "splash", "smoke", "water"),
        positive_hints="consistent small repeating details, legible intentional text where required",
        negative_hints="garbled text, inconsistent pattern, stiff fluid, merged distant faces",
    ),
    ShortcomingSpec(
        id="cg_render_geometry",
        keywords=(
            "blender",
            "octane",
            "unreal",
            "3d render",
            "cgi",
            "raytrace",
            "subsurface scattering",
            "keyshot",
            "marmoset",
            "substance painter",
            "cycles render",
        ),
        positive_hints="believable hard-surface contact, consistent material response, readable form",
        negative_hints="melting mesh, fused geometry, swimming textures, impossible topology",
    ),
    # Digital painting / illustration / games / UI (cel-anime lives in the is_2d_style specs below).
    ShortcomingSpec(
        id="digital_painting",
        keywords=(
            "digital painting",
            "digital art",
            "digital illustration",
            "photoshop",
            "procreate",
            "clip studio",
            "paint tool sai",
            "krita",
            "corel painter",
            "wacom",
            "speedpaint",
            "speed painting",
            "paintover",
            "drawing tablet",
        ),
        positive_hints=(
            "intentional brushwork, readable stroke economy, coherent edge control, "
            "layer-aware color mixing, purposeful soft and hard transitions"
        ),
        negative_hints=(
            "plastic airbrush smear, uniform over-smoothing, muddy midtones, AI soup blend, "
            "soulless gradient fill, noise pretending to be texture"
        ),
    ),
    ShortcomingSpec(
        id="concept_matte_digital",
        keywords=(
            "concept art",
            "key art",
            "matte painting",
            "photobash",
            "photo bash",
            "environment design",
            "keyframe",
            "artstation",
            "set extension",
            "digital collage",
        ),
        positive_hints=(
            "clear design read, consistent perspective across photobashed elements, "
            "matched grain and color cast, intentional focal hierarchy"
        ),
        negative_hints=(
            "floating cutouts, scale mismatch between assets, conflicting light directions, "
            "generic grey sculpt mush, template composition"
        ),
    ),
    ShortcomingSpec(
        id="pixel_digital",
        keywords=(
            "pixel art",
            "pixelart",
            "sprite",
            "spritesheet",
            "sprite sheet",
            "8-bit",
            "16-bit",
            "32-bit",
            "retro game",
            "nes style",
            "snes",
            "gameboy",
            "isometric pixel",
        ),
        positive_hints=(
            "crisp pixel grid, disciplined limited palette, clean clusters, "
            "intentional dithering or AA where appropriate"
        ),
        negative_hints=(
            "blurry between pixels, accidental gradient ramps, subpixel smear, "
            "inconsistent pixel scale, soft anti-alias where hard pixels are required"
        ),
    ),
    ShortcomingSpec(
        id="vector_flat_digital",
        keywords=(
            "vector",
            "vector art",
            "flat design",
            "flat illustration",
            "app icon",
            "ui illustration",
            "infographic",
            "logo design",
            "svg style",
            "material design",
        ),
        positive_hints=(
            "clean vector shapes, consistent corner and stroke discipline, deliberate flat fills, "
            "readable icon silhouette"
        ),
        negative_hints=(
            "wobbly anchors, accidental 3D or photoreal bleed, fuzzy anti-aliased mush, "
            "inconsistent stroke width, messy boolean overlaps"
        ),
    ),
    ShortcomingSpec(
        id="stylized_game_digital",
        keywords=(
            "hand-painted texture",
            "hand painted texture",
            "stylized 3d",
            "stylized 3d render",
            "game asset",
            "texture sheet",
            "albedo map",
            "character turnaround",
            "model sheet",
            "low poly",
            "hand-painted",
        ),
        positive_hints=(
            "readable stylized materials, consistent texel density, clear albedo vs lighting read, "
            "cohesive hand-painted surface direction"
        ),
        negative_hints=(
            "muddy albedo, random noise mud, swimming hand-painted detail, inconsistent brush scale across UV shells"
        ),
    ),
    # Stylized 2D specs: only considered when the caller enables the 2D pack.
    ShortcomingSpec(
        id="style_drift_2d",
        keywords=("anime", "manga", "cel shaded", "cel-shaded", "visual novel", "light novel", "gacha"),
        positive_hints="consistent stylization, single coherent 2D style across the image, stable character design cues",
        negative_hints="style drift, mixed realism and toon, inconsistent proportions between regions",
        is_2d_style=True,
    ),
    ShortcomingSpec(
        id="line_art_2d",
        keywords=("anime", "manga", "cel", "toon", "cartoon", "chibi"),
        positive_hints="clean intentional line art, hair strand clarity, garment contour flow",
        negative_hints="extraneous outlines, inconsistent hair lines, mechanical uniform ink",
        is_2d_style=True,
    ),
    ShortcomingSpec(
        id="cel_lighting_2d",
        keywords=("cel", "flat color", "anime shading", "toon shading", "manga"),
        positive_hints="hard-edged shadow shapes that follow form, disciplined flat colors, coherent toon light",
        negative_hints="random soft gradients in flat-style areas, contradictory toon shadows",
        is_2d_style=True,
    ),
    ShortcomingSpec(
        id="anatomy_2d",
        keywords=("chibi", "anime", "manga", "super deformed"),
        positive_hints="consistent stylized proportions, stable eye highlights, readable stylized pose",
        negative_hints="inconsistent eye size, broken stylized limbs, floating hands in action pose",
        is_2d_style=True,
    ),
    ShortcomingSpec(
        id="composition_2d",
        keywords=("manga", "comic", "cover art", "key visual", "splash art", "pin-up"),
        positive_hints="panel-like clarity, generous negative space where appropriate, strong silhouette",
        negative_hints="overcrowded composition, weak focal read in stylized scene",
        is_2d_style=True,
    ),
    ShortcomingSpec(
        id="medium_2d",
        keywords=("watercolor", "halftone", "screentone", "ink wash", "marker", "copics"),
        positive_hints="medium-consistent texture and bleed, paper or print logic",
        negative_hints="plastic smoothness, random noise pretending to be medium",
        is_2d_style=True,
    ),
    ShortcomingSpec(
        id="extra_2d",
        keywords=("comic strip", "children book", "picture book", "graphic novel"),
        positive_hints="wholesome stylized clarity, consistent line and fill discipline",
        negative_hints="unwanted photoreal drift, kitsch over-polish, garbled lettering in bubbles",
        is_2d_style=True,
    ),
)

_SHORTCOMING_PATTERNS: dict[str, tuple[re.Pattern, ...]] = {
    s.id: compile_keywords(s.keywords) for s in SHORTCOMING_SPECS
}


def spec_by_id(sid: str) -> ShortcomingSpec:
    for s in SHORTCOMING_SPECS:
        if s.id == sid:
            return s
    raise KeyError(sid)


def _eligible(include_2d: bool) -> list[ShortcomingSpec]:
    return [s for s in SHORTCOMING_SPECS if include_2d or not s.is_2d_style]


def detect_shortcoming_ids(prompt: str, *, include_2d: bool) -> tuple[str, ...]:
    """Ids whose keywords match ``prompt`` (word-boundary, case-insensitive)."""
    if not prompt or not prompt.strip():
        return ()
    return tuple(s.id for s in _eligible(include_2d) if matches(prompt, _SHORTCOMING_PATTERNS[s.id]))


def mitigation_fragments(
    prompt: str,
    mode: MitigationMode,
    *,
    include_2d_pack: bool,
) -> tuple[str, str]:
    """``(positive_suffix, negative_suffix)`` for ``mode``; both empty for ``"none"`` or no match.

    ``"auto"`` uses only the specs whose keywords appear in ``prompt``; ``"all"`` uses every eligible spec.
    """
    if mode == "all":
        specs = _eligible(include_2d_pack)
    elif mode == "auto":
        specs = [spec_by_id(i) for i in detect_shortcoming_ids(prompt, include_2d=include_2d_pack)]
    else:
        return "", ""
    if not specs:
        return "", ""
    pos = merge_csv_unique(*(s.positive_hints for s in specs))
    neg = merge_csv_unique(*(s.negative_hints for s in specs))
    return pos, neg

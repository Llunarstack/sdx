"""Prompt packs for creatures, anthro/furry characters, robots/mechs and humanoid monsters.

These are the domains where diffusion mixes species, melts fur into skin or confuses hard surface with
flesh, so every line states species, posture and material boundaries. Selected by
``utils.prompt.subject_addons.suggest_creature_prompt_addons``.
"""

from __future__ import annotations

from utils.safety.minor_guard import YOUTH_NEGATIVE

__all__ = [
    "HUMANOID_MONSTER_PROMPT_KEYWORDS",
    "CREATURE_CHARACTER_RECOMMENDED_PROMPTS_BY_DOMAIN",
    "CREATURE_COMMON_NEGATIVE_ADDON",
    "CREATURE_SFW_CONTEXT_POSITIVE",
    "CREATURE_SFW_NEGATIVE_ADDON",
    "CREATURE_NSFW_CONTEXT_POSITIVE",
    "CREATURE_NSFW_NEGATIVE_ADDON",
    "CREATURE_CHARACTER_SFW_RECOMMENDED_PROMPTS_BY_DOMAIN",
    "CREATURE_CHARACTER_NSFW_RECOMMENDED_PROMPTS_BY_DOMAIN",
]


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

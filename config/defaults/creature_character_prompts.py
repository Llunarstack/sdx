"""
Prompt packs for **creatures**, **anthropomorphic / furry-style characters**, **robots & mechs**,
and **humanoid monsters** — domains where diffusion often mixes species, melts fur into skin, or
confuses hard-surface with flesh.

Use clear **species + posture + material** boundaries in training captions and at inference.

**Tag-style keywords** (``HUMANOID_MONSTER_PROMPT_KEYWORDS``) overlap with vocabulary common on
tag-based image boards (Danbooru-style ``*_tag`` conventions); use the same tokens in training
and inference for better adherence.
"""

from __future__ import annotations

__all__ = [
    "HUMANOID_MONSTER_PROMPT_KEYWORDS",
    "CREATURE_CHARACTER_DOMAIN_NAMES",
    "CREATURE_CHARACTER_RECOMMENDED_PROMPTS_BY_DOMAIN",
    "CREATURE_CHARACTER_RECOMMENDED_NEGATIVE_BY_DOMAIN",
    "CREATURE_COMMON_NEGATIVE_ADDON",
    "CREATURE_SFW_CONTEXT_POSITIVE",
    "CREATURE_SFW_NEGATIVE_ADDON",
    "CREATURE_NSFW_CONTEXT_POSITIVE",
    "CREATURE_NSFW_NEGATIVE_ADDON",
    "CREATURE_CHARACTER_SFW_RECOMMENDED_PROMPTS_BY_DOMAIN",
    "CREATURE_CHARACTER_NSFW_RECOMMENDED_PROMPTS_BY_DOMAIN",
    "CREATURE_CHARACTER_SFW_RECOMMENDED_NEGATIVE_BY_DOMAIN",
    "CREATURE_CHARACTER_NSFW_RECOMMENDED_NEGATIVE_BY_DOMAIN",
    "CREATURE_NSFW_KEYWORDS",
    "CREATURES_PROMPT_TIPS",
    "ANTHRO_FURRY_PROMPT_TIPS",
    "ROBOTS_MECH_PROMPT_TIPS",
    "HUMANOID_MONSTERS_PROMPT_TIPS",
    "CREATURE_TRAINING_CAPTION_TIPS",
    "CREATURE_SFW_SCENARIO_TIPS",
    "CREATURE_NSFW_SCENARIO_TIPS",
    "ANTHRO_FURRY_SFW_TIPS",
    "ANTHRO_FURRY_NSFW_TIPS",
    "ROBOTS_MECH_SFW_TIPS",
    "ROBOTS_MECH_NSFW_TIPS",
    "HUMANOID_MONSTERS_SFW_TIPS",
    "HUMANOID_MONSTERS_NSFW_TIPS",
    "CREATURES_SFW_TIPS",
    "CREATURES_NSFW_TIPS",
]

CREATURE_CHARACTER_DOMAIN_NAMES = ("creatures", "anthro_furry", "robots_mechs", "humanoid_monsters")

# Substrings for tooling / ``suggest_creature_prompt_addons`` (multi-word phrases OK).
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

CREATURE_CHARACTER_RECOMMENDED_NEGATIVE_BY_DOMAIN = {
    "creatures": (
        "merged species, extra limbs random, floating paws, anatomy soup, melted creature, "
        "wrong leg count, two heads accidental, wings from wrong vertebra, scale-fur confusion"
    ),
    "anthro_furry": (
        "human ears on anthro, muzzle too short for species, fur as plastic, skin-fur boundary blur, "
        "paws as human hands by mistake, tail disconnected, duplicate facial features, species unclear"
    ),
    "robots_mechs": (
        "organic skin on metal panels, melted machinery, inconsistent scale bolts, "
        "floating armor, limb count wrong, mecha as rubber suit, wobbly hard-surface, "
        "symmetry error on mechanical parts, human flesh where chrome should be"
    ),
    "humanoid_monsters": (
        "melted horror face, merged eyes, anatomy noise, human monster indistinct, "
        "blobs instead of muscles, random tentacles, low readability silhouette, cute by accident, "
        "horns duplicated, wings from wrong joint, tail from spine mid-back, angel wings on demon by mistake, "
        "extra human ears on elf, wrong ear count, merged succubus wings, floating halo, plastic horns"
    ),
}

CREATURE_COMMON_NEGATIVE_ADDON = (
    "wrong species mix, melted anatomy, duplicate ears, floating limbs, merged fingers, "
    "inconsistent tail, plastic fur, rubber mech, flesh-metal confusion, unreadable silhouette"
)

# --- Rating-specific packs (SFW vs mature/explicit) ---
# SFW: general-audience character art, games, books, mascots — push away from adult themes in the negative.
CREATURE_SFW_CONTEXT_POSITIVE = (
    "sfw, general audience, character design, wholesome, adventure, expressive pose, "
    "costume appropriate, clear silhouette, safe for work, family friendly"
)

CREATURE_SFW_NEGATIVE_ADDON = (
    "nsfw, nude, naked, explicit, sexual, erotic, pornographic, suggestive, "
    "minor, child, loli, shota, underage, school uniform fetish, "
    "wardrobe malfunction, see-through clothes, sheer fabric focus"
)

# NSFW: adult-only creative use — emphasize **coherent anatomy + species** (same topology rules as SFW).
CREATURE_NSFW_CONTEXT_POSITIVE = (
    "adult character, mature themes, explicit anatomy coherent with species, "
    "consistent body plan, same tag vocabulary as training data, 18+, adult only"
)

CREATURE_NSFW_NEGATIVE_ADDON = (
    "minor, underage, child, loli, shota, "
    "melted anatomy, fused limbs, impossible joints, duplicate torsos, merged bodies, "
    "species-anatomy mismatch, rubber skin, identity blur between characters, "
    "wrong hole topology, extra genitals, merged partners, floating hands, disembodied limbs"
)

# Substring hints for ``rating="auto"`` in tooling (subset of typical booru / prompt language).
CREATURE_NSFW_KEYWORDS = (
    "nsfw",
    "nude",
    "naked",
    "explicit",
    "erotic",
    "sexual",
    "18+",
    "18 plus",
    "adult only",
    "mature rating",
    "uncensored",
    "nipple",
    "nipples",
    "areola",
    "genital",
    "penis",
    "vagina",
    "pussy",
    "dick",
    "cock",
    "cum",
    "semen",
    "orgasm",
    "intercourse",
    "sex",
    "bdsm",
    "bondage",
    "after sex",
    "spread legs",
    "on bed nude",
    "topless",
    "bottomless",
    "no panties",
    "no bra",
    "lingerie",
    "see-through",
    "sheer",
    "micro bikini",
    "implied sex",
    "masturbat",
    "fingering",
    "oral sex",
    "fellatio",
    "cunnilingus",
    "paizuri",
    "titfuck",
    "doggy style",
    "cowgirl position",
    "missionary",
    "from behind",
    "penetration",
    "creampie",
    "ahegao",
    "on all fours",
    "straddling",
)

# Per-domain lines when you want the rating baked into the **positive** (optional prepend).
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

CREATURE_CHARACTER_SFW_RECOMMENDED_NEGATIVE_BY_DOMAIN = {
    "creatures": CREATURE_CHARACTER_RECOMMENDED_NEGATIVE_BY_DOMAIN["creatures"]
    + ", gore, dismemberment, exposed organs, sexualized minor, provocative pose, vore",
    "anthro_furry": CREATURE_CHARACTER_RECOMMENDED_NEGATIVE_BY_DOMAIN["anthro_furry"]
    + ", nsfw, nude, sexualized minor, underwear focus, downblouse, upskirt, cameltoe",
    "robots_mechs": CREATURE_CHARACTER_RECOMMENDED_NEGATIVE_BY_DOMAIN["robots_mechs"]
    + ", gore, crushed body horror, sexualized minor, exposed wires as gore substitute",
    "humanoid_monsters": CREATURE_CHARACTER_RECOMMENDED_NEGATIVE_BY_DOMAIN["humanoid_monsters"]
    + ", extreme gore, sexualized minor, snuff, guro, entrails focus",
}

CREATURE_CHARACTER_NSFW_RECOMMENDED_NEGATIVE_BY_DOMAIN = {
    "creatures": CREATURE_CHARACTER_RECOMMENDED_NEGATIVE_BY_DOMAIN["creatures"] + ", " + CREATURE_NSFW_NEGATIVE_ADDON,
    "anthro_furry": CREATURE_CHARACTER_RECOMMENDED_NEGATIVE_BY_DOMAIN["anthro_furry"] + ", " + CREATURE_NSFW_NEGATIVE_ADDON,
    "robots_mechs": CREATURE_CHARACTER_RECOMMENDED_NEGATIVE_BY_DOMAIN["robots_mechs"] + ", " + CREATURE_NSFW_NEGATIVE_ADDON,
    "humanoid_monsters": CREATURE_CHARACTER_RECOMMENDED_NEGATIVE_BY_DOMAIN["humanoid_monsters"]
    + ", "
    + CREATURE_NSFW_NEGATIVE_ADDON,
}

CREATURES_PROMPT_TIPS = [
    "State **leg count and posture**: 'quadruped resting', 'bipedal alien' reduces random limbs.",
    "Name **covering**: scales vs feathers vs hide vs exoskeleton — models confuse blended words.",
    "One **focal creature**; 'herd' or 'pack' often duplicates errors — start with one hero subject.",
    "Add **scale cue**: 'horse-sized', 'cat-sized' helps proportions vs environment.",
    "Symmetry: 'mirrored horns', 'paired tusks' — odd counts often render wrong.",
    "Eyes: 'forward-facing predator eyes' vs 'side prey eyes' — fixes wrong skull read.",
    "Teeth: 'herbivore flat molars' vs 'carnivore fangs' — reduces random dentistry.",
]

CREATURES_SFW_TIPS = [
    "Children's media: 'rounded teeth', 'soft claws', 'big expressive eyes' — keeps tone safe.",
    "Pet-like creatures: 'collar with tag', 'leash held by hero' — anchors scale and story.",
    "No teeth blood: say 'clean muzzle' or 'closed mouth smile' for family art.",
    "Sticker or emoji style: 'thick outline', 'flat colors' — reduces accidental gore.",
    "Reference real animals for **silhouette** (rabbit vs fox vs bear) before fantasy add-ons.",
]

CREATURES_NSFW_TIPS = [
    "Adult feral-style: still tag **adult** and avoid juvenile proportions; use heavy muscle or size cues.",
    "Explicit mount + rider: describe **where hands and reins** go so limbs do not merge.",
    "Multi-partner with creature: number each **face and torso** in the prompt order.",
    "Tentacle / non-human: keep **one continuity** (same sucker size) along each limb.",
    "After explicit, 'glowing eyes dimmed', 'exhausted pant' — optional cooldown tags for coherence.",
]

ANTHRO_FURRY_PROMPT_TIPS = [
    "Specify **species** early: anthro fox, anthro rabbit — avoids generic dog-cat mush.",
    "Choose **feet**: digitigrade paws vs plantigrade — mention if shoes or bare paws.",
    "Say **no human ears** when you want only animal ears (many models add both).",
    "Clothing: 'jacket open over chest fur', 'collar clears neck ruff' reduces neck merge artifacts.",
    "Tail: 'tail visible behind', 'tail relaxed on ground' anchors attachment.",
    "Color: 'red fox fur with white chest', 'grey wolf with amber eyes' — reduces wrong palette.",
    "Hands: 'four fingers and thumb paw', 'hooved hands' — specify if non-human digits matter.",
    "Snout length: 'short rabbit nose', 'long horse muzzle' — species read locks faster.",
]

ROBOTS_MECH_PROMPT_TIPS = [
    "Split **materials**: brushed aluminum vs painted armor vs rubber seals — reduces plastic-flesh look.",
    "Joints: 'ball joint shoulder', 'piston elbow' gives the model hinge vocabulary.",
    "Scale: 'human-height android' vs 'building-sized mech' — avoid ambiguous 'robot' alone.",
    "Face: 'screen face', 'camera lens eyes', or 'humanlike synth skin' — pick one read.",
    "Cables and vents: small readable details beat vague 'high tech'.",
]

HUMANOID_MONSTERS_PROMPT_TIPS = [
    "Horror: **one** distortion axis (elongated limbs, extra eyes, wrong fingers) — stacking many causes mush.",
    "Keep **eyes count** explicit: 'two eyes', 'four eyes symmetric' if you care.",
    "Skin: 'waxy undead', 'leathery demon', 'chitin patches' — separates from normal human skin.",
    "Pose: 'standing contrapposto', 'crouched' — readable pose beats vague 'scary'.",
    "Wings: state **attachment** ('wings from shoulder blades', 'back-mounted feather wings') to reduce floating wings.",
    "Horns: **count and curve** ('two curved horns', 'single broken horn') beat vague 'demon horns'.",
    "Tails: 'demon tail from base of spine', 'thin pointed tail' — avoids random tail origins.",
    "Angels vs demons: use **feather vs bat wing** words explicitly so the model does not blend.",
    "Demi-humans: 'pointed ears only', 'no human ears' for elves, tieflings, and similar.",
    "Monster girls / *-girl tags: keep **species cues** (ears, tail, scales) plus human body plan explicit.",
]

CREATURE_TRAINING_CAPTION_TIPS = [
    "Caption **species + body plan** every time (anthro red fox, digitigrade, two ears).",
    "For robots, repeat **hard-surface words** in diverse scenes so metal does not drift to skin.",
    "Pair monsters with **lighting words** (rim, underlight) so face geometry stays readable.",
    "Negative prompts in data are rare; prefer **positive material boundaries** in captions.",
]

CREATURE_SFW_SCENARIO_TIPS = [
    "Use **rating tags** consistently in data: `sfw`, `general` — mirrors booru / site rules.",
    "For commissions: name **clothing layer** early so the model does not default to nude anthro.",
    "Mascots and games: 'full body', 'turnaround-friendly', 'readable icon silhouette'.",
    "Avoid ambiguous 'adult' (means mature tone vs age); prefer 'general audience' or '18+' explicitly.",
    "Streaming-safe: add 'no cleavage', 'high neckline' if the base model over-sexualizes armor.",
    "Visual novel sprite: 'waist-up', 'neutral A-pose' — reduces odd crops.",
    "Tabletop token: top-down or isometric, creature centered, plain base.",
]

CREATURE_NSFW_SCENARIO_TIPS = [
    "Adult-only datasets: keep **the same species tokens** as your SFW captions (anthro fox, digitigrade) plus explicit tags.",
    "Align with your **tag schema** (e.g. danbooru-style) so train and sample use the same vocabulary.",
    "Explicit + creature: describe **contact points and limbs** (who is where) to reduce merged-body artifacts.",
    "Use negatives for **topology** (fused, extra limbs) more than for style; see CREATURE_NSFW_NEGATIVE_ADDON.",
    "Tag **solo / duo / group** explicitly; crowd scenes amplify merged limbs.",
    "Perspective: 'from side', 'three-quarter view' — fixes impossible penetration angles.",
    "Lighting: 'rim light on sweat' vs 'flat hentai lighting' — pick one render read.",
    "Post titles vs prompt: duplicate key tags in prompt body so T5 sees them before truncation.",
]

ANTHRO_FURRY_SFW_TIPS = [
    "Lead with outfit: 'anthro rabbit in winter coat' before face detail.",
    "Props anchor SFW scenes: 'holding map', 'sitting at cafe table'.",
    "Furry community tags: `clothed`, `fully clothed` if your hub supports them.",
    "School or academy SFW: **uniform modest**, specify age as adult student if needed.",
    "Sports: 'sweatband', 'knee pads' — athletic not fetish unless you tag otherwise.",
    "Sleepover SFW: pajamas onesie, plushies, popcorn — avoid lingerie wording.",
    "Reference sheet: front side back views in prompt text or split generations.",
]

ANTHRO_FURRY_NSFW_TIPS = [
    "State **adult** and species in one line: 'adult anthro wolf, explicit, mature body'.",
    "Separate **muzzle from human face cues** (no second nose); mention 'only animal nose'.",
    "Fur on **torso and limbs** with direction words reduces skin-fur mush in explicit poses.",
    "Knot / feral anatomy: use **same terms your training set** uses; consistency beats euphemism mix.",
    "Pairs: 'anthro A left, anthro B right' — reduces merged muzzles.",
    "Fluids on fur: 'matte wet fur', 'clumped strands' — direction words help.",
    "Mask or hood kink: 'hood back, ears through holes cut' — avoids flat head.",
    "Aftercare: 'blanket over shoulders', 'tail wrapped around partner' — optional cozy tag.",
]

ROBOTS_MECH_SFW_TIPS = [
    "Kid-friendly robot: rounded forms, expressive LED face, no body horror.",
    "Pilot visible: 'helmet on', 'cockpit interior' clarifies scale vs mech.",
    "Classroom tutor bot: rolling base, projector eye, chalk dust — grounded scene.",
    "Robot pet: dog-sized quadruped bot, charging dock, LED tail wag.",
    "Toy line aesthetic: 'action figure joints visible' — readable plastic seams.",
]

ROBOTS_MECH_NSFW_TIPS = [
    "Android explicit: 'synthetic skin seam at neck', 'access panel closed' — boundaries readers expect.",
    "Partial suit: which parts are **chrome vs suit fabric** avoids random flesh patches.",
    "Oil or coolant as metaphor: 'blue drip from shoulder joint' — keep non-gory if preferred.",
    "Voice box glow: 'speaker grille chest lit' — focal when face buried in partner.",
    "Cable insert play: 'port at nape', 'fiber optic subtle' — one port design only.",
    "Twin androids: 'matching serial stamp hip' — tells them apart in explicit crowd.",
]

HUMANOID_MONSTERS_SFW_TIPS = [
    "Stylized villain: 'cartoon menace', 'disney villain silhouette' keeps blood optional.",
    "TTRPG cover: 'monster manual style', readable at thumbnail size.",
    "Halloween costume SFW: visible zipper, human eyes through mask holes.",
    "Cute demon: 'small horns', 'blush', 'oversized sweater' — tone lock.",
    "Paladin vs demon standoff: symmetrical composition, no contact, clear silhouettes.",
]

HUMANOID_MONSTERS_NSFW_TIPS = [
    "Horror-mature: pick **gore level** explicitly; 'bloodless horror' vs 'gory' reduces random entrails.",
    "Explicit + monster: one **primary distortion** (claws, teeth, extra arms) — avoid stacking every trope.",
    "Bite marks: 'healing scrape' vs 'deep wound' — pick one severity.",
    "Claws on partner skin: 'red lines, no ribbons of flesh' if you want lighter guro.",
    "Tail use explicit: describe **prehensile curl** vs passive tail to avoid random third limb.",
    "Wing cloak pose: 'wings wrapped around both figures' — mention overlap order (who in front).",
    "Corruption aesthetic: 'black veins spreading from contact' — single pattern direction.",
]

"""
Special Prompt Helpers for AI Image Generation.

Covers categories that standard prompt engineering guides and the existing
``config/defaults/prompt_domains.py`` system handle poorly:

1. **Weird / surreal / abstract** — dreamlike logic, impossible geometry, Magritte-style
2. **Horror / dark / disturbing** — dread atmosphere, body horror, psychological unease
3. **Narrative / story-driven** — "the moment before X", aftermath, implied history
4. **Emotion / mood-first** — leading with feeling rather than subject description
5. **Hard technical** — extreme foreshortening, unusual perspectives, complex multi-subject
6. **Style fusion** — combining genuinely incompatible styles without collapse
7. **Minimalist / negative-space** — restraint, emptiness as composition tool
8. **NSFW anatomy precision** — per-body-part precision tokens beyond generic anatomy tags
9. **Auto-detection** — ``classify_prompt_category()`` routes to the right helper
10. **Single entry point** — ``apply_special_helpers()`` wires everything together

Design notes
------------
- Pure Python, no torch dependency.  Safe to import at any time.
- Each helper returns ``(positive_addon, negative_addon)`` tuples of comma-separated
  token strings, ready to be merged with the caller's existing prompt/negative via
  ``merge_csv_unique()`` (same helper used in ``ai_image_shortcomings.py``).
- Token choices are grounded in known diffusion model failure modes:
  - SDXL / Illustrious / NoobAI: anatomy collapse, concept bleeding, centering bias
  - Flux: same-face syndrome, grid artifacts at high CFG, over-polished look
  - General: horror defaults to "spooky Halloween", surreal defaults to "colorful dream",
    narrative loses the implied story, emotion gets overridden by subject tags.
- WHY comments explain the reasoning behind non-obvious token choices.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

__all__ = [
    # Category constants
    "CATEGORY_WEIRD",
    "CATEGORY_HORROR",
    "CATEGORY_NARRATIVE",
    "CATEGORY_EMOTION",
    "CATEGORY_TECHNICAL",
    "CATEGORY_STYLE_FUSION",
    "CATEGORY_MINIMALIST",
    "CATEGORY_NSFW_PRECISION",
    "CATEGORY_STANDARD",
    # Per-category token dicts
    "WEIRD_POSITIVE_TOKENS",
    "WEIRD_NEGATIVE_TOKENS",
    "HORROR_POSITIVE_TOKENS",
    "HORROR_NEGATIVE_TOKENS",
    "NARRATIVE_POSITIVE_TOKENS",
    "NARRATIVE_NEGATIVE_TOKENS",
    "EMOTION_POSITIVE_TOKENS",
    "EMOTION_NEGATIVE_TOKENS",
    "TECHNICAL_POSITIVE_TOKENS",
    "TECHNICAL_NEGATIVE_TOKENS",
    "STYLE_FUSION_POSITIVE_TOKENS",
    "STYLE_FUSION_NEGATIVE_TOKENS",
    "MINIMALIST_POSITIVE_TOKENS",
    "MINIMALIST_NEGATIVE_TOKENS",
    "NSFW_PRECISION_POSITIVE_TOKENS",
    "NSFW_PRECISION_NEGATIVE_TOKENS",
    # Per-body-part NSFW precision maps
    "NSFW_BODY_PART_PRECISION",
    # Per-category helper functions
    "weird_helpers",
    "horror_helpers",
    "narrative_helpers",
    "emotion_helpers",
    "technical_helpers",
    "style_fusion_helpers",
    "minimalist_helpers",
    "nsfw_precision_helpers",
    # Detection + wiring
    "classify_prompt_category",
    "apply_special_helpers",
    # Utility
    "merge_csv_unique",
]

# ---------------------------------------------------------------------------
# Category name constants
# ---------------------------------------------------------------------------

CATEGORY_WEIRD: str = "weird"
CATEGORY_HORROR: str = "horror"
CATEGORY_NARRATIVE: str = "narrative"
CATEGORY_EMOTION: str = "emotion"
CATEGORY_TECHNICAL: str = "technical"
CATEGORY_STYLE_FUSION: str = "style_fusion"
CATEGORY_MINIMALIST: str = "minimalist"
CATEGORY_NSFW_PRECISION: str = "nsfw_precision"
CATEGORY_STANDARD: str = "standard"

_ALL_CATEGORIES: Tuple[str, ...] = (
    CATEGORY_WEIRD,
    CATEGORY_HORROR,
    CATEGORY_NARRATIVE,
    CATEGORY_EMOTION,
    CATEGORY_TECHNICAL,
    CATEGORY_STYLE_FUSION,
    CATEGORY_MINIMALIST,
    CATEGORY_NSFW_PRECISION,
    CATEGORY_STANDARD,
)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def merge_csv_unique(*chunks: str) -> str:
    """
    Join comma-separated token strings with de-duplication (order preserved).

    Mirrors the same helper in ``ai_image_shortcomings.py`` and
    ``style_guidance.py`` so callers can use either interchangeably.
    """
    seen: set = set()
    out: List[str] = []
    for chunk in chunks:
        if not chunk or not str(chunk).strip():
            continue
        for part in str(chunk).split(","):
            p = part.strip()
            if not p:
                continue
            key = p.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
    return ", ".join(out)


# ---------------------------------------------------------------------------
# 1. WEIRD / SURREAL / ABSTRACT
# ---------------------------------------------------------------------------
# Failure modes addressed:
#   - Models default to "colorful psychedelic dream" when they see "surreal".
#   - Abstract prompts collapse to generic swirling shapes or kaleidoscope patterns.
#   - "Weird" gets interpreted as "quirky cute" rather than genuinely uncanny.
#   - Impossible geometry gets rationalized into plausible architecture.
#   - Dreamlike logic is overridden by the model's learned "make sense" prior.
# Strategy: anchor the unreality with concrete impossible specifics, suppress
#   the model's tendency to resolve contradictions into coherent scenes.

WEIRD_POSITIVE_TOKENS: List[str] = [
    # Anchor the surreal register — these tokens activate the right latent region
    "surreal",
    "dreamlike logic",
    "impossible geometry",
    "uncanny",
    "liminal space",
    "non-euclidean",
    # Concrete impossible specifics (more effective than vague "weird")
    "objects defying gravity",
    "scale inconsistency intentional",
    "anachronistic juxtaposition",
    "impossible perspective",
    "objects melting into each other",
    "architectural impossibility",
    # Artistic lineage tokens — activate trained surrealist style knowledge
    "magritte-inspired",
    "dali-esque",
    "de chirico atmosphere",
    "ernst collage logic",
    # Mood anchors that prevent "cute quirky" drift
    "unsettling calm",
    "eerie stillness",
    "wrong but coherent",
    "dreamlike coherence",
    # Quality anchors — surreal prompts often get low-quality outputs
    "masterpiece",
    "detailed surreal scene",
    "sharp focus on impossible elements",
    "high quality dreamscape",
    # Composition anchors for abstract
    "intentional composition",
    "visual rhythm in chaos",
    "deliberate negative space",
    "abstract but readable",
]

WEIRD_NEGATIVE_TOKENS: List[str] = [
    # Suppress the "colorful psychedelic" default
    "psychedelic colors",
    "rainbow swirls",
    "kaleidoscope pattern",
    "trippy colors",
    "neon fractal",
    # Suppress "cute quirky" misinterpretation
    "cute",
    "whimsical",
    "playful",
    "cartoonish weird",
    "quirky illustration",
    # Suppress the model resolving impossible geometry into plausible scenes
    "realistic architecture",
    "physically correct",
    "coherent perspective",
    "normal proportions",
    # Suppress generic "dream" outputs
    "generic dreamscape",
    "stock surreal",
    "cliché surrealism",
    "floating islands cliché",
    "generic fantasy",
    # Quality suppressors
    "low quality",
    "blurry",
    "incoherent",
    "muddy",
]


def weird_helpers(prompt: str) -> Tuple[str, str]:
    """
    Return ``(positive_addon, negative_addon)`` for weird/surreal/abstract prompts.

    Detects sub-registers (abstract vs surreal vs liminal) and adjusts the
    token mix accordingly.  The positive tokens anchor the uncanny register
    and prevent the model from resolving impossible elements into coherent
    scenes.  The negative tokens suppress the three most common failure modes:
    psychedelic-color default, cute-quirky misread, and geometry normalization.

    Args:
        prompt: The user's positive prompt string.

    Returns:
        Tuple of (positive_addon, negative_addon) comma-separated token strings.
    """
    p = prompt.lower()

    # Sub-register detection
    is_abstract = any(w in p for w in ("abstract", "non-representational", "geometric", "pattern"))
    is_liminal = any(w in p for w in ("liminal", "backrooms", "empty corridor", "abandoned", "void"))
    is_body_horror_adjacent = any(w in p for w in ("melting", "morphing", "transforming", "fusing"))

    pos_extras: List[str] = []
    neg_extras: List[str] = []

    if is_abstract:
        # Abstract needs composition anchors more than surreal narrative anchors
        pos_extras += [
            "intentional abstract composition",
            "color field discipline",
            "shape language clarity",
            "visual tension",
        ]
        neg_extras += [
            "representational",
            "figurative",
            "narrative scene",
        ]

    if is_liminal:
        # Liminal spaces need the "wrong emptiness" feeling, not just empty rooms
        pos_extras += [
            "liminal atmosphere",
            "wrong emptiness",
            "fluorescent hum feeling",
            "no exit implied",
            "transitional non-place",
        ]
        neg_extras += [
            "cozy",
            "inviting",
            "populated",
            "normal room",
        ]

    if is_body_horror_adjacent:
        # Morphing/melting needs to stay surreal, not drift into gore
        pos_extras += [
            "surreal transformation",
            "dreamlike morphing",
            "impossible anatomy intentional",
        ]
        neg_extras += [
            "gore",
            "blood",
            "visceral",
        ]

    pos = merge_csv_unique(", ".join(WEIRD_POSITIVE_TOKENS), ", ".join(pos_extras))
    neg = merge_csv_unique(", ".join(WEIRD_NEGATIVE_TOKENS), ", ".join(neg_extras))
    return pos, neg


# ---------------------------------------------------------------------------
# 2. HORROR / DARK / DISTURBING
# ---------------------------------------------------------------------------
# Failure modes addressed:
#   - "Horror" defaults to Halloween orange-and-black with jack-o-lanterns.
#   - "Dark" gets interpreted as low-exposure photography, not mood.
#   - "Disturbing" collapses to generic monster or jump-scare imagery.
#   - Psychological horror loses the dread and becomes action-horror.
#   - Body horror gets sanitized into "slightly weird anatomy".
#   - Atmospheric horror loses tension when quality tags pull toward "beautiful".
# Strategy: separate atmospheric dread from visceral horror; use specific
#   cinematographic and literary horror references to activate the right register.

HORROR_POSITIVE_TOKENS: List[str] = [
    # Atmospheric dread anchors
    "atmospheric horror",
    "dread atmosphere",
    "psychological unease",
    "creeping dread",
    "unsettling",
    "foreboding",
    "ominous",
    # Cinematographic horror references (activate trained horror-film knowledge)
    "horror film lighting",
    "chiaroscuro shadows",
    "deep shadows",
    "motivated darkness",
    "shadow detail preserved",
    "high contrast horror",
    # Specific horror sub-registers
    "cosmic horror scale",
    "eldritch",
    "lovecraftian atmosphere",
    "body horror",
    "psychological horror",
    "folk horror",
    "slow burn dread",
    # Texture and detail tokens that sell horror
    "decayed texture",
    "weathered and worn",
    "organic wrongness",
    "uncanny valley intentional",
    "wrong proportions intentional",
    # Composition anchors for horror
    "horror composition",
    "negative space as threat",
    "subject partially obscured",
    "something wrong in the background",
    # Quality anchors — horror needs detail to work
    "detailed horror scene",
    "sharp focus on disturbing elements",
    "masterpiece dark art",
    "high quality horror illustration",
]

HORROR_NEGATIVE_TOKENS: List[str] = [
    # Suppress Halloween-default
    "halloween",
    "jack-o-lantern",
    "orange and black",
    "spooky cute",
    "cartoon horror",
    "campy horror",
    "fun scary",
    # Suppress low-exposure misread of "dark"
    "underexposed",
    "too dark to see",
    "pitch black",
    "crushed blacks",
    # Suppress sanitization of body horror
    "clean anatomy",
    "normal proportions",
    "healthy",
    "pristine",
    # Suppress action-horror drift
    "action pose",
    "heroic",
    "triumphant",
    "dynamic action",
    # Suppress over-polished AI look that kills horror atmosphere
    "plastic skin",
    "airbrushed",
    "perfect lighting",
    "studio lighting",
    "bright and cheerful",
    "vibrant colors",
    "oversaturated",
    # Quality suppressors
    "low quality",
    "blurry",
    "flat",
]


def horror_helpers(prompt: str) -> Tuple[str, str]:
    """
    Return ``(positive_addon, negative_addon)`` for horror/dark/disturbing prompts.

    Detects sub-registers (atmospheric, body horror, psychological, cosmic) and
    adjusts the token mix.  The key insight is that horror quality comes from
    *specificity of wrongness* — vague "scary" tokens produce generic results,
    while specific dread anchors (chiaroscuro, eldritch scale, organic wrongness)
    activate the model's horror-trained knowledge.

    Args:
        prompt: The user's positive prompt string.

    Returns:
        Tuple of (positive_addon, negative_addon) comma-separated token strings.
    """
    p = prompt.lower()

    is_cosmic = any(w in p for w in ("cosmic", "eldritch", "lovecraft", "cthulhu", "void", "ancient evil"))
    is_body_horror = any(w in p for w in ("body horror", "flesh", "mutation", "deformed", "grotesque", "visceral"))
    is_psychological = any(w in p for w in ("psychological", "paranoia", "hallucination", "mental", "mind"))
    is_folk_horror = any(w in p for w in ("folk horror", "cult", "ritual", "pagan", "rural horror", "wicker"))
    is_ghost = any(w in p for w in ("ghost", "spirit", "apparition", "haunted", "specter", "wraith"))

    pos_extras: List[str] = []
    neg_extras: List[str] = []

    if is_cosmic:
        pos_extras += [
            "incomprehensible scale",
            "non-euclidean horror",
            "ancient and indifferent",
            "human insignificance",
            "tentacles as architecture",
            "wrong stars",
        ]
        neg_extras += ["relatable", "human-scale", "friendly alien"]

    if is_body_horror:
        pos_extras += [
            "organic wrongness",
            "flesh texture detail",
            "biological impossibility",
            "visceral detail",
            "wet organic surface",
            "body horror masterpiece",
        ]
        neg_extras += ["clean", "sanitized", "medical illustration clean"]

    if is_psychological:
        pos_extras += [
            "unreliable perspective",
            "visual paranoia",
            "something wrong but undefined",
            "dread without source",
            "psychological tension",
        ]
        neg_extras += ["clear threat", "obvious monster", "jump scare setup"]

    if is_folk_horror:
        pos_extras += [
            "folk horror atmosphere",
            "rural isolation",
            "pagan ritual aesthetic",
            "wicker and bone",
            "harvest dread",
            "community as threat",
        ]
        neg_extras += ["urban", "modern", "technology"]

    if is_ghost:
        pos_extras += [
            "translucent apparition",
            "ectoplasmic glow",
            "presence implied",
            "cold light",
            "spectral detail",
        ]
        neg_extras += ["solid", "opaque", "fully visible", "friendly ghost"]

    pos = merge_csv_unique(", ".join(HORROR_POSITIVE_TOKENS), ", ".join(pos_extras))
    neg = merge_csv_unique(", ".join(HORROR_NEGATIVE_TOKENS), ", ".join(neg_extras))
    return pos, neg


# ---------------------------------------------------------------------------
# 3. NARRATIVE / STORY-DRIVEN
# ---------------------------------------------------------------------------
# Failure modes addressed:
#   - "The moment before X" loses the anticipation and becomes a static portrait.
#   - "Aftermath of Y" loses the implied history and becomes a clean scene.
#   - Story-driven prompts get stripped of narrative tension by quality tags.
#   - Environmental storytelling is ignored in favor of subject rendering.
#   - Implied off-screen elements (the threat, the event) are not rendered.
#   - Time-of-event cues (dust settling, smoke clearing) are missed.
# Strategy: use cinematographic storytelling tokens, environmental narrative
#   anchors, and temporal cue tokens to preserve the story in the image.

NARRATIVE_POSITIVE_TOKENS: List[str] = [
    # Temporal narrative anchors
    "narrative moment",
    "story-driven composition",
    "implied narrative",
    "environmental storytelling",
    "cinematic storytelling",
    # "Moment before" tokens — anticipation and tension
    "moment of anticipation",
    "held breath",
    "tension before action",
    "about to happen",
    "pre-event stillness",
    "coiled tension",
    # "Aftermath" tokens — implied history
    "aftermath atmosphere",
    "story already happened",
    "evidence of events",
    "residual traces",
    "dust settling",
    "smoke clearing",
    "aftermath stillness",
    # Environmental narrative detail
    "story in the details",
    "narrative props",
    "meaningful background elements",
    "worn objects with history",
    "contextual clues",
    "scene-setting details",
    # Cinematographic framing for narrative
    "establishing shot narrative",
    "motivated framing",
    "off-screen implied threat",
    "reaction shot composition",
    "decisive moment",
    # Quality anchors that preserve narrative
    "detailed narrative scene",
    "masterpiece storytelling",
    "cinematic quality",
    "sharp focus on story elements",
]

NARRATIVE_NEGATIVE_TOKENS: List[str] = [
    # Suppress static portrait default
    "static portrait",
    "posed",
    "studio pose",
    "neutral expression",
    "blank background",
    # Suppress clean-scene default for aftermath
    "pristine",
    "undisturbed",
    "clean scene",
    "no damage",
    "perfect condition",
    # Suppress loss of environmental storytelling
    "simple background",
    "plain background",
    "minimal background",
    "background blur only",
    # Suppress generic quality tags overriding narrative mood
    "generic composition",
    "stock photo",
    "commercial photography",
    "product shot",
    # Quality suppressors
    "low quality",
    "blurry",
    "flat",
    "incoherent",
]


def narrative_helpers(prompt: str) -> Tuple[str, str]:
    """
    Return ``(positive_addon, negative_addon)`` for narrative/story-driven prompts.

    Detects whether the prompt is "moment before", "aftermath", "implied history",
    or general narrative, and adjusts tokens accordingly.  The core challenge is
    that diffusion models are trained on static images and lose temporal/narrative
    context — these tokens re-anchor the narrative register.

    Args:
        prompt: The user's positive prompt string.

    Returns:
        Tuple of (positive_addon, negative_addon) comma-separated token strings.
    """
    p = prompt.lower()

    is_moment_before = any(
        phrase in p
        for phrase in (
            "moment before",
            "about to",
            "just before",
            "seconds before",
            "on the verge",
            "brink of",
            "anticipation",
        )
    )
    is_aftermath = any(
        phrase in p
        for phrase in (
            "aftermath",
            "after the",
            "just after",
            "following the",
            "in the wake",
            "ruins of",
            "remains of",
            "wreckage",
        )
    )
    is_memory = any(
        phrase in p for phrase in ("memory", "flashback", "remembering", "nostalgia", "long ago", "faded memory")
    )
    is_discovery = any(
        phrase in p for phrase in ("discovers", "finding", "stumbles upon", "comes across", "first sight of")
    )

    pos_extras: List[str] = []
    neg_extras: List[str] = []

    if is_moment_before:
        pos_extras += [
            "tension before the moment",
            "anticipatory stillness",
            "held breath composition",
            "pre-action energy",
            "coiled readiness",
            "eyes focused on unseen target",
            "body language of anticipation",
        ]
        neg_extras += ["relaxed", "at ease", "aftermath", "post-action"]

    if is_aftermath:
        pos_extras += [
            "aftermath evidence",
            "story already told by environment",
            "dust and debris settling",
            "residual heat shimmer",
            "scattered objects with narrative",
            "absence as presence",
            "what was here before implied",
        ]
        neg_extras += ["pristine", "undamaged", "before the event", "clean"]

    if is_memory:
        pos_extras += [
            "memory-like quality",
            "soft focus edges",
            "warm nostalgic light",
            "faded color palette",
            "dreamlike recall",
            "imperfect recollection",
        ]
        neg_extras += ["sharp crisp present", "high contrast", "cold light"]

    if is_discovery:
        pos_extras += [
            "moment of discovery",
            "wonder and recognition",
            "first encounter framing",
            "subject reacting to revelation",
            "discovery lighting",
        ]
        neg_extras += ["familiar", "routine", "expected"]

    pos = merge_csv_unique(", ".join(NARRATIVE_POSITIVE_TOKENS), ", ".join(pos_extras))
    neg = merge_csv_unique(", ".join(NARRATIVE_NEGATIVE_TOKENS), ", ".join(neg_extras))
    return pos, neg


# ---------------------------------------------------------------------------
# 4. EMOTION / MOOD-FIRST
# ---------------------------------------------------------------------------
# Failure modes addressed:
#   - Emotion tokens get overridden by subject/style tokens (T5 truncation).
#   - "Melancholy" defaults to "sad face" rather than pervasive mood.
#   - "Joyful" defaults to "smiling person" rather than light and color mood.
#   - Nuanced emotions (bittersweet, wistful, resigned) collapse to nearest
#     simple emotion (sad, happy, neutral).
#   - Mood-first prompts lose the mood when quality tags are prepended.
#   - Color temperature and lighting are the primary mood carriers but are
#     often not specified when leading with emotion.
# Strategy: translate emotion into its visual language (color, light, texture,
#   composition) and anchor the mood register before subject tokens.

EMOTION_POSITIVE_TOKENS: List[str] = [
    # Mood-first framing anchors
    "mood-driven composition",
    "emotion as atmosphere",
    "feeling before subject",
    "pervasive mood",
    # Color-temperature mood carriers
    "color temperature matches mood",
    "warm light for warmth",
    "cool desaturated for melancholy",
    "golden light for nostalgia",
    "blue-grey for loneliness",
    # Lighting as mood carrier
    "motivated mood lighting",
    "soft diffuse for tenderness",
    "harsh contrast for tension",
    "rim light for isolation",
    "ambient glow for wonder",
    # Texture and detail as mood carrier
    "texture reinforces mood",
    "worn surfaces for melancholy",
    "soft surfaces for comfort",
    "sharp edges for anxiety",
    # Composition as mood carrier
    "composition reinforces emotion",
    "negative space for loneliness",
    "tight framing for intimacy",
    "wide shot for isolation",
    "low angle for awe",
    # Nuanced emotion anchors
    "bittersweet",
    "wistful",
    "resigned",
    "quietly hopeful",
    "tender",
    "aching",
    "serene melancholy",
    # Quality anchors
    "emotionally resonant",
    "masterpiece mood piece",
    "evocative atmosphere",
]

EMOTION_NEGATIVE_TOKENS: List[str] = [
    # Suppress emotion-override by subject tags
    "neutral expression",
    "blank face",
    "emotionless",
    "stock photo smile",
    "forced smile",
    # Suppress mood collapse to simple emotions
    "generic happy",
    "generic sad",
    "obvious emotion",
    "cliché expression",
    # Suppress lighting that kills mood
    "flat lighting",
    "even lighting",
    "studio lighting",
    "harsh flash",
    "overexposed",
    # Suppress color that kills mood
    "oversaturated",
    "candy colors",
    "neon",
    "garish",
    # Quality suppressors
    "low quality",
    "blurry",
    "flat",
    "generic",
]

# Emotion-to-visual-language lookup for specific emotion tokens
_EMOTION_VISUAL_MAP: Dict[str, Tuple[List[str], List[str]]] = {
    "melancholy": (
        ["desaturated palette", "cool blue-grey light", "rain or mist", "downward gaze", "slumped posture"],
        ["bright colors", "warm light", "upbeat", "energetic"],
    ),
    "joy": (
        ["warm golden light", "saturated but not garish", "open body language", "upward movement", "soft bokeh"],
        ["cold light", "desaturated", "heavy shadows", "closed posture"],
    ),
    "dread": (
        ["cold desaturated light", "deep shadows", "tight framing", "something wrong implied", "stillness"],
        ["warm light", "open space", "bright", "cheerful"],
    ),
    "wonder": (
        ["soft volumetric light", "upward gaze", "scale contrast", "discovery lighting", "wide eyes"],
        ["mundane", "familiar", "routine", "flat lighting"],
    ),
    "loneliness": (
        ["wide shot", "subject small in frame", "empty space dominant", "cool light", "no eye contact"],
        ["crowded", "social", "warm gathering", "intimate"],
    ),
    "tenderness": (
        ["soft diffuse light", "warm palette", "close framing", "gentle touch implied", "soft focus edges"],
        ["harsh", "cold", "distant", "clinical"],
    ),
    "anxiety": (
        ["tight framing", "sharp edges", "cool harsh light", "cluttered background", "tense posture"],
        ["relaxed", "open", "calm", "serene"],
    ),
    "nostalgia": (
        ["warm golden tones", "soft vignette", "faded slightly", "vintage light quality", "memory-like"],
        ["crisp modern", "cold", "high contrast digital"],
    ),
    "awe": (
        ["low angle", "subject dwarfed by environment", "dramatic scale", "god rays", "overwhelmed expression"],
        ["human-scale", "intimate", "close-up", "mundane"],
    ),
    "resignation": (
        ["flat even light", "muted palette", "still posture", "downward gaze", "empty hands"],
        ["dynamic", "energetic", "hopeful", "bright"],
    ),
}


def emotion_helpers(prompt: str) -> Tuple[str, str]:
    """
    Return ``(positive_addon, negative_addon)`` for emotion/mood-first prompts.

    Translates detected emotion keywords into their visual language equivalents
    (color temperature, lighting quality, composition, texture).  This is
    critical because diffusion models respond to visual descriptors, not
    abstract emotion words — "melancholy" alone is weak; "desaturated palette,
    cool blue-grey light, downward gaze" is strong.

    Args:
        prompt: The user's positive prompt string.

    Returns:
        Tuple of (positive_addon, negative_addon) comma-separated token strings.
    """
    p = prompt.lower()

    pos_extras: List[str] = []
    neg_extras: List[str] = []

    for emotion, (visual_pos, visual_neg) in _EMOTION_VISUAL_MAP.items():
        if emotion in p:
            pos_extras.extend(visual_pos)
            neg_extras.extend(visual_neg)

    # Also check for compound/nuanced emotions
    if any(w in p for w in ("bittersweet", "wistful", "poignant")):
        pos_extras += [
            "simultaneous warmth and sadness",
            "golden light with cool shadows",
            "smile that doesn't reach the eyes",
            "beauty tinged with loss",
        ]
    if any(w in p for w in ("serene", "peaceful", "tranquil", "calm")):
        pos_extras += [
            "soft diffuse light",
            "still water or still air",
            "unhurried composition",
            "breathing room in frame",
        ]
        neg_extras += ["chaotic", "busy", "cluttered", "dynamic action"]

    pos = merge_csv_unique(", ".join(EMOTION_POSITIVE_TOKENS), ", ".join(pos_extras))
    neg = merge_csv_unique(", ".join(EMOTION_NEGATIVE_TOKENS), ", ".join(neg_extras))
    return pos, neg


# ---------------------------------------------------------------------------
# 5. HARD TECHNICAL (foreshortening, unusual perspectives, multi-subject)
# ---------------------------------------------------------------------------
# Failure modes addressed:
#   - Extreme foreshortening collapses to "weird anatomy" or gets corrected away.
#   - Worm's-eye / bird's-eye views default to mild tilt rather than extreme angle.
#   - Dutch angle becomes a slight tilt rather than a dramatic roll.
#   - Multi-subject prompts collapse subjects together or merge them.
#   - Complex overlapping figures lose depth and flatten.
#   - Fisheye / ultra-wide distortion gets smoothed out.
#   - POV shots lose the first-person perspective.
#   - Isometric / axonometric views drift toward standard perspective.
# Strategy: use explicit perspective grammar tokens, reinforce with negative
#   tokens that suppress the model's "correct to normal" prior.

TECHNICAL_POSITIVE_TOKENS: List[str] = [
    # Perspective anchors
    "correct perspective",
    "coherent perspective throughout",
    "consistent vanishing points",
    "perspective-accurate foreshortening",
    # Foreshortening specific
    "extreme foreshortening",
    "foreshortening intentional",
    "foreshortened limbs correct",
    "foreshortening anatomy accurate",
    "foreshortened figure",
    # Unusual angle anchors
    "dramatic camera angle",
    "extreme low angle",
    "extreme high angle",
    "worm's eye view",
    "bird's eye view",
    "overhead shot",
    "dutch angle dramatic",
    # Multi-subject anchors
    "multiple subjects clearly separated",
    "distinct subjects no merging",
    "clear spatial separation",
    "each subject individually readable",
    "group composition coherent",
    # Depth and overlap
    "clear depth layers",
    "foreground midground background distinct",
    "overlapping figures with depth",
    "occlusion correct",
    "depth cues consistent",
    # Lens distortion anchors
    "fisheye distortion intentional",
    "wide angle distortion correct",
    "barrel distortion consistent",
    # Quality anchors for technical shots
    "masterpiece technical composition",
    "detailed technical accuracy",
    "sharp focus throughout",
    "high quality complex scene",
]

TECHNICAL_NEGATIVE_TOKENS: List[str] = [
    # Suppress anatomy correction of intentional foreshortening
    "corrected anatomy",
    "normal proportions",
    "standard anatomy",
    "anatomically corrected",
    # Suppress mild-angle default
    "slight tilt",
    "mild angle",
    "standard perspective",
    "eye level default",
    "normal camera angle",
    # Suppress subject merging
    "merged subjects",
    "fused figures",
    "overlapping without depth",
    "subjects blending together",
    "unclear separation",
    # Suppress depth collapse
    "flat composition",
    "no depth",
    "everything same plane",
    "depth of field hiding subjects",
    # Suppress distortion smoothing
    "corrected distortion",
    "undistorted",
    "lens correction applied",
    # Quality suppressors
    "low quality",
    "blurry",
    "incoherent",
    "broken perspective",
]


def technical_helpers(prompt: str) -> Tuple[str, str]:
    """
    Return ``(positive_addon, negative_addon)`` for hard technical prompts.

    Detects the specific technical challenge (foreshortening, unusual angle,
    multi-subject, fisheye, isometric) and applies targeted tokens.  The key
    insight is that models have a strong "correct to normal" prior — they will
    fix foreshortening, normalize angles, and merge subjects unless explicitly
    told not to.  The negative tokens suppress this correction behavior.

    Args:
        prompt: The user's positive prompt string.

    Returns:
        Tuple of (positive_addon, negative_addon) comma-separated token strings.
    """
    p = prompt.lower()

    is_foreshortening = any(
        w in p for w in ("foreshorten", "foreshortening", "reaching toward camera", "pointing at viewer")
    )
    is_low_angle = any(w in p for w in ("from below", "worm", "low angle", "looking up at", "upward angle"))
    is_high_angle = any(w in p for w in ("from above", "bird", "high angle", "looking down", "overhead", "top down"))
    is_dutch = any(w in p for w in ("dutch angle", "tilted camera", "canted angle", "diagonal horizon"))
    is_multi_subject = any(
        w in p for w in ("two people", "three people", "group", "crowd", "multiple characters", "couple", "duo", "trio")
    )
    is_fisheye = any(w in p for w in ("fisheye", "fish eye", "ultra wide", "wide angle distortion"))
    is_isometric = any(w in p for w in ("isometric", "axonometric", "isometric view", "iso perspective"))
    is_pov = any(w in p for w in ("pov", "point of view", "first person", "from my perspective", "viewer's hands"))

    pos_extras: List[str] = []
    neg_extras: List[str] = []

    if is_foreshortening:
        pos_extras += [
            "extreme foreshortening correct",
            "foreshortened arm toward viewer",
            "foreshortening anatomy study",
            "foreshortened perspective accurate",
            "foreshortening not corrected",
        ]
        neg_extras += ["normal arm length", "corrected limb", "standard proportions"]

    if is_low_angle:
        pos_extras += [
            "extreme low angle shot",
            "worm's eye view dramatic",
            "looking up at subject",
            "ceiling or sky visible",
            "subject towers above",
            "dramatic upward perspective",
        ]
        neg_extras += ["eye level", "standard angle", "mild tilt"]

    if is_high_angle:
        pos_extras += [
            "extreme high angle shot",
            "bird's eye view",
            "looking straight down",
            "floor or ground dominant",
            "subject seen from above",
            "overhead perspective accurate",
        ]
        neg_extras += ["eye level", "standard angle", "mild tilt"]

    if is_dutch:
        pos_extras += [
            "dramatic dutch angle",
            "horizon strongly tilted",
            "canted frame intentional",
            "diagonal composition",
        ]
        neg_extras += ["level horizon", "straight camera", "corrected tilt"]

    if is_multi_subject:
        pos_extras += [
            "each subject clearly defined",
            "no subject merging",
            "spatial separation between subjects",
            "group dynamics readable",
            "individual identities preserved",
            "correct relative scale",
        ]
        neg_extras += ["merged figures", "fused subjects", "unclear who is who", "subjects blending"]

    if is_fisheye:
        pos_extras += [
            "fisheye lens distortion correct",
            "barrel distortion intentional",
            "curved horizon",
            "wide angle distortion consistent",
        ]
        neg_extras += ["corrected lens", "straight lines", "undistorted"]

    if is_isometric:
        pos_extras += [
            "true isometric projection",
            "no perspective convergence",
            "parallel lines stay parallel",
            "isometric grid correct",
            "axonometric accuracy",
        ]
        neg_extras += ["perspective convergence", "vanishing point", "standard perspective"]

    if is_pov:
        pos_extras += [
            "first person perspective",
            "viewer's hands visible",
            "pov shot correct",
            "immersive first person view",
            "camera as eyes",
        ]
        neg_extras += ["third person", "observer outside scene", "standard framing"]

    pos = merge_csv_unique(", ".join(TECHNICAL_POSITIVE_TOKENS), ", ".join(pos_extras))
    neg = merge_csv_unique(", ".join(TECHNICAL_NEGATIVE_TOKENS), ", ".join(neg_extras))
    return pos, neg


# ---------------------------------------------------------------------------
# 6. STYLE FUSION (combining incompatible styles)
# ---------------------------------------------------------------------------
# Failure modes addressed:
#   - Two incompatible styles collapse to whichever is more common in training.
#   - "Photorealistic anime" becomes either photo or anime, not both.
#   - Style fusion prompts produce muddy, incoherent blends.
#   - The dominant style overwhelms the secondary style completely.
#   - Style tokens fight each other and produce artifacts.
#   - "2.5D" and "semi-realistic" are understood but other fusions are not.
# Strategy: use explicit fusion grammar tokens, establish a dominant/modifier
#   hierarchy, and suppress the model's tendency to resolve to one style.
# Note: this complements STYLE_MIX_TIPS in prompt_domains.py but goes deeper
#   into the token-level mechanics of making fusions work.

STYLE_FUSION_POSITIVE_TOKENS: List[str] = [
    # Fusion grammar anchors
    "style fusion intentional",
    "hybrid style",
    "deliberate style blend",
    "two styles coexisting",
    "style synthesis",
    # Hierarchy anchors (dominant + modifier pattern)
    "dominant style with secondary influence",
    "primary style with stylistic accent",
    "base style modified by",
    # Known working fusion phrases (trained on these)
    "2.5d",
    "semi-realistic",
    "photorealistic anime",
    "stylized realistic",
    "painterly photograph",
    "illustrated realism",
    "anime 3d hybrid",
    # Coherence anchors
    "coherent style throughout",
    "consistent hybrid aesthetic",
    "unified visual language",
    "style-consistent lighting",
    "style-consistent materials",
    # Quality anchors for fusion
    "masterpiece style fusion",
    "high quality hybrid",
    "detailed style blend",
    "sharp focus on fusion elements",
]

STYLE_FUSION_NEGATIVE_TOKENS: List[str] = [
    # Suppress style collapse to dominant
    "pure anime",
    "pure photorealistic",
    "pure illustration",
    "single style only",
    "style resolved to one",
    # Suppress muddy blend artifacts
    "muddy blend",
    "incoherent style",
    "style clash artifacts",
    "conflicting visual language",
    "style seams visible",
    # Suppress lora-stack artifacts (relevant when using LoRAs for fusion)
    "lora artifact seams",
    "ghosting from adaptation",
    "oversaturated lora bloom",
    # Quality suppressors
    "low quality",
    "blurry",
    "flat",
    "generic",
]

# Known fusion pairs and their recommended grammar tokens
_FUSION_PAIR_TOKENS: Dict[str, Tuple[List[str], List[str]]] = {
    "photorealistic_anime": (
        [
            "photorealistic anime",
            "anime with photorealistic rendering",
            "realistic anime style",
            "anime character photorealistic",
            "3d anime style",
        ],
        ["pure anime flat", "pure photorealistic no anime", "style collapse"],
    ),
    "oil_digital": (
        [
            "oil painting digital hybrid",
            "painterly digital art",
            "digital oil painting",
            "oil paint texture digital",
        ],
        ["pure digital clean", "pure oil painting only"],
    ),
    "watercolor_ink": (
        [
            "watercolor and ink",
            "ink wash with watercolor",
            "watercolor over ink lines",
            "ink and wash technique",
        ],
        ["pure watercolor no lines", "pure ink no color"],
    ),
    "3d_illustration": (
        [
            "3d render illustration style",
            "illustrated 3d",
            "3d with illustration aesthetic",
            "render with painterly finish",
        ],
        ["pure 3d render", "pure illustration flat"],
    ),
    "pixel_painterly": (
        [
            "pixel art with painterly shading",
            "painterly pixel art",
            "pixel art painterly hybrid",
        ],
        ["pure pixel art crisp", "pure painterly no pixels"],
    ),
    "noir_color": (
        [
            "noir with selective color",
            "black and white with color accent",
            "monochrome with color pop",
            "sin city style color",
        ],
        ["full color", "full monochrome"],
    ),
}


def style_fusion_helpers(prompt: str) -> Tuple[str, str]:
    """
    Return ``(positive_addon, negative_addon)`` for style fusion prompts.

    Detects which styles are being fused and applies the appropriate fusion
    grammar tokens.  The dominant/modifier hierarchy is critical — the model
    needs to know which style is the base and which is the accent.  Without
    this, it will collapse to whichever style is more common in its training.

    Args:
        prompt: The user's positive prompt string.

    Returns:
        Tuple of (positive_addon, negative_addon) comma-separated token strings.
    """
    p = prompt.lower()

    pos_extras: List[str] = []
    neg_extras: List[str] = []

    # Detect fusion pairs
    has_photo = any(w in p for w in ("photorealistic", "realistic", "photo", "photograph"))
    has_anime = any(w in p for w in ("anime", "manga", "cel shaded", "toon"))
    has_oil = any(w in p for w in ("oil painting", "oil paint", "painterly"))
    has_digital = any(w in p for w in ("digital art", "digital painting", "digital illustration"))
    has_watercolor = any(w in p for w in ("watercolor", "water color", "aquarelle"))
    has_ink = any(w in p for w in ("ink", "lineart", "line art", "inking"))
    has_3d = any(w in p for w in ("3d render", "3d", "cgi", "blender", "octane"))
    has_illustration = any(w in p for w in ("illustration", "illustrated", "illustrative"))
    has_pixel = any(w in p for w in ("pixel art", "pixelart", "pixel"))
    has_noir = any(w in p for w in ("noir", "black and white", "monochrome", "grayscale"))
    has_color = any(w in p for w in ("color", "colour", "vibrant", "colorful"))

    if has_photo and has_anime:
        extras = _FUSION_PAIR_TOKENS["photorealistic_anime"]
        pos_extras.extend(extras[0])
        neg_extras.extend(extras[1])

    if has_oil and has_digital:
        extras = _FUSION_PAIR_TOKENS["oil_digital"]
        pos_extras.extend(extras[0])
        neg_extras.extend(extras[1])

    if has_watercolor and has_ink:
        extras = _FUSION_PAIR_TOKENS["watercolor_ink"]
        pos_extras.extend(extras[0])
        neg_extras.extend(extras[1])

    if has_3d and has_illustration:
        extras = _FUSION_PAIR_TOKENS["3d_illustration"]
        pos_extras.extend(extras[0])
        neg_extras.extend(extras[1])

    if has_pixel and (has_oil or has_digital):
        extras = _FUSION_PAIR_TOKENS["pixel_painterly"]
        pos_extras.extend(extras[0])
        neg_extras.extend(extras[1])

    if has_noir and has_color:
        extras = _FUSION_PAIR_TOKENS["noir_color"]
        pos_extras.extend(extras[0])
        neg_extras.extend(extras[1])

    pos = merge_csv_unique(", ".join(STYLE_FUSION_POSITIVE_TOKENS), ", ".join(pos_extras))
    neg = merge_csv_unique(", ".join(STYLE_FUSION_NEGATIVE_TOKENS), ", ".join(neg_extras))
    return pos, neg


# ---------------------------------------------------------------------------
# 7. MINIMALIST / NEGATIVE-SPACE
# ---------------------------------------------------------------------------
# Failure modes addressed:
#   - "Minimalist" gets interpreted as "simple low-quality" and produces
#     low-detail outputs rather than intentional restraint.
#   - Negative space gets filled in by the model's "horror vacui" prior.
#   - "White background" becomes a grey gradient or gets populated.
#   - Minimalist compositions lose the intentional emptiness.
#   - Single-subject minimalism gets cluttered with background detail.
#   - The model interprets emptiness as "unfinished" and adds elements.
# Strategy: explicitly frame emptiness as intentional, use design/art
#   vocabulary for minimalism, and aggressively suppress the fill-in prior.

MINIMALIST_POSITIVE_TOKENS: List[str] = [
    # Minimalism framing anchors
    "minimalist",
    "intentional minimalism",
    "deliberate restraint",
    "less is more",
    "negative space as design element",
    "emptiness intentional",
    # Negative space anchors
    "generous negative space",
    "breathing room",
    "open composition",
    "subject isolated",
    "subject floating in space",
    "clean empty background",
    # Design vocabulary for minimalism
    "graphic design minimalism",
    "zen composition",
    "japanese minimalism",
    "wabi-sabi aesthetic",
    "ma concept negative space",
    # Technical minimalism anchors
    "clean white background",
    "pure white background",
    "flat clean background",
    "no background detail",
    "background as void",
    # Subject treatment in minimalism
    "single focal point",
    "subject as only element",
    "isolated subject",
    "subject with space to breathe",
    # Quality anchors for minimalism
    "masterpiece minimalist",
    "high quality minimal",
    "precise minimalism",
    "intentional composition",
]

MINIMALIST_NEGATIVE_TOKENS: List[str] = [
    # Suppress horror vacui (filling empty space)
    "busy background",
    "detailed background",
    "background elements",
    "background clutter",
    "filled composition",
    "horror vacui",
    # Suppress "unfinished" misread
    "incomplete",
    "unfinished",
    "sketch",
    "rough",
    "low detail",
    # Suppress gradient/grey background default
    "gradient background",
    "grey background",
    "vignette",
    "background blur",
    "bokeh background",
    # Suppress added elements
    "extra elements",
    "decorative elements",
    "ornamental",
    "busy",
    "complex",
    "cluttered",
    # Quality suppressors
    "low quality",
    "blurry",
    "flat in wrong way",
    "generic",
]


def minimalist_helpers(prompt: str) -> Tuple[str, str]:
    """
    Return ``(positive_addon, negative_addon)`` for minimalist/negative-space prompts.

    The core challenge is that diffusion models have a strong "fill the frame"
    prior from training on detailed images.  Minimalism requires actively
    suppressing this prior while framing emptiness as intentional design.
    The positive tokens use design vocabulary (ma, wabi-sabi, zen) that the
    model associates with intentional restraint rather than incompleteness.

    Args:
        prompt: The user's positive prompt string.

    Returns:
        Tuple of (positive_addon, negative_addon) comma-separated token strings.
    """
    p = prompt.lower()

    is_white_bg = any(w in p for w in ("white background", "white bg", "isolated on white", "studio white"))
    is_zen = any(w in p for w in ("zen", "japanese", "wabi", "sabi", "ma ", "negative space"))
    is_graphic = any(w in p for w in ("graphic design", "poster", "logo", "icon", "flat design"))
    is_single_subject = any(w in p for w in ("single", "one subject", "solo", "isolated", "alone", "just the"))

    pos_extras: List[str] = []
    neg_extras: List[str] = []

    if is_white_bg:
        pos_extras += [
            "pure white background no gradient",
            "clean white void",
            "white background intentional",
            "subject on white",
            "no shadow on background",
        ]
        neg_extras += ["grey background", "off-white", "cream background", "gradient white"]

    if is_zen:
        pos_extras += [
            "zen minimalism",
            "ma negative space",
            "japanese aesthetic restraint",
            "wabi-sabi imperfect simplicity",
            "empty space as meaning",
        ]
        neg_extras += ["ornate", "decorative", "busy", "western maximalism"]

    if is_graphic:
        pos_extras += [
            "graphic design clarity",
            "clean vector-like simplicity",
            "design-first composition",
            "readable at a glance",
            "strong silhouette",
        ]
        neg_extras += ["painterly texture", "organic noise", "complex rendering"]

    if is_single_subject:
        pos_extras += [
            "single subject only",
            "nothing else in frame",
            "subject as sole element",
            "no supporting elements",
        ]
        neg_extras += ["multiple elements", "supporting objects", "props", "environment detail"]

    pos = merge_csv_unique(", ".join(MINIMALIST_POSITIVE_TOKENS), ", ".join(pos_extras))
    neg = merge_csv_unique(", ".join(MINIMALIST_NEGATIVE_TOKENS), ", ".join(neg_extras))
    return pos, neg


# ---------------------------------------------------------------------------
# 8. NSFW ANATOMY PRECISION
# ---------------------------------------------------------------------------
# Failure modes addressed:
#   - Generic anatomy tags ("bad anatomy", "correct anatomy") are too vague.
#   - Per-body-part precision is needed for NSFW content where anatomy errors
#     are more visible and more consequential.
#   - Models conflate different body parts or produce impossible geometry.
#   - Proportions drift when multiple body parts are described.
#   - Skin texture and surface detail are inconsistent across body regions.
#   - Lighting on curved surfaces (breasts, buttocks, thighs) is often wrong.
#   - Nipple/genital anatomy is frequently malformed without specific tokens.
# Strategy: per-body-part precision token sets that address the specific
#   failure modes for each region, plus surface/lighting anchors.
# Note: these tokens are for legitimate adult content generation.
#   No censorship — the model learns what you show it.

NSFW_PRECISION_POSITIVE_TOKENS: List[str] = [
    # Global anatomy anchors
    "correct anatomy",
    "anatomically accurate",
    "believable proportions",
    "coherent body",
    "natural body proportions",
    # Skin surface anchors
    "natural skin texture",
    "skin pores visible",
    "subtle skin imperfections",
    "believable subsurface scattering",
    "skin tone consistent",
    "skin texture consistent across body",
    # Lighting on curved surfaces
    "correct lighting on curved surfaces",
    "specular highlights follow form",
    "soft shadows in body contours",
    "rim light on body edges",
    "ambient occlusion in body creases",
    # Pose and weight
    "believable weight distribution",
    "gravity-correct pose",
    "natural body contact with surface",
    "skin compression where body contacts surface",
    "natural skin fold at contact points",
    # Quality anchors
    "masterpiece anatomy",
    "detailed body",
    "high quality figure",
    "sharp focus on anatomy",
]

NSFW_PRECISION_NEGATIVE_TOKENS: List[str] = [
    # Global anatomy suppressors
    "bad anatomy",
    "wrong anatomy",
    "impossible anatomy",
    "broken anatomy",
    "deformed body",
    # Proportion errors
    "wrong proportions",
    "mismatched body parts",
    "inconsistent scale",
    "floating body parts",
    # Skin surface errors
    "plastic skin",
    "airbrushed skin",
    "waxy skin",
    "doll-like skin",
    "uniform skin texture",
    "no skin texture",
    # Lighting errors on body
    "flat lighting on body",
    "no shadows on body",
    "wrong specular on skin",
    "inconsistent lighting across body",
    # Quality suppressors
    "low quality",
    "blurry",
    "flat",
    "generic",
]

# Per-body-part precision token maps
# Each entry: body_part -> (positive_tokens, negative_tokens)
# WHY: generic anatomy tags don't target specific regions; per-part tokens
#   activate the model's region-specific knowledge and suppress region-specific
#   failure modes.
NSFW_BODY_PART_PRECISION: Dict[str, Tuple[List[str], List[str]]] = {
    "breasts": (
        [
            # Shape and form
            "natural breast shape",
            "gravity-correct breast position",
            "breast shape follows pose",
            "natural breast sag",
            "breast volume consistent",
            # Surface and texture
            "natural areola",
            "areola correct size",
            "nipple correct position",
            "nipple detail",
            "natural nipple shape",
            "breast skin texture natural",
            # Lighting
            "correct lighting on breast curves",
            "soft shadow under breast",
            "specular highlight on breast correct",
            "subsurface scattering on breast",
        ],
        [
            # Shape errors
            "floating breasts",
            "gravity-defying breasts",
            "wrong breast shape",
            "breast shape inconsistent",
            "breast size inconsistent",
            # Surface errors
            "missing nipples",
            "wrong nipple position",
            "nipple too large",
            "nipple too small",
            "areola wrong color",
            # Lighting errors
            "flat lighting on breasts",
            "no shadow under breast",
        ],
    ),
    "buttocks": (
        [
            # Shape and form
            "natural buttock shape",
            "gravity-correct buttock position",
            "gluteal fold natural",
            "buttock shape follows pose",
            "natural buttock volume",
            # Surface
            "buttock skin texture natural",
            "natural skin compression when seated",
            "natural buttock crease",
            # Lighting
            "correct lighting on buttock curves",
            "soft shadow in gluteal fold",
            "specular on buttock correct",
        ],
        [
            "wrong buttock shape",
            "floating buttocks",
            "buttock shape inconsistent",
            "missing gluteal fold",
            "flat buttocks",
            "no buttock definition",
        ],
    ),
    "genitals_female": (
        [
            # Anatomy
            "correct female genitalia",
            "anatomically accurate vulva",
            "labia natural shape",
            "labia minora visible",
            "labia majora natural",
            "clitoris correct position",
            "vaginal opening correct",
            # Surface
            "natural skin texture on genitals",
            "correct skin color",
            # Lighting
            "correct lighting on genital area",
            "soft shadow in folds",
        ],
        [
            "wrong female anatomy",
            "incorrect vulva",
            "missing labia",
            "wrong labia shape",
            "anatomically incorrect genitals",
            "smooth featureless genitals",
            "barbie anatomy",
        ],
    ),
    "genitals_male": (
        [
            # Anatomy
            "correct male genitalia",
            "anatomically accurate penis",
            "natural penis shape",
            "glans correct",
            "foreskin natural",
            "scrotum natural shape",
            "testicles correct",
            # Surface
            "natural skin texture on genitals",
            "correct skin color",
            "natural skin folds",
            # Lighting
            "correct lighting on genital area",
        ],
        [
            "wrong male anatomy",
            "incorrect penis shape",
            "anatomically incorrect genitals",
            "wrong proportions genitals",
            "missing scrotum",
            "smooth featureless genitals",
        ],
    ),
    "thighs": (
        [
            "natural thigh shape",
            "thigh gap or no gap natural",
            "inner thigh skin texture",
            "thigh compression when seated",
            "natural thigh crease",
            "thigh muscle definition natural",
            "correct lighting on thigh curves",
        ],
        [
            "wrong thigh shape",
            "thigh gap inconsistent",
            "flat thighs",
            "no thigh definition",
            "thigh shape inconsistent",
        ],
    ),
    "abdomen": (
        [
            "natural abdomen shape",
            "belly button correct position",
            "navel natural shape",
            "abdominal muscle definition natural",
            "natural belly softness",
            "skin fold at waist natural",
            "correct lighting on abdomen",
        ],
        [
            "wrong belly button position",
            "missing navel",
            "wrong navel shape",
            "flat abdomen unrealistic",
            "abdomen shape inconsistent",
        ],
    ),
    "neck_shoulders": (
        [
            "natural neck shape",
            "neck muscle definition natural",
            "shoulder width correct",
            "shoulder blade visible naturally",
            "clavicle visible naturally",
            "neck-shoulder transition natural",
            "correct lighting on neck and shoulders",
        ],
        [
            "wrong neck shape",
            "neck too long",
            "neck too short",
            "shoulder width inconsistent",
            "missing clavicle",
            "neck-shoulder merge",
        ],
    ),
    "hands_feet": (
        [
            # Hands
            "correct hands",
            "five fingers",
            "natural hand shape",
            "finger length correct",
            "knuckle detail natural",
            "fingernail detail",
            "hand pose natural",
            # Feet
            "correct feet",
            "five toes",
            "natural foot shape",
            "toe length correct",
            "toenail detail",
            "foot arch natural",
        ],
        [
            "extra fingers",
            "missing fingers",
            "fused fingers",
            "wrong finger count",
            "deformed hands",
            "extra toes",
            "missing toes",
            "fused toes",
            "deformed feet",
        ],
    ),
    "face": (
        [
            "natural facial features",
            "eyes correct size",
            "eyes symmetrical natural",
            "iris detail",
            "pupil correct",
            "eyelash detail",
            "nose correct shape",
            "lips natural shape",
            "teeth visible natural",
            "ear correct position",
            "facial asymmetry natural",
            "skin texture on face",
        ],
        [
            "dead eyes",
            "glassy eyes",
            "wrong eye size",
            "asymmetric eyes wrong",
            "missing iris detail",
            "wrong nose shape",
            "wrong lip shape",
            "uncanny valley face",
            "mask-like expression",
            "perfect symmetry uncanny",
        ],
    ),
}


def nsfw_precision_helpers(prompt: str) -> Tuple[str, str]:
    """
    Return ``(positive_addon, negative_addon)`` for NSFW anatomy precision prompts.

    Detects which body parts are mentioned and applies per-part precision tokens.
    The key insight is that generic anatomy tags ("bad anatomy", "correct anatomy")
    are too vague for NSFW content — the model needs region-specific tokens to
    activate its per-region knowledge and suppress per-region failure modes.

    Per-body-part tokens are drawn from ``NSFW_BODY_PART_PRECISION`` and merged
    with the global ``NSFW_PRECISION_POSITIVE_TOKENS`` / ``NSFW_PRECISION_NEGATIVE_TOKENS``.

    Args:
        prompt: The user's positive prompt string.

    Returns:
        Tuple of (positive_addon, negative_addon) comma-separated token strings.
    """
    p = prompt.lower()

    pos_extras: List[str] = []
    neg_extras: List[str] = []

    # Body part detection keywords
    _part_keywords: Dict[str, List[str]] = {
        "breasts": ["breast", "boob", "tit", "nipple", "areola", "chest", "cleavage", "topless", "nude upper"],
        "buttocks": ["butt", "ass", "buttock", "glute", "rear", "behind", "bottom", "booty"],
        "genitals_female": [
            "vagina",
            "vulva",
            "pussy",
            "labia",
            "clitoris",
            "nude female",
            "naked female",
            "female genitals",
        ],
        "genitals_male": [
            "penis",
            "cock",
            "dick",
            "phallus",
            "erect",
            "nude male",
            "naked male",
            "male genitals",
        ],
        "thighs": ["thigh", "inner thigh", "leg", "legs", "thighs"],
        "abdomen": ["abdomen", "belly", "stomach", "navel", "belly button", "abs", "midriff"],
        "neck_shoulders": ["neck", "shoulder", "shoulders", "clavicle", "collarbone", "décolletage"],
        "hands_feet": ["hand", "hands", "finger", "fingers", "foot", "feet", "toe", "toes"],
        "face": ["face", "portrait", "eyes", "lips", "nose", "expression", "close-up face"],
    }

    for part, keywords in _part_keywords.items():
        if any(kw in p for kw in keywords):
            part_pos, part_neg = NSFW_BODY_PART_PRECISION[part]
            pos_extras.extend(part_pos)
            neg_extras.extend(part_neg)

    pos = merge_csv_unique(", ".join(NSFW_PRECISION_POSITIVE_TOKENS), ", ".join(pos_extras))
    neg = merge_csv_unique(", ".join(NSFW_PRECISION_NEGATIVE_TOKENS), ", ".join(neg_extras))
    return pos, neg


# ---------------------------------------------------------------------------
# 9. AUTO-DETECTION: classify_prompt_category()
# ---------------------------------------------------------------------------
# Detection strategy:
#   - Use word-boundary regex for precision (avoids "horror" matching "horrible").
#   - Score each category by keyword matches; return highest-scoring category.
#   - Tie-breaking: more specific categories win over "standard".
#   - NSFW precision is detected by explicit body-part vocabulary.
#   - Style fusion requires detecting two or more style keywords.
#   - Minimalist requires explicit minimalism vocabulary (not just "simple").

# Detection keyword sets per category
_DETECT_PATTERNS: Dict[str, List[re.Pattern]] = {}


def _compile_detect(terms: List[str]) -> List[re.Pattern]:
    return [re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in terms]


_DETECT_PATTERNS[CATEGORY_WEIRD] = _compile_detect(
    [
        "surreal",
        "surrealism",
        "dreamlike",
        "uncanny",
        "liminal",
        "non-euclidean",
        "impossible",
        "abstract",
        "weird",
        "strange",
        "bizarre",
        "kafkaesque",
        "magritte",
        "dali",
        "de chirico",
        "dreamscape",
        "backrooms",
        "void",
        "melting",
        "morphing",
        "impossible geometry",
        "wrong",
        "unsettling",
    ]
)

_DETECT_PATTERNS[CATEGORY_HORROR] = _compile_detect(
    [
        "horror",
        "scary",
        "terrifying",
        "dread",
        "disturbing",
        "dark atmosphere",
        "eldritch",
        "lovecraft",
        "cthulhu",
        "body horror",
        "psychological horror",
        "folk horror",
        "haunted",
        "ghost",
        "demon",
        "monster",
        "creature",
        "gore",
        "blood",
        "visceral",
        "grotesque",
        "macabre",
        "sinister",
        "ominous",
        "foreboding",
        "creepy",
        "eerie",
        "unsettling darkness",
    ]
)

_DETECT_PATTERNS[CATEGORY_NARRATIVE] = _compile_detect(
    [
        "moment before",
        "just before",
        "about to",
        "aftermath",
        "after the",
        "in the wake",
        "story",
        "narrative",
        "implied",
        "environmental storytelling",
        "memory",
        "flashback",
        "nostalgia",
        "discovery",
        "finds",
        "stumbles upon",
        "the last",
        "the first",
        "the only",
        "witness",
        "survivor",
        "ruins of",
        "remains of",
        "wreckage",
        "evidence of",
    ]
)

_DETECT_PATTERNS[CATEGORY_EMOTION] = _compile_detect(
    [
        "melancholy",
        "joy",
        "joyful",
        "dread",
        "wonder",
        "loneliness",
        "lonely",
        "tenderness",
        "tender",
        "anxiety",
        "anxious",
        "nostalgia",
        "nostalgic",
        "awe",
        "resignation",
        "resigned",
        "bittersweet",
        "wistful",
        "poignant",
        "serene",
        "peaceful",
        "tranquil",
        "aching",
        "yearning",
        "hopeful",
        "despair",
        "grief",
        "elation",
        "euphoria",
        "melancholic",
        "somber",
        "mood",
        "atmosphere",
        "feeling",
        "emotion",
        "emotional",
    ]
)

_DETECT_PATTERNS[CATEGORY_TECHNICAL] = _compile_detect(
    [
        "foreshortening",
        "foreshortened",
        "from below",
        "from above",
        "worm's eye",
        "bird's eye",
        "dutch angle",
        "fisheye",
        "fish eye",
        "ultra wide",
        "isometric",
        "axonometric",
        "pov",
        "point of view",
        "first person",
        "extreme angle",
        "dramatic perspective",
        "overhead",
        "top down",
        "multiple subjects",
        "group composition",
        "crowd",
        "complex composition",
        "overlapping figures",
        "depth layers",
    ]
)

_DETECT_PATTERNS[CATEGORY_MINIMALIST] = _compile_detect(
    [
        "minimalist",
        "minimalism",
        "minimal",
        "negative space",
        "empty",
        "white background",
        "clean background",
        "isolated",
        "zen",
        "wabi-sabi",
        "sparse",
        "simple composition",
        "breathing room",
        "less is more",
        "single subject",
        "void background",
        "pure white",
    ]
)

_DETECT_PATTERNS[CATEGORY_NSFW_PRECISION] = _compile_detect(
    [
        "nsfw",
        "nude",
        "naked",
        "explicit",
        "adult content",
        "18+",
        "breast",
        "nipple",
        "vagina",
        "vulva",
        "penis",
        "genitals",
        "topless",
        "bottomless",
        "nude female",
        "nude male",
        "erotic",
        "sexual",
        "sex",
        "intercourse",
        "penetration",
    ]
)

# Style fusion requires two style keywords — handled separately in classify
_STYLE_KEYWORDS: List[str] = [
    "photorealistic",
    "realistic",
    "anime",
    "manga",
    "cartoon",
    "illustration",
    "oil painting",
    "watercolor",
    "digital art",
    "3d render",
    "pixel art",
    "sketch",
    "ink",
    "cel shaded",
    "painterly",
    "noir",
    "black and white",
]


def classify_prompt_category(prompt: str) -> str:
    """
    Classify a prompt into one of the special helper categories.

    Returns one of: ``"weird"``, ``"horror"``, ``"narrative"``, ``"emotion"``,
    ``"technical"``, ``"style_fusion"``, ``"minimalist"``, ``"nsfw_precision"``,
    ``"standard"``.

    Detection is keyword-based with word-boundary matching.  When multiple
    categories score equally, the more specific category wins.  Style fusion
    requires detecting two or more incompatible style keywords.

    Args:
        prompt: The user's positive prompt string.

    Returns:
        Category string constant (one of the ``CATEGORY_*`` constants).
    """
    if not prompt or not prompt.strip():
        return CATEGORY_STANDARD

    p = prompt.lower()

    # Score each category
    scores: Dict[str, int] = {}
    for category, patterns in _DETECT_PATTERNS.items():
        score = sum(1 for pat in patterns if pat.search(p))
        if score > 0:
            scores[category] = score

    # Style fusion: check for two or more incompatible style keywords
    style_hits = sum(1 for kw in _STYLE_KEYWORDS if re.search(rf"\b{re.escape(kw)}\b", p, re.IGNORECASE))
    if style_hits >= 2:
        scores[CATEGORY_STYLE_FUSION] = scores.get(CATEGORY_STYLE_FUSION, 0) + style_hits

    if not scores:
        return CATEGORY_STANDARD

    # Return highest-scoring category
    # Tie-breaking priority: nsfw_precision > horror > weird > technical > narrative > emotion > style_fusion > minimalist
    priority_order = [
        CATEGORY_NSFW_PRECISION,
        CATEGORY_HORROR,
        CATEGORY_WEIRD,
        CATEGORY_TECHNICAL,
        CATEGORY_NARRATIVE,
        CATEGORY_EMOTION,
        CATEGORY_STYLE_FUSION,
        CATEGORY_MINIMALIST,
    ]

    best_score = max(scores.values())
    best_categories = [cat for cat, score in scores.items() if score == best_score]

    if len(best_categories) == 1:
        return best_categories[0]

    # Tie-break by priority order
    for cat in priority_order:
        if cat in best_categories:
            return cat

    return best_categories[0]


# ---------------------------------------------------------------------------
# 10. MAIN ENTRY POINT: apply_special_helpers()
# ---------------------------------------------------------------------------

# Map category names to their helper functions
_CATEGORY_HELPERS: Dict[str, object] = {
    CATEGORY_WEIRD: weird_helpers,
    CATEGORY_HORROR: horror_helpers,
    CATEGORY_NARRATIVE: narrative_helpers,
    CATEGORY_EMOTION: emotion_helpers,
    CATEGORY_TECHNICAL: technical_helpers,
    CATEGORY_STYLE_FUSION: style_fusion_helpers,
    CATEGORY_MINIMALIST: minimalist_helpers,
    CATEGORY_NSFW_PRECISION: nsfw_precision_helpers,
}


def apply_special_helpers(
    prompt: str,
    negative: str,
    category: str = "auto",
) -> Tuple[str, str]:
    """
    Apply special prompt helpers for the given category and return enriched
    ``(positive_prompt, negative_prompt)`` strings.

    This is the single entry point that wires all category helpers together.
    It merges the helper's token additions into the caller's existing prompt
    and negative using ``merge_csv_unique()`` (de-duplicated, order-preserved).

    Args:
        prompt:
            The user's positive prompt string.  Returned enriched with
            category-appropriate positive tokens appended.
        negative:
            The user's negative prompt string.  Returned enriched with
            category-appropriate negative tokens appended.
        category:
            One of the ``CATEGORY_*`` constants, or ``"auto"`` (default) to
            auto-detect via ``classify_prompt_category()``.  Pass
            ``"standard"`` to skip all helpers and return inputs unchanged.

    Returns:
        ``(enriched_positive, enriched_negative)`` — comma-separated token
        strings ready to pass directly to the sampler.

    Examples::

        # Auto-detect and apply
        pos, neg = apply_special_helpers(
            "the moment before the battle, soldier gripping sword",
            "low quality, blurry",
        )

        # Explicit category
        pos, neg = apply_special_helpers(
            "surreal dreamscape with impossible stairs",
            "low quality",
            category="weird",
        )

        # NSFW with anatomy precision
        pos, neg = apply_special_helpers(
            "nude woman lying on bed, natural lighting",
            "bad anatomy, low quality",
            category="nsfw_precision",
        )
    """
    if not prompt:
        return prompt, negative

    # Resolve category
    resolved = str(category or "auto").lower().strip()
    if resolved == "auto":
        resolved = classify_prompt_category(prompt)

    # Standard / unknown: return unchanged
    if resolved == CATEGORY_STANDARD or resolved not in _CATEGORY_HELPERS:
        return prompt, negative

    # Call the appropriate helper
    helper = _CATEGORY_HELPERS[resolved]
    pos_addon, neg_addon = helper(prompt)  # type: ignore[operator]

    # Merge with caller's existing prompt/negative
    enriched_positive = merge_csv_unique(prompt, pos_addon)
    enriched_negative = merge_csv_unique(negative, neg_addon)

    return enriched_positive, enriched_negative


# ---------------------------------------------------------------------------
# Convenience: apply multiple categories at once
# ---------------------------------------------------------------------------


def apply_multiple_helpers(
    prompt: str,
    negative: str,
    categories: List[str],
) -> Tuple[str, str]:
    """
    Apply multiple category helpers in sequence and merge all additions.

    Useful when a prompt spans multiple categories (e.g. horror + narrative,
    or technical + nsfw_precision).  Each helper's additions are merged
    cumulatively.

    Args:
        prompt:   The user's positive prompt string.
        negative: The user's negative prompt string.
        categories:
            List of ``CATEGORY_*`` constants.  ``"auto"`` and ``"standard"``
            are silently skipped.

    Returns:
        ``(enriched_positive, enriched_negative)``
    """
    pos = str(prompt or "")
    neg = str(negative or "")

    for cat in categories:
        c = str(cat or "").lower().strip()
        if c in ("auto", CATEGORY_STANDARD) or c not in _CATEGORY_HELPERS:
            continue
        helper = _CATEGORY_HELPERS[c]
        pos_addon, neg_addon = helper(pos)  # type: ignore[operator]
        pos = merge_csv_unique(pos, pos_addon)
        neg = merge_csv_unique(neg, neg_addon)

    return pos, neg


# ---------------------------------------------------------------------------
# Convenience: suggest categories for a prompt (for UI / tooling)
# ---------------------------------------------------------------------------


def suggest_categories(prompt: str, *, top_n: int = 3) -> List[str]:
    """
    Return the top-N most likely categories for a prompt, ranked by score.

    Useful for UI tooling that wants to show the user which helpers are
    relevant without automatically applying them.

    Args:
        prompt: The user's positive prompt string.
        top_n:  Maximum number of categories to return (default 3).

    Returns:
        List of category strings, highest-scoring first.  May be shorter
        than ``top_n`` if fewer categories match.  Never includes
        ``"standard"``.
    """
    if not prompt or not prompt.strip():
        return []

    p = prompt.lower()
    scores: Dict[str, int] = {}

    for category, patterns in _DETECT_PATTERNS.items():
        score = sum(1 for pat in patterns if pat.search(p))
        if score > 0:
            scores[category] = score

    # Style fusion scoring
    style_hits = sum(1 for kw in _STYLE_KEYWORDS if re.search(rf"\b{re.escape(kw)}\b", p, re.IGNORECASE))
    if style_hits >= 2:
        scores[CATEGORY_STYLE_FUSION] = scores.get(CATEGORY_STYLE_FUSION, 0) + style_hits

    sorted_cats = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
    return sorted_cats[:top_n]


# ---------------------------------------------------------------------------
# Convenience: get token lists for a category (for inspection / debugging)
# ---------------------------------------------------------------------------


def get_category_tokens(category: str) -> Tuple[List[str], List[str]]:
    """
    Return the base positive and negative token lists for a category.

    Returns the static token lists (not the dynamic per-prompt additions).
    Useful for inspection, debugging, or building UI token pickers.

    Args:
        category: One of the ``CATEGORY_*`` constants.

    Returns:
        ``(positive_tokens, negative_tokens)`` — lists of token strings.
        Returns ``([], [])`` for ``"standard"`` or unknown categories.
    """
    _token_map: Dict[str, Tuple[List[str], List[str]]] = {
        CATEGORY_WEIRD: (WEIRD_POSITIVE_TOKENS, WEIRD_NEGATIVE_TOKENS),
        CATEGORY_HORROR: (HORROR_POSITIVE_TOKENS, HORROR_NEGATIVE_TOKENS),
        CATEGORY_NARRATIVE: (NARRATIVE_POSITIVE_TOKENS, NARRATIVE_NEGATIVE_TOKENS),
        CATEGORY_EMOTION: (EMOTION_POSITIVE_TOKENS, EMOTION_NEGATIVE_TOKENS),
        CATEGORY_TECHNICAL: (TECHNICAL_POSITIVE_TOKENS, TECHNICAL_NEGATIVE_TOKENS),
        CATEGORY_STYLE_FUSION: (STYLE_FUSION_POSITIVE_TOKENS, STYLE_FUSION_NEGATIVE_TOKENS),
        CATEGORY_MINIMALIST: (MINIMALIST_POSITIVE_TOKENS, MINIMALIST_NEGATIVE_TOKENS),
        CATEGORY_NSFW_PRECISION: (NSFW_PRECISION_POSITIVE_TOKENS, NSFW_PRECISION_NEGATIVE_TOKENS),
    }
    return _token_map.get(category, ([], []))

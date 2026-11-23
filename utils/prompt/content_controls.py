"""Content-control tag packs for harder generation targets, with the built-in tag tables."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from config.prompts import (
    ANTI_AI_LOOK_NEGATIVE,
    ANTI_AI_LOOK_NEGATIVE_STRONG,
    NATURAL_LOOK_POSITIVE,
    NATURAL_LOOK_POSITIVE_DEEP,
    dedupe_tags,
)

from utils.prompt.realism import WEAK_DOMAIN_BUCKETS as _WEAK_DOMAIN_BUCKETS
from utils.prompt.realism import WEAK_DOMAIN_PACKS as _WEAK_DOMAIN_PACKS
from utils.prompt.realism import infer_hard_controls as _infer_hard_controls
from utils.prompt.text import append_unique, split_tags

BUILTIN_TAG_TABLES: dict = {
    "adherence_standard_negative": {
        "_": [
            "wrong subject",
            "missing key elements",
            "contradicts the description",
            "omitted props",
            "ignored clothing details",
            "wrong outfit",
            "wrong hair color",
            "prompt drift",
            "wrong accessory set",
            "missing stated background object",
            "invented jewelry or props",
            "wrong material finish",
            "ignored symmetry request",
            "swapped character roles",
            "extra figures not mentioned",
            "wrong time of day",
        ]
    },
    "adherence_standard_positive": {
        "_": [
            "faithful to the prompt",
            "accurate to description",
            "all described elements present",
            "literal scene interpretation",
            "coherent with written details",
            "follow the caption literally",
            "respect every named color and material",
            "preserve stated headcount and roles",
            "keep background elements from the prompt",
            "correct handedness and mirroring",
            "consistent prop placement",
            "no invented accessories",
            "match era and setting cues",
            "honor camera and lens wording",
        ]
    },
    "adherence_strict_extra_negative": {
        "_": [
            "hallucinated details",
            "extra subjects not in description",
            "merged conflicting concepts",
            "anachronistic props",
            "inconsistent character identity",
            "ignored foreground objects",
            "contradictory lighting vs prompt",
            "hallucinated architecture",
            "fabric pattern not as described",
            "wrong vehicle or product model",
            "ignored explicit color words",
            "added logos or watermarks",
            "changed art medium from prompt",
        ]
    },
    "adherence_strict_extra_positive": {
        "_": [
            "strict prompt adherence",
            "no invented objects beyond the description",
            "preserve every mentioned attribute",
            "exact outfit and props as described",
            "zero filler objects not in the prompt",
            "no substitute wardrobe items",
            "exact palette from the description",
            "preserve negative space as described",
            "lock framing to stated crop",
            "verbatim pose keywords honored",
            "no genre drift from the brief",
        ]
    },
    "advanced_pose_positive": {
        "ahegao": [
            "ahegao face",
            "tongue out",
            "heart shaped pupils",
            "rolling eyes",
            "mind break",
            "pleasure overload",
        ],
        "climax": ["orgasm pose", "body shaking", "toes curled", "back arched in pleasure"],
        "dynamic": ["dynamic pose", "action pose", "motion blur on limbs", "weight shifted stance"],
        "erotic": [
            "seductive pose",
            "arched back",
            "legs spread",
            "hands running over body",
            "pose vocabulary honored literally",
        ],
        "sfw": [
            "elegant pose",
            "graceful stance",
            "reading pose",
            "sitting peacefully",
            "walking gracefully",
            "dancing pose",
            "contemplative pose",
        ],
        "struggle": ["struggling pose", "bound wrists", "resisting yet aroused"],
    },
    "angle_negative_shared": {
        "_": ["impossible perspective", "warped viewpoint", "broken vanishing point", "camera angle contradicts prompt"]
    },
    "angle_positive": {
        "bird_eye": [
            "bird's-eye view",
            "strong bird's-eye view",
            "more bird's-eye view",
            "bird's-eye view discipline",
            "overhead shot",
            "strong overhead shot",
            "more overhead shot",
            "overhead shot discipline",
            "coherent top-down geometry",
            "strong coherent top-down geometry",
            "more coherent top-down geometry",
            "coherent top-down geometry discipline",
        ],
        "dutch": ["dutch angle", "tilted camera", "intentional composition tilt"],
        "eye_level": [
            "eye-level shot",
            "strong eye-level shot",
            "more eye-level shot",
            "eye-level shot discipline",
            "natural perspective",
            "strong natural perspective",
            "more natural perspective",
            "perspective consistency",
            "natural perspective discipline",
            "framing matches crop words",
            "strong framing matches crop words",
            "more framing matches crop words",
            "framing matches crop words discipline",
        ],
        "first_person": ["first-person POV", "point-of-view shot", "hands in foreground", "POV matches prompt stance"],
        "high_angle": [
            "high-angle shot",
            "strong high-angle shot",
            "more high-angle shot",
            "high-angle shot discipline",
            "top-down perspective",
            "strong top-down perspective",
            "more top-down perspective",
            "perspective consistency",
            "top-down perspective discipline",
        ],
        "low_angle": [
            "low-angle shot",
            "strong low-angle shot",
            "more low-angle shot",
            "low-angle shot discipline",
            "upward perspective",
            "strong upward perspective",
            "more upward perspective",
            "perspective consistency",
            "upward perspective discipline",
            "correct foreshortening",
            "strong correct foreshortening",
            "more correct foreshortening",
            "correct foreshortening discipline",
        ],
        "over_shoulder": ["over-the-shoulder shot", "foreground shoulder framing"],
        "third_person": ["third-person viewpoint", "camera behind subject"],
        "worm_eye": [
            "worm's-eye view",
            "strong worm's-eye view",
            "more worm's-eye view",
            "worm's-eye view discipline",
            "extreme low-angle perspective",
            "strong extreme low-angle perspective",
            "more extreme low-angle perspective",
        ],
    },
    "artist_composition_negative": {
        "classical": [
            "chaotic value pattern",
            "stiff frozen symmetry",
            "merged ambiguous masses",
            "all edges equally sharp clutter",
            "decorative symmetry without purpose",
            "flat graphic shapes without mass",
            "too much detail everywhere",
            "edge noise everywhere",
            "massing breaks into small fragments",
            "values too close no notan read",
        ],
        "full": [
            "muddy focal point",
            "unclear subject hierarchy",
            "competing focal points",
            "dead corner visual traps",
            "no clear focal point",
            "confusing silhouette read",
            "equal spacing everywhere",
            "aimless eye wandering",
            "tangent edges at frame border",
            "flat muddy composition",
            "cluttered visual noise",
            "accidental center bullseye stiffness",
            "weak framing hierarchy",
            "motif repetition without variation",
            "ambiguous subframes",
            "no dominant subdominant structure",
            "busy background competes with subject",
            "overly centered static framing",
            "twinning mirrored elements by accident",
            "value contrast everywhere no hierarchy",
            "saturation everywhere no hierarchy",
            "cropping cuts important forms awkwardly",
            "conflicting vanishing points",
            "warped perspective grid",
            "impossible geometry",
            "broken horizon continuity",
            "scale inconsistency in depth",
            "shadow direction contradicts light source",
            "overlapping depth order contradictions",
            "floating objects with no ground contact",
            "verticals tilt inconsistently",
            "depth cues contradict each other",
            "camera height inconsistent with horizon",
            "chaotic value pattern",
            "stiff frozen symmetry",
            "merged ambiguous masses",
            "all edges equally sharp clutter",
            "decorative symmetry without purpose",
            "flat graphic shapes without mass",
            "too much detail everywhere",
            "edge noise everywhere",
            "massing breaks into small fragments",
            "values too close no notan read",
        ],
        "lite": [
            "muddy focal point",
            "unclear subject hierarchy",
            "competing focal points",
            "dead corner visual traps",
            "no clear focal point",
            "confusing silhouette read",
            "equal spacing everywhere",
            "aimless eye wandering",
        ],
        "perspective": [
            "conflicting vanishing points",
            "warped perspective grid",
            "impossible geometry",
            "broken horizon continuity",
            "scale inconsistency in depth",
            "shadow direction contradicts light source",
            "overlapping depth order contradictions",
            "floating objects with no ground contact",
            "verticals tilt inconsistently",
            "depth cues contradict each other",
            "camera height inconsistent with horizon",
        ],
        "standard": [
            "muddy focal point",
            "unclear subject hierarchy",
            "tangent edges at frame border",
            "flat muddy composition",
            "cluttered visual noise",
            "accidental center bullseye stiffness",
            "weak framing hierarchy",
            "motif repetition without variation",
            "ambiguous subframes",
            "no dominant subdominant structure",
            "busy background competes with subject",
            "overly centered static framing",
            "twinning mirrored elements by accident",
            "value contrast everywhere no hierarchy",
            "saturation everywhere no hierarchy",
            "cropping cuts important forms awkwardly",
        ],
    },
    "artist_composition_positive": {
        "classical": [
            "academic composition discipline",
            "S-curve rhythm in forms",
            "dynamic symmetry",
            "chiaroscuro friendly value grouping",
            "notan readable lights and darks",
            "arabesque flow between masses",
            "figure ground clarity",
            "lost and found edge rhythm",
            "pyramidal mass composition",
            "repoussoir framing depth",
            "classical economy of shapes",
            "bilateral symmetry used with intent",
            "classical balance of light and dark masses",
            "armature of the rectangle",
            "classical line of action",
            "massing before detail",
            "economy of detail at focal point",
            "edge hierarchy supports focal point",
            "simplify into two or three value groups",
            "silhouette design reads clearly",
        ],
        "full": [
            "rule of thirds composition",
            "clear focal point",
            "intentional visual hierarchy",
            "balanced negative space",
            "readable silhouette separation",
            "triangular grouping of subjects",
            "gestalt closure friendly shapes",
            "clear eye entry path",
            "clear value read at thumbnail",
            "big medium small shape design",
            "rule of odds grouping",
            "dominant shape language",
            "golden ratio friendly framing",
            "leading lines guiding the eye",
            "foreground midground background layers",
            "dynamic asymmetrical balance",
            "rhythm in shapes and value masses",
            "cohesive color harmony in the layout",
            "golden spiral friendly focal placement",
            "repetition with variation in motifs",
            "enclosure and framing hierarchy",
            "diagonal thrust with visual counterbalance",
            "subframing supports primary subject",
            "picture plane clarity",
            "unity with variety in the layout",
            "dominant subdominant subordinate",
            "visual weight balance across frame",
            "controlled saturation hierarchy",
            "warm cool color temperature contrast",
            "shape value color unity",
            "simplified background value masses",
            "clear secondary focal support",
            "intentional cropping and cutoff",
            "rhythm of light accents",
            "rest area for the eye",
            "consistent linear perspective",
            "clear horizon line",
            "stable vanishing point",
            "orthogonal architectural perspective",
            "scale diminution with distance",
            "atmospheric perspective depth",
            "eye level relationship to ground plane",
            "natural foreshortening",
            "one point perspective cues",
            "two point perspective cues",
            "three point perspective cues",
            "overlapping forms show depth order",
            "unified shadow direction for light logic",
            "curvilinear perspective consistency",
            "convincing ground plane contact",
            "object overlap obeys spatial depth",
            "consistent scale landmarks",
            "verticals converge consistently",
            "level camera avoids unintended tilt",
            "depth cues via size overlap and contrast",
            "horizon line matches viewpoint intent",
            "academic composition discipline",
            "S-curve rhythm in forms",
            "dynamic symmetry",
            "chiaroscuro friendly value grouping",
            "notan readable lights and darks",
            "arabesque flow between masses",
            "figure ground clarity",
            "lost and found edge rhythm",
            "pyramidal mass composition",
            "repoussoir framing depth",
            "classical economy of shapes",
            "bilateral symmetry used with intent",
            "classical balance of light and dark masses",
            "armature of the rectangle",
            "classical line of action",
            "massing before detail",
            "economy of detail at focal point",
            "edge hierarchy supports focal point",
            "simplify into two or three value groups",
            "silhouette design reads clearly",
        ],
        "lite": [
            "rule of thirds composition",
            "strong rule of thirds composition",
            "more rule of thirds composition",
            "rule of thirds composition discipline",
            "clear focal point",
            "strong focal point",
            "more clear focal point",
            "clear focal point discipline",
            "intentional visual hierarchy",
            "more intentional visual hierarchy",
            "intentional visual hierarchy and readability",
            "intentional visual hierarchy discipline",
            "balanced negative space",
            "strong balanced negative space",
            "more balanced negative space",
            "balance and stabilityd negative space",
            "balanced negative space discipline",
            "readable silhouette separation",
            "strong readable silhouette separation",
            "more readable silhouette separation",
            "readable silhouette separation discipline",
            "triangular grouping of subjects",
            "strong triangular grouping of subjects",
            "more triangular grouping of subjects",
            "triangular grouping of subjects discipline",
            "gestalt closure friendly shapes",
            "gestalt closure shapes",
            "strong gestalt closure friendly shapes",
            "more gestalt closure friendly shapes",
            "gestalt closure friendly shapes discipline",
            "clear eye entry path",
            "strong eye entry path",
            "more clear eye entry path",
            "clear eye entry path discipline",
            "clear value read at thumbnail",
            "strong value read at thumbnail",
            "more clear value read at thumbnail",
            "clear value read at thumbnail discipline",
            "big medium small shape design",
            "strong big medium small shape design",
            "more big medium small shape design",
            "big medium small shape design discipline",
            "rule of odds grouping",
            "strong rule of odds grouping",
            "more rule of odds grouping",
            "rule of odds grouping discipline",
            "dominant shape language",
            "strong dominant shape language",
            "more dominant shape language",
            "dominant shape language discipline",
        ],
        "perspective": [
            "consistent linear perspective",
            "clear horizon line",
            "stable vanishing point",
            "orthogonal architectural perspective",
            "scale diminution with distance",
            "atmospheric perspective depth",
            "eye level relationship to ground plane",
            "natural foreshortening",
            "one point perspective cues",
            "two point perspective cues",
            "three point perspective cues",
            "overlapping forms show depth order",
            "unified shadow direction for light logic",
            "curvilinear perspective consistency",
            "convincing ground plane contact",
            "object overlap obeys spatial depth",
            "consistent scale landmarks",
            "verticals converge consistently",
            "level camera avoids unintended tilt",
            "depth cues via size overlap and contrast",
            "horizon line matches viewpoint intent",
        ],
        "standard": [
            "rule of thirds composition",
            "strong rule of thirds composition",
            "more rule of thirds composition",
            "rule of thirds composition discipline",
            "clear focal point",
            "strong focal point",
            "more clear focal point",
            "clear focal point discipline",
            "intentional visual hierarchy",
            "more intentional visual hierarchy",
            "intentional visual hierarchy and readability",
            "intentional visual hierarchy discipline",
            "balanced negative space",
            "strong balanced negative space",
            "more balanced negative space",
            "balance and stabilityd negative space",
            "balanced negative space discipline",
            "readable silhouette separation",
            "strong readable silhouette separation",
            "more readable silhouette separation",
            "readable silhouette separation discipline",
            "golden ratio friendly framing",
            "golden ratio framing",
            "strong golden ratio friendly framing",
            "more golden ratio friendly framing",
            "golden ratio friendly framing discipline",
            "leading lines guiding the eye",
            "strong leading lines guiding the eye",
            "more leading lines guiding the eye",
            "leading lines guiding the eye discipline",
            "foreground midground background layers",
            "strong foreground midground background layers",
            "more foreground midground background layers",
            "foreground midground background layers discipline",
            "dynamic asymmetrical balance",
            "strong dynamic asymmetrical balance",
            "more dynamic asymmetrical balance",
            "dynamic asymmetrical balance and stability",
            "dynamic asymmetrical balance discipline",
            "rhythm in shapes and value masses",
            "strong rhythm in shapes and value masses",
            "more rhythm in shapes and value masses",
            "rhythm in shapes and value masses discipline",
            "cohesive color harmony in the layout",
            "strong cohesive color harmony in the layout",
            "more cohesive color harmony in the layout",
            "cohesive color harmony in the layout discipline",
            "golden spiral friendly focal placement",
            "golden spiral focal placement",
            "strong golden spiral friendly focal placement",
            "more golden spiral friendly focal placement",
            "golden spiral friendly focal placement discipline",
            "repetition with variation in motifs",
            "strong repetition with variation in motifs",
            "more repetition with variation in motifs",
            "repetition with variation in motifs discipline",
            "enclosure and framing hierarchy",
            "strong enclosure and framing hierarchy",
            "more enclosure and framing hierarchy",
            "enclosure and framing hierarchy and readability",
            "enclosure and framing hierarchy discipline",
            "diagonal thrust with visual counterbalance",
            "strong diagonal thrust with visual counterbalance",
            "more diagonal thrust with visual counterbalance",
            "diagonal thrust with visual counterbalance and stability",
            "diagonal thrust with visual counterbalance discipline",
            "subframing supports primary subject",
            "strong subframing supports primary subject",
            "more subframing supports primary subject",
            "subframing supports primary subject discipline",
            "picture plane clarity",
            "strong picture plane clarity",
            "more picture plane clarity",
            "picture plane clarity discipline",
            "unity with variety in the layout",
            "strong unity with variety in the layout",
            "more unity with variety in the layout",
            "unity with variety in the layout discipline",
            "dominant subdominant subordinate",
            "strong dominant subdominant subordinate",
            "more dominant subdominant subordinate",
            "visual weight balance across frame",
            "controlled saturation hierarchy",
            "warm cool color temperature contrast",
            "shape value color unity",
            "simplified background value masses",
            "clear secondary focal support",
            "intentional cropping and cutoff",
            "rhythm of light accents",
            "rest area for the eye",
        ],
    },
    "background_negative": {
        "indoor": ["impossible room geometry", "warped walls", "floating furniture"],
        "minimal": ["busy background", "visual clutter"],
        "nature": ["muddy foliage", "texture soup", "background collapse", "wrong biome vs prompt"],
        "outdoor": ["flat backdrop", "collapsed depth", "incoherent horizon"],
        "studio": ["cluttered background", "background noise", "foreground bleed"],
        "urban": ["warped buildings", "broken vanishing point", "street distortion", "invented signage text"],
    },
    "background_positive": {
        "indoor": ["coherent indoor environment", "interior depth cues", "room geometry consistency"],
        "minimal": ["minimal background", "subject emphasis", "clean negative space"],
        "nature": ["natural landscape depth", "organic environment detail", "foliage separation", "flora as described"],
        "outdoor": ["coherent outdoor environment", "natural atmospheric depth", "ground-plane consistency"],
        "studio": ["studio backdrop", "controlled lighting", "clean background separation"],
        "urban": [
            "urban environment detail",
            "architectural perspective",
            "street-scale coherence",
            "architecture matches named city cues",
        ],
    },
    "best_model_tags": {
        "_": [
            "score_9",
            "strong score_9",
            "score_9 discipline",
            "score_8_up",
            "strong score_8_up",
            "score_8_up discipline",
            "score_7_up",
            "strong score_7_up",
            "score_7_up discipline",
            "masterpiece",
            "strong masterpiece",
            "masterpiece discipline",
            "best quality",
            "strong best quality",
            "more best quality",
            "best quality discipline",
            "amazing quality",
            "strong amazing quality",
            "more amazing quality",
            "amazing quality discipline",
            "newest",
            "strong newest",
            "newest discipline",
            "absurdres",
            "strong absurdres",
            "absurdres discipline",
            "highres",
            "strong highres",
            "highres discipline",
            "extremely detailed",
            "strong extremely detailed",
            "more extremely detailed",
            "extremely detailed discipline",
            "ultra detailed",
            "strong ultra detailed",
            "more ultra detailed",
            "ultra detailed discipline",
            "detailed eyes",
            "strong detailed eyes",
            "more detailed eyes",
            "detailed eyes discipline",
            "detailed face",
            "strong detailed face",
            "more detailed face",
            "detailed face discipline",
            "detailed hair",
            "strong detailed hair",
            "more detailed hair",
            "detailed hair discipline",
            "detailed skin",
            "strong detailed skin",
            "more detailed skin",
            "detailed skin discipline",
            "detailed hands",
            "strong detailed hands",
            "more detailed hands",
            "detailed hands discipline",
            "detailed background",
            "strong detailed background",
            "more detailed background",
            "detailed background discipline",
            "fine details",
            "strong fine details",
            "more fine details",
            "fine details discipline",
            "intricate details",
            "strong intricate details",
            "more intricate details",
            "intricate details discipline",
            "sharp focus",
            "strong sharp focus",
            "more sharp focus",
            "sharp focus discipline",
            "volumetric lighting",
            "strong volumetric lighting",
            "more volumetric lighting",
            "volumetric lighting discipline",
            "cinematic lighting",
            "strong cinematic lighting",
            "more cinematic lighting",
            "cinematic lighting discipline",
            "professional lighting",
            "strong professional lighting",
            "more professional lighting",
            "professional lighting discipline",
            "rim lighting",
            "strong rim lighting",
            "more rim lighting",
            "rim lighting discipline",
            "soft shadows",
            "strong soft shadows",
            "more soft shadows",
            "soft shadows discipline",
            "depth of field",
            "strong depth of field",
            "more depth of field",
            "depth of field discipline",
            "dynamic composition",
            "strong dynamic composition",
            "more dynamic composition",
            "dynamic composition discipline",
            "rule of thirds",
            "strong rule of thirds",
            "more rule of thirds",
            "rule of thirds discipline",
            "rich colors",
            "strong rich colors",
            "more rich colors",
            "rich colors discipline",
            "vivid colors",
            "strong vivid colors",
            "more vivid colors",
            "vivid colors discipline",
            "high contrast",
            "strong high contrast",
            "more high contrast",
            "high contrast discipline",
            "film grain subtle",
            "strong film grain subtle",
            "more film grain subtle",
            "film grain subtle discipline",
            "chromatic aberration subtle",
            "strong chromatic aberration subtle",
            "more chromatic aberration subtle",
            "chromatic aberration subtle discipline",
            "tag-accurate rendering",
            "strong tag-accurate rendering",
            "more tag-accurate rendering",
            "tag-accurate rendering discipline",
            "caption consistency",
            "strong caption consistency",
            "more caption consistency",
            "caption consistency discipline",
            "literal object inventory",
            "strong literal object inventory",
            "more literal object inventory",
        ]
    },
    "body_proportion_positive": {
        "exaggerated": ["exaggerated proportions", "hyper sexualized body", "extreme curves", "unrealistic penis size"],
        "hyper": ["hyper proportions", "massive penis", "impossibly long cock", "extreme body modification"],
        "realistic": ["realistic human proportions", "natural body ratios", "ratios match character description"],
    },
    "clothing_negative": {
        "armor": ["merged armor plates", "floating armor parts"],
        "bikini": ["broken straps", "warped swimsuit geometry", "asymmetric cups"],
        "casual": [
            "melted clothing",
            "fused fabric",
            "broken seams",
            "floating clothes",
            "wrong garment category vs prompt",
        ],
        "danbooru": ["melted costume", "fused fabric", "broken accessories", "ignored stated accessory"],
        "fantasy_armor": ["merged armor plates", "floating armor parts", "warped metal"],
        "formal": ["distorted suit", "warped tie", "asymmetric lapels", "invented pattern not mentioned"],
        "lingerie": ["broken lingerie straps", "warped lace", "plastic looking fabric"],
        "nude": ["deformed anatomy", "extra limbs", "broken joints"],
        "sfw": ["melted clothing", "fused fabric", "inconsistent outfit"],
        "streetwear": ["muddy clothing details", "garment fusion", "texture collapse"],
        "swimwear": ["broken straps", "warped swimsuit geometry"],
        "torn": ["badly torn clothes", "inconsistent damage"],
    },
    "clothing_positive": {
        "armor": ["fantasy armor", "revealing armor", "metal bikini armor", "leather straps", "bikini armor"],
        "bikini": ["micro bikini", "string bikini", "swimsuit pulled aside"],
        "casual": [
            "casual outfit",
            "strong casual outfit",
            "more casual outfit",
            "casual outfit discipline",
            "natural fabric folds",
            "strong natural fabric folds",
            "more natural fabric folds",
            "natural fabric folds discipline",
            "realistic cloth drape",
            "strong realistic cloth drape",
            "more realistic cloth drape",
            "realistic cloth drape discipline",
            "detailed clothing texture",
            "strong detailed clothing texture",
            "more detailed clothing texture",
            "detailed clothing texture discipline",
            "fabric colors from the prompt",
            "strong fabric colors from the prompt",
            "more fabric colors from the prompt",
            "fabric colors from the prompt discipline",
        ],
        "danbooru": [
            "maid outfit",
            "school uniform",
            "thighhighs",
            "garter straps",
            "fishnets",
            "latex suit",
            "leather corset",
            "pasties",
            "crotchless panties",
            "microskirt",
            "sailor uniform",
            "cheerleader outfit",
            "gothic lolita",
            "nun habit",
            "shrine maiden",
            "bikini armor",
            "miko outfit",
            "succubus outfit",
            "demon outfit",
            "angel outfit",
            "witch hat and robe",
            "cat ears and tail",
            "bunny suit",
            "maid headdress",
        ],
        "fantasy_armor": [
            "fantasy armor",
            "strong fantasy armor",
            "more fantasy armor",
            "fantasy armor discipline",
            "readable armor segments",
            "strong readable armor segments",
            "more readable armor segments",
            "readable armor segments discipline",
            "material separation",
            "strong material separation",
            "more material separation",
            "material separation discipline",
            "ornate armor details",
            "strong ornate armor details",
            "more ornate armor details",
            "ornate armor details discipline",
        ],
        "formal": [
            "formal outfit",
            "strong formal outfit",
            "more formal outfit",
            "formal outfit discipline",
            "tailored clothing",
            "strong tailored clothing",
            "more tailored clothing",
            "tailored clothing discipline",
            "clean garment lines",
            "strong clean garment lines",
            "more clean garment lines",
            "clean garment lines discipline",
            "elegant dress",
            "strong elegant dress",
            "more elegant dress",
            "elegant dress discipline",
            "lapel and tie as described",
            "strong lapel and tie as described",
            "more lapel and tie as described",
            "lapel and tie as described discipline",
        ],
        "lingerie": [
            "sexy lingerie",
            "strong sexy lingerie",
            "more sexy lingerie",
            "sexy lingerie discipline",
            "lace lingerie",
            "strong lace lingerie",
            "more lace lingerie",
            "lace lingerie discipline",
            "sheer fabric",
            "strong sheer fabric",
            "more sheer fabric",
            "sheer fabric discipline",
            "garter belts",
            "strong garter belts",
            "more garter belts",
            "garter belts discipline",
            "stockings",
            "strong stockings",
            "stockings discipline",
            "bodysuit",
            "strong bodysuit",
            "bodysuit discipline",
            "teddy",
            "strong teddy",
            "teddy discipline",
            "babydoll",
            "strong babydoll",
            "babydoll discipline",
            "corset",
            "strong corset",
            "corset discipline",
            "cage harness",
            "strong cage harness",
            "more cage harness",
            "cage harness discipline",
            "set pieces match tag list",
            "strong set pieces match tag list",
            "more set pieces match tag list",
            "set pieces match tag list discipline",
        ],
        "nude": ["completely nude", "fully naked", "no clothing", "bare skin"],
        "sfw": [
            "casual clothes",
            "elegant gown",
            "adventurer outfit",
            "cozy sweater",
            "historical dress",
            "sportswear",
            "royal attire",
            "fantasy robes",
            "simple tunic",
            "flowing dress",
            "hoodie and shorts",
            "summer dress",
            "winter coat",
            "apron dress",
            "pajamas",
            "uniform",
            "oversized sweater",
            "cardigan",
            "blouse and skirt",
            "sundress",
            "jeans and t-shirt",
            "cute dress",
            "kawaii outfit",
            "pastel clothes",
        ],
        "streetwear": [
            "streetwear style",
            "strong streetwear style",
            "more streetwear style",
            "streetwear style discipline",
            "layered outfit",
            "strong layered outfit",
            "more layered outfit",
            "layered outfit discipline",
            "distinct garment details",
            "strong distinct garment details",
            "more distinct garment details",
            "distinct garment details discipline",
            "urban fashion",
            "strong urban fashion",
            "more urban fashion",
            "urban fashion discipline",
            "brand-agnostic but silhouette-locked to brief",
            "strong brand-agnostic but silhouette-locked to brief",
            "more brand-agnostic but silhouette-locked to brief",
            "brand-agnostic but silhouette-locked to brief discipline",
        ],
        "swimwear": [
            "swimwear",
            "strong swimwear",
            "swimwear discipline",
            "clean silhouette",
            "strong clean silhouette",
            "more clean silhouette",
            "clean silhouette discipline",
            "consistent fabric tension",
            "strong consistent fabric tension",
            "more consistent fabric tension",
            "consistent fabric tension discipline",
            "summer beachwear",
            "strong summer beachwear",
            "more summer beachwear",
            "summer beachwear discipline",
        ],
        "torn": [
            "torn clothes",
            "strong torn clothes",
            "more torn clothes",
            "torn clothes discipline",
            "ripped clothing",
            "strong ripped clothing",
            "clothes in tatters",
            "partially undressed",
            "naked apron",
        ],
    },
    "composition_negative": {
        "cinematic": ["awkward framing", "confusing perspective", "visual clutter", "ignored camera instruction"],
        "group": ["subject fusion", "merged faces", "merged limbs", "missing listed person"],
        "multi_character": [
            "subject fusion",
            "merged faces",
            "merged limbs",
            "same outfit on both",
            "outfit swap",
            "clothes mixed between characters",
            "wrong outfit on wrong person",
            "identical twins when not intended",
            "cloned face on both",
            "merged bodies",
            "conjoined",
            "shared limbs between people",
            "pose copied onto wrong figure",
            "role swap vs description",
        ],
        "scene": ["flat composition", "collapsed depth", "scale mismatch", "wrong landmark or room layout"],
        "single_subject": [
            "duplicate person",
            "cloned subject",
            "double head",
            "extra face",
            "extra figure not in prompt",
        ],
    },
    "composition_positive": {
        "cinematic": [
            "cinematic framing",
            "intentional camera blocking",
            "strong depth cues",
            "shot size matches wording",
        ],
        "group": [
            "multi-character composition",
            "clear subject separation",
            "non-overlapping silhouettes",
            "headcount matches caption",
        ],
        "multi_character": [
            "multi-character composition",
            "clear subject separation",
            "non-overlapping silhouettes",
            "each person has distinct clothing",
            "each figure has own pose",
            "readable silhouettes",
            "spatial separation between subjects",
            "each figure matches their description",
        ],
        "scene": [
            "environment-first composition",
            "coherent depth layering",
            "foreground midground background",
            "environment layout from the prompt",
        ],
        "single_subject": [
            "single subject focus",
            "clean silhouette",
            "clear subject isolation",
            "prompt-specified solo framing",
        ],
    },
    "conflicting_tag_pairs": {
        "_": [
            "front view|||back view",
            "strong front view|||back view",
            "more front view|||back view",
            "front view|||back view discipline",
            "bird's-eye view|||worm's-eye view",
            "strong bird's-eye view|||worm's-eye view",
            "more bird's-eye view|||worm's-eye view",
            "bird's-eye view|||worm's-eye view discipline",
            "top-down perspective|||low-angle shot",
            "strong top-down perspective|||low-angle shot",
            "more top-down perspective|||low-angle shot",
            "top-down perspective|||low-angle shot discipline",
            "daylight|||night scene",
            "strong daylight|||night scene",
            "more daylight|||night scene",
            "daylight|||night scene discipline",
            "indoor|||outdoor",
            "strong indoor|||outdoor",
            "indoor|||outdoor discipline",
            "single subject|||crowd",
            "strong single subject|||crowd",
            "more single subject|||crowd",
            "single subject|||crowd discipline",
            "photoreal|||toon shading",
            "strong photoreal|||toon shading",
            "more photoreal|||toon shading",
            "photoreal|||toon shading discipline",
            "flat color|||volumetric lighting",
            "strong flat color|||volumetric lighting",
            "more flat color|||volumetric lighting",
            "flat color|||volumetric lighting discipline",
            "monochrome|||rainbow palette",
            "strong monochrome|||rainbow palette",
            "more monochrome|||rainbow palette",
            "monochrome|||rainbow palette discipline",
            "minimalist|||maximalist clutter",
            "strong minimalist|||maximalist clutter",
            "more minimalist|||maximalist clutter",
            "minimalist|||maximalist clutter discipline",
            "vintage film|||neon cyberpunk",
            "strong vintage film|||neon cyberpunk",
            "more vintage film|||neon cyberpunk",
            "vintage film|||neon cyberpunk discipline",
            "macro close-up|||wide establishing shot",
            "strong macro close-up|||wide establishing shot",
            "more macro close-up|||wide establishing shot",
            "macro close-up|||wide establishing shot discipline",
            "symmetrical layout|||chaotic asymmetry",
            "strong symmetrical layout|||chaotic asymmetry",
            "more symmetrical layout|||chaotic asymmetry",
            "symmetrical layout|||chaotic asymmetry discipline",
        ]
    },
    "domain_negative": {
        "architecture": [
            "distorted perspective",
            "strong distorted perspective",
            "more distorted perspective",
            "perspective consistency",
            "distorted perspective discipline",
            "collapsed geometry",
            "strong collapsed geometry",
            "more collapsed geometry",
            "collapsed geometry discipline",
            "warped straight lines",
            "strong warped straight lines",
            "more warped straight lines",
            "warped straight lines discipline",
        ],
        "buildings": [
            "bent building lines",
            "strong bent building lines",
            "more bent building lines",
            "bent building lines discipline",
            "impossible architecture",
            "strong impossible architecture",
            "more impossible architecture",
            "impossible architecture discipline",
            "warped windows",
            "strong warped windows",
            "more warped windows",
            "warped windows discipline",
        ],
        "mixed": [
            "object merging",
            "strong object merging",
            "more object merging",
            "object merging discipline",
            "foreground-background confusion",
            "strong foreground-background confusion",
            "more foreground-background confusion",
            "foreground-background confusion discipline",
            "scale mismatch",
            "strong scale mismatch",
            "more scale mismatch",
            "scale mismatch discipline",
            "substituted product or vehicle",
            "strong substituted product or vehicle",
            "more substituted product or vehicle",
            "substituted product or vehicle discipline",
        ],
        "objects": [
            "melted object",
            "strong melted object",
            "more melted object",
            "melted object discipline",
            "fused object",
            "strong fused object",
            "more fused object",
            "fused object discipline",
            "undefined object shape",
            "strong undefined object shape",
            "more undefined object shape",
            "undefined object shape discipline",
        ],
        "vehicles": [
            "deformed wheels",
            "strong deformed wheels",
            "more deformed wheels",
            "deformed wheels discipline",
            "vehicle body distortion",
            "strong vehicle body distortion",
            "more vehicle body distortion",
            "vehicle body distortion discipline",
            "floating car parts",
            "strong floating car parts",
            "more floating car parts",
            "floating car parts discipline",
        ],
    },
    "domain_positive": {
        "architecture": [
            "architectural composition",
            "strong architectural composition",
            "more architectural composition",
            "architectural composition discipline",
            "building scale coherence",
            "strong building scale coherence",
            "more building scale coherence",
            "building scale coherence discipline",
            "clean geometry",
            "strong clean geometry",
            "more clean geometry",
            "clean geometry discipline",
        ],
        "buildings": [
            "architectural perspective",
            "strong architectural perspective",
            "more architectural perspective",
            "perspective consistency",
            "architectural perspective discipline",
            "straight structural lines",
            "strong straight structural lines",
            "more straight structural lines",
            "straight structural lines discipline",
            "facade detail",
            "strong facade detail",
            "more facade detail",
            "facade detail discipline",
        ],
        "mixed": [
            "multi-object scene coherence",
            "strong multi-object scene coherence",
            "more multi-object scene coherence",
            "multi-object scene coherence discipline",
            "clear object hierarchy",
            "strong object hierarchy",
            "more clear object hierarchy",
            "clear object hierarchy and readability",
            "clear object hierarchy discipline",
            "consistent perspective",
            "strong consistent perspective",
            "more consistent perspective",
            "perspective consistency",
            "consistent perspective discipline",
            "object list from prompt intact",
            "strong object list from prompt intact",
            "more object list from prompt intact",
            "object list from prompt intact discipline",
        ],
        "objects": [
            "product-style object clarity",
            "strong product-style object clarity",
            "more product-style object clarity",
            "product-style object clarity discipline",
            "distinct object edges",
            "strong distinct object edges",
            "more distinct object edges",
            "distinct object edges discipline",
            "material realism",
            "strong material realism",
            "more material realism",
            "material realism discipline",
        ],
        "vehicles": [
            "accurate vehicle geometry",
            "strong accurate vehicle geometry",
            "more accurate vehicle geometry",
            "accurate vehicle geometry discipline",
            "coherent wheel/body alignment",
            "strong coherent wheel/body alignment",
            "more coherent wheel/body alignment",
            "coherent wheel/body alignment discipline",
            "automotive detail",
            "strong automotive detail",
            "more automotive detail",
            "automotive detail discipline",
        ],
    },
    "duplicate_subject_negative": {
        "_": ["duplicate subject", "cloned person", "double face", "extra head", "extra torso", "clone not in caption"]
    },
    "environment_detail_positive": {
        "bedroom": [
            "luxurious bedroom",
            "silk sheets",
            "dim romantic lighting",
            "candlelit room",
            "props list from caption only",
        ],
        "danbooru": [
            "bedroom",
            "bathroom",
            "classroom",
            "office",
            "forest",
            "beach",
            "alley",
            "rooftop",
            "throne room",
            "room type matches tag",
        ],
        "dungeon": ["dark dungeon", "chains on wall", "stone floor", "bdsm dungeon"],
        "luxury": ["luxury hotel suite", "penthouse view", "marble floor"],
        "outdoor": ["forest clearing sex", "public risk", "against tree", "moonlit night", "weather matches prompt"],
        "sfw": [
            "cozy living room",
            "sunlit garden",
            "peaceful library",
            "art studio",
            "warm kitchen",
            "mountain cabin",
            "flower field",
        ],
    },
    "hand_negative": {
        "detailed": ["melted fingers", "deformed knuckles", "finger blobs", "wrong hand pose vs prompt"],
        "grip": ["floating hands", "object-hand clipping", "impossible grip"],
        "stable": ["extra fingers", "missing fingers", "fused fingers", "broken hands"],
    },
    "hand_positive": {
        "detailed": [
            "detailed hands",
            "defined knuckles",
            "natural finger joints",
            "visible fingernails",
            "finger count matches gesture description",
        ],
        "grip": ["hands holding object", "physically plausible grip", "clear contact points"],
        "stable": ["natural hands", "five fingers", "clear finger separation", "anatomically correct hands"],
    },
    "human_media_negative": {
        "dslr": ["phone computational mush", "over processed smartphone hdr"],
        "film": ["digital noise pattern", "sensor banding as grain"],
        "photographic": [
            "cg render",
            "game engine screenshot",
            "3d model turntable",
            "fake hdr",
            "look contradicts film vs digital cue",
        ],
    },
    "human_media_positive": {
        "dslr": [
            "dslr photograph",
            "full frame look",
            "natural lens bokeh",
            "pleasant optical vignette",
            "sharp subject natural falloff",
            "real lens character",
        ],
        "film": [
            "35mm film photograph",
            "analog film grain",
            "film color science",
            "subtle halation",
            "organic grain structure",
            "portra palette subtle",
        ],
        "photographic": [
            "authentic photograph",
            "believable exposure",
            "natural dynamic range",
            "real world texture",
            "photographic composition",
            "captured image not rendered",
            "exposure matches stated lighting words",
        ],
    },
    "interaction_intensity_positive": {
        "extreme": ["brutal sex", "violent thrusting", "maximum intensity"],
        "gentle": ["gentle lovemaking", "tender interaction", "intensity matches soft wording"],
        "intense": ["rough sex", "aggressive fucking", "powerful penetration"],
        "passionate": ["passionate sex", "intense thrusting"],
    },
    "lighting_mode_negative": {
        "dramatic_rim": ["flat lighting", "unmotivated highlights"],
        "high_key": ["blown highlights", "washed-out subject"],
        "low_key": ["crushed blacks", "shadow noise blotches"],
        "natural_daylight": ["unnatural color cast", "inconsistent sun direction", "time of day contradicts prompt"],
        "studio_softbox": ["harsh mixed lighting", "uneven exposure patches", "unmotivated fill vs brief"],
    },
    "lighting_mode_positive": {
        "dramatic_rim": [
            "dramatic rim light",
            "directional key light",
            "cinematic shadow sculpting",
            "rim matches stated light source",
        ],
        "high_key": ["high-key lighting", "bright soft tonality", "low shadow harshness"],
        "low_key": ["low-key lighting", "rich shadows", "controlled specular highlights"],
        "natural_daylight": ["natural daylight", "soft global illumination", "plausible bounce light"],
        "studio_softbox": [
            "studio softbox lighting",
            "controlled key/fill ratio",
            "clean light wrap",
            "key direction matches prompt",
        ],
    },
    "lora_scaffold_negative": {
        "blend": ["double adaptation clash", "split style artifact", "muddy lora blend", "ghosting from stacked loras"],
        "character_first": [
            "character face drift",
            "morphing identity",
            "inconsistent character",
            "ignored character tag order",
        ],
        "style_first": ["weak style adherence", "washed lora effect", "conflicting style cues"],
    },
    "lora_scaffold_positive": {
        "blend": [
            "coherent lora fusion",
            "unified adaptation",
            "single harmonized style",
            "consistent trigger strength feel",
            "prompt weights reflected in fusion",
        ],
        "character_first": [
            "locked character design",
            "consistent facial structure",
            "same outfit logic",
            "character model consistency",
        ],
        "style_first": ["dominant lora style", "style transfer coherence", "clear art direction from trigger"],
    },
    "object_layout_negative": {
        "asymmetrical": ["unbalanced clutter", "visual tangles"],
        "foreground_anchor": ["object float", "object-background merge"],
        "rule_of_thirds": ["center lock artifact", "composition collapse"],
        "symmetrical": ["asymmetric distortion", "skewed central axis"],
    },
    "object_layout_positive": {
        "asymmetrical": ["asymmetrical balance", "intentional visual weight distribution"],
        "foreground_anchor": ["foreground anchor object", "clear object silhouette", "depth separation"],
        "rule_of_thirds": ["rule of thirds object placement", "balanced composition"],
        "symmetrical": ["symmetrical layout", "centered structural balance"],
    },
    "objects_interaction_positive": {
        "danbooru": [
            "vibrator",
            "dildo",
            "anal plug",
            "handcuffs",
            "strap on",
            "wand vibrator",
            "anal beads",
            "tentacle",
            "magic wand",
            "book",
            "flower",
            "umbrella",
            "lantern",
            "basket",
            "tea cup",
        ],
        "furniture": [
            "fucked on table",
            "bent over desk",
            "against wall",
            "on bed with sheets gripped",
            "surface matches setting description",
        ],
        "held": [
            "holding object",
            "physically plausible grip",
            "clear contact points",
            "object identity from prompt only",
        ],
        "sex_toy": [
            "using sex toy",
            "dildo insertion",
            "vibrator on clit",
            "plug visible",
            "nipple clamps",
            "ball gag",
            "collar and leash",
            "riding crop",
        ],
        "sfw": [
            "holding book",
            "playing instrument",
            "holding flower",
            "reading book",
            "painting",
            "gardening",
            "holding lantern",
        ],
        "weapon": ["holding sword", "gun in hand", "sword at side", "weapon detail"],
    },
    "one_shot_negative": {
        "_": [
            "duplicate subject",
            "cloned face",
            "double head",
            "extra arms",
            "extra legs",
            "missing limbs",
            "bad hands",
            "mutated hands",
            "too many fingers",
            "fused fingers",
            "blurry",
            "out of focus subject",
            "muddy details",
            "jpeg artifacts",
            "worst quality",
            "low quality",
            "watermark",
            "cropped",
            "bad anatomy",
            "deformed",
            "disconnected limbs",
            "incoherent background",
            "inconsistent lighting",
            "visual clutter",
            "ignored foreground object",
            "scene creep beyond the brief",
            "wrong focal subject",
        ]
    },
    "one_shot_positive": {
        "_": [
            "single subject focus",
            "clear hero subject",
            "coherent scene",
            "consistent lighting direction",
            "correct perspective",
            "sharp subject",
            "well framed",
            "complete anatomy in frame",
            "natural colors",
            "high clarity",
            "subject isolation",
            "clean silhouette",
            "obvious focal point",
            "prompt-faithful composition",
            "caption-driven focal hierarchy",
            "no orphan props",
        ]
    },
    "people_layout_negative": {
        "duo": ["merged faces", "fused limbs", "third person artifact", "third silhouette not mentioned"],
        "group_large": ["subject collapse", "crowd blur soup", "indistinct faces"],
        "group_small": ["face duplication", "subject fusion", "missing limbs"],
        "solo": ["extra person", "background person", "duplicate body"],
    },
    "people_layout_positive": {
        "duo": ["two people", "distinct individuals", "clear separation between subjects"],
        "group_large": ["crowd scene", "layered subject depth", "clear foreground/background grouping"],
        "group_small": ["small group portrait", "distinct faces", "non-overlapping silhouettes"],
        "solo": ["single person", "one subject only", "clear subject focus", "solo count locked to prompt"],
    },
    "perspective_stability_negative": {
        "_": [
            "perspective drift",
            "scale drift",
            "warped geometry",
            "inconsistent scale",
            "ignored focal length cue",
            "warped floor grid vs prompt",
        ]
    },
    "perspective_stability_positive": {
        "_": [
            "consistent perspective",
            "stable vanishing points",
            "coherent scene scale",
            "geometry matches lens description",
        ]
    },
    "pose_naturalness_negative": {
        "dynamic_natural": [
            "impossible action pose",
            "off-balance body",
            "dislocated limbs",
            "unnatural contortion",
            "pose ignores physics note in prompt",
        ],
        "intimate_natural": ["anatomy merge", "impossible body contact", "fused torsos", "broken paired pose geometry"],
        "natural": ["stiff pose", "mannequin posture", "unnatural joint bend", "broken body mechanics"],
    },
    "pose_naturalness_positive": {
        "dynamic_natural": [
            "dynamic but physically plausible pose",
            "grounded center of mass",
            "natural limb articulation",
            "motion with anatomical consistency",
        ],
        "intimate_natural": [
            "coherent interpersonal contact",
            "anatomically plausible paired pose",
            "natural torso and hip alignment",
            "balanced body overlap",
        ],
        "natural": [
            "natural body posture",
            "realistic weight distribution",
            "anatomically plausible joint angles",
            "relaxed non-stiff pose",
            "weight believable per prompt",
        ],
    },
    "pose_negative_shared": {
        "_": [
            "broken limbs",
            "strong broken limbs",
            "more broken limbs",
            "broken limbs discipline",
            "disconnected joints",
            "strong disconnected joints",
            "more disconnected joints",
            "disconnected joints discipline",
            "extra arms",
            "strong extra arms",
            "more extra arms",
            "extra arms discipline",
            "extra legs",
            "strong extra legs",
            "more extra legs",
            "extra legs discipline",
            "twisted anatomy",
            "strong twisted anatomy",
            "more twisted anatomy",
            "twisted anatomy discipline",
        ]
    },
    "pose_positive": {
        "acrobatics": [
            "acrobatic pose",
            "strong acrobatic pose",
            "more acrobatic pose",
            "acrobatic pose discipline",
            "extreme perspective pose",
            "strong extreme perspective pose",
            "more extreme perspective pose",
            "perspective consistency",
            "extreme perspective pose discipline",
            "clear limb articulation",
            "strong limb articulation",
            "more clear limb articulation",
            "clear limb articulation discipline",
            "anatomical coherence",
            "strong anatomical coherence",
            "more anatomical coherence",
            "anatomical coherence discipline",
        ],
        "action": [
            "action pose",
            "strong action pose",
            "more action pose",
            "action pose discipline",
            "dynamic motion",
            "strong dynamic motion",
            "more dynamic motion",
            "dynamic motion discipline",
            "clear body mechanics",
            "strong body mechanics",
            "more clear body mechanics",
            "clear body mechanics discipline",
            "balanced center of gravity",
            "strong balanced center of gravity",
            "more balanced center of gravity",
            "balance and stabilityd center of gravity",
            "balanced center of gravity discipline",
            "motion matches action words",
            "strong motion matches action words",
            "more motion matches action words",
            "motion matches action words discipline",
        ],
        "complex": [
            "dynamic full-body pose",
            "strong dynamic full-body pose",
            "more dynamic full-body pose",
            "dynamic full-body pose discipline",
            "balanced anatomy",
            "strong balanced anatomy",
            "more balanced anatomy",
            "balance and stabilityd anatomy",
            "balanced anatomy discipline",
            "clear limb placement",
            "strong limb placement",
            "more clear limb placement",
            "clear limb placement discipline",
            "foreshortening",
            "strong foreshortening",
            "foreshortening discipline",
            "prompt-specified limb arrangement",
            "strong prompt-specified limb arrangement",
            "more prompt-specified limb arrangement",
            "prompt-specified limb arrangement discipline",
        ],
    },
    "quality_pack_negative": {
        "cinematic": ["flat tonal range", "highlight clipping", "crushed shadows", "unmotivated grade shift"],
        "editorial": [
            "plastic skin",
            "overprocessed skin",
            "fake hdr look",
            "drifted color grade",
            "lost material identity",
        ],
        "illustrative": ["muddy line quality", "shape ambiguity", "texture soup"],
        "micro_detail": [
            "mushy texture",
            "lost fine detail",
            "plastic uniform surfaces",
            "waxy skin everywhere",
            "smeared micro-detail",
            "smudged fine edges",
            "ambiguous surface read",
        ],
        "ultra_clean": ["jpeg artifacts", "macroblocking", "banding", "mushy details"],
    },
    "quality_pack_negative_one_shot": {
        "_": [
            "duplicate subject",
            "cloned face",
            "double head",
            "extra arms",
            "extra legs",
            "missing limbs",
            "bad hands",
            "mutated hands",
            "too many fingers",
            "fused fingers",
            "blurry",
            "out of focus subject",
            "muddy details",
            "jpeg artifacts",
            "worst quality",
            "low quality",
            "watermark",
            "cropped",
            "bad anatomy",
            "deformed",
            "disconnected limbs",
            "incoherent background",
            "inconsistent lighting",
            "visual clutter",
            "bad proportions",
            "noise",
            "text overlay",
            "logo",
            "username",
            "signature",
            "extra digits",
            "subject replaced by clutter",
            "brief abandoned mid-scene",
        ]
    },
    "quality_pack_negative_top": {
        "_": [
            "worst quality",
            "low quality",
            "jpeg artifacts",
            "blurry",
            "noise",
            "bad proportions",
            "watermark",
            "text overlay",
            "logo",
            "username",
            "signature",
            "bad hands",
            "extra digits",
            "mutated hands",
            "ignored composition brief",
            "wrong aspect storytelling",
        ]
    },
    "quality_pack_positive": {
        "cinematic": ["cinematic grade", "filmic contrast", "controlled highlight rolloff", "depth-rich image"],
        "editorial": [
            "editorial photography look",
            "controlled dynamic range",
            "material fidelity",
            "faithful material response",
            "prompt-locked palette",
        ],
        "illustrative": ["clean stylized rendering", "shape readability", "crisp silhouette language"],
        "micro_detail": [
            "high frequency detail",
            "texture fidelity",
            "material read",
            "clean micro-contrast",
            "sharp fine structure",
            "readable micro surface structure",
            "edge-accurate small forms",
        ],
        "ultra_clean": [
            "ultra clean render",
            "high micro-contrast",
            "fine detail retention",
            "artifact-free finish",
            "clean edge discipline",
        ],
    },
    "relationship_negative": {
        "combat": ["weapon clipping", "limb overlap confusion", "pose collision"],
        "neutral": ["awkward overlap", "merged bodies"],
        "romantic": ["anatomy merge", "fused torsos", "broken contact anatomy", "wrong relationship dynamic"],
        "teamwork": ["subject confusion", "unclear action hierarchy"],
    },
    "relationship_positive": {
        "combat": ["combat choreography", "clear attacker-defender spacing", "dynamic tension"],
        "neutral": ["neutral interpersonal spacing", "non-overlapping poses"],
        "romantic": ["romantic interaction", "gentle physical contact", "coherent paired pose"],
        "teamwork": [
            "cooperative group action",
            "shared focal objective",
            "coordinated pose direction",
            "group action matches brief",
        ],
    },
    "sex_view_positive": {
        "female": ["female anatomy consistency", "correct female body proportions"],
        "male": ["male anatomy consistency", "correct male body proportions"],
        "mixed": ["multiple subjects with distinct anatomy", "clear subject separation"],
        "nonbinary": ["androgynous presentation", "anatomical consistency"],
    },
    "skin_detail_negative": {
        "clean_beauty": ["over-smoothed skin", "loss of facial definition"],
        "natural_texture": [
            "waxy skin",
            "airbrushed plastic skin",
            "porcelain skin artifact",
            "ignored freckle or scar note from prompt",
        ],
        "stylized_skin": ["style-inconsistent skin", "skin shading collapse"],
    },
    "skin_detail_positive": {
        "clean_beauty": ["clean beauty skin", "balanced skin smoothing", "retained facial structure"],
        "natural_texture": [
            "natural skin texture",
            "real pores",
            "subtle imperfections",
            "non-waxy skin",
            "skin read matches ethnicity and age cues",
        ],
        "stylized_skin": ["stylized skin rendering", "consistent skin shading language"],
    },
    "style_lock_negative": {
        "_": [
            "style bleed",
            "mixed incompatible styles",
            "style drift",
            "visual style inconsistency",
            "medium drift from prompt",
        ]
    },
    "style_lock_positive": {
        "_": [
            "style consistency",
            "single coherent art direction",
            "uniform rendering style",
            "prompt-locked rendering medium",
            "no medium swap mid-image",
        ]
    },
    "style_negative": {
        "3d": [
            "flat 2d look",
            "strong flat 2d look",
            "more flat 2d look",
            "flat 2d look discipline",
            "painted texture smear",
            "strong painted texture smear",
            "more painted texture smear",
            "painted texture smear discipline",
            "broken geometry",
            "strong broken geometry",
            "more broken geometry",
            "broken geometry discipline",
            "plastic artifacting",
            "strong plastic artifacting",
            "more plastic artifacting",
            "plastic artifacting discipline",
        ],
        "3d_photoreal": ["toy-like plastic look", "flat shading", "non-physical reflections"],
        "anime": ["photographic pores", "uncanny realism", "muddy linework", "style contradicts photoreal request"],
        "painterly": ["photo artifacts", "over-sharp photographic texture", "style inconsistency"],
        "photoreal": [
            "cartoon look",
            "strong cartoon look",
            "more cartoon look",
            "cartoon look discipline",
            "anime lineart",
            "strong anime lineart",
            "more anime lineart",
            "anime lineart discipline",
            "cgi plastic skin",
            "strong cgi plastic skin",
            "more cgi plastic skin",
            "cgi plastic skin discipline",
            "painterly brush strokes",
            "strong painterly brush strokes",
            "more painterly brush strokes",
            "painterly brush strokes discipline",
        ],
        "semi_real": [
            "style collapse",
            "strong style collapse",
            "more style collapse",
            "style collapse discipline",
            "over-cartooned",
            "strong over-cartooned",
            "over-cartooned discipline",
            "over-photographic mismatch",
            "strong over-photographic mismatch",
            "more over-photographic mismatch",
        ],
    },
    "style_positive": {
        "3d": [
            "3d render",
            "strong 3d render",
            "more 3d render",
            "3d render discipline",
            "cg render",
            "strong cg render",
            "more cg render",
            "cg render discipline",
            "solid geometry",
            "strong solid geometry",
            "more solid geometry",
            "solid geometry discipline",
            "consistent material shading",
            "strong consistent material shading",
            "more consistent material shading",
            "consistent material shading discipline",
            "octane-style lighting",
            "strong octane-style lighting",
            "more octane-style lighting",
            "octane-style lighting discipline",
        ],
        "3d_photoreal": [
            "photorealistic 3d render",
            "strong photorealistic 3d render",
            "more photorealistic 3d render",
            "photorealistic 3d render discipline",
            "realistic cg materials",
            "strong realistic cg materials",
            "more realistic cg materials",
            "realistic cg materials discipline",
            "cinematic physically based lighting",
            "strong cinematic physically based lighting",
            "more cinematic physically based lighting",
            "cinematic physically based lighting discipline",
        ],
        "anime": [
            "anime style",
            "strong anime style",
            "more anime style",
            "anime style discipline",
            "clean linework",
            "strong clean linework",
            "more clean linework",
            "clean linework discipline",
            "coherent cel-shading",
            "strong coherent cel-shading",
            "more coherent cel-shading",
            "coherent cel-shading discipline",
        ],
        "painterly": [
            "painterly style",
            "strong painterly style",
            "more painterly style",
            "painterly style discipline",
            "brushwork detail",
            "strong brushwork detail",
            "more brushwork detail",
            "brushwork detail discipline",
            "coherent painted textures",
            "strong coherent painted textures",
            "more coherent painted textures",
            "coherent painted textures discipline",
        ],
        "photoreal": [
            "photorealistic",
            "strong photorealistic",
            "photorealistic discipline",
            "real-world textures",
            "strong real-world textures",
            "more real-world textures",
            "real-world textures discipline",
            "natural skin/material detail",
            "strong natural skin/material detail",
            "more natural skin/material detail",
            "natural skin/material detail discipline",
            "camera-realistic lighting",
            "strong camera-realistic lighting",
            "more camera-realistic lighting",
            "camera-realistic lighting discipline",
            "lens-consistent perspective",
            "strong lens-consistent perspective",
            "more lens-consistent perspective",
            "perspective consistency",
            "lens-consistent perspective discipline",
            "literal texture scale from brief",
            "strong literal texture scale from brief",
            "more literal texture scale from brief",
            "literal texture scale from brief discipline",
        ],
        "semi_real": [
            "semi-realistic",
            "strong semi-realistic",
            "semi-realistic discipline",
            "stylized realism",
            "strong stylized realism",
            "more stylized realism",
            "stylized realism discipline",
            "balanced stylization and realism",
            "strong balanced stylization and realism",
            "more balanced stylization and realism",
            "balance and stabilityd stylization and realism",
            "balanced stylization and realism discipline",
        ],
    },
    "typography_negative": {
        "clean": ["garbled text", "misspelled words", "warped letters", "illegible text"],
        "poster": ["broken headline", "incoherent text layout", "letter overlap artifacts"],
        "ui": ["ui gibberish text", "icon-text overlap", "misaligned labels"],
    },
    "typography_positive": {
        "clean": ["clean typography", "legible lettering", "consistent kerning", "sharp text edges"],
        "poster": ["poster typography", "readable headline", "layout-aligned text blocks"],
        "ui": ["ui text clarity", "readable interface labels", "consistent icon-label spacing"],
    },
    "unwanted_text_negative": {
        "_": [
            "watermark",
            "strong watermark",
            "watermark discipline",
            "logo",
            "strong logo",
            "logo discipline",
            "signature",
            "strong signature",
            "signature discipline",
            "brand mark",
            "strong brand mark",
            "more brand mark",
            "brand mark discipline",
            "caption overlay",
            "strong caption overlay",
            "more caption overlay",
            "caption overlay discipline",
            "subtitle overlay",
            "strong subtitle overlay",
            "more subtitle overlay",
            "subtitle overlay discipline",
            "random text",
            "strong random text",
            "more random text",
            "random text discipline",
            "letters",
            "strong letters",
            "letters discipline",
            "numbers",
            "strong numbers",
            "numbers discipline",
            "typography",
            "strong typography",
            "typography discipline",
            "spurious caption text",
            "strong spurious caption text",
            "more spurious caption text",
            "spurious caption text discipline",
            "unprompted UI chrome",
            "strong unprompted UI chrome",
            "more unprompted UI chrome",
            "unprompted UI chrome discipline",
            "fake metadata overlay",
            "strong fake metadata overlay",
            "more fake metadata overlay",
            "fake metadata overlay discipline",
            "thumbnail studies",
            "gesture drawing clarity",
            "value study planning",
            "shape design clarity",
            "silhouette-first design",
            "composition thumbnail pass",
            "design iteration pass",
            "edge control hierarchy",
            "simplified value grouping",
            "focal point emphasis",
            "clean read at distance",
            "intentional staging",
            "camera blocking clarity",
            "contrast hierarchy",
            "detail hierarchy",
            "large shapes first",
        ]
    },
}


def _flat(pack: str) -> list[str]:
    return list(BUILTIN_TAG_TABLES.get(pack, {}).get("_", []))


def _modes(pack: str) -> dict[str, list[str]]:
    return {mode: list(tags) for mode, tags in BUILTIN_TAG_TABLES.get(pack, {}).items()}


def _dn(pack: str) -> dict[str, list[str]]:
    """Mode pack whose "none" mode still merges (and so dedupes) the prompt."""
    d = _modes(pack)
    d.setdefault("none", [])
    return d


_BEST_MODEL_TAGS: list[str] = _flat("best_model_tags")

_POSE_POSITIVE: dict[str, list[str]] = _modes("pose_positive")
_POSE_NEGATIVE: list[str] = _flat("pose_negative_shared")
_ANGLE_POSITIVE: dict[str, list[str]] = _modes("angle_positive")
_ANGLE_NEGATIVE: list[str] = _flat("angle_negative_shared")
_SEX_VIEW_POSITIVE: dict[str, list[str]] = _modes("sex_view_positive")
_DOMAIN_POSITIVE: dict[str, list[str]] = _modes("domain_positive")
_DOMAIN_NEGATIVE: dict[str, list[str]] = _modes("domain_negative")
# Weak domains (product, food, interior, machinery, nature macro, technical, diagram): utils.prompt.realism.
_DOMAIN_POSITIVE.update({k: list(pos) for k, (pos, _neg) in _WEAK_DOMAIN_PACKS.items()})
_DOMAIN_NEGATIVE.update({k: list(neg) for k, (_pos, neg) in _WEAK_DOMAIN_PACKS.items()})
_BACKGROUND_POSITIVE: dict[str, list[str]] = _modes("background_positive")
_BACKGROUND_NEGATIVE: dict[str, list[str]] = _modes("background_negative")
_PEOPLE_LAYOUT_POSITIVE: dict[str, list[str]] = _modes("people_layout_positive")
_PEOPLE_LAYOUT_NEGATIVE: dict[str, list[str]] = _modes("people_layout_negative")
_RELATIONSHIP_POSITIVE: dict[str, list[str]] = _modes("relationship_positive")
_RELATIONSHIP_NEGATIVE: dict[str, list[str]] = _modes("relationship_negative")
_OBJECT_LAYOUT_POSITIVE: dict[str, list[str]] = _modes("object_layout_positive")
_OBJECT_LAYOUT_NEGATIVE: dict[str, list[str]] = _modes("object_layout_negative")
_HAND_POSITIVE: dict[str, list[str]] = _dn("hand_positive")
_HAND_NEGATIVE: dict[str, list[str]] = _dn("hand_negative")
_POSE_NATURALNESS_POSITIVE: dict[str, list[str]] = _dn("pose_naturalness_positive")
_POSE_NATURALNESS_NEGATIVE: dict[str, list[str]] = _dn("pose_naturalness_negative")
_TYPOGRAPHY_POSITIVE: dict[str, list[str]] = _dn("typography_positive")
_TYPOGRAPHY_NEGATIVE: dict[str, list[str]] = _dn("typography_negative")

_ONE_SHOT_POSITIVE: list[str] = _flat("one_shot_positive")
_ONE_SHOT_NEGATIVE: list[str] = _flat("one_shot_negative")

_QUALITY_PACK_POSITIVE: dict[str, list[str]] = {
    "none": [],
    "top": _BEST_MODEL_TAGS,
    "one_shot": _ONE_SHOT_POSITIVE + _BEST_MODEL_TAGS,
    **_modes("quality_pack_positive"),
}
_QUALITY_PACK_NEGATIVE: dict[str, list[str]] = {
    "none": [],
    "top": _flat("quality_pack_negative_top"),
    "one_shot": _flat("quality_pack_negative_one_shot"),
    **_modes("quality_pack_negative"),
}

_ADHERENCE_PACK_POSITIVE: dict[str, list[str]] = {
    "none": [],
    "standard": _flat("adherence_standard_positive"),
    "strict": _flat("adherence_standard_positive") + _flat("adherence_strict_extra_positive"),
}
_ADHERENCE_PACK_NEGATIVE: dict[str, list[str]] = {
    "standard": _flat("adherence_standard_negative"),
    "strict": _flat("adherence_standard_negative") + _flat("adherence_strict_extra_negative"),
}

_LIGHTING_MODE_POSITIVE: dict[str, list[str]] = _dn("lighting_mode_positive")
_LIGHTING_MODE_NEGATIVE: dict[str, list[str]] = _dn("lighting_mode_negative")
_SKIN_DETAIL_POSITIVE: dict[str, list[str]] = _dn("skin_detail_positive")
_SKIN_DETAIL_NEGATIVE: dict[str, list[str]] = _dn("skin_detail_negative")
_BODY_PROPORTION_POSITIVE: dict[str, list[str]] = _dn("body_proportion_positive")
_INTERACTION_INTENSITY: dict[str, list[str]] = _dn("interaction_intensity_positive")
_CLOTHING_POSITIVE: dict[str, list[str]] = _dn("clothing_positive")
_CLOTHING_NEGATIVE: dict[str, list[str]] = _dn("clothing_negative")
_ADVANCED_POSE_POSITIVE: dict[str, list[str]] = _dn("advanced_pose_positive")
_OBJECTS_INTERACTION_POSITIVE: dict[str, list[str]] = _dn("objects_interaction_positive")
_ENVIRONMENT_DETAIL_POSITIVE: dict[str, list[str]] = _dn("environment_detail_positive")
_STYLE_POSITIVE: dict[str, list[str]] = _modes("style_positive")
_STYLE_NEGATIVE: dict[str, list[str]] = _modes("style_negative")
_STYLE_LOCK_POSITIVE: list[str] = _flat("style_lock_positive")
_STYLE_LOCK_NEGATIVE: list[str] = _flat("style_lock_negative")
_UNWANTED_TEXT_NEGATIVE: list[str] = _flat("unwanted_text_negative")
_COMPOSITION_POSITIVE: dict[str, list[str]] = _modes("composition_positive")
_COMPOSITION_NEGATIVE: dict[str, list[str]] = _modes("composition_negative")
_ARTIST_COMPOSITION_POSITIVE: dict[str, list[str]] = _dn("artist_composition_positive")
_ARTIST_COMPOSITION_NEGATIVE: dict[str, list[str]] = _dn("artist_composition_negative")
_PERSPECTIVE_STABILITY_POSITIVE: list[str] = _flat("perspective_stability_positive")
_PERSPECTIVE_STABILITY_NEGATIVE: list[str] = _flat("perspective_stability_negative")
_DUPLICATE_SUBJECT_NEGATIVE: list[str] = _flat("duplicate_subject_negative")
_CONFLICTING_TAG_PAIRS: list[tuple[str, str]] = [
    (a.strip(), b.strip()) for a, _, b in (r.partition("|||") for r in _flat("conflicting_tag_pairs")) if b
]

_ANTI_AI_LITE_NEGATIVE: list[str] = dedupe_tags(ANTI_AI_LOOK_NEGATIVE.split(","))
_ANTI_AI_STRONG_NEGATIVE: list[str] = dedupe_tags(ANTI_AI_LOOK_NEGATIVE_STRONG.split(","))
_ANTI_AI_LITE_POSITIVE: list[str] = dedupe_tags(NATURAL_LOOK_POSITIVE.split(","))
_ANTI_AI_STRONG_POSITIVE: list[str] = dedupe_tags(NATURAL_LOOK_POSITIVE_DEEP.split(","))

_HUMAN_MEDIA_POSITIVE: dict[str, list[str]] = _dn("human_media_positive")
_HUMAN_MEDIA_NEGATIVE: dict[str, list[str]] = _dn("human_media_negative")
_LORA_SCAFFOLD_POSITIVE: dict[str, list[str]] = _dn("lora_scaffold_positive")
_LORA_SCAFFOLD_NEGATIVE: dict[str, list[str]] = _dn("lora_scaffold_negative")


def _contains_any(haystack: str, needles: Iterable[str]) -> bool:
    """Substring match on lowered prompt (same semantics as prior if/any chains)."""
    return any(n in haystack for n in needles)


def _first_bucket(haystack: str, buckets: Sequence[tuple[Sequence[str], str]]) -> str | None:
    """First matching keyword group wins (order preserves old elif chains)."""
    for keywords, value in buckets:
        if _contains_any(haystack, keywords):
            return value
    return None


def _merge_kv_pack(
    pos: str,
    neg: str,
    key: str,
    pos_map: dict[str, list[str]],
    neg_map: dict[str, list[str]] | None = None,
    *,
    shared_neg: Sequence[str] | None = None,
    pos_only: bool = False,
) -> tuple[str, str]:
    """Append ``pos_map[key]`` and the matching negatives (``shared_neg`` or ``neg_map[key]``); unknown keys skip."""
    if key not in pos_map:
        return pos, neg
    pos = append_unique(pos, pos_map[key])
    if pos_only:
        return pos, neg
    if shared_neg is not None:
        neg = append_unique(neg, shared_neg)
    elif neg_map is not None:
        neg = append_unique(neg, neg_map.get(key, []))
    return pos, neg


def _remove_conflicting_tags(text: str) -> str:
    """Drop the later tag of each contradictory pair."""
    tokens = split_tags(text)
    if not tokens:
        return text
    lowered = [t.lower() for t in tokens]
    to_remove = set()
    for a, b in _CONFLICTING_TAG_PAIRS:
        a_idx = next((i for i, t in enumerate(lowered) if a in t), None)
        b_idx = next((i for i, t in enumerate(lowered) if b in t), None)
        if a_idx is None or b_idx is None:
            continue
        if a_idx < b_idx:
            to_remove.add(b_idx)
        else:
            to_remove.add(a_idx)
    cleaned = [t for i, t in enumerate(tokens) if i not in to_remove]
    return ", ".join(cleaned)


# --- Keyword buckets for `infer_content_controls_from_prompt` (order = first-hit wins) ---

_INFER_SCENE_DOMAIN: tuple[tuple[tuple[str, ...], str], ...] = _WEAK_DOMAIN_BUCKETS + (
    (("car", "vehicle", "truck", "motorcycle", "bicycle", "bus"), "vehicles"),
    (("building", "architecture", "facade", "skyscraper", "interior"), "architecture"),
    (("object", "product shot", "tabletop", "still life"), "objects"),
)

_INFER_VIEW_ANGLE: tuple[tuple[tuple[str, ...], str], ...] = (
    (("first person", "pov", "through the eyes"), "first_person"),
    (("bird eye", "bird's-eye", "overhead", "top down"), "bird_eye"),
    (("low angle", "from below", "worm eye", "worm's-eye"), "low_angle"),
)

# Golden hour / sunset are a time of day, not a rim setup; utils.prompt.art_direction plans that light.
_INFER_LIGHTING: tuple[tuple[tuple[str, ...], str], ...] = (
    (("studio lighting", "white background", "simple background"), "studio_softbox"),
    (("overcast", "cloudy day", "soft daylight"), "natural_daylight"),
)

_INFER_STYLE_MODE: tuple[tuple[tuple[str, ...], str], ...] = (
    (("3d render", "cg", "octane", "blender render", "unreal engine"), "3d"),
    (("photoreal", "raw photo", "dslr", "real photograph"), "photoreal"),
    (("semi realistic", "semi-realistic", "stylized realism", "2.5d"), "semi_real"),
    (("anime", "manga style", "cel shading", "visual novel"), "anime"),
)

_INFER_HUMAN_MEDIA_FILM: tuple[str, ...] = (
    "35mm",
    "kodak",
    "portra",
    "fuji film",
    "film photograph",
    "analog photo",
    "cinestill",
)
_INFER_HUMAN_MEDIA_DSLR: tuple[str, ...] = (
    "dslr",
    "shot on canon",
    "shot on nikon",
    "shot on sony",
    "85mm",
    "50mm lens",
    "full frame",
)
_INFER_HUMAN_MEDIA_PHOTO: tuple[str, ...] = (
    "photorealistic",
    "raw photo",
    "real photograph",
    "realistic photo",
    "iphone photo",
    "smartphone photo realistic",
)

_INFER_ADHERENCE_STANDARD_KW: tuple[str, ...] = (
    "as described",
    "faithful to prompt",
    "accurate to description",
    "per the prompt",
    "match the caption",
    "follow the prompt",
)
_INFER_ADHERENCE_STRICT_KW: tuple[str, ...] = (
    "strict adherence",
    "exactly as described",
    "match every detail",
    "no extra elements",
    "no additions beyond",
)


def infer_content_controls_from_prompt(prompt: str) -> dict[str, str]:
    """
    Infer likely control modes from prompt keywords.
    Returns partial kwargs for `apply_content_controls`.
    """
    raw_prompt = (prompt or "").strip()
    p = raw_prompt.lower()
    out: dict[str, str] = {}

    sd = _first_bucket(p, _INFER_SCENE_DOMAIN)
    if sd:
        out["scene_domain"] = sd

    va = _first_bucket(p, _INFER_VIEW_ANGLE)
    if va:
        out["view_angle"] = va

    if _contains_any(p, ("action pose", "acrobat", "parkour", "jumping", "spinning kick")):
        out["pose_mode"] = "action"

    if "lingerie" in p:
        out["clothing_mode"] = "lingerie"
    elif "bikini" in p:
        out["clothing_mode"] = "bikini"
    elif "torn" in p or "ripped clothes" in p:
        out["clothing_mode"] = "torn"

    tags = [x.strip().lower() for x in raw_prompt.split(",") if x.strip()]
    tagset = set(tags)
    comma_rich = "," in raw_prompt

    multi_noise = _contains_any(p, ("group", "crowd", "orgy")) or (
        bool(tagset & {"2girls", "2boys", "3girls"}) if comma_rich else _contains_any(p, ("2girls", "2boys", "3girls"))
    )
    if comma_rich:
        solo_hit = bool(tagset & {"1girl", "1boy", "1other", "solo"}) or _contains_any(p, ("single girl", "single boy"))
    else:
        solo_hit = _contains_any(p, ("1girl", "1boy", "1other", "single girl", "single boy", "solo"))

    if solo_hit and not multi_noise:
        out["composition_mode"] = "single_subject"
        out["people_layout"] = "solo"
    elif (comma_rich and bool(tagset & {"2girls", "2boys"})) or _contains_any(
        p, ("2girls", "2boys", "couple", "two girls", "two boys", "two people", "duo")
    ):
        out["people_layout"] = "duo"
        if out.get("composition_mode") in (None, "none"):
            out["composition_mode"] = "multi_character"

    if _contains_any(p, ("hand focus", "holding hands", "hands visible", "detailed hands", "fingers")):
        out["hand_mode"] = "stable"
    for key, value in _infer_hard_controls(p).items():  # broader hand / group cues (utils.prompt.realism)
        out.setdefault(key, value)

    lm = _first_bucket(p, _INFER_LIGHTING)
    if lm:
        out["lighting_mode"] = lm

    sm = _first_bucket(p, _INFER_STYLE_MODE)
    if sm:
        out["style_mode"] = sm

    if _contains_any(p, _INFER_HUMAN_MEDIA_FILM):
        out["human_media_mode"] = "film"
    elif _contains_any(p, _INFER_HUMAN_MEDIA_DSLR):
        out["human_media_mode"] = "dslr"
    elif _contains_any(p, _INFER_HUMAN_MEDIA_PHOTO):
        out["human_media_mode"] = "photographic"

    parts = [x.strip() for x in raw_prompt.split(",") if x.strip()]
    n_parts = len(parts)
    plen = len(raw_prompt)
    inferred_adherence = "none"
    if n_parts >= 12 or plen >= 280:
        inferred_adherence = "standard"
    if n_parts >= 22 or plen >= 520:
        inferred_adherence = "strict"
    if _contains_any(p, _INFER_ADHERENCE_STANDARD_KW):
        inferred_adherence = "standard" if inferred_adherence == "none" else inferred_adherence
    if _contains_any(p, _INFER_ADHERENCE_STRICT_KW):
        inferred_adherence = "strict"
    if inferred_adherence != "none":
        out["adherence_pack"] = inferred_adherence

    if _contains_any(
        p,
        (
            "academic composition",
            "classical composition",
            "golden ratio",
            "golden spiral",
            "dynamic symmetry",
            "notan",
            "chiaroscuro composition",
            "arabesque",
        ),
    ):
        out["artist_composition"] = "classical"
    elif _contains_any(
        p,
        (
            "one point perspective",
            "1-point perspective",
            "two point perspective",
            "2-point perspective",
            "three point perspective",
            "3-point perspective",
            "vanishing point",
            "horizon line",
            "orthogonal perspective",
        ),
    ):
        out["artist_composition"] = "perspective"
    elif _contains_any(
        p,
        (
            "rule of thirds",
            "leading lines",
            "negative space",
            "visual weight",
            "focal plane",
        ),
    ):
        out["artist_composition"] = "standard"
    elif _contains_any(p, ("strong composition", "focal point", "visual hierarchy", "picture plane")):
        out["artist_composition"] = "lite"

    return out


def apply_content_controls(
    prompt: str,
    negative_prompt: str,
    *,
    pose_mode: str = "none",
    view_angle: str = "none",
    subject_sex: str = "none",
    scene_domain: str = "none",
    clothing_mode: str = "none",
    background_mode: str = "none",
    people_layout: str = "none",
    relationship_mode: str = "none",
    object_layout: str = "none",
    hand_mode: str = "none",
    pose_naturalness: str = "none",
    typography_mode: str = "none",
    quality_pack: str = "none",
    lighting_mode: str = "none",
    skin_detail_mode: str = "none",
    body_proportion: str = "none",
    interaction_intensity: str = "none",
    advanced_pose: str = "none",
    object_interaction: str = "none",
    environment_type: str = "none",
    style_mode: str = "none",
    style_lock: bool = False,
    anti_style_bleed: bool = False,
    composition_mode: str = "none",
    artist_composition: str = "none",
    anti_duplicate_subjects: bool = False,
    anti_perspective_drift: bool = False,
    cleanup_conflicting_tags: bool = False,
    allow_text_in_image: bool = False,
    one_shot_boost: bool = False,
    anti_ai_pack: str = "none",
    human_media_mode: str = "none",
    lora_scaffold: str = "none",
    adherence_pack: str = "none",
) -> tuple[str, str]:
    """
    Apply optional content controls to prompt + negative prompt.

    Returns `(updated_prompt, updated_negative_prompt)`.
    """
    p = (prompt or "").strip()
    n = (negative_prompt or "").strip()

    if one_shot_boost:
        p = append_unique(p, _ONE_SHOT_POSITIVE)
        n = append_unique(n, _ONE_SHOT_NEGATIVE)

    p, n = _merge_kv_pack(p, n, pose_mode, _POSE_POSITIVE, None, shared_neg=_POSE_NEGATIVE)
    p, n = _merge_kv_pack(p, n, view_angle, _ANGLE_POSITIVE, None, shared_neg=_ANGLE_NEGATIVE)
    p, n = _merge_kv_pack(p, n, subject_sex, _SEX_VIEW_POSITIVE, None, pos_only=True)

    p, n = _merge_kv_pack(p, n, scene_domain, _DOMAIN_POSITIVE, _DOMAIN_NEGATIVE)
    p, n = _merge_kv_pack(p, n, clothing_mode, _CLOTHING_POSITIVE, _CLOTHING_NEGATIVE)
    p, n = _merge_kv_pack(p, n, background_mode, _BACKGROUND_POSITIVE, _BACKGROUND_NEGATIVE)
    p, n = _merge_kv_pack(p, n, people_layout, _PEOPLE_LAYOUT_POSITIVE, _PEOPLE_LAYOUT_NEGATIVE)
    p, n = _merge_kv_pack(p, n, relationship_mode, _RELATIONSHIP_POSITIVE, _RELATIONSHIP_NEGATIVE)
    p, n = _merge_kv_pack(p, n, object_layout, _OBJECT_LAYOUT_POSITIVE, _OBJECT_LAYOUT_NEGATIVE)
    p, n = _merge_kv_pack(p, n, hand_mode, _HAND_POSITIVE, _HAND_NEGATIVE)
    p, n = _merge_kv_pack(p, n, pose_naturalness, _POSE_NATURALNESS_POSITIVE, _POSE_NATURALNESS_NEGATIVE)
    p, n = _merge_kv_pack(p, n, typography_mode, _TYPOGRAPHY_POSITIVE, _TYPOGRAPHY_NEGATIVE)
    p, n = _merge_kv_pack(p, n, quality_pack, _QUALITY_PACK_POSITIVE, _QUALITY_PACK_NEGATIVE)
    p, n = _merge_kv_pack(p, n, adherence_pack, _ADHERENCE_PACK_POSITIVE, _ADHERENCE_PACK_NEGATIVE)
    p, n = _merge_kv_pack(p, n, lighting_mode, _LIGHTING_MODE_POSITIVE, _LIGHTING_MODE_NEGATIVE)
    p, n = _merge_kv_pack(p, n, skin_detail_mode, _SKIN_DETAIL_POSITIVE, _SKIN_DETAIL_NEGATIVE)
    p, n = _merge_kv_pack(p, n, body_proportion, _BODY_PROPORTION_POSITIVE, None, pos_only=True)
    p, n = _merge_kv_pack(p, n, interaction_intensity, _INTERACTION_INTENSITY, None, pos_only=True)
    p, n = _merge_kv_pack(p, n, advanced_pose, _ADVANCED_POSE_POSITIVE, None, pos_only=True)
    p, n = _merge_kv_pack(p, n, object_interaction, _OBJECTS_INTERACTION_POSITIVE, None, pos_only=True)
    p, n = _merge_kv_pack(p, n, environment_type, _ENVIRONMENT_DETAIL_POSITIVE, None, pos_only=True)

    p, n = _merge_kv_pack(p, n, style_mode, _STYLE_POSITIVE, _STYLE_NEGATIVE)

    if style_lock:
        p = append_unique(p, _STYLE_LOCK_POSITIVE)
        n = append_unique(n, _STYLE_LOCK_NEGATIVE)
    elif anti_style_bleed:
        n = append_unique(n, _STYLE_LOCK_NEGATIVE)

    p, n = _merge_kv_pack(p, n, composition_mode, _COMPOSITION_POSITIVE, _COMPOSITION_NEGATIVE)
    p, n = _merge_kv_pack(p, n, artist_composition, _ARTIST_COMPOSITION_POSITIVE, _ARTIST_COMPOSITION_NEGATIVE)

    if anti_duplicate_subjects:
        n = append_unique(n, _DUPLICATE_SUBJECT_NEGATIVE)

    if anti_perspective_drift:
        p = append_unique(p, _PERSPECTIVE_STABILITY_POSITIVE)
        n = append_unique(n, _PERSPECTIVE_STABILITY_NEGATIVE)

    if anti_ai_pack == "lite":
        p = append_unique(p, _ANTI_AI_LITE_POSITIVE)
        n = append_unique(n, _ANTI_AI_LITE_NEGATIVE)
    elif anti_ai_pack == "strong":
        p = append_unique(p, _ANTI_AI_STRONG_POSITIVE)
        n = append_unique(n, _ANTI_AI_STRONG_NEGATIVE)

    p, n = _merge_kv_pack(p, n, human_media_mode, _HUMAN_MEDIA_POSITIVE, _HUMAN_MEDIA_NEGATIVE)
    p, n = _merge_kv_pack(p, n, lora_scaffold, _LORA_SCAFFOLD_POSITIVE, _LORA_SCAFFOLD_NEGATIVE)

    if not allow_text_in_image:
        n = append_unique(n, _UNWANTED_TEXT_NEGATIVE)

    if cleanup_conflicting_tags:
        p = _remove_conflicting_tags(p)

    return p, n

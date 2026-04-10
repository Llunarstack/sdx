# Recommended tags for domains that many models struggle with.
# Use these in training captions and at inference so our model handles 3D, realistic, interior/exterior, anatomy, etc.
# The data/caption pipeline (caption_utils.py) boosts these when present in captions.
# See docs/MODEL_WEAKNESSES.md for what other models suck at and how we prepare/fix.

__all__ = [
    "DOMAIN_NAMES",
    "RECOMMENDED_PROMPTS_BY_DOMAIN",
    "RECOMMENDED_NEGATIVE_BY_DOMAIN",
    "DEFAULT_NEGATIVE_PROMPT",
    "TEXT_IN_IMAGE_NEGATIVE",
    "TEXT_IN_IMAGE_PHRASES",
    "TEXT_IN_IMAGE_PROMPT_TIPS",
    "COMPLEX_PROMPT_TIPS",
    "CHALLENGING_PROMPT_TIPS",
    "ANATOMY_NEGATIVES",
    "HAND_FIX_PROMPT_TIPS",
    "PORTRAIT_ASPECT_TIPS",
    "HARD_STYLE_NAMES",
    "HARD_STYLE_RECOMMENDED_PROMPTS",
    "HARD_STYLE_NEGATIVES",
    "STYLE_MIX_TIPS",
    "LORA_MIX_TIPS",
    "ANTI_AI_LOOK_NEGATIVE",
    "ANTI_AI_LOOK_NEGATIVE_STRONG",
    "NATURAL_LOOK_POSITIVE",
    "NATURAL_LOOK_POSITIVE_DEEP",
    "LORA_STACK_NEGATIVE",
    "PHOTOGRAPHIC_AUTHENTIC_POSITIVE",
    "CONCEPT_BLEEDING_NEGATIVE",
    "CONCEPT_BLEEDING_POSITIVE",
    "ARTIFACT_NEGATIVES",
    "WATERMARK_NEGATIVE_STRONG",
    "FLUX_FACE_DIVERSITY_NEGATIVE",
    "DIVERSITY_POSITIVE",
    "SPATIAL_AWARENESS_TIPS",
    "EMOTION_PROMPT_TIPS",
    "CFG_BURN_TIPS",
    "BACKGROUND_TIPS",
    "CENTERING_TIPS",
    "DISTANT_FACE_TIPS",
    "RESOLUTION_TIPS",
    "SEED_VARIANCE_TIPS",
    "VOCABULARY_TIPS",
    "QUALITY_TAG_DEPENDENCY_TIPS",
    "COLOR_TINT_NEGATIVE",
    "FLUX_GRID_ARTIFACT_TIPS",
    "NEGATIVE_PROMPT_BEST_PRACTICES",
    "FULL_BODY_AND_TWO_HEAD_TIPS",
    "GARBLED_FACE_TIPS",
    "LORA_STRENGTH_TIPS",
    "PROMPT_STRUCTURE_TIPS",
    "ORIGINALITY_POSITIVE_TOKENS",
    # Physics / fluids / transparency (see physics_material_prompts.py)
    "PHYSICS_MATERIAL_DOMAIN_NAMES",
    "PHYSICS_MATERIAL_RECOMMENDED_PROMPTS_BY_DOMAIN",
    "PHYSICS_MATERIAL_RECOMMENDED_NEGATIVE_BY_DOMAIN",
    "PHYSICS_COMMON_NEGATIVE_ADDON",
    "FLUIDS_PROMPT_TIPS",
    "TRANSPARENCY_REFRACTION_TIPS",
    "RIGID_CONTACT_SHADOW_TIPS",
    "PARTICULATE_VOLUMETRIC_TIPS",
    "SOFT_BODY_CLOTH_TIPS",
    "PHYSICS_TRAINING_CAPTION_TIPS",
    # Creatures / anthro / robots / humanoid monsters (see creature_character_prompts.py)
    "HUMANOID_MONSTER_PROMPT_KEYWORDS",
    "CREATURE_CHARACTER_DOMAIN_NAMES",
    "CREATURE_CHARACTER_RECOMMENDED_PROMPTS_BY_DOMAIN",
    "CREATURE_CHARACTER_RECOMMENDED_NEGATIVE_BY_DOMAIN",
    "CREATURE_COMMON_NEGATIVE_ADDON",
    "CREATURES_PROMPT_TIPS",
    "CREATURES_SFW_TIPS",
    "CREATURES_NSFW_TIPS",
    "ANTHRO_FURRY_PROMPT_TIPS",
    "ROBOTS_MECH_PROMPT_TIPS",
    "HUMANOID_MONSTERS_PROMPT_TIPS",
    "CREATURE_TRAINING_CAPTION_TIPS",
    "CREATURE_SFW_CONTEXT_POSITIVE",
    "CREATURE_SFW_NEGATIVE_ADDON",
    "CREATURE_NSFW_CONTEXT_POSITIVE",
    "CREATURE_NSFW_NEGATIVE_ADDON",
    "CREATURE_CHARACTER_SFW_RECOMMENDED_PROMPTS_BY_DOMAIN",
    "CREATURE_CHARACTER_NSFW_RECOMMENDED_PROMPTS_BY_DOMAIN",
    "CREATURE_CHARACTER_SFW_RECOMMENDED_NEGATIVE_BY_DOMAIN",
    "CREATURE_CHARACTER_NSFW_RECOMMENDED_NEGATIVE_BY_DOMAIN",
    "CREATURE_NSFW_KEYWORDS",
    "CREATURE_SFW_SCENARIO_TIPS",
    "CREATURE_NSFW_SCENARIO_TIPS",
    "ANTHRO_FURRY_SFW_TIPS",
    "ANTHRO_FURRY_NSFW_TIPS",
    "ROBOTS_MECH_SFW_TIPS",
    "ROBOTS_MECH_NSFW_TIPS",
    "HUMANOID_MONSTERS_SFW_TIPS",
    "HUMANOID_MONSTERS_NSFW_TIPS",
]

DOMAIN_NAMES = ("3d", "realistic", "interior", "exterior")

# Short recommended prompt prefixes per domain (for inference or data prep).
# Combine with your subject, e.g. RECOMMENDED_PROMPTS_BY_DOMAIN["3d"][0] + ", " + "your subject".
RECOMMENDED_PROMPTS_BY_DOMAIN = {
    "3d": [
        "3d render, octane render, masterpiece, best quality",
        "3d illustration, isometric, clean 3d, high quality",
        "blender 3d, solid shading, detailed, 8k",
    ],
    "realistic": [
        "photorealistic, realistic, raw photo, natural lighting, masterpiece",
        "hyperrealistic, photo, real photography, detailed skin, 8k",
    ],
    "interior": [
        "interior design, indoor, room, modern interior, masterpiece, best quality",
        "living room, furniture, cozy interior, architecture interior, detailed",
    ],
    "exterior": [
        "exterior, outdoor, architecture, building, facade, masterpiece",
        "street view, urban, landscape, outdoors, natural lighting, 8k",
    ],
    "complex": [
        "masterpiece, best quality, detailed, sharp focus, complex composition",
        "ultra detailed, high quality, multiple elements, coherent, 8k",
    ],
    "challenging": [
        "masterpiece, best quality, detailed, sharp focus",
        "surreal, abstract, fantasy, unusual, detailed, high quality",
    ],
}

# Negative prompts that help avoid common failures in each domain.
RECOMMENDED_NEGATIVE_BY_DOMAIN = {
    "3d": "flat, 2d, blurry, bad proportions, deformed",
    "realistic": "unrealistic, cartoon, anime, painting, illustration, blurry, oversaturated",
    "interior": "empty room, cluttered, distorted perspective, wrong scale, blurry",
    "exterior": "flat, wrong perspective, distorted building, blurry, oversaturated",
    "complex": "simple, plain, boring, low quality, blurry, duplicate",
    "challenging": "generic, bland, low quality, blurry, incoherent",
}

# Default negative when user leaves negative prompt empty (sample.py uses this).
# Classic SD-style: quality + anatomy/hands (keep minimal for SDXL-like models if they over-obey).
# Note: "text" here means unwanted watermarks/signatures; for images that should contain legible text, use TEXT_IN_IMAGE_NEGATIVE.
DEFAULT_NEGATIVE_PROMPT = (
    "low quality, worst quality, blurry, jpeg artifacts, watermark, signature, text, "
    "bad anatomy, bad hands, missing fingers, extra fingers, fused fingers, deformed, duplicate"
)

# When the user prompt clearly asks for text in the image (sign, lettering, "says", etc.),
# use this negative instead of DEFAULT_NEGATIVE_PROMPT so we don't suppress desired text.
# Avoids bad text (garbled, misspelled) and watermarks but does NOT include "text".
TEXT_IN_IMAGE_NEGATIVE = (
    "low quality, worst quality, blurry, jpeg artifacts, watermark, signature, "
    "garbled text, misspelled, wrong spelling, illegible, unreadable, "
    "bad anatomy, bad hands, deformed, duplicate"
)

# Phrases that suggest the user wants legible text / lettering in the image (sample.py uses this to switch negative).
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

# Tips for generating images with text (training data and inference).
TEXT_IN_IMAGE_PROMPT_TIPS = [
    "Be explicit: 'sign that says OPEN', 'text reading Hello in bold', 'label that says 24h'.",
    "Quote the exact text in the prompt: 'sign that says \"CAFE\"' so the model gets the spelling.",
    "Add quality cues: 'legible text', 'clear lettering', 'sharp text', 'readable'.",
    "Use negative: garbled text, misspelled, watermark (or leave empty and use --text-in-image so we use TEXT_IN_IMAGE_NEGATIVE).",
]

# Anatomy / people: use when generating characters or full body (from data/caption_utils.py + research).
ANATOMY_NEGATIVES = (
    "bad anatomy, bad hands, missing fingers, extra fingers, fused fingers, mutated hands, "
    "poorly drawn hands, deformed hands, extra limbs, missing limbs, bad face, deformed face, "
    "distorted eyes, duplicate, duplicate head, two heads, merged subjects, blurry, worst quality"
)

# Hand fixes (research: object anchoring and simple poses work better than "no deformed hands").
# Use in training captions and at inference when hands are important.
HAND_FIX_PROMPT_TIPS = [
    "Describe hands in relation to objects: e.g. 'holding a cup', 'hands on keyboard'.",
    "Use simple poses: 'hands clasped', 'arms crossed', 'hands behind back', 'hands in pockets'.",
    "Frame so hands are partially visible or in one clear pose rather than multiple overlapping.",
    "Add positive tags: 'correct hands', 'five fingers', 'natural hands', 'visible hands'.",
]

# Portrait / aspect ratio: avoid "double head" (common when portrait aspect + single head).
PORTRAIT_ASPECT_TIPS = [
    "For portrait aspect (e.g. 9:16), add full-body cues: 'standing', 'long dress', 'legs', 'shoes' so the model fills the frame with one body.",
    "Or use 1:1 (square) for head/shoulder shots to reduce duplicate-head artifacts.",
]

# Complex / long prompts: better adherence and quality (training + inference).
COMPLEX_PROMPT_TIPS = [
    "Put the main subject first: '1girl, red dress, in a forest, ...' so the model prioritizes it.",
    "Use comma-separated tags; add 'masterpiece, best quality, detailed' for stronger adherence.",
    "For long prompts, ensure the most important elements are in the first ~77 tokens (T5 truncation).",
    "Avoid contradicting terms in the same prompt (e.g. 'day' and 'night'); use (word) to emphasize key terms.",
    "Use --boost-quality in sample.py to prepend quality tags when you want maximum adherence.",
]

# Challenging content: NSFW, surreal, abstract, weird (no censorship; improve quality and adherence).
CHALLENGING_PROMPT_TIPS = [
    "Do not censor training data or prompts; the model learns what you show. Use clear, descriptive tags.",
    "For mature/explicit content: use consistent tags in data and at inference so the model learns the mapping.",
    "Surreal/abstract/weird: add 'surreal, detailed, masterpiece' and describe the scene concretely (colors, layout, mood).",
    "Strange compositions: put subject first, then setting, then style; use (important element) for emphasis.",
    "Quality tags help: 'masterpiece, best quality, sharp focus' improve adherence for any challenging prompt.",
]

# --- Hard styles: 3D, photorealistic, style mixes ---
# Many models blur 3D vs 2D, realistic vs illustrated, and mixed styles (2.5D, semi-realistic).
# data/caption_utils.py boosts HARD_STYLE_TAGS when present; use these prompts/negatives at inference.

HARD_STYLE_NAMES = ("3d", "realistic", "3d_realistic", "style_mix")

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

HARD_STYLE_NEGATIVES = {
    "3d": "flat, 2d, blurry, bad proportions, deformed, illustration style, painting",
    "realistic": "unrealistic, cartoon, anime, painting, illustration, blurry, oversaturated, drawn",
    "3d_realistic": "flat, 2d, cartoon, blurry, deformed, wrong lighting, plastic look",
    "style_mix": "blurry, incoherent style, messy, low quality, wrong mix, flat",
}

# Combining multiple styles in one prompt (e.g. 3d + anime, realistic + painterly).
STYLE_MIX_TIPS = [
    "Put the dominant style first: e.g. '3d render, anime style, ...' or 'photorealistic, oil painting style, ...'.",
    "Use explicit mix phrases the model was trained on: '2.5d', 'semi-realistic', 'photorealistic anime', 'stylized realistic'.",
    "Add quality anchors: 'masterpiece, best quality' at the start so the model doesn't blur the mix.",
    "Avoid contradicting pairs in one prompt (e.g. 'flat illustration' + '3d render'); pick one base and one modifier.",
    "Training data: include many captions with the exact mix phrases you want (2.5d, semi-realistic, etc.); they get boosted.",
]

# Using multiple LoRAs (e.g. style + character, or two styles).
LORA_MIX_TIPS = [
    "Use lower scales when stacking LoRAs (e.g. 0.5–0.6 each) so they blend instead of fighting.",
    "Put the dominant LoRA first in --lora; its trigger (--lora-trigger) goes at the start of the prompt.",
    "For style + character: use style LoRA at 0.5–0.7 and character LoRA at 0.6–0.8, with both triggers in the prompt.",
    "If the output is muddy or oversaturated, lower CFG (e.g. 5–6) or use --cfg-rescale 0.7.",
    "Train or fine-tune on mixed-style data so the base model already understands 2.5d/semi-realistic; LoRAs then refine.",
    "At inference: --lora-scaffold blend (or --lora-scaffold-auto) adds fusion-friendly pos/neg tokens; LORA_STACK_NEGATIVE helps when stacking two+ LoRAs.",
]

# --- Natural / non-AI look: reduce plastic, oversmooth, CGI feel ---
# Use with --naturalize in sample.py: adds these to negative and optional positive + post-process (grain).

# Negative: push away from the typical "AI-generated" look (smooth plastic skin, waxy, doll-like, oversaturated).
ANTI_AI_LOOK_NEGATIVE = (
    "oversaturated, plastic skin, smooth skin, airbrushed, waxy, doll-like, "
    "perfect skin, flawless skin, synthetic, artificial, CGI, uncanny, "
    "AI art, generated, digital art, overly smooth, porcelain skin, plastic look, "
    "neural network aesthetic, hyperreal plastic, hdr glow, halo artifacts, oversharpened, "
    "too perfect symmetry, stock photo stiffness, candy saturation, glossy unreal skin"
)

# Stronger de-AI / de-CG pack (use with --anti-ai-pack strong or NATURAL_LOOK_POSITIVE_DEEP).
ANTI_AI_LOOK_NEGATIVE_STRONG = (
    ANTI_AI_LOOK_NEGATIVE
    + ", midjourney look, generic AI illustration, same-face syndrome, "
    "floating specular highlights, subsurface abuse, waxy teeth, dead glassy eyes, "
    "uniform skin porelessness, beauty filter face, snapchat smooth skin, "
    "tilt-shift miniature fake, tilt shift cliché, lens blur fake, computational hdr, "
    "neon edge glow, burnt edges from cfg, posterization, banding in gradients"
)

# Positive: subtle hints for a more natural, photographic feel (optional prepend when --naturalize).
NATURAL_LOOK_POSITIVE = (
    "film grain, natural skin texture, subtle imperfections, raw photo, natural lighting, "
    "believable shadows, ambient occlusion subtle, real camera response"
)

# Deeper natural / human-capture cues (prepend with --naturalize-deep or human-media packs).
NATURAL_LOOK_POSITIVE_DEEP = (
    NATURAL_LOOK_POSITIVE
    + ", authentic photograph, candid moment, imperfect but real, "
    "shot on camera, natural white balance, organic grain, micro-contrast natural, "
    "shallow depth of field natural, lens character subtle"
)

# When stacking LoRAs: negatives that reduce muddy fusion / double-style (pair with lower strengths).
LORA_STACK_NEGATIVE = (
    "double exposure style clash, conflicting adaptations, merged incompatible styles, "
    "lora artifact seams, ghosting from adaptation, oversaturated lora bloom, muddy blend"
)

# Short positive prefix for “real photo not render” (composable with style LoRAs).
PHOTOGRAPHIC_AUTHENTIC_POSITIVE = (
    "authentic photograph, believable lighting, natural materials, real-world texture, "
    "photographic depth, captured image not rendered"
)

# --- Community-reported issues (SDXL, Flux, Illustrious, NoobAI, Z-Image, etc.) ---
# Use these negatives/tips to mitigate concept bleeding, plastic skin, repetitive faces, artifacts, etc.

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

# Over-polished / "too AI" / corporate (Flux, Klein) — we have ANTI_AI_LOOK_NEGATIVE and --naturalize.
# Emotion: nuanced expressions (smug, terrified, hesitant) often default to neutral.
EMOTION_PROMPT_TIPS = [
    "Use explicit emotion words: 'smug smile', 'terrified expression', 'hesitant look', 'contemptuous gaze'.",
    "Put the emotion early in the prompt so it isn't truncated or ignored.",
    "Combine with face/portrait tags: 'portrait, smug expression, raised eyebrow'.",
    "Training: include diverse expressions with clear captions so the model learns nuanced emotions.",
]

# Spatial awareness (behind, next to, under — SDXL preposition failure).
SPATIAL_AWARENESS_TIPS = [
    "Put spatial relations early: 'woman behind the tree', 'cat next to the vase', 'book under the lamp'.",
    "Be explicit: 'X is behind Y', 'X to the left of Y', 'Y in front of X'.",
    "Repeat the relation in different words: 'behind the tree, tree in front of her'.",
]

# V-pred / NoobAI "burn" (CFG too high → burnt colors, neon edges).
CFG_BURN_TIPS = [
    "For v-pred or burn-prone models: keep CFG between 3 and 5.5 (e.g. --cfg-scale 4.5).",
    "Use --cfg-rescale 0.7 (or 0.6) to reduce oversaturation and edge glow.",
    "Lower steps can sometimes reduce burn; try 25–35 steps.",
]

# Background "amnesia" (NoobAI, anime models: blur or impossible geometry).
BACKGROUND_TIPS = [
    "Describe the background explicitly: 'detailed background', 'forest behind', 'urban street', 'interior room'.",
    "Put background after subject: '1girl, red dress, standing in a detailed forest, trees, sunlight'.",
    "Add 'coherent background', 'correct perspective', 'depth' to reduce impossible geometry.",
]

# Centering bias (Klein/Flux: subject always center, passport-photo feel).
CENTERING_TIPS = [
    "Ask for off-center: 'off-center composition', 'rule of thirds', 'subject to the left', 'asymmetric composition'.",
    "Add 'dynamic angle', 'dutch angle', 'looking to the side', 'profile' to avoid straight-on default.",
]

# Distant face "meltdown" (SDXL: faces beyond medium shot smear).
DISTANT_FACE_TIPS = [
    "For SDXL-style models: prefer close-up or medium shot for recognizable faces; use face restorer (e.g. ADetailer) for distant faces.",
    "Add 'detailed face', 'sharp face', 'clear facial features' even for full-body; negative: 'blurry face', 'smeared face', 'alien face'.",
]

# Resolution inflexibility (SDXL: double-head / fragmentation off native buckets).
RESOLUTION_TIPS = [
    "Use native training resolutions (e.g. 1024x1024, 896x1152 for SDXL); straying causes double-heads or fragmented bodies.",
    "sample.py prints a note when output size differs from model native; use --vae-tiling for large decode, but prefer native for composition.",
]

# Low seed variance (Z-Image, some Flux: same composition across seeds).
SEED_VARIANCE_TIPS = [
    "Use --creativity (if the model supports creativity_embed_dim) to increase diversity.",
    "Try multiple seeds and vary prompt slightly (e.g. add 'dynamic pose', 'different angle') to force variation.",
    "Training: ensure dataset has diverse compositions so the model doesn't collapse to one layout.",
]

# Vocabulary / language gaps (Z-Image Qwen encoder, niche tags).
VOCABULARY_TIPS = [
    "Use common, concrete words; avoid niche slang or obscure tags if the model ignores them.",
    "Try synonyms: if 'smug' fails, try 'condescending smile' or 'knowing expression'.",
    "For non-English encoders: translating key terms (e.g. to Chinese for Qwen) can improve adherence for niche objects.",
]

# Anime/stylized models: quality drops without "masterpiece" / "best quality".
QUALITY_TAG_DEPENDENCY_TIPS = [
    "Always prepend quality tags for anime-style models: 'masterpiece, best quality, 1girl, ...' or use --boost-quality.",
    "Training: include quality tags in captions so the model doesn't depend on them at inference; or document that users should use them.",
]

# Orange/green tint (e.g. some hosted generators vs local).
COLOR_TINT_NEGATIVE = "orange tint, green tint, yellow tint, color cast, piss filter, wrong white balance"

# --- Inspired by Stable Diffusion Art, ComfyUI docs, FLUX GitHub, community guides ---

# FLUX grid-like artifact (visible grid in dark areas, worse with Depth/Canny and upscale). GitHub #406.
FLUX_GRID_ARTIFACT_TIPS = [
    "Keep CFG at 3.5 or lower (grid appears when CFG > 3.5).",
    "Use LoRA strength at or below 1.20; higher strength triggers grid artifacts.",
    "Avoid overtrained LoRAs; overtraining is a reported cause of grid patterns.",
    "Grid can appear in early denoising steps and worsen with upscaling/ControlNet; use native resolution when possible.",
]

# Negative prompt best practices (ComfyUI / Stable Diffusion Art: specific > vague, simple, test per checkpoint).
NEGATIVE_PROMPT_BEST_PRACTICES = [
    "Use specific terms (e.g. 'extra fingers', 'blurry face') instead of vague ones like 'bad image'.",
    "Keep negative prompts simple; contradicting or overly long negatives can hurt quality.",
    "Test negatives with your specific checkpoint; what works for one model may not for another.",
    "Weight modifiers (word:0.9) can fine-tune; avoid very high weights (e.g. 1.4) which can cause issues.",
]

# Full-body and two-head (Stable Diffusion Art: describe lower body, use 1:1 or portrait + full-body cues).
FULL_BODY_AND_TWO_HEAD_TIPS = [
    "For full body: add 'standing', 'long dress', 'legs', 'shoes' (describe what you want to see) rather than only 'full body portrait'.",
    "Use portrait aspect for full-body shots so the frame fits one body; use 1:1 for head/shoulder to avoid two-head.",
    "Two-head is common when aspect ratio deviates from 1:1 without full-body cues; either use 1:1 or add standing/legs/dress.",
]

# Garbled faces (Stable Diffusion Art: pixel coverage, Hi-Res Fix, VAE, face restorer, inpainting).
GARBLED_FACE_TIPS = [
    "Faces need enough pixels: use close-up or medium shot, or higher resolution / Hi-Res Fix.",
    "Use an improved VAE if your pipeline supports it; some VAEs fix eye/face issues.",
    "Post-process: face restoration (CodeFormer, GFPGAN) or inpainting/ADetailer for problematic faces.",
]

# LoRA strength (good LoRAs often work at ~1.0; avoid overcooked; FLUX grid at >1.20).
LORA_STRENGTH_TIPS = [
    "Aim for LoRA strength around 1.0 (or 0.8–1.0) unless the model card suggests otherwise.",
    "Strength above 1.20 on FLUX can trigger grid artifacts; keep at or below 1.20.",
    "Overcooked (overtrained) LoRAs can cause artifacts and grid; prefer well-balanced training.",
]

# Prompt structure (Who/What/Where/When — from general prompt-engineering guides).
PROMPT_STRUCTURE_TIPS = [
    "Structure prompts clearly: who (subject), what (action/attributes), where (setting), when/style (lighting, style).",
    "Put the main subject and key details first so they are not truncated (T5 has a token limit).",
    "Be specific and concrete; vague prompts get vague or generic results.",
]

# Originality / novelty: encourage fresher composition/details
# so generations feel less like templated repeats of the same pattern.
# Used by sample.py --originality and train.py --train-originality-prob (via utils/prompt/originality_augment.py).
ORIGINALITY_POSITIVE_TOKENS = [
    "unique composition",
    "original concept",
    "fresh perspective",
    "unexpected details",
    "distinctive lighting",
    "novel scene",
    "inventive composition",
    "creative accidents",
    "new idea",
    "rare details",
    # --- expanded: camera, atmosphere, art-direction (keeps outputs feeling "new") ---
    "unusual angle",
    "dynamic framing",
    "asymmetric balance",
    "bold color story",
    "muted palette twist",
    "dramatic rim light",
    "soft volumetric light",
    "cinematic depth",
    "environmental storytelling",
    "foreground interest",
    "layered depth",
    "tactile materials",
    "hand-crafted look",
    "non-stock pose",
    "candid moment",
    "slice of life",
    "dreamlike twist",
    "subtle surreal touch",
    "editorial fashion vibe",
    "fine art photography",
    "concept art energy",
    "indie aesthetic",
    "one-of-a-kind mood",
    "surprising negative space",
    "rhythm in shapes",
]

# --- Physics, fluids, transparency, rigid & soft bodies (merged into domain dicts below) ---
from .physics_material_prompts import (  # noqa: E402
    FLUIDS_PROMPT_TIPS,
    PARTICULATE_VOLUMETRIC_TIPS,
    PHYSICS_COMMON_NEGATIVE_ADDON,
    PHYSICS_MATERIAL_DOMAIN_NAMES,
    PHYSICS_MATERIAL_RECOMMENDED_NEGATIVE_BY_DOMAIN,
    PHYSICS_MATERIAL_RECOMMENDED_PROMPTS_BY_DOMAIN,
    PHYSICS_TRAINING_CAPTION_TIPS,
    RIGID_CONTACT_SHADOW_TIPS,
    SOFT_BODY_CLOTH_TIPS,
    TRANSPARENCY_REFRACTION_TIPS,
)

RECOMMENDED_PROMPTS_BY_DOMAIN.update(PHYSICS_MATERIAL_RECOMMENDED_PROMPTS_BY_DOMAIN)
RECOMMENDED_NEGATIVE_BY_DOMAIN.update(PHYSICS_MATERIAL_RECOMMENDED_NEGATIVE_BY_DOMAIN)

from .creature_character_prompts import (  # noqa: E402
    ANTHRO_FURRY_NSFW_TIPS,
    ANTHRO_FURRY_PROMPT_TIPS,
    ANTHRO_FURRY_SFW_TIPS,
    CREATURE_CHARACTER_DOMAIN_NAMES,
    CREATURE_CHARACTER_NSFW_RECOMMENDED_NEGATIVE_BY_DOMAIN,
    CREATURE_CHARACTER_NSFW_RECOMMENDED_PROMPTS_BY_DOMAIN,
    CREATURE_CHARACTER_RECOMMENDED_NEGATIVE_BY_DOMAIN,
    CREATURE_CHARACTER_RECOMMENDED_PROMPTS_BY_DOMAIN,
    CREATURE_CHARACTER_SFW_RECOMMENDED_NEGATIVE_BY_DOMAIN,
    CREATURE_CHARACTER_SFW_RECOMMENDED_PROMPTS_BY_DOMAIN,
    CREATURE_COMMON_NEGATIVE_ADDON,
    CREATURE_NSFW_CONTEXT_POSITIVE,
    CREATURE_NSFW_KEYWORDS,
    CREATURE_NSFW_NEGATIVE_ADDON,
    CREATURE_NSFW_SCENARIO_TIPS,
    CREATURE_SFW_CONTEXT_POSITIVE,
    CREATURE_SFW_NEGATIVE_ADDON,
    CREATURE_SFW_SCENARIO_TIPS,
    CREATURE_TRAINING_CAPTION_TIPS,
    CREATURES_NSFW_TIPS,
    CREATURES_PROMPT_TIPS,
    CREATURES_SFW_TIPS,
    HUMANOID_MONSTER_PROMPT_KEYWORDS,
    HUMANOID_MONSTERS_NSFW_TIPS,
    HUMANOID_MONSTERS_PROMPT_TIPS,
    HUMANOID_MONSTERS_SFW_TIPS,
    ROBOTS_MECH_NSFW_TIPS,
    ROBOTS_MECH_PROMPT_TIPS,
    ROBOTS_MECH_SFW_TIPS,
)

RECOMMENDED_PROMPTS_BY_DOMAIN.update(CREATURE_CHARACTER_RECOMMENDED_PROMPTS_BY_DOMAIN)
RECOMMENDED_NEGATIVE_BY_DOMAIN.update(CREATURE_CHARACTER_RECOMMENDED_NEGATIVE_BY_DOMAIN)

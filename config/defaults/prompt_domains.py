"""Shared prompt strings: default/text-in-image negatives, hard-style prefixes and the anti-AI-look lexicon."""

__all__ = [
    "DEFAULT_NEGATIVE_PROMPT",
    "TEXT_IN_IMAGE_NEGATIVE",
    "TEXT_IN_IMAGE_PHRASES",
    "HARD_STYLE_RECOMMENDED_PROMPTS",
    "ANTI_AI_LOOK_NEGATIVE",
    "ANTI_AI_LOOK_NEGATIVE_STRONG",
    "NATURAL_LOOK_POSITIVE",
    "NATURAL_LOOK_POSITIVE_DEEP",
    "LORA_STACK_NEGATIVE",
    "CONCEPT_BLEEDING_NEGATIVE",
    "CONCEPT_BLEEDING_POSITIVE",
    "ARTIFACT_NEGATIVES",
    "WATERMARK_NEGATIVE_STRONG",
    "FLUX_FACE_DIVERSITY_NEGATIVE",
    "DIVERSITY_POSITIVE",
]


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
# Grain and sensor noise are not requested here: the capture finish (utils/quality/camera_isp.py)
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

# PixAI-style emphasis and ReVe-style caption normalization for strong prompt adherence.
# Quality tags (10x boost), anti-blending, count-aware.
import re
import random
from typing import List, Optional, Tuple

# Quality tags that should strongly improve output when present (repeat/boost in caption)
QUALITY_TAGS = [
    "masterpiece", "best quality", "high quality", "highres", "8k", "ultra detailed",
    "absurdres", "detailed", "sharp focus", "professional", "perfect composition",
]

# Person descriptors: order for prompt_from_tags / normalize_tag_order (subject → age → height → build → anatomy → rest).
SUBJECT_PREFIXES = ("1girl", "1boy", "2girls", "2boys", "3girls", "1 other", "solo", "multiple", "woman", "man", "girl", "boy")
AGE_TAGS = (
    "child", "toddler", "teen", "teenager", "young", "adult", "middle-aged", "elderly", "old", "aged",
    "kid", "youth", "minor", "mature", "senior", "little girl", "little boy", "young woman", "young man",
)
HEIGHT_TAGS = (
    "tall", "short", "average height", "very tall", "petite", "giant", "tiny", "towering",
    "tall woman", "tall man", "short woman", "short man", "height difference", "long legs", "short legs",
)
BUILD_BODY_TAGS = (
    "slim", "thin", "skinny", "muscular", "athletic", "curvy", "plus size", "fat", "obese", "chubby",
    "petite", "hourglass", "broad shoulders", "narrow waist", "large breasts", "small breasts",
    "flat chest", "muscular woman", "muscular man", "lean", "stocky", "heavy", "voluptuous",
)
ANATOMY_FRAMING_TAGS = (
    "full body", "upper body", "portrait", "bust", "face focus", "close-up", "cowboy shot",
    "from above", "from below", "profile", "back", "front", "side view", "head to toe",
    "head out of frame", "navel", "cleavage", "ass focus", "foot focus", "hand focus",
)
BODY_PART_TAGS = (
    "long hair", "short hair", "hands", "feet", "visible hands", "arms", "legs", "fingers",
    "bare feet", "open mouth", "smile", "eyes", "hair", "braid", "ponytail", "bangs",
    "correct hands", "five fingers", "visible feet", "crossed arms", "raised arm", "arm up",
)

# Domain tags that many models struggle with — boost when present so our model learns them well.
# Use these in training captions and at inference for 3D, realistic, interior/exterior, etc.
DOMAIN_TAGS = {
    "3d": [
        "3d render", "3d illustration", "octane render", "cinema 4d", "blender",
        "isometric", "low poly", "voxel", "solid shading", "clean 3d",
    ],
    "realistic": [
        "photorealistic", "realistic", "photo", "real photography", "hyperrealistic",
        "raw photo", "natural lighting", "detailed skin", "skin texture",
    ],
    "interior": [
        "interior", "interior design", "indoor", "room", "living room", "bedroom",
        "architecture interior", "furniture", "cozy interior", "modern interior",
    ],
    "exterior": [
        "exterior", "outdoor", "architecture", "building", "facade", "landscape",
        "street view", "urban", "nature", "outdoors",
    ],
    "other_hard": [
        "hands", "correct anatomy", "perspective", "symmetry", "text", "lettering",
        "multiple objects", "complex composition", "depth of field",
    ],
    # Text in image: legible text, signs, labels (many models render text poorly; boost when present).
    "text_in_image": [
        "legible text", "clear text", "readable text", "sharp text", "clear lettering",
        "sign that says", "text that says", "lettering", "written text", "caption",
        "label", "headline", "title", "correct spelling", "readable",
    ],
    # Things other models famously suck at — boost so we learn them (research: SD/SDXL/FLUX limitations)
    "anatomy": [
        "correct anatomy", "correct hands", "five fingers", "proper proportions",
        "full body", "standing", "legs", "feet", "visible hands", "natural pose",
    ],
    "avoid_failures": [
        "single subject", "no duplicate", "clear composition", "readable text",
        "correct number of", "distinct", "well proportioned", "coherent",
    ],
    # Concept bleeding (SDXL etc.): colors/objects bleeding together; boost so model keeps distinct colors/edges.
    "concept_bleed": [
        "distinct colors", "separate colors", "clear separation", "no color bleed", "defined edges",
        "distinct objects", "separate objects", "clean edges", "sharp boundaries",
    ],
    # Complex prompts: multi-element, detailed, long captions (boost for better adherence).
    "complex": [
        "complex composition", "multiple elements", "detailed", "specific",
        "intricate", "layered", "coherent", "full detail", "ultra detailed",
    ],
    # Challenging / surreal / abstract / weird (boost so model learns these; no censorship).
    "challenging": [
        "surreal", "abstract", "fantasy", "unusual", "bizarre", "dreamlike",
        "moody", "atmospheric", "detailed", "masterpiece", "best quality",
    ],
    # Style / artist tags (PixAI, Danbooru, etc.): strong style conditioning.
    "style_artist": [
        "oil painting", "watercolor", "digital painting", "concept art",
        "cel shading", "cinematic", "fantasy art", "character design",
        "in the style of", "art by", "drawn by", "style of",
        "ghibli", "studio ghibli", "miyazaki", "digital art", "official art",
    ],
    # Person descriptors: height, age, body size/build, anatomy & body parts — order and boost for consistent adherence.
    "person_descriptors": (
        list(AGE_TAGS) + list(HEIGHT_TAGS) + list(BUILD_BODY_TAGS)
        + list(ANATOMY_FRAMING_TAGS) + list(BODY_PART_TAGS)
    ),
}
# Flat list for "boost when present" (so model learns these domains strongly)
DOMAIN_TAGS_FLAT = [t for tags in DOMAIN_TAGS.values() for t in tags]

# Hard styles: 3D, photorealistic, and style mixes that most AI image models struggle with.
# Boosted more strongly than general domain tags so the model learns 3D/realistic/mixed looks.
# See config/prompt_domains.py for recommended prompts and negatives per hard style.
HARD_STYLE_TAGS = {
    "3d": [
        "3d render", "3d illustration", "octane render", "cinema 4d", "blender",
        "isometric", "low poly", "voxel", "solid shading", "clean 3d", "cg",
        "unreal engine", "unity", "3d model", "rendered", "subsurface scattering",
    ],
    "realistic": [
        "photorealistic", "realistic", "hyperrealistic", "raw photo", "real photography",
        "natural lighting", "detailed skin", "skin texture", "photo", "real life",
        "8k uhd", "dslr", "film grain", "lens flare", "depth of field", "bokeh",
    ],
    # Style mixes: 2.5D, semi-realistic, anime+realistic, etc. — models often blur these.
    "style_mix": [
        "2.5d", "2.5d style", "semi-realistic", "semi realistic", "anime realistic",
        "photorealistic anime", "realistic anime", "3d anime", "anime 3d",
        "realistic 3d", "3d realistic", "mixed style", "hybrid style", "stylized realistic",
        "illustration realistic", "realistic illustration", "painterly realistic",
    ],
}
HARD_STYLE_TAGS_FLAT = [t for tags in HARD_STYLE_TAGS.values() for t in tags]

# Universal negative prompt snippets that help avoid common diffusion-model failures.
# Use at inference or append to training negative_caption when relevant (hands, anatomy, double head, etc.)
NEGATIVE_ANATOMY = "bad anatomy, bad hands, missing fingers, extra fingers, fused fingers, mutated hands, poorly drawn hands, deformed hands, extra limbs, missing limbs, malformed limbs, disfigured"
NEGATIVE_FACE = "bad face, deformed face, ugly face, distorted eyes, asymmetric eyes, wrong eyes, blurry face, mutated face"
NEGATIVE_COMPOSITION = "duplicate, duplicate head, two heads, merged subjects, cropped, out of frame, bad composition, cut off, missing body parts"
NEGATIVE_QUALITY = "blurry, low quality, worst quality, jpeg artifacts, distorted, oversaturated, underexposed, overexposed"
# Use when you want to avoid bad/unwanted text but NOT when you want text in the image.
NEGATIVE_TEXT = "garbled text, misspelled text, unreadable text, wrong text, watermark, signature"
# When generating images that should contain text, use this (avoids bad text, keeps desired text possible).
NEGATIVE_BAD_TEXT_ONLY = "garbled text, misspelled, wrong spelling, illegible, watermark, signature"
# Combined for "maximum safety" when generating people/anatomy
NEGATIVE_ANATOMY_FULL = f"{NEGATIVE_ANATOMY}, {NEGATIVE_FACE}, {NEGATIVE_COMPOSITION}, {NEGATIVE_QUALITY}"

# Phrases that imply multiple people/crowd — we add anti-blending and respect count
MULTI_PERSON_PHRASES = (
    "2girls", "2boys", "3girls", "multiple girls", "crowd", "group", "room full of people",
    "many people", "several people", "lots of people", "gathering", "audience",
)
ANTI_BLEND_POSITIVE = "distinct characters, no character blending, clear separation, separate individuals"
ANTI_BLEND_NEGATIVE = "character blending, merged figures, fused characters, blur between people"


def normalize_tag(tag: str) -> str:
    """Normalize a single tag: strip, optional underscore to space (for Danbooru-style)."""
    t = tag.strip()
    # Keep underscore for tokenizer consistency with tag boards; optionally normalize to space
    return t


def normalize_tags_string(tags_str: str, underscore_to_space: bool = False) -> str:
    """
    Normalize a comma-separated tag string: strip each tag, optionally replace _ with space.
    Returns a single string "tag1, tag2, tag3" for use in prompts.
    """
    if not (tags_str and tags_str.strip()):
        return ""
    parts = [p.strip() for p in tags_str.split(",") if p.strip()]
    if underscore_to_space:
        parts = [p.replace("_", " ") for p in parts]
    return ", ".join(parts)


def _person_descriptor_bucket(tag: str) -> int:
    """
    Return sort key for person-descriptor ordering: 0=subject, 1=age, 2=height, 3=build, 4=anatomy/framing, 5=body parts, 6=other.
    Used so prompts consistently order subject → age → height → size/build → anatomy/body parts → rest.
    """
    t = tag.lower().strip()
    if any(t.startswith(p) for p in SUBJECT_PREFIXES):
        return 0
    buckets = [(1, AGE_TAGS), (2, HEIGHT_TAGS), (3, BUILD_BODY_TAGS), (4, ANATOMY_FRAMING_TAGS), (5, BODY_PART_TAGS)]
    for bucket_id, term_list in buckets:
        for term in term_list:
            if t == term or t.startswith(term + " ") or t.replace("_", " ") == term:
                return bucket_id
    return 6


def prompt_from_tags(tags: List[str], subject_first: bool = True) -> str:
    """
    Build a prompt string from a list of tags (e.g. from --tags or a tag file).
    When subject_first=True, orders tags: subject → age → height → build/size → anatomy/framing → body parts → rest.
    """
    if not tags:
        return ""
    tags = [normalize_tag(t) for t in tags if normalize_tag(t)]
    if not subject_first or len(tags) <= 1:
        return ", ".join(tags)
    # Stable sort by person-descriptor bucket: subject → age → height → build → anatomy → body parts → other
    indexed = [(t, i) for i, t in enumerate(tags)]
    ordered = sorted(indexed, key=lambda ti: (_person_descriptor_bucket(ti[0]), ti[1]))
    return ", ".join(t[0] for t in ordered)


def apply_pixai_emphasis(caption: str, expand_emphasis: bool = True, expand_deemphasis: bool = False) -> str:
    """
    PixAI-style: (tag) or ((tag)) = emphasize; [tag] = de-emphasize.
    expand_emphasis: (tag) -> repeat tag 1x for stronger focus; ((tag)) -> repeat 2x.
    expand_deemphasis: [tag] -> remove or keep once (we keep once by default).
    """
    if not caption.strip():
        return caption

    # ((word)) -> duplicate twice for strong emphasis
    def replace_double(m):
        t = m.group(1).strip()
        return ", ".join([t] * 3) if expand_emphasis else t

    caption = re.sub(r"\(\(\s*([^)]+)\s*\)\)", replace_double, caption, flags=re.IGNORECASE)

    # (word) -> duplicate once for emphasis
    def replace_single(m):
        t = m.group(1).strip()
        return f"{t}, {t}" if expand_emphasis else t

    caption = re.sub(r"\(\s*([^)]+)\s*\)", replace_single, caption, flags=re.IGNORECASE)

    # [word] -> de-emphasize: keep once (or remove if expand_deemphasis)
    def replace_bracket(m):
        t = m.group(1).strip()
        return "" if expand_deemphasis else t

    caption = re.sub(r"\[\s*([^]]+)\s*\]", replace_bracket, caption, flags=re.IGNORECASE)

    # Normalize commas and spaces
    caption = re.sub(r"\s*,\s*", ", ", caption)
    caption = re.sub(r"\s+", " ", caption).strip()
    return caption


def structured_to_tags(parts: dict, order: Optional[List[str]] = None) -> str:
    """ReVe-style: build caption from structured parts (subject, setting, style, etc.)."""
    if order is None:
        order = ["subject", "setting", "style", "aesthetics", "camera", "extra"]
    return ", ".join(parts.get(k, "").strip() for k in order if parts.get(k, "").strip())


def normalize_tag_order(caption: str, put_subject_first: bool = True) -> str:
    """
    PixAI-style: order tags for better adherence. When put_subject_first=True, uses full person-descriptor order:
    subject → age → height → build/size → anatomy/framing → body parts → other (same as prompt_from_tags).
    """
    if not put_subject_first or "," not in caption:
        return caption
    tags = [t.strip() for t in caption.split(",") if t.strip()]
    indexed = [(t, i) for i, t in enumerate(tags)]
    ordered = sorted(indexed, key=lambda ti: (_person_descriptor_bucket(ti[0]), ti[1]))
    return ", ".join(t[0] for t in ordered)


def boost_hard_style_tags(caption: str, repeat_factor: int = 3) -> str:
    """
    When 3D, photorealistic, or style-mix tags are present, prepend them (repeated) so the model
    strongly learns these hard styles. Use before boost_quality_tags so style anchors the prompt.
    """
    if not caption.strip():
        return caption
    caption_lower = caption.lower()
    found = [t for t in HARD_STYLE_TAGS_FLAT if t in caption_lower]
    if not found:
        return caption
    extra = ", ".join(found * repeat_factor)
    return f"{extra}, {caption}".strip()


def boost_quality_tags(caption: str, repeat_factor: int = 3) -> str:
    """
    When quality tags are present, repeat them at the start so the model strongly associates
    them with high-quality output (10x-style improvement). repeat_factor=3 means prepend 2 extra copies.
    """
    if not caption.strip():
        return caption
    caption_lower = caption.lower()
    found = [q for q in QUALITY_TAGS if q in caption_lower]
    if not found:
        return caption
    # Prepend repeated quality tags so they dominate the sequence
    extra = ", ".join(found * repeat_factor)
    return f"{extra}, {caption}".strip()


def boost_domain_tags(caption: str, repeat_factor: int = 2) -> str:
    """
    When domain tags (3D, realistic, interior, exterior, etc.) are present, repeat them
    so the model learns these hard-to-generate domains well. repeat_factor=2 means prepend 1 extra copy.
    """
    if not caption.strip():
        return caption
    caption_lower = caption.lower()
    found = [t for t in DOMAIN_TAGS_FLAT if t in caption_lower]
    if not found:
        return caption
    extra = ", ".join(found * repeat_factor)
    return f"{extra}, {caption}".strip()


# Default quality prefix for challenging/short prompts (inference or data).
QUALITY_PREFIX = "masterpiece, best quality, "


def prepend_quality_if_short(caption: str, min_tag_count: int = 4) -> str:
    """
    If caption has fewer than min_tag_count comma-separated tags, prepend QUALITY_PREFIX
    so short or vague prompts get stronger adherence. Use in data pipeline or inference.
    """
    if not caption.strip():
        return caption
    tags = [t.strip() for t in caption.split(",") if t.strip()]
    if len(tags) >= min_tag_count:
        return caption
    return f"{QUALITY_PREFIX}{caption}".strip()


def add_anti_blending_and_count(caption: str, negative_prompt: str) -> Tuple[str, str]:
    """
    For multi-person prompts: add anti-blending to positive and blending to negative
    so the model learns to avoid character bleed and respect "room full of people" etc.
    Returns (augmented_caption, augmented_negative_prompt).
    """
    c_lower = caption.lower()
    is_multi = any(p in c_lower for p in MULTI_PERSON_PHRASES) or re.search(r"\d+\s*(girl|boy|person|people)", c_lower)
    if not is_multi:
        return caption, negative_prompt
    # Add anti-blend to positive; add blend terms to negative
    new_cap = f"{caption}, {ANTI_BLEND_POSITIVE}".strip() if ANTI_BLEND_POSITIVE not in caption else caption
    new_neg = f"{negative_prompt}, {ANTI_BLEND_NEGATIVE}".strip() if negative_prompt else ANTI_BLEND_NEGATIVE
    return new_cap, new_neg

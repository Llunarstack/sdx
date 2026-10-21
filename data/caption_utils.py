"""Training-caption helpers: tag emphasis, person-descriptor ordering, tag front-loading, anti-blending,
regional captions and comma-safe truncation (torch-free)."""

import re
from typing import Any

QUALITY_TAGS = [
    "masterpiece",
    "best quality",
    "high quality",
    "highres",
    "8k",
    "ultra detailed",
    "absurdres",
    "detailed",
    "sharp focus",
    "professional",
    "perfect composition",
]

# Person-descriptor buckets, in the order prompt_from_tags / normalize_tag_order put them (unmatched tags go last).
SUBJECT_PREFIXES = (
    "1girl",
    "1boy",
    "2girls",
    "2boys",
    "3girls",
    "1 other",
    "solo",
    "multiple",
    "woman",
    "man",
    "girl",
    "boy",
)
AGE_TAGS = (
    "child",
    "toddler",
    "teen",
    "teenager",
    "young",
    "adult",
    "middle-aged",
    "elderly",
    "old",
    "aged",
    "kid",
    "youth",
    "minor",
    "mature",
    "senior",
    "little girl",
    "little boy",
    "young woman",
    "young man",
)
HEIGHT_TAGS = (
    "tall",
    "short",
    "average height",
    "very tall",
    "petite",
    "giant",
    "tiny",
    "towering",
    "tall woman",
    "tall man",
    "short woman",
    "short man",
    "height difference",
    "long legs",
    "short legs",
)
BUILD_BODY_TAGS = (
    "slim",
    "thin",
    "skinny",
    "muscular",
    "athletic",
    "curvy",
    "plus size",
    "fat",
    "obese",
    "chubby",
    "petite",
    "hourglass",
    "broad shoulders",
    "narrow waist",
    "large breasts",
    "small breasts",
    "flat chest",
    "muscular woman",
    "muscular man",
    "lean",
    "stocky",
    "heavy",
    "voluptuous",
)
ANATOMY_FRAMING_TAGS = (
    "full body",
    "upper body",
    "portrait",
    "bust",
    "face focus",
    "close-up",
    "cowboy shot",
    "from above",
    "from below",
    "profile",
    "back",
    "front",
    "side view",
    "head to toe",
    "head out of frame",
    "navel",
    "cleavage",
    "ass focus",
    "foot focus",
    "hand focus",
)
BODY_PART_TAGS = (
    "long hair",
    "short hair",
    "hands",
    "feet",
    "visible hands",
    "arms",
    "legs",
    "fingers",
    "bare feet",
    "open mouth",
    "smile",
    "eyes",
    "hair",
    "braid",
    "ponytail",
    "bangs",
    "correct hands",
    "five fingers",
    "visible feet",
    "crossed arms",
    "raised arm",
    "arm up",
)

# Domain tags models commonly under-learn; front-loaded when a caption contains them.
DOMAIN_TAGS = {
    "3d": [
        "3d render",
        "3d illustration",
        "octane render",
        "cinema 4d",
        "blender",
        "isometric",
        "low poly",
        "voxel",
        "solid shading",
        "clean 3d",
    ],
    "realistic": [
        "photorealistic",
        "realistic",
        "photo",
        "real photography",
        "hyperrealistic",
        "raw photo",
        "natural lighting",
        "detailed skin",
        "skin texture",
    ],
    "interior": [
        "interior",
        "interior design",
        "indoor",
        "room",
        "living room",
        "bedroom",
        "architecture interior",
        "furniture",
        "cozy interior",
        "modern interior",
    ],
    "exterior": [
        "exterior",
        "outdoor",
        "architecture",
        "building",
        "facade",
        "landscape",
        "street view",
        "urban",
        "nature",
        "outdoors",
    ],
    "other_hard": [
        "hands",
        "correct anatomy",
        "perspective",
        "symmetry",
        "text",
        "lettering",
        "multiple objects",
        "complex composition",
        "depth of field",
    ],
    "text_in_image": [
        "legible text",
        "clear text",
        "readable text",
        "sharp text",
        "clear lettering",
        "sign that says",
        "text that says",
        "lettering",
        "written text",
        "caption",
        "label",
        "headline",
        "title",
        "correct spelling",
        "readable",
    ],
    "anatomy": [
        "correct anatomy",
        "correct hands",
        "five fingers",
        "proper proportions",
        "full body",
        "standing",
        "legs",
        "feet",
        "visible hands",
        "natural pose",
    ],
    "avoid_failures": [
        "single subject",
        "no duplicate",
        "clear composition",
        "readable text",
        "correct number of",
        "distinct",
        "well proportioned",
        "coherent",
    ],
    "concept_bleed": [
        "distinct colors",
        "separate colors",
        "clear separation",
        "no color bleed",
        "defined edges",
        "distinct objects",
        "separate objects",
        "clean edges",
        "sharp boundaries",
    ],
    "complex": [
        "complex composition",
        "multiple elements",
        "detailed",
        "specific",
        "intricate",
        "layered",
        "coherent",
        "full detail",
        "ultra detailed",
    ],
    "challenging": [
        "surreal",
        "abstract",
        "fantasy",
        "unusual",
        "bizarre",
        "dreamlike",
        "moody",
        "atmospheric",
        "detailed",
        "masterpiece",
        "best quality",
    ],
    "style_artist": [
        "oil painting",
        "watercolor",
        "digital painting",
        "concept art",
        "cel shading",
        "cinematic",
        "fantasy art",
        "character design",
        "in the style of",
        "art by",
        "drawn by",
        "style of",
        "ghibli",
        "studio ghibli",
        "miyazaki",
        "digital art",
        "official art",
    ],
    "person_descriptors": (
        list(AGE_TAGS) + list(HEIGHT_TAGS) + list(BUILD_BODY_TAGS) + list(ANATOMY_FRAMING_TAGS) + list(BODY_PART_TAGS)
    ),
}
DOMAIN_TAGS_FLAT = [t for tags in DOMAIN_TAGS.values() for t in tags]

# 3D / photoreal / style-mix tags; front-loaded before quality and domain tags so the style anchors the caption.
HARD_STYLE_TAGS = {
    "3d": [
        "3d render",
        "3d illustration",
        "octane render",
        "cinema 4d",
        "blender",
        "isometric",
        "low poly",
        "voxel",
        "solid shading",
        "clean 3d",
        "cg",
        "unreal engine",
        "unity",
        "3d model",
        "rendered",
        "subsurface scattering",
    ],
    "realistic": [
        "photorealistic",
        "realistic",
        "hyperrealistic",
        "raw photo",
        "real photography",
        "natural lighting",
        "detailed skin",
        "skin texture",
        "photo",
        "real life",
        "8k uhd",
        "dslr",
        "film grain",
        "lens flare",
        "depth of field",
        "bokeh",
    ],
    "style_mix": [
        "2.5d",
        "2.5d style",
        "semi-realistic",
        "semi realistic",
        "anime realistic",
        "photorealistic anime",
        "realistic anime",
        "3d anime",
        "anime 3d",
        "realistic 3d",
        "3d realistic",
        "mixed style",
        "hybrid style",
        "stylized realistic",
        "illustration realistic",
        "realistic illustration",
        "painterly realistic",
    ],
}
HARD_STYLE_TAGS_FLAT = [t for tags in HARD_STYLE_TAGS.values() for t in tags]


# Phrases that imply several people; such captions get the anti-blending pair.
MULTI_PERSON_PHRASES = (
    "2girls",
    "2boys",
    "3girls",
    "multiple girls",
    "crowd",
    "group",
    "room full of people",
    "many people",
    "several people",
    "lots of people",
    "gathering",
    "audience",
)
ANTI_BLEND_POSITIVE = "distinct characters, no character blending, clear separation, separate individuals"
ANTI_BLEND_NEGATIVE = "character blending, merged figures, fused characters, blur between people"


_DESCRIPTOR_BUCKETS = (AGE_TAGS, HEIGHT_TAGS, BUILD_BODY_TAGS, ANATOMY_FRAMING_TAGS, BODY_PART_TAGS)


def _person_descriptor_bucket(tag: str) -> int:
    """Sort key: 0 subject, 1 age, 2 height, 3 build, 4 anatomy/framing, 5 body parts, 6 anything else."""
    t = tag.lower().strip()
    if any(t.startswith(p) for p in SUBJECT_PREFIXES):
        return 0
    for bucket_id, terms in enumerate(_DESCRIPTOR_BUCKETS, start=1):
        for term in terms:
            if t == term or t.startswith(term + " ") or t.replace("_", " ") == term:
                return bucket_id
    return 6


def _order_person_first(tags: list[str]) -> str:
    return ", ".join(sorted(tags, key=_person_descriptor_bucket))


def prompt_from_tags(tags: list[str], subject_first: bool = True) -> str:
    """Join tags into a prompt; ``subject_first`` stable-sorts them subject -> age -> height -> build ->
    anatomy/framing -> body parts -> rest."""
    tags = [t.strip() for t in tags or () if t.strip()]
    return _order_person_first(tags) if subject_first else ", ".join(tags)


def apply_tag_emphasis(caption: str, expand_emphasis: bool = True, expand_deemphasis: bool = False) -> str:
    """Expand Danbooru-style emphasis into repetition T5 can see.

    ``((tag))`` -> three copies and ``(tag)`` -> two (just the tag when ``expand_emphasis`` is off);
    ``[tag]`` is kept once, or dropped when ``expand_deemphasis`` is on.
    """
    if not caption.strip():
        return caption

    def repeat(n: int):
        return lambda m: ", ".join([m.group(1).strip()] * (n if expand_emphasis else 1))

    caption = re.sub(r"\(\(\s*([^)]+)\s*\)\)", repeat(3), caption)
    caption = re.sub(r"\(\s*([^)]+)\s*\)", repeat(2), caption)
    caption = re.sub(r"\[\s*([^]]+)\s*\]", lambda m: "" if expand_deemphasis else m.group(1).strip(), caption)
    caption = re.sub(r"\s*,\s*", ", ", caption)
    return re.sub(r"\s+", " ", caption).strip()


def normalize_tag_order(caption: str, put_subject_first: bool = True) -> str:
    """Reorder a comma-separated caption in person-descriptor order (same order as :func:`prompt_from_tags`)."""
    if not put_subject_first or "," not in caption:
        return caption
    return _order_person_first([t.strip() for t in caption.split(",") if t.strip()])


def front_load_tags(caption: str, vocab) -> str:
    """Move the caption's comma-separated tags found in ``vocab`` (whole tag, case-insensitive) to the front, once
    each; the other tags keep their order. Nothing is repeated: repetition burns encoder tokens and binds boilerplate
    tags to whatever images happen to carry them."""
    vocab_l = {v.lower() for v in vocab}
    head: list[str] = []
    tail: list[str] = []
    seen: set[str] = set()
    for tag in (t.strip() for t in caption.split(",")):
        if not tag:
            continue
        key = tag.lower()
        if key not in vocab_l:
            tail.append(tag)
        elif key not in seen:
            seen.add(key)
            head.append(tag)
    return ", ".join(head + tail) if head else caption


def boost_hard_style_tags(caption: str) -> str:
    """Front-load 3D / photoreal / style-mix tags (:data:`HARD_STYLE_TAGS`) so the style anchors the prompt."""
    return front_load_tags(caption, HARD_STYLE_TAGS_FLAT)


# Prepended during training when ``use_adherence_boost`` is on — nudges T5 toward literal caption following.
ADHERENCE_TAGS: tuple[str, ...] = (
    "faithful to caption",
    "accurate to description",
    "all specified details",
)


def prepend_adherence_boost(caption: str) -> str:
    """Prepend the adherence tags once so long/complex captions anchor on literal conditioning."""
    if not caption.strip():
        return caption
    return f"{', '.join(ADHERENCE_TAGS)}, {caption}".strip()


def apply_training_guidance_to_caption_pair(
    caption: str,
    negative_caption: str = "",
    *,
    shortcomings_mode: str = "none",
    shortcomings_2d: bool = False,
    art_guidance_mode: str = "none",
    anatomy_guidance: str = "none",
    style_guidance_mode: str = "none",
    style_guidance_artists: bool = True,
    include_art_guidance_photography: bool = True,
) -> tuple[str, str]:
    """Shortcomings / art-medium / anatomy / style guidance on a caption pair (PromptStack guidance stage)."""
    from utils.prompt.stack.stages.guidance import apply_training_guidance_pair

    return apply_training_guidance_pair(
        caption,
        negative_caption or "",
        shortcomings_mode=shortcomings_mode,
        shortcomings_2d=shortcomings_2d,
        art_guidance_mode=art_guidance_mode,
        anatomy_guidance=anatomy_guidance,
        style_guidance_mode=style_guidance_mode,
        style_guidance_artists=style_guidance_artists,
        include_art_guidance_photography=include_art_guidance_photography,
    )


def boost_quality_tags(caption: str) -> str:
    """Front-load the caption's own quality tags (:data:`QUALITY_TAGS`), deduplicated."""
    return front_load_tags(caption, QUALITY_TAGS)


def boost_domain_tags(caption: str) -> str:
    """Front-load the caption's domain tags (3D, realistic, interior, anatomy...; :data:`DOMAIN_TAGS`), deduplicated."""
    return front_load_tags(caption, DOMAIN_TAGS_FLAT)


# Key order for dict-style JSONL ``parts`` so subject / clothing / background stay consistent.
REGION_PART_ORDER: tuple[str, ...] = (
    "subject",
    "person",
    "character",
    "characters",
    "outfit",
    "clothing",
    "accessories",
    "body",
    "hands",
    "face",
    "foreground",
    "background",
    "props",
    "lighting",
    "style",
)


_REGION_RANK = {name: i for i, name in enumerate(REGION_PART_ORDER)}


def _region_sort_key(key: str) -> tuple[int, str]:
    return _REGION_RANK.get(key.lower().strip(), len(REGION_PART_ORDER)), key


def format_parts_dict(parts: dict, order: list[str] | None = None) -> str:
    """Turn {'subject': '...', 'background': '...'} into one line for T5."""
    if not parts:
        return ""
    if order:
        keys = [k for k in order if k in parts and str(parts[k]).strip()]
    else:
        keys = sorted(parts.keys(), key=_region_sort_key)
    return " | ".join(f"{k}: {str(parts[k]).strip()}" for k in keys if str(parts[k]).strip())


def _normalize_region_item(item: Any) -> str:
    """List element: str or {label, text|caption|description}."""
    if item is None:
        return ""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        label = (item.get("label") or item.get("name") or item.get("region") or "").strip()
        text = (item.get("text") or item.get("caption") or item.get("description") or "").strip()
        return f"{label}: {text}" if text and label else text
    return str(item).strip()


def format_region_captions_block(regions: Any) -> str:
    """One ``a: x | b: y`` block from ``region_captions`` lists, a ``parts`` dict, or ``{parts, region_captions}``."""
    if regions is None:
        return ""
    if isinstance(regions, dict) and "parts" in regions and "region_captions" in regions:
        a = format_parts_dict(regions["parts"]) if isinstance(regions["parts"], dict) else ""
        b = format_region_captions_block(regions["region_captions"])
        return " | ".join(x for x in (a, b) if x)
    if isinstance(regions, dict) and "region_captions" in regions and isinstance(regions.get("region_captions"), list):
        return format_region_captions_block(regions["region_captions"])
    if isinstance(regions, dict):
        return format_parts_dict(regions)
    if isinstance(regions, list):
        return " | ".join(b for b in map(_normalize_region_item, regions) if b)
    return str(regions).strip()


def merge_region_captions_into_caption(
    base_caption: str,
    regions: Any,
    *,
    mode: str = "append",
    layout_tag: str = "[layout]",
) -> str:
    """Join the global caption and the ``layout_tag``-prefixed region block (``mode``: append | prefix | off)."""
    if mode == "off" or not regions:
        return base_caption
    block = format_region_captions_block(regions)
    if not block:
        return base_caption
    tagged = f"{layout_tag} {block}".strip()
    base_caption = (base_caption or "").strip()
    if mode == "prefix":
        return f"{tagged}. {base_caption}".strip() if base_caption else tagged
    return f"{base_caption}. {tagged}".strip() if base_caption else tagged


QUALITY_PREFIX = "masterpiece, best quality, "


def prepend_quality_if_short(caption: str, min_tag_count: int = 4) -> str:
    """Prepend :data:`QUALITY_PREFIX` to captions with fewer than ``min_tag_count`` comma-separated tags."""
    if not caption.strip() or sum(1 for t in caption.split(",") if t.strip()) >= min_tag_count:
        return caption
    return f"{QUALITY_PREFIX}{caption}".strip()


# Booru-style head counts ("2girls", "1girl, 1boy", "3 people"); two or more means several people.
_HEAD_COUNT_RE = re.compile(r"(?<!\d)(\d+)\s*(?:girls?|boys?|persons?|people)\b")


def add_anti_blending_and_count(caption: str, negative_prompt: str) -> tuple[str, str]:
    """For multi-person captions, add the anti-blending positive and the blending terms to the negative."""
    c_lower = caption.lower()
    head_count = sum(int(n) for n in _HEAD_COUNT_RE.findall(c_lower))
    if head_count < 2 and not any(p in c_lower for p in MULTI_PERSON_PHRASES):
        return caption, negative_prompt
    new_cap = caption if ANTI_BLEND_POSITIVE in caption else f"{caption}, {ANTI_BLEND_POSITIVE}".strip()
    new_neg = f"{negative_prompt}, {ANTI_BLEND_NEGATIVE}".strip() if negative_prompt else ANTI_BLEND_NEGATIVE
    return new_cap, new_neg


def truncate_caption_at_comma_boundary(caption: str, max_len: int) -> str:
    """Keep whole comma-separated tags from the start while the caption fits ``max_len`` characters.

    A single first tag longer than ``max_len`` is hard-cut; ``max_len <= 0`` disables truncation.
    """
    text = (caption or "").strip()
    if max_len <= 0 or len(text) <= max_len:
        return text
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return text[:max_len]
    kept: list[str] = []
    length = 0
    for part in parts:
        add = len(part) + (2 if kept else 0)
        if length + add > max_len:
            break
        kept.append(part)
        length += add
    return ", ".join(kept) if kept else parts[0][:max_len]

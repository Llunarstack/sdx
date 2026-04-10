from __future__ import annotations


def recommend_holy_grail_preset(
    *,
    prompt: str = "",
    style: str = "",
    has_control: bool = False,
    has_lora: bool = False,
) -> str:
    """
    Lightweight heuristic recommender for holy-grail presets.
    """
    text = f"{prompt} {style}".lower()
    if any(k in text for k in ("anime", "manga", "cel", "waifu", "danbooru")):
        return "anime"
    if any(k in text for k in ("illustration", "concept art", "painterly", "comic", "storybook")):
        return "illustration"
    if has_control and has_lora:
        return "balanced"
    if any(k in text for k in ("photo", "photoreal", "cinematic", "dslr", "realistic", "portrait")):
        return "photoreal"
    return "balanced"


"""Decompose a T2I prompt into subject / attire / style / pose / place facets.

Matches the handwritten agentic flow: separate reference searches and style
extraction per tag group (e.g. catgirl vs black lingerie vs eipril style).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

__all__ = [
    "PromptFacets",
    "decompose_prompt_facets",
    "facets_need_clarify",
]

_STYLE_RE = re.compile(
    r"(?:^|,)\s*((?:by|art by|drawn by|painted by|in the style of|style of)\s+[^,]+|"
    r"[^,]*\s+style)\s*(?=,|$)",
    re.IGNORECASE,
)
_ATTIRE_RE = re.compile(
    r"\b("
    r"lingerie|bikini|swimsuit|dress|skirt|shirt|blouse|jacket|coat|hoodie|"
    r"armor|uniform|kimono|hanfu|qipao|suit|tie|stockings|thighhighs|"
    r"gloves|boots|heels|jewelry|necklace|choker|collar|leotard|bodysuit|"
    r"nude|naked|topless|bottomless|clothing|outfit|wear|attire"
    r")\b",
    re.IGNORECASE,
)
_POSE_RE = re.compile(
    r"\b("
    r"standing|sitting|kneeling|lying|reclining|walking|running|jumping|"
    r"looking at viewer|from behind|from side|profile|cowboy shot|full body|"
    r"upper body|portrait|close-up|dynamic pose|contrapposto|arms up|"
    r"crossed arms|hand on hip|leaning|squatting|all fours|spread"
    r")\b",
    re.IGNORECASE,
)
_PLACE_RE = re.compile(
    r"\b("
    r"indoors|outdoors|bedroom|bathroom|kitchen|office|classroom|street|"
    r"park|forest|beach|ocean|city|cyberpunk|neon|cafe|bar|club|studio|"
    r"white background|black background|simple background|detailed background|"
    r"room|interior|exterior|sky|sunset|night|day"
    r")\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class PromptFacets:
    raw: str
    subject: list[str] = field(default_factory=list)
    attire: list[str] = field(default_factory=list)
    style: list[str] = field(default_factory=list)
    pose: list[str] = field(default_factory=list)
    place: list[str] = field(default_factory=list)
    nsfw: bool = False
    artist_hint: str = ""
    missing: list[str] = field(default_factory=list)
    search_queries: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _split_tags(prompt: str) -> list[str]:
    text = str(prompt or "").strip().strip("[]")
    if not text:
        return []
    if "," in text:
        return [t.strip() for t in text.split(",") if t.strip()]
    return [t.strip() for t in re.split(r"\s{2,}|\n+", text) if t.strip()]


def _is_nsfw(text: str) -> bool:
    return bool(
        re.search(
            r"\b(nsfw|nude|naked|lingerie|explicit|uncensored|xxx|porn|hentai)\b",
            text,
            re.IGNORECASE,
        )
    )


def decompose_prompt_facets(prompt: str) -> PromptFacets:
    """Heuristic facet split tuned for danbooru-style and short prose prompts."""
    raw = str(prompt or "").strip()
    facets = PromptFacets(raw=raw, nsfw=_is_nsfw(raw))
    tags = _split_tags(raw)

    for m in _STYLE_RE.finditer(raw):
        phrase = m.group(1).strip(" ,")
        if phrase and phrase.lower() not in {s.lower() for s in facets.style}:
            facets.style.append(phrase)

    artist = ""
    try:
        from config.defaults.style_artists import extract_style_from_text

        artist = str(extract_style_from_text(raw) or "").strip()
    except Exception:
        artist = ""
    if artist:
        facets.artist_hint = artist
        if artist.lower() not in {s.lower() for s in facets.style}:
            facets.style.append(f"{artist} style")

    for tag in tags:
        low = tag.lower()
        if any(s.lower() == low for s in facets.style):
            continue
        if "style" in low or low.startswith("by ") or "art by" in low:
            facets.style.append(tag)
            continue
        if _ATTIRE_RE.search(tag):
            facets.attire.append(tag)
            continue
        if _POSE_RE.search(tag):
            facets.pose.append(tag)
            continue
        if _PLACE_RE.search(tag):
            facets.place.append(tag)
            continue
        facets.subject.append(tag)

    cleaned: list[str] = []
    for t in facets.subject:
        if _ATTIRE_RE.search(t) or _POSE_RE.search(t) or _PLACE_RE.search(t):
            continue
        if "style" in t.lower():
            continue
        cleaned.append(t)
    facets.subject = cleaned or ([tags[0]] if tags else [])

    if not facets.pose:
        facets.missing.append("pose")
    if not facets.place:
        facets.missing.append("place")
    if not facets.style and not facets.artist_hint:
        facets.missing.append("style")

    subj = ", ".join(facets.subject[:6]) or "character"
    if facets.subject:
        facets.search_queries["subject"] = subj
    if facets.attire:
        attire = ", ".join(facets.attire[:4])
        facets.search_queries["attire"] = f"{subj}, {attire}"
        facets.search_queries["attire_only"] = attire
    if facets.style or facets.artist_hint:
        styl = facets.artist_hint or ", ".join(facets.style[:3])
        facets.search_queries["style"] = f"{styl} art style reference"
        facets.search_queries["artist"] = styl
    return facets


def facets_need_clarify(facets: PromptFacets) -> bool:
    return bool(facets.missing)

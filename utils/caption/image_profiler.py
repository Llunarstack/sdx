"""Multi-source image profiling — better than WD/JoyTag alone."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

_VLM_SAFE_EXTRA = frozenset(
    {
        "solo",
        "1girl",
        "1boy",
        "2girls",
        "2boys",
        "outdoors",
        "indoors",
        "sky",
        "night",
        "day",
        "rain",
        "snow",
        "sunset",
        "forest",
        "city",
        "beach",
        "school uniform",
        "long hair",
        "short hair",
        "looking at viewer",
        "smile",
        "standing",
        "sitting",
        "lying",
    }
)


@dataclass
class ImageProfile:
    caption: str
    scene_summary: str = ""
    character_tags: list[str] = field(default_factory=list)
    copyright_tags: list[str] = field(default_factory=list)
    artist_tags: list[str] = field(default_factory=list)
    extra_tags: list[str] = field(default_factory=list)
    is_original_character: bool = False
    confidence: float = 0.0
    sources: list[str] = field(default_factory=list)

    def to_manifest_patch(self) -> dict:
        out = {
            "caption": self.caption,
            "scene_summary": self.scene_summary,
            "tag_confidence": round(self.confidence, 4),
            "tag_sources": list(self.sources),
        }
        if self.character_tags:
            out["character_tags"] = self.character_tags
        if self.copyright_tags:
            out["copyright_tags"] = self.copyright_tags
        if self.artist_tags:
            out["artist_tags"] = self.artist_tags
        if self.is_original_character:
            out["is_original_character"] = True
        return out


def _norm_tag(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower().replace("_", " "))


def _merge_tags(*groups: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for g in groups:
        for t in g:
            n = _norm_tag(t)
            if not n or n in seen:
                continue
            seen.add(n)
            out.append(n)
    return out


def _build_caption(
    *,
    characters: Sequence[str],
    copyrights: Sequence[str],
    artists: Sequence[str],
    general: Sequence[str],
) -> str:
    parts = _merge_tags(characters, copyrights, artists, general)
    return ", ".join(parts)


def _summary_from_tags(
    *,
    characters: Sequence[str],
    copyrights: Sequence[str],
    artists: Sequence[str],
    caption: str,
) -> str:
    bits: list[str] = []
    if characters:
        bits.append(f"Characters: {', '.join(_merge_tags(characters))}.")
    if copyrights:
        bits.append(f"Series: {', '.join(_merge_tags(copyrights))}.")
    if artists:
        bits.append(f"Artist: {', '.join(_merge_tags(artists))}.")
    if caption:
        bits.append(f"Tags: {caption}.")
    return " ".join(bits).strip()


def _parse_row_lists(row: dict, key: str) -> list[str]:
    val = row.get(key)
    if not val:
        return []
    if isinstance(val, list):
        return [str(x) for x in val if str(x).strip()]
    return [t.strip() for t in str(val).split(",") if t.strip()]


def _vlm_scene_summary(image_path: Path, *, device: str = "cuda") -> tuple[str, list[str]]:
    prompt = (
        "Describe this image in detail: who is present, what are they doing, "
        "where is the scene, clothing, expression, composition, lighting, and mood. "
        "If the character looks like a known anime/game character, name them and the series. "
        "If it appears to be an original character, say so."
    )
    try:
        from utils.brain.understand import caption_image_vlm

        summary = caption_image_vlm(str(image_path), user_prompt=prompt, device=device)
    except Exception:
        summary = ""
    if not summary:
        return "", []

    extras: list[str] = []
    low = summary.lower()
    for tag in _VLM_SAFE_EXTRA:
        if tag in low:
            extras.append(tag)
    if re.search(r"\boriginal character\b|\boc\b|\boriginal design\b", low):
        extras.append("original_character")
    return summary, extras


def _apply_reverse_hit(hit, *, characters, copyrights, artists, sources, confidence):
    sources.append(f"{hit.engine}:{hit.similarity:.0f}%")
    confidence = max(confidence, min(0.99, hit.similarity / 100.0))
    if hit.characters:
        characters.extend(hit.characters)
    if hit.material:
        for part in re.split(r"[/|]", hit.material):
            part = part.strip()
            if part and len(part) > 2:
                copyrights.append(part)
    if hit.author and not artists:
        artists.append(hit.author)
    if hit.site == "danbooru" and hit.site_id:
        try:
            from utils.caption.danbooru_lookup import fetch_danbooru_post

            db = fetch_danbooru_post(hit.site_id)
            if db:
                sources.append(f"danbooru_api:{hit.site_id}")
                confidence = max(confidence, 0.98)
                characters.extend(db.character_tags)
                copyrights.extend(db.copyright_tags)
                artists.extend(db.artist_tags)
        except Exception:
            pass
    elif hit.site == "e621" and hit.site_id:
        try:
            from utils.caption.e621_lookup import fetch_e621_post

            e6 = fetch_e621_post(hit.site_id)
            if e6:
                sources.append(f"e621_api:{hit.site_id}")
                confidence = max(confidence, 0.98)
                characters.extend(e6.character_tags)
                copyrights.extend(e6.copyright_tags)
                artists.extend(e6.artist_tags)
        except Exception:
            pass
    return confidence


def profile_image(
    image_path: str | Path,
    *,
    booru_row: dict | None = None,
    use_reverse_search: bool = True,
    use_saucenao: bool = True,
    use_tineye: bool = True,
    use_vlm: bool = True,
    reverse_min_sim: float = 75.0,
    device: str = "cuda",
) -> ImageProfile:
    """Fuse metadata + reverse search + VLM into one training caption."""
    p = Path(image_path)
    has_file = p.is_file()
    row = booru_row or {}

    sources: list[str] = []
    characters = _parse_row_lists(row, "character_tags")
    copyrights = _parse_row_lists(row, "copyright_tags")
    artists = _parse_row_lists(row, "artist_tags")
    general: list[str] = []
    scene_summary = str(row.get("scene_summary") or "").strip()
    confidence = float(row.get("tag_confidence") or 0.0)

    if characters:
        sources.append("booru_character")
        confidence = max(confidence, 0.95)
    if copyrights:
        sources.append("booru_copyright")
        confidence = max(confidence, 0.95)
    if artists:
        sources.append("booru_artist")
        confidence = max(confidence, 0.9)

    raw_caption = str(row.get("caption") or "").strip()
    if raw_caption:
        if not characters and not copyrights:
            general.extend(t.strip() for t in raw_caption.split(",") if t.strip())
            sources.append("booru_caption")
            confidence = max(confidence, 0.85)

    if use_reverse_search and has_file:
        from utils.caption.reverse_search import hit_meets_threshold, reverse_search_file

        for hit in reverse_search_file(p, use_saucenao=use_saucenao, use_tineye=use_tineye):
            if not hit_meets_threshold(hit, reverse_min_sim):
                continue
            confidence = _apply_reverse_hit(
                hit,
                characters=characters,
                copyrights=copyrights,
                artists=artists,
                sources=sources,
                confidence=confidence,
            )
            if characters or copyrights:
                break

    vlm_extras: list[str] = []
    if use_vlm and has_file:
        vlm_summary, vlm_extras = _vlm_scene_summary(p, device=device)
        if vlm_summary:
            scene_summary = vlm_summary
            sources.append("vlm_dense")
            confidence = max(confidence, 0.55)
        general.extend(vlm_extras)

    if not scene_summary:
        scene_summary = _summary_from_tags(
            characters=characters,
            copyrights=copyrights,
            artists=artists,
            caption=raw_caption,
        )
        if scene_summary:
            sources.append("tag_summary")

    is_oc = not characters and not copyrights
    if is_oc:
        if "original_character" in general:
            is_oc = True
        if scene_summary and re.search(r"\boriginal character\b|\boc\b|\bunknown character\b", scene_summary, re.I):
            is_oc = True
    else:
        general = [t for t in general if t != "original_character"]

    if is_oc and "original_character" not in general:
        general.insert(0, "original_character")

    caption = _build_caption(
        characters=characters,
        copyrights=copyrights,
        artists=artists,
        general=general if general else [t.strip() for t in raw_caption.split(",") if t.strip()],
    )
    if not caption and raw_caption:
        caption = raw_caption

    return ImageProfile(
        caption=caption,
        scene_summary=scene_summary,
        character_tags=_merge_tags(characters),
        copyright_tags=_merge_tags(copyrights),
        artist_tags=_merge_tags(artists),
        extra_tags=_merge_tags(general),
        is_original_character=is_oc,
        confidence=confidence,
        sources=list(dict.fromkeys(sources)),
    )


def profile_from_manifest_row(row: dict, data_root: Path, **kwargs) -> ImageProfile:
    rel = row.get("image_path") or ""
    img = data_root / rel if rel else Path("_missing")
    if not img.is_file():
        img = Path(rel) if rel else Path("_missing")
    has_identity = bool(row.get("character_tags") or row.get("copyright_tags"))
    return profile_image(
        img,
        booru_row=row,
        use_reverse_search=kwargs.get("use_reverse_search", True) and not has_identity,
        use_saucenao=kwargs.get("use_saucenao", True),
        use_tineye=kwargs.get("use_tineye", True),
        use_vlm=kwargs.get("use_vlm", True) and img.is_file(),
        device=kwargs.get("device", "cuda"),
        reverse_min_sim=float(kwargs.get("reverse_min_sim", kwargs.get("saucenao_min_sim", 75.0))),
    )

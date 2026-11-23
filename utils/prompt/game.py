"""Game imagery: which asset a prompt asks for, the conventions that asset needs, and its finishing defaults."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from utils.prompt.text import append_unique

_PIXEL_RX = re.compile(
    r"\bpixel[\s-]?art\b|\bpixelart\b|\b(?:1|2|4|8|16|32)[\s-]?bit\b|\bpixel(?:ated)? (?:sprite|game|character|tileset|style)"
)
_GRID_RX = re.compile(r"\b(8|12|16|24|32|48|64|96|128)\s*[x×]\s*\1\b")


@dataclass(frozen=True)
class _Asset:
    name: str
    rx: re.Pattern[str]
    positive: tuple[str, ...]
    negative: tuple[str, ...]
    transparent_bg: bool = False
    tileable: bool = False
    critic: str = ""  # "", "seam", "icon"


def _a(name: str, pattern: str, pos: tuple[str, ...], neg: tuple[str, ...], **kw) -> _Asset:
    return _Asset(name, re.compile(pattern), pos, neg, **kw)


# First match wins, so specific assets come before broad looks (low poly, toon).
_ASSETS: tuple[_Asset, ...] = (
    _a(
        "sprite_sheet",
        r"sprite ?sheet|animation frames|(?:walk|run|attack|idle) (?:cycle|animation)",
        (
            "sprite sheet",
            "evenly spaced grid of frames",
            "same character size and framing in every frame",
            "consistent palette across frames",
            "plain flat background",
        ),
        ("uneven spacing", "overlapping frames", "inconsistent character design", "cropped frames", "perspective"),
        transparent_bg=True,
    ),
    _a(
        "tileset",
        r"tile ?set|tile ?map|tileable|seamless(?:\s+[\w-]+){0,3}\s+(?:texture|tiles?|pattern)|game texture|terrain tiles?",
        ("seamless tileable texture", "even flat lighting", "orthographic top view", "uniform detail density"),
        ("visible seams", "vignette", "perspective", "lighting gradient", "single focal object", "border"),
        tileable=True,
        critic="seam",
    ),
    _a(
        "icon",
        r"\b(?:item|inventory|skill|ability|spell|loot|ui|game|potion|weapon) icons?\b|icon set",
        (
            "single centered object",
            "bold readable silhouette",
            "clear at small size",
            "strong rim light",
            "plain background",
        ),
        ("cluttered background", "tiny details", "cropped object", "text", "multiple objects"),
        transparent_bg=True,
        critic="icon",
    ),
    _a(
        "ui_hud",
        r"\bhud\b|game ui\b|game interface|health bar|inventory screen|main menu|menu screen|minimap|skill tree",
        (
            "clean aligned UI panels",
            "consistent margins",
            "legible labels",
            "clear visual hierarchy",
            "UI framing the screen edges",
        ),
        ("garbled text", "misaligned panels", "overlapping widgets", "illegible lettering", "random symbols"),
    ),
    _a(
        "character_sheet",
        r"character (?:sheet|turnaround|reference sheet|design sheet)|turnaround|model sheet|front,? side,? (?:and )?back",
        (
            "character turnaround",
            "front view, side view, back view",
            "identical design in every view",
            "neutral standing pose",
            "orthographic views",
            "plain background",
        ),
        ("different outfits between views", "inconsistent proportions", "perspective distortion", "dynamic pose"),
    ),
    _a(
        "in_engine",
        r"in[\s-]?game screenshot|gameplay screenshot|game screenshot|in[\s-]?engine|(?:first|third)[\s-]person "
        r"(?:shooter|view|game)|open[\s-]world game|\bgameplay\b|video game (?:scene|still)",
        (
            "real-time rendered game graphics",
            "physically based materials",
            "consistent world scale",
            "dynamic shadows",
            "natural game camera field of view",
            "atmospheric depth",
        ),
        ("pre-rendered cinematic look", "warped architecture", "floating objects", "garbled UI text", "deformed hands"),
    ),
    _a(
        "key_art",
        r"key art|cover art|box art|game poster|splash art|loading screen",
        (
            "dramatic hero composition",
            "clear focal character",
            "readable silhouette",
            "open space for a title",
            "cinematic lighting",
        ),
        ("cluttered composition", "flat lighting", "cropped head", "random text", "watermark"),
    ),
    _a(
        "card_art",
        r"(?:trading|collectible) card|card game art|card illustration|\btcg\b",
        ("full-bleed card illustration", "central subject", "dynamic action", "controlled vivid palette"),
        ("card frame", "rules text", "borders", "watermark"),
    ),
    _a(
        "game_map",
        r"(?:world|game|dungeon|level|overworld) map|top[\s-]down (?:view|game|rpg|map)|side[\s-]?scroller|platformer level",
        ("one consistent orthographic view", "readable walkable paths", "grid aligned layout", "clear landmarks"),
        ("mixed perspectives", "perspective distortion", "unreadable paths"),
    ),
    _a(
        "environment_concept",
        r"environment (?:concept|art|design)|level (?:design|concept)|game (?:environment|level)",
        (
            "environment concept art",
            "clear foreground, midground and background",
            "atmospheric perspective",
            "human figure for scale",
            "readable landmarks",
        ),
        ("flat depth", "muddy values", "no sense of scale"),
    ),
    _a(
        "isometric",
        r"isometric|\biso view\b|diorama",
        ("true isometric projection", "parallel edges", "no vanishing point", "orthographic camera"),
        ("perspective", "fisheye", "vanishing point", "tilted horizon"),
    ),
    _a(
        "voxel",
        r"\bvoxel",
        ("voxel art", "uniform cube grid", "flat color per voxel", "orthographic camera"),
        ("smooth surfaces", "irregular cube sizes", "noisy textures"),
    ),
    _a(
        "low_poly",
        r"low[\s-]?poly",
        ("low poly 3d", "flat shaded faceted polygons", "clean geometric silhouettes", "limited palette"),
        ("smooth subdivision", "high poly detail", "noisy textures"),
    ),
    _a(
        "toon",
        r"(?:cel|toon)[\s-]?shad",
        ("cel shaded", "hard light and shadow bands", "clean ink outlines", "flat color fills"),
        ("soft gradients", "photoreal textures"),
    ),
)

_PIXEL_POS = (
    "pixel art",
    "crisp hard-edged pixels",
    "limited color palette",
    "clean 1-pixel outlines",
    "deliberate dithering only",
)
_PIXEL_NEG = ("anti-aliasing", "blurry", "smooth gradients", "mixed pixel sizes", "jpeg artifacts", "sub-pixel detail")


@dataclass(frozen=True)
class GameProfile:
    asset: str  # one of the _ASSETS names, or "pixel_art" / "sprite"
    pixel_art: bool = False
    grid: int = 0  # requested sprite resolution ("32x32"); 0 = estimate the pixel size from the image
    palette_colors: int = 0  # 0 = keep colours
    transparent_bg: bool = False
    tileable: bool = False
    critic: str = ""
    positive: tuple[str, ...] = field(default_factory=tuple)
    negative: tuple[str, ...] = field(default_factory=tuple)


def _palette_for(p: str) -> int:
    if re.search(r"\b1[\s-]?bit\b|monochrome pixel", p):
        return 2
    if re.search(r"\b2[\s-]?bit\b|\b4[\s-]?colou?rs?\b|game ?boy", p):
        return 4
    if re.search(r"\b8[\s-]?bit\b|\bnes\b", p):
        return 16
    m = re.search(r"\b(\d{1,3})[\s-]?colou?r palette\b", p)
    if m:
        return max(2, min(256, int(m.group(1))))
    return 32


def detect_game_profile(prompt: str) -> GameProfile | None:
    """The game asset a prompt asks for, or None when it isn't game imagery."""
    p = (prompt or "").lower()
    pixel = bool(_PIXEL_RX.search(p))
    asset = next((a for a in _ASSETS if a.rx.search(p)), None)
    if asset is None and not pixel:
        if not re.search(r"\bsprite\b", p):
            return None
        asset = _Asset(
            "sprite",
            re.compile(""),
            ("2d game sprite", "single character", "full body in frame", "plain flat background"),
            ("cropped", "busy background", "perspective"),
            transparent_bg=True,
        )
    grid_m = _GRID_RX.search(p)
    pos = (asset.positive if asset else ()) + (_PIXEL_POS if pixel else ())
    neg = (asset.negative if asset else ()) + (_PIXEL_NEG if pixel else ())
    return GameProfile(
        asset=asset.name if asset else "pixel_art",
        pixel_art=pixel,
        grid=int(grid_m.group(1)) if (pixel and grid_m) else 0,
        palette_colors=_palette_for(p) if pixel else 0,
        transparent_bg=bool(asset and asset.transparent_bg)
        or bool(re.search(r"transparent background|no background", p)),
        tileable=bool(asset and asset.tileable),
        critic="pixel" if pixel else (asset.critic if asset else ""),
        positive=pos,
        negative=neg,
    )


def resolve_game_profile(args) -> GameProfile | None:
    """detect_game_profile on the user's own prompt, with --game-asset / --pixel-size / --palette-colors overrides."""
    mode = str(getattr(args, "game_asset", "auto") or "auto").lower()
    if mode == "off":
        return None
    raw = str(getattr(args, "_raw_prompt_before_compose", "") or getattr(args, "prompt", "") or "")
    prof = detect_game_profile(raw)
    if mode not in ("auto", "") and (prof is None or prof.asset != mode):
        forced = detect_game_profile(f"{raw}, {mode.replace('_', ' ')}") or prof
        prof = forced if forced is not None else GameProfile(asset=mode)
    if prof is None:
        return None
    palette = int(getattr(args, "palette_colors", -1))
    tbg = str(getattr(args, "transparent_bg", "auto") or "auto").lower()
    return GameProfile(
        asset=prof.asset,
        pixel_art=prof.pixel_art or int(getattr(args, "pixel_size", 0) or 0) > 0,
        grid=prof.grid,
        palette_colors=prof.palette_colors if palette < 0 else palette,
        transparent_bg=prof.transparent_bg if tbg == "auto" else tbg == "on",
        tileable=prof.tileable,
        critic=prof.critic,
        positive=prof.positive,
        negative=prof.negative,
    )


def apply_game_prompt(positive: str, negative: str, prof: GameProfile) -> tuple[str, str]:
    return append_unique(positive, prof.positive), append_unique(negative, prof.negative)

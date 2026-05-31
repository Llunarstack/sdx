"""
**Human-made** polish — reduce obvious AI tells in prompts and post-process.

Targets common diffusion artifacts: plastic skin, AI speckles, uniform smoothness,
perfect symmetry, haloed edges, and sterile digital color. Complements ``--naturalize``,
``--less-ai``, and ``utils/quality/artistic_post_process``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Prompt fragments (training-free steering)
# ---------------------------------------------------------------------------

_HUMAN_POS_LITE = (
    "authentic human-made capture, natural skin texture with pores, subtle imperfections, "
    "organic lighting falloff, believable material response, not overprocessed"
)
_HUMAN_NEG_LITE = (
    "ai generated, plastic skin, waxy face, airbrushed, oversmoothed, doll-like, synthetic, "
    "cgi render, uncanny valley, perfect symmetry, ai speckles, white dots, spiky artifacts, "
    "watercolor bleed on photo, jpeg mush, oversaturated hdr"
)
_HUMAN_NEG_STRONG = (
    _HUMAN_NEG_LITE + ", hyperreal plastic, beauty filter, frequency separation seam, ai upscaler halos, "
    "melting fingers, extra limbs, garbled text, floating objects"
)
_HUMAN_POS_STRONG = (
    _HUMAN_POS_LITE + ", shot on real camera, subtle film grain, natural shadow noise, "
    "slight lens character, imperfect but intentional composition"
)


@dataclass(slots=True)
class HumanMadeConfig:
    """Post-process + implicit prompt strength."""

    strength: float = 0.55
    remove_speckles: float = 0.45
    reduce_plastic: float = 0.35
    organic_grain: float = 0.018
    local_tone_jitter: float = 0.04
    halo_reduction: float = 0.25
    asymmetry: float = 0.08
    lost_found_edges: float = 0.12
    value_shadow_lift: float = 0.04
    value_highlight_roll: float = 0.05
    chromatic_aberration: float = 0.06
    vignette: float = 0.08
    micro_detail: float = 0.15
    sss_skin: float = 0.12
    seed: int = 0


def human_made_preset(name: str) -> HumanMadeConfig:
    """``lite`` | ``standard`` | ``strong`` presets."""
    n = str(name or "standard").lower().strip()
    if n == "lite":
        return HumanMadeConfig(strength=0.35, remove_speckles=0.25, reduce_plastic=0.2, organic_grain=0.012)
    if n == "strong":
        return HumanMadeConfig(
            strength=0.85,
            remove_speckles=0.65,
            reduce_plastic=0.55,
            organic_grain=0.024,
            local_tone_jitter=0.07,
            halo_reduction=0.4,
            asymmetry=0.14,
            lost_found_edges=0.2,
            value_shadow_lift=0.06,
            value_highlight_roll=0.08,
            chromatic_aberration=0.1,
            vignette=0.14,
            micro_detail=0.22,
            sss_skin=0.18,
        )
    return HumanMadeConfig()


def human_made_prompt_fragments(preset: str = "standard") -> Tuple[str, str]:
    """Return (positive_addon, negative_addon)."""
    n = str(preset or "standard").lower().strip()
    if n in ("none", "off", "0"):
        return "", ""
    if n == "strong":
        return _HUMAN_POS_STRONG, _HUMAN_NEG_STRONG
    if n == "lite":
        return _HUMAN_POS_LITE, _HUMAN_NEG_LITE
    return _HUMAN_POS_LITE, _HUMAN_NEG_LITE


def apply_human_made_prompt_flags(args: Any) -> None:
    """
    Mutate ``args`` for sampling: enable ``--less-ai``, ``--naturalize``, anti-ai packs
    when ``--human-made`` is set.
    """
    preset = str(getattr(args, "human_made", "none") or "none").lower().strip()
    if preset in ("none", "off", "0", ""):
        return
    if str(getattr(args, "anti_ai_pack", "none") or "none") == "none":
        args.anti_ai_pack = "lite" if preset == "lite" else "strong" if preset == "strong" else "lite"
    if str(getattr(args, "human_media_mode", "none") or "none") == "none":
        args.human_media_mode = "photographic"
    if not getattr(args, "naturalize", False):
        args.naturalize = True
    if float(getattr(args, "naturalize_grain", 0.0) or 0.0) <= 0.0:
        args.naturalize_grain = 0.012 if preset == "lite" else 0.022 if preset == "strong" else 0.016
    if preset in ("standard", "strong") and str(getattr(args, "shortcomings_mitigation", "none") or "none") == "none":
        args.shortcomings_mitigation = "auto"
    if not getattr(args, "less_ai", False):
        args.less_ai = True


def append_human_made_prompt_fragments(prompt: str, negative: str, preset: str) -> Tuple[str, str]:
    """Append human-made pos/neg fragments (call after base prompt is assembled)."""
    pos, neg = human_made_prompt_fragments(preset)
    p = f"{pos}, {prompt}".strip(", ") if pos and prompt else (pos or prompt)
    n = f"{negative}, {neg}".strip(", ") if neg and negative else (neg or negative)
    return p.strip(), n.strip()


def _to_float(img: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(img, dtype=np.float32), 0.0, 255.0)


def _from_float(img: np.ndarray, was_uint: bool) -> np.ndarray:
    out = np.clip(img, 0.0, 255.0)
    return out.astype(np.uint8) if was_uint else out


def remove_ai_speckles(image: np.ndarray, *, strength: float = 0.4) -> np.ndarray:
    """
    Median-blend isolated high-frequency sparkle artifacts (common AI tell).
    """
    s = float(max(0.0, min(1.0, strength)))
    if s <= 0.0:
        return image
    try:
        from scipy.ndimage import median_filter, uniform_filter
    except ImportError:
        return image
    was_uint = image.dtype == np.uint8
    img = _to_float(image)
    gray = img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114
    local_mean = uniform_filter(gray, size=3)
    local_sq = uniform_filter(gray * gray, size=3)
    var = np.clip(local_sq - local_mean * local_mean, 0.0, None)
    # Speckles: high local variance on small scale vs neighbors
    speckle_mask = (var > np.percentile(var, 92)).astype(np.float32)
    speckle_mask = uniform_filter(speckle_mask, size=2)
    med = median_filter(img, size=(3, 3, 1))
    blend = (s * 0.65) * speckle_mask[..., np.newaxis]
    out = img * (1.0 - blend) + med * blend
    return _from_float(out, was_uint)


def reduce_plastic_smoothness(image: np.ndarray, *, strength: float = 0.35) -> np.ndarray:
    """
    Soften waxy low-texture regions while preserving edges (AI oversmooth tell).
    """
    s = float(max(0.0, min(1.0, strength)))
    if s <= 0.0:
        return image
    from PIL import Image, ImageFilter

    was_uint = image.dtype == np.uint8
    img = _to_float(image)
    gray = img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114
    # Low local variance = oversmooth plastic regions
    try:
        from scipy.ndimage import uniform_filter

        local_mean = uniform_filter(gray, size=5)
        local_sq = uniform_filter(gray * gray, size=5)
        var = np.clip(local_sq - local_mean * local_mean, 0.0, None)
        smooth_mask = np.clip(1.0 - var / (var.max() + 1e-6), 0.0, 1.0) ** 1.5
    except ImportError:
        smooth_mask = np.ones_like(gray) * 0.3
    pil = Image.fromarray(img.astype(np.uint8))
    soft = np.array(pil.filter(ImageFilter.GaussianBlur(radius=1.2)), dtype=np.float32)
    blend = (s * 0.5) * smooth_mask[..., np.newaxis]
    out = img * (1.0 - blend) + soft * blend
    return _from_float(out, was_uint)


def local_tone_jitter(image: np.ndarray, *, strength: float = 0.04, seed: int = 0) -> np.ndarray:
    """Break uniform AI lighting with subtle tile-wise luminance variation."""
    s = float(max(0.0, min(1.0, strength)))
    if s <= 0.0:
        return image
    was_uint = image.dtype == np.uint8
    img = _to_float(image)
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    th, tw = max(4, h // 32), max(4, w // 32)
    tiles = rng.normal(0.0, s * 18.0, (th, tw)).astype(np.float32)
    from PIL import Image

    tile_img = Image.fromarray(tiles, mode="F")
    jitter = np.array(tile_img.resize((w, h), Image.BILINEAR), dtype=np.float32)
    lum = img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114
    scale = (lum + jitter) / (lum + 1e-6)
    out = np.clip(img * scale[..., np.newaxis], 0.0, 255.0)
    return _from_float(out, was_uint)


def reduce_edge_halos(image: np.ndarray, *, strength: float = 0.25) -> np.ndarray:
    """Attenuate sharpening halos / white rims around high-contrast edges."""
    s = float(max(0.0, min(1.0, strength)))
    if s <= 0.0:
        return image
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        return image
    was_uint = image.dtype == np.uint8
    img = _to_float(image)
    blur = gaussian_filter(img, sigma=1.1, mode="reflect")
    high = img - blur
    # Halos often show as bright rims on edges
    rim = np.abs(high).mean(axis=2)
    rim_mask = np.clip((rim - np.percentile(rim, 75)) / (rim.max() + 1e-6), 0.0, 1.0)
    corrected = img - s * 0.6 * high * rim_mask[..., np.newaxis]
    return _from_float(corrected, was_uint)


def apply_human_made_pipeline(image: np.ndarray, config: HumanMadeConfig | None = None) -> np.ndarray:
    """Full human-made post-process chain."""
    cfg = config or HumanMadeConfig()
    st = float(max(0.0, min(1.0, cfg.strength)))
    if st <= 0.0:
        return image
    out = image
    if cfg.remove_speckles > 0:
        out = remove_ai_speckles(out, strength=cfg.remove_speckles * st)
    if cfg.reduce_plastic > 0:
        out = reduce_plastic_smoothness(out, strength=cfg.reduce_plastic * st)
    if cfg.halo_reduction > 0:
        out = reduce_edge_halos(out, strength=cfg.halo_reduction * st)
    if cfg.local_tone_jitter > 0:
        out = local_tone_jitter(out, strength=cfg.local_tone_jitter * st, seed=int(cfg.seed))
    if cfg.organic_grain > 0:
        from utils.quality.quality import add_film_grain, naturalize

        out = add_film_grain(out, amount=float(cfg.organic_grain) * st, seed=int(cfg.seed))
        out = naturalize(out, grain_amount=0.0, micro_contrast=1.0 + 0.02 * st, seed=int(cfg.seed))
    try:
        from utils.quality.artistic_post_process import ArtisticPostConfig, apply_artistic_pipeline

        art = ArtisticPostConfig(
            value_structure=cfg.value_shadow_lift > 0 or cfg.value_highlight_roll > 0,
            value_shadow_lift=float(cfg.value_shadow_lift) * st,
            value_highlight_roll=float(cfg.value_highlight_roll) * st,
            asymmetry_strength=float(cfg.asymmetry) * st,
            asymmetry_seed=int(cfg.seed),
            lost_found_strength=float(cfg.lost_found_edges) * st,
            lost_found_seed=int(cfg.seed) + 17,
            sss_strength=float(cfg.sss_skin) * st,
            sss_radius=3.0,
            chromatic_aberration=float(cfg.chromatic_aberration) * st,
            vignette_strength=float(cfg.vignette) * st,
            micro_detail=float(cfg.micro_detail) * st,
        )
        out = apply_artistic_pipeline(out, art)
    except Exception:
        pass
    return out


def config_from_args(args: Any) -> HumanMadeConfig | None:
    """Build config from ``sample.py`` args; None if disabled."""
    preset = str(getattr(args, "human_made", "none") or "none").lower().strip()
    if preset in ("none", "off", "0", ""):
        return None
    cfg = human_made_preset(preset if preset in ("lite", "standard", "strong") else "standard")
    override = float(getattr(args, "human_made_strength", -1.0) or -1.0)
    if override >= 0.0:
        cfg.strength = float(max(0.0, min(1.0, override)))
    cfg.seed = int(getattr(args, "seed", 0) or 0)
    return cfg


__all__ = [
    "HumanMadeConfig",
    "apply_human_made_pipeline",
    "append_human_made_prompt_fragments",
    "apply_human_made_prompt_flags",
    "config_from_args",
    "human_made_preset",
    "human_made_prompt_fragments",
    "local_tone_jitter",
    "reduce_edge_halos",
    "reduce_plastic_smoothness",
    "remove_ai_speckles",
]

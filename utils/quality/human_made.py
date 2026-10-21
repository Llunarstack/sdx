"""
**Human-made** polish: anti-AI prompt fragments plus a deterministic cleanup pass.

The post pass only removes sharpening halos and sets value structure. Capture texture
(sensor noise, film grain, lens) is added once, after best-of-N selection, by the
medium-routed finish in ``utils.quality.camera_isp``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

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
    "melting fingers, extra limbs, garbled text, floating objects, teal and orange grade, "
    "mirror-symmetric face, identical catchlights, clipped hdr highlights, dead-center framing"
)
_HUMAN_POS_STRONG = (
    _HUMAN_POS_LITE + ", shot on real camera, slight lens character, "
    "imperfect but intentional composition, uneven catchlights, "
    "mixed white balance, soft highlight rolloff"
)


@dataclass(slots=True)
class HumanMadeConfig:
    """Strength of the deterministic post pass (prompt fragments come from the preset name)."""

    strength: float = 0.55
    halo_reduction: float = 0.25
    value_shadow_lift: float = 0.04
    value_highlight_roll: float = 0.05


def human_made_preset(name: str) -> HumanMadeConfig:
    """``lite`` | ``standard`` | ``strong`` presets."""
    n = str(name or "standard").lower().strip()
    if n == "lite":
        return HumanMadeConfig(strength=0.35)
    if n == "strong":
        return HumanMadeConfig(strength=0.85, halo_reduction=0.4, value_shadow_lift=0.06, value_highlight_roll=0.08)
    return HumanMadeConfig()


def human_made_prompt_fragments(preset: str = "standard") -> tuple[str, str]:
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
    if preset in ("standard", "strong") and str(getattr(args, "shortcomings_mitigation", "none") or "none") == "none":
        args.shortcomings_mitigation = "auto"
    if not getattr(args, "less_ai", False):
        args.less_ai = True


def append_human_made_prompt_fragments(prompt: str, negative: str, preset: str) -> tuple[str, str]:
    """Append human-made pos/neg fragments after the user's text, so encoder truncation cuts ours first."""
    pos, neg = human_made_prompt_fragments(preset)
    p = f"{prompt}, {pos}".strip(", ") if pos and prompt else (pos or prompt)
    n = f"{negative}, {neg}".strip(", ") if neg and negative else (neg or negative)
    return p.strip(), n.strip()


def reduce_edge_halos(image: np.ndarray, *, strength: float = 0.25) -> np.ndarray:
    """Attenuate sharpening halos / white rims around high-contrast edges."""
    s = float(max(0.0, min(1.0, strength)))
    if s <= 0.0:
        return image
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        return image
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 255.0)
    blur = gaussian_filter(img, sigma=1.1, mode="reflect")
    high = img - blur
    rim = np.abs(high).mean(axis=2)
    rim_mask = np.clip((rim - np.percentile(rim, 75)) / (rim.max() + 1e-6), 0.0, 1.0)
    corrected = np.clip(img - s * 0.6 * high * rim_mask[..., np.newaxis], 0.0, 255.0)
    return np.rint(corrected).astype(np.uint8) if image.dtype == np.uint8 else corrected


def apply_human_made_pipeline(image: np.ndarray, config: HumanMadeConfig | None = None) -> np.ndarray:
    """Deterministic post pass: halo reduction, then shadow lift / highlight roll-off."""
    cfg = config or HumanMadeConfig()
    st = float(max(0.0, min(1.0, cfg.strength)))
    if st <= 0.0:
        return image
    out = reduce_edge_halos(image, strength=cfg.halo_reduction * st)
    from utils.quality.artistic_post_process import apply_value_structure

    return apply_value_structure(
        out, shadow_lift=cfg.value_shadow_lift * st, highlight_roll=cfg.value_highlight_roll * st
    )


def config_from_args(args: Any) -> HumanMadeConfig | None:
    """Build config from ``sample.py`` args; None if disabled."""
    preset = str(getattr(args, "human_made", "none") or "none").lower().strip()
    if preset in ("none", "off", "0", ""):
        return None
    cfg = human_made_preset(preset if preset in ("lite", "standard", "strong") else "standard")
    override = float(getattr(args, "human_made_strength", -1.0) or -1.0)
    if override >= 0.0:
        cfg.strength = float(max(0.0, min(1.0, override)))
    return cfg


__all__ = [
    "HumanMadeConfig",
    "apply_human_made_pipeline",
    "append_human_made_prompt_fragments",
    "apply_human_made_prompt_flags",
    "config_from_args",
    "human_made_preset",
    "human_made_prompt_fragments",
    "reduce_edge_halos",
]

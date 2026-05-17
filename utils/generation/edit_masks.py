"""
Heuristic **inpaint masks** (grayscale: white = regenerate, black = preserve) when no user mask exists.

These are layout priors only—good for refinement loops (“fix the face”, “redo the background”)
until you plug in SAM/detector-driven masks.

Regions match :class:`~utils.generation.iterative_refinement.EditRouter` inpaint hints:
``face``, ``hands``, ``clothing``, ``background``, ``subject`` (generic center mass), ``full``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

from PIL import Image, ImageDraw, ImageFilter

__all__ = [
    "HEURISTIC_EDIT_REGIONS",
    "heuristic_inpaint_mask",
    "normalize_heuristic_region",
    "save_heuristic_mask",
]


HEURISTIC_EDIT_REGIONS = frozenset({"face", "hands", "clothing", "background", "subject", "full"})


def normalize_heuristic_region(region: Optional[str]) -> str:
    """Return a known region token; unrecognized values become ``subject``."""
    key = (region or "subject").strip().lower()
    return key if key in HEURISTIC_EDIT_REGIONS else "subject"


def heuristic_inpaint_mask(
    width: int,
    height: int,
    region: Optional[str],
    *,
    feather_radius: float = 5.0,
) -> Image.Image:
    """
    Build an ``L``-mode mask. **White** pixels are rewritten; **black** pixels are frozen (MDM inpaint).

    Unknown region strings fall back to ``subject``.
    """
    w, h = max(4, int(width)), max(4, int(height))
    rkey = normalize_heuristic_region(region)
    if rkey == "full":
        m = Image.new("L", (w, h), 255)
    elif rkey == "background":
        m = Image.new("L", (w, h), 255)
        draw = ImageDraw.Draw(m)
        sx0, sy0, sx1, sy1 = _subject_ellipse_bbox(w, h)
        draw.ellipse([sx0, sy0, sx1, sy1], fill=0)
    else:
        m = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(m)
        if rkey == "face":
            cx, cy = w / 2, h * 0.30
            rx, ry = w * 0.24, h * 0.20
            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
        elif rkey == "hands":
            draw.rectangle([int(w * 0.10), int(h * 0.56), int(w * 0.90), int(h * 0.98)], fill=255)
        elif rkey == "clothing":
            cx, cy = w / 2, h * 0.48
            rx, ry = w * 0.26, h * 0.40
            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
        elif rkey == "subject":
            sx0, sy0, sx1, sy1 = _subject_ellipse_bbox(w, h)
            draw.ellipse([sx0, sy0, sx1, sy1], fill=255)

    if feather_radius and feather_radius > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=float(feather_radius)))
    return m


def _subject_ellipse_bbox(w: int, h: int) -> Tuple[float, float, float, float]:
    cx, cy = w / 2, h * 0.45
    rx, ry = w * 0.34, h * 0.48
    return cx - rx, cy - ry, cx + rx, cy + ry


def save_heuristic_mask(
    path: Union[str, Path],
    width: int,
    height: int,
    region: Optional[str],
    *,
    feather_radius: float = 5.0,
) -> Path:
    """Write a heuristic mask PNG next to editable assets."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    heuristic_inpaint_mask(width, height, region, feather_radius=feather_radius).save(p, format="PNG")
    return p

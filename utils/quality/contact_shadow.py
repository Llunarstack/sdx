"""Still-image contact-shadow scoring and repair (anti-float heuristic)."""

from __future__ import annotations

import numpy as np

__all__ = ["score_contact_shadow", "apply_contact_shadow", "prompt_wants_ground_contact"]

_GROUND_POSITIVE = (
    "standing",
    "full body",
    "full-body",
    "fullbody",
    "person",
    "people",
    "woman",
    "man",
    "girl",
    "boy",
    "shoes",
    "feet",
    "foot",
    "boots",
    "sneakers",
    "street",
    "floor",
    "pavement",
    "sidewalk",
    "ground",
    "walking",
    "on the floor",
)

_GROUND_NEGATIVE = (
    "aerial",
    "bird's eye",
    "birds eye",
    "from above",
    "overhead",
    "drone view",
    "macro",
    "close-up",
    "close up",
    "closeup",
    "extreme close",
)


def _foot_band(rgb: np.ndarray) -> tuple[np.ndarray, int, int]:
    h = rgb.shape[0]
    y0 = int(h * 0.78)
    return rgb[y0:, :], y0, h


def _subject_mask_lower(rgb: np.ndarray) -> np.ndarray:
    band, _, _ = _foot_band(rgb)
    gray = band.astype(np.float32).mean(axis=2)
    med = float(np.median(gray))
    return (np.abs(gray - med) > 18.0).astype(np.float32)


def score_contact_shadow(rgb: np.ndarray) -> float:
    """Return 0–1 score; higher means contact ambient occlusion is present."""
    band, _, _ = _foot_band(rgb)
    gray = band.astype(np.float32).mean(axis=2)
    mask = _subject_mask_lower(rgb)
    if mask.mean() < 0.02:
        return 0.7

    bh = gray.shape[0]
    cols = np.where(mask.max(axis=0) > 0.5)[0]
    if cols.size == 0:
        return 0.7

    aos: list[float] = []
    step = max(1, len(cols) // 8)
    for c in cols[::step]:
        rows = np.where(mask[:, c] > 0.5)[0]
        if rows.size == 0:
            continue
        tip = int(rows.max())
        under = tip + 1
        far = min(bh - 1, tip + max(3, bh // 6))
        if under >= bh:
            continue
        aos.append(float(gray[far, c] - gray[under, c]) / 255.0)

    if not aos:
        return 0.6

    ao = float(np.mean(aos))
    return float(np.clip(0.4 + ao * 5.0, 0.0, 1.0))


def apply_contact_shadow(rgb: np.ndarray, strength: float = 0.45) -> np.ndarray:
    """Paint a soft elliptical contact shadow under lower-band subject mass."""
    out = rgb.astype(np.float32).copy()
    h, w = out.shape[:2]
    mask = _subject_mask_lower(rgb.astype(np.uint8))
    if mask.mean() < 0.015:
        return rgb.copy()

    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return rgb.copy()

    a = float(np.clip(strength, 0.0, 1.0))
    cx = float(xs.mean())
    y0 = int(h * 0.78)
    cy = y0 + float(ys.mean()) + max(2.0, h * 0.01)
    rx = max(4.0, (xs.max() - xs.min()) * 0.55)
    ry = max(2.0, h * 0.012)
    yy, xx = np.mgrid[0:h, 0:w]
    ell = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    shadow = np.clip(1.0 - ell, 0.0, 1.0)[..., None] * (0.35 * a)
    out = out * (1.0 - shadow)
    return np.clip(out, 0, 255).astype(np.uint8)


def prompt_wants_ground_contact(prompt: str) -> bool:
    """True when the prompt implies feet should meet the ground."""
    text = (prompt or "").lower()
    if not text:
        return False

    if any(k in text for k in _GROUND_NEGATIVE):
        return False

    if "product" in text and "table" in text and "on the floor" not in text and "floor" not in text:
        return False

    return any(k in text for k in _GROUND_POSITIVE)

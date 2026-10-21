"""Human-made likeness scores: rank images by how few generated-image fingerprints they show.

AI stills share a few statistical fingerprints even after CFG: symmetric faces, teal-orange
LUT, clipped HDR highlights, waxy midtones, perfectly centred mass and a period-8/16 lattice
(the last two detectors live in ``utils.quality.ai_look``). These heuristics prefer the
irregularities a photographer or painter leaves in, but never penalise a look the plan
(``utils.quality.art_critic.resolve_plan``) asks for. Capture texture itself is added by
``utils.quality.camera_isp``, after ranking.
"""

from __future__ import annotations

import numpy as np

from utils.quality.quality import pearson
from utils.quality.quality import rgb_to_luminance as _gray

__all__ = [
    "score_human_likeness",
    "score_not_teal_orange",
    "score_highlight_rolloff",
    "score_midtone_entropy",
    "score_off_center_mass",
    "score_mixed_white_balance",
    "score_specular_life",
]


def _as_rgb(img: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[0] < 4 or arr.shape[1] < 4 or arr.shape[2] < 3:
        return None
    return arr[..., :3]


def score_not_teal_orange(rgb: np.ndarray) -> float:
    """Penalize the Instagram teal-highlight / orange-shadow LUT."""
    arr = _as_rgb(rgb)
    if arr is None:
        return 0.5
    x = arr.astype(np.float32)
    lum = _gray(arr)
    lo = lum <= np.percentile(lum, 35)
    hi = lum >= np.percentile(lum, 70)
    if int(lo.sum()) < 16 or int(hi.sum()) < 16:
        return 0.5
    shadow = x[lo].mean(axis=0)
    highlight = x[hi].mean(axis=0)
    # Orange shadows: R >> B. Teal highlights: B >> R.
    orange = float(np.clip((shadow[0] - shadow[2]) / 80.0, 0.0, 1.0))
    teal = float(np.clip((highlight[2] - highlight[0]) / 80.0, 0.0, 1.0))
    lut = orange * teal
    return float(np.clip(1.0 - 0.85 * lut, 0.0, 1.0))


def score_highlight_rolloff(rgb: np.ndarray) -> float:
    """Film-like highlight compression vs HDR clip."""
    arr = _as_rgb(rgb)
    if arr is None:
        return 0.5
    lum = _gray(arr)
    p99 = float(np.percentile(lum, 99.5))
    p90 = float(np.percentile(lum, 90))
    try:
        from utils.ai_accel import quality_highlight_frac

        clip = float(quality_highlight_frac(arr, thr=252.0))
    except Exception:
        clip = float(np.mean(lum >= 252.0))
    # Want bright but not a sheet of 255.
    headroom = float(np.clip((255.0 - p99) / 18.0, 0.0, 1.0))
    span = float(np.clip((p99 - p90) / 40.0, 0.0, 1.0))
    return float(np.clip(0.45 * headroom + 0.35 * (1.0 - min(1.0, clip * 12.0)) + 0.20 * span, 0.0, 1.0))


def score_midtone_entropy(rgb: np.ndarray) -> float:
    """Midtones should carry irregular texture, not waxy flats."""
    arr = _as_rgb(rgb)
    if arr is None:
        return 0.5
    gray = _gray(arr)
    mid = (gray > 40.0) & (gray < 220.0)
    if int(mid.sum()) < 64:
        return 0.5
    patch = gray.copy()
    patch[~mid] = np.median(gray)
    gx = np.abs(np.diff(patch, axis=1, prepend=patch[:, :1]))
    gy = np.abs(np.diff(patch, axis=0, prepend=patch[:1, :]))
    energy = float((gx + gy)[mid].mean())
    # Plastic ~< 2, photo skin ~4–12, oversharpened > 20.
    return float(np.clip(energy / 10.0, 0.0, 1.0))


def score_off_center_mass(rgb: np.ndarray) -> float:
    """Human crops rarely plant the subject on the exact optical axis."""
    arr = _as_rgb(rgb)
    if arr is None:
        return 0.5
    edge = np.abs(np.diff(_gray(arr), axis=1, prepend=0.0)) + np.abs(np.diff(_gray(arr), axis=0, prepend=0.0))
    mass = edge + 1e-3
    h, w = mass.shape
    ys = np.arange(h, dtype=np.float64)[:, None]
    xs = np.arange(w, dtype=np.float64)[None, :]
    cy = float((mass * ys).sum() / mass.sum())
    cx = float((mass * xs).sum() / mass.sum())
    dist = np.hypot((cx / max(w, 1) - 0.5), (cy / max(h, 1) - 0.5))
    return float(np.clip(dist / 0.12, 0.0, 1.0))


def score_mixed_white_balance(rgb: np.ndarray) -> float:
    """Prefer local illuminant variation; a single LUT looks generated."""
    arr = _as_rgb(rgb)
    if arr is None:
        return 0.5
    x = arr.astype(np.float32)
    a = x[..., 0] - x[..., 1]
    b = x[..., 2] - x[..., 1]
    h, w = a.shape
    bs = max(8, min(h, w) // 8)
    ys = list(range(0, h - bs + 1, bs))
    xs = list(range(0, w - bs + 1, bs))
    if len(ys) < 2 or len(xs) < 2:
        return 0.55
    hues: list[float] = []
    for yi in ys:
        for xi in xs:
            hues.append(
                float(np.arctan2(b[yi : yi + bs, xi : xi + bs].mean(), a[yi : yi + bs, xi : xi + bs].mean() + 1e-6))
            )
    std = float(np.std(np.asarray(hues, dtype=np.float64)))
    # Too flat → one filter; chaotic → posterized junk. Sweet spot ~0.08–0.35 rad.
    if std < 0.04:
        return float(np.clip(std / 0.04 * 0.4, 0.0, 0.4))
    if std > 0.55:
        return float(np.clip(1.0 - (std - 0.55) / 0.4, 0.2, 0.7))
    return float(np.clip(0.55 + 0.45 * (std - 0.04) / 0.30, 0.0, 1.0))


def score_specular_life(rgb: np.ndarray) -> float:
    """Catchlights / specks should not be a left-right clone."""
    arr = _as_rgb(rgb)
    if arr is None:
        return 0.5
    lum = _gray(arr)
    thr = float(np.percentile(lum, 98.6))
    hot = lum >= max(thr, 220.0)
    if int(hot.sum()) < 6:
        return 0.58
    h, w = lum.shape
    hw = w // 2
    left = hot[:, :hw]
    right = np.fliplr(hot[:, w - hw :])
    lc = float(left.sum())
    rc = float(right.sum())
    if lc + rc < 8:
        return 0.58
    bal = abs(lc - rc) / (lc + rc)
    corr = pearson(left.astype(np.float32), right.astype(np.float32))
    # Identical bright-spot maps are the doll-eye tell.
    clone = max(0.0, corr)
    return float(np.clip(0.35 * bal + 0.65 * (1.0 - clone), 0.0, 1.0))


def score_human_likeness(rgb: np.ndarray, prompt: str = "", plan: dict | None = None) -> float:
    """[0,1] higher = fewer generated-image fingerprints, skipping any the plan asks for.

    A requested teal-orange grade drops the LUT penalty, any requested grade the single-LUT
    (mixed white balance) penalty, a centred / symmetric / reflected layout the off-centre
    and mirror-symmetry penalties, and periodic content (pixel art, halftone) the lattice.
    """
    from utils.quality.ai_look import score_decoder_grid, score_symmetry
    from utils.quality.art_critic import resolve_plan

    p = resolve_plan(prompt, plan)
    grade = p.get("grade")
    centred = p.get("composition") in ("center", "symmetry")
    terms = {
        "teal_orange": (0.14, None if grade == "teal_orange" else score_not_teal_orange),
        "highlight": (0.12, score_highlight_rolloff),
        "midtone": (0.14, score_midtone_entropy),
        "off_center": (0.10, None if centred else score_off_center_mass),
        "white_balance": (0.12, None if grade else score_mixed_white_balance),
        "specular": (0.10, score_specular_life),
        "lattice": (0.12, None if p.get("periodic") else score_decoder_grid),
        "symmetry": (0.16, None if centred else score_symmetry),
    }
    total = acc = 0.0
    for weight, fn in terms.values():
        v = fn(rgb) if fn is not None else None
        if v is not None:
            total += weight
            acc += weight * float(v)
    return float(np.clip(acc / total, 0.0, 1.0)) if total else 0.5

"""Human-made likeness: rank and finish images so they stop looking generated.

AI stills share a few statistical fingerprints even after CFG/grain:
symmetric faces, teal–orange LUT, clipped HDR highlights, repeating
midtone texture, and perfectly centered mass. These heuristics prefer
the opposite — the same irregularities a photographer or painter leaves in.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "score_human_likeness",
    "score_bilateral_life",
    "score_not_teal_orange",
    "score_highlight_rolloff",
    "score_midtone_entropy",
    "score_off_center_mass",
    "score_lattice_free",
    "score_mixed_white_balance",
    "score_specular_life",
    "apply_sensor_chroma",
    "apply_blue_noise_grain",
    "apply_repeat_break",
    "apply_film_response",
    "apply_local_wb_drift",
    "apply_human_likeness_finish",
]


def _as_rgb(img: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[0] < 8 or arr.shape[1] < 8 or arr.shape[2] < 3:
        return None
    return arr[..., :3]


def _gray(rgb: np.ndarray) -> np.ndarray:
    ch = rgb.astype(np.float32)
    return 0.299 * ch[..., 0] + 0.587 * ch[..., 1] + 0.114 * ch[..., 2]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.size < 8 or y.size != x.size:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if denom < 1e-8:
        return 0.0
    return float(np.clip(np.dot(x, y) / denom, -1.0, 1.0))


def score_bilateral_life(rgb: np.ndarray) -> float:
    """Higher when left/right are related but not a mirror (faces too perfect)."""
    arr = _as_rgb(rgb)
    if arr is None:
        return 0.5
    h, w = arr.shape[:2]
    # Portrait-ish band: upper 70% where faces usually sit.
    band = _gray(arr[: max(8, int(h * 0.72)), :])
    hw = w // 2
    left = band[:, :hw]
    right = np.fliplr(band[:, w - hw :])
    if left.shape != right.shape:
        return 0.5
    corr = max(0.0, _pearson(left, right))
    # ~0.55–0.82 is a lived-in face; >0.95 is the AI doll.
    if corr >= 0.97:
        return 0.12
    if corr >= 0.90:
        return float(np.clip(1.0 - (corr - 0.90) / 0.10, 0.15, 0.55))
    return float(np.clip(0.55 + 0.45 * (0.90 - corr) / 0.90, 0.0, 1.0))


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


def score_lattice_free(rgb: np.ndarray) -> float:
    """Penalize VAE/DiT period-8/16 spectral spikes (the generated-grid tell)."""
    arr = _as_rgb(rgb)
    if arr is None:
        return 0.5
    g = _gray(arr).astype(np.float32)
    g = g - float(g.mean())
    h, w = g.shape
    spec = np.abs(np.fft.rfft2(g))
    med = float(np.median(spec)) + 1e-6

    def _bin(fy: int, fx: int) -> float:
        y0, y1 = max(0, fy - 1), min(spec.shape[0], fy + 2)
        x0, x1 = max(0, fx - 1), min(spec.shape[1], fx + 2)
        return float(spec[y0:y1, x0:x1].max())

    peaks: list[float] = []
    for period in (8, 16):
        fy = int(round(h / period))
        fx = int(round(w / period))
        if fx < spec.shape[1]:
            peaks.append(_bin(0, fx))  # vertical stripes
        if fy < spec.shape[0]:
            peaks.append(_bin(fy, 0))  # horizontal stripes
        if fy < spec.shape[0] and fx < spec.shape[1]:
            peaks.append(_bin(fy, fx))
    ratio = (max(peaks) if peaks else 0.0) / med
    # Irregular photos sit ~1–8; a period-8 lattice is 100× that.
    return float(np.clip(1.0 - (ratio - 12.0) / 40.0, 0.0, 1.0))


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
    corr = _pearson(left.astype(np.float32), right.astype(np.float32))
    # Identical bright-spot maps are the doll-eye tell.
    clone = max(0.0, corr)
    return float(np.clip(0.35 * bal + 0.65 * (1.0 - clone), 0.0, 1.0))


def score_human_likeness(rgb: np.ndarray, prompt: str = "") -> float:
    """[0,1] higher = fewer generated-image fingerprints. Prompt can skip symmetry for mirrors."""
    low = (prompt or "").lower()
    skip_sym = any(k in low for k in ("reflection", "reflected", "bilateral", "mandala", "symmetric"))
    skip_lattice = any(k in low for k in ("pixel art", "checkerboard", "halftone", "led wall"))
    parts = [
        0.14 * score_not_teal_orange(rgb),
        0.12 * score_highlight_rolloff(rgb),
        0.14 * score_midtone_entropy(rgb),
        0.10 * score_off_center_mass(rgb),
        0.12 * score_mixed_white_balance(rgb),
        0.10 * score_specular_life(rgb),
    ]
    parts.append(0.12 * (0.55 if skip_lattice else score_lattice_free(rgb)))
    if skip_sym:
        parts.append(0.16 * 0.55)
    else:
        parts.append(0.16 * score_bilateral_life(rgb))
    return float(np.clip(sum(parts), 0.0, 1.0))


def _to_float(img: np.ndarray) -> tuple[np.ndarray, bool]:
    was = img.dtype == np.uint8
    return np.clip(np.asarray(img, dtype=np.float32), 0.0, 255.0), was


def _from_float(img: np.ndarray, was_uint: bool) -> np.ndarray:
    out = np.clip(img, 0.0, 255.0)
    return out.astype(np.uint8) if was_uint else out


def _blur3(plane: np.ndarray) -> np.ndarray:
    p = np.pad(plane, 1, mode="edge")
    return (
        p[:-2, :-2]
        + p[:-2, 1:-1]
        + p[:-2, 2:]
        + p[1:-1, :-2]
        + p[1:-1, 1:-1]
        + p[1:-1, 2:]
        + p[2:, :-2]
        + p[2:, 1:-1]
        + p[2:, 2:]
    ) / 9.0


def apply_blue_noise_grain(image: np.ndarray, *, amount: float = 0.018, seed: int = 0) -> np.ndarray:
    """High-frequency grain (not tiled white-noise speckle)."""
    a = float(max(0.0, min(0.08, amount)))
    if a <= 0.0:
        return image
    img, was = _to_float(image)
    h, w = img.shape[:2]
    rng = np.random.default_rng(int(seed))
    white = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    high = white - _blur3(white)
    lum = np.clip(_gray(img) / 255.0, 0.05, 1.0)
    # More grain in midtones (film), less in speculars.
    mid = 1.0 - np.abs(lum - 0.45) * 1.4
    grain = high * a * 255.0 * np.clip(mid, 0.15, 1.0)
    out = img + grain[..., None]
    return _from_float(out, was)


def apply_sensor_chroma(image: np.ndarray, *, amount: float = 0.04, seed: int = 0) -> np.ndarray:
    """ISO-like chroma in shadows. Humans read this as a camera, not a render."""
    a = float(max(0.0, min(0.2, amount)))
    if a <= 0.0:
        return image
    img, was = _to_float(image)
    h, w = img.shape[:2]
    rng = np.random.default_rng(int(seed) + 91)
    lum = np.clip(_gray(img) / 255.0, 0.0, 1.0)
    shadow = np.clip(1.0 - lum * 1.6, 0.0, 1.0)
    chroma = rng.normal(0.0, 1.0, (h, w, 3)).astype(np.float32)
    chroma[..., 1] *= 0.55  # green channel is cleaner on real sensors
    out = img + chroma * shadow[..., None] * (a * 28.0)
    return _from_float(out, was)


def apply_repeat_break(image: np.ndarray, *, amount: float = 0.12, seed: int = 0) -> np.ndarray:
    """Break copy-pasted pore/fabric frequency with a tiny high-pass phase shift."""
    a = float(max(0.0, min(1.0, amount)))
    if a <= 0.0:
        return image
    img, was = _to_float(image)
    blur = np.stack([_blur3(img[..., c]) for c in range(3)], axis=-1)
    high = img - blur
    rng = np.random.default_rng(int(seed) + 3)
    h, w = img.shape[:2]
    shift = 1 + int(rng.integers(0, 2))
    rolled = np.roll(high, shift=shift, axis=int(rng.integers(0, 2)))
    out = img + a * 0.35 * (rolled - high)
    return _from_float(out, was)


def apply_film_response(image: np.ndarray, *, amount: float = 0.35) -> np.ndarray:
    """Toe lift + highlight shoulder + shadow desat (print film, not HDR clip)."""
    a = float(max(0.0, min(1.0, amount)))
    if a <= 0.0:
        return image
    img, was = _to_float(image)
    x = img / 255.0
    toe = x + a * 0.032 * (1.0 - x) ** 2
    over = np.maximum(toe - 0.82, 0.0)
    rolled = toe - over + over / (1.0 + a * 1.7 * over)
    lum = np.clip(_gray(img) / 255.0, 0.0, 1.0)
    sat_keep = np.clip(0.52 + 0.48 * lum, 0.52, 1.0)
    mean = rolled.mean(axis=2, keepdims=True)
    chroma = rolled - mean
    out = mean + chroma * (1.0 - a * 0.5 * (1.0 - sat_keep[..., None]))
    return _from_float(out * 255.0, was)


def apply_local_wb_drift(image: np.ndarray, *, amount: float = 0.04, seed: int = 0) -> np.ndarray:
    """Spatially varying white balance — mixed indoor/window light."""
    a = float(max(0.0, min(0.2, amount)))
    if a <= 0.0:
        return image
    img, was = _to_float(image)
    h, w = img.shape[:2]
    rng = np.random.default_rng(int(seed) + 19)
    th, tw = max(3, h // 24), max(3, w // 24)
    field = rng.normal(0.0, 1.0, (th, tw, 3)).astype(np.float32)
    field[..., 1] *= 0.4
    try:
        from PIL import Image

        planes = []
        for c in range(3):
            im = Image.fromarray(field[..., c], mode="F")
            planes.append(np.array(im.resize((w, h), Image.BILINEAR), dtype=np.float32))
        drift = np.stack(planes, axis=-1)
    except Exception:
        drift = np.zeros_like(img)
        drift[: min(th, h), : min(tw, w)] = field[: min(th, h), : min(tw, w)]
    lum = np.clip(_gray(img) / 255.0, 0.0, 1.0)
    mid = 1.0 - np.abs(lum - 0.5) * 1.2
    out = img + drift * a * 9.0 * np.clip(mid, 0.2, 1.0)[..., None]
    return _from_float(out, was)


def apply_human_likeness_finish(
    image: np.ndarray,
    *,
    strength: float = 0.55,
    seed: int = 0,
) -> np.ndarray:
    """Film curve + mixed WB + shadow chroma + blue-noise grain + frequency break."""
    s = float(max(0.0, min(1.0, strength)))
    if s <= 0.0:
        return image
    out = apply_film_response(image, amount=0.40 * s)
    out = apply_local_wb_drift(out, amount=0.028 * s, seed=seed)
    out = apply_repeat_break(out, amount=0.18 * s, seed=seed)
    out = apply_sensor_chroma(out, amount=0.045 * s, seed=seed)
    out = apply_blue_noise_grain(out, amount=0.014 * s, seed=seed)
    return out

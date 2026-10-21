"""Art critic: composition, exposure, local contrast and colour harmony, scored against intent.

``score_art(rgb, plan)`` returns an :class:`ArtReport` whose parts are in [0, 1] (higher is
better; a part without evidence is absent). The plan says what the prompt asked for, so an
intended look is never penalised: a requested grade, a low-/high-key exposure, a centred or
symmetric layout, a dutch angle. ``plan_from_prompt`` derives the plan from prompt keywords;
an explicit plan (e.g. built from the raw user prompt and ``--photo-color-grade``) replaces it
via ``resolve_plan``.

Plan keys: ``grade`` (teal_orange, warm, cool, neon, mono, vivid, muted, mixed), ``key``
(low, high), ``composition`` (thirds, golden, center, symmetry, negative_space, dutch, fill),
``harmony`` (mono, analogous, complementary, split), ``contrast`` (low), ``medium``
(photo, 2d, traditional, 3d, generic) and ``periodic`` (bool: pixel art, halftone and other
deliberately periodic content).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from utils.quality.quality import ramp, rgb_to_luminance

__all__ = [
    "ArtReport",
    "plan_from_prompt",
    "resolve_plan",
    "score_art",
    "score_color_harmony",
    "score_composition",
    "score_exposure",
    "score_horizon",
    "score_local_contrast",
    "spectral_residual_saliency",
]

_GRADE_RULES = (
    ("teal_orange", r"teal[\s-]*(?:and|&|/|-)?[\s-]*orange|orange[\s-]*(?:and|&|/)[\s-]*teal|blockbuster grade"),
    ("mono", r"black[\s-]*and[\s-]*white|b&w|monochrome|gr[ae]yscale|film noir|noir"),
    ("neon", r"neon|cyberpunk|synthwave|vaporwave|blacklight"),
    ("vivid", r"vibrant|vivid|saturated|psychedelic|pop art|technicolou?r|kodachrome"),
    ("muted", r"muted|desaturated|pastel|faded|bleach bypass|washed[\s-]out"),
    ("mixed", r"cinestill|tungsten and daylight|mixed lighting"),
    (
        "warm",
        r"golden hour|sunset|sunrise|tungsten|candle ?lit|candlelight|firelight|warm (?:tones?|grade|light)|sepia|amber",
    ),
    ("cool", r"blue hour|moonlit|moonlight|cool (?:tones?|grade|light)|icy|cold light"),
)
# --photo-color-grade names -> plan grade.
_GRADE_ALIASES = {
    "teal_orange": "teal_orange",
    "kodak_portra": "warm",
    "cinestill_800t": "mixed",
    "noir_bw": "mono",
    "fujifilm_eterna": "muted",
}
_KEY_RULES = (
    ("low", r"low[\s-]*key|chiaroscuro|film noir|noir|dark and moody|moody dark|night scene|at night"),
    ("high", r"high[\s-]*key|bright and airy|airy|overexposed look|white background|seamless white"),
)
_COMPOSITION_RULES = (
    ("dutch", r"dutch angle|dutch tilt|canted|tilted (?:frame|camera|horizon)"),
    ("symmetry", r"symmetric(?:al)?|mirror(?:ed)? symmetry|mandala|wes anderson|bilateral|reflect(?:ed|ion|ions|ing)"),
    ("center", r"cent(?:er|re)d (?:composition|subject|framing)|cent(?:er|re) frame|dead cent(?:er|re)"),
    ("golden", r"golden (?:ratio|spiral|section)"),
    ("thirds", r"rule of thirds"),
    ("negative_space", r"negative space|minimalis[tm]"),
    ("fill", r"close[\s-]*up|extreme close|macro|headshot|tight crop|fills? the frame"),
)
_HARMONY_RULES = (
    ("split", r"split[\s-]*complementary"),
    ("complementary", r"complementary"),
    ("analogous", r"analogous"),
    ("mono", r"monochromatic"),
)
_LOW_CONTRAST_RE = re.compile(r"\b(?:fog(?:gy)?|mist(?:y)?|haz(?:e|y)|soft light|low contrast|overcast|dreamy)\b")
_PERIODIC_RE = re.compile(
    r"\b(?:pixel art|8-bit|16-bit|sprite sheet|pixelated|checkerboard|halftone|led wall|mosaic tiles?)\b"
)


def _first(rules: tuple[tuple[str, str], ...], text: str) -> str | None:
    for name, pattern in rules:
        if re.search(rf"\b(?:{pattern})", text):
            return name
    return None


def plan_from_prompt(prompt: str) -> dict:
    """Intent the prompt states explicitly; keys that are not stated are omitted."""
    text = (prompt or "").lower()
    if not text.strip():
        return {}
    from config.defaults.art_mediums import detect_medium

    plan: dict = {
        "grade": _first(_GRADE_RULES, text),
        "key": _first(_KEY_RULES, text),
        "composition": _first(_COMPOSITION_RULES, text),
        "harmony": _first(_HARMONY_RULES, text),
        "contrast": "low" if _LOW_CONTRAST_RE.search(text) else None,
        "medium": detect_medium(text).family,
        "periodic": bool(_PERIODIC_RE.search(text)),
    }
    return {k: v for k, v in plan.items() if v}


def resolve_plan(prompt: str = "", plan: dict | None = None, *, photo_color_grade: str = "") -> dict:
    """The explicit ``plan`` when given, else the plan derived from ``prompt``; a
    ``--photo-color-grade`` name sets the grade. An explicit plan is not merged with the prompt,
    which downstream is often the enriched prompt full of injected style packs."""
    out = dict(plan) if plan is not None else plan_from_prompt(prompt)
    grade = _GRADE_ALIASES.get(str(photo_color_grade or "").lower().strip())
    if grade:
        out["grade"] = grade
    return out


def spectral_residual_saliency(gray: np.ndarray, size: int = 64) -> np.ndarray:
    """Spectral-residual saliency (Hou & Zhang, CVPR 2007) on a ``size``-px long side, in [0, 1].

    The log-amplitude spectrum minus its 3x3 local mean is the "unexpected" part of the image;
    transforming it back with the original phase and squaring gives the saliency map, smoothed
    by a Gaussian (sigma 3 px at this scale).
    """
    from scipy.ndimage import uniform_filter

    h, w = gray.shape[:2]
    if h >= w:
        sh, sw = size, max(8, round(size * w / h))
    else:
        sh, sw = max(8, round(size * h / w)), size
    small = cv2.resize(np.asarray(gray, dtype=np.float32), (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float64)
    spec = np.fft.fft2(small)
    log_amp = np.log(np.abs(spec) + 1e-9)
    residual = log_amp - uniform_filter(log_amp, size=3, mode="wrap")
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(spec)))) ** 2
    sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), 3.0)
    sal -= sal.min()
    peak = float(sal.max())
    return sal / peak if peak > 1e-12 else np.zeros_like(sal)


def _regions(sal: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Salient regions as (mass, mean saliency, cx, cy) with Hou & Zhang's 3x-mean threshold."""
    mask = (sal >= 3.0 * float(sal.mean())).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = sal.shape
    out = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < 2:
            continue
        out.append((area / (h * w), float(sal[labels == i].mean()), cents[i][0] / w, cents[i][1] / h))
    return out


def score_composition(rgb: np.ndarray, plan: dict | None = None) -> dict[str, float]:
    """Saliency-based layout parts: ``composition``, ``negative_space`` and ``crop``.

    composition - after Liu et al., "Optimizing Photo Composition" (Eurographics 2010): each
      salient region pulls toward its nearest power point (Gaussian falloff, sigma 0.17 of the
      frame) and the saliency-weighted centroid toward the centre (visual balance, sigma 0.2);
      0.6 / 0.4 mix. A centred or symmetric plan scores the centroid against the centre instead
      (sigma 0.08); a golden plan uses the 0.382 / 0.618 points.
    negative_space - fraction of the frame below 15% saliency; 0.30-0.85 scores 1 (>= 0.6 when
      the plan asks for negative space), falling to 0 at 0.15 / 0.97. Skipped for fill / close-up.
    crop - salient mass inside a 3% border band (top band x2, where heads get cut) over all
      salient mass: <= 0.05 scores 1, >= 0.25 scores 0. Skipped for fill / close-up.
    """
    p = plan or {}
    gray = rgb_to_luminance(np.asarray(rgb)[..., :3]) if np.asarray(rgb).ndim == 3 else np.asarray(rgb, np.float32)
    if min(gray.shape[:2]) < 32:
        return {}
    sal = spectral_residual_saliency(gray)
    regions = _regions(sal)
    out: dict[str, float] = {}
    comp = p.get("composition")
    if regions:
        wts = np.array([m * s for m, s, _, _ in regions])
        cents = np.array([(cx, cy) for _, _, cx, cy in regions])
        centroid = (wts[:, None] * cents).sum(0) / wts.sum()
        if comp in ("center", "symmetry"):
            d = float(np.hypot(*(centroid - 0.5)))
            out["composition"] = float(np.exp(-(d**2) / (2 * 0.08**2)))
        else:
            third = (0.382, 0.618) if comp == "golden" else (1 / 3, 2 / 3)
            points = np.array([(x, y) for x in third for y in third])
            dist = np.min(np.hypot(cents[:, None, 0] - points[None, :, 0], cents[:, None, 1] - points[None, :, 1]), 1)
            e_rt = float((wts * np.exp(-(dist**2) / (2 * 0.17**2))).sum() / wts.sum())
            d_vb = float(np.hypot(*(centroid - 0.5)))
            e_vb = float(np.exp(-(d_vb**2) / (2 * 0.2**2)))
            out["composition"] = e_rt if comp in ("thirds", "golden") else 0.6 * e_rt + 0.4 * e_vb
    if comp != "fill":
        neg = float((sal < 0.15).mean())
        if comp == "negative_space":
            out["negative_space"] = ramp(neg, 0.35, 0.6)
        else:
            out["negative_space"] = min(ramp(neg, 0.15, 0.30), ramp(neg, 0.97, 0.85))
        h, w = sal.shape
        bh, bw = max(1, round(0.03 * h)), max(1, round(0.03 * w))
        salient = np.where(sal >= 3.0 * float(sal.mean()), sal, 0.0)
        total = float(salient.sum())
        if total > 1e-9:
            border = 2.0 * salient[:bh].sum() + salient[-bh:].sum() + salient[bh:-bh, :bw].sum()
            border += salient[bh:-bh, -bw:].sum()
            out["crop"] = 1.0 - ramp(float(border) / total, 0.05, 0.25)
    return out


def score_horizon(rgb: np.ndarray, plan: dict | None = None) -> tuple[float | None, float | None]:
    """Accidental horizon tilt from the strongest near-horizontal line; returns (score, tilt_deg).

    Standard Hough transform (Duda & Hart, 1972) over Canny edges at 512 px, restricted to lines
    within 6 deg of horizontal at 0.1 deg resolution; the top-voted line, if it collects at
    least 30% of the width in votes, gives the tilt. Steeper lines are perspective or
    deliberate. |tilt| <= 0.4 deg scores 1, >= 1.5 deg scores 0 (the 1-5 deg range reads as a
    mistake). A dutch-angle plan or no long line gives (None, None).
    """
    if (plan or {}).get("composition") == "dutch":
        return None, None
    arr = np.asarray(rgb)
    gray = rgb_to_luminance(arr[..., :3]) if arr.ndim == 3 else arr.astype(np.float32)
    h, w = gray.shape[:2]
    if min(h, w) < 32:
        return None, None
    scale = 512.0 / max(h, w)
    small = cv2.resize(gray, (max(8, round(w * scale)), max(8, round(h * scale))), interpolation=cv2.INTER_AREA)
    u8 = cv2.GaussianBlur(np.clip(small, 0, 255).astype(np.uint8), (0, 0), 1.5)
    otsu, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(u8, max(10.0, 0.25 * otsu), max(30.0, 0.5 * otsu))
    lines = cv2.HoughLinesWithAccumulator(
        edges, 1, np.pi / 1800, int(0.3 * small.shape[1]), min_theta=np.radians(84.0), max_theta=np.radians(96.0)
    )
    if lines is None or not len(lines):
        return None, None
    tilt = float(np.degrees(lines[0, 0, 1])) - 90.0
    return 1.0 - ramp(abs(tilt), 0.4, 1.5), tilt


def score_exposure(rgb: np.ndarray, plan: dict | None = None) -> tuple[float, float]:
    """Clipping and tonal key against the planned key; returns (score, median luma).

    Highlight clip (luma >= 253) ramps 0.5% -> 5% and shadow clip (<= 2) 1% -> 8%, weighted
    0.6 / 0.4. Key: a mid plan wants the median luma within 0.18-0.82 (0 at 0.06 / 0.94); a
    low-key plan wants it <= 0.35 (0 at 0.6) and tolerates 5% -> 30% crushed shadows; a
    high-key plan wants >= 0.65 (0 at 0.4) and tolerates 3% -> 15% clipped highlights.
    """
    arr = np.asarray(rgb)
    y = (rgb_to_luminance(arr[..., :3]) if arr.ndim == 3 else arr.astype(np.float32)) / 255.0
    hi_clip = float((y >= 253 / 255).mean())
    lo_clip = float((y <= 2 / 255).mean())
    med = float(np.median(y))
    key = (plan or {}).get("key")
    if key == "low":
        clip = 1.0 - 0.6 * ramp(hi_clip, 0.005, 0.05) - 0.4 * ramp(lo_clip, 0.05, 0.30)
        key_ok = ramp(med, 0.6, 0.35)
    elif key == "high":
        clip = 1.0 - 0.6 * ramp(hi_clip, 0.03, 0.15) - 0.4 * ramp(lo_clip, 0.01, 0.08)
        key_ok = ramp(med, 0.4, 0.65)
    else:
        clip = 1.0 - 0.6 * ramp(hi_clip, 0.005, 0.05) - 0.4 * ramp(lo_clip, 0.01, 0.08)
        key_ok = min(ramp(med, 0.06, 0.18), ramp(med, 0.94, 0.82))
    return float(np.clip(clip, 0.0, 1.0) * key_ok), med


def score_local_contrast(rgb: np.ndarray, plan: dict | None = None) -> float | None:
    """Median local RMS contrast (Peli 1990): std / mean luminance in 15x15 windows.

    Flat and lifeless below 0.05 (0 at 0.015), harsh above 0.30 (0 at 0.45). A fog / mist /
    soft-light plan lowers the floor to 0.003 -> 0.015 and keeps the harsh cap.
    """
    arr = np.asarray(rgb)
    y = (rgb_to_luminance(arr[..., :3]) if arr.ndim == 3 else arr.astype(np.float32)) / 255.0
    if min(y.shape[:2]) < 16:
        return None
    mean = cv2.blur(y, (15, 15))
    var = np.maximum(cv2.blur(y * y, (15, 15)) - mean * mean, 0.0)
    c = float(np.median(np.sqrt(var) / (mean + 0.02)))
    lo = (0.003, 0.015) if (plan or {}).get("contrast") == "low" else (0.015, 0.05)
    return min(ramp(c, *lo), ramp(c, 0.45, 0.30))


# Matsuda hue templates as (centre_deg, width_deg) sectors, as used by Cohen-Or et al. 2006.
HUE_TEMPLATES: dict[str, tuple[tuple[float, float], ...]] = {
    "i": ((0.0, 18.0),),
    "V": ((0.0, 93.6),),
    "L": ((0.0, 18.0), (90.0, 79.2)),
    "I": ((0.0, 18.0), (180.0, 18.0)),
    "T": ((0.0, 180.0),),
    "Y": ((0.0, 93.6), (180.0, 18.0)),
    "X": ((0.0, 93.6), (180.0, 93.6)),
}
_HARMONY_TEMPLATES = {"mono": ("i",), "analogous": ("V",), "complementary": ("I", "X"), "split": ("Y", "L")}


def _template_distance(name: str) -> np.ndarray:
    hue = np.arange(360, dtype=np.float64)
    best = np.full(360, np.inf)
    for centre, width in HUE_TEMPLATES[name]:
        diff = np.abs((hue - centre + 180.0) % 360.0 - 180.0)
        best = np.minimum(best, np.maximum(0.0, diff - width / 2.0))
    return best


_TEMPLATE_DIST = {name: _template_distance(name) for name in HUE_TEMPLATES}
_ROT_INDEX = (np.arange(360)[None, :] - np.arange(360)[:, None]) % 360


def score_color_harmony(rgb: np.ndarray, plan: dict | None = None) -> tuple[float | None, str, float]:
    """Hue-template fit (Cohen-Or et al., "Color Harmonization", SIGGRAPH 2006).

    Saturation x value weighted hue histogram; for every Matsuda template and rotation, the mean
    arc distance from each hue to the nearest template sector; the best (template, rotation)
    wins. Returns (score, template, mean deviation in degrees). Uniformly scattered hues fit the
    X template at ~8-10 deg, a two-hue grade at ~0, so <= 2.5 deg scores 1 and >= 9 deg scores 0.
    A planned harmony restricts the templates; near-grey images and mono grades give None.
    """
    p = plan or {}
    if p.get("grade") == "mono":
        return None, "", 0.0
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3 or min(arr.shape[:2]) < 8:
        return None, "", 0.0
    hsv = cv2.cvtColor(np.ascontiguousarray(np.clip(arr[..., :3], 0, 255).astype(np.uint8)), cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.int64) * 2
    s = hsv[..., 1].astype(np.float64) / 255.0
    v = hsv[..., 2].astype(np.float64) / 255.0
    wt = np.where((s > 0.15) & (v > 0.15), s * v, 0.0)
    if wt.sum() < 0.03 * wt.size:
        return None, "", 0.0
    hist = np.bincount(hue.ravel(), weights=wt.ravel(), minlength=360)[:360]
    hist /= hist.sum()
    best_name, best_dev = "", np.inf
    for name in _HARMONY_TEMPLATES.get(p.get("harmony", ""), tuple(HUE_TEMPLATES)):
        dev = float((_TEMPLATE_DIST[name][_ROT_INDEX] @ hist).min())
        if dev < best_dev:
            best_name, best_dev = name, dev
    return 1.0 - ramp(best_dev, 2.5, 9.0), best_name, best_dev


_WEIGHTS = {
    "composition": 0.24,
    "negative_space": 0.08,
    "crop": 0.12,
    "horizon": 0.12,
    "exposure": 0.20,
    "local_contrast": 0.10,
    "harmony": 0.14,
}


@dataclass(slots=True)
class ArtReport:
    score: float
    """Weighted mean of the available parts, in [0, 1]."""
    parts: dict[str, float] = field(default_factory=dict)
    measures: dict[str, object] = field(default_factory=dict)


def score_art(rgb: np.ndarray, plan: dict | None = None, prompt: str = "") -> ArtReport:
    """Every art part for one image, judged against ``resolve_plan(prompt, plan)``."""
    p = resolve_plan(prompt, plan)
    parts = score_composition(rgb, p)
    horizon, tilt = score_horizon(rgb, p)
    exposure, key = score_exposure(rgb, p)
    contrast = score_local_contrast(rgb, p)
    harmony, template, deviation = score_color_harmony(rgb, p)
    parts.update(
        {k: v for k, v in (("horizon", horizon), ("local_contrast", contrast), ("harmony", harmony)) if v is not None}
    )
    parts["exposure"] = exposure
    wsum = sum(_WEIGHTS[k] for k in parts)
    score = sum(_WEIGHTS[k] * v for k, v in parts.items()) / wsum
    measures = {"tilt_deg": tilt, "median_luma": key, "hue_template": template, "hue_deviation_deg": deviation}
    return ArtReport(score=float(score), parts=parts, measures=measures)

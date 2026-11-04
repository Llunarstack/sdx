"""Image critics: AI-look tells, art direction, human-likeness, scene physics, hands, counts, contact shadows, seed variation, and training-manifest curation filters."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils.quality.finishing import pearson, ramp, rgb_to_luminance

# Pixel period of the decoder grid: VAE x8 upsampling under the DiT 2x2 patch (folding by 16
# also covers the x8, x4 and x2 sub-periods).
GRID_PERIOD = 16

# Pooling weights; a missing part (None) drops out and the rest renormalise.
_AI_LOOK_WEIGHTS = {
    "grid": 0.22,
    "noise_floor": 0.12,
    "spectral_slope": 0.12,
    "oversmoothing": 0.16,
    "oversaturation": 0.14,
    "halo": 0.12,
    "symmetry": 0.12,
    "plastic_skin": 0.14,  # utils.quality.realism; None (absent) without visible skin
    "waxy_highlights": 0.08,
}
# Parts that assume a camera capture; skipped for drawn / rendered media.
_CAPTURE_ONLY = ("noise_floor", "spectral_slope", "oversmoothing", "plastic_skin", "waxy_highlights")


def _gray(rgb: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    return rgb_to_luminance(arr[..., :3])


def noise_residual(gray: np.ndarray) -> np.ndarray:
    """Image minus its 3x3 median: the content-suppressed residual forensic detectors look at."""
    g = np.ascontiguousarray(gray, dtype=np.float32)
    return g - cv2.medianBlur(g, 3)


def score_decoder_grid(rgb: np.ndarray, period: int = GRID_PERIOD) -> float | None:
    """Decoder-grid periodicity (VAE upsampling, DiT patches) in the noise residual.

    The median-filter residual (Corvi et al., ICASSP 2023) is folded onto one ``period`` x
    ``period`` tile by averaging over all tiles - the one-image analogue of averaging residuals
    to expose a generator fingerprint. By Parseval, the folded tile's energy is the residual's
    spectral power at the grid-harmonic lattice (multiples of 1/period cycles/px), so this is the
    spectral-peak test of Durall et al. (CVPR 2020) computed without an FFT. A decoder grid or
    nearest/transposed-conv blocking repeats in every tile; content does not. To cancel the noise bias and reject single long edges, the
    folds of the even and odd tile columns (resp. rows) are cross-correlated, as in PRNU
    fingerprint matching (Lukas, Fridrich & Goljan, 2006), and the weaker split counts.
    Both the signed residual (fingerprints, checkerboards) and its magnitude (block edges, the
    Wu & Yuen 1997 blockiness cue) are tested; the result is a z-score under the no-grid null.

    Thresholds: synthetic 1/f^2 and white fields stay below z ~2 and step-edge scenes below ~5,
    so z <= 6 scores 1 and z >= 30 scores 0 (log-linear between). A +-2 level 16-px fingerprint
    on a 256-px 1/f^2 field gives z ~22; nearest-neighbour x8 upsampling gives z ~900.
    """
    gray = _gray(rgb)
    if gray is None or min(gray.shape) < 4 * period:
        return None
    res = noise_residual(gray)
    scale = 1.4826 * float(np.median(np.abs(res - np.median(res)))) + 1e-3
    res = np.clip(res, -3.0 * scale, 3.0 * scale)  # winsorise so strong edges cannot dominate a phase
    th, tw = res.shape[0] // period, res.shape[1] // period
    worst = 0.0
    for x in (res, np.abs(res)):
        tiles = x[: th * period, : tw * period].reshape(th, period, tw, period)
        noise = float(tiles.var()) * period / (th * tw / 2.0) + 1e-12
        z_split = []
        for axis in (0, 2):
            even = [slice(None)] * 4
            odd = [slice(None)] * 4
            even[axis], odd[axis] = slice(0, None, 2), slice(1, None, 2)
            a = tiles[tuple(even)].mean(axis=(0, 2))
            b = tiles[tuple(odd)].mean(axis=(0, 2))
            z_split.append(float(((a - a.mean()) * (b - b.mean())).sum()) / noise)
        worst = max(worst, min(z_split))
    return 1.0 - ramp(float(np.log(max(worst, 1e-6))), float(np.log(6.0)), float(np.log(30.0)))


_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def score_noise_floor(rgb: np.ndarray, block: int = 16) -> float | None:
    """Presence and signal-dependence of a capture noise floor.

    Per-block noise sigma by Immerkaer's fast estimator (CVIU 1996) on the most homogeneous
    quarter of 16x16 blocks, binned by brightness; the per-bin median is the noise level
    function (NLF, Liu et al., CVPR 2006). Two checks:

    * floor: median NLF sigma in 8-bit units. 8-bit quantisation alone gives ~0.3, so < 0.5
      means "smoother than any capture" (0) and >= 1.5 is a normal capture floor (1).
    * shape: real sensors are Poisson-Gaussian (Foi et al., TIP 2008); through the sRGB curve
      read noise lifts the shadows, so sigma falls with brightness, while i.i.d. added grain is
      flat. Coefficient of variation of the NLF across >= 3 bins: <= 0.04 flat (0), >= 0.12
      signal-dependent (1). A simulated Poisson-Gaussian capture of a ramp gives 0.14-0.26,
      constant sigma-2 grain 0.015.
    """
    gray = _gray(rgb)
    if gray is None or min(gray.shape) < 4 * block:
        return None
    h, w = (gray.shape[0] // block) * block, (gray.shape[1] // block) * block
    g = gray[:h, :w]
    conv = np.abs(cv2.filter2D(g, cv2.CV_32F, _IMMERKAER, borderType=cv2.BORDER_REFLECT))
    structure = cv2.GaussianBlur(g, (0, 0), 2.0)
    by, bx = h // block, w // block

    def _blocks(a: np.ndarray) -> np.ndarray:
        return a.reshape(by, block, bx, block).swapaxes(1, 2).reshape(by * bx, block * block)

    sigma = np.sqrt(np.pi / 2.0) / 6.0 * _blocks(conv).mean(axis=1)
    texture = _blocks(structure).std(axis=1)
    mean = _blocks(g).mean(axis=1)
    usable = (mean > 16.0) & (mean < 240.0)
    if int(usable.sum()) < 8:
        return None
    flat = usable & (texture <= np.percentile(texture[usable], 25))
    edges = np.linspace(16.0, 240.0, 7)
    nlf = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = flat & (mean >= lo) & (mean < hi)
        if int(sel.sum()) >= 3:
            nlf.append(float(np.median(sigma[sel])))
    if not nlf:
        return None
    level = float(np.median(nlf))
    floor_score = ramp(level, 0.5, 1.5)
    if len(nlf) < 3:
        return floor_score
    cv = float(np.std(nlf) / max(np.mean(nlf), 1e-6))
    return floor_score * (0.4 + 0.6 * ramp(cv, 0.04, 0.12))


def score_spectral_slope(rgb: np.ndarray) -> float | None:
    """Radially averaged power-spectrum slope against the natural-image 1/f^2 law.

    Natural scenes follow P(f) ~ f^-alpha with alpha ~= 2 (Field 1987; van der Schaaf & van
    Hateren 1996 report 1.88 +- 0.43). Fit over 0.02-0.25 cycles/px (above the layout, below the
    demosaic/JPEG band). |alpha - 2| <= 0.4 scores 1, >= 1.2 scores 0: steeper is the waxy,
    detail-starved look, flatter is oversharpened or noise-dominated.
    """
    gray = _gray(rgb)
    if gray is None or min(gray.shape) < 64:
        return None
    h, w = gray.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    power = np.abs(np.fft.fft2((gray - gray.mean()) * win)) ** 2
    f = np.sqrt(np.fft.fftfreq(h)[:, None] ** 2 + np.fft.fftfreq(w)[None, :] ** 2)
    edges = np.geomspace(0.02, 0.25, 16)
    logf, logp = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (f >= lo) & (f < hi)
        if sel.any():
            logf.append(np.log10(np.sqrt(lo * hi)))
            logp.append(np.log10(power[sel].mean() + 1e-12))
    if len(logf) < 6:
        return None
    alpha = -float(np.polyfit(logf, logp, 1)[0])
    return 1.0 - ramp(abs(alpha - 2.0), 0.4, 1.2)


def score_oversmoothing(rgb: np.ndarray) -> float | None:
    """Missing finest-scale texture where there is mid-scale structure (the waxy look).

    Local variance (7x7) of two difference-of-Gaussian bands, fine (sigma 0 - 1) and mid
    (sigma 1 - 2), over the textured half of the image (mid-band variance above its median);
    the statistic is the median log2(var_fine / var_mid). A 1/f^2 field gives ~+1.6, the same
    field blurred by sigma 0.7 / 1.0 / 1.5 px gives -0.2 / -1.0 / -1.75. Capture optics and
    demosaicing soften by roughly sigma 0.5-0.8, so >= -0.5 scores 1 and <= -2.0 scores 0.
    """
    gray = _gray(rgb)
    if gray is None or min(gray.shape) < 32:
        return None
    b1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
    b2 = cv2.GaussianBlur(gray, (0, 0), 2.0)
    fine = gray - b1
    mid = b1 - b2
    v_fine = cv2.blur(fine * fine, (7, 7))
    v_mid = cv2.blur(mid * mid, (7, 7))
    textured = v_mid > max(float(np.median(v_mid)), 0.25)
    if int(textured.sum()) < 64:
        return None
    ratio = float(np.median(np.log2((v_fine[textured] + 1e-3) / (v_mid[textured] + 1e-3))))
    return 1.0 - ramp(-ratio, 0.5, 2.0)


def score_oversaturation(rgb: np.ndarray, vivid_ok: bool = False) -> float | None:
    """Gamut-hugging chroma, the glossy generated look.

    Two measures, the worse wins: Hasler & Suesstrunk's colourfulness M (2003; their scale puts
    "highly colourful" at 82 and "extremely" at 109), ramping 75 -> 115; and the 99th
    percentile CIELAB chroma C*ab (sRGB primaries reach ~100-130), ramping 80 -> 115.
    ``vivid_ok`` (a requested vivid / neon look) shifts both ramps up by 25.
    """
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3 or min(arr.shape[:2]) < 8:
        return None
    x = arr[..., :3].astype(np.float32)
    rg = x[..., 0] - x[..., 1]
    yb = 0.5 * (x[..., 0] + x[..., 1]) - x[..., 2]
    m = float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))
    from utils.quality.finishing import srgb_to_lab

    step = max(1, max(arr.shape[:2]) // 256)
    lab = srgb_to_lab(x[::step, ::step] / 255.0)
    c99 = float(np.percentile(np.hypot(lab[..., 1], lab[..., 2]), 99))
    shift = 25.0 if vivid_ok else 0.0
    return 1.0 - max(ramp(m, 75.0 + shift, 115.0 + shift), ramp(c99, 80.0 + shift, 115.0 + shift))


def score_halo(rgb: np.ndarray, max_edges: int = 4000) -> float | None:
    """Overshoot / undershoot rims at strong edges (HDR tone-mapping and unsharp-mask halos).

    At Canny edges, sample the luminance profile along the gradient normal. Only step-like edges
    count: >= 20 levels between the far plateaus (5-8 px, resp. 16-24 px for wide HDR glows) and
    each plateau flat to within a quarter of that contrast, so texture is not mistaken for halos.
    The halo is how far the near band (1-3 px, resp. 1-8 px) overshoots the bright plateau plus
    how far it undershoots the dark one, relative to the contrast. An optical edge is monotone
    (~0), in-camera sharpening reaches ~0.1, HDR / heavy unsharp-mask halos 0.3+.
    Median relative overshoot <= 0.10 scores 1, >= 0.35 scores 0; < 30 step edges gives None.
    """
    gray = _gray(rgb)
    if gray is None or min(gray.shape) < 48:
        return None
    u8 = np.clip(gray, 0, 255).astype(np.uint8)
    edges = cv2.Canny(cv2.GaussianBlur(u8, (0, 0), 1.0), 60, 150) > 0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    margin = 26
    edges[:margin] = edges[-margin:] = False
    edges[:, :margin] = edges[:, -margin:] = False
    ys, xs = np.nonzero(edges)
    if ys.size < 50:
        return None
    if ys.size > max_edges:
        pick = np.random.default_rng(0).choice(ys.size, max_edges, replace=False)
        ys, xs = ys[pick], xs[pick]
    ny, nx = gy[ys, xs] / (mag[ys, xs] + 1e-6), gx[ys, xs] / (mag[ys, xs] + 1e-6)
    from scipy.ndimage import map_coordinates

    t = np.arange(-24, 25, dtype=np.float32)
    coords = np.stack([ys[:, None] + ny[:, None] * t[None, :], xs[:, None] + nx[:, None] * t[None, :]])
    prof = map_coordinates(gray, coords.reshape(2, -1), order=1, mode="nearest").reshape(ys.size, t.size)
    c = 24  # index of t == 0

    def _band(a: int, b: int) -> slice:
        return slice(c + a, c + b + 1)

    worst = np.zeros(ys.size, dtype=np.float32)
    keep = np.zeros(ys.size, dtype=bool)
    for near, far in ((3, (5, 8)), (8, (16, 24))):
        bright_band = prof[:, _band(*far)]
        dark_band = prof[:, _band(-far[1], -far[0])]
        bright = np.median(bright_band, axis=1)
        dark = np.median(dark_band, axis=1)
        contrast = bright - dark
        spread = np.maximum(np.ptp(bright_band, axis=1), np.ptp(dark_band, axis=1))
        ok = (contrast >= 20.0) & (spread <= 0.25 * contrast)
        over = prof[:, _band(1, near)].max(axis=1) - bright
        under = dark - prof[:, _band(-near, -1)].min(axis=1)
        rel = (np.maximum(over, 0) + np.maximum(under, 0)) / np.maximum(contrast, 1e-6)
        worst = np.where(ok, np.maximum(worst, rel), worst)
        keep |= ok
    if int(keep.sum()) < 30:
        return None
    return 1.0 - ramp(float(np.median(worst[keep])), 0.10, 0.35)


def _mirror_corr(band: np.ndarray) -> float:
    w = band.shape[1] // 2
    return pearson(band[:, :w], np.fliplr(band[:, -w:])) if w >= 4 else 0.0


def score_symmetry(rgb: np.ndarray) -> float | None:
    """Excessive left-right mirror symmetry of detail (the doll face / cloned-halves tell).

    Mirror correlation of a band-pass (DoG sigma 1 - 4) luminance, which ignores the symmetric
    shading every frontal face has and keeps pores, strands and catchlights. On detected faces
    (OpenCV frontal cascade) the worst face counts: <= 0.55 scores 1, >= 0.85 scores 0.
    Without faces, the whole frame: <= 0.5 scores 1, >= 0.85 scores 0.
    """
    gray = _gray(rgb)
    if gray is None or min(gray.shape) < 16:
        return None
    band = cv2.GaussianBlur(gray, (0, 0), 1.0) - cv2.GaussianBlur(gray, (0, 0), 4.0)
    from utils.quality.test_time_pick import _cv2_face_cascade

    cascade = _cv2_face_cascade()
    faces = []
    if cascade and min(gray.shape) >= 48:
        u8 = np.clip(gray, 0, 255).astype(np.uint8)
        faces = list(cascade.detectMultiScale(u8, scaleFactor=1.12, minNeighbors=5, minSize=(40, 40)))
    if faces:
        corr = max(_mirror_corr(band[y : y + fh, x : x + fw]) for x, y, fw, fh in faces)
        return 1.0 - ramp(corr, 0.55, 0.85)
    return 1.0 - ramp(_mirror_corr(band), 0.5, 0.85)


@dataclass(slots=True)
class AILookReport:
    ai_look: float
    """Pooled generated-look score in [0, 1]; higher looks more generated."""
    parts: dict[str, float] = field(default_factory=dict)
    """Per-detector naturalness in [0, 1] (1 = no tell); detectors without evidence are absent."""

    @property
    def natural(self) -> float:
        return 1.0 - self.ai_look


def ai_look_score(rgb: np.ndarray, prompt: str = "", plan: dict | None = None) -> AILookReport:
    """Run every applicable detector and pool them.

    Pooling is a weighted geometric mean of the naturalness parts (log-linear opinion pool):
    one confident tell (a part near 0) drags the result down even when the rest look fine.
    Capture-only parts (noise floor, spectral slope, oversmoothing) are skipped for 2-D and
    3-D media, the grid for deliberately periodic content (pixel art, halftone), symmetry when
    the plan or prompt asks for it, and the saturation ramps shift up for a vivid / neon grade.
    """
    p = resolve_plan(prompt, plan)
    skip: set[str] = set()
    if p.get("medium") in ("2d", "3d"):
        skip.update(_CAPTURE_ONLY)
    if p.get("periodic"):
        skip.add("grid")
    if p.get("composition") in ("symmetry", "center"):
        skip.add("symmetry")
    detectors = {
        "grid": score_decoder_grid,
        "noise_floor": score_noise_floor,
        "spectral_slope": score_spectral_slope,
        "oversmoothing": score_oversmoothing,
        "oversaturation": lambda im: score_oversaturation(im, vivid_ok=p.get("grade") in ("vivid", "neon")),
        "halo": score_halo,
        "symmetry": score_symmetry,
    }
    from utils.quality.realism import score_plastic_skin, score_waxy_highlights

    detectors.update(plastic_skin=score_plastic_skin, waxy_highlights=score_waxy_highlights)
    parts: dict[str, float] = {}
    for name, fn in detectors.items():
        if name in skip:
            continue
        v = fn(rgb)
        if v is not None:
            parts[name] = float(v)
    if not parts:
        return AILookReport(ai_look=0.5)
    wsum = sum(_AI_LOOK_WEIGHTS[k] for k in parts)
    natural = float(np.exp(sum(_AI_LOOK_WEIGHTS[k] * np.log(max(v, 0.02)) for k, v in parts.items()) / wsum))
    return AILookReport(ai_look=1.0 - natural, parts=parts)


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
    from config.styles import detect_medium

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


_ART_WEIGHTS = {
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
    wsum = sum(_ART_WEIGHTS[k] for k in parts)
    score = sum(_ART_WEIGHTS[k] * v for k, v in parts.items()) / wsum
    measures = {"tilt_deg": tilt, "median_luma": key, "hue_template": template, "hue_deviation_deg": deviation}
    return ArtReport(score=float(score), parts=parts, measures=measures)


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
    lum = rgb_to_luminance(arr)
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
    lum = rgb_to_luminance(arr)
    p99 = float(np.percentile(lum, 99.5))
    p90 = float(np.percentile(lum, 90))
    try:
        from utils.native import quality_highlight_frac

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
    gray = rgb_to_luminance(arr)
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
    edge = np.abs(np.diff(rgb_to_luminance(arr), axis=1, prepend=0.0)) + np.abs(np.diff(rgb_to_luminance(arr), axis=0, prepend=0.0))
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
    lum = rgb_to_luminance(arr)
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


_COLOR_RGB: dict[str, np.ndarray] = {
    "red": np.array([0.78, 0.12, 0.12], dtype=np.float32),
    "blue": np.array([0.12, 0.18, 0.78], dtype=np.float32),
    "green": np.array([0.12, 0.62, 0.18], dtype=np.float32),
    "yellow": np.array([0.82, 0.78, 0.12], dtype=np.float32),
    "orange": np.array([0.88, 0.45, 0.10], dtype=np.float32),
    "purple": np.array([0.48, 0.18, 0.62], dtype=np.float32),
    "pink": np.array([0.88, 0.38, 0.58], dtype=np.float32),
    "white": np.array([0.88, 0.88, 0.88], dtype=np.float32),
    "black": np.array([0.08, 0.08, 0.08], dtype=np.float32),
    "brown": np.array([0.42, 0.26, 0.14], dtype=np.float32),
    "gray": np.array([0.48, 0.48, 0.48], dtype=np.float32),
    "grey": np.array([0.48, 0.48, 0.48], dtype=np.float32),
    "gold": np.array([0.78, 0.62, 0.18], dtype=np.float32),
    "golden": np.array([0.78, 0.62, 0.18], dtype=np.float32),
    "silver": np.array([0.68, 0.70, 0.74], dtype=np.float32),
}
_COLOR_ALT = "|".join(re.escape(k) for k in _COLOR_RGB)
_LEFT_OF_COLORS = re.compile(
    rf"\b({_COLOR_ALT})\b.{{0,40}}?\b(?:to\s+the\s+)?left\s+of\b.{{0,24}}?\b({_COLOR_ALT})\b",
    re.I | re.S,
)
_RIGHT_OF_COLORS = re.compile(
    rf"\b({_COLOR_ALT})\b.{{0,40}}?\b(?:to\s+the\s+)?right\s+of\b.{{0,24}}?\b({_COLOR_ALT})\b",
    re.I | re.S,
)
_LEFT_RIGHT_BIND = re.compile(
    rf"\b({_COLOR_ALT})\b.{{0,48}}?\bon\s+the\s+left\b.{{0,80}}?\b({_COLOR_ALT})\b.{{0,48}}?\bon\s+the\s+right\b",
    re.I | re.S,
)


def _sobel_abs(gray: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = gray[:, 1:] - gray[:, :-1]
    gy[1:, :] = gray[1:, :] - gray[:-1, :]
    return np.abs(gx) + np.abs(gy)


def score_occlusion(rgb: np.ndarray) -> float:
    """Foreground occluder has high-frequency overlay; subject not a smear.

    Combines center-vs-frame edge energy, vertical-bar anisotropy (fences/mesh),
    and overall edge magnitude so a fused blob does not score like a mesh.
    """
    arr = _as_rgb(rgb)
    if arr is None:
        return 0.0
    gray = rgb_to_luminance(arr)
    gx = np.abs(gray[:, 1:] - gray[:, :-1])
    gy = np.abs(gray[1:, :] - gray[:-1, :])
    edge = _sobel_abs(gray)
    h, w = edge.shape
    y0, y1 = int(h * 0.20), max(int(h * 0.20) + 1, int(h * 0.80))
    x0, x1 = int(w * 0.20), max(int(w * 0.20) + 1, int(w * 0.80))
    center = edge[y0:y1, x0:x1]
    mean_center = float(center.mean()) if center.size else 0.0
    mean_all = float(edge.mean())
    fill = float(np.clip(mean_center / (mean_all + 1e-6), 0.0, 1.0))
    gx_m = float(gx.mean())
    gy_m = float(gy.mean())
    aniso = gx_m / (gx_m + gy_m + 1e-6)
    energy = float(np.tanh(mean_all / 18.0))
    return float(np.clip(0.42 * fill + 0.38 * aniso + 0.20 * energy, 0.0, 1.0))


def score_reflection(rgb: np.ndarray) -> float:
    """Lower half resembles a vertically flipped, darker version of upper half
    (table/water reflection) rather than a second identical object.

    Compare upper[::-1] vs lower with a scale/shift; penalize if lower is as
    bright as upper (second object).
    """
    arr = _as_rgb(rgb)
    if arr is None:
        return 0.0
    gray = rgb_to_luminance(arr)
    h = gray.shape[0]
    half = h // 2
    if half < 2:
        return 0.0
    upper = gray[:half]
    lower = gray[half : half + half]
    flipped = upper[::-1]
    # Pearson is cosine of mean-centered vectors (scale + shift invariant).
    corr = max(0.0, pearson(flipped, lower))
    upper_m = float(upper.mean())
    lower_m = float(lower.mean())
    if lower_m >= 0.95 * upper_m:
        darkness_factor = 0.30
    else:
        darkness_factor = 1.0
    return float(np.clip(corr * darkness_factor, 0.0, 1.0))


def _mean_rgb(rgb: np.ndarray) -> np.ndarray:
    ch = rgb.astype(np.float32) / 255.0
    return ch.reshape(-1, 3).mean(axis=0)


def _color_match(mean_rgb: np.ndarray, name: str) -> float:
    target = _COLOR_RGB.get(name.lower())
    if target is None:
        return 0.0
    a = mean_rgb.astype(np.float32)
    b = target
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))


def _parse_left_right_colors(prompt: str) -> tuple[str, str] | None:
    text = prompt or ""
    m = _LEFT_OF_COLORS.search(text)
    if m:
        return m.group(1).lower(), m.group(2).lower()
    m = _LEFT_RIGHT_BIND.search(text)
    if m:
        return m.group(1).lower(), m.group(2).lower()
    m = _RIGHT_OF_COLORS.search(text)
    if m:
        return m.group(2).lower(), m.group(1).lower()
    return None


def score_spatial_color_bind(rgb: np.ndarray, prompt: str) -> float | None:
    """Left/right halves match the colors named in a spatial bind prompt.

    Returns None when the prompt has no parseable color pair so pick-best can skip.
    """
    pair = _parse_left_right_colors(prompt)
    if pair is None:
        return None
    arr = _as_rgb(rgb)
    if arr is None:
        return 0.0
    left_name, right_name = pair
    w = arr.shape[1]
    mid = max(1, w // 2)
    left = _mean_rgb(arr[:, :mid])
    right = _mean_rgb(arr[:, mid:])
    correct = 0.5 * (_color_match(left, left_name) + _color_match(right, right_name))
    swapped = 0.5 * (_color_match(left, right_name) + _color_match(right, left_name))
    return float(np.clip(0.5 + 0.5 * (correct - swapped), 0.0, 1.0))


def _load_rgb_path(image_path: str) -> np.ndarray | None:
    try:
        from PIL import Image

        return np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


def occlusion_score(image_path: str) -> tuple[float | None, dict]:
    """Eval-harness wrapper around ``score_occlusion``."""
    arr = _load_rgb_path(image_path)
    if arr is None:
        return None, {"error": "image_unreadable"}
    return float(score_occlusion(arr)), {"metric": "occlusion"}


def reflection_score(image_path: str) -> tuple[float | None, dict]:
    """Eval-harness wrapper around ``score_reflection``."""
    arr = _load_rgb_path(image_path)
    if arr is None:
        return None, {"error": "image_unreadable"}
    return float(score_reflection(arr)), {"metric": "reflection"}


def spatial_bind_score(image_path: str, prompt: str) -> tuple[float | None, dict]:
    """Eval-harness wrapper around ``score_spatial_color_bind``."""
    arr = _load_rgb_path(image_path)
    if arr is None:
        return None, {"error": "image_unreadable"}
    val = score_spatial_color_bind(arr, prompt)
    if val is None:
        return None, {"skipped": "no_color_pair"}
    return float(val), {"metric": "spatial_color_bind"}


def score_hand_structure(rgb_uint8: np.ndarray) -> float:
    """
    Return score in ~[0, 1]; higher = more local edge structure in hand-ish bands.

    Heuristic: lower 40% of the frame + left/right margins (typical hand locations
    for portraits / full-body). Rewards mid edge density; penalizes both blank mush
    and crunchy noise soup.
    """
    arr = np.asarray(rgb_uint8, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] < 8 or arr.shape[1] < 8:
        return 0.5
    gray = rgb_to_luminance(arr)
    h, w = gray.shape
    # Sobel-ish
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = gray[:, 1:] - gray[:, :-1]
    gy[1:, :] = gray[1:, :] - gray[:-1, :]
    edge = np.sqrt(gx * gx + gy * gy)

    y0 = int(h * 0.55)
    band = edge[y0:, :]
    left = edge[:, : max(1, w // 4)]
    right = edge[:, -max(1, w // 4) :]
    regions = [band, left, right]

    scores: list[float] = []
    for reg in regions:
        m = float(reg.mean())
        # Ideal mid structure ~ 8–40 on 0–255 scale
        if m < 2.0:
            scores.append(0.15)  # mush / blank
        elif m > 80.0:
            scores.append(0.35)  # crunch / noise
        else:
            # Peak around 18
            scores.append(float(np.exp(-((m - 18.0) ** 2) / (2 * 14.0**2))))
    # Asymmetry bonus (perfect mirror faces look fake; hands rarely mirror)
    asym = float(np.abs(left.mean() - right.mean()) / (left.mean() + right.mean() + 1e-3))
    asym_score = float(np.clip(asym * 2.0, 0.0, 1.0))
    base = float(np.mean(scores)) if scores else 0.5
    return float(np.clip(0.75 * base + 0.25 * asym_score, 0.0, 1.0))


@dataclass(slots=True)
class CountBindReport:
    expected_subjects: int
    expected_counts: dict[str, int]
    negation_count: int
    attribute_bindings: int
    score_hint: float  # 0–1 prior before looking at the image


def analyze_prompt_counts(prompt: str) -> CountBindReport:
    """Parse prompt for count/negation/binding complexity."""
    from models.conditioning import PromptParser

    parsed = PromptParser().parse(prompt or "")
    counts = dict(getattr(parsed, "count_constraints", {}) or {})
    triples = list(getattr(parsed, "triples", []) or [])
    negs = list(getattr(parsed, "negation_indices", []) or [])
    n_subj = max(len(triples), len(counts), 1 if (prompt or "").strip() else 0)
    # Harder prompts get a slightly lower prior (need image verification more)
    complexity = min(1.0, 0.15 * len(counts) + 0.1 * len(negs) + 0.05 * len(triples))
    hint = float(max(0.35, 1.0 - 0.4 * complexity))
    return CountBindReport(
        expected_subjects=int(n_subj),
        expected_counts={str(k): int(v) for k, v in counts.items()},
        negation_count=len(negs),
        attribute_bindings=sum(len(getattr(t, "attributes", []) or []) for t in triples),
        score_hint=hint,
    )


def score_count_bind_match(
    rgb_uint8: np.ndarray,
    prompt: str,
    *,
    estimated_people: int | None = None,
) -> float:
    """
    Combine prompt-expected people count with an estimate.

    If ``estimated_people`` is None, tries ``test_time_pick.estimate_people_count``.
    """
    report = analyze_prompt_counts(prompt)
    target = 0
    for v in report.expected_counts.values():
        target = max(target, int(v))
    if target <= 0:
        # Fallback: people words
        pl = (prompt or "").lower()
        for word, n in (("three", 3), ("four", 4), ("five", 5), ("two", 2), ("a pair", 2)):
            if word in pl:
                target = n
                break
    if target <= 0:
        return float(report.score_hint)

    est = estimated_people
    if est is None:
        try:
            from utils.quality.test_time_pick import estimate_people_count

            est = int(estimate_people_count(np.asarray(rgb_uint8)))
        except Exception:
            est = 0
    if est <= 0:
        return float(report.score_hint * 0.7)
    # Soft match
    err = abs(int(est) - int(target))
    return float(max(0.0, 1.0 - 0.35 * err))


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


_SIG_SIZE = 32
_HIST_BINS = 8
# Mean pairwise distance typical for a genuinely diverse same-prompt batch;
# calibrates collapse_score so 1.0 ≈ identical batch, 0.0 ≈ healthy variation.
_REFERENCE_DISTANCE = 0.30


@dataclass(slots=True)
class VariationReport:
    """Batch diversity audit result."""

    mean_distance: float
    min_distance: float
    collapse_score: float
    """0 = healthy variation, 1 = the batch collapsed to one image."""
    near_duplicate_pairs: list[tuple[int, int]] = field(default_factory=list)
    per_image_uniqueness: list[float] = field(default_factory=list)
    """Mean distance from each image to the rest (low = redundant)."""


def image_signature(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(normalized 32x32 luma, per-channel 8-bin histograms) for one RGB image."""
    from PIL import Image

    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img[..., :3])
    small = np.asarray(pil.resize((_SIG_SIZE, _SIG_SIZE), Image.BILINEAR), dtype=np.float32)
    luma = rgb_to_luminance(small) / 255.0
    hists = []
    for c in range(3):
        h, _ = np.histogram(img[..., c], bins=_HIST_BINS, range=(0, 256))
        hists.append(h.astype(np.float32) / max(float(h.sum()), 1.0))
    return luma, np.concatenate(hists)


def signature_distance(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> float:
    """Perceptual distance in [0, ~1]: structure (luma RMS) + palette (hist L1)."""
    luma_d = float(np.sqrt(np.mean((a[0] - b[0]) ** 2)))
    hist_d = float(np.abs(a[1] - b[1]).mean()) * _HIST_BINS / 2.0
    return 0.65 * luma_d + 0.35 * hist_d


def audit_variation(
    images: list[np.ndarray],
    *,
    near_duplicate_threshold: float = 0.05,
) -> VariationReport:
    """Audit a same-prompt batch for seed diversity."""
    n = len(images)
    if n < 2:
        return VariationReport(0.0, 0.0, 0.0, per_image_uniqueness=[0.0] * n)
    sigs = [image_signature(im) for im in images]
    dist = np.zeros((n, n), dtype=np.float64)
    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = signature_distance(sigs[i], sigs[j])
            dist[i, j] = dist[j, i] = d
            if d < near_duplicate_threshold:
                pairs.append((i, j))
    upper = dist[np.triu_indices(n, k=1)]
    mean_d = float(upper.mean())
    min_d = float(upper.min())
    collapse = float(np.clip(1.0 - mean_d / _REFERENCE_DISTANCE, 0.0, 1.0))
    uniqueness = [float(dist[i].sum() / (n - 1)) for i in range(n)]
    return VariationReport(mean_d, min_d, collapse, pairs, uniqueness)


def reseed_indices(report: VariationReport, *, max_collapse: float = 0.6) -> list[int]:
    """
    Which images to regenerate with fresh seeds / stronger exploration.

    From each near-duplicate pair, keep the more unique member and flag the
    other. Empty list when the batch is diverse enough (collapse below cap).
    """
    if report.collapse_score < max_collapse and not report.near_duplicate_pairs:
        return []
    flagged: set[int] = set()
    uniq = report.per_image_uniqueness
    for i, j in report.near_duplicate_pairs:
        keep, drop = (i, j) if uniq[i] >= uniq[j] else (j, i)
        if keep not in flagged:
            flagged.add(drop)
    if not flagged and report.collapse_score >= max_collapse and uniq:
        # Uniform collapse without discrete duplicate pairs: re-roll the most
        # redundant half of the batch.
        order = sorted(range(len(uniq)), key=lambda k: uniq[k])
        flagged.update(order[: max(1, len(order) // 2)])
    return sorted(flagged)


@dataclass(slots=True)
class FilterStats:
    kept: int = 0
    dropped_dup: int = 0
    dropped_caption: int = 0
    dropped_bad_words: int = 0
    dropped_weight: int = 0
    dropped_clip: int = 0
    dropped_aesthetic: int = 0

    @property
    def dropped_total(self) -> int:
        return (
            self.dropped_dup
            + self.dropped_caption
            + self.dropped_bad_words
            + self.dropped_weight
            + self.dropped_clip
            + self.dropped_aesthetic
        )


@dataclass(slots=True)
class FilterConfig:
    dedup: str = ""  # "", "phash", "md5"
    min_caption_len: int = 0
    max_caption_len: int = 0
    bad_words: tuple[str, ...] = ()
    min_weight: float = 0.0
    min_clip_sim: float = 0.0
    min_aesthetic_proxy: float = 0.0
    image_root: Path | None = None
    native_md5: bool = True


def perceptual_hash(path: Path, size: int = 8) -> str:
    """pHash hex string, or ``""`` when imagehash/PIL cannot read the file."""
    try:
        import imagehash
        from PIL import Image

        return str(imagehash.phash(Image.open(path), hash_size=size))
    except Exception:
        return ""


def file_md5(path: Path, *, prefer_native: bool = True) -> str:
    """Streaming MD5 hex (Rust ``file-md5`` when built); ``""`` when the file cannot be read."""
    from utils.native import file_md5_hex

    return file_md5_hex(path, prefer_native_md5=prefer_native)


def _resolve_image(path_str: str, *, jsonl_dir: Path, image_root: Path | None) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    if image_root is not None:
        return (image_root / p).resolve()
    return (jsonl_dir / p).resolve()


def filter_jsonl_row(
    row: dict[str, Any],
    cfg: FilterConfig,
    *,
    seen_hashes: set[str],
    jsonl_dir: Path,
) -> tuple[dict[str, Any] | None, str]:
    """Return (row, reason) or (None, drop_reason)."""
    path = row.get("image_path") or row.get("path") or row.get("image") or ""
    cap = (row.get("caption") or row.get("text") or "").strip()
    if not path or not cap:
        return None, "missing_path_or_caption"
    if cfg.dedup:
        img_path = _resolve_image(str(path), jsonl_dir=jsonl_dir, image_root=cfg.image_root)
        h = perceptual_hash(img_path) if cfg.dedup == "phash" else ""
        h = h or file_md5(img_path, prefer_native=cfg.native_md5)
        if h and h in seen_hashes:
            return None, "dup"
        if h:
            seen_hashes.add(h)
    if cfg.min_caption_len and len(cap) < cfg.min_caption_len:
        return None, "caption_len"
    if cfg.max_caption_len and len(cap) > cfg.max_caption_len:
        return None, "caption_len"
    if cfg.bad_words and any(w in cap.lower() for w in cfg.bad_words):
        return None, "bad_word"
    w = float(row.get("weight", row.get("aesthetic_score", 1.0)))
    if cfg.min_weight > 0 and w < cfg.min_weight:
        return None, "weight"
    if cfg.min_clip_sim > 0 and "clip_sim" in row:
        try:
            if float(row["clip_sim"]) < cfg.min_clip_sim:
                return None, "clip_sim"
        except (TypeError, ValueError):
            pass
    if cfg.min_aesthetic_proxy > 0 and "aesthetic_proxy" in row:
        try:
            if float(row["aesthetic_proxy"]) < cfg.min_aesthetic_proxy:
                return None, "aesthetic"
        except (TypeError, ValueError):
            pass
    return row, ""


_DROP_COUNTERS = {
    "dup": "dropped_dup",
    "caption_len": "dropped_caption",
    "bad_word": "dropped_bad_words",
    "weight": "dropped_weight",
    "clip_sim": "dropped_clip",
    "aesthetic": "dropped_aesthetic",
}


def filter_rows(
    rows: Iterable[dict[str, Any]], cfg: FilterConfig, *, base_dir: Path
) -> tuple[list[dict[str, Any]], FilterStats]:
    """Apply :func:`filter_jsonl_row` to ``rows`` (relative image paths resolve against ``base_dir``)."""
    seen: set[str] = set()
    stats = FilterStats()
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        kept, reason = filter_jsonl_row(row, cfg, seen_hashes=seen, jsonl_dir=base_dir)
        if kept is None:
            counter = _DROP_COUNTERS.get(reason)
            if counter:
                setattr(stats, counter, getattr(stats, counter) + 1)
            continue
        out_rows.append(kept)
        stats.kept += 1
    return out_rows, stats


def filter_jsonl_file(
    input_path: str | Path,
    *,
    config: FilterConfig | None = None,
    output_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], FilterStats]:
    """Filter a JSONL manifest; optionally write output."""
    from data.manifest_utils import iter_manifest_rows

    inp = Path(input_path)
    out_rows, stats = filter_rows(iter_manifest_rows(inp), config or FilterConfig(), base_dir=inp.parent)
    if output_path:
        op = Path(output_path)
        op.parent.mkdir(parents=True, exist_ok=True)
        with op.open("w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_rows, stats

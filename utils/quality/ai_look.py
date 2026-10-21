"""AI-look critic: calibrated detectors for the statistical tells of generated images.

Each ``score_*`` returns a naturalness score in [0, 1] (1 = no tell, 0 = strong tell) or ``None``
when the image carries no evidence either way (too small, no flat patches, no strong edges).
``ai_look_score`` pools the available parts into one ``ai_look`` value (higher = looks generated).

Thresholds are documented per detector. They are priors set from the cited papers and from
synthetic probes (``tests/test_ai.py``), not fitted on a labelled real-vs-generated set.
Score candidates before any stochastic finish (grain, camera ISP): those stages exist to add
exactly the statistics the noise-floor and spectral terms measure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from utils.quality.quality import pearson, ramp, rgb_to_luminance

__all__ = [
    "AILookReport",
    "ai_look_score",
    "noise_residual",
    "score_decoder_grid",
    "score_halo",
    "score_noise_floor",
    "score_oversaturation",
    "score_oversmoothing",
    "score_spectral_slope",
    "score_symmetry",
]

# Pixel period of the decoder grid: VAE x8 upsampling under the DiT 2x2 patch (folding by 16
# also covers the x8, x4 and x2 sub-periods).
GRID_PERIOD = 16

# Pooling weights; a missing part (None) drops out and the rest renormalise.
_WEIGHTS = {
    "grid": 0.22,
    "noise_floor": 0.12,
    "spectral_slope": 0.12,
    "oversmoothing": 0.16,
    "oversaturation": 0.14,
    "halo": 0.12,
    "symmetry": 0.12,
}
# Parts that assume a camera capture; skipped for drawn / rendered media.
_CAPTURE_ONLY = ("noise_floor", "spectral_slope", "oversmoothing")


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
    for lo, hi in zip(edges[:-1], edges[1:]):
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
    for lo, hi in zip(edges[:-1], edges[1:]):
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
    from utils.quality.camera_isp import srgb_to_lab

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
    from utils.quality.art_critic import resolve_plan

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
    parts: dict[str, float] = {}
    for name, fn in detectors.items():
        if name in skip:
            continue
        v = fn(rgb)
        if v is not None:
            parts[name] = float(v)
    if not parts:
        return AILookReport(ai_look=0.5)
    wsum = sum(_WEIGHTS[k] for k in parts)
    natural = float(np.exp(sum(_WEIGHTS[k] * np.log(max(v, 0.02)) for k, v in parts.items()) / wsum))
    return AILookReport(ai_look=1.0 - natural, parts=parts)

"""Image ideas: run once per saved image, after every other finish, on float32 RGB in [0, 1].

Colour work happens in OKLab (Ottosson 2020) or linear light; heavy per-pixel filters use the Rust kernels in
``utils.ideas.kernels`` when built. Strength 1 is the suggested amount.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from utils.ideas import Ctx, idea, kernels
from utils.quality.finishing import linear_to_srgb, srgb_to_linear

_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    np.float32,
)
_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    np.float32,
)
_M1_INV = np.linalg.inv(_M1).astype(np.float32)
_M2_INV = np.linalg.inv(_M2).astype(np.float32)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ _LUMA


def _oklab(rgb: np.ndarray) -> np.ndarray:
    return np.cbrt(srgb_to_linear(rgb) @ _M1.T) @ _M2.T


def _rgb(lab: np.ndarray) -> np.ndarray:
    return linear_to_srgb(((lab @ _M2_INV.T) ** 3) @ _M1_INV.T).astype(np.float32)


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return ndimage.gaussian_filter(img, sigma=(sigma, sigma, 0) if img.ndim == 3 else sigma, mode="reflect")


def _mix(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + float(t) * (b - a)


def _s(ctx: Ctx, hi: float = 1.0) -> float:
    return min(hi, max(0.0, ctx.strength))


def _diag(img: np.ndarray) -> float:
    return math.hypot(img.shape[0], img.shape[1])


def _sobel(gray: np.ndarray) -> np.ndarray:
    return np.hypot(ndimage.sobel(gray, axis=1), ndimage.sobel(gray, axis=0))


def _shift(a: np.ndarray, axis: int, step: int) -> np.ndarray:
    """Neighbour ``step`` (+-1) along ``axis`` with a replicated border."""
    pad = [(0, 0)] * a.ndim
    pad[axis] = (1, 1)
    sl = [slice(None)] * a.ndim
    sl[axis] = slice(1 + step, 1 + step + a.shape[axis])
    return np.pad(a, pad, mode="edge")[tuple(sl)]


def _ink(gray: np.ndarray, amount: float) -> np.ndarray:
    """XDoG line multiplier (Winnemoeller 2012): 1 off the lines, down to ``1 - amount`` on them."""
    d = ndimage.gaussian_filter(gray, 1.0) - 0.98 * ndimage.gaussian_filter(gray, 1.6)
    e = np.where(d >= -0.005, 1.0, 1.0 + np.tanh(60.0 * (d + 0.005)))
    return 1.0 - amount * (1.0 - e)


def _guided(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """He et al. guided filter of ``src`` (H, W, C) by the grey ``guide``."""

    def box(a: np.ndarray) -> np.ndarray:
        return ndimage.uniform_filter(a, size=(2 * r + 1, 2 * r + 1, 1), mode="reflect")

    i = guide[..., None]
    mi, mp = box(i), box(src)
    a = (box(i * src) - mi * mp) / (box(i * i) - mi * mi + eps)
    return box(a) * i + box(mp - a * mi)


# ============================================================================================ quantisation


@idea("image", "Blue-noise dither to 8 bit (void-and-cluster mask): no banding and no visible pattern.", quantizes=True)
def img_blue_noise_dither(rgb, ctx: Ctx):
    h, w, _ = rgb.shape
    mask = kernels.blue_noise(64)
    rng = ctx.rng()
    thr = np.stack(
        [
            np.tile(np.roll(mask, tuple(rng.integers(0, 64, 2)), (0, 1)), (h // 64 + 1, w // 64 + 1))[:h, :w]
            for _ in range(3)
        ],
        -1,
    )
    return np.floor(rgb * 255.0 + 0.5 + _s(ctx) * (thr - 0.5)) / 255.0


@idea(
    "image",
    "Floyd-Steinberg error diffusion to 8 bit (Rust kernel; blue-noise dither without the build).",
    quantizes=True,
)
def img_error_diffusion(rgb, ctx: Ctx):
    out = kernels.error_diffuse(rgb, 256, _s(ctx))
    return img_blue_noise_dither(rgb, ctx) if out is None else out


@idea("image", "Deband: re-smooth stepped gradients in flat regions (skies, walls) before quantisation.")
def img_deband(rgb, ctx: Ctx):
    flat = ndimage.gaussian_filter(np.clip(1.0 - _sobel(_luma(rgb)) / 0.06, 0.0, 1.0), 2.0)
    return rgb + _s(ctx) * flat[..., None] * (_blur(rgb, 2.5) - rgb)


# ============================================================================================ optics & light


@idea("image", "Film halation: a warm red glow bleeding out of bright highlights.")
def img_halation(rgb, ctx: Ctx):
    hi = np.clip((_luma(rgb) - 0.8) / 0.2, 0.0, 1.0) ** 2
    glow = ndimage.gaussian_filter(hi, 0.012 * _diag(rgb))
    return rgb + 0.25 * ctx.strength * glow[..., None] * np.array([1.0, 0.35, 0.12], np.float32) * (1.0 - rgb)


@idea("image", "Multi-scale highlight bloom (screen blend), like light scattering inside a real lens.")
def img_bloom(rgb, ctx: Ctx):
    hi = rgb * (np.clip((_luma(rgb) - 0.7) / 0.3, 0.0, 1.0) ** 2)[..., None]
    glow = sum(_blur(hi, f * _diag(rgb)) for f in (0.005, 0.015, 0.04)) / 3.0
    return 1.0 - (1.0 - rgb) * (1.0 - np.clip(0.35 * ctx.strength * glow, 0.0, 1.0))


@idea("image", "Veiling glare: bright areas lift the black floor slightly, as in a real lens.")
def img_veiling_glare(rgb, ctx: Ctx):
    lin = srgb_to_linear(rgb)
    lum = _luma(lin)
    k = 0.25 * ctx.strength * float((lum * (lum > 0.6)).mean())
    return linear_to_srgb((lin + k) / (1.0 + k))


@idea("image", "Lens corner softness: real optics resolve less detail toward the corners.")
def img_corner_softness(rgb, ctx: Ctx):
    h, w, _ = rgb.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot((yy + 0.5) / h - 0.5, (xx + 0.5) / w - 0.5) / math.hypot(0.5, 0.5)
    wgt = (np.clip((r - 0.55) / 0.45, 0.0, 1.0) ** 2 * _s(ctx, 2.0))[..., None]
    return rgb + wgt * (_blur(rgb, max(0.6, _diag(rgb) / 1500.0)) - rgb)


@idea("image", "Fix lateral chromatic aberration: rescale red and blue about the centre to align with green.")
def img_ca_fix(rgb, ctx: Ctx):
    import cv2

    h, w, _ = rgb.shape
    ref = _sobel(rgb[..., 1])

    def scaled(ch: np.ndarray, k: float) -> np.ndarray:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), 0.0, 1.0 + k)
        return cv2.warpAffine(
            np.ascontiguousarray(ch), m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )

    out = rgb.copy()
    for c in (0, 2):
        best = max(np.linspace(-0.004, 0.004, 9), key=lambda k: float((_sobel(scaled(rgb[..., c], k)) * ref).sum()))
        if best != 0.0:
            out[..., c] = scaled(rgb[..., c], float(best) * _s(ctx))
    return out


@idea("image", "Tilt-shift: blur away from a horizontal focus band (miniature look).")
def img_tilt_shift(rgb, ctx: Ctx):
    h = rgb.shape[0]
    d = np.clip((np.abs((np.arange(h, dtype=np.float32) + 0.5) / h - 0.55) - 0.15) / 0.3, 0.0, 1.0)[:, None, None]
    k = _diag(rgb) / 1000.0 * _s(ctx, 2.0)
    b1, b2 = _blur(rgb, 1.5 * k + 1e-3), _blur(rgb, 4.0 * k + 1e-3)
    return np.where(d < 0.5, rgb + 2 * d * (b1 - rgb), b1 + (2 * d - 1) * (b2 - b1))


# ============================================================================================ colour


@idea("image", "OKLab vibrance: boost low-chroma colours, leave already-saturated ones alone.")
def img_oklab_vibrance(rgb, ctx: Ctx):
    lab = _oklab(rgb)
    c = np.hypot(lab[..., 1], lab[..., 2])
    lab[..., 1:] *= (1.0 + 0.35 * ctx.strength * (1.0 - np.clip(c / 0.25, 0.0, 1.0)))[..., None]
    return _rgb(lab)


@idea("image", "Soft gamut compression in OKLCh: rein in neon chroma while keeping hue and lightness.")
def img_gamut_oklch(rgb, ctx: Ctx):
    lab = _oklab(rgb)
    c = np.hypot(lab[..., 1], lab[..., 2])
    span = 0.1 / max(ctx.strength, 1e-3)
    c2 = np.where(c > 0.2, 0.2 + (c - 0.2) / (1.0 + (c - 0.2) / span), c)
    lab[..., 1:] *= (c2 / np.maximum(c, 1e-6))[..., None]
    return _rgb(lab)


@idea("image", "Gray-world white balance in linear light (removes a global colour cast).")
def img_gray_world_wb(rgb, ctx: Ctx):
    lin = srgb_to_linear(rgb)
    means = lin.reshape(-1, 3).mean(0).clip(1e-4)
    return linear_to_srgb(lin * (1.0 + 0.7 * _s(ctx) * (means.mean() / means - 1.0)))


@idea("image", "White-patch white balance: the brightest 0.5% becomes neutral.")
def img_white_patch_wb(rgb, ctx: Ctx):
    lin = srgb_to_linear(rgb)
    p = np.percentile(lin.reshape(-1, 3), 99.5, axis=0).clip(1e-3)
    return linear_to_srgb(lin * (1.0 + 0.6 * _s(ctx) * (np.clip(p.max() / p, 0.5, 2.0) - 1.0)))


@idea("image", "Zone white balance: neutralise separate colour casts in the shadows and the highlights.")
def img_zone_wb(rgb, ctx: Ctx):
    lab = _oklab(rgb)
    out = lab.copy()
    for mask in (np.clip((0.45 - lab[..., 0]) / 0.2, 0, 1), np.clip((lab[..., 0] - 0.75) / 0.2, 0, 1)):
        total = float(mask.sum())
        if total >= 1.0:
            cast = (lab[..., 1:] * mask[..., None]).reshape(-1, 2).sum(0) / total
            out[..., 1:] -= 0.6 * _s(ctx) * mask[..., None] * cast
    return _rgb(out)


@idea("image", "Split toning: warm highlights and neutral-cool shadows (subtle, not teal-orange).")
def img_split_toning(rgb, ctx: Ctx):
    lum = _luma(rgb)[..., None]
    warm = np.array([0.035, 0.012, -0.025], np.float32)
    cool = np.array([-0.008, 0.0, 0.018], np.float32)
    return rgb + ctx.strength * (np.clip((lum - 0.55) / 0.45, 0, 1) * warm + np.clip((0.45 - lum) / 0.45, 0, 1) * cool)


@idea("image", "Colour harmony (Cohen-Or 2006): pull outlier hues toward an analogous + complementary scheme.")
def img_hue_harmony(rgb, ctx: Ctx):
    lab = _oklab(rgb)
    c = np.hypot(lab[..., 1], lab[..., 2])
    hue = np.arctan2(lab[..., 2], lab[..., 1])
    hist, edges = np.histogram(hue, bins=36, range=(-np.pi, np.pi), weights=c)
    h0 = 0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1])
    centers, widths = np.array([h0, h0 + np.pi]), np.radians([40.0, 20.0])
    d = (hue[..., None] - centers + np.pi) % (2 * np.pi) - np.pi
    k = np.abs(d).argmin(-1)
    dk = np.take_along_axis(d, k[..., None], -1)[..., 0]
    excess = np.sign(dk) * np.clip(np.abs(dk) - widths[k], 0.0, None)
    hue = hue - 0.35 * _s(ctx) * excess * np.clip(c / 0.08, 0.0, 1.0)
    lab[..., 1], lab[..., 2] = c * np.cos(hue), c * np.sin(hue)
    return _rgb(lab)


@idea("image", "Even skin tone: reduce blotchy redness inside detected skin while keeping its texture.")
def img_skin_tone_even(rgb, ctx: Ctx):
    from utils.quality.realism import skin_mask

    mask = ndimage.gaussian_filter(skin_mask(rgb * 255.0).astype(np.float32), 3.0)
    if mask.max() < 0.05:
        return rgb
    lab = _oklab(rgb)
    sigma = max(2.0, _diag(rgb) / 200.0)
    for ch in (1, 2):
        lab[..., ch] -= 0.5 * _s(ctx) * mask * (lab[..., ch] - ndimage.gaussian_filter(lab[..., ch], sigma))
    return _rgb(lab)


@idea("image", "Duotone: map lightness onto a deep-blue to warm-cream ramp.")
def img_duotone(rgb, ctx: Ctx):
    dark, light = np.array([0.10, 0.12, 0.25], np.float32), np.array([0.98, 0.92, 0.80], np.float32)
    return _mix(rgb, dark + (light - dark) * _luma(rgb)[..., None], 0.75 * _s(ctx))


@idea("image", "Soft palette limit: k-means to 12 colours, blended in (poster / cel cohesion).")
def img_palette_limit(rgb, ctx: Ctx):
    from utils.quality.pixel_art import quantize_palette

    q = quantize_palette(rgb * 255.0, 12, seed=int(ctx.seed)).astype(np.float32) / 255.0
    return _mix(rgb, q, 0.5 * _s(ctx))


# ============================================================================================ tone


@idea("image", "Shadow lift: a soft toe that opens crushed blacks without greying the image.")
def img_shadow_lift(rgb, ctx: Ctx):
    return rgb + (0.06 * ctx.strength * np.clip(1.0 - _luma(rgb) / 0.25, 0, 1) ** 2)[..., None]


@idea("image", "Auto levels: stretch the 0.3-99.7 percentile luma range to the full range.")
def img_auto_levels(rgb, ctx: Ctx):
    lo, hi = np.percentile(_luma(rgb), [0.3, 99.7])
    return rgb if hi - lo < 0.05 else _mix(rgb, (rgb - lo) / (hi - lo), 0.7 * _s(ctx))


@idea("image", "Exposure normalise: move the log-average luminance toward photographic mid-grey.")
def img_exposure_normalize(rgb, ctx: Ctx):
    lin = srgb_to_linear(rgb)
    key = float(np.exp(np.log(_luma(lin) + 1e-4).mean()))
    return linear_to_srgb(lin * float(np.clip((0.18 / max(key, 1e-4)) ** (0.5 * _s(ctx)), 0.6, 1.6)))


@idea("image", "CLAHE on lightness (OpenCV): local contrast that never blows out.")
def img_clahe(rgb, ctx: Ctx):
    import cv2

    lab = cv2.cvtColor((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=1.0 + 2.0 * _s(ctx, 2.0), tileGridSize=(8, 8)).apply(lab[..., 0])
    return _mix(rgb, cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0, 0.7)


@idea("image", "Multi-scale retinex on lightness: even illumination and recovered shadow detail.")
def img_retinex(rgb, ctx: Ctx):
    lum = np.clip(_luma(rgb), 1e-3, 1.0)
    r = sum(np.log(lum) - np.log(ndimage.gaussian_filter(lum, f * _diag(rgb)) + 1e-3) for f in (0.01, 0.05, 0.15)) / 3
    lo, hi = np.percentile(r, [1, 99])
    rn = np.clip((r - lo) / max(hi - lo, 1e-4), 0.0, 1.0)
    return rgb + (0.3 * _s(ctx) * (rn - lum))[..., None]


@idea("image", "Dark-channel dehaze (He et al. 2009): removes the milky 'AI haze' from low-contrast renders.")
def img_dehaze(rgb, ctx: Ctx):
    size = max(7, int(0.02 * _diag(rgb)) | 1)
    dark = ndimage.minimum_filter(rgb.min(-1), size=size)
    a = rgb.reshape(-1, 3)[np.argsort(dark.ravel())[-max(1, dark.size // 1000) :]].mean(0).clip(0.3, 1.0)
    t = ndimage.gaussian_filter(1.0 - 0.95 * ndimage.minimum_filter((rgb / a).min(-1), size=size), size / 2)
    return _mix(rgb, (rgb - a) / np.clip(t, 0.1, 1.0)[..., None] + a, 0.35 * _s(ctx))


@idea("image", "Tame plastic specular highlights (small, bright, colourless hot spots).")
def img_specular_tame(rgb, ctx: Ctx):
    lum = _luma(rgb)
    spec = (lum - ndimage.gaussian_filter(lum, 3.0) > 0.05) & (lum > 0.85) & (rgb.max(-1) - rgb.min(-1) < 0.12)
    m = ndimage.gaussian_filter(spec.astype(np.float32), 1.0)
    return rgb - (0.4 * _s(ctx) * m * np.clip(lum - 0.85, 0.0, None))[..., None]


@idea("image", "Saliency vignette: gently darken what the eye should skip (spectral-residual saliency).")
def img_saliency_vignette(rgb, ctx: Ctx):
    import cv2

    from utils.quality.critics import spectral_residual_saliency

    sal = np.asarray(spectral_residual_saliency(_luma(rgb) * 255.0), np.float32)
    sal = ndimage.gaussian_filter(cv2.resize(sal, (rgb.shape[1], rgb.shape[0])), _diag(rgb) / 60.0)
    sal = (sal - sal.min()) / max(float(sal.max() - sal.min()), 1e-6)
    return rgb * (1.0 - 0.15 * _s(ctx, 2.0) * (1.0 - sal))[..., None]


# ============================================================================================ detail & texture


@idea("image", "Edge-aware sharpening: crisp edges without amplifying noise in flat areas.")
def img_edge_sharpen(rgb, ctx: Ctx):
    e = _sobel(_luma(rgb))
    e = np.clip(e / max(float(np.percentile(e, 95)), 1e-4), 0.0, 1.0)
    sharp = rgb + 0.8 * ctx.strength * e[..., None] * (rgb - _blur(rgb, 1.0))
    # Clamp to each pixel's 3x3 range in the original: sharpening without bright/dark halos.
    return np.clip(sharp, ndimage.minimum_filter(rgb, (3, 3, 1)), ndimage.maximum_filter(rgb, (3, 3, 1)))


@idea("image", "Richardson-Lucy deconvolution of luma: recovers detail softened by the VAE decoder.")
def img_richardson_lucy(rgb, ctx: Ctx):
    from scipy.signal import fftconvolve

    lum = np.clip(_luma(rgb), 1e-4, 1.0)
    ax = np.arange(-3, 4)
    psf = np.outer(*(2 * [np.exp(-(ax**2) / (2 * 0.8**2))]))
    psf /= psf.sum()
    est = lum.copy()
    for _ in range(max(1, round(8 * _s(ctx, 2.0)))):
        rel = lum / np.maximum(fftconvolve(est, psf, mode="same"), 1e-4)
        est = np.clip(est * fftconvolve(rel, psf[::-1, ::-1], mode="same"), 0.0, 1.0)
    out = rgb + (0.7 * (est - lum))[..., None]
    # Deconvolution rings at hard edges; the 3x3 range clamp keeps the recovered detail without the halos.
    return np.clip(out, ndimage.minimum_filter(rgb, (3, 3, 1)), ndimage.maximum_filter(rgb, (3, 3, 1)))


@idea("image", "Guided-filter detail boost: lift texture relative to an edge-aware base layer.")
def img_guided_detail(rgb, ctx: Ctx):
    base = _guided(_luma(rgb), rgb, 8, 1e-3)
    return base + (1.0 + 0.6 * ctx.strength) * (rgb - base)


@idea("image", "A-trous wavelet detail boost: lift fine and mid-scale texture separately.")
def img_wavelet_detail(rgb, ctx: Ctx):
    k = np.array([1, 4, 6, 4, 1], np.float32) / 16.0
    c, out = rgb, np.zeros_like(rgb)
    for j, gain in enumerate((0.25, 0.35, 0.2, 0.1)):
        kj = np.zeros(4 * 2**j + 1, np.float32)
        kj[:: 2**j] = k
        nxt = ndimage.convolve1d(ndimage.convolve1d(c, kj, axis=0, mode="reflect"), kj, axis=1, mode="reflect")
        out += (c - nxt) * (1.0 + gain * ctx.strength)
        c = nxt
    return out + c


@idea("image", "Natural-image spectrum: nudge the luma power spectrum toward 1/f^2 (over-smooth or over-busy).")
def img_spectral_1f(rgb, ctx: Ctx):
    lum = _luma(rgb)
    h, w = lum.shape
    spec = np.fft.rfft2(lum - lum.mean())
    f = np.hypot(np.fft.fftfreq(h)[:, None], np.fft.rfftfreq(w)[None, :])
    band = (f > 0.02) & (f < 0.4)
    if band.sum() < 16:
        return rgb
    slope = np.polyfit(np.log(f[band]), np.log(np.abs(spec[band]) ** 2 + 1e-12), 1)[0]
    delta = float(np.clip((-2.0 - slope) / 2.0, -0.5, 0.5)) * 0.5 * ctx.strength
    gain = np.clip((np.maximum(f, 1e-6) / 0.1) ** delta, 0.5, 2.0)
    gain[0, 0] = 1.0
    lum2 = np.fft.irfft2(spec * gain, s=(h, w)) + lum.mean()
    return rgb + (lum2 - lum)[..., None]


@idea("image", "Texture re-injection: band-limited grain only in flat, textureless regions (no plastic look).")
def img_texture_reinject(rgb, ctx: Ctx):
    lum = _luma(rgb)
    var = ndimage.gaussian_filter(lum * lum, 2.0) - ndimage.gaussian_filter(lum, 2.0) ** 2
    flat = np.clip(1.0 - np.sqrt(np.clip(var, 0.0, None)) / 0.02, 0.0, 1.0)
    n = ctx.rng().standard_normal(lum.shape).astype(np.float32)
    band = ndimage.gaussian_filter(n, 0.7) - ndimage.gaussian_filter(n, 1.5)
    return rgb + (0.012 * ctx.strength * flat * band / (band.std() + 1e-6))[..., None]


@idea("image", "Noise consistency: add grain where the image is cleaner than the rest ('one camera' look).")
def img_noise_consistency(rgb, ctx: Ctx):
    lum = _luma(rgb)
    local = ndimage.gaussian_filter(np.abs(lum - ndimage.gaussian_filter(lum, 1.0)), 8.0) * 1.25
    need = np.sqrt(np.clip(np.percentile(local, 70) ** 2 - local**2, 0.0, None))
    return rgb + (_s(ctx, 2.0) * need * ctx.rng().standard_normal(lum.shape).astype(np.float32))[..., None]


# ============================================================================================ smoothing & retouch


@idea("image", "Bilateral smoothing (Rust kernel): removes diffusion mottling while keeping edges.")
def img_bilateral_smooth(rgb, ctx: Ctx):
    return _mix(rgb, kernels.bilateral(rgb, 3, 2.0, 0.08), 0.6 * _s(ctx))


@idea("image", "Despeckle: 3x3 median (Rust kernel) applied only to outlier pixels (hot pixels, sparkles).")
def img_median_despeckle(rgb, ctx: Ctx):
    med = kernels.median3(rgb)
    return np.where(np.abs(rgb - med).max(-1, keepdims=True) > 0.08, _mix(rgb, med, _s(ctx)), rgb)


@idea("image", "Perona-Malik anisotropic diffusion: smooths within regions, never across edges.")
def img_anisotropic_diffusion(rgb, ctx: Ctx):
    out = rgb.copy()
    for _ in range(max(1, round(8 * _s(ctx, 2.0)))):
        lum = _luma(out)
        flux = 0.0
        for axis in (0, 1):
            for step in (1, -1):
                g = np.exp(-(((_shift(lum, axis, step) - lum) / 0.06) ** 2))[..., None]
                flux = flux + g * (_shift(out, axis, step) - out)
        out = out + 0.2 * flux
    return out


@idea("image", "Frequency-separation retouch: even out blotchy skin tone, keep pore-level texture.")
def img_freq_sep_retouch(rgb, ctx: Ctx):
    from utils.quality.realism import skin_mask

    mask = ndimage.gaussian_filter(skin_mask(rgb * 255.0).astype(np.float32), 3.0)[..., None]
    sigma = max(1.5, _diag(rgb) / 300.0)
    low = _blur(rgb, sigma)
    return rgb + 0.5 * _s(ctx) * mask * (_blur(low, 2 * sigma) - low)


@idea("image", "Level the horizon: find a dominant near-horizontal line and rotate it level.")
def img_horizon_level(rgb, ctx: Ctx):
    import cv2

    h, w, _ = rgb.shape
    edges = cv2.Canny((np.clip(_luma(rgb), 0, 1) * 255).astype(np.uint8), 60, 160)
    lines = cv2.HoughLines(edges, 1, np.pi / 720, max(80, w // 4))
    if lines is None:
        return rgb
    tilts = [math.degrees(t) - 90.0 for t in lines[:12, 0, 1] if abs(math.degrees(t) - 90.0) < 6.0]
    angle = float(np.median(tilts)) * _s(ctx) if tilts else 0.0
    if abs(angle) < 0.3:
        return rgb
    rad = math.radians(abs(angle))
    scale = math.cos(rad) + math.sin(rad) * max(h, w) / min(h, w)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    return cv2.warpAffine(rgb.astype(np.float32), m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


# ============================================================================================ looks


@idea("image", "Print halftone: 45-degree dot screen over the image (comic / poster print look).")
def img_halftone(rgb, ctx: Ctx):
    h, w, _ = rgb.shape
    cell = max(4.0, _diag(rgb) / 250.0) * math.sqrt(2.0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = (xx + yy) / cell, (xx - yy) / cell
    dist = np.hypot(u - np.floor(u) - 0.5, v - np.floor(v) - 0.5)
    radius = 0.62 * np.sqrt(np.clip(1.0 - ndimage.gaussian_filter(_luma(rgb), cell / 3.0), 0.0, 1.0))
    ink = np.clip((radius - dist) / 0.08 + 0.5, 0.0, 1.0)
    return _mix(rgb, rgb * (1.0 - ink)[..., None], 0.6 * _s(ctx))


@idea("image", "Ink lines from XDoG edges (illustration / anime line-art emphasis).")
def img_edge_ink(rgb, ctx: Ctx):
    return rgb * _ink(_luma(rgb), 0.7 * _s(ctx))[..., None]


@idea("image", "Cel shading: soft-quantise lightness into four bands, with light ink lines.")
def img_cel_shade(rgb, ctx: Ctx):
    lum = _luma(rgb)
    banded = rgb + (_s(ctx) * 0.85 * (np.round(lum * 4.0) / 4.0 - lum))[..., None]
    return np.clip(banded, 0.0, 1.0) * _ink(lum, 0.4 * _s(ctx))[..., None]


@idea("image", "Paper texture: fibrous multiplicative grain for traditional-media looks.")
def img_paper_texture(rgb, ctx: Ctx):
    n = ctx.rng(salt=7).standard_normal(rgb.shape[:2]).astype(np.float32)
    tex = (
        ndimage.gaussian_filter(n, (0.6, 3.0))
        + ndimage.gaussian_filter(n, (3.0, 0.6))
        + ndimage.gaussian_filter(n, 1.5)
    )
    return rgb * (1.0 + 0.035 * ctx.strength * tex / (tex.std() + 1e-6))[..., None]


@idea("image", "Kuwahara painterly filter (Rust kernel): flat brush-like regions with sharp boundaries.")
def img_kuwahara(rgb, ctx: Ctx):
    return _mix(rgb, kernels.kuwahara(rgb, max(2, round(_diag(rgb) / 400))), _s(ctx))


@idea("image", "Watercolour: painterly flattening, darkened pigment edges and paper granulation.")
def img_watercolor(rgb, ctx: Ctx):
    base = kernels.kuwahara(rgb, max(2, round(_diag(rgb) / 500)))
    edge = _sobel(ndimage.gaussian_filter(_luma(base), 1.0))
    edge = np.clip(edge / max(float(np.percentile(edge, 98)), 1e-4), 0.0, 1.0)
    lab = _oklab(base)
    lab[..., 1:] *= 0.9
    wash = img_paper_texture(np.clip(_rgb(lab) * (1.0 - 0.25 * edge)[..., None], 0.0, 1.0), ctx)
    return _mix(rgb, wash, _s(ctx))


@idea("image", "Pencil sketch: colour-dodge sketch of the lightness, blended over the image.")
def img_pencil_sketch(rgb, ctx: Ctx):
    lum = _luma(rgb)
    blur = ndimage.gaussian_filter(1.0 - lum, max(2.0, _diag(rgb) / 150.0))
    sketch = np.clip(lum / np.maximum(1.0 - blur, 1e-3), 0.0, 1.0)
    return _mix(rgb, np.repeat(sketch[..., None], 3, -1), 0.85 * _s(ctx))

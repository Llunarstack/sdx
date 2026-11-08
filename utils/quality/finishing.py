"""Image finishing: deterministic sRGB helpers (contrast, saturation, grades, filters), the lens / sensor / film capture finish, artistic and human-made post passes, face touch-ups and output resizing."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image


def rgb_to_luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec.601 luma of an (..., 3) array, float32."""
    x = np.asarray(rgb, dtype=np.float32)
    return x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114


def ramp(x: float, lo: float, hi: float) -> float:
    """0 at ``lo``, 1 at ``hi``, linear between (``lo > hi`` gives a falling ramp)."""
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two equally sized arrays; 0 when either is flat."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.size < 2 or y.size != x.size:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if denom < 1e-8:
        return 0.0
    return float(np.clip(np.dot(x, y) / denom, -1.0, 1.0))


def naturalize(
    image: np.ndarray,
    grain_amount: float = 0.015,
    micro_contrast: float = 1.02,
    seed: int | None = None,
) -> np.ndarray:
    """Micro-contrast plus density-space film grain (``camera_isp.simulate_film``, no lens effects).

    ``grain_amount`` 0.015 is ISO 400 colour negative; grain RMS grows with sqrt(ISO), so the
    ISO scales with the square of the amount. Returns the input dtype.
    """
    out = contrast(np.asarray(image, dtype=np.float32), factor=micro_contrast)
    if grain_amount > 0:
        iso = int(np.clip(400.0 * (float(grain_amount) / 0.015) ** 2, 25, 6400))
        out = simulate_film(out, FilmStock("naturalize", iso=iso, lens=NO_LENS), seed=int(seed or 0))
    if image.dtype == np.uint8:
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out


def sharpen(image: np.ndarray, amount: float = 0.5, radius: float = 1.0) -> np.ndarray:
    """
    Unsharp-mask style sharpen. image: (H, W, C) uint8 or float [0,255].
    amount: strength (0 = no change, 1 = strong). radius: blur radius for the mask.
    Requires scipy for gaussian blur; if missing, returns image unchanged.
    """
    if amount <= 0:
        return image
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        return image
    is_float = image.dtype in (np.float32, np.float64)
    if not is_float:
        image = image.astype(np.float32)
    blurred = gaussian_filter(image, sigma=radius, mode="reflect")
    sharp = image + amount * (image - blurred)
    sharp = np.clip(sharp, 0, 255)
    if not is_float:
        sharp = sharp.astype(np.uint8)
    return sharp


def sharpen_pil(pil_image, amount: float = 0.5, radius: float = 1.0):
    """Sharpen a PIL Image. amount 0–1, radius for unsharp (default 1)."""
    arr = np.array(pil_image)
    out = sharpen(arr, amount=amount, radius=radius)
    from PIL import Image

    return Image.fromarray(out.astype(np.uint8) if out.dtype != np.uint8 else out)


def contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Linear contrast: image * factor + (1 - factor) * mean.
    factor > 1 = more contrast, < 1 = less. factor=1 = no change.
    image: (H, W, C) float [0,255] or uint8.
    """
    if abs(factor - 1.0) < 1e-6:
        return image
    is_uint = image.dtype == np.uint8
    if is_uint:
        image = image.astype(np.float32)
    mean = image.mean()
    out = image * factor + (1 - factor) * mean
    out = np.clip(out, 0, 255)
    if is_uint:
        out = out.astype(np.uint8)
    return out


def contrast_pil(pil_image, factor: float):
    """Adjust contrast of a PIL Image. factor > 1 = more contrast."""
    arr = np.array(pil_image, dtype=np.float32)
    out = contrast(arr, factor=factor)
    from PIL import Image

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def saturation_rgb(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """Blend away from / toward Rec.601 gray (same maths as PIL ``ImageEnhance.Color``), in float.

    factor 1.0 = unchanged; 1.05-1.2 adds pop. Returns the input dtype.
    """
    if abs(float(factor) - 1.0) < 1e-6:
        return image
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 255.0)
    gray = rgb_to_luminance(img)[..., None]
    out = np.clip(gray + float(factor) * (img - gray), 0.0, 255.0)
    return np.rint(out).astype(np.uint8) if image.dtype == np.uint8 else out


def gentle_s_curve_luminance(image: np.ndarray, strength: float = 0.0) -> np.ndarray:
    """
    Smoothstep on luminance only, then re-scale RGB to match new luma (hue/sat roughly preserved).
    strength 0 = off; 0.08–0.25 adds punch for photo/3D; 0.04–0.12 for flat illustration/anime.
    """
    if strength <= 0:
        return image
    is_uint = image.dtype == np.uint8
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 255.0)
    lum = rgb_to_luminance(img)
    x = np.clip(lum / 255.0, 0.0, 1.0)
    smooth = 3.0 * x * x - 2.0 * x * x * x
    st = float(min(1.0, max(0.0, strength)))
    y = (1.0 - st) * x + st * smooth
    new_lum = np.clip(y * 255.0, 1e-3, 255.0)
    scale = new_lum / (lum + 1e-3)
    out = np.clip(img * scale[..., np.newaxis], 0.0, 255.0)
    if is_uint:
        return out.astype(np.uint8)
    return out


def chroma_smooth_light(image: np.ndarray, amount: float = 0.0, sigma: float = 1.15) -> np.ndarray:
    """
    Blur chroma (RGB - gray) slightly to reduce speckle/banding in flat fills and skin —
    helps anime/cel styles and noisy photoreal without nuking edges like full RGB blur.
    amount 0 = off; 0.08–0.35 typical blend toward blurred chroma.
    """
    if amount <= 0:
        return image
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        return image
    is_uint = image.dtype == np.uint8
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 255.0)
    lum = rgb_to_luminance(img)
    gray = np.stack([lum, lum, lum], axis=-1)
    chroma = img - gray
    ch_blur = gaussian_filter(chroma, sigma=(sigma, sigma, 0), mode="reflect")
    a = float(min(0.85, max(0.0, amount)))
    out = np.clip(gray + (1.0 - a) * chroma + a * ch_blur, 0.0, 255.0)
    if is_uint:
        return out.astype(np.uint8)
    return out


def luminance_clarity(image: np.ndarray, amount: float = 0.0, radius: float = 1.0) -> np.ndarray:
    """
    Unsharp-mask on luminance only: crisper edges / micro-detail without RGB halos (works across styles).
    amount 0 = off; 0.08–0.35 typical. Needs scipy (same as sharpen()).
    """
    if amount <= 0:
        return image
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        return image
    is_uint = image.dtype == np.uint8
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 255.0)
    lum = rgb_to_luminance(img)
    blur = gaussian_filter(lum, sigma=float(radius), mode="reflect")
    sharp_lum = lum + float(amount) * (lum - blur)
    sharp_lum = np.clip(sharp_lum, 1e-3, 255.0)
    scale = sharp_lum / (lum + 1e-3)
    out = np.clip(img * scale[..., np.newaxis], 0.0, 255.0)
    if is_uint:
        return out.astype(np.uint8)
    return out


def polish_pass(image: np.ndarray, amount: float = 0.0) -> np.ndarray:
    """
    One-knob cross-style finish: mild S-curve + chroma smooth + luminance clarity.
    amount in [0, 1]; try 0.35–0.7. Stacks on top of other passes if you also set --clarity etc.
    """
    if amount <= 0:
        return image
    a = float(min(1.0, max(0.0, amount)))
    out = gentle_s_curve_luminance(image, strength=0.22 * a)
    out = chroma_smooth_light(out, amount=0.32 * a, sigma=1.0 + 0.4 * a)
    return luminance_clarity(out, amount=0.28 * a, radius=0.85 + 0.35 * a)


# Baseline adds for --finishing-preset (added to explicit --clarity / --tone-punch / --chroma-smooth).
FINISHING_PRESET_BASELINES = {
    "none": (0.0, 0.0, 0.0),
    "photo": (0.16, 0.13, 0.06),
    "anime": (0.055, 0.045, 0.24),
    "illustration": (0.11, 0.095, 0.11),
    "characters": (0.14, 0.075, 0.14),
    "painterly": (0.10, 0.11, 0.09),
}


def apply_photo_color_grade(image: np.ndarray, preset: str = "none", strength: float = 0.6) -> np.ndarray:
    """
    Lightweight photographic color grading presets.
    Presets: none|natural|teal_orange|kodak_portra|cinestill_800t|noir_bw|fujifilm_eterna
    """
    p = str(preset or "none").lower().strip()
    if p == "none":
        return image
    s = float(min(1.0, max(0.0, strength)))
    if s <= 0:
        return image

    is_uint = image.dtype == np.uint8
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 255.0)
    lum = rgb_to_luminance(img) / 255.0
    sh = np.clip(1.0 - lum, 0.0, 1.0)[..., None]
    hi = np.clip(lum, 0.0, 1.0)[..., None]

    if p == "natural":
        out = gentle_s_curve_luminance(img, strength=0.08 * s)
        out = saturation_rgb(out, factor=1.0 + 0.03 * s)
    elif p == "teal_orange":
        out = img.copy()
        out[..., 1] += 7.0 * s * sh[..., 0]
        out[..., 2] += 13.0 * s * sh[..., 0]
        out[..., 0] += 14.0 * s * hi[..., 0]
        out = gentle_s_curve_luminance(out, strength=0.12 * s)
    elif p == "kodak_portra":
        out = img.copy()
        out[..., 0] += 10.0 * s
        out[..., 1] += 5.0 * s
        out[..., 2] -= 4.0 * s
        out = saturation_rgb(out, factor=1.0 - 0.05 * s)
        out = gentle_s_curve_luminance(out, strength=0.09 * s)
    elif p == "cinestill_800t":
        out = img.copy()
        out[..., 2] += 10.0 * s * sh[..., 0]
        out[..., 0] += 10.0 * s * hi[..., 0]
        out = contrast(out, factor=1.0 + 0.05 * s)
    elif p == "noir_bw":
        g = rgb_to_luminance(img)
        out = np.stack([g, g, g], axis=-1)
        out = contrast(out, factor=1.0 + 0.14 * s)
    elif p == "fujifilm_eterna":
        out = img.copy()
        out[..., 1] += 5.0 * s
        out[..., 2] += 2.0 * s
        out = saturation_rgb(out, factor=1.0 - 0.08 * s)
        out = gentle_s_curve_luminance(out, strength=0.06 * s)
    else:
        out = img

    out = np.clip(np.asarray(out, dtype=np.float32), 0.0, 255.0)
    return out.astype(np.uint8) if is_uint else out


def apply_photo_filter(image: np.ndarray, filter_name: str = "none", strength: float = 0.5) -> np.ndarray:
    """
    Optical filter simulation. Filters: none|pro_mist|polarizer|nd_long_exposure|vintage_diffusion|clean_digital.

    Mist and vintage diffusion scatter light in linear light (``camera_isp.apply_diffusion_filter``).
    ``nd_long_exposure`` only steers the prompt: motion blur of moving subjects cannot be added
    after the fact. Returns the input dtype.
    """
    f = str(filter_name or "none").lower().strip()
    s = float(min(1.0, max(0.0, strength)))
    if f in ("none", "nd_long_exposure") or s <= 0:
        return image
    out = image
    if f in ("pro_mist", "vintage_diffusion"):
        if f == "pro_mist":
            out = apply_diffusion_filter(out, amount=0.04 + 0.08 * s, radius_frac=0.02)
        else:
            out = apply_diffusion_filter(out, amount=0.10 + 0.15 * s, radius_frac=0.008)
    elif f == "polarizer":
        out = saturation_rgb(out, factor=1.0 + 0.08 * s)
        out = gentle_s_curve_luminance(out, strength=0.08 * s)
    elif f == "clean_digital":
        out = chroma_smooth_light(out, amount=0.08 * s, sigma=0.9)
        out = luminance_clarity(out, amount=0.08 * s, radius=0.9)
    return out


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 sRGB EOTF on values in [0, 1]."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Inverse of ``srgb_to_linear`` (sRGB OETF), clipped to [0, 1]."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB ``[..., 3]`` in [0, 1] -> CIE Lab (D65)."""
    lin = srgb_to_linear(rgb)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = lin @ m.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
    eps = 216 / 24389
    kappa = 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    l = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([l, a, b], axis=-1)


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Inverse of ``srgb_to_lab``, clipped to [0, 1]."""
    l, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (l + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    eps = 216 / 24389
    kappa = 24389 / 27
    xr = np.where(fx**3 > eps, fx**3, (116.0 * fx - 16.0) / kappa)
    yr = np.where(l > kappa * eps, fy**3, l / kappa)
    zr = np.where(fz**3 > eps, fz**3, (116.0 * fz - 16.0) / kappa)
    xyz = np.stack([xr * 0.95047, yr, zr * 1.08883], axis=-1)
    m_inv = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=np.float64,
    )
    return linear_to_srgb(xyz @ m_inv.T)


def _unit(image: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(image, dtype=np.float32)[..., :3] / 255.0, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class Lens:
    """Residual lens character after in-camera corrections."""

    half_fov_deg: float = 23.4  # half the diagonal field of view (50 mm on full frame)
    vignette: float = 0.5  # share of the cos^4 falloff left after lens-shading correction
    ca_red: float = 0.0008  # lateral CA: red magnification - 1 at the frame corner (green = reference)
    ca_blue: float = -0.0006
    psf_sigma: float = 0.45  # optical blur, Gaussian PSF sigma in pixels


NO_LENS = Lens(vignette=0.0, ca_red=0.0, ca_blue=0.0, psf_sigma=0.0)


def user_lens(base: Lens, vignette: float = 0.0, chromatic_aberration: float = 0.0) -> Lens:
    """Apply the 0-1 ``--vignette`` / ``--chromatic-aberration`` knobs to ``base`` (0 keeps its value).

    Chromatic aberration 1.0 is a 0.4% red magnification at the corner (about 3 px on a 1024 px frame).
    """
    if vignette > 0.0:
        base = replace(base, vignette=float(min(1.0, vignette)))
    if chromatic_aberration > 0.0:
        k = 0.004 * float(min(1.0, chromatic_aberration))
        base = replace(base, ca_red=k, ca_blue=-0.8 * k)
    return base


def _radial(h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Offsets from the optical centre and squared radius normalised to 1 at the corner."""
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dy, dx = yy - cy, xx - cx
    return dy, dx, (dx * dx + dy * dy) / max(cx * cx + cy * cy, 1e-6)


def vignette_gain(h: int, w: int, lens: Lens) -> np.ndarray:
    """Natural vignetting (cos^4 law), scaled by the share left after lens-shading correction."""
    t2 = _radial(h, w)[2] * np.float32(np.tan(np.radians(lens.half_fov_deg)) ** 2)
    cos4 = 1.0 / (1.0 + t2) ** 2
    return (1.0 - lens.vignette * (1.0 - cos4)).astype(np.float32)


def lateral_chromatic_aberration(lin: np.ndarray, ca_red: float, ca_blue: float) -> np.ndarray:
    """Lateral CA as per-channel radial magnification ``1 + k r^2`` about the optical centre.

    Cubic-spline resampling at exact (unquantised) sub-pixel positions; green is the reference.
    """
    if ca_red == 0.0 and ca_blue == 0.0:
        return lin
    from scipy.ndimage import map_coordinates

    h, w = lin.shape[:2]
    dy, dx, r2 = _radial(h, w)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    out = lin.copy()
    for c, k in ((0, ca_red), (2, ca_blue)):
        if k == 0.0:
            continue
        s = 1.0 + np.float32(k) * r2
        coords = np.stack([cy + dy / s, cx + dx / s])
        out[..., c] = map_coordinates(lin[..., c], coords, order=3, mode="mirror")
    return out


def apply_lens(lin: np.ndarray, lens: Lens) -> np.ndarray:
    """Vignette, lateral CA and optical blur on a linear-light (H, W, 3) image."""
    h, w = lin.shape[:2]
    out = lin * vignette_gain(h, w, lens)[..., None]
    out = lateral_chromatic_aberration(out, lens.ca_red, lens.ca_blue)
    if lens.psf_sigma > 0.0:
        out = cv2.GaussianBlur(out, (0, 0), lens.psf_sigma, borderType=cv2.BORDER_REFLECT_101)
    return out


def apply_lens_srgb(image: np.ndarray, lens: Lens) -> np.ndarray:
    """``apply_lens`` for an sRGB [0, 255] image (decoded to linear light and back)."""
    lin = apply_lens(srgb_to_linear(_unit(image)), lens)
    return (linear_to_srgb(lin) * 255.0).astype(np.float32)


def apply_diffusion_filter(image: np.ndarray, amount: float, radius_frac: float = 0.02) -> np.ndarray:
    """Mist / soft-focus filter: a share ``amount`` of the light is scattered into a wide Gaussian PSF.

    Done in linear light, so highlights bloom and shadows pick up veiling glare, as with a
    real Pro-Mist or diffusion filter. ``radius_frac`` is the PSF sigma as a share of the diagonal.
    """
    a = float(np.clip(amount, 0.0, 1.0))
    if a <= 0.0:
        return image
    lin = srgb_to_linear(_unit(image))
    h, w = lin.shape[:2]
    sigma = max(0.5, float(radius_frac) * float(np.hypot(h, w)))
    glow = cv2.GaussianBlur(lin, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
    out = (linear_to_srgb((1.0 - a) * lin + a * glow) * 255.0).astype(np.float32)
    return to_uint8(out) if image.dtype == np.uint8 else out


def poisson_gaussian_noise(
    raw: np.ndarray, *, full_well_e: float, read_noise_e: float, rng: np.random.Generator
) -> np.ndarray:
    """Heteroscedastic sensor noise (Foi et al., IEEE TIP 2008): Poisson shot noise plus Gaussian read noise.

    ``raw`` is linear signal normalised to the white level, so
    ``var = raw / full_well_e + (read_noise_e / full_well_e) ** 2``.
    """
    electrons = rng.poisson(np.clip(raw, 0.0, None).astype(np.float64) * full_well_e)
    noisy = electrons + rng.normal(0.0, read_noise_e, raw.shape)
    return (noisy / full_well_e).astype(np.float32)


def mosaic_rggb(rgb: np.ndarray) -> np.ndarray:
    """Sample an (H, W, 3) image through an RGGB Bayer colour filter array."""
    cfa = np.array(rgb[..., 1], dtype=np.float32)
    cfa[0::2, 0::2] = rgb[0::2, 0::2, 0]
    cfa[1::2, 1::2] = rgb[1::2, 1::2, 2]
    return cfa


# Malvar, He & Cutler, "High-quality linear interpolation for demosaicing of Bayer-patterned
# color images" (ICASSP 2004): 5x5 gradient-corrected bilinear kernels.
_MHC_G = (
    np.array([[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1], [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]], np.float32)
    / 8.0
)
_MHC_ROW = (
    np.array(
        [[0, 0, 0.5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1], [0, -1, 0, -1, 0], [0, 0, 0.5, 0, 0]], np.float32
    )
    / 8.0
)
_MHC_COL = np.ascontiguousarray(_MHC_ROW.T)
_MHC_DIAG = (
    np.array(
        [[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5], [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]], np.float32
    )
    / 8.0
)


def demosaic_malvar(cfa: np.ndarray) -> np.ndarray:
    """Malvar-He-Cutler demosaic of an RGGB mosaic to (H, W, 3)."""
    cfa = np.ascontiguousarray(cfa, dtype=np.float32)

    def filt(k: np.ndarray) -> np.ndarray:
        # Reflect-101 keeps the Bayer phase at the borders.
        return cv2.filter2D(cfa, -1, k, borderType=cv2.BORDER_REFLECT_101)

    g_rb, row, col, diag = filt(_MHC_G), filt(_MHC_ROW), filt(_MHC_COL), filt(_MHC_DIAG)
    r, g, b = np.empty_like(cfa), g_rb, np.empty_like(cfa)
    r[0::2, 0::2], b[0::2, 0::2] = cfa[0::2, 0::2], diag[0::2, 0::2]
    r[0::2, 1::2], g[0::2, 1::2], b[0::2, 1::2] = row[0::2, 1::2], cfa[0::2, 1::2], col[0::2, 1::2]
    r[1::2, 0::2], g[1::2, 0::2], b[1::2, 0::2] = col[1::2, 0::2], cfa[1::2, 0::2], row[1::2, 0::2]
    r[1::2, 1::2], b[1::2, 1::2] = diag[1::2, 1::2], cfa[1::2, 1::2]
    return np.stack([r, g, b], axis=-1)


@dataclass(frozen=True, slots=True)
class CameraProfile:
    """Digital camera: lens, sensor, colour pipeline and JPEG engine.

    ``ccm`` maps white-balanced camera RGB to linear sRGB (rows sum to 1, so white is kept); the
    matrices below are representative daylight matrices, not measurements of a specific body.
    Off-diagonals are kept non-positive so the inverse is non-negative and unprocessed raw stays >= 0.
    """

    name: str
    lens: Lens = field(default_factory=Lens)
    iso: int = 100
    base_iso: int = 100
    full_well_e: float = 30000.0  # electrons at base ISO
    read_noise_e: float = 2.5  # pre-amplifier read noise (electrons)
    post_read_noise_e: float = 10.0  # post-amplifier read noise in base-ISO electrons (falls as 1/gain)
    wb_gains: tuple[float, float, float] = (2.0, 1.0, 1.55)
    ccm: tuple[tuple[float, float, float], ...] = ((1.80, -0.66, -0.14), (-0.20, 1.55, -0.35), (-0.02, -0.50, 1.52))
    tone: float = 0.5  # share of smoothstep in the S-shaped tone curve
    cfa: bool = True
    chroma_nr: float = 0.0  # Gaussian sigma on Cb/Cr, pixels
    sharpen: float = 0.5
    sharpen_radius: float = 1.0
    sharpen_threshold: float = 1.0 / 255.0
    jpeg_quality: int = 92


CAMERA_PRESETS: dict[str, CameraProfile] = {
    "phone": CameraProfile(
        "phone",
        lens=Lens(half_fov_deg=38.0, vignette=0.25, ca_red=0.0006, ca_blue=-0.0005, psf_sigma=0.6),
        iso=50,
        base_iso=50,
        full_well_e=15000.0,  # effective, after multi-frame merging
        read_noise_e=1.5,
        post_read_noise_e=3.0,
        wb_gains=(1.9, 1.0, 1.7),
        ccm=((1.65, -0.50, -0.15), (-0.28, 1.58, -0.30), (-0.02, -0.60, 1.62)),
        tone=0.65,
        chroma_nr=2.0,
        sharpen=0.8,
        jpeg_quality=88,
    ),
    "dslr": CameraProfile(
        "dslr",
        lens=Lens(half_fov_deg=23.4, vignette=0.6, ca_red=0.0010, ca_blue=-0.0008, psf_sigma=0.45),
        full_well_e=40000.0,
        read_noise_e=2.5,
        post_read_noise_e=12.0,
        tone=0.45,
        chroma_nr=0.7,
        sharpen=0.4,
        sharpen_radius=0.9,
        jpeg_quality=95,
    ),
    "compact": CameraProfile(
        "compact",
        lens=Lens(half_fov_deg=31.0, vignette=0.4, ca_red=0.0015, ca_blue=-0.0012, psf_sigma=0.65),
        full_well_e=12000.0,
        read_noise_e=3.0,
        post_read_noise_e=8.0,
        wb_gains=(1.95, 1.0, 1.6),
        ccm=((1.70, -0.55, -0.15), (-0.25, 1.50, -0.25), (-0.05, -0.55, 1.60)),
        tone=0.55,
        chroma_nr=1.0,
        sharpen=0.6,
        jpeg_quality=90,
    ),
}


def _tone(x: np.ndarray, t: float) -> np.ndarray:
    """Camera tone curve: blend of identity and smoothstep (Brooks et al. use the full smoothstep)."""
    return (1.0 - t) * x + t * x * x * (3.0 - 2.0 * x)


def _inverse_tone(y: np.ndarray, t: float) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, 4097, dtype=np.float32)
    return np.interp(y, _tone(grid, t), grid).astype(np.float32)


def _chroma_denoise(rgb: np.ndarray, sigma: float) -> np.ndarray:
    """Blur Cb/Cr (BT.601 colour differences) and keep luma, as camera chroma NR does."""
    if sigma <= 0.0:
        return rgb
    y = rgb_to_luminance(rgb)
    cr = cv2.GaussianBlur(rgb[..., 0] - y, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
    cb = cv2.GaussianBlur(rgb[..., 2] - y, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
    r, b = y + cr, y + cb
    return np.stack([r, (y - 0.299 * r - 0.114 * b) / 0.587, b], axis=-1)


def _sharpen_luma(rgb: np.ndarray, amount: float, radius: float, threshold: float) -> np.ndarray:
    """In-camera unsharp mask on luma with coring, so flat-area noise is not amplified."""
    if amount <= 0.0:
        return rgb
    y = rgb_to_luminance(rgb)
    d = y - cv2.GaussianBlur(y, (0, 0), radius, borderType=cv2.BORDER_REFLECT_101)
    d = np.sign(d) * np.maximum(np.abs(d) - threshold, 0.0)
    return rgb + amount * d[..., None]


def jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    """Encode and decode as baseline JPEG with 4:2:0 chroma subsampling; returns uint8."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(to_uint8(image)).save(buf, "JPEG", quality=int(quality), subsampling=2)
    with Image.open(io.BytesIO(buf.getvalue())) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def simulate_capture(image: np.ndarray, profile: CameraProfile, *, seed: int = 0, jpeg: bool = True) -> np.ndarray:
    """Re-capture an sRGB [0, 255] image through ``profile``.

    Returns uint8 when the JPEG stage runs, else float32 [0, 255].
    """
    rng = np.random.default_rng(int(seed))
    ccm = np.asarray(profile.ccm, dtype=np.float32)
    gains = np.asarray(profile.wb_gains, dtype=np.float32)

    lin = srgb_to_linear(_inverse_tone(_unit(image), profile.tone))
    raw = apply_lens((lin @ np.linalg.inv(ccm).T) / gains, profile.lens)

    gain = max(profile.iso, profile.base_iso) / profile.base_iso
    full_well = profile.full_well_e / gain
    read = float(np.hypot(profile.read_noise_e, profile.post_read_noise_e / gain))
    if profile.cfa:
        cfa = poisson_gaussian_noise(mosaic_rggb(raw), full_well_e=full_well, read_noise_e=read, rng=rng)
        # White balance on the mosaic, then clip at the white level (no magenta highlights).
        cfa = np.clip(cfa * mosaic_rggb(np.broadcast_to(gains, raw.shape)), 0.0, 1.0)
        cam = demosaic_malvar(cfa)
    else:
        cam = np.clip(poisson_gaussian_noise(raw, full_well_e=full_well, read_noise_e=read, rng=rng) * gains, 0.0, 1.0)

    out = _tone(linear_to_srgb(cam @ ccm.T), profile.tone)
    out = _chroma_denoise(out, profile.chroma_nr)
    out = _sharpen_luma(out, profile.sharpen, profile.sharpen_radius, profile.sharpen_threshold)
    out = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.float32)
    if jpeg and profile.jpeg_quality > 0:
        return jpeg_roundtrip(out, profile.jpeg_quality)
    return out


@dataclass(frozen=True, slots=True)
class FilmStock:
    """Colour-negative or B&W emulsion, scanned.

    ``granularity`` is diffuse RMS granularity x1000 at net density 1.0 through a 48 um aperture
    (the datasheet figure; Tri-X 400 is 17). 0 derives it from ISO: colour negative ~4*sqrt(ISO/100),
    conventional B&W ~8.5*sqrt(ISO/100).
    """

    name: str
    iso: int = 400
    monochrome: bool = False
    granularity: float = 0.0
    frame_width_mm: float = 36.0
    clump_um: float = 12.0  # grain / dye-cloud clump size
    gamma: float = 0.65  # negative contrast on the straight-line part of the H&D curve
    lens: Lens = field(default_factory=lambda: Lens(vignette=0.6, psf_sigma=0.5))

    def rms_granularity(self) -> float:
        if self.granularity > 0.0:
            return self.granularity
        return (8.5 if self.monochrome else 4.0) * float(np.sqrt(max(self.iso, 6) / 100.0))


_D_RANGE = 2.2  # usable density range of the negative above base + fog
_LOG_E_MID = float(np.log10(0.18)) + 0.6  # H&D inflection two stops over mid grey


def _hd_curve(log_e: np.ndarray, gamma: float) -> np.ndarray:
    """Net negative density vs log10 exposure: logistic H&D curve with mid-slope ``gamma``."""
    k = 4.0 * gamma / _D_RANGE
    return _D_RANGE / (1.0 + np.exp(-k * np.log(10.0) * (log_e - _LOG_E_MID)))


def _hd_inverse(density: np.ndarray, gamma: float) -> np.ndarray:
    k = 4.0 * gamma / _D_RANGE
    s = np.clip(density / _D_RANGE, 1e-6, 1.0 - 1e-6)
    return _LOG_E_MID - np.log(1.0 / s - 1.0) / (k * np.log(10.0))


def simulate_film(image: np.ndarray, stock: FilmStock, *, seed: int = 0) -> np.ndarray:
    """Expose an sRGB [0, 255] image onto ``stock`` and scan it back; returns float32 [0, 255].

    Grain lives in density: each dye layer (one for B&W) gets an independent field whose RMS
    follows Selwyn's law (sigma_D * sqrt(area) constant) scaled to the scan pixel, grows with
    sqrt(net density) (Nutting), and is spatially correlated over the clump size. Density is then
    mapped back through the same H&D curve, so tone is unchanged and only grain is added.
    """
    lin = apply_lens(srgb_to_linear(_unit(image)), stock.lens)
    h, w = lin.shape[:2]
    pitch_um = stock.frame_width_mm * 1000.0 / max(h, w)
    sigma_d = stock.rms_granularity() / 1000.0 * 48.0 / max(pitch_um, stock.clump_um)

    layers = rgb_to_luminance(lin)[..., None] if stock.monochrome else lin
    exposure = np.clip(layers, 1e-4, None)
    density = _hd_curve(np.log10(exposure), stock.gamma)

    rng = np.random.default_rng(int(seed))
    field_ = rng.standard_normal(density.shape).astype(np.float32)
    blur = stock.clump_um / pitch_um
    if blur >= 0.3:
        field_ = cv2.GaussianBlur(field_, (0, 0), blur, borderType=cv2.BORDER_REFLECT_101).reshape(density.shape)
        field_ /= field_.reshape(-1, density.shape[-1]).std(axis=0) + 1e-8
    grainy = density + sigma_d * np.sqrt(np.maximum(density, 0.0)) * field_
    ratio = np.power(10.0, _hd_inverse(grainy, stock.gamma)) / exposure
    return (linear_to_srgb(lin * ratio) * 255.0).astype(np.float32)


_FILM_RE = re.compile(
    r"\b(?:analog(?:ue)? (?:film|photo\w*)|film (?:photo\w*|grain|still|camera|stock|scan|look)|"
    r"(?:35|120|16|8) ?mm film|shot on (?:35|120|16|8) ?mm|super ?8|medium format film|"
    r"black[- ]and[- ]white film|b&w film|monochrome film|expired film|disposable camera|"
    r"kodak|portra|ektar|ektachrome|kodachrome|tri-?x|t-?max|ilford|hp5|cinestill|vision ?3|"
    r"fuji(?:film)? (?:superia|velvia|provia|pro ?400h|c200)|polaroid|instax|lomograph\w*)\b",
    re.I,
)
_MONO_RE = re.compile(
    r"\b(?:black[- ]and[- ]white|b&w|monochrome|tri-?x|t-?max|ilford|hp5|noir|greyscale|grayscale)\b", re.I
)
_ISO_RE = re.compile(r"\biso[ -]?(\d{2,5})\b", re.I)
_PHONE_RE = re.compile(
    r"\b(?:iphone|smartphone|phone (?:photo|camera|pic|snapshot)|camera phone|selfie|android phone|"
    r"pixel \d|galaxy s\d+|snapchat|instagram story)\b",
    re.I,
)
_COMPACT_RE = re.compile(
    r"\b(?:point[- ]and[- ]shoot|compact (?:digital )?camera|digicam|ccd (?:camera|sensor)|"
    r"early digital camera|2000s digital (?:camera|photo)|powershot|cybershot|coolpix)\b",
    re.I,
)
_LOW_LIGHT_RE = re.compile(
    r"\b(?:night\w*|midnight|low[- ]light|dim(?:ly)?[- ]lit|candle\w*|concert|nightclub|moonlight|"
    r"starlight|blue hour|dark (?:room|bar|alley|street))\b",
    re.I,
)
_INDOOR_RE = re.compile(
    r"\b(?:indoors?|interior|dusk|twilight|overcast|rainy|restaurant|cafe|bar|office|bedroom|kitchen)\b", re.I
)

# --photo-grain-style -> (iso, frame width mm); clean_digital stays on the digital camera.
_GRAIN_STYLE_FILM = {"fine_35mm": (100, 36.0), "medium_35mm": (400, 36.0), "heavy_16mm": (500, 10.26)}


def _explicit_iso(prompt: str) -> int:
    m = _ISO_RE.search(prompt or "")
    return int(np.clip(int(m.group(1)), 25, 102400)) if m else 0


def film_stock_for_prompt(prompt: str, grain_style: str = "none", iso: int = 0) -> FilmStock:
    """Pick an emulsion and frame size from film-stock cues in the prompt."""
    p = (prompt or "").lower()
    mono = bool(_MONO_RE.search(p))
    width, clump = 36.0, 12.0
    if re.search(r"\b16 ?mm\b", p):
        stock_iso, width = 500, 10.26
    elif re.search(r"\b(?:super ?8|8 ?mm)\b", p):
        stock_iso, width = 200, 5.79
    elif re.search(r"\b(?:polaroid|instax)\b", p):
        stock_iso, width, clump = 640, 79.0, 15.0
    elif re.search(r"\b(?:120 film|medium format)\b", p):
        stock_iso, width = 400, 56.0
    elif "ektar" in p:
        stock_iso = 100
    elif re.search(r"\bt-?max\b", p):
        stock_iso = 100
    elif re.search(r"\b(?:cinestill|vision ?3|800t|500t)\b", p):
        stock_iso = 800
    else:
        stock_iso = 400
    if grain_style in _GRAIN_STYLE_FILM:
        stock_iso, width = _GRAIN_STYLE_FILM[grain_style]
    stock_iso = int(iso or _explicit_iso(p) or stock_iso)
    granularity = 17.0 if re.search(r"\b(?:tri-?x|hp5)\b", p) else 0.0
    return FilmStock(
        f"{'bw' if mono else 'color'}_{stock_iso}",
        iso=stock_iso,
        monochrome=mono,
        granularity=granularity,
        frame_width_mm=width,
        clump_um=clump,
    )


def _non_photographic(prompt: str) -> bool:
    from config.styles import detect_all_mediums, detect_medium_ids

    families = {m.family for m in detect_all_mediums(prompt)}
    return bool(families & {"2d", "traditional", "3d"}) or bool(detect_medium_ids(prompt, include_photography=False))


def plan_finish(
    prompt: str,
    *,
    camera: str = "auto",
    iso: int = 0,
    grain_style: str = "none",
    human_media: str = "none",
    vignette: float = 0.0,
    chromatic_aberration: float = 0.0,
) -> CameraProfile | FilmStock | None:
    """Route by medium: photo -> digital camera, film -> emulsion, anything else -> None (no capture finish).

    ``camera`` forces a preset (``phone``/``dslr``/``compact``/``film``); ``none`` disables.
    ``vignette`` / ``chromatic_aberration`` are the explicit 0-1 knobs (see ``user_lens``).
    """
    cam = str(camera or "auto").lower().strip()
    style = str(grain_style or "none").lower().strip()
    p = prompt or ""
    if cam in ("none", "off"):
        return None
    if cam == "auto":
        if _non_photographic(p):
            return None
        if style in _GRAIN_STYLE_FILM or str(human_media).lower() == "film" or _FILM_RE.search(p):
            cam = "film"
        elif str(human_media).lower() == "dslr":
            cam = "dslr"
        elif _PHONE_RE.search(p):
            cam = "phone"
        elif _COMPACT_RE.search(p):
            cam = "compact"
        else:
            cam = "dslr"
    if cam == "film":
        plan: CameraProfile | FilmStock = film_stock_for_prompt(p, style, iso)
    else:
        profile = CAMERA_PRESETS.get(cam, CAMERA_PRESETS["dslr"])
        chosen = iso or _explicit_iso(p)
        if not chosen:
            boost = 16 if _LOW_LIGHT_RE.search(p) else 4 if _INDOOR_RE.search(p) else 1
            chosen = profile.base_iso * boost
        plan = replace(profile, iso=int(chosen))
    return replace(plan, lens=user_lens(plan.lens, vignette, chromatic_aberration))


def apply_finish(
    image: np.ndarray,
    plan: CameraProfile | FilmStock | None,
    *,
    seed: int = 0,
    jpeg_quality: int = 0,
    jpeg: bool = True,
) -> np.ndarray:
    """Run a ``plan_finish`` result on an sRGB [0, 255] image (``None`` returns it unchanged)."""
    if isinstance(plan, FilmStock):
        return simulate_film(image, plan, seed=seed)
    if isinstance(plan, CameraProfile):
        if jpeg_quality > 0:
            plan = replace(plan, jpeg_quality=int(np.clip(jpeg_quality, 1, 100)))
        return simulate_capture(image, plan, seed=seed, jpeg=jpeg)
    return image


def to_uint8(image: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Quantise [0, 255] to uint8 once; with ``rng``, TPDF dither (two uniform LSB) to avoid banding."""
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    x = arr.astype(np.float32)
    if rng is not None:
        x = x + (rng.random(x.shape, dtype=np.float32) - rng.random(x.shape, dtype=np.float32))
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def save_rgb(image: np.ndarray, path: str | Path, *, sixteen_bit: bool = False, seed: int = 0) -> np.ndarray:
    """Save an RGB [0, 255] image; float input is dithered to 8 bit or written as 16-bit PNG/TIFF.

    Returns the 8-bit image (for grids and previews).
    """
    path = Path(path)
    u8 = to_uint8(image, np.random.default_rng(int(seed)))
    if sixteen_bit and path.suffix.lower() in (".png", ".tif", ".tiff"):
        u16 = np.clip(np.rint(np.asarray(image, dtype=np.float32) * (65535.0 / 255.0)), 0, 65535).astype(np.uint16)
        ok, buf = cv2.imencode(path.suffix, cv2.cvtColor(u16, cv2.COLOR_RGB2BGR))
        if ok:
            path.write_bytes(buf.tobytes())
            return u8
    from PIL import Image

    Image.fromarray(u8).save(path)
    return u8


@dataclass(slots=True)
class ArtisticPostConfig:
    """All artistic post-processing knobs in one place."""

    composition_mode: str = "none"  # "none"|"rule_of_thirds"|"golden_ratio"|"dynamic_symmetry"
    composition_strength: float = 0.0

    value_structure: bool = False
    value_shadow_lift: float = 0.0  # 0-0.15
    value_highlight_roll: float = 0.0  # 0-0.15
    value_midtone_contrast: float = 0.0  # 0-0.3

    lost_found_strength: float = 0.0  # 0-1
    lost_found_seed: int | None = None

    micro_detail: float = 0.0  # 0-1


def apply_value_structure(
    image: np.ndarray,
    *,
    shadow_lift: float = 0.0,
    highlight_roll: float = 0.0,
    midtone_contrast: float = 0.0,
) -> np.ndarray:
    """
    Enforce value discipline:
    - shadow_lift: raise the black point slightly (0.0-0.15) — prevents crushed shadows
    - highlight_roll: compress highlights (0.0-0.15) — prevents blown-out whites
    - midtone_contrast: boost separation in the midtone range (0.0-0.3)

    All operations are luminance-only to preserve hue/saturation.
    """
    if shadow_lift <= 0 and highlight_roll <= 0 and midtone_contrast <= 0:
        return image

    is_uint = image.dtype == np.uint8
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 255.0)
    lum = rgb_to_luminance(img) / 255.0  # [0,1]

    new_lum = lum.copy()

    # Shadow lift: remap [0, lift_point] -> [lift_floor, lift_point]
    if shadow_lift > 0:
        sl = float(min(0.15, shadow_lift))
        shadow_mask = np.clip(1.0 - lum / max(sl * 3, 0.1), 0.0, 1.0)
        new_lum = new_lum + sl * shadow_mask

    # Highlight roll-off: compress [1-roll_point, 1] -> [1-roll_point, 1-roll_floor]
    if highlight_roll > 0:
        hr = float(min(0.15, highlight_roll))
        hi_mask = np.clip((lum - (1.0 - hr * 3)) / max(hr * 3, 0.01), 0.0, 1.0)
        new_lum = new_lum - hr * hi_mask

    # Midtone contrast: S-curve in [0.2, 0.8] range
    if midtone_contrast > 0:
        mc = float(min(0.3, midtone_contrast))
        mid_mask = np.clip(1.0 - np.abs(lum - 0.5) / 0.3, 0.0, 1.0)
        # Push midtones away from 0.5
        push = np.sign(lum - 0.5) * mc * mid_mask
        new_lum = new_lum + push

    new_lum = np.clip(new_lum, 0.0, 1.0)

    # Rescale RGB channels to match new luminance
    old_lum = np.clip(rgb_to_luminance(img) / 255.0, 1e-4, 1.0)
    scale = (new_lum / old_lum)[..., np.newaxis]
    out = np.clip(img * scale, 0.0, 255.0)

    return out.astype(np.uint8) if is_uint else out


def _rule_of_thirds_map(h: int, w: int) -> np.ndarray:
    """
    Returns (H, W) weight map that is highest at rule-of-thirds intersections.
    Used to nudge the image's visual weight toward these points.
    """
    yy = np.linspace(0, 1, h)
    xx = np.linspace(0, 1, w)
    grid_x, grid_y = np.meshgrid(xx, yy)

    # Four intersection points
    thirds = [1 / 3, 2 / 3]
    weight = np.zeros((h, w), dtype=np.float32)
    sigma = 0.12  # spread of each hotspot
    for ty in thirds:
        for tx in thirds:
            d2 = (grid_y - ty) ** 2 + (grid_x - tx) ** 2
            weight += np.exp(-d2 / (2 * sigma**2))

    return weight / (weight.max() + 1e-8)


def _golden_ratio_map(h: int, w: int) -> np.ndarray:
    """
    Returns (H, W) weight map based on golden ratio spiral approximation.
    Hotspots at golden ratio intersections (phi = 1.618).
    """
    phi = 1.618033988749895
    yy = np.linspace(0, 1, h)
    xx = np.linspace(0, 1, w)
    grid_x, grid_y = np.meshgrid(xx, yy)

    # Golden ratio intersection points
    gx = [1.0 / phi, 1.0 - 1.0 / phi]
    gy = [1.0 / phi, 1.0 - 1.0 / phi]
    weight = np.zeros((h, w), dtype=np.float32)
    sigma = 0.10
    for ty in gy:
        for tx in gx:
            d2 = (grid_y - ty) ** 2 + (grid_x - tx) ** 2
            weight += np.exp(-d2 / (2 * sigma**2))

    return weight / (weight.max() + 1e-8)


def apply_compositional_director(
    image: np.ndarray,
    mode: str = "rule_of_thirds",
    strength: float = 0.15,
) -> np.ndarray:
    """
    Nudge the image's visual weight toward compositional guide points.

    This works by slightly boosting luminance/contrast at the guide intersections
    and very subtly reducing it at the center — counteracting the AI's center bias.

    mode: "rule_of_thirds" | "golden_ratio" | "dynamic_symmetry"
    strength: 0-1 (0.1-0.25 is subtle but effective)
    """
    if strength <= 0 or mode == "none":
        return image

    is_uint = image.dtype == np.uint8
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 255.0)
    h, w = img.shape[:2]
    s = float(min(1.0, max(0.0, strength)))

    if mode == "golden_ratio":
        guide_map = _golden_ratio_map(h, w)
    elif mode == "dynamic_symmetry":
        # Diagonal emphasis: boost along the main diagonals
        yy = np.linspace(0, 1, h)
        xx = np.linspace(0, 1, w)
        grid_x, grid_y = np.meshgrid(xx, yy)
        d1 = np.exp(-((grid_y - grid_x) ** 2) / (2 * 0.08**2))
        d2 = np.exp(-((grid_y - (1 - grid_x)) ** 2) / (2 * 0.08**2))
        guide_map = np.clip(d1 + d2, 0, 1).astype(np.float32)
        guide_map /= guide_map.max() + 1e-8
    else:  # rule_of_thirds (default)
        guide_map = _rule_of_thirds_map(h, w)

    # Center suppression map: reduce center weight slightly
    yy = np.linspace(-1, 1, h)
    xx = np.linspace(-1, 1, w)
    gx, gy = np.meshgrid(xx, yy)
    center_weight = np.exp(-(gx**2 + gy**2) / (2 * 0.4**2))
    center_suppress = 1.0 - 0.08 * s * center_weight

    # Apply: boost guide points, suppress center
    boost = 1.0 + s * 0.12 * guide_map[..., np.newaxis]
    suppress = center_suppress[..., np.newaxis]
    out = np.clip(img * boost * suppress, 0.0, 255.0)

    return out.astype(np.uint8) if is_uint else out


def apply_lost_and_found_edges(
    image: np.ndarray,
    strength: float = 0.3,
    seed: int | None = None,
) -> np.ndarray:
    """
    Vary edge sharpness across the image to mimic human artistic mark-making.

    In traditional art, edges are not uniformly sharp — some are "lost" (soft,
    blending into surroundings) and some are "found" (crisp, defining form).
    AI images tend to have uniformly sharp or uniformly soft edges, which reads
    as mechanical.

    Method:
    - Detect edges using a simple gradient magnitude
    - Generate a smooth random mask that selects which edges to soften
    - Apply local blur only to the "lost" edge regions
    - Keep "found" edges sharp

    strength: 0-1 (0.2-0.5 is natural; above 0.7 becomes painterly)
    """
    if strength <= 0:
        return image

    try:
        from PIL import Image as _PIL
        from scipy.ndimage import gaussian_filter, sobel
    except ImportError:
        return image

    rng = np.random.default_rng(seed)
    is_uint = image.dtype == np.uint8
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 255.0)
    h, w = img.shape[:2]
    s = float(min(1.0, max(0.0, strength)))

    # Detect edges
    gray = rgb_to_luminance(img)
    edge_x = sobel(gray, axis=1)
    edge_y = sobel(gray, axis=0)
    edge_mag = np.sqrt(edge_x**2 + edge_y**2)
    edge_mag = edge_mag / (edge_mag.max() + 1e-8)  # [0, 1]

    # Generate smooth "lost" mask — which edges to soften
    # Use low-frequency noise so the pattern is organic, not random speckle
    grid_h, grid_w = max(4, h // 8), max(4, w // 8)
    noise = rng.uniform(0, 1, (grid_h, grid_w)).astype(np.float32)
    lost_mask = (
        np.array(_PIL.fromarray((noise * 255).astype(np.uint8)).resize((w, h), _PIL.BILINEAR), dtype=np.float32) / 255.0
    )

    # Only apply to actual edge regions (not flat areas)
    lost_mask = lost_mask * edge_mag

    # Threshold: top s*50% of edge pixels get softened
    threshold = np.percentile(lost_mask, 100 - s * 50)
    lost_binary = (lost_mask > threshold).astype(np.float32)

    # Smooth the binary mask to avoid hard transitions
    lost_smooth = gaussian_filter(lost_binary, sigma=2.0)
    lost_smooth = np.clip(lost_smooth, 0.0, 1.0)[..., np.newaxis]

    # Create softened version
    blur_radius = 0.8 + s * 1.5
    img_soft = gaussian_filter(img, sigma=(blur_radius, blur_radius, 0))

    # Blend: lost edges get softened, found edges stay sharp
    out = img * (1.0 - lost_smooth) + img_soft * lost_smooth
    out = np.clip(out, 0.0, 255.0)

    return out.astype(np.uint8) if is_uint else out


def apply_micro_detail(
    image: np.ndarray,
    strength: float = 0.3,
) -> np.ndarray:
    """
    Recover fine texture detail using a high-frequency emphasis on luminance only.
    Unlike standard sharpening, this operates at a very fine scale (radius=0.5)
    and only on luminance — no RGB halos.

    strength: 0-1 (0.2-0.5 is natural; above 0.7 is aggressive)
    """
    if strength <= 0:
        return image

    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        return image

    is_uint = image.dtype == np.uint8
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 255.0)
    s = float(min(1.0, max(0.0, strength)))

    lum = rgb_to_luminance(img)
    blur = gaussian_filter(lum, sigma=0.5, mode="reflect")
    hf = lum - blur  # high-frequency component
    new_lum = np.clip(lum + s * hf * 0.8, 1e-3, 255.0)

    scale = (new_lum / (lum + 1e-3))[..., np.newaxis]
    out = np.clip(img * scale, 0.0, 255.0)

    return out.astype(np.uint8) if is_uint else out


def apply_artistic_pipeline(
    image: np.ndarray,
    cfg: ArtisticPostConfig,
) -> np.ndarray:
    """Value structure -> compositional director -> lost-and-found edges -> micro-detail."""
    out = image
    if (
        cfg.value_structure
        or cfg.value_shadow_lift > 0
        or cfg.value_highlight_roll > 0
        or cfg.value_midtone_contrast > 0
    ):
        out = apply_value_structure(
            out,
            shadow_lift=cfg.value_shadow_lift,
            highlight_roll=cfg.value_highlight_roll,
            midtone_contrast=cfg.value_midtone_contrast,
        )
    if cfg.composition_mode != "none" and cfg.composition_strength > 0:
        out = apply_compositional_director(out, cfg.composition_mode, cfg.composition_strength)
    if cfg.lost_found_strength > 0:
        out = apply_lost_and_found_edges(out, cfg.lost_found_strength, cfg.lost_found_seed)
    if cfg.micro_detail > 0:
        out = apply_micro_detail(out, cfg.micro_detail)
    return out


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


def _detect_face_rects_bgr(bgr: np.ndarray, *, min_side: int = 48) -> list[tuple[int, int, int, int]]:
    from utils.quality.test_time_pick import _cv2_face_cascade

    cascade = _cv2_face_cascade()
    if cascade is None:
        return []
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(min_side, min_side))
    if faces is None or len(faces) == 0:
        return []
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def enhance_faces_in_rgb(
    rgb: np.ndarray,
    *,
    padding: float = 0.25,
    sharpen_amount: float = 0.35,
    micro_contrast: float = 1.04,
    max_faces: int = 4,
    min_face_side: int = 48,
) -> np.ndarray:
    """
    Return a copy of *rgb* (H,W,3 uint8) with up to *max_faces* frontal regions sharpened.

    *padding* expands each bbox by this fraction of max(w,h) before processing.
    *micro_contrast* is passed to ``quality.contrast`` on the patch (1.0 = off).
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return rgb
    if rgb.dtype != np.uint8:
        rgb = np.clip(np.asarray(rgb), 0, 255).astype(np.uint8)

    try:
        import cv2
    except ImportError:
        return rgb

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    rects = _detect_face_rects_bgr(bgr, min_side=min_face_side)
    if not rects:
        return rgb

    out = rgb.copy()
    H, W = out.shape[:2]

    for x, y, w, h in rects[: max(1, int(max_faces))]:
        pad = int(max(w, h) * float(padding))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)
        if x2 <= x1 or y2 <= y1:
            continue

        patch = out[y1:y2, x1:x2].copy()
        ph, pw = patch.shape[:2]
        # Ellipse mask centered on original face (not full padded rect) for soft blend
        fcx = x + w // 2 - x1
        fcy = y + h // 2 - y1
        rx = max(4, int(w * 0.55))
        ry = max(4, int(h * 0.65))

        enhanced = sharpen(patch, amount=float(sharpen_amount))
        if abs(float(micro_contrast) - 1.0) >= 1e-6:
            enhanced = contrast(enhanced.astype(np.float32), factor=float(micro_contrast))
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        mask = np.zeros((ph, pw), dtype=np.float32)
        cv2.ellipse(mask, (fcx, fcy), (rx, ry), 0, 0, 360, 1.0, thickness=-1)
        sigma = max(3.0, min(rx, ry) * 0.25)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
        mask = np.clip(mask, 0.0, 1.0)
        m3 = mask[:, :, np.newaxis]
        blended = (m3 * enhanced.astype(np.float32) + (1.0 - m3) * patch.astype(np.float32)).round()
        out[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    return out


def blend_reference_rgb(
    base: np.ndarray,
    reference: np.ndarray,
    *,
    alpha: float = 0.12,
) -> np.ndarray:
    """
    Crude whole-image linear blend toward a resized *reference* (color/style pull).
    *alpha* in [0, 0.5] recommended; no learned identity lock.
    """
    if base.ndim != 3 or base.shape[2] != 3:
        return base
    a = float(np.clip(alpha, 0.0, 0.5))
    if a <= 0:
        return base

    try:
        from PIL import Image
    except ImportError:
        return base

    b = np.clip(np.asarray(base), 0, 255).astype(np.uint8)
    H, W = b.shape[:2]
    ref = np.clip(np.asarray(reference), 0, 255).astype(np.uint8)
    if ref.ndim != 3 or ref.shape[2] != 3:
        return base
    ref_im = Image.fromarray(ref, mode="RGB").resize((W, H), Image.Resampling.LANCZOS)
    r = np.asarray(ref_im, dtype=np.float32)
    out = (1.0 - a) * b.astype(np.float32) + a * r
    return np.clip(out.round(), 0, 255).astype(np.uint8)


ResizeMode = Literal["stretch", "center_crop", "saliency_crop"]


def _resize_np(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    pil = Image.fromarray(img.astype(np.uint8), mode="RGB")
    pil = pil.resize((int(out_w), int(out_h)), Image.Resampling.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def _center_crop_box(h: int, w: int, out_h: int, out_w: int) -> tuple[int, int, int, int]:
    target_ar = float(out_w) / max(1.0, float(out_h))
    src_ar = float(w) / max(1.0, float(h))
    if src_ar > target_ar:
        ch = h
        cw = max(1, int(round(h * target_ar)))
        y0 = 0
        x0 = max(0, (w - cw) // 2)
    else:
        cw = w
        ch = max(1, int(round(w / max(1e-6, target_ar))))
        x0 = 0
        y0 = max(0, (h - ch) // 2)
    return y0, x0, ch, cw


def _face_boost_map(img: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    try:
        import cv2
    except Exception:
        return np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    out = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    for x, y, fw, fh in faces:
        x2 = min(img.shape[1], x + fw)
        y2 = min(img.shape[0], y + fh)
        out[y:y2, x:x2] += float(strength)
    return out


def _saliency_map(img: np.ndarray, face_bias: float = 0.0) -> np.ndarray:
    # Lightweight saliency: edge magnitude + local contrast + optional face boost.
    arr = img.astype(np.float32) / 255.0
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    gx = np.abs(np.gradient(gray, axis=1))
    gy = np.abs(np.gradient(gray, axis=0))
    edge = gx + gy
    contrast = np.abs(gray - gray.mean())
    sal = 0.7 * edge + 0.3 * contrast
    if face_bias > 0:
        sal = sal + _face_boost_map(img, face_bias)
    sal = sal - sal.min()
    m = float(sal.max())
    if m > 1e-8:
        sal = sal / m
    return sal.astype(np.float32)


def _best_saliency_crop_box(
    img: np.ndarray, out_h: int, out_w: int, face_bias: float = 0.0
) -> tuple[int, int, int, int]:
    h, w = img.shape[:2]
    y0, x0, ch, cw = _center_crop_box(h, w, out_h, out_w)
    sal = _saliency_map(img, face_bias=face_bias)
    if cw == w and ch == h:
        return y0, x0, ch, cw
    if ch == h:
        # Horizontal sliding window
        col = sal.sum(axis=0)
        csum = np.concatenate([[0.0], np.cumsum(col)])
        best_x, best_val = x0, -1e30
        for x in range(0, max(1, w - cw + 1)):
            v = float(csum[x + cw] - csum[x])
            if v > best_val:
                best_val, best_x = v, x
        return 0, int(best_x), ch, cw
    # Vertical sliding window
    row = sal.sum(axis=1)
    csum = np.concatenate([[0.0], np.cumsum(row)])
    best_y, best_val = y0, -1e30
    for y in range(0, max(1, h - ch + 1)):
        v = float(csum[y + ch] - csum[y])
        if v > best_val:
            best_val, best_y = v, y
    return int(best_y), 0, ch, cw


def fit_image_to_size(
    img: np.ndarray,
    out_h: int,
    out_w: int,
    *,
    mode: ResizeMode = "stretch",
    saliency_face_bias: float = 0.0,
) -> np.ndarray:
    """Resize RGB uint8 image using stretch, center-crop, or saliency-crop."""
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("fit_image_to_size expects HxWx3 image")
    if out_h <= 0 or out_w <= 0:
        raise ValueError("output size must be positive")
    h, w = img.shape[:2]
    if h == out_h and w == out_w:
        return img.astype(np.uint8)
    m = str(mode or "stretch").lower()
    if m == "stretch":
        return _resize_np(img, out_h, out_w)
    if m == "center_crop":
        y0, x0, ch, cw = _center_crop_box(h, w, out_h, out_w)
    elif m == "saliency_crop":
        y0, x0, ch, cw = _best_saliency_crop_box(img, out_h, out_w, face_bias=float(saliency_face_bias))
    else:
        raise ValueError(f"Unknown resize mode: {mode}")
    crop = img[y0 : y0 + ch, x0 : x0 + cw]
    return _resize_np(crop, out_h, out_w)

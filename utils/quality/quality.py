"""Deterministic sRGB finishing helpers (contrast, saturation, tone, clarity, grades, filters).

Stochastic capture effects (sensor noise, film grain, lens) live in ``utils.quality.camera_isp``.
"""

import numpy as np


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
        from utils.quality.camera_isp import NO_LENS, FilmStock, simulate_film

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
        from utils.quality.camera_isp import apply_diffusion_filter

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

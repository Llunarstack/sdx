# Quality and post-processing utilities for better-looking images.
# Inspired by ComfyUI, A1111, and common diffusion workflows.
from typing import Optional

import numpy as np


def add_film_grain(image: np.ndarray, amount: float = 0.02, seed: Optional[int] = None) -> np.ndarray:
    """
    Add subtle luminance grain to reduce the plastic/AI look. amount in [0.01, 0.04] is typical.
    image: (H, W, C) uint8 or float [0,255]. Returns same dtype.
    """
    if amount <= 0:
        return image
    rng = np.random.default_rng(seed)
    is_uint = image.dtype == np.uint8
    if is_uint:
        img = image.astype(np.float32)
    else:
        img = np.asarray(image, dtype=np.float32).copy()
    h, w, c = img.shape
    # Slightly less grain in shadows (lum factor) so it looks natural
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    lum = np.clip(gray / 255.0, 0.05, 1.0)
    noise = rng.normal(0, amount * 255, (h, w)).astype(np.float32) * (0.4 + 0.6 * lum)
    for i in range(c):
        img[:, :, i] = np.clip(img[:, :, i] + noise, 0, 255)
    if is_uint:
        return np.clip(img, 0, 255).astype(np.uint8)
    return img


def naturalize(
    image: np.ndarray,
    grain_amount: float = 0.015,
    micro_contrast: float = 1.02,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Light post-process to make the image look less AI-generated: subtle film grain + slight contrast.
    grain_amount: 0 = off, 0.01–0.03 typical; micro_contrast: 1.0 = off, 1.01–1.05 subtle.
    """
    if grain_amount <= 0 and abs(micro_contrast - 1.0) < 1e-6:
        return image
    out = image
    if abs(micro_contrast - 1.0) >= 1e-6:
        out = contrast(out, factor=micro_contrast)
        if out.dtype == np.float32 or out.dtype == np.float64:
            out = np.clip(out, 0, 255)
        else:
            out = out.astype(np.uint8)
    if grain_amount > 0:
        out = add_film_grain(out, amount=grain_amount, seed=seed)
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

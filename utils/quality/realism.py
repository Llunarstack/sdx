"""Photoreal critics and finishing: plastic-skin and waxy-highlight detectors (pooled into
``critics.ai_look_score``), a measured saturation tame-down and highlight shoulder, and a capture-finish lens
tied to the shot spec's focal length and aperture (``utils.prompt.realism.ShotSpec``)."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import cv2
import numpy as np

from utils.quality.finishing import lab_to_srgb, ramp, rgb_to_luminance, srgb_to_lab

# ---------------------------------------------------------------------------------------------------------
# Critics
# ---------------------------------------------------------------------------------------------------------


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    """Skin-tone pixels: the YCrCb box of Chai & Ngan (1999), opened to drop speckle, eroded off edges."""
    u8 = np.clip(np.asarray(rgb, dtype=np.float32)[..., :3], 0, 255).astype(np.uint8)
    ycc = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    m = ((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127) & (y >= 50)).astype(np.uint8)
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return cv2.erode(m, k) > 0


def _fine_level(gray: np.ndarray, sel: np.ndarray) -> float:
    """Robust sigma (1.4826 * MAD, 8-bit units) of the finest band (image minus Gaussian sigma 1) in ``sel``."""
    fine = gray - cv2.GaussianBlur(gray, (0, 0), 1.0)
    v = fine[sel]
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def _skin(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3 or min(arr.shape[:2]) < 48:
        return None
    mask = skin_mask(arr)
    if int(mask.sum()) < max(1500, int(0.02 * mask.size)):
        return None
    return rgb_to_luminance(arr[..., :3].astype(np.float32)), mask


def score_plastic_skin(rgb: np.ndarray) -> float | None:
    """Fine texture left on skin (pores, fuzz, capture noise); the airbrushed generated look has almost none.

    Captured skin keeps a fine-band sigma of ~1.5-3 levels; decoder output is often < 0.5. Sigma <= 0.5
    scores 0, >= 1.5 scores 1. None without enough skin (2% of the frame).
    """
    s = _skin(rgb)
    if s is None:
        return None
    gray, mask = s
    return ramp(_fine_level(gray, mask), 0.5, 1.5)


def score_waxy_highlights(rgb: np.ndarray) -> float | None:
    """Specular sheen on skin that is smooth and clipped (wax / plastic) rather than textured.

    The sheen is the brightest 5% of skin (and >= 170). Real skin highlights break up over pores (fine
    sigma >= 1.4 scores 1, <= 0.4 scores 0); a share of clipped channels above 5% (40% = 0) multiplies in.
    """
    s = _skin(rgb)
    if s is None:
        return None
    gray, mask = s
    near = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    thr = max(float(np.percentile(gray[mask], 95)), 170.0)
    sheen = near & (gray >= thr)
    if int(sheen.sum()) < 100:
        return None
    clipped = float((np.asarray(rgb)[..., :3].max(axis=-1)[sheen] >= 250).mean())
    return ramp(_fine_level(gray, sheen), 0.4, 1.4) * (1.0 - ramp(clipped, 0.05, 0.4))


# ---------------------------------------------------------------------------------------------------------
# Finishing (applied before the capture finish, only for photographic prompts)
# ---------------------------------------------------------------------------------------------------------


def tame_saturation(image: np.ndarray, strength: float = 0.6, *, vivid_ok: bool = False) -> np.ndarray:
    """Soft-knee CIELAB chroma compression, only when the 99th-percentile chroma is past what cameras give.

    Off below C*99 = 80 (105 for a requested vivid look), full strength by +35. Above the knee (C* 60, resp.
    85) chroma follows ``knee + d / (1 + a d / 40)``; hue and lightness are kept, pixels below the knee
    are untouched.
    """
    x = np.asarray(image, dtype=np.float32)[..., :3]
    shift = 25.0 if vivid_ok else 0.0
    step = max(1, max(x.shape[:2]) // 256)
    lab_small = srgb_to_lab(x[::step, ::step] / 255.0)
    c99 = float(np.percentile(np.hypot(lab_small[..., 1], lab_small[..., 2]), 99))
    a = float(strength) * ramp(c99, 80.0 + shift, 115.0 + shift)
    if a <= 0.0:
        return x
    lab = srgb_to_lab(x / 255.0)
    c = np.hypot(lab[..., 1], lab[..., 2])
    knee = 60.0 + shift
    sel = c > knee
    d = c[sel] - knee
    scale = (knee + d / (1.0 + a * d / 40.0)) / c[sel]
    sub = lab[sel]
    sub[:, 1:] *= scale[:, None]
    out = x.copy()
    out[sel] = (lab_to_srgb(sub) * 255.0).astype(np.float32)
    return out


def highlight_rolloff(image: np.ndarray, strength: float = 0.6, knee: float = 0.82) -> np.ndarray:
    """Filmic shoulder above ``knee`` (luminance, 0-1) with the desaturation film shows near clip.

    ``u -> u (1 - k u / 2)`` on the part above the knee (monotone for k <= 1, k = 0.35 * strength), applied
    as a luminance gain so hue holds; near-clipped pixels blend a little toward grey. A no-op when less
    than 0.2% of the frame is above 0.95.
    """
    x = np.asarray(image, dtype=np.float32)[..., :3] / 255.0
    y = rgb_to_luminance(x)
    if float((y > 0.95).mean()) < 0.002 or strength <= 0.0:
        return x * 255.0
    k = 0.35 * float(min(strength, 1.0))
    u = np.clip((y - knee) / (1.0 - knee), 0.0, None)
    y_new = np.where(y > knee, knee + (1.0 - knee) * u * (1.0 - k * np.minimum(u, 1.0) / 2.0), y)
    out = x * (y_new / np.maximum(y, 1e-6))[..., None]
    w = 0.5 * k * np.clip((x.max(axis=-1) - 0.9) / 0.1, 0.0, 1.0) ** 2
    out = out + w[..., None] * (y_new[..., None] - out)
    return np.clip(out, 0.0, 1.0) * 255.0


def realism_correct(image: np.ndarray, strength: float = 0.6, *, vivid_ok: bool = False) -> np.ndarray:
    """Saturation tame-down then highlight shoulder; float32 [0, 255]. Grain, vignette and CA come from the
    capture finish at the shot spec's ISO and focal length (``tune_finish``)."""
    return highlight_rolloff(tame_saturation(image, strength, vivid_ok=vivid_ok), strength)


def lens_for_focal(lens: Any, focal_mm: float, aperture: float, *, keep_vignette: bool = False, keep_ca: bool = False):
    """``finishing.Lens`` with the full-frame field of view of ``focal_mm``; vignetting grows at wide
    apertures (x1.3 at f/1, x0.5 by f/9), lateral CA with shorter focal lengths (x35/f, 0.5-1.6)."""
    half_fov = math.degrees(math.atan(21.63 / max(float(focal_mm), 8.0)))
    lens = replace(lens, half_fov_deg=float(half_fov))
    if not keep_vignette:
        lens = replace(lens, vignette=float(min(1.0, lens.vignette * np.clip(1.4 - 0.1 * aperture, 0.5, 1.3))))
    if not keep_ca:
        s = float(np.clip(35.0 / max(float(focal_mm), 8.0), 0.5, 1.6))
        lens = replace(lens, ca_red=lens.ca_red * s, ca_blue=lens.ca_blue * s)
    return lens


def tune_finish(finish: Any, spec: Any, *, user_vignette: float = 0.0, user_ca: float = 0.0):
    """Fit a ``plan_finish`` result's lens to the shot spec (phones keep their own fixed lens)."""
    if finish is None or spec is None or getattr(spec, "camera", "") == "phone" or not hasattr(finish, "lens"):
        return finish
    lens = lens_for_focal(
        finish.lens, spec.focal_mm, spec.aperture, keep_vignette=user_vignette > 0, keep_ca=user_ca > 0
    )
    return replace(finish, lens=lens)


def realism_strength(args: Any) -> float:
    """Correction strength: 0.6 for ``--realism auto``, 1.0 for ``max``."""
    return 1.0 if str(getattr(args, "realism", "auto") or "auto").lower() == "max" else 0.6

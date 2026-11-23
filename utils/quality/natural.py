"""``--natural``: a critic-gated finishing pass that removes measurable AI-look tells, one at a time.

For each image: score every tell with :func:`utils.quality.critics.ai_look_score` (decoder grid, missing or flat
noise floor, off-natural detail spectrum, over-smoothing, oversaturation, halos, plastic skin, waxy highlights),
then, weakest tell first, try the fixes meant for that tell and keep one only when

* that tell's own score rises,
* the pooled naturalness rises by at least ``MIN_GAIN``, and
* the content does not move (blurred-luma drift under ``MAX_DRIFT``), so no fix can win by repainting the image.

Candidates are judged on a centred 512 px crop (a full-frame score costs ~0.8 s); the full image is scored before
and after, and the pass is undone if the full-frame score got worse. Last, a blue-noise dither takes it to 8 bit
(no banding). Media the critics treat as 2-D/3-D skip the capture tells, so illustrations get no sensor noise.

This removes *measurable* tells only. Content errors (hands, text, physics) and a weak checkpoint are untouched.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage

from utils.quality.critics import GRID_PERIOD, ai_look_score

THRESHOLD = 0.75  # tells scoring at or above this are left alone
MIN_GAIN = 0.005  # pooled naturalness a fix must add
MAX_DRIFT = 0.02  # mean |blurred luma| change allowed per fix, in [0, 1]
MAX_FIXES = 5
CROP = 512

Fix = Callable[[np.ndarray, float, int], np.ndarray]  # (float32 RGB [0, 255], strength, seed) -> same


def _ideas(name: str) -> Fix:
    """An image idea from ``utils.ideas`` as a [0, 255] fix."""

    def fix(img: np.ndarray, s: float, seed: int) -> np.ndarray:
        from utils.ideas import IDEAS, Ctx

        out = IDEAS[name].fn(np.clip(img / 255.0, 0.0, 1.0).astype(np.float32), Ctx(name, s, seed))
        return np.clip(out, 0.0, 1.0).astype(np.float32) * 255.0

    fix.__name__ = name
    return fix


def degrid(img: np.ndarray, s: float, seed: int) -> np.ndarray:
    """Subtract the decoder's periodic fingerprint: the high-pass residual averaged onto one ``GRID_PERIOD`` tile
    (content does not repeat, so it averages out; the grid does), tiled back over the frame."""
    p = GRID_PERIOD
    h, w = (img.shape[0] // p) * p, (img.shape[1] // p) * p
    if h < 2 * p or w < 2 * p:
        return img
    x = img.astype(np.float32) / 255.0
    resid = (x - ndimage.gaussian_filter(x, (p / 2, p / 2, 0), mode="reflect"))[:h, :w]
    tile = resid.reshape(h // p, p, w // p, p, 3).mean((0, 2))
    tile -= tile.mean((0, 1), keepdims=True)
    reps = (-(-img.shape[0] // p), -(-img.shape[1] // p), 1)
    pattern = np.tile(tile, reps)[: img.shape[0], : img.shape[1]]
    return np.clip(x - s * pattern, 0.0, 1.0) * 255.0


def sensor_noise(img: np.ndarray, s: float, seed: int) -> np.ndarray:
    """Poisson-Gaussian sensor noise in linear light (Foi et al. 2008): signal-dependent like a real capture,
    about 2 levels at mid-grey for ``s = 1`` (flat i.i.d. grain is itself a tell)."""
    from utils.quality.finishing import linear_to_srgb, poisson_gaussian_noise, srgb_to_linear

    lin = srgb_to_linear(np.clip(img / 255.0, 0.0, 1.0))
    noisy = poisson_gaussian_noise(
        lin, full_well_e=4000.0 / max(s, 0.05) ** 2, read_noise_e=3.0, rng=np.random.default_rng(seed)
    )
    return linear_to_srgb(np.clip(noisy, 0.0, 1.0)).astype(np.float32) * 255.0


def _tame_saturation(img: np.ndarray, s: float, seed: int) -> np.ndarray:
    from utils.quality.realism import tame_saturation

    return tame_saturation(img, s)


def chroma_scale(img: np.ndarray, s: float, seed: int) -> np.ndarray:
    """Scale every colour's OKLab chroma by ``1 - 0.2 s`` (hue and lightness kept): the global pump-down that
    colourfulness-wide oversaturation needs, where a knee on the peaks alone barely moves it."""
    from utils.ideas.image import _oklab, _rgb

    lab = _oklab(np.clip(img / 255.0, 0.0, 1.0).astype(np.float32))
    lab[..., 1:] *= max(0.0, 1.0 - 0.2 * s)
    return _rgb(lab) * 255.0


def _highlight_rolloff(img: np.ndarray, s: float, seed: int) -> np.ndarray:
    from utils.quality.realism import highlight_rolloff

    return highlight_rolloff(img, s)


def _reduce_halos(img: np.ndarray, s: float, seed: int) -> np.ndarray:
    from utils.quality.finishing import reduce_edge_halos

    return reduce_edge_halos(img, strength=0.3 * s)


TELL_FIXES: dict[str, tuple[Fix, ...]] = {
    "grid": (degrid,),
    "noise_floor": (sensor_noise,),
    "spectral_slope": (_ideas("img_spectral_1f"), _ideas("img_wavelet_detail")),
    "oversmoothing": (_ideas("img_texture_reinject"), _ideas("img_wavelet_detail"), sensor_noise),
    "oversaturation": (_tame_saturation, chroma_scale, _ideas("img_gamut_oklch")),
    "halo": (_reduce_halos,),
    "plastic_skin": (_ideas("img_texture_reinject"), sensor_noise),
    "waxy_highlights": (_ideas("img_specular_tame"), _highlight_rolloff),
}
STRENGTHS = (0.5, 1.0, 2.0)


@dataclass
class NaturalReport:
    before: float
    after: float
    fixes: list[tuple[str, str, float]] = field(default_factory=list)  # (tell, fix, strength)
    tells_before: dict[str, float] = field(default_factory=dict)
    tells_after: dict[str, float] = field(default_factory=dict)
    reverted: bool = False

    def line(self) -> str:
        weak = any(v < THRESHOLD for t, v in self.tells_before.items() if t in TELL_FIXES)
        done = ", ".join(f"{t} via {f} {s:g}" for t, f, s in self.fixes) or ("none passed" if weak else "none needed")
        note = " (reverted: full frame scored worse)" if self.reverted else ""
        return f"Natural pass: naturalness {self.before:.2f} -> {self.after:.2f}; fixes: {done}{note}"


def _crop(img: np.ndarray, size: int = CROP) -> np.ndarray:
    """Centred crop, offset on the grid period so a decoder fingerprint keeps its phase."""
    h, w = img.shape[:2]
    y0 = max(0, (h - size) // 2) // GRID_PERIOD * GRID_PERIOD
    x0 = max(0, (w - size) // 2) // GRID_PERIOD * GRID_PERIOD
    return img[y0 : y0 + size, x0 : x0 + size]


def _drift(a: np.ndarray, b: np.ndarray) -> float:
    """Mean change of blurred perceptual lightness (OKLab L): content moved, not colourfulness or grain."""
    from utils.ideas.image import _oklab

    def lightness(x: np.ndarray) -> np.ndarray:
        return ndimage.gaussian_filter(_oklab(np.clip(x[..., :3] / 255.0, 0.0, 1.0))[..., 0], 2.0)

    return float(np.abs(lightness(a) - lightness(b)).mean())


def _apply(chain: list[tuple[Fix, float]], img: np.ndarray, seed: int) -> np.ndarray:
    for fix, s in chain:
        img = fix(img, s, seed)
    return img


def natural_pass(img: np.ndarray, prompt: str = "", *, seed: int = 0, dither: bool = True):
    """Fix the measurable AI-look tells of one RGB image in [0, 255]; returns ``(image, NaturalReport)``.

    The image comes back as uint8 when ``dither`` (blue-noise to 8 bit), else float32 [0, 255].
    """
    src = np.clip(np.asarray(img, dtype=np.float32)[..., :3], 0.0, 255.0)
    full = ai_look_score(src, prompt)
    work, crop = src, _crop(src)
    crop_rep = ai_look_score(crop, prompt)
    rep = NaturalReport(before=full.natural, after=full.natural, tells_before=dict(full.parts))
    weak = sorted((v, t) for t, v in crop_rep.parts.items() if v < THRESHOLD and t in TELL_FIXES)
    # Cleanups (degrid, smoothing) also strip noise; on photographic media each one is also tried with a real
    # sensor-noise top-up, so two interacting tells can be fixed together.
    photo = "noise_floor" in crop_rep.parts
    for _, tell in weak[:MAX_FIXES]:
        current = crop_rep.parts.get(tell, 1.0)
        if current >= THRESHOLD:  # an earlier fix already cured it
            continue
        best = None
        for fix in TELL_FIXES[tell]:
            for s in STRENGTHS:
                chains = [[(fix, s)]]
                if photo and fix is not sensor_noise:
                    chains.append([(fix, s), (sensor_noise, 1.0)])
                for chain in chains:
                    cand = _apply(chain, crop, seed)
                    if _drift(crop, cand) > MAX_DRIFT:
                        continue
                    r = ai_look_score(cand, prompt)
                    beats = best is None or r.natural > best[0].natural
                    if r.parts.get(tell, 1.0) > current and r.natural >= crop_rep.natural + MIN_GAIN and beats:
                        best = (r, chain, cand)
        if best is not None:
            crop_rep, chain, crop = best
            work = _apply(chain, work, seed)
            rep.fixes.append((tell, "+".join(f.__name__ for f, _ in chain), chain[0][1]))
    if rep.fixes:
        final = ai_look_score(work, prompt)
        if final.natural < full.natural:
            work, final, rep.reverted = src, full, True
        rep.after, rep.tells_after = final.natural, dict(final.parts)
    else:
        rep.tells_after = dict(full.parts)
    if dither:
        from utils.ideas import IDEAS, Ctx

        out = IDEAS["img_blue_noise_dither"].fn(work / 255.0, Ctx("img_blue_noise_dither", 1.0, seed))
        return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8), rep
    return work, rep


def natural_pass_all(args: Any, images: list, report: dict) -> list:
    """``--natural`` over the run's final images (RGBA sprites are left alone); notes each result in ``report``."""
    prompt = str(getattr(args, "_raw_prompt_before_compose", "") or getattr(args, "prompt", "") or "")
    dither = not bool(getattr(args, "save_16bit", False))
    out, notes = [], []
    for k, img in enumerate(images):
        if np.asarray(img).shape[-1] == 4:
            out.append(img)
            continue
        fixed, rep = natural_pass(img, prompt, seed=int(getattr(args, "seed", 0) or 0) + k, dither=dither)
        print(rep.line(), file=sys.stderr)
        notes.append({"before": round(rep.before, 4), "after": round(rep.after, 4), "fixes": rep.fixes})
        out.append(fixed)
    report["natural"] = notes
    return out

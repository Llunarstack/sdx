"""
Best-of-N candidate scoring for ``sample.py --pick-best`` (and every other ranker in the repo).

``score_*`` functions score one RGB uint8 HWC image. A pick metric is a weighted set of terms
(``METRIC_WEIGHTS``); candidates are ranked per term and the weighted mean rank (a Borda count)
picks the winner, so a term's units never matter and a negligible difference cannot swing the
pick the way per-metric min-max scaling did. Terms that have no evidence for a prompt (no
expected text, no count, a requested grade for the cast term...) drop out and the rest
renormalise. Score candidates before any seeded grain / capture finish.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np

from utils.quality.finishing import rgb_to_luminance

try:
    from utils.native import (
        maybe_count_components_native,
        maybe_image_luma_stats_cuda,
        maybe_image_stats_native,
    )
except Exception:  # pragma: no cover - optional native path
    maybe_image_stats_native = None
    maybe_image_luma_stats_cuda = None
    maybe_count_components_native = None

_PEOPLE_WORDS = {
    "people",
    "persons",
    "person",
    "character",
    "characters",
    "subject",
    "subjects",
    "man",
    "men",
    "woman",
    "women",
    "girl",
    "girls",
    "boy",
    "boys",
    "kid",
    "kids",
    "child",
    "children",
}
_WORD_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_PEOPLE_TERMS_PATTERN = r"(?:people|persons|person|characters?|subjects?|men|women|girls?|boys?|kids?|children)"
_NUMBER_WORDS_PATTERN = "|".join(re.escape(word) for word in _WORD_NUMBERS.keys())
_PEOPLE_COUNT_PATTERNS = (
    re.compile(rf"\bexactly\s+(\d+)\s+{_PEOPLE_TERMS_PATTERN}\b"),
    re.compile(rf"\b(\d+)\s+{_PEOPLE_TERMS_PATTERN}\b"),
    re.compile(r"\b(\d+)girls?\b"),
    re.compile(r"\b(\d+)boys?\b"),
)
_PEOPLE_COUNT_WORD_PATTERN = re.compile(rf"\b({_NUMBER_WORDS_PATTERN})\s+{_PEOPLE_TERMS_PATTERN}\b")
_OBJECT_COUNT_PATTERNS = (
    re.compile(r"\bexactly\s+(\d+)\s+([a-z][a-z0-9_-]{2,})\b"),
    re.compile(r"\b(\d+)\s+([a-z][a-z0-9_-]{2,})\b"),
    re.compile(rf"\b({_NUMBER_WORDS_PATTERN})\s+([a-z][a-z0-9_-]{{2,}})\b"),
)
_OCR_WHITESPACE_RE = re.compile(r"\s+")
_MORPH_KERNEL_3 = np.ones((3, 3), dtype=np.uint8)


def rank01(scores: Sequence[float], rel_tol: float = 1e-6) -> list[float]:
    """Average ranks scaled to [0, 1] (worst 0, best 1; a single candidate gets 0.5).

    Values within ``rel_tol`` of each other (relative to their magnitude) tie, so float noise
    cannot order candidates. Invariant to any increasing rescaling of the scores; NaN ranks last.
    """
    x = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=-np.inf)
    n = int(x.size)
    if n == 0:
        return []
    if n == 1:
        return [0.5]
    order = np.argsort(x, kind="stable")
    xs = x[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] - xs[j] <= rel_tol * max(abs(xs[j]), abs(xs[j + 1]), 1e-12):
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return (ranks / (n - 1)).tolist()


def rank_aggregate(term_scores: Mapping[str, Sequence[float] | None], weights: Mapping[str, float]) -> list[float]:
    """Weighted Borda count: per-candidate weighted mean of ``rank01`` over the available terms."""
    lists = {k: v for k, v in term_scores.items() if v is not None and weights.get(k, 0.0) > 0}
    if not lists:
        n = max((len(v) for v in term_scores.values() if v is not None), default=0)
        return [0.5] * n
    total = sum(weights[k] for k in lists)
    ranked = {k: np.asarray(rank01(v)) for k, v in lists.items()}
    return (sum(weights[k] * ranked[k] for k in lists) / total).tolist()


def score_edge_sharpness(rgb_uint8: np.ndarray) -> float:
    """Higher = sharper (Laplacian variance on grayscale)."""
    if maybe_image_stats_native is not None:
        try:
            st = maybe_image_stats_native(rgb_uint8)
            if st and "laplacian_var" in st:
                return float(st["laplacian_var"])
        except Exception:
            pass
    try:
        import cv2
    except ImportError:
        return float(np.std(rgb_uint8.astype(np.float32)))
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] < 3:
        g = rgb_uint8.astype(np.float32)
    else:
        g = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def score_ocr_match(rgb_uint8: np.ndarray, expected: str) -> float:
    """
    Rough [0,1] score: ratio of expected alphanumeric chars found in OCR string.
    Higher = better match to expected text.
    """
    if not expected or not str(expected).strip():
        return 0.5
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return 0.5
    pil = Image.fromarray(rgb_uint8, mode="RGB")
    try:
        txt = pytesseract.image_to_string(pil) or ""
    except Exception:
        return 0.0
    exp = _OCR_WHITESPACE_RE.sub("", str(expected).upper())
    got_raw = _OCR_WHITESPACE_RE.sub("", txt.upper())
    got_chars = set(got_raw)
    if not exp:
        return 0.5
    hit = sum(1 for c in exp if c in got_chars)
    return hit / max(1, len(exp))


def score_clip_similarity(
    rgb_uint8_list: list[np.ndarray],
    prompt: str,
    device: str,
    model_id: str = "openai/clip-vit-base-patch32",
) -> list[float]:
    """Per-image CLIP image-text similarity (higher = better prompt alignment)."""
    if not prompt or not str(prompt).strip():
        return [0.5] * len(rgb_uint8_list)
    try:
        import torch
        from PIL import Image

        model, processor = _clip(model_id, str(device))
    except ImportError:
        return [0.5] * len(rgb_uint8_list)

    pil_images = [Image.fromarray(im, mode="RGB") for im in rgb_uint8_list]
    proc = processor(text=[prompt], images=pil_images, return_tensors="pt", padding=True, truncation=True)
    with torch.inference_mode():
        logits = model(**{k: v.to(device) for k, v in proc.items()}).logits_per_image  # (N, 1)
    return [float(s) for s in logits[:, 0].float().cpu().numpy()]


@lru_cache(maxsize=2)
def _clip(model_id: str, device: str) -> tuple[Any, Any]:
    from transformers import CLIPModel, CLIPProcessor

    processor = cast(Any, CLIPProcessor).from_pretrained(model_id)
    return cast(Any, CLIPModel).from_pretrained(model_id).to(device).eval(), processor


def score_exposure_balance(rgb_uint8: np.ndarray) -> float:
    """
    Lightweight verifier score (LANDSCAPE §2): penalize heavy highlight/shadow clipping.
    Higher = more pixels in a mid-tone band (less blown-out / crushed).
    """
    if maybe_image_luma_stats_cuda is not None:
        try:
            stc = maybe_image_luma_stats_cuda(rgb_uint8, clip_low=2, clip_high=253)
            if stc and "clip_ratio" in stc:
                return float(np.clip(1.0 - float(stc["clip_ratio"]), 0.0, 1.0))
        except Exception:
            pass
    if maybe_image_stats_native is not None:
        try:
            st = maybe_image_stats_native(rgb_uint8, clip_low=2, clip_high=253)
            if st and "clip_ratio" in st:
                return float(np.clip(1.0 - float(st["clip_ratio"]), 0.0, 1.0))
        except Exception:
            pass
    gray = np.mean(rgb_uint8, axis=-1)
    clipped = int(np.count_nonzero((gray <= 2.0) | (gray >= 253.0)))
    n = max(1, int(gray.size))
    return 1.0 - float(clipped) / float(n)


def score_color_cast_neutrality(rgb_uint8: np.ndarray, plan: dict | None = None) -> float | None:
    """[0, 1] midtone colour-cast score; None when the plan asks for a grade (the cast is intended).

    Cast magnitude is the max gray-world gain deviation over midtones: <= 0.05 scores 1, >= 0.35 scores 0.
    """
    if (plan or {}).get("grade"):
        return None
    return float(np.clip(1.0 - (color_cast_magnitude(rgb_uint8) - 0.05) / 0.30, 0.0, 1.0))


def color_cast_magnitude(rgb: np.ndarray) -> float:
    """Max |gray-world gain - 1| over midtones; 0 = neutral. Clipped shadows/highlights carry no cast."""
    img = np.asarray(rgb, dtype=np.float32)
    if img.ndim != 3 or img.shape[2] < 3:
        return 0.0
    luma = rgb_to_luminance(img)
    w = np.clip((luma - 20.0) / 40.0, 0.0, 1.0) * np.clip((235.0 - luma) / 40.0, 0.0, 1.0)
    wsum = float(w.sum())
    if wsum < 1e-3:
        return 0.0
    means = [float((img[..., c] * w).sum() / wsum) for c in range(3)]
    target = sum(means) / 3.0
    if target < 1e-3:
        return 0.0
    return max(abs(target / max(m, 1e-3) - 1.0) for m in means)


def score_dynamic_range_headroom(rgb_uint8: np.ndarray) -> float:
    """
    [0,1] score: high when highlights/shadows retain headroom and local contrast is present.
    """
    x = rgb_uint8.astype(np.float32)
    if x.ndim == 3 and x.shape[2] >= 3:
        lum = rgb_to_luminance(x)
    else:
        lum = x if x.ndim == 2 else x[..., 0]
    p1 = float(np.percentile(lum, 1))
    p99 = float(np.percentile(lum, 99))
    span = max(0.0, p99 - p1)
    span_score = float(np.clip(span / 190.0, 0.0, 1.0))
    clip_penalty = float(np.clip(((2.0 - p1) / 8.0), 0.0, 1.0) + np.clip(((p99 - 253.0) / 8.0), 0.0, 1.0))
    clip_penalty = min(1.0, clip_penalty)
    return float(np.clip(0.78 * span_score + 0.22 * (1.0 - clip_penalty), 0.0, 1.0))


def score_aesthetic_proxy(rgb_uint8: np.ndarray) -> float:
    """
    Heuristic [0,1] (also ``scripts/tools/manifest.py enrich``): mean of exposure balance,
    dynamic-range headroom and, when measurable, decoder-grid freedom.
    """
    from utils.quality.critics import score_decoder_grid

    parts = [float(score_exposure_balance(rgb_uint8)), float(score_dynamic_range_headroom(rgb_uint8))]
    grid = score_decoder_grid(rgb_uint8)
    if grid is not None:
        parts.append(float(grid))
    return float(np.clip(sum(parts) / len(parts), 0.0, 1.0))


def score_photographic_detail_balance(rgb_uint8: np.ndarray) -> float:
    """
    [0,1] score: prefers realistic detail (not mushy, not over-sharpened halos).
    """
    edge = float(score_edge_sharpness(rgb_uint8))
    # Target Laplacian variance band for photo-like sharpness.
    if edge <= 35.0:
        return float(np.clip(edge / 35.0, 0.0, 1.0) * 0.65)
    if edge <= 220.0:
        return float(np.clip(0.75 + 0.25 * ((edge - 35.0) / 185.0), 0.0, 1.0))
    over = min(1.0, (edge - 220.0) / 500.0)
    return float(np.clip(1.0 - 0.55 * over, 0.0, 1.0))


def score_skin_tone_naturalness(rgb_uint8: np.ndarray) -> float:
    """
    [0,1] skin-likeness sanity check in YCbCr space for human-photo prompts.
    Returns neutral 0.5 when no skin-like pixels are present.
    """
    x = rgb_uint8.astype(np.float32)
    if x.ndim != 3 or x.shape[2] < 3:
        return 0.5
    r = x[..., 0]
    g = x[..., 1]
    b = x[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    # Broad skin cluster bounds.
    mask = (cb >= 77.0) & (cb <= 135.0) & (cr >= 133.0) & (cr <= 175.0) & (y >= 35.0) & (y <= 235.0)
    ratio = float(np.mean(mask))
    if ratio < 0.005:
        return 0.5
    # Penalize extremely saturated skin candidates.
    sat_proxy = np.abs(r - g) + np.abs(g - b)
    sat_skin = float(np.percentile(sat_proxy[mask], 90)) if np.any(mask) else 0.0
    sat_score = float(np.clip(1.0 - ((sat_skin - 70.0) / 140.0), 0.0, 1.0))
    ratio_score = float(np.clip(ratio / 0.18, 0.0, 1.0))
    return float(np.clip(0.45 + 0.35 * ratio_score + 0.20 * sat_score, 0.0, 1.0))


def infer_expected_people_count(prompt: str) -> int:
    """
    Infer intended people count from prompt phrases.
    Returns 0 when no count intent is detected.
    """
    prompt_text = (prompt or "").lower()
    if not prompt_text:
        return 0
    for pattern in _PEOPLE_COUNT_PATTERNS:
        match = re.search(pattern, prompt_text)
        if match:
            try:
                return max(0, int(match.group(1)))
            except Exception:
                pass
    number_word_match = re.search(_PEOPLE_COUNT_WORD_PATTERN, prompt_text)
    if number_word_match:
        return int(_WORD_NUMBERS.get(number_word_match.group(1), 0))
    return 0


def infer_expected_object_count(prompt: str) -> tuple[int, str]:
    """
    Infer intended repeated non-people object count from prompt text.
    Returns (count, object_hint). count=0 means no object-count intent detected.
    """
    prompt_text = (prompt or "").lower()
    if not prompt_text:
        return 0, ""
    # Examples: "exactly 7 coins", "5 candles", "two windows"
    for pattern in _OBJECT_COUNT_PATTERNS:
        match = re.search(pattern, prompt_text)
        if not match:
            continue
        raw_number = match.group(1)
        noun = str(match.group(2) or "").strip().lower()
        if not noun or noun in _PEOPLE_WORDS:
            continue
        noun = noun.rstrip("s")
        try:
            if raw_number.isdigit():
                expected_count = int(raw_number)
            else:
                expected_count = int(_WORD_NUMBERS.get(raw_number, 0))
            if expected_count > 0:
                return expected_count, noun
        except Exception:
            pass
    return 0, ""


@lru_cache(maxsize=1)
def _cv2_face_cascade() -> Any:
    """OpenCV frontal-face Haar cascade, or None when OpenCV or the XML is unavailable."""
    try:
        import cv2

        cv2_any = cast(Any, cv2)
        cascade = cv2_any.CascadeClassifier(cv2_any.data.haarcascades + "haarcascade_frontalface_default.xml")
        return None if cascade.empty() else cascade
    except Exception:
        return None


@lru_cache(maxsize=1)
def _cv2_hog_people() -> Any:
    try:
        import cv2

        cv2_any = cast(Any, cv2)
        hog = cv2_any.HOGDescriptor()
        hog.setSVMDetector(cv2_any.HOGDescriptor_getDefaultPeopleDetector())
        return hog
    except Exception:
        return None


def estimate_people_count(rgb_uint8: np.ndarray) -> int:
    """
    Lightweight heuristic person estimate using OpenCV face/body detectors.
    """
    try:
        import cv2
    except ImportError:
        return -1
    try:
        gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    except Exception:
        return -1

    faces_n = 0
    face_cascade = _cv2_face_cascade()
    if face_cascade:
        try:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=4, minSize=(18, 18))
            faces_n = int(len(faces))
        except Exception:
            faces_n = 0

    body_n = 0
    hog = _cv2_hog_people()
    if hog:
        try:
            rects, _weights = hog.detectMultiScale(
                rgb_uint8,
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.05,
            )
            body_n = int(len(rects))
        except Exception:
            body_n = 0

    if faces_n <= 0 and body_n <= 0:
        return 0
    return max(faces_n, body_n)


def score_people_count_match(rgb_uint8: np.ndarray, expected_count: int) -> float:
    """
    [0,1] score where 1 means estimated people count matches expected exactly.
    """
    exp = int(expected_count or 0)
    if exp <= 0:
        return 0.5
    est = estimate_people_count(rgb_uint8)
    if est < 0:
        return 0.5
    err = abs(est - exp)
    denom = max(1, exp)
    return float(np.clip(1.0 - (err / denom), 0.0, 1.0))


def estimate_object_count(rgb_uint8: np.ndarray, object_hint: str = "") -> int:
    """
    Lightweight repeated-object count heuristic.
    Uses Hough circles for circle-like hints, otherwise connected components.
    """
    try:
        import cv2
    except ImportError:
        return -1
    try:
        gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    except Exception:
        return -1
    h, w = gray.shape[:2]
    hint = (object_hint or "").lower().strip()

    if maybe_count_components_native is not None:
        try:
            area_min = max(12, int(h * w * 0.00015))
            area_max = int(h * w * 0.25)
            native_count = maybe_count_components_native(
                rgb_uint8,
                threshold=140,
                min_area=area_min,
                max_area=area_max,
            )
            if (
                native_count is not None
                and native_count >= 0
                and hint not in {"coin", "button", "ball", "circle", "bubble"}
            ):
                return int(native_count)
        except Exception:
            pass

    if hint in {"coin", "button", "ball", "circle", "bubble"}:
        try:
            blur = cv2.GaussianBlur(gray, (7, 7), 1.2)
            circles = cv2.HoughCircles(
                blur,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=max(8.0, float(min(h, w)) * 0.025),
                param1=80,
                param2=22,
                minRadius=max(2, int(min(h, w) * 0.008)),
                maxRadius=max(6, int(min(h, w) * 0.14)),
            )
            if circles is not None:
                return int(circles.shape[1])
        except Exception:
            pass

    try:
        blur = cv2.GaussianBlur(gray, (5, 5), 0.0)
        _thr, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, _MORPH_KERNEL_3, iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, _MORPH_KERNEL_3, iterations=1)
        n_labels, _labels, stats, _cent = cv2.connectedComponentsWithStats(bw, connectivity=8)
        if n_labels <= 1:
            return 0
        area_min = max(12, int(h * w * 0.00015))
        area_max = int(h * w * 0.25)
        areas = stats[1:, cv2.CC_STAT_AREA]  # skip background row 0
        keep_mask = (areas >= area_min) & (areas <= area_max)
        return int(np.count_nonzero(keep_mask))
    except Exception:
        return -1


def score_object_count_match(rgb_uint8: np.ndarray, expected_count: int, object_hint: str = "") -> float:
    """
    [0,1] score where 1 means estimated object count matches expected exactly.
    """
    exp = int(expected_count or 0)
    if exp <= 0:
        return 0.5
    est = estimate_object_count(rgb_uint8, object_hint=object_hint)
    if est < 0:
        return 0.5
    err = abs(est - exp)
    denom = max(1, exp)
    return float(np.clip(1.0 - (err / denom), 0.0, 1.0))


def _maybe_human_scores(prompt: str, rgb_images: Sequence[np.ndarray], plan: dict | None = None) -> list[float] | None:
    """Per-image human-likeness in [0,1]; None for drawn / rendered media."""
    from utils.quality.critics import resolve_plan
    from utils.quality.critics import score_human_likeness

    p = resolve_plan(prompt, plan)
    if p.get("medium") in ("2d", "3d"):
        return None
    return [score_human_likeness(im, prompt, plan=p) for im in rgb_images]


def _maybe_spatial_bind_scores(prompt: str, rgb_images: Sequence[np.ndarray]) -> list[float] | None:
    """Per-image left/right color-bind scores; None when the prompt names no color pair."""
    from utils.quality.critics import score_spatial_color_bind

    scores = [score_spatial_color_bind(im, prompt) for im in rgb_images]
    return None if not scores or scores[0] is None else [float(s or 0.0) for s in scores]


def _per_image(module: str, fn: str, with_prompt: bool = False):
    def score(prompt: str, rgb_images: Sequence[np.ndarray]) -> list[float]:
        f = getattr(import_module(f"utils.quality.{module}"), fn)
        return [float(f(im, prompt) if with_prompt else f(im)) for im in rgb_images]

    return score


_SPECIALTY_PICK = {
    "combo_human": _maybe_human_scores,
    "combo_spatial_bind": _maybe_spatial_bind_scores,
    "combo_contact": _per_image("critics", "score_contact_shadow"),
    "combo_hand": _per_image("critics", "score_hand_structure"),
    "combo_occlusion": _per_image("critics", "score_occlusion"),
    "combo_reflection": _per_image("critics", "score_reflection"),
    "combo_architecture": _per_image("architecture_consistency", "score_architecture_consistency", with_prompt=True),
}


@dataclass(slots=True)
class PickContext:
    """Everything a term may need besides the images."""

    prompt: str = ""
    device: str = "cpu"
    expected_text: str = ""
    clip_model_id: str = "openai/clip-vit-base-patch32"
    expected_count: int = 0
    expected_count_target: str = "auto"
    expected_count_object: str = ""
    vit_ckpt_path: str = ""
    vit_use_adherence: bool = False
    vit_num_ar_blocks: int = -1
    plan: dict = field(default_factory=dict)


def _vit_views(pil: Any, size: int) -> list[Any]:
    """Aspect-preserving views: short side to ``size``, square crops tiling the long side.

    Matches the near-square ``RandomResizedCrop`` the vit_quality head trains on and keeps the
    high-frequency cues a squashing resize would smear.
    """
    from PIL import Image

    w, h = pil.size
    s = size / max(1, min(w, h))
    im = pil.resize((max(size, round(w * s)), max(size, round(h * s))), Image.Resampling.BILINEAR)
    W, H = im.size
    long_side, horizontal = (W, True) if W >= H else (H, False)
    offsets = sorted({0, (long_side - size) // 2, long_side - size})
    return [im.crop((o, 0, o + size, size)) if horizontal else im.crop((0, o, size, o + size)) for o in offsets]


def _score_vit_quality(rgb_images: Sequence[np.ndarray], ctx: PickContext) -> list[float] | None:
    """Per-image vit_quality probability (optionally blended with adherence); None without a checkpoint."""
    ckpt = str(ctx.vit_ckpt_path or "").strip()
    if not ckpt:
        return None
    try:
        import torch
        from PIL import Image
        from vit_quality.model import cached_vit_scorer

        scorer = cached_vit_scorer(ckpt, str(ctx.device))
    except Exception:
        return None
    views: list[Any] = []
    owner: list[int] = []
    for i, im in enumerate(rgb_images):
        for v in _vit_views(Image.fromarray(np.asarray(im, dtype=np.uint8), mode="RGB"), scorer.image_size):
            views.append(scorer.normalize(v))
            owner.append(i)
    out = scorer(torch.stack(views, dim=0), ctx.prompt, ctx.vit_num_ar_blocks)
    per_view = torch.sigmoid(out["quality_logit"].float()).reshape(-1).cpu().numpy()
    if ctx.vit_use_adherence:
        per_view = 0.65 * per_view + 0.35 * out["adherence_score"].float().reshape(-1).cpu().numpy()
    idx = np.asarray(owner)
    return [float(per_view[idx == i].mean()) for i in range(len(rgb_images))]


# Reward-panel backends that judge the image (aesthetic / preference / IQA). Prompt-similarity
# backends (CLIP-H, SigLIP2) are left out: the clip term already measures alignment.
_REWARD_PATHS = {
    "hpsv2": "default_hpsv2_path",
    "pickscore": "default_pickscore_path",
    "image_reward": "default_image_reward_path",
    "cafe_aesthetic": "default_cafe_aesthetic_path",
    "perceptclip": "default_perceptclip_iqa_path",
    "musiq": "default_musiq_path",
    "clip_iqa": "default_clip_iqa_path",
}
_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def _weights_on_disk(path_or_repo: str) -> bool:
    """True when a pretrained folder holds weights or the HF hub cache already has the repo's weights."""
    p = Path(path_or_repo)
    if p.is_dir():
        from utils.modeling.model_paths import has_local_weights

        return has_local_weights(p)
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return any(isinstance(try_to_load_from_cache(path_or_repo, f), str) for f in _WEIGHT_FILES)


@lru_cache(maxsize=1)
def local_reward_weights() -> Any:
    """``HFRewardWeights`` with every backend whose weights are not on disk zeroed; None if none are."""
    from utils.modeling import model_paths
    from utils.modeling.hf_loaders import HFRewardWeights

    w = HFRewardWeights()
    found = False
    for name in (f.name for f in fields(w)):
        fn = _REWARD_PATHS.get(name)
        ok = bool(fn) and _weights_on_disk(getattr(model_paths, fn)())
        found |= ok
        if not ok:
            setattr(w, name, 0.0)
    return w if found else None


def score_reward_models(rgb_images: Sequence[np.ndarray], prompt: str, device: str) -> list[float] | None:
    """Local HF reward-panel composite per image (CAFE, PickScore, HPS, ImageReward... whichever exist)."""
    weights = local_reward_weights()
    if weights is None:
        return None
    from utils.modeling.hf_loaders import HFRewardPanel

    panel = HFRewardPanel(weights, device=device)
    out = [panel.score_breakdown(im, prompt=prompt) for im in rgb_images]
    return None if not any(b.parts for b in out) else [float(b.composite) for b in out]


def _count_scores(rgb_images: Sequence[np.ndarray], ctx: PickContext) -> list[float] | None:
    """Expected-count match; the target is explicit, else inferred (people first, then objects)."""
    mode = str(ctx.expected_count_target or "auto").strip().lower()
    count = int(ctx.expected_count or 0)
    hint = str(ctx.expected_count_object or "").strip().lower()
    if mode == "objects":
        if count <= 0:
            count, inferred = infer_expected_object_count(ctx.prompt)
            hint = hint or inferred
        objects = True
    elif mode == "people":
        count = count if count > 0 else infer_expected_people_count(ctx.prompt)
        objects = False
    elif count > 0:
        objects = bool(hint)
    else:
        count = infer_expected_people_count(ctx.prompt)
        objects = count <= 0
        if objects:
            count, inferred = infer_expected_object_count(ctx.prompt)
            hint = hint or inferred
    if count <= 0:
        return None
    if objects:
        return [score_object_count_match(im, count, object_hint=hint) for im in rgb_images]
    return [score_people_count_match(im, count) for im in rgb_images]


def _each(fn: Callable[[np.ndarray], float | None]) -> Callable[[Sequence[np.ndarray], PickContext], list | None]:
    def score(rgb_images: Sequence[np.ndarray], _ctx: PickContext) -> list | None:
        vals = [fn(im) for im in rgb_images]
        return None if any(v is None for v in vals) else [float(cast(float, v)) for v in vals]

    return score


def _ai_look_natural(rgb_images: Sequence[np.ndarray], ctx: PickContext) -> list[float]:
    from utils.quality.critics import ai_look_score

    return [ai_look_score(im, ctx.prompt, ctx.plan).natural for im in rgb_images]


def _art(rgb_images: Sequence[np.ndarray], ctx: PickContext) -> list[float]:
    from utils.quality.critics import score_art

    return [score_art(im, ctx.plan).score for im in rgb_images]


def _cast(rgb_images: Sequence[np.ndarray], ctx: PickContext) -> list[float] | None:
    if ctx.plan.get("grade"):
        return None
    return [float(cast(float, score_color_cast_neutrality(im))) for im in rgb_images]


def _specialty(fn: Callable[[str, Sequence[np.ndarray]], list[float] | None]) -> Callable:
    return lambda ims, c: fn(c.prompt, ims)


TERMS: dict[str, Callable[[Sequence[np.ndarray], PickContext], list[float] | None]] = {
    **{name.removeprefix("combo_"): _specialty(fn) for name, fn in _SPECIALTY_PICK.items()},
    "clip": lambda ims, c: (
        score_clip_similarity(list(ims), c.prompt, device=c.device, model_id=c.clip_model_id)
        if c.prompt.strip()
        else None
    ),
    "edge": _each(lambda im: score_edge_sharpness(im)),
    "ocr": lambda ims, c: [score_ocr_match(im, c.expected_text) for im in ims] if c.expected_text.strip() else None,
    "vit": _score_vit_quality,
    "aesthetic": _each(lambda im: score_aesthetic_proxy(im)),
    "exposure": _each(lambda im: score_exposure_balance(im)),
    "dynamic": _each(lambda im: score_dynamic_range_headroom(im)),
    "detail": _each(lambda im: score_photographic_detail_balance(im)),
    "skin": _each(lambda im: score_skin_tone_naturalness(im)),
    "cast": _cast,
    "count": _count_scores,
    "human": lambda ims, c: _maybe_human_scores(c.prompt, ims, c.plan),
    "ai_look": _ai_look_natural,
    "art": _art,
    "reward": lambda ims, c: score_reward_models(ims, c.prompt, c.device),
}

# Raw term values that are not already in [0, 1] (absolute scoring, e.g. ``CompositeRanker.score_one``).
_UNIT_SCALE: dict[str, Callable[[float], float]] = {
    "clip": lambda x: float(1.0 / (1.0 + np.exp(-(x - 25.0) / 5.0))),  # CLIP logit = 100 * cosine
    "edge": lambda x: float(np.clip(x / 400.0, 0.0, 1.0)),  # Laplacian variance
}


def unit_scale(term: str, value: float) -> float:
    """Map a raw term value to [0, 1] for absolute (single-image) scoring."""
    fn = _UNIT_SCALE.get(term)
    return fn(float(value)) if fn else float(np.clip(value, 0.0, 1.0))


_SPECIALTY_BASE = {"clip": 0.40, "exposure": 0.14, "ai_look": 0.12, "human": 0.14, "ocr": 0.10}

# Pick metric -> term weights. Single-term metrics return that term's raw scores.
METRIC_WEIGHTS: dict[str, dict[str, float]] = {
    "clip": {"clip": 1.0},
    "edge": {"edge": 1.0},
    "ocr": {"ocr": 1.0},
    "vit": {"vit": 1.0},
    "aesthetic": {"aesthetic": 1.0},
    "natural": {"ai_look": 1.0},
    "art": {"art": 1.0},
    "combo": {"clip": 0.55, "edge": 0.20, "ai_look": 0.25, "ocr": 0.15},
    "combo_vit": {"clip": 0.40, "edge": 0.15, "vit": 0.30, "ai_look": 0.15},
    "combo_vit_hq": {
        "clip": 0.18,
        "edge": 0.08,
        "vit": 0.22,
        "exposure": 0.10,
        "ai_look": 0.18,
        "cast": 0.08,
        "art": 0.16,
    },
    "combo_vit_realism": {
        "clip": 0.16,
        "detail": 0.08,
        "exposure": 0.08,
        "dynamic": 0.08,
        "ai_look": 0.20,
        "cast": 0.06,
        "skin": 0.06,
        "vit": 0.18,
        "art": 0.10,
        "ocr": 0.06,
    },
    "combo_exposure": {"clip": 0.45, "edge": 0.25, "exposure": 0.30},
    "combo_structural": {"clip": 0.35, "edge": 0.20, "exposure": 0.20, "ai_look": 0.25},
    "combo_hq": {"clip": 0.27, "edge": 0.13, "exposure": 0.15, "ai_look": 0.20, "cast": 0.10, "art": 0.15},
    "combo_count": {"clip": 0.45, "edge": 0.25, "count": 0.30, "ocr": 0.10},
    "combo_count_vit": {"clip": 0.38, "edge": 0.22, "count": 0.25, "vit": 0.15, "ocr": 0.08},
    "combo_realism": {
        "clip": 0.24,
        "detail": 0.10,
        "exposure": 0.10,
        "dynamic": 0.08,
        "ai_look": 0.22,
        "cast": 0.06,
        "skin": 0.06,
        "art": 0.14,
        "ocr": 0.04,
    },
    "aesthetic_realism": {
        "detail": 0.14,
        "exposure": 0.14,
        "dynamic": 0.12,
        "ai_look": 0.24,
        "cast": 0.08,
        "skin": 0.08,
        "art": 0.20,
        "ocr": 0.05,
    },
    "combo_natural": {"clip": 0.30, "ai_look": 0.45, "art": 0.25, "ocr": 0.10},
    "combo_art": {"clip": 0.35, "art": 0.45, "ai_look": 0.20, "ocr": 0.10},
    "superior_composite": {"clip": 0.30, "ai_look": 0.25, "art": 0.20, "reward": 0.15, "vit": 0.10, "ocr": 0.10},
    **{name: {**_SPECIALTY_BASE, name.removeprefix("combo_"): 0.20} for name in _SPECIALTY_PICK},
}

# ``--pick-best`` choices (sample CLI, quality policy).
PICK_METRICS: tuple[str, ...] = ("auto", "none", *METRIC_WEIGHTS)


def term_scores(
    rgb_images: Sequence[np.ndarray], terms: Sequence[str], ctx: PickContext
) -> dict[str, list[float] | None]:
    """Raw per-image scores for each named term (None where a term has no evidence)."""
    return {t: TERMS[t](rgb_images, ctx) for t in terms}


def pick_best_indices(
    rgb_images: list[np.ndarray],
    prompt: str,
    metric: str,
    device: str,
    expected_text: str = "",
    clip_model_id: str = "openai/clip-vit-base-patch32",
    expected_count: int = 0,
    expected_count_target: str = "auto",
    expected_count_object: str = "",
    vit_ckpt_path: str = "",
    vit_use_adherence: bool = False,
    vit_num_ar_blocks: int = -1,
    *,
    plan: dict | None = None,
) -> tuple[int, list[float]]:
    """
    Return (best_index, one score per image) for a ``PICK_METRICS`` name.

    ``plan`` (see ``utils.quality.critics``) states the intended look; by default it is derived
    from ``prompt``. Unknown metrics and ``none`` keep candidate 0.
    """
    metric = (metric or "none").lower().strip()
    n = len(rgb_images)
    if n == 0:
        return 0, []
    weights = METRIC_WEIGHTS.get(metric)
    if weights is None:
        return 0, [0.0] * n
    from utils.quality.critics import resolve_plan

    ctx = PickContext(
        prompt=prompt or "",
        device=device,
        expected_text=str(expected_text or ""),
        clip_model_id=clip_model_id,
        expected_count=int(expected_count or 0),
        expected_count_target=expected_count_target,
        expected_count_object=expected_count_object,
        vit_ckpt_path=vit_ckpt_path,
        vit_use_adherence=vit_use_adherence,
        vit_num_ar_blocks=vit_num_ar_blocks,
        plan=resolve_plan(prompt, plan),
    )
    scores = term_scores(rgb_images, list(weights), ctx)
    if len(weights) == 1:
        raw = next(iter(scores.values()))
        combined = [0.5] * n if raw is None else [float(v) for v in raw]
    else:
        combined = rank_aggregate(scores, weights)
    return int(np.argmax(combined)), combined

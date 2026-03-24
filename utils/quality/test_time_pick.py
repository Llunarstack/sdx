"""
Test-time scaling: score candidate RGB images (uint8 HWC) to pick the best sample.
Used by sample.py --pick-best (§11.3 IMPROVEMENTS.md).
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

import numpy as np

_clip_model = None
_clip_processor = None
_clip_model_id = None


def _norm01(scores: Sequence[float]) -> List[float]:
    s = list(scores)
    if not s:
        return []
    lo, hi = min(s), max(s)
    if hi - lo < 1e-8:
        return [0.5] * len(s)
    return [(x - lo) / (hi - lo) for x in s]


def score_edge_sharpness(rgb_uint8: np.ndarray) -> float:
    """Higher = sharper (Laplacian variance on grayscale)."""
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
    exp = re.sub(r"\s+", "", str(expected).upper())
    got = re.sub(r"\s+", "", txt.upper())
    if not exp:
        return 0.5
    hit = sum(1 for c in exp if c in got)
    return hit / max(1, len(exp))


def score_clip_similarity(
    rgb_uint8_list: List[np.ndarray],
    prompt: str,
    device: str,
    model_id: str = "openai/clip-vit-base-patch32",
) -> List[float]:
    """Per-image CLIP image-text similarity (higher = better prompt alignment)."""
    global _clip_model, _clip_processor, _clip_model_id
    if not prompt or not str(prompt).strip():
        return [0.5] * len(rgb_uint8_list)
    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        return [0.5] * len(rgb_uint8_list)

    if _clip_model is None or _clip_model_id != model_id:
        _clip_processor = CLIPProcessor.from_pretrained(model_id)
        _clip_model = CLIPModel.from_pretrained(model_id).to(device)
        _clip_model.eval()
        _clip_model_id = model_id

    pil_images = [Image.fromarray(im, mode="RGB") for im in rgb_uint8_list]
    proc = _clip_processor(text=[prompt], images=pil_images, return_tensors="pt", padding=True)
    proc = {k: v.to(device) for k, v in proc.items()}
    with torch.no_grad():
        out = _clip_model(**proc)
        logits = out.logits_per_image  # (N, 1) when one text duplicated
        if logits.shape[-1] == 1:
            scores = logits[:, 0].float().cpu().numpy().tolist()
        else:
            scores = logits.diag().float().cpu().numpy().tolist()
    return [float(s) for s in scores]


def score_exposure_balance(rgb_uint8: np.ndarray) -> float:
    """
    Lightweight verifier score (LANDSCAPE §2): penalize heavy highlight/shadow clipping.
    Higher = more pixels in a mid-tone band (less blown-out / crushed).
    """
    x = rgb_uint8.astype(np.float32)
    gray = x.mean(axis=-1)
    clipped = np.sum((gray <= 2.0) | (gray >= 253.0))
    n = max(1, gray.size)
    return 1.0 - float(clipped) / float(n)


def score_tiling_artifact_free(rgb_uint8: np.ndarray) -> float:
    """
    Heuristic [0,1]: lower repeated-shift correlation implies fewer tiling/grid artifacts.
    Higher is better (more artifact-free).
    """
    x = rgb_uint8.astype(np.float32)
    if x.ndim == 3:
        g = x.mean(axis=-1)
    else:
        g = x
    h, w = g.shape[:2]
    shifts = [(8, 0), (0, 8), (16, 0), (0, 16)]
    corrs = []
    for dy, dx in shifts:
        if h <= dy + 4 or w <= dx + 4:
            continue
        a = g[0 : h - dy, 0 : w - dx].reshape(-1)
        b = g[dy:h, dx:w].reshape(-1)
        a_std = float(np.std(a))
        b_std = float(np.std(b))
        if a_std < 1e-6 or b_std < 1e-6:
            continue
        c = float(np.corrcoef(a, b)[0, 1])
        if np.isfinite(c):
            corrs.append(abs(c))
    if not corrs:
        return 0.5
    # High repeated-pattern correlation => likely grid/tiling artifact.
    worst = max(corrs)
    return float(np.clip(1.0 - worst, 0.0, 1.0))


def score_color_cast_neutrality(rgb_uint8: np.ndarray) -> float:
    """
    Heuristic [0,1]: penalize strong global channel cast.
    Higher = more balanced channel means.
    """
    x = rgb_uint8.astype(np.float32)
    if x.ndim != 3 or x.shape[2] < 3:
        return 0.5
    means = x.reshape(-1, x.shape[2]).mean(axis=0)[:3]
    spread = float(np.max(means) - np.min(means))
    # 0 spread => perfect neutrality. ~80+ indicates strong cast.
    return float(np.clip(1.0 - spread / 80.0, 0.0, 1.0))


def pick_best_indices(
    rgb_images: List[np.ndarray],
    prompt: str,
    metric: str,
    device: str,
    expected_text: str = "",
    clip_model_id: str = "openai/clip-vit-base-patch32",
) -> Tuple[int, List[float]]:
    """
    Return (best_index, raw_scores_one_per_image).
    metric: none|clip|edge|ocr|combo|combo_exposure|combo_structural|combo_hq
    """
    metric = (metric or "none").lower().strip()
    n = len(rgb_images)
    if n == 0:
        return 0, []
    if metric in ("none", ""):
        return 0, [0.0] * n

    if metric == "edge":
        scores = [score_edge_sharpness(im) for im in rgb_images]
        return int(np.argmax(scores)), scores

    if metric == "ocr":
        scores = [score_ocr_match(im, expected_text) for im in rgb_images]
        return int(np.argmax(scores)), scores

    if metric == "clip":
        scores = score_clip_similarity(rgb_images, prompt, device=device, model_id=clip_model_id)
        return int(np.argmax(scores)), scores

    if metric == "combo":
        s_clip = _norm01(score_clip_similarity(rgb_images, prompt, device=device, model_id=clip_model_id))
        s_edge = _norm01([score_edge_sharpness(im) for im in rgb_images])
        if expected_text and str(expected_text).strip():
            s_ocr = _norm01([score_ocr_match(im, expected_text) for im in rgb_images])
            combined = [0.5 * c + 0.35 * e + 0.15 * o for c, e, o in zip(s_clip, s_edge, s_ocr)]
        else:
            combined = [0.65 * c + 0.35 * e for c, e in zip(s_clip, s_edge)]
        return int(np.argmax(combined)), combined

    if metric == "combo_exposure":
        s_clip = _norm01(score_clip_similarity(rgb_images, prompt, device=device, model_id=clip_model_id))
        s_edge = _norm01([score_edge_sharpness(im) for im in rgb_images])
        s_exp = _norm01([score_exposure_balance(im) for im in rgb_images])
        combined = [0.45 * c + 0.30 * e + 0.25 * x for c, e, x in zip(s_clip, s_edge, s_exp)]
        return int(np.argmax(combined)), combined

    if metric == "combo_structural":
        s_clip = _norm01(score_clip_similarity(rgb_images, prompt, device=device, model_id=clip_model_id))
        s_edge = _norm01([score_edge_sharpness(im) for im in rgb_images])
        s_exp = _norm01([score_exposure_balance(im) for im in rgb_images])
        s_tile = _norm01([score_tiling_artifact_free(im) for im in rgb_images])
        combined = [0.35 * c + 0.25 * e + 0.20 * x + 0.20 * t for c, e, x, t in zip(s_clip, s_edge, s_exp, s_tile)]
        return int(np.argmax(combined)), combined

    if metric == "combo_hq":
        s_clip = _norm01(score_clip_similarity(rgb_images, prompt, device=device, model_id=clip_model_id))
        s_edge = _norm01([score_edge_sharpness(im) for im in rgb_images])
        s_exp = _norm01([score_exposure_balance(im) for im in rgb_images])
        s_tile = _norm01([score_tiling_artifact_free(im) for im in rgb_images])
        s_cast = _norm01([score_color_cast_neutrality(im) for im in rgb_images])
        combined = [
            0.30 * c + 0.20 * e + 0.20 * x + 0.15 * t + 0.15 * k
            for c, e, x, t, k in zip(s_clip, s_edge, s_exp, s_tile, s_cast)
        ]
        return int(np.argmax(combined)), combined

    return 0, [0.0] * n

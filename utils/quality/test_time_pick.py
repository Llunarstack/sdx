"""
Test-time scaling: score candidate RGB images (uint8 HWC) to pick the best sample.
Used by sample.py --pick-best (§11.3 IMPROVEMENTS.md).
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

import numpy as np

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

try:
    from utils.native import maybe_norm01_native, maybe_weighted_sum_native
except Exception:  # pragma: no cover - optional native path
    maybe_norm01_native = None
    maybe_weighted_sum_native = None

_clip_model = None
_clip_processor = None
_clip_model_id = None
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


def _norm01(scores: Sequence[float]) -> List[float]:
    s = list(scores)
    if not s:
        return []
    if maybe_norm01_native is not None:
        try:
            out = maybe_norm01_native([float(x) for x in s])
            if out is not None and len(out) == len(s):
                return [float(x) for x in out]
        except Exception:
            pass
    lo, hi = min(s), max(s)
    if hi - lo < 1e-8:
        return [0.5] * len(s)
    return [(x - lo) / (hi - lo) for x in s]


def _weighted_sum(score_lists: Sequence[Sequence[float]], weights: Sequence[float]) -> List[float]:
    rows = [list(r) for r in score_lists]
    if not rows:
        return []
    cols = len(rows[0])
    if any(len(r) != cols for r in rows):
        raise ValueError("score lists must be rectangular")
    if len(rows) != len(weights):
        raise ValueError("weights length must equal number of score lists")
    if maybe_weighted_sum_native is not None:
        try:
            out = maybe_weighted_sum_native(
                [[float(v) for v in row] for row in rows],
                [float(w) for w in weights],
            )
            if out is not None and len(out) == cols:
                return [float(x) for x in out]
        except Exception:
            pass
    return [float(sum(float(w) * float(rows[r][c]) for r, w in enumerate(weights))) for c in range(cols)]


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


def score_saturation_balance(rgb_uint8: np.ndarray) -> float:
    """
    Heuristic [0,1]: penalize heavily over-saturated results.
    Higher = more natural saturation balance.
    """
    x = rgb_uint8.astype(np.float32)
    if x.ndim != 3 or x.shape[2] < 3:
        return 0.5
    maxc = np.max(x[..., :3], axis=-1)
    minc = np.min(x[..., :3], axis=-1)
    sat = np.where(maxc > 1e-6, (maxc - minc) / np.maximum(maxc, 1e-6), 0.0)
    p95 = float(np.percentile(sat, 95))
    # Keep a comfortable range around ~0.55; strongly penalize very high saturation tails.
    if p95 <= 0.55:
        return 1.0
    return float(np.clip(1.0 - ((p95 - 0.55) / 0.45), 0.0, 1.0))


def infer_expected_people_count(prompt: str) -> int:
    """
    Infer intended people count from prompt phrases.
    Returns 0 when no count intent is detected.
    """
    p = (prompt or "").lower()
    if not p:
        return 0
    people_terms = r"(?:people|persons|person|characters?|subjects?|men|women|girls?|boys?|kids?|children)"
    patterns = [
        rf"\bexactly\s+(\d+)\s+{people_terms}\b",
        rf"\b(\d+)\s+{people_terms}\b",
        r"\b(\d+)girls?\b",
        r"\b(\d+)boys?\b",
    ]
    for pat in patterns:
        m = re.search(pat, p)
        if m:
            try:
                return max(0, int(m.group(1)))
            except Exception:
                pass
    word_terms = "|".join(re.escape(k) for k in _WORD_NUMBERS.keys())
    m2 = re.search(rf"\b({word_terms})\s+{people_terms}\b", p)
    if m2:
        return int(_WORD_NUMBERS.get(m2.group(1), 0))
    return 0


def infer_expected_object_count(prompt: str) -> Tuple[int, str]:
    """
    Infer intended repeated non-people object count from prompt text.
    Returns (count, object_hint). count=0 means no object-count intent detected.
    """
    p = (prompt or "").lower()
    if not p:
        return 0, ""
    number_words = "|".join(re.escape(k) for k in _WORD_NUMBERS.keys())
    # Examples: "exactly 7 coins", "5 candles", "two windows"
    pats = [
        r"\bexactly\s+(\d+)\s+([a-z][a-z0-9_-]{2,})\b",
        r"\b(\d+)\s+([a-z][a-z0-9_-]{2,})\b",
        rf"\b({number_words})\s+([a-z][a-z0-9_-]{{2,}})\b",
    ]
    for pat in pats:
        m = re.search(pat, p)
        if not m:
            continue
        raw_num = m.group(1)
        noun = str(m.group(2) or "").strip().lower()
        if not noun or noun in _PEOPLE_WORDS:
            continue
        noun = noun.rstrip("s")
        try:
            if raw_num.isdigit():
                n = int(raw_num)
            else:
                n = int(_WORD_NUMBERS.get(raw_num, 0))
            if n > 0:
                return n, noun
        except Exception:
            pass
    return 0, ""


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
    try:
        face_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(face_xml)
        if not face_cascade.empty():
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=4, minSize=(18, 18))
            faces_n = int(len(faces))
    except Exception:
        faces_n = 0

    body_n = 0
    try:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
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
            if native_count is not None and native_count >= 0 and hint not in {"coin", "button", "ball", "circle", "bubble"}:
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
        kernel = np.ones((3, 3), dtype=np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
        n_labels, _labels, stats, _cent = cv2.connectedComponentsWithStats(bw, connectivity=8)
        if n_labels <= 1:
            return 0
        area_min = max(12, int(h * w * 0.00015))
        area_max = int(h * w * 0.25)
        keep = 0
        for i in range(1, n_labels):
            a = int(stats[i, cv2.CC_STAT_AREA])
            if area_min <= a <= area_max:
                keep += 1
        return int(keep)
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


def pick_best_indices(
    rgb_images: List[np.ndarray],
    prompt: str,
    metric: str,
    device: str,
    expected_text: str = "",
    clip_model_id: str = "openai/clip-vit-base-patch32",
    expected_count: int = 0,
    expected_count_target: str = "auto",
    expected_count_object: str = "",
) -> Tuple[int, List[float]]:
    """
    Return (best_index, raw_scores_one_per_image).
    metric: none|clip|edge|ocr|combo|combo_exposure|combo_structural|combo_hq|combo_count
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
            combined = _weighted_sum([s_clip, s_edge, s_ocr], [0.5, 0.35, 0.15])
        else:
            combined = _weighted_sum([s_clip, s_edge], [0.65, 0.35])
        return int(np.argmax(combined)), combined

    if metric == "combo_exposure":
        s_clip = _norm01(score_clip_similarity(rgb_images, prompt, device=device, model_id=clip_model_id))
        s_edge = _norm01([score_edge_sharpness(im) for im in rgb_images])
        s_exp = _norm01([score_exposure_balance(im) for im in rgb_images])
        combined = _weighted_sum([s_clip, s_edge, s_exp], [0.45, 0.30, 0.25])
        return int(np.argmax(combined)), combined

    if metric == "combo_structural":
        s_clip = _norm01(score_clip_similarity(rgb_images, prompt, device=device, model_id=clip_model_id))
        s_edge = _norm01([score_edge_sharpness(im) for im in rgb_images])
        s_exp = _norm01([score_exposure_balance(im) for im in rgb_images])
        s_tile = _norm01([score_tiling_artifact_free(im) for im in rgb_images])
        combined = _weighted_sum([s_clip, s_edge, s_exp, s_tile], [0.35, 0.25, 0.20, 0.20])
        return int(np.argmax(combined)), combined

    if metric == "combo_hq":
        s_clip = _norm01(score_clip_similarity(rgb_images, prompt, device=device, model_id=clip_model_id))
        s_edge = _norm01([score_edge_sharpness(im) for im in rgb_images])
        s_exp = _norm01([score_exposure_balance(im) for im in rgb_images])
        s_tile = _norm01([score_tiling_artifact_free(im) for im in rgb_images])
        s_cast = _norm01([score_color_cast_neutrality(im) for im in rgb_images])
        s_sat = _norm01([score_saturation_balance(im) for im in rgb_images])
        combined = _weighted_sum([s_clip, s_edge, s_exp, s_tile, s_cast, s_sat], [0.27, 0.18, 0.18, 0.14, 0.12, 0.11])
        return int(np.argmax(combined)), combined

    if metric == "combo_count":
        s_clip = _norm01(score_clip_similarity(rgb_images, prompt, device=device, model_id=clip_model_id))
        s_edge = _norm01([score_edge_sharpness(im) for im in rgb_images])
        tgt = str(expected_count_target or "auto").strip().lower()
        exp_count = int(expected_count or 0)
        obj_hint = str(expected_count_object or "").strip().lower()
        use_object_scoring = False
        # auto => prefer explicit target, else infer people first, then non-people objects.
        if tgt == "objects":
            if exp_count <= 0:
                exp_count, inf_obj = infer_expected_object_count(prompt)
                if not obj_hint:
                    obj_hint = inf_obj
            use_object_scoring = True
        elif tgt == "people":
            if exp_count <= 0:
                exp_count = infer_expected_people_count(prompt)
            use_object_scoring = False
        else:
            if exp_count > 0:
                # Explicit count in auto mode defaults to people unless object hint exists.
                use_object_scoring = bool(obj_hint)
            else:
                exp_people = infer_expected_people_count(prompt)
                if exp_people > 0:
                    exp_count = exp_people
                    use_object_scoring = False
                else:
                    exp_obj, inf_obj = infer_expected_object_count(prompt)
                    exp_count = exp_obj
                    obj_hint = obj_hint or inf_obj
                    use_object_scoring = True
        if exp_count > 0:
            if use_object_scoring:
                s_cnt = _norm01([score_object_count_match(im, exp_count, object_hint=obj_hint) for im in rgb_images])
            else:
                s_cnt = _norm01([score_people_count_match(im, exp_count) for im in rgb_images])
            if expected_text and str(expected_text).strip():
                s_ocr = _norm01([score_ocr_match(im, expected_text) for im in rgb_images])
                combined = _weighted_sum([s_clip, s_edge, s_cnt, s_ocr], [0.38, 0.22, 0.30, 0.10])
            else:
                combined = _weighted_sum([s_clip, s_edge, s_cnt], [0.45, 0.25, 0.30])
            return int(np.argmax(combined)), combined
        # No count target found; fall back to standard combo behavior.
        if expected_text and str(expected_text).strip():
            s_ocr = _norm01([score_ocr_match(im, expected_text) for im in rgb_images])
            combined = _weighted_sum([s_clip, s_edge, s_ocr], [0.5, 0.35, 0.15])
        else:
            combined = _weighted_sum([s_clip, s_edge], [0.65, 0.35])
        return int(np.argmax(combined)), combined

    return 0, [0.0] * n

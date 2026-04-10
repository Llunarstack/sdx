"""
Test-time scaling: score candidate RGB images (uint8 HWC) to pick the best sample.
Used by sample.py --pick-best (§11.3 IMPROVEMENTS.md).
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence, Tuple, cast

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

_clip_model: Any = None
_clip_processor: Any = None
_clip_model_id: Optional[str] = None
_vit_model: Any = None
_vit_cfg: Any = None
_vit_ckpt: Any = None
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
_CV2_FACE_CASCADE = None  # lazy: cv2.CascadeClassifier or False if unavailable
_CV2_HOG = None  # lazy HOGDescriptor


def _norm01(scores: Sequence[float]) -> List[float]:
    score_array = np.asarray(scores, dtype=np.float64)
    n = int(score_array.size)
    if n == 0:
        return []
    if maybe_norm01_native is not None:
        try:
            score_list = score_array.tolist()
            out = maybe_norm01_native([float(x) for x in score_list])
            if out is not None and len(out) == len(score_list):
                return [float(x) for x in out]
        except Exception:
            pass
    lo = float(np.min(score_array))
    hi = float(np.max(score_array))
    span = hi - lo
    if span < 1e-8:
        return [0.5] * n
    return ((score_array - lo) / span).astype(np.float64).tolist()


def _weighted_sum(score_lists: Sequence[Sequence[float]], weights: Sequence[float]) -> List[float]:
    rows = list(score_lists)
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
    score_matrix = np.asarray(rows, dtype=np.float64)
    weight_array = np.asarray([float(weight) for weight in weights], dtype=np.float64)
    # Matrix-vector multiply is the fastest pure-Python fallback path here.
    return np.matmul(score_matrix.T, weight_array).astype(np.float64).tolist()


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
        _clip_processor = cast(Any, CLIPProcessor).from_pretrained(model_id)
        _clip_model = cast(Any, cast(Any, CLIPModel).from_pretrained(model_id)).to(device)
        _clip_model.eval()
        _clip_model_id = model_id

    pil_images = [Image.fromarray(im, mode="RGB") for im in rgb_uint8_list]
    proc = _clip_processor(text=[prompt], images=pil_images, return_tensors="pt", padding=True)
    proc = {k: v.to(device) for k, v in proc.items()}
    with torch.inference_mode():
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
    gray = np.mean(rgb_uint8, axis=-1)
    clipped = int(np.count_nonzero((gray <= 2.0) | (gray >= 253.0)))
    n = max(1, int(gray.size))
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
        a = g[0 : h - dy, 0 : w - dx].ravel()
        b = g[dy:h, dx:w].ravel()
        n = a.size
        if n < 2:
            continue
        inv_n = 1.0 / float(n)
        ma = float(a.mean())
        mb = float(b.mean())
        cov = float(a @ b) * inv_n - ma * mb
        var_a = float(a @ a) * inv_n - ma * ma
        var_b = float(b @ b) * inv_n - mb * mb
        denom = (max(0.0, var_a) ** 0.5) * (max(0.0, var_b) ** 0.5)
        if denom < 1e-6:
            continue
        c = cov / denom
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
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] < 3:
        return 0.5
    means = np.mean(rgb_uint8[..., :3], axis=(0, 1), dtype=np.float64)
    spread = float(np.max(means) - np.min(means))
    # 0 spread => perfect neutrality. ~80+ indicates strong cast.
    return float(np.clip(1.0 - spread / 80.0, 0.0, 1.0))


def score_saturation_balance(rgb_uint8: np.ndarray) -> float:
    """
    Heuristic [0,1]: penalize heavily over-saturated results.
    Higher = more natural saturation balance.
    """
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] < 3:
        return 0.5
    ch = rgb_uint8[..., :3]
    maxc = np.max(ch, axis=-1).astype(np.float32, copy=False)
    minc = np.min(ch, axis=-1).astype(np.float32, copy=False)
    sat = np.where(maxc > 1e-6, (maxc - minc) / np.maximum(maxc, 1e-6), 0.0)
    p95 = float(np.percentile(sat, 95))
    # Keep a comfortable range around ~0.55; strongly penalize very high saturation tails.
    if p95 <= 0.55:
        return 1.0
    return float(np.clip(1.0 - ((p95 - 0.55) / 0.45), 0.0, 1.0))


def score_dynamic_range_headroom(rgb_uint8: np.ndarray) -> float:
    """
    [0,1] score: high when highlights/shadows retain headroom and local contrast is present.
    """
    x = rgb_uint8.astype(np.float32)
    if x.ndim == 3 and x.shape[2] >= 3:
        lum = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
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
    Heuristic [0,1] aligned with ``scripts/tools/data/manifest_enrich.py``:
    mean of exposure balance, tiling-freeness, and dynamic-range headroom.
    """
    a = float(score_exposure_balance(rgb_uint8))
    b = float(score_tiling_artifact_free(rgb_uint8))
    c = float(score_dynamic_range_headroom(rgb_uint8))
    return float(max(0.0, min(1.0, (a + b + c) / 3.0)))


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


def infer_expected_object_count(prompt: str) -> Tuple[int, str]:
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


def _cv2_face_cascade():
    global _CV2_FACE_CASCADE
    if _CV2_FACE_CASCADE is not None:
        return _CV2_FACE_CASCADE
    try:
        import cv2

        cv2_any = cast(Any, cv2)
        path = cv2_any.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2_any.CascadeClassifier(path)
        _CV2_FACE_CASCADE = cascade if not cascade.empty() else False
    except Exception:
        _CV2_FACE_CASCADE = False
    return _CV2_FACE_CASCADE


def _cv2_hog_people():
    global _CV2_HOG
    if _CV2_HOG is not None:
        return _CV2_HOG
    try:
        import cv2

        cv2_any = cast(Any, cv2)
        hog = cv2_any.HOGDescriptor()
        hog.setSVMDetector(cv2_any.HOGDescriptor_getDefaultPeopleDetector())
        _CV2_HOG = hog
    except Exception:
        _CV2_HOG = False
    return _CV2_HOG


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
    vit_ckpt_path: str = "",
    vit_use_adherence: bool = False,
    vit_num_ar_blocks: int = -1,
) -> Tuple[int, List[float]]:
    """
    Return (best_index, raw_scores_one_per_image).
    metric: aesthetic, combo_vit_hq, combo_vit_realism, combo_count_vit, plus prior clip/vit/combo_* names.
    """
    metric = (metric or "none").lower().strip()
    image_count = len(rgb_images)
    if image_count == 0:
        return 0, []
    if metric in ("none", ""):
        return 0, [0.0] * image_count

    def _score_vit_quality() -> List[float]:
        """
        Return per-image quality score in roughly [0,1] from vit_quality model.
        If checkpoint/deps aren't available, returns neutral 0.5s.
        """
        global _vit_model, _vit_cfg, _vit_ckpt
        ckpt = str(vit_ckpt_path or "").strip()
        if not ckpt:
            return [0.5] * image_count
        try:
            import torch
            from PIL import Image
            from torchvision import transforms
            from vit_quality.checkpoint_utils import load_vit_quality_checkpoint
            from vit_quality.dataset import text_feature_vector
        except Exception:
            return [0.5] * image_count
        try:
            dev = torch.device(device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu")
        except Exception:
            dev = torch.device("cpu")

        try:
            if _vit_model is None or _vit_ckpt != ckpt:
                m, cfg = load_vit_quality_checkpoint(ckpt)
                m = m.to(dev)
                m.eval()
                _vit_model, _vit_cfg, _vit_ckpt = m, cfg, ckpt
        except Exception:
            return [0.5] * image_count

        img_sz = int((_vit_cfg or {}).get("image_size", 224) or 224)
        tfm = transforms.Compose(
            [
                transforms.Resize((img_sz, img_sz), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        txt = text_feature_vector(prompt).unsqueeze(0).to(dev)
        imgs = []
        for im in rgb_images:
            try:
                pil = Image.fromarray(im, mode="RGB")
                imgs.append(tfm(pil))
            except Exception:
                imgs.append(torch.zeros((3, img_sz, img_sz), dtype=torch.float32))
        x = torch.stack(imgs, dim=0).to(dev)
        txt = txt.expand(x.shape[0], -1)
        ar_cond = None
        if bool((_vit_cfg or {}).get("use_ar_conditioning", False)):
            try:
                from utils.architecture.ar_block_conditioning import (
                    ar_conditioning_vector,
                    normalize_num_ar_blocks,
                )

                nb = normalize_num_ar_blocks(vit_num_ar_blocks)
                if nb in (0, 2, 4):
                    ar_cond = ar_conditioning_vector(nb, device=dev, dtype=txt.dtype).expand(x.shape[0], -1)
            except Exception:
                ar_cond = None
        with torch.inference_mode():
            out = _vit_model(x, txt, ar_conditioning=ar_cond)
            q = out.get("quality_logit", None)
            if q is None:
                return [0.5] * image_count
            q_prob = torch.sigmoid(q.float()).detach().cpu().numpy().tolist()
            if vit_use_adherence and "adherence_score" in out:
                a = out["adherence_score"].float().detach().cpu().numpy().tolist()
                return [float(0.65 * qq + 0.35 * aa) for qq, aa in zip(q_prob, a)]
            return [float(v) for v in q_prob]

    if metric == "edge":
        scores = [score_edge_sharpness(im) for im in rgb_images]
        return int(np.argmax(scores)), scores

    if metric == "ocr":
        scores = [score_ocr_match(im, expected_text) for im in rgb_images]
        return int(np.argmax(scores)), scores

    if metric == "clip":
        scores = score_clip_similarity(rgb_images, prompt, device=device, model_id=clip_model_id)
        return int(np.argmax(scores)), scores

    if metric == "vit":
        scores = _score_vit_quality()
        return int(np.argmax(scores)), scores

    if metric == "aesthetic":
        scores = [score_aesthetic_proxy(im) for im in rgb_images]
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

    if metric == "combo_vit":
        s_clip = _norm01(score_clip_similarity(rgb_images, prompt, device=device, model_id=clip_model_id))
        s_edge = _norm01([score_edge_sharpness(im) for im in rgb_images])
        s_vit = _norm01(_score_vit_quality())
        combined = _weighted_sum([s_clip, s_edge, s_vit], [0.45, 0.20, 0.35])
        return int(np.argmax(combined)), combined

    if metric == "combo_vit_hq":
        s_clip = _norm01(score_clip_similarity(rgb_images, prompt, device=device, model_id=clip_model_id))
        s_edge = _norm01([score_edge_sharpness(im) for im in rgb_images])
        s_vit = _norm01(_score_vit_quality())
        s_exp = _norm01([score_exposure_balance(im) for im in rgb_images])
        s_tile = _norm01([score_tiling_artifact_free(im) for im in rgb_images])
        s_cast = _norm01([score_color_cast_neutrality(im) for im in rgb_images])
        s_sat = _norm01([score_saturation_balance(im) for im in rgb_images])
        s_aes = _norm01([score_aesthetic_proxy(im) for im in rgb_images])
        combined = _weighted_sum(
            [s_clip, s_edge, s_vit, s_exp, s_tile, s_cast, s_sat, s_aes],
            [0.16, 0.10, 0.22, 0.12, 0.10, 0.10, 0.10, 0.10],
        )
        return int(np.argmax(combined)), combined

    if metric == "combo_vit_realism":
        s_clip = _norm01(score_clip_similarity(rgb_images, prompt, device=device, model_id=clip_model_id))
        s_det = _norm01([score_photographic_detail_balance(im) for im in rgb_images])
        s_exp = _norm01([score_exposure_balance(im) for im in rgb_images])
        s_dyn = _norm01([score_dynamic_range_headroom(im) for im in rgb_images])
        s_tile = _norm01([score_tiling_artifact_free(im) for im in rgb_images])
        s_cast = _norm01([score_color_cast_neutrality(im) for im in rgb_images])
        s_sat = _norm01([score_saturation_balance(im) for im in rgb_images])
        s_skin = _norm01([score_skin_tone_naturalness(im) for im in rgb_images])
        s_vit = _norm01(_score_vit_quality())
        if expected_text and str(expected_text).strip():
            s_ocr = _norm01([score_ocr_match(im, expected_text) for im in rgb_images])
            combined = _weighted_sum(
                [s_clip, s_det, s_exp, s_dyn, s_tile, s_cast, s_sat, s_skin, s_vit, s_ocr],
                [0.16, 0.10, 0.10, 0.10, 0.08, 0.08, 0.08, 0.06, 0.18, 0.06],
            )
        else:
            combined = _weighted_sum(
                [s_clip, s_det, s_exp, s_dyn, s_tile, s_cast, s_sat, s_skin, s_vit],
                [0.18, 0.11, 0.11, 0.10, 0.09, 0.08, 0.08, 0.07, 0.18],
            )
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

    if metric in ("combo_count", "combo_count_vit"):
        use_vit_cnt = metric == "combo_count_vit"
        s_clip = _norm01(score_clip_similarity(rgb_images, prompt, device=device, model_id=clip_model_id))
        s_edge = _norm01([score_edge_sharpness(im) for im in rgb_images])
        s_vit_cnt = _norm01(_score_vit_quality()) if use_vit_cnt else None
        target_mode = str(expected_count_target or "auto").strip().lower()
        expected_count_value = int(expected_count or 0)
        object_hint = str(expected_count_object or "").strip().lower()
        use_object_scoring = False
        # auto => prefer explicit target, else infer people first, then non-people objects.
        if target_mode == "objects":
            if expected_count_value <= 0:
                expected_count_value, inferred_object_hint = infer_expected_object_count(prompt)
                if not object_hint:
                    object_hint = inferred_object_hint
            use_object_scoring = True
        elif target_mode == "people":
            if expected_count_value <= 0:
                expected_count_value = infer_expected_people_count(prompt)
            use_object_scoring = False
        else:
            if expected_count_value > 0:
                # Explicit count in auto mode defaults to people unless object hint exists.
                use_object_scoring = bool(object_hint)
            else:
                inferred_people_count = infer_expected_people_count(prompt)
                if inferred_people_count > 0:
                    expected_count_value = inferred_people_count
                    use_object_scoring = False
                else:
                    inferred_object_count, inferred_object_hint = infer_expected_object_count(prompt)
                    expected_count_value = inferred_object_count
                    object_hint = object_hint or inferred_object_hint
                    use_object_scoring = True
        if expected_count_value > 0:
            if use_object_scoring:
                s_cnt = _norm01(
                    [score_object_count_match(im, expected_count_value, object_hint=object_hint) for im in rgb_images]
                )
            else:
                s_cnt = _norm01([score_people_count_match(im, expected_count_value) for im in rgb_images])
            if expected_text and str(expected_text).strip():
                s_ocr = _norm01([score_ocr_match(im, expected_text) for im in rgb_images])
                if use_vit_cnt and s_vit_cnt is not None:
                    combined = _weighted_sum([s_clip, s_edge, s_cnt, s_ocr, s_vit_cnt], [0.32, 0.18, 0.26, 0.08, 0.16])
                else:
                    combined = _weighted_sum([s_clip, s_edge, s_cnt, s_ocr], [0.38, 0.22, 0.30, 0.10])
            else:
                if use_vit_cnt and s_vit_cnt is not None:
                    combined = _weighted_sum([s_clip, s_edge, s_cnt, s_vit_cnt], [0.38, 0.22, 0.25, 0.15])
                else:
                    combined = _weighted_sum([s_clip, s_edge, s_cnt], [0.45, 0.25, 0.30])
            return int(np.argmax(combined)), combined
        # No count target found; fall back to standard combo behavior.
        if expected_text and str(expected_text).strip():
            s_ocr = _norm01([score_ocr_match(im, expected_text) for im in rgb_images])
            if use_vit_cnt and s_vit_cnt is not None:
                combined = _weighted_sum([s_clip, s_edge, s_ocr, s_vit_cnt], [0.44, 0.28, 0.12, 0.16])
            else:
                combined = _weighted_sum([s_clip, s_edge, s_ocr], [0.5, 0.35, 0.15])
        else:
            if use_vit_cnt and s_vit_cnt is not None:
                combined = _weighted_sum([s_clip, s_edge, s_vit_cnt], [0.50, 0.30, 0.20])
            else:
                combined = _weighted_sum([s_clip, s_edge], [0.65, 0.35])
        return int(np.argmax(combined)), combined

    if metric == "aesthetic_realism":
        # Photo-style scoring without CLIP (for --pick-auto-no-clip).
        s_det = _norm01([score_photographic_detail_balance(im) for im in rgb_images])
        s_exp = _norm01([score_exposure_balance(im) for im in rgb_images])
        s_dyn = _norm01([score_dynamic_range_headroom(im) for im in rgb_images])
        s_tile = _norm01([score_tiling_artifact_free(im) for im in rgb_images])
        s_cast = _norm01([score_color_cast_neutrality(im) for im in rgb_images])
        s_sat = _norm01([score_saturation_balance(im) for im in rgb_images])
        s_skin = _norm01([score_skin_tone_naturalness(im) for im in rgb_images])
        if expected_text and str(expected_text).strip():
            s_ocr = _norm01([score_ocr_match(im, expected_text) for im in rgb_images])
            combined = _weighted_sum(
                [s_det, s_exp, s_dyn, s_tile, s_cast, s_sat, s_skin, s_ocr],
                [0.184, 0.184, 0.158, 0.105, 0.105, 0.105, 0.105, 0.054],
            )
        else:
            combined = _weighted_sum(
                [s_det, s_exp, s_dyn, s_tile, s_cast, s_sat, s_skin],
                [0.216, 0.203, 0.176, 0.122, 0.108, 0.095, 0.081],
            )
        return int(np.argmax(combined)), combined

    if metric == "combo_realism":
        s_clip = _norm01(score_clip_similarity(rgb_images, prompt, device=device, model_id=clip_model_id))
        s_det = _norm01([score_photographic_detail_balance(im) for im in rgb_images])
        s_exp = _norm01([score_exposure_balance(im) for im in rgb_images])
        s_dyn = _norm01([score_dynamic_range_headroom(im) for im in rgb_images])
        s_tile = _norm01([score_tiling_artifact_free(im) for im in rgb_images])
        s_cast = _norm01([score_color_cast_neutrality(im) for im in rgb_images])
        s_sat = _norm01([score_saturation_balance(im) for im in rgb_images])
        s_skin = _norm01([score_skin_tone_naturalness(im) for im in rgb_images])
        if expected_text and str(expected_text).strip():
            s_ocr = _norm01([score_ocr_match(im, expected_text) for im in rgb_images])
            combined = _weighted_sum(
                [s_clip, s_det, s_exp, s_dyn, s_tile, s_cast, s_sat, s_skin, s_ocr],
                [0.24, 0.14, 0.14, 0.12, 0.08, 0.08, 0.08, 0.08, 0.04],
            )
        else:
            combined = _weighted_sum(
                [s_clip, s_det, s_exp, s_dyn, s_tile, s_cast, s_sat, s_skin],
                [0.26, 0.16, 0.15, 0.13, 0.09, 0.08, 0.07, 0.06],
            )
        return int(np.argmax(combined)), combined

    return 0, [0.0] * image_count

"""Image editing: heuristic and segmentation inpaint masks, image dissection, part compositing, text rendering, edit-skill plans and the post-generation editing loop."""

from __future__ import annotations

import difflib
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from utils.generation.sample_argv import sample_argv

HEURISTIC_EDIT_REGIONS = frozenset({"face", "hands", "clothing", "background", "subject", "full"})


def normalize_heuristic_region(region: str | None) -> str:
    """Return a known region token; unrecognized values become ``subject``."""
    key = (region or "subject").strip().lower()
    return key if key in HEURISTIC_EDIT_REGIONS else "subject"


def heuristic_inpaint_mask(
    width: int,
    height: int,
    region: str | None,
    *,
    feather_radius: float = 5.0,
) -> Image.Image:
    """
    Build an ``L``-mode mask. **White** pixels are rewritten; **black** pixels are frozen (MDM inpaint).

    Unknown region strings fall back to ``subject``.
    """
    w, h = max(4, int(width)), max(4, int(height))
    rkey = normalize_heuristic_region(region)
    if rkey == "full":
        m = Image.new("L", (w, h), 255)
    elif rkey == "background":
        m = Image.new("L", (w, h), 255)
        draw = ImageDraw.Draw(m)
        sx0, sy0, sx1, sy1 = _subject_ellipse_bbox(w, h)
        draw.ellipse([sx0, sy0, sx1, sy1], fill=0)
    else:
        m = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(m)
        if rkey == "face":
            cx, cy = w / 2, h * 0.30
            rx, ry = w * 0.24, h * 0.20
            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
        elif rkey == "hands":
            draw.rectangle([int(w * 0.10), int(h * 0.56), int(w * 0.90), int(h * 0.98)], fill=255)
        elif rkey == "clothing":
            cx, cy = w / 2, h * 0.48
            rx, ry = w * 0.26, h * 0.40
            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
        elif rkey == "subject":
            sx0, sy0, sx1, sy1 = _subject_ellipse_bbox(w, h)
            draw.ellipse([sx0, sy0, sx1, sy1], fill=255)

    if feather_radius and feather_radius > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=float(feather_radius)))
    return m


def _subject_ellipse_bbox(w: int, h: int) -> tuple[float, float, float, float]:
    cx, cy = w / 2, h * 0.45
    rx, ry = w * 0.34, h * 0.48
    return cx - rx, cy - ry, cx + rx, cy + ry


def save_heuristic_mask(
    path: str | Path,
    width: int,
    height: int,
    region: str | None,
    *,
    feather_radius: float = 5.0,
) -> Path:
    """Write a heuristic mask PNG next to editable assets."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    heuristic_inpaint_mask(width, height, region, feather_radius=feather_radius).save(p, format="PNG")
    return p


_PRETRAINED = Path(__file__).resolve().parents[2] / "pretrained"
GDINO_REPO_PATH = _PRETRAINED / "GroundingDINO-Base"
SAM2_REPO_PATH = _PRETRAINED / "SAM2-Hiera-Large"

BBox = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class SegmentationMaskResult:
    mask: Image.Image  # mode "L"
    bbox: BBox | None
    mode: str
    notes: str = ""


_PHRASE_TO_REGION_TERMS = (
    (("background", "backdrop", "sky behind", "environment"), "background"),
    (("face", "facial", "eyes", "eyebrow", "mouth", "lips"), "face"),
    (("hands", "fingers", "palm"), "hands"),
    (("outfit", "dress", "shirt", "jacket", "clothing", "pants"), "clothing"),
)


def phrase_to_fallback_region(text: str) -> str | None:
    """If *text* looks like an edit noun phrase, map to a coarse heuristic region."""
    t = (text or "").strip().lower()
    if not t:
        return None
    if re.search(r"\b(full|everything|whole image|entire image)\b", t):
        return "full"
    for terms, reg in _PHRASE_TO_REGION_TERMS:
        if any(x in t for x in terms):
            return reg
    return None


def _pick_device_preference() -> str:
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


_gdino_pl: Any = None
_sam_pl: Any = None
_gdino_unavailable = object()
_sam_unavailable = object()


def _load_gdino():
    global _gdino_pl
    if _gdino_pl is _gdino_unavailable:
        return None
    if _gdino_pl is not None:
        return _gdino_pl
    try:
        import transformers
    except ImportError:
        _gdino_pl = _gdino_unavailable
        return None
    if not GDINO_REPO_PATH.exists():
        _gdino_pl = _gdino_unavailable
        return None
    dev = _pick_device_preference()
    dev_id = 0 if dev.startswith("cuda") else -1
    try:
        pl = transformers.pipeline(
            task="zero-shot-object-detection",
            model=str(GDINO_REPO_PATH),
            device=dev_id,
        )
    except Exception:
        _gdino_pl = _gdino_unavailable
        return None
    _gdino_pl = pl
    return _gdino_pl


def _load_sam2():
    global _sam_pl
    if _sam_pl is _sam_unavailable:
        return None
    if _sam_pl is not None:
        return _sam_pl
    try:
        import transformers
    except ImportError:
        _sam_pl = _sam_unavailable
        return None
    if not SAM2_REPO_PATH.exists():
        _sam_pl = _sam_unavailable
        return None
    dev = _pick_device_preference()
    dev_id = 0 if dev.startswith("cuda") else -1
    try:
        pl = transformers.pipeline(
            task="mask-generation",
            model=str(SAM2_REPO_PATH),
            device=dev_id,
        )
    except Exception:
        _sam_pl = _sam_unavailable
        return None
    _sam_pl = pl
    return _sam_pl


def _clip_box(box: BBox, w: int, h: int) -> BBox:
    x0, y0, x1, y1 = box
    x0 = max(0, min(w - 1, int(x0)))
    y0 = max(0, min(h - 1, int(y0)))
    x1 = max(x0 + 1, min(w, int(x1)))
    y1 = max(y0 + 1, min(h, int(y1)))
    return (x0, y0, x1, y1)


def _bbox_to_mask(size: tuple[int, int], box: BBox) -> Image.Image:
    w, h = size
    x0, y0, x1, y1 = _clip_box(box, w, h)
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0:y1, x0:x1] = 255
    return Image.fromarray(m, mode="L")


def _mask_from_sam_output(raw: object, fallback: Image.Image, w: int, h: int) -> Image.Image:
    """Normalize SAM pipeline output → ``L`` mask."""
    if isinstance(raw, list) and raw:
        first = raw[0]
        blob = first.get("mask") if isinstance(first, dict) else None
        if blob is None:
            return fallback
        if hasattr(blob, "convert"):
            m = blob.convert("L").resize((w, h), Image.Resampling.NEAREST)
            return m
        ma = np.array(blob).astype(np.uint8)
        if ma.ndim == 3:
            ma = ma[..., 0]
        m_arr = ((ma > 0).astype(np.uint8)) * 255
        return Image.fromarray(m_arr, mode="L").resize((w, h), Image.Resampling.NEAREST)
    return fallback


def mask_for_box(pil_rgb: Image.Image, box: BBox) -> tuple[Image.Image, bool]:
    """``L`` mask for ``box``: SAM2-refined when SAM2 is installed, else the filled box. Returns (mask, sam_used)."""
    w, h = pil_rgb.size
    bbox_mask = _bbox_to_mask((w, h), box)
    sam = _load_sam2()
    if sam is None:
        return bbox_mask, False
    try:
        return _mask_from_sam_output(sam(pil_rgb, input_boxes=[list(box)]), bbox_mask, w, h), True
    except Exception:
        return bbox_mask, False


def detect_box_for_phrase(
    pil_rgb: Image.Image,
    phrase: str,
    *,
    threshold: float = 0.2,
) -> tuple[BBox | None, float]:
    """Run Grounding DINO if available; return (bbox, score)."""
    gdino = _load_gdino()
    w, h = pil_rgb.size
    if gdino is None:
        return None, 0.0
    try:
        det = gdino(pil_rgb, candidate_labels=[phrase.strip()], threshold=float(threshold))
        if isinstance(det, list) and det:
            best = max(det, key=lambda d: float(d.get("score", 0.0)))
            b = best.get("box") or {}
            box = (
                int(b.get("xmin", 0)),
                int(b.get("ymin", 0)),
                int(b.get("xmax", w)),
                int(b.get("ymax", h)),
            )
            return _clip_box(box, w, h), float(best.get("score", 0.0))
    except Exception:
        return None, 0.0
    return None, 0.0


def build_segmentation_mask_for_edit(
    image: Image.Image | str | Path,
    target_phrase: str,
    *,
    feather_radius: float = 4.0,
    gdino_threshold: float = 0.2,
    use_vision_models: bool = True,
) -> SegmentationMaskResult:
    """
    Build a grayscale inpaint mask for *target_phrase* (“the face”, “the sword”, …).

    Args:
        image: RGB source (or path).
        target_phrase: Object / region wording for detection (short is better).
        feather_radius: Gaussian blur radius on ``L`` mask (0 to skip).
        gdino_threshold: Detection confidence floor for Grounding DINO.
        use_vision_models: If False, only keyword → heuristic_masks path.
    """
    phrase = (target_phrase or "").strip()
    if not phrase:
        raise ValueError("target_phrase must be non-empty")

    if isinstance(image, (str, Path)):
        pil = Image.open(Path(image)).convert("RGB")
    else:
        pil = image.convert("RGB")

    w, h = pil.size

    heuristic_label = phrase_to_fallback_region(phrase)
    fallback_region = normalize_heuristic_region(heuristic_label) if heuristic_label else None

    if not use_vision_models:
        if fallback_region:
            mask = heuristic_inpaint_mask(w, h, fallback_region, feather_radius=max(0.0, float(feather_radius)))
            return SegmentationMaskResult(mask=mask, bbox=None, mode="heuristic_only", notes="models disabled")
        mask = heuristic_inpaint_mask(w, h, "subject", feather_radius=max(0.0, float(feather_radius)))
        return SegmentationMaskResult(mask=mask, bbox=None, mode="heuristic_subject_default", notes="models disabled")

    box, score = detect_box_for_phrase(pil, phrase, threshold=gdino_threshold)
    if box is None and fallback_region:
        mask = heuristic_inpaint_mask(w, h, fallback_region, feather_radius=max(0.0, float(feather_radius)))
        return SegmentationMaskResult(
            mask=mask,
            bbox=None,
            mode="heuristic",
            notes="gdino_miss",
        )

    if box is None:
        mask = heuristic_inpaint_mask(w, h, "subject", feather_radius=max(0.0, float(feather_radius)))
        return SegmentationMaskResult(mask=mask, bbox=None, mode="heuristic_subject_default", notes="gdino_miss")

    mask, sam_used = mask_for_box(pil, box)
    mode = "gdino_plus_sam2" if sam_used else "gdino_bbox_only"
    notes = f"score={score:.3f}"

    if feather_radius and feather_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=float(feather_radius)))

    return SegmentationMaskResult(mask=mask, bbox=box, mode=mode, notes=notes)


ImageLike = str | Path


@dataclass(frozen=True, slots=True)
class PartRequest:
    """A single user-requested part."""

    part: str
    source_index: int = 0  # 0-based reference image index
    role: str = "foreground"  # "foreground" | "background" | "style" | "text" | ...
    hint: str = ""  # freeform extra info


@dataclass(slots=True)
class DissectedPart:
    """Result of attempting to extract a requested part."""

    request: PartRequest
    bbox: BBox | None = None
    mask_path: str | None = None
    crop_path: str | None = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


_IMG_REF_RE = re.compile(
    r"\b(?:image|img|ref|reference)\s*(?P<idx>\d+)\b",
    re.IGNORECASE,
)

# Matches patterns like:
# - "use the hat from image 1"
# - "take background from ref2"
# - "use eyes from img 3"
_PART_FROM_IMG_RE = re.compile(
    r"\b(?:use|take|extract|borrow|copy|keep)\s+"
    r"(?:the\s+)?(?P<part>[a-z0-9][a-z0-9 _\\-]{0,48}?)\s+"
    r"(?:from|in)\s+(?:the\s+)?(?:image|img|ref|reference)\s*(?P<idx>\d+)\b",
    re.IGNORECASE,
)

# Background-only shorthand:
_BG_FROM_IMG_RE = re.compile(
    r"\b(?:use|take|keep)\s+(?:the\s+)?background\s+from\s+(?:image|img|ref|reference)\s*(?P<idx>\d+)\b",
    re.IGNORECASE,
)


def parse_part_requests(prompt: str, *, default_source_index: int = 0) -> list[PartRequest]:
    """
    Extract PartRequest entries from a user prompt.

    This is intentionally conservative: if we can't parse a request, we don't invent one.
    """
    p = (prompt or "").strip()
    if not p:
        return []

    reqs: list[PartRequest] = []

    # 1) Explicit "X from image N" requests.
    for m in _PART_FROM_IMG_RE.finditer(p):
        part = (m.group("part") or "").strip()
        idx = int(m.group("idx")) - 1
        if part:
            role = "background" if "background" in part.lower() else "foreground"
            reqs.append(PartRequest(part=part, source_index=max(0, idx), role=role))

    # 2) Background shorthand (avoids "part" being parsed as "background from image").
    for m in _BG_FROM_IMG_RE.finditer(p):
        idx = int(m.group("idx")) - 1
        reqs.append(PartRequest(part="background", source_index=max(0, idx), role="background"))

    # 3) If user references images but doesn't specify parts, emit a soft request.
    # Example: "use image 2 as reference for outfit" (no explicit "from image" phrase).
    if not reqs:
        refs = [int(m.group("idx")) - 1 for m in _IMG_REF_RE.finditer(p)]
        refs = [r for r in refs if r >= 0]
        if refs:
            # Minimal: treat as "style/identity reference" rather than a hard crop request.
            reqs.append(PartRequest(part="reference", source_index=refs[0], role="style"))

    # Deduplicate (same part+idx+role).
    seen = set()
    out: list[PartRequest] = []
    for r in reqs:
        key = (r.part.lower().strip(), int(r.source_index), r.role.lower().strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(r)

    # If no explicit refs, allow default implicit reference if prompt uses "use this image" language.
    if not out and re.search(r"\b(use|keep)\s+(this|the)\s+image\b", p, re.IGNORECASE):
        out.append(PartRequest(part="reference", source_index=int(default_source_index), role="style"))

    return out


def visual_facts_from_requests(
    prompt: str,
    requests: Sequence[PartRequest],
    *,
    num_reference_images: int,
) -> list[str]:
    """
    Build short textual "facts" describing what to extract, suitable for RAG merging.
    """
    facts: list[str] = []
    if not requests:
        return facts

    facts.append("Reference-image dissection requested. Treat these as constraints when generating the final image.")
    facts.append(f"Number of reference images provided: {int(num_reference_images)}")

    for r in requests:
        img_n = int(r.source_index) + 1
        facts.append(f"Use `{r.part}` from reference image {img_n} (role={r.role}).")

    # Add the original prompt as a reminder for dissection intent, without duplicating too much.
    short = (prompt or "").strip()
    if short:
        facts.append(f"User intent: {short[:240] + ('...' if len(short) > 240 else '')}")
    return facts


def dissect_images_to_parts(
    prompt: str,
    reference_images: Sequence[ImageLike],
    *,
    output_dir: str | Path,
    default_source_index: int = 0,
    enable_heavy_models: bool = True,
) -> tuple[list[PartRequest], list[DissectedPart], list[str]]:
    """
    High-level helper used by RAG/generation.

    Returns (requests, parts, facts). With ``enable_heavy_models=False`` the parts carry no masks; without
    Grounding DINO a part's mask is the centre box.
    """
    reqs = parse_part_requests(prompt, default_source_index=default_source_index)
    facts = visual_facts_from_requests(prompt, reqs, num_reference_images=len(reference_images))

    parts: list[DissectedPart] = [DissectedPart(request=r) for r in reqs]

    if not enable_heavy_models:
        return reqs, parts, facts

    import numpy as np
    from PIL import Image

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save_mask_and_crop(pil: Image.Image, mask: Image.Image, stem: str) -> tuple[str, str]:
        mask_p, crop_p = out_dir / f"{stem}_mask.png", out_dir / f"{stem}_crop.png"
        mask.save(mask_p)
        alpha = np.asarray(mask, dtype=np.float32)[..., None] / 255.0
        Image.fromarray((np.asarray(pil, dtype=np.float32) * alpha).round().astype(np.uint8)).save(crop_p)
        return str(mask_p), str(crop_p)

    for i, dp in enumerate(parts):
        r = dp.request
        if not 0 <= r.source_index < len(reference_images):
            dp.metadata["error"] = "bad_source_index"
            continue
        pil = Image.open(Path(reference_images[r.source_index])).convert("RGB")
        w, h = pil.size
        stem = f"ref{r.source_index + 1}_{re.sub(r'[^a-z0-9]+', '_', r.part.lower()).strip('_')}_{i:02d}"

        if r.role == "background" or r.part.lower().strip() == "background":
            dp.mask_path, dp.crop_path = _save_mask_and_crop(pil, Image.new("L", (w, h), 255), stem)
            dp.bbox = (0, 0, w, h)
            dp.confidence = 1.0
            dp.metadata["mode"] = "background_full"
            continue

        box, score = detect_box_for_phrase(pil, r.part, threshold=0.15)
        if box is None:
            # No detector or no hit: the central 60% of the image stands in for the part.
            box, score = (int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)), 0.05
            dp.metadata["bbox_fallback"] = True
        dp.bbox = box
        mask, sam_used = mask_for_box(pil, box)
        dp.mask_path, dp.crop_path = _save_mask_and_crop(pil, mask, stem)
        dp.confidence = float(max(0.0, min(1.0, score)))
        dp.metadata["mode"] = ("center" if dp.metadata.get("bbox_fallback") else "gdino") + (
            "+sam" if sam_used else "+bbox"
        )

    return reqs, parts, facts


@dataclass(frozen=True, slots=True)
class CompositeSpec:
    init_image_path: str
    mask_path: str
    preserved_parts: int
    background_locked: bool


def _load_rgb(path: str | Path, *, size: tuple[int, int] | None = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.Resampling.LANCZOS)
    return img


def _load_mask(path: str | Path, *, size: tuple[int, int]) -> Image.Image:
    m = Image.open(path).convert("L")
    if m.size != size:
        m = m.resize(size, Image.Resampling.NEAREST)
    return m


def build_init_and_inpaint_mask(
    *,
    reference_images: Sequence[str | Path],
    parts: Sequence[DissectedPart],
    output_dir: str | Path,
    target_size: tuple[int, int],
    lock_background_if_requested: bool = True,
) -> CompositeSpec:
    """
    Compose an init image and an inpaint mask from dissection outputs.

    Mask semantics follow `sample.py`: white=inpaint, black=keep.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    W, H = int(target_size[0]), int(target_size[1])
    size = (W, H)

    # Choose background base.
    bg_part = next((p for p in parts if p.request.role == "background" or p.request.part.lower() == "background"), None)
    if bg_part is not None and bg_part.crop_path:
        base = _load_rgb(bg_part.crop_path, size=size)
        background_locked = bool(lock_background_if_requested)
    else:
        # Default: use first ref as a loose base if available; otherwise blank.
        if reference_images:
            base = _load_rgb(reference_images[0], size=size)
        else:
            base = Image.new("RGB", size, (0, 0, 0))
        background_locked = False

    base_arr = np.array(base).astype(np.uint8)

    # Preserve-mask: white means "preserve/keep"; we will invert at end.
    preserve = np.zeros((H, W), dtype=np.uint8)

    if background_locked:
        preserve[:, :] = 255

    preserved_parts = 0

    # Overlay preserved parts.
    for p in parts:
        if p.request.role == "background" or p.request.part.lower() == "background":
            continue
        if not p.crop_path or not p.mask_path:
            continue
        try:
            crop = _load_rgb(p.crop_path, size=size)
            m = _load_mask(p.mask_path, size=size)
        except Exception:
            continue

        crop_arr = np.array(crop).astype(np.uint8)
        m_arr = (np.array(m).astype(np.uint8) >= 128).astype(np.uint8) * 255

        # Composite crop onto base using mask.
        m3 = (m_arr[..., None] / 255.0).astype(np.float32)
        base_arr = (
            (base_arr.astype(np.float32) * (1.0 - m3) + crop_arr.astype(np.float32) * m3).round().astype(np.uint8)
        )

        # Update preserve map (keep these pixels).
        preserve = np.maximum(preserve, m_arr)
        preserved_parts += 1

    init = Image.fromarray(base_arr, mode="RGB")

    # Inpaint mask: white=inpaint, black=keep.
    # If we are not background-locked, we inpaint everything except preserved parts.
    inpaint = (255 - preserve).astype(np.uint8)
    mask = Image.fromarray(inpaint, mode="L")

    init_path = out_dir / "composite_init.png"
    mask_path = out_dir / "composite_mask.png"
    init.save(init_path)
    mask.save(mask_path)

    return CompositeSpec(
        init_image_path=str(init_path),
        mask_path=str(mask_path),
        preserved_parts=int(preserved_parts),
        background_locked=bool(background_locked),
    )


class TextRenderingEngine:
    """Advanced text rendering for AI image generation."""

    def __init__(self):
        self.font_cache = {}
        self.text_styles = {
            "modern": {"weight": "bold", "spacing": 1.2, "kerning": 0.1},
            "classic": {"weight": "normal", "spacing": 1.0, "kerning": 0.05},
            "display": {"weight": "black", "spacing": 1.5, "kerning": 0.2},
            "script": {"weight": "normal", "spacing": 0.9, "kerning": -0.05},
            "monospace": {"weight": "normal", "spacing": 1.0, "kerning": 0.0},
        }

        self.typography_prompts = {
            "logo": "clean typography, professional logo design, vector style, crisp text",
            "poster": "bold typography, eye-catching text, poster design, readable font",
            "banner": "large text, banner style, promotional typography, clear lettering",
            "sign": "signage typography, outdoor sign, readable from distance, bold letters",
            "book": "book typography, serif font, readable text, literary design",
            "magazine": "magazine layout, editorial typography, modern font, clean design",
            "business": "corporate typography, professional font, business card style",
            "artistic": "artistic typography, creative lettering, stylized text, decorative font",
        }

    def extract_text_requirements(self, prompt: str) -> dict[str, Any]:
        """Extract text rendering requirements from prompt."""
        text_info = {
            "has_text": False,
            "text_content": [],
            "text_style": "modern",
            "text_placement": "center",
            "text_size": "medium",
            "text_color": "auto",
            "typography_type": "general",
        }

        # Look for quoted text that should be rendered
        quoted_text = re.findall(r'"([^"]*)"', prompt)
        if quoted_text:
            text_info["has_text"] = True
            text_info["text_content"] = quoted_text

        # Look for text in brackets
        bracketed_text = re.findall(r"\[text:\s*([^\]]*)\]", prompt, re.IGNORECASE)
        if bracketed_text:
            text_info["has_text"] = True
            text_info["text_content"].extend(bracketed_text)

        # Detect typography type
        for typ, keywords in {
            "logo": ["logo", "brand", "company", "business"],
            "poster": ["poster", "advertisement", "ad", "promotional"],
            "banner": ["banner", "header", "title"],
            "sign": ["sign", "signage", "street", "shop"],
            "book": ["book", "novel", "literature", "cover"],
            "magazine": ["magazine", "editorial", "article"],
            "business": ["business card", "corporate", "professional"],
            "artistic": ["artistic", "creative", "decorative", "stylized"],
        }.items():
            if any(keyword in prompt.lower() for keyword in keywords):
                text_info["typography_type"] = typ
                break

        # Detect text style
        style_keywords = {
            "modern": ["modern", "contemporary", "clean", "minimal"],
            "classic": ["classic", "traditional", "serif", "elegant"],
            "display": ["bold", "large", "display", "headline"],
            "script": ["script", "handwritten", "cursive", "calligraphy"],
            "monospace": ["monospace", "code", "technical", "digital"],
        }

        for style, keywords in style_keywords.items():
            if any(keyword in prompt.lower() for keyword in keywords):
                text_info["text_style"] = style
                break

        # Detect placement
        placement_keywords = {
            "top": ["top", "header", "above"],
            "bottom": ["bottom", "footer", "below"],
            "left": ["left", "side"],
            "right": ["right", "side"],
            "center": ["center", "middle", "centered"],
        }

        for placement, keywords in placement_keywords.items():
            if any(keyword in prompt.lower() for keyword in keywords):
                text_info["text_placement"] = placement
                break

        # Detect size
        size_keywords = {
            "small": ["small", "tiny", "fine", "subtitle"],
            "medium": ["medium", "normal", "regular"],
            "large": ["large", "big", "headline", "title"],
            "huge": ["huge", "massive", "giant", "billboard"],
        }

        for size, keywords in size_keywords.items():
            if any(keyword in prompt.lower() for keyword in keywords):
                text_info["text_size"] = size
                break

        return text_info

    def enhance_prompt_for_text(self, prompt: str, text_info: dict[str, Any]) -> str:
        """Enhance prompt with text rendering instructions."""
        if not text_info["has_text"]:
            return prompt

        # Add typography-specific prompts
        typography_enhancement = self.typography_prompts.get(
            text_info["typography_type"], "clear text, readable typography, professional font"
        )

        # Add style-specific enhancements
        style_enhancement = ""
        if text_info["text_style"] == "modern":
            style_enhancement = "sans-serif font, clean lines, modern typography"
        elif text_info["text_style"] == "classic":
            style_enhancement = "serif font, traditional typography, elegant lettering"
        elif text_info["text_style"] == "display":
            style_enhancement = "bold font, display typography, strong lettering"
        elif text_info["text_style"] == "script":
            style_enhancement = "script font, handwritten style, flowing letters"
        elif text_info["text_style"] == "monospace":
            style_enhancement = "monospace font, fixed-width, technical typography"

        # Combine enhancements
        enhanced_prompt = f"{prompt}, {typography_enhancement}, {style_enhancement}"

        # Add text quality boosters
        text_quality_tags = [
            "sharp text",
            "crisp typography",
            "readable font",
            "clear lettering",
            "high resolution text",
            "professional typography",
            "perfect text rendering",
        ]

        enhanced_prompt += f", {', '.join(text_quality_tags[:3])}"

        return enhanced_prompt

    def create_text_mask(self, image_size: tuple[int, int], text_info: dict[str, Any]) -> torch.Tensor | None:
        """Create a mask for text placement guidance."""
        if not text_info["has_text"] or not text_info["text_content"]:
            return None

        width, height = image_size
        mask = torch.zeros((1, height, width), dtype=torch.float32)

        # Define text regions based on placement
        placement = text_info["text_placement"]

        if placement == "top":
            mask[:, : height // 4, :] = 1.0
        elif placement == "bottom":
            mask[:, 3 * height // 4 :, :] = 1.0
        elif placement == "left":
            mask[:, :, : width // 3] = 1.0
        elif placement == "right":
            mask[:, :, 2 * width // 3 :] = 1.0
        elif placement == "center":
            mask[:, height // 3 : 2 * height // 3, width // 4 : 3 * width // 4] = 1.0
        else:
            # Default center placement
            mask[:, height // 3 : 2 * height // 3, width // 4 : 3 * width // 4] = 1.0

        return mask

    def validate_text_rendering(self, image: Image.Image, expected_text: list[str]) -> dict[str, Any]:
        """Validate text rendering quality in generated image."""
        try:
            import pytesseract

            # Extract text from image
            extracted_text = pytesseract.image_to_string(image)

            validation_results = {
                "text_detected": bool(extracted_text.strip()),
                "extracted_text": extracted_text.strip(),
                "expected_text": expected_text,
                "accuracy_score": 0.0,
                "readability_score": 0.0,
                "issues": [],
            }

            if expected_text and extracted_text.strip():
                # Calculate accuracy
                expected_combined = " ".join(expected_text).lower()
                extracted_lower = extracted_text.lower()

                # Simple word-based accuracy
                expected_words = set(expected_combined.split())
                extracted_words = set(extracted_lower.split())

                word_accuracy = 0.0
                if expected_words:
                    word_accuracy = len(expected_words & extracted_words) / len(expected_words)

                # Fuzzy string similarity helps tolerate OCR tokenization differences.
                ratio = difflib.SequenceMatcher(None, expected_combined, extracted_lower).ratio()

                validation_results["accuracy_score"] = max(float(word_accuracy), float(ratio))

                # Readability assessment
                readability = self._assess_readability(extracted_text)
                validation_results["readability_score"] = readability

                # Identify issues
                if validation_results["accuracy_score"] < 0.5:
                    validation_results["issues"].append("Low text accuracy")
                if readability < 0.6:
                    validation_results["issues"].append("Poor readability")
                if len(extracted_text) < 3:
                    validation_results["issues"].append("Very short text detected")

            return validation_results

        except ImportError:
            return {"text_detected": False, "error": "pytesseract not available for text validation"}

    def _assess_readability(self, text: str) -> float:
        """Assess text readability based on character patterns."""
        if not text.strip():
            return 0.0

        # Basic readability metrics
        total_chars = len(text)
        alpha_chars = sum(1 for c in text if c.isalpha())
        space_chars = sum(1 for c in text if c.isspace())

        # Readability score based on character distribution
        alpha_ratio = alpha_chars / total_chars if total_chars > 0 else 0
        space_ratio = space_chars / total_chars if total_chars > 0 else 0

        # Good text should have high alpha ratio and reasonable spacing
        readability = (alpha_ratio * 0.7) + (min(space_ratio * 5, 0.3) * 0.3)

        return min(readability, 1.0)


class TypographyStyler:
    """Typography styling and enhancement system."""

    def __init__(self):
        self.font_families = {
            "sans_serif": ["Arial", "Helvetica", "Open Sans", "Roboto", "Lato"],
            "serif": ["Times", "Georgia", "Playfair", "Merriweather", "Crimson"],
            "display": ["Impact", "Bebas", "Oswald", "Montserrat", "Raleway"],
            "script": ["Pacifico", "Dancing Script", "Great Vibes", "Allura"],
            "monospace": ["Courier", "Monaco", "Consolas", "Source Code Pro"],
        }

        self.color_schemes = {
            "high_contrast": [("#000000", "#FFFFFF"), ("#FFFFFF", "#000000")],
            "warm": [("#8B4513", "#FFF8DC"), ("#CD853F", "#FFFACD")],
            "cool": [("#191970", "#F0F8FF"), ("#4682B4", "#E6F3FF")],
            "vibrant": [("#FF6347", "#FFFAF0"), ("#32CD32", "#F0FFF0")],
            "elegant": [("#2F4F4F", "#F5F5DC"), ("#696969", "#F8F8FF")],
        }

    def suggest_typography_improvements(self, prompt: str, text_info: dict[str, Any]) -> list[str]:
        """Suggest typography improvements for better text rendering."""
        suggestions = []

        if text_info["has_text"]:
            # Font suggestions
            typography_type = text_info["typography_type"]
            if typography_type == "logo":
                suggestions.append("Use bold, sans-serif font for logo clarity")
                suggestions.append("Ensure high contrast between text and background")
            elif typography_type == "poster":
                suggestions.append("Use display font for maximum impact")
                suggestions.append("Consider hierarchical text sizing")
            elif typography_type == "book":
                suggestions.append("Use serif font for better readability")
                suggestions.append("Maintain consistent line spacing")

            # Style suggestions
            if "gradient" in prompt.lower():
                suggestions.append("Avoid gradients on text for better readability")

            if "small" in prompt.lower() and "text" in prompt.lower():
                suggestions.append("Increase text size for better visibility")

            # Color suggestions
            if any(color in prompt.lower() for color in ["red", "blue", "green"]):
                suggestions.append("Ensure sufficient contrast with background color")

        return suggestions

    def generate_text_variations(self, base_prompt: str, text_content: list[str]) -> list[str]:
        """Generate prompt variations optimized for text rendering."""
        variations = []

        for style in ["modern", "classic", "bold", "elegant"]:
            for placement in ["centered", "top", "bottom"]:
                variation = f"{base_prompt}, {style} typography, {placement} text"
                if text_content:
                    variation += f', text says "{text_content[0]}"'
                variation += ", sharp text, readable font, high quality typography"
                variations.append(variation)

        return variations[:6]  # Return top 6 variations


class TextAwareInpainting:
    """Text-aware inpainting for editing text in images."""

    def __init__(self):
        # pytesseract.image_to_data conf is usually an integer in [0, 100]
        self.text_detection_threshold = 60

    def detect_text_regions(self, image: Image.Image) -> list[dict[str, Any]]:
        """Detect text regions in an image for targeted editing."""
        try:
            import pytesseract
            from pytesseract import Output

            # Get detailed text detection data
            data = pytesseract.image_to_data(image, output_type=Output.DICT)

            text_regions = []
            n_boxes = len(data["level"])

            for i in range(n_boxes):
                confidence = int(data["conf"][i])
                if confidence >= int(self.text_detection_threshold):  # Good confidence threshold
                    x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                    text = data["text"][i].strip()

                    if text:  # Only include regions with actual text
                        text_regions.append(
                            {"bbox": (x, y, x + w, y + h), "text": text, "confidence": confidence, "area": w * h}
                        )

            return text_regions

        except ImportError:
            return []

    def create_text_edit_mask(self, image: Image.Image, target_text: str = None) -> Image.Image | None:
        """Create a mask for text editing operations."""
        text_regions = self.detect_text_regions(image)

        if not text_regions:
            return None

        # Decide which regions to use (target_text acts as a focus hint).
        if target_text is not None and str(target_text).strip():
            t = str(target_text).strip().lower()
            focused = [r for r in text_regions if t in str(r.get("text", "")).lower()]
            # If we didn't match anything, fall back to all detected regions.
            if focused:
                regions_to_use = focused
            else:
                regions_to_use = text_regions
        else:
            regions_to_use = text_regions

        # "FreeText-like": build line-level masks rather than per-word rectangles.
        # Cluster word boxes by their vertical centers and then take the union bbox per cluster.
        def _bbox_area(b):
            x1, y1, x2, y2 = b
            return max(0, x2 - x1) * max(0, y2 - y1)

        # Compute typical line height for clustering tolerance.
        heights = [max(1, (r["bbox"][3] - r["bbox"][1])) for r in regions_to_use]
        avg_h = float(np.median(heights)) if heights else 12.0
        # Tolerance for considering two words part of the same line.
        y_tol = max(8.0, avg_h * 0.55)

        regions_sorted = sorted(regions_to_use, key=lambda r: (r["bbox"][1] + r["bbox"][3]) * 0.5)
        clusters: list[list[dict[str, Any]]] = []
        cluster_centers: list[float] = []

        for r in regions_sorted:
            x1, y1, x2, y2 = r["bbox"]
            yc = (y1 + y2) * 0.5
            placed = False
            for ci in range(len(clusters)):
                if abs(yc - cluster_centers[ci]) <= y_tol:
                    clusters[ci].append(r)
                    # Update cluster center (mean)
                    cluster_centers[ci] = float(
                        np.mean([((rr["bbox"][1] + rr["bbox"][3]) * 0.5) for rr in clusters[ci]])
                    )
                    placed = True
                    break
            if not placed:
                clusters.append([r])
                cluster_centers.append(yc)

        # Create mask image: white regions are inpainted; black elsewhere is kept.
        mask = Image.new("L", image.size, 0)  # Black background
        draw = ImageDraw.Draw(mask)

        # Padding proportional to bbox size to better catch broken glyphs / kerning issues.
        pad_x_ratio = 0.08
        pad_y_ratio = 0.10
        min_pad_px = 2

        for cluster in clusters:
            # Optional: ignore tiny clusters (noise).
            union = [10**9, 10**9, -1, -1]
            for r in cluster:
                x1, y1, x2, y2 = r["bbox"]
                union[0] = min(union[0], x1)
                union[1] = min(union[1], y1)
                union[2] = max(union[2], x2)
                union[3] = max(union[3], y2)
            if union[2] <= union[0] or union[3] <= union[1]:
                continue
            ux1, uy1, ux2, uy2 = union
            if _bbox_area((ux1, uy1, ux2, uy2)) < 50:
                continue

            w = ux2 - ux1
            h = uy2 - uy1
            pad_x = max(min_pad_px, int(w * pad_x_ratio))
            pad_y = max(min_pad_px, int(h * pad_y_ratio))
            nx1 = max(0, ux1 - pad_x)
            ny1 = max(0, uy1 - pad_y)
            nx2 = min(image.size[0], ux2 + pad_x)
            ny2 = min(image.size[1], uy2 + pad_y)
            draw.rectangle((nx1, ny1, nx2, ny2), fill=255)

        # Topology refinement: small morphological closing to connect broken glyph strokes.
        try:
            import cv2

            arr = np.array(mask.convert("L"))
            if arr.max() > 0:
                # Close small gaps inside text regions.
                kernel = np.ones((3, 3), dtype=np.uint8)
                arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel, iterations=1)
        except Exception:
            pass

        return mask

    def suggest_text_edits(self, image: Image.Image) -> list[dict[str, Any]]:
        """Suggest possible text edits for an image."""
        text_regions = self.detect_text_regions(image)

        suggestions = []
        for region in text_regions:
            suggestions.append(
                {
                    "original_text": region["text"],
                    "bbox": region["bbox"],
                    "edit_types": ["replace", "remove", "style_change"],
                    "confidence": region["confidence"],
                }
            )

        return suggestions


def create_text_rendering_pipeline():
    """Create a complete text rendering pipeline."""
    return {"engine": TextRenderingEngine(), "styler": TypographyStyler(), "inpainting": TextAwareInpainting()}


SkillKind = Literal[
    "inpaint_region",
    "glyph_text",
    "identity_lock",
]


@dataclass
class EditSkill:
    kind: SkillKind
    prompt: str = ""
    region: str = ""  # face|hands|clothing|background|subject|full
    strength: float = 0.45
    mask_path: str = ""
    init_image: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EditSkillPlan:
    skills: list[EditSkill] = field(default_factory=list)
    sample_argv: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skills": [s.to_dict() for s in self.skills],
            "sample_argv": list(self.sample_argv),
            "notes": list(self.notes),
        }


@dataclass
class IdeogramPlan:
    """Text-in-image + layout compile (Ideogram-like)."""

    prompt: str
    glyph_texts: list[str] = field(default_factory=list)
    design_brief: dict[str, Any] = field(default_factory=dict)
    box_layout: dict[str, Any] | None = None
    palette_hex: list[str] = field(default_factory=list)
    sample_argv: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_edit_skills(
    *,
    init_image: str,
    prompt: str = "",
    fix_regions: list[str] | None = None,
    change_outfit: str = "",
    glyph_text: str = "",
    identity_ref: str = "",
    work_dir: str | Path = "",
    width: int = 1024,
    height: int = 1024,
) -> EditSkillPlan:
    """Build a sequenced edit plan (masks + argv hints)."""
    work = Path(work_dir or "outputs/edit_skills")
    work.mkdir(parents=True, exist_ok=True)
    plan = EditSkillPlan()
    argv_values: dict[str, Any] = {}
    regions = list(fix_regions or [])
    if change_outfit:
        regions.append("clothing")
        prompt = f"{prompt}, {change_outfit}".strip(", ")

    if regions and init_image:
        primary = regions[0] if len(regions) == 1 else "subject"
        for reg in regions:
            mask_path = work / f"mask_{reg}.png"
            try:
                save_heuristic_mask(mask_path, width=width, height=height, region=reg)
            except Exception as exc:
                plan.notes.append(f"mask {reg}: {exc}")
                continue
            skill = EditSkill(
                kind="inpaint_region",
                prompt=prompt or f"fix {reg}, high quality, correct anatomy",
                region=reg,
                mask_path=str(mask_path),
                init_image=init_image,
            )
            plan.skills.append(skill)
        if plan.skills:
            s0 = plan.skills[0]
            argv_values.update(
                init_image=init_image, mask=s0.mask_path, prompt=s0.prompt, strength=s0.strength, inpaint_mode="mdm"
            )
            plan.notes.append(f"primary inpaint region={primary}")

    if glyph_text:
        plan.skills.append(EditSkill(kind="glyph_text", prompt=glyph_text, extra={"text": glyph_text}))
        argv_values.update(text_in_image=True, expected_text=glyph_text)

    if identity_ref:
        plan.skills.append(EditSkill(kind="identity_lock", init_image=identity_ref))
        argv_values["reference_image"] = identity_ref

    plan.sample_argv = sample_argv(argv_values)
    (work / "edit_plan.json").write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    return plan


def plan_ideogram_layout(
    prompt: str,
    *,
    texts: list[str] | None = None,
    palette_hex: list[str] | None = None,
    layout: str = "poster",  # poster | social | logo
) -> IdeogramPlan:
    """Compile Ideogram-like text + layout argv from prompt + explicit strings (the box layout is the caller's to save)."""

    found = TextRenderingEngine().extract_text_requirements(prompt or "")["text_content"]
    glyphs = list(texts or []) or [s.strip() for s in found if s.strip()]

    brief: dict[str, Any] = {
        "intent": layout,
        "prompt": prompt,
        "text_elements": [{"text": t, "role": "headline" if i == 0 else "body"} for i, t in enumerate(glyphs)],
    }
    # Simple vertical stack boxes for text regions
    regions = []
    for i, t in enumerate(glyphs[:4]):
        y0 = 0.08 + i * 0.18
        regions.append({"id": f"text_{i}", "prompt": f'text "{t}"', "box": [0.1, y0, 0.9, y0 + 0.14]})
    box = {"mode": "typography", "regions": regions, "global_prompt": prompt} if regions else None

    argv = sample_argv({"prompt": prompt, "text_in_image": True, "expected_text": ",".join(glyphs)})
    return IdeogramPlan(
        prompt=prompt,
        glyph_texts=glyphs,
        design_brief=brief,
        box_layout=box,
        palette_hex=list(palette_hex or []),
        sample_argv=argv,
        notes=[f"{len(glyphs)} glyph strings", f"layout={layout}"],
    )


_log = logging.getLogger(__name__)

ActionKind = Literal[
    "ocr_fix",
    "inpaint_region",
    "img2img",
    "rag_prompt_delta",
    "art_post",
    "prompt_realign",
]

_STOP = frozenset(
    """
    a an the and or of to in on at for with from by as is are was were be been being
    this that these those it its into over under near very more most less least
    image photo picture scene shot view style high quality detailed masterpiece
    """.split()
)

_ANATOMY_CUES = (
    "hand",
    "hands",
    "finger",
    "fingers",
    "face",
    "eyes",
    "arm",
    "legs",
    "anatomy",
    "portrait",
    "person",
    "woman",
    "man",
    "girl",
    "boy",
)


@dataclass(slots=True)
class EditAction:
    """One concrete edit to apply in the phase."""

    kind: ActionKind
    reason: str
    region: str | None = None  # face | hands | clothing | background | subject | full
    prompt_delta: str = ""
    negative_delta: str = ""
    strength: float = 0.55
    priority: int = 50  # lower runs first


@dataclass(slots=True)
class Diagnosis:
    """Signals from the verifier pass."""

    clip_score: float = 0.0
    sharpness: float = 0.0
    exposure: float = 0.0
    aesthetic: float = 0.0
    gate_passed: bool = False
    gate_failures: list[str] = field(default_factory=list)
    missing_tokens: list[str] = field(default_factory=list)
    expected_text: list[str] = field(default_factory=list)
    needs_ocr: bool = False
    needs_anatomy: bool = False
    piece_labels: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EditingPhaseConfig:
    """Knobs for the editing-phase loop."""

    max_iters: int = 3
    min_clip: float = 0.28
    min_sharpness: float = 0.15
    min_exposure: float = 0.25
    min_aesthetic: float = 0.0
    enable_ocr: bool = True
    enable_pieces: bool = True
    enable_rag: bool = True
    enable_art_post: bool = True
    enable_anatomy: bool = True
    img2img_strength: float = 0.45
    inpaint_strength: float = 0.62
    refine_steps: int = 20
    scheduler: str = "ays_dit"
    solver: str = "dpmpp_2m"
    dry_run: bool = False  # plan only; never call sample.py
    device: str = "cuda"


@dataclass(slots=True)
class EditingPhaseResult:
    """Outcome of ``run_editing_phase``."""

    image_rgb: np.ndarray
    prompt: str
    negative_prompt: str
    diagnosis_history: list[Diagnosis] = field(default_factory=list)
    actions_applied: list[EditAction] = field(default_factory=list)
    iterations: int = 0
    stopped_reason: str = ""
    output_path: str | None = None
    piece_dir: str | None = None


# ---------------------------------------------------------------------------
# Prompt / piece helpers
# ---------------------------------------------------------------------------


def extract_prompt_tokens(prompt: str, *, max_tokens: int = 24) -> list[str]:
    """Content-ish tokens from the prompt (for coverage / piece labels)."""
    raw = re.findall(r"[A-Za-z][A-Za-z0-9'\\-]{2,}", prompt.lower())
    out: list[str] = []
    seen: set[str] = set()
    for t in raw:
        if t in _STOP or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_tokens:
            break
    return out


def expected_render_text(prompt: str) -> list[str]:
    """Quoted / [text:] strings that should be readable in the image."""
    try:
        info = TextRenderingEngine().extract_text_requirements(prompt)
        return list(info.get("text_content") or [])
    except Exception:
        quoted = re.findall(r'"([^"]+)"', prompt)
        bracketed = re.findall(r"\[text:\s*([^\]]*)\]", prompt, flags=re.I)
        return [*(quoted or []), *(bracketed or [])]


def infer_piece_labels(prompt: str, *, max_pieces: int = 6) -> list[str]:
    """
    REVE-style piece list from the prompt.

    Maps common subject words onto heuristic inpaint regions; leftover nouns
    become ``subject`` pieces for documentation / future SAM grounding.
    """
    tokens = extract_prompt_tokens(prompt, max_tokens=32)
    labels: list[str] = []
    region_hits = {
        "face": ("face", "portrait", "head", "eyes", "smile"),
        "hands": ("hand", "hands", "finger", "fingers", "holding"),
        "clothing": ("dress", "shirt", "jacket", "coat", "outfit", "clothing", "armor"),
        "background": ("background", "sky", "landscape", "city", "forest", "room", "street"),
    }
    for region, keys in region_hits.items():
        if any(k in tokens or k in prompt.lower() for k in keys):
            labels.append(region)
    # Always ensure at least subject + background for decomposition UX
    if "subject" not in labels:
        labels.insert(0, "subject")
    if "background" not in labels and len(labels) < max_pieces:
        labels.append("background")
    return labels[:max_pieces]


def missing_tokens_heuristic(prompt: str, caption: str) -> list[str]:
    """Tokens in the prompt that do not appear in an image caption / description."""
    want = extract_prompt_tokens(prompt)
    cap = (caption or "").lower()
    return [t for t in want if t not in cap and not any(t in w for w in cap.split())][:12]


# ---------------------------------------------------------------------------
# Diagnose
# ---------------------------------------------------------------------------


def diagnose_image(
    image_rgb: np.ndarray,
    prompt: str,
    *,
    cfg: EditingPhaseConfig,
    caption: str = "",
) -> Diagnosis:
    """Verifier pass: gates + prompt gaps + text/anatomy cues."""
    from utils.agentic.quality_loop import GateThresholds, QualityGateRunner

    d = Diagnosis()
    d.piece_labels = infer_piece_labels(prompt) if cfg.enable_pieces else []
    d.expected_text = expected_render_text(prompt)
    d.needs_ocr = bool(cfg.enable_ocr and d.expected_text)
    pl = prompt.lower()
    d.needs_anatomy = bool(cfg.enable_anatomy and any(c in pl for c in _ANATOMY_CUES))

    gates = QualityGateRunner(
        GateThresholds(
            min_sharpness=cfg.min_sharpness,
            min_exposure=cfg.min_exposure,
            min_clip=cfg.min_clip,
            min_aesthetic=cfg.min_aesthetic,
        )
    )
    try:
        gr = gates.evaluate(image_rgb, prompt=prompt, device=cfg.device if cfg.device != "cuda" else "cpu")
        d.gate_passed = bool(gr.passed)
        d.gate_failures = list(gr.failures)
        d.clip_score = float(gr.scores.get("clip", 0.0) or 0.0)
        d.sharpness = float(gr.scores.get("sharpness", 0.0) or 0.0)
        d.exposure = float(gr.scores.get("exposure", 0.0) or 0.0)
        d.aesthetic = float(gr.scores.get("aesthetic", 0.0) or 0.0)
    except Exception as e:
        d.notes.append(f"gate_error:{e}")
        d.gate_passed = False

    if caption.strip():
        d.missing_tokens = missing_tokens_heuristic(prompt, caption)
    elif not d.gate_passed and "clip" in d.gate_failures:
        # Without a caption, treat low CLIP as “global missing adherence”
        d.missing_tokens = extract_prompt_tokens(prompt, max_tokens=8)
        d.notes.append("low_clip_without_caption")

    return d


def maybe_caption_image(image_rgb: np.ndarray, *, device: str = "cpu") -> str:
    """Optional light VLM caption (moondream) for prompt-gap detection."""
    try:
        from pathlib import Path

        from PIL import Image

        from utils.modeling.model_paths import default_moondream_path

        path = default_moondream_path()
        local = Path(path)
        if not local.is_dir():
            return ""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(local), trust_remote_code=True)
        _ = tok  # tokenizer required for some moondream builds
        model = AutoModelForCausalLM.from_pretrained(str(local), trust_remote_code=True, low_cpu_mem_usage=True)
        import torch

        dev = device if device != "cuda" else ("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(dev)
        model.eval()
        pil = Image.fromarray(image_rgb.astype(np.uint8))
        if hasattr(model, "encode_image") and hasattr(model, "answer_question"):
            enc = model.encode_image(pil)
            return str(model.answer_question(enc, "Describe the image in detail.") or "")
        return ""
    except Exception as e:
        _log.debug("caption skipped: %s", e)
        return ""


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def plan_edits(diagnosis: Diagnosis, prompt: str, *, cfg: EditingPhaseConfig) -> list[EditAction]:
    """Turn diagnosis into a prioritized action list."""
    actions: list[EditAction] = []

    if diagnosis.needs_ocr and diagnosis.expected_text:
        actions.append(
            EditAction(
                kind="ocr_fix",
                reason=f"expected readable text: {diagnosis.expected_text[:3]}",
                region="subject",
                prompt_delta=", crisp readable typography, legible letters, correct spelling",
                negative_delta="gibberish text, misspelled words, warped letters",
                strength=cfg.inpaint_strength,
                priority=10,
            )
        )

    if diagnosis.needs_anatomy:
        region = "hands" if any(t in prompt.lower() for t in ("hand", "hands", "finger")) else "face"
        actions.append(
            EditAction(
                kind="inpaint_region",
                reason="anatomy / proportion risk from prompt cues",
                region=region,
                prompt_delta=", correct anatomy, natural proportions, fixed perspective",
                negative_delta="deformed hands, extra fingers, bad anatomy, warped face",
                strength=cfg.inpaint_strength,
                priority=20,
            )
        )

    for tok in diagnosis.missing_tokens[:4]:
        region = "background" if tok in ("sky", "forest", "city", "room", "street", "background") else "subject"
        actions.append(
            EditAction(
                kind="inpaint_region",
                reason=f"missing prompt token: {tok}",
                region=region,
                prompt_delta=f", clearly visible {tok}",
                strength=cfg.inpaint_strength,
                priority=30,
            )
        )

    if diagnosis.missing_tokens and cfg.enable_rag:
        join = ", ".join(diagnosis.missing_tokens[:6])
        actions.append(
            EditAction(
                kind="rag_prompt_delta",
                reason="retrieve / invent detail for missing entities",
                prompt_delta=f", rich authentic detail for: {join}",
                priority=35,
            )
        )

    if (not diagnosis.gate_passed) and ("clip" in diagnosis.gate_failures or diagnosis.clip_score < cfg.min_clip):
        actions.append(
            EditAction(
                kind="prompt_realign",
                reason="low prompt alignment",
                region="full",
                prompt_delta=", matches the prompt faithfully, coherent scene",
                strength=cfg.img2img_strength,
                priority=40,
            )
        )

    if not diagnosis.gate_passed and ("sharpness" in diagnosis.gate_failures or "exposure" in diagnosis.gate_failures):
        actions.append(
            EditAction(
                kind="img2img",
                reason="global quality (sharpness/exposure)",
                strength=min(0.35, cfg.img2img_strength),
                prompt_delta=", clean natural photo, cohesive lighting",
                negative_delta="blurry, overexposed, underexposed, artificial, plastic skin",
                priority=45,
            )
        )

    if cfg.enable_art_post and (
        not diagnosis.gate_passed or "aesthetic" in diagnosis.gate_failures or diagnosis.needs_anatomy
    ):
        actions.append(
            EditAction(
                kind="art_post",
                reason="composition / value / naturalize pass",
                priority=90,
            )
        )

    # Deduplicate by (kind, region, prompt_delta)
    seen: set[tuple] = set()
    uniq: list[EditAction] = []
    for a in sorted(actions, key=lambda x: x.priority):
        key = (a.kind, a.region, a.prompt_delta)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(a)
    return uniq


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def _merge_prompt(base: str, delta: str) -> str:
    d = (delta or "").strip().strip(",")
    if not d:
        return base
    if d.lower() in base.lower():
        return base
    return f"{base.rstrip().rstrip(',')}, {d}"


def break_into_piece_masks(
    width: int,
    height: int,
    labels: list[str],
    out_dir: Path,
) -> dict[str, Path]:
    """Write heuristic piece masks (REVE-style separation scaffolding)."""

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for lab in labels:
        p = out_dir / f"piece_{lab}.png"
        save_heuristic_mask(p, width, height, lab)
        paths[lab] = p
    return paths


def apply_art_post(image_rgb: np.ndarray) -> np.ndarray:
    """Lightweight artistic naturalize pass."""
    try:
        from utils.quality.finishing import ArtisticPostConfig, apply_artistic_pipeline

        cfg = ArtisticPostConfig(
            composition_mode="rule_of_thirds",
            composition_strength=0.15,
            value_structure=True,
            value_midtone_contrast=0.08,
            lost_found_strength=0.1,
        )
        return apply_artistic_pipeline(image_rgb, cfg)
    except Exception as e:
        _log.warning("art_post skipped: %s", e)
        return image_rgb


def apply_action(
    action: EditAction,
    *,
    image_rgb: np.ndarray,
    prompt: str,
    negative_prompt: str,
    ckpt: str | None,
    cfg: EditingPhaseConfig,
    work_dir: Path,
    piece_masks: dict[str, Path],
    seed: int | None = None,
) -> tuple[np.ndarray, str, str]:
    """
    Apply one action. Returns ``(image, prompt, negative)``.

    When ``cfg.dry_run`` or ``ckpt`` is missing, returns image unchanged (plan-only).
    """
    prompt2 = _merge_prompt(prompt, action.prompt_delta)
    neg2 = _merge_prompt(negative_prompt, action.negative_delta)

    if action.kind == "rag_prompt_delta":
        # Prompt-only enrichment; retrieval corpora are optional.
        return image_rgb, prompt2, neg2

    if action.kind == "art_post":
        if cfg.dry_run:
            return image_rgb, prompt2, neg2
        return apply_art_post(image_rgb), prompt2, neg2

    if cfg.dry_run or not ckpt:
        return image_rgb, prompt2, neg2

    from PIL import Image

    from utils.generation.sample_argv import run_edit_with_pillow

    h, w = int(image_rgb.shape[0]), int(image_rgb.shape[1])
    init = Image.fromarray(image_rgb.astype(np.uint8))
    mask = None
    strength = float(action.strength)
    if action.kind in ("ocr_fix", "inpaint_region") and action.region:
        mp = piece_masks.get(action.region) or piece_masks.get("subject")
        if mp and mp.is_file():
            mask = Image.open(mp).convert("L")
        else:
            mask = heuristic_inpaint_mask(w, h, action.region or "subject")
    elif action.kind in ("img2img", "prompt_realign"):
        mask = None
        strength = float(action.strength)

    out = run_edit_with_pillow(
        ckpt=ckpt,
        prompt=prompt2,
        negative_prompt=neg2,
        base_image=init,
        mask_image=mask,
        width=w,
        height=h,
        steps=int(cfg.refine_steps),
        cfg_scale=7.0,
        seed=seed,
        img2img_strength=strength,
        device=cfg.device,
        scheduler=cfg.scheduler,
        solver=cfg.solver,
    )
    arr = np.asarray(out.convert("RGB"), dtype=np.uint8)
    return arr, prompt2, neg2


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_editing_phase(
    image: np.ndarray | str | Path,
    prompt: str,
    *,
    ckpt: str | None = None,
    negative_prompt: str = "",
    config: EditingPhaseConfig | None = None,
    work_dir: str | Path | None = None,
    seed: int | None = None,
    caption: str | None = None,
) -> EditingPhaseResult:
    """
    Run the post-gen editing loop until gates pass or ``max_iters``.

    Parameters
    ----------
    image:
        RGB uint8 array or path to an image.
    prompt:
        Positive prompt used for generation / realignment.
    ckpt:
        Checkpoint for img2img/inpaint via ``sample.py``. Optional in dry_run.
    """
    cfg = config or EditingPhaseConfig()
    if isinstance(image, (str, Path)):
        from PIL import Image

        image_rgb = np.asarray(Image.open(image).convert("RGB"), dtype=np.uint8)
        src_path = str(image)
    else:
        image_rgb = np.asarray(image, dtype=np.uint8)
        src_path = None

    wd = Path(work_dir) if work_dir else Path("outputs") / "editing_phase"
    wd.mkdir(parents=True, exist_ok=True)
    pieces_dir = wd / "pieces"

    prompt_cur = prompt
    neg_cur = negative_prompt
    hist: list[Diagnosis] = []
    applied: list[EditAction] = []
    cap = caption if caption is not None else ""

    piece_masks = break_into_piece_masks(
        int(image_rgb.shape[1]),
        int(image_rgb.shape[0]),
        infer_piece_labels(prompt_cur),
        pieces_dir,
    )

    stopped = "max_iters"
    for it in range(max(1, int(cfg.max_iters))):
        if not cap and it == 0 and not cfg.dry_run:
            cap = maybe_caption_image(image_rgb, device=cfg.device)

        diag = diagnose_image(image_rgb, prompt_cur, cfg=cfg, caption=cap)
        hist.append(diag)

        if diag.gate_passed and not diag.needs_ocr and not diag.missing_tokens:
            stopped = "gates_passed"
            break

        actions = plan_edits(diag, prompt_cur, cfg=cfg)
        if not actions:
            stopped = "no_actions"
            break

        # One primary action per iteration (plus optional art_post at end of plan)
        primary = [a for a in actions if a.kind != "art_post"]
        trailer = [a for a in actions if a.kind == "art_post"]
        step_actions = (primary[:1] + trailer[:1]) if primary else trailer[:1]

        for act in step_actions:
            image_rgb, prompt_cur, neg_cur = apply_action(
                act,
                image_rgb=image_rgb,
                prompt=prompt_cur,
                negative_prompt=neg_cur,
                ckpt=ckpt,
                cfg=cfg,
                work_dir=wd,
                piece_masks=piece_masks,
                seed=None if seed is None else int(seed) + it,
            )
            applied.append(act)

        # Refresh piece masks if canvas size unchanged (always true here)
        out_iter = wd / f"iter_{it:02d}.png"
        try:
            from PIL import Image

            Image.fromarray(image_rgb).save(out_iter)
        except Exception:
            pass
    else:
        stopped = "max_iters"

    # Final gate note
    if hist and hist[-1].gate_passed and stopped == "max_iters":
        stopped = "gates_passed"

    final_path = wd / "final.png"
    try:
        from PIL import Image

        Image.fromarray(image_rgb).save(final_path)
        out_p = str(final_path)
    except Exception:
        out_p = src_path

    return EditingPhaseResult(
        image_rgb=image_rgb,
        prompt=prompt_cur,
        negative_prompt=neg_cur,
        diagnosis_history=hist,
        actions_applied=applied,
        iterations=len(hist),
        stopped_reason=stopped,
        output_path=out_p,
        piece_dir=str(pieces_dir),
    )

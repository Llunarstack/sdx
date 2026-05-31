"""
**Understand** reference images: OCR, VLM caption, control maps, tags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageFilter


@dataclass(slots=True)
class ImageUnderstanding:
    path: str
    caption: str = ""
    ocr_text: str = ""
    tags: List[str] = field(default_factory=list)
    control_maps: Dict[str, str] = field(default_factory=dict)
    width: int = 0
    height: int = 0
    source: str = "local"
    metadata: Dict[str, Any] = field(default_factory=dict)


def ocr_image(path: str, *, device: str = "cuda") -> str:
    """OCR via TrOCR / GOT-OCR when weights exist, else pytesseract."""
    try:
        from utils.modeling.hf_loaders import ocr_got, ocr_trocr, ocr_trocr_handwritten

        for fn in (ocr_trocr, ocr_trocr_handwritten, ocr_got):
            text = fn(path, device=device)
            if text:
                return text
    except Exception:
        pass
    try:
        import pytesseract

        img = Image.open(path).convert("RGB")
        return (pytesseract.image_to_string(img) or "").strip()
    except Exception:
        return ""


def caption_image_vlm(path: str, *, user_prompt: str = "", device: str = "cuda") -> str:
    """Caption via HF chain (moondream → BLIP-2 → Florence-2 → …) with heuristic fallback."""
    p = Path(path)
    if not p.is_file():
        return ""
    try:
        from utils.modeling.hf_loaders import caption_image_chain

        cap, _backend = caption_image_chain(str(p), user_prompt=user_prompt, device=device)
        if cap:
            return cap
    except Exception:
        pass
    try:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        arr = np.array(img.resize((64, 64)))
        mean = arr.mean(axis=(0, 1))
        return f"reference image {w}x{h}, average color RGB {mean.astype(int).tolist()}"
    except Exception:
        return f"reference image at {p.name}"


def extract_control_maps(
    path: str,
    output_dir: Path,
    *,
    types: Sequence[str] = ("canny", "softedge", "hed", "depth", "normals"),
    device: str = "cuda",
) -> Dict[str, str]:
    """Build control maps via ``hf_control`` (PIL proxies + HF depth/normals)."""
    try:
        from utils.modeling.hf_control import extract_control_maps_batch

        return extract_control_maps_batch(path, output_dir, types=types, device=device)
    except Exception:
        pass
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem
    maps: Dict[str, str] = {}
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return maps
    if "canny" in types:
        gray = img.convert("L").filter(ImageFilter.FIND_EDGES)
        dest = out_dir / f"{stem}_canny.png"
        gray.save(dest)
        maps["canny"] = str(dest)
    return maps


def understand_image(
    path: str,
    *,
    user_prompt: str = "",
    work_dir: Path,
    source: str = "local",
    run_ocr: bool = True,
    run_vlm: bool = True,
    run_control: bool = True,
    device: str = "cuda",
) -> ImageUnderstanding:
    """Full understanding pass for one reference image."""
    p = Path(path)
    u = ImageUnderstanding(path=str(p.resolve()), source=source)
    if not p.is_file():
        return u
    try:
        img = Image.open(p)
        u.width, u.height = img.size
    except Exception:
        pass
    ctrl_dir = Path(work_dir) / "control_maps"
    if run_ocr:
        u.ocr_text = ocr_image(str(p), device=device)
    if run_vlm:
        u.caption = caption_image_vlm(str(p), user_prompt=user_prompt, device=device)
    if run_control:
        u.control_maps = extract_control_maps(str(p), ctrl_dir, device=device)
    if user_prompt.strip():
        try:
            from utils.modeling.hf_loaders import detect_objects

            tokens = [t.strip() for t in user_prompt.replace(",", " ").split() if len(t.strip()) > 3][:8]
            if tokens:
                hits = detect_objects(str(p), tokens, device=device)
                if hits:
                    u.metadata["detected_objects"] = hits
                    u.tags.append("object_detect")
        except Exception:
            pass
    tags: List[str] = []
    if u.ocr_text:
        tags.append("has_text")
    if u.caption:
        tags.append("vlm_caption")
    if u.control_maps:
        tags.extend(f"control_{k}" for k in u.control_maps)
    u.tags = tags
    return u


def understand_images(
    paths: Sequence[str],
    *,
    user_prompt: str = "",
    work_dir: Path,
    sources: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> List[ImageUnderstanding]:
    """Understand many references in order."""
    srcs = list(sources or ["local"] * len(paths))
    out: List[ImageUnderstanding] = []
    for i, path in enumerate(paths):
        src = srcs[i] if i < len(srcs) else "local"
        out.append(
            understand_image(
                path,
                user_prompt=user_prompt,
                work_dir=work_dir,
                source=src,
                **kwargs,
            )
        )
    return out


__all__ = [
    "ImageUnderstanding",
    "caption_image_vlm",
    "extract_control_maps",
    "ocr_image",
    "understand_image",
    "understand_images",
]

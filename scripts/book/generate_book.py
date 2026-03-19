#!/usr/bin/env python3
"""
Book / comic / manga workflow generator.

This is intentionally a *workflow* (diffusion as sequential planning):
- Page 0: normal text-to-image generation.
- Page i>0 (optional): inpaint from previous page while freezing face region(s) using MDM-style freezing.
- Optional: OCR-check for spelling/legibility and re-inpaint detected text regions until it passes a threshold.

Why workflow-first:
- Adding true multi-page "reasoning" into the core model is architecture/training work.
- You already have the key primitives (inpainting + MDM freeze + OCR text masks), so we can ship a working generator now.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image


def _repo_root() -> Path:
    # scripts/book/generate_book.py -> scripts/book -> scripts -> repo root
    return Path(__file__).resolve().parents[2]


def _sample_py() -> Path:
    return _repo_root() / "sample.py"


def _safe_run(cmd: Sequence[str]) -> None:
    subprocess.run(list(cmd), check=True)


def _apply_book_style(prefix: str, prompt: str) -> str:
    prompt = (prompt or "").strip()
    if not prompt:
        return prefix.strip()
    return f"{prefix.strip()}, {prompt}"


def _maybe_append_text_says(prompt: str, expected_texts: List[str]) -> str:
    if not expected_texts:
        return prompt
    # sample.py has prompt-based detection for "text that says" / "says \""
    extra_parts = [f'text that says "{t}"' for t in expected_texts if t and t.strip()]
    if not extra_parts:
        return prompt
    # If user already wrote quoted text, don't duplicate too aggressively.
    # Still append if none of the exact quoted strings appear.
    lower = prompt.lower()
    for t in expected_texts:
        if t and f'"{t}"'.lower() in lower:
            return prompt
    return f"{prompt.strip()}, {', '.join(extra_parts)}"


def _parse_expected_texts(raw: str) -> List[str]:
    # Accept: "OPEN" or "OPEN,WORLD" or JSON list.
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        if raw.startswith("["):
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


def _load_prompts_from_file(path: Path) -> List[str]:
    """
    Backward-compatible: returns only prompts.
    For per-page expected text overrides, see _load_prompts_with_expected_from_file().
    """
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    prompts: List[str] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        prompts.append(ln.split("|||", 1)[0].strip())
    return prompts


def _load_prompts_with_expected_from_file(path: Path) -> List[Tuple[str, str]]:
    """
    Supports lines of the form:
      prompt...
      or
      prompt...|||expected text (comma-separated or JSON list)
    """
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    specs: List[Tuple[str, str]] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if "|||" in ln:
            left, right = ln.split("|||", 1)
            specs.append((left.strip(), right.strip()))
        else:
            specs.append((ln, ""))
    return specs


def _build_face_keep_mask(init_image: Image.Image, out_path: Path, face_padding: float = 0.2) -> None:
    """
    Create mask where:
    - black (0): face region to KEEP (not inpainted)
    - white (255): everything else to INPAINT
    """
    import cv2
    import numpy as np

    img = init_image.convert("RGB")
    w, h = img.size
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Default: no face detected => inpaint everything (all white)
    mask = np.full((h, w), 255, dtype=np.uint8)

    if len(faces) > 0:
        # Freeze the largest face (most likely the main character).
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        pad_w = int(fw * face_padding)
        pad_h = int(fh * face_padding)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w, x + fw + pad_w)
        y2 = min(h, y + fh + pad_h)
        mask[y1:y2, x1:x2] = 0

    Image.fromarray(mask, mode="L").save(out_path)


def _build_edge_keep_mask(
    init_image: Image.Image,
    out_path: Path,
    canny_thresh1: int = 50,
    canny_thresh2: int = 150,
    dilation_px: int = 3,
) -> None:
    """
    Create keep/inpaint mask by freezing strong edges from the previous page.
    - black (0): keep edges
    - white (255): inpaint elsewhere
    """
    import cv2
    import numpy as np

    img = init_image.convert("RGB")
    w, h = img.size
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, canny_thresh1, canny_thresh2)
    if dilation_px > 0:
        kernel = np.ones((dilation_px * 2 + 1, dilation_px * 2 + 1), dtype=np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

    # edges==255 => keep (black in our mask)
    mask = np.full((h, w), 255, dtype=np.uint8)
    mask[edges > 0] = 0
    Image.fromarray(mask, mode="L").save(out_path)


def _build_speech_bubble_outline_keep_mask_pil(
    init_image: Image.Image,
    ocr_engine: Any,
    *,
    inner_dilate_px: int = 2,
    outer_dilate_px: int = 18,
) -> Optional[Image.Image]:
    """
    Build a KEEP mask (black=keep, white=inpaint) for speech-bubble outlines.

    Strategy:
    - OCR text boxes approximate the *interior*.
    - Dilate interior by an outer radius to approximate the bubble outline/tail.
    - Freeze only the "ring": outer_dilated - inner_dilated.
    - This allows re-rendering text inside the bubble while keeping the bubble shape stable.
    """
    if ocr_engine is None:
        return None

    mask_words_pil = ocr_engine.create_text_edit_mask(init_image, target_text=None)
    if mask_words_pil is None:
        return None

    import cv2
    import numpy as np

    w, h = init_image.size
    inpaint_arr = np.array(mask_words_pil.convert("L"))  # 0 background, 255 text interior (white=inpaint)
    interior = (inpaint_arr > 0).astype(np.uint8) * 255

    inner_dilate_px = max(0, int(inner_dilate_px))
    outer_dilate_px = max(0, int(outer_dilate_px))
    if outer_dilate_px <= inner_dilate_px:
        outer_dilate_px = inner_dilate_px + 1

    def _dilate(arr, radius: int):
        if radius <= 0:
            return arr
        kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
        return cv2.dilate(arr, kernel, iterations=1)

    inner = _dilate(interior, inner_dilate_px)
    outer = _dilate(interior, outer_dilate_px)

    # Keep ring where outer exists but inner does not.
    ring_keep = ((outer > 0) & (inner == 0)).astype(np.uint8) * 255

    keep_mask = np.full((h, w), 255, dtype=np.uint8)  # default: inpaint everywhere else
    keep_mask[ring_keep > 0] = 0  # black=keep bubble outline
    return Image.fromarray(keep_mask, mode="L")


def _build_speech_bubble_outline_keep_mask(
    init_image: Image.Image,
    out_path: Path,
    ocr_engine: Any,
    *,
    inner_dilate_px: int = 2,
    outer_dilate_px: int = 18,
) -> bool:
    m = _build_speech_bubble_outline_keep_mask_pil(
        init_image,
        ocr_engine,
        inner_dilate_px=inner_dilate_px,
        outer_dilate_px=outer_dilate_px,
    )
    if m is None:
        return False
    m.save(out_path)
    return True


def _combine_keep_masks(mask_paths: List[Path], out_path: Path) -> None:
    """
    Combine multiple masks where black pixels represent KEEP.
    If ANY mask says KEEP (black), final mask will be KEEP (black).
    """
    import numpy as np

    if not mask_paths:
        raise ValueError("No masks to combine")

    masks = [Image.open(p).convert("L") for p in mask_paths]
    w, h = masks[0].size

    combined = np.full((h, w), 255, dtype=np.uint8)
    for m in masks:
        if m.size != (w, h):
            m = m.resize((w, h), Image.Resampling.LANCZOS)
        arr = np.array(m)
        keep = arr < 128  # black-ish
        combined[keep] = 0

    Image.fromarray(combined, mode="L").save(out_path)


def _load_user_anchor_mask(
    mask_path: Path,
    out_path: Path,
    mask_type: str,
) -> None:
    """
    mask_type:
    - 'inpaint': white=inpaint, black=keep (same as sample.py)
    - 'keep': white=keep, black=inpaint (will be inverted)
    """
    if mask_type not in {"inpaint", "keep"}:
        raise ValueError("mask_type must be inpaint|keep")

    m = Image.open(mask_path).convert("L")
    if mask_type == "keep":
        # invert so black means keep
        import numpy as np

        arr = np.array(m)
        arr = 255 - arr
        m = Image.fromarray(arr, mode="L")
    m.save(out_path)


def _ocr_text_accuracy(text_engine: Any, image: Image.Image, expected_texts: List[str]) -> float:
    try:
        res = text_engine.validate_text_rendering(image, expected_texts)
        return float(res.get("accuracy_score", 0.0))
    except Exception:
        return 0.0


def _try_ocr_fix(
    *,
    image_path: Path,
    expected_texts: List[str],
    prompt: str,
    ckpt: str,
    out_path: Path,
    sample_steps: int,
    strength: float,
    inpaint_strength: float,
    sample_width: int,
    sample_height: int,
    device: str,
    negative_prompt: str,
    no_neg_filter: bool,
    text_engine: Any,
    ocr_engine: Any,
    max_iters: int,
    threshold: float,
    inpaint_mode: str,
    seed: Optional[int],
    sampler: str,
    text_in_image_flag: bool,
    ocr_mask_dilate: int,
) -> None:
    """
    Repeatedly:
    - OCR validate on current image
    - If accuracy is low: create an OCR mask for detected text regions
      and inpaint only those regions using mdm freezing.
    """
    if not expected_texts:
        # Nothing to fix.
        if image_path != out_path:
            out_path.write_bytes(image_path.read_bytes())
        return

    cur_path = image_path
    cur_img = Image.open(cur_path).convert("RGB")

    for it in range(max_iters + 1):
        acc = _ocr_text_accuracy(text_engine, cur_img, expected_texts)
        if acc >= threshold:
            if cur_path != out_path:
                out_path.write_bytes(cur_path.read_bytes())
            return

        # Create a mask targeting the current text regions.
        # TextAwareInpainting.create_text_edit_mask returns a PIL "L" mask with white rectangles.
        try:
            # If multiple strings are expected, mask all detected text regions.
            target = expected_texts[0] if len(expected_texts) == 1 else None
            mask_pil = ocr_engine.create_text_edit_mask(cur_img, target_text=target)
        except Exception:
            mask_pil = None

        if mask_pil is None:
            # Can't build mask -> stop.
            if cur_path != out_path:
                out_path.write_bytes(cur_path.read_bytes())
            return

        tmp_dir = out_path.parent
        mask_path = tmp_dir / f"{out_path.stem}_ocrmask_{it}.png"

        if mask_pil is not None and ocr_mask_dilate > 0:
            try:
                import cv2
                import numpy as np

                arr = np.array(mask_pil.convert("L"))
                # White regions are inpaint; dilate them slightly for more context.
                kernel = np.ones((ocr_mask_dilate * 2 + 1, ocr_mask_dilate * 2 + 1), dtype=np.uint8)
                dil = cv2.dilate(arr, kernel, iterations=1)
                mask_pil = Image.fromarray(dil, mode="L")
            except Exception:
                # If dilation fails, keep original mask.
                pass

        mask_pil.save(mask_path)

        patched_prompt = _maybe_append_text_says(prompt, expected_texts)

        cmd = [
            sys.executable,
            str(_sample_py()),
            "--ckpt",
            ckpt,
            "--prompt",
            patched_prompt,
            "--out",
            str(out_path),
            "--num",
            "1",
            "--steps",
            str(sample_steps),
            "--width",
            str(sample_width),
            "--height",
            str(sample_height),
            "--seed",
            str(seed if seed is not None else 42),
            "--device",
            device,
            "--strength",
            str(inpaint_strength),
            "--init-image",
            str(cur_path),
            "--mask",
            str(mask_path),
            "--inpaint-mode",
            inpaint_mode,
            "--scheduler",
            sampler,
        ]
        if negative_prompt:
            cmd += ["--negative-prompt", negative_prompt]
        if no_neg_filter:
            cmd += ["--no-neg-filter"]
        if text_in_image_flag:
            cmd += ["--text-in-image"]

        _safe_run(cmd)

        # Loop with new image
        cur_path = out_path
        cur_img = Image.open(cur_path).convert("RGB")

    # If we never reached threshold, still save last output.
    if cur_path != out_path:
        out_path.write_bytes(cur_path.read_bytes())


def main() -> None:
    repo_root = _repo_root()
    sample_py_path = _sample_py()
    if not sample_py_path.exists():
        raise SystemExit(f"sample.py not found at {sample_py_path}")

    parser = argparse.ArgumentParser(description="Generate a multi-page book/comic/manga with OCR+MDM text fixes.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path passed to sample.py")
    parser.add_argument("--output-dir", required=True, help="Directory to write cover/pages into")

    parser.add_argument("--book-type", default="manga", choices=["manga", "comic", "novel_cover", "storyboard"], help="Prompt style preset")
    parser.add_argument("--model-preset", default="anime", choices=["sdxl", "flux", "anime", "zit"], help="sample.py preset flag")

    parser.add_argument(
        "--prompts-file",
        default="",
        help="Text file: one page prompt per line. Optional per-page expected text: use `prompt|||expected_text`.",
    )
    parser.add_argument("--pages", type=int, default=0, help="Number of pages to generate using --page-prompt-template")
    parser.add_argument("--page-prompt-template", default="", help="Template for each page prompt; supports {page} placeholder.")

    parser.add_argument("--cover-prompt", default="", help="Optional cover prompt")
    parser.add_argument("--expected-text", default="", help="Expected text (comma-separated or JSON list). Used for OCR validation + fixes.")
    parser.add_argument("--cover-expected-text", default="", help="Expected cover text (defaults to --expected-text).")
    parser.add_argument("--pages-expected-text", default="", help="Expected page text (defaults to --expected-text).")
    parser.add_argument("--ocr-fix", action="store_true", help="Enable OCR validation + iterative inpainting of text regions.")
    parser.add_argument("--ocr-threshold", type=float, default=0.65, help="Stop when OCR accuracy_score >= this threshold.")
    parser.add_argument("--ocr-iters", type=int, default=2, help="Max OCR repair iterations per page.")
    parser.add_argument("--ocr-mask-dilate", type=int, default=0, help="Dilate OCR mask before inpainting (pixels).")

    parser.add_argument("--anchor-face", action="store_true", help="Freeze detected face region across pages (character consistency).")
    parser.add_argument("--face-anchor-padding", type=float, default=0.25, help="Padding fraction around detected face bbox.")
    parser.add_argument(
        "--anchor-mask",
        type=str,
        default="",
        help="Optional user mask to anchor across pages (mask: white=inpaint, black=keep by default).",
    )
    parser.add_argument(
        "--anchor-mask-type",
        type=str,
        default="inpaint",
        choices=["inpaint", "keep"],
        help="Interpret --anchor-mask values as: inpaint=white=inpaint/black=keep, keep=white=keep/black=inpaint.",
    )
    parser.add_argument("--edge-anchor", action="store_true", help="Additionally freeze strong edges from previous page (reduces distortion).")
    parser.add_argument("--edge-anchor-dilate", type=int, default=3, help="Edge mask dilation radius (pixels).")
    parser.add_argument("--edge-anchor-canny-1", type=int, default=50, help="Canny threshold 1 for edges.")
    parser.add_argument("--edge-anchor-canny-2", type=int, default=150, help="Canny threshold 2 for edges.")
    parser.add_argument("--anchor-speech-bubbles", action="store_true", help="Freeze approximate speech-bubble outlines across pages using OCR text region anchors.")
    parser.add_argument("--speech-bubble-anchor-inner-dilate", type=int, default=2, help="Inner dilation around OCR text (defines bubble interior excluded from keep).")
    parser.add_argument("--speech-bubble-anchor-outer-dilate", type=int, default=18, help="Outer dilation around OCR text (defines bubble outline keep ring).")
    parser.add_argument("--page-inpaint-strength", type=float, default=0.78, help="MDM inpaint strength when generating subsequent pages.")
    parser.add_argument("--text-inpaint-strength", type=float, default=0.55, help="MDM inpaint strength when fixing text.")

    parser.add_argument("--steps", type=int, default=30, help="Inference steps passed to sample.py")
    parser.add_argument("--width", type=int, default=0, help="Output width (0 => model native image_size)")
    parser.add_argument("--height", type=int, default=0, help="Output height (0 => model native image_size)")
    parser.add_argument("--seed", type=int, default=42, help="Base seed (inpainting keeps it deterministic-ish)")
    parser.add_argument("--device", type=str, default="cuda", help="Device passed to sample.py")
    parser.add_argument("--scheduler", type=str, default="ddim", choices=["ddim", "euler"], help="Sampler scheduler.")

    parser.add_argument("--negative-prompt", default="", help="Additional negative prompt passed to sample.py")
    parser.add_argument("--no-neg-filter", action="store_true", help="Disable positive/negative token conflict filtering.")

    parser.add_argument("--character-sheet", type=str, default="", help="Path to character sheet JSON forwarded to sample.py")
    parser.add_argument("--character-prompt-extra", type=str, default="", help="Extra character tokens forwarded to sample.py")
    parser.add_argument("--character-negative-extra", type=str, default="", help="Extra character negative tokens forwarded to sample.py")

    # Optional: keep prompt stable while injecting expected text.
    parser.add_argument("--force-text-quote", action="store_true", help="When OCR fixing, ensure prompt contains text that says \"...\".")
    parser.add_argument("--text-in-image", action="store_true", help="Set sample.py --text-in-image (also helps negatives).")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(exist_ok=True)
    cover_dir = out_dir / "cover"
    cover_dir.mkdir(exist_ok=True)

    expected_texts = _parse_expected_texts(args.expected_text)
    cover_expected_texts = _parse_expected_texts(args.cover_expected_text) if args.cover_expected_text.strip() else expected_texts
    pages_expected_texts = _parse_expected_texts(args.pages_expected_text) if args.pages_expected_text.strip() else expected_texts

    book_prefix_map: Dict[str, str] = {
        "manga": "manga panel page, black and white ink, screentones, clean lineart, high contrast, dynamic composition",
        "comic": "comic book page, inked lineart, screentone shading, crisp outlines, strong silhouettes, panel layout",
        "novel_cover": "book cover design, title typography, author name, professional layout, readable lettering",
        "storyboard": "storyboard frame, manga anime layout, clean thumbnails, camera framing, clear panel borders",
    }
    prompt_prefix = book_prefix_map.get(args.book_type, "")

    # Load OCR engine lazily (so script can run without tesseract).
    text_engine = None
    ocr_engine = None

    # Build page prompts
    prompts: List[str] = []
    page_expected_overrides: List[List[str]] = []
    if args.prompts_file:
        page_specs = _load_prompts_with_expected_from_file(Path(args.prompts_file))
        prompts = [p for (p, _) in page_specs]
        page_expected_overrides = [_parse_expected_texts(exp) for (_, exp) in page_specs]
    if args.pages and args.page_prompt_template:
        for i in range(args.pages):
            prompts.append(args.page_prompt_template.format(page=i))
            page_expected_overrides.append([])
    if not prompts and args.cover_prompt:
        prompts = []
    if not prompts and not args.cover_prompt:
        raise SystemExit("Provide either --prompts-file or (--pages and --page-prompt-template), or provide --cover-prompt.")

    # Initialize OCR lazily (only when needed).
    # - If OCR-fix is requested, we only need OCR if we actually have expected text.
    # - If speech-bubble anchoring is enabled, OCR is always needed (no expected text required).
    if (args.ocr_fix or args.anchor_speech_bubbles) and text_engine is None:
        any_page_expected = any(bool(x) for x in page_expected_overrides)
        needs_expected_text_ocr = args.ocr_fix and (cover_expected_texts or pages_expected_texts or any_page_expected)
        needs_speech_bubble_ocr = args.anchor_speech_bubbles
        if needs_expected_text_ocr or needs_speech_bubble_ocr:
            try:
                from utils.text_rendering import create_text_rendering_pipeline

                pipe = create_text_rendering_pipeline()
                text_engine = pipe["engine"]
                ocr_engine = pipe["inpainting"]
            except Exception as e:
                print(f"WARNING: OCR pipeline unavailable: {e}", file=sys.stderr)
                args.ocr_fix = False
                args.anchor_speech_bubbles = False

    # Resolve size flags: sample.py uses native image_size when width/height are 0.
    width_arg = args.width
    height_arg = args.height

    def sample_generate(
        prompt: str,
        out_path: Path,
        *,
        expected_texts_for_prompt: List[str],
        init_image: Optional[Path] = None,
        mask_path: Optional[Path] = None,
        strength: Optional[float] = None,
    ) -> None:
        prompt_final = _apply_book_style(prompt_prefix, prompt)
        if args.force_text_quote and expected_texts_for_prompt:
            prompt_final = _maybe_append_text_says(prompt_final, expected_texts_for_prompt)

        cmd: List[str] = [
            sys.executable,
            str(sample_py_path),
            "--ckpt",
            args.ckpt,
            "--prompt",
            prompt_final,
            "--out",
            str(out_path),
            "--num",
            "1",
            "--steps",
            str(args.steps),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--preset",
            args.model_preset,
            "--scheduler",
            args.scheduler,
        ]
        if args.character_sheet.strip():
            cmd += ["--character-sheet", args.character_sheet]
        if args.character_prompt_extra.strip():
            cmd += ["--character-prompt-extra", args.character_prompt_extra]
        if args.character_negative_extra.strip():
            cmd += ["--character-negative-extra", args.character_negative_extra]
        if width_arg and width_arg > 0:
            cmd += ["--width", str(width_arg)]
        if height_arg and height_arg > 0:
            cmd += ["--height", str(height_arg)]
        if args.negative_prompt:
            cmd += ["--negative-prompt", args.negative_prompt]
        if args.no_neg_filter:
            cmd += ["--no-neg-filter"]
        if args.text_in_image:
            cmd += ["--text-in-image"]

        if init_image is not None:
            cmd += ["--init-image", str(init_image)]
        if strength is not None:
            cmd += ["--strength", str(strength)]
        if mask_path is not None:
            cmd += ["--mask", str(mask_path)]
            cmd += ["--inpaint-mode", "mdm"]

        _safe_run(cmd)

    # Cover generation
    if args.cover_prompt:
        cover_out = cover_dir / "cover.png"
        sample_generate(
            args.cover_prompt,
            cover_out,
            expected_texts_for_prompt=cover_expected_texts,
        )
        # OCR-fix cover too (optional)
        if args.ocr_fix and cover_expected_texts:
            ocr_out = cover_out
            _try_ocr_fix(
                image_path=cover_out,
                expected_texts=cover_expected_texts,
                prompt=args.cover_prompt,
                ckpt=args.ckpt,
                out_path=ocr_out,
                sample_steps=args.steps,
                strength=args.page_inpaint_strength,
                inpaint_strength=args.text_inpaint_strength,
                sample_width=args.width,
                sample_height=args.height,
                device=args.device,
                negative_prompt=args.negative_prompt,
                no_neg_filter=args.no_neg_filter,
                text_engine=text_engine,
                ocr_engine=ocr_engine,
                max_iters=args.ocr_iters,
                threshold=args.ocr_threshold,
                inpaint_mode="mdm",
                seed=args.seed,
                sampler=args.scheduler,
                text_in_image_flag=args.text_in_image,
                ocr_mask_dilate=args.ocr_mask_dilate,
            )

    # Pages
    prev_path: Optional[Path] = None
    user_anchor_mask_internal: Optional[Path] = None
    if args.anchor_mask.strip():
        user_anchor_mask_internal = out_dir / "user_anchor_mask.png"
        _load_user_anchor_mask(
            Path(args.anchor_mask),
            user_anchor_mask_internal,
            mask_type=args.anchor_mask_type,
        )

    for i, page_prompt in enumerate(prompts):
        page_out = pages_dir / f"page_{i:03d}.png"
        # If this page has an override expected text, use it; otherwise use the global pages_expected_texts.
        page_expected_texts = pages_expected_texts
        if i < len(page_expected_overrides) and page_expected_overrides[i]:
            page_expected_texts = page_expected_overrides[i]

        user_anchor_exists = user_anchor_mask_internal is not None and user_anchor_mask_internal.exists()
        should_anchor = (
            prev_path is not None
            and (args.anchor_face or args.edge_anchor or args.anchor_speech_bubbles or user_anchor_exists)
        )

        if i == 0 or not should_anchor:
            sample_generate(
                page_prompt,
                page_out,
                expected_texts_for_prompt=page_expected_texts,
            )
        else:
            prev_img = Image.open(prev_path).convert("RGB")
            keep_masks: List[Path] = []

            # Freeze regions from the previous page, inpaint the rest for coherence.
            if args.anchor_face:
                face_mask_path = pages_dir / f"face_keep_mask_{i:03d}.png"
                _build_face_keep_mask(prev_img, face_mask_path, face_padding=args.face_anchor_padding)
                keep_masks.append(face_mask_path)

            if args.edge_anchor:
                edge_mask_path = pages_dir / f"edge_keep_mask_{i:03d}.png"
                _build_edge_keep_mask(
                    prev_img,
                    edge_mask_path,
                    canny_thresh1=args.edge_anchor_canny_1,
                    canny_thresh2=args.edge_anchor_canny_2,
                    dilation_px=args.edge_anchor_dilate,
                )
                keep_masks.append(edge_mask_path)

            if args.anchor_speech_bubbles:
                sb_keep_path = pages_dir / f"speech_bubble_keep_mask_{i:03d}.png"
                ok = _build_speech_bubble_outline_keep_mask(
                    prev_img,
                    sb_keep_path,
                    ocr_engine,
                    inner_dilate_px=args.speech_bubble_anchor_inner_dilate,
                    outer_dilate_px=args.speech_bubble_anchor_outer_dilate,
                )
                if ok:
                    keep_masks.append(sb_keep_path)

            if user_anchor_mask_internal is not None and user_anchor_mask_internal.exists():
                keep_masks.append(user_anchor_mask_internal)

            if not keep_masks:
                # Shouldn't happen, but keep it safe: fall back to fresh generation.
                sample_generate(
                    page_prompt,
                    page_out,
                    expected_texts_for_prompt=page_expected_texts,
                )
            else:
                mask_path = pages_dir / f"anchor_keep_mask_{i:03d}.png"
                _combine_keep_masks(keep_masks, mask_path)

                sample_generate(
                    page_prompt,
                    page_out,
                    expected_texts_for_prompt=page_expected_texts,
                    init_image=prev_path,
                    mask_path=mask_path,
                    strength=args.page_inpaint_strength,
                )

        # Optional OCR repair
        if args.ocr_fix and page_expected_texts:
            _try_ocr_fix(
                image_path=page_out,
                expected_texts=page_expected_texts,
                prompt=page_prompt,
                ckpt=args.ckpt,
                out_path=page_out,
                sample_steps=args.steps,
                strength=args.page_inpaint_strength,
                inpaint_strength=args.text_inpaint_strength,
                sample_width=args.width,
                sample_height=args.height,
                device=args.device,
                negative_prompt=args.negative_prompt,
                no_neg_filter=args.no_neg_filter,
                text_engine=text_engine,
                ocr_engine=ocr_engine,
                max_iters=args.ocr_iters,
                threshold=args.ocr_threshold,
                inpaint_mode="mdm",
                seed=args.seed,
                sampler=args.scheduler,
                text_in_image_flag=args.text_in_image,
                ocr_mask_dilate=args.ocr_mask_dilate,
            )

        prev_path = page_out

    print(f"✅ Book generation finished: {out_dir}")


if __name__ == "__main__":
    main()


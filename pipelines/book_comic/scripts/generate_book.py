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
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image


# Book pipeline helpers (pick-best, CFG, post-process) — repo root on sys.path
def _repo_root() -> Path:
    # pipelines/book_comic/scripts/generate_book.py -> ... -> repo root
    return Path(__file__).resolve().parents[3]


def _ensure_repo_on_path() -> None:
    r = _repo_root()
    rs = str(r)
    if rs not in sys.path:
        sys.path.insert(0, rs)


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
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
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
    ocr_extra_flags: Optional[List[str]] = None,
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
        if ocr_extra_flags:
            cmd.extend(ocr_extra_flags)

        _safe_run(cmd)

        # Loop with new image
        cur_path = out_path
        cur_img = Image.open(cur_path).convert("RGB")

    # If we never reached threshold, still save last output.
    if cur_path != out_path:
        out_path.write_bytes(cur_path.read_bytes())


def main() -> None:
    sample_py_path = _sample_py()
    if not sample_py_path.exists():
        raise SystemExit(f"sample.py not found at {sample_py_path}")

    parser = argparse.ArgumentParser(description="Generate a multi-page book/comic/manga with OCR+MDM text fixes.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path passed to sample.py")
    parser.add_argument("--output-dir", required=True, help="Directory to write cover/pages into")

    parser.add_argument(
        "--book-type",
        default="manga",
        choices=["manga", "comic", "novel_cover", "storyboard"],
        help="Prompt style preset",
    )
    parser.add_argument(
        "--model-preset", default="anime", choices=["sdxl", "flux", "anime", "zit"], help="sample.py preset flag"
    )

    parser.add_argument(
        "--prompts-file",
        default="",
        help="Text file: one page prompt per line. Optional per-page expected text: use `prompt|||expected_text`.",
    )
    parser.add_argument("--pages", type=int, default=0, help="Number of pages to generate using --page-prompt-template")
    parser.add_argument(
        "--page-prompt-template", default="", help="Template for each page prompt; supports {page} placeholder."
    )

    parser.add_argument("--cover-prompt", default="", help="Optional cover prompt")
    parser.add_argument(
        "--expected-text",
        default="",
        help="Expected text (comma-separated or JSON list). Used for OCR validation + fixes.",
    )
    parser.add_argument("--cover-expected-text", default="", help="Expected cover text (defaults to --expected-text).")
    parser.add_argument("--pages-expected-text", default="", help="Expected page text (defaults to --expected-text).")
    parser.add_argument(
        "--ocr-fix", action="store_true", help="Enable OCR validation + iterative inpainting of text regions."
    )
    parser.add_argument(
        "--ocr-threshold", type=float, default=0.65, help="Stop when OCR accuracy_score >= this threshold."
    )
    parser.add_argument("--ocr-iters", type=int, default=2, help="Max OCR repair iterations per page.")
    parser.add_argument("--ocr-mask-dilate", type=int, default=0, help="Dilate OCR mask before inpainting (pixels).")

    parser.add_argument(
        "--anchor-face", action="store_true", help="Freeze detected face region across pages (character consistency)."
    )
    parser.add_argument(
        "--face-anchor-padding", type=float, default=0.25, help="Padding fraction around detected face bbox."
    )
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
    parser.add_argument(
        "--edge-anchor",
        action="store_true",
        help="Additionally freeze strong edges from previous page (reduces distortion).",
    )
    parser.add_argument("--edge-anchor-dilate", type=int, default=3, help="Edge mask dilation radius (pixels).")
    parser.add_argument("--edge-anchor-canny-1", type=int, default=50, help="Canny threshold 1 for edges.")
    parser.add_argument("--edge-anchor-canny-2", type=int, default=150, help="Canny threshold 2 for edges.")
    parser.add_argument(
        "--anchor-speech-bubbles",
        action="store_true",
        help="Freeze approximate speech-bubble outlines across pages using OCR text region anchors.",
    )
    parser.add_argument(
        "--speech-bubble-anchor-inner-dilate",
        type=int,
        default=2,
        help="Inner dilation around OCR text (defines bubble interior excluded from keep).",
    )
    parser.add_argument(
        "--speech-bubble-anchor-outer-dilate",
        type=int,
        default=18,
        help="Outer dilation around OCR text (defines bubble outline keep ring).",
    )
    parser.add_argument(
        "--page-inpaint-strength",
        type=float,
        default=0.78,
        help="MDM inpaint strength when generating subsequent pages.",
    )
    parser.add_argument(
        "--text-inpaint-strength", type=float, default=0.55, help="MDM inpaint strength when fixing text."
    )

    parser.add_argument("--steps", type=int, default=30, help="Inference steps passed to sample.py")
    parser.add_argument("--width", type=int, default=0, help="Output width (0 => model native image_size)")
    parser.add_argument("--height", type=int, default=0, help="Output height (0 => model native image_size)")
    parser.add_argument("--seed", type=int, default=42, help="Base seed (inpainting keeps it deterministic-ish)")
    parser.add_argument("--device", type=str, default="cuda", help="Device passed to sample.py")
    parser.add_argument("--scheduler", type=str, default="ddim", choices=["ddim", "euler"], help="Sampler scheduler.")

    parser.add_argument("--negative-prompt", default="", help="Additional negative prompt passed to sample.py")
    parser.add_argument(
        "--no-neg-filter", action="store_true", help="Disable positive/negative token conflict filtering."
    )

    parser.add_argument(
        "--character-sheet", type=str, default="", help="Path to character sheet JSON forwarded to sample.py"
    )
    parser.add_argument(
        "--character-prompt-extra", type=str, default="", help="Extra character tokens forwarded to sample.py"
    )
    parser.add_argument(
        "--character-negative-extra",
        type=str,
        default="",
        help="Extra character negative tokens forwarded to sample.py",
    )

    # Optional: keep prompt stable while injecting expected text.
    parser.add_argument(
        "--force-text-quote", action="store_true", help='When OCR fixing, ensure prompt contains text that says "...".'
    )
    parser.add_argument(
        "--text-in-image", action="store_true", help="Set sample.py --text-in-image (also helps negatives)."
    )

    # --- Accuracy / consistency (uses utils/test_time_pick, utils/quality, data/caption_utils) ---
    parser.add_argument(
        "--book-accuracy",
        default="none",
        choices=["none", "fast", "balanced", "maximum", "production"],
        help="Preset: balanced=2 candidates+combo; maximum=4; production=6+stricter lexicon negatives; none=single sample.",
    )
    parser.add_argument(
        "--sample-candidates",
        type=int,
        default=0,
        help="How many images to draw per page before pick-best (0 = use --book-accuracy preset).",
    )
    parser.add_argument(
        "--pick-best",
        default="auto",
        choices=["auto", "none", "clip", "edge", "ocr", "combo", "combo_count"],
        help="Test-time metric (auto = use preset). combo = CLIP+edge(+OCR if expected text); combo_count also adds people-count verifier.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=0,
        help="Optional explicit count target for combo_count (0 = infer from prompt).",
    )
    parser.add_argument(
        "--expected-count-target",
        type=str,
        default="auto",
        choices=["auto", "people", "objects"],
        help="Count target mode for combo_count.",
    )
    parser.add_argument(
        "--expected-count-object",
        type=str,
        default="",
        help="Optional object hint for combo_count object mode (e.g. coin, candle, window).",
    )
    parser.add_argument("--boost-quality", action="store_true", help="Prepend quality tags (overrides preset off).")
    parser.add_argument("--no-boost-quality", action="store_true", help="Disable boost even if preset would enable it.")
    parser.add_argument("--subject-first", action="store_true", help="Reorder subject tags first (sample.py).")
    parser.add_argument("--no-subject-first", action="store_true")
    parser.add_argument("--save-prompt", action="store_true", help="Write .txt sidecar next to each PNG.")
    parser.add_argument("--prepend-quality-if-short", action="store_true", help="Prepend quality if caption is short.")
    parser.add_argument("--no-prepend-quality-if-short", action="store_true")
    parser.add_argument(
        "--shortcomings-mitigation",
        type=str,
        default="",
        choices=["", "none", "auto", "all"],
        help="Override shortcomings pack sent to sample.py (default from --book-accuracy; auto=keyword match, all=full base pack).",
    )
    parser.add_argument(
        "--shortcomings-2d",
        action="store_true",
        help="Enable 2D-specific shortcomings packs (anime/manga/cartoon) for page generation and OCR repair.",
    )
    parser.add_argument(
        "--no-shortcomings-2d",
        action="store_true",
        help="Disable 2D-specific shortcomings packs even if --book-accuracy preset enables them.",
    )
    parser.add_argument(
        "--art-guidance-mode",
        type=str,
        default="",
        choices=["", "none", "auto", "all"],
        help="Override artist-first medium packs sent to sample.py (default from --book-accuracy).",
    )
    parser.set_defaults(art_guidance_photography=False)
    parser.add_argument(
        "--art-guidance-photography",
        action="store_true",
        help="Include photography packs for --art-guidance-mode auto|all.",
    )
    parser.add_argument(
        "--no-art-guidance-photography",
        action="store_true",
        help="Disable photography packs for --art-guidance-mode auto|all.",
    )
    parser.add_argument(
        "--anatomy-guidance",
        type=str,
        default="",
        choices=["", "none", "lite", "strong"],
        help="Override anatomy/proportion guidance sent to sample.py (default from --book-accuracy).",
    )
    parser.add_argument(
        "--style-guidance-mode",
        type=str,
        default="",
        choices=["", "none", "auto", "all"],
        help="Override style-domain guidance sent to sample.py (default from --book-accuracy).",
    )
    parser.set_defaults(style_guidance_artists=False)
    parser.add_argument(
        "--style-guidance-artists",
        action="store_true",
        help="Enable artist/game-name stabilization cues in style guidance.",
    )
    parser.add_argument(
        "--no-style-guidance-artists",
        action="store_true",
        help="Disable artist/game-name stabilization cues in style guidance.",
    )
    parser.add_argument("--vae-tiling", action="store_true", help="Large outputs: tiled VAE decode.")
    parser.add_argument(
        "--pick-clip-model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="CLIP model id for --pick-best clip/combo.",
    )
    parser.add_argument("--pick-save-all", action="store_true", help="Keep all candidates when using pick-best.")
    parser.add_argument("--grid", action="store_true", help="Save N-up grid when num>1.")
    parser.add_argument("--cfg-scale", type=float, default=0.0, help="0 = sample.py default.")
    parser.add_argument("--cfg-rescale", type=float, default=0.0, help="0 = off unless sample auto-enables.")
    parser.add_argument(
        "--resize-mode",
        type=str,
        default="stretch",
        choices=["stretch", "center_crop", "saliency_crop"],
        help="When --width/--height are set: how sample.py fits output aspect.",
    )
    parser.add_argument(
        "--resize-saliency-face-bias",
        type=float,
        default=0.0,
        help="Extra face priority for --resize-mode saliency_crop.",
    )
    parser.add_argument(
        "--dynamic-threshold-percentile",
        type=float,
        default=0.0,
        help="e.g. 99.5 with --dynamic-threshold-type percentile (0 = off).",
    )
    parser.add_argument(
        "--post-sharpen",
        type=float,
        default=-1.0,
        help="Unsharp strength after each image (-1 = use --book-accuracy preset).",
    )
    parser.add_argument("--post-naturalize", action="store_true", help="Film grain + micro-contrast (less plastic).")
    parser.add_argument("--no-post-naturalize", action="store_true")
    parser.add_argument("--post-grain", type=float, default=-1.0, help="Grain amount (-1 = preset).")
    parser.add_argument("--post-micro-contrast", type=float, default=-1.0, help="-1 = preset.")

    # Lexicon / aspect (see pipelines/book_comic/prompt_lexicon.py, docs/BOOK_COMIC_TECH.md)
    parser.add_argument(
        "--lexicon-style",
        default="none",
        choices=[
            "none",
            "shonen",
            "shoujo",
            "seinen",
            "slice_of_life",
            "chibi",
            "webtoon",
            "manhwa_color",
            "graphic_novel",
            "editorial",
            "light_novel",
            "yonkoma",
            "anime_2d",
            "anime_3d",
            "cartoon_2d",
            "cartoon_3d",
            "web_comic",
            "digital_art",
            "digital_3d",
            "drawing_ink",
            "painting_oil",
            "painting_watercolor",
            "realistic_photo",
            "realistic_painting",
            "fantasy_concept",
            "sci_fi_concept",
            "cyberpunk",
            "steampunk",
            "noir_comic",
            "art_nouveau",
            "ukiyo_e",
            "pixel_retro",
            "voxel_isometric",
            "clay_stop_motion",
            "render_octane",
            "render_eevee",
            "editorial_fashion_photo",
            "black_white_film",
            "baroque",
            "rococo",
            "impressionist",
            "expressionist",
            "cubist",
            "surrealist",
            "art_deco",
            "minimalist",
            "maximalist",
            "anime_shonen_battle",
            "anime_shojo_romance",
            "anime_seinen_gritty",
            "anime_isekai_fantasy",
            "anime_mecha",
            "anime_idol",
            "newspaper_comic",
            "manga_horror",
            "retro_pulp_cover",
            "poster_graphic",
            "archviz_real",
            "product_cg",
            "clay_render_bw",
            "toon_render_hybrid",
            "film_35mm",
            "polaroid_vintage",
            "street_noir_photo",
            "wildlife_naturalist",
        ],
        help="Append style snippet (ink, pacing, format) to the book-type prefix.",
    )
    parser.add_argument(
        "--art-medium-pack",
        default="none",
        choices=[
            "none",
            "digital_painting_pro",
            "drawing_ink_pro",
            "stylized_3d_game",
            "pbr_3d_realism",
            "oil_painting_classic",
            "watercolor_storybook",
            "photo_real_cinematic",
            "anime_2d_pro",
            "anime_3d_pro",
            "cartoon_2d_pro",
            "webcomic_mobile",
            "fantasy_concept_keyart",
            "cyberpunk_noir_panel",
            "mecha_anime_action",
            "editorial_fashion_real",
            "film_noir_bw_real",
            "octane_3d_cinematic",
            "isometric_voxel_world",
            "mixed_media_collage",
            "risograph_poster",
            "concept_sheet_design",
            "visual_dev_story",
            "comic_pencil_storyboard",
            "crosshatch_ink_master",
            "archviz_cinematic",
            "product_cg_studio",
            "mecha_3d_detail",
            "surreal_paint_studio",
            "mural_graphic_large",
            "sports_photo_pro",
            "wedding_photo_editorial",
            "food_photo_editorial",
            "anime_movie_keyart",
            "superhero_action_modern",
            "indie_webtoon_episode",
        ],
        help="One-flag art medium preset pack (digital/drawing/3d/painting/realistic/anime/cartoon/web-comic).",
    )
    parser.add_argument(
        "--art-medium-family",
        default="none",
        choices=[
            "none",
            "digital_art",
            "drawing_art",
            "digital_3d_art",
            "painting_art",
            "realistic_art",
            "anime_cartoon_webcomic",
            "mixed_media_art",
        ],
        help="Broad art medium family helper.",
    )
    parser.add_argument(
        "--art-medium-variant",
        default="none",
        help=(
            "Specific medium variant helper (examples: painting, cel_shaded, pencil, ink, "
            "stylized, pbr_realistic, oil, watercolor, photoreal, anime_2d, anime_3d, web_comic)."
        ),
    )
    parser.add_argument(
        "--art-medium-extra",
        default="",
        help="Optional custom medium fragment merged after selected medium family/variant.",
    )
    parser.add_argument(
        "--aspect-preset",
        default="none",
        choices=[
            "none",
            "square",
            "print_manga",
            "webtoon_tall",
            "widescreen_panel",
            "cover_hd",
            "double_page_spread",
            "print_us_comic",
        ],
        help="Set --width/--height when both are 0 (see prompt_lexicon.ASPECT_PRESETS).",
    )
    parser.set_defaults(lexicon_negative=True)
    parser.add_argument(
        "--no-lexicon-negative",
        dest="lexicon_negative",
        action="store_false",
        help="Do not merge lexicon anti-artifact negatives into --negative-prompt.",
    )
    parser.add_argument(
        "--include-tategaki-hint",
        action="store_true",
        help="Add vertical JP lettering hint to prefix (training data should support JP).",
    )
    parser.add_argument("--include-sfx-hint", action="store_true", help="Add hand-drawn SFX typography hint.")
    parser.add_argument(
        "--include-print-finish",
        action="store_true",
        help="Add print-ready line weight / halftone hint (prompt_lexicon.PRINT_FINISH_HINT).",
    )
    parser.add_argument(
        "--include-cover-spotlight",
        action="store_true",
        help="Add strong focal / title-area hint (covers and pin-ups; COVER_SPOTLIGHT_HINT).",
    )
    parser.add_argument(
        "--book-style-pack",
        default="none",
        choices=["none", "manga_nsfw_action", "webtoon_nsfw_romance", "comic_dialogue_safe", "oc_launch_safe"],
        help="One-flag style bundle that sets artist/OC/NSFW defaults; explicit flags override.",
    )
    parser.add_argument(
        "--humanize-pack",
        default="none",
        choices=["none", "lite", "balanced", "strong", "painterly", "filmic"],
        help="One-flag bundle for anti-AI humanization hints and negatives.",
    )
    parser.add_argument(
        "--auto-humanize",
        action="store_true",
        help="Auto-pick humanization profile from book type/style/safety mode; explicit --humanize-* still override.",
    )
    parser.add_argument(
        "--artist-craft-profile",
        default="none",
        choices=["none", "manga_pro", "western_comic_pro", "webtoon_pro", "children_book", "cinematic_storyboard"],
        help="Artist-facing production helper profile (panel flow, focal hierarchy, readability).",
    )
    parser.add_argument(
        "--artist-pack",
        default="none",
        choices=["none", "manga_cinematic", "comic_dialogue", "webtoon_scroll", "storyboard_fast"],
        help="Preset bundle for artist craft controls; explicit per-control flags override this pack.",
    )
    parser.add_argument(
        "--shot-language",
        default="none",
        choices=["none", "mixed", "cinematic", "manga_dynamic", "dialogue_coverage"],
        help="Shot grammar helper (establishing/medium/close-up rhythm and dialogue coverage).",
    )
    parser.add_argument(
        "--pacing-plan",
        default="none",
        choices=["none", "decompressed", "balanced", "compressed"],
        help="Narrative beat density helper for panel rhythm.",
    )
    parser.add_argument(
        "--lettering-craft",
        default="none",
        choices=["none", "standard", "strict"],
        help="Lettering and balloon placement helper cues.",
    )
    parser.add_argument(
        "--value-plan",
        default="none",
        choices=["none", "bw_hierarchy", "color_script"],
        help="Value structure helper (B/W hierarchy or color script discipline).",
    )
    parser.add_argument(
        "--screentone-plan",
        default="none",
        choices=["none", "clean", "dramatic"],
        help="Screentone/halftone handling helper cues.",
    )
    parser.add_argument(
        "--humanize-profile",
        default="none",
        choices=["none", "lite", "balanced", "strong", "painterly", "filmic"],
        help="Positive humanization profile hint.",
    )
    parser.add_argument(
        "--humanize-imperfection",
        default="none",
        choices=["none", "lite", "balanced", "strong"],
        help="Imperfect hand-made variance level.",
    )
    parser.add_argument(
        "--humanize-materiality",
        default="none",
        choices=["none", "paper", "ink_paper", "canvas", "print", "film"],
        help="Surface/material realism cue for human-made feel.",
    )
    parser.add_argument(
        "--humanize-asymmetry",
        default="none",
        choices=["none", "lite", "balanced", "strong"],
        help="Natural asymmetry cue level.",
    )
    parser.add_argument(
        "--humanize-negative-level",
        default="none",
        choices=["none", "lite", "balanced", "strong"],
        help="Anti-synthetic negative prompt boost level.",
    )
    parser.add_argument(
        "--safety-mode",
        default="",
        choices=["", "none", "sfw", "nsfw"],
        help="Optional override forwarded to sample.py safety scaffolding.",
    )
    parser.add_argument(
        "--nsfw-pack",
        default="",
        choices=["", "none", "soft", "explicit_detail", "romantic", "extreme"],
        help="Optional adult-content stability pack forwarded to sample.py.",
    )
    parser.add_argument(
        "--nsfw-civitai-pack",
        default="",
        choices=["", "none", "hits", "hits_lite", "snippets", "snippets_lite", "action", "complex", "easy", "clothing", "objects", "style"],
        help="Optional Civitai-derived NSFW trigger pack forwarded to sample.py.",
    )
    parser.add_argument(
        "--civitai-trigger-bank",
        default="",
        choices=["", "none", "light", "medium", "heavy", "frequency_light", "frequency_medium", "frequency_heavy"],
        help="Optional trigger bank for sample.py when using --safety-mode nsfw.",
    )
    parser.add_argument("--oc-name", default="", help="Original character name/handle for identity locking.")
    parser.add_argument(
        "--oc-pack",
        default="none",
        choices=["none", "heroine_scifi", "rival_dark", "mentor_classic"],
        help="Preset OC design bundle; explicit --oc-* flags override this pack.",
    )
    parser.add_argument(
        "--oc-archetype",
        default="none",
        choices=["none", "shonen_lead", "cool_rival", "mentor", "antihero", "magical_girl", "noir_detective", "space_pilot"],
        help="Original character archetype helper preset.",
    )
    parser.add_argument("--oc-traits", default="", help="Signature OC traits (hair, eyes, scars, accessories).")
    parser.add_argument("--oc-wardrobe", default="", help="Consistent outfit/costume anchors for the OC.")
    parser.add_argument("--oc-silhouette", default="", help="Silhouette language lock (shape identity cues).")
    parser.add_argument("--oc-color-motifs", default="", help="Color motifs/palette anchors for the OC.")
    parser.add_argument("--oc-expression-sheet", default="", help="Expression anchors for the OC across scenes.")
    parser.add_argument("--oc-negative", default="", help="Negative prompt fragment for OC consistency failures.")

    parser.add_argument(
        "--chapter-break-every",
        type=int,
        default=0,
        help="After every N pages, drop inpaint chain (fresh page like page 0). 0=off.",
    )
    parser.add_argument(
        "--page-context-previous",
        type=int,
        default=0,
        help="Append a short summary of the last N page prompts for continuity. 0=off.",
    )
    parser.add_argument(
        "--page-context-max-chars",
        type=int,
        default=500,
        help="Max characters for rolling page context (with --page-context-previous).",
    )
    parser.add_argument(
        "--panel-layout",
        default="none",
        choices=[
            "none",
            "single",
            "two_panel_horizontal",
            "two_panel_vertical",
            "three_panel_strip",
            "four_koma",
            "splash",
            "grid_2x2",
        ],
        help="Soft panel/grid hint merged into each page prompt (prompt_lexicon.PANEL_LAYOUT_HINTS).",
    )
    parser.add_argument(
        "--narration-prefix",
        default="",
        help="Optional string prepended to every page (series voice, e.g. dark fantasy noir).",
    )
    parser.add_argument(
        "--sample-originality",
        type=float,
        default=0.0,
        help="Forward to sample.py --originality (0–1) for less templated pages; 0=omit flag.",
    )
    parser.add_argument(
        "--sample-creativity",
        type=float,
        default=-1.0,
        help="Forward to sample.py --creativity (0–1). -1 = omit (use checkpoint default).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="If page_NNN.png exists, skip regeneration (still updates inpaint chain + manifest).",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=0,
        help="0-based index to start generating from (earlier prompts still feed --page-context-previous).",
    )
    parser.add_argument(
        "--write-book-manifest",
        action="store_true",
        help="Write book_manifest.json under --output-dir (prompts, seeds, paths, flags).",
    )

    # Cross-page consistency (prompt cues; see pipelines/book_comic/consistency_helpers.py)
    parser.add_argument(
        "--consistency-json",
        default="",
        help="JSON spec for character/props/vehicle/setting/lettering (merged with CLI flags).",
    )
    parser.add_argument(
        "--consistency-character",
        default="",
        help="Freeform recurring protagonist appearance (or use --consistency-json character object).",
    )
    parser.add_argument("--consistency-costume", default="", help="Locked outfit description for every page.")
    parser.add_argument(
        "--consistency-props",
        default="",
        help="Important props; semicolon-separated (each becomes a same-object cue).",
    )
    parser.add_argument("--consistency-vehicle", default="", help="Recurring vehicle description.")
    parser.add_argument("--consistency-setting", default="", help="Continuous location / environment cue.")
    parser.add_argument(
        "--consistency-creature",
        default="",
        help="Recurring pet, mascot, or non-human companion description.",
    )
    parser.add_argument("--consistency-palette", default="", help="Locked color palette hint (comics color script).")
    parser.add_argument(
        "--consistency-lighting",
        default="",
        help="Consistent lighting / key direction (reduces drift across inpaint chain).",
    )
    parser.add_argument(
        "--consistency-visual-extra",
        default="",
        help="Extra freeform tokens appended to the consistency block.",
    )
    parser.add_argument(
        "--consistency-lettering-hard",
        action="store_true",
        help="Add strong legible-lettering positive cues (use with OCR / expected text for dialogue).",
    )
    parser.add_argument(
        "--consistency-negative",
        default=None,
        choices=["none", "light", "strong"],
        help="Append consistency anti-drift negatives (default: none, or JSON negative_level if set).",
    )

    args = parser.parse_args()

    _ensure_repo_on_path()
    from pipelines.book_comic import book_helpers, consistency_helpers, prompt_lexicon

    settings = book_helpers.resolve_book_sample_settings(args)
    _style_cfg = prompt_lexicon.resolve_book_style_controls(
        book_style_pack=str(getattr(args, "book_style_pack", "none") or "none"),
        artist_pack=str(getattr(args, "artist_pack", "none") or "none"),
        oc_pack=str(getattr(args, "oc_pack", "none") or "none"),
        safety_mode=str(getattr(args, "safety_mode", "") or ""),
        nsfw_pack=str(getattr(args, "nsfw_pack", "") or ""),
        nsfw_civitai_pack=str(getattr(args, "nsfw_civitai_pack", "") or ""),
        civitai_trigger_bank=str(getattr(args, "civitai_trigger_bank", "") or ""),
    )
    _auto_h = (
        prompt_lexicon.infer_auto_humanize_controls(
            book_type=str(getattr(args, "book_type", "manga") or "manga"),
            lexicon_style=str(getattr(args, "lexicon_style", "none") or "none"),
            safety_mode=str(_style_cfg.get("safety_mode", "") or ""),
        )
        if bool(getattr(args, "auto_humanize", False))
        else {}
    )
    _human_profile_raw = str(getattr(args, "humanize_profile", "none") or "none")
    _human_imperf_raw = str(getattr(args, "humanize_imperfection", "none") or "none")
    _human_mat_raw = str(getattr(args, "humanize_materiality", "none") or "none")
    _human_asym_raw = str(getattr(args, "humanize_asymmetry", "none") or "none")
    _human_neg_raw = str(getattr(args, "humanize_negative_level", "none") or "none")
    _human_cfg = prompt_lexicon.resolve_humanize_controls(
        humanize_pack=str(getattr(args, "humanize_pack", "none") or "none"),
        humanize_profile=_auto_h.get("humanize_profile", "none") if _human_profile_raw == "none" else _human_profile_raw,
        imperfection_level=_auto_h.get("imperfection_level", "none") if _human_imperf_raw == "none" else _human_imperf_raw,
        materiality_mode=_auto_h.get("materiality_mode", "none") if _human_mat_raw == "none" else _human_mat_raw,
        asymmetry_level=_auto_h.get("asymmetry_level", "none") if _human_asym_raw == "none" else _human_asym_raw,
        negative_level=_auto_h.get("negative_level", "none") if _human_neg_raw == "none" else _human_neg_raw,
    )

    def _cfg_cmd_tail() -> List[str]:
        tail: List[str] = []
        book_helpers.append_optional_sample_flags(
            tail,
            vae_tiling=bool(getattr(args, "vae_tiling", False)),
            pick_clip_model=str(getattr(args, "pick_clip_model", "") or ""),
            pick_save_all=bool(getattr(args, "pick_save_all", False)),
            cfg_scale=float(getattr(args, "cfg_scale", 0.0) or 0.0),
            cfg_rescale=float(getattr(args, "cfg_rescale", 0.0) or 0.0),
            dynamic_threshold_percentile=float(getattr(args, "dynamic_threshold_percentile", 0.0) or 0.0),
            resize_mode=str(getattr(args, "resize_mode", "stretch") or "stretch"),
            resize_saliency_face_bias=float(getattr(args, "resize_saliency_face_bias", 0.0) or 0.0),
            grid=bool(getattr(args, "grid", False)),
        )
        return tail

    def _nsfw_tail() -> List[str]:
        tail: List[str] = []
        sm = str(_style_cfg.get("safety_mode", "") or "").strip().lower()
        if sm in ("none", "sfw", "nsfw"):
            tail.extend(["--safety-mode", sm])
        npack = str(_style_cfg.get("nsfw_pack", "") or "").strip().lower()
        if npack in ("none", "soft", "explicit_detail", "romantic", "extreme"):
            tail.extend(["--nsfw-pack", npack])
        ncp = str(_style_cfg.get("nsfw_civitai_pack", "") or "").strip().lower()
        if ncp in ("none", "hits", "hits_lite", "snippets", "snippets_lite", "action", "complex", "easy", "clothing", "objects", "style"):
            tail.extend(["--nsfw-civitai-pack", ncp])
        ctb = str(_style_cfg.get("civitai_trigger_bank", "") or "").strip().lower()
        if ctb in ("none", "light", "medium", "heavy", "frequency_light", "frequency_medium", "frequency_heavy"):
            tail.extend(["--civitai-trigger-bank", ctb])
        return tail

    ocr_extra = book_helpers.build_extra_ocr_sample_flags(settings) + _cfg_cmd_tail() + _nsfw_tail()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(exist_ok=True)
    cover_dir = out_dir / "cover"
    cover_dir.mkdir(exist_ok=True)

    expected_texts = _parse_expected_texts(args.expected_text)
    cover_expected_texts = (
        _parse_expected_texts(args.cover_expected_text) if args.cover_expected_text.strip() else expected_texts
    )
    pages_expected_texts = (
        _parse_expected_texts(args.pages_expected_text) if args.pages_expected_text.strip() else expected_texts
    )

    book_prefix_map: Dict[str, str] = {
        "manga": "manga panel page, black and white ink, screentones, clean lineart, high contrast, dynamic composition",
        "comic": "comic book page, inked lineart, screentone shading, crisp outlines, strong silhouettes, panel layout",
        "novel_cover": "book cover design, title typography, author name, professional layout, readable lettering",
        "storyboard": "storyboard frame, manga anime layout, clean thumbnails, camera framing, clear panel borders",
    }
    base_prefix = book_prefix_map.get(args.book_type, "")
    prompt_prefix = prompt_lexicon.enhance_book_prefix(
        base_prefix,
        lexicon_style=str(getattr(args, "lexicon_style", "none") or "none"),
        book_type=str(args.book_type),
        include_tategaki_hint=bool(getattr(args, "include_tategaki_hint", False)),
        include_sfx_hint=bool(getattr(args, "include_sfx_hint", False)),
        include_print_finish=bool(getattr(args, "include_print_finish", False)),
        include_cover_spotlight=bool(getattr(args, "include_cover_spotlight", False)),
    )

    # Optional width/height from aspect preset when user did not set both.
    ap = str(getattr(args, "aspect_preset", "none") or "none")
    aw, ah = prompt_lexicon.aspect_dimensions(ap)
    if aw > 0 and ah > 0 and (not int(args.width or 0)) and (not int(args.height or 0)):
        args.width, args.height = aw, ah

    panel_hint_str = prompt_lexicon.panel_layout_hint(str(getattr(args, "panel_layout", "none") or "none"))
    _artist_cfg = prompt_lexicon.resolve_artist_controls(
        artist_pack=str(_style_cfg.get("artist_pack", "none") or "none"),
        craft_profile=str(getattr(args, "artist_craft_profile", "none") or "none"),
        shot_language=str(getattr(args, "shot_language", "none") or "none"),
        pacing_plan=str(getattr(args, "pacing_plan", "none") or "none"),
        lettering_craft=str(getattr(args, "lettering_craft", "none") or "none"),
        value_plan=str(getattr(args, "value_plan", "none") or "none"),
        screentone_plan=str(getattr(args, "screentone_plan", "none") or "none"),
    )
    artist_hint_str = prompt_lexicon.artist_craft_bundle(**_artist_cfg)
    _medium_cfg = prompt_lexicon.resolve_art_medium_controls(
        art_medium_pack=str(getattr(args, "art_medium_pack", "none") or "none"),
        art_medium_family=str(getattr(args, "art_medium_family", "none") or "none"),
        art_medium_variant=str(getattr(args, "art_medium_variant", "none") or "none"),
        art_medium_extra=str(getattr(args, "art_medium_extra", "") or ""),
    )
    medium_hint_str = prompt_lexicon.art_medium_bundle(
        family=str(_medium_cfg.get("family", "none") or "none"),
        variant=str(_medium_cfg.get("variant", "none") or "none"),
        extra=str(_medium_cfg.get("extra", "") or ""),
    )
    human_hint_str = prompt_lexicon.humanize_prompt_bundle(
        humanize_profile=str(_human_cfg.get("humanize_profile", "none") or "none"),
        imperfection_level=str(_human_cfg.get("imperfection_level", "none") or "none"),
        materiality_mode=str(_human_cfg.get("materiality_mode", "none") or "none"),
        asymmetry_level=str(_human_cfg.get("asymmetry_level", "none") or "none"),
    )
    panel_hint_str = prompt_lexicon.merge_prompt_fragments(panel_hint_str, artist_hint_str)
    panel_hint_str = prompt_lexicon.merge_prompt_fragments(panel_hint_str, medium_hint_str)
    panel_hint_str = prompt_lexicon.merge_prompt_fragments(panel_hint_str, human_hint_str)
    _oc_cfg = prompt_lexicon.resolve_oc_controls(
        oc_pack=str(_style_cfg.get("oc_pack", "none") or "none"),
        name=str(getattr(args, "oc_name", "") or ""),
        archetype=str(getattr(args, "oc_archetype", "none") or "none"),
        visual_traits=str(getattr(args, "oc_traits", "") or ""),
        wardrobe=str(getattr(args, "oc_wardrobe", "") or ""),
        silhouette=str(getattr(args, "oc_silhouette", "") or ""),
        color_motifs=str(getattr(args, "oc_color_motifs", "") or ""),
        expression_sheet=str(getattr(args, "oc_expression_sheet", "") or ""),
    )
    oc_block = prompt_lexicon.original_character_bundle(**_oc_cfg)
    narration_p = (getattr(args, "narration_prefix", "") or "").strip()

    consistency_spec: Dict[str, Any] = {}
    cj = str(getattr(args, "consistency_json", "") or "").strip()
    if cj:
        consistency_spec = dict(consistency_helpers.load_consistency_json(Path(cj)))
    consistency_helpers.overlay_cli_on_spec(consistency_spec, args)
    consistency_block = consistency_helpers.positive_block_from_mapping(consistency_spec)
    consistency_neg_level = consistency_helpers.negative_level_from_spec(
        consistency_spec, getattr(args, "consistency_negative", None)
    )
    consistency_neg_fragment = consistency_helpers.consistency_negative_addon(consistency_neg_level)
    chapter_every = max(0, int(getattr(args, "chapter_break_every", 0) or 0))
    context_n = max(0, int(getattr(args, "page_context_previous", 0) or 0))
    context_mc = max(32, int(getattr(args, "page_context_max_chars", 500) or 500))
    start_idx = max(0, int(getattr(args, "start_page", 0) or 0))
    prev_prompts: List[str] = []
    manifest_rows: List[Dict[str, Any]] = []

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
        raise SystemExit(
            "Provide either --prompts-file or (--pages and --page-prompt-template), or provide --cover-prompt."
        )

    # Initialize OCR lazily (only when needed).
    # - If OCR-fix is requested, we only need OCR if we actually have expected text.
    # - If speech-bubble anchoring is enabled, OCR is always needed (no expected text required).
    if (args.ocr_fix or args.anchor_speech_bubbles) and text_engine is None:
        any_page_expected = any(bool(x) for x in page_expected_overrides)
        needs_expected_text_ocr = args.ocr_fix and (cover_expected_texts or pages_expected_texts or any_page_expected)
        needs_speech_bubble_ocr = args.anchor_speech_bubbles
        if needs_expected_text_ocr or needs_speech_bubble_ocr:
            try:
                from utils.generation.text_rendering import create_text_rendering_pipeline

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

    def _merged_negative() -> str:
        base = (
            prompt_lexicon.suggest_negative_addon(
            use_lexicon_negative=True,
            user_negative=(args.negative_prompt or ""),
            production_tier=str(getattr(args, "book_accuracy", "") or "").lower() == "production",
            artist_lettering_strict=str(_artist_cfg.get("lettering_craft", "none") or "none").lower() == "strict",
            ).strip()
            if getattr(args, "lexicon_negative", True)
            else (args.negative_prompt or "").strip()
        )
        if consistency_neg_fragment:
            base = f"{base}, {consistency_neg_fragment}".strip().strip(",")
        hneg = prompt_lexicon.humanize_negative_addon(str(_human_cfg.get("negative_level", "none") or "none"))
        if hneg:
            base = f"{base}, {hneg}".strip().strip(",")
        oc_neg = str(getattr(args, "oc_negative", "") or "").strip()
        if oc_neg:
            base = f"{base}, {oc_neg}".strip().strip(",")
        return base

    def _postprocess_output(path: Path, page_seed: int) -> None:
        book_helpers.apply_postprocess_to_image_file(
            path,
            sharpen_amount=settings.post_sharpen,
            naturalize=settings.post_naturalize,
            grain=settings.post_grain,
            micro_contrast=settings.post_micro_contrast,
            seed=page_seed,
        )

    def sample_generate(
        prompt: str,
        out_path: Path,
        *,
        expected_texts_for_prompt: List[str],
        init_image: Optional[Path] = None,
        mask_path: Optional[Path] = None,
        strength: Optional[float] = None,
        page_seed: Optional[int] = None,
    ) -> None:
        prompt_final = _apply_book_style(prompt_prefix, prompt)
        if args.force_text_quote and expected_texts_for_prompt:
            prompt_final = _maybe_append_text_says(prompt_final, expected_texts_for_prompt)
        prompt_final = book_helpers.enhance_prompt_for_page(prompt_final, settings=settings)

        pick_exp = book_helpers.expected_text_for_pick(expected_texts_for_prompt)

        cmd: List[str] = [
            sys.executable,
            str(sample_py_path),
            "--ckpt",
            args.ckpt,
            "--prompt",
            prompt_final,
            "--out",
            str(out_path),
            "--steps",
            str(args.steps),
            "--seed",
            str(args.seed if page_seed is None else page_seed),
            "--device",
            args.device,
            "--preset",
            args.model_preset,
            "--scheduler",
            args.scheduler,
        ]
        book_helpers.append_sample_py_quality_flags(
            cmd,
            settings,
            pick_expected_text=pick_exp,
            pick_expected_count=int(getattr(args, "expected_count", 0) or 0),
            pick_expected_count_target=str(getattr(args, "expected_count_target", "auto") or "auto"),
            pick_expected_count_object=str(getattr(args, "expected_count_object", "") or ""),
        )
        cmd.extend(_cfg_cmd_tail())
        cmd.extend(_nsfw_tail())
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
        mn = _merged_negative()
        if mn:
            cmd += ["--negative-prompt", mn]
        if args.no_neg_filter:
            cmd += ["--no-neg-filter"]
        if args.text_in_image:
            cmd += ["--text-in-image"]

        orig = float(getattr(args, "sample_originality", 0.0) or 0.0)
        if orig > 0:
            cmd.extend(["--originality", str(max(0.0, min(1.0, orig)))])
        try:
            cr = float(getattr(args, "sample_creativity", -1.0))
        except (TypeError, ValueError):
            cr = -1.0
        if cr >= 0.0:
            cmd.extend(["--creativity", str(max(0.0, min(1.0, cr)))])

        if init_image is not None:
            cmd += ["--init-image", str(init_image)]
        if strength is not None:
            cmd += ["--strength", str(strength)]
        if mask_path is not None:
            cmd += ["--mask", str(mask_path)]
            cmd += ["--inpaint-mode", "mdm"]

        _safe_run(cmd)
        _postprocess_output(out_path, page_seed if page_seed is not None else int(args.seed))

    # Cover generation
    if args.cover_prompt:
        cover_out = cover_dir / "cover.png"
        cover_composed = book_helpers.compose_book_page_prompt(
            user_prompt=args.cover_prompt,
            narration_prefix=narration_p,
            consistency_block=prompt_lexicon.merge_prompt_fragments(consistency_block, oc_block),
            panel_hint=prompt_lexicon.merge_prompt_fragments(artist_hint_str, human_hint_str),
            rolling_context="",
        )
        sample_generate(
            cover_composed,
            cover_out,
            expected_texts_for_prompt=cover_expected_texts,
            page_seed=int(args.seed),
        )
        if args.write_book_manifest:
            manifest_rows.append(
                {
                    "kind": "cover",
                    "path": f"cover/{cover_out.name}",
                    "prompt": cover_composed,
                    "seed": int(args.seed),
                }
            )
        # OCR-fix cover too (optional)
        if args.ocr_fix and cover_expected_texts:
            ocr_out = cover_out
            _try_ocr_fix(
                image_path=cover_out,
                expected_texts=cover_expected_texts,
                prompt=cover_composed,
                ckpt=args.ckpt,
                out_path=ocr_out,
                sample_steps=args.steps,
                strength=args.page_inpaint_strength,
                inpaint_strength=args.text_inpaint_strength,
                sample_width=args.width,
                sample_height=args.height,
                device=args.device,
                negative_prompt=_merged_negative(),
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
                ocr_extra_flags=ocr_extra,
            )
            _postprocess_output(cover_out, int(args.seed))

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

        if i < start_idx:
            prev_prompts.append(page_prompt)
            if page_out.is_file():
                prev_path = page_out
            continue

        rolling = book_helpers.build_rolling_page_context(
            prev_prompts, num_previous=context_n, max_chars=context_mc
        )
        composed_prompt = book_helpers.compose_book_page_prompt(
            user_prompt=page_prompt,
            narration_prefix=narration_p,
            consistency_block=prompt_lexicon.merge_prompt_fragments(consistency_block, oc_block),
            panel_hint=panel_hint_str,
            rolling_context=rolling,
        )

        if chapter_every > 0 and i > 0 and (i % chapter_every == 0):
            prev_path = None

        if args.skip_existing and page_out.is_file():
            prev_prompts.append(page_prompt)
            prev_path = page_out
            if args.write_book_manifest:
                manifest_rows.append(
                    {
                        "kind": "page",
                        "index": i,
                        "path": f"pages/{page_out.name}",
                        "prompt": composed_prompt,
                        "seed": int(args.seed) + i * 9973,
                        "skipped": True,
                    }
                )
            continue

        # If this page has an override expected text, use it; otherwise use the global pages_expected_texts.
        page_expected_texts = pages_expected_texts
        if i < len(page_expected_overrides) and page_expected_overrides[i]:
            page_expected_texts = page_expected_overrides[i]

        user_anchor_exists = user_anchor_mask_internal is not None and user_anchor_mask_internal.exists()
        should_anchor = prev_path is not None and (
            args.anchor_face or args.edge_anchor or args.anchor_speech_bubbles or user_anchor_exists
        )

        page_seed = int(args.seed) + i * 9973

        if i == 0 or not should_anchor:
            sample_generate(
                composed_prompt,
                page_out,
                expected_texts_for_prompt=page_expected_texts,
                page_seed=page_seed,
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
                    composed_prompt,
                    page_out,
                    expected_texts_for_prompt=page_expected_texts,
                    page_seed=page_seed,
                )
            else:
                mask_path = pages_dir / f"anchor_keep_mask_{i:03d}.png"
                _combine_keep_masks(keep_masks, mask_path)

                sample_generate(
                    composed_prompt,
                    page_out,
                    expected_texts_for_prompt=page_expected_texts,
                    init_image=prev_path,
                    mask_path=mask_path,
                    strength=args.page_inpaint_strength,
                    page_seed=page_seed,
                )

        # Optional OCR repair
        if args.ocr_fix and page_expected_texts:
            _try_ocr_fix(
                image_path=page_out,
                expected_texts=page_expected_texts,
                prompt=composed_prompt,
                ckpt=args.ckpt,
                out_path=page_out,
                sample_steps=args.steps,
                strength=args.page_inpaint_strength,
                inpaint_strength=args.text_inpaint_strength,
                sample_width=args.width,
                sample_height=args.height,
                device=args.device,
                negative_prompt=_merged_negative(),
                no_neg_filter=args.no_neg_filter,
                text_engine=text_engine,
                ocr_engine=ocr_engine,
                max_iters=args.ocr_iters,
                threshold=args.ocr_threshold,
                inpaint_mode="mdm",
                seed=page_seed,
                sampler=args.scheduler,
                text_in_image_flag=args.text_in_image,
                ocr_mask_dilate=args.ocr_mask_dilate,
                ocr_extra_flags=ocr_extra,
            )
            _postprocess_output(page_out, page_seed)

        prev_prompts.append(page_prompt)
        prev_path = page_out
        if args.write_book_manifest:
            manifest_rows.append(
                {
                    "kind": "page",
                    "index": i,
                    "path": f"pages/{page_out.name}",
                    "prompt": composed_prompt,
                    "seed": page_seed,
                    "skipped": False,
                }
            )

    if args.write_book_manifest:
        mf = {
            "ckpt": args.ckpt,
            "book_type": args.book_type,
            "model_preset": args.model_preset,
            "book_style_pack": getattr(args, "book_style_pack", ""),
            "humanize_pack": getattr(args, "humanize_pack", ""),
            "auto_humanize": bool(getattr(args, "auto_humanize", False)),
            "lexicon_style": getattr(args, "lexicon_style", ""),
            "art_medium_pack": getattr(args, "art_medium_pack", ""),
            "art_medium_family": str(_medium_cfg.get("family", "none") or "none"),
            "art_medium_variant": str(_medium_cfg.get("variant", "none") or "none"),
            "art_medium_extra": str(_medium_cfg.get("extra", "") or ""),
            "artist_pack": getattr(args, "artist_pack", ""),
            "artist_craft_profile": getattr(args, "artist_craft_profile", ""),
            "shot_language": getattr(args, "shot_language", ""),
            "pacing_plan": getattr(args, "pacing_plan", ""),
            "lettering_craft": getattr(args, "lettering_craft", ""),
            "value_plan": getattr(args, "value_plan", ""),
            "screentone_plan": getattr(args, "screentone_plan", ""),
            "humanize_profile": str(_human_cfg.get("humanize_profile", "none") or "none"),
            "humanize_imperfection": str(_human_cfg.get("imperfection_level", "none") or "none"),
            "humanize_materiality": str(_human_cfg.get("materiality_mode", "none") or "none"),
            "humanize_asymmetry": str(_human_cfg.get("asymmetry_level", "none") or "none"),
            "humanize_negative_level": str(_human_cfg.get("negative_level", "none") or "none"),
            "oc_pack": getattr(args, "oc_pack", ""),
            "panel_layout": getattr(args, "panel_layout", ""),
            "narration_prefix": narration_p,
            "consistency_block": consistency_block,
            "original_character_block": oc_block,
            "consistency_negative_level": consistency_neg_level,
            "consistency_json": cj or None,
            "safety_mode": str(_style_cfg.get("safety_mode", "") or ""),
            "nsfw_pack": str(_style_cfg.get("nsfw_pack", "") or ""),
            "nsfw_civitai_pack": str(_style_cfg.get("nsfw_civitai_pack", "") or ""),
            "civitai_trigger_bank": str(_style_cfg.get("civitai_trigger_bank", "") or ""),
            "shortcomings_mitigation": getattr(settings, "shortcomings_mitigation", "none"),
            "shortcomings_2d": bool(getattr(settings, "shortcomings_2d", False)),
            "art_guidance_mode": getattr(settings, "art_guidance_mode", "none"),
            "art_guidance_photography": bool(getattr(settings, "art_guidance_photography", True)),
            "anatomy_guidance": getattr(settings, "anatomy_guidance", "none"),
            "style_guidance_mode": getattr(settings, "style_guidance_mode", "none"),
            "style_guidance_artists": bool(getattr(settings, "style_guidance_artists", True)),
            "resize_mode": str(getattr(args, "resize_mode", "stretch") or "stretch"),
            "resize_saliency_face_bias": float(getattr(args, "resize_saliency_face_bias", 0.0) or 0.0),
            "chapter_break_every": chapter_every,
            "page_context_previous": context_n,
            "entries": manifest_rows,
        }
        (out_dir / "book_manifest.json").write_text(json.dumps(mf, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Book generation finished: {out_dir}")


if __name__ == "__main__":
    main()

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

Stack: each page calls **repo-root** ``sample.py`` (DiT + diffusion), forwarding pick-best metrics that can load **vit_quality** checkpoints and optional **DiT AR regime** alignment via ``utils.architecture.ar_block_conditioning`` (see ``--pick-vit-ar-blocks`` / ``--pick-vit-ar-from-ckpt``).
Optional **LoRA / DoRA / LyCORIS**, stacked **ControlNet** maps, **holy-grail** scheduling, and **reference / IP-Adapter** weights use the same flags as ``sample.py`` (see ``book_helpers.extend_sample_py_adapter_control_cmd``).
**Hires-fix, finishing presets, latent spectral/domain knobs, refinement, and face/post-reference** flags match ``sample.py`` via ``book_helpers.extend_sample_py_sdx_enhance_cmd`` (OCR repair subprocesses get the same fragment).
**``--book-preflight``** checks that **DiT** and **ViT** checkpoints, LoRAs, and control maps exist and warns on AR / resolution mismatches (``book_model_readiness``).
**``--book-dry-run``** prints the first resolved ``sample.py`` argv and exits (no images written).
**Adherence / quality packs, CLIP guard & monitor, volatile CFG, SAG, dual-stage layout** forward through
``book_helpers.extend_sample_py_adherence_quality_cmd`` (also appended to OCR repair argv).
"""

from __future__ import annotations

import argparse
import json
import shlex
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


def _resolve_pick_report_json_path(raw: str, out_path: Path) -> str:
    """
    Map ``--pick-report-json`` to a concrete file for this page/cover image.

    - ``per-page`` / ``each``: ``<out_stem>_pick_report.json`` next to the PNG.
    - Path ending in ``.json``: that exact file (last page wins if reused).
    - Otherwise: treat as a directory and write ``<dir>/<stem>_pick_report.json``.
    """
    prj = (raw or "").strip()
    if not prj:
        return ""
    low = prj.lower()
    if low in ("per-page", "per_page", "each"):
        return str(out_path.parent / f"{out_path.stem}_pick_report.json")
    p = Path(prj).expanduser()
    if p.suffix.lower() == ".json":
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)
    p.mkdir(parents=True, exist_ok=True)
    return str(p / f"{out_path.stem}_pick_report.json")


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
        "--page-prompt-template",
        default="",
        help="Template per page; placeholders: {page}/{page0} (0-based), {page1}, {total_pages}, {total_pages0}, plus custom {keys} left as-is.",
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

    # --- Accuracy / consistency (uses utils/quality/test_time_pick, data/caption_utils) ---
    parser.add_argument(
        "--book-accuracy",
        default="none",
        choices=["none", "fast", "balanced", "maximum", "production", "production_vit", "production_fidelity"],
        help=(
            "Preset: balanced=2+combo; maximum=4; production=6+combo; production_vit=6+combo_vit_hq; "
            "production_fidelity=8+combo_vit_hq (max pick budget; pair --pick-vit-ckpt + --adherence-pack / CLIP flags); "
            "none=single sample."
        ),
    )
    parser.add_argument(
        "--book-preflight",
        default="warn",
        choices=["off", "warn", "strict"],
        help=(
            "Verify --ckpt / --pick-vit-ckpt / LoRA paths and DiT vs ViT AR alignment before running. "
            "strict=exit on missing files or ViT metrics without a scorer checkpoint."
        ),
    )
    parser.add_argument(
        "--book-dry-run",
        action="store_true",
        help="Print the first resolved sample.py command (shlex-quoted) and exit; no subprocess, no PNGs.",
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
        choices=[
            "auto",
            "none",
            "clip",
            "edge",
            "ocr",
            "vit",
            "aesthetic",
            "combo",
            "combo_vit",
            "combo_vit_hq",
            "combo_vit_realism",
            "combo_count_vit",
            "combo_exposure",
            "combo_structural",
            "combo_hq",
            "combo_count",
            "combo_realism",
            "aesthetic_realism",
        ],
        help=(
            "Forwarded to sample.py --pick-best (ViT metrics need --pick-vit-ckpt). "
            "See utils/quality/test_time_pick.py."
        ),
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
    parser.add_argument(
        "--pick-vit-ckpt",
        default="",
        help="vit_quality checkpoint (best.pt) for sample.py vit / combo_vit* / combo_count_vit pick metrics.",
    )
    parser.add_argument(
        "--pick-vit-use-adherence",
        action="store_true",
        help="Blend ViT adherence head in sample.py when using vit/combo_vit metrics.",
    )
    parser.add_argument(
        "--pick-vit-ar-blocks",
        type=int,
        default=-1,
        help="0/2/4: ViT scorer AR regime matching DiT num_ar_blocks (-1 = unknown one-hot). See utils/architecture/ar_block_conditioning.py.",
    )
    parser.add_argument(
        "--pick-vit-ar-from-ckpt",
        action="store_true",
        help="Set --pick-vit-ar-blocks from --ckpt metadata (overrides explicit --pick-vit-ar-blocks when successful).",
    )
    parser.add_argument(
        "--pick-report-json",
        default="",
        help=(
            "sample.py --pick-report-json: per-page use ``per-page`` (sidecar next to each PNG), "
            "a ``.json`` file path, or a directory (writes ``<stem>_pick_report.json`` there)."
        ),
    )
    parser.add_argument(
        "--pick-auto-no-clip",
        action="store_true",
        help="Forward sample.py --pick-auto-no-clip (auto pick-best branches).",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=0,
        help="sample.py DiT beam search width (with --num 1); 0=off.",
    )
    parser.add_argument("--beam-steps", type=int, default=0, help="Early steps for beam stage (sample.py).")
    parser.add_argument("--beam-metric", default="", help="Metric for beam previews (sample.py).")
    parser.add_argument("--beam2-width", type=int, default=0, help="sample.py second-stage micro-beam width; 0=off.")
    parser.add_argument("--beam2-steps", type=int, default=0, help="Micro-beam denoise steps (sample.py).")
    parser.add_argument("--beam2-metric", default="", help="Metric for micro-beam (sample.py).")
    parser.add_argument(
        "--beam2-at-frac",
        type=float,
        default=0.65,
        help="When to run micro-beam as fraction of steps (sample.py).",
    )
    parser.add_argument(
        "--beam2-noise",
        type=float,
        default=0.03,
        help="Latent noise std for micro-beam branches (sample.py).",
    )
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
            "anime_game_toon_pbr",
            "genshin_like_3d_anime",
            "honkai_starrail_3d_anime",
            "zenless_zone_urban_anime",
            "persona_ui_stylized",
            "arcane_painterly_3d",
            "guiltygear_hybrid",
            "anime_cg_cutscene",
            "digital_fantasy_splash",
            "digital_semi_real_portrait",
            "digital_mobile_game_iconic",
            "pbr_cinematic_keyart",
            "stylized_3d_overwatch_like",
            "unreal_realtime_cinematic",
            "lineart_character_sheet",
            "ink_crosshatch_noir",
            "oil_portrait_classical",
            "gouache_poster_graphic",
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
            "digital_splash_master",
            "digital_portrait_master",
            "mobile_game_illustration_pro",
            "unreal_cinematic_3d",
            "anime_game_3d_pro",
            "hero_stylized_3d_pro",
            "lineart_sheet_pro",
            "ink_noir_pro",
            "storyboard_rough_pro",
            "oil_portrait_master",
            "gouache_poster_master",
            "watercolor_botanical_master",
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
        "--color-render-pack",
        default="none",
        choices=[
            "none",
            "anime_cel_master",
            "painting_value_master",
            "comic_noir_master",
            "pbr_3d_master",
            "toon_3d_master",
            "photo_grade_master",
            "hybrid_2d3d_master",
        ],
        help="One-flag color/render preset for color theory, gradients, shading, and render pipeline cues.",
    )
    parser.add_argument(
        "--color-theory-mode",
        default="none",
        choices=[
            "none",
            "balanced",
            "complementary",
            "split_complementary",
            "analogous",
            "triadic",
            "tetradic",
            "monochrome",
            "warm_cool",
            "gamut_print",
        ],
        help="Color theory helper mode (palette relationship and value/saturation orchestration).",
    )
    parser.add_argument(
        "--gradient-blend-mode",
        default="none",
        choices=["none", "clean", "painterly", "atmospheric", "toon_steps", "volumetric"],
        help="Gradient and blend behavior helper mode.",
    )
    parser.add_argument(
        "--shading-technique",
        default="none",
        choices=["none", "cel", "soft_painterly", "crosshatch", "chiaroscuro", "pbr", "subsurface"],
        help="Shading strategy helper mode for 2D/3D/photo outputs.",
    )
    parser.add_argument(
        "--render-pipeline",
        default="none",
        choices=["none", "illustration_2d", "anime_2d", "toon_3d", "pbr_3d", "cinematic_photo", "hybrid_2d3d"],
        help="Render-pipeline language helper across 2D, 3D, and photographic aesthetics.",
    )
    parser.add_argument(
        "--color-render-extra",
        default="",
        help="Optional custom color/render fragment merged after selected controls.",
    )
    parser.add_argument(
        "--artist-technique-pack",
        default="none",
        choices=["none", "digital_2d_master", "anime_2d_master", "drawing_ink_master", "painting_master", "toon_3d_master", "pbr_3d_master"],
        help="One-flag artist-technique preset for linework/render/shading/material/composition.",
    )
    parser.add_argument(
        "--linework-technique",
        default="none",
        choices=["none", "clean_contour", "expressive_weight", "crosshatch_precision", "sketch_loose", "calligraphic_ink"],
        help="Linework technique emphasis.",
    )
    parser.add_argument(
        "--rendering-technique",
        default="none",
        choices=["none", "cel_anime", "painterly_2d", "toon_3d", "pbr_3d", "hybrid_2d3d"],
        help="Rendering technique emphasis.",
    )
    parser.add_argument(
        "--shading-technique-plan",
        default="none",
        choices=["none", "chiaroscuro", "ambient_occlusion", "rim_bounce", "subsurface_skin", "volumetric_depth"],
        help="Shading strategy emphasis.",
    )
    parser.add_argument(
        "--material-technique",
        default="none",
        choices=["none", "fabric_folds", "metal_surface", "skin_microdetail", "paint_texture", "paper_grain"],
        help="Material rendering technique emphasis.",
    )
    parser.add_argument(
        "--composition-technique",
        default="none",
        choices=["none", "rule_of_thirds", "leading_lines", "depth_layers", "silhouette_focus", "negative_space"],
        help="Composition technique emphasis.",
    )
    parser.add_argument(
        "--artist-technique-extra",
        default="",
        help="Optional extra artist-technique fragment merged after selected technique controls.",
    )
    parser.add_argument(
        "--photo-realism-pack",
        default="none",
        choices=["none", "documentary", "cinematic", "studio_portrait", "film_analog", "night_noir", "product_catalog", "fashion_editorial"],
        help="Photography realism pack forwarded to sample.py.",
    )
    parser.add_argument(
        "--photo-color-grade",
        default="none",
        choices=["none", "natural", "teal_orange", "kodak_portra", "cinestill_800t", "noir_bw", "fujifilm_eterna"],
        help="Photography color-grade profile forwarded to sample.py.",
    )
    parser.add_argument(
        "--photo-lighting-technique",
        default="none",
        choices=["none", "three_point", "golden_hour", "overcast_soft", "motivated_practical", "rim_backlight", "butterfly", "rembrandt"],
        help="Photography lighting technique forwarded to sample.py.",
    )
    parser.add_argument(
        "--photo-filter",
        default="none",
        choices=["none", "pro_mist", "polarizer", "nd_long_exposure", "vintage_diffusion", "clean_digital"],
        help="Photography filter profile forwarded to sample.py.",
    )
    parser.add_argument(
        "--photo-grain-style",
        default="none",
        choices=["none", "fine_35mm", "medium_35mm", "heavy_16mm", "clean_digital"],
        help="Photography grain style forwarded to sample.py.",
    )
    parser.add_argument("--photo-realism-strength", type=float, default=1.0, help="Photo realism prompt strength.")
    parser.set_defaults(auto_photo_realism=True)
    parser.add_argument(
        "--auto-photo-realism",
        dest="auto_photo_realism",
        action="store_true",
        help="Auto-infer photo realism controls from prompt keywords in sample.py.",
    )
    parser.add_argument(
        "--no-auto-photo-realism",
        dest="auto_photo_realism",
        action="store_false",
        help="Disable auto photo realism inference in sample.py.",
    )
    parser.set_defaults(realism_autopilot=True)
    parser.add_argument(
        "--realism-autopilot",
        dest="realism_autopilot",
        action="store_true",
        help="Enable realism autopilot in sample.py (default on).",
    )
    parser.add_argument(
        "--no-realism-autopilot",
        dest="realism_autopilot",
        action="store_false",
        help="Disable realism autopilot in sample.py.",
    )
    parser.set_defaults(photo_postprocess=True)
    parser.add_argument(
        "--photo-postprocess",
        dest="photo_postprocess",
        action="store_true",
        help="Enable photo postprocess in sample.py (default on).",
    )
    parser.add_argument(
        "--no-photo-postprocess",
        dest="photo_postprocess",
        action="store_false",
        help="Disable photo postprocess in sample.py.",
    )
    parser.add_argument("--photo-post-strength", type=float, default=0.6, help="Photo postprocess strength.")

    # --- LoRA / DoRA / LyCORIS, ControlNet, reference (forwarded to sample.py) ---
    parser.add_argument(
        "--style",
        default="",
        help="Optional global style prompt fragment for every page (sample.py --style).",
    )
    parser.add_argument("--style-strength", type=float, default=0.7, help="sample.py --style-strength.")
    parser.add_argument(
        "--auto-style-from-prompt",
        action="store_true",
        help="sample.py: infer --style from prompt keywords when --style is empty.",
    )
    parser.add_argument(
        "--tags",
        default="",
        help="Comma-separated tags prepended by sample.py (Danbooru-style).",
    )
    parser.add_argument("--tags-file", default="", help="Path to tag list for sample.py --tags-file.")
    parser.add_argument("--control-image", default="", help="Single control map path (depth/edge/pose, etc.).")
    parser.add_argument(
        "--control-type",
        default="auto",
        choices=["auto", "unknown", "canny", "depth", "pose", "seg", "lineart", "scribble", "normal", "hed"],
        help="Type when using --control-image (sample.py).",
    )
    parser.add_argument("--control-scale", type=float, default=0.85, help="ControlNet strength (sample.py).")
    parser.add_argument("--control-guidance-start", type=float, default=0.0, help="Control schedule start fraction.")
    parser.add_argument("--control-guidance-end", type=float, default=1.0, help="Control schedule end fraction.")
    parser.add_argument("--control-guidance-decay", type=float, default=1.0, help="Control decay power in schedule.")
    parser.add_argument(
        "--control",
        nargs="*",
        default=None,
        metavar="SPEC",
        help="Stacked controls for sample.py: path:type:scale ... (see sample.py --control).",
    )
    parser.add_argument("--holy-grail", action="store_true", help="sample.py adaptive CFG/control/adapter scheduling.")
    parser.add_argument("--holy-grail-cfg-early-ratio", type=float, default=0.72)
    parser.add_argument("--holy-grail-cfg-late-ratio", type=float, default=1.0)
    parser.add_argument("--holy-grail-control-mult", type=float, default=1.0)
    parser.add_argument("--holy-grail-adapter-mult", type=float, default=1.0)
    parser.add_argument("--holy-grail-no-frontload-control", action="store_true")
    parser.add_argument("--holy-grail-late-adapter-boost", type=float, default=1.15)
    parser.add_argument("--holy-grail-cads-strength", type=float, default=0.0)
    parser.add_argument("--holy-grail-cads-min-strength", type=float, default=0.0)
    parser.add_argument("--holy-grail-cads-power", type=float, default=1.0)
    parser.add_argument("--holy-grail-unsharp-sigma", type=float, default=0.0)
    parser.add_argument("--holy-grail-unsharp-amount", type=float, default=0.0)
    parser.add_argument("--holy-grail-clamp-quantile", type=float, default=0.0)
    parser.add_argument("--holy-grail-clamp-floor", type=float, default=1.0)
    parser.add_argument(
        "--lora",
        nargs="*",
        default=None,
        metavar="SPEC",
        help=(
            "LoRA / DoRA / LyCORIS specs per sample.py: path, path:scale, path:scale:role "
            "(repeat roles: character, style, detail, composition)."
        ),
    )
    parser.add_argument(
        "--no-lora-normalize-scales",
        action="store_true",
        help="sample.py: disable per-layer multi-adapter scale normalization.",
    )
    parser.add_argument("--lora-max-total-scale", type=float, default=1.5, help="Cap total adapter scale per layer.")
    parser.add_argument(
        "--lora-default-role",
        default="style",
        help="Default :role when a --lora spec omits it.",
    )
    parser.add_argument(
        "--lora-role-budgets",
        default="",
        help="Override sample.py role budgets (default uses sample.py built-in string if empty).",
    )
    parser.add_argument(
        "--lora-stage-policy",
        default="auto",
        choices=["off", "auto", "character_focus", "style_focus", "balanced"],
        help="Depth-aware LoRA role routing (sample.py).",
    )
    parser.add_argument(
        "--lora-layers",
        default="all",
        choices=["all", "first", "middle", "last"],
        help="Restrict LoRA to early/mid/late DiT blocks (sample.py).",
    )
    parser.add_argument(
        "--lora-role-stage-weights",
        default="",
        help="Per-role early/mid/late multipliers (see sample.py --lora-role-stage-weights).",
    )
    parser.add_argument("--lora-trigger", default="", help="Prepend trigger tokens when using --lora.")
    parser.add_argument(
        "--lora-scaffold",
        default="none",
        choices=["none", "blend", "character_first", "style_first"],
        help="Prompt scaffolding when using LoRA stacks.",
    )
    parser.add_argument(
        "--lora-scaffold-auto",
        action="store_true",
        help="Use blend scaffolding when --lora set and --lora-scaffold none.",
    )
    parser.add_argument("--reference-image", default="", help="CLIP vision reference image (sample.py).")
    parser.add_argument("--reference-strength", type=float, default=1.0)
    parser.add_argument("--reference-tokens", type=int, default=4)
    parser.add_argument(
        "--reference-clip-model",
        default="openai/clip-vit-large-patch14",
        help="HF CLIP id for --reference-image.",
    )
    parser.add_argument(
        "--reference-adapter-pt",
        default="",
        help="Trained IP-Adapter-style projector checkpoint (sample.py).",
    )

    # --- sample.py SDX polish (hires / latent / post / refine / faces) — see book_helpers.extend_sample_py_sdx_enhance_cmd
    parser.add_argument(
        "--flow-matching-sample",
        action="store_true",
        help="Rectified-flow Euler/Heun sampler (sample.py --flow-matching-sample).",
    )
    parser.add_argument(
        "--flow-solver",
        default="euler",
        choices=["euler", "heun"],
        help="ODE solver with --flow-matching-sample (sample.py --flow-solver).",
    )
    parser.add_argument(
        "--domain-prior-latent",
        type=float,
        default=0.0,
        help="sample.py latent domain prior strength (>0 enables).",
    )
    parser.add_argument(
        "--spectral-coherence-latent",
        type=float,
        default=0.0,
        help="sample.py FFT low-frequency latent blend strength (>0 enables).",
    )
    parser.add_argument(
        "--spectral-coherence-cutoff",
        type=float,
        default=0.15,
        help="Radial cutoff for --spectral-coherence-latent (sample.py default 0.15).",
    )
    parser.add_argument("--hires-fix", action="store_true", help="sample.py latent upscale + short refine pass.")
    parser.add_argument("--hires-scale", type=float, default=1.5, help="Target scale when hires-fix infers size.")
    parser.add_argument("--hires-steps", type=int, default=15, help="Denoising steps for hires pass.")
    parser.add_argument("--hires-strength", type=float, default=0.35, help="Noise level 0–1 for hires pass.")
    parser.add_argument(
        "--hires-cfg-scale",
        type=float,
        default=-1.0,
        help="CFG during hires; <0 = same as main --cfg-scale (sample.py).",
    )
    parser.add_argument(
        "--finishing-preset",
        default="none",
        choices=["none", "photo", "anime", "illustration", "characters", "painterly"],
        help="sample.py cross-style finishing preset (adds baseline clarity/tone/chroma).",
    )
    parser.add_argument(
        "--sharpen",
        type=float,
        default=0.0,
        help="sample.py post: unsharp strength 0–1 (0=off; forwarded to main + OCR repair).",
    )
    parser.add_argument("--contrast", type=float, default=1.0, help="sample.py post: contrast factor (1=off).")
    parser.add_argument("--saturation", type=float, default=1.0, help="sample.py post: saturation factor (1=off).")
    parser.add_argument("--clarity", type=float, default=0.0, help="sample.py post: local contrast / clarity.")
    parser.add_argument("--tone-punch", type=float, default=0.0, dest="tone_punch", help="sample.py tone curve punch.")
    parser.add_argument("--chroma-smooth", type=float, default=0.0, dest="chroma_smooth", help="sample.py chroma smooth.")
    parser.add_argument(
        "--polish",
        type=float,
        default=0.0,
        help="sample.py one-knob polish (S-curve + chroma + clarity + grain).",
    )
    parser.add_argument(
        "--face-enhance",
        action="store_true",
        dest="face_enhance",
        help="sample.py OpenCV face patches sharpen/contrast (needs opencv-python).",
    )
    parser.add_argument("--face-enhance-sharpen", type=float, default=0.35, dest="face_enhance_sharpen")
    parser.add_argument("--face-enhance-contrast", type=float, default=1.04, dest="face_enhance_contrast")
    parser.add_argument("--face-enhance-padding", type=float, default=0.25, dest="face_enhance_padding")
    parser.add_argument("--face-enhance-max", type=int, default=4, dest="face_enhance_max")
    parser.add_argument(
        "--post-reference-image",
        default="",
        dest="post_reference_image",
        help="sample.py whole-frame RGB blend reference (weak color/style pull).",
    )
    parser.add_argument(
        "--post-reference-alpha",
        type=float,
        default=0.0,
        dest="post_reference_alpha",
        help="Blend weight for --post-reference-image (0=off).",
    )
    parser.add_argument(
        "--face-restore-shell",
        default="",
        dest="face_restore_shell",
        help="sample.py shell hook {src}/{dst} after save (e.g. GFPGAN CLI).",
    )
    parser.add_argument("--no-refine", action="store_true", dest="no_refine", help="sample.py: skip latent refinement pass.")
    parser.add_argument("--refine-t", type=int, default=50, dest="refine_t", help="sample.py refinement noise level t.")
    parser.add_argument(
        "--refine-gate",
        default="off",
        choices=["off", "auto"],
        dest="refine_gate",
        help="sample.py: run refinement only when quick quality score is below threshold.",
    )
    parser.add_argument(
        "--refine-gate-threshold",
        type=float,
        default=0.62,
        dest="refine_gate_threshold",
        help="Threshold for --refine-gate auto.",
    )

    # --- Prompt / style fidelity (sample.py packs, CLIP hooks, layout, caching) ---
    parser.add_argument(
        "--quality-pack",
        default="none",
        choices=["none", "top", "one_shot", "ultra_clean", "cinematic", "illustrative", "editorial", "micro_detail"],
        help="sample.py high-quality artifact-control / detail scaffolding.",
    )
    parser.add_argument(
        "--adherence-pack",
        default="none",
        choices=["none", "standard", "strict"],
        help="sample.py literal prompt adherence scaffolding (long complex prompts).",
    )
    parser.add_argument(
        "--clip-guard-threshold",
        type=float,
        default=0.0,
        dest="clip_guard_threshold",
        help="If >0: CLIP cosine gate + short extra denoise (sample.py; slow, needs transformers). Try 0.20–0.28.",
    )
    parser.add_argument(
        "--clip-guard-model",
        default="openai/clip-vit-base-patch32",
        dest="clip_guard_model",
        help="HF CLIP id for --clip-guard-threshold / --clip-monitor-every.",
    )
    parser.add_argument(
        "--clip-guard-t-frac",
        type=float,
        default=0.22,
        dest="clip_guard_t_frac",
        help="Timestep fraction for CLIP-guard re-noising.",
    )
    parser.add_argument(
        "--clip-guard-steps",
        type=int,
        default=12,
        dest="clip_guard_steps",
        help="Extra denoise steps when CLIP guard triggers.",
    )
    parser.add_argument(
        "--clip-monitor-every",
        type=int,
        default=0,
        dest="clip_monitor_every",
        help="If >0: decode every N steps, boost CFG when CLIP cosine is low (sample.py; very slow).",
    )
    parser.add_argument(
        "--clip-monitor-threshold",
        type=float,
        default=0.22,
        dest="clip_monitor_threshold",
        help="CLIP cosine threshold for --clip-monitor-every.",
    )
    parser.add_argument(
        "--clip-monitor-cfg-boost",
        type=float,
        default=0.12,
        dest="clip_monitor_cfg_boost",
        help="CFG multiplicative boost when monitor fires.",
    )
    parser.add_argument(
        "--clip-monitor-rewind",
        type=float,
        default=0.0,
        dest="clip_monitor_rewind",
        help="Soft-rewind latent mix when monitor fires (0=off).",
    )
    parser.add_argument(
        "--volatile-cfg-boost",
        type=float,
        default=0.0,
        dest="volatile_cfg_boost",
        help="Spike-aware CFG multiplier tail (sample.py); try 0.08–0.18.",
    )
    parser.add_argument(
        "--volatile-cfg-quantile",
        type=float,
        default=0.72,
        dest="volatile_cfg_quantile",
        help="Rolling latent-delta quantile for volatile CFG.",
    )
    parser.add_argument(
        "--volatile-cfg-window",
        type=int,
        default=6,
        dest="volatile_cfg_window",
        help="Rolling window length for volatile CFG.",
    )
    parser.add_argument(
        "--sag-scale",
        type=float,
        default=0.0,
        dest="sag_scale",
        help="Self-attention guidance style heuristic (sample.py); ~0.12–0.35; ~2× forward cost.",
    )
    parser.add_argument(
        "--no-auto-expected-text",
        action="store_true",
        dest="no_auto_expected_text",
        help="sample.py: do not infer --expected-text from quoted prompt lines.",
    )
    parser.add_argument(
        "--no-auto-constraint-boost",
        action="store_true",
        dest="no_auto_constraint_boost",
        help="sample.py: do not auto-raise --num when text/count constraints are detected.",
    )
    parser.add_argument(
        "--hard-style",
        default="",
        choices=["", "3d", "realistic", "3d_realistic", "style_mix"],
        help="sample.py style-domain tag pack (empty = omit).",
    )
    parser.add_argument(
        "--dual-stage-layout",
        action="store_true",
        dest="dual_stage_layout",
        help="sample.py: coarse latent layout then upscale + detail pass (no img2img/inpaint).",
    )
    parser.add_argument("--dual-stage-div", type=int, default=2, dest="dual_stage_div", help="Latent divisor for layout stage.")
    parser.add_argument("--dual-layout-steps", type=int, default=24, dest="dual_layout_steps", help="Steps for coarse stage.")
    parser.add_argument("--dual-detail-steps", type=int, default=20, dest="dual_detail_steps", help="Steps after latent upscale.")
    parser.add_argument(
        "--dual-detail-strength",
        type=float,
        default=0.38,
        dest="dual_detail_strength",
        help="Re-noise strength before detail stage.",
    )
    parser.add_argument(
        "--sample-deterministic",
        action="store_true",
        dest="deterministic",
        help="Forward sample.py --deterministic (cudnn reproducibility where supported).",
    )
    parser.add_argument(
        "--sample-no-cache",
        action="store_true",
        dest="no_cache",
        help="Forward sample.py --no-cache (disable T5 encoding cache).",
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
        choices=[
            "none",
            "manga_nsfw_action",
            "webtoon_nsfw_romance",
            "manga_nsfw_surreal",
            "webtoon_nsfw_complex",
            "comic_dialogue_safe",
            "oc_launch_safe",
        ],
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
        "--book-authenticity",
        default="none",
        choices=["none", "lite", "standard", "strong"],
        help=(
            "Extra sequential-art craft positives + anti-AI negatives (manga/webtoon/comic/illustration); "
            "pairs with --humanize-pack. Uses --book-authenticity-medium (auto reads --visual-memory book_style)."
        ),
    )
    parser.add_argument(
        "--book-authenticity-medium",
        default="auto",
        choices=[
            "auto",
            "manga",
            "webtoon",
            "graphic_novel",
            "comic_us",
            "illustration",
            "children",
            "storyboard",
        ],
        help="Which authenticity recipe to use; auto = book_type + lexicon_style + visual-memory book_style.",
    )
    parser.add_argument(
        "--book-challenge-pack",
        default="none",
        choices=[
            "none",
            "mature_coherence",
            "surreal_weird",
            "technical_hard",
            "horror_mood",
            "crowd_hands",
            "max",
        ],
        help=(
            "Extra positive/negative fragments for NSFW narrative fidelity (with --safety-mode nsfw), "
            "surreal OCs, crowds, reflections, etc. See book_challenging_content.py."
        ),
    )
    parser.add_argument(
        "--book-challenge-extra",
        default="",
        help="Freeform fragment merged with --book-challenge-pack positives.",
    )
    parser.add_argument(
        "--user-style-fragment",
        default="",
        help=(
            "Freeform user aesthetic repeated on every page (e.g. gouache texture, specific palette); "
            "layers on top of book/lexicon style and visual-memory user_style_anchor."
        ),
    )
    parser.add_argument(
        "--style-fusion-preset",
        default="none",
        choices=[
            "none",
            "manga_comic",
            "webtoon_manga",
            "graphic_comic",
            "manhwa_western",
            "illustration_manga",
        ],
        help="Hybrid sequential-art idiom (manga×comic, etc.); see book_style_fusion.py. Visual-memory style_mix can also set this.",
    )
    parser.add_argument(
        "--style-secondary",
        default="",
        help=(
            "Secondary idiom to fuse when preset is none: manga, webtoon, graphic_novel, comic_us, "
            "illustration, manhwa (underscores ok)."
        ),
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
        choices=[
            "none",
            "manga_cinematic",
            "comic_dialogue",
            "webtoon_scroll",
            "storyboard_fast",
            "anime_shinkai_cinematic",
            "anime_ghibli_story",
            "anime_trigger_action",
            "anime_nomura_character",
            "game_riot_splash",
            "game_valorant_clean",
            "game_blizzard_heroic",
            "game_fromsoftware_dark",
            "cartoon_pixar_story",
            "cartoon_disney_staging",
            "cartoon_network_graphic",
            "mignola_noir",
            "alex_ross_painterly",
            "frazetta_epic",
            "moebius_worldbuilding",
            "anime3d_genshin_keyart",
            "anime3d_hsr_cinematic",
            "anime3d_zzz_urban",
            "anime3d_arcane_painterly",
        ],
        help="Preset bundle for artist craft controls; explicit per-control flags override this pack.",
    )
    parser.add_argument(
        "--artist-style-profile",
        default="none",
        choices=[
            "none",
            "shinkai_cinematic_anime",
            "miyazaki_ghibli_storybook",
            "trigger_dynamic_action",
            "clamp_fashion_linework",
            "toriyama_clean_adventure",
            "otomo_urban_mecha",
            "nomura_character_design",
            "shinkawa_brush_stealth",
            "riot_splash_fantasy",
            "valorant_clean_shapes",
            "blizzard_cinematic_heroic",
            "fromsoftware_dark_fantasy",
            "zelda_painterly_adventure",
            "pixar_shape_script",
            "disney_animation_staging",
            "cartoon_network_graphic",
            "nickelodeon_expressive",
            "mignola_noir_graphic",
            "alex_ross_painterly_realism",
            "frazetta_epic_fantasy",
            "moebius_line_worldbuilding",
            "niziu_anime_3d_colorist",
            "reiq_anime_3d_sculpt",
            "swd3e2_game_anime_hybrid",
            "alpha_3d_anime_keyart",
        ],
        help="Named artist/studio-inspired style profile for anime, game art, cartoons, and comics.",
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
        "--auto-original-character",
        dest="auto_original_character",
        action="store_true",
        help="Auto-synthesize OC profile when cover/page prompts request an original character (default: on).",
    )
    parser.add_argument(
        "--no-auto-original-character",
        dest="auto_original_character",
        action="store_false",
        help="Disable automatic OC synthesis from prompt intent.",
    )
    parser.set_defaults(auto_original_character=True)
    parser.add_argument(
        "--auto-oc-seed-offset",
        type=int,
        default=0,
        help="Extra deterministic seed offset used by auto-OC synthesis.",
    )

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
    parser.add_argument(
        "--visual-memory",
        default="",
        help=(
            "JSON visual memory for cast/props (proportions, camera, page overrides); "
            "see pipelines/book_comic/visual_memory.py and examples/book_visual_memory.example.json"
        ),
    )

    args = parser.parse_args()
    if getattr(args, "lora", None) is None:
        args.lora = []
    if getattr(args, "control", None) is None:
        args.control = []

    _ensure_repo_on_path()
    from utils.architecture.ar_block_conditioning import read_num_ar_blocks_from_checkpoint

    _dit_ar_nb = -1
    try:
        _dit_ar_nb = int(read_num_ar_blocks_from_checkpoint(Path(args.ckpt)))
    except Exception:
        _dit_ar_nb = -1
    if bool(getattr(args, "pick_vit_ar_from_ckpt", False)) and _dit_ar_nb in (0, 2, 4):
        args.pick_vit_ar_blocks = _dit_ar_nb

    from pipelines.book_comic import (
        book_challenging_content,
        book_helpers,
        book_model_readiness,
        book_style_authenticity,
        book_style_fusion,
        consistency_helpers,
        prompt_lexicon,
    )
    from pipelines.book_comic import visual_memory as book_visual_memory

    settings = book_helpers.resolve_book_sample_settings(args)
    _style_cfg = prompt_lexicon.resolve_book_style_controls(
        book_style_pack=str(getattr(args, "book_style_pack", "none") or "none"),
        artist_pack=str(getattr(args, "artist_pack", "none") or "none"),
        oc_pack=str(getattr(args, "oc_pack", "none") or "none"),
        safety_mode=str(getattr(args, "safety_mode", "") or ""),
        nsfw_pack=str(getattr(args, "nsfw_pack", "") or ""),
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

    _audit_errs, _audit_warns = book_helpers.audit_book_run_flags(
        pick_best=str(settings.pick_best or ""),
        sample_candidates=int(settings.sample_candidates),
        pick_vit_ckpt=str(getattr(args, "pick_vit_ckpt", "") or ""),
        beam_width=int(getattr(args, "beam_width", 0) or 0),
        book_challenge_pack=str(getattr(args, "book_challenge_pack", "none") or "none"),
        safety_mode=str(_style_cfg.get("safety_mode", "") or ""),
        clip_guard_threshold=float(getattr(args, "clip_guard_threshold", 0.0) or 0.0),
        clip_monitor_every=int(getattr(args, "clip_monitor_every", 0) or 0),
        adherence_pack=str(getattr(args, "adherence_pack", "none") or "none"),
        pick_vit_use_adherence=bool(getattr(args, "pick_vit_use_adherence", False)),
    )
    for _aw in _audit_warns:
        print(f"WARNING [book]: {_aw}", file=sys.stderr)
    if _audit_errs:
        raise SystemExit("book run configuration errors:\n" + "\n".join(_audit_errs))

    _pf_errs, _pf_warns = book_model_readiness.run_book_preflight(
        args,
        dit_ar_blocks=_dit_ar_nb,
        mode=str(getattr(args, "book_preflight", "warn") or "warn"),
        resolved_pick_best=str(settings.pick_best or ""),
    )
    for _pw in _pf_warns:
        print(f"WARNING [book preflight]: {_pw}", file=sys.stderr)
    if _pf_errs:
        raise SystemExit("book preflight (strict) failed:\n" + "\n".join(_pf_errs))

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
        return tail

    def _photo_tail() -> List[str]:
        tail: List[str] = []
        prp = str(getattr(args, "photo_realism_pack", "none") or "none").strip().lower()
        if prp != "none":
            tail.extend(["--photo-realism-pack", prp])
        prg = str(getattr(args, "photo_color_grade", "none") or "none").strip().lower()
        if prg != "none":
            tail.extend(["--photo-color-grade", prg])
        prl = str(getattr(args, "photo_lighting_technique", "none") or "none").strip().lower()
        if prl != "none":
            tail.extend(["--photo-lighting-technique", prl])
        prf = str(getattr(args, "photo_filter", "none") or "none").strip().lower()
        if prf != "none":
            tail.extend(["--photo-filter", prf])
        prn = str(getattr(args, "photo_grain_style", "none") or "none").strip().lower()
        if prn != "none":
            tail.extend(["--photo-grain-style", prn])
        prs = float(getattr(args, "photo_realism_strength", 1.0) or 1.0)
        if abs(prs - 1.0) > 1e-6:
            tail.extend(["--photo-realism-strength", str(prs)])
        if not bool(getattr(args, "auto_photo_realism", True)):
            tail.append("--no-auto-photo-realism")
        if not bool(getattr(args, "photo_postprocess", True)):
            tail.append("--no-photo-postprocess")
        pps = float(getattr(args, "photo_post_strength", 0.6) or 0.6)
        if abs(pps - 0.6) > 1e-6:
            tail.extend(["--photo-post-strength", str(pps)])
        if not bool(getattr(args, "realism_autopilot", True)):
            tail.append("--no-realism-autopilot")
        return tail

    ocr_extra = (
        book_helpers.build_extra_ocr_sample_flags(settings)
        + _cfg_cmd_tail()
        + _nsfw_tail()
        + _photo_tail()
        + book_helpers.adapter_control_argv_for_sample(args)
        + book_helpers.sdx_enhance_argv_for_sample(args)
        + book_helpers.adherence_quality_argv_for_sample(args)
    )

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
        artist_style_profile=str(getattr(args, "artist_style_profile", "none") or "none"),
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
    _color_render_cfg = prompt_lexicon.resolve_color_render_controls(
        color_render_pack=str(getattr(args, "color_render_pack", "none") or "none"),
        color_theory_mode=str(getattr(args, "color_theory_mode", "none") or "none"),
        gradient_blend_mode=str(getattr(args, "gradient_blend_mode", "none") or "none"),
        shading_technique=str(getattr(args, "shading_technique", "none") or "none"),
        render_pipeline=str(getattr(args, "render_pipeline", "none") or "none"),
        color_render_extra=str(getattr(args, "color_render_extra", "") or ""),
    )
    color_render_hint_str = prompt_lexicon.color_render_bundle(**_color_render_cfg)
    _technique_cfg = prompt_lexicon.resolve_artist_technique_controls(
        artist_technique_pack=str(getattr(args, "artist_technique_pack", "none") or "none"),
        linework_technique=str(getattr(args, "linework_technique", "none") or "none"),
        rendering_technique=str(getattr(args, "rendering_technique", "none") or "none"),
        shading_technique_plan=str(getattr(args, "shading_technique_plan", "none") or "none"),
        material_technique=str(getattr(args, "material_technique", "none") or "none"),
        composition_technique=str(getattr(args, "composition_technique", "none") or "none"),
        artist_technique_extra=str(getattr(args, "artist_technique_extra", "") or ""),
    )
    technique_hint_str = prompt_lexicon.artist_technique_bundle(**_technique_cfg)
    human_hint_str = prompt_lexicon.humanize_prompt_bundle(
        humanize_profile=str(_human_cfg.get("humanize_profile", "none") or "none"),
        imperfection_level=str(_human_cfg.get("imperfection_level", "none") or "none"),
        materiality_mode=str(_human_cfg.get("materiality_mode", "none") or "none"),
        asymmetry_level=str(_human_cfg.get("asymmetry_level", "none") or "none"),
    )
    panel_hint_str = prompt_lexicon.merge_prompt_fragments(panel_hint_str, artist_hint_str)
    panel_hint_str = prompt_lexicon.merge_prompt_fragments(panel_hint_str, medium_hint_str)
    panel_hint_str = prompt_lexicon.merge_prompt_fragments(panel_hint_str, color_render_hint_str)
    panel_hint_str = prompt_lexicon.merge_prompt_fragments(panel_hint_str, technique_hint_str)
    panel_hint_str = prompt_lexicon.merge_prompt_fragments(panel_hint_str, human_hint_str)
    _vm_early = str(getattr(args, "visual_memory", "") or "").strip()
    _vm_style_peek = book_style_authenticity.peek_visual_memory_book_style(Path(_vm_early)) if _vm_early else ""
    _book_auth_bundle = book_style_authenticity.resolve_authenticity_bundle(
        level=str(getattr(args, "book_authenticity", "none") or "none"),
        medium=str(getattr(args, "book_authenticity_medium", "auto") or "auto"),
        book_type=str(args.book_type),
        lexicon_style=str(getattr(args, "lexicon_style", "none") or "none"),
        visual_memory_book_style=_vm_style_peek,
    )
    if _book_auth_bundle.get("positive"):
        panel_hint_str = prompt_lexicon.merge_prompt_fragments(panel_hint_str, _book_auth_bundle["positive"])
    _book_auth_neg = str(_book_auth_bundle.get("negative") or "").strip()
    _book_auth_effective_medium = str(_book_auth_bundle.get("effective_medium") or "").strip()
    _primary_style_bm = _vm_style_peek or book_style_fusion.primary_style_from_book_type(str(args.book_type))
    _style_fusion_cli = book_style_fusion.fusion_from_cli(
        preset=str(getattr(args, "style_fusion_preset", "none") or "none"),
        secondary=str(getattr(args, "style_secondary", "") or ""),
        primary_book_style=_primary_style_bm,
    )
    _user_style_cli = str(getattr(args, "user_style_fragment", "") or "").strip()
    _oc_name = str(getattr(args, "oc_name", "") or "")
    _oc_archetype = str(getattr(args, "oc_archetype", "none") or "none")
    _oc_traits = str(getattr(args, "oc_traits", "") or "")
    _oc_wardrobe = str(getattr(args, "oc_wardrobe", "") or "")
    _oc_silhouette = str(getattr(args, "oc_silhouette", "") or "")
    _oc_color_motifs = str(getattr(args, "oc_color_motifs", "") or "")
    _oc_expression_sheet = str(getattr(args, "oc_expression_sheet", "") or "")
    _auto_oc_negative = ""

    if bool(getattr(args, "auto_original_character", True)):
        try:
            from utils.prompt.auto_oc import infer_auto_original_character

            _seed_parts = [
                str(getattr(args, "cover_prompt", "") or ""),
                str(getattr(args, "page_prompt_template", "") or ""),
            ]
            _pf = str(getattr(args, "prompts_file", "") or "").strip()
            if _pf:
                try:
                    _specs = _load_prompts_with_expected_from_file(Path(_pf))
                    if _specs:
                        _seed_parts.append(str(_specs[0][0] or ""))
                except Exception:
                    pass
            _seed_prompt = ", ".join([x for x in _seed_parts if str(x).strip()])
            _auto_profile = infer_auto_original_character(
                _seed_prompt,
                seed=int(getattr(args, "seed", 0)) + int(getattr(args, "auto_oc_seed_offset", 0)),
                style_context=", ".join(
                    [
                        str(getattr(args, "lexicon_style", "") or ""),
                        str(getattr(args, "artist_pack", "") or ""),
                        str(getattr(args, "artist_style_profile", "") or ""),
                        str(getattr(args, "art_medium_pack", "") or ""),
                        str(getattr(args, "color_render_pack", "") or ""),
                    ]
                ),
            )
            _explicit_oc = any(
                bool(str(v).strip())
                for v in (
                    _oc_name,
                    _oc_traits,
                    _oc_wardrobe,
                    _oc_silhouette,
                    _oc_color_motifs,
                    _oc_expression_sheet,
                )
            ) or (_oc_archetype.lower().strip() != "none")
            if _auto_profile is not None and not _explicit_oc:
                _oc_name = str(getattr(_auto_profile, "name", "") or "")
                _oc_archetype = str(getattr(_auto_profile, "archetype", "none") or "none")
                _oc_traits = str(getattr(_auto_profile, "visual_traits", "") or "")
                _oc_wardrobe = str(getattr(_auto_profile, "wardrobe", "") or "")
                _oc_silhouette = str(getattr(_auto_profile, "silhouette", "") or "")
                _oc_color_motifs = str(getattr(_auto_profile, "color_motifs", "") or "")
                _oc_expression_sheet = str(getattr(_auto_profile, "expression_sheet", "") or "")
                _auto_oc_negative = str(getattr(_auto_profile, "negative_block", "") or "")
        except Exception:
            pass

    _oc_cfg = prompt_lexicon.resolve_oc_controls(
        oc_pack=str(_style_cfg.get("oc_pack", "none") or "none"),
        name=_oc_name,
        archetype=_oc_archetype,
        visual_traits=_oc_traits,
        wardrobe=_oc_wardrobe,
        silhouette=_oc_silhouette,
        color_motifs=_oc_color_motifs,
        expression_sheet=_oc_expression_sheet,
    )
    oc_block = prompt_lexicon.original_character_bundle(**_oc_cfg)
    narration_p = (getattr(args, "narration_prefix", "") or "").strip()

    consistency_spec: Dict[str, Any] = {}
    cj = str(getattr(args, "consistency_json", "") or "").strip()
    if cj:
        consistency_spec = dict(consistency_helpers.load_consistency_json(Path(cj)))
    consistency_helpers.overlay_cli_on_spec(consistency_spec, args)
    _safety_for_challenge = str(_style_cfg.get("safety_mode", "") or "")
    consistency_block = consistency_helpers.positive_block_from_mapping(
        consistency_spec,
        safety_mode=_safety_for_challenge,
    )
    _challenge_pack_cli = str(getattr(args, "book_challenge_pack", "none") or "none").strip().lower()
    _book_challenge_pos = prompt_lexicon.merge_prompt_fragments(
        book_challenging_content.challenge_pack_positive(
            _challenge_pack_cli,
            safety_mode=_safety_for_challenge,
        ),
        str(getattr(args, "book_challenge_extra", "") or "").strip(),
    )
    consistency_block = prompt_lexicon.merge_prompt_fragments(consistency_block, _book_challenge_pos)
    consistency_neg_level = consistency_helpers.negative_level_from_spec(
        consistency_spec, getattr(args, "consistency_negative", None)
    )
    consistency_neg_fragment = consistency_helpers.consistency_negative_addon(consistency_neg_level)
    _book_challenge_neg = book_challenging_content.challenge_pack_negative(_challenge_pack_cli)

    vm_path = str(getattr(args, "visual_memory", "") or "").strip()
    book_vm = None
    if vm_path:
        try:
            book_vm = book_visual_memory.load_visual_memory(Path(vm_path))
        except Exception as e:
            raise SystemExit(f"--visual-memory: failed to load {vm_path!r}: {e}") from e

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
            prompts.append(
                book_helpers.expand_page_prompt_template(
                    args.page_prompt_template,
                    page_index=i,
                    total_pages=int(args.pages),
                )
            )
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
            production_tier=str(getattr(args, "book_accuracy", "") or "").lower()
            in ("production", "production_vit", "production_fidelity"),
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
        if _book_auth_neg:
            base = f"{base}, {_book_auth_neg}".strip().strip(",")
        if _book_challenge_neg:
            base = f"{base}, {_book_challenge_neg}".strip().strip(",")
        oc_neg = str(getattr(args, "oc_negative", "") or "").strip()
        if _auto_oc_negative:
            base = f"{base}, {_auto_oc_negative}".strip().strip(",")
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
            pick_vit_ckpt=str(getattr(args, "pick_vit_ckpt", "") or ""),
            pick_vit_use_adherence=bool(getattr(args, "pick_vit_use_adherence", False)),
            pick_vit_ar_blocks=int(getattr(args, "pick_vit_ar_blocks", -1) or -1),
            pick_report_json=_resolve_pick_report_json_path(
                str(getattr(args, "pick_report_json", "") or ""),
                out_path,
            ),
            pick_auto_no_clip=bool(getattr(args, "pick_auto_no_clip", False)),
        )
        book_helpers.append_sample_py_beam_flags(
            cmd,
            beam_width=int(getattr(args, "beam_width", 0) or 0),
            beam_steps=int(getattr(args, "beam_steps", 0) or 0),
            beam_metric=str(getattr(args, "beam_metric", "") or ""),
            beam2_width=int(getattr(args, "beam2_width", 0) or 0),
            beam2_steps=int(getattr(args, "beam2_steps", 0) or 0),
            beam2_metric=str(getattr(args, "beam2_metric", "") or ""),
            beam2_at_frac=float(getattr(args, "beam2_at_frac", 0.65) or 0.65),
            beam2_noise=float(getattr(args, "beam2_noise", 0.03) or 0.03),
        )
        cmd.extend(_cfg_cmd_tail())
        cmd.extend(_nsfw_tail())
        cmd.extend(_photo_tail())
        book_helpers.extend_sample_py_adapter_control_cmd(cmd, args)
        book_helpers.extend_sample_py_sdx_enhance_cmd(cmd, args)
        book_helpers.extend_sample_py_adherence_quality_cmd(cmd, args)
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

        if bool(getattr(args, "book_dry_run", False)):
            line = " ".join(shlex.quote(str(x)) for x in cmd)
            print(f"DRY-RUN sample.py:\n  {line}", flush=True)
            sys.exit(0)
        _safe_run(cmd)
        _postprocess_output(out_path, page_seed if page_seed is not None else int(args.seed))

    # Cover generation
    if args.cover_prompt:
        cover_out = cover_dir / "cover.png"
        vm_cover = (
            book_vm.prompt_fragment_for_cover(safety_mode=_safety_for_challenge)
            if book_vm is not None
            else ""
        )
        cover_composed = book_helpers.compose_book_page_prompt(
            user_prompt=args.cover_prompt,
            narration_prefix=narration_p,
            consistency_block=prompt_lexicon.merge_prompt_fragments(consistency_block, oc_block, vm_cover),
            style_fusion_block=_style_fusion_cli,
            user_style_fragment=_user_style_cli,
            panel_hint=panel_hint_str,
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
        vm_page = (
            book_vm.prompt_fragment_for_page(i, safety_mode=_safety_for_challenge)
            if book_vm is not None
            else ""
        )
        composed_prompt = book_helpers.compose_book_page_prompt(
            user_prompt=page_prompt,
            narration_prefix=narration_p,
            consistency_block=prompt_lexicon.merge_prompt_fragments(consistency_block, oc_block, vm_page),
            style_fusion_block=_style_fusion_cli,
            user_style_fragment=_user_style_cli,
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

        page_seed = book_helpers.derive_book_page_seed(int(args.seed), i)

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
            "color_render_pack": getattr(args, "color_render_pack", ""),
            "color_theory_mode": str(_color_render_cfg.get("color_theory_mode", "none") or "none"),
            "gradient_blend_mode": str(_color_render_cfg.get("gradient_blend_mode", "none") or "none"),
            "shading_technique": str(_color_render_cfg.get("shading_technique", "none") or "none"),
            "render_pipeline": str(_color_render_cfg.get("render_pipeline", "none") or "none"),
            "color_render_extra": str(_color_render_cfg.get("color_render_extra", "") or ""),
            "artist_technique_pack": str(getattr(args, "artist_technique_pack", "none") or "none"),
            "linework_technique": str(_technique_cfg.get("linework_technique", "none") or "none"),
            "rendering_technique": str(_technique_cfg.get("rendering_technique", "none") or "none"),
            "shading_technique_plan": str(_technique_cfg.get("shading_technique_plan", "none") or "none"),
            "material_technique": str(_technique_cfg.get("material_technique", "none") or "none"),
            "composition_technique": str(_technique_cfg.get("composition_technique", "none") or "none"),
            "artist_technique_extra": str(_technique_cfg.get("artist_technique_extra", "") or ""),
            "photo_realism_pack": str(getattr(args, "photo_realism_pack", "none") or "none"),
            "photo_color_grade": str(getattr(args, "photo_color_grade", "none") or "none"),
            "photo_lighting_technique": str(getattr(args, "photo_lighting_technique", "none") or "none"),
            "photo_filter": str(getattr(args, "photo_filter", "none") or "none"),
            "photo_grain_style": str(getattr(args, "photo_grain_style", "none") or "none"),
            "photo_realism_strength": float(getattr(args, "photo_realism_strength", 1.0) or 1.0),
            "auto_photo_realism": bool(getattr(args, "auto_photo_realism", True)),
            "photo_postprocess": bool(getattr(args, "photo_postprocess", True)),
            "photo_post_strength": float(getattr(args, "photo_post_strength", 0.6) or 0.6),
            "realism_autopilot": bool(getattr(args, "realism_autopilot", True)),
            "artist_pack": getattr(args, "artist_pack", ""),
            "artist_style_profile": str(_artist_cfg.get("artist_style_profile", "none") or "none"),
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
            "book_authenticity": str(getattr(args, "book_authenticity", "none") or "none"),
            "book_authenticity_medium": str(getattr(args, "book_authenticity_medium", "auto") or "auto"),
            "book_authenticity_effective_medium": _book_auth_effective_medium,
            "book_challenge_pack": str(getattr(args, "book_challenge_pack", "none") or "none"),
            "book_challenge_extra": str(getattr(args, "book_challenge_extra", "") or "").strip() or None,
            "sample_candidates": int(settings.sample_candidates),
            "pick_best": str(settings.pick_best or ""),
            "pick_vit_ckpt": str(getattr(args, "pick_vit_ckpt", "") or "").strip() or None,
            "pick_vit_use_adherence": bool(getattr(args, "pick_vit_use_adherence", False)),
            "pick_vit_ar_blocks": int(getattr(args, "pick_vit_ar_blocks", -1) or -1),
            "pick_vit_ar_from_ckpt": bool(getattr(args, "pick_vit_ar_from_ckpt", False)),
            "pick_report_json": str(getattr(args, "pick_report_json", "") or "").strip() or None,
            "pick_auto_no_clip": bool(getattr(args, "pick_auto_no_clip", False)),
            "beam_width": int(getattr(args, "beam_width", 0) or 0),
            "beam_steps": int(getattr(args, "beam_steps", 0) or 0),
            "beam_metric": str(getattr(args, "beam_metric", "") or "").strip() or None,
            "beam2_width": int(getattr(args, "beam2_width", 0) or 0),
            "beam2_steps": int(getattr(args, "beam2_steps", 0) or 0),
            "beam2_metric": str(getattr(args, "beam2_metric", "") or "").strip() or None,
            "beam2_at_frac": float(getattr(args, "beam2_at_frac", 0.65) or 0.65),
            "beam2_noise": float(getattr(args, "beam2_noise", 0.03) or 0.03),
            "sample_style_prompt": str(getattr(args, "style", "") or "").strip() or None,
            "sample_style_strength": float(getattr(args, "style_strength", 0.7) or 0.7),
            "sample_tags_file": str(getattr(args, "tags_file", "") or "").strip() or None,
            "lora_specs": list(getattr(args, "lora", []) or []),
            "control_stack": list(getattr(args, "control", []) or []),
            "control_image": str(getattr(args, "control_image", "") or "").strip() or None,
            "holy_grail": bool(getattr(args, "holy_grail", False)),
            "reference_image": str(getattr(args, "reference_image", "") or "").strip() or None,
            "reference_adapter_pt": str(getattr(args, "reference_adapter_pt", "") or "").strip() or None,
            "book_preflight": str(getattr(args, "book_preflight", "warn") or "warn"),
            "book_dry_run": bool(getattr(args, "book_dry_run", False)),
            "sdx_enhance": {
                "hires_fix": bool(getattr(args, "hires_fix", False)),
                "finishing_preset": str(getattr(args, "finishing_preset", "none") or "none"),
                "flow_matching_sample": bool(getattr(args, "flow_matching_sample", False)),
                "spectral_coherence_latent": float(getattr(args, "spectral_coherence_latent", 0.0) or 0.0),
                "domain_prior_latent": float(getattr(args, "domain_prior_latent", 0.0) or 0.0),
                "face_enhance": bool(getattr(args, "face_enhance", False)),
                "no_refine": bool(getattr(args, "no_refine", False)),
            },
            "adherence_quality": {
                "quality_pack": str(getattr(args, "quality_pack", "none") or "none"),
                "adherence_pack": str(getattr(args, "adherence_pack", "none") or "none"),
                "clip_guard_threshold": float(getattr(args, "clip_guard_threshold", 0.0) or 0.0),
                "clip_monitor_every": int(getattr(args, "clip_monitor_every", 0) or 0),
                "volatile_cfg_boost": float(getattr(args, "volatile_cfg_boost", 0.0) or 0.0),
                "sag_scale": float(getattr(args, "sag_scale", 0.0) or 0.0),
                "dual_stage_layout": bool(getattr(args, "dual_stage_layout", False)),
                "hard_style": str(getattr(args, "hard_style", "") or "").strip() or None,
            },
            "book_model_stack": book_model_readiness.book_model_stack_snapshot(
                args,
                dit_ar_blocks=_dit_ar_nb,
                vit_cfg=book_model_readiness.peek_vit_config_for_args(args),
            ),
            "user_style_fragment": _user_style_cli or None,
            "style_fusion_preset": str(getattr(args, "style_fusion_preset", "none") or "none"),
            "style_secondary": str(getattr(args, "style_secondary", "") or "").strip() or None,
            "oc_pack": getattr(args, "oc_pack", ""),
            "auto_original_character": bool(getattr(args, "auto_original_character", True)),
            "auto_oc_seed_offset": int(getattr(args, "auto_oc_seed_offset", 0) or 0),
            "panel_layout": getattr(args, "panel_layout", ""),
            "narration_prefix": narration_p,
            "consistency_block": consistency_block,
            "original_character_block": oc_block,
            "consistency_negative_level": consistency_neg_level,
            "consistency_json": cj or None,
            "visual_memory": vm_path or None,
            "visual_memory_entity_ids": book_vm.entity_ids() if book_vm is not None else [],
            "safety_mode": str(_style_cfg.get("safety_mode", "") or ""),
            "nsfw_pack": str(_style_cfg.get("nsfw_pack", "") or ""),
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

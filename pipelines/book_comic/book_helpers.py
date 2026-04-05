"""
Shared helpers for book/comic/manga generation: prompt quality, sample.py flags,
and optional post-process (sharpen / naturalize) using repo utilities.

Used by ``pipelines/book_comic/scripts/generate_book.py``. Run from **repo root**
so ``data``, ``utils`` imports resolve.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, List, Optional, Sequence


@dataclass
class BookAccuracyPreset:
    """Resolved sampling + post settings for a ``--book-accuracy`` tier."""

    sample_candidates: int
    pick_best: str
    boost_quality: bool
    subject_first: bool
    save_prompt_sidecar: bool
    post_sharpen: float
    post_naturalize: bool
    post_grain: float
    post_micro_contrast: float
    prepend_quality_if_short: bool
    shortcomings_mitigation: str
    shortcomings_2d: bool
    art_guidance_mode: str
    art_guidance_photography: bool
    anatomy_guidance: str
    style_guidance_mode: str
    style_guidance_artists: bool


def preset_for_book_accuracy(name: str) -> BookAccuracyPreset:
    """
    Map ``--book-accuracy`` to defaults. ``none`` = minimal extra processing.
    """
    n = (name or "none").lower().strip()
    if n == "fast":
        return BookAccuracyPreset(
            sample_candidates=1,
            pick_best="none",
            boost_quality=False,
            subject_first=False,
            save_prompt_sidecar=False,
            post_sharpen=0.0,
            post_naturalize=False,
            post_grain=0.0,
            post_micro_contrast=1.0,
            prepend_quality_if_short=False,
            shortcomings_mitigation="none",
            shortcomings_2d=False,
            art_guidance_mode="none",
            art_guidance_photography=True,
            anatomy_guidance="none",
            style_guidance_mode="none",
            style_guidance_artists=True,
        )
    if n == "balanced":
        return BookAccuracyPreset(
            sample_candidates=2,
            pick_best="combo",
            boost_quality=True,
            subject_first=True,
            save_prompt_sidecar=True,
            post_sharpen=0.25,
            post_naturalize=True,
            post_grain=0.012,
            post_micro_contrast=1.02,
            prepend_quality_if_short=True,
            shortcomings_mitigation="auto",
            shortcomings_2d=True,
            art_guidance_mode="auto",
            art_guidance_photography=True,
            anatomy_guidance="lite",
            style_guidance_mode="auto",
            style_guidance_artists=True,
        )
    if n == "maximum":
        return BookAccuracyPreset(
            sample_candidates=4,
            pick_best="combo",
            boost_quality=True,
            subject_first=True,
            save_prompt_sidecar=True,
            post_sharpen=0.4,
            post_naturalize=True,
            post_grain=0.018,
            post_micro_contrast=1.03,
            prepend_quality_if_short=True,
            shortcomings_mitigation="auto",
            shortcomings_2d=True,
            art_guidance_mode="auto",
            art_guidance_photography=True,
            anatomy_guidance="lite",
            style_guidance_mode="auto",
            style_guidance_artists=True,
        )
    if n == "production":
        # Heaviest test-time stack: more candidates, same pick metric, slightly stronger print polish.
        return BookAccuracyPreset(
            sample_candidates=6,
            pick_best="combo",
            boost_quality=True,
            subject_first=True,
            save_prompt_sidecar=True,
            post_sharpen=0.5,
            post_naturalize=True,
            post_grain=0.022,
            post_micro_contrast=1.04,
            prepend_quality_if_short=True,
            shortcomings_mitigation="all",
            shortcomings_2d=True,
            art_guidance_mode="all",
            art_guidance_photography=True,
            anatomy_guidance="strong",
            style_guidance_mode="all",
            style_guidance_artists=True,
        )
    # none
    return BookAccuracyPreset(
        sample_candidates=1,
        pick_best="none",
        boost_quality=False,
        subject_first=False,
        save_prompt_sidecar=False,
        post_sharpen=0.0,
        post_naturalize=False,
        post_grain=0.0,
        post_micro_contrast=1.0,
        prepend_quality_if_short=False,
        shortcomings_mitigation="none",
        shortcomings_2d=False,
        art_guidance_mode="none",
        art_guidance_photography=True,
        anatomy_guidance="none",
        style_guidance_mode="none",
        style_guidance_artists=True,
    )


def resolve_book_sample_settings(args: Any) -> BookAccuracyPreset:
    """
    Start from ``--book-accuracy`` preset, then apply CLI overrides (non-zero
    ``--sample-candidates``, explicit ``--pick-best``, post-process floats, etc.).
    """
    base = preset_for_book_accuracy(getattr(args, "book_accuracy", "none"))
    sc = int(getattr(args, "sample_candidates", 0) or 0)
    if sc >= 1:
        base = replace(base, sample_candidates=sc)

    pb = str(getattr(args, "pick_best", "auto") or "auto").strip().lower()
    if pb not in ("auto", ""):
        base = replace(base, pick_best=pb)

    if getattr(args, "no_boost_quality", False):
        base = replace(base, boost_quality=False)
    elif getattr(args, "boost_quality", False):
        base = replace(base, boost_quality=True)

    if getattr(args, "subject_first", False):
        base = replace(base, subject_first=True)
    if getattr(args, "no_subject_first", False):
        base = replace(base, subject_first=False)

    if getattr(args, "save_prompt", False):
        base = replace(base, save_prompt_sidecar=True)

    ps = getattr(args, "post_sharpen", -1.0)
    try:
        psf = float(ps)
    except (TypeError, ValueError):
        psf = -1.0
    if psf >= 0:
        base = replace(base, post_sharpen=psf)

    if getattr(args, "post_naturalize", False):
        base = replace(base, post_naturalize=True)
    if getattr(args, "no_post_naturalize", False):
        base = replace(base, post_naturalize=False)

    pg = getattr(args, "post_grain", -1.0)
    try:
        pgf = float(pg)
    except (TypeError, ValueError):
        pgf = -1.0
    if pgf >= 0:
        base = replace(base, post_grain=pgf)

    pmc = getattr(args, "post_micro_contrast", -1.0)
    try:
        pmcf = float(pmc)
    except (TypeError, ValueError):
        pmcf = -1.0
    if pmcf > 0:
        base = replace(base, post_micro_contrast=pmcf)

    if getattr(args, "prepend_quality_if_short", False):
        base = replace(base, prepend_quality_if_short=True)
    if getattr(args, "no_prepend_quality_if_short", False):
        base = replace(base, prepend_quality_if_short=False)

    sm_raw = getattr(args, "shortcomings_mitigation", "")
    sm = str(sm_raw or "").strip().lower()
    if sm in ("none", "auto", "all") and sm:
        base = replace(base, shortcomings_mitigation=sm)
    if getattr(args, "shortcomings_2d", False):
        base = replace(base, shortcomings_2d=True)
    if getattr(args, "no_shortcomings_2d", False):
        base = replace(base, shortcomings_2d=False)
    ag = str(getattr(args, "art_guidance_mode", "") or "").strip().lower()
    if ag in ("none", "auto", "all") and ag:
        base = replace(base, art_guidance_mode=ag)
    if getattr(args, "no_art_guidance_photography", False):
        base = replace(base, art_guidance_photography=False)
    if getattr(args, "art_guidance_photography", False):
        base = replace(base, art_guidance_photography=True)
    anat = str(getattr(args, "anatomy_guidance", "") or "").strip().lower()
    if anat in ("none", "lite", "strong") and anat:
        base = replace(base, anatomy_guidance=anat)
    sgm = str(getattr(args, "style_guidance_mode", "") or "").strip().lower()
    if sgm in ("none", "auto", "all") and sgm:
        base = replace(base, style_guidance_mode=sgm)
    if getattr(args, "no_style_guidance_artists", False):
        base = replace(base, style_guidance_artists=False)
    if getattr(args, "style_guidance_artists", False):
        base = replace(base, style_guidance_artists=True)

    # Multiple candidates need a pick metric.
    if base.sample_candidates > 1 and base.pick_best in ("none", ""):
        base = replace(base, pick_best="combo")

    return base


def enhance_prompt_for_page(
    prompt_final: str,
    *,
    settings: BookAccuracyPreset,
) -> str:
    """Apply ``prepend_quality_if_short`` from ``data.caption_utils`` when enabled."""
    out = (prompt_final or "").strip()
    if not out:
        return out
    if not settings.prepend_quality_if_short:
        return out
    try:
        from data.caption_utils import prepend_quality_if_short as _pq

        return _pq(out, min_tag_count=4)
    except Exception:
        return out


def expected_text_for_pick(expected_texts: Sequence[str]) -> str:
    """First non-empty expected string for ``--pick-best ocr|combo`` scoring."""
    for t in expected_texts or []:
        s = str(t).strip()
        if s:
            return s
    return ""


def append_sample_py_quality_flags(
    cmd: List[str],
    settings: BookAccuracyPreset,
    *,
    pick_expected_text: str,
    pick_expected_count: int = 0,
    pick_expected_count_target: str = "auto",
    pick_expected_count_object: str = "",
) -> None:
    """Mutate *cmd* with flags aligned with ``sample.py`` (test-time pick, CFG helpers, etc.)."""
    cmd.extend(["--num", str(max(1, settings.sample_candidates))])
    cmd.extend(["--pick-best", settings.pick_best or "none"])
    if settings.boost_quality:
        cmd.append("--boost-quality")
    if settings.subject_first:
        cmd.append("--subject-first")
    if settings.save_prompt_sidecar:
        cmd.append("--save-prompt")
    sm = str(getattr(settings, "shortcomings_mitigation", "none") or "none").lower()
    if sm in ("auto", "all"):
        cmd.extend(["--shortcomings-mitigation", sm])
    if bool(getattr(settings, "shortcomings_2d", False)):
        cmd.append("--shortcomings-2d")
    ag = str(getattr(settings, "art_guidance_mode", "none") or "none").lower()
    if ag in ("auto", "all"):
        cmd.extend(["--art-guidance-mode", ag])
    if not bool(getattr(settings, "art_guidance_photography", True)):
        cmd.append("--no-art-guidance-photography")
    anat = str(getattr(settings, "anatomy_guidance", "none") or "none").lower()
    if anat in ("lite", "strong"):
        cmd.extend(["--anatomy-guidance", anat])
    sgm = str(getattr(settings, "style_guidance_mode", "none") or "none").lower()
    if sgm in ("auto", "all"):
        cmd.extend(["--style-guidance-mode", sgm])
    if not bool(getattr(settings, "style_guidance_artists", True)):
        cmd.append("--no-style-guidance-artists")

    metric = (settings.pick_best or "").lower()
    if metric in ("ocr", "combo") and pick_expected_text.strip():
        cmd.extend(["--expected-text", pick_expected_text.strip()])
    if metric == "combo_count":
        if int(pick_expected_count or 0) > 0:
            cmd.extend(["--expected-count", str(int(pick_expected_count))])
        tgt = str(pick_expected_count_target or "auto").strip().lower()
        if tgt in ("auto", "people", "objects"):
            cmd.extend(["--expected-count-target", tgt])
        obj = str(pick_expected_count_object or "").strip()
        if obj:
            cmd.extend(["--expected-count-object", obj])


def append_optional_sample_flags(
    cmd: List[str],
    *,
    vae_tiling: bool,
    pick_clip_model: str,
    pick_save_all: bool,
    cfg_scale: float,
    cfg_rescale: float,
    dynamic_threshold_percentile: float,
    resize_mode: str,
    resize_saliency_face_bias: float,
    grid: bool,
) -> None:
    if vae_tiling:
        cmd.append("--vae-tiling")
    if pick_clip_model:
        cmd.extend(["--pick-clip-model", pick_clip_model])
    if pick_save_all:
        cmd.append("--pick-save-all")
    if cfg_scale and cfg_scale > 0:
        cmd.extend(["--cfg-scale", str(cfg_scale)])
    if cfg_rescale and cfg_rescale > 0:
        cmd.extend(["--cfg-rescale", str(cfg_rescale)])
    if dynamic_threshold_percentile and dynamic_threshold_percentile > 0:
        cmd.extend(
            [
                "--dynamic-threshold-percentile",
                str(dynamic_threshold_percentile),
                "--dynamic-threshold-type",
                "percentile",
            ]
        )
    rm = str(resize_mode or "stretch").strip().lower()
    if rm in ("stretch", "center_crop", "saliency_crop"):
        cmd.extend(["--resize-mode", rm])
    if float(resize_saliency_face_bias or 0.0) > 0:
        cmd.extend(["--resize-saliency-face-bias", str(float(resize_saliency_face_bias))])
    if grid:
        cmd.append("--grid")


def apply_postprocess_to_image_file(
    path: Path,
    *,
    sharpen_amount: float,
    naturalize: bool,
    grain: float,
    micro_contrast: float,
    seed: Optional[int],
) -> None:
    """
    In-place RGB post-process using ``utils.quality`` (sharpen, naturalize).
    No-op if path missing or all effects off.
    """
    if not path.is_file():
        return
    if sharpen_amount <= 0 and not naturalize and abs(micro_contrast - 1.0) < 1e-6:
        return

    import numpy as np
    from PIL import Image
    from utils.quality import naturalize as _nat
    from utils.quality import sharpen as _sharp

    im = Image.open(path).convert("RGB")
    arr = np.asarray(im, dtype=np.uint8)

    if sharpen_amount > 0:
        arr = _sharp(arr, amount=float(sharpen_amount), radius=1.0)
    if naturalize or grain > 0 or abs(micro_contrast - 1.0) >= 1e-6:
        arr = _nat(
            arr,
            grain_amount=float(grain) if naturalize else 0.0,
            micro_contrast=float(micro_contrast),
            seed=seed,
        )
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB").save(path)


def build_extra_ocr_sample_flags(settings: BookAccuracyPreset) -> List[str]:
    """Extra ``sample.py`` args for OCR repair passes (single-image)."""
    out: List[str] = []
    if settings.boost_quality:
        out.append("--boost-quality")
    if settings.subject_first:
        out.append("--subject-first")
    sm = str(getattr(settings, "shortcomings_mitigation", "none") or "none").lower()
    if sm in ("auto", "all"):
        out.extend(["--shortcomings-mitigation", sm])
    if bool(getattr(settings, "shortcomings_2d", False)):
        out.append("--shortcomings-2d")
    ag = str(getattr(settings, "art_guidance_mode", "none") or "none").lower()
    if ag in ("auto", "all"):
        out.extend(["--art-guidance-mode", ag])
    if not bool(getattr(settings, "art_guidance_photography", True)):
        out.append("--no-art-guidance-photography")
    anat = str(getattr(settings, "anatomy_guidance", "none") or "none").lower()
    if anat in ("lite", "strong"):
        out.extend(["--anatomy-guidance", anat])
    sgm = str(getattr(settings, "style_guidance_mode", "none") or "none").lower()
    if sgm in ("auto", "all"):
        out.extend(["--style-guidance-mode", sgm])
    if not bool(getattr(settings, "style_guidance_artists", True)):
        out.append("--no-style-guidance-artists")
    return out


def compose_book_page_prompt(
    *,
    user_prompt: str,
    narration_prefix: str = "",
    consistency_block: str = "",
    panel_hint: str = "",
    rolling_context: str = "",
) -> str:
    """Merge narration, consistency cues, panel hint, rolling continuity, then the page line."""
    parts: List[str] = []
    for p in (narration_prefix, consistency_block, panel_hint, rolling_context, user_prompt):
        s = (p or "").strip()
        if s:
            parts.append(s)
    return ", ".join(parts)


def build_rolling_page_context(
    previous_page_prompts: Sequence[str],
    *,
    num_previous: int,
    max_chars: int = 500,
) -> str:
    """
    Short summary of the last *num_previous* page prompts for cross-page visual continuity.
    """
    n = int(num_previous)
    if n <= 0 or not previous_page_prompts:
        return ""
    take = [t.strip() for t in previous_page_prompts[-n:] if t and str(t).strip()]
    if not take:
        return ""
    prefix = "visual continuity from prior pages: "
    max_blob = max(0, int(max_chars) - len(prefix))
    if max_blob <= 0:
        return ""
    blob = " · ".join(take)
    if len(blob) > max_blob:
        if max_blob == 1:
            blob = blob[-1:]
        else:
            blob = "…" + blob[-(max_blob - 1) :]
    return prefix + blob

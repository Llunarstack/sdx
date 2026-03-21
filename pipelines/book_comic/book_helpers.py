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


def append_sample_py_quality_flags(cmd: List[str], settings: BookAccuracyPreset, *, pick_expected_text: str) -> None:
    """Mutate *cmd* with flags aligned with ``sample.py`` (test-time pick, CFG helpers, etc.)."""
    cmd.extend(["--num", str(max(1, settings.sample_candidates))])
    cmd.extend(["--pick-best", settings.pick_best or "none"])
    if settings.boost_quality:
        cmd.append("--boost-quality")
    if settings.subject_first:
        cmd.append("--subject-first")
    if settings.save_prompt_sidecar:
        cmd.append("--save-prompt")

    metric = (settings.pick_best or "").lower()
    if metric in ("ocr", "combo") and pick_expected_text.strip():
        cmd.extend(["--expected-text", pick_expected_text.strip()])


def append_optional_sample_flags(
    cmd: List[str],
    *,
    vae_tiling: bool,
    pick_clip_model: str,
    pick_save_all: bool,
    cfg_scale: float,
    cfg_rescale: float,
    dynamic_threshold_percentile: float,
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
    return out

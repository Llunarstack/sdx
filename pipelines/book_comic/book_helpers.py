"""
Shared helpers for book/comic/manga generation: prompt quality, sample.py flags,
and optional post-process (sharpen / naturalize) using repo utilities.

Wiring to the rest of the stack:

- **DiT / diffusion**: ``sample.py`` loads checkpoints via ``utils.checkpoint.checkpoint_loading``.
- **ViT quality pick**: ``--pick-vit-ckpt`` loads ``vit_quality`` checkpoints; metrics ``vit`` / ``combo_vit*`` / ``combo_count_vit``.
- **AR regime**: optional ``--pick-vit-ar-blocks`` (0/2/4) aligns the ViT scorer with DiT ``num_ar_blocks`` via
  ``utils.architecture.ar_block_conditioning`` (see ``docs/AR.md``).
- **Adapters / ControlNet**: ``extend_sample_py_adapter_control_cmd`` forwards ``sample.py`` flags for
  LoRA/DoRA/LyCORIS, stacked ``--control``, ``--control-image``, optional holy-grail scheduling, and
  CLIP reference / IP-Adapter weights.
- **SDX polish (parity with CLI inference)**: ``extend_sample_py_sdx_enhance_cmd`` forwards hires-fix,
  finishing preset, latent research (domain / spectral), flow-matching sampler, refinement gate, face
  enhance, post-reference blend, and PIL post knobs (sharpen / contrast / …) to match ``sample.py``.
- **Prompt / style fidelity**: ``extend_sample_py_adherence_quality_cmd`` forwards ``sample.py`` quality and
  adherence packs, CLIP guard + mid-loop CLIP monitor, volatile CFG, SAG, coarse-to-fine dual-stage layout,
  hard-style tags, and optional T5-cache / deterministic toggles.
- **Dual-model preflight**: ``pipelines.book_comic.book_model_readiness`` validates paths and DiT/ViT
  AR alignment (``generate_book.py --book-preflight``).

Run from **repo root** so ``data``, ``utils``, ``vit_quality`` imports resolve.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pipelines.book_comic.prompt_lexicon import merge_prompt_fragments

# Same stride as ``generate_book.py`` page loop (deterministic per-page seeds).
BOOK_PAGE_SEED_STRIDE: int = 9973

# ``sample.py`` / ``test_time_pick`` metrics that load ``--pick-vit-ckpt`` when scoring.
VIT_DEPENDENT_PICK_METRICS: frozenset[str] = frozenset(
    {"vit", "combo_vit", "combo_vit_hq", "combo_vit_realism", "combo_count_vit"},
)


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
    if n == "production_vit":
        # Like production, but prefer ViT-heavy pick when a checkpoint is supplied via CLI.
        return BookAccuracyPreset(
            sample_candidates=6,
            pick_best="combo_vit_hq",
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
    if n == "production_fidelity":
        # Maximum test-time budget for literal prompt + style adherence (pair --pick-vit-ckpt + fidelity flags).
        return BookAccuracyPreset(
            sample_candidates=8,
            pick_best="combo_vit_hq",
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


def book_accuracy_tier_names() -> Tuple[str, ...]:
    """Known ``--book-accuracy`` values (for docs, validators, and UIs)."""
    return ("none", "fast", "balanced", "maximum", "production", "production_vit", "production_fidelity")


def pick_metric_requires_vit_ckpt(metric: str) -> bool:
    """Whether ``sample.py`` needs ``--pick-vit-ckpt`` for this ``--pick-best`` mode."""
    m = str(metric or "").strip().lower()
    if m == "auto":
        return False
    return m in VIT_DEPENDENT_PICK_METRICS


def derive_book_page_seed(base_seed: int, page_index: int, *, stride: int = BOOK_PAGE_SEED_STRIDE) -> int:
    """Deterministic seed for page *page_index* (0-based), matching ``generate_book``."""
    return int(base_seed) + int(page_index) * int(stride)


def normalize_book_prompt_fragment(text: str) -> str:
    """
    Stable whitespace + light punctuation cleanup for prompts (does not rewrite semantics).

    Collapses internal runs of spaces/newlines; trims trailing/leading commas and spaces.
    """
    s = " ".join((text or "").split())
    return s.strip(" \t,")


def audit_book_run_flags(
    *,
    pick_best: str = "none",
    sample_candidates: int = 1,
    pick_vit_ckpt: str = "",
    beam_width: int = 0,
    book_challenge_pack: str = "none",
    safety_mode: str = "",
    clip_guard_threshold: float = 0.0,
    clip_monitor_every: int = 0,
    adherence_pack: str = "none",
    pick_vit_use_adherence: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Return ``(errors, warnings)`` for common book + ``sample.py`` flag mismatches.

    Errors are reserved for hard inconsistencies; today most issues are warnings so runs can proceed.
    """
    errors: List[str] = []
    warnings: List[str] = []
    pb = str(pick_best or "").strip().lower()
    pv = str(pick_vit_ckpt or "").strip()
    sc = max(1, int(sample_candidates or 1))
    bw = int(beam_width or 0)

    if pick_metric_requires_vit_ckpt(pb) and not pv:
        warnings.append(
            f"--pick-best {pb!r} expects --pick-vit-ckpt (ViT scores fall back to neutral 0.5 without it)."
        )
    if bw > 1 and sc != 1:
        warnings.append(
            f"--beam-width {bw} is intended with a single latent branch (--sample-candidates 1); "
            f"got sample_candidates={sc}."
        )
    ch = str(book_challenge_pack or "none").strip().lower()
    sm = str(safety_mode or "").strip().lower()
    if ch == "mature_coherence" and sm != "nsfw":
        warnings.append(
            "--book-challenge-pack mature_coherence is a no-op unless pipeline safety resolves to nsfw "
            f"(current safety_mode={sm!r})."
        )
    if ch == "max" and sm not in ("nsfw",) and sm:
        warnings.append(
            "--book-challenge-pack max includes mature-coherence cues only when safety_mode=nsfw "
            f"(current safety_mode={sm!r})."
        )
    try:
        cgt = float(clip_guard_threshold or 0.0)
    except (TypeError, ValueError):
        cgt = 0.0
    if cgt > 0.0:
        warnings.append(
            "--clip-guard-threshold > 0 enables decode + extra denoise passes (much slower; needs transformers)."
        )
    try:
        cme = int(clip_monitor_every or 0)
    except (TypeError, ValueError):
        cme = 0
    if cme > 0:
        warnings.append(
            f"--clip-monitor-every {cme} decodes x0 during sampling for CLIP alignment (very slow)."
        )
    adh = str(adherence_pack or "none").strip().lower()
    pbv = str(pick_best or "").strip().lower()
    if adh in ("standard", "strict") and pbv in VIT_DEPENDENT_PICK_METRICS and not pick_vit_use_adherence:
        warnings.append(
            f"--adherence-pack {adh!r} pairs well with --pick-vit-use-adherence when using {pbv!r} pick-best."
        )
    return errors, warnings


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
    pick_vit_ckpt: str = "",
    pick_vit_use_adherence: bool = False,
    pick_vit_ar_blocks: int = -1,
    pick_report_json: str = "",
    pick_auto_no_clip: bool = False,
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
    _ocr_metrics = frozenset(
        {"ocr", "combo", "combo_count", "combo_count_vit", "combo_vit_realism"},
    )
    if metric in _ocr_metrics and pick_expected_text.strip():
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

    pv = str(pick_vit_ckpt or "").strip()
    if pv:
        cmd.extend(["--pick-vit-ckpt", pv])
    if pick_vit_use_adherence:
        cmd.append("--pick-vit-use-adherence")
    try:
        ar_b = int(pick_vit_ar_blocks)
    except (TypeError, ValueError):
        ar_b = -1
    if ar_b >= 0:
        cmd.extend(["--pick-vit-ar-blocks", str(ar_b)])
    prp = str(pick_report_json or "").strip()
    if prp:
        cmd.extend(["--pick-report-json", prp])
    if pick_auto_no_clip:
        cmd.append("--pick-auto-no-clip")


def append_sample_py_beam_flags(
    cmd: List[str],
    *,
    beam_width: int = 0,
    beam_steps: int = 0,
    beam_metric: str = "",
    beam2_width: int = 0,
    beam2_steps: int = 0,
    beam2_metric: str = "",
    beam2_at_frac: float = 0.65,
    beam2_noise: float = 0.03,
) -> None:
    """Forward DiT beam / micro-beam search flags to ``sample.py`` (optional)."""
    bw = int(beam_width or 0)
    if bw > 1:
        cmd.extend(["--beam-width", str(bw)])
        bs = int(beam_steps or 0)
        if bs > 0:
            cmd.extend(["--beam-steps", str(bs)])
        bm = str(beam_metric or "").strip()
        if bm:
            cmd.extend(["--beam-metric", bm])
    b2 = int(beam2_width or 0)
    if b2 > 1:
        cmd.extend(["--beam2-width", str(b2)])
        s2 = int(beam2_steps or 0)
        if s2 > 0:
            cmd.extend(["--beam2-steps", str(s2)])
        m2 = str(beam2_metric or "").strip()
        if m2:
            cmd.extend(["--beam2-metric", m2])
        try:
            af = float(beam2_at_frac)
        except (TypeError, ValueError):
            af = 0.65
        cmd.extend(["--beam2-at-frac", str(af)])
        try:
            bn = float(beam2_noise)
        except (TypeError, ValueError):
            bn = 0.03
        cmd.extend(["--beam2-noise", str(bn)])


def extend_sample_py_adapter_control_cmd(cmd: List[str], args: Any) -> None:
    """
    Append ``sample.py`` flags for style, ControlNet (single or stacked), LoRA/DoRA/LyCORIS,
    optional holy-grail guidance, tags, and reference / IP-Adapter conditioning.

    *args* is any namespace with the same attribute names as ``generate_book`` / ``sample.py``
    (hyphenated CLI options become underscores on the object).
    """
    st = str(getattr(args, "style", "") or "").strip()
    if st:
        cmd.extend(["--style", st])
        cmd.extend(["--style-strength", str(float(getattr(args, "style_strength", 0.7) or 0.7))])
    if bool(getattr(args, "auto_style_from_prompt", False)):
        cmd.append("--auto-style-from-prompt")

    tags = str(getattr(args, "tags", "") or "").strip()
    if tags:
        cmd.extend(["--tags", tags])
    tf = str(getattr(args, "tags_file", "") or "").strip()
    if tf:
        cmd.extend(["--tags-file", tf])

    cimg = str(getattr(args, "control_image", "") or "").strip()
    if cimg:
        cmd.extend(["--control-image", cimg])
        cmd.extend(["--control-type", str(getattr(args, "control_type", "auto") or "auto")])
        cmd.extend(["--control-scale", str(float(getattr(args, "control_scale", 0.85) or 0.85))])
        cmd.extend(["--control-guidance-start", str(float(getattr(args, "control_guidance_start", 0.0) or 0.0))])
        cmd.extend(["--control-guidance-end", str(float(getattr(args, "control_guidance_end", 1.0) or 1.0))])
        cmd.extend(["--control-guidance-decay", str(float(getattr(args, "control_guidance_decay", 1.0) or 1.0))])

    ctrl = getattr(args, "control", None) or []
    if ctrl:
        cmd.extend(["--control"] + [str(x) for x in ctrl])

    if bool(getattr(args, "holy_grail", False)):
        cmd.append("--holy-grail")
        cmd.extend(["--holy-grail-cfg-early-ratio", str(float(getattr(args, "holy_grail_cfg_early_ratio", 0.72) or 0.72))])
        cmd.extend(["--holy-grail-cfg-late-ratio", str(float(getattr(args, "holy_grail_cfg_late_ratio", 1.0) or 1.0))])
        cmd.extend(["--holy-grail-control-mult", str(float(getattr(args, "holy_grail_control_mult", 1.0) or 1.0))])
        cmd.extend(["--holy-grail-adapter-mult", str(float(getattr(args, "holy_grail_adapter_mult", 1.0) or 1.0))])
        if bool(getattr(args, "holy_grail_no_frontload_control", False)):
            cmd.append("--holy-grail-no-frontload-control")
        cmd.extend(
            ["--holy-grail-late-adapter-boost", str(float(getattr(args, "holy_grail_late_adapter_boost", 1.15) or 1.15))]
        )
        cmd.extend(["--holy-grail-cads-strength", str(float(getattr(args, "holy_grail_cads_strength", 0.0) or 0.0))])
        cmd.extend(
            ["--holy-grail-cads-min-strength", str(float(getattr(args, "holy_grail_cads_min_strength", 0.0) or 0.0))]
        )
        cmd.extend(["--holy-grail-cads-power", str(float(getattr(args, "holy_grail_cads_power", 1.0) or 1.0))])
        cmd.extend(["--holy-grail-unsharp-sigma", str(float(getattr(args, "holy_grail_unsharp_sigma", 0.0) or 0.0))])
        cmd.extend(["--holy-grail-unsharp-amount", str(float(getattr(args, "holy_grail_unsharp_amount", 0.0) or 0.0))])
        cmd.extend(["--holy-grail-clamp-quantile", str(float(getattr(args, "holy_grail_clamp_quantile", 0.0) or 0.0))])
        cmd.extend(["--holy-grail-clamp-floor", str(float(getattr(args, "holy_grail_clamp_floor", 1.0) or 1.0))])

    lora_specs = getattr(args, "lora", None) or []
    if lora_specs:
        cmd.extend(["--lora"] + [str(x) for x in lora_specs])
        if bool(getattr(args, "no_lora_normalize_scales", False)):
            cmd.append("--no-lora-normalize-scales")
        cmd.extend(["--lora-max-total-scale", str(float(getattr(args, "lora_max_total_scale", 1.5) or 1.5))])
        ldr = str(getattr(args, "lora_default_role", "style") or "style").strip().lower()
        if ldr != "style":
            cmd.extend(["--lora-default-role", ldr])
        lrb = str(getattr(args, "lora_role_budgets", "") or "").strip()
        if lrb:
            cmd.extend(["--lora-role-budgets", lrb])
        lsp = str(getattr(args, "lora_stage_policy", "auto") or "auto").strip().lower()
        if lsp and lsp != "auto":
            cmd.extend(["--lora-stage-policy", lsp])
        lrsw = str(getattr(args, "lora_role_stage_weights", "") or "").strip()
        if lrsw:
            cmd.extend(["--lora-role-stage-weights", lrsw])
        ll = str(getattr(args, "lora_layers", "all") or "all").strip().lower()
        if ll and ll != "all":
            cmd.extend(["--lora-layers", ll])
        lt = str(getattr(args, "lora_trigger", "") or "").strip()
        if lt:
            cmd.extend(["--lora-trigger", lt])
        lsc = str(getattr(args, "lora_scaffold", "none") or "none").strip().lower()
        if lsc and lsc != "none":
            cmd.extend(["--lora-scaffold", lsc])
        if bool(getattr(args, "lora_scaffold_auto", False)):
            cmd.append("--lora-scaffold-auto")

    rif = str(getattr(args, "reference_image", "") or "").strip()
    if rif:
        cmd.extend(["--reference-image", rif])
        cmd.extend(["--reference-strength", str(float(getattr(args, "reference_strength", 1.0) or 1.0))])
        cmd.extend(["--reference-tokens", str(int(getattr(args, "reference_tokens", 4) or 4))])
        cmd.extend(
            [
                "--reference-clip-model",
                str(getattr(args, "reference_clip_model", "openai/clip-vit-large-patch14") or "openai/clip-vit-large-patch14"),
            ]
        )
    rap = str(getattr(args, "reference_adapter_pt", "") or "").strip()
    if rap:
        cmd.extend(["--reference-adapter-pt", rap])


def extend_sample_py_sdx_enhance_cmd(cmd: List[str], args: Any) -> None:
    """
    Forward ``sample.py`` quality / finish flags (hires, latent priors, post-process, refine, faces).

    Uses the same attribute names as ``sample.py``'s ``argparse`` namespace where possible.
    """
    if bool(getattr(args, "flow_matching_sample", False)):
        cmd.append("--flow-matching-sample")
    fs = str(getattr(args, "flow_solver", "euler") or "euler").strip().lower()
    if bool(getattr(args, "flow_matching_sample", False)) and fs in ("euler", "heun"):
        cmd.extend(["--flow-solver", fs])

    dp = float(getattr(args, "domain_prior_latent", 0.0) or 0.0)
    if dp > 0.0:
        cmd.extend(["--domain-prior-latent", str(dp)])

    scl = float(getattr(args, "spectral_coherence_latent", 0.0) or 0.0)
    if scl > 0.0:
        cmd.extend(["--spectral-coherence-latent", str(scl)])
        cmd.extend(
            ["--spectral-coherence-cutoff", str(float(getattr(args, "spectral_coherence_cutoff", 0.15) or 0.15))]
        )

    if bool(getattr(args, "hires_fix", False)):
        cmd.append("--hires-fix")
        cmd.extend(["--hires-scale", str(float(getattr(args, "hires_scale", 1.5) or 1.5))])
        cmd.extend(["--hires-steps", str(int(getattr(args, "hires_steps", 15) or 15))])
        cmd.extend(["--hires-strength", str(float(getattr(args, "hires_strength", 0.35) or 0.35))])
        cmd.extend(["--hires-cfg-scale", str(float(getattr(args, "hires_cfg_scale", -1.0) or -1.0))])

    fp = str(getattr(args, "finishing_preset", "none") or "none").strip().lower()
    if fp and fp != "none":
        cmd.extend(["--finishing-preset", fp])

    try:
        sh = float(getattr(args, "sharpen", 0.0) or 0.0)
    except (TypeError, ValueError):
        sh = 0.0
    if sh > 0.0:
        cmd.extend(["--sharpen", str(sh)])

    try:
        co = float(getattr(args, "contrast", 1.0) or 1.0)
    except (TypeError, ValueError):
        co = 1.0
    if abs(co - 1.0) > 1e-6:
        cmd.extend(["--contrast", str(co)])

    try:
        sa = float(getattr(args, "saturation", 1.0) or 1.0)
    except (TypeError, ValueError):
        sa = 1.0
    if abs(sa - 1.0) > 1e-6:
        cmd.extend(["--saturation", str(sa)])

    try:
        cl = float(getattr(args, "clarity", 0.0) or 0.0)
    except (TypeError, ValueError):
        cl = 0.0
    if cl > 0.0:
        cmd.extend(["--clarity", str(cl)])

    try:
        tp = float(getattr(args, "tone_punch", 0.0) or 0.0)
    except (TypeError, ValueError):
        tp = 0.0
    if tp > 0.0:
        cmd.extend(["--tone-punch", str(tp)])

    try:
        cs = float(getattr(args, "chroma_smooth", 0.0) or 0.0)
    except (TypeError, ValueError):
        cs = 0.0
    if cs > 0.0:
        cmd.extend(["--chroma-smooth", str(cs)])

    try:
        pol = float(getattr(args, "polish", 0.0) or 0.0)
    except (TypeError, ValueError):
        pol = 0.0
    if pol > 0.0:
        cmd.extend(["--polish", str(pol)])

    if bool(getattr(args, "face_enhance", False)):
        cmd.append("--face-enhance")
        cmd.extend(["--face-enhance-sharpen", str(float(getattr(args, "face_enhance_sharpen", 0.35) or 0.35))])
        cmd.extend(["--face-enhance-contrast", str(float(getattr(args, "face_enhance_contrast", 1.04) or 1.04))])
        cmd.extend(["--face-enhance-padding", str(float(getattr(args, "face_enhance_padding", 0.25) or 0.25))])
        cmd.extend(["--face-enhance-max", str(int(getattr(args, "face_enhance_max", 4) or 4))])

    pri = str(getattr(args, "post_reference_image", "") or "").strip()
    if pri:
        cmd.extend(["--post-reference-image", pri])
        pra = float(getattr(args, "post_reference_alpha", 0.0) or 0.0)
        cmd.extend(["--post-reference-alpha", str(pra)])

    frs = str(getattr(args, "face_restore_shell", "") or "").strip()
    if frs:
        cmd.extend(["--face-restore-shell", frs])

    if bool(getattr(args, "no_refine", False)):
        cmd.append("--no-refine")
    else:
        try:
            rt = int(getattr(args, "refine_t", 50) or 50)
        except (TypeError, ValueError):
            rt = 50
        if rt != 50:
            cmd.extend(["--refine-t", str(rt)])

    rg = str(getattr(args, "refine_gate", "off") or "off").strip().lower()
    if rg != "off":
        cmd.extend(["--refine-gate", rg])
        cmd.extend(
            ["--refine-gate-threshold", str(float(getattr(args, "refine_gate_threshold", 0.62) or 0.62))]
        )


def sdx_enhance_argv_for_sample(args: Any) -> List[str]:
    """Argv fragment (no program name) for OCR repair subprocess parity with main passes."""
    out: List[str] = []
    extend_sample_py_sdx_enhance_cmd(out, args)
    return out


def extend_sample_py_adherence_quality_cmd(cmd: List[str], args: Any) -> None:
    """
    Forward ``sample.py`` prompt packs, CLIP alignment hooks, layout stage, and related inference flags.
    """
    qp = str(getattr(args, "quality_pack", "none") or "none").strip().lower()
    if qp != "none":
        cmd.extend(["--quality-pack", qp])
    ap = str(getattr(args, "adherence_pack", "none") or "none").strip().lower()
    if ap != "none":
        cmd.extend(["--adherence-pack", ap])

    try:
        cgt = float(getattr(args, "clip_guard_threshold", 0.0) or 0.0)
    except (TypeError, ValueError):
        cgt = 0.0
    if cgt > 0.0:
        cmd.extend(["--clip-guard-threshold", str(cgt)])
        cmd.extend(
            ["--clip-guard-model", str(getattr(args, "clip_guard_model", "openai/clip-vit-base-patch32") or "openai/clip-vit-base-patch32")]
        )
        cmd.extend(["--clip-guard-t-frac", str(float(getattr(args, "clip_guard_t_frac", 0.22) or 0.22))])
        cmd.extend(["--clip-guard-steps", str(int(getattr(args, "clip_guard_steps", 12) or 12))])

    try:
        cme = int(getattr(args, "clip_monitor_every", 0) or 0)
    except (TypeError, ValueError):
        cme = 0
    if cme > 0:
        cmd.extend(["--clip-monitor-every", str(cme)])
        cmd.extend(["--clip-monitor-threshold", str(float(getattr(args, "clip_monitor_threshold", 0.22) or 0.22))])
        cmd.extend(["--clip-monitor-cfg-boost", str(float(getattr(args, "clip_monitor_cfg_boost", 0.12) or 0.12))])
        cmd.extend(["--clip-monitor-rewind", str(float(getattr(args, "clip_monitor_rewind", 0.0) or 0.0))])

    try:
        vb = float(getattr(args, "volatile_cfg_boost", 0.0) or 0.0)
    except (TypeError, ValueError):
        vb = 0.0
    if vb > 0.0:
        cmd.extend(["--volatile-cfg-boost", str(vb)])
        cmd.extend(["--volatile-cfg-quantile", str(float(getattr(args, "volatile_cfg_quantile", 0.72) or 0.72))])
        cmd.extend(["--volatile-cfg-window", str(int(getattr(args, "volatile_cfg_window", 6) or 6))])

    try:
        sag = float(getattr(args, "sag_scale", 0.0) or 0.0)
    except (TypeError, ValueError):
        sag = 0.0
    if sag > 0.0:
        cmd.extend(["--sag-scale", str(sag)])

    if bool(getattr(args, "no_auto_expected_text", False)):
        cmd.append("--no-auto-expected-text")
    if bool(getattr(args, "no_auto_constraint_boost", False)):
        cmd.append("--no-auto-constraint-boost")

    hs = str(getattr(args, "hard_style", "") or "").strip().lower()
    if hs in ("3d", "realistic", "3d_realistic", "style_mix"):
        cmd.extend(["--hard-style", hs])

    if bool(getattr(args, "dual_stage_layout", False)):
        cmd.append("--dual-stage-layout")
        cmd.extend(["--dual-stage-div", str(int(getattr(args, "dual_stage_div", 2) or 2))])
        cmd.extend(["--dual-layout-steps", str(int(getattr(args, "dual_layout_steps", 24) or 24))])
        cmd.extend(["--dual-detail-steps", str(int(getattr(args, "dual_detail_steps", 20) or 20))])
        cmd.extend(["--dual-detail-strength", str(float(getattr(args, "dual_detail_strength", 0.38) or 0.38))])

    if bool(getattr(args, "deterministic", False)):
        cmd.append("--deterministic")
    if bool(getattr(args, "no_cache", False)):
        cmd.append("--no-cache")


def adherence_quality_argv_for_sample(args: Any) -> List[str]:
    """Argv fragment for OCR repair parity with main passes (packs + CLIP + layout hooks)."""
    out: List[str] = []
    extend_sample_py_adherence_quality_cmd(out, args)
    return out


def adapter_control_argv_for_sample(args: Any) -> List[str]:
    """Build argv fragment (no program name) for OCR repair and other subprocess reuse."""
    out: List[str] = []
    extend_sample_py_adapter_control_cmd(out, args)
    return out


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
    style_fusion_block: str = "",
    user_style_fragment: str = "",
    panel_hint: str = "",
    rolling_context: str = "",
) -> str:
    """
    Merge narration, consistency (cast/setting/text_continuity from JSON), optional hybrid style fusion,
    user-requested aesthetic fragment, panel/craft hints, rolling continuity, then the page line.

    Uses ``prompt_lexicon.merge_prompt_fragments`` so joining matches visual-memory / consistency helpers.
    """
    raw = merge_prompt_fragments(
        narration_prefix,
        consistency_block,
        style_fusion_block,
        user_style_fragment,
        panel_hint,
        rolling_context,
        user_prompt,
    )
    return normalize_book_prompt_fragment(raw)


class _PageTemplateDict(dict):
    """``str.format_map`` helper: unknown ``{placeholders}`` are left unchanged."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def expand_page_prompt_template(
    template: str,
    *,
    page_index: int,
    total_pages: Optional[int] = None,
    **extra: Any,
) -> str:
    """
    Fill ``--page-prompt-template`` placeholders.

    Built-ins: ``{page}`` / ``{page0}`` (0-based), ``{page1}`` (1-based),
    ``{total_pages}`` / ``{total_pages0}`` (book length when passed), plus any *extra* keys.
    Unknown ``{name}`` tokens are preserved for custom story-specific placeholders.
    """
    t = template or ""
    if not t:
        return ""
    mapping: Dict[str, Any] = {
        "page": int(page_index),
        "page0": int(page_index),
        "page1": int(page_index) + 1,
        "total_pages": "" if total_pages is None else int(total_pages),
        "total_pages0": "" if total_pages is None else max(0, int(total_pages) - 1),
    }
    mapping.update(extra)
    return t.format_map(_PageTemplateDict(mapping))


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

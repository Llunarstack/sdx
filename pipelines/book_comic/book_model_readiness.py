"""
Dual-model readiness for **book** generation: DiT/diffusion checkpoint (``--ckpt``) + optional
**vit_quality** scorer (``--pick-vit-ckpt``). Path validation, AR-regime hints, and manifest hints.

Does not run inference; safe to call from CLI preflight.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils.architecture.ar_block_conditioning import normalize_num_ar_blocks

from pipelines.book_comic.book_helpers import pick_metric_requires_vit_ckpt


def _first_path_from_lora_spec(spec: str) -> str:
    s = (spec or "").strip()
    if not s:
        return ""
    return s.split(":", 1)[0].strip()


def _first_path_from_control_spec(spec: str) -> str:
    s = (spec or "").strip()
    if not s:
        return ""
    return s.split(":", 1)[0].strip()


def collect_missing_path_errors(args: Any) -> List[str]:
    """
    Return fatal path issues (missing files). Intended for ``--book-preflight strict``.
    """
    errors: List[str] = []
    ckpt = Path(str(getattr(args, "ckpt", "") or ""))
    if not ckpt.is_file():
        errors.append(f"--ckpt is not a file: {ckpt}")

    pv = str(getattr(args, "pick_vit_ckpt", "") or "").strip()
    if pv and not Path(pv).is_file():
        errors.append(f"--pick-vit-ckpt is not a file: {pv}")

    for spec in getattr(args, "lora", None) or []:
        p = _first_path_from_lora_spec(str(spec))
        if p and not Path(p).is_file():
            errors.append(f"--lora path not found ({spec!r}): {p}")

    for spec in getattr(args, "control", None) or []:
        p = _first_path_from_control_spec(str(spec))
        if p and not Path(p).is_file():
            errors.append(f"--control path not found ({spec!r}): {p}")

    ci = str(getattr(args, "control_image", "") or "").strip()
    if ci and not Path(ci).is_file():
        errors.append(f"--control-image is not a file: {ci}")

    rif = str(getattr(args, "reference_image", "") or "").strip()
    if rif and not Path(rif).is_file():
        errors.append(f"--reference-image is not a file: {rif}")

    rap = str(getattr(args, "reference_adapter_pt", "") or "").strip()
    if rap and not Path(rap).is_file():
        errors.append(f"--reference-adapter-pt is not a file: {rap}")

    tf = str(getattr(args, "tags_file", "") or "").strip()
    if tf and not Path(tf).is_file():
        errors.append(f"--tags-file is not a file: {tf}")

    cs = str(getattr(args, "character_sheet", "") or "").strip()
    if cs and not Path(cs).is_file():
        errors.append(f"--character-sheet is not a file: {cs}")

    vm = str(getattr(args, "visual_memory", "") or "").strip()
    if vm and not Path(vm).is_file():
        errors.append(f"--visual-memory is not a file: {vm}")

    cj = str(getattr(args, "consistency_json", "") or "").strip()
    if cj and not Path(cj).is_file():
        errors.append(f"--consistency-json is not a file: {cj}")

    return errors


def collect_missing_path_warnings(args: Any) -> List[str]:
    """Same as errors but phrased as warnings (``--book-preflight warn`` does not exit)."""
    return [f"{e}" for e in collect_missing_path_errors(args)]


def collect_dual_model_alignment_warnings(
    args: Any,
    *,
    dit_ar_blocks: int = -1,
    vit_cfg: Dict[str, Any] | None = None,
    resolved_pick_best: str = "",
) -> List[str]:
    """
    Warn when ViT scorer config and DiT / CLI flags disagree (AR regime, missing ViT path, etc.).
    """
    warnings: List[str] = []
    pv = str(getattr(args, "pick_vit_ckpt", "") or "").strip()

    if not pv:
        return warnings

    cfg = vit_cfg if isinstance(vit_cfg, dict) else {}
    use_ar = bool(cfg.get("use_ar_conditioning", False))
    vit_img = cfg.get("image_size", 224)
    try:
        vit_img_i = int(vit_img)
    except (TypeError, ValueError):
        vit_img_i = 224

    ar_cli = int(getattr(args, "pick_vit_ar_blocks", -1) or -1)
    from_ckpt = bool(getattr(args, "pick_vit_ar_from_ckpt", False))

    if use_ar and ar_cli < 0 and not from_ckpt:
        warnings.append(
            "ViT checkpoint uses AR conditioning, but --pick-vit-ar-blocks is unset and "
            "--pick-vit-ar-from-ckpt is off; scorer uses the unknown AR one-hot. "
            "Prefer --pick-vit-ar-from-ckpt or set --pick-vit-ar-blocks to 0/2/4 matching the DiT run."
        )

    dit_n = int(dit_ar_blocks)
    vit_n = normalize_num_ar_blocks(ar_cli)
    if dit_n in (0, 2, 4) and vit_n in (0, 2, 4) and dit_n != vit_n:
        warnings.append(
            f"DiT checkpoint reports num_ar_blocks={dit_n} but --pick-vit-ar-blocks resolves to {vit_n}; "
            "ViT scores may be miscalibrated for this generator."
        )

    w = int(getattr(args, "width", 0) or 0)
    h = int(getattr(args, "height", 0) or 0)
    if w > 0 and h > 0 and max(w, h) > vit_img_i * 2:
        warnings.append(
            f"Output resolution {w}x{h} is much larger than ViT training image_size={vit_img_i}; "
            "pick-best ViT scores are computed on resized crops (still usable, less representative)."
        )

    return warnings


def peek_vit_config_for_args(args: Any) -> Dict[str, Any]:
    """Load ViT config dict when ``--pick-vit-ckpt`` points at a file."""
    pv = str(getattr(args, "pick_vit_ckpt", "") or "").strip()
    if not pv or not Path(pv).is_file():
        return {}
    try:
        from vit_quality.checkpoint_utils import peek_vit_quality_config

        return peek_vit_quality_config(pv)
    except Exception:
        return {}


def book_model_stack_snapshot(
    args: Any,
    *,
    dit_ar_blocks: int = -1,
    vit_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Compact manifest-friendly summary of generator + ViT readiness."""
    cfg = vit_cfg if isinstance(vit_cfg, dict) else {}
    pv = str(getattr(args, "pick_vit_ckpt", "") or "").strip()
    return {
        "dit_ckpt_exists": Path(str(getattr(args, "ckpt", "") or "")).is_file(),
        "dit_num_ar_blocks": int(dit_ar_blocks),
        "pick_vit_ckpt_set": bool(pv),
        "pick_vit_ckpt_exists": bool(pv) and Path(pv).is_file(),
        "vit_use_ar_conditioning": bool(cfg.get("use_ar_conditioning", False)),
        "vit_image_size": int(cfg.get("image_size", 224) or 224),
        "pick_vit_ar_blocks_cli": int(getattr(args, "pick_vit_ar_blocks", -1) or -1),
        "pick_vit_ar_from_ckpt": bool(getattr(args, "pick_vit_ar_from_ckpt", False)),
        "lora_count": len(getattr(args, "lora", None) or []),
        "control_stack_count": len(getattr(args, "control", None) or []),
    }


def run_book_preflight(
    args: Any,
    *,
    dit_ar_blocks: int = -1,
    mode: str = "warn",
    resolved_pick_best: str = "",
) -> Tuple[List[str], List[str]]:
    """
    Run path checks + dual-model alignment. *mode* is ``off`` | ``warn`` | ``strict``.

    Pass *resolved_pick_best* when ``--pick-best auto`` was expanded by ``--book-accuracy`` presets.

    Returns ``(errors, warnings)``. In strict mode, callers should exit if *errors* is non-empty.
    """
    m = str(mode or "warn").strip().lower()
    if m == "off":
        return [], []

    vit_cfg = peek_vit_config_for_args(args)
    align_warns = collect_dual_model_alignment_warnings(
        args,
        dit_ar_blocks=dit_ar_blocks,
        vit_cfg=vit_cfg,
        resolved_pick_best=resolved_pick_best,
    )
    path_issues = collect_missing_path_errors(args)
    pb_eff = str(resolved_pick_best or getattr(args, "pick_best", "") or "").strip().lower()

    if m == "strict":
        errors = list(path_issues)
        pv = str(getattr(args, "pick_vit_ckpt", "") or "").strip()
        if pick_metric_requires_vit_ckpt(pb_eff) and not pv:
            errors.append(
                f"--book-preflight strict requires --pick-vit-ckpt when effective pick-best is {pb_eff!r}"
            )
        return errors, align_warns

    # warn
    warnings = list(align_warns)
    warnings.extend(collect_missing_path_warnings(args))
    return [], warnings

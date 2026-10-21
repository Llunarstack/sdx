"""
Dual-model readiness for **book** generation: DiT/diffusion checkpoint (``--ckpt``) + optional
**vit_quality** scorer (``--pick-vit-ckpt``). Path validation, AR-regime hints, and manifest hints.

Does not run inference; safe to call from CLI preflight.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.architecture.ar_block_conditioning import normalize_num_ar_blocks
from utils.generation.sample_helpers import _parse_control_spec, _parse_lora_spec

from pipelines.book_comic.book_helpers import pick_metric_requires_vit_ckpt

# (args attribute, CLI flag) for optional single-file inputs, checked in this order.
_OPTIONAL_FILE_ARGS: tuple[tuple[str, str], ...] = (
    ("control_image", "--control-image"),
    ("reference_image", "--reference-image"),
    ("reference_adapter_pt", "--reference-adapter-pt"),
    ("tags_file", "--tags-file"),
    ("character_sheet", "--character-sheet"),
    ("visual_memory", "--visual-memory"),
    ("consistency_json", "--consistency-json"),
)


def _str_arg(args: Any, name: str) -> str:
    return str(getattr(args, name, "") or "").strip()


def _cli_ar_blocks(args: Any) -> int:
    """``--pick-vit-ar-blocks`` as an int; ``0`` is a valid regime, so only ``None`` means unset."""
    try:
        return int(getattr(args, "pick_vit_ar_blocks", -1))
    except (TypeError, ValueError):
        return -1


def collect_missing_path_errors(args: Any) -> list[str]:
    """Return fatal path issues (missing files). Intended for ``--book-preflight strict``."""
    errors: list[str] = []
    ckpt = Path(str(getattr(args, "ckpt", "") or ""))
    if not ckpt.is_file():
        errors.append(f"--ckpt is not a file: {ckpt}")

    pv = _str_arg(args, "pick_vit_ckpt")
    if pv and not Path(pv).is_file():
        errors.append(f"--pick-vit-ckpt is not a file: {pv}")

    # Same Windows-drive-safe parsers sample.py uses ("C:\\x.safetensors:0.8" -> "C:\\x.safetensors").
    for flag, parse, specs in (
        ("--lora", _parse_lora_spec, getattr(args, "lora", None)),
        ("--control", _parse_control_spec, getattr(args, "control", None)),
    ):
        for spec in specs or []:
            p = parse(str(spec))[0]
            if p and not Path(p).is_file():
                errors.append(f"{flag} path not found ({spec!r}): {p}")

    for attr, flag in _OPTIONAL_FILE_ARGS:
        value = _str_arg(args, attr)
        if value and not Path(value).is_file():
            errors.append(f"{flag} is not a file: {value}")

    return errors


def collect_dual_model_alignment_warnings(
    args: Any,
    *,
    dit_ar_blocks: int = -1,
    vit_cfg: dict[str, Any] | None = None,
) -> list[str]:
    """Warn when the ViT scorer config disagrees with DiT / CLI flags (AR regime, resolution).

    Returns no warnings when ``--pick-vit-ckpt`` is unset.
    """
    if not _str_arg(args, "pick_vit_ckpt"):
        return []

    warnings: list[str] = []
    cfg = vit_cfg or {}
    try:
        vit_img = int(cfg.get("image_size", 224))
    except (TypeError, ValueError):
        vit_img = 224
    ar_cli = _cli_ar_blocks(args)

    if cfg.get("use_ar_conditioning") and ar_cli < 0 and not getattr(args, "pick_vit_ar_from_ckpt", False):
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
    if w > 0 and h > 0 and max(w, h) > vit_img * 2:
        warnings.append(
            f"Output resolution {w}x{h} is much larger than ViT training image_size={vit_img}; "
            "pick-best ViT scores are computed on resized crops (still usable, less representative)."
        )

    return warnings


def peek_vit_config_for_args(args: Any) -> dict[str, Any]:
    """ViT config dict embedded in ``--pick-vit-ckpt`` ({} when unset, missing or unreadable)."""
    pv = _str_arg(args, "pick_vit_ckpt")
    if not pv:
        return {}
    try:
        from vit_quality.checkpoint_utils import peek_vit_quality_config
    except ImportError:
        return {}
    return peek_vit_quality_config(pv)


def book_model_stack_snapshot(
    args: Any,
    *,
    dit_ar_blocks: int = -1,
    vit_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compact manifest-friendly summary of generator + ViT readiness."""
    cfg = vit_cfg or {}
    pv = _str_arg(args, "pick_vit_ckpt")
    return {
        "dit_ckpt_exists": Path(str(getattr(args, "ckpt", "") or "")).is_file(),
        "dit_num_ar_blocks": int(dit_ar_blocks),
        "pick_vit_ckpt_set": bool(pv),
        "pick_vit_ckpt_exists": bool(pv) and Path(pv).is_file(),
        "vit_use_ar_conditioning": bool(cfg.get("use_ar_conditioning", False)),
        "vit_image_size": int(cfg.get("image_size", 224) or 224),
        "pick_vit_ar_blocks_cli": _cli_ar_blocks(args),
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
) -> tuple[list[str], list[str]]:
    """
    Run path checks + dual-model alignment. *mode* is ``off`` | ``warn`` | ``strict``.

    Pass *resolved_pick_best* when ``--pick-best auto`` was expanded by ``--book-accuracy`` presets.

    Returns ``(errors, warnings)``. In strict mode, callers should exit if *errors* is non-empty;
    in warn mode missing paths are reported as warnings instead.
    """
    m = str(mode or "warn").strip().lower()
    if m == "off":
        return [], []

    align_warns = collect_dual_model_alignment_warnings(
        args, dit_ar_blocks=dit_ar_blocks, vit_cfg=peek_vit_config_for_args(args)
    )
    path_issues = collect_missing_path_errors(args)
    if m != "strict":
        return [], align_warns + path_issues

    pb_eff = str(resolved_pick_best or getattr(args, "pick_best", "") or "").strip().lower()
    if pick_metric_requires_vit_ckpt(pb_eff) and not _str_arg(args, "pick_vit_ckpt"):
        path_issues.append(f"--book-preflight strict requires --pick-vit-ckpt when effective pick-best is {pb_eff!r}")
    return path_issues, align_warns

"""
Helpers for book/comic/manga model training orchestration.

This module focuses on:
- Preset bundles for training flags tuned for sequential art datasets.
- Native preflight checks over JSONL manifests (Rust/Zig tools when available).
- Building train.py command lines from a compact, book-focused CLI.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from utils.native import (
    manifest_fingerprint_line,
    native_stack_status,
    run_rust_jsonl_stats,
    run_rust_jsonl_validate,
)


@dataclass(frozen=True, slots=True)
class BookTrainPreset:
    """Book training defaults for a quality/speed tier."""

    model: str
    image_size: int
    global_batch_size: int
    lr: float
    passes: int
    max_steps: int
    train_shortcomings_mitigation: str
    train_shortcomings_2d: bool
    train_art_guidance_mode: str
    train_anatomy_guidance: str
    train_style_guidance_mode: str
    use_hierarchical_captions: bool
    region_caption_mode: str
    boost_adherence_caption: bool
    num_ar_blocks: int
    ar_block_order: str


_BALANCED_PRESET = BookTrainPreset(
    model="DiT-XL/2-Text",
    image_size=512,
    global_batch_size=2,
    lr=1e-4,
    passes=2,
    max_steps=0,
    train_shortcomings_mitigation="auto",
    train_shortcomings_2d=True,
    train_art_guidance_mode="auto",
    train_anatomy_guidance="lite",
    train_style_guidance_mode="auto",
    use_hierarchical_captions=True,
    region_caption_mode="append",
    boost_adherence_caption=True,
    num_ar_blocks=0,
    ar_block_order="raster",
)

_BOOK_TRAIN_PRESETS: dict[str, BookTrainPreset] = {
    "fast": replace(_BALANCED_PRESET, model="DiT-B/2-Text", passes=1, max_steps=1000),
    "balanced": _BALANCED_PRESET,
    "production": replace(
        _BALANCED_PRESET,
        image_size=768,
        global_batch_size=4,
        lr=8e-5,
        passes=4,
        train_shortcomings_mitigation="all",
        train_art_guidance_mode="all",
        train_anatomy_guidance="strong",
        train_style_guidance_mode="all",
        num_ar_blocks=2,
    ),
}

# Profile name -> (num_ar_blocks, ar_block_order); "auto" / unknown names force nothing.
_BOOK_AR_PROFILES: dict[str, tuple[int, str]] = {
    "none": (0, "raster"),
    "layout": (2, "raster"),
    "strong": (4, "raster"),
    "zorder": (2, "zorder"),
    "vit_layout": (2, "zorder"),
    "vit_strong": (4, "zorder"),
    "comic_snake": (2, "snake"),
    "cinema_spiral": (4, "spiral"),
}

_AR_BLOCK_ORDERS = ("raster", "zorder", "snake", "spiral")

_HUMANIZATION_LITE: dict[str, Any] = {
    "shortcomings_mitigation": "auto",
    "shortcomings_2d": True,
    "art_guidance_mode": "auto",
    "art_guidance_photography": True,
    "anatomy_guidance": "lite",
    "style_guidance_mode": "auto",
    "style_guidance_artists": True,
}

_HUMANIZATION_PACKS: dict[str, dict[str, Any]] = {
    "lite": _HUMANIZATION_LITE,
    "balanced": {**_HUMANIZATION_LITE, "style_guidance_mode": "all"},
    "strong": {
        **_HUMANIZATION_LITE,
        "shortcomings_mitigation": "all",
        "art_guidance_mode": "all",
        "anatomy_guidance": "strong",
        "style_guidance_mode": "all",
    },
}


def _positive_int(args: Any, name: str) -> int:
    """``args.<name>`` as an int, with missing / None / non-positive values collapsed to 0."""
    return max(0, int(getattr(args, name, 0) or 0))


def preset_for_book_train(name: str) -> BookTrainPreset:
    """Resolve ``--book-train-preset`` into concrete defaults (unknown names -> balanced)."""
    return _BOOK_TRAIN_PRESETS.get((name or "balanced").strip().lower(), _BALANCED_PRESET)


def resolve_book_ar_profile(name: str) -> dict[str, Any]:
    """One-flag AR profile -> ``{"num_ar_blocks", "ar_block_order"}``; ``{}`` for auto / unknown."""
    row = _BOOK_AR_PROFILES.get((name or "auto").strip().lower())
    if row is None:
        return {}
    return {"num_ar_blocks": row[0], "ar_block_order": row[1]}


def resolve_book_train_settings(args: Any) -> BookTrainPreset:
    """Apply CLI overrides over ``preset_for_book_train`` defaults.

    Zero means "unset" for the numeric overrides (including ``--num-ar-blocks``): train.py, which
    also calls this via ``training.cli``, defaults those flags to 0 and they must not
    clobber the preset. Use ``--ar-profile none`` to force zero AR blocks.
    """
    overrides: dict[str, Any] = {}
    model = str(getattr(args, "model", "") or "").strip()
    if model:
        overrides["model"] = model
    for name in ("image_size", "global_batch_size", "passes", "max_steps"):
        value = _positive_int(args, name)
        if value:
            overrides[name] = value
    lr = float(getattr(args, "lr", 0.0) or 0.0)
    if lr > 0:
        overrides["lr"] = lr
    overrides.update(resolve_book_ar_profile(str(getattr(args, "ar_profile", "auto") or "auto")))
    num_ar_blocks = _positive_int(args, "num_ar_blocks")
    if num_ar_blocks in (2, 4):
        overrides["num_ar_blocks"] = num_ar_blocks
    ar_block_order = str(getattr(args, "ar_block_order", "") or "").strip().lower()
    if ar_block_order in _AR_BLOCK_ORDERS:
        overrides["ar_block_order"] = ar_block_order
    return replace(preset_for_book_train(getattr(args, "book_train_preset", "balanced")), **overrides)


def build_train_command(
    *,
    root: Path,
    python_exe: str,
    args: Any,
    settings: BookTrainPreset,
    passthrough_train_args: list[str],
) -> list[str]:
    """Build the ``train.py`` command from book-focused settings + passthrough args."""
    train_py = root / "train.py"
    cmd: list[str] = [python_exe, str(train_py)]

    data_path = str(getattr(args, "data_path", "") or "").strip()
    manifest = str(getattr(args, "manifest_jsonl", "") or "").strip()
    if data_path:
        cmd.extend(["--data-path", data_path])
    if manifest:
        cmd.extend(["--manifest-jsonl", manifest])

    cmd.extend(
        [
            "--results-dir",
            str(getattr(args, "results_dir", "results/book_train")),
            "--model",
            settings.model,
            "--image-size",
            str(settings.image_size),
            "--global-batch-size",
            str(settings.global_batch_size),
            "--lr",
            str(settings.lr),
            "--passes",
            str(settings.passes),
            "--train-shortcomings-mitigation",
            settings.train_shortcomings_mitigation,
            "--train-art-guidance-mode",
            settings.train_art_guidance_mode,
            "--train-anatomy-guidance",
            settings.train_anatomy_guidance,
            "--train-style-guidance-mode",
            settings.train_style_guidance_mode,
            "--region-caption-mode",
            settings.region_caption_mode,
            "--num-ar-blocks",
            str(settings.num_ar_blocks),
            "--ar-block-order",
            settings.ar_block_order,
        ]
    )
    if settings.train_shortcomings_2d:
        cmd.append("--train-shortcomings-2d")
    if settings.use_hierarchical_captions:
        cmd.append("--use-hierarchical-captions")
    if settings.boost_adherence_caption:
        cmd.append("--boost-adherence-caption")

    if settings.max_steps > 0:
        cmd.extend(["--max-steps", str(settings.max_steps)])

    for attr, flag in (("dry_run", "--dry-run"), ("no_compile", "--no-compile"), ("no_xformers", "--no-xformers")):
        if getattr(args, attr, False):
            cmd.append(flag)
    if int(getattr(args, "num_workers", -1)) >= 0:
        cmd.extend(["--num-workers", str(int(args.num_workers))])

    resume = str(getattr(args, "resume", "") or "").strip()
    init_from = str(getattr(args, "init_from", "") or "").strip()
    if resume and init_from:
        raise ValueError("Use either --resume or --init-from for book training, not both.")
    if resume:
        cmd.extend(["--resume", resume])
    if init_from:
        cmd.extend(["--init-from", init_from])
    ar_curriculum_mode = str(getattr(args, "ar_curriculum_mode", "none") or "none").strip().lower()
    if ar_curriculum_mode in ("none", "step", "linear"):
        cmd.extend(["--ar-curriculum-mode", ar_curriculum_mode])
    for attr, flag in (
        ("ar_curriculum_warmup_steps", "--ar-curriculum-warmup-steps"),
        ("ar_curriculum_ramp_start", "--ar-curriculum-ramp-start"),
        ("ar_curriculum_ramp_end", "--ar-curriculum-ramp-end"),
    ):
        value = _positive_int(args, attr)
        if value:
            cmd.extend([flag, str(value)])
    # -1 is the "unset" sentinel here; 0 is a real block count (e.g. a curriculum targeting no AR).
    for attr, flag in (
        ("ar_curriculum_start_blocks", "--ar-curriculum-start-blocks"),
        ("ar_curriculum_target_blocks", "--ar-curriculum-target-blocks"),
    ):
        value = getattr(args, attr, None)
        if value is not None and int(value) in (0, 2, 4):
            cmd.extend([flag, str(int(value))])
    ar_order_mix = str(getattr(args, "ar_order_mix", "") or "").strip()
    if ar_order_mix:
        cmd.extend(["--ar-order-mix", ar_order_mix])

    cmd.extend(passthrough_train_args)
    return cmd


def run_native_manifest_preflight(
    manifest_jsonl: Path,
    *,
    min_caption_len: int = 0,
    max_caption_len: int = 0,
) -> dict[str, Any]:
    """
    Validate and fingerprint a manifest, preferring low-level native tools when present.

    - Fingerprint: Zig linecrc binary if present, else byte-identical Python fallback.
    - Validation + stats: Rust sdx-jsonl-tools when present.
    """
    out: dict[str, Any] = {
        "manifest": str(manifest_jsonl),
        "exists": manifest_jsonl.is_file(),
        "fingerprint": "",
        "native": native_stack_status(),
        "rust_validate_ok": None,
        "rust_validate_stdout": "",
        "rust_validate_stderr": "",
        "rust_stats_ok": None,
        "rust_stats_stdout": "",
        "rust_stats_stderr": "",
    }
    if not out["exists"]:
        return out

    out["fingerprint"] = manifest_fingerprint_line(manifest_jsonl)
    _record_native_tool(
        out,
        "rust_validate",
        lambda: run_rust_jsonl_validate(
            manifest_jsonl,
            min_caption_len=int(min_caption_len),
            max_caption_len=int(max_caption_len),
        ),
    )
    _record_native_tool(out, "rust_stats", lambda: run_rust_jsonl_stats(manifest_jsonl))
    return out


def _record_native_tool(out: dict[str, Any], key: str, run: Callable[[], Any]) -> None:
    """Store ``<key>_ok/_stdout/_stderr``; a missing tool leaves ``_ok`` as None with the error text."""
    try:
        result = run()
    except (OSError, ValueError) as exc:
        out[f"{key}_stderr"] = str(exc)
        return
    out[f"{key}_ok"] = result.returncode == 0
    out[f"{key}_stdout"] = (result.stdout or "").strip()
    out[f"{key}_stderr"] = (result.stderr or "").strip()


def build_hf_export_command(
    *,
    root: Path,
    python_exe: str,
    dataset: str,
    out_dir: Path,
    image_field: str = "image",
    caption_field: str = "tag_string",
    split: str = "train",
    config: str = "",
    revision: str = "",
    manifest_name: str = "manifest.jsonl",
    max_samples: int = 0,
    streaming: bool = True,
    shuffle_seed: int | None = None,
) -> list[str]:
    """Build command for ``scripts/tools/manifest.py hf-export``."""
    export_py = root / "scripts" / "tools" / "manifest.py"
    cmd: list[str] = [
        python_exe,
        str(export_py),
        "hf-export",
        "--dataset",
        dataset,
        "--split",
        split,
        "--image-field",
        image_field,
        "--caption-field",
        caption_field,
        "--out-dir",
        str(out_dir),
        "--manifest-name",
        manifest_name,
    ]
    if config:
        cmd.extend(["--config", config])
    if revision:
        cmd.extend(["--revision", revision])
    if max_samples > 0:
        cmd.extend(["--max-samples", str(max_samples)])
    if streaming:
        cmd.append("--streaming")
    if shuffle_seed is not None:
        cmd.extend(["--shuffle-seed", str(int(shuffle_seed))])
    return cmd


def build_caption_normalize_command(
    *,
    root: Path,
    python_exe: str,
    inp_manifest: Path,
    out_manifest: Path,
    unicode_normalize: bool = False,
    workers: int = 0,
    shortcomings_mitigation: str = "auto",
    shortcomings_2d: bool = True,
    art_guidance_mode: str = "auto",
    art_guidance_photography: bool = True,
    anatomy_guidance: str = "lite",
    style_guidance_mode: str = "auto",
    style_guidance_artists: bool = True,
) -> list[str]:
    """Build command for ``scripts/tools/captions.py normalize`` with book defaults."""
    normalize_py = root / "scripts" / "tools" / "captions.py"
    cmd: list[str] = [
        python_exe,
        str(normalize_py),
        "normalize",
        "--in",
        str(inp_manifest),
        "--out",
        str(out_manifest),
        "--shortcomings-mitigation",
        shortcomings_mitigation,
        "--art-guidance-mode",
        art_guidance_mode,
        "--anatomy-guidance",
        anatomy_guidance,
        "--style-guidance-mode",
        style_guidance_mode,
    ]
    if unicode_normalize:
        cmd.append("--unicode-normalize")
    if int(workers) > 0:
        cmd.extend(["--workers", str(int(workers))])
    if shortcomings_2d:
        cmd.append("--shortcomings-2d")
    if not art_guidance_photography:
        cmd.append("--no-art-guidance-photography")
    if not style_guidance_artists:
        cmd.append("--no-style-guidance-artists")
    return cmd


def resolve_train_humanization_pack(name: str) -> dict[str, Any]:
    """Caption-normalization defaults (``lite`` / ``balanced`` / ``strong``) that reduce synthetic artifacts.

    Unknown names (including ``none``) return ``{}``.
    """
    return dict(_HUMANIZATION_PACKS.get((name or "none").strip().lower(), {}))

"""
Helpers for book/comic/manga model training orchestration.

This module focuses on:
- Preset bundles for training flags tuned for sequential art datasets.
- Native preflight checks over JSONL manifests (Rust/Zig tools when available).
- Building train.py command lines from a compact, book-focused CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.native.native_tools import (
    manifest_fingerprint_line,
    native_stack_status,
    run_rust_jsonl_stats,
    run_rust_jsonl_validate,
)


@dataclass(frozen=True)
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


def preset_for_book_train(name: str) -> BookTrainPreset:
    """Resolve ``--book-train-preset`` into concrete defaults."""
    n = (name or "balanced").strip().lower()
    if n == "fast":
        return BookTrainPreset(
            model="DiT-B/2-Text",
            image_size=512,
            global_batch_size=2,
            lr=1e-4,
            passes=1,
            max_steps=1000,
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
    if n == "production":
        return BookTrainPreset(
            model="DiT-XL/2-Text",
            image_size=768,
            global_batch_size=4,
            lr=8e-5,
            passes=4,
            max_steps=0,
            train_shortcomings_mitigation="all",
            train_shortcomings_2d=True,
            train_art_guidance_mode="all",
            train_anatomy_guidance="strong",
            train_style_guidance_mode="all",
            use_hierarchical_captions=True,
            region_caption_mode="append",
            boost_adherence_caption=True,
            num_ar_blocks=2,
            ar_block_order="raster",
        )
    # balanced
    return BookTrainPreset(
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


def resolve_book_ar_profile(name: str) -> Dict[str, Any]:
    """
    Resolve one-flag AR profile for training.
    """
    n = (name or "auto").strip().lower()
    if n == "none":
        return {"num_ar_blocks": 0, "ar_block_order": "raster"}
    if n == "layout":
        return {"num_ar_blocks": 2, "ar_block_order": "raster"}
    if n == "strong":
        return {"num_ar_blocks": 4, "ar_block_order": "raster"}
    if n == "zorder":
        return {"num_ar_blocks": 2, "ar_block_order": "zorder"}
    # auto/unknown: do not force anything; keep preset/default behavior.
    return {}


def resolve_book_train_settings(args: Any) -> BookTrainPreset:
    """Apply CLI overrides over ``preset_for_book_train`` defaults."""
    base = preset_for_book_train(getattr(args, "book_train_preset", "balanced"))
    model = str(getattr(args, "model", "") or "").strip()
    if model:
        base = replace(base, model=model)
    image_size = int(getattr(args, "image_size", 0) or 0)
    if image_size > 0:
        base = replace(base, image_size=image_size)
    gbs = int(getattr(args, "global_batch_size", 0) or 0)
    if gbs > 0:
        base = replace(base, global_batch_size=gbs)
    lr = float(getattr(args, "lr", 0.0) or 0.0)
    if lr > 0:
        base = replace(base, lr=lr)
    passes = int(getattr(args, "passes", -1) or -1)
    if passes >= 0:
        base = replace(base, passes=passes)
    max_steps = int(getattr(args, "max_steps", -1) or -1)
    if max_steps >= 0:
        base = replace(base, max_steps=max_steps)
    ar_prof = resolve_book_ar_profile(str(getattr(args, "ar_profile", "auto") or "auto"))
    if "num_ar_blocks" in ar_prof:
        base = replace(base, num_ar_blocks=int(ar_prof["num_ar_blocks"]))
    if "ar_block_order" in ar_prof:
        base = replace(base, ar_block_order=str(ar_prof["ar_block_order"]))
    nab = int(getattr(args, "num_ar_blocks", -1) or -1)
    if nab in (0, 2, 4):
        base = replace(base, num_ar_blocks=nab)
    abo = str(getattr(args, "ar_block_order", "") or "").strip().lower()
    if abo in ("raster", "zorder"):
        base = replace(base, ar_block_order=abo)
    return base


def build_train_command(
    *,
    root: Path,
    python_exe: str,
    args: Any,
    settings: BookTrainPreset,
    passthrough_train_args: List[str],
) -> List[str]:
    """Build the ``train.py`` command from book-focused settings + passthrough args."""
    train_py = root / "train.py"
    cmd: List[str] = [python_exe, str(train_py)]

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
            str(int(getattr(settings, "num_ar_blocks", 0))),
            "--ar-block-order",
            str(getattr(settings, "ar_block_order", "raster") or "raster"),
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

    if bool(getattr(args, "dry_run", False)):
        cmd.append("--dry-run")
    if bool(getattr(args, "no_compile", False)):
        cmd.append("--no-compile")
    if bool(getattr(args, "no_xformers", False)):
        cmd.append("--no-xformers")
    if int(getattr(args, "num_workers", -1)) >= 0:
        cmd.extend(["--num-workers", str(int(getattr(args, "num_workers")))])

    cmd.extend(passthrough_train_args)
    return cmd


def run_native_manifest_preflight(
    manifest_jsonl: Path,
    *,
    min_caption_len: int = 0,
    max_caption_len: int = 0,
) -> Dict[str, Any]:
    """
    Validate and fingerprint a manifest, preferring low-level native tools when present.

    - Fingerprint: Zig linecrc binary if present, else byte-identical Python fallback.
    - Validation + stats: Rust sdx-jsonl-tools when present.
    """
    out: Dict[str, Any] = {
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
    if not manifest_jsonl.is_file():
        return out

    out["fingerprint"] = manifest_fingerprint_line(manifest_jsonl)

    try:
        r_val = run_rust_jsonl_validate(
            manifest_jsonl,
            min_caption_len=int(min_caption_len),
            max_caption_len=int(max_caption_len),
        )
        out["rust_validate_ok"] = r_val.returncode == 0
        out["rust_validate_stdout"] = (r_val.stdout or "").strip()
        out["rust_validate_stderr"] = (r_val.stderr or "").strip()
    except (FileNotFoundError, OSError, ValueError) as exc:
        out["rust_validate_ok"] = None
        out["rust_validate_stderr"] = str(exc)

    try:
        r_stats = run_rust_jsonl_stats(manifest_jsonl)
        out["rust_stats_ok"] = r_stats.returncode == 0
        out["rust_stats_stdout"] = (r_stats.stdout or "").strip()
        out["rust_stats_stderr"] = (r_stats.stderr or "").strip()
    except (FileNotFoundError, OSError, ValueError) as exc:
        out["rust_stats_ok"] = None
        out["rust_stats_stderr"] = str(exc)

    return out


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
    shuffle_seed: Optional[int] = None,
) -> List[str]:
    """Build command for ``scripts/training/hf_export_to_sdx_manifest.py``."""
    export_py = root / "scripts" / "training" / "hf_export_to_sdx_manifest.py"
    cmd: List[str] = [
        python_exe,
        str(export_py),
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
    shortcomings_mitigation: str = "auto",
    shortcomings_2d: bool = True,
    art_guidance_mode: str = "auto",
    art_guidance_photography: bool = True,
    anatomy_guidance: str = "lite",
    style_guidance_mode: str = "auto",
    style_guidance_artists: bool = True,
) -> List[str]:
    """Build command for ``scripts/tools/normalize_captions.py`` with book defaults."""
    norm_py = root / "scripts" / "tools" / "normalize_captions.py"
    cmd: List[str] = [
        python_exe,
        str(norm_py),
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
    if shortcomings_2d:
        cmd.append("--shortcomings-2d")
    if not art_guidance_photography:
        cmd.append("--no-art-guidance-photography")
    if not style_guidance_artists:
        cmd.append("--no-style-guidance-artists")
    return cmd


def resolve_train_humanization_pack(name: str) -> Dict[str, Any]:
    """
    Higher-level normalization defaults to reduce synthetic artifacts during training.
    """
    n = (name or "none").strip().lower()
    if n == "lite":
        return {
            "shortcomings_mitigation": "auto",
            "shortcomings_2d": True,
            "art_guidance_mode": "auto",
            "art_guidance_photography": True,
            "anatomy_guidance": "lite",
            "style_guidance_mode": "auto",
            "style_guidance_artists": True,
        }
    if n == "strong":
        return {
            "shortcomings_mitigation": "all",
            "shortcomings_2d": True,
            "art_guidance_mode": "all",
            "art_guidance_photography": True,
            "anatomy_guidance": "strong",
            "style_guidance_mode": "all",
            "style_guidance_artists": True,
        }
    # balanced / default
    if n == "balanced":
        return {
            "shortcomings_mitigation": "auto",
            "shortcomings_2d": True,
            "art_guidance_mode": "auto",
            "art_guidance_photography": True,
            "anatomy_guidance": "lite",
            "style_guidance_mode": "all",
            "style_guidance_artists": True,
        }
    return {}

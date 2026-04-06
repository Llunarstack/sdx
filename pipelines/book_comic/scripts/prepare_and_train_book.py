#!/usr/bin/env python3
"""
End-to-end book dataset prep + training launcher.

Pipeline:
1) (Optional) Export from Hugging Face dataset -> SDX manifest
2) (Optional) Normalize captions for book-focused guidance
3) Launch book trainer wrapper (which can run native preflight + train.py)

Pass any extra train.py args after ``--``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.book_comic.book_training_helpers import (  # noqa: E402
    build_caption_normalize_command,
    build_hf_export_command,
    resolve_train_humanization_pack,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare book dataset and train in one command.")

    p.add_argument("--data-path", type=str, default="", help="Existing data root. If --dataset is used, defaults to --out-dir.")
    p.add_argument("--manifest-jsonl", type=str, default="", help="Existing manifest path. Optional when using --dataset.")
    p.add_argument("--out-dir", type=str, default="data/book_train_run", help="Export directory when using --dataset.")
    p.add_argument("--results-dir", type=str, default="results/book_train_run")

    # HF export mode (optional)
    p.add_argument("--dataset", type=str, default="", help="HF dataset id to export before training.")
    p.add_argument("--config", type=str, default="", help="HF config name.")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--revision", type=str, default="")
    p.add_argument("--image-field", type=str, default="image")
    p.add_argument("--caption-field", type=str, default="tag_string")
    p.add_argument("--manifest-name", type=str, default="manifest.jsonl")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--shuffle-seed", type=int, default=None)

    # Caption normalization stage
    p.add_argument("--normalize-captions", action="store_true", help="Run normalize_captions before training.")
    p.add_argument("--normalized-manifest-name", type=str, default="manifest_book_norm.jsonl")
    p.add_argument(
        "--train-humanize-pack",
        type=str,
        default="none",
        choices=["none", "lite", "balanced", "strong"],
        help="Preset normalization bundle to reduce synthetic/AI-looking caption bias.",
    )
    p.add_argument("--shortcomings-mitigation", type=str, default="auto", choices=["none", "auto", "all"])
    p.add_argument("--shortcomings-2d", action="store_true")
    p.add_argument("--art-guidance-mode", type=str, default="auto", choices=["none", "auto", "all"])
    p.add_argument("--no-art-guidance-photography", action="store_true")
    p.add_argument("--anatomy-guidance", type=str, default="lite", choices=["none", "lite", "strong"])
    p.add_argument("--style-guidance-mode", type=str, default="auto", choices=["none", "auto", "all"])
    p.add_argument("--no-style-guidance-artists", action="store_true")

    # Book trainer options (forwarded to train_book_model.py)
    p.add_argument("--book-train-preset", type=str, default="balanced", choices=["fast", "balanced", "production"])
    p.add_argument("--model", type=str, default="")
    p.add_argument("--image-size", type=int, default=0)
    p.add_argument("--global-batch-size", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.0)
    p.add_argument("--passes", type=int, default=-1)
    p.add_argument("--max-steps-train", type=int, default=-1)
    p.add_argument(
        "--ar-profile",
        type=str,
        default="auto",
        choices=["auto", "none", "layout", "strong", "zorder", "vit_layout", "vit_strong", "comic_snake", "cinema_spiral"],
        help="Book trainer AR preset (classic + upgraded ViT-aligned traversal presets).",
    )
    p.add_argument("--num-ar-blocks", type=int, default=-1, choices=[-1, 0, 2, 4])
    p.add_argument("--ar-block-order", type=str, default="", choices=["", "raster", "zorder", "snake", "spiral"])
    p.add_argument("--ar-curriculum-mode", type=str, default="none", choices=["none", "step", "linear"])
    p.add_argument("--ar-curriculum-warmup-steps", type=int, default=0)
    p.add_argument("--ar-curriculum-ramp-start", type=int, default=0)
    p.add_argument("--ar-curriculum-ramp-end", type=int, default=0)
    p.add_argument("--ar-curriculum-start-blocks", type=int, default=-1, choices=[-1, 0, 2, 4])
    p.add_argument("--ar-curriculum-target-blocks", type=int, default=-1, choices=[-1, 0, 2, 4])
    p.add_argument("--ar-order-mix", type=str, default="")
    p.add_argument("--num-workers", type=int, default=-1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--no-xformers", action="store_true")
    p.add_argument("--native-preflight", action="store_true")
    p.add_argument("--strict-native", action="store_true")
    p.add_argument("--manifest-min-caption-len", type=int, default=0)
    p.add_argument("--manifest-max-caption-len", type=int, default=0)
    return p


def _run(cmd: list[str], *, cwd: Path, label: str) -> int:
    print(f"[book_prepare_train] {label}: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd).returncode


def main() -> int:
    raw = sys.argv[1:]
    if "--" in raw:
        i = raw.index("--")
        own_argv = raw[:i]
        passthrough_train_args = raw[i + 1 :]
    else:
        own_argv = raw
        passthrough_train_args = []

    args = _build_parser().parse_args(own_argv)
    py = sys.executable

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = Path(args.manifest_jsonl) if str(args.manifest_jsonl).strip() else None
    data_path = str(args.data_path).strip()

    # Stage 1: optional HF export
    if str(args.dataset).strip():
        export_cmd = build_hf_export_command(
            root=ROOT,
            python_exe=py,
            dataset=str(args.dataset).strip(),
            out_dir=out_dir,
            image_field=str(args.image_field),
            caption_field=str(args.caption_field),
            split=str(args.split),
            config=str(args.config).strip(),
            revision=str(args.revision).strip(),
            manifest_name=str(args.manifest_name),
            max_samples=int(args.max_samples),
            streaming=bool(args.streaming),
            shuffle_seed=args.shuffle_seed,
        )
        rc = _run(export_cmd, cwd=ROOT, label="HF export")
        if rc != 0:
            return rc
        manifest = out_dir / str(args.manifest_name)
        if not data_path:
            data_path = str(out_dir)

    if manifest is None:
        if not str(args.manifest_jsonl).strip():
            print("Error: provide --manifest-jsonl, or --dataset for export mode.", file=sys.stderr)
            return 2
        manifest = Path(args.manifest_jsonl)

    # Stage 2: optional caption normalization
    if bool(args.normalize_captions):
        _humanize = resolve_train_humanization_pack(str(getattr(args, "train_humanize_pack", "none") or "none"))
        shortcomings_mitigation = str(_humanize.get("shortcomings_mitigation", args.shortcomings_mitigation))
        shortcomings_2d = bool(_humanize.get("shortcomings_2d", bool(args.shortcomings_2d)))
        art_guidance_mode = str(_humanize.get("art_guidance_mode", args.art_guidance_mode))
        art_guidance_photography = bool(
            _humanize.get("art_guidance_photography", not bool(args.no_art_guidance_photography))
        )
        anatomy_guidance = str(_humanize.get("anatomy_guidance", args.anatomy_guidance))
        style_guidance_mode = str(_humanize.get("style_guidance_mode", args.style_guidance_mode))
        style_guidance_artists = bool(_humanize.get("style_guidance_artists", not bool(args.no_style_guidance_artists)))

        normalized_manifest = out_dir / str(args.normalized_manifest_name)
        normalize_cmd = build_caption_normalize_command(
            root=ROOT,
            python_exe=py,
            inp_manifest=manifest,
            out_manifest=normalized_manifest,
            shortcomings_mitigation=shortcomings_mitigation,
            shortcomings_2d=shortcomings_2d,
            art_guidance_mode=art_guidance_mode,
            art_guidance_photography=art_guidance_photography,
            anatomy_guidance=anatomy_guidance,
            style_guidance_mode=style_guidance_mode,
            style_guidance_artists=style_guidance_artists,
        )
        rc = _run(normalize_cmd, cwd=ROOT, label="Caption normalize")
        if rc != 0:
            return rc
        manifest = normalized_manifest
        if not data_path:
            data_path = str(out_dir)

    # Stage 3: book trainer wrapper
    train_wrapper = ROOT / "pipelines" / "book_comic" / "scripts" / "train_book_model.py"
    train_cmd: list[str] = [
        py,
        str(train_wrapper),
        "--results-dir",
        str(args.results_dir),
        "--book-train-preset",
        str(args.book_train_preset),
        "--manifest-jsonl",
        str(manifest),
    ]
    if data_path:
        train_cmd.extend(["--data-path", data_path])
    if str(args.model).strip():
        train_cmd.extend(["--model", str(args.model).strip()])
    if int(args.image_size) > 0:
        train_cmd.extend(["--image-size", str(int(args.image_size))])
    if int(args.global_batch_size) > 0:
        train_cmd.extend(["--global-batch-size", str(int(args.global_batch_size))])
    if float(args.lr) > 0:
        train_cmd.extend(["--lr", str(float(args.lr))])
    if int(args.passes) >= 0:
        train_cmd.extend(["--passes", str(int(args.passes))])
    if int(args.max_steps_train) >= 0:
        train_cmd.extend(["--max-steps", str(int(args.max_steps_train))])
    if str(args.ar_profile).strip():
        train_cmd.extend(["--ar-profile", str(args.ar_profile).strip()])
    if int(args.num_ar_blocks) in (0, 2, 4):
        train_cmd.extend(["--num-ar-blocks", str(int(args.num_ar_blocks))])
    if str(args.ar_block_order).strip() in ("raster", "zorder", "snake", "spiral"):
        train_cmd.extend(["--ar-block-order", str(args.ar_block_order).strip()])
    if str(args.ar_curriculum_mode).strip() in ("none", "step", "linear"):
        train_cmd.extend(["--ar-curriculum-mode", str(args.ar_curriculum_mode).strip()])
    if int(args.ar_curriculum_warmup_steps) > 0:
        train_cmd.extend(["--ar-curriculum-warmup-steps", str(int(args.ar_curriculum_warmup_steps))])
    if int(args.ar_curriculum_ramp_start) > 0:
        train_cmd.extend(["--ar-curriculum-ramp-start", str(int(args.ar_curriculum_ramp_start))])
    if int(args.ar_curriculum_ramp_end) > 0:
        train_cmd.extend(["--ar-curriculum-ramp-end", str(int(args.ar_curriculum_ramp_end))])
    if int(args.ar_curriculum_start_blocks) in (0, 2, 4):
        train_cmd.extend(["--ar-curriculum-start-blocks", str(int(args.ar_curriculum_start_blocks))])
    if int(args.ar_curriculum_target_blocks) in (0, 2, 4):
        train_cmd.extend(["--ar-curriculum-target-blocks", str(int(args.ar_curriculum_target_blocks))])
    if str(args.ar_order_mix).strip():
        train_cmd.extend(["--ar-order-mix", str(args.ar_order_mix).strip()])
    if int(args.num_workers) >= 0:
        train_cmd.extend(["--num-workers", str(int(args.num_workers))])
    if bool(args.dry_run):
        train_cmd.append("--dry-run")
    if bool(args.no_compile):
        train_cmd.append("--no-compile")
    if bool(args.no_xformers):
        train_cmd.append("--no-xformers")
    if bool(args.native_preflight):
        train_cmd.append("--native-preflight")
    if bool(args.strict_native):
        train_cmd.append("--strict-native")
    if int(args.manifest_min_caption_len) > 0:
        train_cmd.extend(["--manifest-min-caption-len", str(int(args.manifest_min_caption_len))])
    if int(args.manifest_max_caption_len) > 0:
        train_cmd.extend(["--manifest-max-caption-len", str(int(args.manifest_max_caption_len))])

    if passthrough_train_args:
        train_cmd.extend(["--", *passthrough_train_args])
    return _run(train_cmd, cwd=ROOT, label="Book train")


if __name__ == "__main__":
    raise SystemExit(main())

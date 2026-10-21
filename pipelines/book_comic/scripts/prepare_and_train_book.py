#!/usr/bin/env python3
"""
End-to-end book dataset prep + training launcher.

Pipeline:
1) (Optional) Export from Hugging Face dataset -> SDX manifest
2) (Optional) manifest_gate, then data_quality filter/dedupe
3) (Optional) Normalize captions for book-focused guidance
4) Launch the book trainer wrapper (which can run native preflight + train.py)

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
from pipelines.book_comic.scripts.train_book_model import (  # noqa: E402
    AR_PROFILES,
    add_manifest_gate_args,
    manifest_gate_command,
    split_passthrough,
)

TRAIN_WRAPPER = Path(__file__).resolve().with_name("train_book_model.py")

# Trainer options forwarded to train_book_model.py under the same flag name (``--`` + dest with dashes).
_FORWARD_TEXT = ("model", "ar_profile", "ar_block_order", "ar_curriculum_mode", "ar_order_mix", "resume", "init_from")
_FORWARD_SWITCHES = ("dry_run", "no_compile", "no_xformers", "native_preflight", "strict_native")
# Numeric options forwarded only when above their "keep the preset/default" sentinel.
_FORWARD_NUMERIC = {
    "image_size": 0,
    "global_batch_size": 0,
    "lr": 0.0,
    "passes": -1,
    "num_ar_blocks": -1,
    "ar_curriculum_warmup_steps": 0,
    "ar_curriculum_ramp_start": 0,
    "ar_curriculum_ramp_end": 0,
    "ar_curriculum_start_blocks": -1,
    "ar_curriculum_target_blocks": -1,
    "num_workers": -1,
    "manifest_min_caption_len": 0,
    "manifest_max_caption_len": 0,
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare book dataset and train in one command.")

    p.add_argument(
        "--data-path", type=str, default="", help="Existing data root. If --dataset is used, defaults to --out-dir."
    )
    p.add_argument(
        "--manifest-jsonl", type=str, default="", help="Existing manifest path. Optional when using --dataset."
    )
    p.add_argument("--out-dir", type=str, default="data/book_train_run", help="Export directory when using --dataset.")
    p.add_argument("--results-dir", type=str, default="results/book_train_run")

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

    p.add_argument("--normalize-captions", action="store_true", help="Run normalize_captions before training.")
    p.add_argument("--normalized-manifest-name", type=str, default="manifest_book_norm.jsonl")
    p.add_argument(
        "--normalize-captions-unicode",
        action="store_true",
        help="Before other caption transforms: NFKC + zero-width strip via sdx_native.",
    )
    p.add_argument(
        "--normalize-captions-workers",
        type=int,
        default=0,
        help="If >0, parallelize caption normalization (ProcessPool).",
    )
    p.add_argument(
        "--manifest-gate",
        action="store_true",
        help="Run manifest_gate (prompt lint + hygiene + optional image QC) before filtering/normalizing.",
    )
    add_manifest_gate_args(p)

    p.add_argument(
        "--filter-manifest",
        action="store_true",
        help="Run data_quality to dedupe/filter manifest before normalization/training.",
    )
    p.add_argument("--filter-out-name", type=str, default="manifest_quality.jsonl")
    p.add_argument("--filter-dedup", type=str, default="", choices=["", "phash", "md5"])
    p.add_argument("--filter-min-caption-len", type=int, default=0)
    p.add_argument("--filter-max-caption-len", type=int, default=0)
    p.add_argument("--filter-bad-words", type=str, default="")
    p.add_argument("--filter-min-weight", type=float, default=0.0)
    p.add_argument("--filter-native-validate", action="store_true")
    p.add_argument(
        "--train-humanize-pack",
        type=str,
        default="none",
        choices=["none", "lite", "balanced", "strong"],
        help="Preset normalization bundle to reduce synthetic/AI-looking caption bias (overrides the flags below).",
    )
    p.add_argument("--shortcomings-mitigation", type=str, default="auto", choices=["none", "auto", "all"])
    p.add_argument("--shortcomings-2d", action="store_true")
    p.add_argument("--art-guidance-mode", type=str, default="auto", choices=["none", "auto", "all"])
    p.add_argument("--no-art-guidance-photography", action="store_true")
    p.add_argument("--anatomy-guidance", type=str, default="lite", choices=["none", "lite", "strong"])
    p.add_argument("--style-guidance-mode", type=str, default="auto", choices=["none", "auto", "all"])
    p.add_argument("--no-style-guidance-artists", action="store_true")

    # Forwarded to train_book_model.py.
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
        choices=AR_PROFILES,
        help="Book trainer AR preset (classic + ViT-aligned traversal presets).",
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
    p.add_argument("--resume", type=str, default="", help="Forwarded to book trainer -> train.py --resume.")
    p.add_argument(
        "--init-from",
        type=str,
        dest="init_from",
        default="",
        help="Forwarded to book trainer -> train.py --init-from (SDX base weights, fresh optimizer).",
    )
    return p


def _run(cmd: list[str], *, label: str) -> int:
    print(f"[book_prepare_train] {label}: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=ROOT).returncode


def _cli_flag(dest: str) -> str:
    return "--" + dest.replace("_", "-")


def _data_quality_command(args: argparse.Namespace, manifest: Path, out: Path) -> list[str]:
    cmd = [sys.executable, "-m", "scripts.tools", "data_quality", str(manifest), "--out", str(out)]
    if args.filter_dedup:
        cmd += ["--dedup", args.filter_dedup]
    if args.filter_min_caption_len > 0:
        cmd += ["--min-caption-len", str(args.filter_min_caption_len)]
    if args.filter_max_caption_len > 0:
        cmd += ["--max-caption-len", str(args.filter_max_caption_len)]
    bad_words = args.filter_bad_words.strip()
    if bad_words:
        cmd += ["--bad-words", bad_words]
    if args.filter_min_weight > 0:
        cmd += ["--min-weight", str(args.filter_min_weight)]
    if args.filter_native_validate:
        cmd.append("--native-validate")
    return cmd


def _normalize_command(args: argparse.Namespace, manifest: Path, out: Path) -> list[str]:
    caption_opts = {
        "shortcomings_mitigation": args.shortcomings_mitigation,
        "shortcomings_2d": args.shortcomings_2d,
        "art_guidance_mode": args.art_guidance_mode,
        "art_guidance_photography": not args.no_art_guidance_photography,
        "anatomy_guidance": args.anatomy_guidance,
        "style_guidance_mode": args.style_guidance_mode,
        "style_guidance_artists": not args.no_style_guidance_artists,
    }
    caption_opts.update(resolve_train_humanization_pack(args.train_humanize_pack))
    return build_caption_normalize_command(
        root=ROOT,
        python_exe=sys.executable,
        inp_manifest=manifest,
        out_manifest=out,
        unicode_normalize=args.normalize_captions_unicode,
        workers=args.normalize_captions_workers,
        **caption_opts,
    )


def _trainer_command(args: argparse.Namespace, manifest: Path, data_path: str, passthrough: list[str]) -> list[str]:
    cmd = [
        sys.executable,
        str(TRAIN_WRAPPER),
        "--results-dir",
        args.results_dir,
        "--book-train-preset",
        args.book_train_preset,
        "--manifest-jsonl",
        str(manifest),
    ]
    if data_path:
        cmd += ["--data-path", data_path]
    for dest in _FORWARD_TEXT:
        value = getattr(args, dest).strip()
        if value:
            cmd += [_cli_flag(dest), value]
    for dest, sentinel in _FORWARD_NUMERIC.items():
        value = getattr(args, dest)
        if value > sentinel:
            cmd += [_cli_flag(dest), str(value)]
    if args.max_steps_train >= 0:
        cmd += ["--max-steps", str(args.max_steps_train)]
    cmd += [_cli_flag(dest) for dest in _FORWARD_SWITCHES if getattr(args, dest)]
    if passthrough:
        cmd += ["--", *passthrough]
    return cmd


def main() -> int:
    own_argv, passthrough_train_args = split_passthrough(sys.argv[1:])
    args = _build_parser().parse_args(own_argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_arg = args.manifest_jsonl.strip()
    manifest = Path(manifest_arg) if manifest_arg else None
    data_path = args.data_path.strip()

    dataset = args.dataset.strip()
    if dataset:
        export_cmd = build_hf_export_command(
            root=ROOT,
            python_exe=sys.executable,
            dataset=dataset,
            out_dir=out_dir,
            image_field=args.image_field,
            caption_field=args.caption_field,
            split=args.split,
            config=args.config.strip(),
            revision=args.revision.strip(),
            manifest_name=args.manifest_name,
            max_samples=args.max_samples,
            streaming=args.streaming,
            shuffle_seed=args.shuffle_seed,
        )
        rc = _run(export_cmd, label="HF export")
        if rc != 0:
            return rc
        manifest = out_dir / args.manifest_name
        data_path = data_path or str(out_dir)

    if manifest is None:
        print("Error: provide --manifest-jsonl, or --dataset for export mode.", file=sys.stderr)
        return 2

    if args.manifest_gate:
        rc = _run(manifest_gate_command(args, manifest), label="Manifest gate")
        if rc != 0:
            return rc

    if args.filter_manifest:
        filtered = out_dir / args.filter_out_name
        rc = _run(_data_quality_command(args, manifest, filtered), label="Manifest filter/dedupe")
        if rc != 0:
            return rc
        manifest = filtered

    if args.normalize_captions:
        normalized = out_dir / args.normalized_manifest_name
        rc = _run(_normalize_command(args, manifest, normalized), label="Caption normalize")
        if rc != 0:
            return rc
        manifest = normalized
        data_path = data_path or str(out_dir)

    return _run(_trainer_command(args, manifest, data_path, passthrough_train_args), label="Book train")


if __name__ == "__main__":
    raise SystemExit(main())

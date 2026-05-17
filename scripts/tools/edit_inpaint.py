#!/usr/bin/env python3
"""Inpaint or img2img by delegating to ``sample.py`` (same as the main CLI)."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image
from utils.generation.edit_masks import heuristic_inpaint_mask, save_heuristic_mask
from utils.generation.sample_edit_runner import run_edit_with_pillow, run_sample_inference
from utils.generation.segmentation_to_mask import build_segmentation_mask_for_edit

_HEURISTIC_CHOICES = ("face", "hands", "clothing", "background", "subject", "full")


def _positive_int(label: str, minimum: int = 1):
    def _checker(s: str) -> int:
        v = int(s, 10)
        if v < minimum:
            raise argparse.ArgumentTypeError(f"{label} must be >= {minimum}, got {v}")
        return v

    return _checker


def _unit_interval(s: str) -> float:
    v = float(s)
    if not 0.0 <= v <= 1.0:
        raise argparse.ArgumentTypeError(f"value must be in [0, 1], got {v}")
    return v


def main() -> int:
    ap = argparse.ArgumentParser(
        description="SDX: img2img or inpaint via sample.py (--init-image; optional mask or heuristic region).",
    )
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint .pt path")
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--out", type=str, required=True, help="Output image path")
    ap.add_argument("--negative-prompt", type=str, default="")
    ap.add_argument("--init-image", type=str, required=True, help="Base RGB image path")
    ap.add_argument(
        "--mask",
        type=str,
        default="",
        help="Grayscale mask (white = regenerate). Omit for whole-image img2img.",
    )
    ap.add_argument(
        "--heuristic-mask-region",
        type=str,
        default="",
        choices=["", *_HEURISTIC_CHOICES],
        metavar="REGION",
        help="Coarse inpaint mask (see utils/generation/edit_masks.py). Alternative to --mask.",
    )
    ap.add_argument("--width", type=_positive_int("width"), default=512)
    ap.add_argument("--height", type=_positive_int("height"), default=512)
    ap.add_argument("--steps", type=_positive_int("steps"), default=28)
    ap.add_argument("--cfg-scale", type=float, default=7.0)
    ap.add_argument("--strength", type=_unit_interval, default=0.65, help="Img2img / inpaint strength in [0, 1]")
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--scheduler", type=str, default="ddim")
    ap.add_argument("--solver", type=str, default="ddim")
    ap.add_argument(
        "--inpaint-mode",
        type=str,
        default="mdm",
        choices=("legacy", "mdm"),
        help="Used only when a mask is supplied",
    )
    ap.add_argument(
        "--segment-for",
        type=str,
        default="",
        help=(
            "Infer mask from phrase (Grounding DINO + SAM2 under pretrained/ when available; "
            "else keywords). Exclusive with --mask and --heuristic-mask-region."
        ),
    )
    ap.add_argument(
        "--no-segment-models",
        action="store_true",
        help="For --segment-for: heuristic phrase mapping only; skip DINO/SAM pipelines.",
    )
    ap.add_argument("--from-pillow", action="store_true", help="Use PIL round-trip temp files (same as API runner)")
    args = ap.parse_args()

    seed = None if args.seed < 0 else args.seed
    user_mask = str(args.mask or "").strip()
    heuristic = str(args.heuristic_mask_region or "").strip()
    segment_for = str(args.segment_for or "").strip()

    n_opts = sum(1 for x in (user_mask, heuristic, segment_for) if x)
    if n_opts > 1:
        ap.error("Use at most one of --mask, --heuristic-mask-region, --segment-for.")

    tmp_mask_dir = None
    mask_path_disk: Path | None = None
    try:
        if heuristic:
            tmp_mask_dir = tempfile.mkdtemp(prefix="sdx_inpaint_mask_")
            mask_path_disk = Path(tmp_mask_dir) / "mask.png"
            save_heuristic_mask(mask_path_disk, args.width, args.height, heuristic)
        elif segment_for:
            tmp_mask_dir = tempfile.mkdtemp(prefix="sdx_inpaint_mask_")
            mask_path_disk = Path(tmp_mask_dir) / "mask.png"
            guide = (
                Image.open(args.init_image).convert("RGB").resize((args.width, args.height), Image.Resampling.LANCZOS)
            )
            res = build_segmentation_mask_for_edit(
                guide,
                segment_for,
                feather_radius=4.0,
                use_vision_models=not args.no_segment_models,
            )
            res.mask.save(mask_path_disk)
        elif user_mask:
            mask_path_disk = Path(user_mask)

        if args.from_pillow:
            base = Image.open(args.init_image).convert("RGB")
            if heuristic:
                mask_img = heuristic_inpaint_mask(args.width, args.height, heuristic)
            elif segment_for:
                guide = base.resize((args.width, args.height), Image.Resampling.LANCZOS)
                mask_img = build_segmentation_mask_for_edit(
                    guide,
                    segment_for,
                    feather_radius=4.0,
                    use_vision_models=not args.no_segment_models,
                ).mask
            elif user_mask:
                mask_img = Image.open(user_mask).convert("L")
            else:
                mask_img = None
            pil_out = run_edit_with_pillow(
                ckpt=args.ckpt,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                base_image=base,
                mask_image=mask_img,
                width=args.width,
                height=args.height,
                steps=args.steps,
                cfg_scale=args.cfg_scale,
                seed=seed,
                img2img_strength=args.strength,
                inpaint_mode=args.inpaint_mode,
                device=args.device,
                scheduler=args.scheduler,
                solver=args.solver,
            )
            pil_out.save(args.out)
            print(f"Saved: {args.out}", flush=True)
            return 0

        run_sample_inference(
            ckpt=args.ckpt,
            prompt=args.prompt,
            out_path=args.out,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            seed=seed,
            device=args.device,
            init_image_path=args.init_image,
            mask_image_path=str(mask_path_disk) if mask_path_disk else None,
            strength=args.strength,
            inpaint_mode=args.inpaint_mode,
            scheduler=args.scheduler,
            solver=args.solver,
        )
        print(f"Saved: {args.out}", flush=True)
        return 0
    finally:
        if tmp_mask_dir:
            shutil.rmtree(tmp_mask_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

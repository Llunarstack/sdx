#!/usr/bin/env python3
"""
SDX one-command demo — download weights and generate a sample image.

    python demo.py

No checkpoint required: downloads a small DiT-XL/2 pretrained on ImageNet
from Hugging Face (facebookresearch/DiT), converts it to SDX format, and
runs sample.py with --holy-grail-preset auto.

Options
-------
--ckpt PATH         Use an existing SDX checkpoint instead of downloading.
--prompt TEXT       Override the default demo prompt.
--preset NAME       Holy Grail preset (auto|balanced|photoreal|anime|illustration|aggressive).
--steps N           Denoising steps (default: 40).
--cfg FLOAT         CFG scale (default: 6.5).
--out PATH          Output image path (default: demo_out.png).
--no-download       Skip download even if no local checkpoint found.
--device DEVICE     cuda or cpu (default: cuda if available).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

_DEMO_PROMPT = (
    "cinematic portrait of a young woman, soft window light, "
    "shallow depth of field, film grain, muted tones"
)
_DEMO_NEGATIVE = "blurry, low quality, watermark, text, oversaturated"
_HF_DIT_REPO = "facebookresearch/DiT"
_HF_DIT_FILE = "DiT-XL-2-256x256.pt"  # ~675 MB — smallest publicly available DiT-XL checkpoint
_PRETRAINED_DIR = Path(__file__).resolve().parent / "pretrained"
_DEMO_CKPT_DIR = _PRETRAINED_DIR / "dit-xl-2-imagenet"
_DEMO_CKPT = _DEMO_CKPT_DIR / "dit_xl_2_imagenet_sdx.pt"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_dit_imagenet(dest: Path) -> Path:
    """Download DiT-XL/2-256 from HF and return the local .pt path."""
    dest.mkdir(parents=True, exist_ok=True)
    raw = dest / _HF_DIT_FILE
    if raw.exists() and raw.stat().st_size > 1_000_000:
        print(f"  Already downloaded: {raw}")
        return raw
    print(f"  Downloading {_HF_DIT_REPO}/{_HF_DIT_FILE} …")
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=_HF_DIT_REPO,
            filename=_HF_DIT_FILE,
            local_dir=str(dest),
        )
        return Path(path)
    except Exception as e:
        raise RuntimeError(
            f"Download failed: {e}\n"
            "Install huggingface_hub:  pip install huggingface_hub\n"
            f"Or download manually from https://huggingface.co/{_HF_DIT_REPO} "
            f"and place {_HF_DIT_FILE} in {dest}"
        ) from e


def _convert_dit_imagenet_to_sdx(raw_pt: Path, out_pt: Path) -> Path:
    """
    Wrap a raw DiT-XL/2 ImageNet checkpoint in the SDX checkpoint format
    (adds a minimal config so load_dit_text_checkpoint can load it).

    The original DiT uses class conditioning (num_classes=1000), not text.
    We load it as DiT-XL/2 (class-conditioned) via the models.dit registry
    so sample.py can run it with --class-label for a quick smoke test.
    """
    import torch
    from config.train_config import TrainConfig

    if out_pt.exists() and out_pt.stat().st_size > 1_000_000:
        print(f"  Already converted: {out_pt}")
        return out_pt

    print(f"  Converting {raw_pt.name} → SDX format …")
    raw = torch.load(str(raw_pt), map_location="cpu", weights_only=False)

    # The raw DiT checkpoint is just a state dict (no 'model' key wrapper).
    state = raw if isinstance(raw, dict) and "x_embedder.proj.weight" in raw else raw.get("model", raw)

    # Minimal config — class-conditioned DiT-XL/2, no text encoder.
    cfg = TrainConfig(
        model_name="DiT-XL/2",
        image_size=256,
        text_encoder="",
        vae_model="stabilityai/sd-vae-ft-mse",
        num_timesteps=1000,
        beta_schedule="linear",
        prediction_type="epsilon",
    )

    sdx_ckpt = {
        "model": state,
        "ema": state,
        "config": cfg,
        "steps": 0,
    }
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sdx_ckpt, str(out_pt))
    print(f"  Saved: {out_pt}")
    return out_pt


def _ensure_checkpoint(args) -> tuple[Path, bool]:
    """Return (ckpt_path, is_text_conditioned)."""
    if args.ckpt:
        p = Path(args.ckpt)
        if not p.exists():
            sys.exit(f"Checkpoint not found: {p}")
        return p, True  # assume user checkpoints are text-conditioned

    if _DEMO_CKPT.exists() and _DEMO_CKPT.stat().st_size > 1_000_000:
        print(f"Using cached demo checkpoint: {_DEMO_CKPT}")
        return _DEMO_CKPT, False

    if args.no_download:
        sys.exit(
            "No checkpoint found and --no-download set.\n"
            f"Place a checkpoint at {_DEMO_CKPT} or pass --ckpt PATH."
        )

    print("No checkpoint found — downloading DiT-XL/2 ImageNet weights from HF …")
    raw = _download_dit_imagenet(_DEMO_CKPT_DIR)
    ckpt = _convert_dit_imagenet_to_sdx(raw, _DEMO_CKPT)
    return ckpt, False


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _run_class_conditioned(ckpt: Path, args) -> Path:
    """
    Sample from the class-conditioned DiT-XL/2 ImageNet checkpoint.
    Uses the diffusion engine directly (no T5 encoder needed).
    """
    import torch
    from diffusion import create_diffusion
    from models.dit import DiT_XL_2
    from PIL import Image

    device = torch.device(args.device)
    print(f"\nLoading checkpoint: {ckpt}")

    raw = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    state = raw.get("ema") or raw.get("model")

    # Build class-conditioned DiT-XL/2 (1000 ImageNet classes).
    model = DiT_XL_2(input_size=32, num_classes=1000)
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    diffusion = create_diffusion(
        timestep_respacing="",
        num_timesteps=1000,
        beta_schedule="linear",
        prediction_type="epsilon",
    )

    # Load VAE for decoding.
    print("Loading VAE …")
    from diffusers import AutoencoderKL
    vae_id = "stabilityai/sd-vae-ft-mse"
    try:
        vae = AutoencoderKL.from_pretrained(
            str(_PRETRAINED_DIR / "sd-vae-ft-mse")
            if (_PRETRAINED_DIR / "sd-vae-ft-mse").exists()
            else vae_id
        ).to(device).eval()
    except Exception:
        vae = AutoencoderKL.from_pretrained(vae_id).to(device).eval()

    # Class label 207 = golden retriever — a reliable ImageNet class.
    class_label = getattr(args, "class_label", 207)
    print(f"Sampling class label {class_label} ({args.steps} steps, CFG {args.cfg}) …")

    with torch.no_grad():
        # CFG: duplicate batch for cond/uncond.
        y = torch.tensor([class_label, 1000], device=device)  # 1000 = null class
        def model_fn(x, t, y):
            return model.forward_with_cfg(x, t, y, cfg_scale=args.cfg)

        diffusion.set_timesteps(args.steps, timestep_schedule="ddim")

        mk_cond = {"y": y[:1]}
        mk_uncond = {"y": y[1:]}

        x0 = diffusion.sample_loop(
            model,
            (1, 4, 32, 32),
            model_kwargs_cond=mk_cond,
            model_kwargs_uncond=mk_uncond,
            cfg_scale=args.cfg,
            num_inference_steps=args.steps,
            device=str(device),
            dtype=torch.float32,
        )

        # Decode.
        x0 = x0 / 0.18215
        img_t = vae.decode(x0).sample
        img_t = (img_t * 0.5 + 0.5).clamp(0, 1)
        img_np = img_t[0].permute(1, 2, 0).cpu().float().numpy()
        img_np = (img_np * 255).round().astype("uint8")

    out = Path(args.out)
    Image.fromarray(img_np).save(str(out))
    return out


def _run_text_conditioned(ckpt: Path, args) -> Path:
    """
    Run sample.py as a subprocess for a text-conditioned SDX checkpoint.
    This reuses all of sample.py's logic (LoRA, Holy Grail, etc.) without
    duplicating it here.
    """
    import subprocess

    cmd = [
        sys.executable, "sample.py",
        "--ckpt", str(ckpt),
        "--prompt", args.prompt,
        "--negative-prompt", _DEMO_NEGATIVE,
        "--steps", str(args.steps),
        "--cfg-scale", str(args.cfg),
        "--out", args.out,
        "--holy-grail-preset", args.preset,
    ]
    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"sample.py exited with code {result.returncode}")
    return Path(args.out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SDX one-command demo — download weights and generate a sample image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ckpt", default="", help="Path to an existing SDX checkpoint.")
    parser.add_argument("--prompt", default=_DEMO_PROMPT, help="Generation prompt.")
    parser.add_argument(
        "--preset",
        default="auto",
        choices=["auto", "balanced", "photoreal", "anime", "illustration", "aggressive"],
        help="Holy Grail adaptive sampling preset (default: auto).",
    )
    parser.add_argument("--steps", type=int, default=40, help="Denoising steps.")
    parser.add_argument("--cfg", type=float, default=6.5, help="CFG scale.")
    parser.add_argument("--out", default="demo_out.png", help="Output image path.")
    parser.add_argument("--no-download", action="store_true", help="Skip automatic weight download.")
    parser.add_argument(
        "--device",
        default="cuda" if __import__("torch").cuda.is_available() else "cpu",
        help="Device (default: cuda if available).",
    )
    parser.add_argument(
        "--class-label",
        type=int,
        default=207,
        help="ImageNet class label for the demo checkpoint (default: 207 = golden retriever).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  SDX demo")
    print("=" * 60)
    print(f"  Device : {args.device}")
    print(f"  Prompt : {args.prompt}")
    print(f"  Preset : {args.preset}")
    print(f"  Steps  : {args.steps}  CFG: {args.cfg}")
    print(f"  Output : {args.out}")
    print()

    ckpt, is_text = _ensure_checkpoint(args)

    if is_text:
        out = _run_text_conditioned(ckpt, args)
    else:
        print("Note: demo checkpoint is class-conditioned (ImageNet DiT-XL/2).")
        print("      Pass --ckpt to use your own text-conditioned SDX checkpoint.")
        print("      The Holy Grail preset applies only to text-conditioned checkpoints.\n")
        out = _run_class_conditioned(ckpt, args)

    print(f"\nDone — saved to {out}")
    print()
    print("Next steps:")
    print("  Train your own model:")
    print("    python train.py --data-path datasets/train --results-dir results")
    print()
    print("  Sample with your checkpoint:")
    print("    python sample.py --ckpt results/.../best.pt \\")
    print('      --prompt "your prompt" --holy-grail-preset auto --out out.png')
    print()
    print("  See README.md for the full guide.")


if __name__ == "__main__":
    main()

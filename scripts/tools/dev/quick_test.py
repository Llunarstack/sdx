#!/usr/bin/env python3
"""Smoke test: one forward pass (no training). Verifies imports, config, and model run."""

import os
import sys
from contextlib import redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Smoke test: one DiT forward pass.")
    ap.add_argument(
        "--show-native",
        action="store_true",
        help="Print optional native/ tool discovery (Rust, Zig, Go, CUDA DLLs, libsdx_latent) and exit 0.",
    )
    args, _unknown = ap.parse_known_args()
    if args.show_native:
        from utils.native.native_tools import native_stack_status
        import json as _json

        print(_json.dumps(native_stack_status(), indent=2))
        return 0

    # Suppress xformers/Triton noise on Windows (Triton not available there)
    with open(os.devnull, "w") as devnull:
        with redirect_stderr(devnull):
            import torch
            from diffusion import create_diffusion
            from models import DiT_models_text

            from config import TrainConfig, get_dit_build_kwargs

    cfg = TrainConfig(
        data_path="",
        image_size=256,
        model_name="DiT-XL/2-Text",
        text_encoder="google/t5-v1_1-xxl",
        vae_model="stabilityai/sd-vae-ft-mse",
    )
    latent_size = cfg.image_size // 8
    kwargs = get_dit_build_kwargs(cfg, class_dropout_prob=0.0)
    model_fn = DiT_models_text.get(cfg.model_name) or DiT_models_text["DiT-XL/2-Text"]
    model = model_fn(**kwargs)
    model.eval()

    B, C, H, W = 2, 4, latent_size, latent_size
    x = torch.randn(B, C, H, W)
    t = torch.randint(0, 1000, (B,))
    text_dim = kwargs["text_dim"]
    y = torch.randn(B, 77, text_dim)

    with torch.no_grad():
        out = model(x, t, encoder_hidden_states=y)

    assert out.ndim == 4 and out.shape[0] == B and out.shape[2] == H and out.shape[3] == W, out.shape
    diffusion = create_diffusion(num_timesteps=1000)
    assert diffusion.num_timesteps == 1000

    print("OK: imports, config, DiT forward, diffusion.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

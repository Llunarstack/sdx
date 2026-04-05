#!/usr/bin/env python3
"""
Stable Cascade (diffusers) — optional path separate from DiT+T5 sampling.

Requires: pip install diffusers (and models under model/StableCascade-*).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from utils.modeling.model_paths import default_cascade_decoder_path, default_cascade_prior_path

    p = argparse.ArgumentParser(description="Stable Cascade image generation (Diffusers)")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--out", type=str, default="cascade_out.png")
    p.add_argument("--prior", type=str, default="", help="Prior model path (empty = default)")
    p.add_argument("--decoder", type=str, default="", help="Decoder model path (empty = default)")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    try:
        import torch
        from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
    except ImportError:
        print("Install diffusers and torch: pip install diffusers", file=sys.stderr)
        return 1

    prior_id = args.prior or default_cascade_prior_path()
    dec_id = args.decoder or default_cascade_decoder_path()
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading prior: {prior_id}")
    prior = StableCascadePriorPipeline.from_pretrained(prior_id, torch_dtype=dtype).to(device)
    print(f"Loading decoder: {dec_id}")
    decoder = StableCascadeDecoderPipeline.from_pretrained(dec_id, torch_dtype=dtype).to(device)

    prompt = args.prompt
    prior_output = prior(
        prompt=prompt,
        height=1024,
        width=1024,
        prior_guidance_scale=4.0,
        num_inference_steps=20,
    )
    latents = prior_output.image_embeddings
    decoder_output = decoder(
        image_embeddings=latents,
        prompt=prompt,
        num_inference_steps=10,
        guidance_scale=0.0,
        output_type="pil",
    ).images[0]
    decoder_output.save(args.out)
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

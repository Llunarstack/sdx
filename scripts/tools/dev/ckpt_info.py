#!/usr/bin/env python3
"""Print checkpoint config and basic info without loading the full model (fast)."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Inspect checkpoint: config, keys, step count.")
    parser.add_argument("ckpt", type=str, help="Path to .pt checkpoint")
    parser.add_argument("--keys", action="store_true", help="List top-level keys in checkpoint")
    args = parser.parse_args()

    path = Path(args.ckpt)
    if not path.exists():
        print(f"Not found: {path}", file=sys.stderr)
        return 1

    ckpt = __import__("torch").load(path, map_location="cpu", weights_only=False)
    if args.keys:
        print("Keys:", list(ckpt.keys()))
        return 0

    cfg = ckpt.get("config")
    if cfg is None:
        print("No config in checkpoint.")
        return 0

    # Mirror get_dit_build_kwargs / TrainConfig fields
    attrs = [
        "model_name",
        "image_size",
        "text_encoder",
        "vae_model",
        "latent_scale",
        "num_ar_blocks",
        "use_xformers",
        "style_embed_dim",
        "control_cond_dim",
        "creativity_embed_dim",
        "creativity_jitter_std",
        "train_originality_augment_prob",
        "train_originality_strength",
        "num_timesteps",
        "beta_schedule",
        "prediction_type",
    ]
    print("Config:")
    for a in attrs:
        v = getattr(cfg, a, None)
        if v is not None:
            print(f"  {a}: {v}")
    steps = ckpt.get("steps")
    if steps is not None:
        print(f"  steps: {steps}")
    best_loss = ckpt.get("best_loss")
    if best_loss is not None:
        print(f"  best_loss: {best_loss:.4f}")
    state = ckpt.get("model") or ckpt.get("ema")
    if state is not None:
        n = len(state)
        print(f"  state_dict keys: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

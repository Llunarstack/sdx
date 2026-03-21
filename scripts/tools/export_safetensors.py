#!/usr/bin/env python3
"""
Export a .pt checkpoint's DiT weights to .safetensors for ComfyUI, A1111, or other loaders.
Saves only the model state_dict (ema or model). Optional: --metadata to embed config fields as JSON.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Export .pt checkpoint DiT weights to .safetensors.")
    parser.add_argument("ckpt", type=str, help="Path to .pt checkpoint")
    parser.add_argument(
        "--out", type=str, default="", help="Output path (default: same stem as ckpt with .safetensors)"
    )
    parser.add_argument("--metadata", action="store_true", help="Embed config fields as safetensors metadata")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"Not found: {ckpt_path}", file=sys.stderr)
        return 1

    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as e:
        print(f"Need torch and safetensors: {e}", file=sys.stderr)
        return 1

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("ema") or ckpt.get("model")
    if state is None:
        print("Checkpoint has no 'ema' or 'model' state_dict.", file=sys.stderr)
        return 1

    out_path = Path(args.out) if args.out else ckpt_path.with_suffix(".safetensors")
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = None
    if args.metadata:
        cfg = ckpt.get("config")
        if cfg is not None:
            meta = {}
            for key in ("model_name", "image_size", "text_encoder", "vae_model", "num_timesteps", "prediction_type"):
                v = getattr(cfg, key, None)
                if v is not None:
                    meta[key] = str(v) if not isinstance(v, (int, float, bool)) else v
            if meta:
                metadata = {"sdx_config": json.dumps(meta)}

    save_file(state, str(out_path), metadata=metadata)
    print(f"Saved {len(state)} keys to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

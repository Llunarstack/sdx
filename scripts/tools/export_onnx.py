#!/usr/bin/env python3
"""
Export a .pt checkpoint's DiT to ONNX for fast deployment (e.g. TensorRT, ONNX Runtime).
Exports the model forward: x, t, encoder_hidden_states -> noise prediction.
Run from repo root: python scripts/tools/export_onnx.py path/to/best.pt [--out dit.onnx]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Export DiT from .pt checkpoint to ONNX.")
    parser.add_argument("ckpt", type=str, help="Path to .pt checkpoint")
    parser.add_argument("--out", type=str, default="", help="Output path (default: same stem as ckpt with .onnx)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for export (dynamic if --dynamic-batch)")
    parser.add_argument("--dynamic-batch", action="store_true", help="Use dynamic batch and seq length axes")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"Not found: {ckpt_path}", file=sys.stderr)
        return 1

    import torch
    from models import DiT_models_text

    from config import get_dit_build_kwargs

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config")
    if cfg is None:
        print("Checkpoint has no config.", file=sys.stderr)
        return 1
    model_name = getattr(cfg, "model_name", "DiT-XL/2-Text")
    model_fn = DiT_models_text.get(model_name) or DiT_models_text["DiT-XL/2-Text"]
    model = model_fn(**get_dit_build_kwargs(cfg, class_dropout_prob=0.0))
    state = ckpt.get("ema") or ckpt.get("model")
    model.load_state_dict(state, strict=True)
    model.eval()

    image_size = getattr(cfg, "image_size", 256)
    latent_size = image_size // 8
    te = getattr(cfg, "text_encoder", "").lower()
    text_dim = 4096 if "xxl" in te else (1024 if "xl" in te and "xxl" not in te else 768)
    seq_len = 77  # typical max length

    batch = args.batch
    x = torch.randn(batch, 4, latent_size, latent_size)
    t = torch.zeros(batch, dtype=torch.long)
    enc = torch.randn(batch, seq_len, text_dim)

    out_path = Path(args.out) if args.out else ckpt_path.with_suffix(".onnx")
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "x": {0: "batch"},
            "t": {0: "batch"},
            "encoder_hidden_states": {0: "batch"},
            "out": {0: "batch"},
        }

    with torch.no_grad():
        torch.onnx.export(
            model,
            (x, t, enc),
            str(out_path),
            input_names=["x", "t", "encoder_hidden_states"],
            output_names=["out"],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
        )
    print(f"Exported to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

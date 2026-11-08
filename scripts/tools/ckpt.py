#!/usr/bin/env python3
"""Checkpoint / model inspection and export commands.

    python -m scripts.tools ckpt info results/run/best.pt [--keys]
    python -m scripts.tools ckpt vit-inspect vit_quality/best.pt [--tree --tree-depth 3 | --json]
    python -m scripts.tools ckpt export-onnx best.pt [--out dit.onnx --dynamic-batch]
    python -m scripts.tools ckpt export-safetensors best.pt [--metadata]
    python -m scripts.tools ckpt soup --checkpoints a.pt b.pt --template a.pt --out soup.pt
    python -m scripts.tools ckpt noise-schedule linear --steps 1000 --out sched.csv
    python -m scripts.tools ckpt ar-mask --h 16 --w 16 --blocks 2 --order zorder --compare
    python -m scripts.tools ckpt dit-compare --image-size 512 --models DiT-B/2-Text,DiT-XL/2-Text
    python -m scripts.tools ckpt timestep-preview --modes uniform,logit_normal --samples 200000

Old names (``ckpt_info``, ``vit_inspect``, ``export_onnx``, ``export_safetensors``, ``model_soup``,
``noise_schedule_export``, ``ar_mask_inspect``, ``dit_variant_compare``, ``training_timestep_preview``)
still work as ``python -m scripts.tools <old_name>``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_PROG = "python -m scripts.tools ckpt"


def _load_ckpt(path_str: str):
    """``(path, checkpoint dict)`` or ``(path, None)`` after printing a not-found error."""
    path = Path(path_str)
    if not path.exists():
        print(f"Not found: {path}", file=sys.stderr)
        return path, None
    import torch

    return path, torch.load(path, map_location="cpu", weights_only=False)


def _info_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog=f"{_PROG} info", description="Inspect checkpoint: config, keys, step count.")
    parser.add_argument("ckpt", type=str, help="Path to .pt checkpoint")
    parser.add_argument("--keys", action="store_true", help="List top-level keys in checkpoint")
    args = parser.parse_args(argv)

    _, ckpt = _load_ckpt(args.ckpt)
    if ckpt is None:
        return 1
    if args.keys:
        print("Keys:", list(ckpt.keys()))
        return 0

    cfg = ckpt.get("config")
    if cfg is None:
        print("No config in checkpoint.")
        return 0

    attrs = (
        "model_name",
        "image_size",
        "text_encoder",
        "vae_model",
        "latent_scale",
        "num_ar_blocks",
        "ar_block_order",
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
    )
    print("Config:")
    for a in attrs:
        v = getattr(cfg, a, None)
        if v is not None:
            print(f"  {a}: {v}")
    if ckpt.get("steps") is not None:
        print(f"  steps: {ckpt['steps']}")
    if ckpt.get("best_loss") is not None:
        print(f"  best_loss: {ckpt['best_loss']:.4f}")
    state = ckpt.get("model") or ckpt.get("ema")
    if state is not None:
        print(f"  state_dict keys: {len(state)}")
    return 0


def _vit_inspect_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog=f"{_PROG} vit-inspect", description="Inspect ViT quality checkpoint (params + config)."
    )
    p.add_argument("ckpt", type=str, help="Path to ViT best.pt / last.pt")
    p.add_argument("--json", action="store_true", help="Print config as JSON")
    p.add_argument("--tree", action="store_true", help="Print module tree (see utils/training/monitoring.py)")
    p.add_argument("--tree-depth", type=int, default=3)
    p.add_argument("--use-ema", action="store_true", help="Load EMA weights if present")
    args = p.parse_args(argv)

    from utils.training.monitoring import format_module_tree
    from vit_quality.model import load_vit_quality_checkpoint, vit_model_parameter_report

    path = Path(args.ckpt)
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr)
        return 1

    model, cfg = load_vit_quality_checkpoint(path, use_ema=args.use_ema)
    rep = vit_model_parameter_report(model)

    if args.json:
        out = dict(cfg)
        out["total_parameters"] = rep["total_parameters"]
        out["trainable_parameters"] = rep["trainable_parameters"]
        print(json.dumps(out, indent=2, default=str))
        return 0

    print(f"Checkpoint: {path.resolve()}")
    print(f"Parameters: {rep['total_parameters']:,} total, {rep['trainable_parameters']:,} trainable")
    print("Config:")
    for k in sorted(cfg.keys()):
        print(f"  {k}: {cfg[k]}")
    if args.tree:
        print("\nModule tree:")
        for line in format_module_tree(model, max_depth=int(args.tree_depth)):
            print(line)
    return 0


def _export_onnx_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog=f"{_PROG} export-onnx", description="Export DiT from .pt checkpoint to ONNX.")
    parser.add_argument("ckpt", type=str, help="Path to .pt checkpoint")
    parser.add_argument("--out", type=str, default="", help="Output path (default: same stem as ckpt with .onnx)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for export (dynamic if --dynamic-batch)")
    parser.add_argument("--dynamic-batch", action="store_true", help="Use dynamic batch and seq length axes")
    args = parser.parse_args(argv)

    ckpt_path, ckpt = _load_ckpt(args.ckpt)
    if ckpt is None:
        return 1
    cfg = ckpt.get("config")
    if cfg is None:
        print("Checkpoint has no config.", file=sys.stderr)
        return 1

    import torch
    from config import get_dit_build_kwargs
    from models import DiT_models_text

    model_fn = DiT_models_text.get(getattr(cfg, "model_name", "DiT-XL/2-Text")) or DiT_models_text["DiT-XL/2-Text"]
    build_kw = get_dit_build_kwargs(cfg, class_dropout_prob=0.0)
    model = model_fn(**build_kw)
    model.load_state_dict(ckpt.get("ema") or ckpt.get("model"), strict=True)
    model.eval()

    latent_size = getattr(cfg, "image_size", 256) // 8
    batch = args.batch
    x = torch.randn(batch, int(build_kw.get("in_channels", 4)), latent_size, latent_size)
    t = torch.zeros(batch, dtype=torch.long)
    enc = torch.randn(batch, 77, build_kw["text_dim"])

    out_path = (Path(args.out) if args.out else ckpt_path.with_suffix(".onnx")).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {k: {0: "batch"} for k in ("x", "t", "encoder_hidden_states", "out")}

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


def _export_safetensors_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog=f"{_PROG} export-safetensors", description="Export .pt checkpoint DiT weights to .safetensors."
    )
    parser.add_argument("ckpt", type=str, help="Path to .pt checkpoint")
    parser.add_argument(
        "--out", type=str, default="", help="Output path (default: same stem as ckpt with .safetensors)"
    )
    parser.add_argument("--metadata", action="store_true", help="Embed config fields as safetensors metadata")
    args = parser.parse_args(argv)

    try:
        from safetensors.torch import save_file
    except ImportError as e:
        print(f"Need torch and safetensors: {e}", file=sys.stderr)
        return 1
    ckpt_path, ckpt = _load_ckpt(args.ckpt)
    if ckpt is None:
        return 1
    state = ckpt.get("ema") or ckpt.get("model")
    if state is None:
        print("Checkpoint has no 'ema' or 'model' state_dict.", file=sys.stderr)
        return 1

    out_path = (Path(args.out) if args.out else ckpt_path.with_suffix(".safetensors")).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = None
    cfg = ckpt.get("config") if args.metadata else None
    if cfg is not None:
        meta = {}
        for key in ("model_name", "image_size", "text_encoder", "vae_model", "num_timesteps", "prediction_type"):
            v = getattr(cfg, key, None)
            if v is not None:
                meta[key] = v if isinstance(v, (int, float, bool)) else str(v)
        if meta:
            metadata = {"sdx_config": json.dumps(meta)}

    save_file(state, str(out_path), metadata=metadata)
    print(f"Saved {len(state)} keys to {out_path}")
    return 0


def _soup_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog=f"{_PROG} soup", description="Average multiple DiT checkpoints into one soup weights file."
    )
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--weights", nargs="*", type=float, default=None)
    p.add_argument("--template", required=True, help="Checkpoint for config/metadata template")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    from utils.agentic.quality_loop import save_soup_checkpoint, soup_checkpoints

    avg = soup_checkpoints(args.checkpoints, weights=args.weights)
    save_soup_checkpoint(avg, template_path=args.template, out_path=args.out)
    print(f"Saved model soup ({len(args.checkpoints)} ckpts) -> {args.out}")
    return 0


def _noise_schedule_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog=f"{_PROG} noise-schedule", description="Noise schedule CSV (Rust sdx-noise-schedule)."
    )
    p.add_argument("mode", choices=("linear", "cosine"), help="Schedule family")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--beta-start", type=float, default=1e-4, help="linear only")
    p.add_argument("--beta-end", type=float, default=2e-2, help="linear only")
    p.add_argument("--s", type=float, default=0.008, help="cosine only (offset s)")
    p.add_argument("-o", "--out", type=str, default="", help="Write CSV (default: stdout)")
    args = p.parse_args(argv)

    import utils.native  # noqa: F401  (puts native/_experimental/python on sys.path)

    from sdx_native.native_tools import run_rust_noise_schedule, rust_noise_schedule_exe

    if rust_noise_schedule_exe() is None:
        print(
            "sdx-noise-schedule not built. Run: .\\scripts\\tools\\native\\build_native.ps1 "
            "or: (cd native/rust/sdx-noise-schedule && cargo build --release)",
            file=sys.stderr,
        )
        return 2

    if args.mode == "linear":
        cmd = ["linear", "--steps", str(args.steps), "--beta-start", str(args.beta_start)]
        cmd += ["--beta-end", str(args.beta_end)]
    else:
        cmd = ["cosine", "--steps", str(args.steps), "--s", str(args.s)]
    r = run_rust_noise_schedule(cmd)
    if r.returncode != 0:
        print(r.stderr or r.stdout or "noise-schedule failed", file=sys.stderr)
        return r.returncode or 1
    if args.out:
        Path(args.out).write_text(r.stdout or "", encoding="utf-8")
    else:
        sys.stdout.write(r.stdout or "")
    return 0


def _ar_mask_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog=f"{_PROG} ar-mask", description="Inspect block-causal AR mask.")
    ap.add_argument("--h", type=int, default=16, help="Patch grid height")
    ap.add_argument("--w", type=int, default=16, help="Patch grid width")
    ap.add_argument("--blocks", type=int, default=2, help="Macro-blocks per side (0=full, 2 or 4 typical)")
    ap.add_argument(
        "--order",
        type=str,
        default="raster",
        choices=("raster", "zorder", "snake", "spiral"),
        help="Macro-block visit order",
    )
    ap.add_argument("--compare", action="store_true", help="Also print zorder if order=raster (or vice versa)")
    args = ap.parse_args(argv)

    if args.blocks <= 0:
        print("num_ar_blocks<=0: full attention (no mask).")
        return 0

    import torch
    from models.attention import ar_mask_sparsity_stats, block_visit_order, create_block_causal_mask_2d

    m = create_block_causal_mask_2d(args.h, args.w, args.blocks, block_order=args.order)
    frac, n = ar_mask_sparsity_stats(m)
    print(f"grid={args.h}x{args.w} blocks_per_side={args.blocks} order={args.order!r}")
    print(f"mask shape=({n},{n})  allowed_pair_fraction={frac:.4f}")
    print(f"macro-block visit: {block_visit_order(args.blocks, args.order)}")

    if args.compare:
        other = {"raster": "zorder", "zorder": "snake", "snake": "spiral", "spiral": "raster"}[args.order]
        m2 = create_block_causal_mask_2d(args.h, args.w, args.blocks, block_order=other)
        frac2, _ = ar_mask_sparsity_stats(m2)
        print(f"compare order={other!r} allowed_pair_fraction={frac2:.4f} same_tensor={bool(torch.equal(m, m2))}")
    return 0


def _dit_compare_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog=f"{_PROG} dit-compare", description="Compare DiT variant parameter counts.")
    parser.add_argument("--image-size", type=int, default=256, help="Training image side (latent = /8).")
    parser.add_argument("--text-dim", type=int, default=4096, help="Text embedding dim (default T5-XXL-style 4096).")
    parser.add_argument("--models", type=str, default="", help="Comma-separated DiT_models_text names; empty = all.")
    parser.add_argument("--vae-scale", type=int, default=8, help="Latent scale vs pixels (SD-style VAE: 8).")
    args = parser.parse_args(argv)

    from utils.modeling.architecture import dit_parameter_report, instantiate_dit_text, list_all_dit_registry_names
    from utils.native import dit_patch_size_from_variant_name, get_latent_lib

    names = [x.strip() for x in args.models.split(",") if x.strip()] or list_all_dit_registry_names()
    rows = []
    for name in names:
        try:
            r = dit_parameter_report(instantiate_dit_text(name, image_size=args.image_size, text_dim=args.text_dim))
            rows.append((name, "EnhancedDiT" if name.startswith("EnhancedDiT") else "DiT-text", r))
        except Exception as e:
            rows.append((name, "?", {"error": str(e)}))

    lib = get_latent_lib()
    print(
        f"image_size={args.image_size}  latent_side={args.image_size // args.vae_scale}  vae_scale={args.vae_scale}  "
        f"text_dim={args.text_dim}  libsdx_latent={'yes' if lib.available else 'no (pure Python math)'}\n"
    )
    print(f"{'Model':<36} {'Kind':<14} {'Params':>14} {'FP32 GiB':>10} {'BF16 GiB':>10}  patch  patch_tokens")
    print("-" * 108)
    for name, kind, r in rows:
        if "error" in r:
            print(f"{name:<36} {kind:<14} ERROR: {r['error']}")
            continue
        ps = dit_patch_size_from_variant_name(name)
        npt = lib.num_patch_tokens(args.image_size, args.vae_scale, ps)
        print(
            f"{name:<36} {kind:<14} {r['total_parameters']:>14,} "
            f"{r['size_fp32_gib']:>10.2f} {r['size_bf16_gib']:>10.2f}  "
            f"patch={ps}  patch_tokens={npt}"
        )
    return 0


def _timestep_preview_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog=f"{_PROG} timestep-preview", description="Histogram preview for training timestep sampling."
    )
    parser.add_argument("--num-timesteps", type=int, default=1000, help="Discrete diffusion steps T (e.g. 1000).")
    parser.add_argument("--samples", type=int, default=100_000, help="Monte Carlo sample count.")
    parser.add_argument(
        "--modes",
        type=str,
        default="uniform,logit_normal,high_noise,low_noise",
        help="Comma-separated: uniform | logit_normal | high_noise | low_noise",
    )
    parser.add_argument("--logit-mean", type=float, default=0.0)
    parser.add_argument("--logit-std", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bins", type=int, default=10, help="Number of histogram bins over [0, T-1].")
    args = parser.parse_args(argv)

    import torch
    from diffusion.schedules import sample_training_timesteps

    torch.manual_seed(args.seed)
    T, B, n_bins = int(args.num_timesteps), int(args.samples), int(args.bins)
    if T < 1 or B < 1:
        raise SystemExit("--num-timesteps and --samples must be >= 1")
    # Indices are in [0, T-1]; histogram spans [0, T) so the last bin includes T-1.
    edges = torch.linspace(0, float(T), n_bins + 1).tolist()

    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        tf = sample_training_timesteps(
            B, T, device=torch.device("cpu"), mode=mode, logit_mean=args.logit_mean, logit_std=args.logit_std
        ).float()
        print(f"\n=== mode={mode!r}  (T={T}, n={B}) ===")
        print(f"  mean(t)={tf.mean().item():.2f}  std={tf.std().item():.2f}")
        q = torch.quantile(tf, q=torch.tensor([0.05, 0.5, 0.95]))
        print(f"  quantiles p5/p50/p95: {[round(x, 1) for x in q.tolist()]}")
        hist = torch.histc(tf, bins=n_bins, min=0.0, max=float(T)).tolist()
        mx = max(hist) or 1.0
        print(f"  histogram ({n_bins} bins over [0, {T})):")
        for i, h in enumerate(hist):
            print(
                f"    [{edges[i]:6.1f}, {edges[i + 1]:6.1f})  {100.0 * h / B:5.1f}%  {'#' * max(1, int(40 * h / mx))}"
            )
    return 0


COMMANDS = {
    "info": _info_main,
    "vit-inspect": _vit_inspect_main,
    "export-onnx": _export_onnx_main,
    "export-safetensors": _export_safetensors_main,
    "soup": _soup_main,
    "noise-schedule": _noise_schedule_main,
    "ar-mask": _ar_mask_main,
    "dit-compare": _dit_compare_main,
    "timestep-preview": _timestep_preview_main,
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        print("Commands: " + ", ".join(COMMANDS))
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    return int(COMMANDS[argv[0]](argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())

"""
Preview **training timestep index** distributions (uniform vs logit-normal vs high-noise).

Useful when tuning `--timestep-sample-mode` / FasterDiT-style SNR–timestep thinking: you can
see how often the model will see small vs large `t` before starting a long run.

Usage (from repo root):
    python scripts/tools/training_timestep_preview.py
    python scripts/tools/training_timestep_preview.py --modes uniform,logit_normal,high_noise --samples 200000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from diffusion.timestep_sampling import sample_training_timesteps


def main() -> None:
    parser = argparse.ArgumentParser(description="Histogram preview for training timestep sampling.")
    parser.add_argument("--num-timesteps", type=int, default=1000, help="Discrete diffusion steps T (e.g. 1000).")
    parser.add_argument("--samples", type=int, default=100_000, help="Monte Carlo sample count.")
    parser.add_argument(
        "--modes",
        type=str,
        default="uniform,logit_normal,high_noise",
        help="Comma-separated: uniform | logit_normal | high_noise",
    )
    parser.add_argument("--logit-mean", type=float, default=0.0)
    parser.add_argument("--logit-std", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bins", type=int, default=10, help="Number of histogram bins over [0, T-1].")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    T = int(args.num_timesteps)
    B = int(args.samples)
    if T < 1 or B < 1:
        raise SystemExit("--num-timesteps and --samples must be >= 1")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    n_bins = int(args.bins)
    # Indices are in [0, T-1]; histogram spans [0, T) so the last bin includes T-1.
    bin_edges = torch.linspace(0, float(T), n_bins + 1)

    for mode in modes:
        t = sample_training_timesteps(
            B,
            T,
            device=device,
            mode=mode,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
        )
        tf = t.float()
        print(f"\n=== mode={mode!r}  (T={T}, n={B}) ===")
        print(f"  mean(t)={tf.mean().item():.2f}  std={tf.std().item():.2f}")
        q = torch.quantile(tf, q=torch.tensor([0.05, 0.5, 0.95]))
        print(f"  quantiles p5/p50/p95: {[round(x, 1) for x in q.tolist()]}")
        hist = torch.histc(tf, bins=n_bins, min=0.0, max=float(T))
        mx = float(hist.max().item()) or 1.0
        print(f"  histogram ({n_bins} bins over [0, {T})):")
        for i in range(n_bins):
            lo, hi = bin_edges[i].item(), bin_edges[i + 1].item()
            bar = "#" * max(1, int(40 * hist[i].item() / mx))
            pct = 100.0 * hist[i].item() / B
            print(f"    [{lo:6.1f}, {hi:6.1f})  {pct:5.1f}%  {bar}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Train VideoDiT on short latent clips with flow-matching v2.

Scaffold entry — expects precomputed latents ``(C, T, H, W)`` in a folder or .pt
files. Full VAE encode pipeline can wrap this later.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latents-dir", type=str, required=True, help="Directory of .pt latent clips (C,T,H,W)")
    p.add_argument("--out", type=str, default="results/video_dit/best.pt")
    p.add_argument("--model", type=str, default="PermanentVideoDiT-S/2")
    p.add_argument("--steps", type=int, default=100, help="Optimizer steps (smoke default)")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--context-dim", type=int, default=768)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--permanence-weight", type=float, default=0.1, help="Slot-memory consistency loss weight")
    p.add_argument("--dry-run", action="store_true", help="Build model + one loss step then exit")
    args = p.parse_args()

    import torch
    from diffusion.flow_matching import flow_matching_per_sample_losses_v2
    from models.permanent_video_dit import permanence_consistency_loss
    from pipelines.video.neural_sample import load_video_dit

    device = torch.device(args.device)
    permanent = "Permanent" in str(args.model)
    model = load_video_dit(
        None, model_name=args.model, context_dim=args.context_dim, device=device, permanent=permanent
    )
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    latent_paths = sorted(Path(args.latents_dir).glob("*.pt"))
    if not latent_paths:
        # Synthetic smoke batch
        x0 = torch.randn(args.batch, 4, 8, 16, 16, device=device)
    else:
        clips = [torch.load(p, map_location=device, weights_only=True) for p in latent_paths[: args.batch]]
        x0 = torch.stack([c if c.ndim == 4 else c[0] for c in clips], dim=0).to(device)

    context = torch.randn(x0.shape[0], args.context_dim, device=device)

    n_steps = 1 if args.dry_run else max(1, int(args.steps))
    for step in range(n_steps):
        opt.zero_grad(set_to_none=True)
        eps = torch.randn_like(x0)
        # Pooled context in model_kwargs for VideoDiT(context=...)
        loss = flow_matching_per_sample_losses_v2(
            model,
            x0,
            eps,
            1000,
            {"context": context},
            shift=3.0,
        ).mean()
        if permanent and float(args.permanence_weight) > 0:
            # Extra permanence pass on clean latents (slot trajectory smoothness)
            t0 = torch.zeros(x0.shape[0], device=device)
            out = model(x0, t0, context=context, return_slots=True)
            if isinstance(out, tuple):
                _, slots = out
                loss = loss + float(args.permanence_weight) * permanence_consistency_loss(slots)
        loss.backward()
        opt.step()
        if step % 10 == 0 or args.dry_run:
            print(f"step={step} loss={float(loss):.6f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_name": args.model}, out)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

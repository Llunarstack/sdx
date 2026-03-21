#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    from ViT.backbone_presets import describe_presets_for_help
    from ViT.config import ViTConfig
    from ViT.dataset import ViTManifestDataset, collate_vit_batch
    from ViT.ema import ModelEMA
    from ViT.losses import pairwise_ranking_loss
    from ViT.model import build_vit_model

    p = argparse.ArgumentParser(
        description="Train ViT quality+adherence model on SDX manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(describe_presets_for_help()),
    )
    p.add_argument("--manifest-jsonl", required=True)
    p.add_argument("--image-root", default="")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--model-name",
        default="vit_base_patch16_224",
        help="timm backbone for ViTQualityAdherenceModel (see ViT/backbone_presets.py, ViT/EXCELLENCE_VS_DIT.md).",
    )
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--quality-loss-weight", type=float, default=1.0)
    p.add_argument("--adherence-loss-weight", type=float, default=1.0)
    p.add_argument("--ranking-loss-weight", type=float, default=0.0)
    p.add_argument("--ranking-margin", type=float, default=0.15)
    p.add_argument("--ranking-min-gap", type=float, default=0.05)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--save-ema", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="vit_runs")
    args = p.parse_args()

    cfg = ViTConfig(
        manifest_jsonl=args.manifest_jsonl,
        image_root=args.image_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_name=args.model_name,
        pretrained=not args.no_pretrained,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        quality_loss_weight=args.quality_loss_weight,
        adherence_loss_weight=args.adherence_loss_weight,
        device=args.device,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    seed_all(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ds = ViTManifestDataset(cfg.manifest_jsonl, image_root=cfg.image_root, image_size=cfg.image_size)
    if len(ds) == 0:
        print("Dataset is empty after filtering valid rows.", file=sys.stderr)
        return 2
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_vit_batch,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_vit_model(
        cfg.model_name, pretrained=cfg.pretrained, text_feat_dim=cfg.text_feat_dim, hidden_dim=cfg.hidden_dim
    )
    model.to(device)
    ema = ModelEMA(model, decay=float(args.ema_decay))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    bce = torch.nn.BCEWithLogitsLoss()
    mse = torch.nn.MSELoss()

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history = []

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        n = 0
        for b in dl:
            images = b["images"].to(device, non_blocking=True)
            txt = b["text_features"].to(device, non_blocking=True)
            out = model(images, txt)

            loss = 0.0
            q = b["quality_labels"]
            a = b["adherence_scores"]
            if q is not None:
                q = q.to(device, non_blocking=True)
                loss = loss + cfg.quality_loss_weight * bce(out["quality_logit"], q)
                if args.ranking_loss_weight > 0:
                    loss = loss + float(args.ranking_loss_weight) * pairwise_ranking_loss(
                        out["quality_logit"], q, margin=args.ranking_margin, min_target_gap=args.ranking_min_gap
                    )
            if a is not None:
                a = a.to(device, non_blocking=True)
                loss = loss + cfg.adherence_loss_weight * mse(out["adherence_score"], a)
                if args.ranking_loss_weight > 0:
                    loss = loss + float(args.ranking_loss_weight) * pairwise_ranking_loss(
                        out["adherence_score"], a, margin=args.ranking_margin, min_target_gap=args.ranking_min_gap
                    )

            if isinstance(loss, float):
                # No labels present in this manifest.
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            ema.update(model)

            running += float(loss.item())
            n += 1

        avg_loss = running / max(1, n)
        history.append({"epoch": epoch + 1, "loss": avg_loss})
        print(f"[ViT] epoch {epoch + 1}/{cfg.epochs} loss={avg_loss:.6f}")

        ckpt = {
            "state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "config": vars(cfg),
            "epoch": epoch + 1,
            "loss": avg_loss,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, out_dir / "best.pt")
            if args.save_ema:
                model_ema = build_vit_model(
                    cfg.model_name, pretrained=False, text_feat_dim=cfg.text_feat_dim, hidden_dim=cfg.hidden_dim
                )
                model_ema.to(device)
                ema.copy_to(model_ema)
                torch.save(
                    {
                        "state_dict": model_ema.state_dict(),
                        "config": vars(cfg),
                        "epoch": epoch + 1,
                        "loss": avg_loss,
                        "is_ema": True,
                    },
                    out_dir / "best_ema.pt",
                )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[ViT] done. best_loss={best_loss:.6f} out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Save a randomly initialized Enhanced DiT-XL/2 checkpoint (all enhanced features on) as a training start point."""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
from config.train_config import TrainConfig  # noqa: E402
from models.enhanced_dit import EnhancedDiT_XL_2  # noqa: E402

_FEATURES = ("spatial_control", "anatomy_awareness", "text_rendering", "consistency")


def save_model_checkpoint(out: Path, *, image_size: int = 512) -> Path:
    kwargs = {f"enable_{f}": True for f in _FEATURES}
    model = EnhancedDiT_XL_2(input_size=image_size // 8, **kwargs)
    model.initialize_weights()
    cfg = TrainConfig(model_name="EnhancedDiT-XL/2", image_size=image_size, global_batch_size=32, lr=5e-5)
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    total_params = sum(p.numel() for p in model.parameters())
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg,
            "step": 0,
            "epoch": 0,
            "model_name": "EnhancedDiT-XL/2",
            "total_parameters": total_params,
            "enhanced_features": dict.fromkeys(_FEATURES, True),
        },
        out,
    )
    EnhancedDiT_XL_2(input_size=image_size // 8, **kwargs).load_state_dict(
        torch.load(out, map_location="cpu", weights_only=False)["model"]
    )
    print(f"Saved {out} ({total_params / 1e9:.2f}B params, {out.stat().st_size / 1024**3:.2f} GB); reload verified.")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("checkpoints/enhanced_dit_xl2_random_init.pt"))
    p.add_argument("--image-size", type=int, default=512)
    args = p.parse_args()
    save_model_checkpoint(args.out, image_size=args.image_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

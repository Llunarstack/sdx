#!/usr/bin/env python3
"""EnhancedDiT commands (optional path, not the default ``train.py`` / ``sample.py`` DiT-Text + T5 pipeline).

    python -m scripts.tools enhanced init [--out checkpoints/enhanced_dit_xl2_random_init.pt --image-size 512]
    python -m scripts.tools enhanced train --config cfg.json [--model EnhancedDiT-XL/2 --epochs 100]
    python -m scripts.tools enhanced sample "a cat on a chair" --checkpoint ckpt.pt --output out.png

``init`` saves a randomly initialised EnhancedDiT-XL/2 (all enhanced features on) as a training start point;
``train`` runs ``training/enhanced_trainer.py`` (DDP when launched under torchrun); ``sample`` is the inference path
for EnhancedDiT checkpoints that ``sample.py`` rejects. Model code: ``models/dit_variants.py``.
Environment check: ``python -m toolkit.training.env_health``.

Old names ``save_model_checkpoint``, ``train_enhanced`` and ``sample_enhanced`` still work as
``python -m scripts.tools <old_name>``.
"""

from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch

_PROG = "python -m scripts.tools enhanced"
_FEATURES = ("spatial_control", "anatomy_awareness", "text_rendering", "consistency")


def _add_feature_flags(parser: argparse.ArgumentParser) -> None:
    for f in _FEATURES:
        parser.add_argument(f"--enable-{f.replace('_', '-')}", action="store_true", default=True)


# --- init ----------------------------------------------------------------------------------------------------------


def save_model_checkpoint(out: Path, *, image_size: int = 512) -> Path:
    from config.train_config import TrainConfig
    from models.dit_variants import EnhancedDiT_XL_2

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


def _init_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog=f"{_PROG} init", description="Save a randomly initialized Enhanced DiT-XL/2 checkpoint."
    )
    p.add_argument("--out", type=Path, default=Path("checkpoints/enhanced_dit_xl2_random_init.pt"))
    p.add_argument("--image-size", type=int, default=512)
    args = p.parse_args(argv)
    save_model_checkpoint(args.out, image_size=args.image_size)
    return 0


# --- train ---------------------------------------------------------------------------------------------------------


def _load_train_config(argv: list[str]):
    from config.train_config import TrainConfig
    from models.dit_variants import EnhancedDiT_models

    parser = argparse.ArgumentParser(prog=f"{_PROG} train", description="Train EnhancedDiT models.")
    parser.add_argument("--config", type=str, required=True, help="Training config file")
    parser.add_argument(
        "--model",
        type=str,
        default="EnhancedDiT-XL/2",
        choices=list(EnhancedDiT_models.keys()),
        help="Enhanced DiT model variant",
    )
    _add_feature_flags(parser)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--results-dir", type=str, default="./runs")
    args = parser.parse_args(argv)

    if args.config.endswith(".json"):
        import json

        cfg = TrainConfig(**json.loads(Path(args.config).read_text(encoding="utf-8")))
    else:
        import importlib.util

        spec = importlib.util.spec_from_file_location("config", args.config)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = module.cfg

    cfg.model_name = args.model
    cfg.global_batch_size = args.global_batch_size
    cfg.lr = args.lr
    cfg.epochs = args.epochs
    cfg.image_size = args.image_size
    cfg.results_dir = args.results_dir
    for f in _FEATURES:
        setattr(cfg, f"enable_{f}", getattr(args, f"enable_{f}"))
    return cfg


_BATCH_TENSORS = (
    "images",
    "timesteps",
    "noise",
    "class_labels",
    "spatial_layouts",
    "anatomy_masks",
    "text_tokens",
    "text_positions",
    "typography_styles",
    "character_ids",
    "style_ids",
    "object_counts",
    "anatomy_keypoints",
)


def _train_main(argv: list[str]) -> int:
    cfg = _load_train_config(argv)

    import glob
    import os
    from copy import deepcopy
    from time import time

    import torch.distributed as dist
    from data.t2i_dataset import EnhancedT2IDataset, collate_enhanced
    from models.dit_variants import EnhancedDiT_models
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from training.enhanced_trainer import create_enhanced_trainer
    from utils.training.monitoring import MetricsTracker, ProgressBar, TrainingMetrics, log_gpu_memory, setup_logging

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    assert torch.cuda.is_available(), "CUDA required"
    use_ddp = "RANK" in os.environ or "LOCAL_RANK" in os.environ
    if use_ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        world_size = dist.get_world_size()
    else:
        rank = local_rank = 0
        device = torch.device("cuda", 0)
        world_size = 1
    cfg.world_size = world_size
    cfg.local_rank = local_rank
    torch.manual_seed(cfg.global_seed + rank)

    logger = setup_logging()
    logger.info(f"Starting enhanced DiT training with model: {cfg.model_name}")
    metrics_tracker = None
    if rank == 0:
        Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
        exp_index = len(glob.glob(f"{cfg.results_dir}/*"))
        exp_dir = Path(cfg.results_dir) / f"{exp_index:03d}-{cfg.model_name.replace('/', '-')}"
        ckpt_dir = exp_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        metrics_tracker = MetricsTracker(str(exp_dir))
    log_gpu_memory(logger, "Before model loading: ")

    model = EnhancedDiT_models[cfg.model_name](
        input_size=cfg.image_size // 8,
        **{f"enable_{f}": getattr(cfg, f"enable_{f}") for f in _FEATURES},
    ).to(device)
    if cfg.grad_checkpointing:
        model.enable_gradient_checkpointing()
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.requires_grad_(False)
    ema_params = list(ema_model.parameters())
    raw_model = model
    if use_ddp:
        model = DDP(model, device_ids=[local_rank])
    model_params = list(raw_model.parameters())

    trainer = create_enhanced_trainer(model, device)
    dataset = EnhancedT2IDataset(
        data_path=cfg.data_path,
        manifest_path=cfg.manifest_jsonl,
        image_size=cfg.image_size,
        **{f"enable_{f}": getattr(cfg, f"enable_{f}") for f in _FEATURES},
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    loader = DataLoader(
        dataset,
        batch_size=cfg.global_batch_size // world_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_enhanced,
    )
    logger.info(f"Dataset contains {len(dataset):,} images")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    model.train()
    progress_bar = ProgressBar(len(loader) * cfg.epochs, "Enhanced Training")

    def checkpoint(step: int, epoch: int, **extra) -> dict:
        return {
            "model": raw_model.state_dict(),
            "ema": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "config": cfg,
            **extra,
        }

    step = 0
    for epoch in range(cfg.epochs):
        if use_ddp:
            sampler.set_epoch(epoch)
        for batch in loader:
            start_time = time()
            for name in _BATCH_TENSORS:
                value = getattr(batch, name)
                if value is not None:
                    setattr(batch, name, value.to(device))

            optimizer.zero_grad()
            losses = trainer.training_step(batch)
            total_loss = losses["total"]
            total_loss.backward()
            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                torch._foreach_mul_(ema_params, 0.9999)
                torch._foreach_add_(ema_params, model_params, alpha=1 - 0.9999)

            step_time = time() - start_time
            if rank == 0:
                loss_value = total_loss.item()
                lr = optimizer.param_groups[0]["lr"]
                metrics_tracker.log_step(
                    TrainingMetrics(
                        step=step,
                        epoch=epoch,
                        loss=loss_value,
                        lr=lr,
                        grad_norm=0.0,
                        time_per_step=step_time,
                        samples_per_second=cfg.global_batch_size / step_time,
                        gpu_memory_gb=torch.cuda.memory_allocated() / 1024**3,
                    )
                )
                extra_info = {
                    k: f"{float(losses[k]) if k in losses else 0.0:.4f}"
                    for k in ("spatial", "anatomy", "text", "consistency")
                }
                progress_bar.update(step, loss_value, lr, extra_info)
            step += 1

            if rank == 0 and step % 5000 == 0:
                losses_out = {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
                path = ckpt_dir / f"checkpoint_step_{step:06d}.pt"
                torch.save(checkpoint(step, epoch, losses=losses_out), path)
                logger.info(f"Saved checkpoint: {path}")

    if rank == 0:
        progress_bar.close()
        final_path = ckpt_dir / "final_checkpoint.pt"
        torch.save(checkpoint(step, cfg.epochs), final_path)
        logger.info(f"Saved final checkpoint: {final_path}")
        metrics_tracker.save_summary()
        logger.info("Enhanced training completed!")
    if use_ddp:
        dist.destroy_process_group()
    return 0


# --- sample --------------------------------------------------------------------------------------------------------

_OBJECT_WORDS = (
    "person",
    "woman",
    "man",
    "girl",
    "boy",
    "child",
    "car",
    "house",
    "tree",
    "flower",
    "chair",
    "table",
    "cat",
    "dog",
    "bird",
    "book",
    "cup",
    "bottle",
)
_HUMAN_WORDS = (
    "person",
    "woman",
    "man",
    "girl",
    "boy",
    "child",
    "people",
    "human",
    "figure",
    "character",
    "portrait",
)


def extract_objects_from_prompt(prompt: str) -> list[str]:
    lower = prompt.lower()
    return [obj for obj in _OBJECT_WORDS if obj in lower]


def detect_human_in_prompt(prompt: str) -> bool:
    lower = prompt.lower()
    return any(word in lower for word in _HUMAN_WORDS)


def _scene_boxes(n: int, layout_type: str) -> list[tuple[float, float, float, float]]:
    """``(x, y, w, h)`` per object for the spatial-control input (grid / circle / balanced)."""
    import numpy as np

    boxes = []
    for i in range(n):
        if layout_type == "grid":
            x, y = (i % 3) * 0.3 + 0.1, (i // 3) * 0.3 + 0.1
        elif layout_type == "circle":
            angle = (i / n) * 2 * np.pi
            x, y = 0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)
        else:
            x, y = 0.2 + (i / max(1, n - 1)) * 0.6, 0.3 + 0.4 * np.random.random()
        boxes.append((float(x), float(y), 0.1, 0.1))
    return boxes


def load_enhanced_model(checkpoint_path: str, device: str = "cuda"):
    from models.dit_variants import EnhancedDiT_models

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model_name = getattr(config, "model_name", "EnhancedDiT-XL/2")
    model_fn = EnhancedDiT_models.get(model_name)
    if model_fn is None:
        raise ValueError(f"Unknown enhanced model: {model_name}")
    model = model_fn(
        input_size=config.image_size // 8,
        **{f"enable_{f}": getattr(config, f"enable_{f}", True) for f in _FEATURES},
    )
    model.load_state_dict(checkpoint.get("ema", checkpoint.get("model")), strict=True)
    return model.to(device).eval(), config


def _stable_id(name: str, mod: int) -> int:
    return zlib.crc32(name.encode("utf-8")) % mod


def prepare_enhanced_inputs(prompt: str, args, device: str = "cuda") -> dict:
    inputs: dict = {}
    if args.enable_spatial_control:
        objects = extract_objects_from_prompt(prompt)
        if objects:
            boxes = _scene_boxes(len(objects[:10]), args.layout_type)
            layout = torch.zeros(1, 10, 4, device=device)
            layout[0, : len(boxes)] = torch.tensor(boxes, device=device)
            inputs["spatial_layout"] = layout

    if args.enable_anatomy_awareness and detect_human_in_prompt(prompt):
        side = args.image_size // 8
        mask = torch.zeros(1, side**2, device=device)
        center = side // 2
        mask[0, center * side + center - 10 : center * side + center + 10] = 1.0  # center region = likely figure
        inputs["anatomy_mask"] = mask

    if args.enable_text_rendering and args.text_content:
        text_content = args.text_content.split(",")
        chars = " ".join(text_content)[:50]
        tokens = torch.zeros(1, 50, dtype=torch.long, device=device)
        tokens[0, : len(chars)] = torch.tensor([ord(c) % 1000 for c in chars], dtype=torch.long, device=device)
        inputs["text_tokens"] = tokens
        n = min(len(text_content), 50)
        positions = torch.zeros(1, 50, 2, device=device)
        positions[0, :n, 0] = torch.tensor([0.3 + 0.4 * (i / max(1, len(text_content) - 1)) for i in range(n)])
        positions[0, :n, 1] = 0.5
        inputs["text_positions"] = positions
        style_id = {"modern": 0, "classic": 1, "display": 2, "script": 3}.get(args.typography_style, 0)
        inputs["typography_style"] = torch.tensor([style_id], dtype=torch.long, device=device)

    if args.enable_consistency:
        if args.character_name:
            inputs["character_id"] = torch.tensor([_stable_id(args.character_name, 100)], device=device)
        if args.style_name:
            inputs["style_id"] = torch.tensor([_stable_id(args.style_name, 50)], device=device)
    return inputs


@torch.inference_mode()
def sample_enhanced(model, enhanced_inputs: dict, args, device: str = "cuda") -> torch.Tensor:
    """Simplified cosine-schedule DDIM sampling (class label 0, no text encoder)."""
    import math

    side = args.image_size // 8
    z = torch.randn((1, 4, side, side), device=device, generator=torch.Generator(device=device).manual_seed(args.seed))
    timesteps = torch.linspace(1000, 0, args.steps + 1, device=device)[:-1]
    alphas = torch.cos(timesteps / 1000 * math.pi / 2) ** 2
    y = torch.zeros(1, dtype=torch.long, device=device)
    for i, t in enumerate(timesteps):
        noise_pred = model(z, torch.full((1,), t, device=device, dtype=torch.long), y, **enhanced_inputs)
        alpha_t, alpha_prev = alphas[i], alphas[min(i + 1, len(timesteps) - 1)]
        x0_pred = (z - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        if i < len(timesteps) - 1:
            z = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * noise_pred
        else:
            z = x0_pred
    return z


def _sample_main(argv: list[str]) -> int:
    import numpy as np
    from PIL import Image

    parser = argparse.ArgumentParser(prog=f"{_PROG} sample", description="Enhanced DiT Inference")
    parser.add_argument("prompt", type=str, help="Text prompt")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--output", type=str, default="enhanced_output.png", help="Output path")
    parser.add_argument("--steps", type=int, default=50, help="Sampling steps")
    parser.add_argument("--cfg-scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--image-size", type=int, default=512, help="Image size")
    _add_feature_flags(parser)
    parser.add_argument(
        "--layout-type", type=str, default="balanced", choices=["grid", "circle", "line", "pyramid", "balanced"]
    )
    parser.add_argument("--text-content", type=str, default="", help="Comma-separated text content to render")
    parser.add_argument(
        "--typography-style", type=str, default="modern", choices=["modern", "classic", "display", "script"]
    )
    parser.add_argument("--character-name", type=str, default="", help="Character name for consistency")
    parser.add_argument("--style-name", type=str, default="", help="Style name for consistency")
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading enhanced model from {args.checkpoint}...")
    model, config = load_enhanced_model(args.checkpoint, device)
    print(f"Model loaded: {config.model_name}")

    print("Preparing enhanced features...")
    enhanced_inputs = prepare_enhanced_inputs(args.prompt, args, device)
    print("Enhanced features enabled:")
    for feature, tensor in enhanced_inputs.items():
        print(f"  {feature}: {tensor.shape}")

    print(f"Generating image with prompt: '{args.prompt}'")
    print(f"Using {args.steps} steps, seed {args.seed}")
    latents = sample_enhanced(model, enhanced_inputs, args, device).cpu()

    # No VAE on this path: latents are mapped straight to pixels (first 3 channels).
    image_array = np.clip((latents[0].permute(1, 2, 0).numpy() + 1) * 127.5, 0, 255).astype(np.uint8)
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]
    image = Image.fromarray(image_array).resize((args.image_size, args.image_size), Image.LANCZOS)
    image.save(args.output)
    print(f"Enhanced image saved to {args.output}")

    print("\nGeneration Summary:")
    print(f"  Model: {config.model_name}")
    print(f"  Prompt: {args.prompt}")
    print(f"  Enhanced features used: {len(enhanced_inputs)}")
    if "spatial_layout" in enhanced_inputs:
        print(f"  Objects detected: {extract_objects_from_prompt(args.prompt)}")
    if "text_tokens" in enhanced_inputs:
        print(f"  Text content: {args.text_content}")
        print(f"  Typography style: {args.typography_style}")
    if "character_id" in enhanced_inputs:
        print(f"  Character: {args.character_name}")
    if "style_id" in enhanced_inputs:
        print(f"  Style: {args.style_name}")
    return 0


COMMANDS = {"init": _init_main, "train": _train_main, "sample": _sample_main}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        print("Commands: " + ", ".join(COMMANDS))
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    return int(COMMANDS[argv[0]](argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())

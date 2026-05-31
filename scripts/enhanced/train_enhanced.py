#!/usr/bin/env python3
"""
Enhanced Training Script for Advanced DiT Models
Trains models with built-in precision control, anatomy awareness, text rendering, and consistency features.
"""

import argparse
import glob
import os
import sys
from copy import deepcopy
from pathlib import Path
from time import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Repo root: scripts/enhanced/ -> parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

# Project imports
from config.train_config import TrainConfig
from data.enhanced_dataset import EnhancedT2IDataset, collate_enhanced_batch
from diffusion.gaussian_diffusion import create_diffusion
from models.enhanced_dit import EnhancedDiT_models
from training.enhanced_trainer import create_enhanced_trainer
from utils.training.error_handling import setup_logging
from utils.training.metrics import MetricsTracker, TrainingMetrics

# Enable TF32 on Ampere+ for speed
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def log_gpu_memory(logger, prefix=""):
    """Log GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"{prefix}GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


class ProgressBar:
    """Simple progress bar for training."""

    def __init__(self, total_steps, description="Training"):
        self.total_steps = total_steps
        self.description = description
        self.current_step = 0

    def update(self, step, loss, lr, extra_info=None):
        """Update progress bar."""
        self.current_step = step
        progress = step / self.total_steps * 100

        info_str = (
            f"{self.description}: {progress:.1f}% | Step {step}/{self.total_steps} | Loss: {loss:.4f} | LR: {lr:.2e}"
        )

        if extra_info:
            extra_str = " | ".join([f"{k}: {v}" for k, v in extra_info.items()])
            info_str += f" | {extra_str}"

        print(f"\r{info_str}", end="", flush=True)

        if step % 100 == 0:  # New line every 100 steps
            print()

    def close(self):
        """Close progress bar."""
        print()  # New line


def get_enhanced_config():
    """Get enhanced training configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Training config file")
    parser.add_argument(
        "--model",
        type=str,
        default="EnhancedDiT-XL/2",
        choices=list(EnhancedDiT_models.keys()),
        help="Enhanced DiT model variant",
    )

    # Enhanced feature flags
    parser.add_argument(
        "--enable-spatial-control", action="store_true", default=True, help="Enable spatial control features"
    )
    parser.add_argument(
        "--enable-anatomy-awareness", action="store_true", default=True, help="Enable anatomy awareness features"
    )
    parser.add_argument(
        "--enable-text-rendering", action="store_true", default=True, help="Enable text rendering features"
    )
    parser.add_argument("--enable-consistency", action="store_true", default=True, help="Enable consistency features")

    # Training parameters
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--results-dir", type=str, default="./runs")

    args = parser.parse_args()

    # Load base config
    if args.config.endswith(".json"):
        import json

        with open(args.config, "r") as f:
            config_dict = json.load(f)
        cfg = TrainConfig(**config_dict)
    else:
        import importlib.util

        spec = importlib.util.spec_from_file_location("config", args.config)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        cfg = config_module.cfg

    # Override with command line args
    cfg.model_name = args.model
    cfg.global_batch_size = args.global_batch_size
    cfg.lr = args.lr
    cfg.epochs = args.epochs
    cfg.image_size = args.image_size
    cfg.results_dir = args.results_dir

    # Enhanced feature flags
    cfg.enable_spatial_control = args.enable_spatial_control
    cfg.enable_anatomy_awareness = args.enable_anatomy_awareness
    cfg.enable_text_rendering = args.enable_text_rendering
    cfg.enable_consistency = args.enable_consistency

    return cfg


def main():
    """Main enhanced training function."""
    cfg = get_enhanced_config()

    # Setup distributed training
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
        rank = 0
        local_rank = 0
        device = torch.device("cuda", 0)
        world_size = 1

    cfg.world_size = world_size
    cfg.local_rank = local_rank
    torch.manual_seed(cfg.global_seed + rank)

    # Setup logging
    logger = setup_logging()
    logger.info(f"Starting enhanced DiT training with model: {cfg.model_name}")

    # Create experiment directory
    if rank == 0:
        Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
        exp_index = len(glob.glob(f"{cfg.results_dir}/*"))
        exp_dir = Path(cfg.results_dir) / f"{exp_index:03d}-{cfg.model_name.replace('/', '-')}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = exp_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)

        # Initialize metrics tracker
        metrics_tracker = MetricsTracker(str(exp_dir))
    else:
        exp_dir = Path(cfg.results_dir) / "run"
        metrics_tracker = None

    # Log GPU memory
    log_gpu_memory(logger, "Before model loading: ")

    # Create enhanced model
    logger.info("Creating enhanced DiT model...")
    model_fn = EnhancedDiT_models[cfg.model_name]
    model = model_fn(
        input_size=cfg.image_size // 8,  # Assuming 8x downsampling
        enable_spatial_control=cfg.enable_spatial_control,
        enable_anatomy_awareness=cfg.enable_anatomy_awareness,
        enable_text_rendering=cfg.enable_text_rendering,
        enable_consistency=cfg.enable_consistency,
    ).to(device)

    # Enable gradient checkpointing if specified
    if cfg.grad_checkpointing:
        model.enable_gradient_checkpointing()

    # Create EMA model
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.requires_grad_(False)

    # Wrap with DDP
    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

    # Create diffusion
    logger.info("Creating diffusion...")
    create_diffusion(
        timestep_respacing=cfg.timestep_respacing,
        beta_schedule=cfg.beta_schedule,
    )

    # Create enhanced trainer
    logger.info("Creating enhanced trainer...")
    trainer = create_enhanced_trainer(model, device)

    # Create enhanced dataset
    logger.info("Creating enhanced dataset...")
    dataset = EnhancedT2IDataset(
        data_path=cfg.data_path,
        manifest_path=cfg.manifest_jsonl,
        image_size=cfg.image_size,
        enable_spatial_control=cfg.enable_spatial_control,
        enable_anatomy_awareness=cfg.enable_anatomy_awareness,
        enable_text_rendering=cfg.enable_text_rendering,
        enable_consistency=cfg.enable_consistency,
    )

    # Create data loader
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    loader = DataLoader(
        dataset,
        batch_size=cfg.global_batch_size // world_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_enhanced_batch,
    )

    logger.info(f"Dataset contains {len(dataset):,} images")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    # Training loop
    logger.info("Starting enhanced training...")
    model.train()

    total_steps = len(loader) * cfg.epochs
    progress_bar = ProgressBar(total_steps, "Enhanced Training")

    step = 0
    for epoch in range(cfg.epochs):
        if use_ddp:
            sampler.set_epoch(epoch)

        for batch in loader:
            start_time = time()

            # Move batch to device
            batch.images = batch.images.to(device)
            batch.timesteps = batch.timesteps.to(device)
            batch.noise = batch.noise.to(device)
            batch.class_labels = batch.class_labels.to(device)

            # Move enhanced features to device
            if batch.spatial_layouts is not None:
                batch.spatial_layouts = batch.spatial_layouts.to(device)
            if batch.anatomy_masks is not None:
                batch.anatomy_masks = batch.anatomy_masks.to(device)
            if batch.text_tokens is not None:
                batch.text_tokens = batch.text_tokens.to(device)
            if batch.text_positions is not None:
                batch.text_positions = batch.text_positions.to(device)
            if batch.typography_styles is not None:
                batch.typography_styles = batch.typography_styles.to(device)
            if batch.character_ids is not None:
                batch.character_ids = batch.character_ids.to(device)
            if batch.style_ids is not None:
                batch.style_ids = batch.style_ids.to(device)
            if batch.object_counts is not None:
                batch.object_counts = batch.object_counts.to(device)
            if batch.anatomy_keypoints is not None:
                batch.anatomy_keypoints = batch.anatomy_keypoints.to(device)

            # Training step
            optimizer.zero_grad()

            losses = trainer.training_step(batch)
            total_loss = losses["total"]

            total_loss.backward()

            # Gradient clipping
            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

            optimizer.step()

            # Update EMA
            with torch.no_grad():
                for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                    ema_param.data.mul_(0.9999).add_(param.data, alpha=1 - 0.9999)

            # Logging
            step_time = time() - start_time
            samples_per_second = cfg.global_batch_size / step_time

            if rank == 0:
                # Create metrics
                metrics = TrainingMetrics(
                    step=step,
                    epoch=epoch,
                    loss=total_loss.item(),
                    lr=optimizer.param_groups[0]["lr"],
                    grad_norm=0.0,  # Would calculate actual grad norm
                    time_per_step=step_time,
                    samples_per_second=samples_per_second,
                    gpu_memory_gb=torch.cuda.memory_allocated() / 1024**3,
                )

                if metrics_tracker:
                    metrics_tracker.log_step(metrics)

                # Update progress bar
                extra_info = {
                    "spatial": f"{losses.get('spatial', torch.tensor(0.0)).item():.4f}",
                    "anatomy": f"{losses.get('anatomy', torch.tensor(0.0)).item():.4f}",
                    "text": f"{losses.get('text', torch.tensor(0.0)).item():.4f}",
                    "consistency": f"{losses.get('consistency', torch.tensor(0.0)).item():.4f}",
                }
                progress_bar.update(step, total_loss.item(), optimizer.param_groups[0]["lr"], extra_info)

            step += 1

            # Save checkpoint
            if rank == 0 and step % 5000 == 0:
                checkpoint = {
                    "model": model.state_dict() if not use_ddp else model.module.state_dict(),
                    "ema": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "epoch": epoch,
                    "config": cfg,
                    "losses": {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()},
                }

                checkpoint_path = ckpt_dir / f"checkpoint_step_{step:06d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint: {checkpoint_path}")

    # Final cleanup
    if rank == 0:
        progress_bar.close()

        # Save final checkpoint
        final_checkpoint = {
            "model": model.state_dict() if not use_ddp else model.module.state_dict(),
            "ema": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": cfg.epochs,
            "config": cfg,
        }

        final_path = ckpt_dir / "final_checkpoint.pt"
        torch.save(final_checkpoint, final_path)
        logger.info(f"Saved final checkpoint: {final_path}")

        # Save training summary
        if metrics_tracker:
            metrics_tracker.save_summary()

        logger.info("Enhanced training completed!")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

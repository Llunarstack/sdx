"""Training run checks and monitoring: config validation and memory estimates, the sdx logger, metrics tracking, progress bars, the live terminal dashboard and model summaries."""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from config.train_config import TrainConfig
from diffusion.schedules import ZTSNR_SUFFIX


def validate_train_config(cfg: TrainConfig, *, require_cuda: bool = True) -> list[str]:
    """
    Validate training configuration and return list of warnings/errors.

    ``require_cuda``: when False, missing CUDA becomes a WARNING (for dry-runs / CI / docs tooling).
    """
    warnings = []
    errors = []

    # Data validation
    if not cfg.data_path and not cfg.manifest_jsonl:
        errors.append("Either data_path or manifest_jsonl must be specified")

    rcm = getattr(cfg, "region_caption_mode", "append")
    if rcm not in {"append", "prefix", "off"}:
        errors.append(f"region_caption_mode must be append|prefix|off, got: {rcm}")

    if cfg.data_path and not Path(cfg.data_path).exists():
        errors.append(f"Data path does not exist: {cfg.data_path}")

    if cfg.manifest_jsonl and not Path(cfg.manifest_jsonl).exists():
        errors.append(f"Manifest JSONL does not exist: {cfg.manifest_jsonl}")

    if getattr(cfg, "resume", None) and getattr(cfg, "init_from", None):
        errors.append("Specify only one of resume or init_from (--resume vs --init-from)")

    # Model validation
    from models import DiT_models_text

    if cfg.model_name not in DiT_models_text:
        errors.append(f"Unknown model: {cfg.model_name}. Available: {list(DiT_models_text.keys())}")

    # Autoencoder validation
    ae_type = getattr(cfg, "autoencoder_type", "kl")
    if ae_type not in {"kl", "rae"}:
        errors.append(f"autoencoder_type must be 'kl' or 'rae', got: {ae_type}")

    # REPA validation
    repa_w = float(getattr(cfg, "repa_weight", 0.0))
    if repa_w < 0:
        errors.append(f"repa_weight must be >= 0, got: {repa_w}")
    repa_out_dim = int(getattr(cfg, "repa_out_dim", 768))
    if repa_out_dim <= 0:
        errors.append(f"repa_out_dim must be > 0, got: {repa_out_dim}")
    repa_proj_hidden = int(getattr(cfg, "repa_projector_hidden_dim", 0))
    if repa_proj_hidden < 0:
        errors.append(f"repa_projector_hidden_dim must be >= 0, got: {repa_proj_hidden}")

    # SSM swap validation (hybrid SSM token mixer)
    ssm_every_n = int(getattr(cfg, "ssm_every_n", 0))
    if ssm_every_n < 0:
        errors.append(f"ssm_every_n must be >= 0, got: {ssm_every_n}")
    ssm_kernel = int(getattr(cfg, "ssm_kernel_size", 7))
    if ssm_kernel < 3:
        errors.append(f"ssm_kernel_size must be >= 3, got: {ssm_kernel}")

    # ViT feature validations
    num_register_tokens = int(getattr(cfg, "num_register_tokens", 0))
    if num_register_tokens < 0:
        errors.append(f"num_register_tokens must be >= 0, got: {num_register_tokens}")

    kv_merge_factor = int(getattr(cfg, "kv_merge_factor", 1))
    if kv_merge_factor < 1:
        errors.append(f"kv_merge_factor must be >= 1, got: {kv_merge_factor}")

    token_routing_enabled = bool(getattr(cfg, "token_routing_enabled", False))
    if token_routing_enabled:
        token_routing_strength = float(getattr(cfg, "token_routing_strength", 1.0))
        if token_routing_strength <= 0:
            errors.append(f"token_routing_strength must be > 0, got: {token_routing_strength}")

    # Compatibility: our AR masks are defined over patch tokens only; register tokens and KV-merge
    # change the effective token/kv lengths, so we disallow with num_ar_blocks>0 for now.
    if getattr(cfg, "num_ar_blocks", 0) and int(getattr(cfg, "num_ar_blocks", 0)) > 0:
        if num_register_tokens > 0:
            errors.append("num_register_tokens is not supported together with num_ar_blocks>0 (AR mask mismatch).")
        if kv_merge_factor > 1:
            errors.append("kv_merge_factor>1 is not supported together with num_ar_blocks>0 (AR mask mismatch).")

    ot_reg = float(getattr(cfg, "ot_noise_pair_reg", 0.0))
    if ot_reg < 0:
        errors.append(f"ot_noise_pair_reg must be >= 0, got: {ot_reg}")
    ot_mode = str(getattr(cfg, "ot_noise_pair_mode", "soft"))
    if ot_mode not in {"soft", "hungarian"}:
        errors.append(f"ot_noise_pair_mode must be soft|hungarian, got: {ot_mode}")
    ot_iters = int(getattr(cfg, "ot_noise_pair_iters", 40))
    if ot_iters < 1:
        errors.append(f"ot_noise_pair_iters must be >= 1, got: {ot_iters}")

    flow_fm = bool(getattr(cfg, "flow_matching_training", False))
    mdm_on = float(getattr(cfg, "mdm_mask_ratio", 0.0)) > 0.0 or bool(getattr(cfg, "mdm_mask_schedule", None))
    if flow_fm and mdm_on:
        errors.append(
            "flow_matching_training cannot be used together with MDM masked training (mdm_mask_ratio / mdm_mask_schedule)."
        )
    baw = float(getattr(cfg, "bridge_aux_weight", 0.0))
    if baw < 0:
        errors.append(f"bridge_aux_weight must be >= 0, got: {baw}")
    bal = float(getattr(cfg, "bridge_aux_lambda", 0.2))
    if bal <= 0.0 or bal > 1.0:
        errors.append(f"bridge_aux_lambda must be in (0, 1], got: {bal}")

    # Training length validation
    if cfg.passes <= 0 and cfg.max_steps <= 0 and cfg.epochs <= 0:
        errors.append("Must specify either passes > 0, max_steps > 0, or epochs > 0")

    if cfg.passes > 0 and cfg.max_steps > 0:
        warnings.append("Both passes and max_steps specified. max_steps will cap total training.")

    # Batch size validation
    if cfg.global_batch_size <= 0:
        errors.append("global_batch_size must be positive")

    if cfg.global_batch_size % cfg.world_size != 0:
        warnings.append(f"global_batch_size ({cfg.global_batch_size}) not divisible by world_size ({cfg.world_size})")

    # Learning rate validation
    if cfg.lr <= 0:
        errors.append("Learning rate must be positive")

    if cfg.min_lr >= cfg.lr:
        warnings.append("min_lr should be less than lr for cosine schedule")

    # Memory optimization warnings
    if cfg.image_size > 512 and not cfg.grad_checkpointing:
        warnings.append("Consider enabling grad_checkpointing for large image sizes")

    if cfg.global_batch_size > 64 and not cfg.use_bf16:
        warnings.append("Consider enabling bf16 for large batch sizes")

    # Validation split warnings
    if cfg.val_split > 0 and cfg.val_split >= 0.5:
        warnings.append("Validation split is very large (>= 50%)")

    if cfg.early_stopping_patience > 0 and cfg.val_split <= 0:
        warnings.append("early_stopping_patience requires val_split > 0")

    # AR blocks validation
    if cfg.num_ar_blocks not in [0, 2, 4]:
        errors.append("num_ar_blocks must be 0, 2, or 4")

    # Caption dropout schedule validation
    if cfg.caption_dropout_schedule:
        for _step, prob in cfg.caption_dropout_schedule:
            if not (0 <= prob <= 1):
                errors.append(f"Caption dropout probability must be in [0,1], got {prob}")

    # Refinement validation
    if cfg.refinement_prob < 0 or cfg.refinement_prob > 1:
        errors.append("refinement_prob must be in [0,1]")

    if cfg.refinement_max_t >= cfg.num_timesteps:
        warnings.append("refinement_max_t should be less than num_timesteps")

    # MDM validation
    if getattr(cfg, "mdm_mask_ratio", 0.0) < 0 or getattr(cfg, "mdm_mask_ratio", 0.0) > 1:
        errors.append("mdm_mask_ratio must be in [0,1]")
    if getattr(cfg, "mdm_patch_size", 2) <= 0:
        errors.append("mdm_patch_size must be > 0")
    if getattr(cfg, "mdm_min_mask_patches", 1) < 0:
        errors.append("mdm_min_mask_patches must be >= 0")
    if getattr(cfg, "mdm_mask_schedule", None) is not None:
        for t_step, r in cfg.mdm_mask_schedule:
            if not (0 <= float(r) <= 1):
                errors.append(f"mdm_mask_schedule mask_ratio must be in [0,1], got {r}")
                break
            if int(t_step) < 0:
                errors.append(f"mdm_mask_schedule t_step must be >=0, got {t_step}")
                break
            if int(t_step) >= cfg.num_timesteps:
                warnings.append(
                    f"mdm_mask_schedule contains t_step={t_step} >= num_timesteps={cfg.num_timesteps} (it will be clamped)."
                )

    # MoE validation
    moe_num_experts = getattr(cfg, "moe_num_experts", 0)
    moe_top_k = getattr(cfg, "moe_top_k", 2)
    if moe_num_experts is not None and int(moe_num_experts) < 0:
        errors.append("moe_num_experts must be >= 0")
    if int(moe_top_k) <= 0:
        errors.append("moe_top_k must be > 0")
    if moe_num_experts is not None and int(moe_num_experts) > 0 and int(moe_top_k) > int(moe_num_experts):
        errors.append("moe_top_k cannot exceed moe_num_experts")
    if getattr(cfg, "moe_balance_loss_weight", 0.0) < 0:
        errors.append("moe_balance_loss_weight must be >= 0")

    # Latent grid / model IO sanity (catches bad JSON exports early)
    if int(cfg.image_size) < 8 or int(cfg.image_size) % 8 != 0:
        errors.append(f"image_size must be >= 8 and a multiple of 8 (VAE latent alignment); got {cfg.image_size}")

    te = str(getattr(cfg, "text_encoder", "") or "").strip()
    if not te:
        errors.append("text_encoder must be a non-empty Hugging Face id or local path")

    vm = str(getattr(cfg, "vae_model", "") or "").strip()
    if not vm:
        errors.append("vae_model must be a non-empty Hugging Face id or local path")

    if ae_type == "kl" and float(getattr(cfg, "latent_scale", 0.18215)) <= 0:
        errors.append("latent_scale must be > 0 when autoencoder_type is kl")

    if bool(getattr(cfg, "edit_conditioning", False)):
        if float(getattr(cfg, "img2img_prob", 0.0) or 0.0) > 0:
            errors.append("edit_conditioning and img2img_prob both use the row's source image; pick one")
        if ae_type != "kl":
            errors.append("edit_conditioning needs a KL VAE (the RAE latent bridge is trainable)")
        if int(getattr(cfg, "num_ar_blocks", 0) or 0) > 0:
            errors.append("edit_conditioning is not supported with num_ar_blocks > 0")
        if not str(getattr(cfg, "model_name", "")).startswith("DiT-"):
            errors.append("edit_conditioning needs a DiT-*-Text model (DiT-B/L/XL, DiT-P, DiT-Supreme)")
        if not 0.0 <= float(getattr(cfg, "edit_source_dropout", 0.1)) < 1.0:
            errors.append("edit_source_dropout must be in [0, 1)")

    pred_t = str(getattr(cfg, "prediction_type", "epsilon") or "").lower().strip()
    if pred_t not in ("epsilon", "v", "x0"):
        errors.append(f"prediction_type must be epsilon | v | x0; got {pred_t!r}")

    beta = str(getattr(cfg, "beta_schedule", "linear") or "").lower().strip()
    ztsnr = beta.endswith(ZTSNR_SUFFIX)
    if beta.removesuffix(ZTSNR_SUFFIX) not in ("linear", "cosine", "sigmoid", "squaredcos_cap_v2"):
        warnings.append(
            f"Unusual beta_schedule {beta!r}; supported in-repo: linear, cosine, sigmoid, squaredcos_cap_v2"
        )
    if ztsnr and pred_t == "epsilon" and not bool(getattr(cfg, "flow_matching_training", False)):
        errors.append(
            "zero terminal SNR needs --prediction-type v (or x0): epsilon prediction is undefined at SNR 0 "
            "(Lin et al. 2024)."
        )

    buckets = getattr(cfg, "resolution_buckets", None)
    if buckets:
        for pair in buckets:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                errors.append(f"resolution_buckets entries must be (H, W) pairs; got {pair!r}")
                break
            hi, wi = int(pair[0]), int(pair[1])
            if hi < 8 or wi < 8 or hi % 8 != 0 or wi % 8 != 0:
                errors.append(f"resolution_buckets (H, W) must be >= 8 and multiples of 8; got ({hi}, {wi})")
                break

    crop_mode = str(getattr(cfg, "crop_mode", "center") or "center").lower().strip()
    if crop_mode not in ("center", "random", "largest_center"):
        errors.append(f"crop_mode must be center | random | largest_center; got {crop_mode!r}")

    # Hardware validation
    if not torch.cuda.is_available():
        msg = "CUDA is required for the default training pipeline but is not available"
        if require_cuda:
            errors.append(msg)
        else:
            warnings.append(msg)

    # Combine errors and warnings
    issues = []
    for error in errors:
        issues.append(f"ERROR: {error}")
    for warning in warnings:
        issues.append(f"WARNING: {warning}")

    return issues


def estimate_memory_usage(cfg: TrainConfig) -> dict:
    """Estimate GPU memory usage for training configuration."""
    # Rough estimates based on model size and batch size
    model_memory_map = {
        "DiT-B/2-Text": 1.2,  # GB
        "DiT-L/2-Text": 2.8,  # GB
        "DiT-XL/2-Text": 4.5,  # GB
        "DiT-P/2-Text": 5.2,  # GB
        "DiT-P-L/2-Text": 6.8,  # GB
        "DiT-Supreme/2-Text": 5.8,  # GB
        "DiT-Supreme-L/2-Text": 7.5,  # GB
    }

    base_model_memory = model_memory_map.get(cfg.model_name, 4.5)

    # EMA doubles model memory
    model_memory = base_model_memory * 2

    # Batch memory (rough estimate)
    batch_size_per_gpu = cfg.global_batch_size // cfg.world_size
    image_memory_per_sample = (cfg.image_size**2 * 3 * 4) / (1024**3)  # RGB float32
    latent_memory_per_sample = image_memory_per_sample / 64  # VAE compression ~8x8

    batch_memory = batch_size_per_gpu * (image_memory_per_sample + latent_memory_per_sample)

    # T5 encoder memory
    t5_memory = 2.5 if "xxl" in cfg.text_encoder.lower() else 1.0

    # VAE memory
    vae_memory = 0.5

    # Optimizer states (AdamW: 2x model params)
    optimizer_memory = base_model_memory * 2

    total_estimated = model_memory + batch_memory + t5_memory + vae_memory + optimizer_memory

    return {
        "model_memory_gb": model_memory,
        "batch_memory_gb": batch_memory,
        "t5_memory_gb": t5_memory,
        "vae_memory_gb": vae_memory,
        "optimizer_memory_gb": optimizer_memory,
        "total_estimated_gb": total_estimated,
        "recommended_vram_gb": total_estimated * 1.3,  # 30% buffer
    }


def setup_logging(log_dir: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """``sdx`` logger with a console handler and, when ``log_dir`` is given, ``<log_dir>/sdx.log``."""
    logger = logging.getLogger("sdx")
    logger.setLevel(level)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

    if log_dir:
        log_path = Path(log_dir) / "sdx.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")
        )
        logger.addHandler(file_handler)
    return logger


def log_gpu_memory(logger: logging.Logger, prefix: str = ""):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"{prefix}GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def get_model_info(model: torch.nn.Module) -> dict:
    """Parameter counts, fp32 size in MB and device."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),
        "device": next(model.parameters()).device if total_params > 0 else "unknown",
    }


@dataclass(slots=True)
class TrainingMetrics:
    """Training metrics container."""

    step: int
    epoch: int
    loss: float
    lr: float
    grad_norm: float
    time_per_step: float
    samples_per_second: float
    gpu_memory_gb: float

    # Optional validation metrics
    val_loss: float | None = None
    val_steps: int | None = None

    # Optional refinement metrics
    refinement_loss: float | None = None
    refinement_ratio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


class MetricsTracker:
    """Track and log training metrics."""

    def __init__(self, log_dir: str | None = None):
        self.log_dir = Path(log_dir) if log_dir else None
        self.metrics_history: list[TrainingMetrics] = []
        self.best_loss = float("inf")
        self.best_val_loss = float("inf")
        self.start_time = time.time()
        self.step_start_time = time.time()

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_file = self.log_dir / "metrics.jsonl"

    def log_step(self, metrics: TrainingMetrics):
        """Log metrics for a training step."""
        self.metrics_history.append(metrics)

        # Update best losses
        if metrics.loss < self.best_loss:
            self.best_loss = metrics.loss

        if metrics.val_loss is not None and metrics.val_loss < self.best_val_loss:
            self.best_val_loss = metrics.val_loss

        # Save to file
        if self.log_dir:
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(metrics.to_dict()) + "\n")

    def get_recent_avg_loss(self, window: int = 100) -> float:
        """Get average loss over recent steps."""
        if not self.metrics_history:
            return float("inf")

        recent_metrics = self.metrics_history[-window:]
        return np.mean([m.loss for m in recent_metrics])

    def get_training_speed(self) -> dict[str, float]:
        """Get training speed statistics."""
        if len(self.metrics_history) < 2:
            return {"samples_per_second": 0.0, "steps_per_minute": 0.0}

        recent_metrics = self.metrics_history[-10:]  # Last 10 steps
        avg_samples_per_sec = np.mean([m.samples_per_second for m in recent_metrics])
        avg_time_per_step = np.mean([m.time_per_step for m in recent_metrics])
        steps_per_minute = 60.0 / avg_time_per_step if avg_time_per_step > 0 else 0.0

        return {"samples_per_second": avg_samples_per_sec, "steps_per_minute": steps_per_minute}

    def estimate_time_remaining(self, current_step: int, total_steps: int) -> str:
        """Estimate remaining training time."""
        if len(self.metrics_history) < 5:
            return "Estimating..."

        recent_metrics = self.metrics_history[-10:]
        avg_time_per_step = np.mean([m.time_per_step for m in recent_metrics])

        remaining_steps = total_steps - current_step
        remaining_seconds = remaining_steps * avg_time_per_step

        hours = int(remaining_seconds // 3600)
        minutes = int((remaining_seconds % 3600) // 60)

        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"

    def get_summary(self) -> dict[str, Any]:
        """Get training summary statistics."""
        if not self.metrics_history:
            return {}

        total_time = time.time() - self.start_time
        total_steps = len(self.metrics_history)

        return {
            "total_steps": total_steps,
            "total_time_hours": total_time / 3600,
            "best_loss": self.best_loss,
            "best_val_loss": self.best_val_loss if self.best_val_loss != float("inf") else None,
            "final_loss": self.metrics_history[-1].loss,
            "avg_samples_per_second": np.mean([m.samples_per_second for m in self.metrics_history]),
            "avg_gpu_memory_gb": np.mean([m.gpu_memory_gb for m in self.metrics_history]),
        }

    def save_summary(self):
        """Save training summary to file."""
        if not self.log_dir:
            return

        summary = self.get_summary()
        summary_file = self.log_dir / "training_summary.json"

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)


class ProgressBar:
    """Simple progress bar for training."""

    def __init__(self, total_steps: int, desc: str = "Training"):
        self.total_steps = total_steps
        self.desc = desc
        self.current_step = 0
        self.start_time = time.time()

    def update(self, step: int, loss: float, lr: float, extra_info: dict | None = None):
        """Update progress bar."""
        self.current_step = step
        progress = step / self.total_steps
        bar_length = 40
        filled_length = int(bar_length * progress)

        bar = "█" * filled_length + "-" * (bar_length - filled_length)
        percent = progress * 100

        elapsed_time = time.time() - self.start_time
        if step > 0:
            eta = elapsed_time * (self.total_steps - step) / step
            eta_str = f"{int(eta // 3600):02d}:{int((eta % 3600) // 60):02d}:{int(eta % 60):02d}"
        else:
            eta_str = "--:--:--"

        info_str = f"loss={loss:.4f} lr={lr:.2e}"
        if extra_info:
            info_str += " " + " ".join([f"{k}={v}" for k, v in extra_info.items()])

        print(
            f"\r{self.desc}: |{bar}| {percent:.1f}% [{step}/{self.total_steps}] ETA: {eta_str} {info_str}",
            end="",
            flush=True,
        )

    def close(self):
        """Close progress bar."""
        print()  # New line


def log_system_info():
    """Log system information for debugging."""
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            info[f"gpu_{i}"] = {
                "name": gpu_props.name,
                "memory_gb": gpu_props.total_memory / (1024**3),
                "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
            }

    return info


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

_SPARK = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float], width: int = 40) -> str:
    if not values:
        return ""
    vals = values[-width:]
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span <= 1e-12:
        return _SPARK[0] * len(vals)
    return "".join(_SPARK[min(len(_SPARK) - 1, int((v - lo) / span * (len(_SPARK) - 1)))] for v in vals)


def _fmt_eta(seconds: float) -> str:
    if seconds <= 0 or seconds != seconds:  # <=0 or NaN
        return "--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class LiveDashboard:
    """Accumulates training metrics and renders them live."""

    def __init__(self, total_steps: int, *, model_name: str = "", history: int = 60, enabled: bool = True):
        self.total_steps = int(total_steps) if total_steps and total_steps > 0 else 0
        self.model_name = model_name
        self.start_time = time.time()
        self.loss_hist: deque[float] = deque(maxlen=history)
        self.best_loss = float("inf")
        self.ema_loss: float | None = None
        self.enabled = enabled

        self._live = None
        self._console = None
        if not enabled:
            return
        try:
            from rich.console import Console
            from rich.live import Live

            self._console = Console()
            self._live = Live(self._render(0, 0.0, 0.0, 0.0, {}), console=self._console, refresh_per_second=8)
            self._live.start()
        except Exception:
            # rich missing or non-tty: fall back to plain prints in update().
            self._live = None

    def _trend(self) -> str:
        if len(self.loss_hist) < 6:
            return "warming up"
        recent = list(self.loss_hist)
        first = sum(recent[: len(recent) // 3]) / max(1, len(recent) // 3)
        last = sum(recent[-len(recent) // 3 :]) / max(1, len(recent) // 3)
        if last < first * 0.995:
            return "improving ↓"
        if last > first * 1.02:
            return "rising ↑ (watch LR / divergence)"
        return "flat →"

    def _health(self) -> str:
        if len(self.loss_hist) < 4:
            return "starting"
        trend = self._trend()
        if "rising" in trend:
            return "unstable — check LR / batch"
        if "improving" in trend:
            return "healthy"
        return "plateau — normal late-training"

    def _render(self, step: int, loss: float, steps_per_sec: float, lr: float, extra: dict):
        try:
            from rich.panel import Panel
            from rich.table import Table
        except Exception:
            return ""

        elapsed = time.time() - self.start_time
        frac = (step / self.total_steps) if self.total_steps else 0.0
        eta = (self.total_steps - step) / steps_per_sec if (self.total_steps and steps_per_sec > 0) else 0.0
        bar_w = 34
        filled = int(frac * bar_w)
        bar = "█" * filled + "░" * (bar_w - filled)

        table = Table(show_header=False, box=None, pad_edge=False)
        table.add_column(justify="right", style="cyan", no_wrap=True)
        table.add_column(style="white")

        pct = f"{frac * 100:5.1f}%" if self.total_steps else "  n/a"
        step_str = f"{step:,}/{self.total_steps:,}" if self.total_steps else f"{step:,}"
        table.add_row("progress", f"{bar} {pct}   step {step_str}")
        table.add_row(
            "loss", f"{loss:.4f}   best {self.best_loss:.4f}   ema {self.ema_loss or loss:.4f}   [{self._trend()}]"
        )
        table.add_row("health", self._health())
        table.add_row("trend", _sparkline(list(self.loss_hist)))
        gpu = extra.get("gpu_mem", "")
        table.add_row(
            "throughput", f"{steps_per_sec:.2f} steps/s   {extra.get('img_per_sec', 0.0):.1f} img/s   lr {lr:.2e}"
        )
        table.add_row("time", f"elapsed {_fmt_eta(elapsed)}   eta {_fmt_eta(eta)}" + (f"   gpu {gpu}" if gpu else ""))
        aux = extra.get("aux", "")
        if aux:
            table.add_row("aux", aux)
        title = f"SDX training · {self.model_name}" if self.model_name else "SDX training"
        return Panel(table, title=title, border_style="green")

    def update(self, step: int, loss: float, steps_per_sec: float, lr: float, **extra) -> None:
        if not self.enabled:
            return
        if loss == loss:  # not NaN
            self.loss_hist.append(loss)
            self.best_loss = min(self.best_loss, loss)
            self.ema_loss = loss if self.ema_loss is None else (0.9 * self.ema_loss + 0.1 * loss)

        gpu = ""
        if torch is not None and torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu = f"{used:.1f}/{total:.0f}GB"
        extra.setdefault("gpu_mem", gpu)

        if self._live is not None:
            self._live.update(self._render(step, loss, steps_per_sec, lr, extra))
        else:
            eta = (self.total_steps - step) / steps_per_sec if (self.total_steps and steps_per_sec > 0) else 0.0
            print(
                f"[dash] step {step}/{self.total_steps or '?'} loss={loss:.4f} best={self.best_loss:.4f} "
                f"{steps_per_sec:.2f}it/s eta={_fmt_eta(eta)} {self._trend()} {gpu}",
                flush=True,
            )

    def close(self) -> None:
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None


def print_model_summary(model: nn.Module) -> None:
    """Total / trainable parameters, fp32 size and the parameter share per leaf-module type."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    by_type: dict[str, int] = {}
    n_layers = 0
    for module in model.modules():
        if next(module.children(), None) is not None:
            continue
        n = sum(p.numel() for p in module.parameters())
        by_type[type(module).__name__] = by_type.get(type(module).__name__, 0) + n
        n_layers += n > 0

    print("=" * 80)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 80)
    print(f"Total Parameters: {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print(f"Model Size: {total * 4 / (1024 * 1024):.2f} MB")
    print(f"Number of Layers: {n_layers}")
    print("\nParameter Distribution by Module Type:")
    print("-" * 50)
    for module_type, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
        print(f"{module_type:20s}: {count:>12,} ({100.0 * count / max(total, 1):5.1f}%)")
    print("=" * 80)


def format_module_tree(
    module: nn.Module,
    *,
    max_depth: int = 4,
    max_children: int = 12,
    _depth: int = 0,
    _indent: str = "",
) -> list[str]:
    """Compact ASCII lines: child name, type, subtree param count."""
    lines: list[str] = []
    if _depth == 0:
        lines.append(f"{type(module).__name__}  total_params={sum(p.numel() for p in module.parameters()):,}")
    if _depth >= max_depth:
        return lines

    all_children = list(module.named_children())
    for name, child in all_children[:max_children]:
        lines.append(f"{_indent}{name}  {type(child).__name__}  ({sum(p.numel() for p in child.parameters()):,})")
        lines.extend(
            format_module_tree(
                child, max_depth=max_depth, max_children=max_children, _depth=_depth + 1, _indent=_indent + "  "
            )
        )
    if _depth == 0 and len(all_children) > max_children:
        lines.append(f"{_indent}... ({len(all_children) - max_children} more direct children omitted)")
    return lines

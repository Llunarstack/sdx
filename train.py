"""
Fast training for text-conditioned DiT (PixArt/ReVe-style prompt adherence).
Step-based training (quality improves with steps), refinement (fix imperfections), optional DDP.

Optional profiling: ``--profile-out PATH`` (and ``--profile-sort``, ``--profile-top``) writes cProfile
output plus a sorted text summary (same convention as sample.py).
"""

import glob
import itertools
import logging
import math
import os
import platform
import subprocess
import sys
import warnings
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import time
from typing import Optional

import torch
import torch.distributed as dist

# Project imports (run from repo root: python train.py ...)
from config.train_config import TrainConfig, get_dit_build_kwargs
from data import ResolutionBucketBatchSampler, Text2ImageDataset, collate_t2i
from diffusion import create_diffusion
from diffusion.losses.timestep_loss_weight import get_timestep_loss_weight
from diffusion.timestep_sampling import sample_training_timesteps
from models import DiT_models_text
from models.attention import create_block_causal_mask_2d
from models.rae_latent_bridge import RAELatentBridge
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from training.train_args import build_train_config_from_args
from training.train_cli_parser import build_train_arg_parser
from utils.checkpoint.checkpoint_manager import CheckpointManager
from utils.runtime.jsonutil import dumps as json_dumps
from utils.training.ar_curriculum import parse_ar_order_mix, resolve_ar_for_step
from utils.training.config_validator import estimate_memory_usage, validate_train_config
from utils.training.error_handling import get_model_info, log_gpu_memory, setup_logging
from utils.training.metrics import MetricsTracker, log_system_info


def _maybe_ot_pair_noise(cfg, latents: torch.Tensor, device: torch.device) -> Optional[torch.Tensor]:
    """Sample Gaussian noise and optionally OT-couple it to batch latents (train only)."""
    reg = float(getattr(cfg, "ot_noise_pair_reg", 0.0) or 0.0)
    if reg <= 0.0 or latents.shape[0] < 2:
        return None
    from utils.training.ot_noise_pairing import pair_noise_to_latents

    n = torch.randn_like(latents, device=device, dtype=latents.dtype)
    return pair_noise_to_latents(
        latents,
        n,
        reg=reg,
        n_iters=int(getattr(cfg, "ot_noise_pair_iters", 40)),
        mode=str(getattr(cfg, "ot_noise_pair_mode", "soft")),
    )


from utils.modeling.model_viz import print_model_summary  # noqa: E402
from utils.modeling.text_encoder_bundle import load_text_encoder_bundle  # noqa: E402


def _sample_training_timesteps(cfg, num_timesteps: int, batch_size: int, device: torch.device) -> torch.Tensor:
    """Integer diffusion timestep indices for a training batch."""
    return sample_training_timesteps(
        batch_size,
        int(num_timesteps),
        device=device,
        mode=str(getattr(cfg, "timestep_sample_mode", "uniform")),
        logit_mean=float(getattr(cfg, "timestep_logit_mean", 0.0)),
        logit_std=float(getattr(cfg, "timestep_logit_std", 1.0)),
    )


def _spectral_flow_prediction_training_kwargs(cfg: TrainConfig) -> dict:
    """Kwargs for ``GaussianDiffusion.training_losses`` (SFP prototype; ignored for MDM masked path)."""
    return {
        "use_spectral_sfp_loss": bool(getattr(cfg, "spectral_sfp_loss", False)),
        "spectral_sfp_low_sigma": float(getattr(cfg, "spectral_sfp_low_sigma", 0.22)),
        "spectral_sfp_high_sigma": float(getattr(cfg, "spectral_sfp_high_sigma", 0.22)),
        "spectral_sfp_tau_power": float(getattr(cfg, "spectral_sfp_tau_power", 1.0)),
    }


def _sample_training_t(cfg, num_timesteps: int, batch_size: int, device: torch.device) -> torch.Tensor:
    """Backward-compatible alias for `_sample_training_timesteps`."""
    return _sample_training_timesteps(cfg, num_timesteps, batch_size, device)


def _spectral_sfp_training_kwargs(cfg: TrainConfig) -> dict:
    """Backward-compatible alias for `_spectral_flow_prediction_training_kwargs`."""
    return _spectral_flow_prediction_training_kwargs(cfg)


def _apply_runtime_ar(raw_model, *, num_ar_blocks: int, ar_block_order: str) -> None:
    """Update DiT AR regime on a live model (used by curriculum/order-mix training)."""
    b = int(num_ar_blocks)
    order = str(ar_block_order or "raster").strip().lower()
    setattr(raw_model, "num_ar_blocks", b)
    setattr(raw_model, "ar_block_order", order)
    if b <= 0:
        setattr(raw_model, "_ar_mask", None)
        return
    n_patches = int(getattr(raw_model, "num_patches", 0) or 0)
    if n_patches <= 0:
        setattr(raw_model, "_ar_mask", None)
        return
    p = int(round(n_patches**0.5))
    mask = create_block_causal_mask_2d(p, p, b, block_order=order)
    ref = getattr(raw_model, "pos_embed", None)
    if ref is not None and hasattr(ref, "device"):
        mask = mask.to(device=ref.device)
    setattr(raw_model, "_ar_mask", mask)


def _safe_git_info(repo_root: Path) -> dict:
    """Best-effort git metadata for reproducibility manifests."""
    def _run(args):
        try:
            return subprocess.check_output(args, cwd=str(repo_root), text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {
        "commit": commit or "unknown",
        "branch": branch or "unknown",
        "dirty": bool(status) if status is not None else None,
    }


def _cfg_to_dict(cfg: TrainConfig) -> dict:
    if hasattr(cfg, "__dict__"):
        return dict(cfg.__dict__)
    return {}


def _write_run_manifest(exp_dir: Path, cfg: TrainConfig, logger: logging.Logger) -> None:
    repo_root = Path(__file__).resolve().parent
    cfg_out = exp_dir / "config.train.json"
    manifest_out = exp_dir / "run_manifest.json"
    cfg_dict = _cfg_to_dict(cfg)
    cfg_out.write_text(json_dumps(cfg_dict, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": str(getattr(torch.version, "cuda", None)),
        "cudnn_version": int(torch.backends.cudnn.version() or 0),
        "world_size": int(getattr(cfg, "world_size", 1)),
        "local_rank": int(getattr(cfg, "local_rank", 0)),
        "seed": int(getattr(cfg, "global_seed", 0)),
        "git": _safe_git_info(repo_root),
        "config_file": cfg_out.name,
    }
    manifest_out.write_text(json_dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Run manifest saved: {manifest_out}")


# Enable TF32 on Ampere+ for speed
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Lazy heavy imports (T5, VAE) to speed up startup when only parsing args
_transformers = None


def get_rule_loss(batch, pixel_values, vae, device, cfg):
    """Constitutional/rule-based auxiliary loss (IMPROVEMENTS 8.2). Returns (B,) or None. Placeholder returns zeros."""
    B = pixel_values.shape[0]
    out = torch.zeros(B, device=device, dtype=pixel_values.dtype)
    # Optional: add real rules, e.g. "no text" via OCR on decoded image (decode pixel_values or use batch["path"] with PIL)
    return out


def _size_embed_from_latents(latents: torch.Tensor, device, dtype=torch.float32) -> torch.Tensor:
    """(B, 2) latent grid (H, W) for PixArt-style SizeEmbedder."""
    b, _, h, w = latents.shape
    return torch.stack(
        [
            torch.full((b,), float(h), device=device, dtype=dtype),
            torch.full((b,), float(w), device=device, dtype=dtype),
        ],
        dim=1,
    )


def _negatives_strip_emphasis(negative_captions, train_prompt_emphasis: bool):
    """Align negative T5 input with positive path when ``( )`` / ``[ ]`` stripping is enabled."""
    if not train_prompt_emphasis:
        return negative_captions
    from utils.prompt.prompt_emphasis import parse_prompt_emphasis

    return [parse_prompt_emphasis(n or "")[0] for n in negative_captions]


def get_t5_and_vae(device, cfg: TrainConfig):
    global _transformers
    if _transformers is None:
        import transformers

        _transformers = transformers

    tokenizer = _transformers.AutoTokenizer.from_pretrained(cfg.text_encoder)
    text_encoder = _transformers.T5EncoderModel.from_pretrained(cfg.text_encoder)
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder = text_encoder.to(device)

    # Autoencoder load (VAE=AutoencoderKL, or RAE=AutoencoderRAE).
    ae_type = getattr(cfg, "autoencoder_type", "kl")
    from diffusers import AutoencoderKL, AutoencoderRAE

    if ae_type == "rae":
        ae = AutoencoderRAE.from_pretrained(cfg.vae_model)
    else:
        ae = AutoencoderKL.from_pretrained(cfg.vae_model)

    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False
    ae = ae.to(device)

    return tokenizer, text_encoder, ae


def encode_text(
    captions,
    tokenizer,
    text_encoder,
    device,
    max_length=300,
    dtype=torch.bfloat16,
    text_bundle=None,
    train_fusion: bool = False,
    clip_captions=None,
    segment_texts=None,
):
    """Encode captions to hidden states (B, L, text_dim). Triple mode appends two CLIP tokens (train fusion when train_fusion=True)."""
    if text_bundle is not None:
        return text_bundle.encode(
            captions,
            device,
            max_length=max_length,
            dtype=dtype,
            train_fusion=train_fusion,
            clip_captions=clip_captions,
            segment_texts=segment_texts,
        )
    if segment_texts is not None:
        from utils.modeling.t5_segmented_encode import encode_t5_segment_concat

        return encode_t5_segment_concat(
            segment_texts, tokenizer, text_encoder, device, max_length=max_length, dtype=dtype
        )
    with torch.no_grad():
        tok = tokenizer(
            captions,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tok.input_ids.to(device)
        attention_mask = tok.attention_mask.to(device)
        out = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        return hidden.to(dtype)


@torch.no_grad()
def encode_images_vae(images: torch.Tensor, ae: torch.nn.Module, scale: float = 0.18215) -> torch.Tensor:
    """Encode images to latents.

    Args:
        images: ``(B, 3, H, W)`` float tensor in ``[-1, 1]``.
        ae: AutoencoderKL or AutoencoderRAE instance.
        scale: Latent scale factor (VAE only; RAE handles its own normalisation).

    Returns:
        Latent tensor — ``(B, 4, h, w)`` for KL-VAE, ``(B, C, h, w)`` for RAE.
    """
    enc = ae.encode(images)
    if hasattr(enc, "latent_dist"):
        return enc.latent_dist.sample() * scale
    if hasattr(enc, "latent"):
        return enc.latent
    raise AttributeError(
        f"Autoencoder encode() returned {type(enc).__name__!r} with neither "
        "'latent_dist' nor 'latent' attribute. Check your autoencoder type."
    )


# --- REPA (Representation Alignment) helpers ---
_repa_vision_model = None
_repa_mean = None
_repa_std = None
_repa_target_hw = 224
_repa_encoder_model_id = None


@torch.no_grad()
def _get_repa_vision(device, cfg):
    global _repa_vision_model, _repa_mean, _repa_std, _repa_target_hw, _repa_encoder_model_id
    encoder_id = getattr(cfg, "repa_encoder_model", None)
    if not encoder_id:
        return None
    if _repa_vision_model is not None and _repa_encoder_model_id == encoder_id:
        return _repa_vision_model

    # Lazy import: keep startup fast.
    from transformers import AutoImageProcessor

    if "dinov2" in str(encoder_id).lower():
        from transformers import Dinov2Model

        vision_model = Dinov2Model.from_pretrained(encoder_id)
        # DINOv2 returns last_hidden_state; we will use CLS token.
    elif "clip" in str(encoder_id).lower():
        from transformers import CLIPVisionModelWithProjection

        vision_model = CLIPVisionModelWithProjection.from_pretrained(encoder_id)
        # CLIP returns pooled image embedding in image_embeds.
    else:
        raise ValueError(f"Unsupported REPA encoder_model: {encoder_id}. Use dinov2 or clip checkpoints.")

    processor = AutoImageProcessor.from_pretrained(encoder_id)
    mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
    std = getattr(processor, "image_std", [0.229, 0.224, 0.225])
    _repa_mean = torch.tensor(mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    _repa_std = torch.tensor(std, device=device, dtype=torch.float32).view(1, 3, 1, 1)

    # Target size for resizing if processor expects a fixed input.
    target = getattr(processor, "crop_size", None) or getattr(processor, "size", None)
    hw = 224
    if isinstance(target, dict):
        hw = int(target.get("height") or target.get("shortest_edge") or target.get("width") or 224)
    elif isinstance(target, (int, float)):
        hw = int(target)
    _repa_target_hw = hw

    vision_model.eval()
    for p in vision_model.parameters():
        p.requires_grad = False
    vision_model = vision_model.to(device)

    _repa_vision_model = vision_model
    _repa_encoder_model_id = encoder_id
    return _repa_vision_model


@torch.no_grad()
def _repa_features(pixel_values: torch.Tensor, device, cfg) -> torch.Tensor:
    """pixel_values: (B,3,H,W) in [-1,1] -> features: (B,repa_out_dim)."""
    vision_model = _get_repa_vision(device, cfg)
    if vision_model is None:
        raise RuntimeError("REPA requested but encoder_model could not be loaded.")

    # [-1,1] -> [0,1]
    x = (pixel_values * 0.5 + 0.5).clamp(0.0, 1.0)
    x = torch.nn.functional.interpolate(
        x, size=(_repa_target_hw, _repa_target_hw), mode="bilinear", align_corners=False
    )
    x = (x.to(torch.float32) - _repa_mean) / _repa_std

    enc_id = str(getattr(cfg, "repa_encoder_model", "")).lower()
    if "dinov2" in enc_id:
        out = vision_model(pixel_values=x)
        feat = out.last_hidden_state[:, 0, :]  # CLS token
    else:
        out = vision_model(pixel_values=x)
        feat = out.image_embeds if hasattr(out, "image_embeds") else out.pooler_output
    return feat


def compute_mdm_training_loss(
    diffusion,
    model,
    x_start,
    t,
    model_kwargs,
    *,
    refinement_prob: float,
    refinement_max_t: int,
    noise_offset: float,
    min_snr_gamma: float,
    sample_weights,
    loss_weighting: str,
    loss_weighting_sigma_data: float,
    mdm_mask_ratio: float,
    mdm_mask_schedule,
    mdm_patch_size: int,
    mdm_loss_only_masked: bool,
    mdm_min_mask_patches: int,
    spectral_kwargs: Optional[dict] = None,
    prefetched_noise: Optional[torch.Tensor] = None,
):
    """
    Masked Diffusion Models (MDM) style training.
    - x_t_full is created with standard q_sample.
    - masked patches use x_t_full, unmasked patches are replaced with x_start (clean context).
    - loss is either computed only over masked regions or over full latents.
    """
    device = x_start.device
    B, C, H, W = x_start.shape
    sk = spectral_kwargs if spectral_kwargs is not None else {}

    if refinement_prob > 0 and refinement_max_t > 0 and torch.rand(1, device=device).item() < refinement_prob:
        t = torch.randint(0, min(refinement_max_t, diffusion.num_timesteps), (B,), device=device, dtype=t.dtype)

    # Standard forward diffusion for the whole latent (we'll overwrite parts with clean context).
    if prefetched_noise is not None:
        noise = prefetched_noise.to(device=device, dtype=x_start.dtype)
    else:
        noise = torch.randn_like(x_start, device=device, dtype=x_start.dtype)
    x_t_full = diffusion.q_sample(x_start, t, noise=noise, noise_offset=noise_offset)

    # Random patch masking in latent patch-grid space.
    if mdm_mask_ratio <= 0 and not mdm_mask_schedule:
        # Safety fallback: behave like normal diffusion training.
        return diffusion.training_losses(
            model,
            x_start,
            t,
            model_kwargs=model_kwargs,
            noise=noise,
            refinement_prob=0.0,
            refinement_max_t=0,
            noise_offset=noise_offset,
            min_snr_gamma=min_snr_gamma,
            sample_weights=sample_weights,
            loss_weighting=loss_weighting,
            loss_weighting_sigma_data=loss_weighting_sigma_data,
            **sk,
        )

    mdm_patch_size = int(mdm_patch_size)
    if mdm_patch_size <= 0 or H % mdm_patch_size != 0 or W % mdm_patch_size != 0:
        # If shape isn't divisible, don't try fancy patch masking.
        return diffusion.training_losses(
            model,
            x_start,
            t,
            model_kwargs=model_kwargs,
            noise=noise,
            refinement_prob=0.0,
            refinement_max_t=0,
            noise_offset=noise_offset,
            min_snr_gamma=min_snr_gamma,
            sample_weights=sample_weights,
            loss_weighting=loss_weighting,
            loss_weighting_sigma_data=loss_weighting_sigma_data,
            **sk,
        )

    ph = H // mdm_patch_size
    pw = W // mdm_patch_size
    sched = (
        sorted(((int(ts), float(r)) for ts, r in mdm_mask_schedule), key=lambda x: x[0]) if mdm_mask_schedule else []
    )
    if sched:
        first_t, first_r = sched[0]
        last_t, last_r = sched[-1]
        first_r_t = torch.tensor(first_r, device=device, dtype=torch.float32)
        last_r_t = torch.tensor(last_r, device=device, dtype=torch.float32)
    else:
        first_t = first_r_t = last_t = last_r_t = None

    def _scheduled_ratio_for_t(t_steps: torch.Tensor):
        if not sched:
            return torch.full_like(t_steps, float(mdm_mask_ratio), dtype=torch.float32, device=device)
        t_i = t_steps.to(torch.int64)
        default_r = float(mdm_mask_ratio)
        out = torch.full_like(t_steps, default_r, dtype=torch.float32, device=device)

        # Clamp ends to first/last schedule values.
        if len(sched) >= 1:
            out = torch.where(
                t_i <= int(first_t),
                first_r_t,
                out,
            )
            out = torch.where(
                t_i >= int(last_t),
                last_r_t,
                out,
            )

        # Linear interpolation between each consecutive (t, ratio) pair.
        for j in range(len(sched) - 1):
            t0, r0 = sched[j]
            t1, r1 = sched[j + 1]
            t0i, t1i = int(t0), int(t1)
            if t1i <= t0i:
                continue
            alpha_mask = (t_i >= t0i) & (t_i < t1i)
            tt = (t_i.to(torch.float32) - float(t0i)) / float(max(1, t1i - t0i))
            rr = float(r0) + tt * float(r1 - r0)
            out = torch.where(alpha_mask, rr, out)
        return out

    ratio_t = _scheduled_ratio_for_t(t).view(B, 1, 1, 1).to(device=device)
    patch_mask = (torch.rand(B, 1, ph, pw, device=device) < ratio_t).to(dtype=x_start.dtype)

    # Ensure each sample has at least mdm_min_mask_patches masked patches.
    if mdm_min_mask_patches > 0:
        flat = patch_mask.view(B, -1)
        counts = flat.sum(dim=1)
        zero_or_low = (counts < float(mdm_min_mask_patches)).nonzero(as_tuple=False).view(-1)
        if zero_or_low.numel() > 0:
            for bi in zero_or_low.tolist():
                # Flip random patches until we meet the minimum.
                while flat[bi].sum().item() < float(mdm_min_mask_patches):
                    zeros = (flat[bi] <= 0).nonzero(as_tuple=False).view(-1)
                    if zeros.numel() == 0:
                        break
                    pos = zeros[torch.randint(0, zeros.numel(), (1,), device=device)].item()
                    flat[bi, pos] = 1.0

    # Upsample patch mask to latent pixel resolution (nearest in patch space).
    mask_latent = patch_mask.repeat_interleave(mdm_patch_size, dim=2).repeat_interleave(mdm_patch_size, dim=3)
    mask_latent = mask_latent[..., :H, :W]

    # Replace unmasked regions with clean x_start context (so model learns to inpaint masked areas).
    x_t = mask_latent * x_t_full + (1 - mask_latent) * x_start

    model_out = model(x_t, t, **model_kwargs)
    if model_out.shape != x_start.shape and model_out.shape[1] > x_start.shape[1]:
        model_out = model_out[:, : x_start.shape[1]]

    diffusion._to_device(device)
    sqrt_alpha = diffusion.sqrt_alpha_cumprod.to(device)[t][(...,) + (None,) * (x_start.ndim - 1)]
    sqrt_one = diffusion.sqrt_one_minus_alpha_cumprod.to(device)[t][(...,) + (None,) * (x_start.ndim - 1)]

    if diffusion.prediction_type == "v":
        target = sqrt_alpha * noise - sqrt_one * x_start
    elif diffusion.prediction_type == "x0":
        target = x_start
    else:
        target = noise

    mse = torch.nn.functional.mse_loss(model_out, target, reduction="none")
    # Per-sample reduced loss:
    if mdm_loss_only_masked:
        mask_broadcast = mask_latent.expand_as(mse).to(mse.dtype)
        denom = mask_broadcast.sum(dim=(1, 2, 3)) + 1e-8
        loss_per_sample = (mse * mask_broadcast).sum(dim=(1, 2, 3)) / denom
    else:
        loss_per_sample = mse.mean(dim=tuple(range(1, mse.ndim)))

    # Timestep loss weighting (match diffusion.training_losses semantics).
    snr_t = diffusion.snr.to(device)[t] if hasattr(diffusion, "snr") else None
    alpha = diffusion.alpha_cumprod.to(device)[t]
    weight = get_timestep_loss_weight(
        loss_weighting,
        snr=snr_t,
        alpha_cumprod=alpha,
        min_snr_gamma=min_snr_gamma,
        loss_weighting_sigma_data=loss_weighting_sigma_data,
    )
    loss_per_sample = loss_per_sample * weight

    # Optional sample weighting.
    if sample_weights is not None and sample_weights.shape[0] == loss_per_sample.shape[0]:
        sample_weights = sample_weights.to(device, dtype=loss_per_sample.dtype)
        loss = (loss_per_sample * sample_weights).sum() / (sample_weights.sum() + 1e-8)
    else:
        loss = loss_per_sample.mean()

    return {"loss": loss}


def update_ema(ema_model, model, decay=0.9999):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)


@torch.no_grad()
def eval_val_loss(
    ema_model,
    diffusion,
    val_loader,
    tokenizer,
    text_encoder,
    vae,
    device,
    cfg,
    refinement_prob=0.0,
    refinement_max_t=150,
    max_caption_len=300,
    max_batches=None,
    rae_bridge=None,
    latent_scale=1.0,
    text_bundle=None,
):
    """Average loss over validation set (EMA model, no backward). Used for early stopping and best-by-val checkpoint."""
    ema_model.eval()
    if text_bundle is not None and text_bundle.fusion is not None:
        text_bundle.fusion.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in val_loader:
        if max_batches is not None and n_batches >= max_batches:
            break
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        captions = batch["captions"]
        negative_captions = batch.get("negative_captions") or [""] * len(captions)
        styles = batch.get("styles") or [""] * len(captions)
        control_images = batch.get("control_image")
        control_type_ids = batch.get("control_type_id")
        with torch.amp.autocast("cuda", enabled=cfg.use_bf16, dtype=torch.bfloat16):
            if "latent_values" in batch:
                latents = batch["latent_values"].to(device, non_blocking=True).to(torch.bfloat16)
            else:
                latents = encode_images_vae(pixel_values, vae, latent_scale)
            if rae_bridge is not None and latents.shape[1] != 4:
                latents = rae_bridge.rae_to_dit(latents)
            train_pe = bool(getattr(cfg, "train_prompt_emphasis", False))
            pos_caps = captions
            token_weights = None
            if train_pe:
                from utils.prompt.prompt_emphasis import batch_encoder_token_weights

                pos_caps, token_weights = batch_encoder_token_weights(
                    captions,
                    tokenizer,
                    max_caption_len,
                    device=device,
                    dtype=torch.bfloat16,
                    text_bundle=text_bundle,
                )
            encoder_hidden = encode_text(
                pos_caps,
                tokenizer,
                text_encoder,
                device,
                dtype=torch.bfloat16,
                max_length=max_caption_len,
                text_bundle=text_bundle,
                train_fusion=False,
            )
            neg_caps = _negatives_strip_emphasis(negative_captions, train_pe)
            encoder_hidden_neg = None
            if any(n and n.strip() for n in neg_caps):
                encoder_hidden_neg = encode_text(
                    [n or "" for n in neg_caps],
                    tokenizer,
                    text_encoder,
                    device,
                    dtype=torch.bfloat16,
                    max_length=max_caption_len,
                    text_bundle=text_bundle,
                    train_fusion=False,
                )
            style_embedding = None
            if getattr(cfg, "style_embed_dim", 0) and any(s and s.strip() for s in styles):
                style_embedding = encode_text(
                    [s or "" for s in styles],
                    tokenizer,
                    text_encoder,
                    device,
                    dtype=torch.bfloat16,
                    max_length=max_caption_len,
                    text_bundle=text_bundle,
                    train_fusion=False,
                )
                style_embedding = style_embedding.mean(dim=1)
            t = _sample_training_timesteps(cfg, diffusion.num_timesteps, latents.shape[0], device)
            model_kwargs = {
                "encoder_hidden_states": encoder_hidden,
                "encoder_hidden_states_negative": encoder_hidden_neg,
                "negative_prompt_weight": getattr(cfg, "negative_prompt_weight", 0.5),
            }
            if token_weights is not None:
                model_kwargs["token_weights"] = token_weights
            if style_embedding is not None:
                model_kwargs["style_embedding"] = style_embedding
                model_kwargs["style_strength"] = getattr(cfg, "style_strength", 0.7)
            if control_images is not None:
                model_kwargs["control_image"] = control_images.to(device, non_blocking=True).to(torch.bfloat16)
                model_kwargs["control_scale"] = getattr(cfg, "control_scale", 0.85)
                if control_type_ids is not None:
                    model_kwargs["control_type"] = control_type_ids.to(device, non_blocking=True).to(torch.long)
            if getattr(cfg, "creativity_embed_dim", 0) > 0:
                creativity_max = getattr(cfg, "creativity_max", 1.0)
                # Val: no creativity_jitter_std (smoother metric; jitter is train-only).
                model_kwargs["creativity"] = (
                    torch.rand(latents.shape[0], device=device, dtype=torch.bfloat16) * creativity_max
                )
            if int(getattr(cfg, "size_embed_dim", 0) or 0) > 0:
                model_kwargs["size_embed"] = _size_embed_from_latents(latents, device)
            sample_weights = batch.get("sample_weights")
            if sample_weights is not None:
                sample_weights = sample_weights.to(device)
            ot_noise_val = _maybe_ot_pair_noise(cfg, latents, device)
            _sk_v = _spectral_flow_prediction_training_kwargs(cfg)
            _bw_v = float(getattr(cfg, "bridge_aux_weight", 0.0))

            def _bridge_aux_loss_val() -> torch.Tensor:
                from diffusion.bridge_training import bridge_aux_vp_loss

                t_br = _sample_training_timesteps(cfg, diffusion.num_timesteps, latents.shape[0], device)
                return bridge_aux_vp_loss(
                    diffusion,
                    ema_model,
                    latents,
                    t_br,
                    model_kwargs,
                    mix_lambda=float(getattr(cfg, "bridge_aux_lambda", 0.2)),
                    noise=None,
                    noise_offset=getattr(cfg, "noise_offset", 0.0),
                    min_snr_gamma=getattr(cfg, "min_snr_gamma", 5.0),
                    loss_weighting=getattr(cfg, "loss_weighting", "min_snr"),
                    loss_weighting_sigma_data=getattr(cfg, "loss_weighting_sigma_data", 0.5),
                    **_sk_v,
                )

            if getattr(cfg, "flow_matching_training", False):
                from diffusion.flow_matching import flow_matching_per_sample_losses

                eps_v = (
                    ot_noise_val
                    if ot_noise_val is not None
                    else torch.randn_like(latents, device=device, dtype=latents.dtype)
                )
                fp_v = flow_matching_per_sample_losses(
                    ema_model, latents, eps_v, diffusion.num_timesteps, model_kwargs
                )
                if sample_weights is not None:
                    wv = sample_weights.view(-1).to(dtype=fp_v.dtype)
                    lm_v = (fp_v * wv).sum() / (wv.sum() + 1e-8)
                else:
                    lm_v = fp_v.mean()
                loss_dict = {
                    "loss": lm_v + _bw_v * _bridge_aux_loss_val() if _bw_v > 0.0 else lm_v
                }
            elif float(getattr(cfg, "mdm_mask_ratio", 0.0)) > 0 or getattr(cfg, "mdm_mask_schedule", None):
                loss_dict = compute_mdm_training_loss(
                    diffusion,
                    ema_model,
                    latents,
                    t,
                    model_kwargs,
                    refinement_prob=refinement_prob,
                    refinement_max_t=refinement_max_t,
                    noise_offset=getattr(cfg, "noise_offset", 0.0),
                    min_snr_gamma=getattr(cfg, "min_snr_gamma", 5.0),
                    sample_weights=sample_weights,
                    loss_weighting=getattr(cfg, "loss_weighting", "min_snr"),
                    loss_weighting_sigma_data=getattr(cfg, "loss_weighting_sigma_data", 0.5),
                    mdm_mask_ratio=float(getattr(cfg, "mdm_mask_ratio", 0.0)),
                    mdm_mask_schedule=getattr(cfg, "mdm_mask_schedule", None),
                    mdm_patch_size=int(getattr(cfg, "mdm_patch_size", 2)),
                    mdm_loss_only_masked=bool(getattr(cfg, "mdm_loss_only_masked", True)),
                    mdm_min_mask_patches=int(getattr(cfg, "mdm_min_mask_patches", 1)),
                    spectral_kwargs=_sk_v,
                    prefetched_noise=ot_noise_val,
                )
                if _bw_v > 0.0:
                    loss_dict = {"loss": loss_dict["loss"] + _bw_v * _bridge_aux_loss_val()}
            else:
                loss_dict = diffusion.training_losses(
                    ema_model,
                    latents,
                    t,
                    model_kwargs=model_kwargs,
                    refinement_prob=refinement_prob,
                    refinement_max_t=refinement_max_t,
                    noise_offset=getattr(cfg, "noise_offset", 0.0),
                    min_snr_gamma=getattr(cfg, "min_snr_gamma", 5.0),
                    sample_weights=sample_weights,
                    loss_weighting=getattr(cfg, "loss_weighting", "min_snr"),
                    loss_weighting_sigma_data=getattr(cfg, "loss_weighting_sigma_data", 0.5),
                    noise=ot_noise_val,
                    **_sk_v,
                )
                if _bw_v > 0.0:
                    loss_dict = {"loss": loss_dict["loss"] + _bw_v * _bridge_aux_loss_val()}
            total_loss += loss_dict["loss"].mean().item()
        n_batches += 1
    ema_model.train()
    if text_bundle is not None and text_bundle.fusion is not None:
        text_bundle.fusion.train()
    return total_loss / max(1, n_batches)


def get_lr_cosine(step: int, max_steps: int, warmup_steps: int, base_lr: float, min_lr: float) -> float:
    """Cosine decay to min_lr after warmup. Quality keeps improving with steps (no sudden drops)."""
    if step < warmup_steps:
        return base_lr * (step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, progress)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def get_curriculum_max_length(step: int, steps_list, lengths_list) -> int:
    """Return max caption length for current step (curriculum). Default 300 if not configured."""
    if not steps_list or not lengths_list or len(steps_list) != len(lengths_list):
        return 300
    out = lengths_list[0]
    for s, L in zip(steps_list, lengths_list):
        if step >= s:
            out = L
    return out


def get_scheduled_caption_dropout(step: int, schedule) -> float:
    """IMPROVEMENTS 1.3: schedule is list of (step, prob). Linear interpolate between points. Returns 0 if schedule None/empty."""
    if not schedule or len(schedule) == 0:
        return 0.0
    schedule = sorted(schedule, key=lambda x: x[0])
    if step <= schedule[0][0]:
        return float(schedule[0][1])
    if step >= schedule[-1][0]:
        return float(schedule[-1][1])
    for i in range(len(schedule) - 1):
        s0, p0 = schedule[i][0], float(schedule[i][1])
        s1, p1 = schedule[i + 1][0], float(schedule[i + 1][1])
        if s0 <= step < s1:
            t = (step - s0) / max(1, s1 - s0)
            return p0 + t * (p1 - p0)
    return float(schedule[-1][1])


def _log_sample_image(
    ema_model,
    diffusion,
    tokenizer,
    text_encoder,
    vae,
    device,
    cfg,
    steps,
    tb_writer,
    rae_bridge=None,
    text_bundle=None,
):
    """Generate one sample with EMA and log to TensorBoard/WandB."""
    was_training = ema_model.training
    ema_model.eval()
    try:
        demo_prompt = getattr(cfg, "log_images_prompt", "a photo of a cat")
        with torch.no_grad():
            enc = encode_text(
                [demo_prompt],
                tokenizer,
                text_encoder,
                device,
                max_length=77,
                dtype=torch.bfloat16,
                text_bundle=text_bundle,
                train_fusion=False,
            )
            ae_type = getattr(cfg, "autoencoder_type", "kl")
            latent_scale = getattr(cfg, "latent_scale", 0.18215) if ae_type == "kl" else 1.0
            image_size = getattr(cfg, "image_size", 256)
            latent_size = image_size // 8
            shape = (1, 4, latent_size, latent_size)
            sample_steps = min(20, getattr(cfg, "num_timesteps", 1000) // 25)
            mk_cond = {"encoder_hidden_states": enc}
            if int(getattr(cfg, "size_embed_dim", 0) or 0) > 0:
                mk_cond["size_embed"] = torch.tensor(
                    [[float(latent_size), float(latent_size)]], device=device, dtype=torch.float32
                )
            x0 = diffusion.sample_loop(
                ema_model,
                shape,
                model_kwargs_cond=mk_cond,
                model_kwargs_uncond=None,
                cfg_scale=7.5,
                num_inference_steps=sample_steps,
                eta=0.0,
                device=device,
                dtype=torch.float32,
            )
            if ae_type == "kl":
                x0 = x0 / latent_scale
            elif ae_type == "rae" and rae_bridge is not None:
                x0 = rae_bridge.dit_to_rae(x0)
            img = vae.decode(x0).sample
            img = (img * 0.5 + 0.5).clamp(0, 1)
            img = img[0].permute(1, 2, 0).cpu().numpy()
            img = (img * 255).round().astype("uint8")
    finally:
        ema_model.train(was_training)
    if tb_writer is not None:
        tb_writer.add_image("sample", img, steps, dataformats="HWC")
    if getattr(cfg, "wandb_project", None):
        try:
            import wandb

            wandb.log({"sample": wandb.Image(img)}, step=steps)
        except Exception:
            pass


def create_logger(log_dir):
    if dist.is_initialized() and dist.get_rank() != 0:
        logger = logging.getLogger("train")
        logger.addHandler(logging.NullHandler())
        return logger
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(log_dir) / "log.txt") if log_dir else logging.StreamHandler(),
        ],
    )
    return logging.getLogger("train")


def main(cfg: TrainConfig):
    assert torch.cuda.is_available(), "CUDA required"

    # Enhanced error handling and validation
    logger = setup_logging()

    if bool(getattr(cfg, "strict_warnings", False)):
        warnings.filterwarnings("error", category=UserWarning, module=r"^(train|config|data|diffusion|models|training|utils)(\.|$)")
        warnings.filterwarnings(
            "error", category=FutureWarning, module=r"^(train|config|data|diffusion|models|training|utils)(\.|$)"
        )
        logger.info("Strict warnings enabled: project UserWarning/FutureWarning are treated as errors.")

    # Validate configuration
    logger.info("Validating configuration...")
    config_issues = validate_train_config(cfg)
    if config_issues:
        for issue in config_issues:
            if issue.startswith("ERROR"):
                logger.error(issue)
                raise ValueError(f"Configuration error: {issue}")
            else:
                logger.warning(issue)

    # Log system information
    system_info = log_system_info()
    logger.info(f"System info: {system_info}")

    # Memory estimation
    memory_est = estimate_memory_usage(cfg)
    logger.info(f"Estimated memory usage: {memory_est['total_estimated_gb']:.1f}GB")

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
    # IMPROVEMENTS 5.3: full reproducibility when --deterministic
    if getattr(cfg, "deterministic", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True

    # Experiment dir with enhanced checkpoint management
    if rank == 0:
        Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
        exp_index = len(glob.glob(f"{cfg.results_dir}/*"))
        exp_dir = Path(cfg.results_dir) / f"{exp_index:03d}-{cfg.model_name.replace('/', '-')}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = exp_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)

        # Initialize enhanced utilities
        _checkpoint_manager = CheckpointManager(str(ckpt_dir))
        _metrics_tracker = MetricsTracker(str(exp_dir))

        logger = create_logger(str(exp_dir))
    else:
        exp_dir = Path(cfg.results_dir) / "run"
        logger = create_logger(None)

    if rank == 0 and bool(getattr(cfg, "save_run_manifest", True)):
        try:
            _write_run_manifest(exp_dir, cfg, logger)
        except Exception as e:
            logger.warning(f"Failed to write run manifest: {e}")

    # Log GPU memory before loading models
    log_gpu_memory(logger, "Before model loading: ")

    # T5 + Autoencoder (frozen)
    logger.info("Loading T5 and autoencoder...")
    tokenizer, text_encoder, vae = get_t5_and_vae(device, cfg)
    text_bundle = load_text_encoder_bundle(cfg, device)
    if text_bundle is not None:
        logger.info("Triple text encoder enabled: T5 + CLIP-L + CLIP-bigG (fusion layers trained with DiT).")

    ae_type = getattr(cfg, "autoencoder_type", "kl")
    effective_latent_scale = getattr(cfg, "latent_scale", 0.18215) if ae_type == "kl" else 1.0

    # Compatibility guard: the current DiT architecture in this repo assumes SD-style latents:
    # - channel count = 4
    # - spatial downsample factor = 8 (latent_hw = image_size // 8)
    # Representation Autoencoders (RAE) typically produce different latent channel sizes (e.g. 768),
    # so we fail fast with a clear message instead of crashing later in the diffusion loop.
    rae_bridge = None
    if ae_type == "rae":
        latent_hw_expected = int(cfg.image_size) // 8
        latent_channels_expected = 4
        ae_cfg = getattr(vae, "config", None)
        latent_channels_rae = getattr(ae_cfg, "encoder_hidden_size", None) if ae_cfg is not None else None
        encoder_input_size = getattr(ae_cfg, "encoder_input_size", None) if ae_cfg is not None else None
        encoder_patch_size = getattr(ae_cfg, "encoder_patch_size", None) if ae_cfg is not None else None
        latent_hw_rae = None
        if encoder_input_size is not None and encoder_patch_size is not None:
            try:
                latent_hw_rae = int(encoder_input_size) // int(encoder_patch_size)
            except Exception:
                latent_hw_rae = None

        if latent_channels_rae is not None and int(latent_channels_rae) != latent_channels_expected:
            if getattr(cfg, "rae_use_latent_bridge", True):
                rae_bridge = RAELatentBridge(int(latent_channels_rae), latent_channels_expected).to(device)
                logger.info(
                    f"RAE encoder_hidden_size={latent_channels_rae} -> DiT 4ch via RAELatentBridge "
                    f"(cycle loss weight={getattr(cfg, 'rae_bridge_cycle_weight', 0.0)})."
                )
            else:
                raise ValueError(
                    "AutoencoderRAE selected, but this repo's DiT expects 4-channel SD latents. "
                    f"RAE encoder_hidden_size={latent_channels_rae}, expected={latent_channels_expected}. "
                    "Use default --rae-use-latent-bridge (omit --no-rae-latent-bridge) or change autoencoder."
                )
        if latent_hw_rae is not None and int(latent_hw_rae) != latent_hw_expected:
            raise ValueError(
                "AutoencoderRAE selected, but the RAE latent spatial size doesn't match this repo's assumptions. "
                f"RAE latent_hw={latent_hw_rae}, expected={latent_hw_expected} (image_size//8). "
                "Resize training --image-size or use an RAE matching latent_hw = image_size//8."
            )

    # DiT text model (build from cfg via get_dit_build_kwargs)
    model_fn = DiT_models_text.get(cfg.model_name)
    if model_fn is None:
        raise ValueError(f"Unknown model {cfg.model_name}. Choices: {list(DiT_models_text.keys())}")
    # When caption_dropout_schedule is set, we apply dropout in the loop; use 0 for model's built-in dropout
    build_kw = get_dit_build_kwargs(
        cfg, class_dropout_prob=0.0 if getattr(cfg, "caption_dropout_schedule", None) else None
    )
    model = model_fn(**build_kw).to(device)

    # Log model information
    if rank == 0:
        model_info = get_model_info(model)
        logger.info(f"Model info: {model_info}")
        print_model_summary(model)

    if cfg.grad_checkpointing:
        model.enable_gradient_checkpointing()
    ema = deepcopy(model)
    for p in ema.parameters():
        p.requires_grad = False

    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

    if rank == 0:
        logger.info(f"Model: {cfg.model_name}")

    diffusion = create_diffusion(
        timestep_respacing=cfg.timestep_respacing,
        num_timesteps=cfg.num_timesteps,
        beta_schedule=cfg.beta_schedule,
        prediction_type=getattr(cfg, "prediction_type", "epsilon"),
    )

    opt_params = list(model.parameters())
    if rae_bridge is not None:
        opt_params += list(rae_bridge.parameters())
    if text_bundle is not None and text_bundle.fusion is not None:
        opt_params += list(text_bundle.fusion.parameters())
    opt = torch.optim.AdamW(
        opt_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_bf16)

    # Resume from checkpoint
    start_step = 0
    best_loss = float("inf")
    if getattr(cfg, "resume", None):
        resume_path = Path(cfg.resume)
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
            raw = model.module if use_ddp else model
            raw.load_state_dict(ckpt.get("model", ckpt.get("ema")), strict=True)
            if "ema" in ckpt and ckpt["ema"]:
                ema.load_state_dict(ckpt["ema"], strict=True)
            if "opt" in ckpt and ckpt["opt"]:
                try:
                    opt.load_state_dict(ckpt["opt"])
                except Exception:
                    pass
            start_step = ckpt.get("steps", 0)
            best_loss = ckpt.get("best_loss", float("inf"))
            if rae_bridge is not None and ckpt.get("rae_latent_bridge"):
                try:
                    rae_bridge.load_state_dict(ckpt["rae_latent_bridge"], strict=True)
                    logger.info("Loaded rae_latent_bridge from checkpoint.")
                except Exception as e:
                    logger.warning(f"Could not load rae_latent_bridge: {e}")
            if text_bundle is not None and text_bundle.fusion is not None and ckpt.get("text_encoder_fusion"):
                try:
                    text_bundle.fusion.load_state_dict(ckpt["text_encoder_fusion"], strict=True)
                    logger.info("Loaded text_encoder_fusion from checkpoint.")
                except Exception as e:
                    logger.warning(f"Could not load text_encoder_fusion: {e}")
            logger.info(f"Resumed from {resume_path} at step {start_step}")

    # Data
    data_path = cfg.manifest_jsonl or cfg.data_path
    if not data_path:
        raise ValueError("Set --data-path or --manifest-jsonl")

    if getattr(cfg, "dry_run", False):
        cfg.max_steps = 1
        logger.info("Dry run: 1 step then exit.")

    val_split = getattr(cfg, "val_split", 0.0)
    res_buckets = getattr(cfg, "resolution_buckets", None) or []
    latent_dir = getattr(cfg, "latent_cache_dir", None)
    if res_buckets:
        if use_ddp:
            raise ValueError(
                "--resolution-buckets is not supported with multi-GPU DDP yet. "
                "Use a single GPU or omit --resolution-buckets."
            )
        if val_split > 0 and val_split < 1:
            raise ValueError(
                "--resolution-buckets requires --val-split 0 (no train/val split) in this version."
            )
        for h, w in res_buckets:
            if int(h) % 8 != 0 or int(w) % 8 != 0:
                raise ValueError(f"Resolution bucket ({h},{w}) must be divisible by 8 (VAE stride).")
        if latent_dir:
            logger.warning("Disabling latent cache (--latent-cache-dir) when using resolution buckets.")
            latent_dir = None

    max_steps_early = int(getattr(cfg, "max_steps", 0) or 0)
    passes_early = int(getattr(cfg, "passes", 0) or 0)
    bucket_fixed_assign = max_steps_early > 0 or passes_early > 0

    dataset = Text2ImageDataset(
        data_path,
        image_size=cfg.image_size,
        latent_cache_dir=latent_dir,
        crop_mode=getattr(cfg, "crop_mode", "center"),
        region_caption_mode=getattr(cfg, "region_caption_mode", "append"),
        region_layout_tag=getattr(cfg, "region_layout_tag", "[layout]"),
        use_adherence_boost=bool(getattr(cfg, "boost_adherence_caption", False)),
        train_shortcomings_mitigation=str(getattr(cfg, "train_shortcomings_mitigation", "none") or "none"),
        train_shortcomings_2d=bool(getattr(cfg, "train_shortcomings_2d", False)),
        train_art_guidance_mode=str(getattr(cfg, "train_art_guidance_mode", "none") or "none"),
        train_art_guidance_photography=bool(getattr(cfg, "train_art_guidance_photography", True)),
        train_anatomy_guidance=str(getattr(cfg, "train_anatomy_guidance", "none") or "none"),
        train_style_guidance_mode=str(getattr(cfg, "train_style_guidance_mode", "none") or "none"),
        train_style_guidance_artists=bool(getattr(cfg, "train_style_guidance_artists", True)),
        caption_unicode_normalize=bool(getattr(cfg, "caption_unicode_normalize", False)),
        resolution_buckets=res_buckets if res_buckets else None,
        bucket_seed=int(getattr(cfg, "global_seed", 42)),
        bucket_fixed_assign=bucket_fixed_assign,
        use_hierarchical_captions=bool(getattr(cfg, "use_hierarchical_captions", False)),
        hierarchical_caption_separator=str(getattr(cfg, "hierarchical_caption_separator", " | ")),
        hierarchical_drop_global_p=float(getattr(cfg, "hierarchical_caption_drop_global_p", 0.0)),
        hierarchical_drop_local_p=float(getattr(cfg, "hierarchical_caption_drop_local_p", 0.0)),
        foveated_train_prob=float(getattr(cfg, "foveated_train_prob", 0.0)),
        foveated_crop_frac=float(getattr(cfg, "foveated_crop_frac", 0.55)),
        grounding_mask_soft=bool(getattr(cfg, "grounding_mask_soft", False)),
    )

    def _worker_init(worker_id):
        import numpy as np

        np.random.seed(cfg.global_seed + rank * 1000 + worker_id)
        torch.manual_seed(cfg.global_seed + rank * 1000 + worker_id)

    val_loader = None
    train_dataset = dataset
    if val_split > 0 and val_split < 1:
        n_total = len(dataset)
        n_val = max(1, int(n_total * val_split))
        n_train = n_total - n_val
        train_dataset, val_dataset = random_split(
            dataset, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.global_seed)
        )
        if rank == 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.per_device_batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_t2i,
                persistent_workers=cfg.num_workers > 0,
                worker_init_fn=_worker_init if getattr(cfg, "deterministic", False) else None,
            )
        logger.info(f"Train/val split: {len(train_dataset)} train, {len(val_dataset)} val")
    else:
        val_dataset = None

    sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.global_seed)
        if use_ddp
        else None
    )
    if res_buckets:
        bucket_sampler = ResolutionBucketBatchSampler(
            train_dataset,
            cfg.per_device_batch_size,
            drop_last=True,
            shuffle_batches=True,
            generator=torch.Generator().manual_seed(cfg.global_seed),
        )
        loader = DataLoader(
            train_dataset,
            batch_sampler=bucket_sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_t2i,
            persistent_workers=cfg.num_workers > 0,
            worker_init_fn=_worker_init if getattr(cfg, "deterministic", False) else None,
        )
    else:
        loader = DataLoader(
            train_dataset,
            batch_size=cfg.per_device_batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_t2i,
            persistent_workers=cfg.num_workers > 0,
            worker_init_fn=_worker_init if getattr(cfg, "deterministic", False) else None,
            generator=torch.Generator().manual_seed(cfg.global_seed + rank)
            if getattr(cfg, "deterministic", False)
            else None,
        )
    logger.info(f"Train size: {len(train_dataset)}, batch per device: {cfg.per_device_batch_size}")

    # Training length: passes (best) > max_steps > epochs
    max_steps = getattr(cfg, "max_steps", 0)
    passes = getattr(cfg, "passes", 0)
    if passes > 0:
        if res_buckets:
            steps_per_epoch = max(1, len(loader))
        else:
            steps_per_epoch = max(1, len(train_dataset) // cfg.global_batch_size)
        steps_from_passes = passes * steps_per_epoch
        max_steps = steps_from_passes if max_steps <= 0 else min(steps_from_passes, max_steps)
        logger.info(
            f"Passes-based: {passes} passes × {steps_per_epoch} steps/epoch = {steps_from_passes} steps (max_steps={max_steps})"
        )
    use_step_based = max_steps > 0
    if use_step_based:
        loader = itertools.cycle(loader)
        logger.info(f"Step-based training: max_steps={max_steps}, cosine LR to min_lr={getattr(cfg, 'min_lr', 1e-6)}")
    else:
        logger.info(f"Epoch-based training: epochs={cfg.epochs}")

    # Dynamic AR schedules can mutate attention masks/orders during training.
    ar_curriculum_mode = str(getattr(cfg, "ar_curriculum_mode", "none") or "none").strip().lower()
    ar_order_mix_spec = str(getattr(cfg, "ar_order_mix", "") or "")
    ar_order_mix_list = parse_ar_order_mix(ar_order_mix_spec)
    dynamic_ar_runtime = (ar_curriculum_mode != "none") or bool(ar_order_mix_list)
    raw_model_for_ar = model.module if use_ddp else model
    last_runtime_ar: Optional[tuple[int, str]] = None
    if dynamic_ar_runtime:
        b0, o0 = resolve_ar_for_step(
            start_step,
            base_blocks=int(getattr(cfg, "num_ar_blocks", 0)),
            base_order=str(getattr(cfg, "ar_block_order", "raster") or "raster"),
            curriculum_mode=ar_curriculum_mode,
            warmup_steps=int(getattr(cfg, "ar_curriculum_warmup_steps", 0)),
            ramp_start=int(getattr(cfg, "ar_curriculum_ramp_start", 0)),
            ramp_end=int(getattr(cfg, "ar_curriculum_ramp_end", 0)),
            curriculum_start_blocks=int(getattr(cfg, "ar_curriculum_start_blocks", -1)),
            curriculum_target_blocks=int(getattr(cfg, "ar_curriculum_target_blocks", -1)),
            order_mix=ar_order_mix_spec,
        )
        _apply_runtime_ar(raw_model_for_ar, num_ar_blocks=b0, ar_block_order=o0)
        last_runtime_ar = (int(b0), str(o0))
        if rank == 0:
            logger.info(
                "Dynamic AR enabled: mode=%s warmup=%d ramp=[%d,%d] mix=%s initial=(blocks=%d order=%s)",
                ar_curriculum_mode,
                int(getattr(cfg, "ar_curriculum_warmup_steps", 0)),
                int(getattr(cfg, "ar_curriculum_ramp_start", 0)),
                int(getattr(cfg, "ar_curriculum_ramp_end", 0)),
                ",".join(ar_order_mix_list) if ar_order_mix_list else "(none)",
                b0,
                o0,
            )

    # Compile for speed (PyTorch 2+)
    train_model = model
    if dynamic_ar_runtime and cfg.use_compile:
        logger.warning("Skipping torch.compile because dynamic AR runtime is enabled (curriculum/order-mix).")
    elif cfg.use_compile and hasattr(torch, "compile"):
        try:
            train_model = torch.compile(model, mode="reduce-overhead")
            logger.info("Model compiled with torch.compile(mode='reduce-overhead')")
        except Exception as e:
            logger.warning(f"compile failed: {e}, continuing without compile")

    steps = start_step
    log_steps = 0
    running_loss = 0.0
    running_ag_sum = 0.0
    running_ag_count = 0
    running_cov_sum = 0.0
    running_cov_count = 0
    start_time = time()
    refinement_prob = getattr(cfg, "refinement_prob", 0.0)
    refinement_max_t = getattr(cfg, "refinement_max_t", 150)
    curriculum_steps = getattr(cfg, "curriculum_caption_steps", None) or []
    curriculum_lengths = getattr(cfg, "curriculum_max_lengths", None) or [300]
    save_best = getattr(cfg, "save_best", True)
    val_every = getattr(cfg, "val_every", 2000)
    early_stopping_patience = getattr(cfg, "early_stopping_patience", 0)
    use_val_for_best = val_loader is not None
    best_val_loss = float("inf")
    no_improvement_count = 0
    epoch = -1
    empty_embed_cache = {}  # IMPROVEMENTS 1.3: cache empty caption embed per max_caption_len
    polyak_buf = None  # IMPROVEMENTS 1.5: running average of model weights (initialized when save_polyak > 0)
    tb_writer = None
    if rank == 0 and getattr(cfg, "tensorboard_dir", None):
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter(log_dir=str(Path(cfg.tensorboard_dir) / Path(cfg.results_dir).name))
            logger.info(f"TensorBoard logging to {cfg.tensorboard_dir}")
        except Exception as e:
            logger.warning(f"TensorBoard not available: {e}")
    if rank == 0 and getattr(cfg, "wandb_project", None):
        try:
            import wandb

            wandb.init(project=cfg.wandb_project, config=cfg.__dict__ if hasattr(cfg, "__dict__") else {})
            logger.info(f"WandB project: {cfg.wandb_project}")
        except Exception as e:
            logger.warning(f"WandB not available: {e}")

    while True:
        if not use_step_based:
            epoch += 1
            if epoch >= cfg.epochs:
                break
            if use_ddp:
                sampler.set_epoch(epoch)
            if res_buckets:
                train_dataset.set_epoch(epoch)
        batch_iter = loader

        for batch in batch_iter:
            last_attn_grounding: Optional[float] = None
            last_attn_cov: Optional[float] = None
            if dynamic_ar_runtime:
                b_now, o_now = resolve_ar_for_step(
                    steps,
                    base_blocks=int(getattr(cfg, "num_ar_blocks", 0)),
                    base_order=str(getattr(cfg, "ar_block_order", "raster") or "raster"),
                    curriculum_mode=ar_curriculum_mode,
                    warmup_steps=int(getattr(cfg, "ar_curriculum_warmup_steps", 0)),
                    ramp_start=int(getattr(cfg, "ar_curriculum_ramp_start", 0)),
                    ramp_end=int(getattr(cfg, "ar_curriculum_ramp_end", 0)),
                    curriculum_start_blocks=int(getattr(cfg, "ar_curriculum_start_blocks", -1)),
                    curriculum_target_blocks=int(getattr(cfg, "ar_curriculum_target_blocks", -1)),
                    order_mix=ar_order_mix_spec,
                )
                curr = (int(b_now), str(o_now))
                if curr != last_runtime_ar:
                    _apply_runtime_ar(raw_model_for_ar, num_ar_blocks=curr[0], ar_block_order=curr[1])
                    last_runtime_ar = curr
                    if rank == 0:
                        logger.info("AR runtime update @step=%d -> num_ar_blocks=%d ar_block_order=%s", steps, curr[0], curr[1])
            # LR schedule (step-based: cosine after warmup)
            if use_step_based:
                lr = get_lr_cosine(
                    steps,
                    max_steps,
                    getattr(cfg, "lr_warmup_steps", 500),
                    cfg.lr,
                    getattr(cfg, "min_lr", 1e-6),
                )
                for g in opt.param_groups:
                    g["lr"] = lr

            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            captions = batch["captions"]
            if float(getattr(cfg, "train_originality_augment_prob", 0.0)) > 0:
                import numpy as np
                from utils.prompt.originality_augment import inject_originality_tokens

                prob = float(cfg.train_originality_augment_prob)
                ost = max(0.0, min(1.0, float(getattr(cfg, "train_originality_strength", 0.5))))
                rng_np = np.random.default_rng((int(cfg.global_seed) + int(steps) * 1_000_003 + int(rank)) % (2**32))
                aug: list = []
                for cap in captions:
                    if rng_np.random() < prob:
                        aug.append(inject_originality_tokens(cap or "", ost, rng_np))
                    else:
                        aug.append(cap)
                captions = aug
            negative_captions = batch.get("negative_captions") or [""] * len(captions)
            styles = batch.get("styles") or [""] * len(captions)
            control_images = batch.get("control_image")
            control_type_ids = batch.get("control_type_id")

            max_caption_len = (
                get_curriculum_max_length(steps, curriculum_steps, curriculum_lengths) if curriculum_steps else 300
            )

            with torch.amp.autocast("cuda", enabled=cfg.use_bf16, dtype=torch.bfloat16):
                with torch.no_grad():
                    if "latent_values" in batch:
                        latents = batch["latent_values"].to(device, non_blocking=True).to(torch.bfloat16)
                    else:
                        latents = encode_images_vae(pixel_values, vae, effective_latent_scale)
                    # Img2img training (FLUX/NoobAI-style): sometimes use init_image as x_start so model learns to edit from it
                    init_pixel_values = batch.get("init_pixel_values")
                    if init_pixel_values is not None and getattr(cfg, "img2img_prob", 0) > 0:
                        use_init = torch.rand(latents.shape[0], device=device, dtype=torch.bfloat16) < cfg.img2img_prob
                        if use_init.any():
                            init_latents = encode_images_vae(
                                init_pixel_values.to(device, non_blocking=True), vae, effective_latent_scale
                            )
                            latents = torch.where(use_init.view(-1, 1, 1, 1).expand_as(latents), init_latents, latents)
                    train_pe = bool(getattr(cfg, "train_prompt_emphasis", False))
                    pos_caps = captions
                    token_weights = None
                    if train_pe:
                        from utils.prompt.prompt_emphasis import batch_encoder_token_weights

                        pos_caps, token_weights = batch_encoder_token_weights(
                            captions,
                            tokenizer,
                            max_caption_len,
                            device=device,
                            dtype=torch.bfloat16,
                            text_bundle=text_bundle,
                        )
                    encoder_hidden = encode_text(
                        pos_caps,
                        tokenizer,
                        text_encoder,
                        device,
                        dtype=torch.bfloat16,
                        max_length=max_caption_len,
                        text_bundle=text_bundle,
                        train_fusion=True,
                    )
                    # IMPROVEMENTS 1.3: scheduled caption dropout (replace with empty with prob p); only when schedule set
                    cap_drop_schedule = getattr(cfg, "caption_dropout_schedule", None)
                    if cap_drop_schedule:
                        current_cap_dropout = get_scheduled_caption_dropout(steps, cap_drop_schedule)
                        if current_cap_dropout > 0:
                            B = encoder_hidden.shape[0]
                            if max_caption_len not in empty_embed_cache:
                                empty_embed_cache[max_caption_len] = encode_text(
                                    [""],
                                    tokenizer,
                                    text_encoder,
                                    device,
                                    dtype=torch.bfloat16,
                                    max_length=max_caption_len,
                                    text_bundle=text_bundle,
                                    train_fusion=True,
                                )
                            empty_embed = empty_embed_cache[max_caption_len].expand(B, -1, -1)
                            mask = (torch.rand(B, device=device, dtype=torch.bfloat16) < current_cap_dropout).view(
                                B, 1, 1
                            )
                            encoder_hidden = encoder_hidden * (1 - mask) + empty_embed * mask
                            if token_weights is not None:
                                mw = mask.squeeze(-1)  # (B, 1)
                                token_weights = token_weights * (1 - mw) + mw.expand_as(token_weights)
                    # Negative prompt: encode so model can try really hard not to add those features
                    neg_caps = _negatives_strip_emphasis(negative_captions, train_pe)
                    encoder_hidden_neg = None
                    if any(n and n.strip() for n in neg_caps):
                        encoder_hidden_neg = encode_text(
                            [n or "" for n in neg_caps],
                            tokenizer,
                            text_encoder,
                            device,
                            dtype=torch.bfloat16,
                            max_length=max_caption_len,
                            text_bundle=text_bundle,
                            train_fusion=True,
                        )
                    # Style conditioning (T5-encoded style text, blended with strength)
                    style_embedding = None
                    if getattr(cfg, "style_embed_dim", 0) and any(s and s.strip() for s in styles):
                        style_embedding = encode_text(
                            [s or "" for s in styles],
                            tokenizer,
                            text_encoder,
                            device,
                            dtype=torch.bfloat16,
                            max_length=max_caption_len,
                            text_bundle=text_bundle,
                            train_fusion=True,
                        )
                        style_embedding = style_embedding.mean(dim=1)
                latents_raw = latents
                if rae_bridge is not None and latents_raw.shape[1] != 4:
                    latents = rae_bridge.rae_to_dit(latents_raw)
                    z_cycle = latents_raw.detach()
                else:
                    z_cycle = None
                t = _sample_training_timesteps(cfg, diffusion.num_timesteps, latents.shape[0], device)
                model_kwargs = {
                    "encoder_hidden_states": encoder_hidden,
                    "encoder_hidden_states_negative": encoder_hidden_neg,
                    "negative_prompt_weight": getattr(cfg, "negative_prompt_weight", 0.5),
                }
                if token_weights is not None:
                    model_kwargs["token_weights"] = token_weights
                if style_embedding is not None:
                    model_kwargs["style_embedding"] = style_embedding
                    model_kwargs["style_strength"] = getattr(cfg, "style_strength", 0.7)
                if control_images is not None:
                    model_kwargs["control_image"] = control_images.to(device, non_blocking=True).to(torch.bfloat16)
                    model_kwargs["control_scale"] = getattr(cfg, "control_scale", 0.85)
                    if control_type_ids is not None:
                        model_kwargs["control_type"] = control_type_ids.to(device, non_blocking=True).to(torch.long)
                if getattr(cfg, "creativity_embed_dim", 0) > 0:
                    creativity_max = getattr(cfg, "creativity_max", 1.0)
                    cj = float(getattr(cfg, "creativity_jitter_std", 0.0))
                    cre = torch.rand(latents.shape[0], device=device, dtype=torch.bfloat16) * creativity_max
                    if cj > 0:
                        cre = (cre + torch.randn(latents.shape[0], device=device, dtype=torch.bfloat16) * cj).clamp(
                            0.0, creativity_max
                        )
                    model_kwargs["creativity"] = cre
                if int(getattr(cfg, "size_embed_dim", 0) or 0) > 0:
                    model_kwargs["size_embed"] = _size_embed_from_latents(latents, device)
                sample_weights = batch.get("sample_weights")
                if sample_weights is not None:
                    sample_weights = sample_weights.to(device)
                curriculum_diff_steps = getattr(cfg, "curriculum_difficulty_steps", None) or []
                if curriculum_diff_steps and "difficulty" in batch:
                    diff = batch["difficulty"].to(device)
                    easy_first = getattr(cfg, "curriculum_difficulty_easy_first", True)
                    if steps < curriculum_diff_steps[0]:
                        w = 1.0 - diff if easy_first else 1.0 + diff
                    elif len(curriculum_diff_steps) >= 2 and steps >= curriculum_diff_steps[-1]:
                        w = 1.0 + diff if easy_first else 1.0 - diff
                    else:
                        w = torch.ones_like(diff)
                    if sample_weights is not None:
                        sample_weights = sample_weights * w
                    else:
                        sample_weights = w
                ot_noise = _maybe_ot_pair_noise(cfg, latents, device)
                attn_gw = float(getattr(cfg, "attn_grounding_loss_weight", 0.0))
                is_flow = bool(getattr(cfg, "flow_matching_training", False))
                is_mdm = float(getattr(cfg, "mdm_mask_ratio", 0.0)) > 0 or getattr(
                    cfg, "mdm_mask_schedule", None
                )
                gv = batch.get("grounding_mask_valid")
                has_mask_supervision = "grounding_mask" in batch and (
                    gv is None or bool(gv.any().item())
                )
                use_attn_grounding = (
                    attn_gw > 0.0
                    and has_mask_supervision
                    and not is_flow
                    and not is_mdm
                )
                cov_w = float(getattr(cfg, "attn_token_coverage_loss_weight", 0.0))
                use_attn_coverage = cov_w > 0.0 and not is_flow and not is_mdm
                use_attn_aux = use_attn_grounding or use_attn_coverage
                noise_for_vp = ot_noise
                if use_attn_aux and noise_for_vp is None:
                    noise_for_vp = torch.randn_like(latents, device=device, dtype=latents.dtype)
                _sk_tr = _spectral_flow_prediction_training_kwargs(cfg)
                _bw = float(getattr(cfg, "bridge_aux_weight", 0.0))

                def _bridge_aux_loss() -> torch.Tensor:
                    from diffusion.bridge_training import bridge_aux_vp_loss

                    t_br = _sample_training_timesteps(cfg, diffusion.num_timesteps, latents.shape[0], device)
                    return bridge_aux_vp_loss(
                        diffusion,
                        train_model,
                        latents,
                        t_br,
                        model_kwargs,
                        mix_lambda=float(getattr(cfg, "bridge_aux_lambda", 0.2)),
                        noise=None,
                        noise_offset=getattr(cfg, "noise_offset", 0.0),
                        min_snr_gamma=getattr(cfg, "min_snr_gamma", 5.0),
                        loss_weighting=getattr(cfg, "loss_weighting", "min_snr"),
                        loss_weighting_sigma_data=getattr(cfg, "loss_weighting_sigma_data", 0.5),
                        **_sk_tr,
                    )

                if getattr(cfg, "flow_matching_training", False):
                    from diffusion.flow_matching import flow_matching_per_sample_losses

                    eps_fm = (
                        ot_noise
                        if ot_noise is not None
                        else torch.randn_like(latents, device=device, dtype=latents.dtype)
                    )
                    fp = flow_matching_per_sample_losses(
                        train_model, latents, eps_fm, diffusion.num_timesteps, model_kwargs
                    )
                    if sample_weights is not None:
                        w = sample_weights.view(-1).to(dtype=fp.dtype)
                        loss_main = (fp * w).sum() / (w.sum() + 1e-8)
                    else:
                        loss_main = fp.mean()
                    loss_dict = {
                        "loss": loss_main + _bw * _bridge_aux_loss() if _bw > 0.0 else loss_main
                    }
                elif float(getattr(cfg, "mdm_mask_ratio", 0.0)) > 0 or getattr(cfg, "mdm_mask_schedule", None):
                    loss_dict = compute_mdm_training_loss(
                        diffusion,
                        train_model,
                        latents,
                        t,
                        model_kwargs,
                        refinement_prob=refinement_prob,
                        refinement_max_t=refinement_max_t,
                        noise_offset=getattr(cfg, "noise_offset", 0.0),
                        min_snr_gamma=getattr(cfg, "min_snr_gamma", 5.0),
                        sample_weights=sample_weights,
                        loss_weighting=getattr(cfg, "loss_weighting", "min_snr"),
                        loss_weighting_sigma_data=getattr(cfg, "loss_weighting_sigma_data", 0.5),
                        mdm_mask_ratio=float(getattr(cfg, "mdm_mask_ratio", 0.0)),
                        mdm_mask_schedule=getattr(cfg, "mdm_mask_schedule", None),
                        mdm_patch_size=int(getattr(cfg, "mdm_patch_size", 2)),
                        mdm_loss_only_masked=bool(getattr(cfg, "mdm_loss_only_masked", True)),
                        mdm_min_mask_patches=int(getattr(cfg, "mdm_min_mask_patches", 1)),
                        spectral_kwargs=_sk_tr,
                        prefetched_noise=ot_noise,
                    )
                    if _bw > 0.0:
                        loss_dict = {"loss": loss_dict["loss"] + _bw * _bridge_aux_loss()}
                else:
                    loss_dict = diffusion.training_losses(
                        train_model,
                        latents,
                        t,
                        model_kwargs=model_kwargs,
                        refinement_prob=refinement_prob,
                        refinement_max_t=refinement_max_t,
                        noise_offset=getattr(cfg, "noise_offset", 0.0),
                        min_snr_gamma=getattr(cfg, "min_snr_gamma", 5.0),
                        sample_weights=sample_weights,
                        loss_weighting=getattr(cfg, "loss_weighting", "min_snr"),
                        loss_weighting_sigma_data=getattr(cfg, "loss_weighting_sigma_data", 0.5),
                        noise=noise_for_vp if use_attn_grounding else ot_noise,
                        **_sk_tr,
                    )
                    if _bw > 0.0:
                        loss_dict = {"loss": loss_dict["loss"] + _bw * _bridge_aux_loss()}
                loss = loss_dict["loss"].mean()
                if use_attn_aux and hasattr(train_model, "num_patches"):
                    from utils.training.part_aware_training import (
                        capture_dit_block0_cross_attn,
                        grounding_loss_from_attn,
                        token_coverage_loss_from_attn,
                    )

                    attn_w = capture_dit_block0_cross_attn(
                        train_model=train_model,
                        diffusion=diffusion,
                        latents_bchw=latents,
                        t=t,
                        model_kwargs=model_kwargs,
                        training_noise=noise_for_vp,
                        noise_offset=float(getattr(cfg, "noise_offset", 0.0)),
                    )
                    if use_attn_grounding:
                        gm = batch["grounding_mask"].to(device=device, dtype=torch.float32)
                        gv_t = batch.get("grounding_mask_valid")
                        if gv_t is not None:
                            gv_t = gv_t.to(device=device)
                        ag = grounding_loss_from_attn(
                            attn_w=attn_w,
                            train_model=train_model,
                            grounding_mask_b1hw=gm,
                            token_start=int(getattr(cfg, "attn_grounding_token_start", 0)),
                            token_end=int(getattr(cfg, "attn_grounding_token_end", 0)),
                            sample_valid=gv_t,
                            min_fg_patch_mass=float(getattr(cfg, "attn_grounding_min_fg_patch_mass", 0.0)),
                        )
                        loss = loss + attn_gw * ag
                        last_attn_grounding = float(ag.detach().float().cpu().item())
                    if use_attn_coverage:
                        cov = token_coverage_loss_from_attn(
                            attn_w=attn_w,
                            token_start=int(getattr(cfg, "attn_grounding_token_start", 0)),
                            token_end=int(getattr(cfg, "attn_grounding_token_end", 0)),
                            target_coverage=float(getattr(cfg, "attn_token_coverage_target", 0.025)),
                            sample_valid=None,
                            token_weights=token_weights if token_weights is not None else None,
                        )
                        loss = loss + cov_w * cov
                        last_attn_cov = float(cov.detach().float().cpu().item())
                # MoE router balance auxiliary loss (optional).
                moe_w = float(getattr(cfg, "moe_balance_loss_weight", 0.0))
                moe_aux_loss = getattr(train_model, "moe_aux_loss", None)
                if moe_w > 0 and moe_aux_loss is not None:
                    loss = loss + moe_w * moe_aux_loss.to(loss.dtype)

                # REPA: align DiT internal representation with a frozen vision encoder.
                repa_w = float(getattr(cfg, "repa_weight", 0.0))
                if repa_w > 0:
                    repa_pred = getattr(train_model, "_repa_projected", None)
                    if repa_pred is not None:
                        repa_pred = repa_pred.to(device=device)
                        repa_target = _repa_features(pixel_values, device=device, cfg=cfg).to(dtype=repa_pred.dtype)
                        repa_loss = torch.nn.functional.mse_loss(repa_pred, repa_target)
                        loss = loss + repa_w * repa_loss
                if rae_bridge is not None and z_cycle is not None:
                    cw = float(getattr(cfg, "rae_bridge_cycle_weight", 0.0))
                    if cw > 0:
                        loss = loss + cw * rae_bridge.cycle_loss(z_cycle)
                rule_loss_weight = getattr(cfg, "rule_loss_weight", 0.0)
                if rule_loss_weight > 0:
                    rule_loss = get_rule_loss(batch, pixel_values, vae, device, cfg)
                    if rule_loss is not None and rule_loss.numel() > 0 and rule_loss.isfinite().all():
                        loss = loss + rule_loss_weight * rule_loss.mean()
                loss = loss / cfg.grad_accum_steps

            scaler.scale(loss).backward()

            if (steps + 1) % cfg.grad_accum_steps == 0:
                if cfg.max_grad_norm > 0:
                    scaler.unscale_(opt)
                    gradient_params = list(model.parameters())
                    if rae_bridge is not None:
                        gradient_params += list(rae_bridge.parameters())
                    torch.nn.utils.clip_grad_norm_(gradient_params, cfg.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                update_ema(ema, model.module if use_ddp else model, cfg.ema_decay)
                # IMPROVEMENTS 1.5: Polyak (running average of last N steps)
                save_polyak = getattr(cfg, "save_polyak", 0)
                if save_polyak > 0 and rank == 0:
                    raw = model.module if use_ddp else model
                    sd = raw.state_dict()
                    if polyak_buf is None:
                        polyak_buf = {k: v.detach().cpu().clone() for k, v in sd.items()}
                    else:
                        inv_n = 1.0 / save_polyak
                        for k in polyak_buf:
                            polyak_buf[k].mul_(1.0 - inv_n).add_(sd[k].detach().cpu(), alpha=inv_n)

            running_loss += loss.item() * cfg.grad_accum_steps
            if last_attn_grounding is not None:
                running_ag_sum += last_attn_grounding
                running_ag_count += 1
            if last_attn_cov is not None:
                running_cov_sum += last_attn_cov
                running_cov_count += 1
            log_steps += 1
            steps += 1

            if steps % cfg.log_every == 0:
                torch.cuda.synchronize()
                elapsed = time() - start_time
                steps_per_sec = log_steps / elapsed if elapsed > 0 else 0
                avg_loss = running_loss / log_steps
                if use_ddp:
                    t = torch.tensor([avg_loss], device=device)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    avg_loss = t.item() / world_size
                lr_val = opt.param_groups[0]["lr"] if opt.param_groups else 0.0
                lr_str = f" lr={lr_val:.2e}" if use_step_based else ""
                ag_extra = ""
                avg_ag = None
                if running_ag_count > 0:
                    avg_ag = running_ag_sum / running_ag_count
                    ag_extra = f" attn_grnd={avg_ag:.4f}"
                cov_extra = ""
                avg_cov = None
                if running_cov_count > 0:
                    avg_cov = running_cov_sum / running_cov_count
                    cov_extra = f" attn_cov={avg_cov:.4f}"
                logger.info(
                    f"step={steps:07d} epoch={epoch} loss={avg_loss:.4f} steps/s={steps_per_sec:.2f}{lr_str}{ag_extra}{cov_extra}"
                )
                # IMPROVEMENTS 5.1: TensorBoard / WandB
                if rank == 0:
                    if tb_writer is not None:
                        tb_writer.add_scalar("loss", avg_loss, steps)
                        tb_writer.add_scalar("lr", lr_val, steps)
                        if avg_ag is not None:
                            tb_writer.add_scalar("attn_grounding_aux", avg_ag, steps)
                        if avg_cov is not None:
                            tb_writer.add_scalar("attn_token_coverage_aux", avg_cov, steps)
                    if getattr(cfg, "wandb_project", None):
                        try:
                            import wandb

                            wl = {"loss": avg_loss, "lr": lr_val, "step": steps}
                            if avg_ag is not None:
                                wl["attn_grounding_aux"] = avg_ag
                            if avg_cov is not None:
                                wl["attn_token_coverage_aux"] = avg_cov
                            wandb.log(wl, step=steps)
                        except Exception:
                            pass
                # IMPROVEMENTS 5.1: log sample images to WandB/TensorBoard every log_images_every steps
                log_images_every = getattr(cfg, "log_images_every", 0)
                if (
                    rank == 0
                    and log_images_every > 0
                    and steps > 0
                    and steps % log_images_every == 0
                    and (tb_writer is not None or getattr(cfg, "wandb_project", None))
                ):
                    try:
                        _log_sample_image(
                            ema,
                            diffusion,
                            tokenizer,
                            text_encoder,
                            vae,
                            device,
                            cfg,
                            steps,
                            tb_writer,
                            rae_bridge=rae_bridge,
                            text_bundle=text_bundle,
                        )
                    except Exception as e:
                        logger.warning(f"Log sample image failed: {e}")
                # Save best by train loss only when not using validation (when using val, best is by val loss)
                if save_best and rank == 0 and not use_val_for_best and avg_loss < best_loss:
                    best_loss = avg_loss
                    raw = model.module if use_ddp else model
                    ckpt = {
                        "model": raw.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "steps": steps,
                        "best_loss": best_loss,
                        "config": cfg,
                    }
                    if rae_bridge is not None:
                        ckpt["rae_latent_bridge"] = rae_bridge.state_dict()
                    if text_bundle is not None and text_bundle.fusion is not None:
                        ckpt["text_encoder_fusion"] = text_bundle.fusion.state_dict()
                    path = ckpt_dir / "best.pt"
                    torch.save(ckpt, path)
                    logger.info(f"Best checkpoint saved: {path} (train loss={best_loss:.4f})")
                running_loss = 0.0
                running_ag_sum = 0.0
                running_ag_count = 0
                running_cov_sum = 0.0
                running_cov_count = 0
                log_steps = 0
                start_time = time()

            if steps > 0 and steps % cfg.ckpt_every == 0 and rank == 0:
                raw = model.module if use_ddp else model
                ckpt = {
                    "model": raw.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "steps": steps,
                    "config": cfg,
                }
                if rae_bridge is not None:
                    ckpt["rae_latent_bridge"] = rae_bridge.state_dict()
                if text_bundle is not None and text_bundle.fusion is not None:
                    ckpt["text_encoder_fusion"] = text_bundle.fusion.state_dict()
                path = ckpt_dir / f"{steps:07d}.pt"
                torch.save(ckpt, path)
                logger.info(f"Checkpoint saved: {path}")
                # IMPROVEMENTS 1.5: save Polyak average as polyak.pt
                if polyak_buf is not None and getattr(cfg, "save_polyak", 0) > 0:
                    polyak_ckpt = {"model": polyak_buf, "ema": polyak_buf, "steps": steps, "config": cfg}
                    torch.save(polyak_ckpt, ckpt_dir / "polyak.pt")
                    logger.info(f"Polyak checkpoint saved: {ckpt_dir / 'polyak.pt'}")

            # Validation + early stopping: save best by val loss, stop when no improvement
            should_stop = False
            if val_loader is not None and steps > 0 and steps % val_every == 0 and rank == 0:
                val_loss = eval_val_loss(
                    ema,
                    diffusion,
                    val_loader,
                    tokenizer,
                    text_encoder,
                    vae,
                    device,
                    cfg,
                    refinement_prob=refinement_prob,
                    refinement_max_t=refinement_max_t,
                    max_caption_len=curriculum_lengths[-1] if curriculum_lengths else 300,
                    max_batches=getattr(cfg, "val_max_batches", None),
                    rae_bridge=rae_bridge,
                    latent_scale=effective_latent_scale,
                    text_bundle=text_bundle,
                )
                logger.info(f"Val loss @ step {steps}: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improvement_count = 0
                    raw = model.module if use_ddp else model
                    ckpt = {
                        "model": raw.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "steps": steps,
                        "best_loss": best_val_loss,
                        "config": cfg,
                    }
                    if rae_bridge is not None:
                        ckpt["rae_latent_bridge"] = rae_bridge.state_dict()
                    if text_bundle is not None and text_bundle.fusion is not None:
                        ckpt["text_encoder_fusion"] = text_bundle.fusion.state_dict()
                    path = ckpt_dir / "best.pt"
                    torch.save(ckpt, path)
                    logger.info(f"Best checkpoint saved: {path} (val loss={best_val_loss:.4f})")
                else:
                    no_improvement_count += 1
                    logger.info(f"No val improvement ({no_improvement_count}/{early_stopping_patience})")
                if early_stopping_patience > 0 and no_improvement_count >= early_stopping_patience:
                    should_stop = True
                    logger.info(f"Early stopping: no val improvement for {early_stopping_patience} checks")
            if use_ddp and steps > 0 and steps % val_every == 0:
                stop_tensor = torch.tensor([1 if (rank == 0 and should_stop) else 0], device=device, dtype=torch.long)
                dist.broadcast(stop_tensor, src=0)
                should_stop = stop_tensor.item() == 1
            if should_stop:
                break

            if use_step_based and steps >= max_steps:
                break
        if use_step_based and steps >= max_steps:
            break

    if use_ddp:
        dist.destroy_process_group()
    logger.info("Training done.")


if __name__ == "__main__":
    from utils.runtime.profiling import consume_profile_args, run_with_cprofile

    _argv, _pcfg = consume_profile_args(sys.argv)
    sys.argv = _argv

    def _train_entry() -> None:
        parser = build_train_arg_parser()
        args = parser.parse_args()
        cfg = build_train_config_from_args(args)
        main(cfg)

    if _pcfg is not None:
        run_with_cprofile(_train_entry, _pcfg)
    else:
        _train_entry()

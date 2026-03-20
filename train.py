"""
Fast training for text-conditioned DiT (PixArt/ReVe-style prompt adherence).
Step-based training (quality improves with steps), refinement (fix imperfections), optional DDP.
"""
import argparse
import glob
import itertools
import logging
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
from time import time
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

# Project imports (run from repo root: python train.py ...)
from config.train_config import TrainConfig, get_dit_build_kwargs
try:
    from config.pixai_reference import get_pixai_style_label
except ImportError:
    def get_pixai_style_label(_model_name):
        return "PixAI.art-style"
from data import Text2ImageDataset, collate_t2i
from diffusion import create_diffusion
from diffusion.loss_weighting import get_loss_weight
from models import DiT_models_text
from models.rae_latent_bridge import RAELatentBridge
from utils.checkpoint_manager import CheckpointManager
from utils.error_handling import setup_logging, log_gpu_memory, get_model_info
from utils.config_validator import validate_train_config, estimate_memory_usage
from utils.metrics import MetricsTracker, log_system_info
from utils.model_viz import print_model_summary

# Enable TF32 on Ampere+ for speed
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Lazy heavy imports (T5, VAE) to speed up startup when only parsing args
_transformers = None
_diffusers = None


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


def get_t5_and_vae(device, cfg: TrainConfig):
    global _transformers, _diffusers
    if _transformers is None:
        import transformers
        _transformers = transformers
    if _diffusers is None:
        _diffusers = None

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


@torch.no_grad()
def encode_text(captions, tokenizer, text_encoder, device, max_length=300, dtype=torch.bfloat16):
    """Encode captions to T5 hidden states (B, L, 4096)."""
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
def encode_images_vae(images, ae, scale=0.18215):
    """images: (B,3,H,W) in [-1,1].

    Returns latents:
    - VAE (AutoencoderKL): (B,4,h,w) = encode().latent_dist.sample() * scale
    - RAE (AutoencoderRAE): (B,C,h,w) = encode().latent (normalization handled by checkpoint)
    """
    enc = ae.encode(images)
    if hasattr(enc, "latent_dist"):
        latents = enc.latent_dist.sample()
        return latents * scale
    if hasattr(enc, "latent"):
        return enc.latent
    # Fallback: assume VAE-like interface.
    latents = enc.latent_dist.sample()
    return latents * scale


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
    x = torch.nn.functional.interpolate(x, size=(_repa_target_hw, _repa_target_hw), mode="bilinear", align_corners=False)
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
):
    """
    Masked Diffusion Models (MDM) style training.
    - x_t_full is created with standard q_sample.
    - masked patches use x_t_full, unmasked patches are replaced with x_start (clean context).
    - loss is either computed only over masked regions or over full latents.
    """
    device = x_start.device
    B, C, H, W = x_start.shape

    if refinement_prob > 0 and refinement_max_t > 0 and torch.rand(1, device=device).item() < refinement_prob:
        t = torch.randint(0, min(refinement_max_t, diffusion.num_timesteps), (B,), device=device, dtype=t.dtype)

    # Standard forward diffusion for the whole latent (we'll overwrite parts with clean context).
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
        )

    ph = H // mdm_patch_size
    pw = W // mdm_patch_size
    sched = sorted(((int(ts), float(r)) for ts, r in mdm_mask_schedule), key=lambda x: x[0]) if mdm_mask_schedule else []
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
    if loss_weighting == "min_snr" and min_snr_gamma > 0 and hasattr(diffusion, "snr"):
        snr = diffusion.snr.to(device)[t]
        weight = torch.clamp(snr, max=min_snr_gamma) / (snr + 1e-8)
        loss_per_sample = loss_per_sample * weight
    elif loss_weighting != "min_snr":
        alpha = diffusion.alpha_cumprod.to(device)[t]
        weight = get_loss_weight(alpha, loss_weighting, loss_weighting_sigma_data)
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
):
    """Average loss over validation set (EMA model, no backward). Used for early stopping and best-by-val checkpoint."""
    ema_model.eval()
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
        with torch.amp.autocast("cuda", enabled=cfg.use_bf16, dtype=torch.bfloat16):
            if "latent_values" in batch:
                latents = batch["latent_values"].to(device, non_blocking=True).to(torch.bfloat16)
            else:
                latents = encode_images_vae(pixel_values, vae, latent_scale)
            if rae_bridge is not None and latents.shape[1] != 4:
                latents = rae_bridge.rae_to_dit(latents)
            encoder_hidden = encode_text(
                captions, tokenizer, text_encoder, device, dtype=torch.bfloat16, max_length=max_caption_len
            )
            encoder_hidden_neg = None
            if any(n and n.strip() for n in negative_captions):
                encoder_hidden_neg = encode_text(
                    [n or "" for n in negative_captions],
                    tokenizer, text_encoder, device, dtype=torch.bfloat16, max_length=max_caption_len,
                )
            style_embedding = None
            if getattr(cfg, "style_embed_dim", 0) and any(s and s.strip() for s in styles):
                style_embedding = encode_text(
                    [s or "" for s in styles],
                    tokenizer, text_encoder, device, dtype=torch.bfloat16, max_length=max_caption_len,
                )
                style_embedding = style_embedding.mean(dim=1)
            t = torch.randint(0, diffusion.num_timesteps, (latents.shape[0],), device=device)
            model_kwargs = {
                "encoder_hidden_states": encoder_hidden,
                "encoder_hidden_states_negative": encoder_hidden_neg,
                "negative_prompt_weight": getattr(cfg, "negative_prompt_weight", 0.5),
            }
            if style_embedding is not None:
                model_kwargs["style_embedding"] = style_embedding
                model_kwargs["style_strength"] = getattr(cfg, "style_strength", 0.7)
            if control_images is not None:
                model_kwargs["control_image"] = control_images.to(device, non_blocking=True).to(torch.bfloat16)
                model_kwargs["control_scale"] = getattr(cfg, "control_scale", 0.85)
            if getattr(cfg, "creativity_embed_dim", 0) > 0:
                creativity_max = getattr(cfg, "creativity_max", 1.0)
                model_kwargs["creativity"] = torch.rand(latents.shape[0], device=device, dtype=torch.bfloat16) * creativity_max
            if int(getattr(cfg, "size_embed_dim", 0) or 0) > 0:
                model_kwargs["size_embed"] = _size_embed_from_latents(latents, device)
            sample_weights = batch.get("sample_weights")
            if sample_weights is not None:
                sample_weights = sample_weights.to(device)
            if (
                float(getattr(cfg, "mdm_mask_ratio", 0.0)) > 0
                or getattr(cfg, "mdm_mask_schedule", None)
            ):
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
                )
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
                )
            total_loss += loss_dict["loss"].mean().item()
        n_batches += 1
    ema_model.train()
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


def parse_caption_dropout_schedule(s: Optional[str]):
    """Parse '0,0.2,10000,0.05' -> [(0, 0.2), (10000, 0.05)]. Returns None if s is None/empty."""
    if not s or not str(s).strip():
        return None
    parts = [x.strip() for x in str(s).split(",") if x.strip()]
    if len(parts) % 2 != 0:
        return None
    out = []
    for i in range(0, len(parts), 2):
        out.append((int(parts[i]), float(parts[i + 1])))
    return out if out else None


def parse_mdm_mask_schedule(s: Optional[str]):
    """
    Parse '0,0.05,500,0.25,999,0.35' -> [(0,0.05),(500,0.25),(999,0.35)].
    Returns None if s is None/empty.
    """
    if not s or not str(s).strip():
        return None
    parts = [x.strip() for x in str(s).split(",") if x.strip()]
    if len(parts) % 2 != 0:
        return None
    out = []
    for i in range(0, len(parts), 2):
        out.append((int(float(parts[i])), float(parts[i + 1])))
    return out if out else None


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


def _log_sample_image(ema_model, diffusion, tokenizer, text_encoder, vae, device, cfg, steps, tb_writer, rae_bridge=None):
    """Generate one sample with EMA and log to TensorBoard/WandB (IMPROVEMENTS 5.1)."""
    ema_model.eval()
    demo_prompt = getattr(cfg, "log_images_prompt", "a photo of a cat")
    with torch.no_grad():
        enc = encode_text([demo_prompt], tokenizer, text_encoder, device, max_length=77, dtype=torch.bfloat16)
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
            ema_model, shape,
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
    ema_model.train()
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

    # Log GPU memory before loading models
    log_gpu_memory(logger, "Before model loading: ")

    # T5 + Autoencoder (frozen)
    logger.info("Loading T5 and autoencoder...")
    tokenizer, text_encoder, vae = get_t5_and_vae(device, cfg)

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
    build_kw = get_dit_build_kwargs(cfg, class_dropout_prob=0.0 if getattr(cfg, "caption_dropout_schedule", None) else None)
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
        logger.info(f"Model: {cfg.model_name} — {get_pixai_style_label(cfg.model_name)}")

    diffusion = create_diffusion(
        timestep_respacing=cfg.timestep_respacing,
        num_timesteps=cfg.num_timesteps,
        beta_schedule=cfg.beta_schedule,
        prediction_type=getattr(cfg, "prediction_type", "epsilon"),
    )

    opt_params = list(model.parameters())
    if rae_bridge is not None:
        opt_params += list(rae_bridge.parameters())
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
            logger.info(f"Resumed from {resume_path} at step {start_step}")

    # Data
    data_path = cfg.manifest_jsonl or cfg.data_path
    if not data_path:
        raise ValueError("Set --data-path or --manifest-jsonl")
    dataset = Text2ImageDataset(
        data_path,
        image_size=cfg.image_size,
        latent_cache_dir=getattr(cfg, "latent_cache_dir", None),
        crop_mode=getattr(cfg, "crop_mode", "center"),
    )
    val_split = getattr(cfg, "val_split", 0.0)
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
                num_workers=0,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_t2i,
            )
        logger.info(f"Train/val split: {len(train_dataset)} train, {len(val_dataset)} val")
    else:
        val_dataset = None

    def _worker_init(worker_id):
        import numpy as np
        np.random.seed(cfg.global_seed + rank * 1000 + worker_id)
        torch.manual_seed(cfg.global_seed + rank * 1000 + worker_id)

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.global_seed) if use_ddp else None
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
        generator=torch.Generator().manual_seed(cfg.global_seed + rank) if getattr(cfg, "deterministic", False) else None,
    )
    logger.info(f"Train size: {len(train_dataset)}, batch per device: {cfg.per_device_batch_size}")

    # Training length: passes (best) > max_steps > epochs
    # IMPROVEMENTS 9: --dry-run runs one step and exits
    if getattr(cfg, "dry_run", False):
        cfg.max_steps = 1
        logger.info("Dry run: 1 step then exit.")
    max_steps = getattr(cfg, "max_steps", 0)
    passes = getattr(cfg, "passes", 0)
    if passes > 0:
        steps_per_epoch = max(1, len(train_dataset) // cfg.global_batch_size)
        steps_from_passes = passes * steps_per_epoch
        max_steps = steps_from_passes if max_steps <= 0 else min(steps_from_passes, max_steps)
        logger.info(f"Passes-based: {passes} passes × {steps_per_epoch} steps/epoch = {steps_from_passes} steps (max_steps={max_steps})")
    use_step_based = max_steps > 0
    if use_step_based:
        loader = itertools.cycle(loader)
        logger.info(f"Step-based training: max_steps={max_steps}, cosine LR to min_lr={getattr(cfg, 'min_lr', 1e-6)}")
    else:
        logger.info(f"Epoch-based training: epochs={cfg.epochs}")

    # Compile for speed (PyTorch 2+)
    train_model = model
    if cfg.use_compile and hasattr(torch, "compile"):
        try:
            train_model = torch.compile(model, mode="reduce-overhead")
            logger.info("Model compiled with torch.compile(mode='reduce-overhead')")
        except Exception as e:
            logger.warning(f"compile failed: {e}, continuing without compile")

    steps = start_step
    log_steps = 0
    running_loss = 0.0
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
        batch_iter = loader

        for batch in batch_iter:
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
            negative_captions = batch.get("negative_captions") or [""] * len(captions)
            styles = batch.get("styles") or [""] * len(captions)
            control_images = batch.get("control_image")

            max_caption_len = (
                get_curriculum_max_length(steps, curriculum_steps, curriculum_lengths)
                if curriculum_steps else 300
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
                            init_latents = encode_images_vae(init_pixel_values.to(device, non_blocking=True), vae, effective_latent_scale)
                            latents = torch.where(use_init.view(-1, 1, 1, 1).expand_as(latents), init_latents, latents)
                    encoder_hidden = encode_text(
                        captions, tokenizer, text_encoder, device, dtype=torch.bfloat16, max_length=max_caption_len
                    )
                    # IMPROVEMENTS 1.3: scheduled caption dropout (replace with empty with prob p); only when schedule set
                    cap_drop_schedule = getattr(cfg, "caption_dropout_schedule", None)
                    if cap_drop_schedule:
                        current_cap_dropout = get_scheduled_caption_dropout(steps, cap_drop_schedule)
                        if current_cap_dropout > 0:
                            B = encoder_hidden.shape[0]
                            if max_caption_len not in empty_embed_cache:
                                empty_embed_cache[max_caption_len] = encode_text(
                                    [""], tokenizer, text_encoder, device, dtype=torch.bfloat16, max_length=max_caption_len
                                )
                            empty_embed = empty_embed_cache[max_caption_len].expand(B, -1, -1)
                            mask = (torch.rand(B, device=device, dtype=torch.bfloat16) < current_cap_dropout).view(B, 1, 1)
                            encoder_hidden = encoder_hidden * (1 - mask) + empty_embed * mask
                    # Negative prompt: encode so model can try really hard not to add those features
                    encoder_hidden_neg = None
                    if any(n and n.strip() for n in negative_captions):
                        encoder_hidden_neg = encode_text(
                            [n or "" for n in negative_captions],
                            tokenizer, text_encoder, device, dtype=torch.bfloat16, max_length=max_caption_len,
                        )
                    # Style conditioning (T5-encoded style text, blended with strength)
                    style_embedding = None
                    if getattr(cfg, "style_embed_dim", 0) and any(s and s.strip() for s in styles):
                        style_embedding = encode_text(
                            [s or "" for s in styles],
                            tokenizer, text_encoder, device, dtype=torch.bfloat16, max_length=max_caption_len,
                        )
                        style_embedding = style_embedding.mean(dim=1)
                latents_raw = latents
                if rae_bridge is not None and latents_raw.shape[1] != 4:
                    latents = rae_bridge.rae_to_dit(latents_raw)
                    z_cycle = latents_raw.detach()
                else:
                    z_cycle = None
                t = torch.randint(0, diffusion.num_timesteps, (latents.shape[0],), device=device)
                model_kwargs = {
                    "encoder_hidden_states": encoder_hidden,
                    "encoder_hidden_states_negative": encoder_hidden_neg,
                    "negative_prompt_weight": getattr(cfg, "negative_prompt_weight", 0.5),
                }
                if style_embedding is not None:
                    model_kwargs["style_embedding"] = style_embedding
                    model_kwargs["style_strength"] = getattr(cfg, "style_strength", 0.7)
                if control_images is not None:
                    model_kwargs["control_image"] = control_images.to(device, non_blocking=True).to(torch.bfloat16)
                    model_kwargs["control_scale"] = getattr(cfg, "control_scale", 0.85)
                if getattr(cfg, "creativity_embed_dim", 0) > 0:
                    creativity_max = getattr(cfg, "creativity_max", 1.0)
                    model_kwargs["creativity"] = torch.rand(latents.shape[0], device=device, dtype=torch.bfloat16) * creativity_max
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
                if (
                    float(getattr(cfg, "mdm_mask_ratio", 0.0)) > 0
                    or getattr(cfg, "mdm_mask_schedule", None)
                ):
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
                    )
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
                    )
                loss = loss_dict["loss"].mean()
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
                    _gparams = list(model.parameters())
                    if rae_bridge is not None:
                        _gparams += list(rae_bridge.parameters())
                    torch.nn.utils.clip_grad_norm_(_gparams, cfg.max_grad_norm)
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
                logger.info(
                    f"step={steps:07d} epoch={epoch} loss={avg_loss:.4f} steps/s={steps_per_sec:.2f}{lr_str}"
                )
                # IMPROVEMENTS 5.1: TensorBoard / WandB
                if rank == 0:
                    if tb_writer is not None:
                        tb_writer.add_scalar("loss", avg_loss, steps)
                        tb_writer.add_scalar("lr", lr_val, steps)
                    if getattr(cfg, "wandb_project", None):
                        try:
                            import wandb
                            wandb.log({"loss": avg_loss, "lr": lr_val, "step": steps}, step=steps)
                        except Exception:
                            pass
                # IMPROVEMENTS 5.1: log sample images to WandB/TensorBoard every log_images_every steps
                log_images_every = getattr(cfg, "log_images_every", 0)
                if rank == 0 and log_images_every > 0 and steps > 0 and steps % log_images_every == 0 and (tb_writer is not None or getattr(cfg, "wandb_project", None)):
                    try:
                        _log_sample_image(ema, diffusion, tokenizer, text_encoder, vae, device, cfg, steps, tb_writer, rae_bridge=rae_bridge)
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
                    path = ckpt_dir / "best.pt"
                    torch.save(ckpt, path)
                    logger.info(f"Best checkpoint saved: {path} (train loss={best_loss:.4f})")
                running_loss = 0.0
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="", help="Image folder or manifest JSONL")
    parser.add_argument("--manifest-jsonl", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, default="DiT-XL/2-Text")
    parser.add_argument("--vae-model", type=str, default="stabilityai/sd-vae-ft-mse", help="Autoencoder model id/path (VAE=AutoencoderKL or RAE=AutoencoderRAE)")
    parser.add_argument("--autoencoder-type", type=str, default="kl", choices=["kl", "rae"], help="Autoencoder type: kl=AutoencoderKL, rae=AutoencoderRAE")
    parser.add_argument("--no-rae-latent-bridge", action="store_true", help="When using RAE with C!=4, error out instead of training RAELatentBridge")
    parser.add_argument("--rae-bridge-cycle-weight", type=float, default=0.01, help="Cycle loss weight for RAELatentBridge (0=off)")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--no-bf16", action="store_true", help="Disable bf16")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--no-grad-checkpoint", action="store_true")
    parser.add_argument("--num-ar-blocks", type=int, default=0, help="Block-wise AR (0=off, 2 or 4)")
    parser.add_argument("--no-xformers", action="store_true", help="Disable xformers attention")
    parser.add_argument("--passes", type=int, default=0, help="Train for N full passes over the dataset (recommended). Overrides epochs; use with cosine LR.")
    parser.add_argument("--max-steps", type=int, default=0, help="Cap steps when using --passes, or raw step limit when passes=0 (0=use epochs).")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Min LR for cosine schedule")
    parser.add_argument("--lr-warmup-steps", type=int, default=500)
    parser.add_argument("--refinement-prob", type=float, default=0.25, help="Prob of training on fix-imperfection (small t)")
    parser.add_argument("--refinement-max-t", type=int, default=150)
    parser.add_argument("--img2img-prob", type=float, default=0.0, help="Img2img training: prob to use init_image as x_start (0=off)")
    # MDM (Masked Diffusion Models)-style masked-patch training
    parser.add_argument("--mdm-mask-ratio", type=float, default=0.0, help="MDM training: fraction of latent patches to mask (0=off)")
    parser.add_argument(
        "--mdm-mask-schedule",
        type=str,
        default=None,
        help="MDM training: state-dependent mask ratio schedule as comma pairs: t_step,mask_ratio (e.g. 0,0.05,500,0.25,999,0.35)",
    )
    parser.add_argument("--mdm-patch-size", type=int, default=2, help="MDM training: latent patch size (typically 2, matches DiT patch embed)")
    parser.add_argument("--mdm-min-mask-patches", type=int, default=1, help="MDM training: ensure at least N patches masked per sample")
    parser.add_argument("--no-mdm-loss-only-masked", action="store_true", help="MDM training: include unmasked regions in loss (default is masked-only)")
    # MoE DiT upgrade (MLP-only MoE)
    parser.add_argument("--moe-num-experts", type=int, default=0, help="MoE training: number of FFN experts (0=off)")
    parser.add_argument("--moe-top-k", type=int, default=2, help="MoE routing: top-k experts per token")
    parser.add_argument("--moe-balance-loss-weight", type=float, default=0.0, help="MoE: auxiliary router balance loss weight (0=off)")
    parser.add_argument("--no-save-best", action="store_true", help="Disable saving best checkpoint by loss")
    parser.add_argument("--negative-prompt-weight", type=float, default=0.5, help="Weight for subtracting negative prompt")
    parser.add_argument("--style-embed-dim", type=int, default=0, help="Style conditioning (same as text_dim, e.g. 4096); 0=off")
    parser.add_argument("--style-strength", type=float, default=0.7, help="Style blend strength in training (0.6-0.8)")
    parser.add_argument("--control-cond-dim", type=int, default=0, help="1=enable ControlNet; 0=off")
    parser.add_argument("--control-scale", type=float, default=0.85, help="ControlNet strength in training (0.7-1.0)")
    parser.add_argument("--creativity-embed-dim", type=int, default=0, help="Creativity/diversity knob (0=off; e.g. 64)")
    parser.add_argument("--creativity-max", type=float, default=1.0, help="Training: sample creativity in [0, this]")
    parser.add_argument("--size-embed-dim", type=int, default=0, dest="size_embed_dim", help="PixArt-style latent (H,W) -> timestep embed dim (0=off; DiT still sees native res via pos embed)")
    parser.add_argument("--patch-se", action="store_true", dest="patch_se", help="Zero-init patch channel gate after patch embed (identity at init)")
    parser.add_argument("--patch-se-reduction", type=int, default=8, dest="patch_se_reduction", help="Bottleneck divisor for patch SE MLP")
    parser.add_argument("--curriculum-difficulty-steps", type=str, default=None, help="Comma-sep steps for difficulty curriculum (e.g. 0,5000,10000); use with JSONL 'difficulty' 0-1")
    parser.add_argument("--no-difficulty-easy-first", action="store_true", help="If set, late steps prefer easy (default: early=easy)")
    parser.add_argument("--rule-loss-weight", type=float, default=0.0, help="Constitutional/rule auxiliary loss weight (0=off)")
    # REPA (Representation Alignment)
    parser.add_argument("--repa-weight", type=float, default=0.0, help="REPA auxiliary loss weight (0=off)")
    parser.add_argument("--repa-encoder-model", type=str, default="facebook/dinov2-base", help="Frozen vision encoder: dinov2* or clip* (HF id)")
    parser.add_argument("--repa-out-dim", type=int, default=768, help="Projection output dim; must match encoder embedding dim")
    parser.add_argument("--repa-projector-hidden-dim", type=int, default=0, help="REPA projector hidden dim (0=linear head)")
    # SSM swap (hybrid SSM-like token mixer)
    parser.add_argument("--ssm-every-n", type=int, default=0, help="Replace every Nth self-attention block with SSM-like token mixer (0=off).")
    parser.add_argument("--ssm-kernel-size", type=int, default=7, help="SSM token mixer depthwise conv kernel size (odd >=3).")

    # ViT-Gen features
    parser.add_argument("--num-register-tokens", type=int, default=0, help="Append N learnable register tokens to the patch token stream.")
    parser.add_argument("--use-rope", action="store_true", help="Enable RoPE (rotary positional embeddings) in self-attention.")
    parser.add_argument("--rope-base", type=float, default=10000.0, help="RoPE base frequency (theta).")
    parser.add_argument("--kv-merge-factor", type=int, default=1, help="KV pooling factor for hierarchical patch merging in self-attention (1=off).")
    parser.add_argument("--token-routing-enabled", action="store_true", help="Enable soft per-token routing (gating) in DiT blocks.")
    parser.add_argument("--token-routing-strength", type=float, default=1.0, help="Token routing strength in [0,1] (higher = more gating).")
    parser.add_argument("--beta-schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction-type", type=str, default="epsilon", choices=["epsilon", "v"], help="v = velocity (SD2-style)")
    parser.add_argument("--noise-offset", type=float, default=0.0, help="SD-style noise offset (e.g. 0.1)")
    parser.add_argument("--min-snr-gamma", type=float, default=5.0, help="Min-SNR loss weighting (0=off)")
    parser.add_argument("--loss-weighting", type=str, default="min_snr", choices=["min_snr", "unit", "edm", "v", "eps"], help="Timestep loss weight: min_snr (default) | unit | edm | v | eps")
    parser.add_argument("--loss-weighting-sigma-data", type=float, default=0.5, help="Sigma_data for loss_weighting=edm")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--val-split", type=float, default=0.0, help="Fraction of data for validation (e.g. 0.05); 0=off. Enables best-by-val and early stopping.")
    parser.add_argument("--val-every", type=int, default=2000, help="Evaluate val loss every N steps (when val-split > 0)")
    parser.add_argument("--early-stopping-patience", type=int, default=0, help="Stop after N val checks with no improvement; 0=off")
    parser.add_argument("--val-max-batches", type=int, default=None, help="Max val batches per eval (default: full val set)")
    parser.add_argument("--deterministic", action="store_true", help="Reproducible training (worker seeds)")
    parser.add_argument("--latent-cache-dir", type=str, default=None, help="Use precomputed latents for faster training")
    parser.add_argument("--seed", type=int, default=42)
    # IMPROVEMENTS from docs
    parser.add_argument("--caption-dropout-schedule", type=str, default=None, help="Comma-sep pairs step,prob e.g. 0,0.2,10000,0.05 (decay caption dropout over training)")
    parser.add_argument("--crop-mode", type=str, default="center", choices=["center", "random", "largest_center"], help="Crop strategy for training images (1.2)")
    parser.add_argument("--save-polyak", type=int, default=0, help="Running avg of last N steps; save as polyak.pt every ckpt-every (0=off)")
    parser.add_argument("--wandb-project", type=str, default=None, help="WandB project name (enables WandB logging)")
    parser.add_argument("--tensorboard-dir", type=str, default=None, help="TensorBoard log dir (enables TensorBoard)")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 step and exit (verify setup)")
    parser.add_argument("--log-images-every", type=int, default=0, help="Log a sample image to WandB/TB every N steps (0=off)")
    parser.add_argument("--log-images-prompt", type=str, default="a photo of a cat", help="Prompt for --log-images-every sample")
    args = parser.parse_args()

    cfg = TrainConfig(
        data_path=args.data_path,
        manifest_jsonl=args.manifest_jsonl,
        results_dir=args.results_dir,
        model_name=args.model,
        image_size=args.image_size,
        vae_model=args.vae_model,
        autoencoder_type=args.autoencoder_type,
        rae_use_latent_bridge=not getattr(args, "no_rae_latent_bridge", False),
        rae_bridge_cycle_weight=float(getattr(args, "rae_bridge_cycle_weight", 0.01)),
        global_batch_size=args.global_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        use_bf16=not args.no_bf16,
        use_compile=not args.no_compile,
        grad_checkpointing=not args.no_grad_checkpoint,
        global_seed=args.seed,
        num_ar_blocks=args.num_ar_blocks,
        use_xformers=not args.no_xformers,
        passes=args.passes,
        max_steps=args.max_steps,
        min_lr=args.min_lr,
        lr_warmup_steps=args.lr_warmup_steps,
        refinement_prob=args.refinement_prob,
        refinement_max_t=args.refinement_max_t,
        img2img_prob=args.img2img_prob,
        mdm_mask_ratio=args.mdm_mask_ratio,
        mdm_mask_schedule=parse_mdm_mask_schedule(getattr(args, "mdm_mask_schedule", None)),
        mdm_patch_size=args.mdm_patch_size,
        mdm_min_mask_patches=args.mdm_min_mask_patches,
        mdm_loss_only_masked=not args.no_mdm_loss_only_masked,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_balance_loss_weight=args.moe_balance_loss_weight,
        save_best=not args.no_save_best,
        negative_prompt_weight=args.negative_prompt_weight,
        style_embed_dim=args.style_embed_dim,
        style_strength=args.style_strength,
        control_cond_dim=args.control_cond_dim,
        control_scale=args.control_scale,
        creativity_embed_dim=args.creativity_embed_dim,
        creativity_max=args.creativity_max,
        size_embed_dim=getattr(args, "size_embed_dim", 0),
        patch_se=getattr(args, "patch_se", False),
        patch_se_reduction=getattr(args, "patch_se_reduction", 8),
        curriculum_difficulty_steps=[int(x.strip()) for x in args.curriculum_difficulty_steps.split(",") if x.strip()] if (getattr(args, "curriculum_difficulty_steps", None) and str(args.curriculum_difficulty_steps).strip()) else None,
        curriculum_difficulty_easy_first=not getattr(args, "no_difficulty_easy_first", False),
        rule_loss_weight=args.rule_loss_weight,
        repa_weight=args.repa_weight,
        repa_encoder_model=args.repa_encoder_model,
        repa_out_dim=args.repa_out_dim,
        repa_projector_hidden_dim=args.repa_projector_hidden_dim,
        ssm_every_n=args.ssm_every_n,
        ssm_kernel_size=args.ssm_kernel_size,
        num_register_tokens=args.num_register_tokens,
        use_rope=args.use_rope,
        rope_base=args.rope_base,
        kv_merge_factor=args.kv_merge_factor,
        token_routing_enabled=args.token_routing_enabled,
        token_routing_strength=args.token_routing_strength,
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
        noise_offset=args.noise_offset,
        min_snr_gamma=args.min_snr_gamma,
        loss_weighting=getattr(args, "loss_weighting", "min_snr"),
        loss_weighting_sigma_data=getattr(args, "loss_weighting_sigma_data", 0.5),
        resume=args.resume,
        val_split=args.val_split,
        val_every=args.val_every,
        early_stopping_patience=args.early_stopping_patience,
        val_max_batches=args.val_max_batches,
        deterministic=args.deterministic,
        latent_cache_dir=args.latent_cache_dir,
        caption_dropout_schedule=parse_caption_dropout_schedule(getattr(args, "caption_dropout_schedule", None)),
        crop_mode=getattr(args, "crop_mode", "center"),
        save_polyak=getattr(args, "save_polyak", 0),
        wandb_project=getattr(args, "wandb_project", None),
        tensorboard_dir=getattr(args, "tensorboard_dir", None),
        dry_run=getattr(args, "dry_run", False),
        log_images_every=getattr(args, "log_images_every", 0),
        log_images_prompt=getattr(args, "log_images_prompt", "a photo of a cat"),
    )
    main(cfg)

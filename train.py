"""
Train the text-conditioned DiT (VP diffusion, rectified flow or masked MDM), on one GPU or with DDP.

``python train.py --help`` lists every option. ``--profile-out PATH`` (with ``--profile-sort`` /
``--profile-top``) writes cProfile output plus a sorted text summary, like sample.py.
"""

import logging
import math
import os
import pickle
import platform
import subprocess
import sys
import warnings
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import time

from utils.terminal import configure_stdio_for_console

configure_stdio_for_console()

# ``--help`` exits before ``import torch``: the argparse-only parser is loaded by path because
# ``training/__init__.py`` imports torch (tests/test_cli_entrypoints.py).
if __name__ == "__main__" and any(_h in sys.argv[1:] for _h in ("-h", "--help")):
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location(
        "_sdx_train_cli_parser",
        Path(__file__).resolve().parent / "training" / "train_cli_parser.py",
    )
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _mod.build_train_arg_parser().parse_args()

import torch
import torch.distributed as dist
from config.train_config import TrainConfig, get_dit_build_kwargs
from data import ResolutionBucketBatchSampler, Text2ImageDataset, collate_t2i
from diffusion import create_diffusion
from diffusion.flow_matching import flow_matching_per_sample_losses, shift_timestep_indices
from diffusion.gaussian_diffusion import offset_noise
from diffusion.schedules import sd3_resolution_shift
from diffusion.timestep_sampling import sample_training_timesteps
from models import DiT_models_text
from models.ar_masks_extended import create_block_causal_mask_2d
from models.rae_latent_bridge import RAELatentBridge
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from training.train_args import build_train_config_from_args
from training.train_cli_parser import build_train_arg_parser
from utils.modeling.model_viz import print_model_summary
from utils.modeling.multi_encoder_encode import encode_kwargs_for_captions, prepare_multi_encoder_kwargs
from utils.modeling.text_encoder_bundle import encode_text, load_text_encoder_bundle
from utils.runtime import json_dumps, to_plain_dict
from utils.safety.minor_guard import NegativeText, is_sexual, require_image_allowed
from utils.training.ar_curriculum import parse_ar_order_mix, resolve_ar_for_step
from utils.training.config_validator import estimate_memory_usage, validate_train_config
from utils.training.device_perf import configure_training_cuda_and_cpu, log_cuda_quick_stats, log_training_perf_hints
from utils.training.error_handling import get_model_info, log_gpu_memory, setup_logging
from utils.training.fast_dataloader import (
    dataloader_perf_kwargs,
    maybe_cuda_prefetch_dataloader,
    resolve_training_num_workers,
)
from utils.training.metrics import log_system_info
from utils.training.throughput import (
    create_adamw_optimizer,
    encode_neg_and_style_embeddings,
    to_train_device,
)
from utils.training.timestep_curriculum import resolve_timestep_kwargs_for_step


def _training_noise(cfg, latents: torch.Tensor) -> torch.Tensor:
    """The one noise draw of a training step: Gaussian, optionally OT-paired to the latents, plus the noise offset.

    Every loss term of the step (main loss, attention aux capture) must use this same tensor as its noise/target.
    """
    noise = torch.randn_like(latents)
    reg = float(getattr(cfg, "ot_noise_pair_reg", 0.0) or 0.0)
    if reg > 0.0 and latents.shape[0] >= 2:
        from utils.training.ot_noise_pairing import pair_noise_to_latents

        noise = pair_noise_to_latents(
            latents,
            noise,
            reg=reg,
            n_iters=int(getattr(cfg, "ot_noise_pair_iters", 40)),
            mode=str(getattr(cfg, "ot_noise_pair_mode", "soft")),
        )
    return offset_noise(noise, float(getattr(cfg, "noise_offset", 0.0) or 0.0))


def flow_shift_for_latents(cfg, latents: torch.Tensor) -> float:
    """SD3 resolution shift for this batch: ``cfg.flow_shift`` at the native latent grid, scaled by
    ``sqrt(tokens / native tokens)`` for other buckets -- the same rule ``sample_loop`` uses at inference."""
    native = int(getattr(cfg, "image_size", 256)) // 8
    return sd3_resolution_shift(
        int(latents.shape[-2]) * int(latents.shape[-1]), native * native, float(getattr(cfg, "flow_shift", 1.0) or 1.0)
    )


def flow_spectral_kwargs(cfg) -> dict | None:
    """``spectral_kwargs`` for :func:`flow_matching_per_sample_losses` when ``--spectral-sfp-loss`` is on."""
    if not bool(getattr(cfg, "spectral_sfp_loss", False)):
        return None
    return {
        "low_sigma": float(getattr(cfg, "spectral_sfp_low_sigma", 0.22)),
        "high_sigma": float(getattr(cfg, "spectral_sfp_high_sigma", 0.22)),
        "tau_power": float(getattr(cfg, "spectral_sfp_tau_power", 1.0)),
    }


def _flow_matching_loss(cfg, model, latents, noise, t, model_kwargs, sample_weights, num_timesteps) -> torch.Tensor:
    """Rectified-flow loss with the configured timestep distribution, resolution shift and optional spectral loss."""
    t_flow = shift_timestep_indices(t, num_timesteps, flow_shift_for_latents(cfg, latents))
    per = flow_matching_per_sample_losses(
        model, latents, noise, num_timesteps, model_kwargs, t_idx=t_flow, spectral_kwargs=flow_spectral_kwargs(cfg)
    )
    if sample_weights is None:
        return per.mean()
    w = sample_weights.view(-1).to(dtype=per.dtype)
    return (per * w).sum() / (w.sum() + 1e-8)


def _encode_kw_for_batch(captions: list, text_bundle, *, layout_specs=None) -> dict:
    """Multi-encoder kwargs for a batch (optional per-row layout specs from JSONL)."""
    if layout_specs is not None and any(x is not None for x in layout_specs):
        return prepare_multi_encoder_kwargs(
            captions, text_bundle, layout_specs=layout_specs, flat_full_prompts=captions
        )
    return encode_kwargs_for_captions(captions, text_bundle)


def _sample_training_timesteps(
    cfg, num_timesteps: int, batch_size: int, device: torch.device, *, global_step: int = 0
) -> torch.Tensor:
    """Integer diffusion timestep indices for a training batch (curriculum-aware sampling mode)."""
    kw = resolve_timestep_kwargs_for_step(cfg, int(global_step))
    return sample_training_timesteps(batch_size, int(num_timesteps), device=device, **kw)


def _spectral_flow_prediction_training_kwargs(cfg: TrainConfig) -> dict:
    """Spectral SFP kwargs for ``GaussianDiffusion.training_losses`` (ignored on the MDM masked path)."""
    return {
        "use_spectral_sfp_loss": bool(getattr(cfg, "spectral_sfp_loss", False)),
        "spectral_sfp_low_sigma": float(getattr(cfg, "spectral_sfp_low_sigma", 0.22)),
        "spectral_sfp_high_sigma": float(getattr(cfg, "spectral_sfp_high_sigma", 0.22)),
        "spectral_sfp_tau_power": float(getattr(cfg, "spectral_sfp_tau_power", 1.0)),
    }


def _diffusion_step_loss(
    cfg,
    diffusion,
    model,
    latents: torch.Tensor,
    t: torch.Tensor,
    model_kwargs: dict,
    sample_weights,
    noise: torch.Tensor,
    *,
    refinement_prob: float,
    refinement_max_t: int,
) -> torch.Tensor:
    """Scalar main loss of one batch (flow matching, MDM or VP), all from ``noise``."""
    weighting = _loss_weighting_kwargs(cfg)
    sk = _spectral_flow_prediction_training_kwargs(cfg)
    if getattr(cfg, "flow_matching_training", False):
        return _flow_matching_loss(cfg, model, latents, noise, t, model_kwargs, sample_weights, diffusion.num_timesteps)
    if float(getattr(cfg, "mdm_mask_ratio", 0.0)) > 0 or getattr(cfg, "mdm_mask_schedule", None):
        return compute_mdm_training_loss(
            diffusion,
            model,
            latents,
            t,
            model_kwargs,
            refinement_prob=refinement_prob,
            refinement_max_t=refinement_max_t,
            sample_weights=sample_weights,
            mdm_mask_ratio=float(getattr(cfg, "mdm_mask_ratio", 0.0)),
            mdm_mask_schedule=getattr(cfg, "mdm_mask_schedule", None),
            mdm_patch_size=int(getattr(cfg, "mdm_patch_size", 2)),
            mdm_loss_only_masked=bool(getattr(cfg, "mdm_loss_only_masked", True)),
            mdm_min_mask_patches=int(getattr(cfg, "mdm_min_mask_patches", 1)),
            noise=noise,
            spectral_kwargs=sk,
            **weighting,
        )["loss"]
    return diffusion.training_losses(
        model,
        latents,
        t,
        model_kwargs=model_kwargs,
        refinement_prob=refinement_prob,
        refinement_max_t=refinement_max_t,
        sample_weights=sample_weights,
        noise=noise,
        **weighting,
        **sk,
    )["loss"]


def _loss_weighting_kwargs(cfg) -> dict:
    return {
        "min_snr_gamma": getattr(cfg, "min_snr_gamma", 5.0),
        "loss_weighting": getattr(cfg, "loss_weighting", "min_snr"),
        "loss_weighting_sigma_data": getattr(cfg, "loss_weighting_sigma_data", 0.5),
    }


def _bridge_aux_loss(cfg, diffusion, model, latents, model_kwargs, *, global_step: int) -> torch.Tensor:
    """Weighted VP bridge auxiliary (``--bridge-aux-weight``), or 0. Runs its own forward pass, so call it after
    anything that reads per-forward model attributes (REPA features, MoE aux loss)."""
    bridge_w = float(getattr(cfg, "bridge_aux_weight", 0.0))
    if bridge_w <= 0.0:
        return latents.new_zeros(())
    from diffusion.bridge_training import bridge_aux_vp_loss

    t_br = _sample_training_timesteps(
        cfg, diffusion.num_timesteps, latents.shape[0], latents.device, global_step=global_step
    )
    return bridge_w * bridge_aux_vp_loss(
        diffusion,
        model,
        latents,
        t_br,
        model_kwargs,
        mix_lambda=float(getattr(cfg, "bridge_aux_lambda", 0.2)),
        noise_offset=float(getattr(cfg, "noise_offset", 0.0) or 0.0),
        **_loss_weighting_kwargs(cfg),
        **_spectral_flow_prediction_training_kwargs(cfg),
    )


def _apply_runtime_ar(raw_model, *, num_ar_blocks: int, ar_block_order: str) -> None:
    """Update DiT AR regime on a live model (used by curriculum/order-mix training)."""
    b = int(num_ar_blocks)
    order = str(ar_block_order or "raster").strip().lower()
    raw_model.num_ar_blocks = b
    raw_model.ar_block_order = order
    if b <= 0:
        raw_model._ar_mask = None
        return
    n_patches = int(getattr(raw_model, "num_patches", 0) or 0)
    if n_patches <= 0:
        raw_model._ar_mask = None
        return
    p = int(round(n_patches**0.5))
    mask = create_block_causal_mask_2d(p, p, b, block_order=order)
    ref = getattr(raw_model, "pos_embed", None)
    if ref is not None and hasattr(ref, "device"):
        mask = mask.to(device=ref.device)
    raw_model._ar_mask = mask


def _safe_git_info(repo_root: Path) -> dict:
    """Best-effort git metadata for reproducibility manifests."""

    def _run(args):
        try:
            return subprocess.check_output(args, cwd=str(repo_root), text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {
        "commit": commit or "unknown",
        "branch": branch or "unknown",
        "dirty": bool(status) if status is not None else None,
    }


def _write_run_manifest(exp_dir: Path, cfg: TrainConfig, logger: logging.Logger) -> None:
    """Write ``config.train.json`` and ``run_manifest.json`` (command, environment, git state) for reproducibility."""
    repo_root = Path(__file__).resolve().parent
    cfg_out = exp_dir / "config.train.json"
    manifest_out = exp_dir / "run_manifest.json"
    cfg_out.write_text(json_dumps(to_plain_dict(cfg), indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
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


def _set_tf32(enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


# TF32 on by default (also for the scripts that import this module); ``main`` applies ``--no-tf32``.
_set_tf32(True)


def _size_embed_from_latents(latents: torch.Tensor, device, dtype=torch.float32) -> torch.Tensor:
    """(B, 2) latent grid (H, W) for PixArt-style SizeEmbedder."""
    b, _, h, w = latents.shape
    return torch.tensor([float(h), float(w)], device=device, dtype=dtype).repeat(b, 1)


def _negatives_strip_emphasis(negative_captions, train_prompt_emphasis: bool):
    """Negative captions marked as screened negatives; ``( )`` / ``[ ]`` stripped when emphasis training is on."""
    if not train_prompt_emphasis:
        return [NegativeText(n or "") for n in negative_captions]
    from utils.prompt.prompt_emphasis import parse_prompt_emphasis

    return [NegativeText(parse_prompt_emphasis(n or "")[0]) for n in negative_captions]


def get_t5_and_vae(device, cfg: TrainConfig):
    """Frozen T5 ``(tokenizer, text_encoder)`` and autoencoder (KL VAE or RAE) on ``device``."""
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.text_encoder)
    # Prefer safetensors — transformers blocks torch.load on .bin when torch<2.6 (CVE-2025-32434).
    try:
        text_encoder = transformers.T5EncoderModel.from_pretrained(cfg.text_encoder, use_safetensors=True)
    except (ValueError, OSError) as exc:
        if "safetensors" in str(exc).lower() or "torch.load" in str(exc).lower():
            raise RuntimeError(
                f"T5 load failed for {cfg.text_encoder!r}. Run:\n"
                "  python setup/ensure_t5_safetensors.py\n"
                "or upgrade torch: pip install 'torch>=2.6' --index-url https://download.pytorch.org/whl/cu124"
            ) from exc
        raise
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder = text_encoder.to(device)

    from utils.modeling.autoencoder_loading import get_autoencoder_class

    ae = get_autoencoder_class(getattr(cfg, "autoencoder_type", "kl")).from_pretrained(cfg.vae_model)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False
    ae = ae.to(device)

    return tokenizer, text_encoder, ae


@torch.no_grad()
def encode_images_vae(images: torch.Tensor, ae: torch.nn.Module, scale: float = 0.18215) -> torch.Tensor:
    """Latents of ``(B, 3, H, W)`` images in ``[-1, 1]``: a scaled KL-VAE sample ``(B, 4, h, w)`` or the RAE latent
    ``(B, C, h, w)`` (the RAE normalises itself, so ``scale`` only applies to the VAE)."""
    enc = ae.encode(images)
    if hasattr(enc, "latent_dist"):
        return enc.latent_dist.sample() * scale
    if hasattr(enc, "latent"):
        return enc.latent
    raise AttributeError(
        f"Autoencoder encode() returned {type(enc).__name__!r} with neither "
        "'latent_dist' nor 'latent' attribute. Check your autoencoder type."
    )


# The loaded REPA encoder: (encoder_id, model, mean, std, input_hw); one at a time.
_repa_encoder: tuple | None = None


@torch.no_grad()
def _get_repa_encoder(device, encoder_id: str) -> tuple:
    """Frozen REPA vision encoder with its input normalisation: ``(model, mean, std, input_hw)``."""
    global _repa_encoder
    if _repa_encoder is not None and _repa_encoder[0] == encoder_id:
        return _repa_encoder[1:]
    from transformers import AutoImageProcessor, AutoModel, CLIPVisionModelWithProjection

    enc_lower = encoder_id.lower()
    if "dinov3" in enc_lower or "dinov2" in enc_lower:
        vision_model = AutoModel.from_pretrained(encoder_id)
    elif "clip" in enc_lower:
        vision_model = CLIPVisionModelWithProjection.from_pretrained(encoder_id)
    else:
        raise ValueError(f"Unsupported REPA encoder_model: {encoder_id}. Use dinov3*, dinov2*, or clip* checkpoints.")

    processor = AutoImageProcessor.from_pretrained(encoder_id)
    mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
    std = getattr(processor, "image_std", [0.229, 0.224, 0.225])
    target = getattr(processor, "crop_size", None) or getattr(processor, "size", None)
    hw = 224
    if isinstance(target, dict):
        hw = int(target.get("height") or target.get("shortest_edge") or target.get("width") or 224)
    elif isinstance(target, (int, float)):
        hw = int(target)

    vision_model = vision_model.eval().requires_grad_(False).to(device)
    _repa_encoder = (
        encoder_id,
        vision_model,
        torch.tensor(mean, device=device, dtype=torch.float32).view(1, 3, 1, 1),
        torch.tensor(std, device=device, dtype=torch.float32).view(1, 3, 1, 1),
        hw,
    )
    return _repa_encoder[1:]


@torch.no_grad()
def _repa_features(pixel_values: torch.Tensor, device, cfg) -> torch.Tensor:
    """``(B, D)`` REPA target features of ``(B, 3, H, W)`` pixels in ``[-1, 1]``: the DINO CLS token, or the CLIP
    image embedding."""
    encoder_id = getattr(cfg, "repa_encoder_model", None)
    if not encoder_id:
        raise RuntimeError("REPA requested but --repa-encoder-model is not set.")
    vision_model, mean, std, hw = _get_repa_encoder(device, str(encoder_id))
    x = (pixel_values * 0.5 + 0.5).clamp(0.0, 1.0)
    x = torch.nn.functional.interpolate(x, size=(hw, hw), mode="bilinear", align_corners=False)
    out = vision_model(pixel_values=(x.to(torch.float32) - mean) / std)
    if "dino" in str(encoder_id).lower():
        return out.last_hidden_state[:, 0, :]
    return out.image_embeds if hasattr(out, "image_embeds") else out.pooler_output


def repa_alignment_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """REPA (Yu et al. 2025) alignment term: ``1 - cos(proj(h), y)`` between the projected DiT representation and the
    frozen encoder's (scale-free, unlike MSE), averaged over images or tokens."""
    return 1.0 - torch.nn.functional.cosine_similarity(pred.float(), target.float(), dim=-1).mean()


def compute_mdm_training_loss(
    diffusion,
    model,
    x_start,
    t,
    model_kwargs,
    *,
    refinement_prob: float,
    refinement_max_t: int,
    min_snr_gamma: float,
    sample_weights,
    loss_weighting: str,
    loss_weighting_sigma_data: float,
    mdm_mask_ratio: float,
    mdm_mask_schedule,
    mdm_patch_size: int,
    mdm_loss_only_masked: bool,
    mdm_min_mask_patches: int,
    noise: torch.Tensor,
    spectral_kwargs: dict | None = None,
):
    """Masked-diffusion (MDM) loss: masked patches see ``q_sample(x_start, t, noise)``, the rest keep the clean
    ``x_start`` as context, and the loss covers the masked region only (``mdm_loss_only_masked``) or everything.

    The mask ratio is ``mdm_mask_ratio``, or interpolated over ``t`` from ``mdm_mask_schedule`` ``[(t, ratio), ...]``.
    Without masking (or when the patch size does not tile the latent) this is the plain VP loss.
    """
    device = x_start.device
    B, C, H, W = x_start.shape

    if refinement_prob > 0 and refinement_max_t > 0 and torch.rand((), device=device) < refinement_prob:
        t = torch.randint(0, min(refinement_max_t, diffusion.num_timesteps), (B,), device=device, dtype=t.dtype)

    mdm_patch_size = int(mdm_patch_size)
    no_masking = mdm_mask_ratio <= 0 and not mdm_mask_schedule
    if no_masking or mdm_patch_size <= 0 or H % mdm_patch_size != 0 or W % mdm_patch_size != 0:
        return diffusion.training_losses(
            model,
            x_start,
            t,
            model_kwargs=model_kwargs,
            noise=noise,
            refinement_prob=0.0,
            refinement_max_t=0,
            min_snr_gamma=min_snr_gamma,
            sample_weights=sample_weights,
            loss_weighting=loss_weighting,
            loss_weighting_sigma_data=loss_weighting_sigma_data,
            **(spectral_kwargs or {}),
        )
    x_t_full = diffusion.q_sample(x_start, t, noise=noise)
    ph, pw = H // mdm_patch_size, W // mdm_patch_size

    t_i = t.to(torch.int64)
    ratio = torch.full((B,), float(mdm_mask_ratio), dtype=torch.float32, device=device)
    sched = (
        sorted(((int(ts), float(r)) for ts, r in mdm_mask_schedule), key=lambda x: x[0]) if mdm_mask_schedule else []
    )
    if sched:
        ratio = torch.where(t_i <= sched[0][0], sched[0][1], ratio)
        ratio = torch.where(t_i >= sched[-1][0], sched[-1][1], ratio)
        for (t0, r0), (t1, r1) in zip(sched, sched[1:]):
            if t1 > t0:
                frac = (t_i.to(torch.float32) - t0) / (t1 - t0)
                ratio = torch.where((t_i >= t0) & (t_i < t1), r0 + frac * (r1 - r0), ratio)

    ratio_t = ratio.view(B, 1, 1, 1)
    patch_mask = (torch.rand(B, 1, ph, pw, device=device) < ratio_t).to(dtype=x_start.dtype)

    # Top up every row to at least ``mdm_min_mask_patches`` masked patches.
    if mdm_min_mask_patches > 0:
        flat = patch_mask.view(B, -1)
        counts = flat.sum(dim=1)
        deficit = (float(mdm_min_mask_patches) - counts).clamp(min=0).long()
        rows = (deficit > 0).nonzero(as_tuple=False).view(-1)
        for bi in rows.tolist():
            d = int(deficit[bi])
            zeros = (flat[bi] <= 0).nonzero(as_tuple=False).view(-1)
            if zeros.numel() == 0 or d <= 0:
                continue
            d = min(d, int(zeros.numel()))
            pick = zeros[torch.randperm(zeros.numel(), device=device)[:d]]
            flat[bi, pick] = 1.0

    mask_latent = patch_mask.repeat_interleave(mdm_patch_size, dim=2).repeat_interleave(mdm_patch_size, dim=3)
    x_t = mask_latent * x_t_full + (1 - mask_latent) * x_start

    model_out = model(x_t, t, **model_kwargs)
    if model_out.shape != x_start.shape and model_out.shape[1] > x_start.shape[1]:
        model_out = model_out[:, : x_start.shape[1]]

    target = diffusion.training_target(x_start, noise, t)
    mse = torch.nn.functional.mse_loss(model_out, target, reduction="none")
    if mdm_loss_only_masked:
        mask_broadcast = mask_latent.expand_as(mse).to(mse.dtype)
        denom = mask_broadcast.sum(dim=(1, 2, 3)) + 1e-8
        loss_per_sample = (mse * mask_broadcast).sum(dim=(1, 2, 3)) / denom
    else:
        loss_per_sample = mse.mean(dim=tuple(range(1, mse.ndim)))
    loss_per_sample = loss_per_sample * diffusion.timestep_loss_weight(
        t, loss_weighting, min_snr_gamma, loss_weighting_sigma_data
    )
    if sample_weights is not None and sample_weights.shape[0] == loss_per_sample.shape[0]:
        sample_weights = sample_weights.to(device, dtype=loss_per_sample.dtype)
        loss = (loss_per_sample * sample_weights).sum() / (sample_weights.sum() + 1e-8)
    else:
        loss = loss_per_sample.mean()

    return {"loss": loss}


def ema_decay_at(step: int, decay: float) -> float:
    """EMA warmup ``min(decay, (1 + n) / (10 + n))`` (the schedule of timm / diffusers ``EMAModel``): early on the
    average spans only the recent updates, so short runs do not sample from near-initial weights."""
    return min(float(decay), (1.0 + int(step)) / (10.0 + int(step)))


@torch.no_grad()
def _ema_update_state(avg_sd: dict, live_sd: dict, decay: float) -> None:
    """In-place ``avg = decay * avg + (1 - decay) * live`` over a state dict; non-float entries (step counters) are
    copied."""
    for key, avg_val in avg_sd.items():
        live_val = live_sd.get(key)
        if live_val is None:
            continue
        if avg_val.dtype.is_floating_point:
            avg_val.mul_(decay).add_(live_val.to(device=avg_val.device, dtype=avg_val.dtype), alpha=1.0 - decay)
        else:
            avg_val.copy_(live_val)


def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.9999) -> None:
    """EMA over the whole ``state_dict``: parameters miss registered buffers (norm statistics, fixed positional
    tables). ``decay`` comes from :func:`ema_decay_at`."""
    _ema_update_state(ema_model.state_dict(), model.state_dict(), decay)


def _conditioning_kwargs(
    cfg,
    batch: dict,
    latents: torch.Tensor,
    device,
    *,
    encoder_hidden: torch.Tensor,
    encoder_hidden_neg,
    style_embedding,
    token_weights,
    creativity_jitter_std: float = 0.0,
) -> dict:
    """DiT ``model_kwargs`` of a batch: text and negative states, emphasis token weights, style, control image,
    creativity (uniform in ``[0, creativity_max]``, optionally jittered) and size conditioning."""
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
    control_images = batch.get("control_image")
    if control_images is not None:
        model_kwargs["control_image"] = control_images.to(device, non_blocking=True).to(torch.bfloat16)
        model_kwargs["control_scale"] = getattr(cfg, "control_scale", 0.85)
        control_type_ids = batch.get("control_type_id")
        if control_type_ids is not None:
            model_kwargs["control_type"] = control_type_ids.to(device, non_blocking=True).to(torch.long)
    if getattr(cfg, "creativity_embed_dim", 0) > 0:
        creativity_max = getattr(cfg, "creativity_max", 1.0)
        b = latents.shape[0]
        cre = torch.rand(b, device=device, dtype=torch.bfloat16) * creativity_max
        if creativity_jitter_std > 0:
            jitter = torch.randn(b, device=device, dtype=torch.bfloat16) * creativity_jitter_std
            cre = (cre + jitter).clamp(0.0, creativity_max)
        model_kwargs["creativity"] = cre
    if int(getattr(cfg, "size_embed_dim", 0) or 0) > 0:
        model_kwargs["size_embed"] = _size_embed_from_latents(latents, device)
    return model_kwargs


@torch.inference_mode()
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
    global_step: int = 0,
):
    """Mean loss of the EMA model over the validation set, for early stopping and the best-by-val checkpoint.

    Creativity jitter is train-only, so the metric stays smooth.
    """
    ema_model.eval()
    if text_bundle is not None and text_bundle.fusion is not None:
        text_bundle.fusion.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in val_loader:
        if max_batches is not None and n_batches >= max_batches:
            break
        pixel_values = to_train_device(batch["pixel_values"], device)
        captions = batch["captions"]
        negative_captions = batch.get("negative_captions") or [""] * len(captions)
        styles = batch.get("styles") or [""] * len(captions)
        with torch.amp.autocast("cuda", enabled=cfg.use_bf16, dtype=torch.bfloat16):
            if "latent_values" in batch:
                latents = to_train_device(batch["latent_values"], device, dtype=torch.bfloat16)
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

            layout_batch = batch.get("prompt_layout")

            def _val_enc(caps: list, *, with_layout: bool = False) -> torch.Tensor:
                return encode_text(
                    caps,
                    tokenizer,
                    text_encoder,
                    device,
                    dtype=torch.bfloat16,
                    max_length=max_caption_len,
                    text_bundle=text_bundle,
                    train_fusion=False,
                    **_encode_kw_for_batch(
                        caps,
                        text_bundle,
                        layout_specs=layout_batch if with_layout else None,
                    ),
                )

            encoder_hidden = _val_enc(pos_caps, with_layout=True)
            neg_caps = _negatives_strip_emphasis(negative_captions, train_pe)
            encoder_hidden_neg, style_embedding = encode_neg_and_style_embeddings(
                neg_caps,
                styles,
                style_embed_dim=int(getattr(cfg, "style_embed_dim", 0) or 0),
                encode_fn=_val_enc,
                batch_text_encode=bool(getattr(cfg, "batch_text_encode", True)),
            )
            t = _sample_training_timesteps(
                cfg, diffusion.num_timesteps, latents.shape[0], device, global_step=global_step
            )
            model_kwargs = _conditioning_kwargs(
                cfg,
                batch,
                latents,
                device,
                encoder_hidden=encoder_hidden,
                encoder_hidden_neg=encoder_hidden_neg,
                style_embedding=style_embedding,
                token_weights=token_weights,
            )
            sample_weights = batch.get("sample_weights")
            if sample_weights is not None:
                sample_weights = sample_weights.to(device)
            loss = _diffusion_step_loss(
                cfg,
                diffusion,
                ema_model,
                latents,
                t,
                model_kwargs,
                sample_weights,
                _training_noise(cfg, latents),
                refinement_prob=refinement_prob,
                refinement_max_t=refinement_max_t,
            )
            loss = loss + _bridge_aux_loss(cfg, diffusion, ema_model, latents, model_kwargs, global_step=global_step)
            total_loss += loss.item()
        n_batches += 1
    ema_model.train()
    if text_bundle is not None and text_bundle.fusion is not None:
        text_bundle.fusion.train()
    return total_loss / max(1, n_batches)


def _repeat_epochs(loader, sampler=None):
    """Endless batches for step-based training: a fresh pass (new shuffle, crops, DDP epoch) each time.

    ``itertools.cycle`` would replay -- and keep in memory -- the first epoch's batches forever.
    """
    epoch = 0
    while True:
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def get_lr_cosine(step: int, max_steps: int, warmup_steps: int, base_lr: float, min_lr: float) -> float:
    """Linear warmup, then cosine decay from ``base_lr`` to ``min_lr`` at ``max_steps``."""
    if step < warmup_steps:
        return base_lr * (step / max(1, warmup_steps))
    progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def get_curriculum_max_length(step: int, steps_list, lengths_list) -> int:
    """Max caption length at ``step``: the length of the last curriculum step reached (300 when unconfigured)."""
    if not steps_list or not lengths_list or len(steps_list) != len(lengths_list):
        return 300
    out = lengths_list[0]
    for s, L in zip(steps_list, lengths_list):
        if step >= s:
            out = L
    return out


def get_scheduled_caption_dropout(step: int, schedule) -> float:
    """Caption-dropout probability from ``[(step, prob), ...]``, linearly interpolated and held at the ends (0 if unset)."""
    if not schedule:
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
    wandb_run=None,
):
    """Generate one sample with the EMA model and log it to TensorBoard / the WandB run."""
    was_training = ema_model.training
    ema_model.eval()
    try:
        demo_prompt = getattr(cfg, "log_images_prompt", "a photo of a cat")
        with torch.no_grad():

            def _enc(prompt: str) -> torch.Tensor:
                return encode_text(
                    [prompt],
                    tokenizer,
                    text_encoder,
                    device,
                    max_length=300,
                    dtype=torch.float32,  # sample_loop runs the model in fp32 without autocast
                    text_bundle=text_bundle,
                    train_fusion=False,
                    **_encode_kw_for_batch([prompt], text_bundle),
                )

            enc = _enc(demo_prompt)
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
            # Same uncond as sample.py (the empty prompt the model saw under caption dropout); "euler" index spacing
            # starts at t = T-1, which zero-terminal-SNR schedules require.
            x0 = diffusion.sample_loop(
                ema_model,
                shape,
                model_kwargs_cond=mk_cond,
                model_kwargs_uncond={**mk_cond, "encoder_hidden_states": _enc("")},
                cfg_scale=7.5,
                num_inference_steps=sample_steps,
                eta=0.0,
                timestep_schedule="euler",
                device=device,
                dtype=torch.float32,
                flow_matching_sample=bool(getattr(cfg, "flow_matching_training", False)),
                flow_shift=float(getattr(cfg, "flow_shift", 1.0) or 1.0),
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
    require_image_allowed(img, sexual=is_sexual(demo_prompt))
    if tb_writer is not None:
        tb_writer.add_image("sample", img, steps, dataformats="HWC")
    if wandb_run is not None:
        import wandb

        wandb_run.log({"sample": wandb.Image(img)}, step=steps)


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
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Training requires a CUDA-capable GPU. Run `nvidia-smi` to check driver status."
        )

    logger = setup_logging()

    if bool(getattr(cfg, "strict_warnings", False)):
        warnings.filterwarnings(
            "error", category=UserWarning, module=r"^(train|config|data|diffusion|models|training|utils)(\.|$)"
        )
        warnings.filterwarnings(
            "error", category=FutureWarning, module=r"^(train|config|data|diffusion|models|training|utils)(\.|$)"
        )
        logger.info("Strict warnings enabled: project UserWarning/FutureWarning are treated as errors.")

    logger.info("Validating configuration...")
    for issue in validate_train_config(cfg):
        if issue.startswith("ERROR"):
            logger.error(issue)
            raise ValueError(f"Configuration error: {issue}")
        logger.warning(issue)

    logger.info(f"System info: {log_system_info()}")
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
    if getattr(cfg, "deterministic", False):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)

    _set_tf32(bool(getattr(cfg, "enable_tf32", True)))
    configure_training_cuda_and_cpu(
        deterministic=bool(getattr(cfg, "deterministic", False)),
        cudnn_benchmark=bool(getattr(cfg, "cudnn_benchmark", True)),
        enable_tf32=bool(getattr(cfg, "enable_tf32", True)),
        cpu_num_threads=int(getattr(cfg, "torch_cpu_num_threads", 0) or 0),
        cpu_num_interop_threads=int(getattr(cfg, "torch_cpu_num_interop_threads", 0) or 0),
        logger=logger,
    )
    if rank == 0:
        log_cuda_quick_stats(logger)
        log_training_perf_hints(logger, int(cfg.num_workers), int(cfg.world_size))

    if rank == 0:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = Path(cfg.results_dir) / f"{stamp}-{cfg.model_name.replace('/', '-')}"
        ckpt_dir = exp_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        logger = create_logger(str(exp_dir))
    else:
        exp_dir = Path(cfg.results_dir) / "run"
        logger = create_logger(None)

    if rank == 0 and bool(getattr(cfg, "save_run_manifest", True)):
        try:
            _write_run_manifest(exp_dir, cfg, logger)
        except Exception as e:
            logger.warning(f"Failed to write run manifest: {e}")

    log_gpu_memory(logger, "Before model loading: ")
    logger.info("Loading T5 and autoencoder...")
    tokenizer, text_encoder, vae = get_t5_and_vae(device, cfg)
    text_bundle = load_text_encoder_bundle(cfg, device)
    if text_bundle is not None:
        if text_bundle.mode == "penta":
            logger.info(
                "Penta text encoder enabled: T5 + CLIP-L + CLIP-bigG + CLIP-H + LongCLIP-L "
                "(fusion layers trained with DiT)."
            )
        else:
            logger.info("Triple text encoder enabled: T5 + CLIP-L + CLIP-bigG (fusion layers trained with DiT).")

    ae_type = getattr(cfg, "autoencoder_type", "kl")
    effective_latent_scale = getattr(cfg, "latent_scale", 0.18215) if ae_type == "kl" else 1.0

    # The DiT expects SD-style latents (4 channels, image_size // 8). RAEs usually have other channel counts
    # (e.g. 768), bridged by RAELatentBridge; a spatial mismatch fails fast instead of deep in the diffusion loop.
    rae_bridge = None
    if ae_type == "rae":
        latent_hw_expected = int(cfg.image_size) // 8
        latent_channels_expected = 4
        ae_cfg = getattr(vae, "config", None)
        latent_channels_rae = getattr(ae_cfg, "encoder_hidden_size", None)
        encoder_input_size = getattr(ae_cfg, "encoder_input_size", None)
        encoder_patch_size = getattr(ae_cfg, "encoder_patch_size", None)
        latent_hw_rae = None
        if encoder_input_size is not None and encoder_patch_size is not None:
            try:
                latent_hw_rae = int(encoder_input_size) // int(encoder_patch_size)
            except (TypeError, ValueError):
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

    model_fn = DiT_models_text.get(cfg.model_name)
    if model_fn is None:
        raise ValueError(f"Unknown model {cfg.model_name}. Choices: {list(DiT_models_text.keys())}")
    # Caption dropout happens in the loop (captions -> "" before encoding), never inside the model.
    model = model_fn(**get_dit_build_kwargs(cfg, class_dropout_prob=0.0)).to(device)

    if rank == 0:
        logger.info(f"Model info: {get_model_info(model)}")
        print_model_summary(model)

    if cfg.grad_checkpointing:
        model.enable_gradient_checkpointing()

    # LoRA/DoRA adapter training: wrap target Linears with trainable adapters and
    # freeze the base DiT. Only the adapters end up in the optimizer (below).
    lora_training = bool(getattr(cfg, "lora_train", False))
    if lora_training:
        from models.lora_train import inject_trainable_lora

        _targets = [t.strip() for t in str(getattr(cfg, "lora_target", "") or "").split(",") if t.strip()]
        _trainable, _wrapped = inject_trainable_lora(
            model,
            rank=int(getattr(cfg, "lora_rank", 16)),
            alpha=float(getattr(cfg, "lora_alpha", 16.0)),
            use_dora=bool(getattr(cfg, "lora_dora", False)),
            targets=_targets or None,
        )
        if rank == 0:
            n_train = sum(p.numel() for p in _trainable)
            kind = "DoRA" if getattr(cfg, "lora_dora", False) else "LoRA"
            logger.info(
                f"{kind} training: wrapped {_wrapped} Linear layers, "
                f"rank={getattr(cfg, 'lora_rank', 16)} alpha={getattr(cfg, 'lora_alpha', 16.0)}, "
                f"{n_train:,} trainable params (base frozen)."
            )
        if _wrapped == 0:
            raise ValueError(
                "--lora-train matched no Linear layers. Adjust --lora-target substrings "
                "(default targets: attention qkv/proj + MLP fc)."
            )
        if int(getattr(cfg, "control_cond_dim", 0) or 0) > 0:
            from models.lora_train import unfreeze_control_encoder

            _ctrl_n = unfreeze_control_encoder(model)
            if rank == 0 and _ctrl_n:
                logger.info(
                    f"LoRA + control: unfroze control_encoder ({_ctrl_n:,} params trainable alongside adapters)."
                )

    def _maybe_save_lora_adapter(raw_model, ckpt_path) -> None:
        """Write an adapter-only file next to a full checkpoint (LoRA training)."""
        if not lora_training:
            return
        from models.lora_train import lora_state_dict

        adapter_path = Path(ckpt_path).with_name(Path(ckpt_path).stem + "_lora.pt")
        torch.save(lora_state_dict(raw_model), adapter_path)
        logger.info(f"LoRA adapter saved: {adapter_path}")

    raw_model = model
    ema = deepcopy(model).requires_grad_(False)
    # The fusion layers and the RAE bridge are trained with the DiT and run next to its EMA weights at inference,
    # so they are averaged with the same decay.
    ema_bridge = deepcopy(rae_bridge).requires_grad_(False) if rae_bridge is not None else None
    fusion = text_bundle.fusion if text_bundle is not None else None
    ema_fusion = deepcopy(fusion).requires_grad_(False) if fusion is not None else None
    ema_pairs = [(m, e) for m, e in ((raw_model, ema), (rae_bridge, ema_bridge), (fusion, ema_fusion)) if m is not None]

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

    # Only optimize params that need grad (LoRA training freezes the base model;
    # full training leaves everything trainable, so this is a no-op there).
    opt_params = [p for p in model.parameters() if p.requires_grad]
    if rae_bridge is not None:
        opt_params += list(rae_bridge.parameters())
    if text_bundle is not None and text_bundle.fusion is not None:
        opt_params += list(text_bundle.fusion.parameters())
    opt = create_adamw_optimizer(
        opt_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        fused=bool(getattr(cfg, "fused_adamw", True)),
    )
    # Loss scaling only matters for FP16 underflow; BF16 has FP32's exponent range.
    scaler = torch.amp.GradScaler("cuda", enabled=getattr(cfg, "use_fp16", False))

    start_step = 0
    best_loss = float("inf")

    def _load_ckpt_file(path: Path):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError:  # checkpoints pickle the TrainConfig
            return torch.load(path, map_location="cpu", weights_only=False)

    def _apply_train_checkpoint(ckpt: dict, *, load_optimizer: bool) -> None:
        raw_model.load_state_dict(ckpt.get("model", ckpt.get("ema")), strict=True)
        ema.load_state_dict(ckpt.get("ema") or raw_model.state_dict(), strict=True)
        if load_optimizer and ckpt.get("opt"):
            try:
                opt.load_state_dict(ckpt["opt"])
            except (ValueError, KeyError) as e:
                logger.warning(f"Optimizer state not restored (fresh optimizer): {e}")
        for key, live, shadow in (
            ("rae_latent_bridge", rae_bridge, ema_bridge),
            ("text_encoder_fusion", fusion, ema_fusion),
        ):
            if live is None or not ckpt.get(key):
                continue
            try:
                live.load_state_dict(ckpt[key], strict=True)
                shadow.load_state_dict(ckpt.get(f"ema_{key}") or ckpt[key], strict=True)
                logger.info(f"Loaded {key} from checkpoint.")
            except Exception as e:
                logger.warning(f"Could not load {key}: {e}")

    def _checkpoint_dict(step: int, **extra) -> dict:
        ckpt = {
            "model": raw_model.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "steps": step,
            "config": cfg,
            **extra,
        }
        if rae_bridge is not None:
            ckpt["rae_latent_bridge"] = rae_bridge.state_dict()
            ckpt["ema_rae_latent_bridge"] = ema_bridge.state_dict()
        if fusion is not None:
            ckpt["text_encoder_fusion"] = fusion.state_dict()
            ckpt["ema_text_encoder_fusion"] = ema_fusion.state_dict()
        return ckpt

    if getattr(cfg, "resume", None):
        resume_path = Path(cfg.resume)
        if resume_path.exists():
            ckpt = _load_ckpt_file(resume_path)
            _apply_train_checkpoint(ckpt, load_optimizer=True)
            start_step = ckpt.get("steps", 0)
            best_loss = ckpt.get("best_loss", float("inf"))
            logger.info(f"Resumed from {resume_path} at step {start_step}")
    elif getattr(cfg, "init_from", None):
        init_path = Path(cfg.init_from)
        if not init_path.is_file():
            raise FileNotFoundError(f"--init-from is not a file: {init_path}")
        ckpt = _load_ckpt_file(init_path)
        _apply_train_checkpoint(ckpt, load_optimizer=False)
        logger.info(f"Initialized weights from {init_path} (step=0, fresh optimizer)")

    # Manifest JSONL rows resolve relative image paths against --data-path.
    manifest_path = cfg.manifest_jsonl
    data_root = cfg.data_path
    dataset_path = manifest_path or data_root
    if not dataset_path:
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
            raise ValueError("--resolution-buckets requires --val-split 0 (no train/val split) in this version.")
        stride = 8 * int(getattr(raw_model, "patch_size", 2) or 2)
        for h, w in res_buckets:
            if int(h) % stride != 0 or int(w) % stride != 0:
                raise ValueError(
                    f"Resolution bucket ({h},{w}) must be divisible by {stride} (VAE stride x patch size)."
                )
        if latent_dir:
            logger.warning("Disabling latent cache (--latent-cache-dir) when using resolution buckets.")
            latent_dir = None

    dataset = Text2ImageDataset(
        dataset_path,
        data_root=data_root if manifest_path else None,
        image_size=cfg.image_size,
        latent_cache_dir=latent_dir,
        crop_mode=getattr(cfg, "crop_mode", "center"),
        region_caption_mode=getattr(cfg, "region_caption_mode", "append"),
        region_layout_tag=getattr(cfg, "region_layout_tag", "[layout]"),
        use_adherence_boost=bool(getattr(cfg, "boost_adherence_caption", False)),
        # With prompt-emphasis training, train.py turns ( ) / [ ] into token weights; the dataset must keep them.
        use_tag_emphasis=not bool(getattr(cfg, "train_prompt_emphasis", False)),
        train_shortcomings_mitigation=str(getattr(cfg, "train_shortcomings_mitigation", "none") or "none"),
        train_shortcomings_2d=bool(getattr(cfg, "train_shortcomings_2d", False)),
        train_art_guidance_mode=str(getattr(cfg, "train_art_guidance_mode", "none") or "none"),
        train_art_guidance_photography=bool(getattr(cfg, "train_art_guidance_photography", True)),
        train_anatomy_guidance=str(getattr(cfg, "train_anatomy_guidance", "none") or "none"),
        train_style_guidance_mode=str(getattr(cfg, "train_style_guidance_mode", "none") or "none"),
        train_style_guidance_artists=bool(getattr(cfg, "train_style_guidance_artists", True)),
        caption_unicode_normalize=bool(getattr(cfg, "caption_unicode_normalize", False)),
        resolution_buckets=res_buckets if res_buckets else None,
        use_hierarchical_captions=bool(getattr(cfg, "use_hierarchical_captions", False)),
        hierarchical_caption_separator=str(getattr(cfg, "hierarchical_caption_separator", " | ")),
        hierarchical_drop_global_p=float(getattr(cfg, "hierarchical_caption_drop_global_p", 0.0)),
        hierarchical_drop_local_p=float(getattr(cfg, "hierarchical_caption_drop_local_p", 0.0)),
        foveated_train_prob=float(getattr(cfg, "foveated_train_prob", 0.0)),
        foveated_crop_frac=float(getattr(cfg, "foveated_crop_frac", 0.55)),
        grounding_mask_soft=bool(getattr(cfg, "grounding_mask_soft", False)),
    )

    if len(dataset) == 0:
        raise ValueError(
            f"No training samples found. manifest={manifest_path!r} data_root={data_root!r}. "
            "Expected a folder of images with matching .txt/.caption sidecars "
            "(flat or one level of subfolders), or a manifest .jsonl with "
            "{'image_path', 'caption'} rows resolved against --data-path."
        )

    def _worker_init(worker_id):
        import numpy as np

        np.random.seed(cfg.global_seed + rank * 1000 + worker_id)
        torch.manual_seed(cfg.global_seed + rank * 1000 + worker_id)

    val_loader = None
    train_dataset = dataset
    train_num_workers = resolve_training_num_workers(
        int(cfg.num_workers),
        len(dataset),
        int(cfg.per_device_batch_size),
        auto=bool(getattr(cfg, "auto_num_workers", True)),
    )
    dl_common = dataloader_perf_kwargs(
        num_workers=train_num_workers,
        prefetch_factor=int(getattr(cfg, "prefetch_factor", 4)),
        worker_init_fn=_worker_init if getattr(cfg, "deterministic", False) else None,
    )
    cuda_prefetch = bool(getattr(cfg, "cuda_stream_prefetch", True))
    if rank == 0 and train_num_workers != int(cfg.num_workers):
        logger.info("DataLoader num_workers=%d (requested %d)", train_num_workers, int(cfg.num_workers))
    if rank == 0 and cuda_prefetch and device.type == "cuda":
        logger.info("CUDA stream prefetch enabled (overlap H2D with training step)")
    if rank == 0 and not getattr(cfg, "latent_cache_dir", None):
        logger.info(
            "Tip: precompute VAE latents (python -m scripts.training.precompute_latents) "
            "and pass --latent-cache-dir for 3-10x faster epochs when IO-bound."
        )
    if val_split > 0 and val_split < 1:
        n_total = len(dataset)
        n_val = max(1, int(n_total * val_split))
        n_train = n_total - n_val
        train_dataset, val_dataset = random_split(
            dataset, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.global_seed)
        )
        if rank == 0:
            val_loader = maybe_cuda_prefetch_dataloader(
                DataLoader(
                    val_dataset,
                    batch_size=cfg.per_device_batch_size,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=collate_t2i,
                    **dl_common,
                ),
                device,
                enabled=cuda_prefetch,
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
            collate_fn=collate_t2i,
            **dl_common,
        )
    else:
        loader = DataLoader(
            train_dataset,
            batch_size=cfg.per_device_batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            drop_last=True,
            collate_fn=collate_t2i,
            generator=torch.Generator().manual_seed(cfg.global_seed + rank)
            if getattr(cfg, "deterministic", False)
            else None,
            **dl_common,
        )
    loader = maybe_cuda_prefetch_dataloader(loader, device, enabled=cuda_prefetch)
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
        loader = _repeat_epochs(loader, sampler)
        logger.info(f"Step-based training: max_steps={max_steps}, cosine LR to min_lr={getattr(cfg, 'min_lr', 1e-6)}")
    else:
        logger.info(f"Epoch-based training: epochs={cfg.epochs}")

    dashboard = None
    if rank == 0 and bool(getattr(cfg, "live_dashboard", False)):
        from utils.training.live_dashboard import LiveDashboard

        _dash_total = (
            max_steps
            if use_step_based
            else int(cfg.epochs) * max(1, len(train_dataset) // max(1, cfg.global_batch_size))
        )
        dashboard = LiveDashboard(_dash_total, model_name=cfg.model_name)

    # AR curriculum / order mix change the block-causal mask during training.
    ar_curriculum_mode = str(getattr(cfg, "ar_curriculum_mode", "none") or "none").strip().lower()
    ar_order_mix_spec = str(getattr(cfg, "ar_order_mix", "") or "")
    ar_order_mix_list = parse_ar_order_mix(ar_order_mix_spec)
    dynamic_ar_runtime = (ar_curriculum_mode != "none") or bool(ar_order_mix_list)

    def _ar_at(step: int) -> tuple[int, str]:
        blocks, order = resolve_ar_for_step(
            step,
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
        return int(blocks), str(order)

    last_runtime_ar: tuple[int, str] | None = None
    if dynamic_ar_runtime:
        last_runtime_ar = _ar_at(start_step)
        b0, o0 = last_runtime_ar
        _apply_runtime_ar(raw_model, num_ar_blocks=b0, ar_block_order=o0)
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

    train_model = model
    if dynamic_ar_runtime and cfg.use_compile:
        logger.warning("Skipping torch.compile because dynamic AR runtime is enabled (curriculum/order-mix).")
    elif cfg.use_compile:
        compile_mode = str(getattr(cfg, "compile_mode", "reduce-overhead") or "reduce-overhead")
        try:
            train_model = torch.compile(model, mode=compile_mode)
            logger.info("Model compiled with torch.compile(mode=%r)", compile_mode)
        except Exception as e:
            logger.warning(f"compile failed: {e}, continuing without compile")

    # Attention auxiliaries capture block-0 cross-attention of the VP x_t; not defined for flow/MDM inputs.
    attn_aux_supported = (
        not bool(getattr(cfg, "flow_matching_training", False))
        and not (float(getattr(cfg, "mdm_mask_ratio", 0.0)) > 0 or getattr(cfg, "mdm_mask_schedule", None))
        and hasattr(raw_model, "num_patches")
    )
    attn_aux_on = attn_aux_supported and (
        float(getattr(cfg, "attn_grounding_loss_weight", 0.0)) > 0
        or float(getattr(cfg, "attn_token_coverage_loss_weight", 0.0)) > 0
    )
    if use_ddp and (attn_aux_on or float(getattr(cfg, "bridge_aux_weight", 0.0)) > 0):
        raise ValueError(
            "--bridge-aux-weight and the attention auxiliary losses run a second forward pass per step, "
            "which DDP (one forward per backward) cannot reduce; train them on a single GPU."
        )

    steps = start_step
    log_steps = 0
    running_loss_t: torch.Tensor | None = None
    running_ag_sum_t: torch.Tensor | None = None
    running_ag_count = 0
    running_cov_sum_t: torch.Tensor | None = None
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
    save_polyak = int(getattr(cfg, "save_polyak", 0) or 0)
    polyak_buf = None  # running average of the last ~save_polyak weight snapshots (rank 0)
    tb_writer = None
    if rank == 0 and getattr(cfg, "tensorboard_dir", None):
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter(log_dir=str(Path(cfg.tensorboard_dir) / Path(cfg.results_dir).name))
            logger.info(f"TensorBoard logging to {cfg.tensorboard_dir}")
        except (ImportError, OSError) as e:
            logger.warning(f"TensorBoard not available: {e}")
    wandb_run = None
    if rank == 0 and getattr(cfg, "wandb_project", None):
        try:
            import wandb

            wandb_run = wandb.init(project=cfg.wandb_project, config=to_plain_dict(cfg))
            logger.info(f"WandB project: {cfg.wandb_project}")
        except Exception as e:  # missing package, login or network: train without WandB
            logger.warning(f"WandB not available: {e}")

    while True:
        if not use_step_based:
            epoch += 1
            if epoch >= cfg.epochs:
                break
            if use_ddp:
                sampler.set_epoch(epoch)

        for batch in loader:
            last_attn_grounding_t: torch.Tensor | None = None
            last_attn_cov_t: torch.Tensor | None = None
            if dynamic_ar_runtime:
                curr = _ar_at(steps)
                if curr != last_runtime_ar:
                    _apply_runtime_ar(raw_model, num_ar_blocks=curr[0], ar_block_order=curr[1])
                    last_runtime_ar = curr
                    if rank == 0:
                        logger.info(
                            "AR runtime update @step=%d -> num_ar_blocks=%d ar_block_order=%s", steps, curr[0], curr[1]
                        )
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

            _need_pixels = float(getattr(cfg, "repa_weight", 0.0)) > 0 or "latent_values" not in batch
            pixel_values = to_train_device(batch["pixel_values"], device) if _need_pixels else None
            captions = batch["captions"]
            if float(getattr(cfg, "train_originality_augment_prob", 0.0)) > 0:
                import numpy as np
                from utils.prompt.originality_augment import inject_originality_tokens

                prob = float(cfg.train_originality_augment_prob)
                ost = max(0.0, min(1.0, float(getattr(cfg, "train_originality_strength", 0.5))))
                rng_np = np.random.default_rng((int(cfg.global_seed) + int(steps) * 1_000_003 + int(rank)) % (2**32))
                captions = [
                    inject_originality_tokens(cap or "", ost, rng_np) if rng_np.random() < prob else cap
                    for cap in captions
                ]
            negative_captions = batch.get("negative_captions") or [""] * len(captions)
            styles = batch.get("styles") or [""] * len(captions)

            max_caption_len = (
                get_curriculum_max_length(steps, curriculum_steps, curriculum_lengths) if curriculum_steps else 300
            )
            layout_batch = batch.get("prompt_layout")
            # Per-sample caption dropout to the empty prompt: exactly the uncond sample.py encodes for CFG,
            # through the same tokenizer/encoders/fusion (a zero vector would leave that uncond untrained).
            cap_drop_schedule = getattr(cfg, "caption_dropout_schedule", None)
            p_drop = (
                get_scheduled_caption_dropout(steps, cap_drop_schedule)
                if cap_drop_schedule
                else float(getattr(cfg, "caption_dropout_prob", 0.0) or 0.0)
            )
            if p_drop > 0:
                drop = (torch.rand(len(captions)) < p_drop).tolist()
                captions = ["" if d else c for c, d in zip(captions, drop)]
                if layout_batch is not None:
                    layout_batch = [None if d else spec for spec, d in zip(layout_batch, drop)]

            with torch.amp.autocast("cuda", enabled=cfg.use_bf16, dtype=torch.bfloat16):
                with torch.no_grad():
                    if "latent_values" in batch:
                        latents = to_train_device(batch["latent_values"], device, dtype=torch.bfloat16)
                    else:
                        latents = encode_images_vae(pixel_values, vae, effective_latent_scale)
                    # Img2img training: sometimes start from the row's init image so the model learns to edit it.
                    init_pixel_values = batch.get("init_pixel_values")
                    if init_pixel_values is not None and getattr(cfg, "img2img_prob", 0) > 0:
                        use_init = torch.rand(latents.shape[0], device=device, dtype=torch.bfloat16) < cfg.img2img_prob
                        if use_init.any():
                            nbc = device.type == "cuda"
                            idx = use_init.nonzero(as_tuple=False).view(-1)
                            init_lat_subset = encode_images_vae(
                                init_pixel_values.to(device, non_blocking=nbc)[idx],
                                vae,
                                effective_latent_scale,
                            )
                            latents = latents.clone()
                            latents[idx] = init_lat_subset
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
                    # Frozen encoders run without grad inside the bundle; the fusion layers need it to train.
                    with torch.set_grad_enabled(fusion is not None):
                        encoder_hidden = encode_text(
                            pos_caps,
                            tokenizer,
                            text_encoder,
                            device,
                            dtype=torch.bfloat16,
                            max_length=max_caption_len,
                            text_bundle=text_bundle,
                            train_fusion=True,
                            **_encode_kw_for_batch(pos_caps, text_bundle, layout_specs=layout_batch),
                        )
                    neg_caps = _negatives_strip_emphasis(negative_captions, train_pe)

                    def _enc_batch(caps: list) -> torch.Tensor:
                        return encode_text(
                            caps,
                            tokenizer,
                            text_encoder,
                            device,
                            dtype=torch.bfloat16,
                            max_length=max_caption_len,
                            text_bundle=text_bundle,
                            train_fusion=True,
                            **encode_kwargs_for_captions(caps, text_bundle),
                        )

                    encoder_hidden_neg, style_embedding = encode_neg_and_style_embeddings(
                        neg_caps,
                        styles,
                        style_embed_dim=int(getattr(cfg, "style_embed_dim", 0) or 0),
                        encode_fn=_enc_batch,
                        batch_text_encode=bool(getattr(cfg, "batch_text_encode", True)),
                    )
                latents_raw = latents
                if rae_bridge is not None and latents_raw.shape[1] != 4:
                    latents = rae_bridge.rae_to_dit(latents_raw)
                    z_cycle = latents_raw.detach()
                else:
                    z_cycle = None
                t = _sample_training_timesteps(
                    cfg, diffusion.num_timesteps, latents.shape[0], device, global_step=steps
                )
                model_kwargs = _conditioning_kwargs(
                    cfg,
                    batch,
                    latents,
                    device,
                    encoder_hidden=encoder_hidden,
                    encoder_hidden_neg=encoder_hidden_neg,
                    style_embedding=style_embedding,
                    token_weights=token_weights,
                    creativity_jitter_std=float(getattr(cfg, "creativity_jitter_std", 0.0)),
                )
                sample_weights = batch.get("sample_weights")
                if sample_weights is not None:
                    sample_weights = sample_weights.to(device, non_blocking=device.type == "cuda")
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
                noise = _training_noise(cfg, latents)
                attn_gw = float(getattr(cfg, "attn_grounding_loss_weight", 0.0))
                gv = batch.get("grounding_mask_valid")
                has_mask_supervision = "grounding_mask" in batch and (gv is None or bool(gv.any()))
                use_attn_grounding = attn_gw > 0.0 and has_mask_supervision and attn_aux_supported
                cov_w = float(getattr(cfg, "attn_token_coverage_loss_weight", 0.0))
                use_attn_coverage = cov_w > 0.0 and attn_aux_supported
                use_attn_aux = use_attn_grounding or use_attn_coverage

                loss = _diffusion_step_loss(
                    cfg,
                    diffusion,
                    train_model,
                    latents,
                    t,
                    model_kwargs,
                    sample_weights,
                    noise,
                    refinement_prob=0.0 if use_attn_aux else refinement_prob,
                    refinement_max_t=refinement_max_t,
                )
                # Per-forward attributes of the main pass; read before any other forward overwrites them.
                moe_aux_loss = getattr(raw_model, "moe_aux_loss", None)
                repa_pred = getattr(raw_model, "_repa_projected", None)
                if use_attn_aux:
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
                        training_noise=noise,
                    )
                    if use_attn_grounding:
                        nbc = device.type == "cuda"
                        gm = batch["grounding_mask"].to(device=device, dtype=torch.float32, non_blocking=nbc)
                        gv_t = batch.get("grounding_mask_valid")
                        ag = grounding_loss_from_attn(
                            attn_w=attn_w,
                            train_model=raw_model,
                            grounding_mask_b1hw=gm,
                            token_start=int(getattr(cfg, "attn_grounding_token_start", 0)),
                            token_end=int(getattr(cfg, "attn_grounding_token_end", 0)),
                            sample_valid=gv_t,
                            min_fg_patch_mass=float(getattr(cfg, "attn_grounding_min_fg_patch_mass", 0.0)),
                        )
                        loss = loss + attn_gw * ag
                        last_attn_grounding_t = ag.detach()
                    if use_attn_coverage:
                        cov = token_coverage_loss_from_attn(
                            attn_w=attn_w,
                            token_start=int(getattr(cfg, "attn_grounding_token_start", 0)),
                            token_end=int(getattr(cfg, "attn_grounding_token_end", 0)),
                            target_coverage=float(getattr(cfg, "attn_token_coverage_target", 0.025)),
                            sample_valid=None,
                            token_weights=token_weights,
                        )
                        loss = loss + cov_w * cov
                        last_attn_cov_t = cov.detach()
                moe_w = float(getattr(cfg, "moe_balance_loss_weight", 0.0))
                if moe_w > 0 and moe_aux_loss is not None:
                    loss = loss + moe_w * moe_aux_loss.to(loss.dtype)
                repa_w = float(getattr(cfg, "repa_weight", 0.0))
                if repa_w > 0 and pixel_values is not None and repa_pred is not None:
                    loss = loss + repa_w * repa_alignment_loss(repa_pred, _repa_features(pixel_values, device, cfg))
                loss = loss + _bridge_aux_loss(cfg, diffusion, train_model, latents, model_kwargs, global_step=steps)
                if rae_bridge is not None and z_cycle is not None:
                    cw = float(getattr(cfg, "rae_bridge_cycle_weight", 0.0))
                    if cw > 0:
                        loss = loss + cw * rae_bridge.cycle_loss(z_cycle)
                loss = loss / cfg.grad_accum_steps

            scaler.scale(loss).backward()

            if (steps + 1) % cfg.grad_accum_steps == 0:
                if cfg.max_grad_norm > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(opt_params, cfg.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                decay = ema_decay_at((steps + 1) // cfg.grad_accum_steps, cfg.ema_decay)
                for live, shadow in ema_pairs:
                    update_ema(shadow, live, decay)
                if save_polyak > 0 and rank == 0 and steps % max(1, int(cfg.log_every)) == 0:
                    sd = raw_model.state_dict()
                    if polyak_buf is None:
                        polyak_buf = {k: v.detach().cpu().clone() for k, v in sd.items()}
                    else:
                        _ema_update_state(polyak_buf, sd, 1.0 - 1.0 / save_polyak)

            if running_loss_t is None:
                running_loss_t = torch.zeros((), device=device)
            running_loss_t = running_loss_t + loss.detach() * cfg.grad_accum_steps
            if last_attn_grounding_t is not None:
                if running_ag_sum_t is None:
                    running_ag_sum_t = torch.zeros((), device=device)
                running_ag_sum_t = running_ag_sum_t + last_attn_grounding_t.float()
                running_ag_count += 1
            if last_attn_cov_t is not None:
                if running_cov_sum_t is None:
                    running_cov_sum_t = torch.zeros((), device=device)
                running_cov_sum_t = running_cov_sum_t + last_attn_cov_t.float()
                running_cov_count += 1
            log_steps += 1
            steps += 1

            if steps % cfg.log_every == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time() - start_time
                steps_per_sec = log_steps / elapsed if elapsed > 0 else 0
                avg_loss = (running_loss_t / log_steps).item() if running_loss_t is not None else 0.0
                if use_ddp:
                    loss_sum = torch.tensor([avg_loss], device=device)
                    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
                    avg_loss = loss_sum.item() / world_size
                lr_val = opt.param_groups[0]["lr"] if opt.param_groups else 0.0
                lr_str = f" lr={lr_val:.2e}" if use_step_based else ""
                scalars = {"loss": avg_loss, "lr": lr_val}
                aux_bits = []
                if running_ag_count > 0 and running_ag_sum_t is not None:
                    scalars["attn_grounding_aux"] = (running_ag_sum_t / running_ag_count).item()
                    aux_bits.append(f"attn_grnd={scalars['attn_grounding_aux']:.4f}")
                if running_cov_count > 0 and running_cov_sum_t is not None:
                    scalars["attn_token_coverage_aux"] = (running_cov_sum_t / running_cov_count).item()
                    aux_bits.append(f"attn_cov={scalars['attn_token_coverage_aux']:.4f}")
                aux_str = "".join(f" {b}" for b in aux_bits)
                logger.info(
                    f"step={steps:07d} epoch={epoch} loss={avg_loss:.4f} steps/s={steps_per_sec:.2f}{lr_str}{aux_str}"
                )
                if dashboard is not None:
                    dashboard.update(
                        steps,
                        avg_loss,
                        steps_per_sec,
                        lr_val,
                        img_per_sec=steps_per_sec * cfg.global_batch_size,
                        aux="  ".join(aux_bits),
                    )
                if tb_writer is not None:
                    for name, value in scalars.items():
                        tb_writer.add_scalar(name, value, steps)
                if wandb_run is not None:
                    wandb_run.log({**scalars, "step": steps}, step=steps)
                log_images_every = getattr(cfg, "log_images_every", 0)
                if (
                    rank == 0
                    and log_images_every > 0
                    and steps > 0
                    and steps % log_images_every == 0
                    and (tb_writer is not None or wandb_run is not None)
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
                            wandb_run=wandb_run,
                        )
                    except Exception as e:  # a failed preview must not stop training
                        logger.warning(f"Log sample image failed: {e}")
                # Save best by train loss only when not using validation (when using val, best is by val loss)
                if save_best and rank == 0 and not use_val_for_best and avg_loss < best_loss:
                    best_loss = avg_loss
                    path = ckpt_dir / "best.pt"
                    torch.save(_checkpoint_dict(steps, best_loss=best_loss), path)
                    _maybe_save_lora_adapter(raw_model, path)
                    logger.info(f"Best checkpoint saved: {path} (train loss={best_loss:.4f})")
                if running_loss_t is not None:
                    running_loss_t.zero_()
                running_ag_sum_t = None
                running_ag_count = 0
                running_cov_sum_t = None
                running_cov_count = 0
                log_steps = 0
                start_time = time()

            if steps > 0 and steps % cfg.ckpt_every == 0 and rank == 0:
                path = ckpt_dir / f"{steps:07d}.pt"
                torch.save(_checkpoint_dict(steps), path)
                _maybe_save_lora_adapter(raw_model, path)
                logger.info(f"Checkpoint saved: {path}")
                if polyak_buf is not None:
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
                    global_step=steps,
                )
                logger.info(f"Val loss @ step {steps}: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improvement_count = 0
                    path = ckpt_dir / "best.pt"
                    torch.save(_checkpoint_dict(steps, best_loss=best_val_loss), path)
                    _maybe_save_lora_adapter(raw_model, path)
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

    if dashboard is not None:
        dashboard.close()
    if use_ddp:
        dist.destroy_process_group()
    logger.info("Training done.")


if __name__ == "__main__":
    from utils.runtime import consume_profile_args, run_with_cprofile

    _argv, _pcfg = consume_profile_args(sys.argv)
    sys.argv = _argv

    def _train_entry() -> None:
        parser = build_train_arg_parser()
        args = parser.parse_args()
        from training.book_train_preset import apply_book_train_preset_to_args

        apply_book_train_preset_to_args(args)
        cfg = build_train_config_from_args(args)
        main(cfg)

    if _pcfg is not None:
        run_with_cprofile(_train_entry, _pcfg)
    else:
        _train_entry()

"""Helpers of the ``train.py`` loop: losses (VP / flow / MDM / REPA / bridge), conditioning kwargs, EMA, schedules,
validation, sample logging and run bookkeeping. Distillation / DPO / GRPO scripts reuse them.
"""

import itertools
import json
import logging
import math
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
from config.train_config import TrainConfig
from data.t2i_dataset import Text2ImageDataset
from diffusion.flow_matching import flow_matching_per_sample_losses, shift_timestep_indices
from diffusion.gaussian_diffusion import offset_noise
from diffusion.schedules import sample_training_timesteps, sd3_resolution_shift
from models.attention import create_block_causal_mask_2d
from models.conditioning import RAELatentBridge
from utils.modeling.encoders import encode_kwargs_for_captions, encode_text, prepare_multi_encoder_kwargs
from utils.runtime import json_dumps, to_plain_dict
from utils.safety.minor_guard import NegativeText, is_sexual, require_image_allowed
from utils.training.objectives import resolve_timestep_kwargs_for_step
from utils.training.perf import (
    encode_neg_and_style_embeddings,
    to_train_device,
)


def training_noise(cfg, latents: torch.Tensor) -> torch.Tensor:
    """The one noise draw of a training step: Gaussian, optionally OT-paired to the latents, plus the noise offset.

    Every loss term of the step (main loss, attention aux capture) must use this same tensor as its noise/target.
    """
    noise = torch.randn_like(latents)
    reg = float(getattr(cfg, "ot_noise_pair_reg", 0.0) or 0.0)
    if reg > 0.0 and latents.shape[0] >= 2:
        from utils.training.objectives import pair_noise_to_latents

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


def encode_kw_for_batch(captions: list, text_bundle, *, layout_specs=None) -> dict:
    """Multi-encoder kwargs for a batch (optional per-row layout specs from JSONL)."""
    if layout_specs is not None and any(x is not None for x in layout_specs):
        return prepare_multi_encoder_kwargs(
            captions, text_bundle, layout_specs=layout_specs, flat_full_prompts=captions
        )
    return encode_kwargs_for_captions(captions, text_bundle)


def step_timesteps(
    cfg, num_timesteps: int, batch_size: int, device: torch.device, *, global_step: int = 0
) -> torch.Tensor:
    """Integer diffusion timestep indices for a training batch (curriculum-aware sampling mode)."""
    kw = resolve_timestep_kwargs_for_step(cfg, int(global_step))
    return sample_training_timesteps(batch_size, int(num_timesteps), device=device, **kw)


def _spectral_sfp_kwargs(cfg: TrainConfig) -> dict:
    """Spectral SFP kwargs for ``GaussianDiffusion.training_losses`` (ignored on the MDM masked path)."""
    return {
        "use_spectral_sfp_loss": bool(getattr(cfg, "spectral_sfp_loss", False)),
        "spectral_sfp_low_sigma": float(getattr(cfg, "spectral_sfp_low_sigma", 0.22)),
        "spectral_sfp_high_sigma": float(getattr(cfg, "spectral_sfp_high_sigma", 0.22)),
        "spectral_sfp_tau_power": float(getattr(cfg, "spectral_sfp_tau_power", 1.0)),
    }


def diffusion_step_loss(
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
    sk = _spectral_sfp_kwargs(cfg)
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


def bridge_aux_loss(cfg, diffusion, model, latents, model_kwargs, *, global_step: int) -> torch.Tensor:
    """Weighted VP bridge auxiliary (``--bridge-aux-weight``), or 0. Runs its own forward pass, so call it after
    anything that reads per-forward model attributes (REPA features, MoE aux loss)."""
    bridge_w = float(getattr(cfg, "bridge_aux_weight", 0.0))
    if bridge_w <= 0.0:
        return latents.new_zeros(())
    from diffusion.flow_matching import bridge_aux_vp_loss

    t_br = step_timesteps(cfg, diffusion.num_timesteps, latents.shape[0], latents.device, global_step=global_step)
    return bridge_w * bridge_aux_vp_loss(
        diffusion,
        model,
        latents,
        t_br,
        model_kwargs,
        mix_lambda=float(getattr(cfg, "bridge_aux_lambda", 0.2)),
        noise_offset=float(getattr(cfg, "noise_offset", 0.0) or 0.0),
        **_loss_weighting_kwargs(cfg),
        **_spectral_sfp_kwargs(cfg),
    )


def apply_runtime_ar(raw_model, *, num_ar_blocks: int, ar_block_order: str) -> None:
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


def write_run_manifest(exp_dir: Path, cfg: TrainConfig, logger: logging.Logger) -> None:
    """Write ``config.train.json`` and ``run_manifest.json`` (command, environment, git state) for reproducibility."""
    repo_root = Path(__file__).resolve().parents[1]
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


def set_tf32(enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


# TF32 on by default (also for the scripts that import this module); ``main`` applies ``--no-tf32``.
set_tf32(True)


def _size_embed_from_latents(latents: torch.Tensor, device, dtype=torch.float32) -> torch.Tensor:
    """(B, 2) latent grid (H, W) for PixArt-style SizeEmbedder."""
    b, _, h, w = latents.shape
    return torch.tensor([float(h), float(w)], device=device, dtype=dtype).repeat(b, 1)


def negatives_strip_emphasis(negative_captions, train_prompt_emphasis: bool):
    """Negative captions marked as screened negatives; ``( )`` / ``[ ]`` stripped when emphasis training is on."""
    if not train_prompt_emphasis:
        return [NegativeText(n or "") for n in negative_captions]
    from utils.prompt.text import parse_prompt_emphasis

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

    from utils.modeling.encoders import load_autoencoder

    ae = load_autoencoder(
        getattr(cfg, "autoencoder_type", "kl"),
        cfg.vae_model,
        subfolder=str(getattr(cfg, "vae_subfolder", "") or ""),
        device=device,
    )
    return tokenizer, text_encoder, ae


def check_latent_cache_spec(cache_dir: str | None, spec) -> None:
    """Refuse a ``--latent-cache-dir`` written by ``precompute_latents`` for a different latent space."""
    if not cache_dir:
        return
    path = Path(cache_dir) / "latent_spec.json"
    if not path.is_file():
        return  # caches from before precompute_latents recorded its spec
    cached = json.loads(path.read_text(encoding="utf-8"))
    mine = {"channels": spec.channels, "scale": spec.scale, "shift": spec.shift}
    if any(k in cached and abs(float(cached[k]) - float(v)) > 1e-6 for k, v in mine.items()):
        raise ValueError(
            f"Latent cache {cache_dir} was encoded as {cached} but this run's VAE gives {mine}. "
            "Re-run python -m scripts.tools precompute_latents with the same --vae as --vae-model."
        )


@torch.no_grad()
def encode_images_vae(images: torch.Tensor, ae: torch.nn.Module, scale: float = 0.18215) -> torch.Tensor:
    """Latents of ``(B, 3, H, W)`` images in ``[-1, 1]``: a scaled KL-VAE sample ``(B, C, h, w)`` or the RAE latent
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
def repa_features(pixel_values: torch.Tensor, device, cfg) -> torch.Tensor:
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
        for (t0, r0), (t1, r1) in itertools.pairwise(sched):
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
def ema_update_state(avg_sd: dict, live_sd: dict, decay: float) -> None:
    """In-place ``avg = decay * avg + (1 - decay) * live`` over a state dict; non-float entries (step counters) are
    copied."""
    avgs, lives = [], []
    for key, avg_val in avg_sd.items():
        live_val = live_sd.get(key)
        if live_val is None:
            continue
        if avg_val.dtype.is_floating_point:
            avgs.append(avg_val)
            lives.append(live_val.to(device=avg_val.device, dtype=avg_val.dtype))
        else:
            avg_val.copy_(live_val)
    if avgs:  # one fused multi-tensor kernel per op instead of two launches per tensor
        torch._foreach_mul_(avgs, decay)
        torch._foreach_add_(avgs, lives, alpha=1.0 - decay)


def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.9999) -> None:
    """EMA over the whole ``state_dict``: parameters miss registered buffers (norm statistics, fixed positional
    tables). ``decay`` comes from :func:`ema_decay_at`."""
    ema_update_state(ema_model.state_dict(), model.state_dict(), decay)


def load_dit_weights(module: torch.nn.Module, state: dict, *, allow_new_edit_embed: bool) -> bool:
    """Strict load, except a text-to-image checkpoint may seed an edit model; True when its source tag is fresh."""
    if not allow_new_edit_embed or getattr(module, "edit_type_embed", None) is None or "edit_type_embed" in state:
        module.load_state_dict(state, strict=True)
        return False
    missing, unexpected = module.load_state_dict(state, strict=False)
    if unexpected or set(missing) != {"edit_type_embed"}:
        raise RuntimeError(f"Checkpoint does not fit this model: missing={missing}, unexpected={unexpected}")
    return True


def edit_source_latents(cfg, batch: dict, vae, latent_scale: float, device, *, dropout: float) -> torch.Tensor | None:
    """Source-image latents for an instruction-edit step, or ``None`` for a plain text-to-image step.

    Needs ``edit_conditioning`` and a source image on every row of the batch (``collate_t2i`` only stacks
    ``init_pixel_values`` when all rows have one). The whole step drops the source with prob ``dropout``,
    so "no source" looks the same in training as at inference: the tokens are simply absent.
    """
    src = batch.get("init_pixel_values")
    if not getattr(cfg, "edit_conditioning", False) or src is None or random.random() < dropout:
        return None
    return encode_images_vae(src.to(device, non_blocking=device.type == "cuda"), vae, latent_scale)


def conditioning_kwargs(
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
    edit_latents: torch.Tensor | None = None,
) -> dict:
    """DiT ``model_kwargs`` of a batch: text and negative states, emphasis token weights, style, control image,
    creativity (uniform in ``[0, creativity_max]``, optionally jittered), size and edit-source conditioning."""
    model_kwargs = {
        "encoder_hidden_states": encoder_hidden,
        "encoder_hidden_states_negative": encoder_hidden_neg,
        "negative_prompt_weight": getattr(cfg, "negative_prompt_weight", 0.5),
    }
    if edit_latents is not None:
        model_kwargs["edit_latents"] = edit_latents
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
            edit_latents = edit_source_latents(cfg, batch, vae, latent_scale, device, dropout=0.0)
            train_pe = bool(getattr(cfg, "train_prompt_emphasis", False))
            pos_caps = captions
            token_weights = None
            if train_pe:
                from utils.prompt.text import batch_encoder_token_weights

                pos_caps, token_weights = batch_encoder_token_weights(
                    captions,
                    tokenizer,
                    max_caption_len,
                    device=device,
                    dtype=torch.bfloat16,
                    text_bundle=text_bundle,
                )

            layout_batch = batch.get("prompt_layout")

            def _val_enc(caps: list, *, with_layout: bool = False, layout_batch=layout_batch) -> torch.Tensor:
                return encode_text(
                    caps,
                    tokenizer,
                    text_encoder,
                    device,
                    dtype=torch.bfloat16,
                    max_length=max_caption_len,
                    text_bundle=text_bundle,
                    train_fusion=False,
                    **encode_kw_for_batch(
                        caps,
                        text_bundle,
                        layout_specs=layout_batch if with_layout else None,
                    ),
                )

            encoder_hidden = _val_enc(pos_caps, with_layout=True)
            neg_caps = negatives_strip_emphasis(negative_captions, train_pe)
            encoder_hidden_neg, style_embedding = encode_neg_and_style_embeddings(
                neg_caps,
                styles,
                style_embed_dim=int(getattr(cfg, "style_embed_dim", 0) or 0),
                encode_fn=_val_enc,
                batch_text_encode=bool(getattr(cfg, "batch_text_encode", True)),
            )
            t = step_timesteps(cfg, diffusion.num_timesteps, latents.shape[0], device, global_step=global_step)
            model_kwargs = conditioning_kwargs(
                cfg,
                batch,
                latents,
                device,
                encoder_hidden=encoder_hidden,
                encoder_hidden_neg=encoder_hidden_neg,
                style_embedding=style_embedding,
                token_weights=token_weights,
                edit_latents=edit_latents,
            )
            sample_weights = batch.get("sample_weights")
            if sample_weights is not None:
                sample_weights = sample_weights.to(device)
            loss = diffusion_step_loss(
                cfg,
                diffusion,
                ema_model,
                latents,
                t,
                model_kwargs,
                sample_weights,
                training_noise(cfg, latents),
                refinement_prob=refinement_prob,
                refinement_max_t=refinement_max_t,
            )
            loss = loss + bridge_aux_loss(cfg, diffusion, ema_model, latents, model_kwargs, global_step=global_step)
            total_loss += loss.item()
        n_batches += 1
    ema_model.train()
    if text_bundle is not None and text_bundle.fusion is not None:
        text_bundle.fusion.train()
    return total_loss / max(1, n_batches)


def seed_dataloader_worker(worker_id: int, base_seed: int) -> None:
    """``worker_init_fn`` for ``--deterministic`` (module level, so spawn-based workers can pickle it)."""
    import numpy as np

    np.random.seed(base_seed + worker_id)
    torch.manual_seed(base_seed + worker_id)


def repeat_epochs(loader, sampler=None):
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
    for s, L in zip(steps_list, lengths_list, strict=True):
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


def log_sample_image(
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
                    **encode_kw_for_batch([prompt], text_bundle),
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


def build_rae_bridge(cfg, vae, device, logger):
    """``RAELatentBridge`` from the RAE's channel count to the DiT's ``cfg.latent_channels`` (``None`` when they
    match). The DiT samples latents at ``image_size // 8``; a spatial mismatch fails fast instead of deep in the loop.
    """
    rae_bridge = None
    latent_hw_expected = int(cfg.image_size) // 8
    latent_channels_expected = int(getattr(cfg, "latent_channels", 4) or 4)
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
                f"RAE encoder_hidden_size={latent_channels_rae} -> DiT {latent_channels_expected}ch via RAELatentBridge "
                f"(cycle loss weight={getattr(cfg, 'rae_bridge_cycle_weight', 0.0)})."
            )
        else:
            raise ValueError(
                f"AutoencoderRAE selected, but the DiT expects {latent_channels_expected}-channel latents. "
                f"RAE encoder_hidden_size={latent_channels_rae}, expected={latent_channels_expected}. "
                "Use default --rae-use-latent-bridge (omit --no-rae-latent-bridge) or change autoencoder."
            )
    if latent_hw_rae is not None and int(latent_hw_rae) != latent_hw_expected:
        raise ValueError(
            "AutoencoderRAE selected, but the RAE latent spatial size doesn't match this repo's assumptions. "
            f"RAE latent_hw={latent_hw_rae}, expected={latent_hw_expected} (image_size//8). "
            "Resize training --image-size or use an RAE matching latent_hw = image_size//8."
        )
    return rae_bridge


def setup_lora_training(model, cfg, rank: int, logger) -> None:
    """LoRA/DoRA adapter training: wrap the target Linears with trainable adapters and freeze the base DiT (only the
    adapters, plus the control encoder when present, reach the optimizer)."""
    from models.lora import inject_trainable_lora

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
        from models.lora import unfreeze_control_encoder

        _ctrl_n = unfreeze_control_encoder(model)
        if rank == 0 and _ctrl_n:
            logger.info(f"LoRA + control: unfroze control_encoder ({_ctrl_n:,} params trainable alongside adapters).")


def build_train_dataset(cfg, dataset_path, *, data_root, latent_dir, res_buckets) -> Text2ImageDataset:
    """The training ``Text2ImageDataset`` with every caption / crop / guidance option taken from ``cfg``."""
    return Text2ImageDataset(
        dataset_path,
        data_root=data_root,
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

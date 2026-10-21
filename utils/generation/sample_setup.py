"""Model phase of ``sample.py``: checkpoint, adapters, text encoders, autoencoder, sampling canvas, diffusion."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

import torch
from diffusion import create_diffusion
from diffusion.pixel_perfect import dit_rgb_stride_px, sampling_canvas_hw
from PIL import Image

from utils.generation.sample_helpers import (
    TEXT_MAX_TOKENS,
    _parse_lora_role_budgets,
    _parse_lora_role_stage_weights,
    _parse_lora_spec,
    load_model_from_ckpt,
)


@dataclass
class Models:
    """Everything loaded for one run, plus the latent grid it samples on."""

    device: torch.device
    model: Any
    cfg: Any
    rae_bridge: Any
    tokenizer: Any
    text_encoder: Any
    text_bundle: Any
    vae: Any
    ae_type: str
    latent_scale: float
    image_size: int
    req_w: int
    req_h: int
    canvas_h: int
    canvas_w: int
    diffusion: Any

    @property
    def latent_size(self) -> int:
        return self.image_size // 8

    @property
    def lat_h(self) -> int:
        return self.canvas_h // 8

    @property
    def lat_w(self) -> int:
        return self.canvas_w // 8

    def to_vae_space(self, x0: torch.Tensor) -> torch.Tensor:
        """DiT latent -> autoencoder latent (KL scale, RAE bridge)."""
        if self.ae_type == "kl":
            return x0 / self.latent_scale
        if self.ae_type == "rae" and self.rae_bridge is not None:
            return self.rae_bridge.dit_to_rae(x0)
        return x0

    def decode_unit(self, x0: torch.Tensor) -> torch.Tensor:
        """Decode DiT latents to RGB in ``[0, 1]``."""
        return (self.vae.decode(self.to_vae_space(x0)).sample * 0.5 + 0.5).clamp(0, 1)


def lcm_should_enable_flow_sample(dit_cfg, lcm_cfg) -> bool:
    """An LCM overlay needs the flow sampler only if the base DiT or the overlay was flow-trained."""
    return any(bool(getattr(c, "flow_matching_training", False)) for c in (dit_cfg, lcm_cfg))


def _seed_and_backends(args: Any) -> torch.device:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if getattr(args, "deterministic", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if device.type == "cuda":
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    elif device.type == "cuda":
        from utils.training.device_perf import configure_inference_cuda

        configure_inference_cuda(cudnn_benchmark=True, enable_tf32=True)
    return device


def _apply_lcm(args: Any, model: Any, cfg: Any, device: torch.device) -> None:
    """``--lcm-ckpt``: overlay few-step weights and drop to ``--lcm-steps``."""
    lcm_ckpt = str(getattr(args, "lcm_ckpt", "") or "").strip()
    if not lcm_ckpt:
        return
    from utils.checkpoint.checkpoint_loading import load_dit_text_checkpoint

    lcm_model, lcm_cfg, _, _, _ = load_dit_text_checkpoint(lcm_ckpt, device=str(device), reject_enhanced=True)
    model.load_state_dict(lcm_model.state_dict(), strict=False)
    if int(args.steps) >= 16:
        args.steps = max(2, int(getattr(args, "lcm_steps", 4) or 4))
    if lcm_should_enable_flow_sample(cfg, lcm_cfg) and not getattr(args, "force_vp_sample", False):
        args.flow_matching_sample = True
    sampler = "flow" if getattr(args, "flow_matching_sample", False) else "VP"
    print(f"LCM few-step mode: loaded {lcm_ckpt}, steps={args.steps}, {sampler} sampler.", file=sys.stderr)


def _apply_loras(args: Any, model: Any) -> None:
    """LoRA bank auto-adapters, then every ``--lora path[:scale[:role]]`` with role budgets."""
    if not getattr(args, "no_lora_bank", False):
        try:
            from utils.lora_bank import augment_sample_lora_args, default_bank_index_path

            if getattr(args, "lora_bank", False) or default_bank_index_path().is_file():
                resolved = augment_sample_lora_args(args)
                if resolved:
                    print(f"LoRA bank: auto adapters for {', '.join(resolved)}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: LoRA bank resolve failed: {e}", file=sys.stderr)
    if not args.lora:
        return
    from models.lora import apply_loras

    default_role = str(getattr(args, "lora_default_role", "style") or "style").strip().lower()
    specs, role_counts = [], {}
    for spec in args.lora:
        path, scale, role = _parse_lora_spec(spec, default_role=default_role)
        role = (role or default_role).strip().lower()
        specs.append((path.strip(), float(scale), role))
        role_counts[role] = role_counts.get(role, 0) + 1
    stage_policy = str(getattr(args, "lora_stage_policy", "auto") or "auto")
    normalize = not bool(getattr(args, "no_lora_normalize_scales", False))
    max_total = float(getattr(args, "lora_max_total_scale", 1.5))
    _, num_keys = apply_loras(
        model,
        specs,
        normalize_scales=normalize,
        max_total_scale=max_total,
        role_budgets=_parse_lora_role_budgets(getattr(args, "lora_role_budgets", "")),
        stage_policy=stage_policy,
        layer_group=str(getattr(args, "lora_layers", "all") or "all"),
        role_stage_weights=_parse_lora_role_stage_weights(getattr(args, "lora_role_stage_weights", "")),
    )
    role_mix = ", ".join(f"{k}:{v}" for k, v in sorted(role_counts.items()))
    print(
        f"Applied {len(specs)} adapter(s) (LoRA/DoRA/LyCORIS), {num_keys} layer(s), normalize={normalize}, "
        f"max_total={max_total:.2f}, stage_policy={stage_policy}, roles=[{role_mix}]"
    )


def _load_text_encoders(cfg: Any, fusion_sd: Any, device: torch.device) -> tuple[Any, Any, Any]:
    """(tokenizer, text_encoder, text_bundle); the bundle is set for triple / penta encoder checkpoints."""
    mode = str(getattr(cfg, "text_encoder_mode", "t5") or "t5").lower()
    if mode in ("triple", "penta"):
        from utils.modeling.text_encoder_bundle import attach_fusion_weights, load_text_encoder_bundle

        bundle = load_text_encoder_bundle(cfg, device)
        if bundle is None:
            raise RuntimeError(f"Checkpoint config requests {mode} text encoders but bundle failed to load.")
        if fusion_sd is not None:
            attach_fusion_weights(bundle, fusion_sd)
        names = "T5 + CLIP-L + CLIP-bigG + CLIP-H + LongCLIP-L" if mode == "penta" else "T5 + CLIP-L + CLIP-bigG"
        print(f"{mode.capitalize()} text encoder ({names}) loaded.")
        return bundle.tokenizer, bundle.text_encoder, bundle
    from transformers import AutoTokenizer, T5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_encoder)
    return tokenizer, T5EncoderModel.from_pretrained(cfg.text_encoder).to(device).eval(), None


def _check_rae(cfg: Any, vae: Any, rae_bridge: Any, image_size: int) -> None:
    """This repo's DiT samples SD-style 4-channel latents at ``image_size // 8``; an RAE must map onto that."""
    ae_cfg = getattr(vae, "config", None)
    channels = getattr(ae_cfg, "encoder_hidden_size", None)
    in_size = getattr(ae_cfg, "encoder_input_size", None)
    patch = getattr(ae_cfg, "encoder_patch_size", None)
    if channels is not None and int(channels) != 4 and rae_bridge is None:
        raise ValueError(
            "AutoencoderRAE latent channels != 4 but checkpoint has no rae_latent_bridge. "
            f"encoder_hidden_size={channels}. Train with "
            "`train.py --autoencoder-type rae` (bridge is created automatically) and sample this checkpoint."
        )
    if in_size is not None and patch is not None and int(in_size) // int(patch) != image_size // 8:
        raise ValueError(
            "AutoencoderRAE selected, but the RAE latent spatial size doesn't match this repo's assumptions. "
            f"RAE latent_hw={int(in_size) // int(patch)}, expected={image_size // 8} (image_size//8). "
            "To use RAE properly, update the DiT/diffusion latent spatial assumptions."
        )


def load_models(args: Any) -> Models:
    """Seed, load checkpoint + adapters + encoders + autoencoder, and pick the sampling canvas."""
    device = _seed_and_backends(args)
    print("Loading checkpoint and encoders...")
    model, cfg, rae_bridge, fusion_sd = load_model_from_ckpt(args.ckpt, device)
    _apply_lcm(args, model, cfg, device)
    if getattr(args, "compile_inference", False) and device.type == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("Inference: torch.compile enabled (mode=reduce-overhead).")
        except Exception as e:
            print(f"Inference: torch.compile skipped ({e}).", file=sys.stderr)
    _apply_loras(args, model)

    tokenizer, text_encoder, text_bundle = _load_text_encoders(cfg, fusion_sd, device)
    try:
        n_tok = tokenizer(args.prompt, return_tensors="pt", truncation=False).input_ids.shape[1]
        if n_tok > TEXT_MAX_TOKENS - 50:
            print(
                f"Note: prompt has {n_tok} tokens; T5 truncates at {TEXT_MAX_TOKENS}. "
                "Put key elements first for best adherence.",
                file=sys.stderr,
            )
    except Exception:
        pass

    from utils.modeling.autoencoder_loading import get_autoencoder_class

    ae_type = getattr(cfg, "autoencoder_type", "kl")
    vae = get_autoencoder_class(ae_type).from_pretrained(cfg.vae_model).to(device).eval()
    latent_scale = 1.0 if ae_type == "rae" else getattr(cfg, "latent_scale", 0.18215)
    image_size = getattr(cfg, "image_size", 256)
    if ae_type == "rae":
        _check_rae(cfg, vae, rae_bridge, int(image_size))

    req_w, req_h = int(getattr(args, "width", 0) or 0), int(getattr(args, "height", 0) or 0)
    if not (req_w or req_h) and str(getattr(args, "init_image", "") or "").strip():
        try:
            req_w, req_h = Image.open(args.init_image).size  # img2img keeps the init image's aspect ratio
        except OSError:
            pass
    req_w, req_h = req_w or image_size, req_h or image_size
    if (getattr(args, "vae_tiling", False) or req_w * req_h > 512 * 512) and hasattr(vae, "enable_tiling"):
        try:
            vae.enable_tiling()
            print("Autoencoder tiling enabled (decode in tiles for lower VRAM).")
        except Exception:
            pass
    if ae_type == "rae":
        canvas_h = canvas_w = image_size  # the RAE latent grid is fixed
    else:
        # Hires-fix composes at the native pixel area and reaches the requested size in its own pass.
        canvas_h, canvas_w = sampling_canvas_hw(
            req_h,
            req_w,
            native_px=image_size,
            stride_px=dit_rgb_stride_px(model),
            max_area_ratio=1.0 if getattr(args, "hires_fix", False) else 2.0,
        )
    if (canvas_w, canvas_h) != (image_size, image_size):
        print(f"Sampling canvas {canvas_w}x{canvas_h} (latent {canvas_h // 8}x{canvas_w // 8}).", file=sys.stderr)
    diffusion = create_diffusion(
        timestep_respacing=getattr(cfg, "timestep_respacing", ""),
        num_timesteps=getattr(cfg, "num_timesteps", 1000),
        beta_schedule=getattr(cfg, "beta_schedule", "linear"),
        prediction_type=getattr(cfg, "prediction_type", "epsilon"),
    )
    return Models(
        device=device,
        model=model,
        cfg=cfg,
        rae_bridge=rae_bridge,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        text_bundle=text_bundle,
        vae=vae,
        ae_type=ae_type,
        latent_scale=latent_scale,
        image_size=int(image_size),
        req_w=int(req_w),
        req_h=int(req_h),
        canvas_h=int(canvas_h),
        canvas_w=int(canvas_w),
        diffusion=diffusion,
    )


__all__ = ["Models", "lcm_should_enable_flow_sample", "load_models"]

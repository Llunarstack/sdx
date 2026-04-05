"""
CLIP image–text similarity for optional **inference-time** guard / refine (not training).

Uses ``transformers`` + ``PIL``. If imports fail, helpers return conservative defaults (no refine).
"""

from __future__ import annotations

from typing import Callable

import torch


def tensor_chw01_to_pil(t: torch.Tensor):
    """t: (3, H, W) float [0, 1]."""
    from PIL import Image

    a = (t.detach().float().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype("uint8")
    return Image.fromarray(a)


def clip_image_text_cosine(
    image_chw_01: torch.Tensor,
    prompt: str,
    model_id: str,
    device: torch.device,
) -> float:
    """
    Normalized CLIP cosine similarity in roughly [-1, 1] (typical positives ~0.2–0.35 for arbitrary prompts).
    """
    from transformers import CLIPModel, CLIPProcessor

    pil = tensor_chw01_to_pil(image_chw_01)
    proc = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    inputs = proc(text=[prompt], images=pil, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        img_f = model.get_image_features(pixel_values=inputs["pixel_values"])
        txt_f = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        img_f = img_f / (img_f.norm(dim=-1, keepdim=True) + 1e-8)
        txt_f = txt_f / (txt_f.norm(dim=-1, keepdim=True) + 1e-8)
        return float((img_f * txt_f).sum(dim=-1).squeeze().cpu().item())


def decode_latent_preview_rgb(
    x0: torch.Tensor,
    *,
    vae,
    latent_scale: float,
    ae_type: str,
    rae_bridge,
    batch_index: int = 0,
) -> torch.Tensor:
    """Single image (3, H, W) float 0..1 from first batch element."""
    z = x0[batch_index : batch_index + 1].clone()
    if ae_type == "kl":
        z = z / float(latent_scale)
    elif ae_type == "rae" and rae_bridge is not None:
        z = rae_bridge.dit_to_rae(z)
    with torch.no_grad():
        im = vae.decode(z).sample
    im = (im * 0.5 + 0.5).clamp(0, 1)
    return im[0]


def latent_x0_clip_cosine(
    x0_latent: torch.Tensor,
    *,
    prompt: str,
    model_id: str,
    device: torch.device,
    vae,
    latent_scale: float,
    ae_type: str,
    rae_bridge,
    batch_index: int = 0,
) -> float:
    """CLIP image–text cosine on a **decoded** preview of ``x0_latent`` (first batch element)."""
    rgb = decode_latent_preview_rgb(
        x0_latent,
        vae=vae,
        latent_scale=latent_scale,
        ae_type=ae_type,
        rae_bridge=rae_bridge,
        batch_index=batch_index,
    )
    return clip_image_text_cosine(rgb, prompt, model_id, device)


def maybe_clip_refine_latent(
    x0: torch.Tensor,
    *,
    prompt: str,
    sim_threshold: float,
    model_id: str,
    device: torch.device,
    vae,
    latent_scale: float,
    ae_type: str,
    rae_bridge,
    num_timesteps: int,
    t_frac: float,
    refine_steps: int,
    refiner: Callable[[torch.Tensor, int], torch.Tensor],
) -> torch.Tensor:
    """
    If CLIP cosine < ``sim_threshold``, run ``refiner(x0_clean, t_start)`` (caller supplies
    q_sample + ``sample_loop``). Threshold should be tuned per model id (try 0.18–0.28).
    """
    if sim_threshold <= -2.0:  # sentinel "off" if mis-used
        return x0
    try:
        rgb = decode_latent_preview_rgb(
            x0,
            vae=vae,
            latent_scale=latent_scale,
            ae_type=ae_type,
            rae_bridge=rae_bridge,
            batch_index=0,
        )
        sim = clip_image_text_cosine(rgb, prompt, model_id, device)
    except Exception:
        return x0
    if sim >= sim_threshold:
        return x0
    t_start = int(float(t_frac) * (num_timesteps - 1))
    t_start = min(max(1, t_start), num_timesteps - 1)
    return refiner(x0, t_start, refine_steps)


__all__ = [
    "clip_image_text_cosine",
    "decode_latent_preview_rgb",
    "latent_x0_clip_cosine",
    "maybe_clip_refine_latent",
    "tensor_chw01_to_pil",
]

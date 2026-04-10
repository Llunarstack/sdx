"""
Programmatic text-to-image for loaded stacks (DiT + ``GaussianDiffusion`` + VAE).

``sample.py`` remains the full CLI; this module is for **library-style** calls
(e.g. batch scripts, tests, ``BatchInference``) without duplicating its flags.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

import torch
from PIL import Image

_log = logging.getLogger(__name__)


def tensor_bchw01_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert a single batch image tensor ``(1, 3, H, W)`` in ``[0, 1]`` to RGB PIL."""
    if image.dim() != 4 or int(image.shape[0]) != 1 or int(image.shape[1]) != 3:
        raise ValueError("expected tensor shape (1, 3, H, W)")
    img_np = image[0].permute(1, 2, 0).detach().float().cpu().numpy()
    img_np = (img_np * 255.0).clip(0.0, 255.0).round().astype("uint8")
    return Image.fromarray(img_np, mode="RGB")


@torch.no_grad()
def sample_one_image_pil(
    *,
    model: torch.nn.Module,
    diffusion: Any,
    tokenizer: Any,
    text_encoder: torch.nn.Module,
    vae: torch.nn.Module,
    device: Union[str, torch.device],
    prompt: str,
    negative_prompt: str = "",
    image_size: int = 512,
    cfg_scale: float = 7.5,
    num_inference_steps: int = 50,
    latent_scale: float = 0.18215,
    ae_type: str = "kl",
    max_length: int = 300,
    dtype: torch.dtype = torch.float32,
    text_bundle: Any = None,
    seed: Optional[int] = None,
    rae_bridge: Any = None,
    size_embed_dim: int = 0,
    eval_mode: bool = True,
    **sample_loop_kwargs: Any,
) -> Image.Image:
    """
    Encode ``prompt`` / ``negative_prompt``, run ``diffusion.sample_loop``, decode with ``vae``.

    Extra keyword arguments are forwarded to ``diffusion.sample_loop`` (e.g. ``scheduler``,
    ``solver``, ``eta``). Models that use ``size_embed`` must pass a non-zero
    ``size_embed_dim`` matching the checkpoint (same convention as ``sample.py``).
    """
    # Local import keeps `utils.generation` importable without loading `sample` at package import time.
    from sample import encode_text

    dev = torch.device(device) if isinstance(device, str) else device
    if seed is not None:
        torch.manual_seed(int(seed))
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))

    if eval_mode:
        model.eval()
        vae.eval()
        text_encoder.eval()

    cond = encode_text(
        [prompt],
        tokenizer,
        text_encoder,
        dev,
        max_length=max_length,
        dtype=dtype,
        text_bundle=text_bundle,
    )
    uncond = encode_text(
        [negative_prompt],
        tokenizer,
        text_encoder,
        dev,
        max_length=max_length,
        dtype=dtype,
        text_bundle=text_bundle,
    )
    model_kwargs_cond: dict[str, Any] = {"encoder_hidden_states": cond}
    model_kwargs_uncond: dict[str, Any] = {"encoder_hidden_states": uncond}

    if int(size_embed_dim or 0) > 0:
        lh = max(1, int(image_size) // 8)
        lw = max(1, int(image_size) // 8)
        bsz = int(cond.shape[0])
        sz = torch.tensor([[float(lh), float(lw)]], device=dev, dtype=torch.float32).expand(bsz, -1)
        model_kwargs_cond["size_embed"] = sz
        model_kwargs_uncond["size_embed"] = sz

    latent_size = max(1, int(image_size) // 8)
    shape = (1, 4, latent_size, latent_size)

    loop_kw = dict(sample_loop_kwargs)
    eta = float(loop_kw.pop("eta", 0.0))

    x0 = diffusion.sample_loop(
        model,
        shape,
        model_kwargs_cond=model_kwargs_cond,
        model_kwargs_uncond=model_kwargs_uncond,
        cfg_scale=float(cfg_scale),
        num_inference_steps=int(num_inference_steps),
        eta=eta,
        device=dev,
        dtype=dtype,
        **loop_kw,
    )

    if ae_type == "kl":
        x0 = x0 / float(latent_scale)
    elif ae_type == "rae" and rae_bridge is not None:
        x0 = rae_bridge.dit_to_rae(x0)
    else:
        if ae_type == "rae" and rae_bridge is None:
            _log.warning("ae_type='rae' but rae_bridge is None; decoding DiT latent as-is")

    decoded = vae.decode(x0)
    image = decoded.sample if hasattr(decoded, "sample") else decoded
    image = (image * 0.5 + 0.5).clamp(0, 1)
    return tensor_bchw01_to_pil(image)

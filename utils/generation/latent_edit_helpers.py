"""
Latent-space helpers for **img2img**, **inpainting**, and **outpainting**.

Semantics match ``sample.py`` (``--init-image``, ``--init-latent``, ``--mask``,
``--strength``, ``--inpaint-mode``): use these from scripts, tests, or book tooling
instead of duplicating encode / ``q_sample`` / mask logic.

**Mask convention (inpaint):** white / high values = **region to regenerate**;
black / low = **keep** (known content), same as ``sample.py``'s latent blend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PathLike = Union[str, Path]


def maybe_rae_to_dit(z: torch.Tensor, ae_type: str, rae_bridge: Any) -> torch.Tensor:
    """Map RAE VAE latent to DiT 4-channel space when a bridge is present (same as ``sample.py``)."""
    if z is None or str(ae_type) != "rae" or rae_bridge is None:
        return z
    if int(z.shape[1]) == 4:
        return z
    return rae_bridge.rae_to_dit(z)


def maybe_dit_to_rae_for_decode(z: torch.Tensor, ae_type: str, rae_bridge: Any) -> torch.Tensor:
    """Map DiT latent back to RAE channels before ``vae.decode``."""
    if str(ae_type) == "rae" and rae_bridge is not None:
        return rae_bridge.dit_to_rae(z)
    return z


def latent_hw_from_px(image_size_px: int) -> int:
    return max(1, int(image_size_px) // 8)


def load_rgb_image(path: PathLike) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_square(pil: Image.Image, size_px: int, *, resample: int = Image.Resampling.LANCZOS) -> Image.Image:
    if pil.size == (int(size_px), int(size_px)):
        return pil
    return pil.resize((int(size_px), int(size_px)), resample=resample)


def pil_rgb_to_tensor_m11(pil: Image.Image, *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``(1, 3, H, W)`` in roughly ``[-1, 1]`` (SD VAE convention)."""
    arr = np.array(pil).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    if device is not None:
        t = t.to(device=device, dtype=dtype)
    else:
        t = t.to(dtype=dtype)
    return t


def tensor_m11_bchw_to_pil01(x: torch.Tensor) -> Image.Image:
    """Convert a single ``(1, 3, H, W)`` tensor from model space (~[-1,1]) to RGB PIL."""
    if x.dim() != 4 or int(x.shape[0]) != 1 or int(x.shape[1]) != 3:
        raise ValueError("expected tensor shape (1, 3, H, W)")
    t = x.detach().float().cpu().clamp(-1.0, 1.0)
    t = (t * 0.5 + 0.5).clamp(0.0, 1.0)
    arr = (t[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


@torch.no_grad()
def vae_encode_rgb_to_dit_latent(
    vae: Any,
    x_bchw_m11: torch.Tensor,
    *,
    latent_scale: float,
    ae_type: str = "kl",
    rae_bridge: Any = None,
) -> torch.Tensor:
    """
    Encode an RGB batch in ``[-1,1]`` to **DiT** latent space (``z0``), including RAE bridge if needed.
    """
    enc = vae.encode(x_bchw_m11)
    if hasattr(enc, "latent_dist"):
        z0 = enc.latent_dist.sample() * float(latent_scale)
    else:
        z0 = enc.latent
    return maybe_rae_to_dit(z0, ae_type, rae_bridge)


@torch.no_grad()
def vae_decode_dit_latent_to_tensor01(
    vae: Any,
    z_dit: torch.Tensor,
    *,
    latent_scale: float,
    ae_type: str = "kl",
    rae_bridge: Any = None,
) -> torch.Tensor:
    """Decode DiT latent to ``(B, 3, H, W)`` in ``[0, 1]``."""
    z = z_dit
    if str(ae_type) == "kl":
        z = z / float(latent_scale)
    z = maybe_dit_to_rae_for_decode(z, ae_type, rae_bridge)
    decoded = vae.decode(z)
    image = decoded.sample if hasattr(decoded, "sample") else decoded
    return (image * 0.5 + 0.5).clamp(0, 1)


def load_mask_for_inpaint(
    path: PathLike,
    image_size_px: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Load a grayscale mask, resize to ``image_size_px``, threshold to ``{0,1}``.

    Returns ``(1, 1, H, H)`` with **1 = inpaint / generate** region.
    """
    pil = Image.open(path).convert("L")
    pil = resize_square(pil, image_size_px)
    m = np.array(pil).astype(np.float32) / 255.0
    m = (m >= float(threshold)).astype(np.float32)
    t = torch.from_numpy(m).view(1, 1, int(image_size_px), int(image_size_px))
    if device is not None:
        t = t.to(device=device, dtype=dtype)
    else:
        t = t.to(dtype=dtype)
    return t


def mask_pixel_to_latent(mask_11hh: torch.Tensor, latent_h: int, latent_w: int) -> torch.Tensor:
    """Downsample a pixel mask to latent resolution (nearest)."""
    return F.interpolate(mask_11hh, size=(int(latent_h), int(latent_w)), mode="nearest")


def strength_to_start_timestep(strength: float, num_timesteps: int) -> int:
    """Map ``strength`` in ``[0, 1]`` to a VP training index (same clamp as ``sample.py``)."""
    nt = int(num_timesteps)
    t = int(float(strength) * nt)
    return min(max(1, t), nt - 1)


def flow_matching_noise_mix(z0: torch.Tensor, start_timestep: int, num_timesteps: int) -> torch.Tensor:
    """Linear mix toward Gaussian noise used when ``use_flow_sample`` + img2img in ``sample.py``."""
    den = max(int(num_timesteps) - 1, 1)
    s0 = float(start_timestep) / float(den)
    z_fm = torch.randn_like(z0, device=z0.device, dtype=z0.dtype)
    return (1.0 - s0) * z0 + s0 * z_fm


def build_img2img_initial_latent(
    diffusion: Any,
    z0: torch.Tensor,
    *,
    num_timesteps: int,
    strength: float,
    use_flow_sample: bool = False,
) -> Tuple[torch.Tensor, int]:
    """
    Build ``x_init`` and ``start_timestep`` for plain img2img (no mask).

    VP path: ``q_sample(z0, t_start)``. Flow path: linear mix with noise.
    """
    t_start = strength_to_start_timestep(strength, num_timesteps)
    b = int(z0.shape[0])
    t_vec = torch.tensor([t_start], device=z0.device, dtype=torch.long).expand(b)
    if use_flow_sample:
        x_init = flow_matching_noise_mix(z0, t_start, num_timesteps)
    else:
        x_init = diffusion.q_sample(z0, t_vec)
    return x_init, t_start


@dataclass
class LatentEditInit:
    """Bundle for ``GaussianDiffusion.sample_loop`` (``x_init``, ``start_timestep``, optional inpaint args)."""

    x_init: torch.Tensor
    start_timestep: int
    inpaint_mask: Optional[torch.Tensor] = None
    inpaint_x0: Optional[torch.Tensor] = None
    inpaint_noise: Optional[torch.Tensor] = None

    @property
    def inpaint_freeze_known(self) -> bool:
        """True when MDM-style freeze tensors are all set (see ``sample.py`` ``--inpaint-mode mdm``)."""
        return (
            self.inpaint_mask is not None and self.inpaint_x0 is not None and self.inpaint_noise is not None
        )

    def sample_loop_kwargs(self) -> Dict[str, Any]:
        """Keyword subset to merge into ``diffusion.sample_loop``."""
        return {
            "x_init": self.x_init,
            "start_timestep": self.start_timestep,
            "inpaint_mask": self.inpaint_mask,
            "inpaint_x0": self.inpaint_x0,
            "inpaint_noise": self.inpaint_noise,
            "inpaint_freeze_known": self.inpaint_freeze_known,
        }


def build_inpaint_initial_latent(
    diffusion: Any,
    z0: torch.Tensor,
    mask_latent: torch.Tensor,
    *,
    num_timesteps: int,
    strength: float,
    inpaint_mode: str = "legacy",
) -> LatentEditInit:
    """
    Build initial latent for inpainting.

    ``inpaint_mode``:
      - ``\"legacy\"``: masked region gets pure noise at ``t``; known region gets ``q_sample(z0, t)``.
      - ``\"mdm\"``: different noise for masked vs known at ``t``, plus freeze tensors for ``sample_loop``.
    """
    t_start = strength_to_start_timestep(strength, num_timesteps)
    b = int(z0.shape[0])
    t_vec = torch.tensor([t_start], device=z0.device, dtype=torch.long).expand(b)
    mode = str(inpaint_mode or "legacy").lower().strip()

    if mode == "mdm":
        noise_known = torch.randn_like(z0, device=z0.device, dtype=z0.dtype)
        noise_masked = torch.randn_like(z0, device=z0.device, dtype=z0.dtype)
        x_t_known = diffusion.q_sample(z0, t_vec, noise=noise_known)
        x_t_masked = diffusion.q_sample(z0, t_vec, noise=noise_masked)
        x_init = mask_latent * x_t_masked + (1.0 - mask_latent) * x_t_known
        return LatentEditInit(
            x_init=x_init,
            start_timestep=t_start,
            inpaint_mask=mask_latent,
            inpaint_x0=z0,
            inpaint_noise=noise_known,
        )

    noise = torch.randn_like(z0, device=z0.device, dtype=z0.dtype)
    x_t_full = diffusion.q_sample(z0, t_vec, noise=noise)
    x_init = mask_latent * noise + (1.0 - mask_latent) * x_t_full
    return LatentEditInit(x_init=x_init, start_timestep=t_start, inpaint_mask=None, inpaint_x0=None, inpaint_noise=None)


def build_init_latent_from_tensor(
    diffusion: Any,
    z: torch.Tensor,
    *,
    num_timesteps: int,
    strength: float,
    use_flow_sample: bool = False,
) -> Tuple[torch.Tensor, int]:
    """Same as img2img but ``z`` is already a DiT latent (``--init-latent`` path)."""
    t_start = strength_to_start_timestep(strength, num_timesteps)
    b = int(z.shape[0])
    t_vec = torch.tensor([t_start], device=z.device, dtype=torch.long).expand(b)
    if use_flow_sample:
        x_init = flow_matching_noise_mix(z, t_start, num_timesteps)
    else:
        x_init = diffusion.q_sample(z, t_vec)
    return x_init, t_start


def compose_outpaint_canvas(
    base_rgb: Image.Image,
    target_w: int,
    target_h: int,
    *,
    anchor: str = "center",
    fill_rgb: Tuple[int, int, int] = (128, 128, 128),
) -> Tuple[Image.Image, Image.Image]:
    """
    Place ``base_rgb`` on a larger canvas; return ``(canvas_rgb, mask_L)``.

    The mask is **255** on pixels that are **only canvas fill** (to generate)
    and **0** where the original image was pasted (to keep).
    """
    tw, th = int(target_w), int(target_h)
    bw, bh = base_rgb.size
    if bw > tw or bh > th:
        raise ValueError("base image must fit inside target (resize base first if needed)")
    canvas = Image.new("RGB", (tw, th), color=fill_rgb)
    mask = Image.new("L", (tw, th), 255)
    ax = str(anchor or "center").lower().strip()
    if ax == "center":
        x0 = (tw - bw) // 2
        y0 = (th - bh) // 2
    elif ax == "topleft":
        x0, y0 = 0, 0
    elif ax == "topright":
        x0, y0 = tw - bw, 0
    elif ax == "bottomleft":
        x0, y0 = 0, th - bh
    elif ax == "bottomright":
        x0, y0 = tw - bw, th - bh
    else:
        raise ValueError("anchor must be center|topleft|topright|bottomleft|bottomright")
    canvas.paste(base_rgb, (x0, y0))
    keep = Image.new("L", (bw, bh), 0)
    mask.paste(keep, (x0, y0))
    return canvas, mask


@torch.no_grad()
def prepare_latent_edit_from_paths(
    *,
    vae: Any,
    diffusion: Any,
    init_image_path: PathLike,
    image_size_px: int,
    strength: float,
    num_timesteps: int,
    latent_scale: float,
    ae_type: str = "kl",
    rae_bridge: Any = None,
    mask_path: Optional[PathLike] = None,
    inpaint_mode: str = "legacy",
    use_flow_sample: bool = False,
    device: Optional[torch.device] = None,
    dit_model: Any = None,
) -> LatentEditInit:
    """
    End-to-end: load RGB (and optional mask), encode, build ``LatentEditInit`` for ``sample_loop``.

    If ``mask_path`` is set, ``use_flow_sample`` must be ``False`` (same restriction as ``sample.py``).

    Pass ``dit_model`` (loaded DiT) to assert latent ``H×W`` matches ``model.x_embedder.img_size``
    (block-AR / ViT-scorer workflows share the same latent grid as full txt2img).
    """
    dev = device or torch.device("cpu")
    pil = resize_square(load_rgb_image(init_image_path), image_size_px)
    x = pil_rgb_to_tensor_m11(pil, device=dev, dtype=torch.float32)
    z0 = vae_encode_rgb_to_dit_latent(vae, x, latent_scale=latent_scale, ae_type=ae_type, rae_bridge=rae_bridge)
    lh = latent_hw_from_px(image_size_px)

    if mask_path:
        if use_flow_sample:
            raise ValueError("inpainting requires VP q_sample; set use_flow_sample=False")
        m_pix = load_mask_for_inpaint(mask_path, image_size_px, device=dev, dtype=torch.float32)
        m_lat = mask_pixel_to_latent(m_pix, lh, lh)
        if dit_model is not None:
            from utils.generation.dit_ar_latent_compat import validate_latent_edit_tensors

            validate_latent_edit_tensors(z0, m_lat, dit_model)
        return build_inpaint_initial_latent(
            diffusion,
            z0,
            m_lat,
            num_timesteps=num_timesteps,
            strength=strength,
            inpaint_mode=inpaint_mode,
        )

    if dit_model is not None:
        from utils.generation.dit_ar_latent_compat import validate_latent_edit_tensors

        validate_latent_edit_tensors(z0, None, dit_model)

    x_init, t0 = build_img2img_initial_latent(
        diffusion,
        z0,
        num_timesteps=num_timesteps,
        strength=strength,
        use_flow_sample=use_flow_sample,
    )
    return LatentEditInit(x_init=x_init, start_timestep=t0)


def blend_latents(a: torch.Tensor, b: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Elementwise ``alpha * a + (1 - alpha) * b`` (broadcast-safe)."""
    return alpha * a + (1.0 - alpha) * b


def load_aux_rgb_tensor(
    path: PathLike,
    image_size_px: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Load a sidecar RGB (control / depth / reference) and return ``(1, 3, S, S)`` in ``[-1, 1]``.

    Use the same ``image_size_px`` as the main generation square for alignment with latents.
    """
    pil = resize_square(load_rgb_image(path), int(image_size_px))
    return pil_rgb_to_tensor_m11(pil, device=device, dtype=dtype)


__all__ = [
    "LatentEditInit",
    "blend_latents",
    "build_img2img_initial_latent",
    "build_inpaint_initial_latent",
    "build_init_latent_from_tensor",
    "compose_outpaint_canvas",
    "flow_matching_noise_mix",
    "latent_hw_from_px",
    "load_aux_rgb_tensor",
    "load_mask_for_inpaint",
    "load_rgb_image",
    "mask_pixel_to_latent",
    "maybe_dit_to_rae_for_decode",
    "maybe_rae_to_dit",
    "pil_rgb_to_tensor_m11",
    "prepare_latent_edit_from_paths",
    "resize_square",
    "strength_to_start_timestep",
    "tensor_m11_bchw_to_pil01",
    "vae_decode_dit_latent_to_tensor01",
    "vae_encode_rgb_to_dit_latent",
]

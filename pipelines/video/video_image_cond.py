"""
Video image condition — accurate I2V / FLF / element image understanding.

Without a VAE we still produce:

1. A first-frame latent proxy (downsampled RGB → 4-ch latent-shaped tensor).
2. Identity tokens for PermanentVideoDiT ``identity=`` pathway.
3. An image brief (palette, composition, subject band) injected into the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "ImageCondReport",
    "image_to_latent_proxy",
    "image_to_identity_tokens",
    "image_prompt_brief",
    "prepare_i2v_conditioning",
]


@dataclass(slots=True)
class ImageCondReport:
    brief: str = ""
    palette: list[str] = field(default_factory=list)
    has_file: bool = False
    notes: list[str] = field(default_factory=list)


def _load_rgb(path: str | Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def image_prompt_brief(path: str | Path) -> ImageCondReport:
    """Describe what the model must preserve from the input image."""
    p = Path(path)
    if not p.is_file():
        return ImageCondReport(notes=["missing_image"])
    rgb = _load_rgb(p)
    h, w = rgb.shape[:2]
    # Dominant colors
    small = rgb[:: max(1, h // 32), :: max(1, w // 32)].reshape(-1, 3).astype(np.float32)
    mean = small.mean(axis=0)
    names = []
    # crude name
    r, g, b = mean
    if r > g + 20 and r > b + 20:
        names.append("warm-red dominant")
    elif b > r + 20 and b > g + 10:
        names.append("cool-blue dominant")
    elif g > r + 15 and g > b + 15:
        names.append("green dominant")
    else:
        names.append("neutral palette")
    # Subject band contrast
    cy0, cy1 = int(h * 0.15), int(h * 0.75)
    cx0, cx1 = int(w * 0.2), int(w * 0.8)
    sub = rgb[cy0:cy1, cx0:cx1].astype(np.float32)
    contrast = float(sub.std())
    aspect = "portrait" if h > w * 1.1 else ("landscape" if w > h * 1.1 else "square")
    brief = (
        f"preserve identity and composition from reference image ({aspect} {w}x{h}, "
        f"{names[0]}, subject-band contrast {contrast:.0f}); "
        f"do not invent a different person/product; lock wardrobe colors"
    )
    return ImageCondReport(brief=brief, palette=names, has_file=True, notes=[f"size={w}x{h}"])


def image_to_latent_proxy(
    path: str | Path,
    *,
    height: int = 32,
    width: int = 32,
    channels: int = 4,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    ``(1, C, H, W)`` latent-shaped tensor from an RGB image (VAE stand-in).

    Channel 0–2 = normalized RGB; channel 3 = luminance residual.
    """
    from PIL import Image

    device = torch.device(device)
    p = Path(path)
    if not p.is_file():
        return torch.zeros(1, channels, height, width, device=device)
    im = Image.open(p).convert("RGB").resize((width, height), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0  # H,W,3
    t = torch.from_numpy(arr).permute(2, 0, 1)  # 3,H,W
    if channels >= 4:
        luma = (0.299 * t[0] + 0.587 * t[1] + 0.114 * t[2]).unsqueeze(0)
        # Center to ~N(0,1)-ish for diffusion
        t = torch.cat([t * 2 - 1, luma * 2 - 1], dim=0)
    else:
        t = t[:channels] * 2 - 1
    if t.shape[0] < channels:
        pad = torch.zeros(channels - t.shape[0], height, width)
        t = torch.cat([t, pad], dim=0)
    return t[:channels].unsqueeze(0).to(device)


def image_to_identity_tokens(
    path: str | Path,
    *,
    dim: int = 256,
    num_tokens: int = 4,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """``(1, K, D)`` identity tokens from spatial chroma grid (for PermanentVideoDiT)."""
    device = torch.device(device)
    p = Path(path)
    if not p.is_file():
        return torch.zeros(1, num_tokens, dim, device=device)
    from PIL import Image

    im = Image.open(p).convert("RGB").resize((32, 32), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    tokens = []
    grid = int(max(1, num_tokens**0.5))
    cell = 32 // grid
    for i in range(num_tokens):
        gy, gx = divmod(i, grid)
        y0, x0 = gy * cell, gx * cell
        patch = arr[y0 : y0 + cell, x0 : x0 + cell].reshape(-1, 3).mean(axis=0)
        # Expand 3-D color to dim via seeded projection
        g = torch.Generator(device="cpu")
        g.manual_seed(1000 + i)
        basis = torch.randn(3, dim, generator=g)
        basis = F.normalize(basis, dim=0)
        tok = torch.from_numpy(patch).float() @ basis
        tokens.append(F.normalize(tok, dim=0))
    return torch.stack(tokens, dim=0).unsqueeze(0).to(device)


def prepare_i2v_conditioning(
    image_path: str | Path,
    prompt: str,
    *,
    latent_h: int = 32,
    latent_w: int = 32,
    context_dim: int = 768,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Bundle text + image understanding for neural I2V."""
    from .prompt_ground_graph import parse_prompt_ground, rewrite_prompt_for_adherence
    from .video_text_encode import encode_prompt_context

    brief = image_prompt_brief(image_path)
    g = parse_prompt_ground(prompt)
    rewritten, neg = rewrite_prompt_for_adherence(g)
    full_prompt = f"{rewritten}. {brief.brief}" if brief.brief else rewritten
    context = encode_prompt_context(full_prompt, context_dim=context_dim, device=device)
    latent = image_to_latent_proxy(image_path, height=latent_h, width=latent_w, device=device)
    identity = image_to_identity_tokens(image_path, dim=context_dim, device=device)
    return {
        "prompt": full_prompt,
        "negative": neg,
        "context": context,
        "first_frame_latent": latent,
        "identity": identity,
        "brief": brief,
    }

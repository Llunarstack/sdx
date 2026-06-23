"""
Bridge novel capabilities to sample.py edit modes.

Production paths:
  - Inpaint / img2img: ``sample.py --init-image``, ``--mask``
  - Outpaint: dual-stage / canvas extend (roadmap; use inpaint + larger canvas today)
  - OCR repair: ``utils/generation/sample_edit_runner.py``
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .engine import NovelCapabilitiesEngine


def capabilities_for_args(args: Any) -> List[str]:
    """List novel capability names relevant to current CLI flags."""
    eng = NovelCapabilitiesEngine()
    caps = list(eng.get_capabilities())
    if getattr(args, "mask", ""):
        caps.append("active:inpaint")
    if getattr(args, "init_image", "") or getattr(args, "init_latent", ""):
        caps.append("active:img2img")
    return caps


def inpaint_latent_hook(
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    engine: Optional[NovelCapabilitiesEngine] = None,
) -> torch.Tensor:
    """Research inpaint forward (production path: ``sample.py`` + diffusion inpaint mask)."""
    eng = engine or NovelCapabilitiesEngine()
    return eng.inpainting.inpaint(image, mask)


def sample_flags_for_capability(capability: str) -> Dict[str, Any]:
    """Suggest ``sample.py`` flags for a named novel capability."""
    key = (capability or "").lower().strip()
    if key in ("inpaint", "real-time inpainting", "active:inpaint"):
        return {"mask": "<path>", "init_image": "<path>", "strength": 0.65}
    if key in ("img2img", "image to image", "active:img2img"):
        return {"init_image": "<path>", "strength": 0.55}
    if key in ("outpaint", "infinite outpainting"):
        return {"note": "use larger canvas + inpaint border mask until native outpaint lands"}
    if key in ("magic eraser", "eraser"):
        return {"mask": "<object_mask>", "init_image": "<path>", "segment_prompt": "<object>"}
    return {}


__all__ = ["capabilities_for_args", "inpaint_latent_hook", "sample_flags_for_capability"]

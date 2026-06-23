"""
Bridge multimodal research modules to production edit/generation APIs.

Production paths:
  - ``utils/generation/engine.py`` (MultimodalGenerator)
  - ``sample.py`` img2img / inpaint / control-image
  - ``utils/generation/sample_edit_runner.py``
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .engine import MultimodalFusionEngine


def fusion_from_sample_args(
    *,
    prompt_embedding: Optional[torch.Tensor] = None,
    control_image: Optional[torch.Tensor] = None,
    init_image: Optional[torch.Tensor] = None,
    depth_map: Optional[torch.Tensor] = None,
    sketch: Optional[torch.Tensor] = None,
    engine: Optional[MultimodalFusionEngine] = None,
) -> torch.Tensor:
    """Fuse available modalities (research hook; production uses ``MultimodalGenerator``)."""
    eng = engine or MultimodalFusionEngine()
    return eng.generate_multimodal(
        text=prompt_embedding,
        image=init_image or control_image,
        sketch=sketch,
        depth_map=depth_map,
    )


def describe_active_modalities(args: Any) -> Dict[str, bool]:
    """Inspect an argparse namespace (e.g. from ``sample.py``) for active inputs."""
    return {
        "text": bool(getattr(args, "prompt", "") or getattr(args, "prompt_file", "")),
        "control_image": bool(getattr(args, "control_image", "") or getattr(args, "control", None)),
        "img2img": bool(getattr(args, "init_image", "") or getattr(args, "init_latent", "")),
        "inpaint": bool(getattr(args, "mask", "")),
        "box_layout": bool(getattr(args, "box_layout", "")),
    }


__all__ = ["describe_active_modalities", "fusion_from_sample_args"]

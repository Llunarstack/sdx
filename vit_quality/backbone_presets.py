"""
Suggested timm model names for ViTQualityAdherenceModel.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

TIMM_BACKBONE_PRESETS: List[Tuple[str, str]] = [
    ("vit_small_patch16_224", "Fast baseline; less capacity."),
    ("vit_base_patch16_224", "Default - good speed/quality tradeoff."),
    ("vit_large_patch16_224", "Stronger ViT; needs more VRAM."),
    ("vit_base_patch16_reg4_dinov2.lvd142m", "ViT + register tokens (DINOv2-style) if available in your timm."),
    ("vit_base_patch16_clipl.laion2b", "CLIP-pretrained ViT-L backbone (different d; check --image-size)."),
    ("swin_small_patch4_window7_224", "Hierarchical attention - local structure / textures."),
    ("swin_base_patch4_window7_224", "Larger Swin; use if small details matter for QA."),
    ("convnext_tiny", "ConvNeXt tiny - strong CNN inductive bias (timm)."),
    ("convnext_base", "ConvNeXt base - often strong on natural images."),
]

TIER_ALIASES: Dict[str, str] = {
    "fast": "vit_small_patch16_224",
    "balanced": "vit_base_patch16_224",
    "strong": "vit_large_patch16_224",
}


def describe_presets_for_help() -> str:
    """Compact string for argparse epilog."""
    lines = ["Suggested --model-name values (timm):"]
    for name, why in TIMM_BACKBONE_PRESETS[:6]:
        lines.append(f"  {name} — {why}")
    lines.append(f"Tier shortcuts (not auto-resolved; use full timm name): {TIER_ALIASES}")
    return "\n".join(lines)


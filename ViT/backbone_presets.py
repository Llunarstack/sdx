"""
Suggested timm model names for ViTQualityAdherenceModel (see EXCELLENCE_VS_DIT.md).

All names are passed to timm.create_model(..., num_classes=0, global_pool="avg").
Verify availability with: python -c "import timm; timm.create_model('NAME', pretrained=False, num_classes=0)"
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# (timm model_name, one-line rationale)
TIMM_BACKBONE_PRESETS: List[Tuple[str, str]] = [
    ("vit_small_patch16_224", "Fast baseline; less capacity."),
    ("vit_base_patch16_224", "Default - good speed/quality tradeoff."),
    ("vit_large_patch16_224", "Stronger ViT; needs more VRAM."),
    ("swin_small_patch4_window7_224", "Hierarchical attention - local structure / textures."),
    ("swin_base_patch4_window7_224", "Larger Swin; use if small details matter for QA."),
    ("convnext_tiny", "ConvNeXt tiny - strong CNN inductive bias (timm)."),
    ("convnext_base", "ConvNeXt base - often strong on natural images."),
]

# Named tiers for docs / CLI help text
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

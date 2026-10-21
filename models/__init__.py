"""DiT-family architectures. ``DiT_models_text`` maps every ``--model`` name to its constructor."""

from .dit import DiT_models
from .dit_text import DiT_models_text as _dit_text_models
from .enhanced_dit import EnhancedDiT_models
from .hybrid_dit import HybridDiT_models
from .superior_vit import SuperiorViT_models

DiT_models_text: dict = {
    **_dit_text_models,
    **EnhancedDiT_models,
    **HybridDiT_models,
    **SuperiorViT_models,
}

__all__ = ["DiT_models", "DiT_models_text"]

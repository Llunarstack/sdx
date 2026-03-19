from .dit import DiT_models, DiT_XL_2, DiT_XL_4
from . import dit_text
from .dit_text import DiT_XL_2_Text
from .dit_predecessor import (
    DiT_Predecessor_Text,
    DiT_P_2_Text,
    DiT_P_L_2_Text,
    DiT_Supreme_Text,
    DiT_Supreme_2_Text,
    DiT_Supreme_L_2_Text,
)
from .enhanced_dit import EnhancedDiT_models, EnhancedDiT_XL_2, EnhancedDiT_L_2, EnhancedDiT_B_2

DiT_models_text = {
    **dit_text.DiT_models_text,
    "DiT-P/2-Text": DiT_P_2_Text,
    "DiT-P-L/2-Text": DiT_P_L_2_Text,
    "DiT-Supreme/2-Text": DiT_Supreme_2_Text,
    "DiT-Supreme-L/2-Text": DiT_Supreme_L_2_Text,
    # Enhanced models with built-in advanced features
    **EnhancedDiT_models,
}

__all__ = [
    "DiT_models",
    "DiT_XL_2",
    "DiT_XL_4",
    "DiT_models_text",
    "DiT_XL_2_Text",
    "DiT_Predecessor_Text",
    "DiT_P_2_Text",
    "DiT_P_L_2_Text",
    "DiT_Supreme_Text",
    "DiT_Supreme_2_Text",
    "DiT_Supreme_L_2_Text",
    # Enhanced models
    "EnhancedDiT_models",
    "EnhancedDiT_XL_2",
    "EnhancedDiT_L_2", 
    "EnhancedDiT_B_2",
]

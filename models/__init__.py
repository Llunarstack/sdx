"""DiT and related architectures; `DiT_models_text` merges text-conditioned and enhanced variants."""

from . import dit_text
from .anti_ai_naturalness import (
    AntiAINaturalnessController,
    AsymmetryModule,
    ColorImperfectionModule,
    LineWobbleModule,
    MediumArtifactEncoder,
    MediumProfile,
    RenderNoiseModule,
    StyleFamilyRouter,
    TextureImperfectionModule,
    detect_all_mediums,
    detect_medium,
)
from .camera_perspective import (
    CameraConditioner,
    CameraEmbedder,
    CameraSpec,
    CameraSpecParser,
)
from .cascaded_multimodal_diffusion import CascadedMultimodalDiffusion, CascadedSchedule
from .complex_prompt_handler import (
    ComplexPromptConditioner,
    ConceptFusionModule,
    NSFWAnatomyRouter,
    PhysicsAwareTokenTagger,
    PromptComplexityAnalyzer,
)
from .dit import DiT_models, DiT_XL_2, DiT_XL_4
from .dit_predecessor import (
    DiT_P_2_Text,
    DiT_P_L_2_Text,
    DiT_Predecessor_Text,
    DiT_Supreme_2_Text,
    DiT_Supreme_L_2_Text,
    DiT_Supreme_Text,
)
from .dit_text import DiT_XL_2_Text
from .dynamic_patch import DynamicPatchEmbed, TimestepPatchScheduler
from .enhanced_dit import EnhancedDiT_B_2, EnhancedDiT_L_2, EnhancedDiT_models, EnhancedDiT_XL_2
from .linear_attention import LinearCompressedAttention, LocalWindowAttention
from .long_prompt_encoder import (
    ChunkedTextEncoder,
    HierarchicalPromptParser,
    InlineNegativeExtractor,
    LongPromptController,
    NegativePromptFusion,
)
from .model_enhancements import DropPath, RMSNorm, SE1x1, TokenFiLM
from .multi_character import (
    CharacterSpec,
    InteractionSpec,
    MultiCharacterConditioner,
)
from .native_multimodal_transformer import NativeMultimodalTransformer, concat_padding_masks
from .prompt_adherence import (
    AttributeBindingModule,
    CountConstraint,
    NegationGate,
    PromptAdherenceController,
)
from .rae_latent_bridge import RAELatentBridge
from .register_tokens import JumboToken, RegisterTokens
from .rope2d import RoPE2D, apply_rope2d, build_2d_rope_freqs
from .scene_composer import (
    GlobalSceneConditioner,
    SceneElement,
    SceneGraph,
    SceneGraphEncoder,
    SceneRelation,
)
from .taca import TACA
from .vit_superior import (
    SuperiorViT,
    SuperiorViT_B_2,
    SuperiorViT_L_2,
    SuperiorViT_models,
    SuperiorViT_S_2,
    SuperiorViT_XL_2,
)

DiT_models_text = {
    **dit_text.DiT_models_text,
    "DiT-P/2-Text": DiT_P_2_Text,
    "DiT-P-L/2-Text": DiT_P_L_2_Text,
    "DiT-Supreme/2-Text": DiT_Supreme_2_Text,
    "DiT-Supreme-L/2-Text": DiT_Supreme_L_2_Text,
    # Enhanced models with built-in advanced features
    **EnhancedDiT_models,
    # SuperiorViT — next-gen ViT/DiT with all 2024-2025 improvements
    **SuperiorViT_models,
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
    "NativeMultimodalTransformer",
    "concat_padding_masks",
    "RMSNorm",
    "DropPath",
    "TokenFiLM",
    "SE1x1",
    "CascadedMultimodalDiffusion",
    "CascadedSchedule",
    "RAELatentBridge",
    # SuperiorViT — next-gen ViT/DiT
    "SuperiorViT",
    "SuperiorViT_models",
    "SuperiorViT_XL_2",
    "SuperiorViT_L_2",
    "SuperiorViT_B_2",
    "SuperiorViT_S_2",
    # New building blocks
    "TACA",
    "RoPE2D",
    "build_2d_rope_freqs",
    "apply_rope2d",
    "RegisterTokens",
    "JumboToken",
    "DynamicPatchEmbed",
    "TimestepPatchScheduler",
    "LinearCompressedAttention",
    "LocalWindowAttention",
    # Long prompt + negative handling
    "LongPromptController",
    "HierarchicalPromptParser",
    "InlineNegativeExtractor",
    "ChunkedTextEncoder",
    "NegativePromptFusion",
    # Camera & perspective
    "CameraConditioner",
    "CameraSpecParser",
    "CameraEmbedder",
    "CameraSpec",
    # Complex / NSFW / surreal prompts
    "ComplexPromptConditioner",
    "PromptComplexityAnalyzer",
    "ConceptFusionModule",
    "PhysicsAwareTokenTagger",
    "NSFWAnatomyRouter",
    # Anti-AI naturalness
    "AntiAINaturalnessController",
    "MediumArtifactEncoder",
    "TextureImperfectionModule",
    "AsymmetryModule",
    "LineWobbleModule",
    "RenderNoiseModule",
    "ColorImperfectionModule",
    "StyleFamilyRouter",
    "MediumProfile",
    "detect_medium",
    "detect_all_mediums",
    # Scene composition
    "GlobalSceneConditioner",
    "SceneGraph",
    "SceneElement",
    "SceneRelation",
    "SceneGraphEncoder",
    # Multi-character
    "MultiCharacterConditioner",
    "CharacterSpec",
    "InteractionSpec",
    # Prompt adherence
    "PromptAdherenceController",
    "AttributeBindingModule",
    "NegationGate",
    "CountConstraint",
]

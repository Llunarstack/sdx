"""
sdx.models — DiT-family model architectures.

Core models (always available):
    DiT, DiT_Text, EnhancedDiT, NativeMultimodalTransformer

Optional models (imported if dependencies are satisfied):
    SuperiorViT, CascadedMultimodalDiffusion, and all advanced-feature modules.
    Missing optional modules emit an ImportWarning at package load time.
"""

import warnings
from importlib.util import find_spec

_TORCH_AVAILABLE = find_spec("torch") is not None

# ---------------------------------------------------------------------------
# Core imports — always required; failures here are fatal.
# ---------------------------------------------------------------------------
if _TORCH_AVAILABLE:
    from . import dit_text
    from .dit import DiT_models, DiT_XL_2, DiT_XL_4
    from .dit_text import DiT_XL_2_Text
    from .dynamic_patch import DynamicPatchEmbed, TimestepPatchScheduler
    from .enhanced_dit import EnhancedDiT_B_2, EnhancedDiT_L_2, EnhancedDiT_models, EnhancedDiT_XL_2
    from .linear_attention import LinearCompressedAttention, LocalWindowAttention
    from .model_enhancements import DropPath, RMSNorm, SE1x1, TokenFiLM
    from .native_multimodal_transformer import NativeMultimodalTransformer, concat_padding_masks
    from .rae_latent_bridge import RAELatentBridge
    from .register_tokens import JumboToken, RegisterTokens
    from .rope2d import RoPE2D, apply_rope2d, build_2d_rope_freqs
    from .taca import TACA
else:
    warnings.warn(
        "sdx.models: torch not available; skipping torch-dependent core model imports.",
        ImportWarning,
        stacklevel=2,
    )
    DiT_models_text: dict = {}

# ---------------------------------------------------------------------------
# Optional imports — each wrapped individually so one failure doesn't block
# the rest.  All names are set to None on failure.
# ---------------------------------------------------------------------------

# anti_ai_naturalness
try:
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
except Exception as e:
    warnings.warn(
        f"sdx.models: optional module 'anti_ai_naturalness' could not be imported: {e}",
        ImportWarning,
        stacklevel=2,
    )
    AntiAINaturalnessController = None
    AsymmetryModule = None
    ColorImperfectionModule = None
    LineWobbleModule = None
    MediumArtifactEncoder = None
    MediumProfile = None
    RenderNoiseModule = None
    StyleFamilyRouter = None
    TextureImperfectionModule = None
    detect_all_mediums = None
    detect_medium = None

# camera_perspective
try:
    from .camera_perspective import (
        CameraConditioner,
        CameraEmbedder,
        CameraSpec,
        CameraSpecParser,
    )
except Exception as e:
    warnings.warn(
        f"sdx.models: optional module 'camera_perspective' could not be imported: {e}",
        ImportWarning,
        stacklevel=2,
    )
    CameraConditioner = None
    CameraEmbedder = None
    CameraSpec = None
    CameraSpecParser = None

# cascaded_multimodal_diffusion
try:
    from .cascaded_multimodal_diffusion import CascadedMultimodalDiffusion, CascadedSchedule
except Exception as e:
    warnings.warn(
        f"sdx.models: optional module 'cascaded_multimodal_diffusion' could not be imported: {e}",
        ImportWarning,
        stacklevel=2,
    )
    CascadedMultimodalDiffusion = None
    CascadedSchedule = None

# complex_prompt_handler
try:
    from .complex_prompt_handler import (
        ComplexPromptConditioner,
        ConceptFusionModule,
        NSFWAnatomyRouter,
        PhysicsAwareTokenTagger,
        PromptComplexityAnalyzer,
    )
except Exception as e:
    warnings.warn(
        f"sdx.models: optional module 'complex_prompt_handler' could not be imported: {e}",
        ImportWarning,
        stacklevel=2,
    )
    ComplexPromptConditioner = None
    ConceptFusionModule = None
    NSFWAnatomyRouter = None
    PhysicsAwareTokenTagger = None
    PromptComplexityAnalyzer = None

# dit_text_variants
try:
    from .dit_text_variants import (
        DiT_P_2_Text,
        DiT_P_L_2_Text,
        DiT_Predecessor_Text,
        DiT_Supreme_2_Text,
        DiT_Supreme_L_2_Text,
        DiT_Supreme_Text,
    )
except Exception as e:
    warnings.warn(
        f"sdx.models: optional module 'dit_text_variants' could not be imported: {e}",
        ImportWarning,
        stacklevel=2,
    )
    DiT_P_2_Text = None
    DiT_P_L_2_Text = None
    DiT_Predecessor_Text = None
    DiT_Supreme_2_Text = None
    DiT_Supreme_L_2_Text = None
    DiT_Supreme_Text = None

# long_prompt_encoder
try:
    from .long_prompt_encoder import (
        ChunkedTextEncoder,
        HierarchicalPromptParser,
        InlineNegativeExtractor,
        LongPromptController,
        NegativePromptFusion,
    )
except Exception as e:
    warnings.warn(
        f"sdx.models: optional module 'long_prompt_encoder' could not be imported: {e}",
        ImportWarning,
        stacklevel=2,
    )
    ChunkedTextEncoder = None
    HierarchicalPromptParser = None
    InlineNegativeExtractor = None
    LongPromptController = None
    NegativePromptFusion = None

# multi_character
try:
    from .multi_character import (
        CharacterSpec,
        InteractionSpec,
        MultiCharacterConditioner,
    )
except Exception as e:
    warnings.warn(
        f"sdx.models: optional module 'multi_character' could not be imported: {e}",
        ImportWarning,
        stacklevel=2,
    )
    CharacterSpec = None
    InteractionSpec = None
    MultiCharacterConditioner = None

# prompt_adherence
try:
    from .prompt_adherence import (
        AttributeBindingModule,
        CountConstraint,
        NegationGate,
        PromptAdherenceController,
    )
except Exception as e:
    warnings.warn(
        f"sdx.models: optional module 'prompt_adherence' could not be imported: {e}",
        ImportWarning,
        stacklevel=2,
    )
    AttributeBindingModule = None
    CountConstraint = None
    NegationGate = None
    PromptAdherenceController = None

# scene_composer
try:
    from .scene_composer import (
        GlobalSceneConditioner,
        SceneElement,
        SceneGraph,
        SceneGraphEncoder,
        SceneRelation,
    )
except Exception as e:
    warnings.warn(
        f"sdx.models: optional module 'scene_composer' could not be imported: {e}",
        ImportWarning,
        stacklevel=2,
    )
    GlobalSceneConditioner = None
    SceneElement = None
    SceneGraph = None
    SceneGraphEncoder = None
    SceneRelation = None

# superior_vit
try:
    from .superior_vit import (
        SuperiorViT,
        SuperiorViT_B_2,
        SuperiorViT_L_2,
        SuperiorViT_models,
        SuperiorViT_S_2,
        SuperiorViT_XL_2,
    )
except Exception as e:
    warnings.warn(
        f"sdx.models: optional module 'superior_vit' could not be imported: {e}",
        ImportWarning,
        stacklevel=2,
    )
    SuperiorViT = None
    SuperiorViT_B_2 = None
    SuperiorViT_L_2 = None
    SuperiorViT_models = None
    SuperiorViT_S_2 = None
    SuperiorViT_XL_2 = None

# vit_next_blocks
try:
    from .vit_next_blocks import LayerScale, apply_topk_token_keep
except Exception as e:
    warnings.warn(
        f"sdx.models: optional module 'vit_next_blocks' could not be imported: {e}",
        ImportWarning,
        stacklevel=2,
    )
    LayerScale = None
    apply_topk_token_keep = None

# ---------------------------------------------------------------------------
# Composite model registry
# ---------------------------------------------------------------------------
if _TORCH_AVAILABLE:
    DiT_models_text: dict = {**dit_text.DiT_models_text}

    # Add optional variant models if successfully imported
    _optional_text_models: dict = {}
    if DiT_P_2_Text is not None:
        _optional_text_models["DiT-P/2-Text"] = DiT_P_2_Text
    if DiT_P_L_2_Text is not None:
        _optional_text_models["DiT-P-L/2-Text"] = DiT_P_L_2_Text
    if DiT_Supreme_2_Text is not None:
        _optional_text_models["DiT-Supreme/2-Text"] = DiT_Supreme_2_Text
    if DiT_Supreme_L_2_Text is not None:
        _optional_text_models["DiT-Supreme-L/2-Text"] = DiT_Supreme_L_2_Text
    DiT_models_text.update(_optional_text_models)
    DiT_models_text.update(EnhancedDiT_models)
    if SuperiorViT_models is not None:
        DiT_models_text.update(SuperiorViT_models)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
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
    # vit_next_blocks
    "LayerScale",
    "apply_topk_token_keep",
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

"""
GenerationPipeline — wires every new module into one coherent forward pass.

Call order per denoising step:
  1. LongPromptController   — parse + merge + weight + suppress text embeddings
  2. CameraConditioner      — parse camera spec, embed, apply to image tokens
  3. ComplexPromptConditioner — concept fusion, physics tagging, NSFW routing
  4. MultiCharacterConditioner — character isolation, interaction encoding
  5. GlobalSceneConditioner  — scene graph encoding, occlusion, scale consistency
  6. AntiAINaturalnessController — style-aware imperfection injection
  7. PromptAdherenceController — attribute binding + negation gate on cross-attn

All modules are optional — pass None to skip any of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .anti_ai_naturalness import AntiAINaturalnessController, MediumProfile, detect_medium
from .camera_perspective import CameraConditioner, CameraSpec
from .complex_prompt_handler import ComplexPromptConditioner, PromptComplexityProfile
from .long_prompt_encoder import LongPromptController
from .multi_character import CharacterSpec, InteractionSpec, MultiCharacterConditioner
from .prompt_adherence import PromptAdherenceController
from .scene_composer import GlobalSceneConditioner, SceneGraph


@dataclass
class GenerationContext:
    """
    All optional conditioning inputs for one generation call.
    Pass only what you have — everything else is skipped gracefully.
    """
    # Text
    prompt: str = ""
    negative_prompt: str = ""
    # Pre-encoded text embeddings (B, L, D) — if None, pipeline uses prompt strings only
    pos_text_emb: Optional[torch.Tensor] = None
    neg_text_emb: Optional[torch.Tensor] = None
    # Camera
    camera_spec: Optional[CameraSpec] = None          # if None, parsed from prompt
    depth_map: Optional[torch.Tensor] = None          # (B, 1, H, W)
    # Characters
    character_specs: List[CharacterSpec] = field(default_factory=list)
    interactions: List[InteractionSpec] = field(default_factory=list)
    # Scene
    scene_graph: Optional[SceneGraph] = None
    # Style
    medium_profile: Optional[MediumProfile] = None    # if None, detected from prompt
    naturalness_strength: float = 0.5
    # Complexity
    complexity_profile: Optional[PromptComplexityProfile] = None


class GenerationPipeline(nn.Module):
    """
    Unified generation pipeline that wires all conditioning modules together.

    Usage:
        pipeline = GenerationPipeline(hidden_size=1152, num_heads=16)
        ctx = GenerationContext(
            prompt="anime girl, cel shading, but no extra fingers",
            character_specs=[CharacterSpec("alice", "Alice", bbox=(0.1,0.1,0.9,0.9))],
        )
        # In your denoising loop:
        x, text_emb = pipeline(x, text_emb, t_emb, ctx, h_patches=16, w_patches=16)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        enable_long_prompt: bool = True,
        enable_camera: bool = True,
        enable_complex: bool = True,
        enable_multi_char: bool = True,
        enable_scene: bool = True,
        enable_naturalness: bool = True,
        enable_adherence: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        if enable_long_prompt:
            self.long_prompt = LongPromptController(hidden_size, num_heads)
        else:
            self.long_prompt = None

        if enable_camera:
            self.camera = CameraConditioner(hidden_size)
        else:
            self.camera = None

        if enable_complex:
            self.complex_prompt = ComplexPromptConditioner(hidden_size, num_heads)
        else:
            self.complex_prompt = None

        if enable_multi_char:
            self.multi_char = MultiCharacterConditioner(hidden_size, num_heads)
        else:
            self.multi_char = None

        if enable_scene:
            self.scene = GlobalSceneConditioner(hidden_size)
        else:
            self.scene = None

        if enable_naturalness:
            self.naturalness = AntiAINaturalnessController(hidden_size)
        else:
            self.naturalness = None

        if enable_adherence:
            self.adherence = PromptAdherenceController(hidden_size, num_heads)
        else:
            self.adherence = None

    # ------------------------------------------------------------------
    def prepare_text(
        self,
        ctx: GenerationContext,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Apply long-prompt processing to text embeddings.
        Returns (pos_emb, neg_emb) — both may be None if no embeddings provided.
        """
        pos_emb = ctx.pos_text_emb
        neg_emb = ctx.neg_text_emb

        if self.long_prompt is None or pos_emb is None:
            return pos_emb, neg_emb

        # Weight positive tokens by importance
        pos_emb = self.long_prompt.weight(pos_emb)

        # Suppress positive tokens that overlap with negatives
        if neg_emb is not None:
            pos_emb = self.long_prompt.suppress(pos_emb, neg_emb)
        elif ctx.negative_prompt:
            # No neg embedding provided but we have a string — parse inline negatives
            self.long_prompt.parse(ctx.prompt)
            # inline_negatives are strings; we can't encode them without a text encoder
            # so we skip suppression here (handled at CFG level)

        return pos_emb, neg_emb

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        text_emb: Optional[torch.Tensor],
        t_emb: torch.Tensor,
        ctx: GenerationContext,
        h_patches: int,
        w_patches: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply all conditioning modules to image tokens and text embeddings.

        Args:
            x:         (B, N, D) image tokens.
            text_emb:  (B, L, D) text embeddings (or None).
            t_emb:     (B, D) timestep embedding.
            ctx:       GenerationContext with all optional inputs.
            h_patches, w_patches: Spatial token grid dimensions.

        Returns:
            x:         (B, N, D) conditioned image tokens.
            text_emb:  (B, L', D) possibly augmented text embeddings.
        """
        B = x.shape[0]
        device = x.device

        # 1. Long prompt text processing
        if self.long_prompt is not None and text_emb is not None:
            neg_emb = ctx.neg_text_emb
            text_emb = self.long_prompt.weight(text_emb)
            if neg_emb is not None:
                text_emb = self.long_prompt.suppress(text_emb, neg_emb)

        # 2. Camera conditioning
        if self.camera is not None:
            spec = ctx.camera_spec
            if spec is None and ctx.prompt:
                spec = self.camera.parse(ctx.prompt)
            if spec is not None:
                cam_emb = self.camera.embed(spec, device).expand(B, -1)  # (B, D)
                x = self.camera(x, cam_emb, h_patches, w_patches, ctx.depth_map)
                t_emb = t_emb + cam_emb  # inject camera into timestep conditioning

        # 3. Complex prompt conditioning
        if self.complex_prompt is not None and text_emb is not None:
            profile = ctx.complexity_profile
            if profile is None and ctx.prompt:
                profile = self.complex_prompt.analyze(ctx.prompt)
            x = self.complex_prompt(x, text_emb, profile)

        # 4. Multi-character conditioning
        if self.multi_char is not None and ctx.character_specs:
            x, char_cond = self.multi_char(
                x, t_emb, ctx.character_specs, ctx.interactions,
                h_patches, w_patches, text_emb,
            )
            t_emb = t_emb + char_cond

        # 5. Scene graph conditioning
        if self.scene is not None and text_emb is not None:
            x, text_emb = self.scene(
                x, text_emb, ctx.scene_graph, h_patches, w_patches, device
            )

        # 6. Anti-AI naturalness
        if self.naturalness is not None:
            medium = ctx.medium_profile
            if medium is None and ctx.prompt:
                medium = detect_medium(ctx.prompt)
            x = self.naturalness(
                x, medium, h_patches, w_patches, ctx.naturalness_strength
            )

        return x, text_emb

    # ------------------------------------------------------------------
    def apply_cross_attention_hooks(
        self,
        attn_logits: torch.Tensor,
        value: torch.Tensor,
        text_emb: torch.Tensor,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply prompt adherence hooks inside cross-attention.
        Call this from within your DiT block's cross-attention forward.

        Args:
            attn_logits: (B, H, N, L)
            value:       (B, H, L, D_head)
            text_emb:    (B, L, D)
            spatial_mask:(B, S, N) optional per-subject spatial regions.
        Returns:
            biased_logits, gated_value
        """
        if self.adherence is None:
            return attn_logits, value
        return self.adherence.apply_to_cross_attention(
            attn_logits, value, text_emb, spatial_mask
        )


__all__ = ["GenerationPipeline", "GenerationContext"]

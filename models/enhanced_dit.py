"""EnhancedDiT: class-conditional DiT whose blocks add layout, anatomy, text-rendering and consistency modules,
softmax-blended per block (legacy ``scripts/enhanced`` training stack)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from timm.models.vision_transformer import PatchEmbed

from .dit import DiTBlock, FinalLayer, LabelEmbedder, TimestepEmbedder, get_2d_sincos_pos_embed, unpatchify


class SpatialControlModule(nn.Module):
    """Object layout ``(B, num_objects, 4)`` boxes -> attention toward each object's slot."""

    def __init__(self, hidden_size: int, num_objects: int = 10):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_objects = num_objects
        self.object_pos_embed = nn.Parameter(torch.randn(num_objects, hidden_size) * 0.02)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(4, hidden_size // 2),  # x, y, w, h
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.count_constraint = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.GELU(), nn.Linear(hidden_size // 2, num_objects), nn.Sigmoid()
        )
        self.spatial_attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)

    def forward(self, x: torch.Tensor, spatial_layout: torch.Tensor | None = None) -> torch.Tensor:
        """``x``: ``(B, N, D)``; ``spatial_layout``: ``(B, num_objects, 4)`` as (x, y, w, h)."""
        if spatial_layout is None:
            return x
        spatial_features = self.spatial_encoder(spatial_layout) + self.object_pos_embed.unsqueeze(0)
        # Count head gates each object's values, so slots it deems absent contribute nothing.
        count_weights = self.count_constraint(spatial_features.mean(dim=1))
        values = spatial_features * count_weights.unsqueeze(-1)
        x_attended, _ = self.spatial_attention(x, spatial_features, values)
        return x + 0.3 * x_attended


class AnatomyAwareAttention(nn.Module):
    """Self-attention + constraint MLP residual; inside ``anatomy_mask`` an extra attention over finger slots."""

    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.anatomy_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        # Unused by forward; kept so EnhancedDiT checkpoints keep loading strictly.
        self.body_part_embeddings = nn.Parameter(torch.randn(10, hidden_size) * 0.02)
        self.anatomy_constraint = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.hand_attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.finger_embeddings = nn.Parameter(torch.randn(5, hidden_size) * 0.02)

    def forward(self, x: torch.Tensor, anatomy_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``x``: ``(B, N, D)``; ``anatomy_mask``: ``(B, M)`` region mask on a square grid (resized to ``N``)."""
        B, N, D = x.shape
        x_anatomy, _ = self.anatomy_attention(x, x, x)
        x = x + 0.4 * self.anatomy_constraint(x_anatomy)
        if anatomy_mask is None:
            return x
        x_hands, _ = self.hand_attention(x, x, x)
        finger_features = self.finger_embeddings.unsqueeze(0).expand(B, -1, -1)
        finger_attended, _ = self.hand_attention(x_hands, finger_features, finger_features)
        mask = anatomy_mask
        if mask.shape[1] != N:
            side, target = int(mask.shape[1] ** 0.5), int(N**0.5)
            if side != target:
                mask = F.interpolate(mask.view(B, 1, side, side), size=(target, target), mode="nearest").view(B, -1)
        return torch.where(mask.unsqueeze(-1) > 0.5, x + 0.3 * finger_attended, x)


class TextRenderingLayer(nn.Module):
    """Glyph-token cross-attention: text token ids (+ typography style, + 2D positions) attended by the image."""

    def __init__(self, hidden_size: int, vocab_size: int = 50000):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.text_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.typography_embeddings = nn.Parameter(torch.randn(20, hidden_size) * 0.02)
        self.text_position_net = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size),  # +2 for the (x, y) text position
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.char_attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.text_quality_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2), nn.GELU(), nn.Linear(hidden_size * 2, hidden_size), nn.Dropout(0.1)
        )

    def forward(
        self,
        x: torch.Tensor,
        text_tokens: torch.Tensor | None = None,
        text_positions: torch.Tensor | None = None,
        typography_style: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``text_tokens``: ``(B, L)`` ids; ``text_positions``: ``(B, L, 2)``; ``typography_style``: ``(B,)`` ids."""
        if text_tokens is None:
            return x
        text_features = self.text_embeddings(text_tokens)
        if typography_style is not None:
            text_features = text_features + self.typography_embeddings[typography_style].unsqueeze(1)
        if text_positions is not None:
            text_features = self.text_position_net(torch.cat([text_features, text_positions], dim=-1))
        text_attended, _ = self.char_attention(text_features, text_features, text_features)
        text_enhanced = self.text_quality_net(text_attended)
        x_text_attended, _ = self.char_attention(x, text_enhanced, text_enhanced)
        return x + 0.5 * x_text_attended


class ConsistencyEmbedder(nn.Module):
    """Character / style id embeddings (+ learned memory rows) mixed into every token."""

    def __init__(self, hidden_size: int, max_characters: int = 100, max_styles: int = 50):
        super().__init__()
        self.hidden_size = hidden_size
        self.character_embeddings = nn.Embedding(max_characters, hidden_size)
        self.style_embeddings = nn.Embedding(max_styles, hidden_size)
        self.consistency_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # character + style
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.character_memory = nn.Parameter(torch.randn(max_characters, hidden_size) * 0.02)
        self.style_memory = nn.Parameter(torch.randn(max_styles, hidden_size) * 0.02)

    def forward(
        self, x: torch.Tensor, character_id: torch.Tensor | None = None, style_id: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``character_id`` / ``style_id``: ``(B,)`` ids; a missing one contributes zeros."""
        if character_id is None and style_id is None:
            return x
        B, N, D = x.shape
        char = style = x.new_zeros(B, D)
        if character_id is not None:
            char = self.character_embeddings(character_id) + self.character_memory[character_id]
        if style_id is not None:
            style = self.style_embeddings(style_id) + self.style_memory[style_id]
        consistency = self.consistency_net(torch.cat([char, style], dim=-1))
        return x + 0.3 * consistency.unsqueeze(1).expand(-1, N, -1)


class EnhancedDiTBlock(DiTBlock):
    """``DiTBlock`` followed by the four feature modules, mixed with softmax-normalised learned weights."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **kwargs):
        super().__init__(hidden_size, num_heads, mlp_ratio, **kwargs)
        self.spatial_control = SpatialControlModule(hidden_size)
        self.anatomy_attention = AnatomyAwareAttention(hidden_size, num_heads)
        self.text_rendering = TextRenderingLayer(hidden_size)
        self.consistency_embedder = ConsistencyEmbedder(hidden_size)
        self.feature_blend = nn.Parameter(torch.ones(4) * 0.25)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        spatial_layout: torch.Tensor | None = None,
        anatomy_mask: torch.Tensor | None = None,
        text_tokens: torch.Tensor | None = None,
        text_positions: torch.Tensor | None = None,
        typography_style: torch.Tensor | None = None,
        character_id: torch.Tensor | None = None,
        style_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = super().forward(x, c)
        features = (
            self.spatial_control(x, spatial_layout),
            self.anatomy_attention(x, anatomy_mask),
            self.text_rendering(x, text_tokens, text_positions, typography_style),
            self.consistency_embedder(x, character_id, style_id),
        )
        return sum(w * f for w, f in zip(F.softmax(self.feature_blend, dim=0), features))


class EnhancedDiT(nn.Module):
    """Class-conditional DiT built from ``EnhancedDiTBlock``; each feature module can be switched off."""

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = True,
        enable_spatial_control: bool = True,
        enable_anatomy_awareness: bool = True,
        enable_text_rendering: bool = True,
        enable_consistency: bool = True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.enable_spatial_control = enable_spatial_control
        self.enable_anatomy_awareness = enable_anatomy_awareness
        self.enable_text_rendering = enable_text_rendering
        self.enable_consistency = enable_consistency
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList(
            [EnhancedDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self._grad_checkpointing = False
        self.initialize_weights()

    def enable_gradient_checkpointing(self):
        self._grad_checkpointing = True

    def initialize_weights(self):
        """Base DiT init: sin/cos positions, Xavier patch embed, N(0, 0.02) embeddings, adaLN-Zero + zero output."""
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        spatial_layout: torch.Tensor | None = None,
        anatomy_mask: torch.Tensor | None = None,
        text_tokens: torch.Tensor | None = None,
        text_positions: torch.Tensor | None = None,
        typography_style: torch.Tensor | None = None,
        character_id: torch.Tensor | None = None,
        style_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``x``: ``(B, C, H, W)`` latent, ``t`` / ``y``: ``(B,)``; feature inputs are shaped as in each module."""
        h, w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        x = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(t) + self.y_embedder(y, self.training)
        block_kwargs = dict(
            spatial_layout=spatial_layout if self.enable_spatial_control else None,
            anatomy_mask=anatomy_mask if self.enable_anatomy_awareness else None,
            text_tokens=text_tokens if self.enable_text_rendering else None,
            text_positions=text_positions if self.enable_text_rendering else None,
            typography_style=typography_style if self.enable_text_rendering else None,
            character_id=character_id if self.enable_consistency else None,
            style_id=style_id if self.enable_consistency else None,
        )
        checkpointing = self._grad_checkpointing and self.training
        for block in self.blocks:
            if checkpointing:
                x = torch.utils.checkpoint.checkpoint(block, x, c, use_reentrant=False, **block_kwargs)
            else:
                x = block(x, c, **block_kwargs)
        return unpatchify(self.final_layer(x, c), self.patch_size, self.out_channels, h, w)


def EnhancedDiT_XL_2(**kwargs):
    return EnhancedDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def EnhancedDiT_L_2(**kwargs):
    return EnhancedDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def EnhancedDiT_B_2(**kwargs):
    return EnhancedDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


EnhancedDiT_models: dict = {
    "EnhancedDiT-XL/2": EnhancedDiT_XL_2,
    "EnhancedDiT-L/2": EnhancedDiT_L_2,
    "EnhancedDiT-B/2": EnhancedDiT_B_2,
}

__all__ = [
    "EnhancedDiT",
    "EnhancedDiTBlock",
    "EnhancedDiT_XL_2",
    "EnhancedDiT_L_2",
    "EnhancedDiT_B_2",
    "EnhancedDiT_models",
]

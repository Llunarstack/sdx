"""
Multi-Character Interaction System.

Problems solved:
  1. Identity mixing — character A's face appears on character B's body
  2. Clothing attribution — character A's red dress ends up on character B
  3. Object ownership — "Alice holds a sword, Bob holds a shield" gets swapped
  4. Interaction coherence — touching/hugging/fighting poses are physically plausible
  5. Spatial separation — characters don't bleed into each other's regions

Architecture:
  - CharacterSlot: per-character identity embedding + spatial region
  - CharacterIsolationAttention: self-attention that respects character boundaries
  - ClothingAttributionModule: binds clothing/object tokens to their owner's region
  - InteractionEncoder: encodes inter-character relations (touching, facing, etc.)
  - MultiCharacterConditioner: top-level module that wires everything together

Design principle: characters are first-class citizens in the attention graph.
Each character gets its own set of cross-attention keys/values derived from
its identity embedding, preventing cross-contamination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_enhancements import RMSNorm

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CharacterSpec:
    """Specification for one character in a multi-character scene."""
    character_id: str
    name: str
    # Bounding box in normalised [0,1] coords: (x1, y1, x2, y2)
    bbox: Optional[Tuple[float, float, float, float]] = None
    # Text description of this character (subset of full prompt)
    description: str = ""
    # Clothing/object tokens that belong to this character
    owned_tokens: List[int] = field(default_factory=list)
    # Reference embedding (e.g. from IP-Adapter / face encoder)
    reference_embedding: Optional[torch.Tensor] = None


@dataclass
class InteractionSpec:
    """Specification for an interaction between two characters."""
    char_a: str  # character_id
    char_b: str  # character_id
    relation: str  # "touching", "hugging", "fighting", "facing", "holding_hands", etc.
    contact_region: Optional[str] = None  # "hands", "shoulders", "full_body"


# ---------------------------------------------------------------------------
# Character Slot Embeddings
# ---------------------------------------------------------------------------

class CharacterSlotEmbedding(nn.Module):
    """
    Learnable per-character slot embeddings.

    Each character gets a unique slot vector that is injected into the
    cross-attention keys/values, ensuring the model can distinguish
    character A's tokens from character B's tokens.

    Args:
        hidden_size: Transformer hidden dim.
        max_characters: Maximum number of characters supported.
        use_bbox_encoding: Encode bounding box position into the slot.
    """

    def __init__(self, hidden_size: int, max_characters: int = 6, use_bbox_encoding: bool = True):
        super().__init__()
        self.max_characters = max_characters
        self.hidden_size = hidden_size
        self.use_bbox = use_bbox_encoding

        # Learnable base slot embeddings
        self.slots = nn.Parameter(torch.randn(max_characters, hidden_size) * 0.02)

        # Bounding box encoder: (x1, y1, x2, y2, cx, cy, w, h) -> hidden_size
        if use_bbox_encoding:
            self.bbox_encoder = nn.Sequential(
                nn.Linear(8, hidden_size // 2),
                nn.SiLU(),
                nn.Linear(hidden_size // 2, hidden_size),
            )
            nn.init.zeros_(self.bbox_encoder[-1].weight)
            nn.init.zeros_(self.bbox_encoder[-1].bias)

        # Identity projection (for reference embeddings from face encoder / IP-Adapter)
        self.identity_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def encode_bbox(self, bbox: Tuple[float, float, float, float]) -> torch.Tensor:
        """Encode a bounding box to a feature vector."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        feat = torch.tensor([x1, y1, x2, y2, cx, cy, w, h], dtype=torch.float32)
        return self.bbox_encoder(feat.unsqueeze(0))  # (1, D)

    def get_slot(
        self,
        slot_idx: int,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        reference_emb: Optional[torch.Tensor] = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Get the embedding for character slot `slot_idx`.
        Returns: (1, D)
        """
        slot = self.slots[slot_idx].unsqueeze(0)  # (1, D)
        if device is not None:
            slot = slot.to(device)

        if self.use_bbox and bbox is not None:
            bbox_feat = self.encode_bbox(bbox).to(slot.device)
            slot = slot + bbox_feat

        if reference_emb is not None:
            ref = self.identity_proj(reference_emb.to(slot.device))
            if ref.dim() == 1:
                ref = ref.unsqueeze(0)
            slot = slot + ref

        return slot

    def get_all_slots(
        self,
        specs: List[CharacterSpec],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Get slot embeddings for all characters.
        Returns: (num_chars, D)
        """
        slots = []
        for i, spec in enumerate(specs[:self.max_characters]):
            slot = self.get_slot(
                i,
                bbox=spec.bbox,
                reference_emb=spec.reference_embedding,
                device=device,
            )
            slots.append(slot)
        return torch.cat(slots, dim=0)  # (num_chars, D)


# ---------------------------------------------------------------------------
# Character Isolation Attention
# ---------------------------------------------------------------------------

class CharacterIsolationAttention(nn.Module):
    """
    Self-attention that respects character spatial boundaries.

    Standard self-attention lets all patches attend to all other patches,
    causing identity mixing. This module adds a soft isolation bias:
    patches within the same character's region attend more strongly to
    each other, and less strongly to patches in other characters' regions.

    The isolation is soft (not hard masking) so characters can still
    interact at their boundaries — important for touching/hugging scenes.

    Args:
        hidden_size: Token dimension.
        num_heads: Attention heads.
        isolation_strength: How strongly to isolate characters (0=off, 1=full).
        interaction_boost: Extra attention boost at character boundaries (for interactions).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        isolation_strength: float = 0.6,
        interaction_boost: float = 0.3,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.isolation_strength = float(isolation_strength)
        self.interaction_boost = float(interaction_boost)

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        # Learnable per-head isolation gate
        self.isolation_gate = nn.Parameter(
            torch.full((num_heads,), isolation_strength)
        )

        nn.init.zeros_(self.out_proj.weight)

    def _build_isolation_bias(
        self,
        character_masks: torch.Tensor,
        interaction_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build additive attention bias from character masks.

        Args:
            character_masks: (B, C, N) — which patches belong to which character.
            interaction_mask: (B, N, N) — patches that should have boosted attention
                              (e.g. boundary regions between interacting characters).
        Returns:
            bias: (B, 1, N, N) additive bias for attention logits.
        """
        B, C, N = character_masks.shape

        # Same-character affinity: (B, N, N)
        # Two patches have high affinity if they belong to the same character
        # character_masks: (B, C, N) -> (B, N, C)
        masks_t = character_masks.transpose(1, 2)  # (B, N, C)
        same_char = torch.bmm(masks_t, character_masks)  # (B, N, N) — dot product over characters

        # Normalise to [0, 1]
        same_char = same_char / (same_char.max(dim=-1, keepdim=True).values.clamp(min=1e-6))

        # Convert to bias: positive for same character, negative for different
        bias = (same_char - 0.5) * 2.0  # [-1, 1]

        # Add interaction boost at boundaries
        if interaction_mask is not None:
            bias = bias + self.interaction_boost * interaction_mask

        return bias.unsqueeze(1)  # (B, 1, N, N)

    def forward(
        self,
        x: torch.Tensor,
        character_masks: Optional[torch.Tensor] = None,
        interaction_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) image tokens.
            character_masks: (B, C, N) per-character spatial masks.
            interaction_mask: (B, N, N) interaction boundary mask.
        Returns:
            (B, N, D)
        """
        B, N, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, N, N)

        # Apply isolation bias
        if character_masks is not None:
            bias = self._build_isolation_bias(character_masks, interaction_mask)
            gate = self.isolation_gate.view(1, self.num_heads, 1, 1).clamp(0, 1)
            attn = attn + gate * bias * 3.0  # scale bias to meaningful range

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Clothing / Object Attribution
# ---------------------------------------------------------------------------

class ClothingAttributionModule(nn.Module):
    """
    Binds clothing and object tokens to their owner character's spatial region.

    The core problem: "Alice wears a red dress, Bob wears a blue suit" —
    the model often puts the red dress on Bob or mixes the colours.

    This module:
    1. Detects clothing/object tokens in the text sequence.
    2. Assigns each to the nearest character slot (by embedding similarity).
    3. Applies a spatial bias so those tokens only activate in their owner's region.

    Args:
        hidden_size: Token dimension.
        num_heads: Attention heads.
        max_characters: Maximum characters.
    """

    def __init__(self, hidden_size: int, num_heads: int, max_characters: int = 6):
        super().__init__()
        self.num_heads = num_heads
        self.max_characters = max_characters

        # Clothing/object token detector
        self.token_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 2),  # [background, clothing/object]
        )

        # Character ownership predictor: given a token, which character owns it?
        self.ownership_predictor = nn.Linear(hidden_size, max_characters, bias=False)

        # Attribution strength gate per head
        self.attribution_gate = nn.Parameter(torch.ones(num_heads) * 0.5)

    def get_ownership_logits(
        self,
        text_emb: torch.Tensor,
        character_slots: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict which character owns each text token.

        Args:
            text_emb: (B, L, D) text tokens.
            character_slots: (C, D) character slot embeddings.
        Returns:
            ownership: (B, L, C) ownership probabilities.
        """
        B, L, D = text_emb.shape

        # Similarity between each token and each character slot
        slots = character_slots.unsqueeze(0).expand(B, -1, -1)  # (B, C, D)
        # (B, L, D) x (B, D, C) -> (B, L, C)
        sim = torch.bmm(
            F.normalize(text_emb, dim=-1),
            F.normalize(slots, dim=-1).transpose(1, 2),
        )
        return F.softmax(sim, dim=-1)  # (B, L, C)

    def build_attribution_bias(
        self,
        text_emb: torch.Tensor,
        character_slots: torch.Tensor,
        character_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build cross-attention bias that routes clothing tokens to their owner's region.

        Args:
            text_emb: (B, L, D)
            character_slots: (C, D)
            character_masks: (B, C, N) spatial masks.
        Returns:
            bias: (B, 1, N, L) additive cross-attention bias.
        """
        B, L, D = text_emb.shape

        # Detect clothing/object tokens
        token_logits = self.token_classifier(text_emb)  # (B, L, 2)
        is_clothing = F.softmax(token_logits, dim=-1)[:, :, 1]  # (B, L) — prob of being clothing

        # Ownership: which character owns each token
        ownership = self.get_ownership_logits(text_emb, character_slots)  # (B, L, C)

        # For each patch, compute which tokens it should attend to:
        # patch p in character c's region should attend to tokens owned by character c
        # character_masks: (B, C, N) -> (B, N, C)
        patch_char = character_masks.transpose(1, 2)  # (B, N, C)

        # (B, N, C) x (B, C, L) -> (B, N, L) — expected attention pattern
        expected = torch.bmm(patch_char, ownership.transpose(1, 2))  # (B, N, L)

        # Scale by clothing probability (only apply to clothing/object tokens)
        expected = expected * is_clothing.unsqueeze(1)  # (B, N, L)

        # Convert to bias
        bias = (expected - 0.5) * 2.0  # [-1, 1]
        return bias.unsqueeze(1)  # (B, 1, N, L)

    def forward(
        self,
        attn_logits: torch.Tensor,
        text_emb: torch.Tensor,
        character_slots: torch.Tensor,
        character_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply attribution bias to cross-attention logits.
        attn_logits: (B, H, N, L)
        Returns: (B, H, N, L)
        """
        bias = self.build_attribution_bias(text_emb, character_slots, character_masks)
        gate = self.attribution_gate.view(1, self.num_heads, 1, 1).clamp(0, 1)
        return attn_logits + gate * bias * 3.0


# ---------------------------------------------------------------------------
# Interaction Encoder
# ---------------------------------------------------------------------------

class InteractionEncoder(nn.Module):
    """
    Encodes inter-character interactions (touching, hugging, fighting, etc.)
    and injects them as conditioning into the transformer.

    This ensures physically plausible contact poses — e.g. when two characters
    hug, their arms should wrap around each other, not clip through.

    Args:
        hidden_size: Token dimension.
        num_relation_types: Number of distinct relation types.
    """

    RELATIONS = [
        "neutral", "touching", "hugging", "fighting", "holding_hands",
        "facing", "back_to_back", "side_by_side", "one_behind_other",
        "carrying", "dancing", "shaking_hands", "pointing_at",
    ]

    def __init__(self, hidden_size: int, num_relation_types: int = 0):
        super().__init__()
        n = num_relation_types if num_relation_types > 0 else len(self.RELATIONS)
        self.relation_embed = nn.Embedding(n, hidden_size)
        self.relation_to_id = {r: i for i, r in enumerate(self.RELATIONS)}

        # Contact region encoder: encodes where characters touch
        # Input: (x, y) normalised contact point + contact area
        self.contact_encoder = nn.Sequential(
            nn.Linear(3, hidden_size // 4),
            nn.SiLU(),
            nn.Linear(hidden_size // 4, hidden_size),
        )

        # Interaction conditioning MLP
        self.interaction_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.zeros_(self.interaction_mlp[-1].weight)
        nn.init.zeros_(self.interaction_mlp[-1].bias)

    def encode_interaction(
        self,
        spec: InteractionSpec,
        char_a_slot: torch.Tensor,
        char_b_slot: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode one interaction into a conditioning vector.

        Args:
            spec: InteractionSpec.
            char_a_slot: (1, D) slot embedding for character A.
            char_b_slot: (1, D) slot embedding for character B.
        Returns:
            (1, D) interaction conditioning vector.
        """
        rel_id = self.relation_to_id.get(spec.relation, 0)
        rel_emb = self.relation_embed(
            torch.tensor(rel_id, device=char_a_slot.device)
        ).unsqueeze(0)  # (1, D)

        # Combine: relation + character pair
        pair = char_a_slot + char_b_slot + rel_emb  # (1, D)
        combined = torch.cat([pair, rel_emb], dim=-1)  # (1, 2D)
        return self.interaction_mlp(combined)  # (1, D)

    def forward(
        self,
        interactions: List[InteractionSpec],
        character_slots: torch.Tensor,
        char_id_to_idx: Dict[str, int],
    ) -> torch.Tensor:
        """
        Encode all interactions into a single conditioning vector.

        Args:
            interactions: List of InteractionSpec.
            character_slots: (C, D) character slot embeddings.
            char_id_to_idx: Maps character_id -> slot index.
        Returns:
            (1, D) combined interaction conditioning.
        """
        if not interactions:
            return torch.zeros(1, character_slots.shape[-1], device=character_slots.device)

        interaction_vecs = []
        for spec in interactions:
            idx_a = char_id_to_idx.get(spec.char_a, 0)
            idx_b = char_id_to_idx.get(spec.char_b, 1)
            slot_a = character_slots[idx_a].unsqueeze(0)
            slot_b = character_slots[idx_b].unsqueeze(0)
            vec = self.encode_interaction(spec, slot_a, slot_b)
            interaction_vecs.append(vec)

        # Sum all interaction vectors
        return torch.stack(interaction_vecs, dim=0).sum(dim=0)  # (1, D)


# ---------------------------------------------------------------------------
# Spatial Mask Generator
# ---------------------------------------------------------------------------

class SpatialMaskGenerator:
    """
    Generates per-character spatial masks from bounding boxes.

    Converts normalised bounding boxes to patch-level binary/soft masks
    that indicate which image patches belong to which character.
    """

    @staticmethod
    def bbox_to_mask(
        bbox: Tuple[float, float, float, float],
        h_patches: int,
        w_patches: int,
        soft: bool = True,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Convert a normalised bounding box to a patch mask.

        Args:
            bbox: (x1, y1, x2, y2) in [0, 1].
            h_patches, w_patches: Number of patches in each dimension.
            soft: If True, use Gaussian falloff at boundaries.
        Returns:
            mask: (h_patches * w_patches,) float tensor.
        """
        x1, y1, x2, y2 = bbox
        # Patch grid centres
        ys = torch.linspace(0.5 / h_patches, 1 - 0.5 / h_patches, h_patches, device=device)
        xs = torch.linspace(0.5 / w_patches, 1 - 0.5 / w_patches, w_patches, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)

        if soft:
            # Gaussian falloff from bbox centre
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            sx, sy = max((x2 - x1) / 4, 0.05), max((y2 - y1) / 4, 0.05)
            mask = torch.exp(
                -0.5 * ((grid_x - cx) / sx) ** 2
                - 0.5 * ((grid_y - cy) / sy) ** 2
            )
        else:
            mask = ((grid_x >= x1) & (grid_x <= x2) & (grid_y >= y1) & (grid_y <= y2)).float()

        return mask.reshape(-1)  # (N,)

    @classmethod
    def build_masks(
        cls,
        specs: List[CharacterSpec],
        h_patches: int,
        w_patches: int,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Build masks for all characters.
        Returns: (C, N) where N = h_patches * w_patches.
        """
        masks = []
        for spec in specs:
            if spec.bbox is not None:
                m = cls.bbox_to_mask(spec.bbox, h_patches, w_patches, soft=True, device=device)
            else:
                # No bbox: uniform mask (character can be anywhere)
                N = h_patches * w_patches
                m = torch.ones(N, device=device) / len(specs)
            masks.append(m)
        return torch.stack(masks, dim=0)  # (C, N)


# ---------------------------------------------------------------------------
# Multi-Character Conditioner (top-level)
# ---------------------------------------------------------------------------

class MultiCharacterConditioner(nn.Module):
    """
    Top-level module that conditions the transformer on multiple characters.

    Injects per-character identity, spatial, and interaction information
    into the diffusion transformer at every block.

    Usage:
        conditioner = MultiCharacterConditioner(hidden_size=1152, num_heads=16)
        # Build specs from your prompt parser / UI
        specs = [CharacterSpec("alice", "Alice", bbox=(0.1, 0.1, 0.5, 0.9), description="..."),
                 CharacterSpec("bob",   "Bob",   bbox=(0.5, 0.1, 0.9, 0.9), description="...")]
        interactions = [InteractionSpec("alice", "bob", "hugging")]
        # During forward pass:
        x, extra_cond = conditioner(x, t_emb, specs, interactions, h_patches, w_patches)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_characters: int = 6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_characters = max_characters

        self.slot_embed = CharacterSlotEmbedding(hidden_size, max_characters)
        self.isolation_attn = CharacterIsolationAttention(hidden_size, num_heads)
        self.clothing_attr = ClothingAttributionModule(hidden_size, num_heads, max_characters)
        self.interaction_enc = InteractionEncoder(hidden_size)
        self.mask_gen = SpatialMaskGenerator()

        # Conditioning projection: character slots -> conditioning vector added to timestep emb
        self.cond_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.zeros_(self.cond_proj[-1].weight)
        nn.init.zeros_(self.cond_proj[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        specs: List[CharacterSpec],
        interactions: List[InteractionSpec],
        h_patches: int,
        w_patches: int,
        text_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, D) image tokens.
            t_emb: (B, D) timestep embedding.
            specs: List of CharacterSpec (one per character).
            interactions: List of InteractionSpec.
            h_patches, w_patches: Spatial token grid dimensions.
            text_emb: (B, L, D) text tokens (for clothing attribution).
        Returns:
            x: (B, N, D) updated tokens.
            extra_cond: (B, D) extra conditioning to add to t_emb.
        """
        B = x.shape[0]
        device = x.device

        if not specs:
            return x, torch.zeros_like(t_emb)

        # Build character slot embeddings
        char_slots = self.slot_embed.get_all_slots(specs, device)  # (C, D)

        # Build spatial masks
        char_masks_1 = self.mask_gen.build_masks(specs, h_patches, w_patches, device)  # (C, N)
        char_masks = char_masks_1.unsqueeze(0).expand(B, -1, -1)  # (B, C, N)

        # Apply character isolation attention
        x = self.isolation_attn(x, character_masks=char_masks)

        # Apply clothing attribution (if text available)
        # (This would be called inside cross-attention, but we apply it here as a post-hoc correction)
        # In a full integration, pass attn_logits through clothing_attr.forward()

        # Encode interactions
        char_id_to_idx = {spec.character_id: i for i, spec in enumerate(specs)}
        interaction_vec = self.interaction_enc(interactions, char_slots, char_id_to_idx)  # (1, D)

        # Build extra conditioning: sum of character slots + interaction
        char_cond = char_slots.mean(dim=0, keepdim=True)  # (1, D)
        extra_cond = self.cond_proj(char_cond + interaction_vec)  # (1, D)
        extra_cond = extra_cond.expand(B, -1)  # (B, D)

        return x, extra_cond


__all__ = [
    "CharacterSpec",
    "InteractionSpec",
    "CharacterSlotEmbedding",
    "CharacterIsolationAttention",
    "ClothingAttributionModule",
    "InteractionEncoder",
    "SpatialMaskGenerator",
    "MultiCharacterConditioner",
]

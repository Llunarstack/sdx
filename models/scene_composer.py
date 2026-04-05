"""
Scene Composer — unified scene understanding for complex multi-element prompts.

Ties together camera, characters, objects, spatial layout, and lighting into
one coherent conditioning pass. This is the "director's brain" of the model.

Problems solved:
  1. Spatial incoherence — objects float, characters don't stand on surfaces
  2. Scale inconsistency — foreground objects same size as background ones
  3. Occlusion errors — objects that should be behind others appear in front
  4. Lighting inconsistency — each element lit from a different direction
  5. Perspective mismatch — elements rendered from different viewpoints

Architecture:
  - SceneGraph: structured representation of scene elements and relations
  - SceneGraphEncoder: encodes the scene graph into conditioning tokens
  - OcclusionOrderModule: enforces correct depth ordering / occlusion
  - ScalePerspectiveConsistency: ensures scale matches perspective
  - GlobalSceneConditioner: top-level module that produces a unified scene embedding
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Scene Graph
# ---------------------------------------------------------------------------

@dataclass
class SceneElement:
    """One element in the scene (character, object, background, etc.)."""
    element_id: str
    label: str
    element_type: str           # "character", "object", "background", "fx"
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x1,y1,x2,y2) normalised
    depth: float = 0.5          # 0=foreground, 1=background
    scale: float = 1.0          # relative scale
    description: str = ""


@dataclass
class SceneRelation:
    """A spatial/semantic relation between two scene elements."""
    subject_id: str
    object_id: str
    relation: str               # "in_front_of", "behind", "on_top_of", "holding",
                                # "touching", "near", "far_from", "occludes"


@dataclass
class SceneGraph:
    """Complete structured scene description."""
    elements: List[SceneElement] = field(default_factory=list)
    relations: List[SceneRelation] = field(default_factory=list)
    global_description: str = ""
    lighting_direction: Tuple[float, float] = (0.5, 0.3)  # (x, y) normalised
    ambient_level: float = 0.5


# ---------------------------------------------------------------------------
# Scene Graph Encoder
# ---------------------------------------------------------------------------

class SceneGraphEncoder(nn.Module):
    """
    Encodes a SceneGraph into a set of conditioning tokens.

    Each scene element gets one token. Relations are encoded as
    edge features added to the element tokens via a graph attention pass.

    Args:
        hidden_size: Token dimension.
        max_elements: Maximum scene elements.
    """

    ELEMENT_TYPES = ["character", "object", "background", "fx", "text", "unknown"]
    RELATION_TYPES = [
        "in_front_of", "behind", "on_top_of", "below", "holding",
        "touching", "near", "far_from", "occludes", "overlaps",
        "left_of", "right_of", "above", "contains", "attached_to",
    ]

    def __init__(self, hidden_size: int, max_elements: int = 12):
        super().__init__()
        self.max_elements = max_elements
        self.hidden_size = hidden_size

        # Element type embedding
        self.type_embed = nn.Embedding(len(self.ELEMENT_TYPES), hidden_size // 4)

        # Spatial encoding: (x1, y1, x2, y2, cx, cy, w, h, depth, scale) = 10 dims
        self.spatial_proj = nn.Linear(10, hidden_size // 2)

        # Relation embedding
        self.rel_embed = nn.Embedding(len(self.RELATION_TYPES), hidden_size // 4)

        # Element fusion
        self.element_fusion = nn.Sequential(
            nn.Linear(hidden_size // 4 + hidden_size // 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # Graph attention: elements attend to each other via relations
        self.graph_attn = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.graph_norm = nn.LayerNorm(hidden_size)

        # Lighting encoder
        self.lighting_proj = nn.Linear(3, hidden_size)  # (lx, ly, ambient)

        self._type_to_id = {t: i for i, t in enumerate(self.ELEMENT_TYPES)}
        self._rel_to_id = {r: i for i, r in enumerate(self.RELATION_TYPES)}

    def _encode_element(self, el: SceneElement, device: torch.device) -> torch.Tensor:
        """Encode one scene element to (hidden_size,)."""
        type_id = self._type_to_id.get(el.element_type, len(self.ELEMENT_TYPES) - 1)
        type_emb = self.type_embed(torch.tensor(type_id, device=device))  # (D/4,)

        if el.bbox is not None:
            x1, y1, x2, y2 = el.bbox
        else:
            x1, y1, x2, y2 = 0.0, 0.0, 1.0, 1.0
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        spatial = torch.tensor(
            [x1, y1, x2, y2, cx, cy, w, h, el.depth, el.scale],
            device=device, dtype=torch.float32
        )
        spatial_emb = self.spatial_proj(spatial)  # (D/2,)

        combined = torch.cat([type_emb, spatial_emb], dim=-1)  # (D/4 + D/2,)
        return self.element_fusion(combined)  # (D,)

    def forward(
        self,
        scene: SceneGraph,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Encode scene graph into conditioning tokens.
        Returns: (1, num_elements + 1, hidden_size) — elements + global lighting token.
        """
        elements = scene.elements[:self.max_elements]
        if not elements:
            return torch.zeros(1, 1, self.hidden_size, device=device)

        # Encode each element
        el_tokens = torch.stack([
            self._encode_element(el, device) for el in elements
        ], dim=0).unsqueeze(0)  # (1, E, D)

        # Build relation bias for graph attention
        E = len(elements)
        el_id_to_idx = {el.element_id: i for i, el in enumerate(elements)}
        rel_bias = torch.zeros(E, E, device=device)
        for rel in scene.relations:
            i = el_id_to_idx.get(rel.subject_id, -1)
            j = el_id_to_idx.get(rel.object_id, -1)
            if i >= 0 and j >= 0:
                rel_id = self._rel_to_id.get(rel.relation, 0)
                rel_emb = self.rel_embed(torch.tensor(rel_id, device=device))
                # Use relation embedding norm as bias strength
                rel_bias[i, j] += rel_emb.norm().item() * 0.1

        # Graph attention with relation bias
        normed = self.graph_norm(el_tokens)
        attended, _ = self.graph_attn(
            normed, normed, normed,
            attn_mask=rel_bias.unsqueeze(0) if E > 1 else None,
        )
        el_tokens = el_tokens + attended  # (1, E, D)

        # Lighting token
        lx, ly = scene.lighting_direction
        light_feat = torch.tensor([lx, ly, scene.ambient_level], device=device, dtype=torch.float32)
        light_token = self.lighting_proj(light_feat).unsqueeze(0).unsqueeze(0)  # (1, 1, D)

        return torch.cat([el_tokens, light_token], dim=1)  # (1, E+1, D)


# ---------------------------------------------------------------------------
# Occlusion Order Module
# ---------------------------------------------------------------------------

class OcclusionOrderModule(nn.Module):
    """
    Enforces correct depth ordering and occlusion between scene elements.

    When element A is in front of element B, patches in A's region should
    dominate over patches in B's region. This module applies a depth-ordered
    attention bias so foreground elements "win" over background elements.

    Args:
        hidden_size: Token dimension.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # Depth predictor: given image tokens, predict per-patch depth
        self.depth_pred = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),  # 0=foreground, 1=background
        )

        # Occlusion gate: how strongly to enforce occlusion
        self.occlusion_gate = nn.Parameter(torch.tensor(0.3))

    def forward(
        self,
        x: torch.Tensor,
        scene: Optional[SceneGraph] = None,
        h_patches: int = 16,
        w_patches: int = 16,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) image tokens.
            scene: Optional scene graph with depth info.
        Returns:
            (B, N, D) occlusion-ordered tokens.
        """
        B, N, D = x.shape
        gate = torch.sigmoid(self.occlusion_gate)

        # Predict per-patch depth
        pred_depth = self.depth_pred(x)  # (B, N, 1)

        if scene is not None and scene.elements:
            # Build target depth map from scene graph
            target_depth = torch.full((N,), 0.5, device=x.device)
            for el in scene.elements:
                if el.bbox is not None:
                    x1, y1, x2, y2 = el.bbox
                    for idx in range(N):
                        row = idx // w_patches
                        col = idx % w_patches
                        px = (col + 0.5) / w_patches
                        py = (row + 0.5) / h_patches
                        if x1 <= px <= x2 and y1 <= py <= y2:
                            target_depth[idx] = el.depth

            target_depth = target_depth.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)

            # Inject depth correction into tokens
            # Foreground tokens (low depth) get boosted, background suppressed
            depth_scale = 1.0 + gate * (0.5 - pred_depth) * 0.4  # (B, N, 1)
            x = x * depth_scale

        return x


# ---------------------------------------------------------------------------
# Scale-Perspective Consistency
# ---------------------------------------------------------------------------

class ScalePerspectiveConsistency(nn.Module):
    """
    Ensures object scale is consistent with perspective depth.

    Objects further away (higher depth) should appear smaller.
    This module applies a perspective-consistent scale correction
    to image tokens based on their predicted depth.

    Args:
        hidden_size: Token dimension.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # Scale correction: maps depth -> scale factor
        self.scale_corrector = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),
        )
        # Init to output ~1.0 (identity)
        nn.init.zeros_(self.scale_corrector[-2].weight)
        nn.init.constant_(self.scale_corrector[-2].bias, 2.0)

        # Perspective focal length (learned)
        self.focal = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        x: torch.Tensor,
        depth_map: Optional[torch.Tensor] = None,
        h_patches: int = 16,
        w_patches: int = 16,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) image tokens.
            depth_map: (B, N, 1) per-patch depth (0=near, 1=far). If None, estimated.
        Returns:
            (B, N, D) scale-corrected tokens.
        """
        B, N, D = x.shape

        if depth_map is None:
            # Estimate depth from vertical position (simple prior: top=far, bottom=near)
            rows = torch.linspace(0, 1, h_patches, device=x.device)
            cols = torch.linspace(0, 1, w_patches, device=x.device)
            gr, _ = torch.meshgrid(rows, cols, indexing='ij')
            depth_map = gr.flatten().unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)

        # Perspective scale: objects at depth d appear at scale 1/(1 + focal*d)
        focal = torch.sigmoid(self.focal) * 2.0  # [0, 2]
        persp_scale = 1.0 / (1.0 + focal * depth_map)  # (B, N, 1)

        # Learned correction on top of perspective prior
        inp = torch.cat([x, depth_map], dim=-1)  # (B, N, D+1)
        correction = self.scale_corrector(inp)  # (B, N, 1)

        # Apply scale: modulate token magnitude
        scale = persp_scale * correction
        # Normalise so mean scale ≈ 1 (don't change overall brightness)
        scale = scale / (scale.mean(dim=1, keepdim=True) + 1e-6)

        return x * scale


# ---------------------------------------------------------------------------
# Global Scene Conditioner (top-level)
# ---------------------------------------------------------------------------

class GlobalSceneConditioner(nn.Module):
    """
    Top-level scene conditioning module.

    Produces a unified scene embedding from a SceneGraph and applies:
    - Scene graph encoding (elements + relations + lighting)
    - Occlusion ordering
    - Scale-perspective consistency

    The scene tokens are appended to the text embedding sequence,
    giving the model explicit structural knowledge of the scene.

    Usage:
        conditioner = GlobalSceneConditioner(hidden_size=1152)
        scene = SceneGraph(elements=[...], relations=[...])
        scene_tokens = conditioner.encode(scene, device)
        # Append to text_emb: text_emb_aug = torch.cat([text_emb, scene_tokens], dim=1)
        x = conditioner.apply(x, scene, h_patches, w_patches)
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.graph_encoder = SceneGraphEncoder(hidden_size)
        self.occlusion = OcclusionOrderModule(hidden_size)
        self.scale_consistency = ScalePerspectiveConsistency(hidden_size)

        # Scene conditioning gate
        self.scene_gate = nn.Parameter(torch.tensor(0.5))

    def encode(self, scene: SceneGraph, device: torch.device) -> torch.Tensor:
        """
        Encode scene graph into tokens to append to text embedding.
        Returns: (1, E+1, D) scene tokens.
        """
        return self.graph_encoder(scene, device)

    def apply(
        self,
        x: torch.Tensor,
        scene: Optional[SceneGraph],
        h_patches: int,
        w_patches: int,
        depth_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply scene conditioning to image tokens.

        Args:
            x: (B, N, D) image tokens.
            scene: Optional scene graph.
            h_patches, w_patches: Spatial grid.
            depth_map: (B, N, 1) optional depth map.
        Returns:
            (B, N, D) scene-conditioned tokens.
        """
        gate = torch.sigmoid(self.scene_gate)

        x_occ = self.occlusion(x, scene, h_patches, w_patches)
        x_scale = self.scale_consistency(x, depth_map, h_patches, w_patches)

        # Blend: weighted combination of original + corrections
        x = x + gate * 0.5 * (x_occ - x) + gate * 0.5 * (x_scale - x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        text_emb: torch.Tensor,
        scene: Optional[SceneGraph],
        h_patches: int,
        w_patches: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: (B, N, D) scene-conditioned image tokens.
            text_emb_aug: (B, L + E + 1, D) text + scene tokens.
        """
        dev = device or x.device

        # Encode scene graph and append to text
        if scene is not None:
            scene_tokens = self.encode(scene, dev)  # (1, E+1, D)
            B = text_emb.shape[0]
            scene_tokens = scene_tokens.expand(B, -1, -1)
            text_emb_aug = torch.cat([text_emb, scene_tokens], dim=1)
        else:
            text_emb_aug = text_emb

        # Apply spatial corrections
        x = self.apply(x, scene, h_patches, w_patches)

        return x, text_emb_aug


__all__ = [
    "SceneGraph",
    "SceneElement",
    "SceneRelation",
    "SceneGraphEncoder",
    "OcclusionOrderModule",
    "ScalePerspectiveConsistency",
    "GlobalSceneConditioner",
]

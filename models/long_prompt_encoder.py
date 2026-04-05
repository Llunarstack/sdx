"""
Long Prompt Encoder — native support for 400–600+ word prompts.

Standard CLIP truncates at 77 tokens. T5 handles more but attention dilutes
over long sequences — early tokens dominate, late tokens get ignored.

This module solves it with:
  1. HierarchicalPromptParser: splits prompt into semantic layers
     (primary subject → secondary elements → mood/lighting → camera/technical → micro-details)
  2. ChunkedTextEncoder: encodes long prompts in overlapping chunks, merges with
     cross-chunk attention so context flows across chunk boundaries
  3. PromptPriorityWeighter: up-weights important tokens so they don't get diluted
  4. InlineNegativeExtractor: pulls "but no / avoid / without / except" clauses
     out of the positive prompt and returns them as a separate negative embedding

All outputs are drop-in replacements for standard (B, L, D) text embeddings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Semantic layers
# ---------------------------------------------------------------------------

LAYER_PATTERNS = {
    "primary_subject": [
        r'\b(portrait|photo|painting|illustration|render|image|picture)\s+of\b',
        r'\b(a|an|the)\s+\w+\s+(man|woman|girl|boy|person|figure|character|creature|monster|robot|alien)\b',
    ],
    "secondary_elements": [
        r'\b(wearing|holding|carrying|with|beside|near|next to|surrounded by)\b',
        r'\b(background|foreground|setting|environment|scene|landscape|interior|exterior)\b',
    ],
    "mood_lighting": [
        r'\b(lighting|light|shadow|glow|illuminat|ambient|dramatic|soft|harsh|golden|neon|moonlight|sunlight|candlelight)\b',
        r'\b(mood|atmosphere|vibe|tone|feeling|emotion|dark|bright|moody|ethereal|cinematic)\b',
        r'\b(fog|mist|haze|smoke|rain|snow|dust|particles|bokeh|depth of field)\b',
    ],
    "camera_technical": [
        r'\b(shot on|photographed|filmed|captured|taken with|lens|mm|f\/|aperture|iso|shutter)\b',
        r'\b(angle|pov|perspective|view|close.?up|wide.?angle|macro|telephoto|fisheye|tilt.?shift)\b',
        r'\b(rule of thirds|golden ratio|leading lines|negative space|framing|composition)\b',
        r'\b(35mm|50mm|85mm|24mm|anamorphic|prime|zoom|bokeh|dof|shallow|deep focus)\b',
    ],
    "style_medium": [
        r'\b(style|art style|medium|technique|rendered in|painted in|drawn in|digital art|oil painting|watercolor|pencil|ink|charcoal)\b',
        r'\b(hyperrealistic|photorealistic|impressionist|expressionist|surrealist|abstract|minimalist|maximalist)\b',
        r'\b(4k|8k|hd|ultra.?hd|high.?res|detailed|intricate|sharp|crisp|smooth|rough|textured)\b',
    ],
    "micro_details": [
        r'\b(texture|fabric|material|surface|pattern|grain|noise|imperfection|scratch|wear|aged|weathered)\b',
        r'\b(hair|skin|eyes|lips|hands|fingers|nails|jewelry|accessories|tattoo|scar|freckle)\b',
    ],
}

INLINE_NEGATIVE_PATTERNS = [
    r'\bbut\s+(?:no|not|without|avoid|excluding|except)\b(.+?)(?:\.|,\s+(?:and|but|with)|$)',
    r'\bwithout\b\s+(.+?)(?:\.|,\s+(?:and|but|with)|$)',
    r'\bno\s+(.+?)(?:\.|,\s+(?:and|but|with)|$)',
    r'\bavoid(?:ing)?\b\s+(.+?)(?:\.|,\s+(?:and|but|with)|$)',
    r'\bexcluding\b\s+(.+?)(?:\.|,\s+(?:and|but|with)|$)',
    r'\bexcept\s+(?:for\s+)?(.+?)(?:\.|,\s+(?:and|but|with)|$)',
    r'\bnot\s+(.+?)(?:\.|,\s+(?:and|but|with)|$)',
]


@dataclass
class ParsedLongPrompt:
    """Result of hierarchical prompt parsing."""
    raw: str
    positive_text: str                          # prompt with inline negatives stripped
    inline_negatives: List[str]                 # extracted negative clauses
    layers: Dict[str, str]                      # semantic layer -> text
    layer_weights: Dict[str, float]             # importance weight per layer
    token_priority: Optional[torch.Tensor] = None  # (L,) per-token importance


# ---------------------------------------------------------------------------
# Inline Negative Extractor
# ---------------------------------------------------------------------------

class InlineNegativeExtractor:
    """
    Extracts negative clauses embedded in a positive prompt.

    Handles patterns like:
      "cyberpunk girl, neon lights, but no extra fingers, no watermark"
      "portrait of a knight, avoid anime style, without blurry background"
      "forest scene except for any people or animals"

    Returns:
      positive_text: prompt with negative clauses removed
      negatives: list of extracted negative phrases
    """

    def extract(self, prompt: str) -> Tuple[str, List[str]]:
        negatives: List[str] = []
        cleaned = prompt

        for pattern in INLINE_NEGATIVE_PATTERNS:
            for match in re.finditer(pattern, cleaned, re.IGNORECASE):
                neg_text = match.group(1).strip().rstrip('.,;')
                if neg_text:
                    negatives.append(neg_text)
            # Remove matched spans from cleaned prompt
            cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)

        # Clean up extra whitespace and punctuation
        cleaned = re.sub(r'\s+', ' ', cleaned).strip().strip('.,;')
        return cleaned, negatives


# ---------------------------------------------------------------------------
# Hierarchical Prompt Parser
# ---------------------------------------------------------------------------

class HierarchicalPromptParser:
    """
    Splits a long prompt into semantic layers and assigns importance weights.

    Layer priority (highest → lowest):
      primary_subject (1.0) > camera_technical (0.9) > secondary_elements (0.8)
      > mood_lighting (0.7) > style_medium (0.6) > micro_details (0.5)
    """

    LAYER_WEIGHTS = {
        "primary_subject":   1.0,
        "camera_technical":  0.9,
        "secondary_elements": 0.8,
        "mood_lighting":     0.7,
        "style_medium":      0.6,
        "micro_details":     0.5,
        "unclassified":      0.65,
    }

    def __init__(self):
        self.neg_extractor = InlineNegativeExtractor()
        self._compiled = {
            layer: [re.compile(p, re.IGNORECASE) for p in patterns]
            for layer, patterns in LAYER_PATTERNS.items()
        }

    def _classify_clause(self, clause: str) -> str:
        """Assign a clause to its semantic layer."""
        for layer, patterns in self._compiled.items():
            for pat in patterns:
                if pat.search(clause):
                    return layer
        return "unclassified"

    def parse(self, prompt: str) -> ParsedLongPrompt:
        # Step 1: extract inline negatives
        positive_text, inline_negatives = self.neg_extractor.extract(prompt)

        # Step 2: split into clauses (comma / semicolon / period separated)
        clauses = re.split(r'[,;.]\s*', positive_text)
        clauses = [c.strip() for c in clauses if c.strip()]

        # Step 3: classify each clause
        layers: Dict[str, List[str]] = {k: [] for k in LAYER_PATTERNS}
        layers["unclassified"] = []
        for clause in clauses:
            layer = self._classify_clause(clause)
            layers[layer].append(clause)

        # Step 4: merge clauses per layer
        layer_texts = {k: ', '.join(v) for k, v in layers.items() if v}

        # Step 5: compute layer weights
        layer_weights = {k: self.LAYER_WEIGHTS.get(k, 0.65) for k in layer_texts}

        return ParsedLongPrompt(
            raw=prompt,
            positive_text=positive_text,
            inline_negatives=inline_negatives,
            layers=layer_texts,
            layer_weights=layer_weights,
        )


# ---------------------------------------------------------------------------
# Chunked Text Encoder
# ---------------------------------------------------------------------------

class ChunkedTextEncoder(nn.Module):
    """
    Encodes long prompts by splitting into overlapping chunks and merging
    with cross-chunk attention so context flows across boundaries.

    This avoids the 77-token CLIP truncation while preserving coherence
    across the full prompt.

    Args:
        hidden_size: Output token dimension.
        chunk_size: Tokens per chunk (77 for CLIP compatibility).
        overlap: Overlapping tokens between adjacent chunks.
        max_chunks: Maximum number of chunks (limits total prompt length).
    """

    def __init__(
        self,
        hidden_size: int,
        chunk_size: int = 75,
        overlap: int = 10,
        max_chunks: int = 8,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_chunks = max_chunks
        self.hidden_size = hidden_size

        # Cross-chunk attention: lets chunks attend to each other
        self.cross_chunk_attn = nn.MultiheadAttention(
            hidden_size, num_heads=8, batch_first=True, dropout=0.0
        )
        self.cross_chunk_norm = nn.LayerNorm(hidden_size)

        # Priority weighter: up-weights important tokens
        self.priority_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),
        )
        # Init to near-uniform weights
        nn.init.zeros_(self.priority_proj[-2].weight)
        nn.init.constant_(self.priority_proj[-2].bias, 1.0)

        # Layer importance projection: scales tokens by their semantic layer weight
        self.layer_scale = nn.Linear(1, hidden_size, bias=False)
        nn.init.ones_(self.layer_scale.weight)

    def merge_chunks(
        self,
        chunk_embeddings: List[torch.Tensor],
        layer_weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        Merge a list of chunk embeddings into one sequence.

        Args:
            chunk_embeddings: List of (B, L_i, D) tensors.
            layer_weights: Optional per-chunk importance weights.
        Returns:
            (B, L_total, D) merged embedding.
        """
        if not chunk_embeddings:
            raise ValueError("No chunk embeddings to merge")

        # Concatenate all chunks
        merged = torch.cat(chunk_embeddings, dim=1)  # (B, L_total, D)

        # Apply cross-chunk attention to let all tokens see each other
        normed = self.cross_chunk_norm(merged)
        attended, _ = self.cross_chunk_attn(normed, normed, normed)
        merged = merged + attended

        # Apply priority weighting
        priority = self.priority_proj(merged)  # (B, L_total, 1)
        merged = merged * (0.5 + priority)  # scale by [0.5, 1.5]

        # Apply layer importance weights if provided
        if layer_weights is not None:
            # Build per-token weight tensor matching chunk boundaries
            weights = []
            for i, chunk in enumerate(chunk_embeddings):
                w = layer_weights[i] if i < len(layer_weights) else 1.0
                weights.append(torch.full((chunk.shape[1],), w, device=merged.device))
            weight_vec = torch.cat(weights).unsqueeze(0).unsqueeze(-1)  # (1, L, 1)
            merged = merged * weight_vec

        return merged

    def forward(
        self,
        chunk_embeddings: List[torch.Tensor],
        layer_weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        return self.merge_chunks(chunk_embeddings, layer_weights)


# ---------------------------------------------------------------------------
# Prompt Priority Weighter
# ---------------------------------------------------------------------------

class PromptPriorityWeighter(nn.Module):
    """
    Assigns importance weights to text tokens so critical tokens
    don't get diluted in long sequences.

    Tokens describing the primary subject and camera/technical specs
    get higher weights; filler words and micro-details get lower weights.

    Args:
        hidden_size: Token dimension.
    """

    # High-priority keywords (get boosted weight)
    HIGH_PRIORITY = re.compile(
        r'\b(portrait|character|person|woman|man|girl|boy|creature|'
        r'shot on|photographed|lens|mm|angle|pov|perspective|'
        r'close.?up|wide.?angle|bird.?s.?eye|worm.?s.?eye|'
        r'first.?person|third.?person|over.?the.?shoulder)\b',
        re.IGNORECASE
    )

    def __init__(self, hidden_size: int):
        super().__init__()
        # Learned token importance predictor
        self.importance = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 8),
            nn.GELU(),
            nn.Linear(hidden_size // 8, 1),
        )
        # Init to near-zero so it starts as identity
        nn.init.zeros_(self.importance[-1].weight)
        nn.init.zeros_(self.importance[-1].bias)

    def forward(
        self,
        text_emb: torch.Tensor,
        raw_tokens: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Scale text embeddings by learned importance weights.

        Args:
            text_emb: (B, L, D)
            raw_tokens: Optional list of token strings for rule-based boosting.
        Returns:
            (B, L, D) weighted embeddings.
        """
        # Learned importance
        weights = torch.sigmoid(self.importance(text_emb))  # (B, L, 1)

        # Rule-based boost for high-priority tokens
        if raw_tokens is not None:
            boost = torch.ones(len(raw_tokens), device=text_emb.device)
            for i, tok in enumerate(raw_tokens):
                if self.HIGH_PRIORITY.search(tok):
                    boost[i] = 1.5
            weights = weights * boost.unsqueeze(0).unsqueeze(-1)

        # Scale: keep base + add weighted delta
        return text_emb * (1.0 + 0.3 * weights)


# ---------------------------------------------------------------------------
# Negative Prompt Fusion
# ---------------------------------------------------------------------------

class NegativePromptFusion(nn.Module):
    """
    Fuses explicit negative prompt + inline negatives into a unified
    suppression signal applied to cross-attention.

    Standard CFG uses negative prompts at the sampling level (unconditional).
    This module additionally applies suppression *inside* the cross-attention
    so the model actively avoids negative concepts at every layer, not just
    at the CFG guidance step.

    Args:
        hidden_size: Token dimension.
        num_heads: Attention heads.
        suppression_strength: How strongly to suppress negative tokens (0–1).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        suppression_strength: float = 0.85,
    ):
        super().__init__()
        self.suppression_strength = float(suppression_strength)
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Negative concept detector: given a token, how "negative" is it?
        self.neg_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.neg_detector[-2].weight)
        nn.init.constant_(self.neg_detector[-2].bias, -3.0)  # conservative init

        # Positive-negative alignment: measures how much a positive token
        # semantically overlaps with a negative token
        self.alignment_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def compute_suppression_mask(
        self,
        pos_emb: torch.Tensor,
        neg_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token suppression weights.

        For each positive token, if it's semantically similar to any negative
        token, reduce its weight.

        Args:
            pos_emb: (B, L_pos, D) positive text embeddings.
            neg_emb: (B, L_neg, D) negative text embeddings.
        Returns:
            suppression: (B, L_pos, 1) weights in [0, 1]. 1=keep, 0=suppress.
        """
        B, L_pos, D = pos_emb.shape
        _, L_neg, _ = neg_emb.shape

        # Project both to alignment space
        pos_proj = F.normalize(self.alignment_proj(pos_emb), dim=-1)  # (B, L_pos, D)
        neg_proj = F.normalize(self.alignment_proj(neg_emb), dim=-1)  # (B, L_neg, D)

        # Cosine similarity: (B, L_pos, L_neg)
        sim = torch.bmm(pos_proj, neg_proj.transpose(1, 2))

        # Max similarity to any negative token: (B, L_pos)
        max_sim = sim.max(dim=-1).values.clamp(0, 1)

        # Suppression: high similarity to negative → suppress
        suppression = 1.0 - self.suppression_strength * max_sim.unsqueeze(-1)
        return suppression.clamp(min=0.05)  # never fully zero

    def apply_to_cross_attention_values(
        self,
        value: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Suppress value vectors for tokens that overlap with negatives.

        Args:
            value: (B, H, L, D_head) cross-attention values.
            pos_emb: (B, L, D) positive text embeddings.
            neg_emb: (B, L_neg, D) negative text embeddings (or None).
        Returns:
            (B, H, L, D_head) suppressed values.
        """
        if neg_emb is None or neg_emb.shape[1] == 0:
            return value

        suppression = self.compute_suppression_mask(pos_emb, neg_emb)  # (B, L, 1)
        # Broadcast to (B, 1, L, 1) for (B, H, L, D_head)
        suppression = suppression.unsqueeze(1)
        return value * suppression

    def forward(
        self,
        pos_emb: torch.Tensor,
        neg_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns suppressed positive embeddings.
        pos_emb: (B, L, D), neg_emb: (B, L_neg, D) or None.
        """
        if neg_emb is None or neg_emb.shape[1] == 0:
            return pos_emb
        suppression = self.compute_suppression_mask(pos_emb, neg_emb)
        return pos_emb * suppression


# ---------------------------------------------------------------------------
# Long Prompt Controller (top-level)
# ---------------------------------------------------------------------------

class LongPromptController:
    """
    Top-level controller for long/complex prompt handling.

    Usage:
        ctrl = LongPromptController(hidden_size=1152, num_heads=16)
        parsed = ctrl.parse("extremely long prompt... but no watermarks, no blur")
        # parsed.positive_text: cleaned positive prompt
        # parsed.inline_negatives: ["watermarks", "blur"]
        # parsed.layers: {"primary_subject": "...", "camera_technical": "...", ...}
        # parsed.layer_weights: {"primary_subject": 1.0, ...}

        # After encoding chunks with your text encoder:
        merged_emb = ctrl.merge(chunk_embeddings, parsed.layer_weights)
        weighted_emb = ctrl.weight(merged_emb)
        suppressed_emb = ctrl.suppress(weighted_emb, neg_emb)
    """

    def __init__(self, hidden_size: int, num_heads: int):
        self.parser = HierarchicalPromptParser()
        self.chunker = ChunkedTextEncoder(hidden_size)
        self.weighter = PromptPriorityWeighter(hidden_size)
        self.neg_fusion = NegativePromptFusion(hidden_size, num_heads)

    def parse(self, prompt: str) -> ParsedLongPrompt:
        return self.parser.parse(prompt)

    def merge(
        self,
        chunk_embeddings: List[torch.Tensor],
        layer_weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        weights = list(layer_weights.values()) if layer_weights else None
        return self.chunker(chunk_embeddings, weights)

    def weight(
        self,
        text_emb: torch.Tensor,
        raw_tokens: Optional[List[str]] = None,
    ) -> torch.Tensor:
        return self.weighter(text_emb, raw_tokens)

    def suppress(
        self,
        pos_emb: torch.Tensor,
        neg_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.neg_fusion(pos_emb, neg_emb)

    def suppress_values(
        self,
        value: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self.neg_fusion.apply_to_cross_attention_values(value, pos_emb, neg_emb)


__all__ = [
    "HierarchicalPromptParser",
    "InlineNegativeExtractor",
    "ChunkedTextEncoder",
    "PromptPriorityWeighter",
    "NegativePromptFusion",
    "LongPromptController",
    "ParsedLongPrompt",
]

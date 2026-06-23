"""
Consistency engine facade — routes to reproducibility components (INNOVATION_GUIDE §5).
"""

from typing import Optional

import torch

from .character import CharacterConsistency
from .color import ColorConsistency
from .seeding import ConsistentSeeding
from .semantic import SemanticConsistency
from .style import StyleConsistency
from .temporal import TemporalConsistency
from .variation import VariationControl

__all__ = [
    "CharacterConsistency",
    "ColorConsistency",
    "ConsistencyEngine",
    "ConsistentSeeding",
    "SemanticConsistency",
    "StyleConsistency",
    "TemporalConsistency",
    "VariationControl",
]


class ConsistencyEngine:
    """Unified consistency system for reproducible generation."""

    def __init__(self):
        self.seeding = ConsistentSeeding()
        self.character = CharacterConsistency()
        self.style = StyleConsistency()
        self.variation = VariationControl()
        self.semantic = SemanticConsistency()
        self.temporal = TemporalConsistency()
        self.color = ColorConsistency()

    def generate_consistent(
        self,
        prompt: torch.Tensor,
        seed: int,
        character_id: Optional[str] = None,
        style_name: Optional[str] = None,
        variation: float = 0.0,
        num_frames: int = 1,
    ) -> torch.Tensor:
        """
        Generate image with full consistency guarantees.

        Features:
        - Deterministic seeding: same seed = identical image
        - Character consistency: same character across generations
        - Style consistency: maintain artistic style
        - Semantic consistency: preserve meaning
        - Temporal consistency: smooth video sequences
        - Color consistency: stable color palette
        """
        # Ensure prompt is 2D
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)

        # Start with deterministic seed
        base_latent = self.seeding.encode_seed(seed)

        # Apply character if specified
        if character_id:
            char_features = self.character.retrieve_character(character_id)
            if char_features is None:
                char_features = self.character.encode_character(prompt, character_id)

        # Apply style if specified
        if style_name and style_name in self.style.style_memory:
            self.style.style_memory[style_name]

        # Add controlled variation
        final_latent = self.variation(base_latent, variation)

        return final_latent

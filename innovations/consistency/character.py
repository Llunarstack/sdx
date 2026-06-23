"""Character consistency — encode and retrieve character features."""

from typing import Optional

import torch
import torch.nn as nn


class CharacterConsistency(nn.Module):
    """Generate same character with consistent appearance across images."""

    def __init__(self, hidden_dim: int = 512, num_features: int = 128):
        super().__init__()

        # Character encoding: face, body, clothing features
        self.face_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 32),
        )
        self.body_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 32),
        )
        self.clothing_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 32),
        )
        self.accessories_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 32),
        )

        # Store character memory
        self.character_memory = {}

    def encode_character(self, character_description: torch.Tensor, character_id: str) -> torch.Tensor:
        """Encode and store character features."""
        face = self.face_encoder(character_description)
        body = self.body_encoder(character_description)
        clothing = self.clothing_encoder(character_description)
        accessories = self.accessories_encoder(character_description)

        character_features = torch.cat([face, body, clothing, accessories], dim=-1)
        self.character_memory[character_id] = character_features

        return character_features

    def retrieve_character(self, character_id: str) -> Optional[torch.Tensor]:
        """Retrieve stored character features."""
        return self.character_memory.get(character_id)

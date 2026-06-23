"""Semantic decomposition — objects, style, composition, materials, actions, mood."""

from typing import Dict

import torch
import torch.nn as nn


class SemanticDecomposer(nn.Module):
    """Break down prompts into constituent semantic elements."""

    def __init__(self, vocab_size: int = 50000, hidden_dim: int = 768):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)

        # Object detector: what objects are mentioned
        self.object_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )

        # Style extractor: artistic style, mood, lighting
        self.style_extractor = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 128),
        )

        # Composition analyzer: layout, perspective, framing
        self.composition_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 64),
        )

        # Material/texture specifier
        self.material_identifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )

        # Temporal/action intent
        self.action_detector = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 64),
        )

        # Emotional/atmospheric intent
        self.mood_detector = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 64),
        )

    def forward(self, prompt_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decompose prompt into semantic components."""
        embedded = self.embeddings(prompt_tokens)
        pooled = embedded.mean(dim=1)  # Average pooling

        return {
            "objects": self.object_classifier(pooled),
            "style": self.style_extractor(pooled),
            "composition": self.composition_analyzer(pooled),
            "materials": self.material_identifier(pooled),
            "actions": self.action_detector(pooled),
            "mood": self.mood_detector(pooled),
        }

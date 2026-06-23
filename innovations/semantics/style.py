"""Style transfer understanding — classify, intensity, and blend artistic styles."""

from typing import Dict

import torch
import torch.nn as nn


class StyleTransferUnderstanding(nn.Module):
    """Understand and apply specific artistic styles with precision."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.styles = [
            "photorealism",
            "oil_painting",
            "watercolor",
            "pencil_sketch",
            "digital_art",
            "anime",
            "comic",
            "abstract",
            "surreal",
            "cyberpunk",
        ]

        self.style_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, len(self.styles)),
        )

        # Style intensity (how much to apply)
        self.style_intensity = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Style blending (mix multiple styles)
        self.style_mixer = nn.Linear(len(self.styles), len(self.styles))

    def forward(self, semantic_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Parse and understand artistic style requirements."""
        style_logits = self.style_classifier(semantic_features)
        style_probs = torch.softmax(style_logits, dim=-1)

        intensity = self.style_intensity(semantic_features)

        # Mix styles if multiple detected
        blended = self.style_mixer(style_probs)
        blended = torch.softmax(blended, dim=-1)

        return {
            "primary_style": style_probs,
            "style_intensity": intensity,
            "blended_styles": blended,
            "style_names": self.styles,
        }

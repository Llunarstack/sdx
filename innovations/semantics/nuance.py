"""Nuance capture — scale, spatial layout, quantity, temporal, environment, depth."""

from typing import Dict

import torch
import torch.nn as nn


class NuanceCapture(nn.Module):
    """Capture subtle semantic nuances that most models miss."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()

        # Relative size relationships (big vs small, large vs tiny)
        self.relative_scale = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 32),
        )

        # Spatial relationships (above, below, left, right, center, scattered)
        self.spatial_relationships = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 64),
        )

        # Quantity indicators (one, few, many, countless)
        self.quantity_descriptor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 16),
        )

        # Temporal modifiers (sunrise, midday, sunset, midnight)
        self.temporal_descriptor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 32),
        )

        # Weather/environmental conditions
        self.environment_descriptor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 48),
        )

        # Depth of field and focus (sharp, blurred, shallow, deep)
        self.depth_descriptor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 16),
        )

    def forward(self, semantic_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract nuanced details from semantic features."""
        # Get first feature as base for processing
        first_feature = next(iter(semantic_features.values()))
        combined = first_feature  # Use single feature instead of concatenating

        return {
            "scale_relationships": self.relative_scale(combined),
            "spatial": self.spatial_relationships(combined),
            "quantities": self.quantity_descriptor(combined),
            "temporal": self.temporal_descriptor(combined),
            "environment": self.environment_descriptor(combined),
            "depth_of_field": self.depth_descriptor(combined),
        }

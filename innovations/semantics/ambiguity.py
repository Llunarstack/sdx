"""Contextual ambiguity resolution — references, metaphor, implied context."""

from typing import Dict

import torch
import torch.nn as nn


class ContextualAmbiguityResolver(nn.Module):
    """Resolve ambiguous prompts through contextual reasoning."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()

        # Reference resolution (pronoun disambiguation)
        self.reference_resolver = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, hidden_dim),
        )

        # Metaphor/simile interpreter
        self.metaphor_interpreter = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )

        # Implied context generator
        self.context_generator = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, hidden_dim),
        )

    def forward(
        self,
        prompt_embedding: torch.Tensor,
        previous_context: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """Resolve ambiguities and infer missing context."""
        if previous_context is None:
            previous_context = torch.zeros_like(prompt_embedding)

        resolved = self.reference_resolver(torch.cat([prompt_embedding, previous_context], dim=-1))
        metaphor = self.metaphor_interpreter(prompt_embedding)
        context = self.context_generator(prompt_embedding)

        return {
            "resolved_references": resolved,
            "metaphor_interpretation": metaphor,
            "implied_context": context,
        }

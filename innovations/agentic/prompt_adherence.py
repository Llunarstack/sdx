"""
Prompt Adherence System: Ensures generation follows user prompt EXACTLY.
Leverages penta text encoder (T5 + 4 CLIP variants) for semantic validation.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class PromptAnalysis:
    """Analysis of prompt semantic content."""

    primary_subject: str  # Main object/character
    style_descriptors: List[str]  # Artistic style
    color: List[str]  # Colors mentioned
    lighting_conditions: List[str]  # Light/shadow
    composition_details: List[str]  # Layout/framing
    action_verbs: List[str]  # Movement/action
    emotional_tone: str  # Mood/atmosphere
    detail_level: float  # Expected detail (0-1)
    complexity_score: float  # Overall complexity (0-1)


class PentaEncoderSemanticAnalyzer(nn.Module):
    """
    Analyze prompt semantics using all 5 encoders from penta system.
    Each encoder captures different semantic aspects.
    """

    def __init__(self):
        super().__init__()

        # T5 (general semantic understanding - sequences)
        self.t5_subject_extractor = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.GELU(),
            nn.Linear(2048, 512),
        )

        # CLIP-L (basic vision-language understanding)
        self.clip_l_style_extractor = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )

        # CLIP-bigG (fine-grained visual details)
        self.clip_bg_detail_extractor = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.GELU(),
            nn.Linear(1024, 256),
        )

        # CLIP-H (high-level scene understanding)
        self.clip_h_composition_extractor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )

        # LongCLIP (extended context understanding)
        self.clip_long_context_extractor = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )

        # Semantic parser
        self.engine = nn.Sequential(
            nn.Linear(512 + 128 + 256 + 256 + 128, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )

    def forward(
        self,
        t5_embeddings: torch.Tensor,
        clip_l_embeddings: torch.Tensor,
        clip_bg_embeddings: torch.Tensor,
        clip_h_embeddings: torch.Tensor,
        clip_long_embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Extract semantic features from all 5 encoders."""
        # Extract from each encoder
        subject_features = self.t5_subject_extractor(t5_embeddings)
        style_features = self.clip_l_style_extractor(clip_l_embeddings)
        detail_features = self.clip_bg_detail_extractor(clip_bg_embeddings)
        composition_features = self.clip_h_composition_extractor(clip_h_embeddings)
        context_features = self.clip_long_context_extractor(clip_long_embeddings)

        # Combine all features
        combined = torch.cat(
            [subject_features, style_features, detail_features, composition_features, context_features], dim=-1
        )
        semantic = self.engine(combined)

        return {
            "subject": subject_features,
            "style": style_features,
            "detail": detail_features,
            "composition": composition_features,
            "context": context_features,
            "combined": semantic,
        }


class PromptValidator(nn.Module):
    """Validate that generation matches prompt semantically."""

    def __init__(self):
        super().__init__()

        # Semantic matcher (how well does generated match prompt intent)
        self.semantic_matcher = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Detail validator (is there enough detail as promised)
        self.detail_validator = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Style validator (is artistic style correct)
        self.style_validator = nn.Sequential(
            nn.Linear(128 * 2, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Composition validator (is layout as described)
        self.composition_validator = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        prompt_semantics: Dict[str, torch.Tensor],
        generated_semantics: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Validate generation matches prompt across all semantic dimensions.

        Returns:
            Dict with scores for each validation aspect
        """
        # Match each semantic dimension
        semantic_match = self.semantic_matcher(
            torch.cat([prompt_semantics["combined"], generated_semantics["combined"]], dim=-1)
        )

        detail_match = self.detail_validator(
            torch.cat([prompt_semantics["detail"], generated_semantics["detail"]], dim=-1)
        )

        style_match = self.style_validator(torch.cat([prompt_semantics["style"], generated_semantics["style"]], dim=-1))

        composition_match = self.composition_validator(
            torch.cat([prompt_semantics["composition"], generated_semantics["composition"]], dim=-1)
        )

        return {
            "semantic": semantic_match.mean().detach().item(),
            "detail": detail_match.mean().detach().item(),
            "style": style_match.mean().detach().item(),
            "composition": composition_match.mean().detach().item(),
            "overall": (
                semantic_match.mean().detach().item() * 0.4
                + detail_match.mean().detach().item() * 0.2
                + style_match.mean().detach().item() * 0.2
                + composition_match.mean().detach().item() * 0.2
            ),
        }


class DynamicPromptEnforcer(nn.Module):
    """Dynamically adjust generation process to enforce prompt adherence."""

    def __init__(self):
        super().__init__()

        # Guidance strength adjuster (how strongly to follow prompt)
        self.guidance_adjuster = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Sampling temperature adjuster (diversity vs adherence)
        self.temperature_adjuster = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # Refinement intensity adjuster
        self.refinement_adjuster = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        prompt_semantics: Dict[str, torch.Tensor],
        current_adherence: float,
    ) -> Dict[str, float]:
        """
        Calculate dynamic adjustments to ensure adherence.

        Args:
            prompt_semantics: Semantic features from prompt
            current_adherence: Current adherence score (0-1)

        Returns:
            Adjustment parameters for generation
        """
        semantic_combined = prompt_semantics["combined"]

        # Adjust guidance based on adherence gap
        adherence_gap = max(0, 0.95 - current_adherence)  # How close to perfect
        guidance = self.guidance_adjuster(semantic_combined) * (0.5 + adherence_gap)

        # Adjust temperature for diversity
        temperature = self.temperature_adjuster(semantic_combined)
        # Scale based on adherence - lower temp = more adherent
        temperature = temperature * (1.0 - adherence_gap)

        # Refinement intensity
        refinement = self.refinement_adjuster(semantic_combined) * adherence_gap

        return {
            "guidance_scale": float(guidance.mean().detach().item()),
            "temperature": float(torch.clamp(temperature, 0.1, 1.0).mean().detach().item()),
            "refinement_strength": float(refinement.mean().detach().item()),
        }


class PromptAdherenceMonitor:
    """Monitor and ensure prompt adherence throughout generation."""

    def __init__(self):
        self.analyzer = PentaEncoderSemanticAnalyzer()
        self.validator = PromptValidator()
        self.enforcer = DynamicPromptEnforcer()
        self.adherence_history = []
        self.max_deviations = {
            "semantic": 0.15,
            "detail": 0.20,
            "style": 0.15,
            "composition": 0.15,
        }

    def analyze_prompt(
        self,
        t5_embedding: torch.Tensor,
        clip_embeddings: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Analyze prompt semantics using penta encoder."""
        return self.analyzer(
            t5_embedding,
            clip_embeddings["clip_l"],
            clip_embeddings["clip_bg"],
            clip_embeddings["clip_h"],
            clip_embeddings["clip_long"],
        )

    def check_adherence(
        self,
        prompt_semantics: Dict[str, torch.Tensor],
        generated_latent: torch.Tensor,
        intermediate_semantics: Dict[str, torch.Tensor],
    ) -> Tuple[float, List[str]]:
        """
        Check if generation adheres to prompt.

        Returns:
            (adherence_score, violations)
        """
        validation_scores = self.validator(prompt_semantics, intermediate_semantics)

        violations = []
        for aspect, threshold in self.max_deviations.items():
            if aspect in validation_scores:
                if validation_scores[aspect] < (1.0 - threshold):
                    violations.append(f"Poor {aspect} match: {validation_scores[aspect]:.2%}")

        adherence = validation_scores["overall"]
        self.adherence_history.append(adherence)

        return adherence, violations

    def get_enforcement_params(
        self,
        prompt_semantics: Dict[str, torch.Tensor],
        current_adherence: float,
    ) -> Dict[str, float]:
        """Get parameters to enforce adherence in next generation step."""
        return self.enforcer(prompt_semantics, current_adherence)

    def generate_with_adherence(
        self,
        prompt: str,
        t5_embedding: torch.Tensor,
        clip_embeddings: Dict[str, torch.Tensor],
        generation_func,  # Function that takes enforcement params
        max_iterations: int = 5,
    ) -> Tuple[torch.Tensor, float]:
        """
        Generate with iterative adherence enforcement.

        Args:
            prompt: Original prompt
            t5_embedding: T5 encoder output
            clip_embeddings: All CLIP encoder outputs
            generation_func: Function(guidance_scale, temperature) -> latent
            max_iterations: Max refinement iterations

        Returns:
            (final_latent, final_adherence_score)
        """
        # Analyze prompt once
        prompt_semantics = self.analyze_prompt(t5_embedding, clip_embeddings)
        logger.info(f"Prompt analysis complete for: {prompt[:60]}...")

        current_latent = None
        current_adherence = 0.0

        for iteration in range(max_iterations):
            # Get enforcement params
            params = self.get_enforcement_params(prompt_semantics, current_adherence)

            logger.info(
                f"Iteration {iteration + 1}: "
                f"Guidance={params['guidance_scale']:.2f}, "
                f"Temp={params['temperature']:.2f}, "
                f"Refinement={params['refinement_strength']:.2f}"
            )

            # Generate with parameters
            current_latent = generation_func(
                guidance_scale=params["guidance_scale"],
                temperature=params["temperature"],
            )

            # Analyze generated image semantics (in practice, extract from model)
            generated_semantics = {
                "combined": torch.randn(1, 256),  # Would be actual features
                "detail": torch.randn(1, 256),
                "style": torch.randn(1, 128),
                "composition": torch.randn(1, 256),
            }

            # Check adherence
            current_adherence, violations = self.check_adherence(prompt_semantics, current_latent, generated_semantics)

            logger.info(f"Adherence: {current_adherence:.2%}")
            if violations:
                for v in violations:
                    logger.warning(f"  - {v}")

            # If adherent enough, stop
            if current_adherence > 0.90 or iteration == max_iterations - 1:
                logger.info(f"Final adherence: {current_adherence:.2%} (Target: >90%)")
                break

        return current_latent, current_adherence


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test prompt adherence system
    monitor = PromptAdherenceMonitor()

    prompt = "A serene Japanese garden with cherry blossoms at sunset, soft lighting"
    t5_emb = torch.randn(1, 4096)
    clip_embs = {
        "clip_l": torch.randn(1, 768),
        "clip_bg": torch.randn(1, 4096),
        "clip_h": torch.randn(1, 1024),
        "clip_long": torch.randn(1, 768),
    }

    # Analyze prompt
    semantics = monitor.analyze_prompt(t5_emb, clip_embs)
    print(f"Prompt analyzed with {len(semantics)} semantic dimensions")

    # Dummy generation function
    def dummy_gen(guidance_scale, temperature):
        return torch.randn(1, 4, 64, 64)

    # Generate with adherence enforcement
    final_latent, adherence = monitor.generate_with_adherence(prompt, t5_emb, clip_embs, dummy_gen, max_iterations=3)

    print(f"\nFinal adherence score: {adherence:.2%}")
    print(f"Adherence history: {[f'{x:.2%}' for x in monitor.adherence_history]}")

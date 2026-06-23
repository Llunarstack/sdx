"""
Prompt Optimization Agent: Automatically improves user prompts for better generation.
Expands vague descriptions, adds technical details, and optimizes for model understanding.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class PromptAnalysis:
    """Analysis of a prompt."""

    original_prompt: str
    length: int
    coverage_score: float  # 0-1 (how much detail provided)
    vagueness_score: float  # 0-1 (how vague it is)
    specificity_score: float  # 0-1 (how specific)
    technical_depth: float  # 0-1 (technical terminology)
    clarity_score: float  # 0-1 (how clear)
    missing_aspects: List[str]  # What's missing
    suggested_keywords: List[str]  # Keywords to add


class PromptAnalyzer(nn.Module):
    """Analyzes prompts to identify gaps and opportunities."""

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        # Prompt feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )

        # Coverage scorer (how much detail)
        self.coverage_scorer = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Vagueness detector
        self.vagueness_detector = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Specificity scorer
        self.specificity_scorer = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Technical depth analyzer
        self.technical_analyzer = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Missing aspect detector (10 common aspects)
        self.aspect_detector = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 10),  # lighting, color, texture, mood, composition, etc.
        )

        self.aspects = [
            "lighting",
            "color",
            "texture",
            "mood",
            "composition",
            "camera_angle",
            "scale",
            "detail_level",
            "environment",
            "atmosphere",
        ]

    def forward(self, embedding: torch.Tensor, original_text: str = "") -> PromptAnalysis:
        """Analyze prompt comprehensively."""
        features = self.feature_extractor(embedding)

        # Score aspects
        coverage = float(self.coverage_scorer(features).squeeze().detach())
        vagueness = float(self.vagueness_detector(features).squeeze().detach())
        specificity = float(self.specificity_scorer(features).squeeze().detach())
        technical = float(self.technical_analyzer(features).squeeze().detach())
        clarity = (specificity + (1 - vagueness)) / 2

        # Detect missing aspects
        aspect_logits = self.aspect_detector(features)
        aspect_scores = torch.sigmoid(aspect_logits[0]).detach()
        missing = []
        for idx, score in enumerate(aspect_scores):
            if float(score) < 0.4:  # Low score = missing
                missing.append(self.aspects[idx])

        # Suggest keywords based on gaps
        keywords = []
        if vagueness > 0.6:
            keywords.extend(["detailed", "intricate", "elaborate"])
        if coverage < 0.5:
            keywords.extend(["composition", "arrangement", "setting"])
        if float(aspect_logits[0][0].detach()) < 0.3:  # lighting
            keywords.extend(["golden hour", "cinematic lighting"])

        return PromptAnalysis(
            original_prompt=original_text,
            length=len(original_text.split()),
            coverage_score=coverage,
            vagueness_score=vagueness,
            specificity_score=specificity,
            technical_depth=technical,
            clarity_score=clarity,
            missing_aspects=missing,
            suggested_keywords=keywords,
        )


class PromptEnhancer(nn.Module):
    """Enhances prompts with better technical vocabulary."""

    def __init__(self):
        super().__init__()

        # Enhancement suggestion generator (input is 4096-dim from feature extractor)
        self.feature_reducer = nn.Sequential(
            nn.Linear(4096, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )

        self.subject_enhancer = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )

        self.style_enhancer = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )

        self.technical_enhancer = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )

        # Enhancement vocabulary
        self.style_adjectives = [
            "photorealistic",
            "cinematic",
            "artistic",
            "hyperdetailed",
            "professional",
            "masterpiece",
            "award-winning",
            "stunning",
            "breathtaking",
        ]

        self.technical_terms = [
            "physically-based rendering",
            "ray tracing",
            "volumetric lighting",
            "subsurface scattering",
            "high contrast",
            "sharp focus",
            "bokeh",
            "depth of field",
            "color graded",
        ]

        self.quality_boosters = [
            "8k",
            "4k",
            "ultra high definition",
            "intricate details",
            "high quality",
            "professional quality",
            "museum quality",
        ]

    def forward(
        self,
        prompt_features: torch.Tensor,
        analysis: PromptAnalysis,
    ) -> str:
        """Generate enhanced prompt."""
        # Reduce dimensionality
        reduced_features = self.feature_reducer(prompt_features)

        # Generate enhancements
        self.subject_enhancer(reduced_features)
        self.style_enhancer(reduced_features)
        self.technical_enhancer(reduced_features)

        enhancements = []

        # Add quality boosters if not highly detailed
        if analysis.coverage_score < 0.7:
            enhancements.append(self.quality_boosters[0])

        # Add style suggestions
        if analysis.specificity_score < 0.6:
            enhancements.append(self.style_adjectives[0])

        # Add technical terms
        if analysis.technical_depth < 0.5:
            enhancements.append(self.technical_terms[0])

        # Construct enhanced prompt
        enhanced = analysis.original_prompt
        if enhancements:
            enhanced += ", " + ", ".join(enhancements)

        return enhanced


class PromptExpander(nn.Module):
    """Expands concise prompts with detail."""

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        # Context generator
        self.context_generator = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )

        # Detail synthesizer
        self.detail_synthesizer = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )

        # Context phrases for different scenarios
        self.context_phrases = {
            "lighting": [
                "bathed in golden sunlight",
                "lit by dramatic studio lighting",
                "softly illuminated",
                "brightly lit",
                "backlit",
            ],
            "environment": [
                "in a lush forest",
                "in an urban setting",
                "on a pristine beach",
                "in a cozy interior",
                "against a mountain backdrop",
            ],
            "style": [
                "rendered in photorealistic style",
                "painted in impressionist style",
                "in the style of classical art",
                "with digital art aesthetics",
                "with watercolor textures",
            ],
            "mood": [
                "conveying a sense of serenity",
                "with dramatic atmosphere",
                "cheerful and vibrant",
                "mysterious and moody",
                "peaceful and tranquil",
            ],
        }

    def forward(
        self,
        prompt_embedding: torch.Tensor,
        analysis: PromptAnalysis,
    ) -> str:
        """Expand prompt with contextual details."""
        self.context_generator(prompt_embedding)

        expanded = analysis.original_prompt

        # Add details for missing aspects
        additions = []

        if "lighting" in analysis.missing_aspects:
            additions.append(self.context_phrases["lighting"][0])

        if "environment" in analysis.missing_aspects:
            additions.append(self.context_phrases["environment"][0])

        if "mood" in analysis.missing_aspects:
            additions.append(self.context_phrases["mood"][0])

        if additions:
            expanded += " " + ", ".join(additions)

        return expanded


class PromptOptimizationSystem:
    """Unified prompt optimization system."""

    def __init__(self, hidden_dim: int = 4096):
        self.analyzer = PromptAnalyzer(hidden_dim)
        self.enhancer = PromptEnhancer()
        self.expander = PromptExpander(hidden_dim)

        self.optimization_history = []

    def analyze_prompt(
        self,
        prompt: str,
        prompt_embedding: torch.Tensor,
    ) -> PromptAnalysis:
        """Analyze a prompt."""
        return self.analyzer(prompt_embedding, prompt)

    def enhance_prompt(
        self,
        prompt: str,
        prompt_embedding: torch.Tensor,
    ) -> str:
        """Enhance prompt with better vocabulary."""
        analysis = self.analyze_prompt(prompt, prompt_embedding)
        return self.enhancer(prompt_embedding, analysis)

    def expand_prompt(
        self,
        prompt: str,
        prompt_embedding: torch.Tensor,
    ) -> str:
        """Expand vague prompt with details."""
        analysis = self.analyze_prompt(prompt, prompt_embedding)
        return self.expander(prompt_embedding, analysis)

    def optimize_prompt(
        self,
        prompt: str,
        prompt_embedding: torch.Tensor,
        target_coverage: float = 0.8,
        target_specificity: float = 0.8,
    ) -> Dict:
        """Fully optimize prompt."""
        analysis = self.analyze_prompt(prompt, prompt_embedding)

        optimized = prompt

        # Expand if not detailed enough
        if analysis.coverage_score < target_coverage:
            optimized = self.expander(prompt_embedding, analysis)

        # Enhance with technical terms
        if analysis.technical_depth < 0.6:
            optimized = self.enhancer(prompt_embedding, analysis)

        # Analyze result
        reanalyzed = self.analyzer(prompt_embedding, optimized)

        optimization_record = {
            "original": prompt,
            "optimized": optimized,
            "original_analysis": {
                "coverage": analysis.coverage_score,
                "vagueness": analysis.vagueness_score,
                "specificity": analysis.specificity_score,
                "technical_depth": analysis.technical_depth,
                "clarity": analysis.clarity_score,
            },
            "optimized_analysis": {
                "coverage": reanalyzed.coverage_score,
                "vagueness": reanalyzed.vagueness_score,
                "specificity": reanalyzed.specificity_score,
                "technical_depth": reanalyzed.technical_depth,
                "clarity": reanalyzed.clarity_score,
            },
            "improvements": {
                "coverage_gain": reanalyzed.coverage_score - analysis.coverage_score,
                "vagueness_reduction": analysis.vagueness_score - reanalyzed.vagueness_score,
                "specificity_gain": reanalyzed.specificity_score - analysis.specificity_score,
                "technical_gain": reanalyzed.technical_depth - analysis.technical_depth,
            },
        }

        self.optimization_history.append(optimization_record)
        return optimization_record

    def get_optimization_stats(self) -> Dict:
        """Get statistics on optimizations performed."""
        if not self.optimization_history:
            return {
                "total_optimizations": 0,
                "average_improvements": {},
            }

        total = len(self.optimization_history)
        avg_coverage_gain = sum(r["improvements"]["coverage_gain"] for r in self.optimization_history) / total
        avg_vagueness_reduction = (
            sum(r["improvements"]["vagueness_reduction"] for r in self.optimization_history) / total
        )
        avg_specificity_gain = sum(r["improvements"]["specificity_gain"] for r in self.optimization_history) / total
        avg_technical_gain = sum(r["improvements"]["technical_gain"] for r in self.optimization_history) / total

        return {
            "total_optimizations": total,
            "average_improvements": {
                "coverage": avg_coverage_gain,
                "vagueness_reduction": avg_vagueness_reduction,
                "specificity": avg_specificity_gain,
                "technical_depth": avg_technical_gain,
            },
        }

    def recommend_optimizations(self, analysis: PromptAnalysis) -> List[str]:
        """Recommend what optimizations to apply."""
        recommendations = []

        if analysis.coverage_score < 0.6:
            recommendations.append("Expand with more contextual details")

        if analysis.vagueness_score > 0.6:
            recommendations.append("Add more specific descriptors")

        if analysis.technical_depth < 0.5:
            recommendations.append("Include technical rendering terms")

        if "lighting" in analysis.missing_aspects:
            recommendations.append("Specify lighting conditions")

        if "mood" in analysis.missing_aspects:
            recommendations.append("Add emotional/atmospheric descriptions")

        return recommendations


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    system = PromptOptimizationSystem()

    # Test with a simple prompt
    simple_prompt = "a dog"
    embedding = torch.randn(1, 4096)

    print("=== Prompt Optimization Demo ===\n")
    print(f"Original: {simple_prompt}")

    analysis = system.analyze_prompt(simple_prompt, embedding)
    print("\nAnalysis:")
    print(f"  Coverage: {analysis.coverage_score:.2%}")
    print(f"  Vagueness: {analysis.vagueness_score:.2%}")
    print(f"  Specificity: {analysis.specificity_score:.2%}")
    print(f"  Missing: {', '.join(analysis.missing_aspects)}")

    enhanced = system.enhance_prompt(simple_prompt, embedding)
    print(f"\nEnhanced: {enhanced}")

    expanded = system.expand_prompt(simple_prompt, embedding)
    print(f"\nExpanded: {expanded}")

    optimization = system.optimize_prompt(simple_prompt, embedding)
    print(f"\nFully Optimized: {optimization['optimized']}")
    print(f"Coverage improvement: {optimization['improvements']['coverage_gain']:+.2%}")

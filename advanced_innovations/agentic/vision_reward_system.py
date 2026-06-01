"""
VisionReward System: Fine-grained multi-dimensional human preference learning.
Based on research: https://arxiv.org/abs/2412.21059

Learns detailed human preferences across multiple visual dimensions:
- Aesthetic quality (composition, color, lighting)
- Detail richness (texture, sharpness, complexity)
- Semantic alignment (prompt matching accuracy)
- Technical excellence (artifacts, distortion, noise)
- Emotional impact (mood, atmosphere, engagement)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class PreferenceDimension:
    """A single preference dimension score."""
    name: str
    score: float  # 0-1
    confidence: float  # 0-1 how sure we are
    reasoning: str


@dataclass
class MultiDimensionalReward:
    """Complete multi-dimensional reward assessment."""
    aesthetic_quality: PreferenceDimension
    detail_richness: PreferenceDimension
    semantic_alignment: PreferenceDimension
    technical_excellence: PreferenceDimension
    emotional_impact: PreferenceDimension
    overall_score: float
    user_preference_strength: float  # How strong the user's preference signal is


class AestheticQualityModule(nn.Module):
    """Evaluates composition, color harmony, and lighting."""

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        self.composition_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.color_harmony_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.lighting_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, image_features: torch.Tensor) -> PreferenceDimension:
        """Score aesthetic quality across three sub-dimensions."""
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)

        composition = float(self.composition_scorer(image_features).squeeze())
        color = float(self.color_harmony_scorer(image_features).squeeze())
        lighting = float(self.lighting_scorer(image_features).squeeze())
        confidence = float(self.confidence_net(image_features).squeeze())

        # Weighted combination
        score = (composition * 0.4 + color * 0.35 + lighting * 0.25)

        reasoning = (
            f"Composition: {composition:.0%}, "
            f"Color harmony: {color:.0%}, "
            f"Lighting: {lighting:.0%}"
        )

        return PreferenceDimension(
            name="Aesthetic Quality",
            score=score,
            confidence=confidence,
            reasoning=reasoning,
        )


class DetailRichnessModule(nn.Module):
    """Evaluates texture detail, sharpness, and complexity."""

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        self.texture_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.sharpness_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.complexity_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, image_features: torch.Tensor) -> PreferenceDimension:
        """Score detail richness."""
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)

        texture = float(self.texture_scorer(image_features).squeeze())
        sharpness = float(self.sharpness_scorer(image_features).squeeze())
        complexity = float(self.complexity_scorer(image_features).squeeze())
        confidence = float(self.confidence_net(image_features).squeeze())

        score = (texture * 0.4 + sharpness * 0.35 + complexity * 0.25)

        reasoning = (
            f"Texture detail: {texture:.0%}, "
            f"Sharpness: {sharpness:.0%}, "
            f"Complexity: {complexity:.0%}"
        )

        return PreferenceDimension(
            name="Detail Richness",
            score=score,
            confidence=confidence,
            reasoning=reasoning,
        )


class SemanticAlignmentModule(nn.Module):
    """Evaluates prompt matching accuracy."""

    def __init__(self):
        super().__init__()

        self.subject_alignment = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.style_alignment = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.context_alignment = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.confidence_net = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        prompt_features: torch.Tensor,
        image_features: torch.Tensor,
    ) -> PreferenceDimension:
        """Score semantic alignment between prompt and image."""
        if prompt_features.dim() == 1:
            prompt_features = prompt_features.unsqueeze(0)
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)

        # Take first 512 dims
        pf = prompt_features[:, :512]
        imf = image_features[:, :512]

        subject = float(self.subject_alignment(pf).squeeze())
        style = float(self.style_alignment(imf).squeeze())
        context = float(self.context_alignment(pf).squeeze())
        confidence = float(self.confidence_net(pf).squeeze())

        score = (subject * 0.4 + style * 0.35 + context * 0.25)

        reasoning = (
            f"Subject match: {subject:.0%}, "
            f"Style match: {style:.0%}, "
            f"Context: {context:.0%}"
        )

        return PreferenceDimension(
            name="Semantic Alignment",
            score=score,
            confidence=confidence,
            reasoning=reasoning,
        )


class TechnicalExcellenceModule(nn.Module):
    """Evaluates artifacts, distortion, and noise."""

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        self.artifact_detector = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.distortion_detector = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.noise_detector = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, image_features: torch.Tensor) -> PreferenceDimension:
        """Score technical excellence (lower is better for artifacts)."""
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)

        artifacts = 1.0 - float(self.artifact_detector(image_features).squeeze())
        distortion = 1.0 - float(self.distortion_detector(image_features).squeeze())
        noise = 1.0 - float(self.noise_detector(image_features).squeeze())
        confidence = float(self.confidence_net(image_features).squeeze())

        score = (artifacts * 0.4 + distortion * 0.35 + noise * 0.25)

        reasoning = (
            f"Artifact-free: {artifacts:.0%}, "
            f"Distortion-free: {distortion:.0%}, "
            f"Noise-free: {noise:.0%}"
        )

        return PreferenceDimension(
            name="Technical Excellence",
            score=score,
            confidence=confidence,
            reasoning=reasoning,
        )


class EmotionalImpactModule(nn.Module):
    """Evaluates mood, atmosphere, and engagement."""

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        self.mood_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.atmosphere_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.engagement_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, image_features: torch.Tensor) -> PreferenceDimension:
        """Score emotional impact."""
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)

        mood = float(self.mood_scorer(image_features).squeeze())
        atmosphere = float(self.atmosphere_scorer(image_features).squeeze())
        engagement = float(self.engagement_scorer(image_features).squeeze())
        confidence = float(self.confidence_net(image_features).squeeze())

        score = (mood * 0.4 + atmosphere * 0.35 + engagement * 0.25)

        reasoning = (
            f"Mood strength: {mood:.0%}, "
            f"Atmosphere: {atmosphere:.0%}, "
            f"Engagement: {engagement:.0%}"
        )

        return PreferenceDimension(
            name="Emotional Impact",
            score=score,
            confidence=confidence,
            reasoning=reasoning,
        )


class VisionRewardSystem:
    """Complete vision reward system with multi-dimensional preference learning."""

    def __init__(self, hidden_dim: int = 4096):
        self.aesthetic_module = AestheticQualityModule(hidden_dim)
        self.detail_module = DetailRichnessModule(hidden_dim)
        self.semantic_module = SemanticAlignmentModule()
        self.technical_module = TechnicalExcellenceModule(hidden_dim)
        self.emotional_module = EmotionalImpactModule(hidden_dim)

        self.reward_history = []

    def evaluate_image(
        self,
        image_features: torch.Tensor,
        prompt_features: Optional[torch.Tensor] = None,
        user_rating: Optional[float] = None,
    ) -> MultiDimensionalReward:
        """Evaluate image across all preference dimensions."""
        # Evaluate each dimension
        aesthetic = self.aesthetic_module(image_features)
        detail = self.detail_module(image_features)
        technical = self.technical_module(image_features)
        emotional = self.emotional_module(image_features)

        # Semantic alignment needs prompt
        if prompt_features is not None:
            semantic = self.semantic_module(prompt_features, image_features)
        else:
            semantic = PreferenceDimension(
                name="Semantic Alignment",
                score=0.5,
                confidence=0.0,
                reasoning="No prompt provided",
            )

        # Calculate overall score
        overall = (
            aesthetic.score * 0.25 +
            detail.score * 0.2 +
            semantic.score * 0.25 +
            technical.score * 0.15 +
            emotional.score * 0.15
        )

        # Preference strength based on user rating
        pref_strength = (user_rating / 5.0) if user_rating else 0.5

        reward = MultiDimensionalReward(
            aesthetic_quality=aesthetic,
            detail_richness=detail,
            semantic_alignment=semantic,
            technical_excellence=technical,
            emotional_impact=emotional,
            overall_score=overall,
            user_preference_strength=pref_strength,
        )

        self.reward_history.append(reward)
        return reward

    def get_improvement_suggestions(self, reward: MultiDimensionalReward) -> List[str]:
        """Suggest which dimensions need improvement."""
        suggestions = []

        if reward.aesthetic_quality.score < 0.7:
            suggestions.append("Improve composition and color harmony")

        if reward.detail_richness.score < 0.7:
            suggestions.append("Add more texture detail and sharpness")

        if reward.semantic_alignment.score < 0.7:
            suggestions.append("Better align with prompt intent")

        if reward.technical_excellence.score < 0.7:
            suggestions.append("Reduce artifacts and distortion")

        if reward.emotional_impact.score < 0.7:
            suggestions.append("Strengthen mood and atmosphere")

        return suggestions

    def get_detailed_report(self, reward: MultiDimensionalReward) -> Dict:
        """Generate detailed preference report."""
        return {
            "overall_score": reward.overall_score,
            "preference_strength": reward.user_preference_strength,
            "dimensions": {
                "aesthetic_quality": {
                    "score": reward.aesthetic_quality.score,
                    "confidence": reward.aesthetic_quality.confidence,
                    "reasoning": reward.aesthetic_quality.reasoning,
                },
                "detail_richness": {
                    "score": reward.detail_richness.score,
                    "confidence": reward.detail_richness.confidence,
                    "reasoning": reward.detail_richness.reasoning,
                },
                "semantic_alignment": {
                    "score": reward.semantic_alignment.score,
                    "confidence": reward.semantic_alignment.confidence,
                    "reasoning": reward.semantic_alignment.reasoning,
                },
                "technical_excellence": {
                    "score": reward.technical_excellence.score,
                    "confidence": reward.technical_excellence.confidence,
                    "reasoning": reward.technical_excellence.reasoning,
                },
                "emotional_impact": {
                    "score": reward.emotional_impact.score,
                    "confidence": reward.emotional_impact.confidence,
                    "reasoning": reward.emotional_impact.reasoning,
                },
            },
            "suggestions": self.get_improvement_suggestions(reward),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    system = VisionRewardSystem()

    # Test evaluation
    image_features = torch.randn(1, 4096)
    prompt_features = torch.randn(1, 4096)

    reward = system.evaluate_image(image_features, prompt_features, user_rating=4.5)

    print("=== VisionReward Evaluation ===\n")
    report = system.get_detailed_report(reward)

    print(f"Overall Score: {report['overall_score']:.1%}")
    print(f"Preference Strength: {report['preference_strength']:.1%}\n")

    print("Dimension Scores:")
    for dim_name, dim_data in report["dimensions"].items():
        print(
            f"  {dim_name.replace('_', ' ').title()}: {dim_data['score']:.1%} "
            f"({dim_data['reasoning']})"
        )

    print("\nImprovement Suggestions:")
    for suggestion in report["suggestions"]:
        print(f"  - {suggestion}")

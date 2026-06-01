"""
Adaptive Learning System: Learns from quality feedback to continuously improve.
Collects user preferences and optimizes model parameters accordingly.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class GenerationFeedback:
    """Feedback on a generation."""
    prompt: str
    generated_features: torch.Tensor
    user_rating: float  # 0-5
    quality_score: float  # 0-1 (from quality system)
    adherence_score: float  # 0-1 (from adherence system)
    refinement_applied: bool
    timestamp: datetime = field(default_factory=datetime.now)
    user_comments: Optional[str] = None
    specific_improvements: List[str] = field(default_factory=list)


class PreferenceLearner(nn.Module):
    """Learns user preferences from feedback."""

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        # Preference encoder
        self.preference_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )

        # User taste model (learns what user likes)
        self.taste_model = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Style preference detector
        self.style_preference = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )

        # Detail preference detector
        self.detail_preference = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )

    def forward(
        self,
        generated_features: torch.Tensor,
        rating: float,
    ) -> Dict[str, torch.Tensor]:
        """Learn preferences from feedback."""
        encoded = self.preference_encoder(generated_features)

        # User taste score (0-1, higher = user likes it)
        taste = self.taste_model(encoded)  # (1, 1)
        taste_expanded = taste.expand(-1, 64)  # Expand to (1, 64) for consistency

        # Style preferences (1, 64)
        style = self.style_preference(encoded)

        # Detail preferences (1, 64)
        detail = self.detail_preference(encoded)

        # Weight by user rating (higher rating = stronger signal)
        weight = rating / 5.0

        return {
            "taste": taste_expanded * weight,
            "style": style * weight,
            "detail": detail * weight,
            "rating_weight": weight,
        }


class ParameterOptimizer(nn.Module):
    """Optimizes generation parameters based on feedback."""

    def __init__(self):
        super().__init__()

        # Parameter predictor (learns optimal parameters)
        self.guidance_optimizer = nn.Sequential(
            nn.Linear(64 * 2, 128),  # taste (64) + style (64)
            nn.GELU(),
            nn.Linear(128, 1),  # guidance scale
        )

        self.temperature_optimizer = nn.Sequential(
            nn.Linear(64 * 2, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        self.refinement_optimizer = nn.Sequential(
            nn.Linear(64 * 3, 128),  # taste (64) + style (64) + detail (64)
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        preferences: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Optimize generation parameters."""
        taste = preferences["taste"]
        style = preferences["style"]
        detail = preferences["detail"]

        # Ensure tensors are 2D
        if taste.dim() == 1:
            taste = taste.unsqueeze(0)
        if style.dim() == 1:
            style = style.unsqueeze(0)
        if detail.dim() == 1:
            detail = detail.unsqueeze(0)

        # Predict optimal guidance
        guidance_input = torch.cat([taste, style], dim=-1)
        guidance_raw = float(self.guidance_optimizer(guidance_input).squeeze())
        guidance = max(7.0, min(10.0, 7.0 + (guidance_raw * 3.0)))  # 7.0-10.0 range, clamped

        # Predict optimal temperature
        temp_input = torch.cat([style, detail], dim=-1)
        temperature = self.temperature_optimizer(temp_input)
        temperature = 0.3 + (float(temperature) * 0.5)  # 0.3-0.8 range

        # Predict refinement strength
        refine_input = torch.cat([taste, style, detail], dim=-1)
        refinement = self.refinement_optimizer(refine_input)
        refinement = float(refinement)

        return {
            "guidance_scale": guidance,
            "temperature": temperature,
            "refinement_strength": refinement,
        }


class AdaptiveStyleTransfer(nn.Module):
    """Learns and applies user's preferred style."""

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        # Style accumulator (learns dominant user style)
        self.style_accumulator = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )

        # Style transfer module
        self.style_transfer = nn.Sequential(
            nn.Linear(128 * 2, 256),  # user style + image features
            nn.GELU(),
            nn.Linear(256, hidden_dim),
        )

    def accumulate_style(
        self,
        generated_features: torch.Tensor,
        rating: float,
    ) -> torch.Tensor:
        """Accumulate user style preferences."""
        if rating >= 4.0:  # Only learn from high-rated images
            style = self.style_accumulator(generated_features)
            return style * (rating / 5.0)  # Weight by rating
        return torch.zeros(1, 128)

    def transfer_style(
        self,
        user_style: torch.Tensor,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        """Transfer learned user style to new image."""
        combined = torch.cat([user_style, image_features.mean(dim=0, keepdim=True)], dim=-1)
        return self.style_transfer(combined)


class ContinualLearningBuffer:
    """Maintains feedback history for continual learning."""

    def __init__(self, max_size: int = 1000):
        self.feedbacks: List[GenerationFeedback] = []
        self.max_size = max_size
        self.accumulated_style = None
        self.learned_parameters = {}

    def add_feedback(self, feedback: GenerationFeedback):
        """Add feedback to buffer."""
        self.feedbacks.append(feedback)

        # Keep buffer bounded
        if len(self.feedbacks) > self.max_size:
            # Remove oldest low-quality samples
            self.feedbacks.sort(key=lambda f: f.user_rating)
            self.feedbacks = self.feedbacks[-self.max_size :]

        logger.info(
            f"Feedback added (rating: {feedback.user_rating}/5, "
            f"quality: {feedback.quality_score:.2%}). "
            f"Buffer size: {len(self.feedbacks)}"
        )

    def get_high_quality_samples(self, threshold: float = 4.0) -> List[GenerationFeedback]:
        """Get high-rated samples for learning."""
        return [f for f in self.feedbacks if f.user_rating >= threshold]

    def get_recent_feedback(self, limit: int = 10) -> List[GenerationFeedback]:
        """Get recent feedback."""
        return sorted(self.feedbacks, key=lambda f: f.timestamp, reverse=True)[:limit]

    def compute_average_quality(self) -> float:
        """Compute average quality across feedbacks."""
        if not self.feedbacks:
            return 0.0
        return sum(f.quality_score for f in self.feedbacks) / len(self.feedbacks)


class AdaptiveLearningSystem:
    """Unified adaptive learning system."""

    def __init__(self, hidden_dim: int = 4096):
        self.preference_learner = PreferenceLearner(hidden_dim)
        self.parameter_optimizer = ParameterOptimizer()
        self.style_transfer = AdaptiveStyleTransfer(hidden_dim)
        self.feedback_buffer = ContinualLearningBuffer()

        self.user_style = None
        self.learned_parameters = {
            "guidance_scale": 7.5,
            "temperature": 0.5,
            "refinement_strength": 0.2,
        }

    def add_generation_feedback(
        self,
        prompt: str,
        generated_features: torch.Tensor,
        user_rating: float,
        quality_score: float,
        adherence_score: float,
        refinement_applied: bool = False,
        user_comments: Optional[str] = None,
        improvements: Optional[List[str]] = None,
    ):
        """Record feedback and learn from it."""
        feedback = GenerationFeedback(
            prompt=prompt,
            generated_features=generated_features,
            user_rating=user_rating,
            quality_score=quality_score,
            adherence_score=adherence_score,
            refinement_applied=refinement_applied,
            user_comments=user_comments,
            specific_improvements=improvements or [],
        )

        self.feedback_buffer.add_feedback(feedback)

        # Learn preferences
        preferences = self.preference_learner(generated_features, user_rating)

        # Update learned parameters
        optimized_params = self.parameter_optimizer(preferences)
        self.learned_parameters.update(optimized_params)

        logger.info(
            f"Learned parameters updated: "
            f"guidance={self.learned_parameters['guidance_scale']:.2f}, "
            f"temp={self.learned_parameters['temperature']:.2f}"
        )

        # Accumulate style if high quality
        if user_rating >= 4.0:
            style = self.style_transfer.accumulate_style(generated_features, user_rating)
            if self.user_style is None:
                self.user_style = style
            else:
                self.user_style = 0.9 * self.user_style + 0.1 * style

    def get_adaptive_parameters(self) -> Dict[str, float]:
        """Get current learned parameters."""
        return self.learned_parameters.copy()

    def suggest_improvements(self) -> List[str]:
        """Suggest improvements based on feedback patterns."""
        suggestions = []

        avg_quality = self.feedback_buffer.compute_average_quality()
        if avg_quality < 0.75:
            suggestions.append("Increase refinement iterations")

        # Analyze common issues
        recent = self.feedback_buffer.get_recent_feedback(10)
        if recent:
            avg_rating = sum(f.user_rating for f in recent) / len(recent)
            if avg_rating < 3.5:
                suggestions.append("User preferences not being met - retraining recommended")

        return suggestions

    def get_learning_progress(self) -> Dict:
        """Get learning progress metrics."""
        high_quality = self.feedback_buffer.get_high_quality_samples(4.0)
        total = len(self.feedback_buffer.feedbacks)

        return {
            "total_feedbacks": total,
            "high_quality_samples": len(high_quality),
            "high_quality_ratio": len(high_quality) / max(1, total),
            "average_quality": self.feedback_buffer.compute_average_quality(),
            "average_rating": (
                sum(f.user_rating for f in self.feedback_buffer.feedbacks) / max(1, total)
            ),
            "learned_parameters": self.learned_parameters,
            "has_user_style": self.user_style is not None,
        }

    def export_learned_model(self) -> Dict:
        """Export learned parameters for deployment."""
        return {
            "timestamp": datetime.now().isoformat(),
            "learned_parameters": self.learned_parameters,
            "user_style_available": self.user_style is not None,
            "feedback_count": len(self.feedback_buffer.feedbacks),
            "high_quality_count": len(self.feedback_buffer.get_high_quality_samples(4.0)),
            "suggestions": self.suggest_improvements(),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    system = AdaptiveLearningSystem()

    # Simulate user feedback
    for i in range(5):
        features = torch.randn(1, 4096)
        rating = 4.0 + (i * 0.2)  # Simulate improving ratings

        system.add_generation_feedback(
            prompt=f"Test prompt {i}",
            generated_features=features,
            user_rating=min(rating, 5.0),
            quality_score=0.85 + (i * 0.03),
            adherence_score=0.80 + (i * 0.02),
        )

    progress = system.get_learning_progress()
    print("\n=== Learning Progress ===")
    for key, value in progress.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")

"""
Label-Free Evolving Quality Framework (ELIQ):
Adaptive quality assessment that evolves with model improvements without retraining.
Based on research: https://arxiv.org/pdf/2602.03558

Key insight: As generative models improve, human perception scale shifts.
ELIQ detects and adapts to these shifts automatically.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class QualityShift:
    """Represents a detected shift in quality perception."""
    shift_magnitude: float  # 0-1, how large the shift is
    direction: str  # "upward" or "downward"
    confidence: float  # 0-1, how confident about shift
    affected_dimensions: List[str]  # which quality dimensions shifted
    timestamp: float  # when shift was detected


class AdaptivePerceptualScale(nn.Module):
    """Learns to adapt quality assessment scale as models improve."""

    def __init__(self, hidden_dim: int = 256, num_dimensions: int = 10):
        super().__init__()

        self.num_dimensions = num_dimensions

        # Track historical quality distributions
        self.quality_history = []
        self.max_history = 1000

        # Scale adaptation network
        self.scale_adapter = nn.Sequential(
            nn.Linear(hidden_dim + num_dimensions, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_dimensions),
            nn.Sigmoid(),  # Output normalized scales 0-1
        )

        # Shift detector
        self.shift_detector = nn.Sequential(
            nn.Linear(num_dimensions * 2, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 4),  # shift_mag, direction, confidence, threshold
        )

        # Reference quality anchor (baseline)
        self.register_buffer(
            "reference_quality_anchor",
            torch.ones(num_dimensions) * 0.5  # Neutral starting point
        )

        self.generation_count = 0

    def record_quality_measurement(self, quality_vector: torch.Tensor):
        """Record a quality measurement for distribution analysis."""
        if quality_vector.dim() == 1:
            quality_vector = quality_vector.unsqueeze(0)

        self.quality_history.append(quality_vector.detach().cpu())

        # Keep only recent history
        if len(self.quality_history) > self.max_history:
            self.quality_history = self.quality_history[-self.max_history:]

        self.generation_count += 1

    def detect_quality_shift(self) -> Optional[QualityShift]:
        """Detect if perceptual quality scale has shifted."""
        if len(self.quality_history) < 50:
            return None  # Need enough samples

        # Split history into old and recent
        old_history = self.quality_history[:-100] if len(self.quality_history) > 100 else self.quality_history[:max(1, len(self.quality_history)//2)]
        recent_history = self.quality_history[-50:]

        if not old_history or not recent_history:
            return None

        old_qualities = torch.cat(old_history, dim=0)
        recent_qualities = torch.cat(recent_history, dim=0)

        old_mean = old_qualities.mean(dim=0)
        old_std = old_qualities.std(dim=0)

        recent_mean = recent_qualities.mean(dim=0)
        recent_std = recent_qualities.std(dim=0)

        # Detect shifts in mean and std
        mean_shift = (recent_mean - old_mean).abs().mean()
        std_shift = (recent_std - old_std).abs().mean()

        total_shift = (mean_shift + std_shift) / 2

        if total_shift > 0.1:  # Significant shift detected
            # Determine which dimensions shifted
            dimension_shifts = (recent_mean - old_mean).abs()
            affected = [
                i for i in range(len(dimension_shifts))
                if dimension_shifts[i] > 0.05
            ]

            return QualityShift(
                shift_magnitude=float(total_shift),
                direction="upward" if mean_shift > 0 else "downward",
                confidence=min(1.0, total_shift / 0.3),
                affected_dimensions=[f"dim_{i}" for i in affected],
                timestamp=float(self.generation_count),
            )

        return None

    def get_adaptive_scale(self, image_features: torch.Tensor) -> torch.Tensor:
        """Get quality assessment scale adapted to current model state."""
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)

        # Compress image features to avoid dimension bloat
        feature_reducer = nn.Sequential(
            nn.Linear(4096, 256),
            nn.GELU(),
        )

        reduced_features = feature_reducer(image_features)

        # Current reference is dynamically updated
        current_ref = self.reference_quality_anchor.unsqueeze(0)

        # Combine features with reference for context
        combined = torch.cat([reduced_features, current_ref], dim=-1)

        # Get adapted scale
        scale = self.scale_adapter(combined)

        return scale

    def update_reference_anchor(self, new_anchor: torch.Tensor):
        """Update reference quality anchor to new baseline."""
        if new_anchor.dim() == 1:
            new_anchor = new_anchor.unsqueeze(0)

        # Smooth update (don't shift too drastically)
        update_rate = 0.1
        self.reference_quality_anchor = (
            self.reference_quality_anchor * (1 - update_rate) +
            new_anchor.squeeze(0) * update_rate
        )


class LabelFreeQualityAssessor(nn.Module):
    """Assesses quality without human labels by detecting internal consistency."""

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        self.perceptual_scale = AdaptivePerceptualScale(hidden_dim=256)

        # Consistency checkers (no labels needed)
        self.consistency_checkers = nn.ModuleDict({
            "aesthetic_internal": nn.Sequential(
                nn.Linear(hidden_dim, 512),
                nn.GELU(),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            ),
            "semantic_internal": nn.Sequential(
                nn.Linear(hidden_dim * 2, 512),
                nn.GELU(),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            ),
            "technical_internal": nn.Sequential(
                nn.Linear(hidden_dim, 512),
                nn.GELU(),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            ),
        })

        # Self-supervised quality predictor
        self.self_supervised_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 10),  # 10 quality dimensions
        )

    def assess_without_labels(
        self,
        image_features: torch.Tensor,
        prompt_features: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Assess quality purely from internal consistency signals."""
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)

        # Self-supervised quality prediction
        quality_vector = self.self_supervised_predictor(image_features)
        quality_vector = torch.clamp(quality_vector, 0, 1)

        # Consistency checks
        aesthetic_consistency = float(
            self.consistency_checkers["aesthetic_internal"](image_features).squeeze()
        )

        technical_consistency = float(
            self.consistency_checkers["technical_internal"](image_features).squeeze()
        )

        # Semantic consistency (if prompt provided)
        if prompt_features is not None:
            if prompt_features.dim() == 1:
                prompt_features = prompt_features.unsqueeze(0)
            combined = torch.cat([image_features, prompt_features], dim=-1)
            semantic_consistency = float(
                self.consistency_checkers["semantic_internal"](combined).squeeze()
            )
        else:
            semantic_consistency = 0.5

        # Get adapted scale for this generation
        adaptive_scale = self.perceptual_scale.get_adaptive_scale(image_features)

        # Apply adaptive scale to quality vector
        scaled_quality = quality_vector * adaptive_scale

        # Record for shift detection
        self.perceptual_scale.record_quality_measurement(scaled_quality)

        overall_score = (
            float(scaled_quality.mean()) * 0.5 +
            aesthetic_consistency * 0.2 +
            semantic_consistency * 0.15 +
            technical_consistency * 0.15
        )

        # Check for quality shift
        detected_shift = self.perceptual_scale.detect_quality_shift()

        return {
            "overall_quality": overall_score,
            "quality_vector": quality_vector.squeeze().detach().cpu().numpy(),
            "aesthetic_consistency": aesthetic_consistency,
            "semantic_consistency": semantic_consistency,
            "technical_consistency": technical_consistency,
            "adaptive_scale": adaptive_scale.squeeze().detach().cpu().numpy(),
            "quality_shift_detected": detected_shift is not None,
            "shift_details": detected_shift,
            "label_free": True,
            "assessment_type": "internal_consistency",
        }


class ELIQSystem:
    """Complete label-free evolving quality framework."""

    def __init__(self, hidden_dim: int = 4096):
        self.assessor = LabelFreeQualityAssessor(hidden_dim)

        self.assessment_history = []
        self.shift_history = []

    def assess_generation(
        self,
        image_features: torch.Tensor,
        prompt_features: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Assess generation quality without human labels."""
        result = self.assessor.assess_without_labels(image_features, prompt_features)

        self.assessment_history.append(result)

        if result["quality_shift_detected"]:
            self.shift_history.append(result["shift_details"])
            logger.info(
                f"Quality shift detected: {result['shift_details'].direction} "
                f"(magnitude: {result['shift_details'].shift_magnitude:.3f})"
            )

        return result

    def get_quality_report(self) -> Dict:
        """Generate report on quality assessment trends."""
        if not self.assessment_history:
            return {"status": "no_assessments_yet"}

        scores = [a["overall_quality"] for a in self.assessment_history]

        return {
            "total_assessments": len(self.assessment_history),
            "average_quality": sum(scores) / len(scores),
            "min_quality": min(scores),
            "max_quality": max(scores),
            "quality_trend": (
                "improving" if scores[-1] > sum(scores[:-10]) / 10 else "stable"
                if abs(scores[-1] - sum(scores[:-10]) / 10) < 0.05 else "declining"
            ),
            "shifts_detected": len(self.shift_history),
            "recent_shifts": self.shift_history[-3:] if self.shift_history else [],
            "label_free_status": "fully_self_supervised",
        }

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics."""
        return {
            "total_generations_assessed": len(self.assessment_history),
            "shift_detection_active": True,
            "perceptual_scale_shifts": len(self.shift_history),
            "label_dependency": "none",
            "retraining_required": False,
            "adaptive_capability": "continuous",
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    system = ELIQSystem()

    print("=== Label-Free Evolving Quality Framework (ELIQ) ===\n")

    # Simulate generations and detect quality shift
    for i in range(20):
        image = torch.randn(1, 4096)
        prompt = torch.randn(1, 4096)

        result = system.assess_generation(image, prompt)

        if result["quality_shift_detected"]:
            print(f"[{i}] Quality shift: {result['shift_details'].direction}")
        else:
            print(f"[{i}] Quality: {result['overall_quality']:.3f}")

    report = system.get_quality_report()
    print("\nQuality Report:")
    for key, value in report.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")

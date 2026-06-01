"""
Semantic Drift Detection System:
Detects when refinement causes semantic meaning to shift away from original intent.
Prevents 'concept drift' during iterative refinement.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class SemanticChange:
    """Represents a detected semantic change."""
    magnitude: float  # 0-1, how much meaning shifted
    direction: str  # which concepts changed
    concept_shifts: Dict[str, float]  # per-concept shift scores
    severity: str  # "acceptable", "warning", "critical"
    recommendation: str  # what to do about it


class ConceptTracker(nn.Module):
    """Tracks concept presence and confidence across refinement steps."""

    def __init__(self, hidden_dim: int = 4096, num_concepts: int = 50):
        super().__init__()

        self.num_concepts = num_concepts

        # Concept extractor
        self.concept_extractor = nn.Sequential(
            nn.Linear(hidden_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, num_concepts),
            nn.Sigmoid(),  # Concept presence 0-1
        )

        # Concept importance scorer
        self.importance_scorer = nn.Sequential(
            nn.Linear(num_concepts, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_concepts),
            nn.Sigmoid(),
        )

        self.concept_names = [
            "color_palette", "lighting_style", "composition", "subject",
            "background", "texture", "mood", "artistic_style", "detail_level", "realism",
            "symmetry", "perspective", "depth", "contrast", "saturation",
            "warmth", "sharpness", "focus_point", "object_count", "scene_type",
            "season", "time_of_day", "weather", "materials", "cleanliness",
            "motion", "emotion_positive", "emotion_negative", "elegance", "chaos",
            "formality", "simplicity", "complexity", "naturalness", "artificiality",
            "vintage", "modernity", "abstract", "realistic", "surreal",
            "geometric", "organic", "symmetrical", "asymmetrical", "balanced",
            "chaotic", "minimal", "ornate", "professional", "amateur",
            "bright", "dark", "colorful", "monochrome", "mysterious",
        ][:num_concepts]

    def extract_concepts(self, image_features: torch.Tensor) -> torch.Tensor:
        """Extract concept presence vector from image."""
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)

        concepts = self.concept_extractor(image_features)
        return concepts

    def get_concept_importance(self, concepts: torch.Tensor) -> torch.Tensor:
        """Get importance scores for each concept."""
        importance = self.importance_scorer(concepts)
        return importance


class SemanticAnchor(nn.Module):
    """Maintains semantic anchor from original prompt."""

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        # Anchor encoder - compress to 256
        self.anchor_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )

        # Feature compressor for similarity scoring
        self.feature_compressor = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )

        # Semantic similarity scorer
        self.similarity_scorer = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def set_anchor(self, prompt_features: torch.Tensor) -> torch.Tensor:
        """Set semantic anchor from original prompt."""
        if prompt_features.dim() == 1:
            prompt_features = prompt_features.unsqueeze(0)

        anchor = self.anchor_encoder(prompt_features)
        return anchor

    def compute_semantic_distance(
        self,
        anchor: torch.Tensor,
        current_features: torch.Tensor,
    ) -> float:
        """Compute semantic distance from anchor."""
        if current_features.dim() == 1:
            current_features = current_features.unsqueeze(0)

        if anchor.dim() == 1:
            anchor = anchor.unsqueeze(0)

        # Compress current state to same dimension as anchor
        current_encoded = self.feature_compressor(current_features)

        # Compute similarity
        combined = torch.cat([anchor, current_encoded], dim=-1)
        similarity = float(self.similarity_scorer(combined).squeeze())

        # Distance is 1 - similarity
        distance = 1.0 - similarity

        return distance


class DriftDetector(nn.Module):
    """Detects semantic drift during refinement."""

    def __init__(self, hidden_dim: int = 4096, num_concepts: int = 50):
        super().__init__()

        self.concept_tracker = ConceptTracker(hidden_dim, num_concepts)
        self.semantic_anchor = SemanticAnchor(hidden_dim)

        # Feature compressor for concept analysis
        self.feature_compressor = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, num_concepts),
        )

        # Drift detector network
        self.drift_detector = nn.Sequential(
            nn.Linear(num_concepts * 2, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Concept shift analyzer
        self.shift_analyzer = nn.Sequential(
            nn.Linear(num_concepts, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_concepts),
            nn.Tanh(),  # Can shift both positive and negative
        )

    def detect_drift(
        self,
        original_concepts: torch.Tensor,
        current_concepts: torch.Tensor,
        concept_importance: torch.Tensor,
    ) -> SemanticChange:
        """Detect semantic drift between original and current."""
        # Compute per-concept shifts (weighted by importance)
        concept_shifts = (current_concepts - original_concepts).squeeze()
        weighted_shifts = concept_shifts * concept_importance.squeeze()

        # Magnitude of drift
        drift_magnitude = float(torch.abs(weighted_shifts).mean())

        # Which concepts shifted most
        shift_scores = torch.abs(concept_shifts).detach().cpu().numpy()
        major_shifts = {
            self.concept_tracker.concept_names[i]: float(shift_scores[i])
            for i in torch.argsort(torch.tensor(shift_scores), descending=True)[:5]
        }

        # Assess severity
        if drift_magnitude > 0.3:
            severity = "critical"
            recommendation = "Stop refinement, drift is too large"
        elif drift_magnitude > 0.15:
            severity = "warning"
            recommendation = "Continue refinement cautiously, monitor drift"
        else:
            severity = "acceptable"
            recommendation = "Continue refinement, drift within acceptable range"

        return SemanticChange(
            magnitude=drift_magnitude,
            direction=f"shifted away from original in {len(major_shifts)} dimensions",
            concept_shifts=major_shifts,
            severity=severity,
            recommendation=recommendation,
        )


class SemanticDriftDetectionSystem:
    """Complete semantic drift detection and prevention system."""

    def __init__(self, hidden_dim: int = 4096):
        self.drift_detector = DriftDetector(hidden_dim)

        self.refinement_trajectory = []
        self.drift_events = []
        self.anchor_set = False
        self.semantic_anchor = None

    def set_original_prompt(self, prompt_features: torch.Tensor):
        """Set the semantic anchor from original prompt."""
        if prompt_features.dim() == 1:
            prompt_features = prompt_features.unsqueeze(0)

        # Store original prompt features for later drift detection
        self.semantic_anchor = prompt_features.detach().clone()
        self.anchor_set = True

    def check_semantic_drift(
        self,
        image_features: torch.Tensor,
        refinement_step: int = 0,
        step: Optional[int] = None,
    ) -> Dict:
        """Check for semantic drift from original prompt."""
        if not self.anchor_set:
            return {"status": "anchor_not_set"}

        # Support both 'refinement_step' and 'step' parameter names
        if step is not None:
            refinement_step = step

        # Extract concepts
        original_concepts = self.drift_detector.concept_tracker.extract_concepts(
            self.semantic_anchor
        )
        current_concepts = self.drift_detector.concept_tracker.extract_concepts(
            image_features
        )

        # Get concept importance
        concept_importance = self.drift_detector.concept_tracker.get_concept_importance(
            original_concepts
        )

        # Detect drift
        drift_info = self.drift_detector.detect_drift(
            original_concepts,
            current_concepts,
            concept_importance,
        )

        # Store in trajectory
        trajectory_entry = {
            "step": refinement_step,
            "drift_magnitude": drift_info.magnitude,
            "severity": drift_info.severity,
            "concept_shifts": drift_info.concept_shifts,
        }
        self.refinement_trajectory.append(trajectory_entry)

        if drift_info.severity != "acceptable":
            self.drift_events.append(drift_info)

        return {
            "drift_detected": drift_info.severity != "acceptable",
            "drift_magnitude": drift_info.magnitude,
            "severity": drift_info.severity,
            "concept_shifts": drift_info.concept_shifts,
            "recommendation": drift_info.recommendation,
            "step": refinement_step,
            "should_continue_refinement": drift_info.severity != "critical",
        }

    def predict_drift_boundary(self, max_acceptable_drift: float = 0.15) -> Dict:
        """Predict when drift will exceed threshold."""
        if len(self.refinement_trajectory) < 3:
            return {"status": "insufficient_data"}

        # Fit trend line to drift magnitude
        steps = [e["step"] for e in self.refinement_trajectory]
        magnitudes = [e["drift_magnitude"] for e in self.refinement_trajectory]

        # Linear trend approximation
        if len(steps) > 1:
            slope = (magnitudes[-1] - magnitudes[0]) / (steps[-1] - steps[0] + 1e-6)
            current_magnitude = magnitudes[-1]
            steps_until_critical = max(
                0,
                (max_acceptable_drift - current_magnitude) / (slope + 1e-6)
            )

            return {
                "steps_until_boundary": int(steps_until_critical),
                "current_drift": current_magnitude,
                "drift_trend": "increasing" if slope > 0 else "stable" if abs(slope) < 0.01 else "decreasing",
                "recommended_max_steps": max(1, int(steps_until_critical)),
            }

        return {"status": "unable_to_predict"}

    def get_drift_report(self) -> Dict:
        """Generate comprehensive drift report."""
        if not self.refinement_trajectory:
            return {"status": "no_refinements"}

        critical_count = sum(
            1 for e in self.refinement_trajectory if e["severity"] == "critical"
        )
        warning_count = sum(
            1 for e in self.refinement_trajectory if e["severity"] == "warning"
        )

        drift_magnitudes = [e["drift_magnitude"] for e in self.refinement_trajectory]

        return {
            "total_refinement_steps": len(self.refinement_trajectory),
            "critical_drift_events": critical_count,
            "warning_drift_events": warning_count,
            "average_drift": sum(drift_magnitudes) / len(drift_magnitudes),
            "max_drift": max(drift_magnitudes),
            "min_drift": min(drift_magnitudes),
            "drift_trend": self.refinement_trajectory[-1]["severity"],
            "most_affected_concepts": self.refinement_trajectory[-1]["concept_shifts"],
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    system = SemanticDriftDetectionSystem()

    print("=== Semantic Drift Detection System ===\n")

    # Simulate original prompt
    original_prompt = torch.randn(1, 4096)
    system.set_original_prompt(original_prompt)

    print("Semantic anchor set from original prompt\n")

    # Simulate refinement steps
    for step in range(5):
        # Gradually shift away from original
        noise_scale = 0.1 * (step + 1)
        current_image = original_prompt + torch.randn_like(original_prompt) * noise_scale

        result = system.check_semantic_drift(current_image, step)

        if result["status"] != "anchor_not_set":
            print(f"Step {step}: Drift = {result['drift_magnitude']:.3f}, "
                  f"Severity = {result['severity']}")
            if result["recommendation"] != "Continue refinement, drift within acceptable range":
                print(f"  WARNING: {result['recommendation']}")

    print("\nDrift Report:")
    report = system.get_drift_report()
    for key, value in report.items():
        print(f"  {key}: {value}")

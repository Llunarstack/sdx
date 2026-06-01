"""
Generation-Specific Artifact Detector:
Detects GAN artifacts (checkerboard, mode collapse) and diffusion artifacts (speckles, banding).
Identifies and localizes problem areas with surgical precision.
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GANArtifactDetector(nn.Module):
    """Detects GAN-specific artifacts: checkerboard patterns, mode collapse."""

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        # Checkerboard pattern detector (repeating grid patterns)
        self.checkerboard_detector = nn.Sequential(
            nn.Linear(hidden_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        # Mode collapse detector (repeated textures)
        self.mode_collapse_detector = nn.Sequential(
            nn.Linear(hidden_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        # Frequency artifact detector (high-frequency noise patterns)
        self.frequency_detector = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        # Localization network (where are artifacts?)
        self.artifact_localizer = nn.Sequential(
            nn.Linear(hidden_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),  # 8x8 spatial locations
        )

    def forward(self, image_features: torch.Tensor) -> Dict:
        """Detect GAN artifacts in image."""
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)

        # Detect checkerboard pattern
        checkerboard_score = float(self.checkerboard_detector(image_features).squeeze())

        # Detect mode collapse
        mode_collapse_score = float(self.mode_collapse_detector(image_features).squeeze())

        # Detect frequency artifacts
        frequency_score = float(self.frequency_detector(image_features).squeeze())

        # Localize artifacts
        artifact_locations = self.artifact_localizer(image_features)
        artifact_locations = torch.sigmoid(artifact_locations).squeeze()

        # Overall GAN artifact score (weighted combination)
        overall_gan_score = (
            checkerboard_score * 0.4 +
            mode_collapse_score * 0.35 +
            frequency_score * 0.25
        )

        return {
            "overall_gan_artifact_score": overall_gan_score,
            "checkerboard_pattern_score": checkerboard_score,
            "mode_collapse_score": mode_collapse_score,
            "frequency_artifact_score": frequency_score,
            "artifact_locations": artifact_locations.detach().cpu().numpy(),
            "severity": (
                "critical" if overall_gan_score > 0.7
                else "high" if overall_gan_score > 0.5
                else "moderate" if overall_gan_score > 0.3
                else "low"
            ),
        }


class DiffusionArtifactDetector(nn.Module):
    """Detects diffusion model artifacts: speckles, color banding, blur."""

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        # Speckle detector (Karras artifacts)
        self.speckle_detector = nn.Sequential(
            nn.Linear(hidden_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        # Color banding detector
        self.banding_detector = nn.Sequential(
            nn.Linear(hidden_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        # Over-smoothing detector
        self.smoothing_detector = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        # Temporal consistency detector (for sequential generations)
        self.temporal_inconsistency_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        # Localization for diffusion artifacts
        self.diffusion_localizer = nn.Sequential(
            nn.Linear(hidden_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),  # 8x8 spatial locations
        )

    def forward(
        self,
        image_features: torch.Tensor,
        prev_frame: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Detect diffusion artifacts in image."""
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)

        # Detect speckles
        speckle_score = float(self.speckle_detector(image_features).squeeze())

        # Detect color banding
        banding_score = float(self.banding_detector(image_features).squeeze())

        # Detect over-smoothing
        smoothing_score = float(self.smoothing_detector(image_features).squeeze())

        # Temporal inconsistency (if previous frame available)
        temporal_score = 0.0
        if prev_frame is not None:
            if prev_frame.dim() == 1:
                prev_frame = prev_frame.unsqueeze(0)
            combined = torch.cat([image_features, prev_frame], dim=-1)
            temporal_score = float(self.temporal_inconsistency_detector(combined).squeeze())

        # Localize artifacts
        diffusion_locations = self.diffusion_localizer(image_features)
        diffusion_locations = torch.sigmoid(diffusion_locations).squeeze()

        # Overall diffusion artifact score
        overall_diffusion_score = (
            speckle_score * 0.35 +
            banding_score * 0.35 +
            smoothing_score * 0.2 +
            temporal_score * 0.1
        )

        return {
            "overall_diffusion_artifact_score": overall_diffusion_score,
            "speckle_score": speckle_score,
            "banding_score": banding_score,
            "smoothing_score": smoothing_score,
            "temporal_inconsistency_score": temporal_score,
            "artifact_locations": diffusion_locations.detach().cpu().numpy(),
            "severity": (
                "critical" if overall_diffusion_score > 0.7
                else "high" if overall_diffusion_score > 0.5
                else "moderate" if overall_diffusion_score > 0.3
                else "low"
            ),
        }


class ArtifactRemediationSuggester(nn.Module):
    """Suggests remediations for detected artifacts."""

    def __init__(self):
        super().__init__()

        # Maps artifact type to remediation strategies
        self.remediation_map = {
            "checkerboard_pattern": [
                {"strategy": "increase_diversity", "strength": 0.7},
                {"strategy": "reduce_guidance_scale", "strength": 0.5},
                {"strategy": "increase_seed_noise", "strength": 0.6},
            ],
            "mode_collapse": [
                {"strategy": "increase_diversity_weight", "strength": 0.8},
                {"strategy": "modify_prompt_variety", "strength": 0.6},
                {"strategy": "ensemble_different_models", "strength": 0.9},
            ],
            "frequency_artifacts": [
                {"strategy": "increase_num_steps", "strength": 0.7},
                {"strategy": "reduce_learning_rate", "strength": 0.6},
                {"strategy": "post_process_smoothing", "strength": 0.5},
            ],
            "speckles": [
                {"strategy": "increase_diffusion_steps", "strength": 0.8},
                {"strategy": "reduce_noise_scale", "strength": 0.7},
                {"strategy": "post_process_denoise", "strength": 0.8},
            ],
            "color_banding": [
                {"strategy": "add_dithering", "strength": 0.9},
                {"strategy": "increase_color_depth", "strength": 0.7},
                {"strategy": "post_process_smoothing", "strength": 0.6},
            ],
            "over_smoothing": [
                {"strategy": "reduce_num_steps", "strength": 0.5},
                {"strategy": "increase_guidance_scale", "strength": 0.6},
                {"strategy": "sharpen_details", "strength": 0.7},
            ],
        }

    def get_remediation_suggestions(
        self,
        artifact_type: str,
        severity: str,
    ) -> List[Dict]:
        """Get remediation suggestions for artifact."""
        if artifact_type not in self.remediation_map:
            return []

        suggestions = self.remediation_map[artifact_type].copy()

        # Adjust strengths based on severity
        severity_multipliers = {
            "critical": 1.5,
            "high": 1.2,
            "moderate": 0.9,
            "low": 0.5,
        }

        multiplier = severity_multipliers.get(severity, 1.0)

        for suggestion in suggestions:
            suggestion["strength"] = min(1.0, suggestion["strength"] * multiplier)

        return suggestions


class GenerationArtifactDetectionSystem:
    """Complete artifact detection and remediation system."""

    def __init__(self, hidden_dim: int = 4096):
        self.gan_detector = GANArtifactDetector(hidden_dim)
        self.diffusion_detector = DiffusionArtifactDetector(hidden_dim)
        self.remediation_suggester = ArtifactRemediationSuggester()

        self.detection_history = []

    def detect_artifacts(
        self,
        image_features: torch.Tensor,
        prev_frame: Optional[torch.Tensor] = None,
        generation_method: str = "unknown",
    ) -> Dict:
        """Detect both GAN and diffusion artifacts."""
        gan_results = self.gan_detector(image_features)
        diffusion_results = self.diffusion_detector(image_features, prev_frame)

        # Overall artifact score (average of both)
        overall_artifact_score = (
            gan_results["overall_gan_artifact_score"] * 0.5 +
            diffusion_results["overall_diffusion_artifact_score"] * 0.5
        )

        # Determine if generation is clean enough
        artifact_free = overall_artifact_score < 0.2

        # Identify dominant artifact type
        if gan_results["overall_gan_artifact_score"] > diffusion_results["overall_diffusion_artifact_score"]:
            dominant_artifact = (
                "checkerboard_pattern" if gan_results["checkerboard_pattern_score"] > 0.6
                else "mode_collapse" if gan_results["mode_collapse_score"] > 0.6
                else "frequency_artifacts"
            )
        else:
            dominant_artifact = (
                "speckles" if diffusion_results["speckle_score"] > 0.6
                else "color_banding" if diffusion_results["banding_score"] > 0.6
                else "over_smoothing"
            )

        # Get remediation suggestions
        severity = (
            "critical" if overall_artifact_score > 0.7
            else "high" if overall_artifact_score > 0.5
            else "moderate" if overall_artifact_score > 0.3
            else "low"
        )

        remediation_suggestions = self.remediation_suggester.get_remediation_suggestions(
            dominant_artifact,
            severity,
        )

        result = {
            "overall_artifact_score": overall_artifact_score,
            "artifact_free": artifact_free,
            "severity": severity,
            "gan_artifacts": gan_results,
            "diffusion_artifacts": diffusion_results,
            "dominant_artifact_type": dominant_artifact,
            "remediation_suggestions": remediation_suggestions,
            "assessment_confidence": (
                "high" if overall_artifact_score > 0.5 or overall_artifact_score < 0.2
                else "medium"
            ),
        }

        self.detection_history.append(result)
        return result

    def get_artifact_report(self) -> Dict:
        """Generate artifact detection report."""
        if not self.detection_history:
            return {"status": "no_detections"}

        artifact_free_count = sum(1 for d in self.detection_history if d["artifact_free"])
        avg_artifact_score = sum(d["overall_artifact_score"] for d in self.detection_history) / len(self.detection_history)

        artifact_types = {}
        for result in self.detection_history:
            artifact_type = result["dominant_artifact_type"]
            artifact_types[artifact_type] = artifact_types.get(artifact_type, 0) + 1

        return {
            "total_detections": len(self.detection_history),
            "artifact_free_percentage": artifact_free_count / len(self.detection_history) * 100,
            "average_artifact_score": avg_artifact_score,
            "artifact_type_distribution": artifact_types,
            "most_common_artifact": max(artifact_types, key=artifact_types.get),
            "critical_artifacts_detected": sum(
                1 for d in self.detection_history if d["severity"] == "critical"
            ),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    system = GenerationArtifactDetectionSystem()

    print("=== Generation-Specific Artifact Detector ===\n")

    # Test detection
    image = torch.randn(1, 4096)

    result = system.detect_artifacts(image)

    print(f"Overall artifact score: {result['overall_artifact_score']:.3f}")
    print(f"Artifact-free: {result['artifact_free']}")
    print(f"Severity: {result['severity']}")
    print(f"Dominant artifact: {result['dominant_artifact_type']}")
    print("\nRemediation suggestions:")
    for suggestion in result["remediation_suggestions"]:
        print(f"  - {suggestion['strategy']}: {suggestion['strength']:.1%}")

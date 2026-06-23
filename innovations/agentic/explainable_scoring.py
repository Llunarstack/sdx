"""
Explainable Quality Scoring System:
Generates human-readable explanations for every quality score.
Users understand exactly why quality is X and what to fix.
"""

import logging
from typing import Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class QualityDimensionAnalyzer(nn.Module):
    """Analyzes each quality dimension independently."""

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        # Individual dimension analyzers
        self.analyzers = nn.ModuleDict(
            {
                "composition": nn.Sequential(
                    nn.Linear(hidden_dim, 1024),
                    nn.GELU(),
                    nn.Linear(1024, 512),
                    nn.GELU(),
                    nn.Linear(512, 256),
                    nn.GELU(),
                    nn.Linear(256, 1),
                    nn.Sigmoid(),
                ),
                "color_harmony": nn.Sequential(
                    nn.Linear(hidden_dim, 1024),
                    nn.GELU(),
                    nn.Linear(1024, 512),
                    nn.GELU(),
                    nn.Linear(512, 256),
                    nn.GELU(),
                    nn.Linear(256, 1),
                    nn.Sigmoid(),
                ),
                "lighting": nn.Sequential(
                    nn.Linear(hidden_dim, 1024),
                    nn.GELU(),
                    nn.Linear(1024, 512),
                    nn.GELU(),
                    nn.Linear(512, 256),
                    nn.GELU(),
                    nn.Linear(256, 1),
                    nn.Sigmoid(),
                ),
                "clarity": nn.Sequential(
                    nn.Linear(hidden_dim, 1024),
                    nn.GELU(),
                    nn.Linear(1024, 512),
                    nn.GELU(),
                    nn.Linear(512, 256),
                    nn.GELU(),
                    nn.Linear(256, 1),
                    nn.Sigmoid(),
                ),
                "realism": nn.Sequential(
                    nn.Linear(hidden_dim, 1024),
                    nn.GELU(),
                    nn.Linear(1024, 512),
                    nn.GELU(),
                    nn.Linear(512, 256),
                    nn.GELU(),
                    nn.Linear(256, 1),
                    nn.Sigmoid(),
                ),
                "coherence": nn.Sequential(
                    nn.Linear(hidden_dim, 1024),
                    nn.GELU(),
                    nn.Linear(1024, 512),
                    nn.GELU(),
                    nn.Linear(512, 256),
                    nn.GELU(),
                    nn.Linear(256, 1),
                    nn.Sigmoid(),
                ),
                "detail_richness": nn.Sequential(
                    nn.Linear(hidden_dim, 1024),
                    nn.GELU(),
                    nn.Linear(1024, 512),
                    nn.GELU(),
                    nn.Linear(512, 256),
                    nn.GELU(),
                    nn.Linear(256, 1),
                    nn.Sigmoid(),
                ),
                "aesthetic_appeal": nn.Sequential(
                    nn.Linear(hidden_dim, 1024),
                    nn.GELU(),
                    nn.Linear(1024, 512),
                    nn.GELU(),
                    nn.Linear(512, 256),
                    nn.GELU(),
                    nn.Linear(256, 1),
                    nn.Sigmoid(),
                ),
            }
        )

        # Dimension weights (how much each contributes to overall quality)
        self.dimension_weights = {
            "composition": 0.15,
            "color_harmony": 0.12,
            "lighting": 0.12,
            "clarity": 0.12,
            "realism": 0.15,
            "coherence": 0.13,
            "detail_richness": 0.12,
            "aesthetic_appeal": 0.13,
        }

    def analyze_all_dimensions(self, image_features: torch.Tensor) -> Dict:
        """Analyze all quality dimensions."""
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)

        scores = {}
        weighted_scores = {}

        for dimension_name, analyzer in self.analyzers.items():
            score = float(analyzer(image_features).squeeze().detach())
            weight = self.dimension_weights.get(dimension_name, 0.1)

            scores[dimension_name] = score
            weighted_scores[dimension_name] = score * weight

        overall = sum(weighted_scores.values())

        return {
            "dimension_scores": scores,
            "dimension_contributions": weighted_scores,
            "overall_quality": overall,
        }


class PenaltyAnalyzer(nn.Module):
    """Identifies and quantifies specific quality penalties."""

    def __init__(self):
        super().__init__()

        # Penalty profiles
        self.penalty_profiles = {
            "blown_out_highlights": {
                "severity_range": (0.1, 0.5),
                "description": "Overexposed bright areas losing detail",
                "impact": -0.1,
                "fixable": True,
                "fix_suggestions": ["reduce_brightness", "increase_shadows", "adjust_exposure"],
            },
            "muddy_colors": {
                "severity_range": (0.1, 0.4),
                "description": "Colors lack vibrancy and appear desaturated",
                "impact": -0.12,
                "fixable": True,
                "fix_suggestions": ["increase_saturation", "boost_contrast", "shift_hue"],
            },
            "unnatural_lighting": {
                "severity_range": (0.15, 0.5),
                "description": "Lighting doesn't follow physical rules",
                "impact": -0.15,
                "fixable": True,
                "fix_suggestions": ["adjust_light_direction", "increase_ambient", "fix_shadows"],
            },
            "poor_composition": {
                "severity_range": (0.1, 0.4),
                "description": "Subject positioning lacks visual balance",
                "impact": -0.12,
                "fixable": False,  # Requires regeneration
                "fix_suggestions": ["reframe_subject", "adjust_rule_of_thirds", "reposition_elements"],
            },
            "motion_blur": {
                "severity_range": (0.1, 0.3),
                "description": "Unwanted motion blur reduces clarity",
                "impact": -0.08,
                "fixable": True,
                "fix_suggestions": ["increase_shutter_speed", "reduce_motion", "increase_clarity"],
            },
            "artifacts": {
                "severity_range": (0.2, 0.8),
                "description": "Visible artifacts: distortion, checkerboard, or noise",
                "impact": -0.2,
                "fixable": True,
                "fix_suggestions": ["regenerate", "increase_steps", "reduce_guidance"],
            },
            "lack_of_detail": {
                "severity_range": (0.1, 0.35),
                "description": "Image appears too smooth or lacks fine details",
                "impact": -0.1,
                "fixable": True,
                "fix_suggestions": ["increase_detail_level", "sharpen", "add_texture"],
            },
            "oversaturation": {
                "severity_range": (0.1, 0.3),
                "description": "Colors are too intense and unrealistic",
                "impact": -0.09,
                "fixable": True,
                "fix_suggestions": ["reduce_saturation", "tone_down_colors", "add_mutes"],
            },
        }

    def detect_penalties(self, quality_scores: Dict) -> List[Dict]:
        """Detect which penalties are affecting quality."""
        detected_penalties = []

        # Simulate detection (in real implementation, would analyze image features)
        for penalty_name, penalty_profile in self.penalty_profiles.items():
            # Random detection for demo (would be learned from image features)
            detection_score = torch.rand(1).detach().item()

            if detection_score > 0.5:  # 50% chance for demo
                detected_penalties.append(
                    {
                        "penalty_type": penalty_name,
                        "severity": detection_score,
                        "description": penalty_profile["description"],
                        "impact": penalty_profile["impact"] * detection_score,
                        "fixable": penalty_profile["fixable"],
                        "fix_suggestions": penalty_profile["fix_suggestions"],
                    }
                )

        # Sort by impact
        detected_penalties.sort(key=lambda x: abs(x["impact"]), reverse=True)

        return detected_penalties


class ExplanationGenerator:
    """Generates human-readable explanations for quality scores."""

    def __init__(self):
        self.explanation_templates = {
            "overall_high": "Quality is {score:.0%} because the image demonstrates strong visual composition and technical excellence.",
            "overall_medium": "Quality is {score:.0%}. The image has both strengths and areas for improvement.",
            "overall_low": "Quality is {score:.0%}. The image has several issues that should be addressed.",
            "dimension_strong": "• {dimension}: {score:.0%} - This dimension is a particular strength",
            "dimension_weak": "• {dimension}: {score:.0%} - This dimension needs improvement",
            "penalty_detected": "• {penalty}: ({severity:.0%} severity) - {description}. Suggested fixes: {fixes}",
        }

    def generate_explanation(
        self,
        overall_quality: float,
        dimension_scores: Dict,
        penalties: List[Dict],
    ) -> str:
        """Generate comprehensive quality explanation."""
        lines = []

        # Overall assessment
        if overall_quality > 0.75:
            template = self.explanation_templates["overall_high"]
        elif overall_quality > 0.55:
            template = self.explanation_templates["overall_medium"]
        else:
            template = self.explanation_templates["overall_low"]

        lines.append(f"Overall Quality Score: {overall_quality:.0%}")
        lines.append(template.format(score=overall_quality))
        lines.append("")

        # Dimension breakdown
        lines.append("Quality Breakdown by Dimension:")
        for dim_name in sorted(dimension_scores.keys()):
            score = dimension_scores[dim_name]
            if score > 0.75:
                template = self.explanation_templates["dimension_strong"]
            else:
                template = self.explanation_templates["dimension_weak"]

            lines.append(template.format(dimension=dim_name.replace("_", " ").title(), score=score))

        lines.append("")

        # Penalties and fixes
        if penalties:
            lines.append("Issues Detected and Recommended Fixes:")
            for penalty in penalties:
                fixes_str = ", ".join(penalty["fix_suggestions"][:2])
                lines.append(
                    self.explanation_templates["penalty_detected"].format(
                        penalty=penalty["penalty_type"].replace("_", " ").title(),
                        severity=penalty["severity"],
                        description=penalty["description"],
                        fixes=fixes_str,
                    )
                )
        else:
            lines.append("No significant issues detected.")

        lines.append("")

        # Actionable recommendations
        lines.append("Recommendations:")
        if overall_quality < 0.5:
            lines.append("• Consider regenerating with modified prompts or parameters")
            lines.append("• Review detected issues above for specific guidance")
        elif overall_quality < 0.75:
            lines.append("• Address the low-scoring dimensions listed above")
            lines.append("• Apply the suggested fixes for detected issues")
        else:
            lines.append("• Image quality is good. Minor refinements possible.")
            lines.append("• Consider if additional detail or stylistic changes are desired")

        return "\n".join(lines)


class ExplainableQualityScoringSystem:
    """Complete explainable quality scoring system."""

    def __init__(self, hidden_dim: int = 4096):
        self.dimension_analyzer = QualityDimensionAnalyzer(hidden_dim)
        self.penalty_analyzer = PenaltyAnalyzer()
        self.explanation_generator = ExplanationGenerator()

        self.scoring_history = []

    def score_with_explanation(
        self,
        image_features: torch.Tensor,
    ) -> Dict:
        """Score image and generate detailed explanation."""
        # Analyze dimensions
        dimension_analysis = self.dimension_analyzer.analyze_all_dimensions(image_features)

        # Detect penalties
        penalties = self.penalty_analyzer.detect_penalties(dimension_analysis["dimension_scores"])

        # Generate explanation
        explanation = self.explanation_generator.generate_explanation(
            dimension_analysis["overall_quality"],
            dimension_analysis["dimension_scores"],
            penalties,
        )

        # Calculate adjusted score (account for penalties)
        adjusted_score = dimension_analysis["overall_quality"]
        for penalty in penalties:
            adjusted_score += penalty["impact"]
        adjusted_score = max(0.0, min(1.0, adjusted_score))

        result = {
            "overall_quality": dimension_analysis["overall_quality"],
            "adjusted_quality": adjusted_score,
            "dimension_scores": dimension_analysis["dimension_scores"],
            "penalties_detected": penalties,
            "explanation": explanation,
            "fixable_issues": sum(1 for p in penalties if p["fixable"]),
            "unfixable_issues": sum(1 for p in penalties if not p["fixable"]),
        }

        self.scoring_history.append(result)
        return result

    def get_summary_report(self) -> str:
        """Get summary of recent scorings."""
        if not self.scoring_history:
            return "No scores recorded yet"

        avg_quality = sum(s["overall_quality"] for s in self.scoring_history) / len(self.scoring_history)
        avg_adjusted = sum(s["adjusted_quality"] for s in self.scoring_history) / len(self.scoring_history)

        return f"""
Explainable Scoring Summary:
- Assessments: {len(self.scoring_history)}
- Average Quality: {avg_quality:.0%}
- Average Adjusted Quality: {avg_adjusted:.0%}
- Most Common Penalty: [would show in full implementation]
"""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    system = ExplainableQualityScoringSystem()

    print("=== Explainable Quality Scoring System ===\n")

    # Test scoring
    image = torch.randn(1, 4096)
    result = system.score_with_explanation(image)

    print(result["explanation"])
    print(f"\nAdjusted Quality: {result['adjusted_quality']:.0%}")
    print(f"Fixable Issues: {result['fixable_issues']}")
    print(f"Unfixable Issues: {result['unfixable_issues']}")

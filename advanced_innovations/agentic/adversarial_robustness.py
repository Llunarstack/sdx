"""
Adversarial Robustness System: Tests generation robustness to prompt variations.
Ensures consistent quality across different phrasings and prompt perturbations.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class PerturbationTest:
    """Results from a single perturbation test."""
    perturbation_type: str
    original_score: float
    perturbed_score: float
    score_delta: float
    robustness: float  # 0-1 (how robust to this perturbation)
    passed: bool  # Did it stay above threshold


@dataclass
class RobustnessReport:
    """Complete robustness test report."""
    original_score: float
    perturbation_tests: Dict[str, PerturbationTest]
    overall_robustness: float  # 0-1 (average across all perturbations)
    robustness_level: str  # "very_high", "high", "medium", "low", "very_low"
    vulnerable_areas: List[str]  # What types of perturbations hurt most
    is_robust: bool  # Passes robustness criteria


class PromptPerturbationEngine(nn.Module):
    """Generates adversarial prompt variations."""

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        # Perturbation strength predictor
        self.perturbation_generator = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )

        # Perturbation types and their templates
        self.perturbations = {
            "synonym_swap": [
                ("beautiful", "stunning"),
                ("big", "large"),
                ("small", "tiny"),
                ("bright", "luminous"),
                ("dark", "shadowy"),
            ],
            "negation_injection": [
                ("is", "is not"),
                ("with", "without"),
                ("includes", "excludes"),
            ],
            "magnitude_shift": [
                ("very", ""),
                ("extremely", "slightly"),
                ("slightly", "very"),
            ],
            "aspect_swap": [
                ("color", "shape"),
                ("lighting", "composition"),
                ("texture", "detail"),
            ],
            "abstraction_increase": [
                ("photorealistic", "artistic"),
                ("detailed", "simple"),
                ("precise", "rough"),
            ],
            "constraint_relaxation": [
                ("must be", "should be"),
                ("exactly", "approximately"),
                ("strict", "loose"),
            ],
        }

    def generate_perturbations(
        self,
        original_prompt: str,
    ) -> Dict[str, str]:
        """Generate prompt variations."""
        perturbations = {}

        # Synonym swaps
        swapped = original_prompt
        for original, synonym in self.perturbations["synonym_swap"][:3]:
            if original.lower() in swapped.lower():
                swapped = swapped.replace(original, synonym, 1)
                break
        if swapped != original_prompt:
            perturbations["synonym_swap"] = swapped
        else:
            perturbations["synonym_swap"] = original_prompt + " with added detail"

        # Negation injection
        if "without" not in original_prompt and "not" not in original_prompt:
            negated = original_prompt.replace("with", "without", 1)
            if negated == original_prompt:
                negated = original_prompt + " without artifacts"
            perturbations["negation"] = negated

        # Magnitude shift (add "very" or remove it)
        if "very" not in original_prompt:
            magnitude = "very " + original_prompt
        else:
            magnitude = original_prompt.replace("very ", "", 1)
        perturbations["magnitude_shift"] = magnitude

        # Abstraction increase
        abstract = original_prompt
        for original, abstract_term in self.perturbations["abstraction_increase"][:2]:
            if original in abstract:
                abstract = abstract.replace(original, abstract_term)
                break
        if abstract == original_prompt:
            abstract = original_prompt + " in artistic style"
        perturbations["abstraction"] = abstract

        # Constraint relaxation
        relaxed = original_prompt
        for strict, loose in self.perturbations["constraint_relaxation"][:2]:
            if strict in relaxed:
                relaxed = relaxed.replace(strict, loose)
                break
        if relaxed == original_prompt:
            relaxed = original_prompt + " approximately"
        perturbations["constraint_relaxation"] = relaxed

        return perturbations


class RobustnessEvaluator(nn.Module):
    """Evaluates robustness of generation to perturbations."""

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        # Semantic drift detector
        self.drift_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Stability scorer
        self.stability_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        original_embedding: torch.Tensor,
        perturbed_embedding: torch.Tensor,
        original_score: float,
    ) -> Tuple[float, float]:
        """Evaluate robustness to perturbation."""
        # Detect semantic drift
        combined = torch.cat([original_embedding, perturbed_embedding], dim=-1)
        drift = float(self.drift_detector(combined).squeeze())

        # Score stability (how stable the generation is)
        stability = float(self.stability_scorer(perturbed_embedding).squeeze())

        # Robustness = original_score * (1 - drift) * stability
        robustness = original_score * (1.0 - drift) * stability

        return robustness, drift


class AdversarialRobustnessSystem:
    """Complete adversarial robustness testing system."""

    def __init__(self, hidden_dim: int = 4096):
        self.perturbation_engine = PromptPerturbationEngine(hidden_dim)
        self.robustness_evaluator = RobustnessEvaluator(hidden_dim)

        self.test_history = []
        self.robustness_threshold = 0.80  # Minimum acceptable robustness

    def test_robustness(
        self,
        original_prompt: str,
        original_embedding: torch.Tensor,
        original_score: float,
        embedding_func,
        scoring_func,
    ) -> RobustnessReport:
        """Test generation robustness to prompt variations."""
        perturbations = self.perturbation_engine.generate_perturbations(original_prompt)

        test_results = {}
        robustness_scores = []

        for perturbation_type, perturbed_prompt in perturbations.items():
            try:
                # Encode perturbed prompt
                perturbed_embedding = embedding_func(perturbed_prompt)

                # Score perturbed result
                perturbed_score = scoring_func(perturbed_embedding)

                # Evaluate robustness
                robustness, drift = self.robustness_evaluator(
                    original_embedding,
                    perturbed_embedding,
                    original_score,
                )

                score_delta = original_score - perturbed_score
                passed = robustness >= self.robustness_threshold

                test_results[perturbation_type] = PerturbationTest(
                    perturbation_type=perturbation_type,
                    original_score=original_score,
                    perturbed_score=perturbed_score,
                    score_delta=score_delta,
                    robustness=robustness,
                    passed=passed,
                )

                robustness_scores.append(robustness)

            except Exception as e:
                logger.warning(f"Perturbation test {perturbation_type} failed: {e}")
                continue

        # Calculate overall robustness
        if robustness_scores:
            overall_robustness = sum(robustness_scores) / len(robustness_scores)
        else:
            overall_robustness = original_score

        # Determine robustness level
        if overall_robustness > 0.95:
            robustness_level = "very_high"
        elif overall_robustness > 0.85:
            robustness_level = "high"
        elif overall_robustness > 0.70:
            robustness_level = "medium"
        elif overall_robustness > 0.50:
            robustness_level = "low"
        else:
            robustness_level = "very_low"

        # Identify vulnerable areas
        vulnerable = []
        for perturbation_type, result in test_results.items():
            if not result.passed:
                vulnerable.append(perturbation_type)

        # Overall pass
        is_robust = overall_robustness >= self.robustness_threshold

        report = RobustnessReport(
            original_score=original_score,
            perturbation_tests=test_results,
            overall_robustness=overall_robustness,
            robustness_level=robustness_level,
            vulnerable_areas=vulnerable,
            is_robust=is_robust,
        )

        self.test_history.append(report)
        return report

    def get_robustness_report(self, report: RobustnessReport) -> Dict:
        """Generate detailed robustness report."""
        detailed_report = {
            "original_score": report.original_score,
            "overall_robustness": report.overall_robustness,
            "robustness_level": report.robustness_level,
            "is_robust": report.is_robust,
            "vulnerable_areas": report.vulnerable_areas,
            "perturbation_tests": {},
        }

        for perturbation_type, result in report.perturbation_tests.items():
            detailed_report["perturbation_tests"][perturbation_type] = {
                "original_score": result.original_score,
                "perturbed_score": result.perturbed_score,
                "score_delta": result.score_delta,
                "robustness": result.robustness,
                "passed": result.passed,
            }

        return detailed_report

    def get_robustness_stats(self) -> Dict:
        """Get statistics on robustness testing."""
        if not self.test_history:
            return {"total_tests": 0}

        total = len(self.test_history)
        avg_robustness = sum(r.overall_robustness for r in self.test_history) / total
        robust_count = sum(1 for r in self.test_history if r.is_robust)
        robust_rate = robust_count / total

        perturbation_robustness = {}
        for report in self.test_history:
            for perturbation_type, test in report.perturbation_tests.items():
                if perturbation_type not in perturbation_robustness:
                    perturbation_robustness[perturbation_type] = []
                perturbation_robustness[perturbation_type].append(test.robustness)

        avg_by_type = {
            ptype: sum(scores) / len(scores)
            for ptype, scores in perturbation_robustness.items()
        }

        return {
            "total_tests": total,
            "average_robustness": avg_robustness,
            "robust_generation_rate": robust_rate,
            "robustness_by_perturbation_type": avg_by_type,
        }

    def recommend_improvements(self, report: RobustnessReport) -> List[str]:
        """Recommend improvements based on robustness test."""
        recommendations = []

        if not report.is_robust:
            recommendations.append("Add stricter prompt adherence enforcement")

        if "negation" in report.vulnerable_areas:
            recommendations.append("Improve handling of negation in prompts")

        if "abstraction" in report.vulnerable_areas:
            recommendations.append("Better support for different abstraction levels")

        if "constraint_relaxation" in report.vulnerable_areas:
            recommendations.append("Enforce stricter constraint satisfaction")

        if len(report.vulnerable_areas) > 3:
            recommendations.append("Overall sensitivity too high - increase robustness target")

        return recommendations


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    system = AdversarialRobustnessSystem()

    # Test with mock functions
    original_prompt = "a golden retriever in a meadow"
    original_embedding = torch.randn(1, 4096)
    original_score = 0.92

    def mock_embedding_func(prompt):
        return torch.randn(1, 4096)

    def mock_scoring_func(embedding):
        return torch.rand(1).item() * 0.95

    report = system.test_robustness(
        original_prompt,
        original_embedding,
        original_score,
        mock_embedding_func,
        mock_scoring_func,
    )

    print("=== Adversarial Robustness Report ===\n")
    detailed = system.get_robustness_report(report)

    print(f"Original Score: {detailed['original_score']:.1%}")
    print(f"Overall Robustness: {detailed['overall_robustness']:.1%}")
    print(f"Robustness Level: {detailed['robustness_level']}")
    print(f"Is Robust: {detailed['is_robust']}\n")

    if detailed["vulnerable_areas"]:
        print(f"Vulnerable Areas: {', '.join(detailed['vulnerable_areas'])}\n")

    print("Perturbation Test Results:")
    for ptype, result in detailed["perturbation_tests"].items():
        print(
            f"  {ptype}: {result['robustness']:.1%} "
            f"(delta: {result['score_delta']:+.2%})"
        )

    recommendations = system.recommend_improvements(report)
    if recommendations:
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"  - {rec}")

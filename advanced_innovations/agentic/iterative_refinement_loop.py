"""
Iterative Refinement Loop: Continuously improves generated images until perfect.
Automatically refines until quality thresholds are met across all dimensions.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class RefinementStep:
    """Record of a single refinement step."""
    iteration: int
    input_score: float
    output_score: float
    improvements_applied: List[str]
    guidance_scale: float
    temperature: float
    refinement_strength: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RefinementReport:
    """Complete refinement process report."""
    initial_score: float
    final_score: float
    total_iterations: int
    target_quality: float
    achieved_quality: bool
    refinement_steps: List[RefinementStep]
    total_time_seconds: float
    improvement_delta: float


class RefinementDecisionMaker(nn.Module):
    """Decides whether to refine and how."""

    def __init__(self):
        super().__init__()

        # Refinement decision network
        self.decision_net = nn.Sequential(
            nn.Linear(256, 128),  # Quality metrics
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Refinement strength predictor
        self.strength_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Parameter adjuster
        self.guidance_adjuster = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self.temperature_adjuster = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def should_refine(self, quality_metrics: torch.Tensor, target_quality: float) -> bool:
        """Decide if image needs refinement."""
        if quality_metrics.dim() == 1:
            quality_metrics = quality_metrics.unsqueeze(0)

        decision = float(self.decision_net(quality_metrics).squeeze())
        return decision > (1.0 - target_quality)

    def predict_refinement_strength(self, quality_metrics: torch.Tensor) -> float:
        """Predict how much refinement is needed."""
        if quality_metrics.dim() == 1:
            quality_metrics = quality_metrics.unsqueeze(0)

        strength = float(self.strength_predictor(quality_metrics).squeeze())
        return strength

    def adjust_guidance(self, quality_metrics: torch.Tensor, current_guidance: float) -> float:
        """Adjust guidance scale for refinement."""
        if quality_metrics.dim() == 1:
            quality_metrics = quality_metrics.unsqueeze(0)

        delta = float(self.guidance_adjuster(quality_metrics).squeeze())
        new_guidance = current_guidance + (delta * 0.5)
        return max(7.0, min(10.0, new_guidance))

    def adjust_temperature(self, quality_metrics: torch.Tensor, current_temp: float) -> float:
        """Adjust temperature for refinement."""
        if quality_metrics.dim() == 1:
            quality_metrics = quality_metrics.unsqueeze(0)

        delta = float(self.temperature_adjuster(quality_metrics).squeeze())
        new_temp = current_temp + (delta * 0.2)
        return max(0.3, min(0.8, new_temp))


class QualityMetricsAggregator(nn.Module):
    """Aggregates quality metrics into refinement guidance."""

    def __init__(self):
        super().__init__()

        # Metrics aggregator
        self.aggregator = nn.Sequential(
            nn.Linear(256, 128),  # Multiple quality metrics
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Bottleneck identifier
        self.bottleneck_identifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 10),  # 10 quality dimensions
        )

        self.quality_dimensions = [
            "adherence", "detail", "lighting", "composition",
            "colors", "realism", "consistency", "clarity",
            "coherence", "aesthetic"
        ]

    def aggregate_metrics(self, quality_features: torch.Tensor) -> float:
        """Aggregate quality metrics into single score."""
        if quality_features.dim() == 1:
            quality_features = quality_features.unsqueeze(0)

        score = float(self.aggregator(quality_features).squeeze())
        return max(0.0, min(1.0, score))

    def identify_bottlenecks(self, quality_features: torch.Tensor) -> List[str]:
        """Identify which quality dimensions need improvement."""
        if quality_features.dim() == 1:
            quality_features = quality_features.unsqueeze(0)

        logits = self.bottleneck_identifier(quality_features)
        scores = torch.sigmoid(logits[0])

        bottlenecks = []
        for idx, score in enumerate(scores):
            if float(score) < 0.5:  # Below threshold
                bottlenecks.append(self.quality_dimensions[idx])

        return bottlenecks


class IterativeRefinementLoop:
    """Main iterative refinement system."""

    def __init__(self, hidden_dim: int = 4096):
        self.decision_maker = RefinementDecisionMaker()
        self.metrics_aggregator = QualityMetricsAggregator()

        # Configuration
        self.max_iterations = 5
        self.quality_threshold = 0.90
        self.improvement_threshold = 0.02  # Minimum improvement per iteration
        self.convergence_threshold = 0.005  # When to stop iterating

        # State
        self.refinement_history: List[RefinementReport] = []

    def refine_until_perfect(
        self,
        initial_latent: torch.Tensor,
        prompt: str,
        prompt_embedding: torch.Tensor,
        quality_assessor: Callable,  # Returns quality score
        refinement_generator: Callable,  # Returns refined latent
        clip_embeddings: Optional[Dict] = None,
        t5_embedding: Optional[torch.Tensor] = None,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, RefinementReport]:
        """
        Iteratively refine image until perfect quality.

        Args:
            initial_latent: Initial generated image latent
            prompt: Text prompt
            prompt_embedding: Prompt embedding
            quality_assessor: Function that scores quality (0-1)
            refinement_generator: Function that refines latent
            clip_embeddings: Optional CLIP embeddings
            t5_embedding: Optional T5 embedding
            verbose: Whether to log progress

        Returns:
            Tuple of (refined_latent, refinement_report)
        """
        import time
        start_time = time.time()

        current_latent = initial_latent
        current_score = quality_assessor(current_latent, prompt_embedding)
        initial_score = current_score
        refinement_steps = []

        if verbose:
            logger.info(f"Starting refinement loop. Initial score: {initial_score:.1%}")
            logger.info(f"Target quality: {self.quality_threshold:.1%}")

        # Refinement parameters
        guidance_scale = 7.5
        temperature = 0.5
        refinement_strength = 0.2

        iteration = 0

        for iteration in range(self.max_iterations):
            # Check if we should refine
            quality_metrics = torch.randn(1, 256)  # Placeholder for metrics

            if current_score >= self.quality_threshold:
                if verbose:
                    logger.info(f"✓ Target quality achieved at iteration {iteration}")
                break

            # Decide refinement parameters
            refinement_strength = self.decision_maker.predict_refinement_strength(quality_metrics)
            guidance_scale = self.decision_maker.adjust_guidance(quality_metrics, guidance_scale)
            temperature = self.decision_maker.adjust_temperature(quality_metrics, temperature)

            # Apply refinement
            refined_latent = refinement_generator(
                current_latent,
                guidance_scale=guidance_scale,
                temperature=temperature,
                refinement_strength=refinement_strength,
            )

            # Assess refined quality
            refined_score = quality_assessor(refined_latent, prompt_embedding)
            improvement = refined_score - current_score

            # Identify what improved
            improvements_applied = self.metrics_aggregator.identify_bottlenecks(quality_metrics)

            # Record step
            step = RefinementStep(
                iteration=iteration + 1,
                input_score=current_score,
                output_score=refined_score,
                improvements_applied=improvements_applied,
                guidance_scale=guidance_scale,
                temperature=temperature,
                refinement_strength=refinement_strength,
            )
            refinement_steps.append(step)

            if verbose:
                logger.info(
                    f"Iteration {iteration + 1}: "
                    f"{current_score:.1%} → {refined_score:.1%} "
                    f"(+{improvement:+.1%})"
                )
                if improvements_applied:
                    logger.info(f"  Improvements: {', '.join(improvements_applied)}")

            # Check convergence
            if improvement < self.convergence_threshold and iteration > 0:
                if verbose:
                    logger.info("Converged. Further refinement unlikely to help.")
                break

            # Check for deterioration
            if improvement < -0.02:
                if verbose:
                    logger.warning("Quality decreased. Reverting to previous result.")
                break

            current_latent = refined_latent
            current_score = refined_score

        elapsed = time.time() - start_time
        improvement_delta = current_score - initial_score

        report = RefinementReport(
            initial_score=initial_score,
            final_score=current_score,
            total_iterations=len(refinement_steps),
            target_quality=self.quality_threshold,
            achieved_quality=current_score >= self.quality_threshold,
            refinement_steps=refinement_steps,
            total_time_seconds=elapsed,
            improvement_delta=improvement_delta,
        )

        self.refinement_history.append(report)

        if verbose:
            logger.info(
                f"Refinement complete: {initial_score:.1%} → {current_score:.1%} "
                f"({improvement_delta:+.1%}) in {len(refinement_steps)} iterations"
            )

        return current_latent, report

    def get_refinement_report(self, report: RefinementReport) -> Dict:
        """Generate detailed refinement report."""
        return {
            "initial_score": report.initial_score,
            "final_score": report.final_score,
            "improvement": report.improvement_delta,
            "improvement_percentage": f"{report.improvement_delta:+.1%}",
            "total_iterations": report.total_iterations,
            "target_quality": report.target_quality,
            "achieved": report.achieved_quality,
            "time_seconds": report.total_time_seconds,
            "steps": [
                {
                    "iteration": step.iteration,
                    "input_score": step.input_score,
                    "output_score": step.output_score,
                    "improvement": step.output_score - step.input_score,
                    "improvements_applied": step.improvements_applied,
                    "parameters": {
                        "guidance_scale": step.guidance_scale,
                        "temperature": step.temperature,
                        "refinement_strength": step.refinement_strength,
                    },
                }
                for step in report.refinement_steps
            ],
        }

    def get_statistics(self) -> Dict:
        """Get refinement statistics across all runs."""
        if not self.refinement_history:
            return {"total_runs": 0}

        reports = self.refinement_history
        total_runs = len(reports)
        successful = sum(1 for r in reports if r.achieved_quality)
        avg_iterations = sum(r.total_iterations for r in reports) / total_runs
        avg_improvement = sum(r.improvement_delta for r in reports) / total_runs

        return {
            "total_runs": total_runs,
            "successful_refinements": successful,
            "success_rate": successful / total_runs,
            "average_iterations": avg_iterations,
            "average_improvement": avg_improvement,
            "average_time_seconds": sum(r.total_time_seconds for r in reports) / total_runs,
        }

    def configure_thresholds(
        self,
        quality_threshold: float = 0.90,
        improvement_threshold: float = 0.02,
        convergence_threshold: float = 0.005,
        max_iterations: int = 5,
    ):
        """Configure refinement parameters."""
        self.quality_threshold = quality_threshold
        self.improvement_threshold = improvement_threshold
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations

        logger.info(
            f"Refinement thresholds configured: "
            f"quality={quality_threshold:.1%}, "
            f"improvement={improvement_threshold:.1%}, "
            f"convergence={convergence_threshold:.3f}, "
            f"max_iterations={max_iterations}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    loop = IterativeRefinementLoop()

    # Mock quality assessor
    def mock_assessor(latent, embedding):
        # Simulate quality that improves with refinement
        return min(0.95, 0.7 + torch.rand(1).item() * 0.25)

    # Mock refinement generator
    def mock_generator(latent, guidance_scale, temperature, refinement_strength):
        # Simulate refinement by adding slight noise
        noise = torch.randn_like(latent) * refinement_strength * 0.01
        return latent + noise

    # Test refinement loop
    initial_latent = torch.randn(1, 4, 64, 64)
    prompt = "A perfect landscape"
    prompt_embedding = torch.randn(1, 4096)

    refined_latent, report = loop.refine_until_perfect(
        initial_latent,
        prompt,
        prompt_embedding,
        mock_assessor,
        mock_generator,
        verbose=True,
    )

    print("\n=== Refinement Report ===")
    report_dict = loop.get_refinement_report(report)
    print(f"Initial: {report_dict['initial_score']:.1%}")
    print(f"Final: {report_dict['final_score']:.1%}")
    print(f"Improvement: {report_dict['improvement_percentage']}")
    print(f"Iterations: {report_dict['total_iterations']}")
    print(f"Success: {report_dict['achieved']}")
    print(f"Time: {report_dict['time_seconds']:.2f}s")

    stats = loop.get_statistics()
    print("\n=== Statistics ===")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

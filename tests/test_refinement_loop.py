"""
Tests for iterative refinement loop system.
"""

import pytest
import torch
from innovations.agentic import IterativeRefinementLoop


class TestIterativeRefinementLoop:
    """Test iterative refinement loop."""

    def setup_method(self):
        """Initialize system."""
        self.loop = IterativeRefinementLoop()

    def test_initialization(self):
        """Test system initialization."""
        assert self.loop is not None
        assert self.loop.max_iterations == 5
        assert self.loop.quality_threshold == 0.90
        assert self.loop.decision_maker is not None
        assert self.loop.metrics_aggregator is not None

    def test_basic_refinement(self):
        """Test basic refinement loop."""
        latent = torch.randn(1, 4, 64, 64)
        embedding = torch.randn(1, 4096)
        prompt = "Test prompt"

        call_count = [0]
        quality_scores = [0.70, 0.75, 0.80, 0.85, 0.92]

        def quality_assessor(lat, emb):
            score = quality_scores[min(call_count[0], len(quality_scores) - 1)]
            call_count[0] += 1
            return score

        def refinement_gen(lat, guidance_scale, temperature, refinement_strength):
            return lat + torch.randn_like(lat) * 0.001

        refined_latent, report = self.loop.refine_until_perfect(
            latent,
            prompt,
            embedding,
            quality_assessor,
            refinement_gen,
            verbose=False,
        )

        assert report is not None
        assert report.initial_score == 0.70
        assert report.final_score >= 0.70

    def test_convergence(self):
        """Test convergence detection."""
        latent = torch.randn(1, 4, 64, 64)
        embedding = torch.randn(1, 4096)

        scores = [0.88, 0.89, 0.891, 0.8912]

        def quality_assessor(lat, emb):
            idx = min(len(self.loop.refinement_history), 0)
            return scores[min(idx + 1, len(scores) - 1)]

        def refinement_gen(lat, guidance_scale, temperature, refinement_strength):
            return lat + torch.randn_like(lat) * 0.0001

        refined_latent, report = self.loop.refine_until_perfect(
            latent,
            "Test",
            embedding,
            quality_assessor,
            refinement_gen,
            verbose=False,
        )

        # Should converge early due to small improvements
        assert report.total_iterations <= 5

    def test_quality_threshold_met(self):
        """Test when quality threshold is already met."""
        latent = torch.randn(1, 4, 64, 64)
        embedding = torch.randn(1, 4096)

        def quality_assessor(lat, emb):
            return 0.95  # Already above threshold

        def refinement_gen(lat, guidance_scale, temperature, refinement_strength):
            return lat

        refined_latent, report = self.loop.refine_until_perfect(
            latent,
            "Test",
            embedding,
            quality_assessor,
            refinement_gen,
            verbose=False,
        )

        assert report.achieved_quality
        assert report.total_iterations <= 1

    def test_improvement_tracking(self):
        """Test improvement tracking."""
        latent = torch.randn(1, 4, 64, 64)
        embedding = torch.randn(1, 4096)

        iteration_scores = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
        iter_idx = [0]

        def quality_assessor(lat, emb):
            score = iteration_scores[min(iter_idx[0], len(iteration_scores) - 1)]
            iter_idx[0] += 1
            return score

        def refinement_gen(lat, guidance_scale, temperature, refinement_strength):
            return lat + torch.randn_like(lat) * 0.001

        refined_latent, report = self.loop.refine_until_perfect(
            latent,
            "Test",
            embedding,
            quality_assessor,
            refinement_gen,
            verbose=False,
        )

        assert report.improvement_delta > 0
        assert report.final_score >= report.initial_score

    def test_refinement_parameters(self):
        """Test refinement parameters are tracked."""
        latent = torch.randn(1, 4, 64, 64)
        embedding = torch.randn(1, 4096)

        def quality_assessor(lat, emb):
            return 0.85

        def refinement_gen(lat, guidance_scale, temperature, refinement_strength):
            assert 7.0 <= guidance_scale <= 10.0
            assert 0.3 <= temperature <= 0.8
            assert 0.0 <= refinement_strength <= 1.0
            return lat + torch.randn_like(lat) * refinement_strength * 0.01

        refined_latent, report = self.loop.refine_until_perfect(
            latent,
            "Test",
            embedding,
            quality_assessor,
            refinement_gen,
            verbose=False,
        )

        # Verify parameters were used
        for step in report.refinement_steps:
            assert 7.0 <= step.guidance_scale <= 10.0
            assert 0.3 <= step.temperature <= 0.8

    def test_report_generation(self):
        """Test report generation."""
        latent = torch.randn(1, 4, 64, 64)
        embedding = torch.randn(1, 4096)

        def quality_assessor(lat, emb):
            return 0.92

        def refinement_gen(lat, guidance_scale, temperature, refinement_strength):
            return lat

        refined_latent, report = self.loop.refine_until_perfect(
            latent,
            "Test prompt",
            embedding,
            quality_assessor,
            refinement_gen,
            verbose=False,
        )

        detailed_report = self.loop.get_refinement_report(report)

        assert "initial_score" in detailed_report
        assert "final_score" in detailed_report
        assert "improvement_percentage" in detailed_report
        assert "total_iterations" in detailed_report
        assert "achieved" in detailed_report
        assert "time_seconds" in detailed_report
        assert "steps" in detailed_report

    def test_statistics(self):
        """Test statistics tracking."""
        latent = torch.randn(1, 4, 64, 64)
        embedding = torch.randn(1, 4096)

        for run in range(3):
            def quality_assessor(lat, emb):
                return 0.85 + (run * 0.03)

            def refinement_gen(lat, **kwargs):
                return lat

            self.loop.refine_until_perfect(
                latent,
                f"Test {run}",
                embedding,
                quality_assessor,
                refinement_gen,
                verbose=False,
            )

        stats = self.loop.get_statistics()

        assert stats["total_runs"] == 3
        assert "success_rate" in stats
        assert "average_iterations" in stats
        assert "average_improvement" in stats
        assert "average_time_seconds" in stats

    def test_max_iterations(self):
        """Test maximum iterations limit."""
        latent = torch.randn(1, 4, 64, 64)
        embedding = torch.randn(1, 4096)

        # Set low quality threshold to force refinement
        self.loop.configure_thresholds(quality_threshold=0.999, max_iterations=3)

        iteration_count = [0]

        def quality_assessor(lat, emb):
            return 0.80

        def refinement_gen(lat, guidance_scale, temperature, refinement_strength):
            iteration_count[0] += 1
            return lat + torch.randn_like(lat) * 0.001

        refined_latent, report = self.loop.refine_until_perfect(
            latent,
            "Test",
            embedding,
            quality_assessor,
            refinement_gen,
            verbose=False,
        )

        # Should not exceed max iterations
        assert report.total_iterations <= 3

    def test_configuration(self):
        """Test threshold configuration."""
        self.loop.configure_thresholds(
            quality_threshold=0.95,
            improvement_threshold=0.01,
            convergence_threshold=0.001,
            max_iterations=7,
        )

        assert self.loop.quality_threshold == 0.95
        assert self.loop.improvement_threshold == 0.01
        assert self.loop.convergence_threshold == 0.001
        assert self.loop.max_iterations == 7

    def test_improvement_calculation(self):
        """Test improvement calculation."""
        latent = torch.randn(1, 4, 64, 64)
        embedding = torch.randn(1, 4096)

        initial_quality = 0.70
        final_quality = 0.90

        score_idx = [0]

        def quality_assessor(lat, emb):
            scores = [initial_quality, 0.75, 0.80, 0.85, final_quality]
            idx = min(score_idx[0], len(scores) - 1)
            score_idx[0] += 1
            return scores[idx]

        def refinement_gen(lat, **kwargs):
            return lat + torch.randn_like(lat) * 0.001

        refined_latent, report = self.loop.refine_until_perfect(
            latent,
            "Test",
            embedding,
            quality_assessor,
            refinement_gen,
            verbose=False,
        )

        expected_improvement = final_quality - initial_quality
        assert abs(report.improvement_delta - expected_improvement) < 0.01

    def test_decision_maker(self):
        """Test refinement decision making."""
        from innovations.agentic import RefinementDecisionMaker

        maker = RefinementDecisionMaker()

        quality_metrics = torch.randn(1, 256)

        # Test strength prediction
        strength = maker.predict_refinement_strength(quality_metrics)
        assert 0.0 <= strength <= 1.0

        # Test guidance adjustment
        new_guidance = maker.adjust_guidance(quality_metrics, 7.5)
        assert 7.0 <= new_guidance <= 10.0

        # Test temperature adjustment
        new_temp = maker.adjust_temperature(quality_metrics, 0.5)
        assert 0.3 <= new_temp <= 0.8

    def test_metrics_aggregator(self):
        """Test metrics aggregation."""
        from innovations.agentic import QualityMetricsAggregator

        agg = QualityMetricsAggregator()

        quality_features = torch.randn(1, 256)

        # Test aggregation
        score = agg.aggregate_metrics(quality_features)
        assert 0.0 <= score <= 1.0

        # Test bottleneck identification
        bottlenecks = agg.identify_bottlenecks(quality_features)
        assert isinstance(bottlenecks, list)


class TestRefinementPerformance:
    """Performance tests for refinement loop."""

    def test_refinement_speed(self):
        """Test refinement speed."""
        loop = IterativeRefinementLoop()
        latent = torch.randn(1, 4, 64, 64)
        embedding = torch.randn(1, 4096)

        import time
        start = time.time()

        def quality_assessor(lat, emb):
            return 0.92

        def refinement_gen(lat, **kwargs):
            return lat + torch.randn_like(lat) * 0.0001

        refined, report = loop.refine_until_perfect(
            latent,
            "Test",
            embedding,
            quality_assessor,
            refinement_gen,
            verbose=False,
        )

        elapsed = time.time() - start

        # Should complete quickly
        assert elapsed < 5.0

    def test_memory_efficiency(self):
        """Test memory efficiency of refinement."""
        loop = IterativeRefinementLoop()
        torch.randn(1, 4, 64, 64)
        embedding = torch.randn(1, 4096)

        large_latent = torch.randn(1, 16, 512, 512)

        def quality_assessor(lat, emb):
            return 0.90

        def refinement_gen(lat, **kwargs):
            return lat

        refined, report = loop.refine_until_perfect(
            large_latent,
            "Test",
            embedding,
            quality_assessor,
            refinement_gen,
            verbose=False,
        )

        # Should not crash with large tensors
        assert refined.shape == large_latent.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

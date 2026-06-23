"""
Tests for cutting-edge research-based systems:
- VisionReward (multi-dimensional preference learning)
- PerceptualMetrics (LPIPS, DINO, DreamSim)
- RLHF Agent (reinforcement learning from human feedback)
- Flow Matching Consistency (temporal coherence)
"""

import pytest
import torch
from innovations.agentic import (
    FlowMatchingConsistencySystem,
    PerceptualMetricsSystem,
    RLHFAgent,
    VisionRewardSystem,
)


class TestVisionRewardSystem:
    """Test multi-dimensional vision reward system."""

    def setup_method(self):
        self.system = VisionRewardSystem()

    def test_initialization(self):
        assert self.system is not None
        assert self.system.aesthetic_module is not None
        assert self.system.detail_module is not None

    def test_multi_dimensional_evaluation(self):
        image_features = torch.randn(1, 4096)
        prompt_features = torch.randn(1, 4096)

        reward = self.system.evaluate_image(image_features, prompt_features, user_rating=4.5)

        assert reward is not None
        assert 0 <= reward.overall_score <= 1
        assert reward.aesthetic_quality.score >= 0
        assert reward.detail_richness.score >= 0

    def test_improvement_suggestions(self):
        image_features = torch.randn(1, 4096)
        reward = self.system.evaluate_image(image_features, user_rating=3.0)

        suggestions = self.system.get_improvement_suggestions(reward)
        assert isinstance(suggestions, list)

    def test_detailed_report(self):
        image_features = torch.randn(1, 4096)
        reward = self.system.evaluate_image(image_features)

        report = self.system.get_detailed_report(reward)

        assert "overall_score" in report
        assert "dimensions" in report
        assert "aesthetic_quality" in report["dimensions"]
        assert "suggestions" in report


class TestPerceptualMetricsSystem:
    """Test LPIPS, DINO, DreamSim metrics."""

    def setup_method(self):
        self.system = PerceptualMetricsSystem()

    def test_lpips_metric(self):
        img1 = torch.randn(1, 4096)
        img2 = torch.randn(1, 4096)

        score = self.system.lpips(img1, img2)

        assert 0 <= score <= 1

    def test_dino_metric(self):
        img1 = torch.randn(1, 4096)
        img2 = torch.randn(1, 4096)

        score = self.system.dino(img1, img2)

        assert 0 <= score <= 1

    def test_dreamsim_metric(self):
        img1 = torch.randn(1, 4096)
        img2 = torch.randn(1, 4096)

        score = self.system.dreamsim(img1, img2)

        assert 0 <= score <= 1

    def test_full_evaluation(self):
        reference = torch.randn(1, 4096)
        test_image = torch.randn(1, 4096)

        metrics = self.system.evaluate(reference, test_image)

        assert "lpips" in metrics
        assert "dino" in metrics
        assert "dreamsim" in metrics
        assert "ensemble" in metrics

    def test_ranking(self):
        reference = torch.randn(1, 4096)
        test_images = [torch.randn(1, 4096) for _ in range(3)]

        rankings = self.system.rank_images(reference, test_images)

        assert len(rankings) == 3
        assert rankings[0][1] >= rankings[1][1]

    def test_quality_report(self):
        metrics = {
            "lpips": 0.1,
            "lpips_normalized": 0.9,
            "dino": 0.85,
            "dreamsim": 0.88,
            "ensemble": 0.90,
            "perceptual_agreement": 0.88,
            "human_correlation": "0.9616",
        }

        report = self.system.get_quality_report(metrics)

        assert "overall_perceptual_score" in report
        assert "assessment" in report


class TestRLHFAgent:
    """Test RLHF (Reinforcement Learning from Human Feedback) agent."""

    def setup_method(self):
        self.agent = RLHFAgent()

    def test_initialization(self):
        assert self.agent is not None
        assert self.agent.reward_model is not None
        assert self.agent.policy_optimizer is not None

    def test_record_preference(self):
        img_a = torch.randn(1, 4096)
        img_b = torch.randn(1, 4096)

        self.agent.record_preference(img_a, img_b, preference=0, confidence=0.9)

        assert len(self.agent.preference_history) == 1

    def test_multiple_preferences(self):
        for i in range(10):
            img_a = torch.randn(1, 4096)
            img_b = torch.randn(1, 4096)
            pref = torch.randint(0, 2, (1,)).item()

            self.agent.record_preference(img_a, img_b, preference=pref, confidence=0.8)

        assert len(self.agent.preference_history) == 10

    def test_train_reward_model(self):
        # Record preferences first
        for _ in range(5):
            img_a = torch.randn(1, 4096)
            img_b = torch.randn(1, 4096)
            self.agent.record_preference(img_a, img_b, preference=0)

        result = self.agent.train_reward_model()

        assert "status" in result
        assert result["status"] == "trained" or result["status"] == "Not enough comparisons yet"

    def test_rank_by_preference(self):
        # Record some preferences
        for _ in range(5):
            img_a = torch.randn(1, 4096)
            img_b = torch.randn(1, 4096)
            self.agent.record_preference(img_a, img_b, preference=0)

        test_images = [torch.randn(1, 4096) for _ in range(3)]
        rankings = self.agent.rank_by_preference(test_images)

        assert len(rankings) == 3

    def test_learning_progress(self):
        progress = self.agent.get_learning_progress()

        assert "stage" in progress
        assert "comparisons_collected" in progress

    def test_policy_improvements(self):
        # Record preferences
        for _ in range(8):
            img_a = torch.randn(1, 4096)
            img_b = torch.randn(1, 4096)
            self.agent.record_preference(img_a, img_b, preference=0)

        current_image = torch.randn(1, 4096)
        improvements = self.agent.get_policy_improvements(current_image)

        assert "suggested_adjustments" in improvements


class TestFlowMatchingConsistencySystem:
    """Test flow matching with temporal consistency."""

    def setup_method(self):
        self.system = FlowMatchingConsistencySystem()

    def test_initialization(self):
        assert self.system is not None
        assert self.system.velocity_net is not None
        assert self.system.tpc_module is not None

    def test_velocity_field(self):
        x = torch.randn(1, 4096)
        t = torch.tensor(0.5)

        velocity, consistency = self.system.velocity_net(x, t)

        assert velocity.shape == x.shape
        assert 0 <= consistency <= 1

    def test_temporal_pair_consistency(self):
        x = torch.randn(1, 4096)

        time_pairs = [(0.0, 0.1), (0.5, 0.6), (0.9, 1.0)]
        metrics = self.system.tpc_module(x, time_pairs)

        assert "total_consistency_loss" in metrics
        assert "average_consistency" in metrics

    def test_curriculum_model(self):
        initial_state = torch.randn(1, 4096)

        final_state, gen_metrics = self.system.curriculum_model.generate_with_consistency(
            initial_state,
            num_steps=8,
        )

        assert final_state.shape == initial_state.shape
        assert "average_consistency" in gen_metrics

    def test_full_generation(self):
        initial_state = torch.randn(1, 4096)
        prompt_cond = torch.randn(1, 4096)

        result = self.system.generate_with_temporal_consistency(
            initial_state,
            prompt_conditioning=prompt_cond,
            num_steps=10,
        )

        assert "final_state" in result
        assert "overall_consistency_score" in result

    def test_consistency_report(self):
        initial_state = torch.randn(1, 4096)

        generation = self.system.generate_with_temporal_consistency(
            initial_state,
            num_steps=8,
        )

        report = self.system.get_consistency_report(generation)

        assert "overall_consistency_score" in report
        assert "path_quality" in report


class TestResearchSystemsPerformance:
    """Performance tests for research systems."""

    def test_vision_reward_speed(self):
        system = VisionRewardSystem()
        image_features = torch.randn(1, 4096)

        import time

        start = time.time()

        system.evaluate_image(image_features)

        elapsed = time.time() - start

        # Should be fast
        assert elapsed < 1.0

    def test_metrics_speed(self):
        system = PerceptualMetricsSystem()
        img1 = torch.randn(1, 4096)
        img2 = torch.randn(1, 4096)

        import time

        start = time.time()

        system.evaluate(img1, img2)

        elapsed = time.time() - start

        # All metrics together should complete quickly
        assert elapsed < 2.0

    def test_flow_matching_speed(self):
        system = FlowMatchingConsistencySystem()
        initial = torch.randn(1, 4096)

        import time

        start = time.time()

        system.generate_with_temporal_consistency(initial, num_steps=10)

        elapsed = time.time() - start

        # Flow matching should be fast
        assert elapsed < 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

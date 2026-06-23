"""
Comprehensive tests for agentic systems.
Tests all new agent modules: visual reasoning, adaptive learning, prompt optimization,
ensemble validation, and adversarial robustness.
"""

import pytest
import torch
from innovations.agentic import (
    AdaptiveLearningSystem,
    AdversarialRobustnessSystem,
    EnsembleValidationSystem,
    PromptOptimizationSystem,
    VisualReasoningSystem,
)


class TestVisualReasoningSystem:
    """Test visual reasoning system."""

    def setup_method(self):
        """Initialize system."""
        self.system = VisualReasoningSystem()

    def test_initialization(self):
        """Test system initialization."""
        assert self.system is not None
        assert self.system.agent is not None
        assert len(self.system.reasoning_cache) == 0

    def test_analyze_image(self):
        """Test image analysis."""
        embedding = torch.randn(1, 4096)
        result = self.system.analyze_generated_image(embedding, embedding)

        assert "reasoning" in result
        assert "alignment_with_intent" in result
        assert "scene_description" in result

        assert 0 <= result["alignment_with_intent"] <= 1

    def test_scene_description(self):
        """Test scene description generation."""
        embedding = torch.randn(1, 4096)
        result = self.system.analyze_generated_image(embedding, embedding)
        description = result["scene_description"]

        assert isinstance(description, str)
        assert len(description) > 0

    def test_consistency_validation(self):
        """Test visual consistency validation."""
        embedding1 = torch.randn(1, 4096)
        embedding2 = torch.randn(1, 4096)

        reasoning1 = self.system.agent(embedding1)
        reasoning2 = self.system.agent(embedding2)

        consistency = self.system.validate_visual_consistency(reasoning1, reasoning2)

        assert 0 <= consistency <= 1


class TestAdaptiveLearningSystem:
    """Test adaptive learning system."""

    def setup_method(self):
        """Initialize system."""
        self.system = AdaptiveLearningSystem()

    def test_initialization(self):
        """Test system initialization."""
        assert self.system is not None
        assert self.system.preference_learner is not None
        assert self.system.parameter_optimizer is not None
        assert len(self.system.feedback_buffer.feedbacks) == 0

    def test_add_feedback(self):
        """Test adding feedback."""
        features = torch.randn(1, 4096)

        self.system.add_generation_feedback(
            prompt="Test prompt",
            generated_features=features,
            user_rating=4.5,
            quality_score=0.88,
            adherence_score=0.90,
        )

        assert len(self.system.feedback_buffer.feedbacks) == 1

    def test_learned_parameters(self):
        """Test parameter learning."""
        features = torch.randn(1, 4096)

        for i in range(3):
            self.system.add_generation_feedback(
                prompt=f"Test {i}",
                generated_features=features,
                user_rating=3.5 + (i * 0.5),
                quality_score=0.8 + (i * 0.05),
                adherence_score=0.85 + (i * 0.03),
            )

        params = self.system.get_adaptive_parameters()

        assert "guidance_scale" in params
        assert "temperature" in params
        assert "refinement_strength" in params
        assert 7.0 <= params["guidance_scale"] <= 10.0

    def test_learning_progress(self):
        """Test learning progress tracking."""
        features = torch.randn(1, 4096)

        for i in range(5):
            self.system.add_generation_feedback(
                prompt=f"Test {i}",
                generated_features=features,
                user_rating=4.0 + (i * 0.2),
                quality_score=0.85 + (i * 0.03),
                adherence_score=0.80 + (i * 0.02),
            )

        progress = self.system.get_learning_progress()

        assert progress["total_feedbacks"] == 5
        assert progress["high_quality_samples"] > 0
        assert 0 <= progress["high_quality_ratio"] <= 1


class TestPromptOptimizationSystem:
    """Test prompt optimization system."""

    def setup_method(self):
        """Initialize system."""
        self.system = PromptOptimizationSystem()

    def test_initialization(self):
        """Test system initialization."""
        assert self.system is not None
        assert self.system.analyzer is not None
        assert self.system.enhancer is not None
        assert self.system.expander is not None

    def test_analyze_prompt(self):
        """Test prompt analysis."""
        prompt = "a dog"
        embedding = torch.randn(1, 4096)

        analysis = self.system.analyze_prompt(prompt, embedding)

        assert analysis.original_prompt == prompt
        assert 0 <= analysis.coverage_score <= 1
        assert 0 <= analysis.vagueness_score <= 1
        assert 0 <= analysis.specificity_score <= 1

    def test_enhance_prompt(self):
        """Test prompt enhancement."""
        prompt = "a dog"
        embedding = torch.randn(1, 4096)

        enhanced = self.system.enhance_prompt(prompt, embedding)

        assert isinstance(enhanced, str)
        # Enhanced should have more words
        assert len(enhanced.split()) >= len(prompt.split())

    def test_expand_prompt(self):
        """Test prompt expansion."""
        prompt = "a bird"
        embedding = torch.randn(1, 4096)

        expanded = self.system.expand_prompt(prompt, embedding)

        assert isinstance(expanded, str)
        assert len(expanded) > 0

    def test_full_optimization(self):
        """Test complete optimization pipeline."""
        prompt = "a cat"
        embedding = torch.randn(1, 4096)

        optimization = self.system.optimize_prompt(prompt, embedding)

        assert "original" in optimization
        assert "optimized" in optimization
        assert optimization["original"] == prompt
        assert len(optimization["optimized"]) > len(prompt)


class TestEnsembleValidator:
    """Test ensemble validation system."""

    def setup_method(self):
        """Initialize system."""
        self.system = EnsembleValidationSystem()

    def test_initialization(self):
        """Test system initialization."""
        assert self.system is not None
        assert self.system.semantic_validator is not None
        assert self.system.detail_validator is not None
        assert self.system.aesthetic_validator is not None
        assert self.system.realistic_validator is not None

    def test_validation(self):
        """Test full validation."""
        prompt_emb = torch.randn(1, 4096)
        generated_emb = torch.randn(1, 4096)

        result = self.system.validate(prompt_emb, generated_emb)

        assert result is not None
        assert 0 <= result.overall_score <= 1
        assert 0 <= result.confidence <= 1
        assert result.consensus_level in ["perfect", "strong", "moderate", "weak"]

    def test_validator_scores(self):
        """Test individual validator scores."""
        prompt_emb = torch.randn(1, 4096)
        generated_emb = torch.randn(1, 4096)

        result = self.system.validate(prompt_emb, generated_emb)

        validators = result.validator_scores
        assert "semantic" in validators
        assert "detail" in validators
        assert "aesthetic" in validators
        assert "realistic" in validators

        for validator in validators.values():
            assert 0 <= validator.score <= 1

    def test_validation_report(self):
        """Test validation report generation."""
        prompt_emb = torch.randn(1, 4096)
        generated_emb = torch.randn(1, 4096)

        result = self.system.validate(prompt_emb, generated_emb)
        report = self.system.get_validator_report(result)

        assert "overall_score" in report
        assert "confidence" in report
        assert "recommendation" in report
        assert "validators" in report


class TestAdversarialRobustness:
    """Test adversarial robustness system."""

    def setup_method(self):
        """Initialize system."""
        self.system = AdversarialRobustnessSystem()

    def test_initialization(self):
        """Test system initialization."""
        assert self.system is not None
        assert self.system.perturbation_engine is not None
        assert self.system.robustness_evaluator is not None

    def test_perturbation_generation(self):
        """Test perturbation generation."""
        prompt = "a beautiful golden retriever in a meadow"

        perturbations = self.system.perturbation_engine.generate_perturbations(prompt)

        assert isinstance(perturbations, dict)
        assert len(perturbations) > 0
        # Perturbations should be different from original
        for pert_type, pert_prompt in perturbations.items():
            assert pert_prompt != prompt

    def test_robustness_testing(self):
        """Test robustness testing."""
        prompt = "a scenic landscape"
        embedding = torch.randn(1, 4096)
        original_score = 0.88

        def mock_embedding(p):
            return torch.randn(1, 4096)

        def mock_score(e):
            return torch.rand(1).item() * 0.95

        report = self.system.test_robustness(
            prompt,
            embedding,
            original_score,
            mock_embedding,
            mock_score,
        )

        assert report is not None
        assert 0 <= report.overall_robustness <= 1
        assert report.robustness_level in ["very_high", "high", "medium", "low", "very_low"]

    def test_robustness_report(self):
        """Test robustness report generation."""
        prompt = "a landscape"
        embedding = torch.randn(1, 4096)
        original_score = 0.90

        def mock_embedding(p):
            return torch.randn(1, 4096)

        def mock_score(e):
            return 0.85 + torch.rand(1).item() * 0.1

        report = self.system.test_robustness(
            prompt,
            embedding,
            original_score,
            mock_embedding,
            mock_score,
        )

        detailed_report = self.system.get_robustness_report(report)

        assert "original_score" in detailed_report
        assert "overall_robustness" in detailed_report
        assert "robustness_level" in detailed_report
        assert "vulnerable_areas" in detailed_report


class TestIntegrationWithPipeline:
    """Test integration of agentic systems with main pipeline."""

    def test_all_systems_accessible_from_pipeline(self):
        """Test that all agentic systems are accessible."""
        from innovations.pipeline import create_advanced_pipeline

        pipeline = create_advanced_pipeline(enable_all=True)
        status = pipeline.get_status()

        # Check that all new systems are available
        assert status.get("visual_reasoning", False)
        assert status.get("adaptive_learning", False)
        assert status.get("prompt_optimization", False)
        assert status.get("ensemble_validator", False)
        assert status.get("robustness_testing", False)


class TestPerformance:
    """Performance and stress tests."""

    def test_visual_reasoning_performance(self):
        """Test visual reasoning speed."""
        system = VisualReasoningSystem()
        embedding = torch.randn(1, 4096)

        import time

        start = time.time()
        system.analyze_generated_image(embedding, embedding)
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 1.0  # 1 second max

    def test_adaptive_learning_batch_feedback(self):
        """Test adaptive learning with multiple feedback."""
        system = AdaptiveLearningSystem()

        for i in range(20):
            features = torch.randn(1, 4096)
            system.add_generation_feedback(
                prompt=f"Prompt {i}",
                generated_features=features,
                user_rating=3.0 + (i % 3),
                quality_score=0.8 + (i % 20) * 0.01,
                adherence_score=0.75 + (i % 20) * 0.01,
            )

        # Should handle buffer management
        assert len(system.feedback_buffer.feedbacks) <= 20

    def test_ensemble_validator_consistency(self):
        """Test ensemble validator consistency."""
        system = EnsembleValidationSystem()

        # Run multiple validations with same inputs
        embedding_p = torch.randn(1, 4096)
        embedding_g = torch.randn(1, 4096)

        results = []
        for _ in range(3):
            result = system.validate(embedding_p, embedding_g)
            results.append(result.overall_score)

        # Scores should be identical (deterministic for same input)
        assert all(abs(r - results[0]) < 0.01 for r in results)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_feedback_buffer(self):
        """Test system with no feedback."""
        system = AdaptiveLearningSystem()

        progress = system.get_learning_progress()
        assert progress["total_feedbacks"] == 0

    def test_single_feedback(self):
        """Test system with single feedback."""
        system = AdaptiveLearningSystem()

        features = torch.randn(1, 4096)
        system.add_generation_feedback(
            prompt="Test",
            generated_features=features,
            user_rating=5.0,
            quality_score=1.0,
            adherence_score=1.0,
        )

        params = system.get_adaptive_parameters()
        assert len(params) > 0

    def test_extreme_ratings(self):
        """Test with extreme user ratings."""
        system = AdaptiveLearningSystem()

        features = torch.randn(1, 4096)

        # Very high rating
        system.add_generation_feedback(
            prompt="Excellent",
            generated_features=features,
            user_rating=5.0,
            quality_score=0.99,
            adherence_score=0.99,
        )

        # Very low rating
        system.add_generation_feedback(
            prompt="Poor",
            generated_features=features,
            user_rating=1.0,
            quality_score=0.2,
            adherence_score=0.15,
        )

        progress = system.get_learning_progress()
        assert progress["total_feedbacks"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

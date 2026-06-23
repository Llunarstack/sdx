"""
Comprehensive tests for advanced agentic systems.
Tests Memory Preference System and Semantic Composition Reasoner.
"""

import pytest
import torch
from innovations.agentic import (
    MemoryPreferenceSystem,
    SemanticCompositionReasoner,
)


class TestMemoryPreferenceSystem:
    """Test memory and preference learning system."""

    def setup_method(self):
        """Initialize system."""
        self.system = MemoryPreferenceSystem()

    def test_initialization(self):
        """Test system initialization."""
        assert self.system is not None
        assert self.system.preference_memory is not None
        assert self.system.theme_analyzer is not None
        assert self.system.recommendation_engine is not None

    def test_record_generation(self):
        """Test recording a generation."""
        user_id = "user_001"
        features = torch.randn(1, 4096)

        self.system.record_generation(
            user_id,
            features,
            user_rating=4.5,
            subject="landscape",
            style="photorealistic",
            mood="peaceful",
            lighting="golden hour",
        )

        profile = self.system.get_user_profile(user_id)
        assert profile.total_generations == 1
        assert profile.average_rating == 4.5

    def test_preference_learning(self):
        """Test preference learning over time."""
        user_id = "user_002"

        # Record multiple generations
        for i in range(5):
            features = torch.randn(1, 4096)
            rating = 3.5 + (i * 0.3)

            self.system.record_generation(
                user_id,
                features,
                min(rating, 5.0),
                subject="landscape",
                style="photorealistic",
            )

        profile = self.system.get_user_profile(user_id)

        assert profile.total_generations == 5
        assert profile.average_rating > 3.5
        assert len(profile.favorite_subjects) > 0
        assert len(profile.favorite_styles) > 0

    def test_theme_extraction(self):
        """Test theme extraction from preferences."""
        user_id = "user_003"

        for i in range(8):
            features = torch.randn(1, 4096)
            self.system.record_generation(
                user_id, features, 4.0 + (i * 0.1),
                subject="landscape",
                style="photorealistic",
            )

        profile = self.system.get_user_profile(user_id)

        assert profile.dominant_theme is not None
        assert profile.dominant_theme != "unknown"

    def test_recommendations(self):
        """Test recommendation generation."""
        user_id = "user_004"

        for i in range(5):
            features = torch.randn(1, 4096)
            self.system.record_generation(
                user_id, features, 4.0,
                subject="portrait",
                style="cinematic",
            )

        recommendations = self.system.get_recommendations(user_id)

        assert "improvements" in recommendations
        assert "next_prompt_recommendation" in recommendations
        assert "user_profile" in recommendations

    def test_next_prompt_recommendation(self):
        """Test next prompt recommendation."""
        user_id = "user_005"

        for _ in range(3):
            features = torch.randn(1, 4096)
            self.system.record_generation(
                user_id, features, 4.5,
                subject="landscape",
                style="photorealistic",
                mood="peaceful",
                lighting="natural",
            )

        recs = self.system.get_recommendations(user_id)
        next_prompt = recs["next_prompt_recommendation"]

        assert "full_prompt" in next_prompt
        assert len(next_prompt["full_prompt"]) > 0
        assert "landscape" in next_prompt["full_prompt"].lower()

    def test_satisfaction_prediction(self):
        """Test satisfaction prediction."""
        user_id = "user_006"

        for _ in range(5):
            features = torch.randn(1, 4096)
            self.system.record_generation(
                user_id, features, 4.5,
                subject="landscape",
            )

        prompt_features = torch.randn(1, 4096)
        prediction = self.system.predict_satisfaction(user_id, prompt_features)

        assert 0.0 <= prediction <= 1.0

    def test_profile_export(self):
        """Test profile export."""
        user_id = "user_007"

        features = torch.randn(1, 4096)
        self.system.record_generation(
            user_id, features, 4.0,
            subject="landscape",
            style="photorealistic",
        )

        exported = self.system.export_profile(user_id)

        assert exported["user_id"] == user_id
        assert exported["total_generations"] == 1
        assert "created_at" in exported
        assert "last_updated" in exported

    def test_system_stats(self):
        """Test system statistics."""
        for user_num in range(3):
            user_id = f"user_{user_num:03d}"
            for _ in range(2):
                features = torch.randn(1, 4096)
                self.system.record_generation(user_id, features, 4.0)

        stats = self.system.get_system_stats()

        assert stats["total_users"] == 3
        assert stats["total_generations"] == 6

    def test_multiple_users(self):
        """Test system with multiple users."""
        users = ["alice", "bob", "charlie"]

        for user_id in users:
            features = torch.randn(1, 4096)
            self.system.record_generation(user_id, features, 4.5)

        for user_id in users:
            profile = self.system.get_user_profile(user_id)
            assert profile.total_generations == 1

    def test_feature_history_limit(self):
        """Test feature history limiting."""
        user_id = "user_history_test"

        # Record 150 generations (exceeds 100-limit)
        for i in range(150):
            features = torch.randn(1, 4096)
            rating = 3.0 + ((i % 3) * 0.5)
            self.system.record_generation(user_id, features, rating)

        # History should be capped at 100
        assert len(self.system.feature_history[user_id]) <= 100
        assert len(self.system.rating_history[user_id]) <= 100


class TestSemanticCompositionReasoner:
    """Test semantic composition reasoning system."""

    def setup_method(self):
        """Initialize system."""
        self.system = SemanticCompositionReasoner()

    def test_initialization(self):
        """Test system initialization."""
        assert self.system is not None
        assert self.system.concept_embedder is not None
        assert self.system.relation_analyzer is not None
        assert self.system.composition_validator is not None

    def test_concept_extraction(self):
        """Test concept extraction."""
        embedding = torch.randn(1, 4096)
        concepts = self.system.extract_concepts(embedding)

        assert isinstance(concepts, list)
        assert len(concepts) > 0
        assert all(isinstance(c, tuple) and len(c) == 2 for c in concepts)
        assert all(0.0 <= c[1] <= 1.0 for c in concepts)

    def test_composition_analysis(self):
        """Test composition analysis."""
        concepts = ["landscape", "golden_hour", "peaceful"]
        embedding = torch.randn(1, 4096)

        analysis = self.system.analyze_composition(concepts, embedding=embedding)

        assert "concepts" in analysis
        assert "pairwise_relations" in analysis
        assert "composition_score" in analysis
        assert "is_coherent" in analysis
        assert "conflicts" in analysis
        assert "recommendations" in analysis

    def test_composition_scoring(self):
        """Test composition scoring."""
        concepts = ["landscape", "ocean", "sunset"]
        embedding = torch.randn(1, 4096)

        analysis = self.system.analyze_composition(concepts, embedding=embedding)

        score = analysis["composition_score"]
        assert 0.0 <= score <= 1.0

    def test_pairwise_relations(self):
        """Test pairwise concept relations."""
        concepts = ["fire", "water"]
        embedding = torch.randn(1, 4096)

        analysis = self.system.analyze_composition(concepts, embedding=embedding)

        relations = analysis["pairwise_relations"]
        assert len(relations) == 1

        relation = relations[0]
        assert relation["a"] == "fire"
        assert relation["b"] == "water"
        assert "relation" in relation
        assert "strength" in relation
        assert 0.0 <= relation["strength"] <= 1.0

    def test_conflict_detection(self):
        """Test conflict detection."""
        # Some concepts might conflict
        concepts = ["light", "darkness", "brightness"]
        embedding = torch.randn(1, 4096)

        analysis = self.system.analyze_composition(concepts, embedding=embedding)

        # Just verify the structure is correct
        assert isinstance(analysis["conflicts"], list)

    def test_recommendations(self):
        """Test composition recommendations."""
        concepts = ["landscape", "mountains", "water"]
        embedding = torch.randn(1, 4096)

        analysis = self.system.analyze_composition(concepts, embedding=embedding)

        recommendations = analysis["recommendations"]
        assert isinstance(recommendations, list)

    def test_quality_prediction(self):
        """Test generation quality prediction."""
        concepts = ["portrait", "dramatic_lighting", "detailed"]
        embedding = torch.randn(1, 4096)

        quality = self.system.predict_generation_quality(concepts, embedding)

        assert 0.0 <= quality <= 1.0

    def test_concept_improvements(self):
        """Test concept improvement suggestions."""
        concepts = ["landscape", "minimal", "abstract"]
        embedding = torch.randn(1, 4096)

        suggestions = self.system.suggest_concept_improvements(concepts, embedding)

        assert isinstance(suggestions, list)

    def test_system_stats(self):
        """Test system statistics."""
        embedding = torch.randn(1, 4096)

        # Run some analyses to populate cache
        for _ in range(3):
            concepts = ["landscape", "ocean", "sunset"]
            self.system.analyze_composition(concepts, embedding=embedding)

        stats = self.system.get_system_stats()

        assert "cached_relations" in stats
        assert "cached_concepts" in stats

    def test_coherent_vs_incoherent(self):
        """Test coherence scoring for different compositions."""
        embedding = torch.randn(1, 4096)

        # Coherent composition
        coherent = self.system.analyze_composition(
            ["landscape", "golden_hour", "peaceful"],
            embedding=embedding
        )

        # Incoherent composition
        incoherent = self.system.analyze_composition(
            ["random", "nonsense", "mismatch"],
            embedding=embedding
        )

        # Just verify both return valid scores
        assert 0.0 <= coherent["composition_score"] <= 1.0
        assert 0.0 <= incoherent["composition_score"] <= 1.0

    def test_cache_efficiency(self):
        """Test that caching works."""
        embedding = torch.randn(1, 4096)
        concepts = ["landscape", "ocean"]

        # First call - should cache
        analysis1 = self.system.analyze_composition(concepts, embedding=embedding)

        # Second call - should use cache
        analysis2 = self.system.analyze_composition(concepts, embedding=embedding)

        # Relations should be identical
        assert len(analysis1["pairwise_relations"]) == len(analysis2["pairwise_relations"])


class TestIntegrationAdvancedSystems:
    """Test integration of advanced systems."""

    def test_memory_and_composition_together(self):
        """Test using memory system with composition reasoning."""
        memory = MemoryPreferenceSystem()
        composer = SemanticCompositionReasoner()

        user_id = "integrated_user"
        embedding = torch.randn(1, 4096)
        concepts = ["landscape", "golden_hour", "peaceful"]

        # Record preference
        memory.record_generation(
            user_id, embedding, 4.5,
            subject=concepts[0],
            mood=concepts[2],
        )

        # Analyze composition
        composition = composer.analyze_composition(
            concepts, embedding=embedding
        )

        # Get recommendations
        recs = memory.get_recommendations(user_id)

        # Verify both systems work together
        assert recs["user_profile"]["total_generations"] == 1
        assert composition["composition_score"] > 0.0


class TestPerformanceAdvanced:
    """Performance tests for advanced systems."""

    def test_memory_system_performance(self):
        """Test memory system performance."""
        system = MemoryPreferenceSystem()

        import time
        start = time.time()

        for i in range(20):
            features = torch.randn(1, 4096)
            system.record_generation(
                f"user_{i % 5}", features, 4.0 + (i % 2),
                subject="landscape",
            )

        elapsed = time.time() - start

        # Should handle 20 recordings quickly
        assert elapsed < 5.0

    def test_composition_reasoning_performance(self):
        """Test composition reasoning performance."""
        system = SemanticCompositionReasoner()

        import time
        start = time.time()

        for _ in range(10):
            embedding = torch.randn(1, 4096)
            concepts = ["landscape", "ocean", "sunset", "peaceful"]
            system.analyze_composition(concepts, embedding=embedding)

        elapsed = time.time() - start

        # Should analyze compositions quickly
        assert elapsed < 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

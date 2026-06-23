"""
Semantic Composition Reasoner: Understands how visual concepts combine and interact.
Enables logical reasoning about multi-concept generations and concept compatibility.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ConceptRelation:
    """Relationship between two concepts."""

    concept_a: str
    concept_b: str
    relation_type: str  # "enhances", "conflicts", "neutral", "supports"
    strength: float  # 0-1 compatibility score
    reasoning: str


class ConceptEmbedder(nn.Module):
    """Embeds visual concepts into semantic space."""

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        self.concept_embedder = nn.Sequential(
            nn.Linear(hidden_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )

        # Concept classifier for common visual concepts
        self.concept_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 50),  # 50 common concepts
        )

        self.concept_names = [
            "person",
            "animal",
            "building",
            "nature",
            "water",
            "sky",
            "fire",
            "light",
            "shadow",
            "color",
            "texture",
            "pattern",
            "geometric",
            "organic",
            "abstract",
            "realistic",
            "surreal",
            "fantastical",
            "dark",
            "bright",
            "calm",
            "chaotic",
            "ordered",
            "wild",
            "peaceful",
            "energetic",
            "static",
            "dynamic",
            "soft",
            "sharp",
            "warm",
            "cold",
            "wet",
            "dry",
            "transparent",
            "opaque",
            "smooth",
            "rough",
            "detailed",
            "simple",
            "minimalist",
            "maximalist",
            "vintage",
            "modern",
            "ancient",
            "futuristic",
            "natural",
            "artificial",
            "organic",
            "mechanical",
            "beautiful",
            "ugly",
            "elegant",
            "crude",
            "balanced",
        ]

    def embed_concept(self, embedding: torch.Tensor) -> torch.Tensor:
        """Embed concept representation."""
        return self.concept_embedder(embedding)

    def extract_concepts(
        self,
        embedding: torch.Tensor,
        threshold: float = 0.3,
    ) -> List[Tuple[str, float]]:
        """Extract dominant concepts from embedding."""
        features = self.concept_embedder(embedding)
        logits = self.concept_classifier(features)
        scores = torch.sigmoid(logits[0]).detach()

        concepts = []
        for idx, score in enumerate(scores):
            if float(score) > threshold:
                concepts.append((self.concept_names[idx], float(score)))

        # Sort by confidence
        concepts.sort(key=lambda x: x[1], reverse=True)
        return concepts[:10]  # Return top 10


class ConceptRelationAnalyzer(nn.Module):
    """Analyzes relationships between concepts."""

    def __init__(self):
        super().__init__()

        # Relation classifier
        self.relation_classifier = nn.Sequential(
            nn.Linear(256 * 2, 256),  # Two concept embeddings
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 4),  # 4 relation types: enhances, conflicts, neutral, supports
        )

        # Compatibility scorer
        self.compatibility_scorer = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.relation_types = ["enhances", "conflicts", "neutral", "supports"]

    def analyze_relation(
        self,
        concept_a_emb: torch.Tensor,
        concept_b_emb: torch.Tensor,
        concept_a_name: str,
        concept_b_name: str,
    ) -> ConceptRelation:
        """Analyze relationship between two concepts."""
        # Combine embeddings
        combined = torch.cat([concept_a_emb, concept_b_emb], dim=-1)

        # Classify relation type
        relation_logits = self.relation_classifier(combined)
        relation_idx = int(torch.argmax(relation_logits[0]))
        relation_type = self.relation_types[relation_idx]

        # Score compatibility
        compatibility = float(self.compatibility_scorer(combined).squeeze().detach())

        # Reasoning
        reasoning_map = {
            "enhances": f"{concept_a_name} and {concept_b_name} work well together, each strengthening the other",
            "conflicts": f"{concept_a_name} and {concept_b_name} are in tension; combining them may reduce clarity",
            "neutral": f"{concept_a_name} and {concept_b_name} can coexist independently without strong interaction",
            "supports": f"{concept_a_name} supports {concept_b_name}; together they create a coherent composition",
        }

        return ConceptRelation(
            concept_a=concept_a_name,
            concept_b=concept_b_name,
            relation_type=relation_type,
            strength=compatibility,
            reasoning=reasoning_map[relation_type],
        )


class CompositionValidator(nn.Module):
    """Validates multi-concept compositions."""

    def __init__(self):
        super().__init__()

        # Composition coherence scorer
        self.coherence_scorer = nn.Sequential(
            nn.Linear(256 * 5, 256),  # Up to 5 concepts
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Conflict detector
        self.conflict_detector = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def score_composition(self, concept_embeddings: List[torch.Tensor]) -> float:
        """Score how well concepts compose together."""
        if len(concept_embeddings) == 0:
            return 0.5

        # Pad to 5 concepts
        padded = concept_embeddings[:5]
        while len(padded) < 5:
            padded.append(torch.zeros_like(padded[0]))

        combined = torch.cat(padded, dim=-1)
        score = float(self.coherence_scorer(combined).squeeze().detach())
        return score

    def detect_conflicts(
        self,
        concepts: List[str],
        concept_relations: List[ConceptRelation],
    ) -> List[str]:
        """Detect conflicting concepts."""
        conflicts = []

        for relation in concept_relations:
            if relation.relation_type == "conflicts" and relation.strength < 0.4:
                conflicts.append(
                    f"'{relation.concept_a}' conflicts with '{relation.concept_b}' "
                    f"(compatibility: {relation.strength:.1%})"
                )

        return conflicts


class SemanticCompositionReasoner:
    """Unified semantic composition reasoning system."""

    def __init__(self, hidden_dim: int = 4096):
        self.concept_embedder = ConceptEmbedder(hidden_dim)
        self.relation_analyzer = ConceptRelationAnalyzer()
        self.composition_validator = CompositionValidator()

        self.concept_cache: Dict[str, torch.Tensor] = {}
        self.relation_cache: Dict[Tuple[str, str], ConceptRelation] = {}

    def extract_concepts(self, embedding: torch.Tensor) -> List[Tuple[str, float]]:
        """Extract concepts from embedding."""
        return self.concept_embedder.extract_concepts(embedding)

    def analyze_composition(
        self,
        concepts: List[str],
        concept_embeddings: Optional[List[torch.Tensor]] = None,
        embedding: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Analyze multi-concept composition."""
        if embedding is not None and concept_embeddings is None:
            # Extract concepts from embedding
            concept_embeddings = [self.concept_embedder.embed_concept(embedding) for _ in range(len(concepts))]

        if not concept_embeddings:
            concept_embeddings = [torch.zeros(1, 256) for _ in concepts]

        # Analyze pairwise relations
        relations = []
        for i, c_a in enumerate(concepts):
            for j, c_b in enumerate(concepts[i + 1 :], i + 1):
                cache_key = tuple(sorted([c_a, c_b]))
                if cache_key not in self.relation_cache:
                    relation = self.relation_analyzer.analyze_relation(
                        concept_embeddings[i],
                        concept_embeddings[j],
                        c_a,
                        c_b,
                    )
                    self.relation_cache[cache_key] = relation
                else:
                    relation = self.relation_cache[cache_key]

                relations.append(relation)

        # Score overall composition
        composition_score = self.composition_validator.score_composition(concept_embeddings)

        # Detect conflicts
        conflicts = self.composition_validator.detect_conflicts(concepts, relations)

        # Recommendations
        recommendations = self._generate_recommendations(concepts, relations, composition_score)

        return {
            "concepts": concepts,
            "pairwise_relations": [
                {
                    "a": r.concept_a,
                    "b": r.concept_b,
                    "relation": r.relation_type,
                    "strength": r.strength,
                    "reasoning": r.reasoning,
                }
                for r in relations
            ],
            "composition_score": composition_score,
            "is_coherent": composition_score > 0.7,
            "conflicts": conflicts,
            "recommendations": recommendations,
        }

    def _generate_recommendations(
        self,
        concepts: List[str],
        relations: List[ConceptRelation],
        score: float,
    ) -> List[str]:
        """Generate recommendations to improve composition."""
        recommendations = []

        # Check for conflicts
        conflict_count = sum(1 for r in relations if r.relation_type == "conflicts")
        if conflict_count > 1:
            recommendations.append("Multiple concept conflicts detected. Consider removing conflicting pairs.")

        # If score is low, suggest refinement
        if score < 0.6:
            recommendations.append("Composition is fragmented. Try grouping related concepts together.")

        # If score is high, suggest expansion
        if score > 0.85 and len(concepts) < 5:
            recommendations.append("Strong composition. Consider adding complementary concepts.")

        # Suggest enhancing pairs
        enhancing = [r for r in relations if r.relation_type == "enhances"]
        if enhancing:
            pair = enhancing[0]
            recommendations.append(
                f"Emphasize the pairing of '{pair.concept_a}' with '{pair.concept_b}' for stronger impact."
            )

        return recommendations

    def predict_generation_quality(
        self,
        concepts: List[str],
        embedding: torch.Tensor,
    ) -> float:
        """Predict generation quality based on concept composition."""
        analysis = self.analyze_composition(concepts, embedding=embedding)

        # Base score from composition
        base = analysis["composition_score"]

        # Penalize conflicts
        conflict_penalty = len(analysis["conflicts"]) * 0.1
        base = max(0.0, base - conflict_penalty)

        return base

    def suggest_concept_improvements(
        self,
        concepts: List[str],
        embedding: torch.Tensor,
    ) -> List[str]:
        """Suggest concept improvements."""
        analysis = self.analyze_composition(concepts, embedding=embedding)

        suggestions = []

        # Remove conflicting concepts
        for conflict in analysis["conflicts"]:
            # Extract concept names from conflict message
            if "conflicts with" in conflict:
                parts = conflict.split("conflicts with")
                c_a = parts[0].strip().replace("'", "")
                suggestions.append(f"Consider removing '{c_a}' due to conflicts")

        # Add supporting concepts
        supporting = [r for r in analysis["pairwise_relations"] if r["relation"] == "supports"]
        if supporting and len(concepts) < 5:
            suggestions.append("Add more supporting concepts for better cohesion")

        return suggestions

    def get_system_stats(self) -> Dict:
        """Get reasoning system statistics."""
        return {
            "cached_relations": len(self.relation_cache),
            "cached_concepts": len(self.concept_cache),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    system = SemanticCompositionReasoner()

    # Test composition analysis
    embedding = torch.randn(1, 4096)
    concepts = ["landscape", "golden_hour", "peaceful", "detailed"]

    print("=== Semantic Composition Analysis ===\n")

    analysis = system.analyze_composition(concepts, embedding=embedding)

    print(f"Concepts: {', '.join(concepts)}")
    print(f"Composition Score: {analysis['composition_score']:.1%}")
    print(f"Is Coherent: {analysis['is_coherent']}\n")

    print("Pairwise Relations:")
    for rel in analysis["pairwise_relations"]:
        print(f"  {rel['a']} <{rel['relation']}> {rel['b']} ({rel['strength']:.1%})")

    if analysis["conflicts"]:
        print("\nConflicts:")
        for conflict in analysis["conflicts"]:
            print(f"  - {conflict}")

    print("\nRecommendations:")
    for rec in analysis["recommendations"]:
        print(f"  - {rec}")

    # Predict quality
    quality = system.predict_generation_quality(concepts, embedding)
    print(f"\nPredicted Generation Quality: {quality:.1%}")

    # Get suggestions
    suggestions = system.suggest_concept_improvements(concepts, embedding)
    if suggestions:
        print("\nConcept Improvement Suggestions:")
        for sug in suggestions:
            print(f"  - {sug}")

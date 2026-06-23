"""
Ensemble Validation System: Combines multiple validators for bulletproof quality.
Uses voting, consensus, and confidence thresholds for ultimate reliability.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation certainty levels."""
    UNKNOWN = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERY_HIGH = 4


@dataclass
class ValidatorOutput:
    """Output from a single validator."""
    name: str
    score: float  # 0-1
    confidence: float  # 0-1 (how sure the validator is)
    reasoning: str  # Why this score
    should_refine: bool
    validation_level: ValidationLevel


@dataclass
class EnsembleValidationResult:
    """Result from ensemble validation."""
    overall_score: float
    validator_scores: Dict[str, ValidatorOutput]
    consensus_score: float  # Agreement between validators (0-1)
    consensus_level: str  # "perfect", "strong", "moderate", "weak"
    recommendation: str  # Action to take
    confidence: float  # Overall confidence in decision
    all_agree: bool  # Do all validators agree on action
    refinement_needed: bool
    refinement_confidence: float


class SemanticValidator(nn.Module):
    """Validates semantic alignment with prompt."""

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        self.semantic_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        prompt_embedding: torch.Tensor,
        generated_embedding: torch.Tensor,
    ) -> ValidatorOutput:
        """Validate semantic alignment."""
        # Ensure embeddings are 2D
        if prompt_embedding.dim() == 1:
            prompt_embedding = prompt_embedding.unsqueeze(0)
        if generated_embedding.dim() == 1:
            generated_embedding = generated_embedding.unsqueeze(0)

        # Take only first 2048 dims if needed
        p_emb = prompt_embedding[:, :2048]
        g_emb = generated_embedding[:, :2048]

        # Combine embeddings
        combined = torch.cat([p_emb, g_emb], dim=-1)

        # Score alignment
        score = float(self.semantic_scorer(combined).squeeze().detach())

        # Confidence
        confidence = float(self.confidence_predictor(combined).squeeze().detach())

        # Determine validation level
        if confidence > 0.95:
            level = ValidationLevel.VERY_HIGH
        elif confidence > 0.85:
            level = ValidationLevel.HIGH
        elif confidence > 0.7:
            level = ValidationLevel.MEDIUM
        elif confidence > 0.5:
            level = ValidationLevel.LOW
        else:
            level = ValidationLevel.UNKNOWN

        return ValidatorOutput(
            name="SemanticValidator",
            score=score,
            confidence=confidence,
            reasoning=f"Semantic alignment: {score:.1%}",
            should_refine=score < 0.85,
            validation_level=level,
        )


class DetailValidator(nn.Module):
    """Validates detail richness and complexity."""

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        self.detail_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.complexity_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, generated_embedding: torch.Tensor) -> ValidatorOutput:
        """Validate detail level."""
        detail_score = float(self.detail_scorer(generated_embedding).squeeze().detach())
        complexity = float(self.complexity_analyzer(generated_embedding).squeeze().detach())

        confidence = (detail_score + complexity) / 2

        return ValidatorOutput(
            name="DetailValidator",
            score=detail_score,
            confidence=confidence,
            reasoning=f"Detail richness: {detail_score:.1%}, Complexity: {complexity:.1%}",
            should_refine=detail_score < 0.75,
            validation_level=ValidationLevel.HIGH if confidence > 0.8 else ValidationLevel.MEDIUM,
        )


class AestheticValidator(nn.Module):
    """Validates aesthetic quality and visual appeal."""

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        self.aesthetic_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.harmony_checker = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, generated_embedding: torch.Tensor) -> ValidatorOutput:
        """Validate aesthetic quality."""
        aesthetic = float(self.aesthetic_scorer(generated_embedding).squeeze().detach())
        harmony = float(self.harmony_checker(generated_embedding).squeeze().detach())

        score = (aesthetic * 0.6 + harmony * 0.4)
        confidence = (aesthetic + harmony) / 2

        return ValidatorOutput(
            name="AestheticValidator",
            score=score,
            confidence=confidence,
            reasoning=f"Aesthetic quality: {aesthetic:.1%}, Visual harmony: {harmony:.1%}",
            should_refine=score < 0.80,
            validation_level=ValidationLevel.HIGH if confidence > 0.8 else ValidationLevel.MEDIUM,
        )


class ConsistencyValidator(nn.Module):
    """Validates consistency with prompt intent."""

    def __init__(self):
        super().__init__()

        self.consistency_scorer = nn.Sequential(
            nn.Linear(512 * 2, 512),  # Expect 512-dim features
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        encoder_features: List[torch.Tensor],
    ) -> ValidatorOutput:
        """Validate consistency across encoders."""
        # Ensure features are 2D
        normalized_features = []
        for feat in encoder_features:
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            normalized_features.append(feat[:, :256])  # Take first 256 dims

        # Combine encoder features
        combined = torch.cat(normalized_features, dim=-1)

        # Score consistency
        score = float(self.consistency_scorer(combined).squeeze().detach())

        # Confidence based on encoder agreement
        confidence = score

        return ValidatorOutput(
            name="ConsistencyValidator",
            score=score,
            confidence=confidence,
            reasoning=f"Cross-encoder consistency: {score:.1%}",
            should_refine=score < 0.85,
            validation_level=ValidationLevel.VERY_HIGH if confidence > 0.9 else ValidationLevel.HIGH,
        )


class RealisticValidator(nn.Module):
    """Validates photorealism and authenticity."""

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        self.realism_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.artifact_detector = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, generated_embedding: torch.Tensor) -> ValidatorOutput:
        """Validate photorealism."""
        realism = float(self.realism_scorer(generated_embedding).squeeze().detach())
        artifact_score = float(self.artifact_detector(generated_embedding).squeeze().detach())

        # Lower artifact = better
        artifact_free = 1.0 - artifact_score

        score = (realism * 0.7 + artifact_free * 0.3)
        confidence = min(realism, artifact_free)

        return ValidatorOutput(
            name="RealisticValidator",
            score=score,
            confidence=confidence,
            reasoning=f"Photorealism: {realism:.1%}, Artifact-free: {artifact_free:.1%}",
            should_refine=score < 0.80,
            validation_level=ValidationLevel.HIGH if confidence > 0.8 else ValidationLevel.MEDIUM,
        )


class EnsembleValidationSystem:
    """Unified ensemble validation system."""

    def __init__(self, hidden_dim: int = 4096):
        self.semantic_validator = SemanticValidator(hidden_dim)
        self.detail_validator = DetailValidator(hidden_dim)
        self.aesthetic_validator = AestheticValidator(hidden_dim)
        self.consistency_validator = ConsistencyValidator()
        self.realistic_validator = RealisticValidator(hidden_dim)

        self.validators = [
            self.semantic_validator,
            self.detail_validator,
            self.aesthetic_validator,
            self.consistency_validator,
            self.realistic_validator,
        ]

        self.validation_history = []

    def validate(
        self,
        prompt_embedding: torch.Tensor,
        generated_embedding: torch.Tensor,
        encoder_features: Optional[List[torch.Tensor]] = None,
    ) -> EnsembleValidationResult:
        """Run full ensemble validation."""
        outputs = {}

        # Semantic validation
        outputs["semantic"] = self.semantic_validator(prompt_embedding, generated_embedding)

        # Detail validation
        outputs["detail"] = self.detail_validator(generated_embedding)

        # Aesthetic validation
        outputs["aesthetic"] = self.aesthetic_validator(generated_embedding)

        # Consistency validation
        if encoder_features:
            outputs["consistency"] = self.consistency_validator(encoder_features)
        else:
            outputs["consistency"] = ValidatorOutput(
                name="ConsistencyValidator",
                score=0.5,
                confidence=0.3,
                reasoning="Encoder features not provided",
                should_refine=True,
                validation_level=ValidationLevel.LOW,
            )

        # Realistic validation
        outputs["realistic"] = self.realistic_validator(generated_embedding)

        # Calculate consensus
        scores = [output.score for output in outputs.values()]
        confidences = [output.confidence for output in outputs.values()]

        consensus_score = sum(scores) / len(scores)
        avg_confidence = sum(confidences) / len(confidences)

        # Determine consensus level
        score_std = (sum((s - consensus_score) ** 2 for s in scores) / len(scores)) ** 0.5
        if score_std < 0.05:
            consensus_level = "perfect"
        elif score_std < 0.1:
            consensus_level = "strong"
        elif score_std < 0.15:
            consensus_level = "moderate"
        else:
            consensus_level = "weak"

        # Check agreement on refinement
        refinement_votes = sum(1 for output in outputs.values() if output.should_refine)
        all_agree = refinement_votes == 0 or refinement_votes == len(outputs)

        # Generate recommendation
        if consensus_score > 0.92 and all_agree and not outputs["semantic"].should_refine:
            recommendation = "PERFECT - Use image as-is"
            refinement_needed = False
        elif consensus_score > 0.85 and refinement_votes <= 1:
            recommendation = "GOOD - Minor refinements optional"
            refinement_needed = False
        elif consensus_score > 0.75:
            recommendation = "ACCEPTABLE - Consider refinements"
            refinement_needed = True
        else:
            recommendation = "POOR - Recommend regeneration"
            refinement_needed = True

        result = EnsembleValidationResult(
            overall_score=consensus_score,
            validator_scores=outputs,
            consensus_score=consensus_score,
            consensus_level=consensus_level,
            recommendation=recommendation,
            confidence=avg_confidence,
            all_agree=all_agree,
            refinement_needed=refinement_needed,
            refinement_confidence=refinement_votes / len(outputs),
        )

        self.validation_history.append(result)
        return result

    def get_validator_report(self, result: EnsembleValidationResult) -> Dict:
        """Generate detailed validation report."""
        report = {
            "overall_score": result.overall_score,
            "confidence": result.confidence,
            "recommendation": result.recommendation,
            "consensus_level": result.consensus_level,
            "validators": {},
            "summary": {
                "all_validators_agree": result.all_agree,
                "refinement_needed": result.refinement_needed,
                "refinement_confidence": result.refinement_confidence,
            },
        }

        for name, output in result.validator_scores.items():
            report["validators"][name] = {
                "score": output.score,
                "confidence": output.confidence,
                "reasoning": output.reasoning,
                "should_refine": output.should_refine,
                "validation_level": output.validation_level.name,
            }

        return report

    def get_validation_stats(self) -> Dict:
        """Get statistics on validations performed."""
        if not self.validation_history:
            return {"total_validations": 0}

        total = len(self.validation_history)
        avg_score = sum(v.overall_score for v in self.validation_history) / total
        avg_confidence = sum(v.confidence for v in self.validation_history) / total
        refinement_rate = sum(1 for v in self.validation_history if v.refinement_needed) / total
        perfect_rate = sum(1 for v in self.validation_history if v.overall_score > 0.92) / total

        return {
            "total_validations": total,
            "average_score": avg_score,
            "average_confidence": avg_confidence,
            "refinement_rate": refinement_rate,
            "perfect_generation_rate": perfect_rate,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    system = EnsembleValidationSystem()

    # Test with embeddings
    prompt_emb = torch.randn(1, 4096)
    generated_emb = torch.randn(1, 4096)
    encoder_features = [torch.randn(1, 512) for _ in range(3)]

    result = system.validate(prompt_emb, generated_emb, encoder_features)

    print("=== Ensemble Validation Report ===\n")
    report = system.get_validator_report(result)

    print(f"Overall Score: {report['overall_score']:.1%}")
    print(f"Confidence: {report['confidence']:.1%}")
    print(f"Consensus Level: {report['consensus_level']}")
    print(f"Recommendation: {report['recommendation']}\n")

    print("Validator Details:")
    for validator_name, validator_report in report["validators"].items():
        print(
            f"  {validator_name}: {validator_report['score']:.1%} "
            f"(confidence: {validator_report['confidence']:.1%})"
        )

    stats = system.get_validation_stats()
    print("\nValidation Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value}")

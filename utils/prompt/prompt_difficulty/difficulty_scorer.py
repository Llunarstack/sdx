"""Prompt difficulty scoring for understanding what's hard to generate."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class PromptDifficultyAnalysis:
    """Analysis of prompt difficulty."""

    overall_score: float
    complexity_level: str
    component_scores: dict
    challenging_aspects: list[str]
    recommendations: list[str]


class PromptDifficultyScorer:
    """Scores prompt difficulty to understand generation complexity."""

    def __init__(self):
        self.complexity_keywords = {
            "high": [
                "intricate",
                "photorealistic",
                "exact",
                "precise",
                "specific",
                "detailed",
                "complex",
                "accurate",
                "flawless",
                "perfect",
            ],
            "medium": [
                "artistic",
                "styled",
                "interesting",
                "unusual",
                "creative",
                "fantastical",
                "abstract",
                "creative",
            ],
            "low": ["simple", "basic", "minimalist", "plain", "generic"],
        }

        self.spatial_keywords = [
            "position",
            "left",
            "right",
            "center",
            "top",
            "bottom",
            "beside",
            "behind",
            "front",
            "background",
            "layout",
            "arrange",
            "composition",
        ]

        self.quality_keywords = [
            "high quality",
            "photorealistic",
            "professional",
            "cinematic",
            "detailed",
            "sharp",
            "clear",
            "vibrant",
            "hd",
            "4k",
            "masterpiece",
        ]

        self.anatomical_keywords = [
            "hand",
            "face",
            "eye",
            "mouth",
            "expression",
            "pose",
            "gesture",
            "anatomy",
            "realistic face",
        ]

        self.text_keywords = ["text", "writing", "word", "letter", "sign", "caption", "number"]

    def score_prompt(self, prompt: str) -> PromptDifficultyAnalysis:
        """Analyze prompt difficulty across multiple dimensions."""
        prompt_lower = prompt.lower()

        scores = {
            "length_complexity": self._score_length(prompt),
            "semantic_complexity": self._score_semantic_complexity(prompt_lower),
            "spatial_complexity": self._score_spatial_requirements(prompt_lower),
            "anatomical_difficulty": self._score_anatomical_requirements(prompt_lower),
            "text_difficulty": self._score_text_requirements(prompt_lower),
            "quality_requirements": self._score_quality_expectations(prompt_lower),
            "negative_prompt_length": self._score_negative_complexity(prompt),
        }

        overall = sum(scores.values()) / len(scores)

        challenging = self._identify_challenging_aspects(prompt_lower, scores)
        recommendations = self._generate_recommendations(scores, challenging)

        level = self._classify_level(overall)

        return PromptDifficultyAnalysis(
            overall_score=overall,
            complexity_level=level,
            component_scores=scores,
            challenging_aspects=challenging,
            recommendations=recommendations,
        )

    def _score_length(self, prompt: str) -> float:
        """Score based on prompt length."""
        words = len(prompt.split())

        if words < 5:
            return 0.1
        elif words < 15:
            return 0.3
        elif words < 50:
            return 0.6
        elif words < 100:
            return 0.8
        else:
            return 1.0

    def _score_semantic_complexity(self, prompt: str) -> float:
        """Score semantic complexity - unusual concepts, multiple subjects."""
        score = 0.0

        high_count = sum(1 for kw in self.complexity_keywords["high"] if kw in prompt)
        medium_count = sum(1 for kw in self.complexity_keywords["medium"] if kw in prompt)

        score += high_count * 0.2
        score += medium_count * 0.1

        comma_count = prompt.count(",")
        semicolon_count = prompt.count(";")
        score += min(1.0, (comma_count + semicolon_count) * 0.1)

        unusual_chars = len([c for c in prompt if ord(c) > 127])
        score += min(0.3, unusual_chars * 0.01)

        return min(1.0, score)

    def _score_spatial_requirements(self, prompt: str) -> float:
        """Score spatial/compositional complexity."""
        spatial_keywords_found = sum(1 for kw in self.spatial_keywords if kw in prompt)

        score = min(1.0, spatial_keywords_found * 0.15)

        if re.search(r"\d+\s*(people|person|character|object)", prompt):
            score += 0.3

        return min(1.0, score)

    def _score_anatomical_requirements(self, prompt: str) -> float:
        """Score anatomical difficulty - hands, faces, expressions."""
        score = 0.0

        if any(kw in prompt for kw in ["hand", "hands"]):
            score += 0.3
        if any(kw in prompt for kw in ["face", "expression", "expression"]):
            score += 0.2
        if any(kw in prompt for kw in ["pose", "gesture"]):
            score += 0.2
        if "eye" in prompt or "eyes" in prompt:
            score += 0.15

        realistic_face_matches = len(re.findall(r"realistic.*face|face.*realistic", prompt))
        score += realistic_face_matches * 0.2

        return min(1.0, score)

    def _score_text_requirements(self, prompt: str) -> float:
        """Score difficulty of text-in-image requirements (hardest task)."""
        score = 0.0

        if any(kw in prompt for kw in self.text_keywords):
            score = 0.7

        text_matches = re.findall(r'["\']([^"\']{2,})["\']', prompt)
        if text_matches:
            score = 0.8

        specific_text = re.search(r"(say|write|display|show).*[\"'].*[\"']", prompt)
        if specific_text:
            score = 0.9

        return min(1.0, score)

    def _score_quality_expectations(self, prompt: str) -> float:
        """Score quality expectations - high quality requirements increase difficulty."""
        score = 0.0

        quality_count = sum(1 for kw in self.quality_keywords if kw in prompt)
        score += min(0.5, quality_count * 0.15)

        if "masterpiece" in prompt or "award" in prompt:
            score += 0.3

        return min(1.0, score)

    def _score_negative_complexity(self, prompt: str) -> float:
        """Score based on negative prompt complexity."""
        negative_start = prompt.find("negative:")
        if negative_start == -1:
            return 0.0

        negative = prompt[negative_start:].lower()
        negative_terms = [t.strip() for t in negative.split(",") if t.strip()]

        score = min(1.0, len(negative_terms) * 0.1)
        return score

    def _identify_challenging_aspects(self, prompt: str, scores: dict) -> list[str]:
        """Identify specific challenging aspects of the prompt."""
        aspects = []

        if scores["text_difficulty"] > 0.6:
            aspects.append("Text generation (very difficult)")
        if scores["anatomical_difficulty"] > 0.5:
            aspects.append("Anatomical correctness (hands/faces)")
        if scores["spatial_complexity"] > 0.5:
            aspects.append("Complex spatial layout")
        if scores["quality_requirements"] > 0.6:
            aspects.append("High quality expectations")
        if scores["semantic_complexity"] > 0.7:
            aspects.append("Unusual/complex concepts")

        return aspects

    def _generate_recommendations(self, scores: dict, aspects: list[str]) -> list[str]:
        """Generate recommendations for improving generation."""
        recommendations = []

        if scores["text_difficulty"] > 0.6:
            recommendations.append("Consider using ControlNet for text placement guidance")
            recommendations.append("Break text requirement into separate prompt")

        if scores["anatomical_difficulty"] > 0.5:
            recommendations.append("Use detailed hand/face prompts or LoRA for anatomy")
            recommendations.append("Increase sampling steps for anatomical detail")

        if scores["spatial_complexity"] > 0.5:
            recommendations.append("Use spatial layout DSL for precise composition")
            recommendations.append("Consider multi-stage generation (coarse → detail)")

        if scores["quality_requirements"] > 0.6:
            recommendations.append("Enable high-resolution refinement (hires-fix)")
            recommendations.append("Increase CFG guidance for quality enforcement")

        if scores["semantic_complexity"] > 0.7:
            recommendations.append("Simplify prompt or use reference images")
            recommendations.append("Consider ensemble generation with multiple prompts")

        if not recommendations:
            recommendations.append("Prompt is straightforward - use baseline settings")

        return recommendations

    def _classify_level(self, score: float) -> str:
        """Classify difficulty level."""
        if score < 0.2:
            return "trivial"
        elif score < 0.4:
            return "easy"
        elif score < 0.6:
            return "moderate"
        elif score < 0.8:
            return "difficult"
        else:
            return "extremely_difficult"

    def batch_score_prompts(self, prompts: list[str]) -> list[PromptDifficultyAnalysis]:
        """Score multiple prompts."""
        return [self.score_prompt(prompt) for prompt in prompts]

    def generate_report(self, analysis: PromptDifficultyAnalysis) -> str:
        """Generate human-readable report."""
        report = f"""
Prompt Difficulty Analysis
==========================

Overall Difficulty Score: {analysis.overall_score:.2f}/1.00
Complexity Level: {analysis.complexity_level.upper()}

Component Breakdown:
"""
        for component, score in analysis.component_scores.items():
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            report += f"  {component:.<35} {bar} {score:.2f}\n"

        report += "\nChallenging Aspects:\n"
        for aspect in analysis.challenging_aspects:
            report += f"  • {aspect}\n"

        report += "\nRecommendations:\n"
        for rec in analysis.recommendations:
            report += f"  • {rec}\n"

        return report

"""
Advanced Memory & Preference System: Builds rich user profiles with detailed preference learning.
Tracks dominant themes, remembers what users like, and recommends improvements.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class UserPreferenceProfile:
    """Complete user preference profile."""
    user_id: str
    total_generations: int = 0
    favorite_subjects: Dict[str, float] = field(default_factory=dict)  # subject -> preference score
    favorite_styles: Dict[str, float] = field(default_factory=dict)  # style -> preference score
    favorite_moods: Dict[str, float] = field(default_factory=dict)  # mood -> preference score
    preferred_lighting: Dict[str, float] = field(default_factory=dict)  # lighting -> score
    preferred_colors: List[str] = field(default_factory=list)
    quality_threshold: float = 4.0  # User's acceptable quality bar
    average_rating: float = 0.0
    satisfaction_rate: float = 0.0  # Percentage of high-quality generations
    dominant_theme: Optional[str] = None
    secondary_themes: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


class PreferenceMemory(nn.Module):
    """Memory module that learns and stores detailed preferences."""

    def __init__(self, hidden_dim: int = 4096, max_profiles: int = 100):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_profiles = max_profiles

        # Preference encoder (learns what makes a user like something)
        self.preference_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )

        # Subject affinity predictor
        self.subject_affinity = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )

        # Style affinity predictor
        self.style_affinity = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )

        # Mood affinity predictor
        self.mood_affinity = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )

        # Theme extractor (finds dominant patterns)
        self.theme_extractor = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )

        # Profiles storage
        self.profiles: Dict[str, UserPreferenceProfile] = {}

    def get_or_create_profile(self, user_id: str) -> UserPreferenceProfile:
        """Get existing profile or create new one."""
        if user_id not in self.profiles:
            if len(self.profiles) >= self.max_profiles:
                # Remove oldest profile
                oldest_id = min(
                    self.profiles.keys(),
                    key=lambda uid: self.profiles[uid].last_updated
                )
                del self.profiles[oldest_id]

            self.profiles[user_id] = UserPreferenceProfile(user_id=user_id)

        return self.profiles[user_id]

    def update_profile(
        self,
        user_id: str,
        generated_features: torch.Tensor,
        user_rating: float,
        subject: Optional[str] = None,
        style: Optional[str] = None,
        mood: Optional[str] = None,
        lighting: Optional[str] = None,
    ):
        """Update user profile with new generation data."""
        profile = self.get_or_create_profile(user_id)

        # Encode features
        encoded = self.preference_encoder(generated_features)

        # Extract affinities
        subject_aff = self.subject_affinity(encoded)
        style_aff = self.style_affinity(encoded)
        mood_aff = self.mood_affinity(encoded)

        # Update subject preferences
        if subject:
            weight = user_rating / 5.0
            if subject not in profile.favorite_subjects:
                profile.favorite_subjects[subject] = 0.0
            profile.favorite_subjects[subject] = (
                0.9 * profile.favorite_subjects[subject] +
                0.1 * float(subject_aff.mean()) * weight
            )

        # Update style preferences
        if style:
            weight = user_rating / 5.0
            if style not in profile.favorite_styles:
                profile.favorite_styles[style] = 0.0
            profile.favorite_styles[style] = (
                0.9 * profile.favorite_styles[style] +
                0.1 * float(style_aff.mean()) * weight
            )

        # Update mood preferences
        if mood:
            weight = user_rating / 5.0
            if mood not in profile.favorite_moods:
                profile.favorite_moods[mood] = 0.0
            profile.favorite_moods[mood] = (
                0.9 * profile.favorite_moods[mood] +
                0.1 * float(mood_aff.mean()) * weight
            )

        # Update lighting preferences
        if lighting:
            weight = user_rating / 5.0
            if lighting not in profile.preferred_lighting:
                profile.preferred_lighting[lighting] = 0.0
            profile.preferred_lighting[lighting] = (
                0.9 * profile.preferred_lighting[lighting] +
                0.1 * weight
            )

        # Update stats
        profile.total_generations += 1
        old_avg = profile.average_rating
        profile.average_rating = (
            (old_avg * (profile.total_generations - 1) + user_rating) /
            profile.total_generations
        )

        # Update satisfaction rate
        high_quality = sum(1 for _ in range(1) if user_rating >= profile.quality_threshold)
        profile.satisfaction_rate = high_quality / max(1, profile.total_generations)

        profile.last_updated = datetime.now()


class ThemeAnalyzer(nn.Module):
    """Analyzes and identifies dominant themes in user preferences."""

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        self.theme_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )

        self.theme_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 20),  # 20 theme types
        )

        self.themes = [
            "nature", "portrait", "landscape", "abstract", "surreal",
            "fantasy", "sci-fi", "minimalist", "maximalist", "geometric",
            "organic", "vintage", "modern", "cinematic", "painterly",
            "photorealistic", "stylized", "fantasy", "dark", "bright"
        ]

    def extract_themes(
        self,
        features_list: List[torch.Tensor],
        ratings_list: List[float],
    ) -> Tuple[str, List[str]]:
        """Extract dominant themes from user's preferences."""
        if not features_list:
            return "unknown", []

        # Combine all features weighted by rating
        combined = None
        total_weight = 0.0

        for features, rating in zip(features_list, ratings_list):
            weight = rating / 5.0
            if combined is None:
                combined = features * weight
            else:
                combined = combined + features * weight
            total_weight += weight

        if combined is not None and total_weight > 0:
            combined = combined / total_weight

            # Analyze themes
            analyzed = self.theme_analyzer(combined)
            theme_logits = self.theme_classifier(analyzed)
            theme_scores = torch.softmax(theme_logits[0], dim=0)

            # Get top themes
            top_indices = torch.argsort(theme_scores, descending=True)[:3]
            dominant = self.themes[int(top_indices[0])]
            secondary = [
                self.themes[int(idx)] for idx in top_indices[1:3]
                if float(theme_scores[idx]) > 0.1
            ]

            return dominant, secondary

        return "unknown", []


class RecommendationEngine(nn.Module):
    """Recommends improvements and next prompts based on preferences."""

    def __init__(self):
        super().__init__()

        self.recommendation_scorer = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def recommend_improvements(
        self,
        profile: UserPreferenceProfile,
        last_rating: float,
    ) -> List[str]:
        """Recommend improvements based on user profile."""
        recommendations = []

        # If satisfaction is low, recommend quality improvements
        if profile.satisfaction_rate < 0.7:
            recommendations.append("Focus on higher quality generations")

        # Recommend exploring secondary themes
        if profile.secondary_themes:
            theme = profile.secondary_themes[0]
            recommendations.append(f"Try combining {theme} with {profile.dominant_theme}")

        # Recommend style exploration if limited
        if len(profile.favorite_styles) < 3:
            recommendations.append("Explore more artistic styles")

        # If last generation was rated low, suggest alternatives
        if last_rating < 3.0:
            recommendations.append(f"Try emphasizing {profile.dominant_theme} elements")

        return recommendations

    def recommend_next_prompt(
        self,
        profile: UserPreferenceProfile,
    ) -> Dict[str, str]:
        """Recommend next prompt based on preferences."""
        recommendation = {
            "subject": max(profile.favorite_subjects.items(), default=("landscape", 0.5))[0]
            if profile.favorite_subjects else "landscape",
            "style": max(profile.favorite_styles.items(), default=("photorealistic", 0.5))[0]
            if profile.favorite_styles else "photorealistic",
            "mood": max(profile.favorite_moods.items(), default=("peaceful", 0.5))[0]
            if profile.favorite_moods else "peaceful",
            "lighting": max(profile.preferred_lighting.items(), default=("natural", 0.5))[0]
            if profile.preferred_lighting else "natural",
        }

        # Construct prompt
        parts = [
            recommendation["subject"],
            f"in a {recommendation['mood']} mood",
            f"with {recommendation['lighting']} lighting",
            f"in {recommendation['style']} style",
        ]

        recommendation["full_prompt"] = ", ".join(parts)

        return recommendation


class MemoryPreferenceSystem:
    """Unified memory and preference system."""

    def __init__(self, hidden_dim: int = 4096):
        self.preference_memory = PreferenceMemory(hidden_dim)
        self.theme_analyzer = ThemeAnalyzer(hidden_dim)
        self.recommendation_engine = RecommendationEngine()

        self.feature_history: Dict[str, List[torch.Tensor]] = {}
        self.rating_history: Dict[str, List[float]] = {}

    def record_generation(
        self,
        user_id: str,
        generated_features: torch.Tensor,
        user_rating: float,
        subject: Optional[str] = None,
        style: Optional[str] = None,
        mood: Optional[str] = None,
        lighting: Optional[str] = None,
    ):
        """Record a generation with all preferences."""
        # Update profile
        self.preference_memory.update_profile(
            user_id,
            generated_features,
            user_rating,
            subject=subject,
            style=style,
            mood=mood,
            lighting=lighting,
        )

        # Store in history
        if user_id not in self.feature_history:
            self.feature_history[user_id] = []
            self.rating_history[user_id] = []

        self.feature_history[user_id].append(generated_features)
        self.rating_history[user_id].append(user_rating)

        # Keep last 100
        if len(self.feature_history[user_id]) > 100:
            self.feature_history[user_id] = self.feature_history[user_id][-100:]
            self.rating_history[user_id] = self.rating_history[user_id][-100:]

        # Update themes
        profile = self.preference_memory.get_or_create_profile(user_id)
        dominant, secondary = self.theme_analyzer.extract_themes(
            self.feature_history[user_id],
            self.rating_history[user_id],
        )
        profile.dominant_theme = dominant
        profile.secondary_themes = secondary

        logger.info(
            f"Recorded generation for {user_id}: "
            f"rating={user_rating}/5, theme={dominant}, "
            f"satisfaction={profile.satisfaction_rate:.1%}"
        )

    def get_user_profile(self, user_id: str) -> UserPreferenceProfile:
        """Get user's preference profile."""
        return self.preference_memory.get_or_create_profile(user_id)

    def get_recommendations(self, user_id: str) -> Dict:
        """Get recommendations for user."""
        profile = self.preference_memory.get_or_create_profile(user_id)

        last_rating = self.rating_history.get(user_id, [0.0])[-1] if user_id in self.rating_history else 0.0

        improvements = self.recommendation_engine.recommend_improvements(profile, last_rating)
        next_prompt = self.recommendation_engine.recommend_next_prompt(profile)

        return {
            "improvements": improvements,
            "next_prompt_recommendation": next_prompt,
            "user_profile": {
                "total_generations": profile.total_generations,
                "average_rating": profile.average_rating,
                "satisfaction_rate": profile.satisfaction_rate,
                "dominant_theme": profile.dominant_theme,
                "favorite_subjects": dict(sorted(
                    profile.favorite_subjects.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]),
                "favorite_styles": dict(sorted(
                    profile.favorite_styles.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]),
            },
        }

    def predict_satisfaction(
        self,
        user_id: str,
        prompt_features: torch.Tensor,
    ) -> float:
        """Predict whether user will be satisfied with a prompt."""
        profile = self.preference_memory.get_or_create_profile(user_id)

        # Simple prediction: if generation would align with preferences
        if not profile.favorite_subjects:
            return 0.5  # Unknown

        # Average preference strength
        avg_subject_pref = sum(profile.favorite_subjects.values()) / len(profile.favorite_subjects)
        avg_style_pref = sum(profile.favorite_styles.values()) / len(profile.favorite_styles) if profile.favorite_styles else 0.5

        # Combine with user's quality threshold
        prediction = (avg_subject_pref * 0.4 + avg_style_pref * 0.4 + profile.quality_threshold / 5.0 * 0.2)

        return min(1.0, max(0.0, prediction))

    def export_profile(self, user_id: str) -> Dict:
        """Export user profile for analysis."""
        profile = self.preference_memory.get_or_create_profile(user_id)

        return {
            "user_id": user_id,
            "total_generations": profile.total_generations,
            "average_rating": profile.average_rating,
            "satisfaction_rate": profile.satisfaction_rate,
            "dominant_theme": profile.dominant_theme,
            "secondary_themes": profile.secondary_themes,
            "favorite_subjects": profile.favorite_subjects,
            "favorite_styles": profile.favorite_styles,
            "favorite_moods": profile.favorite_moods,
            "preferred_lighting": profile.preferred_lighting,
            "created_at": profile.created_at.isoformat(),
            "last_updated": profile.last_updated.isoformat(),
        }

    def get_system_stats(self) -> Dict:
        """Get system-wide statistics."""
        if not self.preference_memory.profiles:
            return {"total_users": 0}

        profiles = list(self.preference_memory.profiles.values())
        avg_rating = sum(p.average_rating for p in profiles) / len(profiles)
        avg_satisfaction = sum(p.satisfaction_rate for p in profiles) / len(profiles)

        return {
            "total_users": len(profiles),
            "average_user_rating": avg_rating,
            "average_satisfaction_rate": avg_satisfaction,
            "total_generations": sum(p.total_generations for p in profiles),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    system = MemoryPreferenceSystem()

    # Simulate user interactions
    user_id = "user_001"

    for i in range(10):
        features = torch.randn(1, 4096)
        rating = 3.5 + (i * 0.15)  # Improving ratings

        system.record_generation(
            user_id,
            features,
            min(rating, 5.0),
            subject="landscape",
            style="photorealistic",
            mood="peaceful",
            lighting="golden hour",
        )

    # Get recommendations
    recommendations = system.get_recommendations(user_id)

    print("\n=== User Preference Profile ===")
    print(f"Total Generations: {recommendations['user_profile']['total_generations']}")
    print(f"Average Rating: {recommendations['user_profile']['average_rating']:.2f}/5")
    print(f"Satisfaction Rate: {recommendations['user_profile']['satisfaction_rate']:.1%}")
    print(f"Dominant Theme: {recommendations['user_profile']['dominant_theme']}")

    print("\n=== Recommendations ===")
    for rec in recommendations["improvements"]:
        print(f"- {rec}")

    print("\n=== Next Prompt Recommendation ===")
    next_prompt = recommendations["next_prompt_recommendation"]
    print(f"Suggested: {next_prompt['full_prompt']}")

    # System stats
    stats = system.get_system_stats()
    print("\n=== System Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

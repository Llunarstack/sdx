"""
Consistency & Memory System - Address the lack of memory and consistency across generations.
Implements character consistency, style memory, and scene continuity.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


@dataclass
class CharacterProfile:
    """Stores consistent character information."""

    name: str
    description: str
    physical_features: Dict[str, str]
    clothing: Dict[str, str]
    style_tags: List[str]
    reference_images: List[str] = None
    generation_history: List[str] = None
    consistency_score: float = 1.0

    def __post_init__(self):
        if self.reference_images is None:
            self.reference_images = []
        if self.generation_history is None:
            self.generation_history = []


@dataclass
class StyleProfile:
    """Stores consistent style information."""

    name: str
    style_description: str
    color_palette: List[str]
    artistic_elements: List[str]
    technical_parameters: Dict[str, Any]
    reference_prompts: List[str]
    consistency_keywords: List[str]

    def to_prompt_addition(self) -> str:
        """Convert style profile to prompt addition."""
        elements = [
            self.style_description,
            f"color palette: {', '.join(self.color_palette[:3])}",
            ", ".join(self.artistic_elements[:3]),
            ", ".join(self.consistency_keywords[:3]),
        ]
        return ", ".join(elements)


@dataclass
class SceneContext:
    """Stores scene context for continuity."""

    scene_id: str
    location: str
    lighting: str
    atmosphere: str
    time_of_day: str
    weather: str
    objects: List[str]
    characters: List[str]
    style_profile: str
    reference_prompt: str


class ConsistencyManager:
    """Manage consistency across multiple generations."""

    def __init__(self, storage_path: str = "./assets/consistency"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.characters_file = self.storage_path / "characters.json"
        self.styles_file = self.storage_path / "styles.json"
        self.scenes_file = self.storage_path / "scenes.json"

        self.characters = self._load_characters()
        self.styles = self._load_styles()
        self.scenes = self._load_scenes()

        # Consistency enhancement templates
        self.consistency_enhancers = {
            "character": [
                "consistent character design",
                "same person throughout",
                "character consistency",
                "identical appearance",
                "recognizable features",
            ],
            "style": [
                "consistent art style",
                "unified visual style",
                "same artistic approach",
                "coherent style",
                "matching aesthetic",
            ],
            "scene": [
                "consistent environment",
                "same location",
                "matching lighting",
                "coherent atmosphere",
                "unified scene",
            ],
        }

    def _load_characters(self) -> Dict[str, CharacterProfile]:
        """Load character profiles from storage."""
        if self.characters_file.exists():
            with open(self.characters_file, "r") as f:
                data = json.load(f)
                return {name: CharacterProfile(**profile_data) for name, profile_data in data.items()}
        return {}

    def _load_styles(self) -> Dict[str, StyleProfile]:
        """Load style profiles from storage."""
        if self.styles_file.exists():
            with open(self.styles_file, "r") as f:
                data = json.load(f)
                return {name: StyleProfile(**profile_data) for name, profile_data in data.items()}
        return {}

    def _load_scenes(self) -> Dict[str, SceneContext]:
        """Load scene contexts from storage."""
        if self.scenes_file.exists():
            with open(self.scenes_file, "r") as f:
                data = json.load(f)
                return {scene_id: SceneContext(**scene_data) for scene_id, scene_data in data.items()}
        return {}

    def save_all(self):
        """Save all consistency data to storage."""
        # Save characters
        with open(self.characters_file, "w") as f:
            json.dump({name: asdict(profile) for name, profile in self.characters.items()}, f, indent=2)

        # Save styles
        with open(self.styles_file, "w") as f:
            json.dump({name: asdict(profile) for name, profile in self.styles.items()}, f, indent=2)

        # Save scenes
        with open(self.scenes_file, "w") as f:
            json.dump({scene_id: asdict(context) for scene_id, context in self.scenes.items()}, f, indent=2)

    def create_character_profile(self, name: str, description: str, reference_prompt: str = None) -> CharacterProfile:
        """Create a new character profile from description."""
        # Parse physical features from description
        physical_features = self._extract_physical_features(description)
        clothing = self._extract_clothing_info(description)
        style_tags = self._extract_style_tags(description)

        profile = CharacterProfile(
            name=name,
            description=description,
            physical_features=physical_features,
            clothing=clothing,
            style_tags=style_tags,
        )

        if reference_prompt:
            profile.generation_history.append(reference_prompt)

        self.characters[name] = profile
        self.save_all()

        return profile

    def create_style_profile(self, name: str, style_description: str, reference_prompt: str = None) -> StyleProfile:
        """Create a new style profile."""
        color_palette = self._extract_colors(style_description)
        artistic_elements = self._extract_artistic_elements(style_description)
        consistency_keywords = self._generate_consistency_keywords(style_description)

        profile = StyleProfile(
            name=name,
            style_description=style_description,
            color_palette=color_palette,
            artistic_elements=artistic_elements,
            technical_parameters={},
            reference_prompts=[reference_prompt] if reference_prompt else [],
            consistency_keywords=consistency_keywords,
        )

        self.styles[name] = profile
        self.save_all()

        return profile

    def create_scene_context(self, scene_id: str, location: str, base_prompt: str) -> SceneContext:
        """Create a new scene context."""
        # Extract scene elements from prompt
        lighting = self._extract_lighting(base_prompt)
        atmosphere = self._extract_atmosphere(base_prompt)
        time_of_day = self._extract_time_of_day(base_prompt)
        weather = self._extract_weather(base_prompt)
        objects = self._extract_objects(base_prompt)

        context = SceneContext(
            scene_id=scene_id,
            location=location,
            lighting=lighting,
            atmosphere=atmosphere,
            time_of_day=time_of_day,
            weather=weather,
            objects=objects,
            characters=[],
            style_profile="",
            reference_prompt=base_prompt,
        )

        self.scenes[scene_id] = context
        self.save_all()

        return context

    def generate_consistent_prompt(
        self, base_prompt: str, character_name: str = None, style_name: str = None, scene_id: str = None
    ) -> str:
        """Generate a prompt with consistency enhancements."""
        prompt_parts = [base_prompt]

        # Add character consistency
        if character_name and character_name in self.characters:
            character = self.characters[character_name]
            character_consistency = self._build_character_consistency(character)
            prompt_parts.append(character_consistency)

            # Add character-specific enhancers
            prompt_parts.extend(self.consistency_enhancers["character"][:2])

        # Add style consistency
        if style_name and style_name in self.styles:
            style = self.styles[style_name]
            style_consistency = style.to_prompt_addition()
            prompt_parts.append(style_consistency)

            # Add style-specific enhancers
            prompt_parts.extend(self.consistency_enhancers["style"][:2])

        # Add scene consistency
        if scene_id and scene_id in self.scenes:
            scene = self.scenes[scene_id]
            scene_consistency = self._build_scene_consistency(scene)
            prompt_parts.append(scene_consistency)

            # Add scene-specific enhancers
            prompt_parts.extend(self.consistency_enhancers["scene"][:2])

        return ", ".join(prompt_parts)

    def _extract_physical_features(self, description: str) -> Dict[str, str]:
        """Extract physical features from character description."""
        features = {}
        description_lower = description.lower()

        # Hair
        hair_colors = ["blonde", "brown", "black", "red", "gray", "white", "silver"]
        hair_styles = ["long", "short", "curly", "straight", "wavy", "braided"]

        for color in hair_colors:
            if color in description_lower:
                features["hair_color"] = color
                break

        for style in hair_styles:
            if style in description_lower and "hair" in description_lower:
                features["hair_style"] = style
                break

        # Eyes
        eye_colors = ["blue", "brown", "green", "hazel", "gray", "amber"]
        for color in eye_colors:
            if f"{color} eyes" in description_lower:
                features["eye_color"] = color
                break

        # Build
        builds = ["tall", "short", "slim", "athletic", "muscular", "petite"]
        for build in builds:
            if build in description_lower:
                features["build"] = build
                break

        return features

    def _extract_clothing_info(self, description: str) -> Dict[str, str]:
        """Extract clothing information from description."""
        clothing = {}
        description_lower = description.lower()

        # Clothing items
        clothing_items = {
            "shirt": ["shirt", "blouse", "top"],
            "pants": ["pants", "jeans", "trousers"],
            "dress": ["dress", "gown"],
            "jacket": ["jacket", "coat", "blazer"],
            "shoes": ["shoes", "boots", "sneakers"],
        }

        for category, items in clothing_items.items():
            for item in items:
                if item in description_lower:
                    clothing[category] = item
                    break

        # Colors
        colors = ["red", "blue", "green", "black", "white", "gray", "brown", "yellow"]
        for color in colors:
            if color in description_lower:
                clothing["primary_color"] = color
                break

        return clothing

    def _extract_style_tags(self, description: str) -> List[str]:
        """Extract style-related tags from description."""
        style_keywords = [
            "realistic",
            "anime",
            "cartoon",
            "photorealistic",
            "artistic",
            "detailed",
            "simple",
            "complex",
            "stylized",
            "natural",
        ]

        tags = []
        description_lower = description.lower()

        for keyword in style_keywords:
            if keyword in description_lower:
                tags.append(keyword)

        return tags

    def _extract_colors(self, description: str) -> List[str]:
        """Extract color palette from style description."""
        colors = []
        color_names = [
            "red",
            "blue",
            "green",
            "yellow",
            "orange",
            "purple",
            "pink",
            "black",
            "white",
            "gray",
            "brown",
            "cyan",
            "magenta",
            "lime",
        ]

        description_lower = description.lower()
        for color in color_names:
            if color in description_lower:
                colors.append(color)

        return colors[:5]  # Limit to 5 colors

    def _extract_artistic_elements(self, description: str) -> List[str]:
        """Extract artistic elements from style description."""
        elements = []
        artistic_terms = [
            "painterly",
            "sketchy",
            "smooth",
            "textured",
            "detailed",
            "minimalist",
            "ornate",
            "geometric",
            "organic",
            "abstract",
            "realistic",
            "stylized",
            "vintage",
            "modern",
            "classical",
        ]

        description_lower = description.lower()
        for term in artistic_terms:
            if term in description_lower:
                elements.append(term)

        return elements[:5]

    def _generate_consistency_keywords(self, description: str) -> List[str]:
        """Generate keywords that help maintain style consistency."""
        base_keywords = ["consistent style", "unified aesthetic", "coherent design"]

        # Add specific keywords based on description
        if "anime" in description.lower():
            base_keywords.extend(["anime style", "cel shading", "clean lines"])
        elif "realistic" in description.lower():
            base_keywords.extend(["photorealistic", "natural lighting", "detailed"])
        elif "artistic" in description.lower():
            base_keywords.extend(["artistic style", "creative", "expressive"])

        return base_keywords[:5]

    def _extract_lighting(self, prompt: str) -> str:
        """Extract lighting information from prompt."""
        lighting_terms = {
            "bright": "bright lighting",
            "dark": "dark lighting",
            "soft": "soft lighting",
            "harsh": "harsh lighting",
            "natural": "natural lighting",
            "artificial": "artificial lighting",
            "dramatic": "dramatic lighting",
            "sunset": "sunset lighting",
            "sunrise": "sunrise lighting",
        }

        prompt_lower = prompt.lower()
        for term, lighting in lighting_terms.items():
            if term in prompt_lower:
                return lighting

        return "natural lighting"

    def _extract_atmosphere(self, prompt: str) -> str:
        """Extract atmosphere from prompt."""
        atmosphere_terms = {
            "peaceful": "peaceful atmosphere",
            "tense": "tense atmosphere",
            "mysterious": "mysterious atmosphere",
            "cheerful": "cheerful atmosphere",
            "melancholy": "melancholy atmosphere",
            "energetic": "energetic atmosphere",
        }

        prompt_lower = prompt.lower()
        for term, atmosphere in atmosphere_terms.items():
            if term in prompt_lower:
                return atmosphere

        return "neutral atmosphere"

    def _extract_time_of_day(self, prompt: str) -> str:
        """Extract time of day from prompt."""
        time_terms = ["morning", "afternoon", "evening", "night", "dawn", "dusk", "noon", "midnight"]

        prompt_lower = prompt.lower()
        for time in time_terms:
            if time in prompt_lower:
                return time

        return "daytime"

    def _extract_weather(self, prompt: str) -> str:
        """Extract weather from prompt."""
        weather_terms = ["sunny", "cloudy", "rainy", "snowy", "foggy", "stormy", "clear"]

        prompt_lower = prompt.lower()
        for weather in weather_terms:
            if weather in prompt_lower:
                return weather

        return "clear"

    def _extract_objects(self, prompt: str) -> List[str]:
        """Extract objects from prompt."""
        # This is simplified - would use NLP in practice
        common_objects = [
            "tree",
            "car",
            "house",
            "chair",
            "table",
            "book",
            "flower",
            "mountain",
            "river",
            "building",
            "bridge",
            "road",
            "window",
        ]

        objects = []
        prompt_lower = prompt.lower()

        for obj in common_objects:
            if obj in prompt_lower:
                objects.append(obj)

        return objects

    def _build_character_consistency(self, character: CharacterProfile) -> str:
        """Build character consistency prompt addition."""
        consistency_parts = []

        # Add physical features
        for feature, value in character.physical_features.items():
            consistency_parts.append(f"{value} {feature.replace('_', ' ')}")

        # Add clothing
        for item, value in character.clothing.items():
            consistency_parts.append(f"{value}")

        # Add style tags
        consistency_parts.extend(character.style_tags[:3])

        # Add character name for reference
        consistency_parts.append(f"character named {character.name}")

        return ", ".join(consistency_parts)

    def _build_scene_consistency(self, scene: SceneContext) -> str:
        """Build scene consistency prompt addition."""
        consistency_parts = [
            f"location: {scene.location}",
            scene.lighting,
            scene.atmosphere,
            f"time: {scene.time_of_day}",
            f"weather: {scene.weather}",
        ]

        # Add objects
        if scene.objects:
            consistency_parts.append(f"objects: {', '.join(scene.objects[:3])}")

        return ", ".join(consistency_parts)

    def update_character_history(self, character_name: str, new_prompt: str):
        """Update character generation history."""
        if character_name in self.characters:
            self.characters[character_name].generation_history.append(new_prompt)
            self.save_all()

    def get_character_variations(self, character_name: str) -> List[str]:
        """Get prompt variations for character consistency."""
        if character_name not in self.characters:
            return []

        character = self.characters[character_name]
        base_consistency = self._build_character_consistency(character)

        variations = [
            f"{base_consistency}, front view, clear features",
            f"{base_consistency}, side profile, detailed",
            f"{base_consistency}, three quarter view, natural pose",
            f"{base_consistency}, full body, standing pose",
            f"{base_consistency}, portrait, close up",
        ]

        return variations

    def analyze_consistency_drift(self, character_name: str) -> Dict[str, Any]:
        """Analyze how much a character has drifted from original design."""
        if character_name not in self.characters:
            return {"error": "Character not found"}

        character = self.characters[character_name]
        history = character.generation_history

        if len(history) < 2:
            return {"drift_score": 0.0, "analysis": "Insufficient history"}

        # Simple drift analysis based on prompt similarity
        original_prompt = history[0]
        recent_prompts = history[-3:]  # Last 3 prompts

        # Calculate similarity (simplified)
        original_words = set(original_prompt.lower().split())

        similarities = []
        for prompt in recent_prompts:
            prompt_words = set(prompt.lower().split())
            similarity = len(original_words & prompt_words) / len(original_words | prompt_words)
            similarities.append(similarity)

        avg_similarity = sum(similarities) / len(similarities)
        drift_score = 1.0 - avg_similarity

        return {
            "drift_score": drift_score,
            "consistency_score": character.consistency_score,
            "analysis": f"Character has {'high' if drift_score > 0.5 else 'low'} drift from original design",
            "recommendations": self._get_consistency_recommendations(drift_score),
        }

    def _get_consistency_recommendations(self, drift_score: float) -> List[str]:
        """Get recommendations for improving consistency."""
        if drift_score > 0.7:
            return [
                "Character has drifted significantly from original design",
                "Consider using reference images for consistency",
                "Add more specific physical feature descriptions",
                "Use character name consistently in prompts",
            ]
        elif drift_score > 0.4:
            return [
                "Moderate drift detected",
                "Reinforce key character features in prompts",
                "Use consistent style tags",
            ]
        else:
            return ["Good consistency maintained", "Continue using current approach"]


class ReferenceImageManager:
    """Manage reference images for consistency."""

    def __init__(self, storage_path: str = "./reference_images"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def save_reference_image(self, image: Image.Image, reference_id: str, metadata: Dict[str, Any] = None) -> str:
        """Save a reference image with metadata."""
        image_path = self.storage_path / f"{reference_id}.png"
        metadata_path = self.storage_path / f"{reference_id}_metadata.json"

        # Save image
        image.save(image_path)

        # Save metadata
        if metadata is None:
            metadata = {}

        metadata.update(
            {
                "reference_id": reference_id,
                "image_path": str(image_path),
                "created_at": str(Path().cwd()),  # Placeholder timestamp
                "image_size": image.size,
            }
        )

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return str(image_path)

    def load_reference_image(self, reference_id: str) -> Optional[Tuple[Image.Image, Dict[str, Any]]]:
        """Load a reference image with its metadata."""
        image_path = self.storage_path / f"{reference_id}.png"
        metadata_path = self.storage_path / f"{reference_id}_metadata.json"

        if not image_path.exists():
            return None

        # Load image
        image = Image.open(image_path)

        # Load metadata
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

        return image, metadata

    def create_reference_prompt(self, reference_id: str) -> Optional[str]:
        """Create a prompt based on reference image metadata."""
        result = self.load_reference_image(reference_id)
        if not result:
            return None

        image, metadata = result

        # Build prompt from metadata
        prompt_parts = []

        if "character_name" in metadata:
            prompt_parts.append(f"character: {metadata['character_name']}")

        if "style" in metadata:
            prompt_parts.append(f"style: {metadata['style']}")

        if "description" in metadata:
            prompt_parts.append(metadata["description"])

        # Add reference consistency enhancers
        prompt_parts.extend(
            ["consistent with reference", "matching appearance", "same character design", "reference accuracy"]
        )

        return ", ".join(prompt_parts)


def create_consistency_system(storage_path: str = "./assets/consistency"):
    """Create complete consistency system."""
    return {
        "consistency_manager": ConsistencyManager(storage_path),
        "reference_manager": ReferenceImageManager(storage_path + "/references"),
    }

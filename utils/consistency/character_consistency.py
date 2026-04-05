"""
Character Consistency System - Core implementation for maintaining character identity across generations.
Provides character profile management, consistency validation, and training integration.
"""

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None


@dataclass
class PhysicalFeatures:
    """Physical feature specifications for a character."""

    # Face features
    face_shape: str = "oval"  # oval, round, square, heart, diamond
    eye_color: str = "brown"  # brown, blue, green, hazel, gray
    eye_shape: str = "almond"  # almond, round, hooded, monolid
    eyebrow_shape: str = "arched"  # arched, straight, rounded
    nose_shape: str = "straight"  # straight, aquiline, button, wide
    lip_shape: str = "medium"  # thin, medium, full
    skin_tone: str = "medium"  # fair, light, medium, tan, dark
    facial_hair: str = "none"  # none, mustache, beard, goatee
    distinctive_marks: List[str] = None

    # Hair features
    hair_color: str = "brown"
    hair_style: str = "medium"
    hair_texture: str = "straight"  # straight, wavy, curly
    hair_length: str = "medium"  # short, medium, long

    # Body features
    height: str = "average"  # short, average, tall
    build: str = "average"  # slim, average, athletic, heavy
    proportions: str = "balanced"
    body_marks: List[str] = None

    def __post_init__(self):
        if self.distinctive_marks is None:
            self.distinctive_marks = []
        if self.body_marks is None:
            self.body_marks = []


@dataclass
class StylePreferences:
    """Style and appearance preferences for a character."""

    clothing_style: str = "casual"  # casual, formal, vintage, modern
    color_palette: List[str] = None
    accessories: List[str] = None
    preferred_expressions: List[str] = None

    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = ["blue", "white", "black"]
        if self.accessories is None:
            self.accessories = []
        if self.preferred_expressions is None:
            self.preferred_expressions = ["neutral", "smile"]


@dataclass
class CharacterProfile:
    """Complete character profile with all defining features."""

    character_id: str
    name: str
    version: str = "1.0"
    created_date: str = None
    last_updated: str = None

    physical_features: PhysicalFeatures = None
    style_preferences: StylePreferences = None

    reference_images: List[str] = None
    face_embedding: Optional[np.ndarray] = None
    body_embedding: Optional[np.ndarray] = None
    style_embedding: Optional[np.ndarray] = None

    consistency_score: float = 0.0
    generation_count: int = 0

    def __post_init__(self):
        if self.created_date is None:
            self.created_date = datetime.now().isoformat()
        if self.last_updated is None:
            self.last_updated = self.created_date
        if self.physical_features is None:
            self.physical_features = PhysicalFeatures()
        if self.style_preferences is None:
            self.style_preferences = StylePreferences()
        if self.reference_images is None:
            self.reference_images = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary for serialization."""
        data = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        if self.face_embedding is not None:
            data["face_embedding"] = self.face_embedding.tolist()
        if self.body_embedding is not None:
            data["body_embedding"] = self.body_embedding.tolist()
        if self.style_embedding is not None:
            data["style_embedding"] = self.style_embedding.tolist()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CharacterProfile":
        """Create profile from dictionary."""
        # Convert lists back to numpy arrays
        if "face_embedding" in data and data["face_embedding"] is not None:
            data["face_embedding"] = np.array(data["face_embedding"])
        if "body_embedding" in data and data["body_embedding"] is not None:
            data["body_embedding"] = np.array(data["body_embedding"])
        if "style_embedding" in data and data["style_embedding"] is not None:
            data["style_embedding"] = np.array(data["style_embedding"])

        # Create nested dataclasses
        if "physical_features" in data:
            data["physical_features"] = PhysicalFeatures(**data["physical_features"])
        if "style_preferences" in data:
            data["style_preferences"] = StylePreferences(**data["style_preferences"])

        return cls(**data)


class FaceEncoder(nn.Module):
    """Neural network for encoding facial features into embeddings."""

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Convolutional feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Facial landmark processor
        self.landmark_processor = nn.Sequential(
            nn.Linear(468 * 2, 256),  # 468 facial landmarks * 2 coordinates
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Feature fusion and embedding
        self.fusion = nn.Sequential(
            nn.Linear(512 * 4 * 4 + 128, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, face_image: torch.Tensor, landmarks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode face image and landmarks into embedding.

        Args:
            face_image: Face image tensor [B, 3, H, W]
            landmarks: Facial landmarks [B, 468, 2]

        Returns:
            Face embedding [B, embedding_dim]
        """
        # Extract visual features
        visual_features = self.feature_extractor(face_image)
        visual_features = visual_features.view(visual_features.size(0), -1)

        # Process landmarks if available
        if landmarks is not None:
            landmark_features = self.landmark_processor(landmarks.view(landmarks.size(0), -1))
            combined_features = torch.cat([visual_features, landmark_features], dim=1)
        else:
            # Use zero padding if no landmarks
            landmark_features = torch.zeros(visual_features.size(0), 128, device=visual_features.device)
            combined_features = torch.cat([visual_features, landmark_features], dim=1)

        # Generate final embedding
        embedding = self.fusion(combined_features)
        return F.normalize(embedding, p=2, dim=1)


class BodyEncoder(nn.Module):
    """Neural network for encoding body features and proportions."""

    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Body feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Body pose processor
        self.pose_processor = nn.Sequential(
            nn.Linear(33 * 3, 128),  # 33 body keypoints * 3 coordinates
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Embedding generation
        self.embedding_net = nn.Sequential(
            nn.Linear(256 * 4 * 4 + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, body_image: torch.Tensor, pose_keypoints: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode body image and pose into embedding.

        Args:
            body_image: Body image tensor [B, 3, H, W]
            pose_keypoints: Body pose keypoints [B, 33, 3]

        Returns:
            Body embedding [B, embedding_dim]
        """
        # Extract visual features
        visual_features = self.feature_extractor(body_image)
        visual_features = visual_features.view(visual_features.size(0), -1)

        # Process pose if available
        if pose_keypoints is not None:
            pose_features = self.pose_processor(pose_keypoints.view(pose_keypoints.size(0), -1))
            combined_features = torch.cat([visual_features, pose_features], dim=1)
        else:
            # Use zero padding if no pose
            pose_features = torch.zeros(visual_features.size(0), 64, device=visual_features.device)
            combined_features = torch.cat([visual_features, pose_features], dim=1)

        # Generate embedding
        embedding = self.embedding_net(combined_features)
        return F.normalize(embedding, p=2, dim=1)


class ConsistencyValidator:
    """Validates character consistency across generated images."""

    def __init__(self, face_encoder: FaceEncoder, body_encoder: BodyEncoder):
        self.face_encoder = face_encoder
        self.body_encoder = body_encoder

        # Consistency thresholds
        self.face_similarity_threshold = 0.85
        self.body_similarity_threshold = 0.80
        self.color_similarity_threshold = 0.75

    def validate_character_consistency(
        self, generated_image: torch.Tensor, character_profile: CharacterProfile
    ) -> Dict[str, float]:
        """
        Validate consistency of generated image against character profile.

        Args:
            generated_image: Generated image tensor [3, H, W]
            character_profile: Character profile to validate against

        Returns:
            Dictionary of consistency scores
        """
        scores = {}

        # Extract face region and validate
        face_region = self._extract_face_region(generated_image)
        if face_region is not None and character_profile.face_embedding is not None:
            with torch.no_grad():
                face_embedding = self.face_encoder(face_region.unsqueeze(0))
                face_similarity = self._cosine_similarity(
                    face_embedding.detach().cpu().numpy(), character_profile.face_embedding.reshape(1, -1)
                )
                scores["face_similarity"] = float(face_similarity[0])
        else:
            scores["face_similarity"] = 0.0

        # Extract body region and validate
        body_region = self._extract_body_region(generated_image)
        if body_region is not None and character_profile.body_embedding is not None:
            with torch.no_grad():
                body_embedding = self.body_encoder(body_region.unsqueeze(0))
                body_similarity = self._cosine_similarity(
                    body_embedding.detach().cpu().numpy(), character_profile.body_embedding.reshape(1, -1)
                )
                scores["body_similarity"] = float(body_similarity[0])
        else:
            scores["body_similarity"] = 0.0

        # Validate color consistency
        color_score = self._validate_color_consistency(generated_image, character_profile)
        scores["color_consistency"] = color_score

        # Calculate overall consistency score
        weights = {"face_similarity": 0.4, "body_similarity": 0.3, "color_consistency": 0.3}
        overall_score = sum(scores[key] * weights[key] for key in weights.keys())
        scores["overall_consistency"] = overall_score

        # Determine consistency level
        if overall_score >= 0.9:
            scores["consistency_level"] = "excellent"
        elif overall_score >= 0.8:
            scores["consistency_level"] = "good"
        elif overall_score >= 0.7:
            scores["consistency_level"] = "fair"
        else:
            scores["consistency_level"] = "poor"

        return scores

    def _extract_face_region(self, image: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract face region from image."""
        # Ensure image is 3D [C, H, W]
        if image.dim() == 4:
            image = image.squeeze(0)  # Remove batch dimension if present
        elif image.dim() != 3:
            return None

        # Convert to numpy for OpenCV processing
        img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Use OpenCV face detection (simplified)
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                # Take the largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face_region = img_np[y : y + h, x : x + w]

                # Resize to standard size and convert back to tensor
                face_region = cv2.resize(face_region, (224, 224))
                face_tensor = torch.from_numpy(face_region).permute(2, 0, 1).float() / 255.0
                return face_tensor
        except Exception:
            # If face detection fails, use center crop as fallback
            h, w = image.shape[1], image.shape[2]
            center_h, center_w = h // 2, w // 2
            face_size = min(h, w) // 3

            top = max(0, center_h - face_size // 2)
            bottom = min(h, center_h + face_size // 2)
            left = max(0, center_w - face_size // 2)
            right = min(w, center_w + face_size // 2)

            face_region = image[:, top:bottom, left:right]

            # Resize to standard size
            face_region = F.interpolate(
                face_region.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
            ).squeeze(0)

            return face_region

        return None

    def _extract_body_region(self, image: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract body region from image."""
        # Ensure image is 3D [C, H, W]
        if image.dim() == 4:
            image = image.squeeze(0)  # Remove batch dimension if present
        elif image.dim() != 3:
            return None

        # For now, use the full image as body region
        # In practice, would use pose estimation to extract body
        return F.interpolate(image.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)

    def _validate_color_consistency(self, image: torch.Tensor, character_profile: CharacterProfile) -> float:
        """Validate color palette consistency."""
        # Extract dominant colors from image
        img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        dominant_colors = self._extract_dominant_colors(img_np, k=5)

        # Compare with character's preferred color palette
        color_palette = character_profile.style_preferences.color_palette
        if not color_palette:
            return 1.0  # No preference, so consistent

        # Convert color names to RGB values (simplified)
        palette_colors = [self._color_name_to_rgb(color) for color in color_palette]

        # Calculate color similarity
        max_similarities = []
        for dom_color in dominant_colors:
            similarities = [self._color_similarity(dom_color, pal_color) for pal_color in palette_colors]
            max_similarities.append(max(similarities))

        return np.mean(max_similarities)

    def _extract_dominant_colors(self, image: np.ndarray, k: int = 5) -> List[Tuple[int, int, int]]:
        """Extract dominant colors using K-means clustering."""
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)

        # Use K-means to find dominant colors if available
        if KMeans is not None:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(pixels)
                # Return cluster centers as dominant colors
                return [tuple(map(int, color)) for color in kmeans.cluster_centers_]
            except Exception:
                pass

        # Fallback: use simple color sampling
        # Sample random pixels and return most common colors
        sample_size = min(1000, len(pixels))
        indices = np.random.choice(len(pixels), sample_size, replace=False)
        sampled_pixels = pixels[indices]

        # Simple binning approach
        colors = []
        for i in range(0, len(sampled_pixels), len(sampled_pixels) // k):
            if i + len(sampled_pixels) // k < len(sampled_pixels):
                color_group = sampled_pixels[i : i + len(sampled_pixels) // k]
                avg_color = np.mean(color_group, axis=0)
                colors.append(tuple(map(int, avg_color)))

        return colors[:k]

    def _color_name_to_rgb(self, color_name: str) -> Tuple[int, int, int]:
        """Convert color name to RGB values."""
        color_map = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "orange": (255, 165, 0),
            "purple": (128, 0, 128),
            "pink": (255, 192, 203),
            "brown": (165, 42, 42),
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "gray": (128, 128, 128),
            "navy": (0, 0, 128),
            "gold": (255, 215, 0),
            "silver": (192, 192, 192),
        }
        return color_map.get(color_name.lower(), (128, 128, 128))

    def _color_similarity(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
        """Calculate similarity between two RGB colors."""
        # Use Euclidean distance in RGB space
        distance = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))
        max_distance = np.sqrt(3 * 255**2)  # Maximum possible distance
        return 1.0 - (distance / max_distance)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between embeddings."""
        return np.dot(a, b.T) / (np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b, axis=1, keepdims=True))


class CharacterDatabase:
    """Database for storing and managing character profiles."""

    def __init__(self, database_path: str = "./character_database"):
        self.database_path = Path(database_path)
        self.database_path.mkdir(exist_ok=True)

        # Initialize encoders
        self.face_encoder = FaceEncoder()
        self.body_encoder = BodyEncoder()
        self.validator = ConsistencyValidator(self.face_encoder, self.body_encoder)

        # Load existing characters
        self.characters: Dict[str, CharacterProfile] = {}
        self._load_characters()

    def create_character(
        self,
        name: str,
        reference_images: List[str],
        physical_features: Optional[PhysicalFeatures] = None,
        style_preferences: Optional[StylePreferences] = None,
    ) -> CharacterProfile:
        """
        Create a new character profile.

        Args:
            name: Character name
            reference_images: List of reference image paths
            physical_features: Physical feature specifications
            style_preferences: Style preferences

        Returns:
            Created character profile
        """
        # Generate unique character ID
        character_id = f"char_{uuid.uuid4().hex[:8]}"

        # Create profile
        profile = CharacterProfile(
            character_id=character_id,
            name=name,
            physical_features=physical_features or PhysicalFeatures(),
            style_preferences=style_preferences or StylePreferences(),
            reference_images=reference_images,
        )

        # Extract embeddings from reference images
        if reference_images:
            profile.face_embedding, profile.body_embedding = self._extract_embeddings_from_references(reference_images)

        # Save profile
        self.characters[character_id] = profile
        self._save_character(profile)

        return profile

    def update_character(self, character_id: str, updates: Dict[str, Any]) -> CharacterProfile:
        """Update an existing character profile."""
        if character_id not in self.characters:
            raise ValueError(f"Character {character_id} not found")

        profile = self.characters[character_id]

        # Update fields
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)

        profile.last_updated = datetime.now().isoformat()

        # Re-extract embeddings if reference images changed
        if "reference_images" in updates:
            profile.face_embedding, profile.body_embedding = self._extract_embeddings_from_references(
                profile.reference_images
            )

        # Save updated profile
        self._save_character(profile)

        return profile

    def get_character(self, character_id: str) -> Optional[CharacterProfile]:
        """Get character profile by ID."""
        return self.characters.get(character_id)

    def list_characters(self, filters: Optional[Dict[str, Any]] = None) -> List[CharacterProfile]:
        """List all characters with optional filtering."""
        characters = list(self.characters.values())

        if filters:
            # Apply filters (simplified implementation)
            if "name" in filters:
                characters = [c for c in characters if filters["name"].lower() in c.name.lower()]
            if "min_consistency_score" in filters:
                characters = [c for c in characters if c.consistency_score >= filters["min_consistency_score"]]

        return characters

    def delete_character(self, character_id: str) -> bool:
        """Delete a character profile."""
        if character_id not in self.characters:
            return False

        # Remove from memory
        del self.characters[character_id]

        # Remove file
        profile_path = self.database_path / f"{character_id}.json"
        if profile_path.exists():
            profile_path.unlink()

        return True

    def validate_consistency(self, image: torch.Tensor, character_id: str) -> Dict[str, float]:
        """Validate consistency of generated image against character."""
        if character_id not in self.characters:
            raise ValueError(f"Character {character_id} not found")

        profile = self.characters[character_id]
        return self.validator.validate_character_consistency(image, profile)

    def _extract_embeddings_from_references(self, reference_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract face and body embeddings from reference images."""
        face_embeddings = []
        body_embeddings = []

        for img_path in reference_paths:
            try:
                # Load image
                image = Image.open(img_path).convert("RGB")
                img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

                # Extract face embedding
                face_region = self.validator._extract_face_region(img_tensor)
                if face_region is not None:
                    with torch.no_grad():
                        face_emb = self.face_encoder(face_region.unsqueeze(0))
                        face_embeddings.append(face_emb.cpu().numpy())

                # Extract body embedding
                body_region = self.validator._extract_body_region(img_tensor)
                if body_region is not None:
                    with torch.no_grad():
                        body_emb = self.body_encoder(body_region.unsqueeze(0))
                        body_embeddings.append(body_emb.cpu().numpy())

            except Exception as e:
                print(f"Error processing reference image {img_path}: {e}")
                continue

        # Average embeddings
        face_embedding = np.mean(face_embeddings, axis=0) if face_embeddings else None
        body_embedding = np.mean(body_embeddings, axis=0) if body_embeddings else None

        return face_embedding, body_embedding

    def _save_character(self, profile: CharacterProfile):
        """Save character profile to disk."""
        profile_path = self.database_path / f"{profile.character_id}.json"
        with open(profile_path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)

    def _load_characters(self):
        """Load all character profiles from disk."""
        for profile_file in self.database_path.glob("*.json"):
            try:
                with open(profile_file, "r") as f:
                    data = json.load(f)
                profile = CharacterProfile.from_dict(data)
                self.characters[profile.character_id] = profile
            except Exception as e:
                print(f"Error loading character profile {profile_file}: {e}")


def create_character_consistency_system(database_path: str = "./character_database") -> CharacterDatabase:
    """Create and initialize the character consistency system."""
    return CharacterDatabase(database_path)

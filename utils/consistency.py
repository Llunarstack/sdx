"""Character and style consistency: character profiles, face/body encoders and validation, the character database, consistency training losses, prompt customization, and mixed-style harmonization."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.prompts import dedupe_tags
from PIL import Image

from utils.prompt.text import str_list

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None


@dataclass(slots=True)
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
    distinctive_marks: list[str] = None

    # Hair features
    hair_color: str = "brown"
    hair_style: str = "medium"
    hair_texture: str = "straight"  # straight, wavy, curly
    hair_length: str = "medium"  # short, medium, long

    # Body features
    height: str = "average"  # short, average, tall
    build: str = "average"  # slim, average, athletic, heavy
    proportions: str = "balanced"
    body_marks: list[str] = None

    def __post_init__(self):
        if self.distinctive_marks is None:
            self.distinctive_marks = []
        if self.body_marks is None:
            self.body_marks = []


@dataclass(slots=True)
class StylePreferences:
    """Style and appearance preferences for a character."""

    clothing_style: str = "casual"  # casual, formal, vintage, modern
    color_palette: list[str] = None
    accessories: list[str] = None
    preferred_expressions: list[str] = None

    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = ["blue", "white", "black"]
        if self.accessories is None:
            self.accessories = []
        if self.preferred_expressions is None:
            self.preferred_expressions = ["neutral", "smile"]


@dataclass(slots=True)
class CharacterProfile:
    """Complete character profile with all defining features."""

    character_id: str
    name: str
    version: str = "1.0"
    created_date: str = None
    last_updated: str = None

    physical_features: PhysicalFeatures = None
    style_preferences: StylePreferences = None

    reference_images: list[str] = None
    face_embedding: np.ndarray | None = None
    body_embedding: np.ndarray | None = None
    style_embedding: np.ndarray | None = None

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

    def to_dict(self) -> dict[str, Any]:
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
    def from_dict(cls, data: dict[str, Any]) -> CharacterProfile:
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

    def forward(self, face_image: torch.Tensor, landmarks: torch.Tensor | None = None) -> torch.Tensor:
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

    def forward(self, body_image: torch.Tensor, pose_keypoints: torch.Tensor | None = None) -> torch.Tensor:
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
    ) -> dict[str, float]:
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

    def _extract_face_region(self, image: torch.Tensor) -> torch.Tensor | None:
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

    def _extract_body_region(self, image: torch.Tensor) -> torch.Tensor | None:
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

    def _extract_dominant_colors(self, image: np.ndarray, k: int = 5) -> list[tuple[int, int, int]]:
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

    def _color_name_to_rgb(self, color_name: str) -> tuple[int, int, int]:
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

    def _color_similarity(self, color1: tuple[int, int, int], color2: tuple[int, int, int]) -> float:
        """Calculate similarity between two RGB colors."""
        # Use Euclidean distance in RGB space
        distance = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2, strict=True)))
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
        self.characters: dict[str, CharacterProfile] = {}
        self._load_characters()

    def create_character(
        self,
        name: str,
        reference_images: list[str],
        physical_features: PhysicalFeatures | None = None,
        style_preferences: StylePreferences | None = None,
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

    def update_character(self, character_id: str, updates: dict[str, Any]) -> CharacterProfile:
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

    def get_character(self, character_id: str) -> CharacterProfile | None:
        """Get character profile by ID."""
        return self.characters.get(character_id)

    def list_characters(self, filters: dict[str, Any] | None = None) -> list[CharacterProfile]:
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

    def validate_consistency(self, image: torch.Tensor, character_id: str) -> dict[str, float]:
        """Validate consistency of generated image against character."""
        if character_id not in self.characters:
            raise ValueError(f"Character {character_id} not found")

        profile = self.characters[character_id]
        return self.validator.validate_character_consistency(image, profile)

    def _extract_embeddings_from_references(self, reference_paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
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
                with open(profile_file) as f:
                    data = json.load(f)
                profile = CharacterProfile.from_dict(data)
                self.characters[profile.character_id] = profile
            except Exception as e:
                print(f"Error loading character profile {profile_file}: {e}")


class CharacterConsistencyLoss(nn.Module):
    """
    Comprehensive loss function for character consistency training.
    Combines multiple loss components to enforce character identity preservation.
    """

    def __init__(self, face_encoder: FaceEncoder, body_encoder: BodyEncoder, embedding_dim: int = 512):
        super().__init__()
        self.face_encoder = face_encoder
        self.body_encoder = body_encoder
        self.embedding_dim = embedding_dim

        # Loss weights
        self.face_weight = 0.3
        self.body_weight = 0.2
        self.color_weight = 0.2
        self.style_weight = 0.15
        self.reference_weight = 0.15

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=0.2)

    def forward(
        self,
        generated_images: torch.Tensor,
        character_profiles: list[CharacterProfile],
        reference_images: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Calculate character consistency loss.

        Args:
            generated_images: Generated images [B, 3, H, W]
            character_profiles: List of character profiles for each image
            reference_images: Reference images for comparison [B, 3, H, W]

        Returns:
            Dictionary of loss components
        """
        device = generated_images.device

        losses = {}

        # Facial consistency loss
        face_loss = self._compute_facial_consistency_loss(generated_images, character_profiles)
        losses["facial_consistency"] = face_loss

        # Body consistency loss
        body_loss = self._compute_body_consistency_loss(generated_images, character_profiles)
        losses["body_consistency"] = body_loss

        # Color consistency loss
        color_loss = self._compute_color_consistency_loss(generated_images, character_profiles)
        losses["color_consistency"] = color_loss

        # Style consistency loss
        style_loss = self._compute_style_consistency_loss(generated_images, character_profiles)
        losses["style_consistency"] = style_loss

        # Reference similarity loss
        if reference_images is not None:
            ref_loss = self._compute_reference_similarity_loss(generated_images, reference_images)
            losses["reference_similarity"] = ref_loss
        else:
            losses["reference_similarity"] = torch.tensor(0.0, device=device)

        # Total weighted loss
        total_loss = (
            self.face_weight * losses["facial_consistency"]
            + self.body_weight * losses["body_consistency"]
            + self.color_weight * losses["color_consistency"]
            + self.style_weight * losses["style_consistency"]
            + self.reference_weight * losses["reference_similarity"]
        )

        losses["total_consistency"] = total_loss

        return losses

    def _compute_facial_consistency_loss(
        self, generated_images: torch.Tensor, character_profiles: list[CharacterProfile]
    ) -> torch.Tensor:
        """Compute facial feature consistency loss."""
        device = generated_images.device

        if not character_profiles or character_profiles[0].face_embedding is None:
            return torch.tensor(0.0, device=device)

        # Extract face regions from generated images
        face_regions = []
        target_embeddings = []

        for i, profile in enumerate(character_profiles):
            if profile.face_embedding is not None:
                # Extract face region (simplified - would use face detection in practice)
                face_region = self._extract_face_region(generated_images[i])
                if face_region is not None:
                    face_regions.append(face_region)
                    target_embeddings.append(torch.from_numpy(profile.face_embedding).to(device))

        if not face_regions:
            return torch.tensor(0.0, device=device)

        # Stack face regions and target embeddings
        face_batch = torch.stack(face_regions)
        target_batch = torch.stack(target_embeddings)

        # Generate embeddings for extracted faces
        with torch.no_grad():
            generated_embeddings = self.face_encoder(face_batch)

        # Calculate cosine similarity loss
        target_labels = torch.ones(len(face_regions), device=device)
        face_loss = self.cosine_loss(generated_embeddings, target_batch, target_labels)

        return face_loss

    def _compute_body_consistency_loss(
        self, generated_images: torch.Tensor, character_profiles: list[CharacterProfile]
    ) -> torch.Tensor:
        """Compute body consistency loss."""
        device = generated_images.device

        if not character_profiles or character_profiles[0].body_embedding is None:
            return torch.tensor(0.0, device=device)

        # Extract body regions and target embeddings
        body_regions = []
        target_embeddings = []

        for i, profile in enumerate(character_profiles):
            if profile.body_embedding is not None:
                # Use full image as body region (simplified)
                body_region = F.interpolate(
                    generated_images[i : i + 1], size=(224, 224), mode="bilinear", align_corners=False
                ).squeeze(0)
                body_regions.append(body_region)
                target_embeddings.append(torch.from_numpy(profile.body_embedding).to(device))

        if not body_regions:
            return torch.tensor(0.0, device=device)

        # Stack regions and targets
        body_batch = torch.stack(body_regions)
        target_batch = torch.stack(target_embeddings)

        # Generate embeddings
        with torch.no_grad():
            generated_embeddings = self.body_encoder(body_batch)

        # Calculate loss
        target_labels = torch.ones(len(body_regions), device=device)
        body_loss = self.cosine_loss(generated_embeddings, target_batch, target_labels)

        return body_loss

    def _compute_color_consistency_loss(
        self, generated_images: torch.Tensor, character_profiles: list[CharacterProfile]
    ) -> torch.Tensor:
        """Compute color palette consistency loss."""
        device = generated_images.device

        total_loss = torch.tensor(0.0, device=device)
        valid_samples = 0

        for i, profile in enumerate(character_profiles):
            color_palette = profile.style_preferences.color_palette
            if not color_palette:
                continue

            # Extract dominant colors from generated image
            image = generated_images[i]
            dominant_colors = self._extract_dominant_colors_tensor(image)

            # Convert color palette to RGB tensors
            palette_colors = (
                torch.tensor(
                    [self._color_name_to_rgb(color) for color in color_palette], device=device, dtype=torch.float32
                )
                / 255.0
            )

            # Calculate color matching loss
            color_loss = self._color_matching_loss(dominant_colors, palette_colors)
            total_loss += color_loss
            valid_samples += 1

        return total_loss / max(valid_samples, 1)

    def _compute_style_consistency_loss(
        self, generated_images: torch.Tensor, character_profiles: list[CharacterProfile]
    ) -> torch.Tensor:
        """Compute style consistency loss."""
        # Simplified style loss - would be more sophisticated in practice
        device = generated_images.device
        return torch.tensor(0.0, device=device)

    def _compute_reference_similarity_loss(
        self, generated_images: torch.Tensor, reference_images: torch.Tensor
    ) -> torch.Tensor:
        """Compute similarity loss with reference images."""
        # Use perceptual loss between generated and reference images
        # Simplified implementation using MSE
        return self.mse_loss(generated_images, reference_images)

    def _extract_face_region(self, image: torch.Tensor) -> torch.Tensor | None:
        """Extract face region from image tensor."""
        # Simplified face extraction - would use proper face detection
        # For now, assume center crop contains face
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

    def _extract_dominant_colors_tensor(self, image: torch.Tensor, k: int = 5) -> torch.Tensor:
        """Extract dominant colors from image tensor."""
        # Reshape image to list of pixels
        pixels = image.view(3, -1).transpose(0, 1)  # [N, 3]

        # Simple clustering using K-means (simplified)
        # In practice, would use proper clustering
        indices = torch.randperm(pixels.size(0))[:1000]  # Sample pixels
        sampled_pixels = pixels[indices]

        # Return mean colors as approximation
        return sampled_pixels.mean(dim=0, keepdim=True).repeat(k, 1)

    def _color_name_to_rgb(self, color_name: str) -> list[int]:
        """Convert color name to RGB values."""
        color_map = {
            "red": [255, 0, 0],
            "green": [0, 255, 0],
            "blue": [0, 0, 255],
            "yellow": [255, 255, 0],
            "orange": [255, 165, 0],
            "purple": [128, 0, 128],
            "pink": [255, 192, 203],
            "brown": [165, 42, 42],
            "black": [0, 0, 0],
            "white": [255, 255, 255],
            "gray": [128, 128, 128],
            "navy": [0, 0, 128],
            "gold": [255, 215, 0],
            "silver": [192, 192, 192],
        }
        return color_map.get(color_name.lower(), [128, 128, 128])

    def _color_matching_loss(self, dominant_colors: torch.Tensor, palette_colors: torch.Tensor) -> torch.Tensor:
        """Calculate loss for color palette matching."""
        # Calculate pairwise distances between dominant and palette colors
        distances = torch.cdist(dominant_colors, palette_colors, p=2)

        # Find minimum distance for each dominant color to palette
        min_distances, _ = torch.min(distances, dim=1)

        # Return average minimum distance as loss
        return min_distances.mean()


class TripletConsistencyLoss(nn.Module):
    """
    Triplet loss for character consistency training.
    Ensures same character embeddings are closer than different character embeddings.
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)

    def forward(
        self, anchor_embeddings: torch.Tensor, positive_embeddings: torch.Tensor, negative_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate triplet loss for character consistency.

        Args:
            anchor_embeddings: Embeddings from generated images [B, D]
            positive_embeddings: Embeddings from same character references [B, D]
            negative_embeddings: Embeddings from different characters [B, D]

        Returns:
            Triplet loss value
        """
        return self.triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for character consistency.
    Pulls same character embeddings together and pushes different characters apart.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate contrastive loss.

        Args:
            embedding1: First set of embeddings [B, D]
            embedding2: Second set of embeddings [B, D]
            labels: 1 for same character, 0 for different [B]

        Returns:
            Contrastive loss value
        """
        # Calculate Euclidean distance
        distances = F.pairwise_distance(embedding1, embedding2)

        # Contrastive loss formula
        positive_loss = labels * torch.pow(distances, 2)
        negative_loss = (1 - labels) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)

        loss = torch.mean(positive_loss + negative_loss)
        return loss


class PerceptualConsistencyLoss(nn.Module):
    """
    Perceptual loss for character consistency using pre-trained features.
    Uses VGG features to compare high-level visual similarity.
    """

    def __init__(self):
        super().__init__()
        # Use VGG16 features for perceptual comparison
        vgg = torch.hub.load("pytorch/vision:v0.10.0", "vgg16", pretrained=True)
        self.features = vgg.features[:16]  # Use up to conv3_3

        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False

        self.mse_loss = nn.MSELoss()

    def forward(self, generated_images: torch.Tensor, reference_images: torch.Tensor) -> torch.Tensor:
        """
        Calculate perceptual loss between generated and reference images.

        Args:
            generated_images: Generated images [B, 3, H, W]
            reference_images: Reference images [B, 3, H, W]

        Returns:
            Perceptual loss value
        """
        # Extract VGG features
        gen_features = self.features(generated_images)
        ref_features = self.features(reference_images)

        # Calculate MSE loss in feature space
        return self.mse_loss(gen_features, ref_features)


class AdversarialConsistencyLoss(nn.Module):
    """
    Adversarial loss for character consistency.
    Uses a discriminator to distinguish between consistent and inconsistent character generations.
    """

    def __init__(self, discriminator_hidden_size: int = 256):
        super().__init__()

        # Character consistency discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(512, discriminator_hidden_size),  # Face embedding size
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(discriminator_hidden_size, discriminator_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(discriminator_hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        self.bce_loss = nn.BCELoss()

    def forward(self, generated_embeddings: torch.Tensor, reference_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculate adversarial consistency loss.

        Args:
            generated_embeddings: Embeddings from generated images [B, D]
            reference_embeddings: Embeddings from reference images [B, D]

        Returns:
            Adversarial loss value
        """
        batch_size = generated_embeddings.size(0)
        device = generated_embeddings.device

        # Create labels (1 for consistent, 0 for inconsistent)
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # Discriminator predictions
        real_pred = self.discriminator(reference_embeddings)
        fake_pred = self.discriminator(generated_embeddings)

        # Adversarial loss
        real_loss = self.bce_loss(real_pred, real_labels)
        fake_loss = self.bce_loss(fake_pred, fake_labels)

        # Generator loss (want discriminator to think generated is real)
        generator_loss = self.bce_loss(fake_pred, real_labels)

        return {"discriminator_loss": real_loss + fake_loss, "generator_loss": generator_loss}


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss for character sequences.
    Ensures character consistency across multiple frames or related images.
    """

    def __init__(self, temporal_weight: float = 0.5):
        super().__init__()
        self.temporal_weight = temporal_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, sequence_embeddings: torch.Tensor, character_embedding: torch.Tensor) -> torch.Tensor:
        """
        Calculate temporal consistency loss.

        Args:
            sequence_embeddings: Embeddings from sequence of images [T, B, D]
            character_embedding: Target character embedding [B, D]

        Returns:
            Temporal consistency loss
        """
        sequence_length = sequence_embeddings.size(0)

        # Calculate consistency loss for each frame
        frame_losses = []
        for t in range(sequence_length):
            frame_loss = self.mse_loss(sequence_embeddings[t], character_embedding)
            frame_losses.append(frame_loss)

        # Average across frames
        temporal_loss = torch.stack(frame_losses).mean()

        # Add temporal smoothness (adjacent frames should be similar)
        if sequence_length > 1:
            smoothness_losses = []
            for t in range(sequence_length - 1):
                smoothness_loss = self.mse_loss(sequence_embeddings[t], sequence_embeddings[t + 1])
                smoothness_losses.append(smoothness_loss)

            smoothness_loss = torch.stack(smoothness_losses).mean()
            temporal_loss += self.temporal_weight * smoothness_loss

        return temporal_loss


class StyleInvariantLoss(nn.Module):
    """
    Style-invariant loss for character consistency across different art styles.
    Ensures character identity is preserved when style changes.
    """

    def __init__(self, style_encoder_dim: int = 256):
        super().__init__()

        # Style encoder to extract style-invariant features
        self.style_invariant_encoder = nn.Sequential(
            nn.Linear(512, style_encoder_dim),  # Face embedding size
            nn.ReLU(),
            nn.Linear(style_encoder_dim, style_encoder_dim),
            nn.ReLU(),
            nn.Linear(style_encoder_dim, style_encoder_dim // 2),
            nn.LayerNorm(style_encoder_dim // 2),
        )

        self.cosine_loss = nn.CosineEmbeddingLoss()

    def forward(self, style1_embeddings: torch.Tensor, style2_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculate style-invariant consistency loss.

        Args:
            style1_embeddings: Character embeddings in first style [B, D]
            style2_embeddings: Character embeddings in second style [B, D]

        Returns:
            Style-invariant loss
        """
        # Extract style-invariant features
        invariant1 = self.style_invariant_encoder(style1_embeddings)
        invariant2 = self.style_invariant_encoder(style2_embeddings)

        # Calculate cosine similarity loss
        batch_size = invariant1.size(0)
        target_labels = torch.ones(batch_size, device=invariant1.device)

        return self.cosine_loss(invariant1, invariant2, target_labels)


class MultiScaleConsistencyLoss(nn.Module):
    """
    Multi-scale consistency loss for character features at different resolutions.
    Ensures consistency across different levels of detail.
    """

    def __init__(self, scales: list[int] | None = None):
        super().__init__()
        self.scales = scales if scales is not None else [64, 128, 256]
        self.mse_loss = nn.MSELoss()

    def forward(self, generated_image: torch.Tensor, reference_image: torch.Tensor) -> torch.Tensor:
        """
        Calculate multi-scale consistency loss.

        Args:
            generated_image: Generated image [B, 3, H, W]
            reference_image: Reference image [B, 3, H, W]

        Returns:
            Multi-scale consistency loss
        """
        total_loss = torch.tensor(0.0, device=generated_image.device)

        for scale in self.scales:
            # Resize images to current scale
            gen_scaled = F.interpolate(generated_image, size=(scale, scale), mode="bilinear", align_corners=False)
            ref_scaled = F.interpolate(reference_image, size=(scale, scale), mode="bilinear", align_corners=False)

            # Calculate MSE loss at this scale
            scale_loss = self.mse_loss(gen_scaled, ref_scaled)
            total_loss += scale_loss

        return total_loss / len(self.scales)


def create_consistency_loss_system(face_encoder: FaceEncoder, body_encoder: BodyEncoder) -> dict[str, nn.Module]:
    """Create complete consistency loss system."""
    return {
        "character_consistency_loss": CharacterConsistencyLoss(face_encoder, body_encoder),
        "triplet_loss": TripletConsistencyLoss(),
        "contrastive_loss": ContrastiveLoss(),
        "perceptual_loss": PerceptualConsistencyLoss(),
        "adversarial_loss": AdversarialConsistencyLoss(),
        "temporal_loss": TemporalConsistencyLoss(),
        "style_invariant_loss": StyleInvariantLoss(),
        "multiscale_loss": MultiScaleConsistencyLoss(),
    }


class ConsistencyLossManager:
    """
    Manager class for coordinating multiple consistency loss functions.
    Provides unified interface for training integration.
    """

    def __init__(
        self, face_encoder: FaceEncoder, body_encoder: BodyEncoder, loss_weights: dict[str, float] | None = None
    ):
        self.loss_functions = create_consistency_loss_system(face_encoder, body_encoder)

        # Default loss weights
        self.loss_weights = loss_weights or {
            "character_consistency_loss": 1.0,
            "triplet_loss": 0.5,
            "contrastive_loss": 0.3,
            "perceptual_loss": 0.7,
            "adversarial_loss": 0.2,
            "temporal_loss": 0.4,
            "style_invariant_loss": 0.3,
            "multiscale_loss": 0.5,
        }

    def compute_total_loss(
        self,
        generated_images: torch.Tensor,
        character_profiles: list[CharacterProfile],
        reference_images: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Compute total consistency loss from all components.

        Args:
            generated_images: Generated images [B, 3, H, W]
            character_profiles: Character profiles for each image
            reference_images: Reference images [B, 3, H, W]
            **kwargs: Additional arguments for specific loss functions

        Returns:
            Dictionary of loss components and total loss
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=generated_images.device)

        # Character consistency loss
        if "character_consistency_loss" in self.loss_functions:
            char_losses = self.loss_functions["character_consistency_loss"](
                generated_images, character_profiles, reference_images
            )
            losses.update(char_losses)
            total_loss += self.loss_weights["character_consistency_loss"] * char_losses["total_consistency"]

        # Perceptual loss
        if reference_images is not None and "perceptual_loss" in self.loss_functions:
            perceptual_loss = self.loss_functions["perceptual_loss"](generated_images, reference_images)
            losses["perceptual_loss"] = perceptual_loss
            total_loss += self.loss_weights["perceptual_loss"] * perceptual_loss

        # Multi-scale loss
        if reference_images is not None and "multiscale_loss" in self.loss_functions:
            multiscale_loss = self.loss_functions["multiscale_loss"](generated_images, reference_images)
            losses["multiscale_loss"] = multiscale_loss
            total_loss += self.loss_weights["multiscale_loss"] * multiscale_loss

        # Add other losses as needed based on available data
        # (triplet, contrastive, adversarial, temporal, style_invariant)

        losses["total_consistency_loss"] = total_loss
        return losses

    def update_loss_weights(self, new_weights: dict[str, float]):
        """Update loss weights during training."""
        self.loss_weights.update(new_weights)

    def get_loss_statistics(self) -> dict[str, Any]:
        """Get statistics about loss components."""
        return {
            "available_losses": list(self.loss_functions.keys()),
            "loss_weights": self.loss_weights,
            "total_losses": len(self.loss_functions),
        }


def build_character_prompt_additions(
    profile: dict[str, Any],
    *,
    uncensored_mode: bool = False,
    character_strength: float = 1.0,
) -> tuple[str, str]:
    """
    Build `(positive_additions, negative_additions)` from a character profile dict.
    """
    positive_tokens: list[str] = []
    negative_tokens: list[str] = []

    # Disambiguate multi-character / layout (optional JSON keys on character sheet)
    subject_label = str(profile.get("subject_label", "") or profile.get("character_slot", "")).strip()
    spatial_anchor = str(profile.get("spatial_anchor", "") or profile.get("screen_position", "")).strip()
    if subject_label:
        positive_tokens.append(f"subject label: {subject_label}")
    if spatial_anchor:
        positive_tokens.append(f"screen position: {spatial_anchor}")

    # Backward-compatible keys
    positive_tokens.extend(str_list(profile.get("prompt")))
    positive_tokens.extend(str_list(profile.get("positive")))
    positive_tokens.extend(str_list(profile.get("appearance")))
    positive_tokens.extend(str_list(profile.get("style_tags")))
    positive_tokens.extend(str_list(profile.get("clothing")))
    positive_tokens.extend(str_list(profile.get("accessories")))
    negative_tokens.extend(str_list(profile.get("negative")))
    negative_tokens.extend(str_list(profile.get("negative_prompt")))

    # Rich character keys (original or existing/canon)
    positive_tokens.extend(str_list(profile.get("character_name")))
    positive_tokens.extend(str_list(profile.get("source")))
    positive_tokens.extend(str_list(profile.get("franchise")))
    positive_tokens.extend(str_list(profile.get("aliases")))
    positive_tokens.extend(str_list(profile.get("identity_tokens")))
    positive_tokens.extend(str_list(profile.get("canon_traits")))
    positive_tokens.extend(str_list(profile.get("face_features")))
    positive_tokens.extend(str_list(profile.get("hair_features")))
    positive_tokens.extend(str_list(profile.get("body_features")))
    positive_tokens.extend(str_list(profile.get("wardrobe")))
    positive_tokens.extend(str_list(profile.get("materials")))
    positive_tokens.extend(str_list(profile.get("signature_items")))
    positive_tokens.extend(str_list(profile.get("pose_preferences")))
    positive_tokens.extend(str_list(profile.get("camera_preferences")))
    positive_tokens.extend(str_list(profile.get("environment_preferences")))
    positive_tokens.extend(str_list(profile.get("personality_cues")))
    positive_tokens.extend(str_list(profile.get("lighting_preferences")))
    positive_tokens.extend(str_list(profile.get("artist_style_mix")))

    negative_tokens.extend(str_list(profile.get("avoid_tokens")))
    negative_tokens.extend(str_list(profile.get("identity_drift_negative")))

    # Gender presentation controls (opt-in metadata)
    gender_pres = str(profile.get("gender_presentation", "") or "").strip().lower()
    if gender_pres in {"androgynous", "androgynous presentation", "gender-ambiguous", "gender ambiguous"}:
        positive_tokens.append("androgynous presentation")
    elif gender_pres in {"male", "male-presenting", "man-presenting"}:
        positive_tokens.append("male-presenting")
    elif gender_pres in {"female", "female-presenting", "woman-presenting"}:
        positive_tokens.append("female-presenting")

    # Optional strict identity locks (for preexisting/canon stability)
    positive_tokens.extend(str_list(profile.get("lock_tokens")))
    negative_tokens.extend(str_list(profile.get("anti_lock_break_tokens")))

    # Legacy safety behavior becomes optional: only apply when not uncensored.
    if not uncensored_mode:
        lowered = [t.lower() for t in positive_tokens]
        if any("futa" in t or "trap" in t for t in lowered):
            positive_tokens = [
                "androgynous presentation" if ("futa" in t.lower() or "trap" in t.lower()) else t
                for t in positive_tokens
            ]
            negative_tokens.append("explicit genital content")

    # Uncensored mode: remove anti-explicit blocks if present in provided negatives.
    if uncensored_mode:
        filtered_neg = []
        for t in negative_tokens:
            tl = t.lower()
            if any(x in tl for x in ("censorship bars", "mosaic censor", "explicit genital content", "sfw only")):
                continue
            filtered_neg.append(t)
        negative_tokens = filtered_neg

    positive_tokens = dedupe_tags(positive_tokens)
    negative_tokens = dedupe_tags(negative_tokens)

    # Identity strength: >1 repeats high-priority identity cues to improve adherence.
    strength = max(0.5, min(float(character_strength), 2.0))
    if strength > 1.0:
        identity_focus = str_list(profile.get("identity_tokens")) + str_list(profile.get("character_name"))
        identity_focus = dedupe_tags(identity_focus)
        if identity_focus:
            for tok in identity_focus:
                positive_tokens.append(f"({tok})")

    return ", ".join(positive_tokens), ", ".join(negative_tokens)


class StyleType(Enum):
    """Style type classifications."""

    REALISTIC_3D = "realistic_3d"
    ANIME_2D = "anime_2d"
    CARTOON_2D = "cartoon_2d"
    ARTISTIC_2D = "artistic_2d"
    PHOTOGRAPHIC = "photographic"
    PAINTERLY = "painterly"
    SKETCH = "sketch"
    MIXED = "mixed"


class StyleConflictLevel(Enum):
    """Levels of style conflict severity."""

    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    INCOMPATIBLE = "incompatible"


@dataclass(slots=True)
class StyleComponent:
    """Individual style component (LoRA, prompt element, etc.)."""

    name: str
    type: StyleType
    strength: float
    source: str  # "lora", "prompt", "embedding", etc.
    priority: int = 1
    compatibility_tags: list[str] = None

    def __post_init__(self):
        if self.compatibility_tags is None:
            self.compatibility_tags = []


@dataclass(slots=True)
class StyleAnalysis:
    """Analysis of style components and conflicts."""

    components: list[StyleComponent]
    dominant_style: StyleType
    conflict_level: StyleConflictLevel
    conflicts: list[dict[str, Any]]
    harmonization_needed: bool
    recommended_adjustments: list[dict[str, Any]]


class StyleCompatibilityMatrix:
    """Defines compatibility between different style types."""

    def __init__(self):
        # Compatibility scores (0.0 = incompatible, 1.0 = perfect match)
        self.compatibility = {
            (StyleType.REALISTIC_3D, StyleType.REALISTIC_3D): 1.0,
            (StyleType.REALISTIC_3D, StyleType.PHOTOGRAPHIC): 0.9,
            (StyleType.REALISTIC_3D, StyleType.ANIME_2D): 0.2,
            (StyleType.REALISTIC_3D, StyleType.CARTOON_2D): 0.1,
            (StyleType.REALISTIC_3D, StyleType.ARTISTIC_2D): 0.4,
            (StyleType.REALISTIC_3D, StyleType.PAINTERLY): 0.6,
            (StyleType.REALISTIC_3D, StyleType.SKETCH): 0.3,
            (StyleType.ANIME_2D, StyleType.ANIME_2D): 1.0,
            (StyleType.ANIME_2D, StyleType.CARTOON_2D): 0.7,
            (StyleType.ANIME_2D, StyleType.ARTISTIC_2D): 0.8,
            (StyleType.ANIME_2D, StyleType.PAINTERLY): 0.6,
            (StyleType.ANIME_2D, StyleType.SKETCH): 0.5,
            (StyleType.ANIME_2D, StyleType.PHOTOGRAPHIC): 0.1,
            (StyleType.CARTOON_2D, StyleType.CARTOON_2D): 1.0,
            (StyleType.CARTOON_2D, StyleType.ARTISTIC_2D): 0.8,
            (StyleType.CARTOON_2D, StyleType.SKETCH): 0.7,
            (StyleType.CARTOON_2D, StyleType.PAINTERLY): 0.5,
            (StyleType.CARTOON_2D, StyleType.PHOTOGRAPHIC): 0.1,
            (StyleType.ARTISTIC_2D, StyleType.ARTISTIC_2D): 1.0,
            (StyleType.ARTISTIC_2D, StyleType.PAINTERLY): 0.9,
            (StyleType.ARTISTIC_2D, StyleType.SKETCH): 0.8,
            (StyleType.ARTISTIC_2D, StyleType.PHOTOGRAPHIC): 0.3,
            (StyleType.PHOTOGRAPHIC, StyleType.PHOTOGRAPHIC): 1.0,
            (StyleType.PHOTOGRAPHIC, StyleType.PAINTERLY): 0.4,
            (StyleType.PHOTOGRAPHIC, StyleType.SKETCH): 0.2,
            (StyleType.PAINTERLY, StyleType.PAINTERLY): 1.0,
            (StyleType.PAINTERLY, StyleType.SKETCH): 0.8,
            (StyleType.SKETCH, StyleType.SKETCH): 1.0,
        }

        # Make matrix symmetric
        symmetric_compatibility = {}
        for (style1, style2), score in self.compatibility.items():
            symmetric_compatibility[(style1, style2)] = score
            symmetric_compatibility[(style2, style1)] = score

        self.compatibility = symmetric_compatibility

    def get_compatibility(self, style1: StyleType, style2: StyleType) -> float:
        """Get compatibility score between two styles."""
        return self.compatibility.get((style1, style2), 0.0)


class StyleDetector:
    """Detects style types from prompts, LoRAs, and other inputs."""

    def __init__(self):
        # Style detection patterns
        self.style_patterns = {
            StyleType.REALISTIC_3D: [
                r"\b(realistic|photorealistic|3d|render|cgi|unreal engine)\b",
                r"\b(hyperrealistic|lifelike|detailed|high resolution)\b",
                r"\b(octane render|blender|maya|cinema 4d)\b",
            ],
            StyleType.ANIME_2D: [
                r"\b(anime|manga|japanese animation|cel shading)\b",
                r"\b(kawaii|moe|bishojo|bishonen)\b",
                r"\b(studio ghibli|makoto shinkai|anime style)\b",
            ],
            StyleType.CARTOON_2D: [
                r"\b(cartoon|animated|disney|pixar style)\b",
                r"\b(toon|cel animation|flat colors)\b",
                r"\b(comic book|graphic novel|illustration)\b",
            ],
            StyleType.ARTISTIC_2D: [
                r"\b(artistic|stylized|abstract|impressionist)\b",
                r"\b(watercolor|oil painting|acrylic|digital art)\b",
                r"\b(concept art|fantasy art|sci-fi art)\b",
            ],
            StyleType.PHOTOGRAPHIC: [
                r"\b(photograph|photo|camera|lens|aperture)\b",
                r"\b(portrait|landscape|macro|street photography)\b",
                r"\b(canon|nikon|sony|dslr|mirrorless)\b",
            ],
            StyleType.PAINTERLY: [
                r"\b(painting|painted|brush strokes|canvas)\b",
                r"\b(van gogh|monet|picasso|renaissance)\b",
                r"\b(oil on canvas|watercolor|gouache|tempera)\b",
            ],
            StyleType.SKETCH: [
                r"\b(sketch|drawing|pencil|charcoal|ink)\b",
                r"\b(line art|outline|draft|study)\b",
                r"\b(graphite|conte|pastel drawing)\b",
            ],
        }

        # LoRA style classifications (would be loaded from config)
        self.lora_styles = {
            "realistic_vision": StyleType.REALISTIC_3D,
            "anime_diffusion": StyleType.ANIME_2D,
            "cartoon_style": StyleType.CARTOON_2D,
            "artistic_vision": StyleType.ARTISTIC_2D,
            "photo_realistic": StyleType.PHOTOGRAPHIC,
            "oil_painting": StyleType.PAINTERLY,
            "sketch_style": StyleType.SKETCH,
        }

    def detect_prompt_styles(self, prompt: str) -> list[StyleComponent]:
        """Detect styles from text prompt."""
        components = []
        prompt_lower = prompt.lower()

        for style_type, patterns in self.style_patterns.items():
            strength = 0.0
            matches = []

            for pattern in patterns:
                pattern_matches = re.findall(pattern, prompt_lower, re.IGNORECASE)
                if pattern_matches:
                    strength += len(pattern_matches) * 0.3
                    matches.extend(pattern_matches)

            if strength > 0:
                # Normalize strength
                strength = min(strength, 1.0)

                component = StyleComponent(
                    name=f"prompt_{style_type.value}",
                    type=style_type,
                    strength=strength,
                    source="prompt",
                    priority=2,
                    compatibility_tags=matches,
                )
                components.append(component)

        return components

    def detect_lora_styles(self, lora_configs: list[dict[str, Any]]) -> list[StyleComponent]:
        """Detect styles from LoRA configurations."""
        components = []

        for lora_config in lora_configs:
            lora_name = lora_config.get("name", "").lower()
            lora_strength = lora_config.get("strength", 1.0)

            # Try to match LoRA name to known styles
            detected_style = None
            for known_lora, style_type in self.lora_styles.items():
                if known_lora in lora_name:
                    detected_style = style_type
                    break

            # If no direct match, try pattern matching on LoRA name
            if not detected_style:
                for style_type, patterns in self.style_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, lora_name, re.IGNORECASE):
                            detected_style = style_type
                            break
                    if detected_style:
                        break

            if detected_style:
                component = StyleComponent(
                    name=lora_name,
                    type=detected_style,
                    strength=lora_strength,
                    source="lora",
                    priority=3,  # LoRAs have higher priority than prompts
                    compatibility_tags=[lora_name],
                )
                components.append(component)

        return components

    def detect_embedding_styles(self, embeddings: list[dict[str, Any]]) -> list[StyleComponent]:
        """Detect styles from textual inversions/embeddings."""
        components = []

        for embedding in embeddings:
            embedding_name = embedding.get("name", "").lower()
            embedding_strength = embedding.get("strength", 1.0)

            # Similar pattern matching as LoRAs
            detected_style = None
            for style_type, patterns in self.style_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, embedding_name, re.IGNORECASE):
                        detected_style = style_type
                        break
                if detected_style:
                    break

            if detected_style:
                component = StyleComponent(
                    name=embedding_name,
                    type=detected_style,
                    strength=embedding_strength,
                    source="embedding",
                    priority=2,
                    compatibility_tags=[embedding_name],
                )
                components.append(component)

        return components


class StyleConflictAnalyzer:
    """Analyzes conflicts between different style components."""

    def __init__(self):
        self.compatibility_matrix = StyleCompatibilityMatrix()

    def analyze_conflicts(self, components: list[StyleComponent]) -> StyleAnalysis:
        """Analyze style conflicts and provide recommendations."""
        if not components:
            return StyleAnalysis(
                components=[],
                dominant_style=StyleType.REALISTIC_3D,
                conflict_level=StyleConflictLevel.NONE,
                conflicts=[],
                harmonization_needed=False,
                recommended_adjustments=[],
            )

        # Group components by style type
        style_groups = {}
        for component in components:
            if component.type not in style_groups:
                style_groups[component.type] = []
            style_groups[component.type].append(component)

        # Calculate total strength per style
        style_strengths = {}
        for style_type, group_components in style_groups.items():
            total_strength = sum(comp.strength * comp.priority for comp in group_components)
            style_strengths[style_type] = total_strength

        # Determine dominant style
        dominant_style = max(style_strengths.keys(), key=lambda x: style_strengths[x])

        # Analyze conflicts
        conflicts = []
        max_conflict_score = 0.0

        style_types = list(style_groups.keys())
        for i, style1 in enumerate(style_types):
            for style2 in style_types[i + 1 :]:
                compatibility = self.compatibility_matrix.get_compatibility(style1, style2)
                conflict_score = 1.0 - compatibility

                if conflict_score > 0.3:  # Significant conflict
                    strength1 = style_strengths[style1]
                    strength2 = style_strengths[style2]

                    conflict = {
                        "style1": style1,
                        "style2": style2,
                        "conflict_score": conflict_score,
                        "strength1": strength1,
                        "strength2": strength2,
                        "severity": self._get_conflict_severity(conflict_score, strength1, strength2),
                    }
                    conflicts.append(conflict)
                    max_conflict_score = max(max_conflict_score, conflict_score)

        # Determine overall conflict level
        conflict_level = self._determine_conflict_level(max_conflict_score, len(conflicts))

        # Generate recommendations
        recommendations = self._generate_recommendations(
            components, style_groups, style_strengths, conflicts, dominant_style
        )

        return StyleAnalysis(
            components=components,
            dominant_style=dominant_style,
            conflict_level=conflict_level,
            conflicts=conflicts,
            harmonization_needed=conflict_level != StyleConflictLevel.NONE,
            recommended_adjustments=recommendations,
        )

    def _get_conflict_severity(self, conflict_score: float, strength1: float, strength2: float) -> str:
        """Determine severity of individual conflict."""
        combined_strength = strength1 + strength2
        weighted_conflict = conflict_score * combined_strength

        if weighted_conflict > 2.0:
            return "severe"
        elif weighted_conflict > 1.0:
            return "moderate"
        else:
            return "minor"

    def _determine_conflict_level(self, max_conflict_score: float, num_conflicts: int) -> StyleConflictLevel:
        """Determine overall conflict level."""
        if max_conflict_score == 0.0:
            return StyleConflictLevel.NONE
        elif max_conflict_score < 0.3:
            return StyleConflictLevel.MINOR
        elif max_conflict_score < 0.6:
            return StyleConflictLevel.MODERATE
        elif max_conflict_score < 0.8:
            return StyleConflictLevel.SEVERE
        else:
            return StyleConflictLevel.INCOMPATIBLE

    def _generate_recommendations(
        self,
        components: list[StyleComponent],
        style_groups: dict[StyleType, list[StyleComponent]],
        style_strengths: dict[StyleType, float],
        conflicts: list[dict[str, Any]],
        dominant_style: StyleType,
    ) -> list[dict[str, Any]]:
        """Generate harmonization recommendations."""
        recommendations = []

        # If there are severe conflicts, recommend reducing conflicting styles
        for conflict in conflicts:
            if conflict["severity"] in ["severe", "moderate"]:
                weaker_style = (
                    conflict["style1"] if conflict["strength1"] < conflict["strength2"] else conflict["style2"]
                )
                stronger_style = conflict["style2"] if weaker_style == conflict["style1"] else conflict["style1"]

                if weaker_style != dominant_style:
                    recommendations.append(
                        {
                            "type": "reduce_strength",
                            "target_style": weaker_style,
                            "reason": f"Conflicts with dominant {stronger_style.value} style",
                            "suggested_reduction": 0.3 if conflict["severity"] == "moderate" else 0.5,
                        }
                    )

        # Recommend style bridging for compatible styles
        compatible_pairs = []
        for conflict in conflicts:
            if conflict["conflict_score"] < 0.4:  # Somewhat compatible
                compatible_pairs.append((conflict["style1"], conflict["style2"]))

        if compatible_pairs:
            recommendations.append(
                {
                    "type": "style_bridging",
                    "compatible_pairs": compatible_pairs,
                    "reason": "Use transitional elements to blend compatible styles",
                }
            )

        # Recommend dominant style reinforcement
        if len(style_groups) > 2:
            recommendations.append(
                {
                    "type": "reinforce_dominant",
                    "target_style": dominant_style,
                    "reason": "Strengthen dominant style to reduce conflicts",
                    "suggested_boost": 0.2,
                }
            )

        return recommendations


class StyleHarmonizer:
    """Harmonizes conflicting styles to prevent weird-looking images."""

    def __init__(self):
        self.detector = StyleDetector()
        self.analyzer = StyleConflictAnalyzer()

        # Style transition mappings for bridging incompatible styles
        self.style_bridges = {
            (StyleType.REALISTIC_3D, StyleType.ANIME_2D): ["semi-realistic", "stylized realism", "3d anime style"],
            (StyleType.REALISTIC_3D, StyleType.CARTOON_2D): ["stylized 3d", "pixar style", "semi-realistic cartoon"],
            (StyleType.ANIME_2D, StyleType.PHOTOGRAPHIC): [
                "photorealistic anime",
                "live action anime style",
                "realistic manga",
            ],
            (StyleType.CARTOON_2D, StyleType.PHOTOGRAPHIC): [
                "live action cartoon",
                "realistic cartoon style",
                "stylized photography",
            ],
        }

    def harmonize_styles(
        self,
        prompt: str,
        lora_configs: list[dict[str, Any]] = None,
        embeddings: list[dict[str, Any]] = None,
        user_preferences: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """
        Main harmonization function that analyzes and resolves style conflicts.

        Returns:
            Dictionary with harmonized prompt, adjusted LoRA strengths, and metadata
        """
        lora_configs = lora_configs or []
        embeddings = embeddings or []
        user_preferences = user_preferences or {}

        # Detect all style components
        prompt_styles = self.detector.detect_prompt_styles(prompt)
        lora_styles = self.detector.detect_lora_styles(lora_configs)
        embedding_styles = self.detector.detect_embedding_styles(embeddings)

        all_components = prompt_styles + lora_styles + embedding_styles

        # Analyze conflicts
        analysis = self.analyzer.analyze_conflicts(all_components)

        # Apply harmonization if needed
        if analysis.harmonization_needed:
            harmonized_result = self._apply_harmonization(prompt, lora_configs, embeddings, analysis, user_preferences)
        else:
            harmonized_result = {
                "harmonized_prompt": prompt,
                "adjusted_loras": lora_configs,
                "adjusted_embeddings": embeddings,
                "changes_made": [],
            }

        # Add analysis metadata
        harmonized_result.update(
            {
                "style_analysis": {
                    "dominant_style": analysis.dominant_style.value,
                    "conflict_level": analysis.conflict_level.value,
                    "num_conflicts": len(analysis.conflicts),
                    "harmonization_applied": analysis.harmonization_needed,
                },
                "detected_styles": [
                    {"name": comp.name, "type": comp.type.value, "strength": comp.strength, "source": comp.source}
                    for comp in all_components
                ],
                "conflicts": [
                    {
                        "style1": conflict["style1"].value,
                        "style2": conflict["style2"].value,
                        "severity": conflict["severity"],
                        "conflict_score": conflict["conflict_score"],
                    }
                    for conflict in analysis.conflicts
                ],
            }
        )

        return harmonized_result

    def _apply_harmonization(
        self,
        prompt: str,
        lora_configs: list[dict[str, Any]],
        embeddings: list[dict[str, Any]],
        analysis: StyleAnalysis,
        user_preferences: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply harmonization based on analysis recommendations."""

        harmonized_prompt = prompt
        adjusted_loras = lora_configs.copy()
        adjusted_embeddings = embeddings.copy()
        changes_made = []

        # Get user preferences
        max_strength_reduction = user_preferences.get("max_strength_reduction", 0.5)
        allow_prompt_modification = user_preferences.get("allow_prompt_modification", True)

        # Apply recommendations
        for recommendation in analysis.recommended_adjustments:
            if recommendation["type"] == "reduce_strength":
                target_style = recommendation["target_style"]
                reduction = min(recommendation["suggested_reduction"], max_strength_reduction)

                # Reduce LoRA strengths for this style
                for lora in adjusted_loras:
                    lora_name = lora.get("name", "").lower()
                    if self._lora_matches_style(lora_name, target_style):
                        original_strength = lora.get("strength", 1.0)
                        new_strength = original_strength * (1.0 - reduction)
                        lora["strength"] = max(new_strength, 0.1)  # Minimum strength
                        changes_made.append(
                            f"Reduced {lora_name} LoRA strength from {original_strength:.2f} to {new_strength:.2f}"
                        )

                # Reduce embedding strengths for this style
                for embedding in adjusted_embeddings:
                    embedding_name = embedding.get("name", "").lower()
                    if self._embedding_matches_style(embedding_name, target_style):
                        original_strength = embedding.get("strength", 1.0)
                        new_strength = original_strength * (1.0 - reduction)
                        embedding["strength"] = max(new_strength, 0.1)
                        changes_made.append(
                            f"Reduced {embedding_name} embedding strength from {original_strength:.2f} to {new_strength:.2f}"
                        )

            elif recommendation["type"] == "style_bridging" and allow_prompt_modification:
                # Add bridging terms to prompt
                bridge_terms = self._get_bridge_terms(recommendation["compatible_pairs"])
                if bridge_terms:
                    harmonized_prompt = self._add_bridge_terms(harmonized_prompt, bridge_terms)
                    changes_made.append(f"Added style bridging terms: {', '.join(bridge_terms)}")

            elif recommendation["type"] == "reinforce_dominant" and allow_prompt_modification:
                dominant_style = recommendation["target_style"]
                reinforcement_terms = self._get_reinforcement_terms(dominant_style)
                if reinforcement_terms:
                    harmonized_prompt = self._add_reinforcement_terms(harmonized_prompt, reinforcement_terms)
                    changes_made.append(f"Added {dominant_style.value} reinforcement terms")

        # Apply conflict-specific fixes
        if analysis.conflict_level == StyleConflictLevel.SEVERE:
            harmonized_prompt, additional_changes = self._apply_severe_conflict_fixes(
                harmonized_prompt, analysis, user_preferences
            )
            changes_made.extend(additional_changes)

        return {
            "harmonized_prompt": harmonized_prompt,
            "adjusted_loras": adjusted_loras,
            "adjusted_embeddings": adjusted_embeddings,
            "changes_made": changes_made,
        }

    def _lora_matches_style(self, lora_name: str, style_type: StyleType) -> bool:
        """Check if LoRA name matches a style type."""
        patterns = self.detector.style_patterns.get(style_type, [])
        for pattern in patterns:
            if re.search(pattern, lora_name, re.IGNORECASE):
                return True
        return False

    def _embedding_matches_style(self, embedding_name: str, style_type: StyleType) -> bool:
        """Check if embedding name matches a style type."""
        return self._lora_matches_style(embedding_name, style_type)

    def _get_bridge_terms(self, compatible_pairs: list[tuple[StyleType, StyleType]]) -> list[str]:
        """Get bridging terms for compatible style pairs."""
        bridge_terms = []
        for style1, style2 in compatible_pairs:
            pair_bridges = self.style_bridges.get((style1, style2)) or self.style_bridges.get((style2, style1))
            if pair_bridges:
                bridge_terms.extend(pair_bridges[:1])  # Add one bridge term per pair
        return bridge_terms

    def _add_bridge_terms(self, prompt: str, bridge_terms: list[str]) -> str:
        """Add bridging terms to prompt."""
        if not bridge_terms:
            return prompt

        # Add terms at the beginning with moderate weight
        bridge_text = ", ".join(f"({term}:0.8)" for term in bridge_terms)
        return f"{bridge_text}, {prompt}"

    def _get_reinforcement_terms(self, style_type: StyleType) -> list[str]:
        """Get reinforcement terms for a style type."""
        reinforcement_map = {
            StyleType.REALISTIC_3D: ["highly detailed", "photorealistic", "sharp focus"],
            StyleType.ANIME_2D: ["anime style", "cel shading", "clean lines"],
            StyleType.CARTOON_2D: ["cartoon style", "flat colors", "simple shading"],
            StyleType.ARTISTIC_2D: ["artistic", "stylized", "creative"],
            StyleType.PHOTOGRAPHIC: ["photograph", "professional photography", "high quality"],
            StyleType.PAINTERLY: ["painted", "brush strokes", "artistic medium"],
            StyleType.SKETCH: ["sketch", "line art", "drawing"],
        }
        return reinforcement_map.get(style_type, [])

    def _add_reinforcement_terms(self, prompt: str, reinforcement_terms: list[str]) -> str:
        """Add reinforcement terms to prompt."""
        if not reinforcement_terms:
            return prompt

        # Add terms with higher weight to reinforce dominant style
        reinforcement_text = ", ".join(f"({term}:1.1)" for term in reinforcement_terms[:2])
        return f"{prompt}, {reinforcement_text}"

    def _apply_severe_conflict_fixes(
        self, prompt: str, analysis: StyleAnalysis, user_preferences: dict[str, Any]
    ) -> tuple[str, list[str]]:
        """Apply fixes for severe style conflicts."""
        changes_made = []

        # For severe conflicts, add negative prompts to suppress conflicting elements
        conflicting_styles = set()
        for conflict in analysis.conflicts:
            if conflict["severity"] == "severe":
                # Suppress the weaker conflicting style
                weaker_style = (
                    conflict["style1"] if conflict["strength1"] < conflict["strength2"] else conflict["style2"]
                )
                conflicting_styles.add(weaker_style)

        if conflicting_styles and user_preferences.get("allow_negative_prompts", True):
            negative_terms = []
            for style in conflicting_styles:
                if style != analysis.dominant_style:
                    style_negative_terms = self._get_negative_terms(style)
                    negative_terms.extend(style_negative_terms[:2])  # Limit negative terms

            if negative_terms:
                # Add negative terms to suppress conflicting styles
                negative_text = ", ".join(negative_terms)
                prompt = f"{prompt} [negative: {negative_text}]"
                changes_made.append(f"Added negative terms to suppress conflicts: {negative_text}")

        return prompt, changes_made

    def _get_negative_terms(self, style_type: StyleType) -> list[str]:
        """Get negative terms to suppress a style type."""
        negative_map = {
            StyleType.REALISTIC_3D: ["flat", "2d", "cartoon", "anime"],
            StyleType.ANIME_2D: ["realistic", "photographic", "3d render"],
            StyleType.CARTOON_2D: ["realistic", "detailed", "photographic"],
            StyleType.PHOTOGRAPHIC: ["painted", "drawn", "artistic", "stylized"],
            StyleType.PAINTERLY: ["photographic", "digital", "clean"],
            StyleType.SKETCH: ["colored", "painted", "photographic"],
        }
        return negative_map.get(style_type, [])


def create_style_harmonization_system() -> StyleHarmonizer:
    """Create and initialize the style harmonization system."""
    return StyleHarmonizer()

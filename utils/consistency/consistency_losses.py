"""
Character Consistency Loss Functions - Specialized loss functions for training character consistency.
Integrates with the Enhanced DiT training pipeline to enforce character identity preservation.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .character_consistency import BodyEncoder, CharacterProfile, FaceEncoder


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
        character_profiles: List[CharacterProfile],
        reference_images: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
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
        self, generated_images: torch.Tensor, character_profiles: List[CharacterProfile]
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
        self, generated_images: torch.Tensor, character_profiles: List[CharacterProfile]
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
        self, generated_images: torch.Tensor, character_profiles: List[CharacterProfile]
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
        self, generated_images: torch.Tensor, character_profiles: List[CharacterProfile]
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

    def _extract_face_region(self, image: torch.Tensor) -> Optional[torch.Tensor]:
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

    def _color_name_to_rgb(self, color_name: str) -> List[int]:
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

    def __init__(self, scales: List[int] = [64, 128, 256]):
        super().__init__()
        self.scales = scales
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


def create_consistency_loss_system(face_encoder: FaceEncoder, body_encoder: BodyEncoder) -> Dict[str, nn.Module]:
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
        self, face_encoder: FaceEncoder, body_encoder: BodyEncoder, loss_weights: Optional[Dict[str, float]] = None
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
        character_profiles: List[CharacterProfile],
        reference_images: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
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

    def update_loss_weights(self, new_weights: Dict[str, float]):
        """Update loss weights during training."""
        self.loss_weights.update(new_weights)

    def get_loss_statistics(self) -> Dict[str, Any]:
        """Get statistics about loss components."""
        return {
            "available_losses": list(self.loss_functions.keys()),
            "loss_weights": self.loss_weights,
            "total_losses": len(self.loss_functions),
        }

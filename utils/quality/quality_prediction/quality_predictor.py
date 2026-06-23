"""Predict and optimize image quality during generation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QualityPredictor(nn.Module):
    """Predict image quality from latents or images."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
        )

        self.quality_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.diversity_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.aesthetics_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict quality metrics.

        Returns: (overall_quality, diversity_score, aesthetics_score)
        """
        if x.dim() == 4:
            b, c, h, w = x.shape
            x = F.adaptive_avg_pool2d(x, 1).view(b, -1)

        features = self.backbone(x)

        quality = self.quality_head(features)
        diversity = self.diversity_head(features)
        aesthetics = self.aesthetics_head(features)

        return quality, diversity, aesthetics


class PerceptualQualityLoss(nn.Module):
    """Loss for perceptual quality based on high-level features."""

    def __init__(self, pretrained_model: nn.Module):
        super().__init__()
        self.feature_extractor = pretrained_model
        self.feature_extractor.eval()

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, generated: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Compute perceptual quality loss."""
        gen_features = self.feature_extractor(generated)
        ref_features = self.feature_extractor(reference)

        loss = F.mse_loss(gen_features, ref_features)
        return loss


class ImageSharpnessPredictor:
    """Predict sharpness/clarity of generated image."""

    @staticmethod
    def compute_laplacian_variance(image: torch.Tensor) -> torch.Tensor:
        """Measure sharpness via Laplacian variance."""
        if image.dim() == 4:
            image = image.mean(dim=1)

        kernel = torch.tensor([[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]], dtype=torch.float32, device=image.device)

        laplacian = F.conv2d(image.unsqueeze(1), kernel.unsqueeze(1), padding=1)

        variance = laplacian.var()
        return variance

    @staticmethod
    def compute_edge_density(image: torch.Tensor) -> torch.Tensor:
        """Measure edge density (detail level)."""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=image.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=image.device)

        if image.dim() == 4:
            image = image.mean(dim=1)

        edges_x = F.conv2d(image.unsqueeze(1), sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
        edges_y = F.conv2d(image.unsqueeze(1), sobel_y.unsqueeze(0).unsqueeze(0), padding=1)

        edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2)

        return edge_magnitude.mean()


class ColorQuality:
    """Assess color quality of generated images."""

    @staticmethod
    def compute_color_saturation(image: torch.Tensor) -> torch.Tensor:
        """Compute color saturation level."""
        if image.dim() == 4:
            if image.shape[1] != 3:
                image = image.mean(dim=1, keepdim=True)

            hsv = ImageSharpnessPredictor._rgb_to_hsv(image)
            saturation = hsv[:, 1]  # S channel
            return saturation.mean()

        return torch.tensor(0.5)

    @staticmethod
    def compute_color_diversity(image: torch.Tensor) -> torch.Tensor:
        """Measure color diversity (histogram entropy)."""
        if image.dim() == 4:
            image = image.view(image.shape[0], -1)
        else:
            image = image.flatten()

        hist = torch.histc(image, bins=256, min=0, max=1)
        hist = hist / hist.sum()
        entropy = -(hist[hist > 0] * torch.log(hist[hist > 0])).sum()

        return entropy

    @staticmethod
    def _rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
        """Convert RGB to HSV for color analysis."""
        r, g, b = image[:, 0], image[:, 1], image[:, 2]

        max_c = torch.stack([r, g, b], dim=1).max(dim=1)[0]
        min_c = torch.stack([r, g, b], dim=1).min(dim=1)[0]

        v = max_c
        c = max_c - min_c
        s = c / (v + 1e-8)

        h = torch.zeros_like(v)
        mask_r = max_c == r
        mask_g = max_c == g
        mask_b = max_c == b

        h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / (c[mask_r] + 1e-8)) % 6)
        h[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / (c[mask_g] + 1e-8)) + 2)
        h[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / (c[mask_b] + 1e-8)) + 4)

        return torch.stack([h, s, v], dim=1)


class SemanticQualityAssessment:
    """Assess semantic correctness of generated images."""

    def __init__(self, clip_model: nn.Module):
        self.clip = clip_model
        self.clip.eval()

    def compute_prompt_alignment(self, image: torch.Tensor, prompt: str) -> torch.Tensor:
        """Measure alignment between image and prompt."""
        image_features = self.clip.encode_image(image)
        text_features = self.clip.encode_text(prompt)

        similarity = F.cosine_similarity(image_features, text_features)
        return similarity

    def compute_object_presence(self, image: torch.Tensor, object_names: list[str]) -> dict[str, float]:
        """Check for presence of specific objects."""
        image_features = self.clip.encode_image(image)

        scores = {}
        for obj_name in object_names:
            text_features = self.clip.encode_text(f"a {obj_name}")
            similarity = F.cosine_similarity(image_features, text_features)
            scores[obj_name] = float(similarity)

        return scores


class QualityOptimizer:
    """Optimize generation towards target quality metrics."""

    def __init__(self, quality_predictor: QualityPredictor):
        self.predictor = quality_predictor

    def optimize_latents_for_quality(
        self, latents: torch.Tensor, num_iterations: int = 5, target_quality: float = 0.9
    ) -> torch.Tensor:
        """Optimize latents to maximize predicted quality."""
        latents = latents.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([latents], lr=0.01)

        for _ in range(num_iterations):
            quality_score, _, _ = self.predictor(latents)

            loss = F.mse_loss(quality_score, torch.tensor(target_quality))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return latents.detach()

    def score_generation(self, latents: torch.Tensor) -> dict:
        """Score generation on multiple quality dimensions."""
        with torch.no_grad():
            quality, diversity, aesthetics = self.predictor(latents)

        return {
            "overall_quality": float(quality.mean()),
            "diversity": float(diversity.mean()),
            "aesthetics": float(aesthetics.mean()),
            "combined_score": float((quality.mean() + diversity.mean() + aesthetics.mean()) / 3.0),
        }

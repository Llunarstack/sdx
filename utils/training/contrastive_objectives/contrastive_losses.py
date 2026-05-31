"""Contrastive learning objectives for improving image-text alignment."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class NTXentLoss(torch.nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

    Widely used in contrastive learning (SimCLR, CLIP, etc.).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent loss between two views.

        Args:
            z_i: Embeddings from first view [B, D]
            z_j: Embeddings from second view [B, D]

        Returns:
            Scalar loss
        """
        batch_size = z_i.shape[0]

        z = torch.cat([z_i, z_j], dim=0)
        z = F.normalize(z, dim=1)

        similarity_matrix = torch.matmul(z, z.t())
        similarity_matrix = similarity_matrix / self.temperature

        pos_mask = torch.zeros(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=z_i.device)
        for i in range(batch_size):
            pos_mask[i, batch_size + i] = True
            pos_mask[batch_size + i, i] = True

        neg_mask = ~pos_mask
        neg_mask.fill_diagonal_(False)

        pos = similarity_matrix[pos_mask].view(2 * batch_size, 1)

        neg = similarity_matrix[neg_mask].view(2 * batch_size, -1)

        logits = torch.cat([pos, neg], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z_i.device)

        loss = F.cross_entropy(logits, labels)
        return loss


class AlignmentLoss(torch.nn.Module):
    """Loss encouraging alignment between image and text embeddings."""

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, image_embed: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
        """Compute alignment loss.

        Maximizes similarity between matched pairs while minimizing
        similarity between mismatched pairs.

        Args:
            image_embed: Image embeddings [B, D]
            text_embed: Text embeddings [B, D]

        Returns:
            Scalar loss
        """
        batch_size = image_embed.shape[0]

        image_embed = F.normalize(image_embed, dim=1)
        text_embed = F.normalize(text_embed, dim=1)

        similarity = torch.matmul(image_embed, text_embed.t())

        pos_sim = torch.diagonal(similarity)

        neg_sim = similarity.clone()
        neg_sim.fill_diagonal_(0.0)

        max_neg = neg_sim.max(dim=1)[0]

        loss = torch.clamp(self.margin + max_neg - pos_sim, min=0.0).mean()

        return loss


class UniformityLoss(torch.nn.Module):
    """Loss encouraging uniform distribution of embeddings on hypersphere.

    Helps prevent mode collapse and improves representation diversity.
    """

    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute uniformity loss.

        Args:
            embeddings: Normalized embeddings [B, D]

        Returns:
            Scalar loss
        """
        n = embeddings.shape[0]

        embeddings = F.normalize(embeddings, dim=1)

        x_prod = torch.matmul(embeddings, embeddings.t())

        mask = torch.eye(n, dtype=torch.bool, device=embeddings.device)
        x_prod = x_prod[~mask].view(n, -1)

        approx_log_partition = torch.logsumexp(x_prod / self.temperature, dim=1).mean()

        return -approx_log_partition


class TripletLoss(torch.nn.Module):
    """Triplet loss for metric learning.

    Uses anchor-positive-negative triplets to learn embeddings.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """Compute triplet loss.

        Args:
            anchor: Anchor embeddings [B, D]
            positive: Positive embeddings [B, D]
            negative: Negative embeddings [B, D]

        Returns:
            Scalar loss
        """
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negative = F.normalize(negative, dim=1)

        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)

        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0).mean()

        return loss


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Loss (SupCon).

    Extension of NT-Xent for supervised learning with multiple positives.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute supervised contrastive loss.

        Args:
            features: Feature representations [B, D]
            labels: Class labels [B]

        Returns:
            Scalar loss
        """
        batch_size = features.shape[0]

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.t())
        similarity_matrix = similarity_matrix / self.temperature

        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        pos_mask.fill_diagonal_(0)

        neg_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()

        log_probs = F.log_softmax(similarity_matrix, dim=1)

        pos_logps = (pos_mask * log_probs).sum(1) / pos_mask.sum(1)

        neg_logps = (neg_mask * log_probs).sum(1) / neg_mask.sum(1)

        loss = -(pos_logps + neg_logps).mean()

        return loss


class ImageTextMatchingLoss(torch.nn.Module):
    """Loss specifically designed for image-text matching in diffusion.

    Combines multiple contrastive objectives optimized for text-to-image.
    """

    def __init__(
        self,
        alignment_weight: float = 0.5,
        nt_xent_weight: float = 0.3,
        uniformity_weight: float = 0.2,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.alignment_loss = AlignmentLoss(margin=0.5)
        self.nt_xent_loss = NTXentLoss(temperature=temperature)
        self.uniformity_loss = UniformityLoss(temperature=2.0)

        self.alignment_weight = alignment_weight
        self.nt_xent_weight = nt_xent_weight
        self.uniformity_weight = uniformity_weight

    def forward(
        self,
        image_embed: torch.Tensor,
        text_embed: torch.Tensor,
        image_features_aug: torch.Tensor | None = None,
        text_features_aug: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute combined image-text matching loss.

        Args:
            image_embed: Image embeddings [B, D]
            text_embed: Text embeddings [B, D]
            image_features_aug: Augmented image embeddings (optional)
            text_features_aug: Augmented text embeddings (optional)

        Returns:
            Scalar loss
        """
        loss = 0.0

        if self.alignment_weight > 0:
            align_loss = self.alignment_loss(image_embed, text_embed)
            loss += self.alignment_weight * align_loss

        if self.nt_xent_weight > 0 and image_features_aug is not None and text_features_aug is not None:
            nt_xent = self.nt_xent_loss(image_features_aug, text_features_aug)
            loss += self.nt_xent_weight * nt_xent

        if self.uniformity_weight > 0:
            combined_embed = torch.cat([image_embed, text_embed], dim=0)
            unif_loss = self.uniformity_loss(combined_embed)
            loss += self.uniformity_weight * unif_loss

        return loss

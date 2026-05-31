"""Contrastive learning objectives."""

from .contrastive_losses import AlignmentLoss, ImageTextMatchingLoss, NTXentLoss, SupConLoss, TripletLoss, UniformityLoss

__all__ = [
    "AlignmentLoss",
    "ImageTextMatchingLoss",
    "NTXentLoss",
    "SupConLoss",
    "TripletLoss",
    "UniformityLoss",
]

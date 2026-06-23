"""Adaptive training that improves during training (learns better per-sample)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AdaptiveTrainingConfig:
    """Configuration for adaptive training."""

    enable_loss_scaling: bool = True
    enable_curriculum: bool = True
    enable_meta_learning: bool = False
    loss_scale_factor: float = 0.5
    curriculum_schedule: str = "linear"  # linear, exponential, cosine


class AdaptiveLossScaling:
    """Dynamically scale loss per sample based on difficulty."""

    def __init__(self, base_loss_scale: float = 1.0, adaptation_rate: float = 0.01):
        self.base_loss_scale = base_loss_scale
        self.adaptation_rate = adaptation_rate
        self.loss_history = {}

    def compute_sample_weights(self, losses: torch.Tensor, batch_ids: list[str]) -> torch.Tensor:
        """Compute per-sample weight based on loss history.

        Harder samples (higher loss) get higher weight.
        """
        weights = torch.ones_like(losses)

        for i, bid in enumerate(batch_ids):
            if bid not in self.loss_history:
                self.loss_history[bid] = []

            self.loss_history[bid].append(float(losses[i]))

            if len(self.loss_history[bid]) > 1:
                avg_loss = sum(self.loss_history[bid]) / len(self.loss_history[bid])
                weights[i] = avg_loss

        weights = weights / (weights.mean() + 1e-8)
        return weights

    def weighted_loss(self, losses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Compute weighted loss."""
        return (losses * weights).mean()


class CurriculumLearning:
    """Progress from easy to hard examples during training."""

    def __init__(self, difficulty_fn, schedule: str = "linear"):
        self.difficulty_fn = difficulty_fn
        self.schedule = schedule
        self.progress = 0.0

    def update_progress(self, current_step: int, total_steps: int):
        """Update curriculum progress."""
        self.progress = current_step / total_steps

    def get_difficulty_threshold(self) -> float:
        """Get current difficulty threshold based on schedule."""
        if self.schedule == "linear":
            return self.progress

        elif self.schedule == "exponential":
            return 1.0 - (1.0 - self.progress) ** 2

        elif self.schedule == "cosine":
            return 0.5 * (1.0 - torch.cos(torch.tensor(self.progress * 3.14159)).item())

        return self.progress

    def filter_batch(self, batch: tuple, labels: torch.Tensor) -> tuple:
        """Filter batch to include only appropriate difficulty samples."""
        difficulties = self.difficulty_fn(labels)

        threshold = self.get_difficulty_threshold()

        mask = difficulties <= threshold
        if mask.sum() == 0:
            mask = torch.ones_like(mask)

        filtered_batch = tuple(b[mask] if isinstance(b, torch.Tensor) else b for b in batch)
        return filtered_batch


class MetaLearning:
    """Learn to learn - adapt learning rate per layer."""

    def __init__(self, model: nn.Module, meta_lr: float = 0.001):
        self.model = model
        self.meta_lr = meta_lr
        self.layer_lrs = {}

        for name, param in model.named_parameters():
            self.layer_lrs[name] = 0.001

    def compute_layer_importance(self, gradients: dict[str, torch.Tensor]) -> dict[str, float]:
        """Compute importance of each layer's gradients."""
        importance = {}

        for name, grad in gradients.items():
            if grad is not None:
                importance[name] = float((grad**2).mean())

        return importance

    def update_layer_lrs(self, importances: dict[str, float]):
        """Update learning rates based on importance."""
        if not importances:
            return

        max_importance = max(importances.values())

        for name in self.layer_lrs:
            if name in importances:
                norm_importance = importances[name] / (max_importance + 1e-8)
                self.layer_lrs[name] *= 1.0 + self.meta_lr * (norm_importance - 0.5)

                self.layer_lrs[name] = max(1e-5, min(0.1, self.layer_lrs[name]))

    def get_per_layer_optimizer(self, model: nn.Module) -> dict:
        """Create per-layer optimizer with adapted learning rates."""
        param_groups = []

        for name, param in model.named_parameters():
            param_groups.append({"params": [param], "lr": self.layer_lrs.get(name, 0.001)})

        return param_groups


class AdaptiveWeightDecay:
    """Adaptive weight decay per layer."""

    def __init__(self, model: nn.Module, base_wd: float = 0.01):
        self.model = model
        self.base_wd = base_wd
        self.layer_wds = {}

        for name, _ in model.named_parameters():
            self.layer_wds[name] = base_wd

    def compute_layer_weight_decay(self, weight_norms: dict[str, float]) -> dict[str, float]:
        """Compute weight decay per layer based on weight magnitude."""
        wds = {}

        for name, norm in weight_norms.items():
            if norm > 1.0:
                wds[name] = self.base_wd * (1.0 + norm)
            else:
                wds[name] = self.base_wd * norm

        return wds


class DynamicBatchNormalization:
    """Adaptive batch norm that tracks statistics more carefully."""

    def __init__(self, num_features: int, momentum: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive normalization."""
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

            x_norm = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        return self.weight * x_norm + self.bias


class GradientAdaptation:
    """Adapt gradients based on training dynamics."""

    def __init__(self, adaptation_strength: float = 0.1):
        self.adaptation_strength = adaptation_strength
        self.gradient_history = {}

    def adapt_gradients(self, gradients: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Adapt gradients to reduce instability."""
        adapted = {}

        for name, grad in gradients.items():
            if name not in self.gradient_history:
                self.gradient_history[name] = grad.clone().detach()

            prev_grad = self.gradient_history[name]

            grad_similarity = F.cosine_similarity(grad.flatten(), prev_grad.flatten(), dim=0)

            if grad_similarity < 0.5:
                adapted[name] = grad * (1.0 - self.adaptation_strength) + prev_grad * self.adaptation_strength
            else:
                adapted[name] = grad

            self.gradient_history[name] = grad.clone().detach()

        return adapted

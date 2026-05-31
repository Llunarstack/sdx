"""Multi-model ensemble training for improved quality through committee learning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass
class EnsembleTrainingConfig:
    """Configuration for ensemble training."""

    ensemble_size: int = 3
    voting_strategy: str = "consensus"
    diversity_weight: float = 0.1
    consensus_threshold: float = 0.7
    enable_knowledge_distillation: bool = True
    distillation_temperature: float = 4.0


class EnsembleTrainer:
    """Trains multiple models jointly with diversity encouragement."""

    def __init__(self, models: list[torch.nn.Module], config: EnsembleTrainingConfig):
        """Initialize ensemble trainer.

        Args:
            models: List of models to train in ensemble
            config: Ensemble training configuration
        """
        self.models = models
        self.config = config
        self.ensemble_size = len(models)
        self.training_stats = {"disagreement": [], "loss_variance": []}

    def forward_ensemble(self, x: torch.Tensor, timesteps: torch.Tensor, conditions: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through all ensemble members.

        Args:
            x: Input tensor
            timesteps: Timestep conditioning
            conditions: Text/control conditioning

        Returns:
            List of predictions from each model
        """
        predictions = []
        for model in self.models:
            pred = model(x, timesteps, conditions)
            predictions.append(pred)
        return predictions

    def ensemble_forward_with_consensus(
        self, x: torch.Tensor, timesteps: torch.Tensor, conditions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Forward pass with consensus voting.

        Returns:
            (consensus_prediction, ensemble_variance, disagreement_score)
        """
        predictions = self.forward_ensemble(x, timesteps, conditions)
        predictions_stacked = torch.stack(predictions)

        consensus_pred = predictions_stacked.mean(dim=0)

        variance = predictions_stacked.var(dim=0).mean()

        disagreement = self._compute_disagreement(predictions)

        self.training_stats["disagreement"].append(float(disagreement))

        return consensus_pred, variance, disagreement

    def _compute_disagreement(self, predictions: list[torch.Tensor]) -> torch.Tensor:
        """Compute disagreement between ensemble members.

        Higher disagreement = more diverse predictions.
        """
        if len(predictions) < 2:
            return torch.tensor(0.0)

        disagreement = 0.0
        pair_count = 0

        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                dist = F.mse_loss(predictions[i], predictions[j])
                disagreement += dist
                pair_count += 1

        return disagreement / max(1, pair_count)

    def ensemble_loss(
        self,
        predictions: list[torch.Tensor],
        targets: torch.Tensor,
        variance: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ensemble loss with diversity regularization.

        Combines:
        1. Individual model losses
        2. Consensus loss
        3. Diversity encouragement loss
        """
        individual_losses = []
        for pred in predictions:
            loss = F.mse_loss(pred, targets)
            individual_losses.append(loss)

        consensus_pred = torch.stack(predictions).mean(dim=0)
        consensus_loss = F.mse_loss(consensus_pred, targets)

        diversity_loss = -self.config.diversity_weight * variance.mean()

        total_loss = (sum(individual_losses) / len(individual_losses)) + consensus_loss + diversity_loss

        return total_loss

    def train_step(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        timesteps: torch.Tensor,
        conditions: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> dict:
        """Single training step for ensemble.

        Args:
            x: Input tensor
            targets: Target outputs
            timesteps: Timestep conditioning
            conditions: Text conditioning
            optimizer: Optimizer for all models

        Returns:
            Loss dictionary
        """
        optimizer.zero_grad()

        predictions, variance, disagreement = self.ensemble_forward_with_consensus(x, timesteps, conditions)

        loss = self.ensemble_loss(
            [self.forward_ensemble(x, timesteps, conditions)[i] for i in range(len(self.models))],
            targets,
            variance,
        )

        if self.config.enable_knowledge_distillation:
            kd_loss = self._compute_knowledge_distillation_loss(predictions, self.config.distillation_temperature)
            loss = loss + 0.1 * kd_loss

        loss.backward()
        optimizer.step()

        return {
            "total_loss": float(loss),
            "ensemble_disagreement": float(disagreement),
            "variance": float(variance),
        }

    def _compute_knowledge_distillation_loss(self, predictions: list[torch.Tensor], temperature: float) -> torch.Tensor:
        """Knowledge distillation between ensemble members.

        Each model learns from others' predictions.
        """
        if len(predictions) < 2:
            return torch.tensor(0.0, device=predictions[0].device)

        consensus = torch.stack(predictions).mean(dim=0)

        kd_loss = 0.0
        for pred in predictions:
            log_probs = F.log_softmax(pred / temperature, dim=-1)
            soft_targets = F.softmax(consensus / temperature, dim=-1)
            kd_loss += F.kl_div(log_probs, soft_targets, reduction="batchmean") * (temperature ** 2)

        return kd_loss / len(predictions)

    def get_consensus_model(self) -> torch.Tensor:
        """Get averaged model weights (model soup).

        Returns consensus by averaging all models.
        """
        averaged_state = None

        for model in self.models:
            state = model.state_dict()
            if averaged_state is None:
                averaged_state = {k: v.clone() for k, v in state.items()}
            else:
                for k in averaged_state:
                    averaged_state[k] = averaged_state[k] + state[k]

        for k in averaged_state:
            averaged_state[k] = averaged_state[k] / len(self.models)

        return averaged_state

    def save_ensemble(self, checkpoint_dir: str | Path) -> None:
        """Save all ensemble models."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), checkpoint_dir / f"model_{i}.pt")

        consensus_state = self.get_consensus_model()
        torch.save(consensus_state, checkpoint_dir / "consensus_model.pt")

    def load_ensemble(self, checkpoint_dir: str | Path) -> None:
        """Load all ensemble models."""
        checkpoint_dir = Path(checkpoint_dir)

        for i, model in enumerate(self.models):
            state = torch.load(checkpoint_dir / f"model_{i}.pt")
            model.load_state_dict(state)

    def get_training_summary(self) -> dict:
        """Get summary of ensemble training statistics."""
        return {
            "ensemble_size": self.ensemble_size,
            "avg_disagreement": sum(self.training_stats["disagreement"]) / max(1, len(self.training_stats["disagreement"])),
            "disagreement_trend": self.training_stats["disagreement"][-10:] if self.training_stats["disagreement"] else [],
        }

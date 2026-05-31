"""Online hard negative mining for improved training data selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass
class HardNegativeExample:
    """Represents a hard negative example for training."""

    image_path: str
    caption: str
    difficulty_score: float
    loss_at_discovery: float
    discovery_step: int
    reason: str


class HardNegativeMiner:
    """Discovers hard examples during training for curriculum-based learning."""

    def __init__(
        self,
        difficulty_threshold: float = 0.7,
        max_hard_negatives: int = 1000,
        loss_smoothing: float = 0.95,
    ):
        """Initialize hard negative miner.

        Args:
            difficulty_threshold: Threshold for considering an example "hard" (0-1)
            max_hard_negatives: Maximum hard negatives to keep in memory
            loss_smoothing: EMA smoothing for loss history
        """
        self.difficulty_threshold = difficulty_threshold
        self.max_hard_negatives = max_hard_negatives
        self.loss_smoothing = loss_smoothing

        self.hard_negatives: list[HardNegativeExample] = []
        self.loss_history = {}
        self.step = 0

    def evaluate_difficulty(self, model_output: torch.Tensor, target: torch.Tensor, loss: float) -> float:
        """Compute difficulty score for an example.

        Difficulty is based on:
        - Loss magnitude (higher = harder)
        - Prediction uncertainty (entropy of model predictions)
        - Prediction-target divergence
        """
        normalized_loss = min(1.0, loss / 10.0)

        if model_output.shape[-1] > 1:
            probs = F.softmax(model_output, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
            entropy_score = float(entropy) / 8.0
        else:
            entropy_score = 0.0

        divergence = float(torch.abs(model_output - target).mean())
        normalized_divergence = min(1.0, divergence)

        difficulty = 0.5 * normalized_loss + 0.3 * entropy_score + 0.2 * normalized_divergence

        return min(1.0, float(difficulty))

    def record_batch(
        self,
        image_paths: list[str],
        captions: list[str],
        model_outputs: torch.Tensor,
        targets: torch.Tensor,
        batch_losses: torch.Tensor,
    ) -> list[HardNegativeExample]:
        """Record hard examples from a training batch.

        Args:
            image_paths: Paths to images in batch
            captions: Captions for images
            model_outputs: Model predictions
            targets: Target values
            batch_losses: Per-sample losses

        Returns:
            New hard negatives discovered
        """
        new_hard = []

        for i, (img_path, caption) in enumerate(zip(image_paths, captions)):
            loss = float(batch_losses[i])
            model_out = model_outputs[i : i + 1]
            target = targets[i : i + 1]

            difficulty = self.evaluate_difficulty(model_out, target, loss)

            if difficulty > self.difficulty_threshold:
                reason = self._classify_difficulty_reason(difficulty, loss, model_out, target)

                example = HardNegativeExample(
                    image_path=img_path,
                    caption=caption,
                    difficulty_score=difficulty,
                    loss_at_discovery=loss,
                    discovery_step=self.step,
                    reason=reason,
                )

                new_hard.append(example)
                self.hard_negatives.append(example)

                if img_path not in self.loss_history:
                    self.loss_history[img_path] = loss
                else:
                    ema_loss = self.loss_smoothing * self.loss_history[img_path] + (1 - self.loss_smoothing) * loss
                    self.loss_history[img_path] = ema_loss

        if len(self.hard_negatives) > self.max_hard_negatives:
            self.hard_negatives = sorted(
                self.hard_negatives, key=lambda x: x.difficulty_score, reverse=True
            )[: self.max_hard_negatives]

        self.step += 1
        return new_hard

    def _classify_difficulty_reason(
        self, difficulty: float, loss: float, model_out: torch.Tensor, target: torch.Tensor
    ) -> str:
        """Classify why an example is hard."""
        reasons = []

        if loss > 2.0:
            reasons.append("high_loss")

        if model_out.shape[-1] > 1:
            entropy = -torch.sum(
                F.softmax(model_out, dim=-1) * torch.log(F.softmax(model_out, dim=-1) + 1e-8), dim=-1
            )
            if float(entropy) > 2.0:
                reasons.append("high_uncertainty")

        divergence = float(torch.abs(model_out - target).mean())
        if divergence > 0.5:
            reasons.append("prediction_divergence")

        if not reasons:
            reasons.append("multi_factor")

        return "|".join(reasons)

    def get_hard_negatives_for_curriculum(self, num_samples: int = 32) -> list[tuple[str, str]]:
        """Get hard negatives for curriculum-based training.

        Returns samples ranked by difficulty, useful for hard-negative mining curriculum.
        """
        if not self.hard_negatives:
            return []

        sorted_hard = sorted(self.hard_negatives, key=lambda x: x.difficulty_score, reverse=True)
        return [(ex.image_path, ex.caption) for ex in sorted_hard[:num_samples]]

    def export_hard_negatives(self, output_path: str | Path) -> None:
        """Export discovered hard negatives to file for inspection/reuse."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write("image_path,caption,difficulty_score,loss,discovery_step,reason\n")
            for ex in sorted(self.hard_negatives, key=lambda x: x.difficulty_score, reverse=True):
                caption_safe = ex.caption.replace(",", ";").replace("\n", " ")
                f.write(
                    f"{ex.image_path},{caption_safe},{ex.difficulty_score:.4f},"
                    f"{ex.loss_at_discovery:.4f},{ex.discovery_step},{ex.reason}\n"
                )

    def get_difficulty_distribution(self) -> dict:
        """Get statistics on difficulty distribution."""
        if not self.hard_negatives:
            return {"total": 0}

        scores = [ex.difficulty_score for ex in self.hard_negatives]
        return {
            "total_hard_negatives": len(self.hard_negatives),
            "mean_difficulty": float(sum(scores) / len(scores)),
            "max_difficulty": float(max(scores)),
            "min_difficulty": float(min(scores)),
            "easy_threshold": float(sum(1 for s in scores if s < 0.5) / len(scores) * 100),
            "medium_threshold": float(sum(1 for s in scores if 0.5 <= s < 0.8) / len(scores) * 100),
            "hard_threshold": float(sum(1 for s in scores if s >= 0.8) / len(scores) * 100),
        }

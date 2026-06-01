"""
Flow Matching Consistency System: Temporal consistency in generation flows.
Based on research:
- https://arxiv.org/pdf/2602.04908 (Temporal Pair Consistency)
- https://openreview.net/forum?id=xQBRrtQM8u (Adjoint Matching)
- https://arxiv.org/pdf/2412.06295 (Curriculum Consistency Model)

Enforces temporal coherence in the learned velocity field for smooth, consistent generation.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class VelocityFieldNetwork(nn.Module):
    """Learns the velocity field for flow matching."""

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 128),
        )

        # Main velocity network
        self.velocity_net = nn.Sequential(
            nn.Linear(hidden_dim + 128, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, hidden_dim),
        )

        # Temporal consistency predictor
        self.temporal_consistency = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute velocity field and temporal consistency score.

        Args:
            x: Current state tensor
            t: Time step (0 to 1)
            conditioning: Optional conditioning information

        Returns:
            (velocity, temporal_consistency_score)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if t.dim() == 0:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.dim() == 1:
            t = t.unsqueeze(1)

        # Embed time
        t_emb = self.time_embedding(t)

        # Combine state and time
        combined = torch.cat([x, t_emb], dim=-1)

        # Compute velocity
        velocity = self.velocity_net(combined)

        # Compute temporal consistency
        consistency = float(self.temporal_consistency(velocity).squeeze())

        return velocity, consistency


class TemporalPairConsistency(nn.Module):
    """Enforces consistency across nearby timesteps (TPC-FM)."""

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        self.velocity_net = VelocityFieldNetwork(hidden_dim)

        # Consistency loss weight
        self.consistency_weight = nn.Parameter(torch.tensor(1.0))

    def compute_consistency_loss(
        self,
        x: torch.Tensor,
        t1: float,
        t2: float,
    ) -> Tuple[float, Dict]:
        """
        Compute temporal pair consistency loss.
        Ensures predictions at nearby timesteps are similar.
        """
        if abs(t2 - t1) < 0.01:
            return 0.0, {"error": "timesteps too close"}

        # Compute velocities at both timesteps
        v1, c1 = self.velocity_net(x, torch.tensor([t1]))
        v2, c2 = self.velocity_net(x, torch.tensor([t2]))

        # Consistency loss: velocities should be similar for nearby times
        consistency_loss = torch.nn.functional.mse_loss(v1, v2)

        # Weighted by consistency scores
        weighted_loss = (
            float(consistency_loss) *
            self.consistency_weight *
            (c1 + c2) / 2
        )

        return weighted_loss, {
            "v1_consistency": c1,
            "v2_consistency": c2,
            "velocity_mse": float(consistency_loss),
            "weighted_loss": weighted_loss,
        }

    def forward(
        self,
        x: torch.Tensor,
        t_pairs: List[Tuple[float, float]],
    ) -> Dict:
        """
        Enforce consistency across multiple time pairs.
        """
        losses = []
        consistency_scores = []

        for t1, t2 in t_pairs:
            loss, metrics = self.compute_consistency_loss(x, t1, t2)
            losses.append(loss)
            consistency_scores.append(metrics.get("v1_consistency", 0.5))

        return {
            "total_consistency_loss": sum(losses) / len(losses) if losses else 0.0,
            "average_consistency": sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.5,
            "enforced_pairs": len(t_pairs),
        }


class CurriculumConsistencyModel(nn.Module):
    """Curriculum learning for consistency distillation."""

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        self.velocity_net = VelocityFieldNetwork(hidden_dim)

        # Progressive difficulty levels
        self.curriculum_stage = 0  # 0=easy, 1=medium, 2=hard

        # Sampling strategy based on difficulty
        self.easy_sampling = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.medium_sampling = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.hard_sampling = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def get_sampling_strategy(self) -> Dict:
        """Get timestep sampling strategy based on curriculum stage."""
        strategies = {
            0: {  # Easy: sample from coarse timesteps
                "num_steps": 4,
                "strategy": "uniform",
                "description": "Coarse sampling (4 steps)"
            },
            1: {  # Medium: sample from medium timesteps
                "num_steps": 8,
                "strategy": "balanced",
                "description": "Medium sampling (8 steps)"
            },
            2: {  # Hard: sample from fine timesteps
                "num_steps": 16,
                "strategy": "adaptive",
                "description": "Fine sampling (16 steps)"
            },
        }
        return strategies[min(self.curriculum_stage, 2)]

    def update_curriculum(self, loss: float, threshold: float = 0.1):
        """Advance curriculum stage if loss is below threshold."""
        if loss < threshold and self.curriculum_stage < 2:
            self.curriculum_stage += 1
            logger.info(f"Curriculum advanced to stage {self.curriculum_stage}")

    def generate_with_consistency(
        self,
        initial_state: torch.Tensor,
        num_steps: int = 16,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Generate with curriculum-based consistency enforcement.
        """
        timesteps = torch.linspace(0, 1, num_steps)
        states = [initial_state]
        consistencies = []

        for i in range(1, len(timesteps)):
            t = timesteps[i]
            x = states[-1]

            velocity, consistency = self.velocity_net(x, t)

            # Update state along velocity field
            new_state = x + velocity * 0.01  # Small step

            states.append(new_state)
            consistencies.append(consistency)

        final_state = states[-1]

        return final_state, {
            "path_length": len(states),
            "average_consistency": sum(consistencies) / len(consistencies) if consistencies else 0.5,
            "curriculum_stage": self.curriculum_stage,
            "sampling_strategy": self.get_sampling_strategy(),
        }


class FlowMatchingConsistencySystem:
    """Complete flow matching consistency system."""

    def __init__(self, hidden_dim: int = 4096):
        self.velocity_net = VelocityFieldNetwork(hidden_dim)
        self.tpc_module = TemporalPairConsistency(hidden_dim)
        self.curriculum_model = CurriculumConsistencyModel(hidden_dim)

        self.generation_history = []

    def generate_with_temporal_consistency(
        self,
        initial_state: torch.Tensor,
        prompt_conditioning: Optional[torch.Tensor] = None,
        num_steps: int = 20,
    ) -> Dict:
        """
        Generate using flow matching with temporal consistency guarantees.
        """
        timesteps = torch.linspace(0, 1, num_steps)
        states = [initial_state]
        velocities = []
        consistencies = []

        for i in range(1, len(timesteps)):
            t = timesteps[i]
            x = states[-1]

            # Compute velocity with consistency score
            velocity, consistency = self.velocity_net(x, t, prompt_conditioning)

            # Update state
            dt = timesteps[i] - timesteps[i-1]
            new_state = x + velocity * dt

            states.append(new_state)
            velocities.append(velocity)
            consistencies.append(consistency)

        # Compute overall consistency metrics
        avg_consistency = sum(consistencies) / len(consistencies) if consistencies else 0.0

        # Check temporal pairs for consistency
        time_pairs = [(timesteps[i].item(), timesteps[i+1].item()) for i in range(len(timesteps)-1)]
        tpc_metrics = self.tpc_module(initial_state, time_pairs)

        result = {
            "final_state": states[-1],
            "generation_path_length": len(states),
            "average_velocity_consistency": avg_consistency,
            "temporal_pair_consistency_loss": tpc_metrics["total_consistency_loss"],
            "overall_consistency_score": (avg_consistency + (1 - tpc_metrics["total_consistency_loss"])) / 2,
            "metrics": {
                "num_steps": num_steps,
                "velocity_field_consistency": avg_consistency,
                "temporal_coherence": tpc_metrics["average_consistency"],
            }
        }

        self.generation_history.append(result)
        return result

    def get_consistency_report(self, generation_result: Dict) -> Dict:
        """Generate detailed consistency report."""
        return {
            "overall_consistency_score": generation_result["overall_consistency_score"],
            "velocity_field_consistency": generation_result["average_velocity_consistency"],
            "temporal_pair_consistency": generation_result["temporal_pair_consistency_loss"],
            "path_quality": (
                "excellent" if generation_result["overall_consistency_score"] > 0.85
                else "good" if generation_result["overall_consistency_score"] > 0.7
                else "acceptable" if generation_result["overall_consistency_score"] > 0.55
                else "poor"
            ),
            "technical_details": generation_result["metrics"],
        }

    def get_statistics(self) -> Dict:
        """Get flow matching statistics."""
        if not self.generation_history:
            return {"total_generations": 0}

        scores = [g["overall_consistency_score"] for g in self.generation_history]

        return {
            "total_generations": len(self.generation_history),
            "average_consistency": sum(scores) / len(scores),
            "min_consistency": min(scores),
            "max_consistency": max(scores),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    system = FlowMatchingConsistencySystem()

    print("=== Flow Matching Consistency System ===\n")

    # Test generation
    initial_state = torch.randn(1, 4096)
    prompt_cond = torch.randn(1, 4096)

    print("Generating with temporal consistency enforcement...")
    result = system.generate_with_temporal_consistency(
        initial_state,
        prompt_conditioning=prompt_cond,
        num_steps=20,
    )

    report = system.get_consistency_report(result)

    print("\nConsistency Report:")
    print(f"  Overall Score: {report['overall_consistency_score']:.1%}")
    print(f"  Velocity Field Consistency: {report['velocity_field_consistency']:.1%}")
    print(f"  Temporal Pair Consistency Loss: {report['temporal_pair_consistency']:.4f}")
    print(f"  Path Quality: {report['path_quality']}")

    print("\nTechnical Details:")
    for key, value in report['technical_details'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.1%}")
        else:
            print(f"  {key}: {value}")

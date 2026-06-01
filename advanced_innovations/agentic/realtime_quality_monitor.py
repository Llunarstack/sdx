"""
Real-Time Quality Monitoring Stream:
Continuous quality scoring during generation with early stopping capability.
Enables mid-generation redirection based on quality trajectory.
"""

import logging
from collections import deque
from typing import Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class StreamingQualityScorer(nn.Module):
    """Scores quality of generation at each timestep."""

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        # Timestep-aware quality assessment
        self.timestep_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 128),
        )

        # Multi-scale quality scorer
        self.quality_scorer = nn.Sequential(
            nn.Linear(hidden_dim + 128, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 10),  # 10 quality dimensions
            nn.Sigmoid(),
        )

        # Confidence scorer (how confident about quality assessment)
        self.confidence_scorer = nn.Sequential(
            nn.Linear(hidden_dim + 128, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def score_at_timestep(
        self,
        image_state: torch.Tensor,
        timestep: float,  # 0-1
    ) -> Dict:
        """Score quality at a specific timestep during generation."""
        if image_state.dim() == 1:
            image_state = image_state.unsqueeze(0)

        # Encode timestep
        t_tensor = torch.tensor([[timestep]])
        t_emb = self.timestep_encoder(t_tensor)

        # Combine state and timestep
        combined = torch.cat([image_state, t_emb], dim=-1)

        # Score quality
        quality_vector = self.quality_scorer(combined)
        confidence = float(self.confidence_scorer(combined).squeeze())

        overall_quality = float(quality_vector.mean())

        # Extract individual quality dimensions safely
        q_vec = quality_vector.squeeze().detach()

        return {
            "timestep": timestep,
            "quality_vector": q_vec.cpu().numpy(),
            "overall_quality": overall_quality,
            "confidence": confidence,
            "quality_dimensions": {
                "detail": float(q_vec[0]) if q_vec.dim() > 0 else overall_quality,
                "coherence": float(q_vec[1]) if q_vec.dim() > 0 and len(q_vec) > 1 else overall_quality,
                "colors": float(q_vec[2]) if q_vec.dim() > 0 and len(q_vec) > 2 else overall_quality,
                "composition": float(q_vec[3]) if q_vec.dim() > 0 and len(q_vec) > 3 else overall_quality,
                "lighting": float(q_vec[4]) if q_vec.dim() > 0 and len(q_vec) > 4 else overall_quality,
                "realism": float(q_vec[5]) if q_vec.dim() > 0 and len(q_vec) > 5 else overall_quality,
                "clarity": float(q_vec[6]) if q_vec.dim() > 0 and len(q_vec) > 6 else overall_quality,
                "consistency": float(q_vec[7]) if q_vec.dim() > 0 and len(q_vec) > 7 else overall_quality,
                "artifact_free": float(q_vec[8]) if q_vec.dim() > 0 and len(q_vec) > 8 else overall_quality,
                "aesthetic": float(q_vec[9]) if q_vec.dim() > 0 and len(q_vec) > 9 else overall_quality,
            },
        }


class QualityTrajectoryAnalyzer:
    """Analyzes quality evolution across timesteps."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.quality_history = deque(maxlen=window_size)

    def record_quality(self, quality_score: Dict):
        """Record quality measurement."""
        self.quality_history.append(quality_score)

    def get_quality_trend(self) -> Dict:
        """Analyze quality trend."""
        if len(self.quality_history) < 2:
            return {"status": "insufficient_data"}

        scores = [q["overall_quality"] for q in self.quality_history]

        # Trend detection
        if len(scores) > 1:
            recent_change = scores[-1] - scores[-2]
            avg_change = sum([scores[i] - scores[i-1] for i in range(1, len(scores))]) / (len(scores) - 1)

            trend = (
                "improving" if avg_change > 0.02 else
                "declining" if avg_change < -0.02 else
                "stable"
            )

            return {
                "trend": trend,
                "average_change_per_step": avg_change,
                "recent_change": recent_change,
                "current_quality": scores[-1],
                "average_quality": sum(scores) / len(scores),
            }

        return {"trend": "unknown"}

    def predict_final_quality(self, total_steps: int) -> Dict:
        """Predict final quality based on current trajectory."""
        if len(self.quality_history) < 3:
            return {"status": "insufficient_data"}

        scores = [q["overall_quality"] for q in self.quality_history]
        timesteps = [q["timestep"] for q in self.quality_history]

        # Linear extrapolation
        if len(timesteps) > 1:
            slope = (scores[-1] - scores[0]) / (timesteps[-1] - timesteps[0] + 1e-6)
            current_t = timesteps[-1]
            predicted_final = scores[-1] + slope * (1.0 - current_t)

            return {
                "predicted_final_quality": predicted_final,
                "current_quality": scores[-1],
                "trajectory_slope": slope,
                "expected_quality_level": (
                    "excellent" if predicted_final > 0.85
                    else "good" if predicted_final > 0.7
                    else "acceptable" if predicted_final > 0.55
                    else "poor"
                ),
            }

        return {"status": "unable_to_predict"}


class EarlyStoppingDecider(nn.Module):
    """Decides whether to stop generation early based on quality trajectory."""

    def __init__(self):
        super().__init__()

        # Early stopping predictor
        self.stopping_predictor = nn.Sequential(
            nn.Linear(10 * 2, 256),  # quality_vector from 2 timesteps
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Threshold settings
        self.quality_threshold = 0.8
        self.deterioration_threshold = 0.05  # Stop if quality drops 5% in one step
        self.stagnation_threshold = 0.005  # Stop if quality improves <0.5% over 5 steps

    def should_stop_early(
        self,
        quality_history: List[float],
        current_quality: float,
        timestep: float,
    ) -> Dict:
        """Determine if generation should stop early."""
        reasons = []
        stop = False

        # Check 1: Quality threshold reached
        if current_quality > self.quality_threshold:
            reasons.append("quality_threshold_reached")
            stop = True

        # Check 2: Quality deterioration
        if len(quality_history) > 0:
            last_quality = quality_history[-1]
            deterioration = last_quality - current_quality
            if deterioration > self.deterioration_threshold:
                reasons.append("quality_deteriorating")
                stop = True

        # Check 3: Stagnation (no improvement)
        if len(quality_history) >= 5:
            recent_scores = quality_history[-5:]
            avg_improvement = (recent_scores[-1] - recent_scores[0]) / 5
            if avg_improvement < self.stagnation_threshold and current_quality > 0.5:
                reasons.append("stagnated_improvement")
                stop = True

        # Check 4: Diminishing returns
        if len(quality_history) >= 3:
            recent_change = quality_history[-1] - quality_history[-2]
            if recent_change < 0.001 and timestep > 0.7:
                reasons.append("diminishing_returns")
                stop = False  # Only suggest, don't force stop

        return {
            "should_stop": stop,
            "reasons": reasons,
            "current_quality": current_quality,
            "timestep_progress": timestep,
            "confidence": float(len(reasons) / 4),  # More reasons = more confident
        }


class RealTimeQualityMonitoringSystem:
    """Complete real-time quality monitoring with streaming analysis."""

    def __init__(self, hidden_dim: int = 4096):
        self.scorer = StreamingQualityScorer(hidden_dim)
        self.trajectory_analyzer = QualityTrajectoryAnalyzer(window_size=10)
        self.early_stopper = EarlyStoppingDecider()

        self.generation_stream = []
        self.quality_history = []

    def monitor_generation_step(
        self,
        image_state: torch.Tensor,
        timestep: float,
        step_number: int = 0,
    ) -> Dict:
        """Monitor quality at each generation step."""
        # Score at this timestep
        quality_score = self.scorer.score_at_timestep(image_state, timestep)

        # Record in trajectory
        self.trajectory_analyzer.record_quality(quality_score)

        # Extract just scores for history (from already stored stream entries)
        quality_only = [e["quality_score"]["overall_quality"] for e in self.generation_stream]

        # Check early stopping
        early_stop_decision = self.early_stopper.should_stop_early(
            quality_only,
            quality_score["overall_quality"],
            timestep,
        )

        # Predict final quality
        final_prediction = self.trajectory_analyzer.predict_final_quality(100)

        # Get quality trend safely
        trend_info = self.trajectory_analyzer.get_quality_trend()
        trend = trend_info.get("trend", "stable")

        # Store in stream
        stream_entry = {
            "step": step_number,
            "timestep": timestep,
            "quality_score": quality_score,
            "early_stop_decision": early_stop_decision,
            "final_prediction": final_prediction,
        }
        self.generation_stream.append(stream_entry)

        return {
            "monitoring_active": True,
            "step": step_number,
            "timestep_progress": f"{timestep:.1%}",
            "current_quality": f"{quality_score['overall_quality']:.1%}",
            "quality_trend": trend,
            "early_stop_recommended": early_stop_decision["should_stop"],
            "early_stop_reasons": early_stop_decision["reasons"],
            "predicted_final_quality": final_prediction.get("predicted_final_quality", 0),
            "quality_dimensions": quality_score["quality_dimensions"],
        }

    def get_real_time_report(self) -> Dict:
        """Get current real-time monitoring report."""
        if not self.generation_stream:
            return {"status": "no_monitoring_data"}

        quality_scores = [e["quality_score"]["overall_quality"] for e in self.generation_stream]
        timestamps = [e["timestep"] for e in self.generation_stream]

        stopping_decisions = [e["early_stop_decision"] for e in self.generation_stream]
        should_stop_count = sum(1 for s in stopping_decisions if s["should_stop"])

        # Get quality trend safely
        trend_info = self.trajectory_analyzer.get_quality_trend()
        trend = trend_info.get("trend", "stable")

        return {
            "total_steps_monitored": len(self.generation_stream),
            "current_quality": quality_scores[-1],
            "average_quality": sum(quality_scores) / len(quality_scores),
            "max_quality": max(quality_scores),
            "quality_trend": trend,
            "early_stop_recommended_at_steps": [
                i for i, e in enumerate(self.generation_stream)
                if e["early_stop_decision"]["should_stop"]
            ],
            "estimated_time_saved": f"{sum(1 - t for t in timestamps if t > 0.7) / max(1, len(timestamps))*100:.0f}%",
            "monitoring_efficiency": "high" if should_stop_count > 0 else "normal",
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    system = RealTimeQualityMonitoringSystem()

    print("=== Real-Time Quality Monitoring System ===\n")

    # Simulate generation with 20 steps
    for step in range(20):
        timestep = step / 20
        # Simulate quality improving then plateauing
        base_quality = min(0.9, 0.3 + timestep * 0.7 - (max(0, timestep - 0.7) ** 2) * 0.5)
        noise = torch.randn(1, 4096) * 0.1
        image_state = torch.ones(1, 4096) * base_quality + noise

        result = system.monitor_generation_step(image_state, timestep, step)

        print(f"[Step {step:2d}] Quality: {result['current_quality']}, "
              f"Trend: {result['quality_trend']}, "
              f"Stop: {result['early_stop_recommended']}")

    print("\nMonitoring Report:")
    report = system.get_real_time_report()
    for key, value in report.items():
        print(f"  {key}: {value}")

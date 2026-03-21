"""
Checkpoint management, model comparison, and model merging utilities.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class CheckpointManager:
    """Manage model checkpoints with metadata and versioning."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load checkpoint metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {"checkpoints": {}, "best_checkpoint": None, "latest_checkpoint": None}

    def _save_metadata(self):
        """Save checkpoint metadata."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        loss: float,
        config: Any,
        ema_model: Optional[nn.Module] = None,
        is_best: bool = False,
        additional_info: Optional[Dict] = None,
    ) -> str:
        """Save a checkpoint with metadata."""
        timestamp = datetime.now().isoformat()
        checkpoint_name = f"checkpoint_step_{step:06d}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Prepare checkpoint data
        checkpoint_data = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "loss": loss,
            "config": config,
            "timestamp": timestamp,
        }

        if ema_model is not None:
            checkpoint_data["ema"] = ema_model.state_dict()

        if additional_info:
            checkpoint_data.update(additional_info)

        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)

        # Calculate file hash for integrity
        file_hash = self._calculate_file_hash(checkpoint_path)

        # Update metadata
        self.metadata["checkpoints"][checkpoint_name] = {
            "step": step,
            "loss": loss,
            "timestamp": timestamp,
            "file_size_mb": checkpoint_path.stat().st_size / (1024 * 1024),
            "file_hash": file_hash,
            "is_best": is_best,
            "additional_info": additional_info or {},
        }

        # Update best and latest
        if is_best or self.metadata["best_checkpoint"] is None:
            self.metadata["best_checkpoint"] = checkpoint_name

        self.metadata["latest_checkpoint"] = checkpoint_name

        self._save_metadata()

        # Create symlinks for easy access
        self._create_symlinks()

        return str(checkpoint_path)

    def load_checkpoint(
        self, checkpoint_name: Optional[str] = None, load_best: bool = False, load_latest: bool = False
    ) -> Dict[str, Any]:
        """Load a checkpoint."""
        if load_best:
            checkpoint_name = self.metadata["best_checkpoint"]
        elif load_latest:
            checkpoint_name = self.metadata["latest_checkpoint"]

        if not checkpoint_name:
            raise ValueError("No checkpoint specified or available")

        checkpoint_path = self.checkpoint_dir / checkpoint_name

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Verify integrity
        expected_hash = self.metadata["checkpoints"].get(checkpoint_name, {}).get("file_hash")
        if expected_hash:
            actual_hash = self._calculate_file_hash(checkpoint_path)
            if actual_hash != expected_hash:
                raise ValueError(f"Checkpoint integrity check failed: {checkpoint_path}")

        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    def list_checkpoints(self, sort_by: str = "step") -> List[Dict[str, Any]]:
        """List all checkpoints with metadata."""
        checkpoints = []

        for name, metadata in self.metadata["checkpoints"].items():
            checkpoint_info = {"name": name, "path": str(self.checkpoint_dir / name), **metadata}
            checkpoints.append(checkpoint_info)

        # Sort checkpoints
        if sort_by == "step":
            checkpoints.sort(key=lambda x: x["step"])
        elif sort_by == "loss":
            checkpoints.sort(key=lambda x: x["loss"])
        elif sort_by == "timestamp":
            checkpoints.sort(key=lambda x: x["timestamp"])

        return checkpoints

    def cleanup_old_checkpoints(self, keep_best: int = 3, keep_recent: int = 5):
        """Clean up old checkpoints, keeping best and recent ones."""
        checkpoints = self.list_checkpoints(sort_by="step")

        # Identify checkpoints to keep
        keep_names = set()

        # Keep best checkpoints
        best_checkpoints = sorted(checkpoints, key=lambda x: x["loss"])[:keep_best]
        keep_names.update(cp["name"] for cp in best_checkpoints)

        # Keep recent checkpoints
        recent_checkpoints = checkpoints[-keep_recent:]
        keep_names.update(cp["name"] for cp in recent_checkpoints)

        # Always keep the best and latest
        if self.metadata["best_checkpoint"]:
            keep_names.add(self.metadata["best_checkpoint"])
        if self.metadata["latest_checkpoint"]:
            keep_names.add(self.metadata["latest_checkpoint"])

        # Remove old checkpoints
        removed_count = 0
        for checkpoint in checkpoints:
            if checkpoint["name"] not in keep_names:
                checkpoint_path = Path(checkpoint["path"])
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    removed_count += 1

                # Remove from metadata
                del self.metadata["checkpoints"][checkpoint["name"]]

        self._save_metadata()
        print(f"Cleaned up {removed_count} old checkpoints")

    def compare_checkpoints(self, checkpoint1: str, checkpoint2: str) -> Dict[str, Any]:
        """Compare two checkpoints."""
        cp1_meta = self.metadata["checkpoints"].get(checkpoint1, {})
        cp2_meta = self.metadata["checkpoints"].get(checkpoint2, {})

        comparison = {
            "checkpoint1": checkpoint1,
            "checkpoint2": checkpoint2,
            "step_diff": cp2_meta.get("step", 0) - cp1_meta.get("step", 0),
            "loss_diff": cp2_meta.get("loss", float("inf")) - cp1_meta.get("loss", float("inf")),
            "time_diff": cp2_meta.get("timestamp", "") > cp1_meta.get("timestamp", ""),
            "size_diff_mb": cp2_meta.get("file_size_mb", 0) - cp1_meta.get("file_size_mb", 0),
        }

        return comparison

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _create_symlinks(self):
        """Create symlinks for best and latest checkpoints."""
        try:
            # Best checkpoint symlink
            if self.metadata["best_checkpoint"]:
                best_link = self.checkpoint_dir / "best_checkpoint.pt"
                best_target = self.checkpoint_dir / self.metadata["best_checkpoint"]

                if best_link.exists() or best_link.is_symlink():
                    best_link.unlink()
                best_link.symlink_to(best_target.name)

            # Latest checkpoint symlink
            if self.metadata["latest_checkpoint"]:
                latest_link = self.checkpoint_dir / "latest_checkpoint.pt"
                latest_target = self.checkpoint_dir / self.metadata["latest_checkpoint"]

                if latest_link.exists() or latest_link.is_symlink():
                    latest_link.unlink()
                latest_link.symlink_to(latest_target.name)

        except OSError:
            # Symlinks might not be supported on all systems
            pass


def merge_checkpoints(
    checkpoint_paths: List[str],
    output_path: str,
    weights: Optional[List[float]] = None,
    merge_method: str = "weighted_average",
) -> str:
    """Merge multiple checkpoints using different strategies."""
    if not checkpoint_paths:
        raise ValueError("No checkpoint paths provided")

    if weights is None:
        weights = [1.0 / len(checkpoint_paths)] * len(checkpoint_paths)

    if len(weights) != len(checkpoint_paths):
        raise ValueError("Number of weights must match number of checkpoints")

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Load first checkpoint as base
    base_checkpoint = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=False)
    merged_state_dict = {}

    # Initialize merged state dict
    for key in base_checkpoint["model"].keys():
        merged_state_dict[key] = torch.zeros_like(base_checkpoint["model"][key])

    # Merge model parameters
    for i, (checkpoint_path, weight) in enumerate(zip(checkpoint_paths, weights)):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        for key in merged_state_dict.keys():
            if key in checkpoint["model"]:
                if merge_method == "weighted_average":
                    merged_state_dict[key] += weight * checkpoint["model"][key]
                elif merge_method == "max":
                    if i == 0:
                        merged_state_dict[key] = checkpoint["model"][key]
                    else:
                        merged_state_dict[key] = torch.max(merged_state_dict[key], checkpoint["model"][key])
                elif merge_method == "min":
                    if i == 0:
                        merged_state_dict[key] = checkpoint["model"][key]
                    else:
                        merged_state_dict[key] = torch.min(merged_state_dict[key], checkpoint["model"][key])

    # Create merged checkpoint
    merged_checkpoint = {
        "model": merged_state_dict,
        "config": base_checkpoint["config"],
        "merge_info": {
            "source_checkpoints": checkpoint_paths,
            "weights": weights,
            "merge_method": merge_method,
            "merge_timestamp": datetime.now().isoformat(),
        },
    }

    # Copy EMA if available in base checkpoint
    if "ema" in base_checkpoint:
        merged_ema = {}
        for key in base_checkpoint["ema"].keys():
            merged_ema[key] = torch.zeros_like(base_checkpoint["ema"][key])

        for i, (checkpoint_path, weight) in enumerate(zip(checkpoint_paths, weights)):
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if "ema" in checkpoint:
                for key in merged_ema.keys():
                    if key in checkpoint["ema"]:
                        merged_ema[key] += weight * checkpoint["ema"][key]

        merged_checkpoint["ema"] = merged_ema

    # Save merged checkpoint
    torch.save(merged_checkpoint, output_path)
    print(f"Merged checkpoint saved to {output_path}")

    return output_path


def analyze_checkpoint_differences(checkpoint1_path: str, checkpoint2_path: str) -> Dict[str, Any]:
    """Analyze differences between two checkpoints."""
    cp1 = torch.load(checkpoint1_path, map_location="cpu", weights_only=False)
    cp2 = torch.load(checkpoint2_path, map_location="cpu", weights_only=False)

    analysis = {"parameter_differences": {}, "statistics": {}, "config_differences": {}}

    # Compare model parameters
    model1 = cp1["model"]
    model2 = cp2["model"]

    total_diff = 0.0
    total_params = 0
    max_diff = 0.0
    max_diff_param = ""

    for key in model1.keys():
        if key in model2:
            diff = torch.abs(model1[key] - model2[key])
            param_diff = torch.mean(diff).item()
            param_max_diff = torch.max(diff).item()

            analysis["parameter_differences"][key] = {
                "mean_diff": param_diff,
                "max_diff": param_max_diff,
                "std_diff": torch.std(diff).item(),
                "shape": list(model1[key].shape),
            }

            total_diff += param_diff * model1[key].numel()
            total_params += model1[key].numel()

            if param_max_diff > max_diff:
                max_diff = param_max_diff
                max_diff_param = key

    # Overall statistics
    analysis["statistics"] = {
        "total_parameters": total_params,
        "average_difference": total_diff / total_params if total_params > 0 else 0.0,
        "max_difference": max_diff,
        "max_difference_parameter": max_diff_param,
        "step_difference": cp2.get("step", 0) - cp1.get("step", 0),
        "loss_difference": cp2.get("loss", 0) - cp1.get("loss", 0),
    }

    # Compare configs if available
    if "config" in cp1 and "config" in cp2:
        config1 = cp1["config"]
        config2 = cp2["config"]

        # Simple config comparison (assumes configs are dicts or have __dict__)
        try:
            if hasattr(config1, "__dict__"):
                config1_dict = config1.__dict__
            else:
                config1_dict = config1

            if hasattr(config2, "__dict__"):
                config2_dict = config2.__dict__
            else:
                config2_dict = config2

            for key in set(config1_dict.keys()) | set(config2_dict.keys()):
                val1 = config1_dict.get(key, "MISSING")
                val2 = config2_dict.get(key, "MISSING")

                if val1 != val2:
                    analysis["config_differences"][key] = {"checkpoint1": val1, "checkpoint2": val2}
        except Exception:
            analysis["config_differences"] = {"error": "Could not compare configs"}

    return analysis


def extract_lora_from_checkpoint(checkpoint_path: str, output_path: str, rank: int = 16, alpha: float = 16.0) -> str:
    """Extract LoRA weights from a fine-tuned checkpoint (experimental)."""
    # This is a simplified LoRA extraction - real implementation would be more complex
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # For now, just save the checkpoint in LoRA format with metadata
    lora_checkpoint = {
        "lora_weights": checkpoint["model"],  # Simplified - real LoRA would decompose weights
        "rank": rank,
        "alpha": alpha,
        "base_checkpoint": checkpoint_path,
        "extraction_timestamp": datetime.now().isoformat(),
    }

    torch.save(lora_checkpoint, output_path)
    print(f"LoRA weights extracted to {output_path}")

    return output_path

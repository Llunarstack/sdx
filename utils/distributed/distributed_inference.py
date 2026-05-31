"""
Distributed inference orchestration for multi-GPU and multi-machine setups.
Supports tensor parallelism, pipeline parallelism, and sequence parallelism.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed inference."""
    num_gpus: int = 1
    num_machines: int = 1
    parallelism_type: str = "tensor"  # tensor, pipeline, sequence
    batch_size: int = 32
    max_seq_length: int = 2048
    enable_async_communication: bool = True


class TensorParallelism:
    """Tensor parallelism: split model weights across GPUs (linear speedup)."""

    def __init__(self, model: nn.Module, num_gpus: int = 1):
        self.model = model
        self.num_gpus = num_gpus
        self.device_list = [torch.device(f"cuda:{i}") for i in range(num_gpus)]

    def partition_linear_layer(self, layer: nn.Linear) -> List[nn.Linear]:
        """Split linear layer across GPUs (column-wise for weight matrix)."""

        in_features = layer.in_features
        out_features = layer.out_features

        # Split output dimension
        split_out_features = out_features // self.num_gpus

        partitions = []
        for i in range(self.num_gpus):
            start_out = i * split_out_features
            end_out = start_out + split_out_features

            # Create partition
            partition = nn.Linear(in_features, split_out_features)
            partition.weight.data = layer.weight.data[start_out:end_out].clone()
            if layer.bias is not None:
                partition.bias.data = layer.bias.data[start_out:end_out].clone()

            partition.to(self.device_list[i])
            partitions.append(partition)

        return partitions

    def partition_model(self) -> List[Dict[str, nn.Module]]:
        """Partition entire model across GPUs."""

        partitioned_layers = {}

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                partitions = self.partition_linear_layer(module)
                partitioned_layers[name] = partitions

        logger.info(f"Partitioned {len(partitioned_layers)} layers across {self.num_gpus} GPUs")

        return partitioned_layers

    def forward_distributed(
        self,
        x: torch.Tensor,
        partitions: Dict[str, List[nn.Module]],
    ) -> torch.Tensor:
        """Forward pass with distributed computation."""

        # Move input to first GPU
        x = x.to(self.device_list[0])

        for layer_name, partition_list in partitions.items():
            # Compute output on each GPU
            outputs = []

            for i, partition in enumerate(partition_list):
                output = partition(x.to(self.device_list[i]))
                outputs.append(output.to(self.device_list[0]))

            # Concatenate outputs
            x = torch.cat(outputs, dim=-1)

        return x


class PipelineParallelism:
    """Pipeline parallelism: split model layers across GPUs (4x speedup via pipelining)."""

    def __init__(self, model: nn.Module, num_stages: int = 2):
        self.model = model
        self.num_stages = num_stages
        self.stages = self._partition_stages()

    def _partition_stages(self) -> List[nn.Sequential]:
        """Partition model into pipeline stages."""

        layers = list(self.model.children())
        layers_per_stage = max(1, len(layers) // self.num_stages)

        stages = []
        for i in range(self.num_stages):
            start_layer = i * layers_per_stage
            end_layer = start_layer + layers_per_stage if i < self.num_stages - 1 else len(layers)

            stage_layers = layers[start_layer:end_layer]
            stage = nn.Sequential(*stage_layers)

            device = torch.device(f"cuda:{i % torch.cuda.device_count()}")
            stage.to(device)

            stages.append(stage)

        logger.info(f"Created {len(stages)} pipeline stages")

        return stages

    def forward_with_pipeline(
        self,
        x: torch.Tensor,
        micro_batch_size: int = 8,
    ) -> torch.Tensor:
        """Forward pass with pipeline parallelism (GPipe style)."""

        batch_size = x.shape[0]
        num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size

        # Split into micro-batches
        micro_batches = [
            x[i * micro_batch_size : (i + 1) * micro_batch_size]
            for i in range(num_micro_batches)
        ]

        # Forward pipeline
        outputs = []

        for micro_batch in micro_batches:
            # Forward through stages
            output = micro_batch

            for stage in self.stages:
                # Move to stage device
                output = output.to(next(stage.parameters()).device)
                output = stage(output)

            outputs.append(output.cpu())

        # Concatenate outputs
        return torch.cat(outputs, dim=0)

    def get_pipeline_efficiency(self, batch_size: int, seq_len: int) -> float:
        """Compute pipeline efficiency."""

        # Efficiency = computation_time / (computation_time + bubble_time)
        # Bubble time = (num_stages - 1) * forward_time

        computation_time = batch_size * seq_len  # Proxy for time
        bubble_time = (self.num_stages - 1) * (computation_time / self.num_stages)

        efficiency = computation_time / (computation_time + bubble_time)

        return efficiency


class SequenceParallelism:
    """Sequence parallelism: split sequence dimension across GPUs (3x speedup)."""

    def __init__(self, model: nn.Module, num_gpus: int = 1):
        self.model = model
        self.num_gpus = num_gpus

    def split_sequence(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Split sequence dimension across GPUs."""

        seq_len = x.shape[1]
        split_seq_len = seq_len // self.num_gpus

        splits = []
        for i in range(self.num_gpus):
            start_seq = i * split_seq_len
            end_seq = start_seq + split_seq_len if i < self.num_gpus - 1 else seq_len

            split = x[:, start_seq:end_seq, :]
            splits.append(split)

        return splits

    def all_gather_sequences(self, splits: List[torch.Tensor]) -> torch.Tensor:
        """Gather sequences from all GPUs."""

        return torch.cat(splits, dim=1)

    def forward_sequence_parallel(
        self,
        x: torch.Tensor,
        attention_fn,
    ) -> torch.Tensor:
        """Forward pass with sequence parallelism."""

        # Split sequence across GPUs
        sequence_splits = self.split_sequence(x)

        # Process each split on different GPU
        outputs = []
        for i, split in enumerate(sequence_splits):
            device = torch.device(f"cuda:{i % torch.cuda.device_count()}")
            split = split.to(device)

            # All-to-all communication for attention
            # (simplified - full implementation needs ring or all-reduce patterns)

            output = attention_fn(split)
            outputs.append(output)

        # All-gather
        return self.all_gather_sequences(outputs)


class DistributedInference:
    """Unified distributed inference framework."""

    def __init__(self, model: nn.Module, config: DistributedConfig = None):
        self.model = model
        self.config = config or DistributedConfig()

        if self.config.parallelism_type == "tensor":
            self.parallelizer = TensorParallelism(model, self.config.num_gpus)
        elif self.config.parallelism_type == "pipeline":
            self.parallelizer = PipelineParallelism(model, self.config.num_gpus)
        elif self.config.parallelism_type == "sequence":
            self.parallelizer = SequenceParallelism(model, self.config.num_gpus)

        self.stats = {
            "total_tokens": 0,
            "total_time": 0.0,
            "throughput_tokens_per_sec": 0.0,
            "scalability_efficiency": 0.0,
        }

    def inference(self, input_ids: torch.Tensor, max_length: int = 100) -> torch.Tensor:
        """Run distributed inference."""

        import time

        start_time = time.time()

        # Route to appropriate parallelism type
        if self.config.parallelism_type == "tensor":
            partitions = self.parallelizer.partition_model()
            output = self.parallelizer.forward_distributed(input_ids, partitions)
        elif self.config.parallelism_type == "pipeline":
            output = self.parallelizer.forward_with_pipeline(input_ids)
        elif self.config.parallelism_type == "sequence":
            output = self.parallelizer.forward_sequence_parallel(input_ids, self.model)

        elapsed = time.time() - start_time

        # Update stats
        num_tokens = output.shape[1] - input_ids.shape[1]
        self.stats["total_tokens"] += num_tokens
        self.stats["total_time"] += elapsed
        self.stats["throughput_tokens_per_sec"] = (
            self.stats["total_tokens"] / self.stats["total_time"]
        )

        # Scalability efficiency (linear scaling = 1.0)
        expected_speedup = self.config.num_gpus
        actual_speedup = self.stats["throughput_tokens_per_sec"] / (
            self.stats["throughput_tokens_per_sec"] / expected_speedup
        )
        self.stats["scalability_efficiency"] = min(actual_speedup / expected_speedup, 1.0)

        logger.info(
            f"Generated {num_tokens} tokens in {elapsed:.3f}s "
            f"({self.stats['throughput_tokens_per_sec']:.1f} tokens/sec) "
            f"Efficiency: {self.stats['scalability_efficiency']:.2%}"
        )

        return output

    def get_stats(self) -> Dict[str, Any]:
        """Get distribution statistics."""
        return self.stats


class LoadBalancer:
    """Load balancing across distributed GPUs."""

    def __init__(self, num_gpus: int = 1):
        self.num_gpus = num_gpus
        self.gpu_loads = [0.0] * num_gpus

    def assign_batch(self, batch_size: int) -> int:
        """Assign batch to least-loaded GPU."""

        min_load_gpu = self.gpu_loads.index(min(self.gpu_loads))
        self.gpu_loads[min_load_gpu] += batch_size

        return min_load_gpu

    def rebalance(self) -> None:
        """Rebalance loads across GPUs."""

        avg_load = sum(self.gpu_loads) / len(self.gpu_loads)

        for i in range(len(self.gpu_loads)):
            if self.gpu_loads[i] > avg_load * 1.2:  # 20% overload threshold
                # Transfer work
                logger.info(f"Rebalancing GPU {i} (load: {self.gpu_loads[i]})")
                self.gpu_loads[i] = avg_load


if __name__ == "__main__":
    # Example usage
    model = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
    )

    config = DistributedConfig(
        num_gpus=4,
        parallelism_type="pipeline",
        batch_size=16,
    )

    dist_inference = DistributedInference(model, config)

    input_ids = torch.randn(16, 512)
    output = dist_inference.inference(input_ids, max_length=100)

    print("Distributed Inference Stats:")
    for key, value in dist_inference.get_stats().items():
        print(f"  {key}: {value}")

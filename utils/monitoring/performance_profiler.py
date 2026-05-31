"""
Real-time performance monitoring and profiling with automatic kernel recommendations.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class OperationMetrics:
    """Metrics for a single operation."""
    name: str
    operation_type: str
    num_calls: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    total_flops: float = 0.0
    throughput_gflops: float = 0.0


class OperationProfiler:
    """Profile individual operations and identify bottlenecks."""

    def __init__(self):
        self.metrics: Dict[str, OperationMetrics] = defaultdict(
            lambda: OperationMetrics(name="", operation_type="")
        )
        self.operation_hooks = {}

    def profile_module(self, module: nn.Module, module_name: str = "") -> None:
        """Register hooks to profile all operations in a module."""

        for name, submodule in module.named_modules():
            full_name = f"{module_name}/{name}" if module_name else name

            if isinstance(submodule, (nn.Linear, nn.Conv2d, nn.MultiheadAttention)):
                self._register_hook(submodule, full_name)

    def _register_hook(self, module: nn.Module, name: str) -> None:
        """Register forward/backward hooks."""

        def forward_hook(module, input, output):
            # Compute FLOPs
            flops = self._estimate_flops(module, input, output)

            # Create metrics entry
            if name not in self.metrics:
                self.metrics[name] = OperationMetrics(
                    name=name,
                    operation_type=module.__class__.__name__,
                )

            metric = self.metrics[name]
            metric.num_calls += 1
            metric.total_flops += flops

            # Time will be updated in backward hook
            return output

        def backward_hook(module, grad_input, grad_output):
            elapsed_ms = time.perf_counter() * 1000  # Rough approximation
            metric = self.metrics[name]

            metric.total_time_ms += elapsed_ms
            metric.avg_time_ms = metric.total_time_ms / metric.num_calls
            metric.min_time_ms = min(metric.min_time_ms, elapsed_ms)
            metric.max_time_ms = max(metric.max_time_ms, elapsed_ms)

            if metric.total_time_ms > 0:
                metric.throughput_gflops = (metric.total_flops / metric.total_time_ms) / 1e6

        module.register_forward_hook(forward_hook)
        module.register_full_backward_hook(backward_hook)

    def _estimate_flops(
        self,
        module: nn.Module,
        input_data: torch.Tensor,
        output: torch.Tensor,
    ) -> float:
        """Estimate FLOPs for operation."""

        if isinstance(module, nn.Linear):
            # FLOPs = 2 * input_features * output_features * batch_size
            batch_size = input_data[0].shape[0]
            return 2 * module.in_features * module.out_features * batch_size

        elif isinstance(module, nn.Conv2d):
            # FLOPs = 2 * kernel_ops * output_size
            batch_size = input_data[0].shape[0]
            output_h, output_w = output.shape[-2:]
            kernel_ops = (
                module.in_channels * module.kernel_size[0] * module.kernel_size[1]
            )
            return 2 * kernel_ops * output_h * output_w * batch_size

        return 0.0

    def get_bottlenecks(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get most time-consuming operations."""

        sorted_ops = sorted(
            self.metrics.items(),
            key=lambda x: x[1].total_time_ms,
            reverse=True,
        )

        return [(name, metrics.total_time_ms) for name, metrics in sorted_ops[:top_k]]

    def report(self) -> str:
        """Generate profiling report."""

        report = "=== Operation Profiling Report ===\n\n"

        # Summary statistics
        total_time = sum(m.total_time_ms for m in self.metrics.values())
        total_flops = sum(m.total_flops for m in self.metrics.values())

        report += f"Total Time: {total_time:.3f}ms\n"
        report += f"Total FLOPs: {total_flops / 1e9:.3f}B\n"
        report += f"Throughput: {(total_flops / total_time) / 1e9:.3f} GFLOP/s\n\n"

        # Per-operation breakdown
        report += "Operation Breakdown:\n"
        report += f"{'Name':<40} {'Type':<15} {'Calls':<8} {'Time':<10} {'FLOPs':<12}\n"
        report += "-" * 85 + "\n"

        for name, metric in sorted(
            self.metrics.items(),
            key=lambda x: x[1].total_time_ms,
            reverse=True,
        ):
            percentage = 100 * metric.total_time_ms / total_time if total_time > 0 else 0
            report += (
                f"{name:<40} {metric.operation_type:<15} {metric.num_calls:<8} "
                f"{metric.total_time_ms:>8.3f}ms ({percentage:>5.1f}%) "
                f"{metric.total_flops/1e9:>10.3f}B\n"
            )

        return report


class KernelRecommender:
    """Recommend native kernels based on profiling data."""

    def __init__(self):
        self.recommendations = {}
        self.thresholds = {
            "quantization_threshold_ms": 0.1,
            "softmax_threshold_ms": 0.5,
            "attention_threshold_ms": 2.0,
            "activation_threshold_ms": 0.2,
        }

    def analyze_metrics(self, metrics: Dict[str, OperationMetrics]) -> Dict[str, str]:
        """Analyze metrics and recommend kernels."""

        recommendations = {}

        for name, metric in metrics.items():
            if metric.total_time_ms < 0.01:
                continue  # Skip negligible operations

            recommendation = self._get_recommendation(name, metric)
            if recommendation:
                recommendations[name] = recommendation

        return recommendations

    def _get_recommendation(self, operation_name: str, metric: OperationMetrics) -> Optional[str]:
        """Get kernel recommendation for operation."""

        if "linear" in operation_name.lower() or metric.operation_type == "Linear":
            if metric.total_time_ms > self.thresholds["quantization_threshold_ms"]:
                return "RUST_QUANTIZATION"
            elif metric.total_flops > 1e8:
                return "CUDA_LINEAR"

        elif "attention" in operation_name.lower():
            if metric.total_time_ms > self.thresholds["attention_threshold_ms"]:
                return "GO_FLASH_ATTENTION_V2"
            elif metric.total_time_ms > self.thresholds["attention_threshold_ms"] * 0.5:
                return "RUST_PARALLEL_ATTENTION"

        elif "conv" in operation_name.lower() or metric.operation_type == "Conv2d":
            if metric.total_time_ms > 1.0:
                return "CUDA_CONVOLUTION"
            else:
                return "RUST_SIMD_CONV"

        elif "relu" in operation_name.lower() or "gelu" in operation_name.lower():
            if metric.total_time_ms > self.thresholds["activation_threshold_ms"]:
                return "RUST_GELU_SIMD"
            else:
                return "JULIA_MULTI_THREADED"

        elif "softmax" in operation_name.lower():
            if metric.total_time_ms > self.thresholds["softmax_threshold_ms"]:
                return "RUST_SOFTMAX_FAST"

        return None

    def generate_recommendation_report(
        self,
        recommendations: Dict[str, str],
        metrics: Dict[str, OperationMetrics],
    ) -> str:
        """Generate detailed recommendation report."""

        report = "=== Kernel Optimization Recommendations ===\n\n"

        # Group by recommendation
        by_kernel = defaultdict(list)
        for operation_name, kernel in recommendations.items():
            by_kernel[kernel].append((operation_name, metrics[operation_name].total_time_ms))

        # Sort by potential speedup
        for kernel, operations in sorted(
            by_kernel.items(),
            key=lambda x: sum(t for _, t in x[1]),
            reverse=True,
        ):
            total_time = sum(time for _, time in operations)
            report += f"\n{kernel} (Potential Speedup: ~3-5x)\n"
            report += f"  Total Time to Optimize: {total_time:.3f}ms\n"
            report += "  Operations:\n"

            for op_name, op_time in sorted(operations, key=lambda x: x[1], reverse=True):
                report += f"    - {op_name}: {op_time:.3f}ms\n"

        return report


class RuntimeMonitor:
    """Monitor runtime metrics during inference."""

    def __init__(self):
        self.metrics = {
            "total_time": 0.0,
            "total_tokens": 0,
            "throughput_tokens_per_sec": 0.0,
            "memory_peak_mb": 0.0,
            "memory_avg_mb": 0.0,
        }
        self.operation_times = defaultdict(list)

    def record_operation(self, operation_name: str, elapsed_ms: float) -> None:
        """Record operation timing."""
        self.operation_times[operation_name].append(elapsed_ms)

    def update_metrics(self, num_tokens: int, elapsed_ms: float) -> None:
        """Update overall metrics."""
        self.metrics["total_time"] += elapsed_ms
        self.metrics["total_tokens"] += num_tokens
        self.metrics["throughput_tokens_per_sec"] = (
            self.metrics["total_tokens"] / (self.metrics["total_time"] / 1000.0)
        )

    def get_memory_usage(self) -> Tuple[float, float]:
        """Get peak and average GPU memory usage."""
        try:
            peak = torch.cuda.max_memory_allocated() / 1e6
            current = torch.cuda.memory_allocated() / 1e6
            return peak, current
        except RuntimeError:
            return 0.0, 0.0

    def report(self) -> str:
        """Generate monitoring report."""

        report = "=== Runtime Monitoring Report ===\n\n"

        peak_mem, current_mem = self.get_memory_usage()

        report += f"Total Time: {self.metrics['total_time']:.3f}ms\n"
        report += f"Total Tokens: {self.metrics['total_tokens']}\n"
        report += f"Throughput: {self.metrics['throughput_tokens_per_sec']:.1f} tokens/sec\n"
        report += f"Memory (Peak/Current): {peak_mem:.1f}MB / {current_mem:.1f}MB\n\n"

        report += "Top Operations:\n"
        for op_name, times in sorted(
            self.operation_times.items(),
            key=lambda x: np.mean(x[1]),
            reverse=True,
        )[:10]:
            avg_time = np.mean(times)
            report += f"  {op_name}: {avg_time:.3f}ms avg ({np.mean(times) * len(times):.1f}ms total)\n"

        return report


class PerformanceOptimizer:
    """Unified performance optimization framework."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.profiler = OperationProfiler()
        self.recommender = KernelRecommender()
        self.monitor = RuntimeMonitor()

        self.profiler.profile_module(model)

    def analyze(self) -> Dict[str, any]:
        """Analyze model performance and generate recommendations."""

        report = self.profiler.report()
        bottlenecks = self.profiler.get_bottlenecks(top_k=5)
        recommendations = self.recommender.analyze_metrics(self.profiler.metrics)
        rec_report = self.recommender.generate_recommendation_report(
            recommendations,
            self.profiler.metrics,
        )

        return {
            "profiling_report": report,
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "recommendation_report": rec_report,
        }

    def get_runtime_stats(self) -> str:
        """Get runtime statistics."""
        return self.monitor.report()


if __name__ == "__main__":
    # Example usage
    model = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
    )

    optimizer = PerformanceOptimizer(model)

    # Run inference
    x = torch.randn(32, 512)
    output = model(x)

    # Analyze
    analysis = optimizer.analyze()
    print(analysis["profiling_report"])
    print(analysis["recommendation_report"])

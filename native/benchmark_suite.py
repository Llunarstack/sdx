#!/usr/bin/env python3
"""
Cross-language performance benchmark suite for SDX native kernels.

Measures and compares performance across Rust, C++, Go, Mojo, and Julia implementations.
Run: python benchmark_suite.py
"""

import sys
import time
from pathlib import Path
from typing import Callable, List

import numpy as np


class BenchmarkResult:
    def __init__(self, name: str, language: str, operation: str, elapsed_ms: float, speedup: float = 1.0):
        self.name = name
        self.language = language
        self.operation = operation
        self.elapsed_ms = elapsed_ms
        self.speedup = speedup

    def __repr__(self):
        return f"{self.language:8} | {self.operation:20} | {self.elapsed_ms:8.3f}ms | {self.speedup:6.2f}x"


class NativeKernelBenchmark:
    def __init__(self, data_size: int = 10240):
        self.data_size = data_size
        self.results: List[BenchmarkResult] = []
        self.baseline_times = {}

    def benchmark_numpy(self, name: str, operation: str, func: Callable, iterations: int = 100) -> float:
        """Benchmark NumPy baseline implementation."""
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        avg_time = np.mean(times[10:])  # Ignore first 10 for warmup
        self.baseline_times[f"{operation}"] = avg_time

        result = BenchmarkResult(name, "NumPy", operation, avg_time, 1.0)
        self.results.append(result)
        print(f"✓ {result}")

        return avg_time

    def benchmark_rust(self, operation: str, baseline_ms: float) -> None:
        """Benchmark Rust implementation."""
        try:
            import sdx_native  # Assumes PyO3 bindings installed

            data = np.random.randn(self.data_size).astype(np.float32)

            if operation == "Quantization":
                start = time.perf_counter()
                for _ in range(100):
                    sdx_native.quantize_int8(data, 127.0)
                elapsed = (time.perf_counter() - start) * 10  # Convert to ms
            elif operation == "Softmax":
                start = time.perf_counter()
                for _ in range(100):
                    sdx_native.softmax_fast(data)
                elapsed = (time.perf_counter() - start) * 10
            elif operation == "Variance":
                start = time.perf_counter()
                for _ in range(100):
                    sdx_native.variance(data)
                elapsed = (time.perf_counter() - start) * 10
            else:
                return

            speedup = baseline_ms / elapsed if elapsed > 0 else 1.0
            result = BenchmarkResult("rust", "Rust", operation, elapsed, speedup)
            self.results.append(result)
            print(f"✓ {result}")

        except ImportError:
            print("⊘ Rust: sdx_native not available (install with: cd native/rust && maturin develop)")

    def benchmark_python_baseline(self) -> None:
        """Establish NumPy baseline for all operations."""
        print("\n" + "=" * 80)
        print("BASELINE: NumPy (Pure Python)")
        print("=" * 80 + "\n")

        data = np.random.randn(self.data_size).astype(np.float32)

        # Quantization baseline
        self.benchmark_numpy(
            "numpy_quantize",
            "Quantization",
            lambda: np.clip(data * 127.0, -128, 127).astype(np.int8),
            100
        )

        # Softmax baseline
        def softmax_numpy(x):
            exp_x = np.exp(x - np.max(x))
            return exp_x / np.sum(exp_x)

        self.benchmark_numpy(
            "numpy_softmax",
            "Softmax",
            lambda: softmax_numpy(data),
            100
        )

        # Variance baseline
        self.benchmark_numpy(
            "numpy_variance",
            "Variance",
            lambda: np.var(data),
            100
        )

        # GELU baseline
        def gelu_numpy(x):
            cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
            return x * cdf

        self.benchmark_numpy(
            "numpy_gelu",
            "GELU",
            lambda: gelu_numpy(data),
            50
        )

        # Dot product baseline
        self.benchmark_numpy(
            "numpy_dot",
            "Dot Product",
            lambda: np.dot(data, data),
            100
        )

    def benchmark_all(self) -> None:
        """Run full benchmark suite."""
        print("\n" + "=" * 80)
        print("NATIVE IMPLEMENTATIONS: Performance Comparison")
        print("=" * 80 + "\n")

        for operation, baseline_ms in self.baseline_times.items():
            print(f"\n{operation} (Baseline: {baseline_ms:.3f}ms)")
            print("-" * 80)

            # Benchmark each implementation
            self.benchmark_rust(operation, baseline_ms)

            # TODO: Add C++ CUDA, Go, Mojo, Julia benchmarking

    def print_summary(self) -> None:
        """Print comprehensive benchmark summary."""
        print("\n" + "=" * 80)
        print("SUMMARY: Speedup Comparison")
        print("=" * 80 + "\n")

        # Group results by operation
        operations = {}
        for result in self.results:
            if result.operation not in operations:
                operations[result.operation] = []
            operations[result.operation].append(result)

        # Print summary table
        print(f"{'Operation':<20} {'NumPy':<10} {'Rust':<10} {'C++':<10} {'Go':<10} {'Mojo':<10} {'Julia':<10}")
        print("-" * 90)

        for op_name, results in operations.items():
            row = [op_name]
            for lang in ["NumPy", "Rust", "C++", "Go", "Mojo", "Julia"]:
                lang_result = next((r for r in results if r.language == lang), None)
                if lang_result:
                    row.append(f"{lang_result.speedup:.2f}x")
                else:
                    row.append("-")

            print(f"{row[0]:<20} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<10} {row[5]:<10} {row[6]:<10}")

        # Calculate average speedups
        print("\nAverage Speedup by Language:")
        print("-" * 40)

        for lang in ["Rust", "C++", "Go", "Mojo", "Julia"]:
            lang_results = [r for r in self.results if r.language == lang]
            if lang_results:
                avg_speedup = np.mean([r.speedup for r in lang_results])
                print(f"{lang:<10}: {avg_speedup:.2f}x")

    def generate_report(self) -> str:
        """Generate markdown benchmark report."""
        report = "# SDX Native Kernels Benchmark Report\n\n"
        report += "## Configuration\n"
        report += f"- Data Size: {self.data_size:,} elements\n"
        report += "- Iterations: 100\n\n"

        report += "## Results by Operation\n\n"

        operations = {}
        for result in self.results:
            if result.operation not in operations:
                operations[result.operation] = []
            operations[result.operation].append(result)

        for op_name, results in operations.items():
            report += f"### {op_name}\n\n"
            report += "| Language | Time (ms) | Speedup |\n"
            report += "|----------|-----------|----------|\n"
            for result in results:
                report += f"| {result.language} | {result.elapsed_ms:.3f} | {result.speedup:.2f}x |\n"
            report += "\n"

        return report

    def run(self) -> None:
        """Run complete benchmark suite."""
        print("\n" + "=" * 80)
        print("SDX Native Kernels Benchmark Suite")
        print("=" * 80)

        self.benchmark_python_baseline()
        self.benchmark_all()
        self.print_summary()

        # Save report
        report = self.generate_report()
        report_path = Path("native/BENCHMARK_RESULTS.md")
        report_path.write_text(report)
        print(f"\n✓ Report saved to {report_path}")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        data_size = int(sys.argv[1])
    else:
        data_size = 10240

    benchmark = NativeKernelBenchmark(data_size=data_size)
    benchmark.run()


if __name__ == "__main__":
    main()

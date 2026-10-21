#!/usr/bin/env python3
"""Time a few vector kernels in NumPy and in the Rust ``sdx_native`` PyO3 extension (``native/rust``).

Run: ``python native/benchmark_suite.py [data_size]``; the markdown report goes to
``native/BENCHMARK_RESULTS.md``. Without the built extension only the NumPy baseline is timed.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPORT_PATH = Path(__file__).with_name("BENCHMARK_RESULTS.md")
ITERATIONS = 100
WARMUP = 10


@dataclass
class BenchmarkResult:
    language: str
    operation: str
    elapsed_ms: float
    speedup: float = 1.0

    def __str__(self) -> str:
        return f"{self.language:8} | {self.operation:20} | {self.elapsed_ms:8.3f}ms | {self.speedup:6.2f}x"


def time_ms(func: Callable[[], object], iterations: int = ITERATIONS, warmup: int = WARMUP) -> float:
    """Mean wall time per call in milliseconds, after ``warmup`` untimed calls."""
    for _ in range(warmup):
        func()
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    return (time.perf_counter() - start) * 1000.0 / iterations


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def numpy_ops(data: np.ndarray) -> dict[str, Callable[[], object]]:
    return {
        "Quantization": lambda: np.clip(data * 127.0, -128, 127).astype(np.int8),
        "Softmax": lambda: _softmax(data),
        "Variance": lambda: np.var(data),
        "GELU": lambda: _gelu(data),
        "Dot Product": lambda: np.dot(data, data),
    }


def rust_ops(data: np.ndarray) -> dict[str, Callable[[], object]]:
    """Rust counterparts of ``numpy_ops``, or ``{}`` when the extension is not built.

    The pure-Python helper package ``native/_experimental/python/sdx_native`` shares the import name, so the
    extension is recognised by its ``py_*`` functions.
    """
    try:
        import sdx_native
    except ImportError:
        return {}
    if not hasattr(sdx_native, "py_variance"):
        return {}
    return {
        "Quantization": lambda: sdx_native.py_quantize_int8(data, 127.0),
        "Softmax": lambda: sdx_native.py_softmax_fast(data),
        "Variance": lambda: sdx_native.py_variance(data),
        "GELU": lambda: sdx_native.py_gelu_batch(data),
        "Dot Product": lambda: sdx_native.py_dot_product(data, data),
    }


def run_benchmarks(data_size: int) -> list[BenchmarkResult]:
    data = np.random.randn(data_size).astype(np.float32)
    rust = rust_ops(data)
    if not rust:
        print("skip Rust: sdx_native extension not built (cd native/rust && maturin develop)")
    results: list[BenchmarkResult] = []
    for operation, func in numpy_ops(data).items():
        baseline = time_ms(func)
        results.append(BenchmarkResult("NumPy", operation, baseline))
        print(results[-1])
        if operation in rust:
            elapsed = time_ms(rust[operation])
            results.append(BenchmarkResult("Rust", operation, elapsed, baseline / elapsed if elapsed > 0 else 1.0))
            print(results[-1])
    return results


def render_report(results: list[BenchmarkResult], data_size: int) -> str:
    lines = [
        "# SDX Native Kernels Benchmark Report",
        "",
        "## Configuration",
        f"- Data Size: {data_size:,} elements",
        f"- Iterations: {ITERATIONS} (after {WARMUP} warm-up calls)",
        "",
        "## Results by Operation",
        "",
    ]
    for operation in dict.fromkeys(r.operation for r in results):
        lines += [f"### {operation}", "", "| Language | Time (ms) | Speedup |", "|----------|-----------|---------|"]
        lines += [
            f"| {r.language} | {r.elapsed_ms:.3f} | {r.speedup:.2f}x |" for r in results if r.operation == operation
        ]
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    data_size = int(args[0]) if args else 10240
    results = run_benchmarks(data_size)
    rust_speedups = [r.speedup for r in results if r.language == "Rust"]
    if rust_speedups:
        print(f"\nAverage Rust speedup: {np.mean(rust_speedups):.2f}x")
    REPORT_PATH.write_text(render_report(results, data_size), encoding="utf-8")
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()

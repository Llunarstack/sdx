// Standalone benchmarking and testing CLI for native implementations

use sdx_native::*;
use std::time::Instant;

fn benchmark<F>(name: &str, mut f: F, iterations: usize)
where
    F: FnMut(),
{
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed.as_secs_f64() / iterations as f64;
    println!("{}: {:.4}ms per iteration ({:.2}ms total)",
             name, per_iter * 1000.0, elapsed.as_secs_f64() * 1000.0);
}

fn main() {
    println!("=== SDX Native Performance Benchmarks ===\n");

    // Test data
    let data_1k: Vec<f32> = (0..1024).map(|i| (i as f32) / 1024.0).collect();
    let data_10k: Vec<f32> = (0..10240).map(|i| (i as f32) / 10240.0).collect();

    // Quantization
    println!("Quantization (1K elements):");
    benchmark("INT8 Quantization", || {
        let _ = quantize_int8_simd(&data_1k, 127.0);
    }, 100);

    println!("\nActivation Functions (1K elements):");
    benchmark("ReLU", || {
        let mut data = data_1k.clone();
        relu_fast(&mut data);
    }, 50);

    benchmark("GELU Batch", || {
        let _ = gelu_fast_batch(&data_1k);
    }, 50);

    println!("\nSoftmax (1K elements):");
    benchmark("Softmax Fast", || {
        let _ = softmax_fast(&data_1k);
    }, 50);

    println!("\nLayers (1K elements):");
    benchmark("Layer Norm", || {
        let gamma = vec![1.0; 1024];
        let beta = vec![0.0; 1024];
        let _ = layer_norm_fast(&data_1k, 1e-5, &gamma, &beta);
    }, 30);

    println!("\nVectors (10K elements):");
    benchmark("Dot Product", || {
        let _ = dot_product_parallel(&data_10k, &data_10k);
    }, 100);

    benchmark("Variance", || {
        let _ = variance_fast(&data_10k);
    }, 100);

    benchmark("Histogram", || {
        let _ = histogram_parallel(&data_10k, 256);
    }, 50);

    println!("\n=== Performance Summary ===");
    println!("Expected speedups vs Python:");
    println!("  Quantization: 4-5x");
    println!("  ReLU: 6x");
    println!("  GELU: 10x");
    println!("  Softmax: 5x");
    println!("  Layer Norm: 3x");
    println!("  Dot Product: 2x");
    println!("  Variance: 4x");
    println!("  Histogram: 5x");
}

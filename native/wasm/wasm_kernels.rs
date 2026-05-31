// WebAssembly kernels for browser-based image generation
// Compile with: wasm-pack build --target web
// Use in JavaScript/TypeScript for client-side inference

use wasm_bindgen::prelude::*;
use std::f32;

#[wasm_bindgen]
pub struct WasmQuantizer {
    scale: f32,
}

#[wasm_bindgen]
impl WasmQuantizer {
    #[wasm_bindgen(constructor)]
    pub fn new(scale: f32) -> WasmQuantizer {
        WasmQuantizer { scale }
    }

    /// Quantize floating-point array to INT8
    #[wasm_bindgen]
    pub fn quantize(&self, data: &[f32]) -> Vec<i8> {
        data.iter()
            .map(|&x| {
                let scaled = (x * self.scale).clamp(-128.0, 127.0);
                scaled as i8
            })
            .collect()
    }

    /// Dequantize INT8 array back to float32
    #[wasm_bindgen]
    pub fn dequantize(&self, data: &[i8]) -> Vec<f32> {
        data.iter()
            .map(|&x| (x as f32) / self.scale)
            .collect()
    }
}

#[wasm_bindgen]
pub struct WasmActivation;

#[wasm_bindgen]
impl WasmActivation {
    /// Fast GELU activation
    #[wasm_bindgen]
    pub fn gelu(data: &[f32]) -> Vec<f32> {
        data.iter()
            .map(|&x| {
                let cubic = x * x * x;
                let arg = 0.7978845608f32 * (x + 0.044715f32 * cubic);
                let cdf = 0.5f32 * (1.0f32 + arg.tanh());
                x * cdf
            })
            .collect()
    }

    /// ReLU activation
    #[wasm_bindgen]
    pub fn relu(data: &[f32]) -> Vec<f32> {
        data.iter().map(|&x| x.max(0.0)).collect()
    }

    /// Leaky ReLU activation
    #[wasm_bindgen]
    pub fn leaky_relu(data: &[f32], alpha: f32) -> Vec<f32> {
        data.iter()
            .map(|&x| if x > 0.0 { x } else { alpha * x })
            .collect()
    }

    /// Sigmoid activation
    #[wasm_bindgen]
    pub fn sigmoid(data: &[f32]) -> Vec<f32> {
        data.iter()
            .map(|&x| 1.0f32 / (1.0f32 + (-x).exp()))
            .collect()
    }

    /// Tanh activation
    #[wasm_bindgen]
    pub fn tanh(data: &[f32]) -> Vec<f32> {
        data.iter().map(|&x| x.tanh()).collect()
    }
}

#[wasm_bindgen]
pub struct WasmNormalization;

#[wasm_bindgen]
impl WasmNormalization {
    /// Numerically stable softmax
    #[wasm_bindgen]
    pub fn softmax(data: &[f32]) -> Vec<f32> {
        if data.is_empty() {
            return vec![];
        }

        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let exp_vals: Vec<f32> = data
            .iter()
            .map(|&x| (x - max_val).exp())
            .collect();

        let sum_exp: f32 = exp_vals.iter().sum();

        exp_vals.iter().map(|&x| x / sum_exp).collect()
    }

    /// Layer normalization
    #[wasm_bindgen]
    pub fn layer_norm(
        data: &[f32],
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Vec<f32> {
        let mean = data.iter().sum::<f32>() / data.len() as f32;

        let variance = data
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / data.len() as f32;

        let std_dev = (variance + eps).sqrt();

        data.iter()
            .zip(gamma.iter().zip(beta.iter()))
            .map(|(&x, (&g, &b))| ((x - mean) / std_dev) * g + b)
            .collect()
    }

    /// Batch normalization (inference mode)
    #[wasm_bindgen]
    pub fn batch_norm(
        data: &[f32],
        running_mean: &[f32],
        running_var: &[f32],
        eps: f32,
    ) -> Vec<f32> {
        data.iter()
            .zip(running_mean.iter().zip(running_var.iter()))
            .map(|(&x, (&mean, &var))| {
                (x - mean) / (var + eps).sqrt()
            })
            .collect()
    }
}

#[wasm_bindgen]
pub struct WasmLinearAlgebra;

#[wasm_bindgen]
impl WasmLinearAlgebra {
    /// Dot product
    #[wasm_bindgen]
    pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// L2 norm
    #[wasm_bindgen]
    pub fn l2_norm(data: &[f32]) -> f32 {
        data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Cosine similarity
    #[wasm_bindgen]
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot = Self::dot_product(a, b);
        let norm_a = Self::l2_norm(a);
        let norm_b = Self::l2_norm(b);

        if norm_a > 1e-8 && norm_b > 1e-8 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    /// Vector addition
    #[wasm_bindgen]
    pub fn vector_add(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
    }

    /// Vector subtraction
    #[wasm_bindgen]
    pub fn vector_sub(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
    }

    /// Scalar multiplication
    #[wasm_bindgen]
    pub fn scalar_mul(data: &[f32], scalar: f32) -> Vec<f32> {
        data.iter().map(|&x| x * scalar).collect()
    }
}

#[wasm_bindgen]
pub struct WasmAttention;

#[wasm_bindgen]
impl WasmAttention {
    /// Scaled dot-product attention (simplified 2D version)
    #[wasm_bindgen]
    pub fn scaled_attention(
        query: &[f32],
        key: &[f32],
        value: &[f32],
        dim: usize,
        scale: f32,
    ) -> Vec<f32> {
        let seq_len = query.len() / dim;

        // Compute attention scores
        let mut scores = vec![0.0; seq_len * seq_len];

        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut dot = 0.0;
                for d in 0..dim {
                    dot += query[i * dim + d] * key[j * dim + d];
                }
                scores[i * seq_len + j] = dot * scale;
            }
        }

        // Apply softmax per row
        for i in 0..seq_len {
            let row_start = i * seq_len;
            let row_end = row_start + seq_len;
            let max_score = scores[row_start..row_end]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);

            let mut sum_exp = 0.0;
            for j in row_start..row_end {
                scores[j] = (scores[j] - max_score).exp();
                sum_exp += scores[j];
            }

            for j in row_start..row_end {
                scores[j] /= sum_exp;
            }
        }

        // Multiply by values
        let mut output = vec![0.0; seq_len * dim];

        for i in 0..seq_len {
            for d in 0..dim {
                let mut val = 0.0;
                for j in 0..seq_len {
                    val += scores[i * seq_len + j] * value[j * dim + d];
                }
                output[i * dim + d] = val;
            }
        }

        output
    }
}

/// Utility functions for array manipulation

#[wasm_bindgen]
pub fn array_mean(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f32>() / data.len() as f32
}

#[wasm_bindgen]
pub fn array_std(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let mean = array_mean(data);
    let variance = data
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f32>()
        / data.len() as f32;
    variance.sqrt()
}

#[wasm_bindgen]
pub fn array_max(data: &[f32]) -> f32 {
    data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}

#[wasm_bindgen]
pub fn array_min(data: &[f32]) -> f32 {
    data.iter().cloned().fold(f32::INFINITY, f32::min)
}

#[wasm_bindgen]
pub fn array_sum(data: &[f32]) -> f32 {
    data.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization() {
        let quantizer = WasmQuantizer::new(127.0);
        let data = vec![0.5, -0.5, 1.0];
        let quantized = quantizer.quantize(&data);
        let dequantized = quantizer.dequantize(&quantized);

        for (orig, approx) in data.iter().zip(dequantized.iter()) {
            assert!((orig - approx).abs() < 0.01);
        }
    }

    #[test]
    fn test_softmax() {
        let data = vec![1.0, 2.0, 3.0];
        let softmax = WasmNormalization::softmax(&data);
        let sum: f32 = softmax.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        let similarity = WasmLinearAlgebra::cosine_similarity(&a, &b);
        assert!((similarity - 1.0).abs() < 1e-5);
    }
}

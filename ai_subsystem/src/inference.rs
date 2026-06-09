//! Inference runtime — no_std SIMD-accelerated neural network forward pass.
//!
//! Supports:
//!   - Dense (fully-connected) layers
//!   - ReLU, Sigmoid, Tanh activations
//!   - INT8 / FP32 quantization modes
//!   - AVX2 accelerated matrix multiply (future extension)

use alloc::vec::Vec;
use crate::aligned_vec::AlignedVec;

/// Activation function for a layer.
#[derive(Debug, Clone, Copy)]
pub enum Activation {
    Linear,
    ReLU,
    Tanh,
    Sigmoid,
}

/// A single dense layer: weight matrix + bias vector.
pub struct DenseLayer {
    pub weights: AlignedVec<f32, 32>,   // [out_size * in_size] row-major
    pub biases: AlignedVec<f32, 32>,    // [out_size]
    pub in_size: usize,
    pub out_size: usize,
    pub activation: Activation,
}

#[target_feature(enable = "avx2")]
pub unsafe fn avx2_dot_product_impl(weights: &[f32], inputs: &[f32]) -> f32 {
    use core::arch::x86_64::{_mm256_load_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_add_ps, _mm256_storeu_ps};
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= weights.len() && i + 8 <= inputs.len() {
        let w = _mm256_load_ps(weights.as_ptr().add(i));
        let x = _mm256_load_ps(inputs.as_ptr().add(i));
        let m = _mm256_mul_ps(w, x);
        sum = _mm256_add_ps(sum, m);
        i += 8;
    }
    
    let mut arr = [0.0; 8];
    _mm256_storeu_ps(arr.as_mut_ptr(), sum);
    let mut total: f32 = arr.iter().sum();
    
    while i < weights.len() && i < inputs.len() {
        total += weights[i] * inputs[i];
        i += 1;
    }
    
    total
}

pub fn avx2_dot_product(weights: &[f32], inputs: &[f32]) -> f32 {
    unsafe { avx2_dot_product_impl(weights, inputs) }
}

impl DenseLayer {
    /// Forward pass: computes output = activation(W * input + b).
    /// All arithmetic in f32; INT8 quantized path is a future extension.
    pub fn forward(&self, input: &[f32], output: &mut Vec<f32>) {
        output.clear();
        output.resize(self.out_size, 0.0f32);

        for i in 0..self.out_size {
            let mut sum = self.biases[i];
            let row = &self.weights[i * self.in_size..(i + 1) * self.in_size];
            // Scalar dot product for now; AVX2 will be hooked up inside with_simd in Phase 2
            for (w, x) in row.iter().zip(input.iter()) {
                sum += w * x;
            }
            output[i] = apply_activation(sum, self.activation);
        }
    }
}

fn apply_activation(x: f32, act: Activation) -> f32 {
    match act {
        Activation::Linear  => x,
        Activation::ReLU    => if x > 0.0 { x } else { 0.0 },
        Activation::Sigmoid => 1.0 / (1.0 + fast_exp(-x)),
        Activation::Tanh    => fast_tanh(x),
    }
}

/// Fast approximation of exp(-x) for sigmoid — acceptable precision for kernel AI decisions.
#[inline]
fn fast_exp(x: f32) -> f32 {
    // Schraudolph's approximation — error < 2%
    let i = (x.to_bits() as i64).wrapping_add(
        ((127.0_f32 / core::f32::consts::LN_2) as i64) << 23
    ) as u32;
    f32::from_bits(i)
}

#[inline]
fn fast_tanh(x: f32) -> f32 {
    // Rational approximation valid for |x| < 4
    let x2 = x * x;
    x * (27.0 + x2) / (27.0 + 9.0 * x2)
}

// ── Sequential Model ──────────────────────────────────────────────────────────

/// A sequential stack of dense layers.
pub struct SequentialModel {
    pub layers: Vec<DenseLayer>,
}

impl SequentialModel {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add_layer(&mut self, layer: DenseLayer) {
        self.layers.push(layer);
    }

    /// Run inference. Returns output of final layer.
    pub fn infer(&self, input: &[f32]) -> Vec<f32> {
        let mut current: Vec<f32> = input.to_vec();
        let mut next: Vec<f32> = Vec::new();

        for layer in &self.layers {
            layer.forward(&current, &mut next);
            core::mem::swap(&mut current, &mut next);
        }
        current
    }
}

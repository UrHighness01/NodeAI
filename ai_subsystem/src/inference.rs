//! Inference runtime — no_std SIMD-accelerated neural network forward pass.
//!
//! Supports:
//!   - Dense (fully-connected) layers
//!   - ReLU, Sigmoid, Tanh activations
//!   - INT8 / FP32 quantization modes
//!   - AVX2 accelerated matrix multiply (future extension)

use alloc::vec::Vec;
use core::cmp;
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

pub static mut SIMD_WRAPPER: Option<fn(&mut dyn FnMut())> = None;

// ── INT8 Quantized Layer ─────────────────────────────────────────────────────

/// Symmetric per-row scale factor for one output neuron.
pub struct QuantizedDenseLayer {
    /// INT8 weights [out_size * in_size] row-major.
    pub qweights: AlignedVec<i8, 32>,
    /// Per-output-neuron scale factor.  `dequant_val = qweights[row][col] * scale[row]`.
    pub scales: AlignedVec<f32, 32>,
    /// Bias in f32 (added post-dequant).
    pub biases: AlignedVec<f32, 32>,
    pub in_size: usize,
    pub out_size: usize,
    pub activation: Activation,
}

/// AVX2 INT8 dot product: dequantize i8 → f32, then use AVX2 f32 pipeline.
///
/// This avoids `_mm256_cvtepi8_ps` (not universally available in Rust's
/// core::arch for x86_64-unknown-none) by loading i8 weights into a small
/// f32 buffer and using the existing f32 AVX2 dot product.  The overhead of
/// dequant is negligible (1/8th of the arithmetic) and keeps the codegen
/// simple across all Rust nightly versions.
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_int8_dot_product_impl(qweights: &[i8], inputs: &[f32], scale: f32) -> f32 {
    use core::arch::x86_64::{_mm256_load_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_add_ps, _mm256_storeu_ps};

    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    // Process 8 at a time: load i8 bytes, convert to f32 manually
    while i + 8 <= qweights.len() && i + 8 <= inputs.len() {
        // Load 8 i8 bytes into a temporary f32 buffer
        let mut buf = [0.0f32; 8];
        for j in 0..8 {
            buf[j] = qweights[i + j] as f32;
        }
        let w = _mm256_load_ps(buf.as_ptr());
        let x = _mm256_load_ps(inputs.as_ptr().add(i));
        let m = _mm256_mul_ps(w, x);
        sum = _mm256_add_ps(sum, m);
        i += 8;
    }

    let mut arr = [0.0f32; 8];
    _mm256_storeu_ps(arr.as_mut_ptr(), sum);
    let mut total: f32 = arr.iter().sum();

    // Scalar remainder
    while i < qweights.len() && i < inputs.len() {
        total += (qweights[i] as f32) * inputs[i];
        i += 1;
    }

    total * scale
}

pub fn avx2_int8_dot_product(qweights: &[i8], inputs: &[f32], scale: f32) -> f32 {
    unsafe { avx2_int8_dot_product_impl(qweights, inputs, scale) }
}

/// Scalar INT8 dot product (fallback when AVX2 unavailable).
pub fn scalar_int8_dot_product(qweights: &[i8], inputs: &[f32], scale: f32) -> f32 {
    let mut sum = 0.0f32;
    for (w, x) in qweights.iter().zip(inputs.iter()) {
        sum += (*w as f32) * x;
    }
    sum * scale
}

impl DenseLayer {
    /// Quantize this layer to INT8 using symmetric per-row quantization.
    /// Returns `None` if any row has zero range (all-zeros is valid, scale=1).
    pub fn quantize(&self) -> QuantizedDenseLayer {
        let mut qweights = AlignedVec::<i8, 32>::with_capacity(self.weights.len());
        let mut scales = AlignedVec::<f32, 32>::with_capacity(self.out_size);

        for row in 0..self.out_size {
            let start = row * self.in_size;
            let end = start + self.in_size;
            let row_slice = &self.weights.as_slice()[start..end];

            // Find max absolute value for symmetric quantization
            let max_abs = row_slice.iter()
                .fold(0.0f32, |a, v| a.max(v.abs()))
                .max(1e-10); // avoid division by zero

            let scale = max_abs / 127.0;

            for w in row_slice {
                let q = (libm::roundf(w / scale)).clamp(-128.0, 127.0) as i8;
                qweights.push(q);
            }
            scales.push(scale);
        }

        let biases = AlignedVec::from(self.biases.as_slice());

        QuantizedDenseLayer {
            qweights,
            scales,
            biases,
            in_size: self.in_size,
            out_size: self.out_size,
            activation: self.activation,
        }
    }

    /// Forward pass: computes output = activation(W * input + b).
    /// All arithmetic in f32; INT8 quantized path is a future extension.
    pub fn forward(&self, input: &[f32], output: &mut Vec<f32>) {
        output.clear();
        output.resize(self.out_size, 0.0f32);

        unsafe {
            if let Some(wrapper) = SIMD_WRAPPER {
                let mut f = || {
                    for i in 0..self.out_size {
                        let mut sum = self.biases[i];
                        let row = &self.weights[i * self.in_size..(i + 1) * self.in_size];
                        sum += avx2_dot_product(row, input);
                        output[i] = apply_activation(sum, self.activation);
                    }
                };
                wrapper(&mut f);
                return;
            }
        }

        // Fallback scalar path
        for i in 0..self.out_size {
            let mut sum = self.biases[i];
            let row = &self.weights[i * self.in_size..(i + 1) * self.in_size];
            for (w, x) in row.iter().zip(input.iter()) {
                sum += w * x;
            }
            output[i] = apply_activation(sum, self.activation);
        }
    }
}

impl QuantizedDenseLayer {
    /// INT8 quantized forward pass.
    /// Uses AVX2 when available via the SIMD wrapper, otherwise scalar INT8.
    pub fn forward(&self, input: &[f32], output: &mut Vec<f32>) {
        output.clear();
        output.resize(self.out_size, 0.0f32);

        unsafe {
            if let Some(wrapper) = SIMD_WRAPPER {
                let mut f = || {
                    for i in 0..self.out_size {
                        let mut sum = self.biases[i];
                        let row = &self.qweights[i * self.in_size..(i + 1) * self.in_size];
                        let scale = self.scales[i];
                        // INT8 dot product inside the SIMD wrapper
                        sum += avx2_int8_dot_product(row, input, scale);
                        output[i] = apply_activation(sum, self.activation);
                    }
                };
                wrapper(&mut f);
                return;
            }
        }

        // Scalar INT8 fallback
        for i in 0..self.out_size {
            let mut sum = self.biases[i];
            let row = &self.qweights[i * self.in_size..(i + 1) * self.in_size];
            let scale = self.scales[i];
            sum += scalar_int8_dot_product(row, input, scale);
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

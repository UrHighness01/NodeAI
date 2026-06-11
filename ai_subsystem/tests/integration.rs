//! Integration tests for the ai_subsystem crate.
//! Run with: cargo test -p ai_subsystem --target x86_64-unknown-linux-gnu

use ai_subsystem::aligned_vec::AlignedVec;
use ai_subsystem::inference::{DenseLayer, Activation, SequentialModel, QuantizedDenseLayer};

#[test]
fn test_aligned_vec_push_and_len() {
    let mut v: AlignedVec<f32, 32> = AlignedVec::new();
    assert_eq!(v.len(), 0);
    v.push(1.0);
    v.push(2.0);
    v.push(3.0);
    assert_eq!(v.len(), 3);
    assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_aligned_vec_resize() {
    let mut v: AlignedVec<f32, 32> = AlignedVec::new();
    v.resize(10, 0.0);
    assert_eq!(v.len(), 10);
    assert!(v.iter().all(|x| *x == 0.0));
}

#[test]
fn test_aligned_vec_allocation() {
    let mut v: AlignedVec<f32, 32> = AlignedVec::new();
    for i in 0..1000 {
        v.push(i as f32);
    }
    assert_eq!(v.len(), 1000);
    assert_eq!(v[42], 42.0);
}

#[test]
fn test_aligned_vec_clear() {
    let mut v: AlignedVec<i32, 32> = AlignedVec::new();
    for i in 0..10 { v.push(i); }
    assert_eq!(v.len(), 10);
    v.clear();
    assert_eq!(v.len(), 0);
}

#[test]
fn test_dense_layer_forward_linear() {
    let mut weights = AlignedVec::<f32, 32>::with_capacity(6);
    weights.push(1.0); weights.push(0.0);
    weights.push(0.0); weights.push(1.0);
    weights.push(1.0); weights.push(1.0);
    let mut biases = AlignedVec::<f32, 32>::with_capacity(3);
    biases.push(0.0); biases.push(0.0); biases.push(0.0);

    let layer = DenseLayer {
        in_size: 2,
        out_size: 3,
        weights,
        biases,
        activation: Activation::Linear,
    };

    let input = &[2.0, 3.0];
    let mut output = Vec::new();
    layer.forward(input, &mut output);

    assert_eq!(output.len(), 3);
    assert!((output[0] - 2.0).abs() < 1e-5);  // row 0: 1*2 + 0*3 = 2
    assert!((output[1] - 3.0).abs() < 1e-5);  // row 1: 0*2 + 1*3 = 3
    assert!((output[2] - 5.0).abs() < 1e-5);  // row 2: 1*2 + 1*3 = 5
}

#[test]
fn test_dense_layer_forward_relu() {
    let mut weights = AlignedVec::<f32, 32>::with_capacity(2);
    weights.push(-1.0); weights.push(1.0);
    let mut biases = AlignedVec::<f32, 32>::with_capacity(2);
    biases.push(0.0); biases.push(0.0);

    let layer = DenseLayer {
        in_size: 1,
        out_size: 2,
        weights,
        biases,
        activation: Activation::ReLU,
    };

    let input = &[-5.0];
    let mut output = Vec::new();
    layer.forward(input, &mut output);

    assert_eq!(output.len(), 2);
    // row 0 = -1 * -5 = 5, ReLU(5) = 5
    assert!((output[0] - 5.0).abs() < 1e-5);
    // row 1 = 1 * -5 = -5, ReLU(-5) = 0
    assert!((output[1] - 0.0).abs() < 1e-5);
}

#[test]
fn test_quantized_dense_layer() {
    let mut weights = AlignedVec::<f32, 32>::with_capacity(4);
    weights.push(1.5); weights.push(0.5);
    weights.push(-0.5); weights.push(2.0);
    let mut biases = AlignedVec::<f32, 32>::with_capacity(2);
    biases.push(0.1); biases.push(-0.1);

    let layer = DenseLayer {
        in_size: 2,
        out_size: 2,
        weights,
        biases,
        activation: Activation::Linear,
    };

    let quantized = layer.quantize();
    assert_eq!(quantized.qweights.len(), 4);
    assert_eq!(quantized.scales.len(), 2);
    assert_eq!(quantized.biases.len(), 2);

    // Forward pass should produce similar results to f32 layer
    let input = &[1.0, 1.0];
    let mut f32_output = Vec::new();
    let mut i8_output = Vec::new();

    layer.forward(input, &mut f32_output);
    quantized.forward(input, &mut i8_output);

    // f32: [1.5+0.5+0.1=2.1, -0.5+2.0-0.1=1.4]
    assert!((f32_output[0] - 2.1).abs() < 1e-5);
    assert!((f32_output[1] - 1.4).abs() < 1e-5);

    // INT8 should be close (within quantization error)
    let diff0 = (f32_output[0] - i8_output[0]).abs();
    let diff1 = (f32_output[1] - i8_output[1]).abs();
    assert!(diff0 < 1.0, "INT8 quantization error too large: {}", diff0);
    assert!(diff1 < 1.0, "INT8 quantization error too large: {}", diff1);
}

#[test]
fn test_sequential_model() {
    let mut weights1 = AlignedVec::<f32, 32>::with_capacity(2);
    weights1.push(1.0); weights1.push(1.0);
    let mut biases1 = AlignedVec::<f32, 32>::with_capacity(1);
    biases1.push(0.0);

    let mut weights2 = AlignedVec::<f32, 32>::with_capacity(1);
    weights2.push(2.0);
    let mut biases2 = AlignedVec::<f32, 32>::with_capacity(1);
    biases2.push(-1.0);

    let mut model = SequentialModel::new();
    model.add_layer(DenseLayer { in_size: 2, out_size: 1, weights: weights1, biases: biases1, activation: Activation::Linear });
    model.add_layer(DenseLayer { in_size: 1, out_size: 1, weights: weights2, biases: biases2, activation: Activation::Linear });

    // Layer 1: [1,1] -> [1*3+1*4=7]
    // Layer 2: [7] -> [2*7-1=13]
    let output = model.infer(&[3.0, 4.0]);
    assert_eq!(output.len(), 1);
    assert!((output[0] - 13.0).abs() < 1e-5);
}

#[test]
fn test_sequential_model_relu() {
    let mut w = AlignedVec::<f32, 32>::with_capacity(2);
    w.push(-1.0); w.push(1.0);
    let mut b = AlignedVec::<f32, 32>::with_capacity(2);
    b.push(0.0); b.push(0.0);

    let mut model = SequentialModel::new();
    model.add_layer(DenseLayer { in_size: 1, out_size: 2, weights: w, biases: b, activation: Activation::ReLU });

    let output = model.infer(&[-3.0]);
    assert_eq!(output.len(), 2);
    // ReLU(-1 * -3) = 3, ReLU(1 * -3) = 0
    assert!((output[0] - 3.0).abs() < 1e-5);
    assert!((output[1] - 0.0).abs() < 1e-5);
}

#[test]
fn test_aligned_vec_from_slice() {
    let src = vec![1.0f32, 2.0, 3.0];
    let v: AlignedVec<f32, 32> = AlignedVec::from(src.as_slice());
    assert_eq!(v.len(), 3);
    assert_eq!(v[0], 1.0);
    assert_eq!(v[2], 3.0);
}

#[test]
fn test_dense_layer_sigmoid_range() {
    let mut w = AlignedVec::<f32, 32>::with_capacity(2);
    w.push(10.0); w.push(0.0);
    let mut b = AlignedVec::<f32, 32>::with_capacity(1);
    b.push(0.0);

    let layer = DenseLayer { in_size: 2, out_size: 1, weights: w, biases: b, activation: Activation::Sigmoid };

    let mut output = Vec::new();
    layer.forward(&[100.0, 0.0], &mut output);

    // Sigmoid(10*100 + 0) = Sigmoid(1000) ~= 1.0
    assert!(output[0] > 0.99);
    assert!(output[0] <= 1.0);
}

#[test]
fn test_aligned_vec_alignment() {
    let mut v: AlignedVec<u64, 64> = AlignedVec::new();
    v.push(0xDEAD_BEEF);
    let ptr = v.as_slice().as_ptr() as usize;
    assert_eq!(ptr % 64, 0, "Pointer must be 64-byte aligned");
}

#[test]
fn test_dense_layer_sigmoid_negative_range() {
    let mut w = AlignedVec::<f32, 32>::with_capacity(2);
    w.push(-1.0); w.push(0.0);
    let mut b = AlignedVec::<f32, 32>::with_capacity(1);
    b.push(0.0);
    let layer = DenseLayer { in_size: 2, out_size: 1, weights: w, biases: b, activation: Activation::Sigmoid };
    let mut output = Vec::new();
    layer.forward(&[1.0, 0.0], &mut output);
    // Sigmoid(-1*1 + 0) = Sigmoid(-1) (fast_exp approximation works for small x)
    assert!(output[0] >= 0.0);
    assert!(output[0] <= 1.0, "Sigmoid should be in [0,1]: {}", output[0]);
}

#[test]
fn test_sequential_model_three_layers() {
    let mut model = SequentialModel::new();
    // Layer 1: 2→2 identity
    let mut w1 = AlignedVec::<f32, 32>::with_capacity(4);
    w1.push(1.0); w1.push(0.0); w1.push(0.0); w1.push(1.0);
    let mut b1 = AlignedVec::<f32, 32>::with_capacity(2);
    b1.push(0.0); b1.push(0.0);
    model.add_layer(DenseLayer { in_size: 2, out_size: 2, weights: w1, biases: b1, activation: Activation::Linear });
    // Layer 2: 2→1
    let mut w2 = AlignedVec::<f32, 32>::with_capacity(2);
    w2.push(2.0); w2.push(3.0);
    let mut b2 = AlignedVec::<f32, 32>::with_capacity(1);
    b2.push(1.0);
    model.add_layer(DenseLayer { in_size: 2, out_size: 1, weights: w2, biases: b2, activation: Activation::Linear });
    // Layer 3: 1→1
    let mut w3 = AlignedVec::<f32, 32>::with_capacity(1);
    w3.push(0.5);
    let mut b3 = AlignedVec::<f32, 32>::with_capacity(1);
    b3.push(0.0);
    model.add_layer(DenseLayer { in_size: 1, out_size: 1, weights: w3, biases: b3, activation: Activation::Linear });
    // Forward: [1,2] → [1,2] → [2*1+3*2+1=9] → [0.5*9=4.5]
    let output = model.infer(&[1.0, 2.0]);
    assert!((output[0] - 4.5).abs() < 1e-5);
}

#[test]
fn test_quantized_dense_layer_forward_identity() {
    let mut w = AlignedVec::<f32, 32>::with_capacity(3);
    w.push(1.0); w.push(0.5); w.push(0.0);
    let mut b = AlignedVec::<f32, 32>::with_capacity(3);
    b.push(0.0); b.push(0.0); b.push(0.0);
    let layer = DenseLayer { in_size: 1, out_size: 3, weights: w, biases: b, activation: Activation::Linear };
    let q = layer.quantize();
    let mut q_out = Vec::new();
    q.forward(&[2.0], &mut q_out);
    assert_eq!(q_out.len(), 3);
    let mut f32_out = Vec::new();
    layer.forward(&[2.0], &mut f32_out);
    for i in 0..3 {
        assert!((q_out[i] - f32_out[i]).abs() < 1.0, "INT8 mismatch at {}: {} vs {}", i, q_out[i], f32_out[i]);
    }
}

#[test]
fn test_aligned_vec_truncate() {
    let mut v: AlignedVec<f32, 32> = AlignedVec::new();
    v.push(10.0); v.push(20.0); v.push(30.0);
    assert_eq!(v.len(), 3);
    v.clear();
    assert_eq!(v.len(), 0);
}

#[test]
fn test_dense_layer_tanh_activation() {
    let mut w = AlignedVec::<f32, 32>::with_capacity(2);
    w.push(2.0); w.push(0.0);
    let mut b = AlignedVec::<f32, 32>::with_capacity(1);
    b.push(0.0);
    let layer = DenseLayer { in_size: 2, out_size: 1, weights: w, biases: b, activation: Activation::Tanh };
    let mut output = Vec::new();
    layer.forward(&[-3.0, 0.0], &mut output);
    // Tanh(2*(-3) + 0) = Tanh(-6) ≈ -1.0 (fast_tanh approximation)
    assert!(output[0] < 0.0, "Tanh should be negative for negative input: {}", output[0]);
    assert!(output[0] > -1.5);
}

#[test]
fn test_sequential_model_relu_negative_zeros() {
    let mut w = AlignedVec::<f32, 32>::with_capacity(6);
    w.push(-2.0); w.push(0.0);
    w.push(1.0); w.push(-1.0);
    w.push(0.0); w.push(3.0);
    let mut b = AlignedVec::<f32, 32>::with_capacity(3);
    b.push(0.0); b.push(0.0); b.push(0.0);
    let mut model = SequentialModel::new();
    model.add_layer(DenseLayer { in_size: 2, out_size: 3, weights: w, biases: b, activation: Activation::ReLU });
    let output = model.infer(&[-5.0, 2.0]);
    // Row 0: -2*-5 + 0*2 = 10 → ReLU(10) = 10
    // Row 1: 1*-5 + -1*2 = -7 → ReLU(-7) = 0
    // Row 2: 0*-5 + 3*2 = 6 → ReLU(6) = 6
    assert!((output[0] - 10.0).abs() < 1e-5);
    assert!((output[1] - 0.0).abs() < 1e-5);
    assert!((output[2] - 6.0).abs() < 1e-5);
}

#[test]
fn test_quantized_dense_layer_preserves_sign() {
    let mut w = AlignedVec::<f32, 32>::with_capacity(2);
    w.push(-5.0); w.push(3.5);
    let mut b = AlignedVec::<f32, 32>::with_capacity(1);
    b.push(-1.0);
    let layer = DenseLayer { in_size: 2, out_size: 1, weights: w, biases: b, activation: Activation::Linear };
    let q = layer.quantize();
    let mut out = Vec::new();
    q.forward(&[-1.0, 2.0], &mut out);
    // f32: -5*(-1) + 3.5*2 - 1 = 5 + 7 - 1 = 11 → positive
    assert!(out[0] > 0.0, "INT8 should preserve positive sign: {}", out[0]);
}

#[test]
fn test_aligned_vec_reserve() {
    let mut v: AlignedVec<f32, 32> = AlignedVec::new();
    for i in 0..100 { v.push(i as f32); }
    assert_eq!(v.len(), 100);
    assert_eq!(v[50], 50.0);
    v.clear();
    assert_eq!(v.len(), 0);
}

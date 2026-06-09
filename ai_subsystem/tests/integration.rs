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

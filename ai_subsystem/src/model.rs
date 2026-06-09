//! AI model loader — validates and loads model weights from disk at boot.

use alloc::{string::String, vec::Vec};
use crate::inference::{DenseLayer, SequentialModel, Activation};

/// Minimal NodeAI model file header.
/// Binary format: MAGIC(4) | VERSION(2) | LAYER_COUNT(2) | layers...
const MAGIC: &[u8; 4] = b"NAIM";
const FORMAT_VERSION: u16 = 1;

#[derive(Debug)]
pub enum ModelLoadError {
    BadMagic,
    UnsupportedVersion(u16),
    TruncatedData,
    InvalidShape,
}

/// Load a NodeAI model from a raw byte slice.
/// Returns a validated SequentialModel ready for inference.
pub fn load_from_bytes(data: &[u8]) -> Result<SequentialModel, ModelLoadError> {
    if data.len() < 8 {
        return Err(ModelLoadError::TruncatedData);
    }
    if &data[0..4] != MAGIC {
        return Err(ModelLoadError::BadMagic);
    }
    let version = u16::from_le_bytes([data[4], data[5]]);
    if version != FORMAT_VERSION {
        return Err(ModelLoadError::UnsupportedVersion(version));
    }
    let layer_count = u16::from_le_bytes([data[6], data[7]]) as usize;

    let mut model = SequentialModel::new();
    let mut cursor = 8usize;

    for _ in 0..layer_count {
        if cursor + 9 > data.len() {
            return Err(ModelLoadError::TruncatedData);
        }
        let in_size  = u32::from_le_bytes(data[cursor..cursor+4].try_into().unwrap()) as usize;
        let out_size = u32::from_le_bytes(data[cursor+4..cursor+8].try_into().unwrap()) as usize;
        let act_byte = data[cursor + 8];
        cursor += 9;

        let weight_count = in_size * out_size;
        let bias_count   = out_size;
        let bytes_needed = (weight_count + bias_count) * 4;

        if cursor + bytes_needed > data.len() {
            return Err(ModelLoadError::TruncatedData);
        }

        let mut weights = crate::aligned_vec::AlignedVec::with_capacity(weight_count);
        for i in 0..weight_count {
            let off = cursor + i * 4;
            weights.push(f32::from_le_bytes(data[off..off+4].try_into().unwrap()));
        }
        cursor += weight_count * 4;

        let mut biases = crate::aligned_vec::AlignedVec::with_capacity(bias_count);
        for i in 0..bias_count {
            let off = cursor + i * 4;
            biases.push(f32::from_le_bytes(data[off..off+4].try_into().unwrap()));
        }
        cursor += bias_count * 4;

        let activation = match act_byte {
            0 => Activation::Linear,
            1 => Activation::ReLU,
            2 => Activation::Tanh,
            3 => Activation::Sigmoid,
            _ => Activation::ReLU,
        };

        model.add_layer(DenseLayer { weights, biases, in_size, out_size, activation });
    }

    Ok(model)
}

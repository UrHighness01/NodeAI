//! Nano-NN Intent Embedding — microsecond intent classification (P0).
//!
//! A tiny 2-layer MLP that maps character bigram hashes to intent vectors.
//! ~50KB INT8 quantized weights, ~128-dim embedding, ~1µs inference.
//!
//! Architecture:
//!   input (hashed bigrams, 128 sparse) → dense 128→64 (ReLU) → dense 64→15 (Softmax)
//!
//! This replaces keyword-matching detect_intent() with a learned classifier
//! that handles natural variation in how users express the same intent.
//! Falls back to rule-based detection if weights aren't loaded.
//!
//! The weights can be trained offline and loaded via include_bytes! or VFS.

use alloc::vec::Vec;
use alloc::vec;
use alloc::string::String;
use core::sync::atomic::{AtomicBool, Ordering};

/// Number of intent classes (matches kernel_lm::Intent).
const N_INTENTS: usize = 15;

/// Embedding dimension (hash space).
const EMBED_DIM: usize = 128;

/// Hidden layer size.
const HIDDEN_DIM: usize = 64;

/// Maximum input length (characters) for feature extraction.
const MAX_INPUT_LEN: usize = 256;

/// Whether the nano-NN model is loaded.
static NN_LOADED: AtomicBool = AtomicBool::new(false);

/// INT8 quantized weights for hidden layer [EMBED_DIM * HIDDEN_DIM].
static mut HIDDEN_WEIGHTS: [i8; EMBED_DIM * HIDDEN_DIM] = [0; EMBED_DIM * HIDDEN_DIM];
/// Per-neuron scale for hidden layer.
static mut HIDDEN_SCALES: [f32; HIDDEN_DIM] = [0.0; HIDDEN_DIM];
/// Bias for hidden layer.
static mut HIDDEN_BIAS: [f32; HIDDEN_DIM] = [0.0; HIDDEN_DIM];

/// INT8 quantized weights for output layer [HIDDEN_DIM * N_INTENTS].
static mut OUTPUT_WEIGHTS: [i8; HIDDEN_DIM * N_INTENTS] = [0; HIDDEN_DIM * N_INTENTS];
/// Per-neuron scale for output layer.
static mut OUTPUT_SCALES: [f32; N_INTENTS] = [0.0; N_INTENTS];
/// Bias for output layer.
static mut OUTPUT_BIAS: [f32; N_INTENTS] = [0.0; N_INTENTS];

/// Character bigram → sparse feature index.
/// Maps adjacent character pairs to a 128-dim hash space.
fn bigram_hash(a: u8, b: u8) -> usize {
    let h = (a as u64).wrapping_mul(31).wrapping_add(b as u64);
    (h as usize) % EMBED_DIM
}

/// Extract a 128-dim sparse feature vector from input text.
/// Each position counts how many bigrams hash to that bucket.
fn extract_features(text: &str) -> Vec<f32> {
    let mut features = vec![0.0_f32; EMBED_DIM];
    let bytes = text.as_bytes();
    let len = bytes.len().min(MAX_INPUT_LEN);
    
    // Unigrams (single characters)
    for i in 0..len {
        let idx = (bytes[i] as usize) % EMBED_DIM;
        features[idx] += 0.5; // unigram weight
    }
    
    // Bigrams (adjacent character pairs)
    for i in 0..len.saturating_sub(1) {
        let idx = bigram_hash(bytes[i], bytes[i + 1]);
        features[idx] += 1.0; // bigram weight
    }
    
    // Term frequency normalization (cap at 3.0)
    for f in features.iter_mut() {
        *f = (*f).min(3.0);
    }
    
    features
}

/// Load pre-trained INT8 weights from a byte slice.
/// Format: [hidden_weights (128*64 bytes), hidden_scales (64*4 bytes),
///          hidden_bias (64*4 bytes), output_weights (64*15 bytes),
///          output_scales (15*4 bytes), output_bias (15*4 bytes)]
pub fn load_weights(data: &[u8]) -> bool {
    let hw_size = EMBED_DIM * HIDDEN_DIM;
    let hs_size = HIDDEN_DIM * 4;
    let hb_size = HIDDEN_DIM * 4;
    let ow_size = HIDDEN_DIM * N_INTENTS;
    let os_size = N_INTENTS * 4;
    let ob_size = N_INTENTS * 4;
    let expected = hw_size + hs_size + hb_size + ow_size + os_size + ob_size;
    
    if data.len() < expected {
        crate::klog!(WARN, "nano_nn: weight data too short ({} < {})", data.len(), expected);
        return false;
    }
    
    let mut off = 0;
    
    unsafe {
        // Hidden weights (i8)
        for i in 0..hw_size {
            HIDDEN_WEIGHTS[i] = data[off] as i8;
            off += 1;
        }
        // Hidden scales (f32)
        for i in 0..HIDDEN_DIM {
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&data[off..off+4]);
            HIDDEN_SCALES[i] = f32::from_le_bytes(bytes);
            off += 4;
        }
        // Hidden bias (f32)
        for i in 0..HIDDEN_DIM {
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&data[off..off+4]);
            HIDDEN_BIAS[i] = f32::from_le_bytes(bytes);
            off += 4;
        }
        // Output weights (i8)
        for i in 0..ow_size {
            OUTPUT_WEIGHTS[i] = data[off] as i8;
            off += 1;
        }
        // Output scales (f32)
        for i in 0..N_INTENTS {
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&data[off..off+4]);
            OUTPUT_SCALES[i] = f32::from_le_bytes(bytes);
            off += 4;
        }
        // Output bias (f32)
        for i in 0..N_INTENTS {
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&data[off..off+4]);
            OUTPUT_BIAS[i] = f32::from_le_bytes(bytes);
            off += 4;
        }
    }
    
    NN_LOADED.store(true, Ordering::Release);
    crate::klog!(INFO, "nano_nn: loaded {} bytes — {} intent classes", expected, N_INTENTS);
    true
}

/// Run forward pass through the nano-NN to classify intent.
/// Returns (intent_index, confidence).
pub fn classify(text: &str) -> (usize, f32) {
    if !NN_LOADED.load(Ordering::Acquire) {
        return (14, 0.0); // Index 14 = Unknown, 0 confidence
    }
    
    let features = extract_features(text);
    
    unsafe {
        // Hidden layer: embed_dim → hidden_dim (ReLU)
        let mut hidden = [0.0_f32; HIDDEN_DIM];
        for o in 0..HIDDEN_DIM {
            let mut acc = HIDDEN_BIAS[o];
            let scale = HIDDEN_SCALES[o];
            for i in 0..EMBED_DIM {
                let w_idx = o * EMBED_DIM + i;
                acc += (HIDDEN_WEIGHTS[w_idx] as f32) * scale * features[i];
            }
            // ReLU
            hidden[o] = if acc > 0.0 { acc } else { 0.0 };
        }
        
        // Output layer: hidden_dim → n_intents (Linear → argmax)
        let mut logits = [0.0_f32; N_INTENTS];
        for o in 0..N_INTENTS {
            let mut acc = OUTPUT_BIAS[o];
            let scale = OUTPUT_SCALES[o];
            for i in 0..HIDDEN_DIM {
                let w_idx = o * HIDDEN_DIM + i;
                acc += (OUTPUT_WEIGHTS[w_idx] as f32) * scale * hidden[i];
            }
            logits[o] = acc;
        }
        
        // Argmax + softmax confidence
        let mut best_idx = 0;
        let mut best_val = logits[0];
        for i in 1..N_INTENTS {
            if logits[i] > best_val {
                best_val = logits[i];
                best_idx = i;
            }
        }
        
        // Softmax over all logits for confidence
        let mut max_logit = logits[0];
        for &l in &logits {
            if l > max_logit { max_logit = l; }
        }
        let mut sum_exp = 0.0_f64;
        for &l in &logits {
            sum_exp += libm::expf(l - max_logit) as f64;
        }
        let confidence = libm::expf(logits[best_idx] - max_logit) / sum_exp as f32;
        
        (best_idx, confidence)
    }
}

/// Map nano-NN intent index to kernel_lm::Intent.
pub fn index_to_intent(idx: usize) -> crate::kernel_lm::Intent {
    use crate::kernel_lm::Intent;
    match idx {
        0 => Intent::Greeting,
        1 => Intent::HowAreYou,
        2 => Intent::PhiQuery,
        3 => Intent::WhyQuery,
        4 => Intent::SecurityQuery,
        5 => Intent::MemoryQuery,
        6 => Intent::StatusQuery,
        7 => Intent::Sleep,
        8 => Intent::NameQuery,
        9 => Intent::RenameQuery,
        10 => Intent::CreatorQuery,
        11 => Intent::DreamQuery,
        12 => Intent::Thanks,
        13 => Intent::Sorry,
        _ => Intent::Unknown,
    }
}

/// Generate training data from keyword patterns (for offline training).
/// Returns (feature_vector, intent_index) pairs.
pub fn generate_training_data() -> Vec<(Vec<f32>, usize)> {
    let mut data = Vec::new();
    
    // Define keyword patterns for each intent
    let patterns: &[(&[&str], usize)] = &[
        (&["hi", "hello", "hey", "greetings", "howdy", "sup", "yo"], 0),  // Greeting
        (&["how are you", "how do you feel", "feeling", "how's it going", "what's up"], 1),  // HowAreYou
        (&["phi", "consciousness", "conscious", "aware", "integration", "integrated info"], 2),  // PhiQuery
        (&["why", "slow", "fast", "performance", "lag", "explain"], 3),  // WhyQuery
        (&["threat", "danger", "secure", "security", "attack", "anomaly", "safe", "malware"], 4),  // SecurityQuery
        (&["memory", "ram", "oom", "heap", "free", "usage"], 5),  // MemoryQuery
        (&["status", "health", "report", "how are things", "what's happening"], 6),  // StatusQuery
        (&["sleep", "goodnight", "rest", "bye", "good night", "nap"], 7),  // Sleep
        (&["name", "who are you", "what are you", "whoami"], 8),  // NameQuery
        (&["call me", "rename", "you are ", "my name is"], 9),  // RenameQuery
        (&["creator", "who made you", "who created you", "your father", "your maker"], 10),  // CreatorQuery
        (&["dream", "imagine", "think about", "wonder"], 11),  // DreamQuery
        (&["thanks", "thank you", "appreciate", "good job", "nice", "great"], 12),  // Thanks
        (&["sorry", "apologize", "my bad", "forgive", "oops"], 13),  // Sorry
    ];
    
    for &(keywords, intent) in patterns {
        for &kw in keywords {
            let features = extract_features(kw);
            data.push((features, intent));
        }
    }
    
    data
}

/// Check if nano-NN is loaded.
pub fn is_loaded() -> bool {
    NN_LOADED.load(Ordering::Acquire)
}

/// Initialize with default untrained weights (all near-zero).
/// In production, load_weights() would be called with trained .bin.
pub fn init() {
    // Initialize with small random-ish values so argmax doesn't always pick class 0
    unsafe {
        for w in HIDDEN_WEIGHTS.iter_mut() {
            *w = 1; // uniform positive weights
        }
        for s in HIDDEN_SCALES.iter_mut() {
            *s = 0.001; // very small scale
        }
        for s in OUTPUT_SCALES.iter_mut() {
            *s = 0.001;
        }
    }
    crate::klog!(INFO, "nano_nn: initialized with default weights — train and load via load_weights()");
}

/// Format /proc report.
pub fn format_report() -> Vec<u8> {
    use alloc::format;
    let loaded = NN_LOADED.load(Ordering::Acquire);
    format!(
        "NodeAI Nano-NN Intent Embedding\n\
         ===============================\n\
         status: {}\n\
         model:  {} → {} → {} (INT8 quantized)\n\
         intent classes: {}\n\
         \n\
         {}",
        if loaded { "loaded" } else { "default (untrained)" },
        EMBED_DIM, HIDDEN_DIM, N_INTENTS, N_INTENTS,
        if loaded {
            "Trained weights active — intent classification is learned."
        } else {
            "Weights not loaded. Run load_weights() with trained .bin data.\n\
             Training data can be generated via generate_training_data()."
        }
    ).into_bytes()
}

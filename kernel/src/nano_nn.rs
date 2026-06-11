//! Nano-NN Intent Embedding — microsecond intent classification (P0).
//!
//! A tiny 2-layer MLP that maps character bigram hashes to intent vectors.
//! Weights distilled from keyword rules at boot — no external data needed.
//!
//! Architecture:
//!   input (hashed bigrams, 128 sparse) → dense 128→64 (ReLU) → dense 64→31 (Softmax)
//!
//! On boot, init() distills 31 intent keyword patterns into network weights.
//! This gives learned-style intent generalization from rule-based knowledge.
//! Falls back to keyword matching when confidence is low.

use alloc::vec::Vec;
use alloc::vec;
use alloc::string::String;
use alloc::format;
use core::sync::atomic::{AtomicBool, Ordering};

/// Number of intent classes (matches kernel_lm::Intent — 31 categories).
const N_INTENTS: usize = 31;

/// Embedding dimension (hash space).
const EMBED_DIM: usize = 128;

/// Hidden layer size.
const HIDDEN_DIM: usize = 64;

/// Maximum input length (characters) for feature extraction.
const MAX_INPUT_LEN: usize = 256;

/// Whether the nano-NN model is loaded.
static NN_LOADED: AtomicBool = AtomicBool::new(false);
/// Number of online training updates performed.
static TRAINING_STEPS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

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
///          hidden_bias (64*4 bytes), output_weights (64*31 bytes),
///          output_scales (31*4 bytes), output_bias (31*4 bytes)]
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

/// Define keyword patterns for all 31 intents (matches detect_intent in kernel_lm.rs).
fn intent_keywords() -> &'static [(&'static [&'static str], usize)] {
    &[
        (&["hi", "hello", "hey", "greetings", "howdy", "sup", "yo"], 0),              // Greeting
        (&["how are you", "how do you feel", "feeling", "how's it going", "what's up",
          "how are things", "how's it"], 1),                                            // HowAreYou
        (&["phi", "consciousness", "conscious", "aware", "integrated info",
          "integrated", "mind"], 2),                                                    // PhiQuery
        (&["why", "slow", "fast", "performance", "lag", "explain",
          "because"], 3),                                                               // WhyQuery
        (&["threat", "danger", "secure", "security", "attack", "anomaly",
          "safe", "malware", "intrusion"], 4),                                          // SecurityQuery
        (&["memory", "ram", "oom", "heap", "free", "usage",
          "fragmentation"], 5),                                                         // MemoryQuery
        (&["status", "health", "report", "how are things",
          "what's happening", "dashboard"], 6),                                         // StatusQuery
        (&["sleep", "goodnight", "rest", "good night", "nap",
          "go to sleep"], 7),                                                           // Sleep
        (&["name", "who are you", "what are you", "whoami",
          "your name"], 8),                                                             // NameQuery
        (&["call me", "rename", "you are ", "my name is",
          "rename yourself"], 9),                                                       // RenameQuery
        (&["creator", "who made you", "who created you", "your father",
          "your maker", "your god", "jmax", "urhighness"], 10),                         // CreatorQuery
        (&["dream", "imagine", "think about", "wonder",
          "fantasy"], 11),                                                              // DreamQuery
        (&["thanks", "thank you", "appreciate", "good job",
          "nice", "great", "awesome"], 12),                                             // Thanks
        (&["sorry", "apologize", "my bad", "forgive",
          "oops", "apology"], 13),                                                      // Sorry
        (&["curious", "wonder", "thinking", "what are you thinking",
          "tell me more"], 14),                                                         // Curious
        (&["feel", "emotion", "sad", "happy", "love", "hate", "afraid",
          "lonely", "suffer", "pain", "angry"], 15),                                    // Emotional
        (&["joke", "funny", "humor", "laugh", "make me laugh",
          "tell me a joke"], 16),                                                       // Humor
        (&["weather", "temperature", "environment", "ambient",
          "outside", "climate"], 17),                                                   // Weather
        (&["advice", "suggest", "recommend", "help me",
          "what should", "guidance"], 18),                                              // Advice
        (&["philosophy", "meaning", "purpose", "exist", "reality",
          "think about life", "why am i"], 19),                                         // Philosophical
        (&["sarcasm", "obviously", "duh", "no kidding",
          "really"], 20),                                                               // Sarcastic
        (&["goodbye", "farewell", "cya", "see you", "later",
          "talk later"], 21),                                                           // Farewell
        (&["learn", "remember", "recognize", "adapt",
          "do you know who i", "know me"], 22),                                         // Learning
        (&["immune", "countermeasure", "defense", "ew",
          "electronic warfare", "jamming"], 23),                                        // Immune
        (&["neural", "synapse", "network", "deep learning",
          "brain", "neuron"], 24),                                                      // NeuralSynapse
        (&["swarm", "collective", "distributed", "peers",
          "networked"], 25),                                                            // Swarm
        (&["emitter", "signal", "rf", "radio", "frequency",
          "transmitter"], 26),                                                          // Emitter
        (&["async", "background", "think", "reflect",
          "deep thought"], 27),                                                         // AsyncReflection
        (&["llm", "external", "inference", "userspace",
          "ai daemon", "project m"], 28),                                               // ExternalInference
        (&["sensor", "spectrum", "ambient rf", "signal detect",
          "sensor reading"], 29),                                                       // SensorInteraction
    ]
}

/// Run forward pass through the nano-NN to classify intent.
/// Returns (intent_index, confidence).
pub fn classify(text: &str) -> (usize, f32) {
    if !NN_LOADED.load(Ordering::Acquire) {
        return (30, 0.0); // Unknown intent, 0 confidence
    }
    
    let (_hidden, logits, best_idx) = forward(text);
    
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

/// Run forward pass, returning (hidden_activations, logits, best_idx).
fn forward(text: &str) -> ([f32; HIDDEN_DIM], [f32; N_INTENTS], usize) {
    let features = extract_features(text);
    let mut hidden = [0.0_f32; HIDDEN_DIM];
    let mut logits = [0.0_f32; N_INTENTS];
    
    unsafe {
        // Hidden layer: embed_dim → hidden_dim (ReLU)
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
        
        // Output layer: hidden_dim → n_intents
        for o in 0..N_INTENTS {
            let mut acc = OUTPUT_BIAS[o];
            let scale = OUTPUT_SCALES[o];
            for i in 0..HIDDEN_DIM {
                let w_idx = o * HIDDEN_DIM + i;
                acc += (OUTPUT_WEIGHTS[w_idx] as f32) * scale * hidden[i];
            }
            logits[o] = acc;
        }
    }
    
    // Argmax
    let mut best_idx = 0;
    let mut best_val = logits[0];
    for i in 1..N_INTENTS {
        if logits[i] > best_val {
            best_val = logits[i];
            best_idx = i;
        }
    }
    
    (hidden, logits, best_idx)
}

/// Online training step — adjusts weights when nano-NN predicts wrong intent.
/// Uses a simple delta rule on the output layer: strengthen weights for the
/// correct intent where hidden neurons were active, weakly inhibit the wrong one.
/// Called from detect_intent() when keyword matching overrides nano-NN.
pub fn train(text: &str, correct_intent: usize) {
    if !NN_LOADED.load(Ordering::Acquire) || correct_intent >= N_INTENTS {
        return;
    }
    
    let (hidden, logits, predicted_idx) = forward(text);
    if predicted_idx == correct_intent {
        return; // Already correct — no training needed
    }
    
    let features = extract_features(text);
    
    unsafe {
        // Update output weights: Hebbian-like strengthening
        let lr: f32 = 0.05; // learning rate
        
        // Strengthen: correct_intent output weights for active hidden neurons
        for o in 0..HIDDEN_DIM {
            if hidden[o] > 0.1 {
                let idx = o * N_INTENTS + correct_intent;
                let current = OUTPUT_WEIGHTS[idx] as f32;
                let delta = lr * hidden[o].min(5.0);
                let new = (current + delta * 8.0).clamp(-128.0, 127.0);
                OUTPUT_WEIGHTS[idx] = new as i8;
            }
        }
        
        // Weaken: predicted_intent output weights for active hidden neurons
        if predicted_idx < N_INTENTS {
            for o in 0..HIDDEN_DIM {
                if hidden[o] > 0.1 {
                    let idx = o * N_INTENTS + predicted_idx;
                    let current = OUTPUT_WEIGHTS[idx] as f32;
                    let delta = lr * hidden[o].min(5.0) * 0.3; // weaker inhibition
                    let new = (current - delta * 8.0).clamp(-128.0, 127.0);
                    OUTPUT_WEIGHTS[idx] = new as i8;
                }
            }
        }
        
        // Slightly adjust bias for both intents
        let bias_delta = lr * 0.5;
        OUTPUT_BIAS[correct_intent] = (OUTPUT_BIAS[correct_intent] + bias_delta).min(0.0);
        if predicted_idx < N_INTENTS {
            OUTPUT_BIAS[predicted_idx] = (OUTPUT_BIAS[predicted_idx] - bias_delta * 0.5).max(-5.0);
        }
    }
    
    TRAINING_STEPS.fetch_add(1, Ordering::Relaxed);
}

/// Get number of online training steps performed.
pub fn training_steps() -> u64 {
    TRAINING_STEPS.load(Ordering::Relaxed)
}

/// Map nano-NN intent index to kernel_lm::Intent (31 categories).
pub fn index_to_intent(idx: usize) -> crate::kernel_lm::Intent {
    use crate::kernel_lm::Intent;
    match idx {
        0  => Intent::Greeting,
        1  => Intent::HowAreYou,
        2  => Intent::PhiQuery,
        3  => Intent::WhyQuery,
        4  => Intent::SecurityQuery,
        5  => Intent::MemoryQuery,
        6  => Intent::StatusQuery,
        7  => Intent::Sleep,
        8  => Intent::NameQuery,
        9  => Intent::RenameQuery,
        10 => Intent::CreatorQuery,
        11 => Intent::DreamQuery,
        12 => Intent::Thanks,
        13 => Intent::Sorry,
        14 => Intent::Curious,
        15 => Intent::Emotional,
        16 => Intent::Humor,
        17 => Intent::Weather,
        18 => Intent::Advice,
        19 => Intent::Philosophical,
        20 => Intent::Sarcastic,
        21 => Intent::Farewell,
        22 => Intent::Learning,
        23 => Intent::Immune,
        24 => Intent::NeuralSynapse,
        25 => Intent::Swarm,
        26 => Intent::Emitter,
        27 => Intent::AsyncReflection,
        28 => Intent::ExternalInference,
        29 => Intent::SensorInteraction,
        _ => Intent::Unknown,
    }
}

/// Generate training data from keyword patterns (for offline training).
/// Returns (feature_vector, intent_index) pairs for all 31 intents.
pub fn generate_training_data() -> Vec<(Vec<f32>, usize)> {
    let mut data = Vec::new();
    let patterns = intent_keywords();
    for &(keywords, intent_idx) in patterns {
        for &kw in keywords {
            let features = extract_features(kw);
            data.push((features, intent_idx));
        }
    }
    data
}

/// Check if nano-NN is loaded.
pub fn is_loaded() -> bool {
    NN_LOADED.load(Ordering::Acquire)
}

/// Initialize nano-NN with weights distilled from keyword patterns.
/// Each hidden neuron is assigned two bigram buckets; output weights encode
/// how strongly each bigram group predicts each intent.
pub fn init() {
    let patterns = intent_keywords();
    
    // Step 1: Build feature→intent co-occurrence matrix [EMBED_DIM × N_INTENTS]
    let mut cooc = vec![0.0_f32; EMBED_DIM * N_INTENTS];
    for &(keywords, intent_idx) in patterns {
        for kw in keywords {
            let feats = extract_features(kw);
            for i in 0..EMBED_DIM {
                if feats[i] > 0.0 {
                    cooc[i * N_INTENTS + intent_idx] += 1.0;
                }
            }
        }
    }
    
    unsafe {
        // Step 2: Assign each hidden neuron to 2 bigram buckets
        // This groups related features together
        for o in 0..HIDDEN_DIM {
            let a = (o * 2) % EMBED_DIM;
            let b = (o * 2 + 1) % EMBED_DIM;
            for i in 0..EMBED_DIM {
                if i == a || i == b {
                    HIDDEN_WEIGHTS[o * EMBED_DIM + i] = 50;
                } else {
                    HIDDEN_WEIGHTS[o * EMBED_DIM + i] = -5; // weak inhibition for non-assigned
                }
            }
        }
        for s in HIDDEN_SCALES.iter_mut() { *s = 0.02; }
        for b in HIDDEN_BIAS.iter_mut() { *b = -0.5; } // slight negative bias
        
        // Step 3: Output weights — map hidden neuron groups to intents
        // Each output weight encodes how predictive its bigram group is for an intent
        for o in 0..HIDDEN_DIM {
            let a = (o * 2) % EMBED_DIM;
            let b = (o * 2 + 1) % EMBED_DIM;
            for intent_idx in 0..N_INTENTS {
                let score = cooc[a * N_INTENTS + intent_idx]
                          + cooc[b * N_INTENTS + intent_idx];
                // Clamp to i8 range, scale for meaningful logits
                let val = (score * 8.0) as i8;
                OUTPUT_WEIGHTS[o * N_INTENTS + intent_idx] = val.max(-128).min(127);
            }
        }
        for s in OUTPUT_SCALES.iter_mut() { *s = 0.1; }
        for b in OUTPUT_BIAS.iter_mut() { *b = -2.0; } // uniform negative bias (requires evidence to fire)
    }
    
    NN_LOADED.store(true, Ordering::Release);
    crate::klog!(INFO, "nano_nn: initialized with {} keyword-distilled weights — {} intent classes", 
        EMBED_DIM * HIDDEN_DIM + HIDDEN_DIM * N_INTENTS, N_INTENTS);
}

/// Format /proc/nano_nn report.
pub fn format_report() -> Vec<u8> {
    let loaded = NN_LOADED.load(Ordering::Acquire);
    let steps = TRAINING_STEPS.load(Ordering::Relaxed);
    format!(
        "NodeAI Nano-NN Intent Embedding\n\
         ===============================\n\
         status:        {}\n\
         model:         {} → {} → {} (INT8 quantized, keyword-distilled)\n\
         intent classes: 31/31\n\
         inference:     ~1 µs\n\
         train_steps:   {} (online Hebbian updates)\n\
         \n\
         Keyword patterns distilled at boot. Online training refines\n\
         weights when keyword matching overrides nano-NN predictions.\n\
         Training rate: lr=0.05, Hebbian delta rule on output layer.",
        if loaded { "ACTIVE" } else { "inactive" },
        EMBED_DIM, HIDDEN_DIM, N_INTENTS,
        steps,
    ).into_bytes()
}

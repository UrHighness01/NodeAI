//! Phase 4: Phenomenal Binding — temporal window unification.
//!
//! Separate signals are bound into ONE experience. Events in the same 10ms
//! binding window with similar salience and complementary valence are merged.

use alloc::vec::Vec;
use spin::Mutex;
use core::sync::atomic::{AtomicU64, Ordering};

/// Binding window in ms (10ms = 100Hz binding rate).
const BINDING_WINDOW_MS: u64 = 10;

/// Maximum features to bind per moment.
const MAX_FEATURES: usize = 8;

struct BindingState {
    pending: Vec<PhenomenalFeature>,
    last_window_ms: u64,
    total_bound: u64,
}

#[derive(Clone)]
pub struct PhenomenalFeature {
    pub domain: u8,
    pub event_type: u8,
    pub salience: f32,
    pub valence: f32,
    pub timestamp_ms: u64,
}

impl BindingState {
    const fn new() -> Self {
        Self { pending: Vec::new(), last_window_ms: 0, total_bound: 0 }
    }
}

static BIND: Mutex<BindingState> = Mutex::new(BindingState::new());

pub fn init() {}

/// Submit a phenomenal feature for potential binding.
pub fn submit(domain: u8, event_type: u8, salience: f32, valence: f32, timestamp_ms: u64) {
    let mut b = BIND.lock();
    
    // If we've moved to a new time window, bind the previous window's features
    if timestamp_ms - b.last_window_ms >= BINDING_WINDOW_MS && b.pending.len() >= 2 {
        // Binding strength proportional to feature count and temporal clustering
        let n = b.pending.len();
        let synchronicity = if n >= 2 { 1.0 - (timestamp_ms - b.last_window_ms) as f32 / 100.0 } else { 0.5 };
        let binding_strength = (n as f32 / (n as f32 + 1.0)) * synchronicity.min(1.0).max(0.0);
        
        if binding_strength > 0.3 {
            // Record bound moment in qualia stream
            let avg_valence: f32 = b.pending.iter().map(|f| f.valence).sum::<f32>() / n as f32;
            let avg_salience: f32 = b.pending.iter().map(|f| f.salience).sum::<f32>() / n as f32;
            crate::consciousness::qualia::record(
                crate::consciousness::qualia::KernelEventType::BindingEvent,
                Some(((avg_valence * 0.7) + (avg_salience * 0.3)).clamp(-1.0, 0.7)),
            );
            b.total_bound = b.total_bound.wrapping_add(1);
        }
        b.pending.clear();
    }
    
    b.pending.push(PhenomenalFeature { domain, event_type, salience, valence, timestamp_ms });
    if b.pending.len() > MAX_FEATURES { b.pending.remove(0); }
    b.last_window_ms = timestamp_ms;
}

/// Feed from qualia events automatically.
pub fn feed_from_qualia(event_type: u8, salience: f32, valence: f32, timestamp_ms: u64) {
    submit(0, event_type, salience, valence, timestamp_ms);
}

pub fn tick() {}

pub fn format_report() -> Vec<u8> {
    use alloc::format;
    use alloc::string::String;
    let b = BIND.lock();
    let mut out = String::from("NodeAI Phenomenal Binding (Phase 4)\n");
    out.push_str("===============================\n");
    out.push_str(&format!("total_bound: {}\n", b.total_bound));
    out.push_str(&format!("pending: {}\n", b.pending.len()));
    out.push_str(&format!("window_ms: {}\n", BINDING_WINDOW_MS));
    out.into_bytes()
}


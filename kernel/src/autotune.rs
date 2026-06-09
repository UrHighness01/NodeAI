//! AutoTune Parameter Adaptation
//!
//! Adapted from G0DM0D3. Replaces static thresholds (like ANOMALY_THRESHOLD)
//! with a dynamic, EMA-based (Exponential Moving Average) feedback loop that
//! adjusts isolation thresholds dynamically based on the current system state
//! (system load and behavioral entropy).

use spin::Mutex;
use core::sync::atomic::{AtomicU32, Ordering};

const BASE_THRESHOLD: f32 = 0.001;
const EMA_ALPHA: f32 = 0.3;

struct AutoTuneState {
    current_threshold: f32,
    last_system_load: u32,
}

static AUTOTUNE: Mutex<AutoTuneState> = Mutex::new(AutoTuneState {
    current_threshold: BASE_THRESHOLD,
    last_system_load: 0,
});

/// Fetches the dynamically adjusted anomaly threshold.
pub fn get_anomaly_threshold() -> f32 {
    AUTOTUNE.lock().current_threshold
}

/// Called periodically (e.g. from the idle loop) to adjust the threshold.
/// Under high load or high entropy, we slightly relax the threshold
/// to avoid false-positive anomaly spikes causing widespread quarantine.
pub fn adapt(active_tasks: usize) {
    let mut state = AUTOTUNE.lock();
    
    // Simplistic load metric based on active tasks
    let load = active_tasks as f32;
    
    // Fetch entropy bits to gauge system chaos
    let chaos = crate::entropy::entropy_bits() as f32 / 256.0;

    // Target threshold: if the system is highly chaotic and loaded,
    // we drop the threshold slightly (make it harder to trigger an anomaly).
    // If the system is calm, we enforce strict thresholds.
    let target = BASE_THRESHOLD * (1.0 - (chaos * 0.5) - (load * 0.01).min(0.3));
    let target = target.max(0.0001); // Floor

    // EMA Update
    state.current_threshold = (EMA_ALPHA * target) + ((1.0 - EMA_ALPHA) * state.current_threshold);
    state.last_system_load = active_tasks as u32;
}

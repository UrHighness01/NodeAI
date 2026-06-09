//! AutoTune Parameter Adaptation — AI meta-tuner
//!
//! Adapted from G0DM0D3. Replaces static thresholds (like ANOMALY_THRESHOLD)
//! with a dynamic, EMA-based feedback loop that adjusts isolation thresholds.
//!
//! Extended in Round 34 with cross-modal coupling and information-bottleneck
//! retention as additional signal sources.  When the scheduler coherence
//! strongly predicts memory (high coupling), the system tightens thresholds
//! preemptively.  When anomaly is the dominant retained signal (high retention),
//! the system relaxes thresholds to avoid false-positive storms.

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
///
/// AI Meta-Tuner (R34): cross-modal coupling and info_bottleneck retention
/// provide additional signals:
///   - High sched→mem coupling → tighten thresholds preemptively (anticipate memory pressure)
///   - High anomaly retention → relax thresholds (anomaly is the dominant signal, avoid spam)
///   - High memory coupling → tighten (memory predicts other domain changes)
pub fn adapt(active_tasks: usize) {
    let mut state = AUTOTUNE.lock();

    let load = active_tasks as f32;
    let chaos = crate::entropy::entropy_bits() as f32 / 256.0;

    // ── AI Meta-Tuner: cross-modal coupling signals ──────────────────────────
    let sched_to_mem = crate::cross_modal::coupling(
        crate::cross_modal::Domain::Scheduler,
        crate::cross_modal::Domain::Memory,
        2,
    );
    let mem_to_anom = crate::cross_modal::coupling(
        crate::cross_modal::Domain::Memory,
        crate::cross_modal::Domain::Anomaly,
        1,
    );

    // ── Information Bottleneck retention signal ──────────────────────────────
    let ret = crate::info_bottleneck::all_retention();
    // anomaly_retention: fraction of information retained for anomaly domain
    let anomaly_retention = ret[crate::cross_modal::Domain::Anomaly as usize];
    let memory_retention = ret[crate::cross_modal::Domain::Memory as usize];

    // ── Composite modulation ─────────────────────────────────────────────────
    // Tighten threshold (lower = stricter) when sched→mem coupling is high
    // (predicts memory pressure) or memory retention is high.
    // Relax threshold (higher = looser) when anomaly retention is high
    // (anomaly is dominant; avoid false positives).
    let coupling_mod = (sched_to_mem * 0.3 - mem_to_anom * 0.2).max(-0.3).min(0.3);
    let retention_mod = (anomaly_retention * 0.4 - memory_retention * 0.2).max(-0.3).min(0.3);
    let total_mod = coupling_mod + retention_mod;

    let target = BASE_THRESHOLD * (1.0 - (chaos * 0.5) - (load * 0.01).min(0.3) - total_mod);
    let target = target.max(0.00005).min(0.01); // sane bounds

    state.current_threshold = (EMA_ALPHA * target) + ((1.0 - EMA_ALPHA) * state.current_threshold);
    state.last_system_load = active_tasks as u32;
}

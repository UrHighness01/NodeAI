//! Live kernel tuning table — AI-adjustable parameters validated by the safety engine.
//!
//! All parameters are stored as atomics. The AI engine proposes changes;
//! the safety engine clamps them to safe ranges before application.
//! Changes take effect immediately with no reboot required.
//!
//! Parameters are readable from userspace at /ai/tunables and writable
//! via sys_ai_query(QUERY_SET_TUNABLE, ...).

use core::sync::atomic::{AtomicI32, AtomicU64, Ordering};

// ── Scheduler tunables ────────────────────────────────────────────────────────

/// Base time quantum in ms (range: 1–100).
pub static QUANTUM_MS:       AtomicU64 = AtomicU64::new(10);
/// Latency-intent priority bias (range: -20–0).
pub static LATENCY_BIAS:     AtomicI32 = AtomicI32::new(-15);
/// Batch-intent priority bias (range: 0–20).
pub static BATCH_BIAS:       AtomicI32 = AtomicI32::new(10);
/// AI priority adjustment cap per tick (range: 1–20).
pub static AI_NICE_CAP:      AtomicI32 = AtomicI32::new(5);

// ── Memory tunables ───────────────────────────────────────────────────────────

/// Demand-paging stack guard pages (0 = off, N = warn on N pages from stack top).
pub static STACK_GUARD_PAGES: AtomicU64 = AtomicU64::new(2);

// ── Anomaly detector tunables ─────────────────────────────────────────────────

/// Anomaly alert streak threshold (default 3: alert after 3 rare transitions).
pub static ANOMALY_STREAK:   AtomicU64 = AtomicU64::new(3);
/// Anomaly score decay per normal transition (× 1000 for fixed-point).
pub static ANOMALY_DECAY:    AtomicU64 = AtomicU64::new(20); // 0.020

// ── Public API ────────────────────────────────────────────────────────────────

/// Apply an AI-proposed tunable change.
/// Returns Ok(new_value) if accepted, Err(reason) if rejected by safety bounds.
pub fn apply(name: &str, value: i64) -> Result<i64, &'static str> {
    match name {
        "quantum_ms" => {
            let v = value.clamp(1, 100) as u64;
            QUANTUM_MS.store(v, Ordering::Relaxed);
            crate::scheduler::set_quantum_ms(v);
            Ok(v as i64)
        }
        "latency_bias" => {
            let v = value.clamp(-20, 0) as i32;
            LATENCY_BIAS.store(v, Ordering::Relaxed);
            Ok(v as i64)
        }
        "batch_bias" => {
            let v = value.clamp(0, 20) as i32;
            BATCH_BIAS.store(v, Ordering::Relaxed);
            Ok(v as i64)
        }
        "ai_nice_cap" => {
            let v = value.clamp(1, 20) as i32;
            AI_NICE_CAP.store(v, Ordering::Relaxed);
            Ok(v as i64)
        }
        "anomaly_streak" => {
            let v = value.clamp(1, 100) as u64;
            ANOMALY_STREAK.store(v, Ordering::Relaxed);
            Ok(v as i64)
        }
        _ => Err("unknown tunable"),
    }
}

/// Get the current value of a tunable.
pub fn get(name: &str) -> i64 {
    match name {
        "quantum_ms" => QUANTUM_MS.load(Ordering::Relaxed) as i64,
        "latency_bias" => LATENCY_BIAS.load(Ordering::Relaxed) as i64,
        "batch_bias" => BATCH_BIAS.load(Ordering::Relaxed) as i64,
        "ai_nice_cap" => AI_NICE_CAP.load(Ordering::Relaxed) as i64,
        "anomaly_streak" => ANOMALY_STREAK.load(Ordering::Relaxed) as i64,
        _ => 0,
    }
}

/// Format the current tunable table for /ai/tunables.
pub fn format_table() -> alloc::vec::Vec<u8> {
    alloc::format!(
        "# NodeAI live tunables (AI-adjustable, safety-clamped)\n\
         quantum_ms       = {}\n\
         latency_bias     = {}\n\
         batch_bias       = {}\n\
         ai_nice_cap      = {}\n\
         stack_guard_pages= {}\n\
         anomaly_streak   = {}\n\
         anomaly_decay    = {:.3}\n",
        QUANTUM_MS.load(Ordering::Relaxed),
        LATENCY_BIAS.load(Ordering::Relaxed),
        BATCH_BIAS.load(Ordering::Relaxed),
        AI_NICE_CAP.load(Ordering::Relaxed),
        STACK_GUARD_PAGES.load(Ordering::Relaxed),
        ANOMALY_STREAK.load(Ordering::Relaxed),
        ANOMALY_DECAY.load(Ordering::Relaxed) as f32 / 1000.0,
    ).into_bytes()
}

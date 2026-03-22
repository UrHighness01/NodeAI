//! Predictive Hibernation — Phase 29.
//!
//! Tracks user activity patterns and predicts future idle windows so the
//! kernel can:
//!   - Pre-warn subsystems before hibernation so they can flush cleanly.
//!   - Pre-warm caches and device state just before the predicted wake time.
//!   - Avoid waking the user with spurious activity when the system is idle.
//!
//! The predictor uses a simple usage-frequency grid bucketed by:
//!   - Hour of day (0-23) × Weekday (0-6)  →  168-bucket table
//!
//! Each bucket stores: `(activity_seconds, sample_count)`.
//!
//! Prediction: look at the current time-bucket; if activity_seconds/sample_count
//! falls below `IDLE_THRESHOLD_SECS`, recommend hibernation.

use alloc::{vec::Vec, string::String, format};
use alloc::borrow::ToOwned;
use spin::Mutex;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Number of time buckets (24 hours × 7 days).
const BUCKETS: usize = 24 * 7;

/// Mean activity below this (seconds per sample) → predict idle.
const IDLE_THRESHOLD_SECS: u64 = 30;

/// How often the activity ticker fires (ms).
const TICK_MS: u64 = 60_000; // 1 minute

// ── Data model ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Default)]
struct Bucket {
    activity_secs: u64,
    samples:       u64,
}

struct HibState {
    buckets:     [Bucket; BUCKETS],
    /// Unix timestamp (seconds) of next predicted wake, 0 = unknown.
    next_wake:   u64,
    /// Hibernate is armed when this is true.
    armed:       bool,
}

// Safety: only Sync/Send because we guard behind Mutex.
unsafe impl Send for HibState {}

static HIB: Mutex<HibState> = Mutex::new(HibState {
    buckets:   [Bucket { activity_secs: 0, samples: 0 }; BUCKETS],
    next_wake: 0,
    armed:     false,
});

static ENABLED: AtomicBool = AtomicBool::new(false);
/// Monotonic ms at which next tick is due.
static NEXT_TICK_MS: AtomicU64 = AtomicU64::new(0);

// ── Init ──────────────────────────────────────────────────────────────────────

pub fn init() {
    NEXT_TICK_MS.store(crate::scheduler::uptime_ms() + TICK_MS, Ordering::Relaxed);
    // Try to restore saved model from disk.
    load_model();
    ENABLED.store(true, Ordering::Relaxed);
    crate::klog!(INFO, "predictive_hibernate: predictor ready");
}

// ── Tick ──────────────────────────────────────────────────────────────────────

/// Called from the scheduler or idle loop; records activity and checks if
/// it is a good time to hibernate.
pub fn tick() {
    if !ENABLED.load(Ordering::Relaxed) { return; }
    let now = crate::scheduler::uptime_ms();
    if now < NEXT_TICK_MS.load(Ordering::Relaxed) { return; }
    NEXT_TICK_MS.store(now + TICK_MS, Ordering::Relaxed);

    let (hour, dow) = current_hour_dow();
    let bucket_idx  = (dow as usize) * 24 + (hour as usize);

    let active = is_system_active();
    let mut state = HIB.lock();
    let b = &mut state.buckets[bucket_idx];
    b.samples += 1;
    if active {
        b.activity_secs += 60; // 1 minute of activity this tick
    }
    let mean = if b.samples == 0 { 60 } else { b.activity_secs / b.samples };
    let should_hib = mean < IDLE_THRESHOLD_SECS && !active;
    if should_hib && !state.armed {
        state.armed = true;
        drop(state);
        on_predict_idle(hour, dow);
    } else if !should_hib {
        state.armed = false;
    }
}

// ── Record activity ───────────────────────────────────────────────────────────

/// Call this whenever user activity is detected (key / mouse / network IO).
pub fn record_activity() {
    if !ENABLED.load(Ordering::Relaxed) { return; }
    let (hour, dow) = current_hour_dow();
    let idx = (dow as usize) * 24 + (hour as usize);
    let mut state = HIB.lock();
    let b = &mut state.buckets[idx];
    b.activity_secs = b.activity_secs.saturating_add(5); // 5 s of activity
    state.armed = false;
}

// ── Pre-wake ──────────────────────────────────────────────────────────────────

/// Called when waking from sleep/hibernate — pre-warm subsystems.
pub fn on_wake() {
    crate::klog!(INFO, "predictive_hibernate: wake — pre-warming subsystems");
    // Prime VFS page cache with recently-accessed files.
    crate::vfs::prefetch_recently_used();
    crate::ai_engine::wake_hint();
}

// ── Predictions ───────────────────────────────────────────────────────────────

/// Return true if the model predicts we will be idle for the next window.
pub fn predict_idle_next_hour() -> bool {
    let (hour, dow) = current_hour_dow();
    let next_hour   = (hour + 1) % 24;
    let idx = (dow as usize) * 24 + (next_hour as usize);
    let state = HIB.lock();
    let b = state.buckets[idx];
    if b.samples < 3 { return false; }
    (b.activity_secs / b.samples) < IDLE_THRESHOLD_SECS
}

// ── Model persistence ─────────────────────────────────────────────────────────

const MODEL_PATH: &str = "/var/lib/predictive_hib.bin";

fn save_model() {
    let state = HIB.lock();
    let mut buf: Vec<u8> = Vec::with_capacity(BUCKETS * 16 + 4);
    buf.extend_from_slice(b"PHIB");
    for b in &state.buckets {
        buf.extend_from_slice(&b.activity_secs.to_le_bytes());
        buf.extend_from_slice(&b.samples.to_le_bytes());
    }
    let _ = crate::vfs::write_file(MODEL_PATH, &buf);
}

fn load_model() {
    let data = match crate::vfs::read_file(MODEL_PATH) {
        Ok(d) => d,
        Err(_) => return,
    };
    if data.len() < 4 || &data[..4] != b"PHIB" { return; }
    let mut state = HIB.lock();
    let mut off = 4usize;
    for i in 0..BUCKETS {
        if off + 16 > data.len() { break; }
        let act  = u64::from_le_bytes(data[off..off+8].try_into().unwrap_or([0;8]));
        let samp = u64::from_le_bytes(data[off+8..off+16].try_into().unwrap_or([0;8]));
        state.buckets[i] = Bucket { activity_secs: act, samples: samp };
        off += 16;
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn is_system_active() -> bool {
    // Check for running user processes (non-kernel threads).
    crate::scheduler::user_process_count() > 0
}

fn current_hour_dow() -> (u8, u8) {
    // Use RTC or uptime modulo for an approximate hour.
    let secs = crate::scheduler::uptime_ms() / 1000;
    let hour = ((secs / 3600) % 24) as u8;
    let dow  = ((secs / 86400) % 7) as u8;
    (hour, dow)
}

fn on_predict_idle(hour: u8, dow: u8) {
    crate::klog!(INFO, "predictive_hibernate: predicting idle at h={} dow={}", hour, dow);
    save_model();
    crate::power::prepare_hibernate();
}

// ── Public query API ──────────────────────────────────────────────────────────

pub fn stats() -> String {
    let state = HIB.lock();
    let (hour, dow) = current_hour_dow();
    let idx   = (dow as usize) * 24 + (hour as usize);
    let b     = state.buckets[idx];
    let mean  = if b.samples == 0 { 0 } else { b.activity_secs / b.samples };
    format!(
        "hib: h={} dow={} mean_act={}s samples={} armed={}",
        hour, dow, mean, b.samples, state.armed
    )
}

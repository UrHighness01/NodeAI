//! Immune Covertness Budget — frequency exposure tracking & stealth management.
//!
//! Tracks how long the system has stayed on a single frequency, estimates
//! exposure (detectability) over time, and recommends frequency hops.
//! The goal is to blend with the background noise floor and avoid emitter
//! fingerprinting by adversarial spectrum monitors.
//!
//! Call tick() every 100ms alongside the main EW tick cycle.
//! Use exposure_pct() and should_hop() for template placeholders.

use alloc::string::String;
use alloc::format;
use alloc::vec::Vec;
use spin::Mutex;

/// Maximum ticks on one frequency before exposure is critical.
const MAX_TICKS_ON_FREQ: u64 = 500; // 50s at 100ms tick

/// Default tick interval for linear exposure ramp.
const TICK_EXPOSURE_DELTA: f32 = 0.002; // 0.2% per tick → 100% at 500 ticks

struct CovertState {
    /// Ticks spent on current frequency.
    ticks_on_freq: u64,
    /// Total frequency hops executed.
    total_hops: u64,
    /// Current exposure estimate (0.0–1.0).
    exposure: f32,
    /// Peak exposure this session.
    peak_exposure: f32,
    /// Low observability mode flag.
    stealth_mode: bool,
}

static STATE: Mutex<Option<CovertState>> = Mutex::new(None);

/// Initialize the covertness budget tracker.
pub fn init() {
    let mut lock = STATE.lock();
    *lock = Some(CovertState {
        ticks_on_freq: 0,
        total_hops: 0,
        exposure: 0.0,
        peak_exposure: 0.0,
        stealth_mode: false,
    });
    crate::klog!(INFO, "immune_covert: covertness budget initialized");
}

/// Tick the covertness budget — called every 100ms.
/// Increases exposure over time spent on frequency.
pub fn tick() {
    let mut lock = STATE.lock();
    let state = match &mut *lock {
        Some(s) => s,
        None => return,
    };

    state.ticks_on_freq = state.ticks_on_freq.saturating_add(1);

    if state.stealth_mode {
        // In stealth mode, exposure rises slower
        state.exposure = (state.exposure + TICK_EXPOSURE_DELTA * 0.3).min(1.0);
    } else {
        state.exposure = (state.exposure + TICK_EXPOSURE_DELTA).min(1.0);
    }

    if state.exposure > state.peak_exposure {
        state.peak_exposure = state.exposure;
    }

    // Auto-hop if exposure is critical
    if state.exposure >= 0.95 {
        execute_hop_inner(state);
    }
}

/// Check if a frequency hop is recommended.
pub fn should_hop() -> bool {
    let lock = STATE.lock();
    match &*lock {
        Some(s) => s.exposure >= 0.7,
        None => false,
    }
}

/// Execute a frequency hop — resets exposure.
pub fn execute_hop() {
    let mut lock = STATE.lock();
    if let Some(ref mut state) = &mut *lock {
        execute_hop_inner(state);
    }
}

fn execute_hop_inner(state: &mut CovertState) {
    state.ticks_on_freq = 0;
    state.exposure = 0.0;
    state.total_hops = state.total_hops.saturating_add(1);
}

/// Get current exposure percentage (0–100).
pub fn exposure_pct() -> u8 {
    let lock = STATE.lock();
    match &*lock {
        Some(s) => (s.exposure * 100.0) as u8,
        None => 0,
    }
}

/// Get peak exposure percentage this session.
pub fn peak_exposure_pct() -> u8 {
    let lock = STATE.lock();
    match &*lock {
        Some(s) => (s.peak_exposure * 100.0) as u8,
        None => 0,
    }
}

/// Get total hops this session.
pub fn total_hops() -> u64 {
    let lock = STATE.lock();
    match &*lock {
        Some(s) => s.total_hops,
        None => 0,
    }
}

/// Get ticks on current frequency.
pub fn ticks_on_freq() -> u64 {
    let lock = STATE.lock();
    match &*lock {
        Some(s) => s.ticks_on_freq,
        None => 0,
    }
}

/// Enable or disable stealth mode.
pub fn set_stealth(enabled: bool) {
    let mut lock = STATE.lock();
    if let Some(ref mut state) = &mut *lock {
        state.stealth_mode = enabled;
    }
}

/// Report for /proc/immune_covert.
pub fn format_report() -> Vec<u8> {
    let lock = STATE.lock();
    match &*lock {
        Some(s) => {
            let status = if s.stealth_mode { "STEALTH (low observability)" }
                         else if s.exposure > 0.8 { "CRITICAL — hop recommended" }
                         else if s.exposure > 0.5 { "ELEVATED — monitoring" }
                         else { "NOMINAL" };

            let hop_icon = if s.exposure >= 0.7 { "⚠" } else { "✓" };

            format!(
                "Covertness Budget\n\
                 =================\n\
                 status:        {}\n\
                 exposure:      {}% (peak: {}%)\n\
                 ticks_on_freq: {}\n\
                 hops_executed: {}\n\
                 stealth_mode:  {}\n\
                 next_hop_at:   {} ticks\n",
                status,
                (s.exposure * 100.0) as u8,
                (s.peak_exposure * 100.0) as u8,
                s.ticks_on_freq,
                s.total_hops,
                s.stealth_mode,
                if s.exposure >= 0.7 {
                    0
                } else {
                    ((0.7 - s.exposure) / TICK_EXPOSURE_DELTA) as u64
                },
            ).into_bytes()
        }
        None => format!("Covertness Budget\nNot initialized\n").into_bytes(),
    }
}

/// Human-readable covertness summary (for templates).
pub fn covert_summary() -> String {
    let lock = STATE.lock();
    match &*lock {
        Some(s) => {
            if s.exposure > 0.9 {
                format!("Critical exposure at {}% — need to hop now!", (s.exposure * 100.0) as u8)
            } else if s.exposure > 0.7 {
                format!("Exposure at {}% — frequency hop recommended", (s.exposure * 100.0) as u8)
            } else if s.exposure > 0.4 {
                format!("Exposure rising — currently {}%", (s.exposure * 100.0) as u8)
            } else {
                format!("Covert — exposure only {}%", (s.exposure * 100.0) as u8)
            }
        }
        None => String::from("Covertness system offline"),
    }
}

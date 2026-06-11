//! APIC Timer Calibration & Diagnostics — timer frequency monitoring.
//!
//! Tracks APIC timer drift over time by comparing expected tick count
//! against actual uptime. Provides calibration stability metrics and
//! tick delivery statistics.
//!
//! Call tick() every 100ms. /proc/timer_calib for diagnostics.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};
use spin::Mutex;

/// Whether calibration monitoring is active.
static CALIB_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Tracked calibration state.
struct CalibState {
    /// Total ticks expected (based on timer frequency).
    expected_ticks: u64,
    /// Actual ticks counted from interrupt handler increments.
    actual_ticks: u64,
    /// Measured APIC timer frequency in Hz.
    measured_freq_hz: u64,
    /// Nominal frequency target (100 Hz = 10ms ticks).
    nominal_freq_hz: u64,
    /// Maximum observed drift in milliseconds.
    max_drift_ms: u64,
    /// Total corrections applied.
    corrections: u64,
    /// Last stable reading time.
    last_calib_tick: u64,
    /// Whether the timer is stable (drift < 5%).
    stable: bool,
}

static STATE: Mutex<Option<CalibState>> = Mutex::new(None);

/// Initialize the timer calibration monitor.
pub fn init() {
    let state = CalibState {
        expected_ticks: 0,
        actual_ticks: 0,
        measured_freq_hz: 100,
        nominal_freq_hz: 100,
        max_drift_ms: 0,
        corrections: 0,
        last_calib_tick: 0,
        stable: true,
    };

    let mut lock = STATE.lock();
    *lock = Some(state);
    CALIB_ACTIVE.store(true, Ordering::Release);
    crate::klog!(INFO, "timer_calib: APIC timer calibration monitor initialized");
}

/// Record a timer tick — called from the APIC timer interrupt handler.
pub fn record_tick() {
    let mut lock = STATE.lock();
    if let Some(ref mut s) = &mut *lock {
        s.actual_ticks = s.actual_ticks.saturating_add(1);
    }
}

/// Tick the calibration monitor — called every 100ms.
/// Compares expected vs actual ticks to detect drift.
pub fn tick() {
    if !CALIB_ACTIVE.load(Ordering::Acquire) { return; }

    let mut lock = STATE.lock();
    let state = match &mut *lock {
        Some(s) => s,
        None => return,
    };

    let uptime_ms = crate::scheduler::uptime_ms();
    let expected = uptime_ms / 10; // At 100 Hz, 1 tick per 10ms

    state.expected_ticks = expected;

    if state.actual_ticks > 0 && expected > 0 {
        let drift = if state.actual_ticks > expected {
            state.actual_ticks - expected
        } else {
            expected - state.actual_ticks
        };

        if drift > state.max_drift_ms {
            state.max_drift_ms = drift;
        }

        // Drift > 5% of expected ticks = unstable
        let drift_pct = (drift as f64) / (expected as f64).max(1.0);
        state.stable = drift_pct < 0.05;

        // If drift > 10%, correct the frequency estimate
        if drift_pct > 0.10 && expected > 100 {
            let correction = (state.actual_ticks as f64 / expected as f64 * 100.0) as u64;
            state.measured_freq_hz = correction.max(50).min(200);
            state.corrections = state.corrections.saturating_add(1);
        }
    }

    state.last_calib_tick = uptime_ms / 100;
}

/// Get current APIC timer frequency.
pub fn measured_freq() -> u64 {
    let lock = STATE.lock();
    match &*lock {
        Some(s) => s.measured_freq_hz,
        None => 100,
    }
}

/// Whether the APIC timer is stable.
pub fn is_stable() -> bool {
    let lock = STATE.lock();
    match &*lock {
        Some(s) => s.stable,
        None => true,
    }
}

/// Get maximum observed drift in ticks.
pub fn max_drift() -> u64 {
    let lock = STATE.lock();
    match &*lock {
        Some(s) => s.max_drift_ms,
        None => 0,
    }
}

/// Format /proc/timer_calib report.
pub fn format_report() -> Vec<u8> {
    let active = CALIB_ACTIVE.load(Ordering::Acquire);
    if !active {
        return format!("APIC Timer Calibration\nNot initialized\n").into_bytes();
    }
    let lock = STATE.lock();
    match &*lock {
        Some(s) => {
            let stability = if s.stable { "STABLE" } else { "DRIFT DETECTED" };
            let drift_pct = if s.expected_ticks > 0 {
                let d = if s.actual_ticks > s.expected_ticks {
                    s.actual_ticks - s.expected_ticks
                } else {
                    s.expected_ticks - s.actual_ticks
                };
                (d as f64) / (s.expected_ticks as f64).max(1.0) * 100.0
            } else { 0.0 };

            format!(
                "APIC Timer Calibration & Diagnostics\n\
                 =====================================\n\
                 status:          {}\n\
                 nominal_freq:    {} Hz ({}ms period)\n\
                 measured_freq:   {} Hz\n\
                 expected_ticks:  {}\n\
                 actual_ticks:    {}\n\
                 drift_pct:       {:.2}%\n\
                 max_drift:       {} ticks\n\
                 corrections:     {}\n\
                 last_check:      tick #{}\n\
                 \n\
                 Drift analysis:\n\
                 - Under 5%: stable\n\
                 - 5-10%:  degraded\n\
                 - Over 10%: unstable (auto-correction triggered)\n",
                stability,
                s.nominal_freq_hz,
                1000 / s.nominal_freq_hz.max(1),
                s.measured_freq_hz,
                s.expected_ticks,
                s.actual_ticks,
                drift_pct,
                s.max_drift_ms,
                s.corrections,
                s.last_calib_tick,
            ).into_bytes()
        }
        None => format!("APIC Timer Calibration\nUninitialized\n").into_bytes(),
    }
}

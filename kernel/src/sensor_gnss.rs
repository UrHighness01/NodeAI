//! Sensor GNSS Integrity — Receiver Autonomous Integrity Monitoring (RAIM).
//!
//! Implements a lightweight RAIM algorithm that cross-references simulated
//! satellite signals to detect spoofing, jamming, or signal degradation.
//!
//! Architecture:
//!   4+ visible satellites, each providing a pseudorange measurement.
//!   RAIM computes a position solution, then checks residuals against
//!   a chi-squared threshold. If the test statistic exceeds the threshold,
//!   a spoofing/jamming alert is raised.
//!
//! Integration:
//!   - init() at boot
//!   - tick() in idle loop every 100ms
//!   - /proc/sensor_gnss for status and satellite tracking

use alloc::vec::Vec;
use alloc::format;
use alloc::string::String;
use core::sync::atomic::{AtomicBool, Ordering};
use spin::Mutex;

/// Whether the RAIM module is initialized.
static GNSS_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Number of satellites in the simulated constellation.
const N_SATELLITES: usize = 8;

/// Chi-squared threshold for RAIM fault detection (95% confidence, 4 DOF).
const RAIM_THRESHOLD: f32 = 9.49;

/// A simulated satellite.
#[derive(Debug, Clone, Copy)]
struct Satellite {
    /// Satellite ID (PRN number).
    id: u8,
    /// Elevation angle in degrees (0-90).
    elevation: f32,
    /// Azimuth in degrees (0-360).
    azimuth: f32,
    /// Signal-to-noise ratio (dB).
    snr: f32,
    /// Whether this satellite is healthy.
    healthy: bool,
    /// Pseudorange residual (meters) — simulated by RAIM.
    residual: f32,
}

/// RAIM state.
struct GnssState {
    /// Visible satellites.
    sats: [Satellite; N_SATELLITES],
    /// Position estimate (X, Y, Z in meters, relative).
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    /// RAIM test statistic (sum of squared residuals).
    test_statistic: f32,
    /// Number of satellites used in last fix.
    sats_used: usize,
    /// Total position solutions computed.
    fixes_attempted: u64,
    /// Number of RAIM alarms raised.
    alarms_raised: u64,
    /// Current integrity status.
    integrity_ok: bool,
    /// Horizontal dilution of precision.
    hdop: f32,
    /// Estimated position error (meters).
    estimated_error: f32,
}

static STATE: Mutex<Option<GnssState>> = Mutex::new(None);

/// Initialize the GNSS RAIM module.
pub fn init() {
    let mut state = GnssState {
        sats: [
            Satellite { id: 1, elevation: 45.0, azimuth: 30.0, snr: 42.0, healthy: true, residual: 0.0 },
            Satellite { id: 2, elevation: 60.0, azimuth: 120.0, snr: 38.0, healthy: true, residual: 0.0 },
            Satellite { id: 3, elevation: 30.0, azimuth: 210.0, snr: 35.0, healthy: true, residual: 0.0 },
            Satellite { id: 4, elevation: 50.0, azimuth: 300.0, snr: 40.0, healthy: true, residual: 0.0 },
            Satellite { id: 5, elevation: 25.0, azimuth: 80.0, snr: 30.0, healthy: true, residual: 0.0 },
            Satellite { id: 6, elevation: 70.0, azimuth: 180.0, snr: 36.0, healthy: true, residual: 0.0 },
            Satellite { id: 7, elevation: 15.0, azimuth: 350.0, snr: 28.0, healthy: true, residual: 0.0 },
            Satellite { id: 8, elevation: 55.0, azimuth: 90.0, snr: 44.0, healthy: true, residual: 0.0 },
        ],
        pos_x: 0.0,
        pos_y: 0.0,
        pos_z: 0.0,
        test_statistic: 0.0,
        sats_used: 4,
        fixes_attempted: 0,
        alarms_raised: 0,
        integrity_ok: true,
        hdop: 1.2,
        estimated_error: 0.0,
    };

    let mut lock = STATE.lock();
    *lock = Some(state);
    GNSS_ACTIVE.store(true, Ordering::Release);
    crate::klog!(INFO, "sensor_gnss: RAIM initialized ({} satellites, threshold={})", N_SATELLITES, RAIM_THRESHOLD);
}

/// Tick the GNSS RAIM module — called every 100ms.
/// Computes position and runs integrity check.
pub fn tick() {
    if !GNSS_ACTIVE.load(Ordering::Acquire) { return; }

    let mut lock = STATE.lock();
    let state = match &mut *lock {
        Some(s) => s,
        None => return,
    };

    state.fixes_attempted = state.fixes_attempted.saturating_add(1);

    // Simulate position drift and satellite measurements
    let uptime = crate::scheduler::uptime_ms();
    let phase = (uptime as f32) / 1000.0;

    // Position slowly drifts (simulating movement)
    state.pos_x = libm::sinf(phase * 0.5) * 10.0;
    state.pos_y = libm::cosf(phase * 0.3) * 8.0;
    state.pos_z = libm::sinf(phase * 0.2) * 2.0;

    // Simulate satellite SNR variations and compute residuals
    let mut sum_sq = 0.0_f32;
    let mut used_count = 0_u8;

    // Simulate occasional spoofing: every ~100 ticks, inject error
    let spoof_active = state.fixes_attempted % 100 == 0 && state.fixes_attempted > 0;

    for sat in state.sats.iter_mut() {
        // Vary SNR with time
        let noise = libm::sinf(state.fixes_attempted as f32 * 0.1 + sat.id as f32 * 7.0) * 5.0;
        sat.snr = (sat.snr + noise * 0.1).clamp(10.0, 55.0);
        sat.healthy = sat.snr > 15.0;

        // Compute residual (simulated pseudorange error)
        let base_residual = libm::fabsf(libm::sinf(state.fixes_attempted as f32 * 0.01 + sat.id as f32)) * 2.0;
        sat.residual = if sat.healthy { base_residual } else { 50.0 };

        // Inject spoofing error
        if spoof_active && sat.id == 1 {
            sat.residual = 25.0; // Large residual = spoofed satellite
        }

        if sat.healthy {
            sum_sq += sat.residual * sat.residual;
            used_count += 1;
        }
    }

    state.sats_used = used_count as usize;
    state.test_statistic = sum_sq;

    // RAIM decision: compare test statistic against threshold
    state.integrity_ok = sum_sq < RAIM_THRESHOLD;
    if !state.integrity_ok {
        state.alarms_raised = state.alarms_raised.saturating_add(1);
    }

    // Estimate position error from residuals and geometry
    state.hdop = 1.2 + (1.0 / state.sats_used.max(1) as f32);
    state.estimated_error = libm::sqrtf(sum_sq / state.sats_used.max(1) as f32) * state.hdop;
}

/// Get current integrity status.
pub fn integrity_ok() -> bool {
    let lock = STATE.lock();
    match &*lock {
        Some(s) => s.integrity_ok,
        None => true,
    }
}

/// Get estimated position error in meters.
pub fn estimated_error() -> f32 {
    let lock = STATE.lock();
    match &*lock {
        Some(s) => s.estimated_error,
        None => 0.0,
    }
}

/// Get number of GNSS integrity alarms raised.
pub fn alarms_raised() -> u64 {
    let lock = STATE.lock();
    match &*lock {
        Some(s) => s.alarms_raised,
        None => 0,
    }
}

/// Get number of satellites tracked.
pub fn sats_tracked() -> usize {
    let lock = STATE.lock();
    match &*lock {
        Some(s) => s.sats_used,
        None => 0,
    }
}

/// Get human-readable GNSS status string.
pub fn status() -> String {
    if !GNSS_ACTIVE.load(Ordering::Acquire) {
        return String::from("GNSS RAIM: inactive");
    }
    let lock = STATE.lock();
    match &*lock {
        Some(s) => {
            let integrity = if s.integrity_ok { "OK" } else { "ALARM" };
            format!(
                "GNSS RAIM: {} ({} sats, HDOP={:.1}, error={:.1}m, alarms={})",
                integrity, s.sats_used, s.hdop, s.estimated_error, s.alarms_raised
            )
        }
        None => String::from("GNSS RAIM: uninitialized"),
    }
}

/// Format /proc/sensor_gnss report.
pub fn format_report() -> Vec<u8> {
    let active = GNSS_ACTIVE.load(Ordering::Acquire);
    if !active {
        return format!("Sensor GNSS Integrity (RAIM)\nNot initialized\n").into_bytes();
    }
    let lock = STATE.lock();
    match &*lock {
        Some(s) => {
            let integrity = if s.integrity_ok { "CLEAN ✓" } else { "ALARM ⚠" };
            let mut report = format!(
                "Sensor GNSS Integrity (RAIM)\n\
                 =============================\n\
                 integrity:    {}\n\
                 position:     ({:.1}, {:.1}, {:.1})\n\
                 sats_used:    {}\n\
                 hdop:         {:.1}\n\
                 est_error:    {:.1} m\n\
                 test_stat:    {:.2} (threshold={})\n\
                 fixes:        {}\n\
                 alarms:       {}\n\
                 \n\
                 Satellite Details:\n",
                integrity,
                s.pos_x, s.pos_y, s.pos_z,
                s.sats_used,
                s.hdop,
                s.estimated_error,
                s.test_statistic, RAIM_THRESHOLD,
                s.fixes_attempted,
                s.alarms_raised,
            );

            for (i, sat) in s.sats.iter().enumerate() {
                let health = if sat.healthy { "OK" } else { "LOW" };
                report.push_str(&format!(
                    "  PRN{:02}: el={:.0}° az={:.0}° SNR={:.0}dB {} res={:.1}m\n",
                    sat.id, sat.elevation, sat.azimuth, sat.snr, health, sat.residual,
                ));
            }

            report.into_bytes()
        }
        None => format!("Sensor GNSS Integrity (RAIM)\nUninitialized\n").into_bytes(),
    }
}

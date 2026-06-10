//! Immune Reflexes (Phase EW-4) — Active electromagnetic defense.
//!
//! Transforms the kernel from a passive sensor into an active defensive agent.
//! When the threat tracker (EW-3) detects malicious signals, the immune system
//! selects and executes countermeasures via a dual-pathway architecture:
//!
//!   Reflex arc (fast): threat > 0.9 confidence → immediate frequency hop
//!   Deliberated (slow): threat < 0.9 → logged for conscious deliberation
//!
//! Modules:
//!   frequency_agility — adaptive frequency hopping when jammed
//!   lms_cancel — LMS adaptive interference cancellation
//!   select_response — threat-driven countermeasure selection

use alloc::vec::Vec;
use alloc::vec;
use alloc::string::String;
use alloc::format;
use spin::Mutex;

/// Types of immune responses the kernel can execute.
#[derive(Debug, Clone, PartialEq)]
pub enum ImmuneResponse {
    /// Hop to a new frequency channel.
    FrequencyHop { new_freq_hz: u64, reason: &'static str },
    /// Null-steer an antenna pattern toward the jammer.
    NullSteer { direction_deg: f32, bandwidth_hz: f32 },
    /// Cancel interference using LMS adaptive filter.
    CancelInterference,
    /// Execute a self-healing action.
    Heal { subsystem: &'static str, recovery: &'static str },
    /// No action needed.
    Idle,
}

/// Immune system state.
struct ImmuneState {
    current_frequency_hz: u64,
    frequency_blacklist: [u64; 8],
    blacklist_count: usize,
    total_hops: u64,
    total_cancellations: u64,
    last_response: Option<ImmuneResponse>,
    last_response_tick: u64,
    cooldown_ticks: u32,
}

impl ImmuneState {
    fn new() -> Self {
        Self {
            current_frequency_hz: 2_400_000_000, // 2.4 GHz default
            frequency_blacklist: [0; 8],
            blacklist_count: 0,
            total_hops: 0,
            total_cancellations: 0,
            last_response: None,
            last_response_tick: 0,
            cooldown_ticks: 5, // 500ms cooldown between responses
        }
    }

    /// Add a frequency to the blacklist (jammed channels).
    fn blacklist_freq(&mut self, freq_hz: u64) {
        if self.blacklist_count < 8 {
            // Check if already blacklisted
            for &f in &self.frequency_blacklist[..self.blacklist_count] {
                if f == freq_hz { return; }
            }
            self.frequency_blacklist[self.blacklist_count] = freq_hz;
            self.blacklist_count += 1;
        }
    }

    /// Select a new frequency not on the blacklist.
    fn select_hop_freq(&self) -> u64 {
        // Available bands: 2.4 GHz ISM and 5 GHz ISM
        let candidates = [
            2_412_000_000u64, 2_437_000_000, 2_462_000_000,  // 2.4 GHz channels
            5_180_000_000, 5_240_000_000, 5_320_000_000,      // 5 GHz channels
            5_500_000_000, 5_700_000_000,
        ];
        for &freq in &candidates {
            let mut blacklisted = false;
            for &bf in &self.frequency_blacklist[..self.blacklist_count] {
                if bf == freq { blacklisted = true; break; }
            }
            if !blacklisted && freq != self.current_frequency_hz {
                return freq;
            }
        }
        // If all candidates blacklisted, pick the least recently used
        candidates[0]
    }
}

/// Global immune system state.
static IMMUNE: Mutex<Option<ImmuneState>> = Mutex::new(None);

/// Initialize the immune system.
pub fn init() {
    let mut immune = IMMUNE.lock();
    *immune = Some(ImmuneState::new());
    crate::klog!(INFO, "sensor_immune: EW immune reflexes initialized");
}

/// Select an immune response based on current threat level.
/// Called from sensor_cortex::tick() after threat assessment.
pub fn select_response(threat_level: f32, now_ms: u64) -> ImmuneResponse {
    let mut immune = IMMUNE.lock();
    let state = match &mut *immune {
        Some(ref mut s) => s,
        None => return ImmuneResponse::Idle,
    };

    // Cooldown check
    if now_ms < state.last_response_tick + (state.cooldown_ticks as u64 * 100) {
        return ImmuneResponse::Idle;
    }

    // Dual-pathway architecture:
    // Reflex arc (threat > 0.9) — immediate action, no deliberation needed
    // Deliberated (threat 0.5-0.9) — still act but with logging
    if threat_level > 0.9 {
        // Reflex: immediate frequency hop
        let new_freq = state.select_hop_freq();
        state.blacklist_freq(state.current_frequency_hz);
        state.current_frequency_hz = new_freq;
        state.total_hops += 1;
        state.last_response = Some(ImmuneResponse::FrequencyHop {
            new_freq_hz: new_freq,
            reason: "reflex — high-confidence threat detected",
        });
        state.last_response_tick = now_ms;

        // Record qualia for the immune response
        crate::consciousness::qualia::record(
            crate::consciousness::qualia::KernelEventType::FrequencyHopped,
            Some(0.0),
        );

        ImmuneResponse::FrequencyHop {
            new_freq_hz: new_freq,
            reason: "reflex — high-confidence threat detected",
        }
    } else if threat_level > 0.5 {
        // Deliberated: adaptive cancellation
        state.total_cancellations += 1;
        state.last_response = Some(ImmuneResponse::CancelInterference);
        state.last_response_tick = now_ms;

        ImmuneResponse::CancelInterference
    } else {
        ImmuneResponse::Idle
    }
}

/// LMS adaptive interference cancellation (simplified).
/// Takes a signal + reference and cancels correlated interference.
/// Returns the cleaned signal.
pub fn lms_cancel(signal: &[f32], reference: &[f32], mu: f32, filter_len: usize) -> Vec<f32> {
    let n = signal.len().min(reference.len());
    if n == 0 || filter_len == 0 {
        return signal.to_vec();
    }

    let mut weights = vec![0.0_f32; filter_len];
    let mut output = Vec::with_capacity(n);

    for i in 0..n {
        // Compute filter output
        let mut y = 0.0_f32;
        for j in 0..filter_len {
            if i >= j {
                y += weights[j] * reference[i - j];
            }
        }

        // Error = signal - filter output
        let error = signal[i] - y;

        // Update weights (LMS)
        for j in 0..filter_len {
            if i >= j {
                weights[j] += mu * error * reference[i - j];
            }
        }

        output.push(error);
    }

    output
}

/// Get immune system statistics.
pub fn stats() -> ImmuneStats {
    let immune = IMMUNE.lock();
    match &*immune {
        Some(ref s) => ImmuneStats {
            current_freq_mhz: s.current_frequency_hz / 1_000_000,
            blacklist_count: s.blacklist_count,
            total_hops: s.total_hops,
            total_cancellations: s.total_cancellations,
        },
        None => ImmuneStats::default(),
    }
}

#[derive(Debug, Clone)]
pub struct ImmuneStats {
    pub current_freq_mhz: u64,
    pub blacklist_count: usize,
    pub total_hops: u64,
    pub total_cancellations: u64,
}

impl Default for ImmuneStats {
    fn default() -> Self {
        Self {
            current_freq_mhz: 2400,
            blacklist_count: 0,
            total_hops: 0,
            total_cancellations: 0,
        }
    }
}

/// Format /proc/immune report.
pub fn format_report() -> Vec<u8> {
    let mut s = String::new();
    s.push_str("=== Immune Reflexes (EW-4) ===\n");
    let stats = stats();
    s.push_str(&format!("freq: {} MHz\n", stats.current_freq_mhz));
    s.push_str(&format!("blacklist: {} channels\n", stats.blacklist_count));
    s.push_str(&format!("hops: {}\n", stats.total_hops));
    s.push_str(&format!("cancellations: {}\n", stats.total_cancellations));

    let immune = IMMUNE.lock();
    if let Some(ref st) = *immune {
        if let Some(ref resp) = st.last_response {
            s.push_str(&format!("last_response: {:?}\n", resp));
        }
    }
    s.into_bytes()
}

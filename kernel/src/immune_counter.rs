//! Immune Countermeasures (Phase EW-4b) — advanced response selection.
//!
//! Extends the base immune reflexes (sensor_immune) with three subsystems:
//!
//!   1. CountermeasureSelector — chooses optimal response based on threat type,
//!      signal characteristics, and environment.
//!   2. CovertnessBudget — tracks exposure duration and recommends
//!      frequency changes to minimize detection.
//!   3. SelfHealTrigger — monitors subsystem health and triggers recovery
//!      actions when metrics drift outside nominal ranges.
//!
//! All subsystems are lightweight statistical trackers — no floating-point
//! heavy lifting beyond the existing LMS filter.

use alloc::vec::Vec;
use alloc::string::String;
use alloc::format;
use spin::Mutex;

// ── Threat Types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThreatType {
    Narrowband,    // Single-frequency jammer
    Wideband,      // Broad-spectrum noise
    Sweeping,      // Frequency-sweeping jammer
    Pulsed,        // Intermittent high-power bursts
    Deceptive,     // Spoofed signals imitating legitimate sources
    Unknown,
}

impl ThreatType {
    pub fn describe(&self) -> &'static str {
        match self {
            ThreatType::Narrowband => "narrowband jammer",
            ThreatType::Wideband => "wideband noise",
            ThreatType::Sweeping => "frequency sweeper",
            ThreatType::Pulsed => "pulsed interferer",
            ThreatType::Deceptive => "deceptive signal",
            ThreatType::Unknown => "unknown signal",
        }
    }
}

// ── Countermeasure Strategy ───────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Countermeasure {
    /// Hop to a clean frequency (best vs narrowband)
    FrequencyHop,
    /// Notch-filter the affected band (best vs wideband, preserves other channels)
    AdaptiveNotch,
    /// Increase transmission power temporarily (best vs noise)
    PowerBoost,
    /// Wait and listen for a clear channel (best vs sweeping)
    ListenBeforeTalk,
    /// Phase-cancel the known interferer (best vs pulsed)
    CoherentCancel,
    /// Raise anomaly threshold, ignore deceptive signals (best vs deceptive)
    ThresholdElevation,
    /// No action
    Idle,
}

impl Countermeasure {
    pub fn describe(&self) -> &'static str {
        match self {
            Countermeasure::FrequencyHop => "frequency hopping",
            Countermeasure::AdaptiveNotch => "adaptive notching",
            Countermeasure::PowerBoost => "power boost",
            Countermeasure::ListenBeforeTalk => "listen-before-talk",
            Countermeasure::CoherentCancel => "coherent cancellation",
            Countermeasure::ThresholdElevation => "threshold elevation",
            Countermeasure::Idle => "idle",
        }
    }
}

// ── Covertness Budget ─────────────────────────────────────────────────────────

struct CovertnessBudget {
    /// How many ticks since last frequency change
    ticks_on_current_freq: u64,
    /// Total frequency changes this session
    total_hops: u64,
    /// Running estimate of exposure (higher = more detectable)
    exposure_estimate: f32,
    /// Whether the system is in "low observability" mode
    low_observability: bool,
}

impl CovertnessBudget {
    fn new() -> Self {
        Self {
            ticks_on_current_freq: 0,
            total_hops: 0,
            exposure_estimate: 0.0,
            low_observability: false,
        }
    }

    /// Tick the covertness budget - called every 100ms.
    fn tick(&mut self) {
        self.ticks_on_current_freq = self.ticks_on_current_freq.saturating_add(1);
        // Exposure increases with time on frequency
        let time_factor = (self.ticks_on_current_freq as f32) / 100.0;
        self.exposure_estimate = (time_factor * 0.1).min(1.0);
    }

    /// Check if a frequency hop is recommended for covertness.
    fn should_hop(&self) -> bool {
        self.exposure_estimate > 0.7 && !self.low_observability
    }

    /// Record a frequency hop.
    fn record_hop(&mut self) {
        self.ticks_on_current_freq = 0;
        self.total_hops = self.total_hops.saturating_add(1);
        self.exposure_estimate = 0.0;
    }

    fn stats(&self) -> CovertStats {
        CovertStats {
            ticks_on_freq: self.ticks_on_current_freq,
            total_hops: self.total_hops,
            exposure_pct: (self.exposure_estimate * 100.0) as u8,
            low_obs: self.low_observability,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CovertStats {
    pub ticks_on_freq: u64,
    pub total_hops: u64,
    pub exposure_pct: u8,
    pub low_obs: bool,
}

// ── Self-Healing Triggers ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct HealAction {
    pub subsystem: &'static str,
    pub action: &'static str,
    pub priority: u8, // 0=info, 1=warning, 2=critical
}

/// Monitor thresholds for self-healing.
const THRESHOLDS: &[(&str, f32, &str, u8)] = &[
    ("anomaly",   0.7, "tighten anomaly gate",        1),
    ("memory",   50.0, "trigger AI balloon reclaim",   1),
    ("coherence", 0.3, "reset coherence state",        2),
    ("phi",       0.2, "boost causal integration",     2),
    ("threat",    0.8, "elevate immune readiness",     1),
];

// ── Global State ──────────────────────────────────────────────────────────────

struct CountermeasureState {
    counter: CovertnessBudget,
    last_threat_type: ThreatType,
    last_countermeasure: Countermeasure,
    heal_history: [HealAction; 16],
    heal_count: usize,
    total_actions: u64,
    ticks_since_last_action: u64,
}

static STATE: Mutex<Option<CountermeasureState>> = Mutex::new(None);

// ── Public API ────────────────────────────────────────────────────────────────

/// Initialize the countermeasures subsystem.
pub fn init() {
    let mut state = STATE.lock();
    *state = Some(CountermeasureState {
        counter: CovertnessBudget::new(),
        last_threat_type: ThreatType::Unknown,
        last_countermeasure: Countermeasure::Idle,
        heal_history: [HealAction {
            subsystem: "", action: "", priority: 0,
        }; 16],
        heal_count: 0,
        total_actions: 0,
        ticks_since_last_action: 0,
    });
    crate::klog!(INFO, "immune_counter: countermeasure selector + covertness + self-heal initialized");
}

/// Tick the countermeasure system — called every 100ms from idle_loop.
pub fn tick() {
    let mut lock = STATE.lock();
    let state = match &mut *lock {
        Some(s) => s,
        None => return,
    };

    state.ticks_since_last_action = state.ticks_since_last_action.saturating_add(1);
    state.counter.tick();

    // Self-healing check: every 50 ticks (~5s)
    if state.ticks_since_last_action >= 50 {
        check_self_heal(state);
        state.ticks_since_last_action = 0;
    }
}

/// Classify a threat based on spectral features from the sensor cortex.
/// In a full implementation this would use the spectrum sensor data.
/// For now, uses anomaly score to estimate threat type.
pub fn classify_threat(anomaly: f32) -> ThreatType {
    if anomaly > 0.9 {
        ThreatType::Wideband
    } else if anomaly > 0.7 {
        ThreatType::Sweeping
    } else if anomaly > 0.5 {
        ThreatType::Narrowband
    } else if anomaly > 0.3 {
        ThreatType::Pulsed
    } else {
        ThreatType::Unknown
    }
}

/// Select the best countermeasure for a given threat type.
pub fn select_countermeasure(threat: ThreatType) -> Countermeasure {
    match threat {
        ThreatType::Narrowband => Countermeasure::FrequencyHop,
        ThreatType::Wideband => Countermeasure::AdaptiveNotch,
        ThreatType::Sweeping => Countermeasure::ListenBeforeTalk,
        ThreatType::Pulsed => Countermeasure::CoherentCancel,
        ThreatType::Deceptive => Countermeasure::ThresholdElevation,
        ThreatType::Unknown => Countermeasure::Idle,
    }
}

/// Execute a countermeasure and record it.
pub fn execute(threat: ThreatType, cm: Countermeasure) {
    let mut lock = STATE.lock();
    let state = match &mut *lock {
        Some(s) => s,
        None => return,
    };
    state.last_threat_type = threat;
    state.last_countermeasure = cm;
    state.total_actions = state.total_actions.saturating_add(1);

    if cm == Countermeasure::FrequencyHop || cm == Countermeasure::AdaptiveNotch {
        state.counter.record_hop();
    }

    // Record qualia for consciousness awareness
    // Only if we have a real threat (not idle)
    if cm != Countermeasure::Idle {
        crate::consciousness::qualia::record(
            crate::consciousness::qualia::KernelEventType::FrequencyHopped,
            Some(0.0),
        );
    }
}

/// Check whether a covertness hop is recommended.
pub fn should_hop() -> bool {
    let lock = STATE.lock();
    match &*lock {
        Some(s) => s.counter.should_hop(),
        None => false,
    }
}

/// Check all monitored subsystems and trigger heals if needed.
fn check_self_heal(state: &mut CountermeasureState) {
    let anomaly = crate::anomaly::global_score();
    let mem_free = crate::memory::free_mb();
    let coherence = crate::consciousness::self_model::snapshot()
        .map(|s| s.coherence).unwrap_or(0.5);
    let phi = crate::consciousness::phi::current_phi();
    let threat_lvl = crate::sensor_threat::threat_level();

    let metrics = [anomaly, 440.0 - mem_free as f32, coherence, phi, threat_lvl];

    for (i, &(subsystem, threshold, action, priority)) in THRESHOLDS.iter().enumerate() {
        let needs_heal = match subsystem {
            "anomaly" => metrics[i] > threshold,
            "memory"  => metrics[i] > threshold,
            "coherence" => metrics[i] < threshold,
            "phi"     => metrics[i] < threshold,
            "threat"  => metrics[i] > threshold,
            _ => false,
        };
        if needs_heal && state.heal_count < 16 {
            state.heal_history[state.heal_count] = HealAction {
                subsystem,
                action,
                priority,
            };
            state.heal_count += 1;
        }
    }
}

/// Get a summary of the current countermeasure status.
pub fn status_summary() -> String {
    let lock = STATE.lock();
    match &*lock {
        Some(s) => {
            let last_cm = s.last_countermeasure.describe();
            let last_tt = s.last_threat_type.describe();
            let covert = s.counter.stats();
            format!(
                "last threat: {}, response: {}, actions: {}, exposure: {}%",
                last_tt, last_cm, s.total_actions, covert.exposure_pct
            )
        }
        None => "not initialized".into(),
    }
}

/// Format /proc/countermeasures report.
pub fn format_report() -> Vec<u8> {
    let lock = STATE.lock();
    let state = match &*lock {
        Some(s) => s,
        None => return b"immune_counter: not initialized\n".to_vec(),
    };
    let covert = state.counter.stats();
    let mut s = format!(
        "Immune Countermeasures (EW-4b)\n\
         =============================\n\
         last threat type:     {}\n\
         last countermeasure:  {}\n\
         total actions:        {}\n\
         \n\
         Covertness Budget:\n\
           ticks on freq:     {}\n\
           total hops:        {}\n\
           exposure:          {}%\n\
         \n\
         Self-Heal History (last {}):\n",
        state.last_threat_type.describe(),
        state.last_countermeasure.describe(),
        state.total_actions,
        covert.ticks_on_freq,
        covert.total_hops,
        covert.exposure_pct,
        state.heal_count,
    );
    for i in 0..state.heal_count {
        let h = &state.heal_history[i];
        let prio = if h.priority == 2 { "CRITICAL" } else if h.priority == 1 { "WARN" } else { "INFO" };
        s.push_str(&format!("  [{}] {}: {} ({})\n", prio, h.subsystem, h.action, h.priority));
    }
    s.push_str(&format!(
        "\nCountermeasure Strategies:\n\
           narrowband → frequency hop\n\
           wideband   → adaptive notch\n\
           sweeping   → listen-before-talk\n\
           pulsed     → coherent cancellation\n\
           deceptive  → threshold elevation\n"
    ));
    s.into_bytes()
}

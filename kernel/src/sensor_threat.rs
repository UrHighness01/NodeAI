//! Threat Detection (Phase EW-3) — CFAR + Simplified JPDA Tracker.
//!
//! Transforms raw spectrum sensing data into actionable threat intelligence.
//! CFAR detects targets in FFT magnitude data at a constant false-alarm rate.
//! JPDA associates measurements to tracks and maintains persistent threat state.
//!
//! Each tracked threat becomes a persisent object in the kernel's global workspace
//! and generates qualia (ThreatTrackBorn, ThreatTrackLost) for the consciousness stream.

use alloc::vec::Vec;
use alloc::vec;
use alloc::string::String;
use alloc::format;
use spin::Mutex;

/// A single tracked threat in the EM environment.
#[derive(Debug, Clone)]
pub struct ThreatState {
    /// Normalized frequency bin of the threat (0..1).
    pub frequency: f32,
    /// Signal power in dBm.
    pub power_dbm: f32,
    /// Confidence 0..1 (track maturity).
    pub confidence: f32,
    /// Classification hint.
    pub classification: ThreatClass,
    /// Track age in ticks.
    pub age_ticks: u32,
    /// Number of times this threat was detected.
    pub detection_count: u32,
    /// Tick when first seen.
    pub first_seen_tick: u64,
    /// Tick when last seen.
    pub last_seen_tick: u64,
    /// Is this track currently alive?
    pub alive: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ThreatClass {
    Unknown,
    Noise,
    Narrowband,   // CW / narrow signal
    Wideband,     // Spread spectrum / wide modulation
    Sweeping,     // Frequency-agile / sweeping
    Pulsed,       // Radar-like pulsed emission
    Continuous,   // Constant carrier
}

impl ThreatClass {
    pub fn name(&self) -> &'static str {
        match self {
            ThreatClass::Unknown => "unknown",
            ThreatClass::Noise => "noise",
            ThreatClass::Narrowband => "narrowband",
            ThreatClass::Wideband => "wideband",
            ThreatClass::Sweeping => "sweeping",
            ThreatClass::Pulsed => "pulsed",
            ThreatClass::Continuous => "continuous",
        }
    }
}

/// Configuration for the CFAR detector.
pub struct CfarConfig {
    pub guard_cells: usize,
    pub reference_cells: usize,
    pub pfa: f32,           // Probability of false alarm
    pub min_power: f32,     // Minimum signal power to consider
}

impl Default for CfarConfig {
    fn default() -> Self {
        Self {
            guard_cells: 2,
            reference_cells: 8,
            pfa: 0.01,      // 1% false alarm rate
            min_power: -50.0, // dBm
        }
    }
}

/// CFAR detector — detects targets in FFT magnitude data.
/// Returns indices (frequency bins) where targets are detected.
pub fn cfar_detect(magnitudes: &[f32], config: &CfarConfig) -> Vec<usize> {
    let mut detections = Vec::new();
    let n = magnitudes.len();
    if n < config.guard_cells * 2 + config.reference_cells * 2 + 1 {
        return detections;
    }

    // Compute the CFAR threshold factor from PFA
    // For a cell-averaging CFAR with N reference cells:
    //   threshold_factor = N * (PFA^(-1/N) - 1)
    let n_ref = (2 * config.reference_cells) as f32;
    let alpha = n_ref * (libm::powf(config.pfa, -1.0 / n_ref) - 1.0);

    for i in config.guard_cells + config.reference_cells..n - config.guard_cells - config.reference_cells {
        // Sum the reference cells (left and right of guard window)
        let mut sum_power = 0.0_f64;
        let left_start = i - config.guard_cells - config.reference_cells;
        let left_end = i - config.guard_cells;
        let right_start = i + config.guard_cells + 1;
        let right_end = (i + config.guard_cells + 1 + config.reference_cells).min(n);

        for j in left_start..left_end {
            sum_power += magnitudes[j] as f64;
        }
        for j in right_start..right_end {
            sum_power += magnitudes[j] as f64;
        }

        let avg_noise = sum_power / (2.0 * config.reference_cells as f64);
        let threshold = (avg_noise as f32) * alpha;

        if magnitudes[i] > threshold && magnitudes[i] > config.min_power {
            detections.push(i);
        }
    }
    detections
}

/// Simplified JPDA tracker state.
struct JpdaTracker {
    tracks: Vec<ThreatState>,
    next_id: u32,
    tick: u64,
    max_coast_ticks: u32,  // How many ticks to keep a track without detection
}

impl JpdaTracker {
    fn new() -> Self {
        Self {
            tracks: Vec::new(),
            next_id: 1,
            tick: 0,
            max_coast_ticks: 10, // ~1 second at 100ms tick
        }
    }

    /// Update tracks with new measurements (frequency bin indices from CFAR).
    fn update(&mut self, measurements: &[(usize, f32)], total_bins: usize) {
        self.tick += 1;

        // Mark all tracks as not-updated this tick
        for t in &mut self.tracks {
            t.alive = false;
        }

        // Simple nearest-neighbor association (simplified JPDA):
        // For each measurement, find the closest track by frequency.
        // Unassociated measurements become new tracks.
        let mut used_measurements = vec![false; measurements.len()];

        for track in &mut self.tracks {
            if !track.alive {
                continue; // Skip dead tracks
            }

            let mut best_dist = f32::MAX;
            let mut best_idx = None;

            for (i, &(bin, power)) in measurements.iter().enumerate() {
                if used_measurements[i] { continue; }
                let freq = bin as f32 / total_bins as f32;
                let dist = (track.frequency - freq).abs();
                // Maximum association distance: 3 frequency bins
                if dist < (3.0 / total_bins as f32) && dist < best_dist {
                    best_dist = dist;
                    best_idx = Some(i);
                }
            }

            if let Some(idx) = best_idx {
                let (bin, power) = measurements[idx];
                let new_freq = bin as f32 / total_bins as f32;
                // EMA smooth the frequency
                track.frequency = track.frequency * 0.7 + new_freq * 0.3;
                track.power_dbm = track.power_dbm * 0.7 + power * 0.3;
                track.confidence = (track.confidence + 0.1).min(1.0);
                track.age_ticks += 1;
                track.detection_count += 1;
                track.last_seen_tick = self.tick;
                track.alive = true;
                used_measurements[idx] = true;
            } else {
                // Coast: no measurement this tick
                track.age_ticks += 1;
                track.confidence *= 0.95; // Slowly decay
                let ticks_since_seen = (self.tick - track.last_seen_tick) as u32;
                if track.age_ticks.saturating_sub(ticks_since_seen) > self.max_coast_ticks {
                    track.alive = false; // Lost track
                }
            }
        }

        // New tracks from unassociated measurements
        for (i, &(bin, power)) in measurements.iter().enumerate() {
            if used_measurements[i] { continue; }
            let freq = bin as f32 / total_bins as f32;

            // Don't create tracks for noise-level signals
            if power < -55.0 { continue; }

            let classification = if power > -30.0 {
                ThreatClass::Continuous
            } else if power > -40.0 {
                ThreatClass::Narrowband
            } else {
                ThreatClass::Unknown
            };

            self.tracks.push(ThreatState {
                frequency: freq,
                power_dbm: power,
                confidence: 0.3, // Initial confidence is low
                classification,
                age_ticks: 1,
                detection_count: 1,
                first_seen_tick: self.tick,
                last_seen_tick: self.tick,
                alive: true,
            });
        }

        // Prune old lost tracks (keep for history, mark as dead)
        for t in &mut self.tracks {
            if !t.alive && t.age_ticks > self.max_coast_ticks * 3 {
                // Keep in list but will be filtered out by alive_tracks()
            }
        }

        // Limit track count
        while self.tracks.len() > 16 {
            // Remove the lowest-confidence dead track
            let mut worst_idx = None;
            let mut worst_conf = f32::MAX;
            for (i, t) in self.tracks.iter().enumerate() {
                if !t.alive && t.confidence < worst_conf {
                    worst_conf = t.confidence;
                    worst_idx = Some(i);
                }
            }
            if let Some(idx) = worst_idx {
                self.tracks.swap_remove(idx);
            } else {
                break;
            }
        }
    }

    fn alive_tracks(&self) -> Vec<&ThreatState> {
        self.tracks.iter().filter(|t| t.alive).collect()
    }
}

/// Global threat tracker state.
static THREAT_TRACKER: Mutex<Option<JpdaTracker>> = Mutex::new(None);

/// Initialize the threat tracker.
pub fn init() {
    let mut tracker = THREAT_TRACKER.lock();
    *tracker = Some(JpdaTracker::new());
    crate::klog!(INFO, "sensor_threat: CFAR+JPDA threat tracker initialized");
}

/// Run one tick of the threat detection pipeline.
/// Takes FFT magnitudes (e.g., from periodogram), runs CFAR then JPDA update.
pub fn tick(magnitudes: &[f32]) {
    let mut tracker = THREAT_TRACKER.lock();
    let tracker = match &mut *tracker {
        Some(ref mut t) => t,
        None => return,
    };

    if magnitudes.len() < 16 {
        return;
    }

    let config = CfarConfig::default();
    let detections = cfar_detect(magnitudes, &config);

    // Build measurement pairs: (bin_index, power_dbm)
    let measurements: Vec<(usize, f32)> = detections
        .iter()
        .map(|&bin| (bin, magnitudes[bin]))
        .collect();

    let previous_track_count = tracker.alive_tracks().len();

    tracker.update(&measurements, magnitudes.len());

    // Generate qualia for track births and deaths
    let current_tracks = tracker.alive_tracks();
    let current_count = current_tracks.len();

    // New track born
    if current_count > previous_track_count && current_count > 0 {
        crate::consciousness::qualia::record(
            crate::consciousness::qualia::KernelEventType::ThreatTrackBorn,
            Some(-0.5),
        );
    }

    // All tracks lost
    if current_count == 0 && previous_track_count > 0 {
        crate::consciousness::qualia::record(
            crate::consciousness::qualia::KernelEventType::ThreatTrackLost,
            Some(0.2),
        );
    }
}

/// Get current tracked threats.
pub fn active_threats() -> Vec<ThreatState> {
    let tracker = THREAT_TRACKER.lock();
    match &*tracker {
        Some(ref t) => t.alive_tracks().into_iter().cloned().collect(),
        None => Vec::new(),
    }
}

/// Get total number of tracks ever created.
pub fn track_count() -> usize {
    let tracker = THREAT_TRACKER.lock();
    match &*tracker {
        Some(ref t) => t.tracks.len(),
        None => 0,
    }
}

/// Compute a threat level (0..1) from active tracks.
pub fn threat_level() -> f32 {
    let tracker = THREAT_TRACKER.lock();
    match &*tracker {
        Some(ref t) => {
            let alive = t.alive_tracks();
            if alive.is_empty() {
                return 0.0;
            }
            // Threat level = mean confidence weighted by power
            let mut weighted_sum = 0.0;
            let mut total_weight = 0.0;
            for track in &alive {
                let power_weight = ((track.power_dbm + 60.0) / 30.0).clamp(0.0, 1.0);
                weighted_sum += track.confidence * power_weight;
                total_weight += power_weight;
            }
            if total_weight > 0.0 {
                (weighted_sum / total_weight).min(1.0)
            } else {
                0.0
            }
        }
        None => 0.0,
    }
}

/// Format /proc/sensor_threat report.
pub fn format_report() -> Vec<u8> {
    let mut s = String::new();
    s.push_str("=== Threat Detection (CFAR+JPDA) ===\n");
    s.push_str(&format!("threat_level: {:.3}\n", threat_level()));
    s.push_str(&format!("active_tracks: {}\n", active_threats().len()));
    s.push_str(&format!("total_tracks: {}\n", track_count()));

    let threats = active_threats();
    for (i, t) in threats.iter().enumerate() {
        s.push_str(&format!(
            "  [{}] freq={:.3} power={:.1}dBm conf={:.2} class={} age={} det={}\n",
            i, t.frequency, t.power_dbm, t.confidence,
            t.classification.name(), t.age_ticks, t.detection_count,
        ));
    }
    if threats.is_empty() {
        s.push_str("  (no active threats)\n");
    }
    s.into_bytes()
}

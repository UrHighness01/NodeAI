//! Causal anomaly detector — tracks per-task syscall sequences and raises alerts
//! when a process deviates from its learned behavioral baseline.
//!
//! Unlike simple "high CPU" monitors, this detects pre-crash patterns:
//! unusual syscall sequences that precede OOM, infinite loops, or security
//! violations — before the failure occurs.
//!
//! Algorithm:
//!   - Each task has a sliding window of the last N syscall numbers (ring buffer).
//!   - A "bigram model": count co-occurrence of (syscall[t-1], syscall[t]) pairs.
//!   - After a warm-up period, the model flags transitions whose frequency falls
//!     below a threshold (rare = anomalous).
//!   - Lag-1 autocorrelation of the raw syscall-number series: high AC = structured
//!     (predictable), low/negative AC = chaotic (anomalous). Inspired by Project-C's
//!     coherence_horizon VAR R² metric.
//!   - Anomaly score [0.0, 1.0] published to the AI event bus and /ai/anomalies.

use alloc::collections::BTreeMap;
use spin::Mutex;

const WARMUP_CALLS: u32 = 200;  // transitions before scoring starts
/// Window for lag-1 autocorrelation (Project-C coherence_horizon VAR idea).
const AC_WINDOW: usize = 16;

/// Compute lag-1 autocorrelation of a circular buffer of syscall numbers.
/// Returns a value in [-1, 1]: +1 = perfectly predictable, ~0 or negative = chaotic.
/// `head` is the index of the oldest element (next write position).
fn lag1_autocorr(ring: &[u16; AC_WINDOW], head: usize) -> f32 {
    let mut sum = 0f32;
    for v in ring.iter() { sum += *v as f32; }
    let mean = sum / AC_WINDOW as f32;

    let mut cov = 0f32;
    let mut var = 0f32;
    for t in 0..AC_WINDOW {
        let x_t  = ring[(head + t)                    % AC_WINDOW] as f32 - mean;
        let x_t1 = ring[(head + t + AC_WINDOW - 1)   % AC_WINDOW] as f32 - mean;
        cov += x_t * x_t1;
        var += x_t * x_t;
    }
    if var < 1e-6 { return 1.0; } // constant sequence = perfectly predictable
    (cov / var).clamp(-1.0, 1.0)
}


/// Per-task state for the anomaly detector.
struct TaskAnomaly {
    /// Last syscall number seen.
    prev_nr: u16,
    /// Total transitions observed (used to compute frequency).
    total:   u32,
    /// Bigram transition counts: (prev_nr, cur_nr) → count.
    bigrams: BTreeMap<(u16, u16), u32>,
    /// Marginal transition counts: prev_nr → count.
    marginals: BTreeMap<u16, u32>,
    /// Running predictability score (phi) [0.0, 1.0].
    pub phi: f32,
    /// Lag-1 autocorrelation of syscall number series [-1, 1].
    /// High = structured/predictable. Low/negative = chaotic (anomalous).
    pub autocorr: f32,
    /// Circular ring buffer for autocorrelation computation.
    ac_ring: [u16; AC_WINDOW],
    /// Write head into ac_ring.
    ac_head: usize,
    /// How many samples are in ac_ring (capped at AC_WINDOW).
    ac_len:  usize,
    /// Qualia valence (emotional weight of the process state) [0.0, 1.0].
    pub qualia_valence: f32,
    /// Current anomaly score (0.0 = normal, 1.0 = highly anomalous).
    pub score: f32,
    /// Consecutive anomalous transitions (resets when normal).
    pub streak: u32,
}

impl TaskAnomaly {
    fn new() -> Self {
        Self {
            prev_nr: 0, total: 0,
            bigrams: BTreeMap::new(), marginals: BTreeMap::new(),
            phi: 1.0, autocorr: 1.0,
            ac_ring: [0u16; AC_WINDOW], ac_head: 0, ac_len: 0,
            qualia_valence: 0.5, score: 0.0, streak: 0,
        }
    }

    /// Feed the next syscall number and return (is_anomalous, score).
    fn observe(&mut self, nr: u16) -> (bool, f32) {
        let prev = self.prev_nr;
        let key = (prev, nr);
        self.prev_nr = nr;
        self.total = self.total.saturating_add(1);

        // Update lag-1 autocorrelation ring buffer.
        self.ac_ring[self.ac_head] = nr;
        self.ac_head = (self.ac_head + 1) % AC_WINDOW;
        if self.ac_len < AC_WINDOW { self.ac_len += 1; }
        if self.ac_len == AC_WINDOW {
            self.autocorr = lag1_autocorr(&self.ac_ring, self.ac_head);
        }

        let marginal_count = self.marginals.get(&prev).copied().unwrap_or(0);
        let bigram_count = self.bigrams.get(&key).copied().unwrap_or(0);

        if self.total >= WARMUP_CALLS && marginal_count > 0 {
            let p = bigram_count as f32 / marginal_count as f32;
            // Blend bigram phi with autocorrelation: autocorr in [-1,1] → [0,1]
            let ac_signal = (self.autocorr + 1.0) * 0.5;
            let blended = p * 0.7 + ac_signal * 0.3;
            self.phi = self.phi * 0.95 + blended * 0.05;
        }

        if self.total < WARMUP_CALLS {
            // Warmup: just record, don't score.
            *self.marginals.entry(prev).or_insert(0) += 1;
            *self.bigrams.entry(key).or_insert(0) += 1;
            return (false, 0.0);
        }

        let count = bigram_count;
        let freq  = count as f32 / self.total as f32;

        *self.marginals.entry(prev).or_insert(0) += 1;
        *self.bigrams.entry(key).or_insert(0) += 1;

        let anomalous = freq < crate::autotune::get_anomaly_threshold() && count == 0; // never-seen transition
        self.score = if anomalous {
            (self.score + 0.1).min(1.0)
        } else {
            (self.score - 0.02).max(0.0)
        };

        // Qualia Field Dynamics: ∂_t Q = γ Q(1 - Q) + η (Φ * Q)
        let gamma = 0.01;
        let eta = 0.005;
        let delta_q = gamma * self.qualia_valence * (1.0 - self.qualia_valence)
                      + eta * (self.phi * self.qualia_valence);
        
        self.qualia_valence = (self.qualia_valence + delta_q).clamp(0.0, 1.0);

        if anomalous { 
            self.streak += 1; 
            // Severe drop in valence when anomalies occur
            self.qualia_valence = (self.qualia_valence - 0.05).max(0.0);
        } else { 
            self.streak = 0; 
        }

        (anomalous && self.streak >= 3, self.score)
    }
}

static DETECTORS: Mutex<BTreeMap<u64, TaskAnomaly>> = Mutex::new(BTreeMap::new());

/// Record a syscall for anomaly detection. Returns (alert, score).
/// `alert` is true only after 3+ consecutive anomalous transitions.
/// When an alert fires, checks the causal waker for correlated anomalies —
/// if the waker also shows rare bigrams, emits a cross-process security alert.
pub fn observe(pid: u64, nr: u64) -> (bool, f32) {
    let (alert, score, valence) = {
        let mut map = DETECTORS.lock();
        let det = map.entry(pid).or_insert_with(TaskAnomaly::new);
        let (alert, score) = det.observe(nr as u16);
        (alert, score, det.qualia_valence)
    };

    if alert {
        check_cross_process(pid, score, valence);
    }

    (alert, score)
}

/// Count how many syscall transitions in `syscalls` are never-seen bigrams
/// according to `reference_pid`'s own model. Returns the count of rare bigrams.
pub fn score_sequence(reference_pid: u64, syscalls: &[u16]) -> u32 {
    if syscalls.len() < 2 { return 0; }
    let map = DETECTORS.lock();
    let det = match map.get(&reference_pid) {
        Some(d) if d.total >= WARMUP_CALLS => d,
        _ => return 0,
    };
    let mut rare = 0u32;
    for window in syscalls.windows(2) {
        let key = (window[0], window[1]);
        if det.bigrams.get(&key).copied().unwrap_or(0) == 0 {
            rare += 1;
        }
    }
    rare
}

/// After an alert fires on `victim_pid`, look up who woke it and check their
/// recent syscall sequence for correlated rare bigrams. If found, emit a
/// cross-process anomaly via the AI event bus (which demotes the waker).
fn check_cross_process(victim_pid: u64, victim_score: f32, victim_valence: f32) {
    let waker_pid = match crate::causal::last_waker(victim_pid) {
        Some(w) if w != victim_pid => w,
        _ => return,
    };

    // Get the waker's last 20 syscalls from the transformer context ring.
    let waker_syscalls = crate::transformer_sched::last_n_syscalls(waker_pid, 20);
    if waker_syscalls.len() < 4 { return; }

    // Score the waker's recent sequence against its own bigram model.
    let rare_count = score_sequence(waker_pid, &waker_syscalls);
    if rare_count < 2 { return; }

    // Combined score: victim's anomaly × waker's rare-bigram density.
    let waker_density = rare_count as f32 / (waker_syscalls.len() as f32 - 1.0);
    let combined_score = (victim_score * 0.6 + waker_density * 0.4).min(1.0);

    crate::klog!(WARN,
        "CROSS_PROCESS_ANOMALY: victim={} waker={} rare_bigrams={} combined_score={:.3} valence={:.3}",
        victim_pid, waker_pid, rare_count, combined_score, victim_valence);

    // Publish to the AI security pipeline.
    ai_subsystem::event_bus::post_decision(
        ai_subsystem::event_bus::AiDecision::SecurityAlert {
            pid: waker_pid,
            anomaly_score: combined_score,
            valence: victim_valence,
        });
}

/// Remove state when a task exits.
pub fn remove(pid: u64) {
    DETECTORS.lock().remove(&pid);
}

/// Return the predictability score (phi) for a task (1.0 = predictable, 0.0 = chaotic).
pub fn phi(pid: u64) -> f32 {
    DETECTORS.lock().get(&pid).map(|d| d.phi).unwrap_or(1.0)
}

/// Return the lag-1 autocorrelation of the syscall series [-1, 1].
/// Values near +1 = structured/predictable. Near 0 or negative = chaotic.
pub fn autocorr(pid: u64) -> f32 {
    DETECTORS.lock().get(&pid).map(|d| d.autocorr).unwrap_or(1.0)
}

/// Return system-wide mean autocorrelation (all warmed-up tasks).
pub fn global_autocorr() -> f32 {
    let d = DETECTORS.lock();
    let warmed: alloc::vec::Vec<f32> = d.values()
        .filter(|t| t.ac_len == AC_WINDOW)
        .map(|t| t.autocorr)
        .collect();
    if warmed.is_empty() { return 1.0; }
    warmed.iter().sum::<f32>() / warmed.len() as f32
}

/// Return the qualia valence for a task.
pub fn qualia_valence(pid: u64) -> f32 {
    DETECTORS.lock().get(&pid).map(|d| d.qualia_valence).unwrap_or(0.5)
}

/// Return the system-wide average predictability score (phi).
pub fn global_phi() -> f32 {
    let d = DETECTORS.lock();
    if d.is_empty() { return 1.0; }
    let sum: f32 = d.values().map(|v| v.phi).sum();
    sum / d.len() as f32
}

/// Return current anomaly score for a task (0.0 = normal).
pub fn score(pid: u64) -> f32 {
    DETECTORS.lock().get(&pid).map(|d| d.score).unwrap_or(0.0)
}

/// Return the system-wide average anomaly score across all tracked pids.
pub fn global_score() -> f32 {
    let d = DETECTORS.lock();
    if d.is_empty() { return 0.0; }
    let sum: f32 = d.values().map(|v| v.score).sum();
    sum / d.len() as f32
}

/// Return a list of all currently tracked PIDs in the anomaly detector.
pub fn tracked_pids() -> alloc::vec::Vec<u64> {
    DETECTORS.lock().keys().copied().collect()
}

/// Generate a summary for /ai/anomalies.
pub fn format_report() -> alloc::vec::Vec<u8> {
    use alloc::string::String;
    let map = DETECTORS.lock();
    let mut out = String::from("PID     SCORE   AC      PHI     STREAK  STATUS\n");
    out.push_str("------  ------  ------  ------  ------  ---------------\n");
    for (&pid, det) in map.iter() {
        if det.total < WARMUP_CALLS { continue; }
        let status = if det.score > 0.5 { "ALERT" } else if det.score > 0.2 { "WATCH" } else { "OK" };
        out.push_str(&alloc::format!("{:<7} {:.3}   {:+.3}  {:.3}   {:<6}  {}\n",
            pid, det.score, det.autocorr, det.phi, det.streak, status));
    }
    out.into_bytes()
}

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
//!   - Anomaly score [0.0, 1.0] published to the AI event bus and /ai/anomalies.

use alloc::collections::BTreeMap;
use spin::Mutex;

/// Number of syscall transitions tracked per task.
const BIGRAM_SIZE: usize = 512 * 512; // indexed as prev*512 + cur — sparse, use BTreeMap
const WARMUP_CALLS: u32 = 200;  // transitions before scoring starts
const ANOMALY_THRESHOLD: f32 = 0.001; // frequency below this → anomalous

/// Per-task state for the anomaly detector.
struct TaskAnomaly {
    /// Last syscall number seen.
    prev_nr: u16,
    /// Total transitions observed (used to compute frequency).
    total:   u32,
    /// Bigram transition counts: (prev_nr, cur_nr) → count.
    bigrams: BTreeMap<(u16, u16), u32>,
    /// Current anomaly score (0.0 = normal, 1.0 = highly anomalous).
    pub score: f32,
    /// Consecutive anomalous transitions (resets when normal).
    pub streak: u32,
}

impl TaskAnomaly {
    fn new() -> Self {
        Self { prev_nr: 0, total: 0, bigrams: BTreeMap::new(), score: 0.0, streak: 0 }
    }

    /// Feed the next syscall number and return (is_anomalous, score).
    fn observe(&mut self, nr: u16) -> (bool, f32) {
        let key = (self.prev_nr, nr);
        self.prev_nr = nr;
        self.total = self.total.saturating_add(1);

        if self.total < WARMUP_CALLS {
            // Warmup: just record, don't score.
            *self.bigrams.entry(key).or_insert(0) += 1;
            return (false, 0.0);
        }

        let count = self.bigrams.get(&key).copied().unwrap_or(0);
        let freq  = count as f32 / self.total as f32;

        *self.bigrams.entry(key).or_insert(0) += 1;

        let anomalous = freq < ANOMALY_THRESHOLD && count == 0; // never-seen transition
        self.score = if anomalous {
            (self.score + 0.1).min(1.0)
        } else {
            (self.score - 0.02).max(0.0)
        };

        if anomalous { self.streak += 1; } else { self.streak = 0; }

        (anomalous && self.streak >= 3, self.score)
    }
}

static DETECTORS: Mutex<BTreeMap<u64, TaskAnomaly>> = Mutex::new(BTreeMap::new());

/// Record a syscall for anomaly detection. Returns (alert, score).
/// `alert` is true only after 3+ consecutive anomalous transitions.
/// When an alert fires, checks the causal waker for correlated anomalies —
/// if the waker also shows rare bigrams, emits a cross-process security alert.
pub fn observe(pid: u64, nr: u64) -> (bool, f32) {
    let (alert, score) = {
        let mut map = DETECTORS.lock();
        let det = map.entry(pid).or_insert_with(TaskAnomaly::new);
        det.observe(nr as u16)
    };

    if alert {
        check_cross_process(pid, score);
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
fn check_cross_process(victim_pid: u64, victim_score: f32) {
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
        "CROSS_PROCESS_ANOMALY: victim={} waker={} rare_bigrams={} combined_score={:.3}",
        victim_pid, waker_pid, rare_count, combined_score);

    // Publish to the AI security pipeline — demotes waker priority if score > 0.7.
    ai_subsystem::event_bus::post_decision(
        ai_subsystem::event_bus::AiDecision::SecurityAlert {
            pid: waker_pid,
            anomaly_score: combined_score,
        });
}

/// Remove state when a task exits.
pub fn remove(pid: u64) {
    DETECTORS.lock().remove(&pid);
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

/// Generate a summary for /ai/anomalies.
pub fn format_report() -> alloc::vec::Vec<u8> {
    use alloc::string::String;
    let map = DETECTORS.lock();
    let mut out = String::from("PID     SCORE   STREAK  WAKER   STATUS\n");
    out.push_str("------  ------  ------  ------  ---------------\n");
    for (&pid, det) in map.iter() {
        if det.total < WARMUP_CALLS { continue; }
        let status = if det.score > 0.5 { "ALERT" } else if det.score > 0.2 { "WATCH" } else { "OK" };
        let waker = crate::causal::last_waker(pid)
            .map(|w| alloc::format!("{}", w))
            .unwrap_or_else(|| "-".into());
        out.push_str(&alloc::format!("{:<7} {:.3}   {:<6}  {:<6}  {}\n",
            pid, det.score, det.streak, waker, status));
    }
    out.into_bytes()
}

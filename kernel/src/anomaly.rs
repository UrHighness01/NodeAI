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
pub fn observe(pid: u64, nr: u64) -> (bool, f32) {
    let mut map = DETECTORS.lock();
    let det = map.entry(pid).or_insert_with(TaskAnomaly::new);
    det.observe(nr as u16)
}

/// Remove state when a task exits.
pub fn remove(pid: u64) {
    DETECTORS.lock().remove(&pid);
}

/// Return current anomaly score for a task (0.0 = normal).
pub fn score(pid: u64) -> f32 {
    DETECTORS.lock().get(&pid).map(|d| d.score).unwrap_or(0.0)
}

/// Generate a summary for /ai/anomalies.
pub fn format_report() -> alloc::vec::Vec<u8> {
    use alloc::string::String;
    let map = DETECTORS.lock();
    let mut out = String::from("PID     SCORE   STREAK  STATUS\n");
    out.push_str("------  ------  ------  --------\n");
    for (&pid, det) in map.iter() {
        if det.total < WARMUP_CALLS { continue; }
        let status = if det.score > 0.5 { "ALERT" } else if det.score > 0.2 { "WATCH" } else { "OK" };
        out.push_str(&alloc::format!("{:<7} {:.3}   {:<6}  {}\n",
            pid, det.score, det.streak, status));
    }
    out.into_bytes()
}

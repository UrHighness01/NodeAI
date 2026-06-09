//! Temporal Novelty Detection — process behavioral drift monitor (Project-C port).
//!
//! Tracks whether a process is visiting novel or familiar states by computing
//! the Mahalanobis-like distance of each new `[coherence, anomaly_score, syscall_rate]`
//! observation from its own recent history window.
//!
//! High novelty = process behaviour is deviating from its baseline (exploring,
//! drifting, or potentially compromised).  Low novelty = process is in a familiar
//! behavioural regime.
//!
//! This is a simplified kernel-appropriate version of Project-C's PCA-based
//! novelty_detector.py — we use incremental mean/variance instead of SVD.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use spin::Mutex;

/// Sliding window size (number of past observations to keep).
const WINDOW_SIZE: usize = 64;

/// Dimensionality of the state vector.
const N_DIMS: usize = 3;

/// Threshold in normalized units above which a state is "novel".
const NOVELTY_THRESHOLD: f32 = 2.0;

/// A single N_DIMS state vector.
type StateVec = [f32; N_DIMS];

/// Per-process novelty tracker.
struct ProcessNovelty {
    /// Sliding window of past state vectors.
    history: Vec<StateVec>,
    /// Window index for circular buffer.
    cursor: usize,
    /// Running sum of state vectors (for O(1) mean).
    sum: StateVec,
    /// Running sum of squared state vectors (for O(1) variance).
    sum_sq: StateVec,
    /// Total observations ever (for cold-start handling).
    total_obs: u64,
    /// Number of novel states detected so far.
    novel_count: u64,
    /// Number of states above threshold in last window.
    recent_novel: u64,
}

impl ProcessNovelty {
    fn new() -> Self {
        Self {
            history: Vec::with_capacity(WINDOW_SIZE),
            cursor: 0,
            sum: [0.0; N_DIMS],
            sum_sq: [0.0; N_DIMS],
            total_obs: 0,
            novel_count: 0,
            recent_novel: 0,
        }
    }

    /// Record a new observation and return its novelty score.
    fn observe(&mut self, state: StateVec) -> f32 {
        self.total_obs += 1;

        // Update running statistics
        if self.history.len() < WINDOW_SIZE {
            self.history.push(state);
        } else {
            // Remove old value from sums
            let old = self.history[self.cursor];
            for i in 0..N_DIMS {
                self.sum[i] -= old[i];
                self.sum_sq[i] -= old[i] * old[i];
            }
            self.history[self.cursor] = state;
        }
        self.cursor = (self.cursor + 1) % WINDOW_SIZE;

        for i in 0..N_DIMS {
            self.sum[i] += state[i];
            self.sum_sq[i] += state[i] * state[i];
        }

        // Compute novelty score: normalized Euclidean distance from mean
        let n = self.history.len() as f32;
        if n < 10.0 {
            return 0.0; // cold start — not enough data
        }

        let mut dist_sq = 0.0f32;
        for i in 0..N_DIMS {
            let mean = self.sum[i] / n;
            // Variance with Bessel correction
            let var = (self.sum_sq[i] / n - mean * mean).max(1e-10);
            let dev = state[i] - mean;
            dist_sq += (dev * dev) / var;
        }

        let novelty = libm::sqrtf(dist_sq);

        if novelty > NOVELTY_THRESHOLD {
            self.novel_count += 1;
            self.recent_novel += 1;
        }

        // Decay recent_novel after WINDOW_SIZE steps
        if self.total_obs % (WINDOW_SIZE as u64) == 0 {
            self.recent_novel = 0;
        }

        novelty
    }

    /// Fraction of recent observations that were novel.
    fn recent_novelty_rate(&self) -> f32 {
        if self.total_obs < 10 { 0.0 }
        else { self.recent_novel as f32 / (WINDOW_SIZE as f32).min(self.total_obs as f32) }
    }
}

/// Global novelty state keyed by PID.
static NOVELTY_MAP: Mutex<BTreeMap<u64, ProcessNovelty>> = Mutex::new(BTreeMap::new());

/// Record a syscall observation for novelty tracking.
/// Called from `syscall_dispatch` alongside anomaly and coherence.
pub fn observe(pid: u64, coherence: f32, anomaly_score: f32) -> f32 {
    let mut map = NOVELTY_MAP.lock();
    let tracker = map.entry(pid).or_insert_with(ProcessNovelty::new);

    // Build state vector: [coherence, anomaly_score, syscall_rate_approx]
    // syscall_rate = tracker.total_obs as f32 is an approximation; the real
    // rate would need a timer, but total_obs gives relative ordering.
    let syscall_rate = (tracker.total_obs as f32).min(1e6) / 1e6; // normalize
    let state = [coherence, anomaly_score, syscall_rate];

    tracker.observe(state)
}

/// Return the novelty score for a PID (0 = not enough data, > 2 = novel).
pub fn score(pid: u64) -> f32 {
    let map = NOVELTY_MAP.lock();
    if let Some(state) = map.get(&pid) {
        state.recent_novelty_rate()
    } else {
        0.0
    }
}

/// Remove state when a process exits.
pub fn remove(pid: u64) {
    NOVELTY_MAP.lock().remove(&pid);
}

/// Format a /proc report.
pub fn format_report() -> Vec<u8> {
    use alloc::format;
    use alloc::string::String;

    let map = NOVELTY_MAP.lock();
    let mut out = String::from("NodeAI Novelty Detector (Project-C)\n");
    out.push_str("=================================\n");
    out.push_str(&format!("tracked_pids: {}\n", map.len()));

    // Top 16 by total_obs
    let mut entries: Vec<(u64, u64)> = map.iter()
        .map(|(pid, t)| (*pid, t.total_obs))
        .collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1));

    for (pid, _obs) in entries.iter().take(16) {
        if let Some(t) = map.get(pid) {
            out.push_str(&format!(
                "  pid={} novelty_rate={:.3} total_obs={}\n",
                pid, t.recent_novelty_rate(), t.total_obs
            ));
        }
    }

    out.into_bytes()
}

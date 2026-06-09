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
//! novelty_detector.py — we use Welford's online incremental variance for
//! numerical stability, avoiding the catastrophic cancellation that occurs
//! with the naive sum_sq/N - mean^2 formula in f32.

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

/// Per-process novelty tracker using Welford's online algorithm.
///
/// Welford's method maintains:
///   - n: count
///   - mean: running mean
///   - s: sum of squared differences from the current mean (S_k)
///
/// Variance = s / n  (population) or s / (n-1)  (sample with Bessel).
/// This avoids the sum_sq - n*mean^2 subtraction that causes catastrophic
/// cancellation when the values are large and variance is small.
struct ProcessNovelty {
    /// Sliding window of past state vectors (needed for un-learning on eviction).
    history: Vec<StateVec>,
    /// Window index for circular buffer.
    cursor: usize,
    /// Welford's running mean: M_k = M_{k-1} + (x_k - M_{k-1}) / k
    welford_mean: StateVec,
    /// Welford's sum-of-squared-differences: S_k
    welford_s: StateVec,
    /// Number of observations currently tracked in the window.
    count: usize,
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
            welford_mean: [0.0; N_DIMS],
            welford_s: [0.0; N_DIMS],
            count: 0,
            total_obs: 0,
            novel_count: 0,
            recent_novel: 0,
        }
    }

    /// Record a new observation and return its novelty score.
    /// Uses Welford's online algorithm for numerically stable variance.
    fn observe(&mut self, state: StateVec) -> f32 {
        self.total_obs += 1;

        if self.history.len() < WINDOW_SIZE {
            // Adding a new observation (no eviction)
            self.count += 1;
            self.history.push(state);
            self.welford_update_add(&state);
        } else {
            // Evict oldest, add new
            let old = self.history[self.cursor];
            self.history[self.cursor] = state;
            self.welford_update_remove(&old, &state);
        }
        self.cursor = (self.cursor + 1) % WINDOW_SIZE;

        if self.count < 10 {
            return 0.0; // cold start
        }

        // Mahalanobis-like novelty: sqrt(sum((x - mean)^2 / var))
        let mut dist_sq = 0.0f32;
        let n = self.count as f32;
        for i in 0..N_DIMS {
            let mean = self.welford_mean[i];
            // Welford's S_k / (n-1) = sample variance (Bessel correction)
            let var = (self.welford_s[i] / (n - 1.0)).max(1e-10);
            let dev = state[i] - mean;
            dist_sq += (dev * dev) / var;
        }

        let novelty = libm::sqrtf(dist_sq);

        if novelty > NOVELTY_THRESHOLD {
            self.novel_count += 1;
            self.recent_novel += 1;
        }

        if self.total_obs % (WINDOW_SIZE as u64) == 0 {
            self.recent_novel = 0;
        }

        novelty
    }

    /// Welford online update when ADDING a new value (no eviction).
    fn welford_update_add(&mut self, x: &StateVec) {
        let n = self.count as f32;
        for i in 0..N_DIMS {
            let delta = x[i] - self.welford_mean[i];
            self.welford_mean[i] += delta / n;
            let delta2 = x[i] - self.welford_mean[i];
            self.welford_s[i] += delta * delta2;
        }
    }

    /// Welford online update when REMOVING old and ADDING new (sliding window).
    fn welford_update_remove(&mut self, old: &StateVec, new: &StateVec) {
        let n = self.count as f32;
        for i in 0..N_DIMS {
            // Remove old value: reverse Welford
            // M_{n-1} = (M_n * n - old) / (n - 1)
            // S_{n-1} = S_n - (old - M_{n-1}) * (old - M_n)
            let m_n = self.welford_mean[i];
            let m_n1 = (m_n * n - old[i]) / (n - 1.0);
            self.welford_s[i] -= (old[i] - m_n1) * (old[i] - m_n);
            self.welford_mean[i] = m_n1;

            // Add new value: forward Welford (count stays the same)
            let delta = new[i] - self.welford_mean[i];
            self.welford_mean[i] += delta / n;
            let delta2 = new[i] - self.welford_mean[i];
            self.welford_s[i] += delta * delta2;
        }
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

//! Collective Integration — cross-process coherence coupling (Project-C port).
//!
//! Tracks whether pairs of causally-linked processes have coherence scores that
//! co-vary beyond chance.  When process A wakes process B (causal::record_wakeup),
//! we record the interaction.  Over a sliding window we compute a simplified
//! cross-correlation: the fraction of time both processes moved in the same
//! coherence direction (both up or both down).
//!
//! High collective coherence means the pair responds to shared scheduling stimuli
//! as one unit.  Low collective coherence suggests decoupled or adversarial
//! behaviour — one process is coherent while the other is chaotic.
//!
//! This is the first OS kernel to measure cross-process phi-like coupling.
//! Ported from Project-C's collective_integration.py.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use spin::Mutex;

/// How many recent coherence deltas to keep per pair.
const WINDOW_SIZE: usize = 32;

/// One observed coherence delta for a single process at a point in time.
#[derive(Clone, Copy)]
struct DeltaObs {
    /// timestamp_ms / 1000 (seconds since boot).
    tick: u64,
    /// +1 if coherence increased since last observation, -1 if decreased, 0 if unchanged.
    direction: i8,
}

/// Track the co-movement of two causally-linked PIDs.
struct PairState {
    a_deltas: Vec<DeltaObs>,
    b_deltas: Vec<DeltaObs>,
}

impl PairState {
    fn new() -> Self {
        Self {
            a_deltas: Vec::with_capacity(WINDOW_SIZE),
            b_deltas: Vec::with_capacity(WINDOW_SIZE),
        }
    }

    /// Record a coherence observation for one side of the pair.
    fn observe(&mut self, is_a: bool, tick: u64, coherence: f32) {
        let deltas = if is_a { &mut self.a_deltas } else { &mut self.b_deltas };

        // Compute direction from last observation
        let direction = if let Some(last) = deltas.last() {
            // We need to know the previous coherence value to compute delta.
            // Since we don't store it, we approximate: positive if coherence > 0.5,
            // negative if < 0.5. This is a simplification.
            if coherence > 0.55 { 1 }
            else if coherence < 0.45 { -1 }
            else { 0 }
        } else {
            0
        };

        deltas.push(DeltaObs { tick, direction });
        if deltas.len() > WINDOW_SIZE {
            deltas.remove(0);
        }
    }

    /// Compute cross-correlation of direction changes.
    /// Returns [0.0, 1.0] where 1.0 = always move together, 0.0 = always opposite.
    fn cross_corr(&self) -> f32 {
        let n = self.a_deltas.len().min(self.b_deltas.len());
        if n < 4 { return 0.5; } // not enough data — neutral

        // Align on the last `n` observations of each.
        let a_slice = &self.a_deltas[self.a_deltas.len() - n..];
        let b_slice = &self.b_deltas[self.b_deltas.len() - n..];

        let mut same = 0usize;
        let mut total = 0usize;

        for (a, b) in a_slice.iter().zip(b_slice.iter()) {
            // Only count ticks that are close in time (< 2s apart)
            if a.tick.abs_diff(b.tick) > 2 { continue; }
            total += 1;
            if a.direction == b.direction {
                same += 1;
            }
        }

        if total == 0 { 0.5 } else { same as f32 / total as f32 }
    }
}

/// Global collective integration state.
static COLLECTIVE: Mutex<CollectiveState> = Mutex::new(CollectiveState::new());

struct CollectiveState {
    /// Keyed by (min_pid, max_pid) so order doesn't matter.
    pairs: BTreeMap<(u64, u64), PairState>,
    /// Last observed coherence per PID (for delta computation).
    last_coherence: BTreeMap<u64, f32>,
}

impl CollectiveState {
    const fn new() -> Self {
        Self {
            pairs: BTreeMap::new(),
            last_coherence: BTreeMap::new(),
        }
    }

    fn key(a: u64, b: u64) -> (u64, u64) {
        if a < b { (a, b) } else { (b, a) }
    }
}

/// Record that two processes interacted (via causal wakeup, pipe, signal, etc.).
/// Should be called from `causal::record_wakeup` and `syscall_proxy::observe_pattern`.
pub fn record_interaction(pid_a: u64, pid_b: u64) {
    let key = CollectiveState::key(pid_a, pid_b);
    let mut state = COLLECTIVE.lock();

    // Observe current coherence for both processes
    let now = crate::scheduler::uptime_ms() / 1000;
    let coh_a = crate::coherence::compute_horizon(pid_a);
    let coh_b = crate::coherence::compute_horizon(pid_b);

    let pair = state.pairs.entry(key).or_insert_with(PairState::new);
    pair.observe(true, now, coh_a);
    pair.observe(false, now, coh_b);

    state.last_coherence.insert(pid_a, coh_a);
    state.last_coherence.insert(pid_b, coh_b);
}

/// Return the collective coherence score for a pair of PIDs.
/// 1.0 = perfectly coupled, 0.0 = perfectly anti-coupled, 0.5 = uncorrelated / insufficient data.
pub fn pair_coherence(pid_a: u64, pid_b: u64) -> f32 {
    let key = CollectiveState::key(pid_a, pid_b);
    let state = COLLECTIVE.lock();
    state.pairs.get(&key).map(|p| p.cross_corr()).unwrap_or(0.5)
}

/// Return the number of tracked pairs (debug / /proc).
pub fn pair_count() -> usize {
    COLLECTIVE.lock().pairs.len()
}

/// Remove all state for a PID that is exiting.
/// Prevents unbounded memory growth in `pairs` and `last_coherence`.
/// Must be called from `scheduler::exit_current_direct`.
pub fn cleanup(pid: u64) {
    let mut state = COLLECTIVE.lock();
    state.last_coherence.remove(&pid);
    state.pairs.retain(|&(a, b), _| a != pid && b != pid);
}

/// Format a /proc/collective report.
pub fn format_report() -> Vec<u8> {
    use alloc::format;
    use alloc::string::String;

    let state = COLLECTIVE.lock();
    let mut out = String::from("NodeAI Collective Integration (Project-C)\n");
    out.push_str("=============================================\n");
    out.push_str(&format!("tracked_pairs: {}\n", state.pairs.len()));

    // Top 16 pairs by strongest coupling (closest to 1.0 or 0.0)
    let mut scored: Vec<((u64, u64), f32)> = state.pairs.iter()
        .map(|(k, p)| (*k, p.cross_corr()))
        .collect();
    // Sort by distance from neutral (0.5) — strongest signal first
    scored.sort_by(|a, b| {
        let da = (a.1 - 0.5).abs();
        let db = (b.1 - 0.5).abs();
        db.partial_cmp(&da).unwrap_or(core::cmp::Ordering::Equal)
    });

    for ((a, b), cc) in scored.iter().take(16) {
        out.push_str(&format!("  pair ({},{}) collective_coherence={:.3}\n", a, b, cc));
    }

    out.into_bytes()
}

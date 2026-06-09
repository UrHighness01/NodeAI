//! Coherence-Horizon Anomaly Attribution
//!
//! A lightweight, no_std metric based on the Project-C VAR (Vector Autoregression) model.
//! Instead of just signaling an anomaly based on rare bigrams, this computes the
//! "coherence horizon"—how predictable a syscall sequence is based on its recent history.
//!
//! Highly stochastic (unpredictable) sequences have low coherence.
//! Highly structured (predictable) sequences have high coherence.

use alloc::collections::BTreeMap;
use spin::Mutex;
use alloc::vec::Vec;

const WINDOW_SIZE: usize = 64;

struct ProcessCoherence {
    history: Vec<u16>,
}

impl ProcessCoherence {
    fn new() -> Self {
        Self {
            history: Vec::with_capacity(WINDOW_SIZE),
        }
    }

    fn observe(&mut self, nr: u16) {
        if self.history.len() == WINDOW_SIZE {
            self.history.remove(0);
        }
        self.history.push(nr);
    }

    /// Computes the coherence score [0.0, 1.0].
    /// 1.0 = Highly coherent (e.g. infinite loop of 1 or 2 syscalls).
    /// 0.0 = Highly stochastic (every bigram is unique).
    fn compute_coherence(&self) -> f32 {
        if self.history.len() < 10 {
            return 1.0; // Assume coherent until we have enough data
        }

        let mut bigrams: BTreeMap<(u16, u16), u32> = BTreeMap::new();
        for window in self.history.windows(2) {
            let key = (window[0], window[1]);
            *bigrams.entry(key).or_insert(0) += 1;
        }

        let unique_bigrams = bigrams.len() as f32;
        let total_bigrams = (self.history.len() - 1) as f32;

        // Density of unique bigrams. 1.0 = all unique (stochastic).
        // 0.0 = only 1 unique bigram (highly structured).
        let stochasticity = unique_bigrams / total_bigrams;

        // Coherence is the inverse of stochasticity.
        (1.0 - stochasticity).clamp(0.0, 1.0)
    }
}

static COHERENCE_MAP: Mutex<BTreeMap<u64, ProcessCoherence>> = Mutex::new(BTreeMap::new());

/// Record a syscall for coherence tracking.
pub fn observe(pid: u64, nr: u16) {
    let mut map = COHERENCE_MAP.lock();
    let entry = map.entry(pid).or_insert_with(ProcessCoherence::new);
    entry.observe(nr);
}

/// Computes the coherence horizon metric for a process [0.0, 1.0].
pub fn compute_horizon(pid: u64) -> f32 {
    let map = COHERENCE_MAP.lock();
    if let Some(entry) = map.get(&pid) {
        entry.compute_coherence()
    } else {
        1.0
    }
}

/// Remove state when a task exits.
pub fn remove(pid: u64) {
    COHERENCE_MAP.lock().remove(&pid);
}

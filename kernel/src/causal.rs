//! Causal process wakeup graph — live DAG of which processes unblock which.
//!
//! Every time process A causes process B to transition from blocked→runnable
//! (via futex_wake, pipe write, socket send, or waitpid), we record the edge
//! A → B. The graph is a rolling window of the last N_EDGES causal events,
//! stored as a circular buffer of (waker_pid, wakee_pid, uptime_ms) triples.
//!
//! The scheduler reads this to understand causality chains — a burst of
//! anomalous syscalls from B is more interesting if we know A triggered it.
//! /ai/causal_graph exposes the edge list and a per-PID wakeup fanout count.
//!
//! This is novel: Linux has wake_up_source for profiling but never builds a
//! queryable live DAG exposed to both the scheduler and the AI subsystem.

use spin::Mutex;

const N_EDGES: usize = 512;

#[derive(Clone, Copy, Default)]
struct Edge {
    waker:     u64,
    wakee:     u64,
    uptime_ms: u64,
}

struct CausalGraph {
    edges:  [Edge; N_EDGES],
    head:   usize, // next write position (circular)
    count:  usize, // total edges written (may exceed N_EDGES)
}

impl CausalGraph {
    const fn new() -> Self {
        Self {
            edges:  [Edge { waker: 0, wakee: 0, uptime_ms: 0 }; N_EDGES],
            head:   0,
            count:  0,
        }
    }

    fn record(&mut self, waker: u64, wakee: u64, uptime_ms: u64) {
        self.edges[self.head] = Edge { waker, wakee, uptime_ms };
        self.head = (self.head + 1) % N_EDGES;
        self.count = self.count.saturating_add(1);
    }

    /// Return the most recent waker for `wakee`, if any in the live window.
    fn last_waker(&self, wakee: u64) -> Option<u64> {
        let len = self.count.min(N_EDGES);
        // Walk backwards from head.
        for i in (0..len).rev() {
            let idx = (self.head + N_EDGES - 1 - i) % N_EDGES;
            if self.edges[idx].wakee == wakee {
                return Some(self.edges[idx].waker);
            }
        }
        None
    }

    fn format_report(&self, now_ms: u64) -> alloc::vec::Vec<u8> {
        use alloc::string::String;
        let len = self.count.min(N_EDGES);
        let mut out = String::from(
            "WAKER   WAKEE   AGE_MS\n\
             ------  ------  -------\n");
        let show = len.min(32);
        for i in 0..show {
            let idx = (self.head + N_EDGES - show + i) % N_EDGES;
            let e = &self.edges[idx];
            if e.waker == 0 && e.wakee == 0 { continue; }
            let age = now_ms.saturating_sub(e.uptime_ms);
            out.push_str(&alloc::format!("{:<7} {:<7} {}\n", e.waker, e.wakee, age));
        }
        out.push_str(&alloc::format!("\ntotal_edges={}\n", self.count));
        out.into_bytes()
    }
}

static GRAPH: Mutex<CausalGraph> = Mutex::new(CausalGraph::new());

/// Record that `waker_pid` caused `wakee_pid` to become runnable.
/// Called from futex_wake, pipe_write_wake, socket_send_wake, and sys_exit
/// (which wakes the parent waiting in waitpid).
pub fn record_wakeup(waker_pid: u64, wakee_pid: u64) {
    let uptime_ms = crate::scheduler::uptime_ms();
    GRAPH.lock().record(waker_pid, wakee_pid, uptime_ms);
    // Also stamp the wakee's Task so the anomaly detector can read it.
    let mut tasks = crate::scheduler::TASKS.lock();
    if let Some(task) = tasks.get_mut(&wakee_pid) {
        task.woke_by = Some(waker_pid);
    }
}

/// Look up who most recently woke a given PID (for anomaly enrichment).
pub fn last_waker(pid: u64) -> Option<u64> {
    GRAPH.lock().last_waker(pid)
}

/// Predict the most likely next wakee for `pid` and the probability (frequency ratio).
/// Looks at the last 10 edges where `pid` is the waker; returns the mode wakee
/// and how often it appeared. Used for predictive producer-priority boosting.
pub fn predict_next_wake(pid: u64) -> Option<(u64, f32)> {
    use alloc::collections::BTreeMap;
    let graph = GRAPH.lock();
    let len = graph.count.min(N_EDGES).min(10);
    let mut freq: BTreeMap<u64, u32> = BTreeMap::new();
    for i in 0..len {
        let idx = (graph.head + N_EDGES - 1 - i) % N_EDGES;
        let e = &graph.edges[idx];
        if e.waker == pid && e.wakee != 0 {
            *freq.entry(e.wakee).or_insert(0) += 1;
        }
    }
    freq.into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(wakee, count)| (wakee, count as f32 / 10.0))
}

/// Predict which PIDs are likely to need the CPU soon because `waker_pid` is
/// about to run. Looks backwards through the edge buffer for any PID that
/// `waker_pid` has woken ≥2 times in the last 64 edges — those are its
/// habitual consumers (pipe readers, futex waiters, etc.).
///
/// Returns at most 4 candidates (cheapest to pre-enqueue).
pub fn predict_successors(waker_pid: u64) -> alloc::vec::Vec<u64> {
    use alloc::collections::BTreeMap;
    let graph = GRAPH.lock();
    let len = graph.count.min(N_EDGES).min(64); // scan last 64 edges
    let mut freq: BTreeMap<u64, u32> = BTreeMap::new();
    for i in 0..len {
        let idx = (graph.head + N_EDGES - 1 - i) % N_EDGES;
        let e = &graph.edges[idx];
        if e.waker == waker_pid && e.wakee != 0 {
            *freq.entry(e.wakee).or_insert(0) += 1;
        }
    }
    let mut candidates: alloc::vec::Vec<(u64, u32)> = freq.into_iter()
        .filter(|(_, count)| *count >= 2)
        .collect();
    candidates.sort_by(|a, b| b.1.cmp(&a.1)); // most frequent first
    candidates.truncate(4);
    candidates.into_iter().map(|(pid, _)| pid).collect()
}

/// Format for /ai/causal_graph.
pub fn format_report() -> alloc::vec::Vec<u8> {
    use alloc::string::String;
    let now = crate::scheduler::uptime_ms();

    // Edge history (releases lock before predict_next_wake calls).
    let mut out = {
        let bytes = GRAPH.lock().format_report(now);
        String::from_utf8(bytes).unwrap_or_default()
    };

    // Predictive producer table — computed outside the graph lock.
    out.push_str("\nPRODUCER  LIKELY_WAKEE  PROB  BOOST\n");
    out.push_str("--------  ------------  ----  -----\n");
    // Collect unique waker PIDs from recent edges.
    let wakers: alloc::vec::Vec<u64> = {
        let graph = GRAPH.lock();
        let len = graph.count.min(N_EDGES).min(32);
        let mut seen = alloc::collections::BTreeSet::new();
        for i in 0..len {
            let idx = (graph.head + N_EDGES - 1 - i) % N_EDGES;
            let w = graph.edges[idx].waker;
            if w != 0 { seen.insert(w); }
        }
        seen.into_iter().collect()
    };
    for waker in wakers {
        if let Some((wakee, prob)) = predict_next_wake(waker) {
            if prob >= 0.3 {
                let boost = if prob >= 0.5 { "yes(-5)" } else { "no" };
                out.push_str(&alloc::format!(
                    "{:<9} {:<13} {:.2}  {}\n", waker, wakee, prob, boost));
            }
        }
    }
    out.into_bytes()
}

//! Causal Intervention — targeted root-cause namespace escalation (Project-C port).
//!
//! When the system detects a process with sustained high anomaly score or dropping
//! coherence, instead of escalating containment on the symptomatic process alone,
//! we walk the causal wakeup chain to find the *root* process that started the chain.
//! The root receives a higher containment level than the symptom.
//!
//! This prevents the classic self-healing failure mode where the system punishes
//! a downstream process for upstream misbehaviour.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

/// How far up the causal chain to walk.
const MAX_CHAIN_DEPTH: usize = 8;

/// Minimum anomaly score to trigger an intervention.
const INTERVENTION_THRESHOLD: f32 = 0.4;

/// How often to run intervention scans (in ticks of telemetry::tick, ~10ms each).
/// Every 500 ticks = ~5 seconds.
const SCAN_INTERVAL_TICKS: u64 = 500;

static LAST_SCAN_TICK: AtomicU64 = AtomicU64::new(0);

/// Find the causal root of a process by walking the waker chain.
/// Returns the root PID (the process with no waker, or at the chain end).
fn find_causal_root(pid: u64) -> u64 {
    let chain = crate::causal::waker_chain(pid, MAX_CHAIN_DEPTH);
    // Last element in the chain is the oldest ancestor (the root waker)
    *chain.last().unwrap_or(&pid)
}

/// Perform a causal intervention scan. Called periodically from telemetry::tick.
///
/// For each tracked process with anomaly score above threshold, walk its causal
/// chain and escalate containment on the root process rather than the symptom.
pub fn tick() {
    let now = crate::scheduler::uptime_ms() / 10; // convert ms to ~10ms ticks
    let last = LAST_SCAN_TICK.load(Ordering::Relaxed);
    if now < last + SCAN_INTERVAL_TICKS { return; }
    LAST_SCAN_TICK.store(now, Ordering::Relaxed);

    let pids = crate::scheduler::all_pids();
    if pids.is_empty() { return; }

    // Collect (symptom_pid, root_pid, score) for all above-threshold processes
    let mut interventions: Vec<(u64, u64, f32)> = Vec::new();

    for pid in &pids {
        let score = crate::anomaly::score(*pid);
        if score > INTERVENTION_THRESHOLD {
            let root = find_causal_root(*pid);
            interventions.push((*pid, root, score));
        }
    }

    if interventions.is_empty() { return; }

    // Apply containment escalation: root gets higher level than symptom.
    // Root gets a strong containment push proportional to score.
    // Symptom gets a milder push (it's the victim, not the cause).
    for (symptom, root, score) in &interventions {
        if *root != *symptom {
            // Root is the cause: escalate strongly
            let root_score = (*score + 0.2).min(1.0);
            crate::namespaces::update(*root, root_score);
            crate::klog!(INFO,
                "causal_intervention: root pid={} escalated (score={:.3}, from symptom pid={})",
                root, root_score, symptom
            );

            // Symptom is downstream: milder escalation
            let symptom_score = (*score * 0.5).min(1.0);
            crate::namespaces::update(*symptom, symptom_score);
            crate::klog!(INFO,
                "causal_intervention: symptom pid={} contained (score={:.3}, root pid={})",
                symptom, symptom_score, root
            );
        } else {
            // PID is its own root (no causal chain) — escalate normally
            crate::namespaces::update(*symptom, *score);
            crate::klog!(INFO,
                "causal_intervention: self-root pid={} escalated (score={:.3})",
                symptom, score
            );
        }
    }
}

/// Format a /proc report.
pub fn format_report() -> Vec<u8> {
    use alloc::format;
    use alloc::string::String;

    let pids = crate::scheduler::all_pids();
    let mut out = String::from("NodeAI Causal Intervention (Project-C)\n");
    out.push_str("=====================================\n");
    out.push_str(&format!("scan_interval: {} ticks (~{}s)\n", SCAN_INTERVAL_TICKS, SCAN_INTERVAL_TICKS / 100));
    out.push_str(&format!("threshold: {:.1}\n", INTERVENTION_THRESHOLD));

    // Show current above-threshold processes and their causal roots
    for pid in pids.iter().take(32) {
        let score = crate::anomaly::score(*pid);
        if score > INTERVENTION_THRESHOLD {
            let root = find_causal_root(*pid);
            let level = crate::namespaces::level_of(*pid);
            out.push_str(&format!(
                "  pid={} score={:.3} root={} level={:?}\n",
                pid, score, root, level
            ));
        }
    }

    out.into_bytes()
}

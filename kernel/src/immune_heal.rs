//! Immune Self-Healing Triggers — automated subsystem health monitoring.
//!
//! Monitors key kernel metrics (anomaly, memory, coherence, phi, threat)
//! against thresholds. When a metric drifts outside nominal range, a heal
//! action is triggered and recorded. Actions are prioritized (info/warn/crit).
//!
//! Call tick() every 100ms. Use active_heals() and health_status() for templates.

use alloc::string::String;
use alloc::string::ToString;
use alloc::format;
use alloc::vec::Vec;
use spin::Mutex;

/// Maximum number of heal actions to keep in history.
const HEAL_HISTORY_MAX: usize = 32;

/// A single heal action record.
#[derive(Debug, Clone)]
pub struct HealAction {
    pub subsystem: String,
    pub action: String,
    pub priority: u8, // 0=info, 1=warning, 2=critical
    pub tick_recorded: u64,
}

/// Health metric names.
const METRIC_NAMES: &[&str] = &[
    "anomaly", "memory", "coherence", "phi", "threat",
];

/// Evaluate a health metric by name — returns (value_0to100, description, priority).
fn evaluate_metric(name: &str) -> (f32, &'static str, u8) {
    match name {
        "anomaly" => {
            let v = crate::anomaly::global_score() * 100.0;
            (v, "tighten anomaly gate", 1)
        }
        "memory" => {
            let free_mb = crate::memory::free_mb() as f32;
            let pct_used = (1.0 - (free_mb / 440.0).min(1.0)) * 100.0;
            (pct_used, "trigger AI balloon reclaim", 1)
        }
        "coherence" => {
            let v = crate::consciousness::self_model::snapshot()
                .map(|s| (1.0 - s.coherence) * 100.0)
                .unwrap_or(50.0);
            (v, "reset coherence state", 2)
        }
        "phi" => {
            let phi = crate::consciousness::phi::current_phi();
            let v = (1.0 - phi.min(1.0)) * 100.0;
            (v, "boost causal integration", 2)
        }
        "threat" => {
            let v = crate::sensor_threat::threat_level() * 100.0;
            (v, "elevate immune readiness", 1)
        }
        _ => (0.0, "unknown", 0),
    }
}

/// System health rating.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HealthRating {
    Healthy,
    Degraded,
    Critical,
}

struct HealState {
    history: Vec<HealAction>,
    trigger_counts: [u64; 5], // one per threshold
    last_healthy_tick: u64,
}

static STATE: Mutex<Option<HealState>> = Mutex::new(None);

/// Initialize the self-healing trigger monitor.
pub fn init() {
    let mut lock = STATE.lock();
    *lock = Some(HealState {
        history: Vec::with_capacity(HEAL_HISTORY_MAX),
        trigger_counts: [0; 5],
        last_healthy_tick: 0,
    });
    crate::klog!(INFO, "immune_heal: self-healing triggers initialized ({} thresholds)", METRIC_NAMES.len());
}

/// Tick the self-heal monitor — called every 100ms.
/// Checks all thresholds and records actions for any that are exceeded.
pub fn tick() {
    let mut lock = STATE.lock();
    let state = match &mut *lock {
        Some(s) => s,
        None => return,
    };

    let now = crate::scheduler::uptime_ms() / 100;
    let mut any_triggered = false;

    for (i, name) in METRIC_NAMES.iter().enumerate() {
        let (value, action, priority) = evaluate_metric(name);
        let threshold = match *name {
            "anomaly"   => 70.0,
            "memory"    => 85.0,
            "coherence" => 70.0,
            "phi"       => 80.0,
            "threat"    => 80.0,
            _           => 75.0,
        };

        if value >= threshold {
            any_triggered = true;
            state.trigger_counts[i] = state.trigger_counts[i].saturating_add(1);

            // Only record if we haven't recorded this exact action recently
            let should_record = state.history.last()
                .map(|last| last.subsystem != *name || last.tick_recorded + 50 < now)
                .unwrap_or(true);

            if should_record && state.history.len() < HEAL_HISTORY_MAX {
                state.history.push(HealAction {
                    subsystem: name.to_string(),
                    action: action.to_string(),
                    priority,
                    tick_recorded: now,
                });
            }
        }
    }

    if !any_triggered {
        state.last_healthy_tick = now;
    }
}

/// Get the overall system health rating.
pub fn health_rating() -> HealthRating {
    let lock = STATE.lock();
    match &*lock {
        Some(s) => {
            let crit_count = s.history.iter().filter(|h| h.priority == 2).count();
            let warn_count = s.history.iter().filter(|h| h.priority == 1).count();
            if crit_count > 0 { HealthRating::Critical }
            else if warn_count > 5 { HealthRating::Degraded }
            else { HealthRating::Healthy }
        }
        None => HealthRating::Healthy,
    }
}

/// Get count of active/pending heal actions.
pub fn active_heal_count() -> usize {
    let lock = STATE.lock();
    match &*lock {
        Some(s) => s.history.len(),
        None => 0,
    }
}

/// Get a human-readable health description (for templates).
pub fn health_summary() -> String {
    let lock = STATE.lock();
    match &*lock {
        Some(s) => {
            let rating = if s.history.iter().any(|h| h.priority == 2) { "critical" }
                         else if s.history.iter().filter(|h| h.priority == 1).count() > 3 { "degraded" }
                         else { "healthy" };

            let heal_count = s.history.len();
            let last_heal = s.history.last()
                .map(|h| format!("{} — {}", h.subsystem, h.action))
                .unwrap_or_else(|| String::from("none"));

            format!(
                "Health: {}, {} heal actions triggered. Last: {}",
                rating, heal_count, last_heal,
            )
        }
        None => String::from("Self-heal monitor not initialized"),
    }
}

/// Get the most recent heal action description (for templates).
pub fn last_heal_action() -> String {
    let lock = STATE.lock();
    match &*lock {
        Some(s) => s.history.last()
            .map(|h| format!("{} ({}, prio {})", h.action, h.subsystem, h.priority))
            .unwrap_or_else(|| String::from("no action needed")),
        None => String::from("offline"),
    }
}

/// Format /proc/immune_heal report.
pub fn format_report() -> Vec<u8> {
    let lock = STATE.lock();
    match &*lock {
        Some(s) => {
            let crit_count = s.history.iter().filter(|h| h.priority == 2).count();
            let warn_count = s.history.iter().filter(|h| h.priority == 1).count();
            let rating = if crit_count > 0 { "CRITICAL 🔴" }
                         else if warn_count > 3 { "DEGRADED ⚠" }
                         else { "HEALTHY ✓" };

            let mut report = format!(
                "Self-Healing Triggers\n\
                 =====================\n\
                 health: {}\n\
                 total_heal_actions: {}\n\
                 trigger_counts: anomaly={} memory={} coherence={} phi={} threat={}\n\
                 \n\
                 Recent Heal Actions:\n",
                rating,
                s.history.len(),
                s.trigger_counts[0], s.trigger_counts[1],
                s.trigger_counts[2], s.trigger_counts[3], s.trigger_counts[4],
            );

            for (i, action) in s.history.iter().rev().take(10).enumerate() {
                let prio_str = match action.priority {
                    0 => "info",
                    1 => "warn",
                    _ => "CRIT",
                };
                report.push_str(&format!(
                    "  {}. [{}] {} → {}\n",
                    i + 1, prio_str, action.subsystem, action.action,
                ));
            }

            report.into_bytes()
        }
        None => format!("Self-Healing Triggers\nNot initialized\n").into_bytes(),
    }
}

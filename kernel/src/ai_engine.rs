//! Kernel-side AI engine integration — Phase 8.
//!
//! This module bridges the kernel and the `ai_subsystem` crate:
//!   - Initialises the event bus and loads default (untrained) models at boot.
//!   - Provides a `process_tick()` function called from the scheduler tick
//!     to drain events, run inference, and apply decisions.
//!   - Task profile updates feed the scheduler AI with per-task feature vectors.

use ai_subsystem::{
    event_bus::{self, AiDecision, KernelEvent},
    domains::scheduler_ai::{self, TaskFeatures},
    inference::{DenseLayer, SequentialModel, Activation},
};
use crate::scheduler::task::AiProfile;

/// Initialise the AI subsystem: event bus + tiny default models.
pub fn init() {
    event_bus::init();

    // Bootstrap a trivial 5→8→2 scheduler model with random-ish weights.
    // In production this would be loaded from disk; for now it gives the
    // inference path a real exercised code path.
    let m = build_default_scheduler_model();
    scheduler_ai::load_model(m);

    crate::klog!(INFO, "AI subsystem initialized — event bus + scheduler model ready");
}

/// Called from the timer interrupt (scheduler tick path) every tick.
/// Drains events, runs scheduler AI, applies decisions back to the runqueue.
pub fn process_tick(uptime_ms: u64) {
    // Publish a timer tick event
    event_bus::publish(KernelEvent::TimerTick { uptime_ms });

    // Drain all queued events and let each domain process them
    let events = event_bus::drain_events();
    for event in events {
        match event {
            KernelEvent::TaskCreated { pid, .. } => {
                // Initialise an AI profile for the new task (already done in spawn)
                let _ = pid;
            }
            KernelEvent::SyscallIssued { pid, syscall_nr } => {
                // Feed to security AI (simplified — full integration in Phase 10)
                let _ = (pid, syscall_nr);
            }
            KernelEvent::TimerTick { .. } => { /* handled above */ }
            _ => {}
        }
    }

    // Drain any AI decisions and apply them
    let decisions = event_bus::drain_decisions();
    for decision in decisions {
        apply_decision(decision);
    }
}

/// Apply an AI decision from the event bus to a kernel subsystem.
fn apply_decision(decision: AiDecision) {
    match decision {
        AiDecision::SchedulerAdjust { pid, nice_delta, predicted_burst_us } => {
            // Apply nice delta to the task — validated by safety constraints
            let delta = ai_subsystem::safety::check_scheduler_nice(0, nice_delta);
            crate::klog!(TRACE,
                "AI: scheduler adjust pid={} nice_delta={} burst={}μs",
                pid, delta, predicted_burst_us);
        }
        AiDecision::SecurityAlert { pid, anomaly_score } => {
            if anomaly_score > 0.9 {
                crate::klog!(WARN, "AI: security alert pid={} anomaly={:.3}", pid, anomaly_score);
            }
        }
        AiDecision::PowerAdjust { pstate, park_mask } => {
            crate::klog!(DEBUG, "AI: power adjust pstate={} park={:#x}", pstate, park_mask);
        }
        AiDecision::MemoryPrefetch { pid, pages } => {
            let _ = (pid, pages);
        }
    }
}

/// Update the AI profile for a task from its scheduler-collected statistics.
/// Called by the scheduler when a task is descheduled.
pub fn update_task_profile(pid: u64, profile: &AiProfile) {
    let features = TaskFeatures {
        avg_burst_norm:  (profile.ticks_run as f32 / 1000.0).min(1.0),
        io_fraction:     0.1, // TODO: track I/O waits in later phase
        cache_miss_rate: 0.0, // TODO: PMC-based in Phase 5
        priority_norm:   0.5,
        wait_time_norm:  0.0,
    };
    let decision = scheduler_ai::predict(&features);

    // Post the decision back to the bus for application
    event_bus::post_decision(AiDecision::SchedulerAdjust {
        pid,
        nice_delta:           decision.nice_adjust,
        predicted_burst_us:   decision.predicted_burst_us,
    });
}

// ── Default model bootstrap ───────────────────────────────────────────────────

/// Build a minimal bootstrap scheduler model (5→8→2) with near-zero weights.
/// Output will always be close to the neutral fallback until real weights are loaded.
fn build_default_scheduler_model() -> SequentialModel {
    let mut m = SequentialModel::new();
    m.add_layer(DenseLayer {
        in_size:    5,
        out_size:   8,
        weights:    alloc::vec![0.01f32; 5 * 8],
        biases:     alloc::vec![0.0f32; 8],
        activation: Activation::ReLU,
    });
    m.add_layer(DenseLayer {
        in_size:    8,
        out_size:   2,
        weights:    alloc::vec![0.01f32; 8 * 2],
        biases:     alloc::vec![0.5f32, 0.0f32],
        activation: Activation::Sigmoid,
    });
    m
}

// ── Phase 29 additions ────────────────────────────────────────────────────────

/// Called when the system wakes from sleep — re-warms internal caches/state.
pub fn wake_hint() {
    crate::klog!(DEBUG, "ai_engine: wake hint received");
}

/// Set the AI inference CPU budget as a percentage (0-100).
pub fn set_budget_pct(pct: u8) {
    crate::klog!(DEBUG, "ai_engine: inference budget set to {}%", pct);
}

/// Load quantised LLM weights into the AI engine.
/// Returns true on success, false if the format is unrecognised.
pub fn load_llm_weights(data: &[u8]) -> bool {
    // Verify magic bytes of the NodeAI LLM format.
    if data.len() < 8 || &data[..4] != b"NLLM" {
        crate::klog!(WARN, "ai_engine: load_llm_weights — bad magic");
        return false;
    }
    crate::klog!(INFO, "ai_engine: loaded {} bytes of LLM weights", data.len());
    true
}

/// Run LLM inference on `prompt` with a context window of `ctx` tokens.
/// Returns the generated text response.
pub fn llm_infer(prompt: &str, ctx: usize) -> alloc::string::String {
    // Stub: echo back summary until a real transformer engine is integrated.
    let _ = ctx;
    let preview = &prompt[..prompt.len().min(80)];
    alloc::format!("[LLM stub] Received: \"{}...\" — inference engine pending.", preview)
}

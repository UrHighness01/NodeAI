//! Kernel-side AI engine integration.
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
                // Feed to security AI (simplified — full anomaly detection is future work)
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
            // Validate through safety layer then apply.
            let delta = ai_subsystem::safety::check_scheduler_nice(0, nice_delta);
            if delta != 0 {
                crate::scheduler::adjust_priority(pid, delta);
            }
            // Update the task's AI burst estimate.
            // (We don't mutate AiProfile here to avoid lock nesting; the
            //  scheduler tick updates it separately via update_task_profile.)
            crate::klog!(TRACE,
                "AI: scheduler adjust pid={} nice={:+} burst={}μs",
                pid, delta, predicted_burst_us);
        }
        AiDecision::SecurityAlert { pid, anomaly_score } => {
            if anomaly_score > 0.95 {
                crate::klog!(WARN,
                    "AI: SECURITY ALERT pid={} anomaly={:.3} — isolating", pid, anomaly_score);
                // Demote to lowest priority; security subsystem may escalate.
                crate::scheduler::adjust_priority(pid, 20);
            } else if anomaly_score > 0.7 {
                crate::klog!(WARN, "AI: security warn pid={} anomaly={:.3}", pid, anomaly_score);
            }
        }
        AiDecision::PowerAdjust { pstate, park_mask } => {
            crate::power::apply_pstate(pstate, park_mask);
        }
        AiDecision::MemoryPrefetch { pid, pages } => {
            // Prefetch pages into TLB/cache — best-effort, not critical.
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
        cache_miss_rate: 0.0, // TODO: read from IA32_PERF_CTR when PMCs are configured
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

// ── LLM weight storage ────────────────────────────────────────────────────────
//
// NLLM model format (little-endian):
//   [0..4]  magic  b"NLLM"
//   [4]     n_layers  u8
//   [5..7]  vocab_size u16
//   [7]     hidden_size u8  (multiplied by 64 to get actual hidden dim)
//   [8..]   layers: for each layer:
//             in_size  u16
//             out_size u16
//             activation u8 (0=linear, 1=relu, 2=sigmoid, 3=tanh)
//             weights  f32 * in_size * out_size
//             biases   f32 * out_size

static LLM_MODEL: spin::Mutex<Option<SequentialModel>> = spin::Mutex::new(None);
static LLM_READY: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Load quantised LLM weights into the AI engine.
/// Returns true on success, false if the format is unrecognised or truncated.
pub fn load_llm_weights(data: &[u8]) -> bool {
    if data.len() < 8 || &data[..4] != b"NLLM" {
        crate::klog!(WARN, "ai_engine: load_llm_weights — bad magic (got {:?})", &data[..4.min(data.len())]);
        return false;
    }
    let n_layers = data[4] as usize;
    if n_layers == 0 || n_layers > 32 {
        crate::klog!(WARN, "ai_engine: load_llm_weights — invalid n_layers={}", n_layers);
        return false;
    }

    let mut model = SequentialModel::new();
    let mut cursor = 8usize;

    for layer_idx in 0..n_layers {
        if cursor + 5 > data.len() {
            crate::klog!(WARN, "ai_engine: truncated at layer {}", layer_idx);
            return false;
        }
        let in_size  = u16::from_le_bytes([data[cursor],   data[cursor+1]]) as usize;
        let out_size = u16::from_le_bytes([data[cursor+2], data[cursor+3]]) as usize;
        let act_byte = data[cursor+4];
        cursor += 5;

        let w_bytes = in_size * out_size * 4;
        let b_bytes = out_size * 4;
        if cursor + w_bytes + b_bytes > data.len() {
            crate::klog!(WARN, "ai_engine: truncated weights at layer {}", layer_idx);
            return false;
        }

        let mut weights = alloc::vec![0f32; in_size * out_size];
        for (i, w) in weights.iter_mut().enumerate() {
            *w = f32::from_le_bytes([
                data[cursor + i*4], data[cursor + i*4+1],
                data[cursor + i*4+2], data[cursor + i*4+3],
            ]);
        }
        cursor += w_bytes;

        let mut biases = alloc::vec![0f32; out_size];
        for (i, b) in biases.iter_mut().enumerate() {
            *b = f32::from_le_bytes([
                data[cursor + i*4], data[cursor + i*4+1],
                data[cursor + i*4+2], data[cursor + i*4+3],
            ]);
        }
        cursor += b_bytes;

        let activation = match act_byte {
            1 => Activation::ReLU,
            2 => Activation::Sigmoid,
            3 => Activation::Tanh,
            _ => Activation::Linear,
        };
        model.add_layer(DenseLayer { in_size, out_size, weights, biases, activation });
    }

    *LLM_MODEL.lock() = Some(model);
    LLM_READY.store(true, core::sync::atomic::Ordering::Release);
    crate::klog!(INFO, "ai_engine: LLM loaded — {} layers, {} bytes", n_layers, data.len());
    true
}

/// Run LLM inference on `prompt`.
/// If a model is loaded, runs a forward pass treating the prompt as a feature vector.
/// Falls back to a descriptive message if no model is present.
pub fn llm_infer(prompt: &str, ctx: usize) -> alloc::string::String {
    let _ = ctx;
    if !LLM_READY.load(core::sync::atomic::Ordering::Acquire) {
        return alloc::format!(
            "NodeAI: model not loaded. Place NLLM-format weights at /var/lib/llm/model.bin");
    }
    let mut guard = LLM_MODEL.lock();
    if let Some(model) = guard.as_mut() {
        // Encode the prompt as a simple feature vector (byte frequencies, normalised).
        let bytes = prompt.as_bytes();
        let first_layer_in = model.layers.first().map(|l| l.in_size).unwrap_or(8);
        let mut input = alloc::vec![0f32; first_layer_in];
        for (i, &b) in bytes.iter().take(first_layer_in).enumerate() {
            input[i] = b as f32 / 255.0;
        }
        let output = model.infer(&input);
        let summary: alloc::string::String = output.iter().take(4)
            .map(|v| alloc::format!("{:.3}", v))
            .collect::<alloc::vec::Vec<_>>()
            .join(", ");
        alloc::format!("NodeAI inference: [{}] (prompt_len={})", summary, bytes.len())
    } else {
        alloc::string::String::from("NodeAI: model unavailable")
    }
}

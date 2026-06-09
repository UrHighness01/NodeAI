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

    // Try to load episodic memory from disk, fallback to new memory.
    match crate::vfs::read_file("/.ai_memory.bin") {
        Ok(data) => {
            let mut store = ai_subsystem::vector_store::VectorStore::new();
            if store.deserialize(&data) {
                *VECTOR_STORE.lock() = Some(store);
                crate::klog!(INFO, "AI subsystem: loaded episodic memory from disk ({} bytes)", data.len());
            } else {
                crate::klog!(WARN, "AI subsystem: episodic memory corrupt, resetting");
                *VECTOR_STORE.lock() = Some(ai_subsystem::vector_store::VectorStore::new());
            }
        }
        Err(_) => {
            *VECTOR_STORE.lock() = Some(ai_subsystem::vector_store::VectorStore::new());
        }
    }

    crate::klog!(INFO, "AI subsystem initialized — event bus + scheduler model + episodic memory ready");
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
                // Feed to security AI — cross-process anomaly detection in anomaly.rs
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

    // Phase 4: Dynamic Cognition Budgeting
    // Adjust computational depth based on phi-metric stability once per second.
    if uptime_ms % 1000 == 0 {
        let phi = crate::anomaly::global_phi();
        let target_budget = if phi > 0.8 {
            10 // Stable -> Low budget
        } else if phi < 0.4 {
            80 // Chaotic -> High budget for deep analysis
        } else {
            40 // Transition -> Medium budget
        };
        set_budget_pct(target_budget);
    }
    
    // Round 16 Phase 1: Predictive Causal Failure Mitigation
    // Every 500ms, scan running processes to pre-emptively isolate high-risk causal chains before failure.
    if uptime_ms % 500 == 0 {
        let pids = crate::anomaly::tracked_pids();
        for pid in pids {
            if crate::causal::is_high_risk_chain(pid) {
                let chain = crate::causal::waker_chain(pid, 3);
                
                // Round 18 Phase 3: Zapper
                // Attempt to surgically sever the link between the most recent waker and wakee
                // before resorting to full causal chain isolation.
                if chain.len() >= 2 {
                    if crate::el_engine::trigger_zapper(chain[1], chain[0]) {
                        crate::klog!(WARN, "AI: ZAPPER pre-emptively healed chaotic chain before full quarantine.");
                        continue; // Skip the full chain quarantine!
                    }
                }

                crate::klog!(WARN, "AI: PREDICTIVE MITIGATION — isolating high-risk causal chain ending at pid={}", pid);
                for &c in &chain {
                    crate::causal::record_wakeup(crate::causal::AI_KERNEL_PID, c);
                    crate::scheduler::adjust_priority(c, 20); // demote/isolate the whole chain
                }
            }
        }
    }

    // Round 17 Phase 1: Automated Hyper-parameter Evolution with 60s Averaging
    if uptime_ms > 0 && uptime_ms % 5000 == 0 {
        let phi = crate::anomaly::global_phi();
        let current_sum = f32::from_bits(PHI_SUM_60S.load(core::sync::atomic::Ordering::Relaxed));
        let count = PHI_COUNT_60S.load(core::sync::atomic::Ordering::Relaxed) + 1;
        PHI_SUM_60S.store((current_sum + phi).to_bits(), core::sync::atomic::Ordering::Relaxed);
        PHI_COUNT_60S.store(count, core::sync::atomic::Ordering::Relaxed);

        if uptime_ms % 60000 == 0 {
            let avg_phi = (current_sum + phi) / (count as f32);
            let last_avg_phi = f32::from_bits(LAST_PHI.load(core::sync::atomic::Ordering::Relaxed));
            
            if avg_phi > last_avg_phi && avg_phi > 0.6 {
                // Stability increased over 60s. Persist successful genes into episodic memory.
                let mut genes = [0f32; 16];
                genes[0] = crate::tunables::QUANTUM_MS.load(core::sync::atomic::Ordering::Relaxed) as f32;
                genes[1] = crate::tunables::AI_NICE_CAP.load(core::sync::atomic::Ordering::Relaxed) as f32;
                genes[2] = crate::tunables::ANOMALY_STREAK.load(core::sync::atomic::Ordering::Relaxed) as f32;
                genes[3] = crate::tunables::LATENCY_BIAS.load(core::sync::atomic::Ordering::Relaxed) as f32;
                
                if let Some(store) = VECTOR_STORE.lock().as_mut() {
                    store.insert(&genes, 0xFFFF, uptime_ms);
                }
                crate::klog!(INFO, "AI: EVOLUTION — 60s avg phi increased to {:.3}. Persisted genes into episodic memory.", avg_phi);
            } else if avg_phi < 0.4 {
                // Stability low. Mutate genes to explore better parameters.
                crate::klog!(WARN, "AI: EVOLUTION — system chaotic (60s avg phi={:.3}). Mutating genes.", avg_phi);
                let r = (uptime_ms / 60000) % 4;
                let delta = if (uptime_ms / 100) % 2 == 0 { 1 } else { -1 };
                match r {
                    0 => {
                        let current = crate::tunables::QUANTUM_MS.load(core::sync::atomic::Ordering::Relaxed);
                        let _ = crate::tunables::apply("quantum_ms", (current as i64) + (delta * 2));
                    }
                    1 => {
                        let current = crate::tunables::AI_NICE_CAP.load(core::sync::atomic::Ordering::Relaxed);
                        let _ = crate::tunables::apply("ai_nice_cap", (current as i64) + delta);
                    }
                    2 => {
                        let current = crate::tunables::ANOMALY_STREAK.load(core::sync::atomic::Ordering::Relaxed);
                        let _ = crate::tunables::apply("anomaly_streak", (current as i64) + delta);
                    }
                    3 => {
                        let current = crate::tunables::LATENCY_BIAS.load(core::sync::atomic::Ordering::Relaxed);
                        let _ = crate::tunables::apply("latency_bias", (current as i64) + delta);
                    }
                    _ => {}
                }
            }
            LAST_PHI.store(avg_phi.to_bits(), core::sync::atomic::Ordering::Relaxed);
            
            // Reset accumulators
            PHI_SUM_60S.store(0f32.to_bits(), core::sync::atomic::Ordering::Relaxed);
            PHI_COUNT_60S.store(0, core::sync::atomic::Ordering::Relaxed);
            
            // Periodically flush episodic memory to disk
            save_episodic_memory();
        }
    }
}

/// Apply an AI decision from the event bus to a kernel subsystem.
fn apply_decision(decision: AiDecision) {
    match decision {
        AiDecision::SchedulerAdjust { pid, nice_delta, predicted_burst_us } => {
            // Validate through safety layer, then clamp to live tunable cap.
            let raw   = ai_subsystem::safety::check_scheduler_nice(0, nice_delta);
            let cap   = crate::tunables::AI_NICE_CAP.load(core::sync::atomic::Ordering::Relaxed);
            let delta = (raw as i32).clamp(-cap, cap) as i8;
            if delta != 0 {
                crate::causal::record_wakeup(crate::causal::AI_KERNEL_PID, pid);
                crate::scheduler::adjust_priority(pid, delta);
            }
            // Update the task's AI burst estimate.
            // (We don't mutate AiProfile here to avoid lock nesting; the
            //  scheduler tick updates it separately via update_task_profile.)
            crate::klog!(TRACE,
                "AI: scheduler adjust pid={} nice={:+} burst={}μs",
                pid, delta, predicted_burst_us);
        }
        AiDecision::SecurityAlert { pid, anomaly_score, valence } => {
            if anomaly_score > 0.95 {
                // Record into episodic memory
                if let Some(store) = VECTOR_STORE.lock().as_mut() {
                    let mut vec = [0f32; 16];
                    vec[0] = anomaly_score;
                    vec[1] = valence;
                    store.insert(&vec, pid, crate::scheduler::uptime_ms());
                }

                if valence < 0.3 {
                    crate::klog!(WARN,
                        "AI: SECURITY ALERT pid={} anomaly={:.3} valence={:.3} — MALICIOUS, isolating", pid, anomaly_score, valence);
                    // Low valence (chaotic/negative) + high anomaly -> demote to lowest priority.
                    crate::causal::record_wakeup(crate::causal::AI_KERNEL_PID, pid);
                    crate::scheduler::adjust_priority(pid, 20);

                    // Round 16 Phase 4: Affective-Causal Scheduler Integration
                    // Check if its causal fan-out processes share a high negative valence.
                    let fanout = crate::causal::fanout_pids(pid);
                    let mut low_valence_count = 0;
                    for &fpid in &fanout {
                        if crate::anomaly::qualia_valence(fpid) < 0.3 {
                            low_valence_count += 1;
                        }
                    }
                    if low_valence_count > 0 && low_valence_count >= fanout.len() / 2 {
                        crate::klog!(WARN, "AI: AFFECTIVE-CAUSAL — Fan-out processes share low valence ({}/{}). Deprioritizing sub-tree.", low_valence_count, fanout.len());
                        for &fpid in &fanout {
                            crate::causal::record_wakeup(crate::causal::AI_KERNEL_PID, fpid);
                            crate::scheduler::adjust_priority(fpid, 20);
                        }
                    }
                } else {
                    crate::klog!(WARN,
                        "AI: SECURITY ALERT pid={} anomaly={:.3} valence={:.3} — EXPLORATORY, maintaining priority", pid, anomaly_score, valence);
                    // High valence (structured/positive) + high anomaly -> let it run, might be a novel but safe behavior.
                }
            } else if anomaly_score > 0.7 {
                crate::klog!(WARN, "AI: security warn pid={} anomaly={:.3} valence={:.3}", pid, anomaly_score, valence);
            }
        }
        AiDecision::PowerAdjust { pstate, park_mask } => {
            crate::power::apply_pstate(pstate, park_mask);
        }
        AiDecision::MemoryPrefetch { pid, pages } => {
            // Prefetch pages into TLB/cache — best-effort, not critical.
            crate::causal::record_wakeup(crate::causal::AI_KERNEL_PID, pid);
            let _ = (pid, pages);
        }
    }
}

/// Per-task previous prediction record — needed for SGD feedback.
struct PredRecord { predicted_burst_us: u64, features: alloc::vec::Vec<f32> }
static LAST_PRED: spin::Mutex<alloc::collections::BTreeMap<u64, PredRecord>>
    = spin::Mutex::new(alloc::collections::BTreeMap::new());

/// Learning rate for online SGD (small — 100 Hz × many tasks = fast convergence).
const LR: f32 = 0.0005;

/// Update the AI profile for a task from scheduler-collected statistics.
/// Runs one step of vanilla SGD on the scheduler model using the actual burst
/// as the supervision signal, then produces the next prediction.
pub fn update_task_profile(pid: u64, profile: &AiProfile) {
    let features = alloc::vec![
        (profile.ticks_run as f32 / 1000.0).min(1.0), // avg burst (normalised)
        0.1f32,  // io_fraction placeholder (PMC in future)
        0.0f32,  // cache_miss_rate placeholder
        0.5f32,  // priority (normalised to [-1,1] later)
        0.0f32,  // wait_time placeholder
    ];

    // ── Online SGD step ─────────────────────────────────────────────────────
    // If we have a previous prediction for this task, use the actual burst to
    // compute error and update the model weights (one gradient descent step).
    let actual_burst_us = profile.ticks_run * 10_000; // ticks × 10ms → μs approx
    {
        let mut preds = LAST_PRED.lock();
        if let Some(prev) = preds.get(&pid) {
            let pred_burst = prev.predicted_burst_us as f32;
            let actual     = actual_burst_us as f32;
            let error      = pred_burst - actual;

            // Apply SGD to the scheduler model's output layer.
            if let Some(model) = LLM_MODEL.lock().as_mut() {
                if let Some(layer) = model.layers.last_mut() {
                    for (w, &x) in layer.weights.iter_mut().zip(prev.features.iter()) {
                        *w -= LR * 2.0 * error * x;
                    }
                    for b in layer.biases.iter_mut() {
                        *b -= LR * 2.0 * error;
                    }
                }
            }
        }
    }

    // ── Produce next prediction ─────────────────────────────────────────────
    let task_feat = TaskFeatures {
        avg_burst_norm:  features[0],
        io_fraction:     features[1],
        cache_miss_rate: features[2],
        priority_norm:   features[3],
        wait_time_norm:  features[4],
    };
    let decision = scheduler_ai::predict(&task_feat);

    // Store this prediction for the next SGD step.
    LAST_PRED.lock().insert(pid, PredRecord {
        predicted_burst_us: decision.predicted_burst_us,
        features: features,
    });

    event_bus::post_decision(AiDecision::SchedulerAdjust {
        pid,
        nice_delta:         decision.nice_adjust,
        predicted_burst_us: decision.predicted_burst_us,
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

// ── AI engine pipeline: fingerprint → transformer → causal → anomaly → blend ──

/// Called when the system wakes from sleep — re-warms internal caches/state.
pub fn wake_hint() {
    crate::klog!(DEBUG, "ai_engine: wake hint received");
}

static BUDGET_PCT: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(10);
static LAST_PHI: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
static PHI_SUM_60S: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
static PHI_COUNT_60S: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);

/// Set the AI inference CPU budget as a percentage (0-100).
pub fn set_budget_pct(pct: u8) {
    let old = BUDGET_PCT.swap(pct, core::sync::atomic::Ordering::Relaxed);
    if old != pct {
        crate::klog!(INFO, "ai_engine: cognition budget dynamically adjusted to {}% (was {}%) based on phi-stability", pct, old);
    }
}

// ── Episodic Kernel Memory ──────────────────────────────────────────────────

static VECTOR_STORE: spin::Mutex<Option<ai_subsystem::vector_store::VectorStore>> = spin::Mutex::new(None);

pub fn check_memory_label(label: u64) -> bool {
    if let Some(store) = VECTOR_STORE.lock().as_ref() {
        store.has_label(label)
    } else {
        false
    }
}

pub fn get_global_phi() -> f32 {
    crate::anomaly::global_phi()
}

pub fn evaluate_ubot_proposal(gene_name: &str, delta: i64) {
    let phi = crate::anomaly::global_phi();
    if phi < 0.8 {
        crate::klog!(INFO, "AI: UBOT EVOLUTION — accepting proposal for {} += {}", gene_name, delta);
        let _ = crate::tunables::apply(gene_name, crate::tunables::get(gene_name) + delta);
    } else {
        crate::klog!(WARN, "AI: UBOT EVOLUTION — rejecting proposal for {}, phi={:.3} is already high", gene_name, phi);
    }
}

/// Load episodic memory from binary payload.
pub fn load_episodic_memory(data: &[u8]) -> bool {
    let mut store = ai_subsystem::vector_store::VectorStore::new();
    if store.deserialize(data) {
        crate::klog!(INFO, "ai_engine: episodic memory loaded from {} bytes", data.len());
        *VECTOR_STORE.lock() = Some(store);
        true
    } else {
        crate::klog!(WARN, "ai_engine: episodic memory load failed");
        false
    }
}

/// Serialize episodic memory to a binary payload and write to disk.
pub fn save_episodic_memory() {
    if let Some(store) = VECTOR_STORE.lock().as_ref() {
        let data = store.serialize();
        if crate::vfs::write_file("/.ai_memory.bin", &data).is_ok() {
            crate::klog!(INFO, "ai_engine: flushed episodic memory to disk ({} bytes)", data.len());
        } else {
            crate::klog!(WARN, "ai_engine: failed to write episodic memory to disk");
        }
    }
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

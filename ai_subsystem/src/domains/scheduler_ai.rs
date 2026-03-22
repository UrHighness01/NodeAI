//! Scheduler AI domain — predicts CPU burst length and priority adjustments.

use crate::inference::SequentialModel;
use spin::Mutex;

/// Feature vector fed to the scheduler model.
/// All values normalized to [0, 1].
#[derive(Default)]
pub struct TaskFeatures {
    /// Average CPU burst length (normalized over historical max).
    pub avg_burst_norm: f32,
    /// I/O wait fraction (0 = CPU-bound, 1 = I/O-bound).
    pub io_fraction: f32,
    /// L2 cache miss rate (0 = perfect cache, 1 = all misses).
    pub cache_miss_rate: f32,
    /// Current static priority normalized (0 = lowest, 1 = highest).
    pub priority_norm: f32,
    /// Time since last scheduled (normalized, for starvation detection).
    pub wait_time_norm: f32,
}

/// AI decision output for a task.
pub struct SchedulerDecision {
    /// Predicted burst duration in microseconds.
    pub predicted_burst_us: u64,
    /// Priority adjustment recommended by AI [-20, 20].
    pub nice_adjust: i8,
    /// Preferred CPU affinity hint (None = let OS decide).
    pub cpu_hint: Option<u32>,
}

static MODEL: Mutex<Option<SequentialModel>> = Mutex::new(None);

/// Load the scheduler AI model. Called once at boot.
pub fn load_model(model: SequentialModel) {
    *MODEL.lock() = Some(model);
}

/// Query the AI for a scheduling decision on a task.
/// Falls back to deterministic defaults if no model is loaded.
pub fn predict(features: &TaskFeatures) -> SchedulerDecision {
    let input = [
        features.avg_burst_norm,
        features.io_fraction,
        features.cache_miss_rate,
        features.priority_norm,
        features.wait_time_norm,
    ];

    let guard = MODEL.lock();
    if let Some(model) = guard.as_ref() {
        let output = model.infer(&input);
        // output[0]: normalized burst prediction (0..1 maps to 0..10_000 μs)
        // output[1]: nice adjustment normalized (-1..1 maps to -20..20)
        let burst = (output.get(0).copied().unwrap_or(0.5) * 10_000.0) as u64;
        let nice = (output.get(1).copied().unwrap_or(0.0) * 20.0).clamp(-20.0, 20.0) as i8;
        SchedulerDecision {
            predicted_burst_us: burst,
            nice_adjust: nice,
            cpu_hint: None, // TODO: add NUMA-aware placement output
        }
    } else {
        // Deterministic fallback: no prediction, no adjustment
        SchedulerDecision {
            predicted_burst_us: 4000, // default 4 ms timeslice
            nice_adjust: 0,
            cpu_hint: None,
        }
    }
}

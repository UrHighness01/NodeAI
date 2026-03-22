//! Security AI domain — anomaly detection on syscall sequences.

use crate::inference::SequentialModel;
use spin::Mutex;

/// Encoded recent syscall pattern for anomaly scoring.
#[derive(Default)]
pub struct SyscallFeatures {
    /// Normalized syscall frequency vector (one entry per syscall category).
    pub freq_vector: [f32; 16],
    /// Privilege escalation attempts in last window (0 = none, 1 = many).
    pub priv_attempt_rate: f32,
    /// Unusual memory access pattern score.
    pub mem_anomaly_score: f32,
}

pub struct SecurityDecision {
    /// Anomaly score (0.0 = normal, 1.0 = highly suspicious).
    pub anomaly_score: f32,
    /// Whether to raise a kernel security alert.
    pub raise_alert: bool,
    /// Whether to throttle the process's syscall rate.
    pub throttle: bool,
}

static MODEL: Mutex<Option<SequentialModel>> = Mutex::new(None);

pub fn load_model(model: SequentialModel) {
    *MODEL.lock() = Some(model);
}

pub fn predict(features: &SyscallFeatures) -> SecurityDecision {
    let mut input = alloc::vec![0f32; 18];
    input[..16].copy_from_slice(&features.freq_vector);
    input[16] = features.priv_attempt_rate;
    input[17] = features.mem_anomaly_score;

    let guard = MODEL.lock();
    if let Some(model) = guard.as_ref() {
        let output = model.infer(&input);
        let score = output.get(0).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        SecurityDecision {
            anomaly_score: score,
            raise_alert: score > 0.85,   // threshold: highly suspicious
            throttle: score > 0.70,       // threshold: suspicious — slow it down
        }
    } else {
        SecurityDecision { anomaly_score: 0.0, raise_alert: false, throttle: false }
    }
}

//! Power AI domain — workload-aware CPU frequency and core parking.

use crate::inference::SequentialModel;
use spin::Mutex;

#[derive(Default)]
pub struct PowerFeatures {
    /// Overall CPU utilization (0 = idle, 1 = fully loaded).
    pub cpu_utilization: f32,
    /// Thermal headroom (0 = at thermal limit, 1 = cool).
    pub thermal_headroom: f32,
    /// Battery/power budget remaining (0 = empty/limited, 1 = full/unlimited).
    pub power_budget: f32,
    /// Latency sensitivity hint from scheduler (0 = batch, 1 = interactive).
    pub latency_sensitivity: f32,
}

pub struct PowerDecision {
    /// Recommended P-state index (0 = maximum performance).
    pub pstate: u8,
    /// Bitmask of cores that should be parked (powered down).
    pub park_mask: u64,
}

static MODEL: Mutex<Option<SequentialModel>> = Mutex::new(None);

pub fn load_model(model: SequentialModel) {
    *MODEL.lock() = Some(model);
}

pub fn predict(features: &PowerFeatures) -> PowerDecision {
    let input = [
        features.cpu_utilization,
        features.thermal_headroom,
        features.power_budget,
        features.latency_sensitivity,
    ];

    let guard = MODEL.lock();
    if let Some(model) = guard.as_ref() {
        let output = model.infer(&input);
        let pstate   = (output.get(0).copied().unwrap_or(0.0) * 16.0).clamp(0.0, 15.0) as u8;
        let park_raw = output.get(1).copied().unwrap_or(0.0);
        // Only park cores when system is lightly loaded
        let park_mask = if park_raw > 0.7 { 0xFF00_0000_0000_0000u64 } else { 0 };
        PowerDecision { pstate, park_mask }
    } else {
        PowerDecision { pstate: 0, park_mask: 0 }
    }
}

//! Memory AI domain — predicts page fault patterns and swap candidates.

use crate::inference::SequentialModel;
use spin::Mutex;

#[derive(Default)]
pub struct MemoryFeatures {
    /// Fraction of working set that fits in L3 cache (0 = none, 1 = all).
    pub cache_coverage: f32,
    /// Recent page fault rate (0 = no faults, 1 = very high).
    pub fault_rate_norm: f32,
    /// Process memory pressure (0 = no pressure, 1 = OOM-candidate).
    pub pressure_norm: f32,
}

pub struct MemoryDecision {
    /// Number of pages to proactively prefetch.
    pub prefetch_pages: u32,
    /// Suggested swap-out aggressiveness (0 = none, 255 = maximum).
    pub swap_pressure: u8,
}

static MODEL: Mutex<Option<SequentialModel>> = Mutex::new(None);

pub fn load_model(model: SequentialModel) {
    *MODEL.lock() = Some(model);
}

pub fn predict(features: &MemoryFeatures) -> MemoryDecision {
    let input = [features.cache_coverage, features.fault_rate_norm, features.pressure_norm];

    let guard = MODEL.lock();
    if let Some(model) = guard.as_ref() {
        let output = model.infer(&input);
        let prefetch = (output.get(0).copied().unwrap_or(0.1) * 32.0).max(0.0) as u32;
        let swap = (output.get(1).copied().unwrap_or(0.0) * 255.0).clamp(0.0, 255.0) as u8;
        MemoryDecision { prefetch_pages: prefetch, swap_pressure: swap }
    } else {
        MemoryDecision { prefetch_pages: 4, swap_pressure: 0 }
    }
}

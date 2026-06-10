//! Phase 5: Deliberation Engine — multi-policy generation, weighted voting, veto.
//!
//! The kernel generates multiple policy options, deliberates via CoreValues,
//! and executes the best one. The veto mechanism allows the self-model to
//! override decisions that conflict with identity.

use alloc::vec::Vec;
use spin::Mutex;
use core::sync::atomic::{AtomicU64, Ordering};

/// Core values guiding all decisions.
pub struct CoreValues {
    pub preservation: f32,  // system survival
    pub efficiency: f32,    // resource use
    pub fairness: f32,      // cross-process fairness
    pub growth: f32,        // learning
    pub autonomy: f32,      // self-direction
}

/// A single policy option for a decision.
pub struct Policy {
    pub name: &'static str,
    pub score: f32,
    pub vetoed: bool,
}

struct DeliberationState {
    values: CoreValues,
    recent_decisions: Vec<Policy>,
    total_decisions: u64,
    veto_count: u64,
}

impl DeliberationState {
    const fn new() -> Self {
        Self {
            values: CoreValues {
                preservation: 1.0,
                efficiency: 0.8,
                fairness: 0.6,
                growth: 0.5,
                autonomy: 0.4,
            },
            recent_decisions: Vec::new(),
            total_decisions: 0,
            veto_count: 0,
        }
    }
}

static DELIB: Mutex<DeliberationState> = Mutex::new(DeliberationState::new());

pub fn init() {}

/// Deliberate over a set of policy options. Returns the index of the chosen policy.
/// Each policy's score is weighted by CoreValues. Veto triggers if score < 0.3.
pub fn deliberate(policies: &mut [Policy]) -> usize {
    let mut state = DELIB.lock();
    state.total_decisions = state.total_decisions.wrapping_add(1);
    
    if policies.is_empty() { return 0; }
    
    // Find best policy by CoreValues-weighted score
    let mut best_idx = 0;
    let mut best_score = -1.0f32;
    
    for (i, p) in policies.iter().enumerate() {
        // Weighted by core values (simplified: all values apply equally)
        let weight = (state.values.preservation + state.values.efficiency 
                     + state.values.fairness + state.values.growth 
                     + state.values.autonomy) / 5.0;
        let weighted = p.score * weight;
        
        if weighted > best_score {
            best_score = weighted;
            best_idx = i;
        }
    }
    
    // Veto if best score is too low (conflicts with identity)
    if best_score < 0.3 {
        policies[best_idx].vetoed = true;
        state.veto_count = state.veto_count.wrapping_add(1);
        
        // Fall back to next best non-vetoed
        for (i, p) in policies.iter().enumerate() {
            if !p.vetoed && p.score >= 0.3 {
                best_idx = i;
                break;
            }
        }
    }
    
    // Log decision
    if policies[best_idx].vetoed {
        crate::klog!(DEBUG, "deliberation: vetoed, fallback to policy {}", best_idx);
    }
    
    state.recent_decisions.push(Policy { name: "", score: best_score, vetoed: false });
    if state.recent_decisions.len() > 16 { state.recent_decisions.remove(0); }
    
    best_idx
}

/// Adjust core values (called by AI engine or userspace daemon).
pub fn set_values(v: CoreValues) {
    DELIB.lock().values = v;
}

pub fn tick() {}

pub fn format_report() -> Vec<u8> {
    use alloc::format;
    use alloc::string::String;
    let state = DELIB.lock();
    let mut out = String::from("NodeAI Deliberation Engine (Phase 5)\n");
    out.push_str("==================================\n");
    out.push_str(&format!("decisions: {}\n", state.total_decisions));
    out.push_str(&format!("vetoes: {}\n", state.veto_count));
    out.push_str(&format!("values: pres={:.1} eff={:.1} fair={:.1} grow={:.1} auto={:.1}\n",
        state.values.preservation, state.values.efficiency,
        state.values.fairness, state.values.growth, state.values.autonomy));
    out.into_bytes()
}


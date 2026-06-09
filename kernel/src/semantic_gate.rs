//! Semantic Syscall Gatekeeper
//!
//! Evaluates the semantic intent of syscalls based on the process's causal lineage.
//! Uses a fast-path cache to avoid blocking on transformer inference.

use alloc::sync::Arc;
use spin::Mutex;
use alloc::collections::VecDeque;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Decision {
    Permitted,
    Denied,
    NeedsEvaluation,
}

pub struct SemanticGatekeeper {
    cache: crate::hot_lock::HotMap<[u8; 8], Decision>,
    queue: Mutex<VecDeque<EvaluationRequest>>,
}

#[derive(Clone)]
pub struct EvaluationRequest {
    pub pid: u64,
    pub sys_nr: u64,
    pub path: alloc::string::String,
}

pub static GATEKEEPER: spin::Once<SemanticGatekeeper> = spin::Once::new();

pub fn get_gatekeeper() -> &'static SemanticGatekeeper {
    GATEKEEPER.call_once(|| SemanticGatekeeper::new())
}

impl SemanticGatekeeper {
    pub fn new() -> Self {
        Self {
            cache: crate::hot_lock::HotMap::new(),
            queue: Mutex::new(VecDeque::new()),
        }
    }

    /// Fast-path semantic check.
    pub fn check(&self, pid: u64, sys_nr: u64, path: &str) -> Decision {
        let key = Self::hash_key(pid, sys_nr, path);
        if let Some(decision) = self.cache.get(&key) {
            decision
        } else {
            // Cache miss: Push to queue for background evaluation and fail-open.
            self.queue.lock().push_back(EvaluationRequest { pid, sys_nr, path: alloc::string::String::from(path) });
            self.cache.insert(key, Decision::NeedsEvaluation);
            Decision::Permitted // Fail-open asynchronously
        }
    }

    fn hash_key(pid: u64, sys_nr: u64, path: &str) -> [u8; 8] {
        // A simple hash for the Semantic Cache key
        let mut h = pid.wrapping_mul(0x9E3779B97F4A7C15);
        h ^= sys_nr.wrapping_mul(0xC6A4A7935BD1E995);
        let path_hash = path.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        h ^= path_hash.wrapping_mul(0x811C9DC5);
        h.to_le_bytes()
    }

    /// Process the evaluation queue in the background.
    pub fn process_queue(&self) {
        let req = { self.queue.lock().pop_front() };
        if let Some(r) = req {
            // Call the INT8 Quantized Transformer Inference
            let safe = crate::semantic_sandbox::is_semantically_safe(r.pid, &r.path);
            let decision = if safe {
                Decision::Permitted
            } else {
                crate::klog!(WARN, "SemanticGatekeeper: Denying future access to {} for pid {}", r.path, r.pid);
                Decision::Denied
            };

            let key = Self::hash_key(r.pid, r.sys_nr, &r.path);
            self.cache.insert(key, decision);
        }
    }
}

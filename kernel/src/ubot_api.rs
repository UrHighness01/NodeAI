//! Ubot Autonomous Agent API
//!
//! Exposes syscalls for Ubot agents running in user-space to interface
//! with the kernel's Semantic Sandbox and Evolution Loop.

use alloc::string::String;
use alloc::vec::Vec;
use crate::syscall::{EPERM, EINVAL};
use crate::scheduler::current_pid;

/// Syscall: Propose a hyper-parameter mutation to the AI engine.
///
/// Arg0: address of the gene name (UTF-8 string)
/// Arg1: length of the gene name
/// Arg2: proposed i64 delta value
pub fn sys_ubot_propose_mutation(name_ptr: u64, name_len: u64, delta: u64) -> i64 {
    let pid = current_pid();
    let profile = crate::namespaces::profile_of(pid);

    if !profile.mutation_propose {
        crate::klog!(WARN, "UBOT: pid={} denied mutation proposal (missing SemanticProfile capability)", pid);
        return EPERM;
    }

    if name_len > 64 { return EINVAL; }

    const USER_END: u64 = 0x0000_8000_0000_0000;
    if name_ptr == 0 || name_ptr.saturating_add(name_len) > USER_END {
        return crate::syscall::EFAULT;
    }

    let slice = unsafe { core::slice::from_raw_parts(name_ptr as *const u8, name_len as usize) };
    if let Ok(gene_name) = core::str::from_utf8(slice) {
        crate::klog!(INFO, "UBOT: pid={} proposed mutation {} += {}", pid, gene_name, delta as i64);
        
        // Push the proposal to the AI Engine for evaluation
        crate::ai_engine::evaluate_ubot_proposal(gene_name, delta as i64);
        0
    } else {
        EINVAL
    }
}

/// Syscall: Query the current global Phi (system stability) score.
/// Returns phi multiplied by 1000 (fixed-point).
pub fn sys_ubot_query_phi() -> i64 {
    let pid = current_pid();
    let profile = crate::namespaces::profile_of(pid);

    if !profile.causal_graph_read {
        crate::klog!(WARN, "UBOT: pid={} denied phi query (missing SemanticProfile capability)", pid);
        return EPERM;
    }

    let current_phi = crate::ai_engine::get_global_phi();
    (current_phi * 1000.0) as i64
}

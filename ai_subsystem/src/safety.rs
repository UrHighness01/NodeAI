//! Safety constraint engine — hard rules that the AI can never override.
//!
//! Every AI decision is passed through this validator before being acted on.
//! If a decision violates a constraint, it is replaced with a safe default
//! and the violation is logged in the audit log.

use crate::audit::{log_constraint_violation, AuditEvent};

/// Result of constraint checking.
pub enum ConstraintResult {
    Allowed,
    Blocked { reason: &'static str },
}

// ── Scheduler constraints ─────────────────────────────────────────────────────

/// The AI must not assign a higher priority than the original task's ceiling.
pub fn check_scheduler_nice(original_nice: i32, proposed_adjust: i8) -> i8 {
    let result = original_nice + proposed_adjust as i32;
    // Never let AI boost priority into real-time range via nice (< -20)
    if result < -20 {
        log_constraint_violation(AuditEvent::PriorityHardCap { original_nice, proposed_adjust });
        return (-20 - original_nice).max(-20).min(20) as i8;
    }
    proposed_adjust
}

// ── Memory constraints ─────────────────────────────────────────────────────────

/// The AI must not cause eviction of critical kernel pages.
/// Returns true if a swap-out of `page_flags` is permitted.
pub fn check_swap_permitted(page_flags: u64) -> bool {
    const KERNEL_PAGE_BIT: u64 = 1 << 0;
    const LOCKED_PAGE_BIT: u64 = 1 << 1;
    const AI_MODEL_PAGE_BIT: u64 = 1 << 2;

    if page_flags & (KERNEL_PAGE_BIT | LOCKED_PAGE_BIT | AI_MODEL_PAGE_BIT) != 0 {
        log_constraint_violation(AuditEvent::IllegalSwapAttempt { page_flags });
        return false;
    }
    true
}

// ── Security constraints ───────────────────────────────────────────────────────

/// The AI may raise a security alert but cannot kill a process without human-approval flag.
pub fn check_process_kill_allowed(ai_requested_kill: bool, human_approval: bool) -> bool {
    if ai_requested_kill && !human_approval {
        log_constraint_violation(AuditEvent::UnauthorizedKillAttempt);
        return false;
    }
    ai_requested_kill && human_approval
}

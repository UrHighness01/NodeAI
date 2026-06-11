//! Security subsystem — basic hardening, execve validation, KASLR.
//!
//! Implements:
//!   - SMEP / SMAP enforcement (CR4 bits)
//!   - Capability system (64-bit bitmask per task)
//!   - Seccomp-style syscall filter table per process
//!   - KASLR helpers (randomised base offsets)
//!   - Stack-canary management for kernel stacks

use core::sync::atomic::{AtomicU64, Ordering};

// ── Capability definitions ────────────────────────────────────────────────────

/// Capabilities that a process may hold (bit positions).
pub mod cap {
    pub const SYS_ADMIN:   u64 = 1 <<  0; // catch-all administrative
    pub const NET_RAW:     u64 = 1 <<  1; // raw socket access
    pub const SYS_PTRACE:  u64 = 1 <<  2; // ptrace other processes
    pub const DAC_OVERRIDE:u64 = 1 <<  3; // bypass DAC file permission checks
    pub const SETUID:      u64 = 1 <<  4; // change UID/GID
    pub const SYS_MODULE:  u64 = 1 <<  5; // load/unload kernel modules
    pub const NET_BIND:    u64 = 1 <<  6; // bind to ports < 1024
    pub const SYS_BOOT:    u64 = 1 <<  7; // reboot / power off
    pub const SYS_NICE:    u64 = 1 <<  8; // raise process priority
    pub const AI_OVERRIDE: u64 = 1 << 62; // can override AI decisions
    pub const KERNEL:      u64 = u64::MAX; // kernel-internal full capabilities
}

// ── Per-process seccomp filter ────────────────────────────────────────────────

/// Seccomp action for a matching syscall.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeccompAction {
    Allow,
    Block,
    Kill,
}

/// A simple syscall whitelist/blacklist.
pub struct SyscallFilter {
    /// Bitset of allowed syscall numbers (covers 0..255 cheaply).
    allowed: [u64; 4],  // 4 × 64 = 256 syscalls
    default_action: SeccompAction,
}

impl SyscallFilter {
    /// Empty filter — default allow everything.
    pub const fn allow_all() -> Self {
        Self { allowed: [u64::MAX; 4], default_action: SeccompAction::Allow }
    }

    /// Default-deny filter (like a whitelist).
    pub const fn deny_all() -> Self {
        Self { allowed: [0; 4], default_action: SeccompAction::Block }
    }

    /// Allow a specific syscall number.
    pub fn allow(&mut self, nr: u64) {
        if nr < 256 { self.allowed[(nr / 64) as usize] |= 1 << (nr % 64); }
    }

    /// Block a specific syscall number.
    pub fn block(&mut self, nr: u64) {
        if nr < 256 { self.allowed[(nr / 64) as usize] &= !(1 << (nr % 64)); }
    }

    /// Check if a syscall number is permitted.
    pub fn check(&self, nr: u64) -> SeccompAction {
        if nr < 256 {
            let allowed = self.allowed[(nr / 64) as usize] & (1 << (nr % 64)) != 0;
            if allowed { SeccompAction::Allow } else { self.default_action }
        } else {
            self.default_action
        }
    }
}

// ── SMEP / SMAP ───────────────────────────────────────────────────────────────

/// Enable SMEP (bit 20) and SMAP (bit 21) in CR4.
/// Must be called after paging is fully set up.
pub fn enable_smep_smap() {
    // Check CPUID structured extended features for SMEP (EBX bit 7) and
    // SMAP (EBX bit 20) before touching CR4.  Writing unsupported bits
    // causes a #GP which can triple-fault if the exception stack is not ready.
    let ebx = unsafe {
        let mut out: u32;
        core::arch::asm!(
            "push rbx",       // rbx is callee-saved, preserve it
            "cpuid",
            "mov {0:e}, ebx",
            "pop rbx",
            out(reg) out,
            inout("eax") 7u32 => _,
            in("ecx") 0u32,
            options(nostack),
        );
        out
    };
    let smep = (ebx & (1 << 7))  != 0;
    let smap = (ebx & (1 << 20)) != 0;

    if !smep && !smap {
        crate::klog!(WARN, "Security: CPU does not support SMEP/SMAP — skipping");
        return;
    }

    unsafe {
        let mut cr4: u64;
        core::arch::asm!("mov {}, cr4", out(reg) cr4);
        if smep { cr4 |= 1 << 20; }
        if smap { cr4 |= 1 << 21; }
        core::arch::asm!("mov cr4, {}", in(reg) cr4);
    }
    crate::klog!(INFO, "Security: SMEP={} SMAP={} enabled in CR4", smep, smap);
}

/// Returns true if SMEP is currently active.
pub fn smep_active() -> bool {
    let cr4: u64;
    unsafe { core::arch::asm!("mov {}, cr4", out(reg) cr4); }
    cr4 & (1 << 20) != 0
}

// ── Stack canary ─────────────────────────────────────────────────────────────

/// A per-kernel-stack canary value placed at the base of every kernel stack.
/// If this is ever overwritten, the stack overflowed.
pub const STACK_CANARY: u64 = 0xDEAD_BEEF_CAFE_BABE;

/// Write a canary at the given stack base (lowest address of the guard area).
pub unsafe fn place_stack_canary(stack_base: u64) {
    core::ptr::write_volatile(stack_base as *mut u64, STACK_CANARY);
}

/// Returns true if the canary at `stack_base` is intact.
pub unsafe fn check_stack_canary(stack_base: u64) -> bool {
    core::ptr::read_volatile(stack_base as *const u64) == STACK_CANARY
}

// ── KASLR ─────────────────────────────────────────────────────────────────────

/// KASLR base offset applied at boot.
/// Seeded from the bootloader handoff page (limine boot_info). Falls back to 0.
static KASLR_OFFSET: AtomicU64 = AtomicU64::new(0);

pub fn set_kaslr_offset(offset: u64) {
    KASLR_OFFSET.store(offset, Ordering::Relaxed);
}

pub fn kaslr_offset() -> u64 {
    KASLR_OFFSET.load(Ordering::Relaxed)
}

// ── Initialise ────────────────────────────────────────────────────────────────

use alloc::collections::BTreeMap;
use spin::RwLock;

/// Dynamic capabilities and security state for a task.
#[derive(Debug, Clone)]
pub struct SecurityContext {
    pub capabilities: u64,
}

impl Default for SecurityContext {
    fn default() -> Self {
        // By default, a process inherits no special capabilities unless granted.
        // We will grant KERNEL capabilities to init (PID 0 and 1).
        Self { capabilities: 0 }
    }
}

static SECURITY_CONTEXTS: RwLock<BTreeMap<u64, SecurityContext>> = RwLock::new(BTreeMap::new());

/// Initialize the security context for a new task. Inherits from parent if specified.
pub fn init_task_context(pid: u64, parent_pid: Option<u64>) {
    let caps = if pid == 0 || pid == 1 {
        cap::KERNEL
    } else if let Some(ppid) = parent_pid {
        SECURITY_CONTEXTS.read().get(&ppid).map(|c| c.capabilities).unwrap_or(0)
    } else {
        0
    };
    
    SECURITY_CONTEXTS.write().insert(pid, SecurityContext { capabilities: caps });
}

/// Remove a task's security context on exit.
pub fn cleanup_task_context(pid: u64) {
    SECURITY_CONTEXTS.write().remove(&pid);
}

/// Check if a task has a specific capability.
pub fn has_capability(pid: u64, cap: u64) -> bool {
    // Treat init (PID 0) or PID 1 as always having full capabilities just in case.
    if pid == 0 || pid == 1 { return true; }
    
    if let Some(ctx) = SECURITY_CONTEXTS.read().get(&pid) {
        ctx.capabilities & cap == cap
    } else {
        false
    }
}

/// Minimum PID eligible for coherence-horizon confinement.
/// Kernel-internal threads (PID < 8) are never confinable.
const MIN_CONFINABLE_PID: u64 = 8;

/// Escalate namespace confinement for a task whose syscall autocorrelation has
/// dropped below the coherence threshold for too many consecutive windows.
/// Revokes non-essential capabilities atomically. Idempotent after first call.
pub fn escalate_confinement(pid: u64) {
    if pid < MIN_CONFINABLE_PID { return; }

    // AI_OVERRIDE intentionally excluded — AI oversight is never revoked.
    const REVOKE_MASK: u64 = cap::NET_BIND | cap::SYS_MODULE | cap::SETUID | cap::SYS_BOOT;

    // Single write lock: check-then-mutate atomically — no TOCTOU.
    let revoked = {
        let mut w = SECURITY_CONTEXTS.write();
        if let Some(ctx) = w.get_mut(&pid) {
            let revoked = ctx.capabilities & REVOKE_MASK;
            if revoked == 0 { return; } // already confined, nothing to revoke
            ctx.capabilities &= !REVOKE_MASK;
            revoked
        } else {
            return;
        }
    };

    crate::klog!(WARN,
        "COHERENCE_HORIZON: pid={} syscall autocorr below threshold — \
         confinement escalated, revoked caps={:#x}", pid, revoked);

    // Only call invalidate_handles for network caps (the only kind that can
    // have open handles in the current FD subsystem).
    if revoked & cap::NET_BIND != 0 {
        crate::syscall::invalidate_handles(pid, cap::NET_RAW);
    }
}

/// Revoke a specific capability from a task dynamically.
pub fn revoke_capability(pid: u64, cap: u64) {
    if let Some(ctx) = SECURITY_CONTEXTS.write().get_mut(&pid) {
        ctx.capabilities &= !cap;
        crate::klog!(WARN, "SECURITY: Revoked capability {:#x} from pid={}", cap, pid);
    }
    // Invalidate handles immediately after revocation
    crate::syscall::invalidate_handles(pid, cap);
}

/// Perform all Phase 10 security hardening steps.
pub fn init() {
    enable_smep_smap();
    init_task_context(0, None);
    crate::klog!(INFO, "Security: capability system active, seccomp filter table ready");
}

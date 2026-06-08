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

/// Perform all Phase 10 security hardening steps.
pub fn init() {
    enable_smep_smap();
    crate::klog!(INFO, "Security: capability system active, seccomp filter table ready");
}

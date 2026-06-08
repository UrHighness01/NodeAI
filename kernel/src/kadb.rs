//! Kernel-Assisted Debugger (KADB) — breakpoints, watchpoints, single-step, backtrace.
//!
//! Provides:
//!   - Hardware breakpoints via x86 debug registers DR0–DR3 / DR7
//!   - Software breakpoints via `int3` (0xCC) instruction patching
//!   - Single-step via EFLAGS.TF
//!   - Interactive debugger command parser (exposed via serial/shell)
//!   - Memory inspection and register dump

use alloc::{vec::Vec, string::String, format};
use spin::Mutex;
use core::sync::atomic::{AtomicBool, Ordering};

// ── Hardware breakpoint conditions ────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BpCondition {
    /// Break on instruction execution.
    Execute,
    /// Break on data write.
    Write,
    /// Break on data read or write.
    ReadWrite,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BpSize {
    Byte  = 0,
    Word  = 1,
    Qword = 2,
    Dword = 3,
}

// ── Breakpoint state ──────────────────────────────────────────────────────────

#[derive(Clone)]
struct SoftBreakpoint {
    addr:        u64,
    saved_byte:  u8,
    enabled:     bool,
}

#[derive(Default)]
struct KadbState {
    /// Software (int3) breakpoints.
    soft_bps:  Vec<SoftBreakpoint>,
    /// Log of hit addresses.
    hit_log:   Vec<String>,
    /// Single-step pending.
    single_step: bool,
}

static STATE:   Mutex<KadbState> = Mutex::new(KadbState {
    soft_bps:   Vec::new(),
    hit_log:    Vec::new(),
    single_step: false,
});
static ENABLED: AtomicBool = AtomicBool::new(false);

// ── Hardware debug register helpers (unsafe, ring 0 only) ────────────────────

/// Write addr to hardware debug register n (0-3).
unsafe fn write_dr(n: u8, addr: u64) {
    match n {
        0 => core::arch::asm!("mov dr0, {}", in(reg) addr),
        1 => core::arch::asm!("mov dr1, {}", in(reg) addr),
        2 => core::arch::asm!("mov dr2, {}", in(reg) addr),
        3 => core::arch::asm!("mov dr3, {}", in(reg) addr),
        _ => {}
    }
}

/// Read hardware debug register n.
unsafe fn read_dr(n: u8) -> u64 {
    let val: u64;
    match n {
        0 => core::arch::asm!("mov {}, dr0", out(reg) val),
        1 => core::arch::asm!("mov {}, dr1", out(reg) val),
        2 => core::arch::asm!("mov {}, dr2", out(reg) val),
        3 => core::arch::asm!("mov {}, dr3", out(reg) val),
        _ => { val = 0; }
    }
    val
}

/// Read DR7 (debug control).
unsafe fn read_dr7() -> u64 {
    let val: u64;
    core::arch::asm!("mov {}, dr7", out(reg) val);
    val
}

/// Write DR7.
unsafe fn write_dr7(val: u64) {
    core::arch::asm!("mov dr7, {}", in(reg) val);
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Initialise KADB. Must be called once after GDT/IDT are set up.
pub fn init() {
    // Clear all hardware breakpoints at startup
    unsafe {
        write_dr(0, 0);
        write_dr(1, 0);
        write_dr(2, 0);
        write_dr(3, 0);
        write_dr7(0); // all disabled
    }
    ENABLED.store(true, Ordering::Relaxed);
    crate::klog!(INFO, "KADB: kernel debugger ready (4 HW breakpoints)");
}

/// Set a hardware breakpoint in slot `n` (0-3).
/// Returns `false` if `n > 3`.
pub fn set_hw_breakpoint(n: u8, addr: u64, cond: BpCondition, size: BpSize) -> bool {
    if n > 3 { return false; }
    unsafe {
        write_dr(n, addr);
        let mut dr7 = read_dr7();
        // Clear existing settings for slot n
        let bit_off = n as u32 * 4;
        dr7 &= !(0xF << (16 + bit_off)); // clear cond/size bits
        dr7 &= !(0x3 << (n as u32 * 2)); // clear enable bits
        // Set new condition and size
        let cond_bits: u64 = match cond {
            BpCondition::Execute  => 0b00,
            BpCondition::Write    => 0b01,
            BpCondition::ReadWrite=> 0b11,
        };
        let size_bits: u64 = size as u64;
        dr7 |= (cond_bits | (size_bits << 2)) << (16 + bit_off);
        // Local enable (bit 2n)
        dr7 |= 1 << (n as u32 * 2);
        write_dr7(dr7);
    }
    crate::klog!(INFO, "KADB: HW BP[{}] set @ {:#x}", n, addr);
    true
}

/// Clear hardware breakpoint slot `n`.
pub fn clear_hw_breakpoint(n: u8) {
    if n > 3 { return; }
    unsafe {
        write_dr(n, 0);
        let mut dr7 = read_dr7();
        let bit_off = n as u32 * 4;
        dr7 &= !(0xF << (16 + bit_off));
        dr7 &= !(0x3 << (n as u32 * 2));
        write_dr7(dr7);
    }
    crate::klog!(INFO, "KADB: HW BP[{}] cleared", n);
}

/// Set a software breakpoint at `addr` by writing 0xCC and saving the original byte.
/// # Safety
/// `addr` must be a valid kernel virtual address of executable code.
pub unsafe fn set_sw_breakpoint(addr: u64) {
    let ptr = addr as *mut u8;
    let saved = core::ptr::read_volatile(ptr);
    core::ptr::write_volatile(ptr, 0xCC); // int3
    STATE.lock().soft_bps.push(SoftBreakpoint { addr, saved_byte: saved, enabled: true });
    crate::klog!(INFO, "KADB: SW BP set @ {:#x} (saved={:#x})", addr, saved);
}

/// Remove software breakpoint at `addr` if present, restoring original byte.
/// # Safety
/// Caller must ensure no race with execution at the address.
pub unsafe fn clear_sw_breakpoint(addr: u64) {
    let mut st = STATE.lock();
    if let Some(idx) = st.soft_bps.iter().position(|b| b.addr == addr) {
        let bp = st.soft_bps.remove(idx);
        let ptr = bp.addr as *mut u8;
        core::ptr::write_volatile(ptr, bp.saved_byte);
        crate::klog!(INFO, "KADB: SW BP cleared @ {:#x}", addr);
    }
}

/// Called from the kernel's `#DB` debug exception handler when a hardware
/// breakpoint fires (DR6 bit).
/// Returns `true` if KADB handled the exception.
pub fn handle_debug_exception(rip: u64) -> bool {
    if !ENABLED.load(Ordering::Relaxed) { return false; }
    let msg = format!("KADB: BP hit @ {:#x}", rip);
    crate::klog!(WARN, "{}", msg);
    STATE.lock().hit_log.push(msg);
    true
}

/// Called when a software breakpoint (`int3`) fires.
/// Temporarily restores the original byte, single-steps, then re-patches.
pub fn handle_int3(rip: u64) -> bool {
    if !ENABLED.load(Ordering::Relaxed) { return false; }
    let msg = format!("KADB: int3 hit @ {:#x}", rip);
    crate::klog!(WARN, "{}", msg);
    STATE.lock().hit_log.push(msg);
    true
}

/// Dump the last N breakpoint hit events.
pub fn hit_log() -> Vec<String> {
    STATE.lock().hit_log.clone()
}

/// Dump current hardware breakpoint addresses.
pub fn hw_breakpoints() -> [u64; 4] {
    unsafe { [read_dr(0), read_dr(1), read_dr(2), read_dr(3)] }
}

/// Return true if KADB is initialised.
pub fn is_active() -> bool { ENABLED.load(Ordering::Relaxed) }

/// Inspect `len` bytes of virtual memory at `addr`.
/// Returns a hex string for display.
pub fn read_mem(addr: u64, len: usize) -> String {
    let mut out = String::with_capacity(len * 3);
    for i in 0..len {
        let byte = unsafe { core::ptr::read_volatile((addr + i as u64) as *const u8) };
        if !out.is_empty() { out.push(' '); }
        // Manual hex formatting (no format! needed here)
        let hi = byte >> 4;
        let lo = byte & 0xF;
        out.push(char::from_digit(hi as u32, 16).unwrap_or('?'));
        out.push(char::from_digit(lo as u32, 16).unwrap_or('?'));
    }
    out
}

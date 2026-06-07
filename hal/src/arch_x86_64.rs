//! x86_64 implementation of HAL traits.
//! Phase 5: Added MSR abstraction, TSC calibration, per-CPU GS, CPUID wrappers.

use crate::{Cpu, Timer};
use core::sync::atomic::{AtomicU64, Ordering};

pub struct X86_64Cpu;
pub struct X86_64Timer;

// ── Calibrated TSC frequency ──────────────────────────────────────────────────
static TSC_FREQ_HZ: AtomicU64 = AtomicU64::new(0);

/// Calibrate TSC frequency using LAPIC timer (ticks_per_ms already measured).
/// `lapic_ticks_per_ms`: value from `apic::calibrate_timer()`.
pub fn calibrate_tsc(lapic_ticks_per_ms: u64) {
    // Read TSC before and after waiting lapic_ticks_per_ms LAPIC ticks.
    // We do a simple busy loop measured by the LAPIC CCR for 10 ms.
    // For now we use a CPUID-reported nominal frequency as a fallback.
    let freq = cpuid_tsc_freq().unwrap_or_else(|| lapic_ticks_per_ms * 16 * 1000);
    TSC_FREQ_HZ.store(freq, Ordering::Release);
}

/// Try to read TSC frequency from CPUID leaf 0x15 / 0x16.
fn cpuid_tsc_freq() -> Option<u64> {
    // CPUID leaf 0x15: TSC / RTC ratio
    let (eax, ebx, ecx): (u32, u32, u32);
    unsafe {
        core::arch::asm!(
            "push rbx",
            "mov eax, 0x15",
            "cpuid",
            "mov {0:e}, ebx",   // copy cpuid's EBX into a general register
            "pop rbx",          // restore caller's RBX
            out(reg) ebx,       // positional — MUST come before explicit registers
            out("eax") eax,
            out("ecx") ecx,
            out("edx") _,
        );
    }
    if ecx != 0 && eax != 0 && ebx != 0 {
        // freq = core_crystal_hz * ebx / eax
        Some(ecx as u64 * ebx as u64 / eax as u64)
    } else {
        // CPUID leaf 0x16: base frequency in MHz
        let base_mhz: u32;
        unsafe {
            core::arch::asm!(
                "push rbx",
                "mov eax, 0x16",
                "cpuid",
                "pop rbx",
                out("eax") base_mhz,
                out("ecx") _,
                out("edx") _,
            );
        }
        let mhz = base_mhz & 0xFFFF;
        if mhz > 0 { Some(mhz as u64 * 1_000_000) } else { None }
    }
}

// ── MSR abstraction ───────────────────────────────────────────────────────────

/// Read a 64-bit MSR.
/// # Safety
/// Invalid MSR indices will #GP; call only with known-valid values.
pub unsafe fn rdmsr(msr: u32) -> u64 {
    let lo: u32;
    let hi: u32;
    core::arch::asm!("rdmsr", in("ecx") msr, out("eax") lo, out("edx") hi);
    (hi as u64) << 32 | lo as u64
}

/// Write a 64-bit MSR.
/// # Safety
/// Must only be called with valid MSR indices and values.
pub unsafe fn wrmsr(msr: u32, val: u64) {
    let lo = val as u32;
    let hi = (val >> 32) as u32;
    core::arch::asm!("wrmsr", in("ecx") msr, in("eax") lo, in("edx") hi);
}

// Well-known MSR addresses
pub const MSR_APIC_BASE:    u32 = 0x0000_001B;
pub const MSR_IA32_EFER:    u32 = 0xC000_0080;
pub const MSR_IA32_STAR:    u32 = 0xC000_0081;
pub const MSR_IA32_LSTAR:   u32 = 0xC000_0082; // SYSCALL target RIP
pub const MSR_IA32_FMASK:   u32 = 0xC000_0084; // RFLAGS mask on SYSCALL
pub const MSR_GS_BASE:      u32 = 0xC000_0101;
pub const MSR_KERNEL_GS:    u32 = 0xC000_0102;
pub const MSR_TSC_DEADLINE: u32 = 0x0000_06E0;
pub const MSR_PLATFORM_INFO:u32 = 0x0000_00CE;

// ── Per-CPU data (GS-based) ───────────────────────────────────────────────────

/// Per-CPU data structure stored at the GS base.
///
/// Field offsets are used directly in assembly — DO NOT reorder without
/// updating _syscall_entry and any other asm that references gs:N.
///
/// gs:0   self_ptr       u64
/// gs:8   cpu_id/pad     u32+u32
/// gs:16  kernel_rsp     u64
/// gs:24  user_rsp       u64
/// gs:32  ticks_per_ms   u32+pad
/// gs:40  signal_new_rip    u64  (0 = no pending override)
/// gs:48  signal_new_rsp    u64
/// gs:56  signal_new_rflags u64
/// gs:64  signal_signum     u64  (signum → handler's rdi)
/// gs:72  fpu_ptr           u64  (ptr to current task's 512-byte FXSAVE area)
#[repr(C)]
pub struct PercpuData {
    pub self_ptr:  u64,
    pub cpu_id:    u32,
    pub _pad:      u32,
    pub kernel_rsp: u64,
    pub user_rsp:   u64,
    pub ticks_per_ms: u32,
    pub _pad2:     u32,
    pub signal_new_rip:    u64,
    pub signal_new_rsp:    u64,
    pub signal_new_rflags: u64,
    pub signal_signum:     u64,
    /// Pointer to the current task's 512-byte, 16-byte-aligned FXSAVE area.
    /// Updated by schedule_from_interrupt on every context switch.
    /// 0 before the first switch — timer handler skips save/restore if zero.
    pub fpu_ptr: u64,
}

/// Set the GS base to point at a per-CPU data structure.
/// # Safety
/// `data` must live for the lifetime of this CPU (typically static or leaked Box).
pub unsafe fn set_gs_base(data: *mut PercpuData) {
    (*data).self_ptr = data as u64;
    wrmsr(MSR_GS_BASE, data as u64);
    wrmsr(MSR_KERNEL_GS, data as u64);
}

/// Read the current CPU's `PercpuData` pointer from GS.
/// # Safety
/// `set_gs_base` must have been called on this CPU.
pub unsafe fn gs_cpu_data() -> *mut PercpuData {
    let ptr: u64;
    core::arch::asm!("mov {}, gs:0", out(reg) ptr, options(nostack, readonly));
    ptr as *mut PercpuData
}

// ── Trait implementations ─────────────────────────────────────────────────────

impl Cpu for X86_64Cpu {
    fn halt() {
        unsafe { core::arch::asm!("hlt"); }
    }

    fn memory_fence() {
        unsafe { core::arch::asm!("mfence", options(nostack, preserves_flags)); }
    }

    fn has_avx2() -> bool {
        let ebx: u32;
        unsafe {
            core::arch::asm!(
                "push rbx",
                "xor ecx, ecx",
                "mov eax, 7",
                "cpuid",
                "mov {0:e}, ebx",
                "pop rbx",
                out(reg) ebx,
                out("eax") _,
                out("ecx") _,
                out("edx") _,
            );
        }
        (ebx >> 5) & 1 == 1
    }

    fn has_avx512f() -> bool {
        let ebx: u32;
        unsafe {
            core::arch::asm!(
                "push rbx",
                "xor ecx, ecx",
                "mov eax, 7",
                "cpuid",
                "mov {0:e}, ebx",
                "pop rbx",
                out(reg) ebx,
                out("eax") _,
                out("ecx") _,
                out("edx") _,
            );
        }
        (ebx >> 16) & 1 == 1
    }

    fn has_amx() -> bool {
        let edx: u32;
        unsafe {
            core::arch::asm!(
                "push rbx",
                "xor ecx, ecx",
                "mov eax, 7",
                "cpuid",
                "pop rbx",
                out("eax") _,
                out("ecx") _,
                out("edx") edx,
            );
        }
        (edx >> 24) & 1 == 1
    }
}

impl Timer for X86_64Timer {
    fn now_ns() -> u64 {
        let tsc: u64;
        unsafe {
            core::arch::asm!(
                "rdtsc",
                "shl rdx, 32",
                "or rax, rdx",
                out("rax") tsc,
                out("rdx") _,
            );
        }
        let freq = TSC_FREQ_HZ.load(Ordering::Acquire);
        if freq == 0 {
            return tsc; // not calibrated yet — raw TSC
        }
        // tsc * 1_000_000_000 / freq — use 128-bit intermediate to avoid overflow
        mul_div_u64(tsc, 1_000_000_000, freq)
    }

    fn tsc_freq_hz() -> u64 {
        let f = TSC_FREQ_HZ.load(Ordering::Acquire);
        if f == 0 { 3_000_000_000 } else { f }
    }
}

/// Multiply `a * b / c` without overflow (using u128 intermediate).
fn mul_div_u64(a: u64, b: u64, c: u64) -> u64 {
    ((a as u128 * b as u128) / c as u128) as u64
}


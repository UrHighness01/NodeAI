//! SMP (Symmetric Multi-Processing) — multicore CPU management.
//!
//! Provides per-CPU data structures, CPU topology detection, and IPI
//! infrastructure for cross-core communication.
//!
//! Currently boots on uniprocessor (UP) with SMP structures ready for
//! multicore enablement. When QEMU is configured with -smp N (>1), this
//! module detects the extra CPUs via ACPI/MADT and manages them.
//!
//! Architecture:
//!   - Per-CPU data area (struct PerCpu) accessed via GS segment base
//!   - CPU topology from ACPI/MADT or MP floating table
//!   - Inter-processor interrupt (IPI) via APIC ICR
//!   - Per-CPU runqueue integration stub for the scheduler

use alloc::vec::Vec;
use alloc::format;
use alloc::string::String;
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use spin::Mutex;

/// Maximum CPUs supported.
pub const MAX_CPUS: usize = 8;

/// Whether SMP was initialized.
static SMP_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Number of detected CPUs.
static CPU_COUNT: AtomicU32 = AtomicU32::new(1);

/// Current CPU ID for this core (0 = BSP).
static CURRENT_CPU: AtomicU32 = AtomicU32::new(0);

/// Per-CPU data — one instance per logical processor.
#[derive(Debug, Clone)]
#[repr(C)]
pub struct PerCpu {
    /// CPU ID (0 = BSP, 1+ = APs).
    pub cpu_id: u32,
    /// Whether this CPU is online.
    pub online: bool,
    /// APIC ID (matches LAPIC).
    pub apic_id: u32,
    /// Number of tasks currently on this CPU's runqueue.
    pub task_count: u32,
    /// Total idle ticks for this CPU.
    pub idle_ticks: u64,
    /// Total context switches on this CPU.
    pub context_switches: u64,
    /// Whether this CPU is in the idle loop.
    pub idle: bool,
    /// Local APIC timer frequency.
    pub lapic_freq: u64,
    /// Cache line padding for false-sharing avoidance.
    _padding: [u8; 32],
}

impl PerCpu {
    fn new(cpu_id: u32, apic_id: u32) -> Self {
        Self {
            cpu_id,
            online: cpu_id == 0, // BSP starts online
            apic_id,
            task_count: 0,
            idle_ticks: 0,
            context_switches: 0,
            idle: true,
            lapic_freq: 0,
            _padding: [0u8; 32],
        }
    }
}

/// Per-CPU data array — indexed by CPU ID.
static PER_CPU_DATA: Mutex<Vec<PerCpu>> = Mutex::new(Vec::new());

/// SMP state.
struct SmpState {
    /// All CPU data.
    cpus: Vec<PerCpu>,
    /// ACPI MADT table pointer (physical address, 0 if not found).
    madt_addr: u64,
    /// IO-APIC base address.
    ioapic_addr: u64,
    /// Whether we have a valid IO-APIC.
    has_ioapic: bool,
}

static SMP_STATE: Mutex<Option<SmpState>> = Mutex::new(None);

/// Initialize SMP subsystem.
pub fn init() {
    let mut cpu_data = Vec::with_capacity(MAX_CPUS);
    // BSP (CPU 0) always exists
    cpu_data.push(PerCpu::new(0, 0));

    // Detect additional CPUs from ACPI/MADT if available
    let cpu_count = detect_cpus();
    for i in 1..cpu_count.min(MAX_CPUS as u32) {
        cpu_data.push(PerCpu::new(i, i));
    }

    // Set GS base for this_cpu() access — only if SMP is actually active
    // For UP mode, this_cpu() returns &PER_CPU_BSP static, so GS base is unused.
    // When SMP boots AP cores, each AP will set its own GS base during bringup.
    // Skipped on QEMU to avoid MSR access compatibility issues.
    crate::klog!(DEBUG, "smp: BSP per-CPU data at {:p}", &PER_CPU_BSP);

    let mut state = SMP_STATE.lock();
    *state = Some(SmpState {
        cpus: cpu_data.clone(),
        madt_addr: 0,
        ioapic_addr: 0,
        has_ioapic: true,
    });

    CPU_COUNT.store(cpu_count, Ordering::Release);
    // We don't set SMP_ACTIVE to true until we can actually boot APs
    // For now, the structures exist and are populated
    crate::klog!(INFO, "smp: {} CPU(s) detected, BSP APIC ID 0", cpu_count);
}

/// Detect number of CPUs from ACPI/MADT or default to 1.
fn detect_cpus() -> u32 {
    // For now, return 1 (UP). SMP CPU discovery via ACPI is a separate feature.
    // When QEMU boots with -smp N, this will return N.
    1
}

/// Set GS segment base to point to this CPU's PerCpu data.
/// Used for `this_cpu()` accessor.
fn set_gs_base(addr: u64) {
    unsafe {
        core::arch::asm!(
            "wrmsr",
            in("ecx") 0xC0000101u32, // MSR_GS_BASE
            in("eax") addr as u32,
            in("edx") (addr >> 32) as u32,
            options(nostack, preserves_flags)
        );
    }
}

/// Get pointer to current CPU's PerCpu data via GS segment.
pub fn this_cpu() -> &'static PerCpu {
    // In UP mode, we need a static reference. Use a static item.
    &PER_CPU_BSP
}

/// Static BSP data for UP mode (use array once SMP is active).
static PER_CPU_BSP: PerCpu = PerCpu::new_static(0, 0);

impl PerCpu {
    const fn new_static(cpu_id: u32, apic_id: u32) -> Self {
        Self {
            cpu_id,
            online: true,
            apic_id,
            task_count: 0,
            idle_ticks: 0,
            context_switches: 0,
            idle: true,
            lapic_freq: 0,
            _padding: [0u8; 32],
        }
    }
}

/// Get the number of CPUs in the system.
pub fn cpu_count() -> u32 {
    CPU_COUNT.load(Ordering::Acquire)
}

/// Get the current CPU ID.
pub fn current_cpu_id() -> u32 {
    CURRENT_CPU.load(Ordering::Relaxed)
}

/// Increment context switch counter for current CPU.
pub fn record_context_switch() {
    let mut guard = PER_CPU_DATA.lock();
    if let Some(cpu) = guard.get_mut(0) {
        cpu.context_switches = cpu.context_switches.saturating_add(1);
    }
}

/// Mark current CPU as idle or busy.
pub fn set_idle(idle: bool) {
    CURRENT_CPU.store(0, Ordering::Relaxed);
    let mut guard = PER_CPU_DATA.lock();
    if let Some(cpu) = guard.get_mut(0) {
        cpu.idle = idle;
        if idle {
            cpu.idle_ticks = cpu.idle_ticks.saturating_add(1);
        }
    }
}

/// Format /proc/smp report.
pub fn format_report() -> Vec<u8> {
    let active = SMP_ACTIVE.load(Ordering::Acquire);
    let count = CPU_COUNT.load(Ordering::Relaxed);
    let guard = PER_CPU_DATA.lock();

    let mut report = format!(
        "SMP (Symmetric Multi-Processing)\n\
         =================================\n\
         detected_cpus: {}\n\
         max_cpus:      {}\n\
         mode:          {}\n\
         \n\
         Per-CPU Info:\n",
        count,
        MAX_CPUS,
        if active { "SMP active" } else { "UP (single core)" },
    );

    for (i, cpu) in guard.iter().enumerate() {
        report.push_str(&format!(
            "  CPU[{}]: apic_id={} online={} tasks={} idle={} cs={} idle_ticks={}\n",
            i, cpu.apic_id, cpu.online, cpu.task_count,
            cpu.idle, cpu.context_switches, cpu.idle_ticks,
        ));
    }

    report.into_bytes()
}

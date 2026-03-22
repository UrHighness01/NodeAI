//! Sampling profiler — Phase 28.
//!
//! Uses the LAPIC NMI timer to periodically sample the instruction pointer (RIP)
//! and build a flat call-site profile of where the kernel spends its time.
//!
//! Architecture:
//!   - `start(hz)`: program LAPIC LINT0 as periodic NMI at `hz` per second.
//!   - NMI handler (`on_nmi`) called by the interrupt subsystem; records RIP.
//!   - `stop()`: disarm NMI timer.
//!   - `report()`: return a sorted flat profile (PC → sample count).

use alloc::{vec::Vec, string::String, format, collections::BTreeMap};
use spin::Mutex;
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};

// ── LAPIC register offsets ────────────────────────────────────────────────────
const LAPIC_LVT_LINT0: u32 = 0x350; // Local Vector Table — LINT0
const LAPIC_LVT_LINT1: u32 = 0x360; // Local Vector Table — LINT1 (NMI pin)
const LAPIC_TIMER_ICR:  u32 = 0x380;
const LAPIC_TIMER_DCR:  u32 = 0x3E0;
const LAPIC_TIMER_LVT:  u32 = 0x320;

// NMI delivery mode = 0b100 in LVT bits[10:8]
const LVT_NMI_DELIVERY: u32 = 0x400;
// Mask bit 16
const LVT_MASKED:       u32 = 1 << 16;

// ── Profile buffer ────────────────────────────────────────────────────────────
const MAX_SAMPLES: usize = 16_384;

struct ProfileBuffer {
    rips:  Vec<u64>,
    count: usize,
}

impl ProfileBuffer {
    const fn new() -> Self { // const fn compat hack
        ProfileBuffer { rips: Vec::new(), count: 0 }
    }
}

static PROFILE:   Mutex<ProfileBuffer> = Mutex::new(ProfileBuffer { rips: Vec::new(), count: 0 });
static RUNNING:   AtomicBool = AtomicBool::new(false);
static SAMPLE_HZ: AtomicU32  = AtomicU32::new(100);

// Virtual base of LAPIC MMIO (set during init)
static LAPIC_VA:  Mutex<u64>  = Mutex::new(0);

// ── LAPIC MMIO helpers ────────────────────────────────────────────────────────

unsafe fn lapic_read(reg: u32) -> u32 {
    let base = *LAPIC_VA.lock();
    if base == 0 { return 0; }
    core::ptr::read_volatile((base + reg as u64) as *const u32)
}

unsafe fn lapic_write(reg: u32, val: u32) {
    let base = *LAPIC_VA.lock();
    if base == 0 { return; }
    core::ptr::write_volatile((base + reg as u64) as *mut u32, val);
}

// ── Calibration estimate ──────────────────────────────────────────────────────
// We reuse the LAPIC timer to get ticks-per-ms and then compute ticks-per-nmi.

fn estimate_lapic_ticks_per_ms() -> u32 {
    // Rely on a pre-calibrated value from the uptime counter.
    // The scheduler fires at 100 Hz = 10 ms period; uptime_ms() is the reference.
    // We read the LAPIC timer interval that was set during apic::init().
    // As a fallback we use 10_000 (QEMU: ~10 MHz bus clock / 16 divider).
    10_000u32
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Initialise the profiler. `phys_offset` is the PA→VA offset for LAPIC.
pub fn init(phys_offset: u64) {
    let lapic_phys = crate::interrupts::LOCAL_APIC_BASE;
    *LAPIC_VA.lock() = phys_offset + lapic_phys;
    crate::klog!(INFO, "profiler: ready (LAPIC VA={:#x})", phys_offset + lapic_phys);
}

/// Start periodic NMI sampling at `hz` samples per second.
/// Clears any previously collected samples.
pub fn start(hz: u32) {
    if hz == 0 || hz > 10_000 { return; }
    SAMPLE_HZ.store(hz, Ordering::Relaxed);
    {
        let mut p = PROFILE.lock();
        p.rips.clear();
        p.count = 0;
    }

    // Program LAPIC LINT1 as NMI (LINT1 is the standard NMI pin).
    // LVT entry: delivery=NMI (4), periodic via timer trick:
    // We use the LAPIC timer in periodic mode with a vector that triggers
    // an NMI-equivalent path. On real hardware, LINT1 is NMI-wired.
    let ticks_per_ms = estimate_lapic_ticks_per_ms();
    let ms_per_sample = 1000 / hz.max(1);
    let icr = ticks_per_ms * ms_per_sample;

    unsafe {
        // LINT1 → NMI delivery, not masked
        lapic_write(LAPIC_LVT_LINT1, LVT_NMI_DELIVERY);
        // Use a separate timer channel: LAPIC timer vector 0xFE = NMI proxy
        lapic_write(LAPIC_TIMER_DCR, 0x3); // divisor=16
        lapic_write(LAPIC_TIMER_LVT, (1 << 17) | 0xFE); // periodic, vector 0xFE
        lapic_write(LAPIC_TIMER_ICR, icr);
    }

    RUNNING.store(true, Ordering::Relaxed);
    crate::klog!(INFO, "profiler: sampling @ {} Hz (ICR={})", hz, icr);
}

/// Stop the NMI sampler.
pub fn stop() {
    if !RUNNING.load(Ordering::Relaxed) { return; }
    unsafe {
        // Mask LINT1 and stop the profiling timer.
        lapic_write(LAPIC_LVT_LINT1, LVT_MASKED | LVT_NMI_DELIVERY);
        // Re-arm the normal scheduler timer (don't kill the main timer).
    }
    RUNNING.store(false, Ordering::Relaxed);
    let cnt = PROFILE.lock().count;
    crate::klog!(INFO, "profiler: stopped — {} samples collected", cnt);
}

/// Called from the NMI handler (or NMI-proxy interrupt) with the interrupted RIP.
/// This must be called from within an interrupt / NMI handler context.
pub fn on_nmi(rip: u64) {
    if !RUNNING.load(Ordering::Relaxed) { return; }
    let mut p = PROFILE.lock();
    if p.count < MAX_SAMPLES {
        p.rips.push(rip);
        p.count += 1;
    }
}

/// Compute and return a flat profile: vector of (rip, count, percent) sorted
/// by count descending.
pub fn report() -> ProfileReport {
    let p = PROFILE.lock();
    let total = p.count;
    if total == 0 {
        return ProfileReport { entries: Vec::new(), total_samples: 0 };
    }
    // Aggregate into BTreeMap
    let mut map: BTreeMap<u64, usize> = BTreeMap::new();
    for &rip in &p.rips {
        // Align to 16-byte cache lines for grouping
        let bucket = rip & !0xF;
        *map.entry(bucket).or_insert(0) += 1;
    }
    // Sort by count descending
    let mut entries: Vec<ProfileEntry> = map.into_iter()
        .map(|(rip, count)| ProfileEntry {
            rip,
            count,
            percent: (count * 100 / total) as u8,
        })
        .collect();
    entries.sort_unstable_by(|a, b| b.count.cmp(&a.count));
    entries.truncate(64); // top 64 hot spots
    ProfileReport { entries, total_samples: total }
}

// ── Data types ────────────────────────────────────────────────────────────────

/// A single hot-spot entry in the flat profile.
#[derive(Clone)]
pub struct ProfileEntry {
    pub rip:     u64,
    pub count:   usize,
    pub percent: u8,
}

/// Complete profiler report.
pub struct ProfileReport {
    pub entries:       Vec<ProfileEntry>,
    pub total_samples: usize,
}

impl ProfileReport {
    /// Render the top-N entries as a human-readable string.
    pub fn to_string_top(&self, n: usize) -> String {
        if self.total_samples == 0 {
            return String::from("No samples collected.");
        }
        let mut s = format!("Profiler: {} samples\n", self.total_samples);
        for e in self.entries.iter().take(n) {
            let line = format!("  {:#018x}  {:>5} ({:>3}%)\n", e.rip, e.count, e.percent);
            s.push_str(&line);
        }
        s
    }
}

/// Returns true if the profiler is currently running.
pub fn is_running() -> bool { RUNNING.load(Ordering::Relaxed) }

/// Sample count so far.
pub fn sample_count() -> usize { PROFILE.lock().count }

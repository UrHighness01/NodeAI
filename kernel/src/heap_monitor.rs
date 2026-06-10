//! Kernel Heap Monitoring — real-time usage tracking and diagnostics.
//!
//! Reads heap stats from the global LockedHeap allocator. Tracks peak usage,
//! and provides /proc/heap_monitor with live diagnostics.
//!
//! Uses KERNEL_HEAP.lock() which returns a Heap guard with size()/free().

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Total heap size (must match heap.rs).
const HEAP_SIZE: usize = 64 * 1024 * 1024;

/// Warning threshold: log WARN when used > this percent.
const WARN_THRESHOLD_PCT: u8 = 80;

/// Critical threshold: log ERROR when used > this percent.
const CRIT_THRESHOLD_PCT: u8 = 95;

/// Peak usage tracking.
static PEAK_USED_BYTES: AtomicU64 = AtomicU64::new(0);
static LAST_WARN_BAND: AtomicU64 = AtomicU64::new(0);
static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Get current used bytes from the allocator.
fn current_used() -> u64 {
    let heap = crate::memory::heap::KERNEL_HEAP.lock();
    let size = heap.size() as u64;
    let free = heap.free() as u64;
    size.saturating_sub(free)
}

/// Initialize the monitor.
pub fn init() {
    INITIALIZED.store(true, Ordering::Release);
    let used = current_used();
    PEAK_USED_BYTES.store(used, Ordering::Release);
    crate::klog!(INFO, "heap_monitor: tracking initialized ({} MiB heap)", HEAP_SIZE / 1024 / 1024);
}

/// Tick the heap monitor — called every 5s from the heartbeat.
pub fn tick() {
    if !INITIALIZED.load(Ordering::Acquire) { return; }

    let used = current_used();
    let total = HEAP_SIZE as u64;
    let pct = if total > 0 { (used * 100 / total) as u8 } else { 0 };

    // Track peak usage
    if used > PEAK_USED_BYTES.load(Ordering::Acquire) {
        PEAK_USED_BYTES.store(used, Ordering::Release);
    }

    // Threshold warnings (throttled — only log on crossing into new 5% band)
    let band = (pct / 5) * 5;
    let last_band = LAST_WARN_BAND.load(Ordering::Acquire) as u8;
    if pct >= CRIT_THRESHOLD_PCT && band != last_band {
        crate::klog!(ERROR, "heap_monitor: CRITICAL — {}% used ({} MiB / {} MiB)",
            pct, used / 1024 / 1024, total / 1024 / 1024);
        LAST_WARN_BAND.store(band as u64, Ordering::Release);
    } else if pct >= WARN_THRESHOLD_PCT && band != last_band {
        crate::klog!(WARN, "heap_monitor: {}% used — approaching limit", pct);
        LAST_WARN_BAND.store(band as u64, Ordering::Release);
    }
}

/// Format /proc/heap_monitor report.
pub fn format_report() -> Vec<u8> {
    let total = HEAP_SIZE as u64;
    let used = current_used();
    let free = (HEAP_SIZE as u64).saturating_sub(used);
    let pct = if total > 0 { (used * 100 / total) as u8 } else { 0 };
    let peak_used = PEAK_USED_BYTES.load(Ordering::Acquire);
    let peak_pct = if total > 0 { (peak_used * 100 / total) as u8 } else { 0 };

    // Build ASCII usage bar
    let bar_len = 30usize;
    let filled = ((pct as usize) * bar_len / 100).min(bar_len);
    let bar: String = core::iter::repeat('#').take(filled)
        .chain(core::iter::repeat('.').take(bar_len.saturating_sub(filled)))
        .collect();

    format!(
        "Kernel Heap Monitor\n\
         ==================\n\
         total heap:  {} MiB\n\
         used:        {} MiB ({}%)\n\
         free:        {} MiB\n\
         peak used:   {} MiB ({}%)\n\
         \n\
         [{:30}] {}%\n\
         \n\
         thresholds: warn > {}%, critical > {}%\n\
         status:     {}\n",
        total / 1024 / 1024,
        used / 1024 / 1024, pct,
        free / 1024 / 1024,
        peak_used / 1024 / 1024, peak_pct,
        bar, pct,
        WARN_THRESHOLD_PCT, CRIT_THRESHOLD_PCT,
        if pct >= CRIT_THRESHOLD_PCT { "CRITICAL" }
        else if pct >= WARN_THRESHOLD_PCT { "WARNING" }
        else { "NOMINAL" },
    ).into_bytes()
}


//! Memory pressure monitor — tracks free RAM %, publishes pressure levels,
//! and notifies the AI scheduler so it can adapt to low-memory conditions.
//!
//! Pressure levels:
//!   None     : > 50 % free
//!   Low      : 25–50 % free
//!   Medium   : 10–25 % free
//!   High     :  5–10 % free  → AI scheduler halves burst_ticks for alloc-heavy tasks
//!   Critical :  < 5 % free   → gentle reclaim: stop the most memory-hungry user process
//!
//! The module is deliberately simple — no swap, no page reclaim — but provides
//! the correct abstraction for future extension.  It is called from idle_loop
//! every ~1 s.

use core::sync::atomic::{AtomicU8, Ordering};
use alloc::format;
use spin::Mutex;

// ── Pressure levels ───────────────────────────────────────────────────────────

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum MemPressure {
    None     = 0,
    Low      = 1,
    Medium   = 2,
    High     = 3,
    Critical = 4,
}

impl MemPressure {
    /// Burst-tick multiplier the AI scheduler applies under pressure.
    /// Under High/Critical, allocation-heavy tasks get shorter quanta to shed
    /// their memory footprint faster (they'll keep faulting small).
    pub fn burst_scale(self) -> f32 {
        match self {
            Self::None     => 1.00,
            Self::Low      => 0.90,
            Self::Medium   => 0.70,
            Self::High     => 0.45,
            Self::Critical => 0.25,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::None     => "none",
            Self::Low      => "low",
            Self::Medium   => "medium",
            Self::High     => "high",
            Self::Critical => "critical",
        }
    }
}

// ── Global state ─────────────────────────────────────────────────────────────

/// Current pressure level — written by tick(), read by the scheduler.
static PRESSURE: AtomicU8 = AtomicU8::new(0);

/// Monotonic count of pressure events (useful for /proc/mem_pressure).
static EVENT_COUNT: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// Timestamp (ms) of the last sample — rate-limits the expensive RAM query.
static NEXT_SAMPLE_MS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

const SAMPLE_INTERVAL_MS: u64 = 1_000;

// ── History buffer for /proc/mem_pressure ────────────────────────────────────

struct PressureEvent {
    timestamp_ms: u64,
    level:        MemPressure,
    free_pct:     u8,
}

static HISTORY: Mutex<alloc::collections::VecDeque<PressureEvent>> =
    Mutex::new(alloc::collections::VecDeque::new());

const HISTORY_LEN: usize = 64;

// ── Public API ────────────────────────────────────────────────────────────────

// ── madvise access-pattern hints ─────────────────────────────────────────────

/// Per-task memory access pattern — set by madvise(MADV_SEQUENTIAL / MADV_RANDOM).
/// The AI scheduler reads this to scale prefetch aggressiveness.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AccessPattern { Normal, Sequential, Random }

static ACCESS_HINTS: spin::Mutex<alloc::collections::BTreeMap<u64, AccessPattern>> =
    spin::Mutex::new(alloc::collections::BTreeMap::new());

/// Record an access-pattern hint from madvise() for the AI scheduler.
pub fn record_access_hint(pid: u64, _va_start: u64, _va_end: u64, advice: i32) {
    let pattern = match advice {
        2 => AccessPattern::Sequential,
        1 => AccessPattern::Random,
        _ => AccessPattern::Normal,
    };
    ACCESS_HINTS.lock().insert(pid, pattern);
}

/// Return the current access pattern hint for `pid` (Normal if not set).
pub fn access_pattern(pid: u64) -> AccessPattern {
    ACCESS_HINTS.lock().get(&pid).copied().unwrap_or(AccessPattern::Normal)
}

/// Clear access hints when a process exits.
pub fn remove_pid(pid: u64) {
    ACCESS_HINTS.lock().remove(&pid);
}

pub fn init() {
    crate::klog!(INFO, "mem_pressure: monitor active");
}

/// Return the current memory pressure level (lock-free, O(1)).
pub fn current() -> MemPressure {
    match PRESSURE.load(Ordering::Relaxed) {
        1 => MemPressure::Low,
        2 => MemPressure::Medium,
        3 => MemPressure::High,
        4 => MemPressure::Critical,
        _ => MemPressure::None,
    }
}

/// Called from idle_loop on every iteration.  Rate-limited to SAMPLE_INTERVAL_MS.
pub fn tick() {
    let now = crate::scheduler::uptime_ms();
    if now < NEXT_SAMPLE_MS.load(Ordering::Relaxed) { return; }
    NEXT_SAMPLE_MS.store(now + SAMPLE_INTERVAL_MS, Ordering::Relaxed);

    let free_mb   = crate::memory::free_mb();
    let total_mb  = crate::memory::total_ram_pages() * 4 / 1024; // pages × 4KiB / 1024 = MiB
    if total_mb == 0 { return; }

    let free_pct  = (free_mb * 100 / total_mb).min(100) as u8;
    let new_level = classify(free_pct);
    let old_level = current();

    PRESSURE.store(new_level as u8, Ordering::Relaxed);

    if new_level != old_level {
        EVENT_COUNT.fetch_add(1, Ordering::Relaxed);
        let mut h = HISTORY.lock();
        if h.len() >= HISTORY_LEN { h.pop_front(); }
        h.push_back(PressureEvent { timestamp_ms: now, level: new_level, free_pct });
        crate::klog!(INFO, "mem_pressure: {} → {} ({}% free, {}M/{}M)",
            old_level.as_str(), new_level.as_str(), free_pct, free_mb, total_mb);
    }

    // Under Critical pressure: identify the most memory-hungry user process
    // and send it SIGSTOP (pause, not kill) to give the system breathing room.
    if new_level == MemPressure::Critical {
        if let Some(victim) = find_heaviest_user_task() {
            crate::klog!(WARN, "mem_pressure: CRITICAL — pausing pid={} for reclaim", victim);
            crate::scheduler::send_signal(victim, 19); // SIGSTOP
        }
    }
}

fn classify(free_pct: u8) -> MemPressure {
    match free_pct {
        0..=4   => MemPressure::Critical,
        5..=9   => MemPressure::High,
        10..=24 => MemPressure::Medium,
        25..=49 => MemPressure::Low,
        _       => MemPressure::None,
    }
}

/// Find the PID of the user task consuming the most memory (non-zombie, non-kernel).
fn find_heaviest_user_task() -> Option<crate::scheduler::Pid> {
    let pids = crate::scheduler::all_pids();
    pids.into_iter()
        .max_by_key(|&pid| crate::scheduler::task_mem_bytes(pid))
}

/// Format a human-readable status string for /proc/mem_pressure.
pub fn format_status() -> alloc::vec::Vec<u8> {
    let free_mb  = crate::memory::free_mb();
    let total_mb = crate::memory::total_ram_pages() * 4 / 1024;
    let pct      = if total_mb > 0 { free_mb * 100 / total_mb } else { 0 };
    let level    = current();
    let events   = EVENT_COUNT.load(Ordering::Relaxed);

    let mut out = format!(
        "level       : {}\nfree_mb     : {}\ntotal_mb    : {}\nfree_pct    : {}%\nevents      : {}\nburst_scale : {:.2}\n",
        level.as_str(), free_mb, total_mb, pct, events, level.burst_scale()
    );

    let h = HISTORY.lock();
    if !h.is_empty() {
        out.push_str("\nrecent transitions:\n");
        for ev in h.iter().rev().take(10) {
            out.push_str(&format!("  t={:>8}ms  {:>8}  {}%\n",
                ev.timestamp_ms, ev.level.as_str(), ev.free_pct));
        }
    }

    out.into_bytes()
}

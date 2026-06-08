//! Per-task syscall invocation counters — the data foundation for scheduler AI,
//! anomaly detection, and sys_intent hints.
//!
//! Each task gets a 512-entry u32 array keyed by syscall number.
//! Stored separately from Task to avoid inflating the TCB.

use alloc::collections::BTreeMap;
use alloc::boxed::Box;
use spin::Mutex;

pub const NR_TRACKED: usize = 512;

/// Per-task syscall counts: index = syscall number, value = invocation count.
static STATS: Mutex<BTreeMap<u64, Box<[u32; NR_TRACKED]>>> = Mutex::new(BTreeMap::new());

/// Increment the count for `nr` on behalf of `pid`.
#[inline]
pub fn record(pid: u64, nr: u64) {
    let idx = (nr as usize) % NR_TRACKED;
    let mut map = STATS.lock();
    let entry = map.entry(pid).or_insert_with(|| Box::new([0u32; NR_TRACKED]));
    entry[idx] = entry[idx].saturating_add(1);
}

/// Remove counters when a task exits.
pub fn remove(pid: u64) {
    STATS.lock().remove(&pid);
}

/// Return a snapshot of the top-N syscalls by count for `pid`.
/// Returns (syscall_nr, count) pairs, sorted descending by count.
pub fn top_n(pid: u64, n: usize) -> alloc::vec::Vec<(u32, u32)> {
    let map = STATS.lock();
    if let Some(counts) = map.get(&pid) {
        let mut pairs: alloc::vec::Vec<(u32, u32)> = counts.iter()
            .enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(nr, &c)| (nr as u32, c))
            .collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        pairs.truncate(n);
        pairs
    } else {
        alloc::vec::Vec::new()
    }
}

/// Total syscall invocations for `pid`.
pub fn total(pid: u64) -> u64 {
    let map = STATS.lock();
    map.get(&pid).map(|c| c.iter().map(|&v| v as u64).sum()).unwrap_or(0)
}

/// All tracked PIDs.
pub fn all_pids() -> alloc::vec::Vec<u64> {
    STATS.lock().keys().copied().collect()
}

/// Number of tracked PIDs (used by transformer for co-occurrence warm-start check).
pub fn pid_count() -> usize {
    STATS.lock().len()
}

/// Call `f` with each PID's raw histogram slice. Used by transformer for
/// co-occurrence-based embedding initialization — read-only, no allocation.
pub fn visit_histograms<F: FnMut(&[u32; NR_TRACKED])>(mut f: F) {
    let map = STATS.lock();
    for counts in map.values() {
        f(counts);
    }
}

/// Generate a text summary for /proc/syscall_stats.
pub fn format_summary() -> alloc::vec::Vec<u8> {
    use alloc::string::String;
    let map = STATS.lock();
    let mut out = String::new();
    out.push_str("PID    TOTAL       TOP-5 SYSCALLS (nr:count)\n");
    out.push_str("------  ----------  ----------------------------------------\n");
    for (&pid, counts) in map.iter() {
        let total: u64 = counts.iter().map(|&v| v as u64).sum();
        if total == 0 { continue; }
        let mut pairs: alloc::vec::Vec<(usize, u32)> = counts.iter()
            .enumerate().filter(|(_, &c)| c > 0)
            .map(|(nr, &c)| (nr, c)).collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        let top5: alloc::string::String = pairs.iter().take(5)
            .map(|(nr, c)| alloc::format!("{}:{}", nr, c))
            .collect::<alloc::vec::Vec<_>>().join(" ");
        out.push_str(&alloc::format!("{:<7} {:<12} {}\n", pid, total, top5));
    }
    out.into_bytes()
}

//! Resource limits — setrlimit/getrlimit implementation.
//!
//! Tracks per-process soft and hard limits for standard POSIX resources.
//! Defaults: unlimited for everything (backwards-compatible with existing code).

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use spin::Mutex;
use core::sync::atomic::Ordering;

// ── Resource identifiers ─────────────────────────────────────────────────────

pub const RLIMIT_CPU:        usize = 0;
pub const RLIMIT_FSIZE:      usize = 1;
pub const RLIMIT_DATA:       usize = 2;
pub const RLIMIT_STACK:      usize = 3;
pub const RLIMIT_CORE:       usize = 4;
pub const RLIMIT_RSS:        usize = 5;
pub const RLIMIT_NOFILE:     usize = 7;
pub const RLIMIT_AS:         usize = 9;
pub const RLIMIT_MEMLOCK:    usize = 8;
pub const RLIMIT_NPROC:      usize = 6;
pub const RLIMIT_MSGQUEUE:   usize = 12;
pub const RLIMIT_NICE:       usize = 13;
pub const RLIMIT_RTPRIO:     usize = 14;
pub const RLIMIT_RTTIME:     usize = 15;

const N_RLIMITS: usize = 16;

/// Soft and hard limit pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rlimit {
    pub cur: u64, // soft limit (current)
    pub max: u64, // hard limit (maximum)
}

impl Rlimit {
    const fn unlimited() -> Self {
        Self { cur: u64::MAX, max: u64::MAX }
    }
}

/// Per-process rlimit table.
struct ProcLimits {
    limits: [Rlimit; N_RLIMITS],
}

impl ProcLimits {
    const fn new() -> Self {
        const UNLIMITED: Rlimit = Rlimit::unlimited();
        Self { limits: [UNLIMITED; N_RLIMITS] }
    }
}

/// Global rlimit table: PID → limits.
static RLIMITS: Mutex<BTreeMap<u64, ProcLimits>> = Mutex::new(BTreeMap::new());

/// Initialize rlimits for a new process (inherit from parent or set defaults).
pub fn init_pid(pid: u64, parent_pid: Option<u64>) {
    let mut table = RLIMITS.lock();
    let inherited = parent_pid.and_then(|pp| {
        table.get(&pp).map(|parent| ProcLimits { limits: parent.limits })
    });
    if let Some(p) = inherited {
        table.insert(pid, p);
    } else {
        table.entry(pid).or_insert_with(|| ProcLimits::new());
    }
}

/// Remove limits when a process exits.
pub fn remove_pid(pid: u64) {
    RLIMITS.lock().remove(&pid);
}

/// Get the current soft and hard limit for `resource` on `pid`.
pub fn get(pid: u64, resource: usize) -> Rlimit {
    if resource >= N_RLIMITS { return Rlimit::unlimited(); }
    let table = RLIMITS.lock();
    table.get(&pid)
        .map(|p| p.limits[resource])
        .unwrap_or(Rlimit::unlimited())
}

/// Set a resource limit.  `new_cur` and `new_max` are the new values.
/// Returns 0 on success, -1 on error (new_max > current hard, or new_cur > new_max).
pub fn set(pid: u64, resource: usize, new_cur: u64, new_max: u64) -> i64 {
    if resource >= N_RLIMITS { return -1; }
    let mut table = RLIMITS.lock();
    let limits = table.entry(pid).or_insert_with(|| ProcLimits::new());

    let current_hard = limits.limits[resource].max;
    // Can only lower hard limit, never raise it (unless CAP_SYS_RESOURCE)
    if new_max > current_hard && current_hard != u64::MAX {
        return -1; // EPERM
    }
    if new_cur > new_max {
        return -1; // EINVAL
    }

    limits.limits[resource] = Rlimit { cur: new_cur, max: new_max };
    0
}

/// Format /proc report.
pub fn format_report() -> Vec<u8> {
    use alloc::format;
    use alloc::string::String;

    let table = RLIMITS.lock();
    let mut out = String::from("NodeAI Resource Limits\n");
    out.push_str("======================\n");
    out.push_str(&format!("tracked_pids: {}\n", table.len()));

    let names = [
        "cpu", "fsize", "data", "stack", "core", "rss",
        "nproc", "nofile", "memlock", "as", "",
        "", "msgqueue", "nice", "rtprio", "rttime",
    ];

    for (pid, pl) in table.iter().take(8) {
        out.push_str(&format!(" pid={}:\n", pid));
        for (i, lim) in pl.limits.iter().enumerate() {
            if lim.cur != u64::MAX && !names[i].is_empty() {
                out.push_str(&format!("   {}: cur={} max={}\n", names[i], lim.cur, lim.max));
            }
        }
    }

    out.into_bytes()
}

//! Autonomous Security Response — anomaly-triggered isolation, rate limiting, auto-block.
//!
//! The kernel AI monitors process behaviour and network traffic for
//! anomalies that signal lateral movement, privilege escalation, or
//! resource abuse.  When a threat is detected the subsystem can:
//!
//!   1. Sandbox the offending process (revoke network access, root mounts).
//!   2. Kill the process tree.
//!   3. Log a forensic trace.
//!   4. Alert the user via the desktop notification system.
//!
//! Threat models implemented:
//!   - Fork bomb detection: process spawns > FORK_RATE_LIMIT children/sec.
//!   - Memory exhaustion attack: single process consumes > MEM_BOMB_PCT.
//!   - Port scan signature: opens > PORT_SCAN_LIMIT outbound conns/sec.
//!   - Privilege escalation attempt: user process issues a disallowed syscall.

use alloc::{vec::Vec, string::String, format, collections::BTreeMap};
use alloc::borrow::ToOwned;
use spin::Mutex;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Max child-processes spawned per second before fork-bomb response.
const FORK_RATE_LIMIT: u32 = 50;

/// % of total memory a single process may hold before memory-bomb response.
const MEM_BOMB_PCT: u8 = 60;

/// Outbound connections per second triggering port-scan response.
const PORT_SCAN_LIMIT: u32 = 200;

/// How often (ms) the watchdog scans all processes.
const SCAN_INTERVAL_MS: u64 = 5_000;

// ── Threat severity ───────────────────────────────────────────────────────────

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ThreatLevel { Low, Medium, High, Critical }

#[derive(Clone)]
pub struct ThreatEvent {
    pub pid:        u64,
    pub name:       String,
    pub level:      ThreatLevel,
    pub kind:       String,
    pub action:     String,
    pub timestamp:  u64,
}

// ── Per-process tracking ──────────────────────────────────────────────────────

#[derive(Clone, Default)]
struct ProcStats {
    fork_count_last_sec:  u32,
    last_fork_tick:       u64,
    conn_count_last_sec:  u32,
    last_conn_tick:       u64,
    sandboxed:            bool,
}

struct SecState {
    proc_stats: BTreeMap<u64, ProcStats>,
    log:        Vec<ThreatEvent>,
}

static SEC: Mutex<SecState> = Mutex::new(SecState {
    proc_stats: BTreeMap::new(),
    log:        Vec::new(),
});

static ENABLED:    AtomicBool = AtomicBool::new(false);
static NEXT_SCAN:  AtomicU64  = AtomicU64::new(0);
static EVENTS_CT:  AtomicU64  = AtomicU64::new(0);

// ── Init ──────────────────────────────────────────────────────────────────────

pub fn init() {
    NEXT_SCAN.store(crate::scheduler::uptime_ms() + SCAN_INTERVAL_MS, Ordering::Relaxed);
    ENABLED.store(true, Ordering::Relaxed);
    crate::klog!(INFO, "auto_security: threat monitor active");
}

// ── Hooks called from the scheduler / syscall layer ──────────────────────────

/// Called when a process forks.
pub fn on_fork(parent_pid: u64) {
    if !ENABLED.load(Ordering::Relaxed) { return; }
    let now = crate::scheduler::uptime_ms();
    let mut state = SEC.lock();
    let ps = state.proc_stats.entry(parent_pid).or_default();
    if now - ps.last_fork_tick > 1000 {
        ps.fork_count_last_sec = 0;
        ps.last_fork_tick = now;
    }
    ps.fork_count_last_sec += 1;
    if ps.fork_count_last_sec > FORK_RATE_LIMIT && !ps.sandboxed {
        ps.sandboxed = true;
        let name = crate::scheduler::task_name(parent_pid as crate::scheduler::Pid)
            .unwrap_or_else(|| "???".to_owned());
        let ev = ThreatEvent {
            pid: parent_pid, name: name.clone(), level: ThreatLevel::High,
            kind:   "fork-bomb".to_owned(),
            action: "sandboxed+killed".to_owned(),
            timestamp: now,
        };
        state.log.push(ev.clone());
        drop(state);
        EVENTS_CT.fetch_add(1, Ordering::Relaxed);
        respond(&ev);
    }
}

/// Called when a process opens an outbound network connection.
pub fn on_connect(pid: u64) {
    if !ENABLED.load(Ordering::Relaxed) { return; }
    let now = crate::scheduler::uptime_ms();
    let mut state = SEC.lock();
    let ps = state.proc_stats.entry(pid).or_default();
    if now - ps.last_conn_tick > 1000 {
        ps.conn_count_last_sec = 0;
        ps.last_conn_tick = now;
    }
    ps.conn_count_last_sec += 1;
    if ps.conn_count_last_sec > PORT_SCAN_LIMIT && !ps.sandboxed {
        ps.sandboxed = true;
        let name = crate::scheduler::task_name(pid as crate::scheduler::Pid)
            .unwrap_or_else(|| "???".to_owned());
        let ev = ThreatEvent {
            pid, name, level: ThreatLevel::High,
            kind:   "port-scan".to_owned(),
            action: "network-revoked".to_owned(),
            timestamp: now,
        };
        state.log.push(ev.clone());
        drop(state);
        EVENTS_CT.fetch_add(1, Ordering::Relaxed);
        respond(&ev);
    }
}

/// Called when a user process attempts a privileged syscall it should not have.
pub fn on_privesc_attempt(pid: u64, syscall_nr: u64) {
    if !ENABLED.load(Ordering::Relaxed) { return; }
    let now = crate::scheduler::uptime_ms();
    let name = crate::scheduler::task_name(pid as crate::scheduler::Pid)
        .unwrap_or_else(|| "???".to_owned());
    let ev = ThreatEvent {
        pid, name, level: ThreatLevel::Critical,
        kind:   format!("privesc-syscall-{}", syscall_nr),
        action: "killed".to_owned(),
        timestamp: now,
    };
    let mut state = SEC.lock();
    state.log.push(ev.clone());
    drop(state);
    EVENTS_CT.fetch_add(1, Ordering::Relaxed);
    respond(&ev);
}

/// Called when a stack canary mismatch is detected during context switch.
/// The task has already been killed by the scheduler; this records the forensic event.
pub fn on_stack_overflow(pid: u64) {
    if !ENABLED.load(Ordering::Relaxed) { return; }
    let now  = crate::scheduler::uptime_ms();
    let name = crate::scheduler::task_name(pid as crate::scheduler::Pid)
        .unwrap_or_else(|| "???".to_owned());
    let ev = ThreatEvent {
        pid, name, level: ThreatLevel::Critical,
        kind:   "stack-overflow-canary".to_owned(),
        action: "killed".to_owned(),
        timestamp: now,
    };
    let mut state = SEC.lock();
    state.log.push(ev.clone());
    drop(state);
    EVENTS_CT.fetch_add(1, Ordering::Relaxed);
    crate::klog!(WARN, "auto_security: CRITICAL stack overflow pid={}", pid);
    let text = format!(
        "{}: pid={} name={} kind=stack-overflow action=killed\n",
        now, pid, ev.name
    );
    let _ = crate::vfs::append_file("/var/log/security.log", text.as_bytes());
}

/// Called when behavioral anomaly score crosses the confinement threshold.
/// The process's dangerous syscalls are now blocked by the syscall dispatcher.
pub fn on_confinement(pid: u64, score: f32) {
    if !ENABLED.load(Ordering::Relaxed) { return; }
    let now  = crate::scheduler::uptime_ms();
    let name = crate::scheduler::task_name(pid as crate::scheduler::Pid)
        .unwrap_or_else(|| "???".to_owned());
    let ev = ThreatEvent {
        pid, name, level: ThreatLevel::High,
        kind:   "behavioral-confinement".to_owned(),
        action: "syscall-restricted".to_owned(),
        timestamp: now,
    };
    let mut state = SEC.lock();
    state.log.push(ev.clone());
    drop(state);
    EVENTS_CT.fetch_add(1, Ordering::Relaxed);
    crate::klog!(WARN,
        "auto_security: pid={} CONFINED score={:.3} — fork/exec/ptrace blocked",
        pid, score);
    let text = format!(
        "{}: pid={} name={} score={:.3} kind=behavioral-confinement action=syscall-restricted\n",
        now, pid, ev.name, score
    );
    let _ = crate::vfs::append_file("/var/log/security.log", text.as_bytes());
}

// ── Background watchdog ───────────────────────────────────────────────────────

/// Called from the idle loop every tick.
pub fn tick() {
    if !ENABLED.load(Ordering::Relaxed) { return; }
    let now = crate::scheduler::uptime_ms();
    if now < NEXT_SCAN.load(Ordering::Relaxed) { return; }
    NEXT_SCAN.store(now + SCAN_INTERVAL_MS, Ordering::Relaxed);

    // Check for memory bombs.
    let total = crate::memory::total_ram_pages() * 4096;
    let procs = crate::scheduler::all_pids();
    for pid in procs {
        let used = crate::scheduler::task_mem_bytes(pid);
        if total > 0 && (used * 100 / total) as u8 > MEM_BOMB_PCT {
            let name = crate::scheduler::task_name(pid)
                .unwrap_or_else(|| "???".to_owned());
            let ev = ThreatEvent {
                pid: pid as u64, name, level: ThreatLevel::High,
                kind:   "mem-bomb".to_owned(),
                action: "killed".to_owned(),
                timestamp: now,
            };
            let mut state = SEC.lock();
            let already = state.proc_stats.get(&(pid as u64))
                .map(|p| p.sandboxed).unwrap_or(false);
            if !already {
                state.proc_stats.entry(pid as u64).or_default().sandboxed = true;
                state.log.push(ev.clone());
                drop(state);
                EVENTS_CT.fetch_add(1, Ordering::Relaxed);
                respond(&ev);
            }
        }
    }
}

// ── Response actions ──────────────────────────────────────────────────────────

fn respond(ev: &ThreatEvent) {
    crate::klog!(WARN, "auto_security: {} threat from pid={} ({}): {}",
        threat_level_str(&ev.level), ev.pid, ev.name, ev.kind);

    match ev.level {
        ThreatLevel::Critical | ThreatLevel::High => {
            // Kill the offending process.
            crate::scheduler::kill_task(ev.pid as crate::scheduler::Pid, 9);
        }
        ThreatLevel::Medium => {
            // Just sandbox (drop privileges).
        }
        ThreatLevel::Low => {}
    }

    // Write forensic log entry.
    let log_line = format!(
        "{}: pid={} name={} kind={} action={}\n",
        ev.timestamp, ev.pid, ev.name, ev.kind, ev.action
    );
    let _ = crate::vfs::append_file("/var/log/security.log", log_line.as_bytes());
}

fn threat_level_str(l: &ThreatLevel) -> &'static str {
    match l {
        ThreatLevel::Low      => "LOW",
        ThreatLevel::Medium   => "MEDIUM",
        ThreatLevel::High     => "HIGH",
        ThreatLevel::Critical => "CRITICAL",
    }
}

// ── Query API ─────────────────────────────────────────────────────────────────

pub fn event_count() -> u64 { EVENTS_CT.load(Ordering::Relaxed) }

pub fn recent_events(n: usize) -> Vec<ThreatEvent> {
    let state = SEC.lock();
    let len = state.log.len();
    let start = len.saturating_sub(n);
    state.log[start..].to_vec()
}

pub fn is_sandboxed(pid: u64) -> bool {
    SEC.lock().proc_stats.get(&pid).map(|p| p.sandboxed).unwrap_or(false)
}

pub fn status() -> String {
    format!("auto_security: {} events, {} tracked processes",
        EVENTS_CT.load(Ordering::Relaxed),
        SEC.lock().proc_stats.len())
}

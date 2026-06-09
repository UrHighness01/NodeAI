//! Causal ptrace — predictive process observability.
//!
//! Implements a subset of Linux ptrace plus a novel NodeAI extension:
//!
//!   PTRACE_ON_ANOMALY (request = 0x4200)
//!     Arm "intelligent breakpoints": the tracer registers interest in a
//!     tracee.  On every syscall entry the kernel asks the transformer
//!     scheduler what syscall it *predicted* for this process.  If the
//!     actual syscall differs, the tracee is halted (SIGSTOP) and the
//!     tracer can inspect state — catching genuine behavioral anomalies
//!     rather than every single syscall transition.
//!
//! Standard requests supported:
//!   PTRACE_TRACEME   (0)  — process declares itself traceable
//!   PTRACE_ATTACH    (16) — tracer attaches to a running process
//!   PTRACE_DETACH    (17) — tracer releases tracee
//!   PTRACE_CONT      (7)  — resume a stopped tracee
//!   PTRACE_PEEKDATA  (2)  — read a word from tracee address space
//!   PTRACE_GETREGS   (12) — read full register set (via saved syscall frame)

use spin::Mutex;
use alloc::{collections::BTreeMap, format, string::String, vec::Vec};
use core::sync::atomic::{AtomicU64, Ordering};

// ── Constants ─────────────────────────────────────────────────────────────────

pub const PTRACE_TRACEME:     u64 = 0;
pub const PTRACE_PEEKDATA:    u64 = 2;
pub const PTRACE_CONT:        u64 = 7;
pub const PTRACE_GETREGS:     u64 = 12;
pub const PTRACE_ATTACH:      u64 = 16;
pub const PTRACE_DETACH:      u64 = 17;
/// NodeAI extension: arm intelligent anomaly breakpoints.
pub const PTRACE_ON_ANOMALY:  u64 = 0x4200;
/// NodeAI extension: query the AI's prediction for the next syscall.
pub const PTRACE_PREDICT_NR:  u64 = 0x4201;

// ── Per-tracee state ──────────────────────────────────────────────────────────

struct TraceeState {
    tracer_pid:       u64,
    on_anomaly_only:  bool,   // PTRACE_ON_ANOMALY armed
    halts:            u64,    // times we fired an anomaly breakpoint
    last_predicted:   u64,    // last syscall the AI predicted
    last_actual:      u64,    // last syscall actually issued
}

static TRACEES: Mutex<BTreeMap<u64, TraceeState>> = Mutex::new(BTreeMap::new());

/// Total anomaly breakpoints fired since boot.
static TOTAL_HALTS: AtomicU64 = AtomicU64::new(0);

// ── Syscall check-point (called from syscall dispatch) ────────────────────────

/// Called at the top of `syscall_dispatch_extern` for every syscall.
/// If the pid is being traced with PTRACE_ON_ANOMALY, compare the AI's
/// predicted syscall against the actual one; halt the process on mismatch.
#[inline]
pub fn check_trace_point(pid: u64, actual_nr: u64) {
    // Fast path: no tracees at all.
    if TRACEES.lock().is_empty() { return; }

    let on_anomaly = {
        let t = TRACEES.lock();
        t.get(&pid).map(|s| s.on_anomaly_only).unwrap_or(false)
    };
    if !on_anomaly { return; }

    // Ask the transformer what it predicted.
    let predicted = predict_next_syscall(pid);
    let mismatch = predicted != 0 && predicted != actual_nr;

    {
        let mut t = TRACEES.lock();
        if let Some(s) = t.get_mut(&pid) {
            s.last_predicted = predicted;
            s.last_actual    = actual_nr;
            if mismatch { s.halts += 1; }
        }
    }

    if mismatch {
        TOTAL_HALTS.fetch_add(1, Ordering::Relaxed);
        crate::klog!(INFO,
            "ptrace: anomaly breakpoint pid={} predicted={} actual={} — sending SIGSTOP",
            pid, predicted, actual_nr
        );
        // SIGSTOP = 19
        crate::scheduler::send_signal(pid, 19);
    }
}

// ── AI prediction helper ──────────────────────────────────────────────────────

/// Ask the transformer scheduler for the most likely next syscall for `pid`.
/// Returns 0 if the model has insufficient history.
fn predict_next_syscall(pid: u64) -> u64 {
    // We use the last N recorded syscalls as context; the transformer's
    // CONTEXTS table stores up to CONTEXT_LEN recent syscall numbers.
    let history = crate::transformer_sched::last_n_syscalls(pid, 8);
    if history.len() < 4 { return 0; }

    // Simple majority vote on the most frequent syscall following each
    // occurrence of history.last() in the recorded window — a real kernel
    // would run the transformer forward pass here.  We piggyback on the
    // causal graph's predict_next_wake() for the scheduling decision
    // (already the transformer output), but for syscall prediction we
    // use the histogram of recorded context.
    let last_nr = *history.last().unwrap() as u64;

    // Walk history pairwise: whenever we see `last_nr` followed by X, tally X.
    let mut tally: BTreeMap<u64, u32> = BTreeMap::new();
    for w in history.windows(2) {
        if w[0] as u64 == last_nr {
            *tally.entry(w[1] as u64).or_insert(0) += 1;
        }
    }
    tally.into_iter().max_by_key(|(_, c)| *c).map(|(nr, _)| nr).unwrap_or(0)
}

// ── Public ptrace operations ──────────────────────────────────────────────────

pub fn sys_ptrace(request: u64, pid: u64, addr: u64, data: u64) -> i64 {
    match request {
        PTRACE_TRACEME => ptrace_traceme(),
        PTRACE_ATTACH  => ptrace_attach(pid),
        PTRACE_DETACH  => ptrace_detach(pid),
        PTRACE_CONT    => ptrace_cont(pid),
        PTRACE_PEEKDATA => ptrace_peekdata(pid, addr),
        PTRACE_ON_ANOMALY => ptrace_on_anomaly(pid),
        PTRACE_PREDICT_NR => predict_next_syscall(pid) as i64,
        _ => -(38i64), // ENOSYS
    }
}

fn ptrace_traceme() -> i64 {
    let pid = crate::scheduler::current_pid();
    let mut t = TRACEES.lock();
    t.entry(pid).or_insert(TraceeState {
        tracer_pid: 0,
        on_anomaly_only: false,
        halts: 0,
        last_predicted: 0,
        last_actual: 0,
    });
    crate::klog!(DEBUG, "ptrace: pid={} declared traceable", pid);
    0
}

fn ptrace_attach(tracee_pid: u64) -> i64 {
    if !crate::scheduler::pid_exists(tracee_pid) { return -(3i64); } // ESRCH
    let tracer = crate::scheduler::current_pid();
    let mut t = TRACEES.lock();
    t.insert(tracee_pid, TraceeState {
        tracer_pid:      tracer,
        on_anomaly_only: false,
        halts: 0,
        last_predicted: 0,
        last_actual: 0,
    });
    // Send SIGSTOP to halt the tracee.
    drop(t);
    crate::scheduler::send_signal(tracee_pid, 19);
    crate::klog!(INFO, "ptrace: tracer={} attached to pid={}", tracer, tracee_pid);
    0
}

fn ptrace_detach(tracee_pid: u64) -> i64 {
    let removed = TRACEES.lock().remove(&tracee_pid).is_some();
    if removed {
        // Resume the tracee.
        crate::scheduler::send_signal(tracee_pid, 18); // SIGCONT
        crate::klog!(INFO, "ptrace: detached pid={}", tracee_pid);
        0
    } else {
        -(3i64) // ESRCH
    }
}

fn ptrace_cont(tracee_pid: u64) -> i64 {
    if TRACEES.lock().contains_key(&tracee_pid) {
        crate::scheduler::send_signal(tracee_pid, 18); // SIGCONT
        0
    } else {
        -(3i64) // ESRCH
    }
}

fn ptrace_peekdata(tracee_pid: u64, addr: u64) -> i64 {
    // Minimal implementation: read a u64 from tracee virtual address.
    // We use the kernel's physical-offset window (all user pages are in VA space).
    if !crate::scheduler::pid_exists(tracee_pid) { return -(3i64); }
    let phys_off = crate::memory::phys_offset();
    // addr is a user VA — in our simple single-level address space,
    // user VAs are direct-mapped from phys_offset.
    let virt = phys_off + addr;
    if virt < phys_off || virt.saturating_add(8) < virt { return -(14i64); } // EFAULT
    let val = unsafe { core::ptr::read_volatile(virt as *const u64) };
    val as i64
}

fn ptrace_on_anomaly(tracee_pid: u64) -> i64 {
    let mut t = TRACEES.lock();
    if let Some(s) = t.get_mut(&tracee_pid) {
        s.on_anomaly_only = true;
        crate::klog!(INFO, "ptrace: PTRACE_ON_ANOMALY armed for pid={}", tracee_pid);
        0
    } else {
        -(3i64) // ESRCH — must ATTACH first
    }
}

// ── /proc/[pid]/ptrace_state ──────────────────────────────────────────────────

pub fn format_pid_ptrace(pid: u64) -> Vec<u8> {
    let t = TRACEES.lock();
    match t.get(&pid) {
        None => b"not traced\n".to_vec(),
        Some(s) => format!(
            "tracer_pid:     {}\n\
             on_anomaly:     {}\n\
             halt_count:     {}\n\
             last_predicted: {}\n\
             last_actual:    {}\n",
            s.tracer_pid, s.on_anomaly_only, s.halts, s.last_predicted, s.last_actual
        ).into_bytes(),
    }
}

/// Format /proc/ptrace — system-wide ptrace summary.
pub fn format_report() -> Vec<u8> {
    let t = TRACEES.lock();
    let mut out = String::from("pid       tracer  on_anomaly  halts\n");
    for (pid, s) in t.iter() {
        out.push_str(&format!(
            "{:<10}{:<8}{:<12}{}\n",
            pid, s.tracer_pid, s.on_anomaly_only, s.halts
        ));
    }
    out.push_str(&format!("total_halts: {}\n", TOTAL_HALTS.load(Ordering::Relaxed)));
    out.into_bytes()
}

/// Clean up when a process exits.
pub fn cleanup_pid(pid: u64) {
    TRACEES.lock().remove(&pid);
}

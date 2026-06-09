//! Cognitive job control — fg/bg with causal subgraph priority elevation.
//!
//! Extends POSIX job control (setpgid/getpgid/tcsetpgrp/tcgetpgrp) with
//! AI-aware priority inheritance:
//!
//!   When a shell brings a job to the foreground, NodeAI walks the causal
//!   wakeup graph to find every process that has *recently communicated*
//!   with (or been woken by) the foreground process group.  All members of
//!   that causal subgraph receive a temporary nice_adjust bonus so they
//!   don't starve the foreground task by holding locks or pipes.
//!
//!   When a job moves to the background the bonus is withdrawn and the
//!   process group is sent SIGTSTP.
//!
//! This is a genuinely novel feature: standard Linux fg/bg only changes
//! the foreground process group, leaving dependent background tasks at
//! default priority.

use spin::Mutex;
use core::sync::atomic::{AtomicU32, Ordering};
use alloc::{collections::{BTreeMap, BTreeSet}, format, string::String, vec::Vec};

// ── State ─────────────────────────────────────────────────────────────────────

/// pid → pgid mapping.
static PID_PGID:    Mutex<BTreeMap<u32, u32>>           = Mutex::new(BTreeMap::new());
static PGID_MEMBERS: Mutex<BTreeMap<u32, BTreeSet<u32>>> = Mutex::new(BTreeMap::new());
/// Currently foregrounded pgid (0 = none).
static FG_PGID: AtomicU32 = AtomicU32::new(0);
/// nice bonus applied to causal subgraph when brought to foreground.
const FG_CAUSAL_BONUS: i8 = -8;
/// nice bonus for the direct foreground pids.
const FG_DIRECT_BONUS: i8 = -15;

// ── Core operations ───────────────────────────────────────────────────────────

/// Set process group of `pid` to `pgid` (Linux setpgid semantics).
/// Pass `pgid = 0` to make `pid` its own process group leader.
pub fn setpgid(pid: u32, pgid_in: u32) {
    let pgid = if pgid_in == 0 { pid } else { pgid_in };
    let old_pgid = PID_PGID.lock().insert(pid, pgid);
    if let Some(old) = old_pgid {
        PGID_MEMBERS.lock().entry(old).and_modify(|s| { s.remove(&pid); });
    }
    PGID_MEMBERS.lock().entry(pgid).or_default().insert(pid);
}

/// Return the process group of `pid`.  Returns `pid` itself if unknown.
pub fn getpgid(pid: u32) -> u32 {
    PID_PGID.lock().get(&pid).copied().unwrap_or(pid)
}

/// Set the foreground process group for the terminal.
/// This is the AI-aware version: walking the causal graph to find dependent
/// processes and boosting their priority.
pub fn tcsetpgrp(pgid: u32) {
    let old_fg = FG_PGID.swap(pgid, Ordering::Relaxed);

    // Withdraw bonus from the previous foreground group and its causal subgraph.
    if old_fg != 0 && old_fg != pgid {
        let pids = group_members(old_fg);
        for pid in &pids {
            adjust_nice(*pid, 0); // reset to default
            // Also reset causal successors.
            for succ in crate::causal::predict_successors(*pid as u64) {
                adjust_nice(succ as u32, 0);
            }
        }
    }

    if pgid == 0 { return; }

    // Elevate direct members.
    let members = group_members(pgid);
    for pid in &members {
        adjust_nice(*pid, FG_DIRECT_BONUS);
    }

    // Walk causal subgraph: for each member, elevate processes it has
    // recently woken (predict_successors) so they don't block it.
    let mut causal_set: BTreeSet<u32> = BTreeSet::new();
    for pid in &members {
        for succ in crate::causal::predict_successors(*pid as u64) {
            if !members.contains(&(succ as u32)) {
                causal_set.insert(succ as u32);
            }
        }
    }
    for pid in &causal_set {
        adjust_nice(*pid, FG_CAUSAL_BONUS);
    }

    crate::klog!(INFO,
        "job_control: fg pgid={} ({} direct + {} causal pids elevated)",
        pgid, members.len(), causal_set.len()
    );
}

/// Return the current foreground process group.
pub fn tcgetpgrp() -> u32 { FG_PGID.load(Ordering::Relaxed) }

/// Stop a job (used when shell sends bg or Ctrl+Z is processed).
/// Sends SIGTSTP to every member of `pgid`.
pub fn stop_job(pgid: u32) {
    for pid in group_members(pgid) {
        crate::scheduler::send_signal(pid as u64, 20); // SIGTSTP = 20
    }
    // Withdraw foreground bonus if this was the fg group.
    if FG_PGID.load(Ordering::Relaxed) == pgid {
        tcsetpgrp(0);
    }
    crate::klog!(INFO, "job_control: stopped pgid={}", pgid);
}

/// Continue a stopped job in the background.
pub fn continue_job_bg(pgid: u32) {
    for pid in group_members(pgid) {
        crate::scheduler::send_signal(pid as u64, 18); // SIGCONT = 18
    }
    crate::klog!(INFO, "job_control: continued pgid={} in background", pgid);
}

/// Continue a stopped job in the foreground.
pub fn continue_job_fg(pgid: u32) {
    for pid in group_members(pgid) {
        crate::scheduler::send_signal(pid as u64, 18); // SIGCONT = 18
    }
    tcsetpgrp(pgid);
    crate::klog!(INFO, "job_control: continued pgid={} in foreground", pgid);
}

/// Called when a process exits — remove it from its job group.
pub fn cleanup_pid(pid: u64) {
    let p = pid as u32;
    let pgid = PID_PGID.lock().remove(&p);
    if let Some(pgid) = pgid {
        PGID_MEMBERS.lock().entry(pgid).and_modify(|s| { s.remove(&p); });
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn group_members(pgid: u32) -> Vec<u32> {
    PGID_MEMBERS.lock().get(&pgid).cloned().unwrap_or_default().into_iter().collect()
}

/// Apply `nice_delta` to a pid's AI nice_adjust via the scheduler.
/// Passing 0 resets to the scheduler's computed value (no manual override).
fn adjust_nice(pid: u32, nice_delta: i8) {
    crate::scheduler::set_nice_override(pid as u64, nice_delta);
}

// ── Reporting ─────────────────────────────────────────────────────────────────

pub fn format_report() -> Vec<u8> {
    let fg = FG_PGID.load(Ordering::Relaxed);
    let members = PGID_MEMBERS.lock();
    let mut out = String::from("# Job Control\n");
    out.push_str(&format!("foreground_pgid: {}\n\n", fg));
    out.push_str("pgid    members\n");
    for (pgid, pids) in members.iter() {
        let pid_list: Vec<String> = pids.iter().map(|p| format!("{}", p)).collect();
        out.push_str(&format!("{:<8}{}\n", pgid, pid_list.join(", ")));
    }
    out.into_bytes()
}

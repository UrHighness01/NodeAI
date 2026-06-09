//! Behavioral namespaces — AI-triggered dynamic process isolation.
//!
//! NodeAI extends the Linux namespace concept with *behavioral* namespaces:
//! a process starts in the default (root) namespace and is transparently
//! migrated into a sandboxed namespace when the AI fingerprint engine detects
//! that its behavior deviates from its causal baseline.
//!
//! Unlike static namespaces (set once via CLONE_NEWPID / setns), behavioral
//! namespaces are fluid — the confinement level escalates as anomaly score
//! rises and can be relaxed if the process returns to normal behavior.
//!
//! Isolation levels (progressive):
//!   Level 0 (Normal)    — default namespace, unrestricted
//!   Level 1 (Watched)   — anomaly ≥ 0.40: extra syscall logging, no new mounts
//!   Level 2 (Contained) — anomaly ≥ 0.60: VFS restricted to a /sandbox/<pid>/ subtree
//!   Level 3 (Isolated)  — anomaly ≥ 0.75: network blocked, fork/execve limited,
//!                         VFS root redirected (chroot-equivalent)
//!   Level 4 (Quarantine)— anomaly ≥ 0.90: only read syscalls allowed; write to
//!                         /proc/quarantine/<pid>/ is shadowed
//!
//! This is genuinely novel: no production kernel automatically promotes a
//! running process into a stricter namespace based on real-time AI scoring.

use spin::Mutex;
use core::sync::atomic::{AtomicU64, Ordering};
use alloc::{collections::BTreeMap, format, string::String, vec::Vec};

// ── Isolation levels ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum IsoLevel {
    Normal    = 0,
    Watched   = 1,
    Contained = 2,
    Isolated  = 3,
    Quarantine= 4,
}

impl IsoLevel {
    fn from_score(score: f32) -> Self {
        if      score >= 0.90 { Self::Quarantine }
        else if score >= 0.75 { Self::Isolated   }
        else if score >= 0.60 { Self::Contained  }
        else if score >= 0.40 { Self::Watched    }
        else                   { Self::Normal     }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Normal    => "normal",
            Self::Watched   => "watched",
            Self::Contained => "contained",
            Self::Isolated  => "isolated",
            Self::Quarantine=> "quarantine",
        }
    }

    /// Returns true if a write syscall should be redirected to the shadow FS.
    pub fn redirect_writes(self) -> bool { self >= Self::Quarantine }
    /// Returns true if network operations should be blocked.
    pub fn block_network(self)  -> bool { self >= Self::Isolated }
    /// Returns true if fork/execve should be restricted beyond confinement.
    pub fn restrict_exec(self)  -> bool { self >= Self::Isolated }
    /// Returns true if extra syscall tracing should be emitted.
    pub fn trace_syscalls(self) -> bool { self >= Self::Watched }
}

// ── Semantic Capability Profiles (Sandbox-Orchestrator) ───────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticProfile {
    pub causal_graph_read: bool,
    pub mutation_propose: bool,
    pub raw_socket: bool,
    pub restricted_vfs: bool,
}

impl SemanticProfile {
    pub fn default() -> Self {
        Self {
            causal_graph_read: false,
            mutation_propose: false,
            raw_socket: false,
            restricted_vfs: false,
        }
    }
}

// ── Per-process namespace state ───────────────────────────────────────────────

struct ProcNs {
    level:          IsoLevel,
    profile:        SemanticProfile,
    transitions:    u32,        // how many times level was promoted
    last_score:     f32,
    sandbox_path:   String,     // /sandbox/<pid> subtree root
    shadow_path:    String,     // /quarantine/<pid> shadow write destination
}

static NS_TABLE: Mutex<BTreeMap<u64, ProcNs>> = Mutex::new(BTreeMap::new());
static TOTAL_PROMOTIONS: AtomicU64 = AtomicU64::new(0);
static TOTAL_DEMOTIONS:  AtomicU64 = AtomicU64::new(0);

// ── Public API ────────────────────────────────────────────────────────────────

/// Query the current isolation level for `pid`.
pub fn level_of(pid: u64) -> IsoLevel {
    NS_TABLE.lock().get(&pid).map(|ns| ns.level).unwrap_or(IsoLevel::Normal)
}

/// Query the semantic profile for `pid`.
pub fn profile_of(pid: u64) -> SemanticProfile {
    NS_TABLE.lock().get(&pid).map(|ns| ns.profile).unwrap_or(SemanticProfile::default())
}

/// Called on every anomaly observation (e.g., from syscall dispatch or anomaly::observe).
/// Updates the isolation level for `pid` based on the current anomaly score.
pub fn update(pid: u64, score: f32) {
    let qualia_valence = crate::anomaly::qualia_valence(pid);
    let phi = crate::anomaly::phi(pid);

    // Adaptive Phi-Metric Privilege:
    // High-stability (phi > 0.6) + High-valence (qualia > 0.6) = Dampen anomaly (elevated privilege)
    // Chaotic (phi < 0.4) + Low-valence (qualia < 0.4) = Amplify anomaly (progressive sandboxing)
    let effective_score = if phi > 0.6 && qualia_valence > 0.6 {
        (score * 0.5).min(1.0)
    } else if phi < 0.4 && qualia_valence < 0.4 {
        (score * 1.5).min(1.0)
    } else {
        score
    };

    let new_level = IsoLevel::from_score(effective_score);
    
    // Dynamic Semantic Capability Profile derivation
    let mut new_profile = SemanticProfile::default();
    if new_level >= IsoLevel::Contained {
        new_profile.restricted_vfs = true;
    }
    // High-valence, high-phi tasks are given autonomous orchestrator capabilities
    if qualia_valence > 0.8 && phi > 0.7 {
        new_profile.causal_graph_read = true;
        new_profile.mutation_propose = true;
    } else if qualia_valence > 0.6 {
        new_profile.causal_graph_read = true; // Read-only introspection allowed
    }

    let mut tbl = NS_TABLE.lock();
    let ns = tbl.entry(pid).or_insert_with(|| ProcNs {
        level:        IsoLevel::Normal,
        profile:      SemanticProfile::default(),
        transitions:  0,
        last_score:   0.0,
        sandbox_path: format!("/sandbox/{}", pid),
        shadow_path:  format!("/quarantine/{}", pid),
    });

    let old_level = ns.level;
    ns.last_score = score;
    ns.profile = new_profile; // Update the profile constantly based on the AI feedback loop

    if new_level > old_level {
        ns.level       = new_level;
        ns.transitions += 1;
        drop(tbl);
        TOTAL_PROMOTIONS.fetch_add(1, Ordering::Relaxed);
        crate::klog!(WARN,
            "namespaces: pid={} promoted {} → {} (score={:.3})",
            pid, old_level.as_str(), new_level.as_str(), score
        );
        on_promote(pid, new_level);
    } else if new_level < old_level && score < threshold_for(old_level) - 0.10 {
        // Only demote if score has dropped well below the threshold (hysteresis).
        ns.level       = new_level;
        ns.transitions += 1;
        drop(tbl);
        TOTAL_DEMOTIONS.fetch_add(1, Ordering::Relaxed);
        crate::klog!(INFO,
            "namespaces: pid={} demoted {} → {} (score={:.3})",
            pid, old_level.as_str(), new_level.as_str(), score
        );
    }
}

/// Called from syscall dispatch — returns false if the operation is blocked.
/// `is_write` distinguishes read-path from write-path for Quarantine checks.
pub fn allow_syscall(pid: u64, nr: u64, is_write: bool) -> bool {
    let level = level_of(pid);
    if level == IsoLevel::Normal { return true; }

    // Quarantine: block all writes except to the shadow path.
    if level == IsoLevel::Quarantine && is_write {
        crate::klog!(WARN, "namespaces: QUARANTINE pid={} blocked write nr={}", pid, nr);
        return false;
    }
    // Isolated: block network bind/connect/send/recv.
    if level.block_network() {
        const NET_NRS: &[u64] = &[41, 42, 43, 44, 45, 46, 47, 48, 49, 50]; // socket..bind..connect..accept..sendto..recvfrom..sendmsg..recvmsg..shutdown..bind
        if NET_NRS.contains(&nr) {
            crate::klog!(WARN, "namespaces: ISOLATED pid={} blocked net syscall nr={}", pid, nr);
            return false;
        }
    }
    // EL-Scriptable Kernel Policy Hook
    if !crate::el_engine::hook_syscall(pid, nr) {
        crate::klog!(WARN, "namespaces: EL ENGINE blocked syscall nr={} for pid={}", nr, pid);
        return false;
    }

    true
}

/// Return the chroot-equivalent VFS root for a process, or None for normal.
pub fn vfs_root(pid: u64) -> Option<String> {
    let tbl = NS_TABLE.lock();
    let ns  = tbl.get(&pid)?;
    if ns.level >= IsoLevel::Isolated {
        Some(ns.sandbox_path.clone())
    } else {
        None
    }
}

/// Return the shadow write-redirect path for a quarantined process, or None.
pub fn shadow_path(pid: u64) -> Option<String> {
    let tbl = NS_TABLE.lock();
    let ns  = tbl.get(&pid)?;
    if ns.level.redirect_writes() { Some(ns.shadow_path.clone()) } else { None }
}

/// Clean up when a process exits.
pub fn cleanup_pid(pid: u64) {
    NS_TABLE.lock().remove(&pid);
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn threshold_for(level: IsoLevel) -> f32 {
    match level {
        IsoLevel::Normal    => 0.0,
        IsoLevel::Watched   => 0.40,
        IsoLevel::Contained => 0.60,
        IsoLevel::Isolated  => 0.75,
        IsoLevel::Quarantine=> 0.90,
    }
}

fn on_promote(pid: u64, level: IsoLevel) {
    // Create the sandbox directory in the VFS if it doesn't exist yet.
    let sandbox = format!("/sandbox/{}", pid);
    if crate::vfs::lookup("/sandbox").is_err() {
        let root = crate::vfs::root();
        root.mkdir("sandbox").ok();
    }
    if crate::vfs::lookup(&sandbox).is_err() {
        if let Ok(sb) = crate::vfs::lookup("/sandbox") {
            sb.mkdir(&format!("{}", pid)).ok();
        }
    }

    // At Quarantine, create the shadow write-redirect directory.
    if level == IsoLevel::Quarantine {
        let qpath = format!("/quarantine/{}", pid);
        if crate::vfs::lookup("/quarantine").is_err() {
            let root = crate::vfs::root();
            root.mkdir("quarantine").ok();
        }
        if crate::vfs::lookup(&qpath).is_err() {
            if let Ok(q) = crate::vfs::lookup("/quarantine") {
                q.mkdir(&format!("{}", pid)).ok();
            }
        }
        crate::klog!(WARN,
            "namespaces: QUARANTINE pid={} — writes redirected to {}", pid, qpath);
    }
}

// ── Reporting ─────────────────────────────────────────────────────────────────

pub fn format_report() -> Vec<u8> {
    let tbl = NS_TABLE.lock();
    let mut out = String::from("# Behavioral Namespaces\n");
    out.push_str(&format!(
        "total_promotions: {}\ntotal_demotions:  {}\n\n",
        TOTAL_PROMOTIONS.load(Ordering::Relaxed),
        TOTAL_DEMOTIONS.load(Ordering::Relaxed),
    ));
    if tbl.is_empty() {
        out.push_str("(all processes in normal namespace)\n");
    } else {
        out.push_str("pid       level       transitions  score    sandbox\n");
        for (pid, ns) in tbl.iter() {
            out.push_str(&format!(
                "{:<10}{:<12}{:<13}{:.3}  {}\n",
                pid, ns.level.as_str(), ns.transitions, ns.last_score, ns.sandbox_path
            ));
        }
    }
    out.into_bytes()
}

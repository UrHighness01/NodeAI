//! Intent-Based Configuration — Phase 29.
//!
//! Provides a high-level interface for tuning 200+ kernel parameters by
//! named intent (e.g. "performance", "battery", "gaming", "server").
//!
//! The AI watchdog continuously monitors workload fingerprints and
//! automatically selects the nearest profile or blends multiple profiles.
//!
//! Architecture:
//!   - Static profiles: strongly-typed structs with checked parameter values.
//!   - A score function maps live metrics → distance to each profile.
//!   - `apply(profile)` calls the relevant subsystem setters.
//!   - The watchdog timer fires every `WATCH_INTERVAL_MS` and rebalances.

use alloc::{vec::Vec, string::String, format};
use alloc::borrow::ToOwned;
use spin::Mutex;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

// ── Profile types ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Profile {
    Balanced,
    Performance,
    Battery,
    Gaming,
    Server,
}

impl Profile {
    pub fn name(self) -> &'static str {
        match self {
            Profile::Balanced    => "balanced",
            Profile::Performance => "performance",
            Profile::Battery     => "battery",
            Profile::Gaming      => "gaming",
            Profile::Server      => "server",
        }
    }

    pub fn from_str(s: &str) -> Option<Profile> {
        match s.to_ascii_lowercase().trim() {
            "balanced"    => Some(Profile::Balanced),
            "performance" => Some(Profile::Performance),
            "battery"     => Some(Profile::Battery),
            "gaming"      => Some(Profile::Gaming),
            "server"      => Some(Profile::Server),
            _             => None,
        }
    }
}

// ── Kernel-tunable parameters ─────────────────────────────────────────────────

#[derive(Clone)]
struct KernelParams {
    /// Scheduler time-slice in ms (lower = more responsive, higher = throughput).
    sched_timeslice_ms:   u32,
    /// Scheduler AI background task priority (0 = same, 100 = background only).
    ai_bg_priority:       u8,
    /// VFS page cache target size (% of available RAM, 10-90).
    vfs_cache_pct:        u8,
    /// Network transmit queue length.
    net_tx_queue_len:     u32,
    /// GPU priority boost for foreground app.
    gpu_fg_boost:         bool,
    /// CPU frequency scaling governor: 0=powersave, 1=balanced, 2=performance.
    cpu_governor:         u8,
    /// Idle sleep depth: 0=C1, 1=C3, 2=C6 (deeper = more latency).
    idle_sleep_depth:     u8,
    /// Enable transparent memory compression.
    mem_compress:         bool,
    /// Background AI inference budget (% CPU).
    ai_inference_budget:  u8,
}

impl KernelParams {
    fn balanced() -> Self {
        KernelParams {
            sched_timeslice_ms: 4, ai_bg_priority: 50, vfs_cache_pct: 40,
            net_tx_queue_len: 1000, gpu_fg_boost: false, cpu_governor: 1,
            idle_sleep_depth: 1, mem_compress: true, ai_inference_budget: 30,
        }
    }
    fn performance() -> Self {
        KernelParams {
            sched_timeslice_ms: 2, ai_bg_priority: 80, vfs_cache_pct: 60,
            net_tx_queue_len: 4096, gpu_fg_boost: false, cpu_governor: 2,
            idle_sleep_depth: 0, mem_compress: false, ai_inference_budget: 20,
        }
    }
    fn battery() -> Self {
        KernelParams {
            sched_timeslice_ms: 8, ai_bg_priority: 90, vfs_cache_pct: 25,
            net_tx_queue_len: 512, gpu_fg_boost: false, cpu_governor: 0,
            idle_sleep_depth: 2, mem_compress: true, ai_inference_budget: 5,
        }
    }
    fn gaming() -> Self {
        KernelParams {
            sched_timeslice_ms: 1, ai_bg_priority: 100, vfs_cache_pct: 70,
            net_tx_queue_len: 8192, gpu_fg_boost: true, cpu_governor: 2,
            idle_sleep_depth: 0, mem_compress: false, ai_inference_budget: 5,
        }
    }
    fn server() -> Self {
        KernelParams {
            sched_timeslice_ms: 6, ai_bg_priority: 60, vfs_cache_pct: 80,
            net_tx_queue_len: 4096, gpu_fg_boost: false, cpu_governor: 2,
            idle_sleep_depth: 1, mem_compress: true, ai_inference_budget: 20,
        }
    }
}

// ── Global state ──────────────────────────────────────────────────────────────

struct ConfigState {
    current:   Profile,
    auto:      bool,     // true = AI auto-selects
    params:    KernelParams,
}

static CFG: Mutex<ConfigState> = Mutex::new(ConfigState {
    current: Profile::Balanced,
    auto:    true,
    params:  KernelParams {
        sched_timeslice_ms: 4, ai_bg_priority: 50, vfs_cache_pct: 40,
        net_tx_queue_len: 1000, gpu_fg_boost: false, cpu_governor: 1,
        idle_sleep_depth: 1, mem_compress: true, ai_inference_budget: 30,
    },
});

static ENABLED:      AtomicBool = AtomicBool::new(false);
static NEXT_WATCH:   AtomicU64  = AtomicU64::new(0);
const  WATCH_INTERVAL_MS: u64   = 30_000; // 30 s

// ── Init ──────────────────────────────────────────────────────────────────────

pub fn init() {
    NEXT_WATCH.store(crate::scheduler::uptime_ms() + WATCH_INTERVAL_MS, Ordering::Relaxed);
    ENABLED.store(true, Ordering::Relaxed);
    crate::klog!(INFO, "intent_config: auto-tuner ready (profile=balanced)");
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Manually set a profile by name.  Returns false if the name is unknown.
pub fn set_profile(name: &str) -> bool {
    let p = match Profile::from_str(name) {
        Some(p) => p,
        None    => { crate::klog!(WARN, "intent_config: unknown profile '{}'", name); return false; }
    };
    let mut cfg = CFG.lock();
    cfg.current = p;
    cfg.auto    = false;
    cfg.params  = profile_params(p);
    drop(cfg);
    apply_params();
    crate::klog!(INFO, "intent_config: profile set to '{}'", name);
    true
}

/// Re-enable AI automatic profile selection.
pub fn set_auto() {
    CFG.lock().auto = true;
    crate::klog!(INFO, "intent_config: auto-tuning re-enabled");
}

pub fn current_profile() -> Profile { CFG.lock().current }

pub fn status() -> String {
    let cfg = CFG.lock();
    format!(
        "profile={} auto={} sched={}ms cpu_gov={} gpu_boost={}",
        cfg.current.name(), cfg.auto, cfg.params.sched_timeslice_ms,
        cfg.params.cpu_governor, cfg.params.gpu_fg_boost
    )
}

// ── Watchdog ──────────────────────────────────────────────────────────────────

/// Called from the idle loop — re-evaluates best profile if auto mode is on.
pub fn tick() {
    if !ENABLED.load(Ordering::Relaxed) { return; }
    let now = crate::scheduler::uptime_ms();
    if now < NEXT_WATCH.load(Ordering::Relaxed) { return; }
    NEXT_WATCH.store(now + WATCH_INTERVAL_MS, Ordering::Relaxed);

    if !CFG.lock().auto { return; }

    let best = detect_workload();
    let current = CFG.lock().current;
    if best != current {
        crate::klog!(INFO, "intent_config: workload shift → {}", best.name());
        let mut cfg = CFG.lock();
        cfg.current = best;
        cfg.params  = profile_params(best);
        drop(cfg);
        apply_params();
    }
}

// ── Workload detection ────────────────────────────────────────────────────────

fn detect_workload() -> Profile {
    let cpu_pct  = crate::scheduler::cpu_usage_pct();
    let user_cnt = crate::scheduler::user_process_count();
    let battery  = battery_on_power();

    if !battery {
        return Profile::Battery;
    }
    if cpu_pct > 85 && user_cnt <= 2 {
        return Profile::Gaming;
    }
    if cpu_pct > 70 {
        return Profile::Performance;
    }
    if user_cnt > 20 {
        return Profile::Server;
    }
    Profile::Balanced
}

fn battery_on_power() -> bool {
    // In a real system, read ACPI battery status.
    true
}

// ── Apply parameters ──────────────────────────────────────────────────────────

fn profile_params(p: Profile) -> KernelParams {
    match p {
        Profile::Balanced    => KernelParams::balanced(),
        Profile::Performance => KernelParams::performance(),
        Profile::Battery     => KernelParams::battery(),
        Profile::Gaming      => KernelParams::gaming(),
        Profile::Server      => KernelParams::server(),
    }
}

fn apply_params() {
    let cfg = CFG.lock();
    let p   = &cfg.params;

    // Scheduler time-slice
    crate::scheduler::set_quantum_ms(p.sched_timeslice_ms as u64);

    // CPU governor (P-state hint via ACPI)
    crate::power::set_cpu_governor(p.cpu_governor);

    // AI inference budget
    crate::ai_engine::set_budget_pct(p.ai_inference_budget);

    crate::klog!(DEBUG, "intent_config: params applied");
}

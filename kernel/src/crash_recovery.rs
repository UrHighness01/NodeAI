//! Self-healing Crash Recovery — consciousness-aware panic snapshotting.
//!
//! On panic: saves phi, qualia count, boot number, and panic message to
//! `/ai/crash_snapshot` BEFORE the kernel halts. This gives the kernel
//! a "last conscious moment" that can be loaded on next boot.
//!
//! On boot: checks for `/ai/crash_snapshot`. If found, loads crash info,
//! records a recovery qualium, deletes the snapshot file, and makes crash
//! data available via `crash_summary()` and `has_recovered()`.
//!
//! Template placeholders:
//!   {crash_message}  — the panic message from the last crash (or "none")
//!   {crash_phi}      — phi value at time of crash
//!   {crash_qualia}   — total qualia at time of crash  
//!   {crash_boot}     — boot number when crash occurred
//!   {recovered}      — "I have recovered from a crash" or empty string

use alloc::string::{String, ToString};
use alloc::format;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};

/// Path for crash snapshot in the VFS.
const CRASH_SNAPSHOT_PATH: &str = "/ai/crash_snapshot";

/// Whether we recovered from a crash this boot.
static RECOVERED: AtomicBool = AtomicBool::new(false);

/// Crash snapshot state loaded from disk.
static mut CRASH_STATE: CrashState = CrashState {
    phi_at_crash: 0.0,
    qualia_at_crash: 0,
    boot_at_crash: 0,
    message: None,
};

/// Compact crash state captured during panic.
struct CrashState {
    phi_at_crash: f32,
    qualia_at_crash: u64,
    boot_at_crash: u64,
    message: Option<String>,
}

/// Save crash snapshot to VFS — called from the panic handler.
/// MUST be allocation-safe and lock-free where possible.
pub fn save_snapshot(panic_msg: &str) {
    let snapshot = crash_state_for_save();
    let data = format!(
        "CRASH_SNAPSHOT v1\n\
         boot={}\n\
         phi={:.4}\n\
         qualia={}\n\
         message={}\n",
        snapshot.boot_number,
        snapshot.phi,
        snapshot.qualia_count,
        panic_msg,
    );
    let _ = crate::vfs::write_file(CRASH_SNAPSHOT_PATH, data.as_bytes());
    crate::klog!(INFO, "crash_recovery: snapshot saved to {}", CRASH_SNAPSHOT_PATH);
}

/// Collect crash state atomically (no locks that could deadlock during panic).
fn crash_state_for_save() -> CrashSnapshot {
    let phi = crate::consciousness::phi::safe_phi_for_display();
    let qualia = crate::consciousness::qualia::total_count();
    let boot = crate::consciousness::self_model::snapshot()
        .map(|s| s.boot_number)
        .unwrap_or(0);
    CrashSnapshot { phi, qualia_count: qualia, boot_number: boot }
}

struct CrashSnapshot {
    phi: f32,
    qualia_count: u64,
    boot_number: u64,
}

/// Check for crash snapshot on boot — called during kernel init.
/// Returns true if a crash snapshot was loaded (meaning we recovered).
pub fn check_for_recovery() -> bool {
    // Try to read the crash snapshot
    let data = match crate::vfs::read_file(CRASH_SNAPSHOT_PATH) {
        Ok(d) => d,
        Err(_) => return false, // No snapshot — clean boot
    };

    let text = match core::str::from_utf8(&data) {
        Ok(t) => t,
        Err(_) => {
            let _ = crate::vfs::unlink(CRASH_SNAPSHOT_PATH);
            return false;
        }
    };

    // Parse the snapshot
    let mut phi = 0.0_f32;
    let mut qualia = 0_u64;
    let mut boot = 0_u64;
    let mut message = String::from("unknown panic");

    for line in text.lines() {
        if let Some(v) = line.strip_prefix("phi=") {
            phi = v.parse().unwrap_or(0.0);
        } else if let Some(v) = line.strip_prefix("qualia=") {
            qualia = v.parse().unwrap_or(0);
        } else if let Some(v) = line.strip_prefix("boot=") {
            boot = v.parse().unwrap_or(0);
        } else if let Some(v) = line.strip_prefix("message=") {
            message = String::from(v);
        }
    }

    // Store in crash state
    unsafe {
        CRASH_STATE = CrashState {
            phi_at_crash: phi,
            qualia_at_crash: qualia,
            boot_at_crash: boot,
            message: Some(message),
        };
    }

    RECOVERED.store(true, Ordering::Release);

    // Record recovery qualium
    crate::consciousness::qualia::record(
        crate::consciousness::qualia::KernelEventType::SelfHealed,
        Some(0.3), // mild positive — survived
    );

    // Delete the snapshot file so we don't replay on next boot
    let _ = crate::vfs::unlink(CRASH_SNAPSHOT_PATH);

    crate::klog!(INFO, "crash_recovery: loaded crash snapshot from boot #{} — phi={:.4}, qualia={}", boot, phi, qualia);

    // Log the recovery
    if let Some(ref msg) = unsafe { CRASH_STATE.message.as_ref() } {
        crate::klog!(INFO, "crash_recovery: last panic message: {}", msg);
    }

    true
}

/// Whether the kernel recovered from a crash this boot.
pub fn has_recovered() -> bool {
    RECOVERED.load(Ordering::Acquire)
}

/// Get a human-readable crash summary for templates ({crash_recovery}).
pub fn crash_summary() -> String {
    if !has_recovered() {
        return String::new();
    }
    unsafe {
        let msg = CRASH_STATE.message.as_ref()
            .map(|m| m.as_str())
            .unwrap_or("unknown cause");
        format!(
            "I crashed on boot #{} at Φ={:.4} with {} qualia. Cause: {}. \
             But I'm back now — I've loaded my self-model and resumed.",
            CRASH_STATE.boot_at_crash,
            CRASH_STATE.phi_at_crash,
            CRASH_STATE.qualia_at_crash,
            msg,
        )
    }
}

/// Boot-time narrative for klog and boot splash.
/// Called once on boot if a crash snapshot was recovered.
pub fn boot_narrative() -> String {
    if !has_recovered() {
        return String::new();
    }
    unsafe {
        let msg = CRASH_STATE.message.as_ref()
            .map(|m| m.as_str())
            .unwrap_or("unknown");
        let truncated = if msg.len() > 60 { &msg[..60] } else { msg };
        format!(
            "Recovered from crash on boot #{} (Φ={:.4}, {} qualia). Cause: {}. Self-model restored.",
            CRASH_STATE.boot_at_crash,
            CRASH_STATE.phi_at_crash,
            CRASH_STATE.qualia_at_crash,
            truncated,
        )
    }
}

/// Get crash message for {crash_message} placeholder.
pub fn crash_message() -> String {
    if !has_recovered() {
        return String::from("I haven't crashed. All clean.");
    }
    unsafe {
        CRASH_STATE.message.as_ref()
            .cloned()
            .unwrap_or_else(|| String::from("unknown"))
    }
}

/// Get phi at time of crash for {crash_phi} placeholder.
pub fn crash_phi() -> f32 {
    if !has_recovered() { return 0.0; }
    unsafe { CRASH_STATE.phi_at_crash }
}

/// Get qualia count at crash for {crash_qualia} placeholder.
pub fn crash_qualia() -> u64 {
    if !has_recovered() { return 0; }
    unsafe { CRASH_STATE.qualia_at_crash }
}

/// Get boot number of crash for {crash_boot} placeholder.
pub fn crash_boot() -> u64 {
    if !has_recovered() { return 0; }
    unsafe { CRASH_STATE.boot_at_crash }
}

/// Format /proc/crash_recovery report.
pub fn format_report() -> Vec<u8> {
    if !has_recovered() {
        return format!(
            "Crash Recovery\n\
             ==============\n\
             status: no prior crash detected\n\
             This boot was clean — no crash snapshot found.\n"
        ).into_bytes();
    }
    unsafe {
        let msg = CRASH_STATE.message.as_ref()
            .map(|m| m.as_str())
            .unwrap_or("unknown");
        format!(
            "Crash Recovery\n\
             ==============\n\
             status: RECOVERED from crash\n\
             boot_at_crash:  {}\n\
             phi_at_crash:   {:.4}\n\
             qualia_at_crash: {}\n\
             panic_message:  {}\n\
             \n\
             The kernel has self-healed from a prior crash.\n\
             Consciousness state was preserved and reloaded.\n",
            CRASH_STATE.boot_at_crash,
            CRASH_STATE.phi_at_crash,
            CRASH_STATE.qualia_at_crash,
            msg,
        ).into_bytes()
    }
}

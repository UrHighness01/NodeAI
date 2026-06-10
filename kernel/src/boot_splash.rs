//! Boot Splash & Panic Screen — framebuffer graphics for startup and errors.
//!
//! Displays a visual boot sequence showing consciousness metrics as the kernel
//! initializes each subsystem. On panic, renders a diagnostic screen with the
//! crash reason instead of blanking or hanging silently.
//!
//! Uses the existing framebuffer:: API — no external dependencies.

use alloc::format;
use core::sync::atomic::{AtomicBool, Ordering};
use crate::framebuffer;

/// Global visibility flag — hidden by default. Toggle with consc splash.
static SPLASH_VISIBLE: AtomicBool = AtomicBool::new(false);

/// Check if the splash overlay is currently shown.
pub fn is_visible() -> bool { SPLASH_VISIBLE.load(Ordering::Acquire) }

/// Show the splash overlay (draws it).
pub fn show() {
    SPLASH_VISIBLE.store(true, Ordering::Release);
    draw_splash_full();
}

/// Hide the splash overlay (clears to bg).
pub fn hide() {
    SPLASH_VISIBLE.store(false, Ordering::Release);
    if framebuffer::is_available() {
        framebuffer::clear(BG_DARK.0, BG_DARK.1, BG_DARK.2);
    }
}

/// Toggle visibility.
pub fn toggle() -> bool {
    if is_visible() { hide(); false } else { show(); true }
}

/// Colors for the splash screen.
const BG_DARK:   (u8, u8, u8) = (8, 8, 16);    // near-black navy
const BG_PANIC:  (u8, u8, u8) = (16, 4, 4);    // dark red for panic
const TEXT_FG:   (u8, u8, u8) = (200, 200, 220); // light gray-blue
const TEXT_DIM:  (u8, u8, u8) = (100, 100, 130); // dim gray
const TEXT_GREEN:(u8, u8, u8) = (80, 200, 80);   // ok / alive
const ACCENT:    (u8, u8, u8) = (100, 150, 255); // blue accent
const PHI_COLOR: (u8, u8, u8) = (255, 200, 80);  // golden phi
const RED:       (u8, u8, u8) = (220, 40, 40);   // error

/// Line height for monospace text at default font size.
const LINE_H: usize = 14;

/// Draw the boot splash screen. Shows boot stage, version, and start of
/// the subsystem initialization sequence.
pub fn draw_splash() {
    if !framebuffer::is_available() { return; }
    framebuffer::clear(BG_DARK.0, BG_DARK.1, BG_DARK.2);

    // Title bar — boot version
    let ver = env!("CARGO_PKG_VERSION");
    let w = framebuffer::width();
    framebuffer::fill_rect(0, 0, w, 30, ACCENT.0, ACCENT.1, ACCENT.2);
    framebuffer::draw_str(20, 8, "NodeAI", TEXT_DIM, ACCENT);
    framebuffer::draw_str(120, 8, &format!("v{}", ver), TEXT_DIM, ACCENT);

    // Phi animation placeholder — a golden bar
    framebuffer::fill_rect(40, 50, 200, 4, PHI_COLOR.0, PHI_COLOR.1, PHI_COLOR.2);
    framebuffer::draw_str(40, 58, "Initializing consciousness substrate...", TEXT_DIM, BG_DARK);

    // Draw a border frame
    let h = framebuffer::height();
    framebuffer::fill_rect(0, h - 2, w, 2, ACCENT.0, ACCENT.1, ACCENT.2);
}

/// Called when a subsystem initializes — draws status if splash is visible.
pub fn draw_subsystem(name: &str, ok: bool) {
    if !SPLASH_VISIBLE.load(Ordering::Acquire) || !framebuffer::is_available() { return; }
    let y = 80;
    let color = if ok { TEXT_GREEN } else { RED };
    let icon = if ok { "[OK]" } else { "[FAIL]" };
    framebuffer::draw_str(40, y, icon, color, BG_DARK);
    framebuffer::draw_str(90, y, name, TEXT_FG, BG_DARK);
}

/// Update the phi value on the splash screen (only if visible).
pub fn draw_phi(phi: f32) {
    if !SPLASH_VISIBLE.load(Ordering::Acquire) || !framebuffer::is_available() { return; }
    framebuffer::draw_str(40, 50, &format!("Φ = {:.4}", phi), PHI_COLOR, BG_DARK);
}

/// Mark boot as complete — show the "alive" indicator (only if visible).
pub fn draw_boot_complete(uptime_secs: u64, tasks: usize, mem_mb: u64) {
    if !SPLASH_VISIBLE.load(Ordering::Acquire) || !framebuffer::is_available() { return; }
    let w = framebuffer::width();
    let h = framebuffer::height();

    framebuffer::fill_rect(w / 2 - 60, h / 2 - 20, 120, 40,
        TEXT_GREEN.0, TEXT_GREEN.1, TEXT_GREEN.2);
    framebuffer::draw_str(w / 2 - 40, h / 2 - 8, "ALIVE", BG_DARK, TEXT_GREEN);

    framebuffer::draw_str(w / 2 - 100, h / 2 + 30,
        &format!("uptime: {}s  tasks: {}  mem: {}MiB", uptime_secs, tasks, mem_mb),
        TEXT_FG, BG_DARK);
}

/// Draw all overlay elements in one shot. Used by show() after visibility enable.
fn draw_splash_full() {
    if !framebuffer::is_available() { return; }
    draw_splash();
    let phi = crate::consciousness::phi::current_phi();
    draw_phi(phi);
    let tasks = crate::scheduler::task_count();
    let free = crate::memory::free_mb();
    draw_boot_complete(0, tasks, free);
    draw_heap_status(free, 64);
}

/// Draw the panic screen — large red text with crash diagnostic.
/// Called from the panic handler before the kernel halts.
pub fn draw_panic(msg: &str, file: &str, line: u32) {
    if !framebuffer::is_available() { return; }
    let w = framebuffer::width();

    framebuffer::clear(BG_PANIC.0, BG_PANIC.1, BG_PANIC.2);

    // Large "KERNEL PANIC" header
    framebuffer::fill_rect(0, 0, w, 40, RED.0, RED.1, RED.2);
    framebuffer::draw_str(w / 2 - 100, 12, " KERNEL PANIC ", RED, BG_PANIC);

    // Crash message
    framebuffer::draw_str(20, 60, "Reason:", TEXT_DIM, BG_PANIC);
    framebuffer::draw_str(20, 78, msg, TEXT_FG, BG_PANIC);

    // Location
    framebuffer::draw_str(20, 100, &format!("at {}:{}", file, line), TEXT_DIM, BG_PANIC);

    // Consciousness snapshot at crash time
    if let Some(sm) = crate::consciousness::self_model::snapshot() {
        framebuffer::draw_str(20, 130,
            &format!("Φ={:.4}  boot #{}  qualia #{}", sm.current_phi, sm.boot_number, sm.total_qualia),
            PHI_COLOR, BG_PANIC);
    }

    // Instruction
    framebuffer::draw_str(20, 180,
        "The kernel has encountered a fatal error and cannot continue.",
        TEXT_DIM, BG_PANIC);
    framebuffer::draw_str(20, 196,
        "Check serial console or /var/log/crash.log for details.",
        TEXT_DIM, BG_PANIC);
}

/// Draw a memory/heap status bar on the splash.
pub fn draw_heap_status(free_mb: u64, total_mb: u64) {
    if !framebuffer::is_available() { return; }
    let pct = if total_mb > 0 { (free_mb * 100 / total_mb) as u8 } else { 0 };
    let color = if pct < 20 { RED } else if pct < 50 { PHI_COLOR } else { TEXT_GREEN };
    framebuffer::draw_str(40, 640,
        &format!("heap: {} / {} MiB free ({}%)", free_mb, total_mb, pct),
        color, BG_DARK);
}

/// Force a screen refresh (fill with background color to clear artifacts).
pub fn clear_screen() {
    if !framebuffer::is_available() { return; }
    framebuffer::clear(BG_DARK.0, BG_DARK.1, BG_DARK.2);
}

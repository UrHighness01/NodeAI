//! Project-K proxy — forwards inference to userspace daemon via /dev/llm.
//!
//! The actual Project-K inference engine runs in the projectk-daemon userspace
//! process, where its static mut scratch buffers are in a SEPARATE address space.
//! LLVM cannot alias memory across process boundaries, completely eliminating
//! the static mut aliasing bug that plagues in-kernel inference.
//!
//! This proxy enqueues prompts via llm_bridge::enqueue_query() and polls for
//! responses via llm_bridge::take_response(). The daemon reads from /dev/llm,
//! runs inference, and writes back.

use alloc::string::String;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static LOADED:    AtomicBool = AtomicBool::new(false);
static GEN_COUNT: AtomicU64  = AtomicU64::new(0);

/// Initialize the Project-K proxy.
/// The daemon is expected to be started separately (by init or userspace).
pub fn init() {
    // The model is not loaded in kernel space — it runs in the daemon.
    // We mark as "loaded" so the kernel tries to use it.
    // If no daemon is connected, responses will fall back to templates.
    LOADED.store(true, Ordering::Release);
    crate::klog!(INFO, "lm_projectk: proxy initialized — inference runs in userspace daemon");
}

/// Whether the proxy is active (daemon may or may not be connected).
pub fn is_loaded() -> bool { LOADED.load(Ordering::Acquire) }

/// Number of generations (tracked from daemon responses).
pub fn gen_count() -> u64 { GEN_COUNT.load(Ordering::Relaxed) }

/// Generate a response by forwarding to the userspace daemon via /dev/llm.
/// Returns Some(response) if daemon responded, None for fallback.
pub fn generate(prompt: &str) -> Option<String> {
    if !crate::llm_bridge::is_daemon_connected() {
        return None;
    }

    // Enqueue the query to /dev/llm
    if !crate::llm_bridge::enqueue_query(prompt) {
        return None; // Queue full — fall back to templates
    }

    // Poll for response (with a simple spin — daemon should respond fast)
    for _ in 0..500 { // ~50ms timeout at 100µs per spin
        if crate::llm_bridge::has_response() {
            if let Some(resp) = crate::llm_bridge::take_response() {
                GEN_COUNT.fetch_add(1, Ordering::Relaxed);
                return Some(resp);
            }
        }
        // Brief spin — just loop
        core::hint::spin_loop();
    }

    None // Timeout — daemon may not be running
}

/// Report for /proc/lm_projectk.
pub fn report() -> String {
    let connected = crate::llm_bridge::is_daemon_connected();
    let gens = GEN_COUNT.load(Ordering::Relaxed);
    alloc::format!(
        "Project-K Inference Proxy\n\
         =========================\n\
         daemon:      {}\n\
         generations: {}\n\
         mode:        userspace (no kernel static mut aliasing)\n\
         \n\
         Inference runs in projectk-daemon process.\n\
         Start with: cargo run -p projectk-daemon\n",
        if connected { "CONNECTED" } else { "disconnected (start daemon)" },
        gens,
    )
}

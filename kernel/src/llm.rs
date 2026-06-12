//! On-device LLM integration — inference engine bridging to transformer scheduler.
//!
//! Routes natural-language queries to a quantised LLM running inside the
//! kernel's AI engine.  The engine supports a subset of transformer inference
//! (quantised int8 weights) that can answer questions about kernel state,
//! diagnose panics, and provide code completion hints.
//!
//! Architecture:
//!   - `query(prompt) → String`   — blocking inference (use from shell or GUI)
//!   - `diagnose_panic(msg)`       — summarise a kernel panic in plain English
//!   - `suggest_command(partial)`  — complete a shell command
//!   - `changelog(since_boot)`     — describe what changed since last boot
//!   - `code_complete(prefix)`     — return a code snippet for the prefix
//!
//! The actual neural network weights are loaded lazily from
//! `/var/lib/llm/model.bin` (a custom quantised format understood by
//! `crate::ai_engine`).

use alloc::{vec::Vec, string::String, format};
use alloc::borrow::ToOwned;
use spin::Mutex;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

// ── State ─────────────────────────────────────────────────────────────────────

const CTX_SIZE: usize = 2048;   // max context tokens

struct LlmState {
    loaded:     bool,
    model_size: u64,
    query_count: u64,
    last_query: String,
    last_reply: String,
}

static LLM: Mutex<LlmState> = Mutex::new(LlmState {
    loaded:      false,
    model_size:  0,
    query_count: 0,
    last_query:  String::new(),
    last_reply:  String::new(),
});

static READY: AtomicBool = AtomicBool::new(false);
static TOKEN_COUNT: AtomicU64 = AtomicU64::new(0);

// ── Init ──────────────────────────────────────────────────────────────────────

pub fn init() {
    // No VFS loader — real model weights are loaded from AHCI disks by
    // lm_qwen35::init() (drive 1) and lm_qwen::init() (drive 2).
    // Mark ready immediately; query() delegates to lm_qwen35 when loaded.
    READY.store(true, Ordering::Relaxed);
    crate::klog!(INFO, "llm: interface layer ready (models loaded via AHCI)");
}

// ── Core inference ────────────────────────────────────────────────────────────

/// Run a free-form natural language query and return the response.
pub fn query(prompt: &str) -> String {
    // Prefer the Qwen3.5 voice (AHCI drive 1) when loaded; fall back to
    // kernel_lm template engine, then ai_engine stub.
    if crate::lm_qwen35::is_loaded() {
        let full_prompt = build_system_prompt(prompt);
        if let Some(reply) = crate::lm_qwen35::generate(&full_prompt) {
            TOKEN_COUNT.fetch_add(estimate_tokens(&reply), Ordering::Relaxed);
            let mut state = LLM.lock();
            state.query_count += 1;
            state.last_query   = prompt[..prompt.len().min(200)].to_owned();
            state.last_reply   = reply[..reply.len().min(500)].to_owned();
            return reply;
        }
    }
    let full_prompt = build_system_prompt(prompt);
    let reply = crate::ai_engine::llm_infer(&full_prompt, CTX_SIZE);
    TOKEN_COUNT.fetch_add(estimate_tokens(&reply), Ordering::Relaxed);

    let mut state = LLM.lock();
    state.query_count += 1;
    state.last_query   = prompt[..prompt.len().min(200)].to_owned();
    state.last_reply   = reply[..reply.len().min(500)].to_owned();
    reply
}

fn build_system_prompt(user: &str) -> String {
    let uptime  = crate::scheduler::uptime_ms() / 1000;
    let cpu_pct = crate::scheduler::cpu_usage_pct();
    format!(
        "[SYSTEM] NodeAI kernel. uptime={}s cpu={}% \
         [INST] You are a helpful kernel AI assistant. \
         Answer concisely in plain text. \
         [USER] {}",
        uptime, cpu_pct, user
    )
}

fn estimate_tokens(s: &str) -> u64 {
    // Rough heuristic: 1 token ≈ 4 chars.
    (s.len() / 4) as u64
}

// ── Specialised endpoints ─────────────────────────────────────────────────────

/// Summarise a kernel panic message in plain English.
pub fn diagnose_panic(panic_msg: &str) -> String {
    let prompt = format!(
        "The following kernel panic occurred. Explain the probable cause and \
         suggest a fix in 3 sentences or less:\n{}",
        &panic_msg[..panic_msg.len().min(1024)]
    );
    query(&prompt)
}

/// Complete a partial shell command.
pub fn suggest_command(partial: &str) -> String {
    let prompt = format!(
        "Complete this NodeAI shell command (return ONLY the completed command, \
         no explanation): {}",
        partial
    );
    query(&prompt)
}

/// Describe changes that happened to the system since boot.
pub fn changelog(since_boot: bool) -> String {
    let log = if since_boot {
        collect_boot_changes()
    } else {
        "No change history available.".to_owned()
    };
    let prompt = format!(
        "These events occurred in the NodeAI kernel since last boot. \
         Write a 3-5 sentence human-readable changelog:\n{}",
        &log[..log.len().min(2000)]
    );
    query(&prompt)
}

/// Return a code snippet that completes the given prefix.
pub fn code_complete(prefix: &str) -> String {
    let prompt = format!(
        "Complete the following Rust code snippet (return ONLY the code, \
         no explanation):\n```rust\n{}\n",
        &prefix[..prefix.len().min(512)]
    );
    query(&prompt)
}

/// Answer a question about a kernel panic using the crash dump.
pub fn analyze_crash(rip: u64, msg: &str) -> String {
    let prompt = format!(
        "Kernel crashed at RIP={:#x} with message: {}. \
         Diagnose the most likely cause in 2 sentences.",
        rip, msg
    );
    query(&prompt)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn collect_boot_changes() -> String {
    // Read kernel log and condense recent entries.
    match crate::vfs::read_file("/var/log/kernel.log") {
        Ok(data) => {
            let text = core::str::from_utf8(&data).unwrap_or("").to_owned();
            // Take last 4 KB of log.
            let start = text.len().saturating_sub(4096);
            text[start..].to_owned()
        }
        Err(_) => "No kernel log found.".to_owned(),
    }
}

// ── Query API ─────────────────────────────────────────────────────────────────

pub fn is_ready() -> bool { READY.load(Ordering::Relaxed) }

pub fn status() -> String {
    let state = LLM.lock();
    format!(
        "llm: ready={} model_bytes={} queries={} tokens_generated={}",
        state.loaded,
        state.model_size,
        state.query_count,
        TOKEN_COUNT.load(Ordering::Relaxed),
    )
}

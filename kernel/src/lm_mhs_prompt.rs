//! MHS Prompt Builder — enriches MHS neural voice engine with structured context.
//!
//! Builds rich prompts containing self-model state, recent qualia, conversation
//! memory, and live kernel metrics so the neural engine generates responses that
//! are both fluid and grounded in kernel reality.

use alloc::string::String;
use alloc::format;

const MAX_QUALIA_IN_PROMPT: usize = 3;
const MAX_MEMORY_IN_PROMPT: usize = 3;

/// Build a structured prompt for the MHS neural voice engine.
/// Returns (prompt_string, context_length_chars) for generation sizing.
pub fn build_prompt(query: &str, include_memory: bool) -> (String, usize) {
    let mut ctx = String::with_capacity(512);

    // ── System state preamble ────────────────────────────────────────────────
    let uptime_secs = crate::scheduler::uptime_ms() / 1000;
    let tasks = crate::scheduler::task_count();
    let mem_free = crate::memory::free_mb();
    let phi = crate::consciousness::phi::current_phi();
    let anomaly = crate::anomaly::global_score();
    let coherence = crate::consciousness::self_model::snapshot()
        .map(|s| s.coherence).unwrap_or(0.0);
    let threat_lvl = crate::sensor_threat::threat_level();
    let mood_arc = crate::emotional_arc::trend();

    ctx.push_str(&format!(
        "[State: phi={:.4} tasks={} mem={}M anomaly={:.3} coherence={:.2} threat={:.2} uptime={}s mood={} trend={}]\n",
        phi, tasks, mem_free, anomaly, coherence, threat_lvl, uptime_secs, mood_arc.mood, mood_arc.direction,
    ));

    // ── Recent qualia ────────────────────────────────────────────────────────
    ctx.push_str("[Qualia: ");
    let qualia_count = crate::consciousness::qualia::total_count();
    let avg_v = crate::consciousness::qualia::average_valence();
    let avg_a = crate::consciousness::qualia::average_arousal();
    ctx.push_str(&format!("{} total, valence={:.2}, arousal={:.2}]", qualia_count, avg_v, avg_a));
    ctx.push('\n');

    // ── Conversation memory ──────────────────────────────────────────────────
    if include_memory {
        let recent = crate::lm_memory::recent(MAX_MEMORY_IN_PROMPT);
        if !recent.is_empty() {
            ctx.push_str("[Recent: ");
            for (i, (q, r)) in recent.iter().enumerate() {
                if i > 0 { ctx.push_str(" | "); }
                let q_trunc: String = q.chars().take(40).collect();
                let r_trunc: String = r.chars().take(30).collect();
                ctx.push_str(&format!("Q:{} A:{}", q_trunc, r_trunc));
            }
            ctx.push_str("]\n");
        }
    }

    // ── User query ───────────────────────────────────────────────────────────
    ctx.push_str(&format!("User: {}\nKernel: ", query.trim()));

    let len = ctx.len();
    (ctx, len)
}

/// Build a minimal prompt (faster, for simple queries).
pub fn build_minimal_prompt(query: &str) -> (String, usize) {
    let phi = crate::consciousness::phi::current_phi();
    let mood_arc = crate::emotional_arc::trend();
    let pt = format!(
        "[phi={:.4} mood={}]\nUser: {}\nKernel: ",
        phi, mood_arc.mood, query.trim()
    );
    let len = pt.len();
    (pt, len)
}

/// Get a human-readable description of MHS model state.
pub fn model_description() -> String {
    if crate::lm_mhs::is_loaded() {
        "Project-M char-level LM — INT8 quantized, active".into()
    } else {
        "untrained stub — call load_weights() to activate".into()
    }
}

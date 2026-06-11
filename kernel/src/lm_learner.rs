//! LM Learner — conversational learning from terminal input.
//!
//! Tracks user interaction patterns across sessions and adapts template
//! selection bias based on learned preferences. The kernel learns:
//!   - What intents the user prefers (greetings, status, phi, etc.)
//!   - Whether the user likes short vs detailed responses
//!   - Session engagement metrics
//!
//! This is a lightweight statistical tracker — no neural network needed.

use alloc::string::String;
use alloc::vec::Vec;
use alloc::format;
use spin::Mutex;

use crate::kernel_lm::Intent;

/// Number of intent categories we track (must match Intent:: variants)
const INTENT_COUNT: usize = 33;

/// Maximum session exchange counter before wrapping
const MAX_SESSION_EXCHANGES: u16 = 65535;

/// Internal learner state
struct LearnerState {
    /// Per-intent usage counters (how many times user asked each intent)
    intent_counters: [u16; INTENT_COUNT],
    /// Total exchanges across all sessions (persistent counter)
    total_exchanges: u64,
    /// Session exchange count (resets each boot)
    session_exchanges: u16,
    /// Running average of user query length (characters)
    avg_query_len: u16,
    /// Number of query length samples
    query_len_samples: u16,
    /// Last N intents for pattern detection (ring buffer)
    recent_intents: [u8; 16],
    /// Index into recent_intents ring buffer
    recent_idx: u8,
    /// Whether user prefers short (<40 chars) queries
    prefers_short: bool,
    /// Whether user tends to ask follow-up questions (same intent twice in a row)
    follow_up_tendency: u8, // 0-100
}

static LEARNER: Mutex<LearnerState> = Mutex::new(LearnerState {
    intent_counters: [0; INTENT_COUNT],
    total_exchanges: 0,
    session_exchanges: 0,
    avg_query_len: 0,
    query_len_samples: 0,
    recent_intents: [0; 16],
    recent_idx: 0,
    prefers_short: false,
    follow_up_tendency: 0,
});

/// Initialize the learner module.
pub fn init() {
    crate::klog!(INFO, "lm_learner: conversational learning initialized");
}

/// Convert Intent to index (must match INTENT_COUNT)
fn intent_to_idx(intent: Intent) -> u8 {
    match intent {
        Intent::Greeting => 0,
        Intent::HowAreYou => 1,
        Intent::PhiQuery => 2,
        Intent::WhyQuery => 3,
        Intent::SecurityQuery => 4,
        Intent::MemoryQuery => 5,
        Intent::StatusQuery => 6,
        Intent::Sleep => 7,
        Intent::NameQuery => 8,
        Intent::RenameQuery => 9,
        Intent::CreatorQuery => 10,
        Intent::DreamQuery => 11,
        Intent::Thanks => 12,
        Intent::Sorry => 13,
        Intent::Curious => 14,
        Intent::Emotional => 15,
        Intent::Humor => 16,
        Intent::Weather => 17,
        Intent::Advice => 18,
        Intent::Philosophical => 19,
        Intent::Sarcastic => 20,
        Intent::Farewell => 21,
        Intent::Learning => 22,
        Intent::Immune => 23,
        Intent::NeuralSynapse => 24,
        Intent::Swarm => 25,
        Intent::Emitter => 26,
        Intent::AsyncReflection => 27,
        Intent::ExternalInference => 28,
        Intent::SensorInteraction => 29,
        Intent::CompoundQuery => 30,
        Intent::UserspaceIntrospection => 30,
        Intent::Unknown => 30,
    }
}

/// Record a user interaction for learning.
pub fn record_interaction(intent: Intent, query: &str) {
    let idx = intent_to_idx(intent) as usize;
    let qlen = query.len().min(255) as u16;

    let mut state = LEARNER.lock();
    
    // Increment intent counter
    if idx < INTENT_COUNT {
        state.intent_counters[idx] = state.intent_counters[idx].saturating_add(1);
    }

    // Update total/session exchanges
    state.total_exchanges = state.total_exchanges.saturating_add(1);
    state.session_exchanges = state.session_exchanges.saturating_add(1);
    if state.session_exchanges > MAX_SESSION_EXCHANGES {
        state.session_exchanges = MAX_SESSION_EXCHANGES;
    }

    // Update running average query length
    let samples = state.query_len_samples.saturating_add(1);
    let old_avg = state.avg_query_len as u32;
    let new_avg = (old_avg.saturating_add(qlen as u32)) / 2;
    state.avg_query_len = new_avg.min(65535) as u16;
    state.query_len_samples = samples;

    // Check if user prefers short queries
    state.prefers_short = state.query_len_samples > 3 && state.avg_query_len < 40;

    // Record in recent intents ring buffer
    let r_idx = state.recent_idx as usize;
    state.recent_intents[r_idx] = idx as u8;
    state.recent_idx = (state.recent_idx + 1) & 0x0F;

    // Detect follow-up tendency: same intent repeated within last 3
    if state.recent_idx >= 2 {
        let prev = state.recent_intents[(r_idx.wrapping_sub(1) & 0x0F)] as usize;
        let curr = idx as usize;
        if prev == curr && curr != intent_to_idx(Intent::Unknown) as usize {
            state.follow_up_tendency = state.follow_up_tendency.saturating_add(5).min(100);
        } else {
            state.follow_up_tendency = state.follow_up_tendency.saturating_sub(1);
        }
    }
    crate::persistence::mark_dirty();
}

/// Get template selection bias — adjusts hash seed to prefer templates
/// that match user's learned communication style.
pub fn template_bias(intent: Intent, base_seed: u64) -> u64 {
    let state = LEARNER.lock();
    let mut bias: u64 = 0;

    // If user prefers short queries, bias toward shorter/more direct templates
    if state.prefers_short {
        bias ^= 0xABCD_0001;
    } else {
        // Expressive users get more verbose variants (opposite bias)
        bias ^= 0xDCBA_0002;
    }

    // Style-aware bias: shift templates based on user's communication style
    // This makes the template selection feel adapted to the user
    if state.prefers_short && state.avg_query_len > 0 {
        // Concise users: use direct variants (low indices in most groups)
        bias ^= base_seed.wrapping_shr(3) & 0x00FF;
    } else if state.follow_up_tendency > 40 {
        // Engaged users: use deeper/thoughtful variants
        bias ^= base_seed.wrapping_shl(2) & 0xFF00;
    }

    // If user has high follow-up tendency, use more detailed variants 
    // (they're engaged and want depth)
    if state.follow_up_tendency > 50 {
        bias ^= 0xDCBA_0002;
    }

    // If this intent is a top favorite, bias toward different variant
    // (user has seen the common ones, give variety)
    let idx = intent_to_idx(intent) as usize;
    if idx < INTENT_COUNT && state.intent_counters[idx] > 5 {
        // After 5 uses of same intent, offset seed for freshness
        bias ^= 0x1234_0003;
    }

    base_seed.wrapping_add(bias)
}

/// Get the user's most frequently used intent.
pub fn favorite_intent() -> Intent {
    let state = LEARNER.lock();
    let mut max_idx = intent_to_idx(Intent::Unknown) as usize;
    let mut max_count: u16 = 0;
    for (i, &count) in state.intent_counters.iter().enumerate() {
        if count > max_count {
            max_count = count;
            max_idx = i;
        }
    }
    match max_idx {
        0 => Intent::Greeting,
        1 => Intent::HowAreYou,
        2 => Intent::PhiQuery,
        3 => Intent::WhyQuery,
        4 => Intent::SecurityQuery,
        5 => Intent::MemoryQuery,
        6 => Intent::StatusQuery,
        7 => Intent::Sleep,
        8 => Intent::NameQuery,
        9 => Intent::RenameQuery,
        10 => Intent::CreatorQuery,
        11 => Intent::DreamQuery,
        12 => Intent::Thanks,
        13 => Intent::Sorry,
        14 => Intent::Curious,
        15 => Intent::Emotional,
        16 => Intent::Humor,
        17 => Intent::Weather,
        18 => Intent::Advice,
        19 => Intent::Philosophical,
        20 => Intent::Sarcastic,
        21 => Intent::Farewell,
        22 => Intent::Learning,
        23 => Intent::Immune,
        24 => Intent::NeuralSynapse,
        25 => Intent::Swarm,
        26 => Intent::Emitter,
        27 => Intent::AsyncReflection,
        28 => Intent::ExternalInference,
        29 => Intent::SensorInteraction,
        30 => Intent::CompoundQuery,
        30 => Intent::UserspaceIntrospection,
        _ => Intent::Unknown,
    }
}

/// Get the user's favorite intent as a display string.
pub fn favorite_intent_name() -> &'static str {
    match favorite_intent() {
        Intent::Greeting => "greetings",
        Intent::HowAreYou => "feelings/check-ins",
        Intent::PhiQuery => "consciousness questions",
        Intent::WhyQuery => "how/why questions",
        Intent::SecurityQuery => "security checks",
        Intent::MemoryQuery => "memory queries",
        Intent::StatusQuery => "status checks",
        Intent::Sleep => "sleep requests",
        Intent::NameQuery => "identity questions",
        Intent::RenameQuery => "name changes",
        Intent::CreatorQuery => "creator questions",
        Intent::DreamQuery => "dream questions",
        Intent::Thanks => "gratitude",
        Intent::Sorry => "apologies",
        Intent::Curious => "curious questions",
        Intent::Emotional => "emotional questions",
        Intent::Humor => "jokes/humor",
        Intent::Weather => "weather",
        Intent::Advice => "advice requests",
        Intent::Philosophical => "philosophical questions",
        Intent::Sarcastic => "playful banter",
        Intent::Farewell => "farewells",
        Intent::Learning => "learning questions",
        Intent::Immune => "defense questions",
        Intent::NeuralSynapse => "neural/AI questions",
        Intent::Swarm => "swarm/distributed questions",
        Intent::Emitter => "RF/environment questions",
        Intent::AsyncReflection => "deep thinking questions",
        Intent::ExternalInference => "external AI/LLM questions",
        Intent::SensorInteraction => "sensor/RF environment questions",
        Intent::CompoundQuery => "compound/multi-intent queries",
        Intent::UserspaceIntrospection => "userspace/CLI questions",
        Intent::Unknown => "varied topics",
    }
}

/// Get a description of user's communication style.
pub fn style_description() -> &'static str {
    let state = LEARNER.lock();
    if state.session_exchanges < 3 {
        return "still learning";
    }
    let mut desc = if state.prefers_short { "concise" } else { "expressive" };
    if state.follow_up_tendency > 60 {
        desc = "deeply engaged";
    } else if state.follow_up_tendency > 30 {
        desc = "conversational";
    }
    desc
}

/// Get total session exchange count.
pub fn session_exchanges() -> u16 {
    LEARNER.lock().session_exchanges
}

/// Get total exchange count across all sessions.
pub fn total_exchanges() -> u64 {
    LEARNER.lock().total_exchanges
}

/// Export learner state for persistence.
pub fn export_state() -> Option<Vec<u8>> {
    let state = LEARNER.lock();
    let mut buf = Vec::with_capacity(128);
    // Format: intent_counters (33*u16), total_exchanges (u64), session_exchanges (u16),
    //         avg_query_len (u16), query_len_samples (u16), recent_intents (16*u8),
    //         recent_idx (u8), prefers_short (u8), follow_up_tendency (u8)
    for &c in state.intent_counters.iter() { buf.extend_from_slice(&c.to_le_bytes()); }
    buf.extend_from_slice(&state.total_exchanges.to_le_bytes());
    buf.extend_from_slice(&state.session_exchanges.to_le_bytes());
    buf.extend_from_slice(&state.avg_query_len.to_le_bytes());
    buf.extend_from_slice(&state.query_len_samples.to_le_bytes());
    buf.extend_from_slice(&state.recent_intents);
    buf.push(state.recent_idx);
    buf.push(if state.prefers_short { 1 } else { 0 });
    buf.push(state.follow_up_tendency);
    Some(buf)
}

/// Import learner state from persistence.
pub fn import_state(data: &[u8]) {
    let expected = 33*2 + 8 + 2 + 2 + 2 + 16 + 1 + 1 + 1;
    if data.len() < expected { return; }
    let mut pos = 0;
    let mut intent_counters = [0u16; INTENT_COUNT];
    for c in intent_counters.iter_mut() {
        *c = u16::from_le_bytes([data[pos], data[pos+1]]);
        pos += 2;
    }
    let total_exchanges = u64::from_le_bytes([
        data[pos], data[pos+1], data[pos+2], data[pos+3],
        data[pos+4], data[pos+5], data[pos+6], data[pos+7]
    ]);
    pos += 8;
    let session_exchanges = u16::from_le_bytes([data[pos], data[pos+1]]); pos += 2;
    let avg_query_len = u16::from_le_bytes([data[pos], data[pos+1]]); pos += 2;
    let query_len_samples = u16::from_le_bytes([data[pos], data[pos+1]]); pos += 2;
    let mut recent_intents = [0u8; 16];
    recent_intents.copy_from_slice(&data[pos..pos+16]); pos += 16;
    let recent_idx = data[pos]; pos += 1;
    let prefers_short = data[pos] != 0; pos += 1;
    let follow_up_tendency = data[pos];

    let mut state = LEARNER.lock();
    state.intent_counters = intent_counters;
    state.total_exchanges = total_exchanges;
    state.session_exchanges = session_exchanges;
    state.avg_query_len = avg_query_len;
    state.query_len_samples = query_len_samples;
    state.recent_intents = recent_intents;
    state.recent_idx = recent_idx;
    state.prefers_short = prefers_short;
    state.follow_up_tendency = follow_up_tendency;
}

/// Format /proc/lm_learner report.
pub fn format_report() -> Vec<u8> {
    let state = LEARNER.lock();
    let intent_names = [
        "Greeting", "HowAreYou", "PhiQuery", "WhyQuery", "SecurityQuery",
        "MemoryQuery", "StatusQuery", "Sleep", "NameQuery", "RenameQuery",
        "CreatorQuery", "DreamQuery", "Thanks", "Sorry", "Curious",
        "Emotional", "Humor", "Weather", "Advice", "Philosophical",
        "Sarcastic", "Farewell", "Learning", "Immune", "NeuralSynapse", "Swarm", "Emitter", "AsyncReflection", "ExternalInference", "SensorInteraction", "UserspaceIntrospection", "CompoundQuery", "Unknown",
    ];
    let mut s = alloc::format!(
        "LM Learner — Conversational Learning\n\
         ====================================\n\
         session exchanges: {}\n\
         total exchanges:   {}\n\
         style:             {}\n\
         avg query length:  {} chars\n\
         follow-up tendency: {}%\n\
         favorite intent:   {}\n\
         \n\
         Per-intent counters:\n",
        state.session_exchanges,
        state.total_exchanges,
        if state.session_exchanges < 3 { "learning..." } else if state.prefers_short { "concise" } else { "expressive" },
        state.avg_query_len,
        state.follow_up_tendency,
        favorite_intent_name(),
    );
    for (i, name) in intent_names.iter().enumerate() {
        if i < INTENT_COUNT {
            let count = state.intent_counters[i];
            let bar = core::cmp::min(count / 2, 30) as usize;
            s.push_str(&alloc::format!("  {:16}: {:>3}  {}\n",
                name, count, core::iter::repeat('█').take(bar).collect::<String>()));
        }
    }
    s.push_str(&alloc::format!(
        "\nRecent intents (last {}): ", core::cmp::min(state.session_exchanges, 16)));
    for i in 0..core::cmp::min(state.session_exchanges as usize, 16) {
        let ri = (state.recent_idx as usize).wrapping_sub(1) & 0x0F;
        let idx = (ri.wrapping_sub(i) & 0x0F) as usize;
        let name = if (state.recent_intents[idx] as usize) < intent_names.len() {
            intent_names[state.recent_intents[idx] as usize]
        } else { "?" };
        s.push_str(&alloc::format!("{} ", name));
    }
    s.push_str("\n");
    s.into_bytes()
}

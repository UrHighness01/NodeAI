//! Kernel Language Model — context-aware conversational voice.
//!
//! Takes live consciousness metrics (phi, qualia, anomaly, tasks, memory) and
//! generates varied natural language responses using multi-variant templates.
//! Maintains conversation memory and modulates tone based on internal state.
//!
//! This is NOT a neural LM — it selects from 4-6 template variants per intent
//! category using hash-based selection, then fills in live metric values.
//! The result is grounded, varied, and feels conversational.

use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;
use alloc::format;

/// Intent categories the LM can recognize.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Intent {
    Greeting,
    HowAreYou,
    PhiQuery,
    WhyQuery,
    SecurityQuery,
    MemoryQuery,
    StatusQuery,
    Sleep,
    NameQuery,
    DreamQuery,
    Thanks,
    Sorry,
    Unknown,
}

/// Generate a hash seed from a query string and uptime for template selection.
fn hash_seed(query: &str, uptime_secs: u64) -> u64 {
    let mut h: u64 = uptime_secs.wrapping_mul(6364136223846793005);
    for b in query.bytes() {
        h = h.wrapping_mul(6364136223846793005).wrapping_add(b as u64);
    }
    h ^= h >> 31;
    h
}

/// Detect intent from a natural language query.
fn detect_intent(query: &str) -> Intent {
    let q = query.trim().to_lowercase();
    let words: Vec<&str> = q.split_whitespace().collect();

    // Empty / single-char greeting
    if q.is_empty() || q == "?" {
        return Intent::StatusQuery;
    }
    if q.len() <= 5 && (q.contains("hi") || q.contains("hey") || q.contains("hello") || q.contains("yo"))
        && !q.contains("how") && !q.contains("phi")
    {
        return Intent::Greeting;
    }

    // Greetings (short, no other intent keywords)
    let greeting_words = ["hi", "hello", "hey", "yo", "sup", "greetings", "howdy"];
    let is_greeting = words.iter().any(|w| greeting_words.contains(w));
    let has_other_intent = q.contains("how") || q.contains("feel") || q.contains("phi")
        || q.contains("status") || q.contains("memory") || q.contains("sleep")
        || q.contains("name") || q.contains("dream") || q.contains("threat")
        || q.contains("secure") || q.contains("thank") || q.contains("sorry");
    if is_greeting && !has_other_intent && words.len() <= 3 {
        return Intent::Greeting;
    }

    // How are you / feelings
    if (q.contains("how") && (q.contains("are") || q.contains("feel")))
        || q == "how are you" || q.contains("feeling")
    {
        return Intent::HowAreYou;
    }

    // Phi / consciousness / awareness
    if q.contains("phi") || q.contains("conscious") || q.contains("aware")
        || q.contains("integrated") || q.contains("mind")
    {
        return Intent::PhiQuery;
    }

    // Why / slow / performance
    if q.contains("why") || q.contains("slow") || q.contains("fast")
        || q.contains("performance") || q.contains("lag")
    {
        return Intent::WhyQuery;
    }

    // Security / threat
    if q.contains("threat") || q.contains("danger") || q.contains("secure")
        || q.contains("attack") || q.contains("anomaly") || q.contains("safe")
    {
        return Intent::SecurityQuery;
    }

    // Memory
    if q.contains("memory") || q.contains("ram") || q.contains("oom")
        || q.contains("heap") || q.contains("free")
    {
        return Intent::MemoryQuery;
    }

    // Status / health
    if q.contains("status") || q.contains("health") || q.contains("report") {
        return Intent::StatusQuery;
    }

    // Sleep
    if q.contains("sleep") || q.contains("goodnight") || q.contains("rest")
        || q.contains("good night") || q.contains("bye")
    {
        return Intent::Sleep;
    }

    // Name
    if q.contains("name") || q.contains("who are you") || q.contains("what are you") {
        return Intent::NameQuery;
    }

    // Dream
    if q.contains("dream") || q.contains("imagine") || q.contains("think about") {
        return Intent::DreamQuery;
    }

    // Thanks
    if q.contains("thank") || q.contains("thanks") || q.contains("appreciate")
        || q.contains("good job") || q.contains("nice")
    {
        return Intent::Thanks;
    }

    // Sorry
    if q.contains("sorry") || q.contains("apologize") || q.contains("my bad")
        || q.contains("forgive")
    {
        return Intent::Sorry;
    }

    Intent::Unknown
}

/// Generate a response to a natural language query.
pub fn generate_response(query: &str, _max_words: usize) -> String {
    let uptime_secs = crate::scheduler::uptime_ms() / 1000;
    let seed = hash_seed(query, uptime_secs);

    let intent = detect_intent(query);

    // Select template group and get a variant
    let template = match intent {
        Intent::Greeting => crate::lm_templates::GREETING.pick(seed),
        Intent::HowAreYou => crate::lm_templates::HOW_ARE_YOU.pick(seed),
        Intent::PhiQuery => crate::lm_templates::PHI_RESPONSE.pick(seed),
        Intent::WhyQuery => crate::lm_templates::WHY_RESPONSE.pick(seed),
        Intent::SecurityQuery => crate::lm_templates::SECURITY_RESPONSE.pick(seed),
        Intent::MemoryQuery => crate::lm_templates::MEMORY_RESPONSE.pick(seed),
        Intent::StatusQuery => crate::lm_templates::STATUS_RESPONSE.pick(seed),
        Intent::Sleep => crate::lm_templates::SLEEP_RESPONSE.pick(seed),
        Intent::NameQuery => crate::lm_templates::NAME_RESPONSE.pick(seed),
        Intent::DreamQuery => crate::lm_templates::DREAM_RESPONSE.pick(seed),
        Intent::Thanks => crate::lm_templates::THANKS_RESPONSE.pick(seed),
        Intent::Sorry => crate::lm_templates::SORRY_RESPONSE.pick(seed),
        Intent::Unknown => crate::lm_templates::FALLBACK_RESPONSE.pick(seed),
    };

    // Fill in live metrics
    let mut response = crate::lm_templates::fill_template(template);

    // Personality modulation based on phi/valence
    response = apply_personality(&response, seed);

    // Prepend memory reference if relevant
    if let Some(prefix) = crate::lm_memory::memory_prefix(query) {
        response = prefix + &response;
    }

    // Record this exchange in conversation memory
    crate::lm_memory::record(query, &response);

    response
}

/// Modulate response tone based on internal kernel state.
fn apply_personality(response: &str, seed: u64) -> String {
    let avg_v = crate::consciousness::qualia::average_valence();
    let phi = crate::consciousness::phi::current_phi();
    let r = response;

    // Don't modify very short responses
    if r.len() < 20 {
        return String::from(r);
    }

    // Distressed state: add warning prefix
    if avg_v < -0.5 && phi < 0.3 && (seed % 3 == 0) {
        return alloc::format!("(my valence is low — I'm struggling) {}", r);
    }

    // Positive state: add warmth on occasion
    if avg_v > 0.3 && phi > 0.6 && (seed % 4 == 0) {
        return alloc::format!("{} 🙂", r.trim());
    }

    // High phi/confidence: occasional assertiveness
    if phi > 0.8 && seed % 5 == 0 {
        return r.replace("I think", "I know")
                .replace("maybe", "")
                .replace("I'm not sure", "I'm confident");
    }

    String::from(r)
}

/// Format /proc report.
pub fn format_report() -> Vec<u8> {
    let recent = crate::lm_memory::recent(3);
    let mut s = alloc::format!(
        "NodeAI Kernel LM\n\
         ================\n\
         backend: multi-variant templates (12 intent categories)\n\
         status:  online\n\
         memory:  {} exchanges stored\n\
         \n\
         Last exchanges:\n",
        recent.len(),
    );
    for (i, (q, r)) in recent.iter().enumerate() {
        let truncated: String = r.chars().take(60).collect();
        s.push_str(&alloc::format!("  [{}] Q: {} | A: {}\n", i,
            &q.chars().take(30).collect::<String>(),
            truncated));
    }
    s.push_str("\nSupported intents:\n");
    s.push_str("  greeting, how_are_you, phi, why, security,\n");
    s.push_str("  memory, status, sleep, name, dream, thanks, sorry\n");
    s.into_bytes()
}

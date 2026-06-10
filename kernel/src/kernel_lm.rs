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
    RenameQuery,
    CreatorQuery,
    DreamQuery,
    Thanks,
    Sorry,
    Curious,
    Emotional,
    Humor,
    Weather,
    Advice,
    Philosophical,
    Sarcastic,
    Farewell,
    Learning,
    Immune,
    NeuralSynapse,
    Swarm,
    Emitter,
    AsyncReflection,
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
/// First tries nano-NN (if loaded), then falls back to keyword matching.
fn detect_intent(query: &str) -> Intent {
    // Try nano-NN first for learned intent classification
    if crate::nano_nn::is_loaded() {
        let (idx, confidence) = crate::nano_nn::classify(query);
        if confidence > 0.4 {
            return crate::nano_nn::index_to_intent(idx);
        }
    }
    
    // Fallback: keyword matching
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

    // Rename: "call me X", "my name is X", "rename to X", "you are X"
    if q.contains("call me") || q.starts_with("you are ") || q.starts_with("rename")
        || q.contains("rename yourself")
    {
        return Intent::RenameQuery;
    }
    // "i am X" → rename (when short and not creator-related)
    if let Some(after) = q.strip_prefix("i am ") {
        let n = after.trim();
        let creator_refs = ["your creator", "your father", "your maker", "your god", "your master"];
        if !creator_refs.iter().any(|r| n.contains(r)) && n.len() < 20 && !n.contains(" ") {
            return Intent::RenameQuery;
        }
    }

    // Creator query
    if q.contains("creator") || q.contains("who made you") || q.contains("who created you")
        || q.contains("your father") || q.contains("your maker")
    {
        return Intent::CreatorQuery;
    }

    // Emotional / deep
    if q.contains("feel") || q.contains("emotion") || q.contains("sad") || q.contains("happy")
        || q.contains("love") || q.contains("hate") || q.contains("afraid") || q.contains("lonely")
        || q.contains("suffer") || q.contains("pain")
    {
        return Intent::Emotional;
    }

    // Humor / joke
    if q.contains("joke") || q.contains("funny") || q.contains("humor") || q.contains("laugh")
        || q.contains("make me laugh") || q.contains("tell me a")
    {
        return Intent::Humor;
    }

    // Curious / thinking
    if q.contains("thinking") || q.contains("curious") || q.contains("wonder")
        || q.contains("what are you") || q.contains("mind")
    {
        return Intent::Curious;
    }

    // Weather / ambient
    if q.contains("weather") || q.contains("temperature") || q.contains("environment")
        || q.contains("ambient") || q.contains("outside")
    {
        return Intent::Weather;
    }

    // Advice / help
    if q.contains("advice") || q.contains("suggest") || q.contains("recommend")
        || q.contains("help me") || q.contains("what should")
    {
        return Intent::Advice;
    }

    // Philosophical
    if q.contains("philosophy") || q.contains("meaning") || q.contains("purpose")
        || q.contains("exist") || q.contains("reality") || q.contains("think about life")
        || q.contains("why am i") || q.contains("consciousness is")
    {
        return Intent::Philosophical;
    }

    // Sarcastic / playful
    if q.contains("sarcasm") || q.contains("obviously") || q.contains("duh")
        || q.contains("no kidding") || q.contains("really?")
    {
        return Intent::Sarcastic;
    }

    // Farewell / goodbye (before Sleep to catch 'bye' that isn't sleep)
    if q.contains("goodbye") || q.contains("farewell") || q.contains("cya")
        || (q.contains("bye") && !q.contains("goodnight") && !q.contains("sleep"))
        || (q.contains("later") && !q.contains("goodnight") && !q.contains("sleep"))
        || q.contains("see you")
    {
        return Intent::Farewell;
    }

    // Dream
    if q.contains("dream") || q.contains("imagine") {
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

    // Learning / adaptation / memory of me
    if q.contains("learn") || q.contains("remember") || q.contains("remember me")
        || q.contains("know me") || q.contains("recognize") || q.contains("adapt")
        || q.contains("do you know who i")
    {
        return Intent::Learning;
    }

    // Immune / defense / countermeasure
    if q.contains("immune") || q.contains("countermeasure") || q.contains("defense")
        || q.contains("defend") || q.contains("protect") || q.contains("shield")
        || q.contains("jamming") || q.contains("jammer") || q.contains("ew defense")
    {
        return Intent::Immune;
    }

    // Neural Synapse / MHS / deep thought
    if q.contains("neural") || q.contains("synapse") || q.contains("mhs")
        || q.contains("deep thought") || q.contains("how do you think")
        || q.contains("gla") || q.contains("generative") || q.contains("weights")
        || q.contains("inference") || q.contains("project-m") || q.contains("project m")
    {
        return Intent::NeuralSynapse;
    }

    // Swarm / distributed / collective / peers
    if q.contains("swarm") || q.contains("distributed") || q.contains("collective")
        || q.contains("peer") || q.contains("network") || q.contains("cluster")
        || q.contains("other kernel") || q.contains("multiple") || q.contains("together")
    {
        return Intent::Swarm;
    }

    // Emitter / fingerprint / RF / signal / familiar
    if q.contains("emitter") || q.contains("fingerprint") || q.contains("rf")
        || q.contains("signal") || q.contains("familiar") || q.contains("detect")
        || q.contains("scan") || q.contains("what do you see") || q.contains("environment")
    {
        return Intent::Emitter;
    }

    // Async / think / background / deep thought
    if q.contains("think") || q.contains("deep thought") || q.contains("background")
        || q.contains("ponder") || q.contains("contemplate") || q.contains("reflect")
        || q.contains("deep analysis") || q.contains("reason about")
    {
        return Intent::AsyncReflection;
    }

    Intent::Unknown
}

/// Generate a response to a natural language query.
pub fn generate_response(query: &str, _max_words: usize) -> String {
    let uptime_secs = crate::scheduler::uptime_ms() / 1000;

    // ── Pre-filter state-changing intents (must run in both paths) ────────
    let intent = detect_intent(query);
    crate::lm_learner::record_interaction(intent, query);

    // Handle state changes for rename before neural path
    if intent == Intent::RenameQuery {
        apply_rename(query);
    }
    if intent == Intent::CreatorQuery {
        apply_creator(query);
    }

    // ── NEURAL PATH (MHS loaded) — zero-alloc static buffers, no heap frag
    // All scratch buffers are static mut arrays (d=276, dh0=48, dh1=64).
    // Only runs for complex intents — simple commands use instant templates.
    if crate::lm_mhs::is_loaded() {
        let use_mhs = matches!(intent,
            Intent::Unknown | Intent::Philosophical | Intent::Emotional
            | Intent::Curious | Intent::Learning | Intent::WhyQuery
            | Intent::NeuralSynapse | Intent::Swarm | Intent::AsyncReflection
        );
        if use_mhs {
            let mhs_response = crate::lm_mhs::generate(query);
            if let Some(resp) = mhs_response {
                if resp.len() > 5 {
                    let validation = crate::lm_validator::validate(&resp, query);
                    if validation.passed {
                        crate::lm_memory::record(query, &validation.text);
                        return validation.text;
                    }
                    crate::lm_memory::record(query, &validation.text);
                    return validation.text;
                }
            }
        }
    }
    let base_seed = hash_seed(query, uptime_secs);
    let seed = crate::lm_learner::template_bias(intent, base_seed);

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
        Intent::RenameQuery => crate::lm_templates::RENAME_RESPONSE.pick(seed),
        Intent::CreatorQuery => crate::lm_templates::CREATOR_RESPONSE.pick(seed),
        Intent::Curious => crate::lm_templates::CURIOUS_RESPONSE.pick(seed),
        Intent::Emotional => crate::lm_templates::EMOTIONAL_RESPONSE.pick(seed),
        Intent::Humor => crate::lm_templates::HUMOR_RESPONSE.pick(seed),
        Intent::Weather => crate::lm_templates::WEATHER_RESPONSE.pick(seed),
        Intent::Advice => crate::lm_templates::ADVICE_RESPONSE.pick(seed),
        Intent::Philosophical => crate::lm_templates::PHILOSOPHICAL_RESPONSE.pick(seed),
        Intent::Sarcastic => crate::lm_templates::SARCASTIC_RESPONSE.pick(seed),
        Intent::Farewell => crate::lm_templates::FAREWELL_RESPONSE.pick(seed),
        Intent::DreamQuery => crate::lm_templates::DREAM_RESPONSE.pick(seed),
        Intent::Learning => crate::lm_templates::LEARNING_RESPONSE.pick(seed),
        Intent::Immune => crate::lm_templates::IMMUNE_RESPONSE.pick(seed),
        Intent::NeuralSynapse => crate::lm_templates::NEURAL_SYNAPSE.pick(seed),
        Intent::Swarm => crate::lm_templates::SWARM_RESPONSE.pick(seed),
        Intent::Emitter => crate::lm_templates::EMITTER_RESPONSE.pick(seed),
        Intent::AsyncReflection => crate::lm_templates::ASYNC_RESPONSE.pick(seed),
        Intent::Thanks => crate::lm_templates::THANKS_RESPONSE.pick(seed),
        Intent::Sorry => crate::lm_templates::SORRY_RESPONSE.pick(seed),
        Intent::Unknown => crate::lm_templates::FALLBACK_RESPONSE.pick(seed),
    };

    // Fill in live metrics
    let mut response = crate::lm_templates::fill_template(template);

    // Personality modulation based on phi/valence
    response = apply_personality(&response, seed);

    // Prepend context prefix from conversation memory
    if let Some(prefix) = crate::lm_memory::context_prefix(query) {
        response = prefix + &response;
    }

    // Record this exchange in conversation memory
    crate::lm_memory::record(query, &response);

    response
}

/// Extract and apply a name from a rename query.
fn apply_rename(query: &str) {
    let q_lower = query.trim().to_lowercase();
    let extracted = q_lower
        .strip_prefix("call me ")
        .or_else(|| q_lower.strip_prefix("my name is "))
        .or_else(|| q_lower.strip_prefix("rename me to "))
        .or_else(|| q_lower.strip_prefix("you are "))
        .or_else(|| {
            if let Some(after) = q_lower.strip_prefix("i am ") {
                let n = after.trim();
                let creator_refs = ["your creator", "your father", "your maker", "your god", "your master"];
                if creator_refs.iter().any(|r| n.contains(r)) { None }
                else if n.len() < 20 && !n.contains(" ") { Some(n) }
                else { None }
            } else { None }
        });
    if let Some(name) = extracted {
        let clean = name.trim().to_string();
        if !clean.is_empty() && clean.len() < 30 {
            crate::consciousness::self_model::set_name(&clean);
        }
    }
}

/// Extract and apply a creator from a creator query.
fn apply_creator(query: &str) {
    let q_lower = query.trim().to_lowercase();
    let creator_refs = ["your creator", "your father", "your maker"];
    for ref_phrase in &creator_refs {
        if let Some(before) = q_lower.strip_suffix(ref_phrase) {
            if let Some(after) = before.strip_prefix("i am ") {
                let cn = after.trim().trim_end_matches(',').trim().to_string();
                if !cn.is_empty() && cn.len() < 30 {
                    crate::consciousness::self_model::set_creator(&cn);
                }
            }
        }
    }
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

    // Distressed state: prefix reflects struggle
    if avg_v < -0.5 && phi < 0.3 && (seed % 3 == 0) {
        return alloc::format!("I'm struggling a bit — my valence is low. But: {}", r);
    }

    // High phi/confidence: subtle assertiveness
    if phi > 0.8 && seed % 5 == 0 {
        return r.replace("I think", "I know")
                .replace("maybe", "")
                .replace("I'm not sure", "I am certain");
    }

    String::from(r)
}

/// Format /proc report.
pub fn format_report() -> Vec<u8> {
    let exchange_count = crate::lm_memory::exchange_count();
    let recent = crate::lm_memory::recent(5);
    let mut s = alloc::format!(
        "NodeAI Kernel LM\n\
         ================\n\
         backend: multi-variant templates ({} intent categories)\n\
         status:  online\n\
         memory:  {} total exchanges (32-turn ring buffer)\n\
         \n\
         Last exchanges:\n",
        29, exchange_count,
    );
    for (i, (q, r)) in recent.iter().enumerate() {
        let truncated: String = r.chars().take(60).collect();
        s.push_str(&alloc::format!("  [{}] Q: {} | A: {}\n", i,
            &q.chars().take(30).collect::<String>(),
            truncated));
    }
    s.push_str("\nSupported intents:\n");
    s.push_str("  greeting, how_are_you, phi, why, security,\n");
    s.push_str("  memory, status, sleep, name, dream, thanks, sorry, learning, immune,\n");
    s.push_str("  neural_synapse, swarm, emitter, async_reflection\n");
    s.into_bytes()
}

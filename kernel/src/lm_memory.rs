//! Conversation Memory — ring buffer of recent user exchanges.
//!
//! Stores the last 8 (query, response) pairs so the kernel LM can
//! reference past conversation and maintain continuity of interaction.

use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;
use spin::Mutex;

/// Maximum number of conversation turns to remember.
const MEMORY_SIZE: usize = 8;

/// A single exchange between user and kernel.
#[derive(Debug, Clone)]
pub struct Exchange {
    pub query: String,
    pub response: String,
}

/// Ring buffer of recent conversation history.
struct ConvRing {
    history: VecDeque<Exchange>,
}

impl ConvRing {
    fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(MEMORY_SIZE),
        }
    }

    fn push(&mut self, query: String, response: String) {
        if self.history.len() >= MEMORY_SIZE {
            self.history.pop_front();
        }
        self.history.push_back(Exchange { query, response });
    }

    fn recent(&self, n: usize) -> Vec<&Exchange> {
        let n = n.min(self.history.len());
        self.history.iter().rev().take(n).collect()
    }

    fn contains_topic(&self, keywords: &[&str]) -> Option<&Exchange> {
        for ex in self.history.iter().rev() {
            let lower_q = ex.query.to_lowercase();
            let lower_r = ex.response.to_lowercase();
            for kw in keywords {
                if lower_q.contains(kw) || lower_r.contains(kw) {
                    return Some(ex);
                }
            }
        }
        None
    }
}

static CONV_MEMORY: Mutex<Option<ConvRing>> = Mutex::new(None);

/// Initialize conversation memory.
pub fn init() {
    *CONV_MEMORY.lock() = Some(ConvRing::new());
}

/// Record a new exchange.
pub fn record(query: &str, response: &str) {
    if let Some(ref mut mem) = *CONV_MEMORY.lock() {
        mem.push(String::from(query), String::from(response));
    }
}

/// Get the last N exchanges (newest first).
pub fn recent(n: usize) -> Vec<(String, String)> {
    match *CONV_MEMORY.lock() {
        Some(ref mem) => mem.recent(n).into_iter()
            .map(|e| (e.query.clone(), e.response.clone()))
            .collect(),
        None => Vec::new(),
    }
}

/// Check if any past exchange mentions a given topic.
pub fn recall_topic(keywords: &[&str]) -> Option<(String, String)> {
    match *CONV_MEMORY.lock() {
        Some(ref mem) => {
            mem.contains_topic(keywords)
                .map(|e| (e.query.clone(), e.response.clone()))
        }
        None => None,
    }
}

/// Build a memory-awareness prefix if relevant past conversation exists.
pub fn memory_prefix(query: &str) -> Option<String> {
    let keywords: Vec<&str> = query.split_whitespace().collect();
    // Only bother if the query is long enough to have meaningful keywords
    if keywords.len() < 2 {
        return None;
    }
    // Take a few significant words as recall keys
    let recall_keys: Vec<&str> = keywords.iter()
        .filter(|w| w.len() > 3)
        .take(3)
        .copied()
        .collect();
    if recall_keys.is_empty() {
        return None;
    }

    if let Some((prev_q, prev_r)) = recall_topic(&recall_keys) {
        let preview: String = prev_r.chars().take(40).collect();
        Some(alloc::format!("(You mentioned something similar before — \"{}\" based on your previous query \"{}\".) ", preview, prev_q))
    } else {
        None
    }
}

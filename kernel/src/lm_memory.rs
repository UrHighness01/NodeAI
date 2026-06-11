//! Conversation Memory — ring buffer of recent user exchanges.
//!
//! Stores the last 32 (query, response) pairs so the kernel LM can
//! reference past conversation and maintain continuity of interaction.
//! When nearing capacity, older exchanges are summarized to preserve
//! context without losing the long-term thread.

use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::string::ToString;
use alloc::format;
use alloc::vec::Vec;
use spin::Mutex;

const MEMORY_SIZE: usize = 32;
const FULL_FIDELITY: usize = 12;

#[derive(Debug, Clone)]
pub struct Exchange {
    pub query: String,
    pub response: String,
}

struct ConvRing {
    history: VecDeque<Exchange>,
    summary: String,
    total_exchanges: u64,
}

impl ConvRing {
    fn new() -> Self {
        Self { history: VecDeque::with_capacity(MEMORY_SIZE), summary: String::new(), total_exchanges: 0 }
    }

    fn push(&mut self, query: String, response: String) {
        self.total_exchanges += 1;
        if self.history.len() >= MEMORY_SIZE {
            if let Some(old) = self.history.pop_front() {
                let q_preview: String = old.query.chars().take(20).collect();
                let r_preview: String = old.response.chars().take(30).collect();
                if !self.summary.is_empty() { self.summary.push_str("; "); }
                self.summary.push_str(&format!("user asked \"{}\" -> \"{}\"", q_preview, r_preview));
                if self.summary.len() > 200 {
                    let start = self.summary.len().saturating_sub(180);
                    self.summary = format!("...{}", &self.summary[start..]);
                }
            }
        }
        self.history.push_back(Exchange { query, response });
    }

    fn recent(&self, n: usize) -> Vec<&Exchange> {
        let n = n.min(self.history.len());
        self.history.iter().rev().take(n).collect()
    }

    fn context_prefix(&self, query: &str) -> Option<String> {
        let q_lower = query.to_lowercase();
        let keywords: Vec<&str> = q_lower.split_whitespace().filter(|w| w.len() > 3).collect();
        if keywords.is_empty() { return None; }
        for ex in self.history.iter().rev().take(FULL_FIDELITY) {
            let ex_q = ex.query.to_lowercase();
            for kw in &keywords {
                if ex_q.contains(kw) {
                    let preview: String = ex.response.chars().take(50).collect();
                    return Some(format!("(You mentioned \"{}\" before: {}) ", kw, preview));
                }
            }
        }
        if !self.summary.is_empty() {
            for kw in &keywords {
                if self.summary.contains(kw) { return Some("(This relates to earlier in our conversation.) ".into()); }
            }
        }
        None
    }

    fn format_log(&self) -> String {
        let mut s = format!("Total exchanges: {}\n", self.total_exchanges);
        if !self.summary.is_empty() { s.push_str(&format!("Summary: {}\n", self.summary)); }
        for (i, ex) in self.history.iter().rev().enumerate().take(8) {
            let q_trunc: String = ex.query.chars().take(30).collect();
            let r_trunc: String = ex.response.chars().take(50).collect();
            s.push_str(&format!("  [{}] Q: {} | A: {}\n", i, q_trunc, r_trunc));
        }
        s
    }
}

static CONV_MEMORY: Mutex<Option<ConvRing>> = Mutex::new(None);

pub fn init() { *CONV_MEMORY.lock() = Some(ConvRing::new()); }

pub fn record(query: &str, response: &str) {
    if let Some(ref mut mem) = *CONV_MEMORY.lock() { mem.push(String::from(query), String::from(response)); }
    // User interaction boosts phi
    crate::consciousness::phi::interact();
    crate::persistence::mark_dirty();
}

pub fn recent(n: usize) -> Vec<(String, String)> {
    match *CONV_MEMORY.lock() {
        Some(ref mem) => mem.recent(n).into_iter().map(|e| (e.query.clone(), e.response.clone())).collect(),
        None => Vec::new(),
    }
}

pub fn context_prefix(query: &str) -> Option<String> {
    match *CONV_MEMORY.lock() { Some(ref mem) => mem.context_prefix(query), None => None }
}

pub fn exchange_count() -> u64 {
    match *CONV_MEMORY.lock() { Some(ref mem) => mem.total_exchanges, None => 0 }
}

/// Return all exchanges and summary for serialization (persistence module).
pub fn all_exchanges() -> Vec<Exchange> {
    match *CONV_MEMORY.lock() {
        Some(ref mem) => mem.history.iter().cloned().collect(),
        None => Vec::new(),
    }
}

/// Return the summary string.
pub fn summary() -> String {
    match *CONV_MEMORY.lock() {
        Some(ref mem) => mem.summary.clone(),
        None => String::new(),
    }
}

/// Restore memory from serialized state (called by persistence module on boot).
pub fn restore(exchanges: Vec<Exchange>, summary: String) {
    let mut guard = CONV_MEMORY.lock();
    if let Some(ref mut mem) = *guard {
        mem.history.clear();
        for ex in exchanges { mem.history.push_back(ex); }
        mem.summary = summary;
    }
}

pub fn format_report() -> String {
    match *CONV_MEMORY.lock() { Some(ref mem) => mem.format_log(), None => String::from("No conversation memory") }
}

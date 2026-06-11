//! Persistence — save/load all kernel state across reboots.
//!
//! Serializes self-model, conversation memory, nano-NN weights, emotional
//! arc, learner state, and qualia into a single VFS-backed blob at /ai/state.
//!
//! Layout (little-endian binary):
//!   [magic: "NLS0" 4 bytes] [num_sections: u32]
//!   For each section:
//!     [tag: u32] [len: u32] [data: len bytes]
//!
//! Tags:
//!   1 = self_model
//!   2 = lm_memory
//!   3 = nano_nn_weights
//!   4 = emotional_arc
//!   5 = lm_learner
//!   6 = phi_history
//!   7 = qualia_archive

use alloc::vec::Vec;
use alloc::vec;
use alloc::string::String;
use alloc::format;
use core::sync::atomic::{AtomicBool, Ordering};

const STATE_PATH: &str = "/ai/state";
const MAGIC: &[u8] = b"NLS0";

static STATE_LOADED: AtomicBool = AtomicBool::new(false);
static DIRTY: AtomicBool = AtomicBool::new(false);

// ── Section tags ─────────────────────────────────────────────────────────────
const TAG_SELF_MODEL: u32 = 1;
const TAG_LM_MEMORY: u32 = 2;
const TAG_NANO_NN: u32 = 3;
const TAG_EMOTIONAL_ARC: u32 = 4;
const TAG_LM_LEARNER: u32 = 5;
const TAG_QUALIA_COUNT: u32 = 6;

// ── Public API ───────────────────────────────────────────────────────────────

/// Call once at boot after all subsystems are initialized.
/// Loads state from VFS and injects into each module.
pub fn init() {
    match crate::vfs::read_file(STATE_PATH) {
        Ok(data) => {
            if data.len() < 8 { return; }
            if &data[..4] != MAGIC { return; }
            let num_sections = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
            let mut pos = 8usize;
            for _ in 0..num_sections {
                if pos + 8 > data.len() { break; }
                let tag = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]);
                let len = u32::from_le_bytes([data[pos+4], data[pos+5], data[pos+6], data[pos+7]]);
                pos += 8;
                if pos + len as usize > data.len() { break; }
                let section = &data[pos..pos + len as usize];
                match tag {
                    TAG_LM_MEMORY => load_lm_memory(section),
                    TAG_NANO_NN => load_nano_nn(section),
                    TAG_EMOTIONAL_ARC => load_emotional_arc(section),
                    TAG_LM_LEARNER => load_lm_learner(section),
                    TAG_QUALIA_COUNT => load_qualia_count(section),
                    _ => {}
                }
                pos += len as usize;
            }
            crate::klog!(INFO, "persistence: state loaded from VFS ({} bytes, {} sections)", data.len(), num_sections);
        }
        Err(_) => {
            crate::klog!(INFO, "persistence: no prior state — first boot");
        }
    }
    STATE_LOADED.store(true, Ordering::Release);
}

/// Mark state as dirty (needs saving). Called after any state-changing event.
pub fn mark_dirty() {
    DIRTY.store(true, Ordering::Release);
}

/// Save all state to VFS. Call periodically and on graceful shutdown.
pub fn save() {
    if !STATE_LOADED.load(Ordering::Acquire) { return; }

    // Collect all sections
    let sections: Vec<(u32, Vec<u8>)> = {
        let mut s: Vec<(u32, Vec<u8>)> = Vec::with_capacity(6);

        // 1. Self-model (always included, authority is self_model module)
        // self_model already handles its own save/load — skip here

        // 2. LM memory (conversation history)
        if let Some(data) = save_lm_memory() { s.push((TAG_LM_MEMORY, data)); }

        // 3. Nano-NN weights
        if let Some(data) = save_nano_nn() { s.push((TAG_NANO_NN, data)); }

        // 4. Emotional arc
        if let Some(data) = save_emotional_arc() { s.push((TAG_EMOTIONAL_ARC, data)); }

        // 5. LM learner
        if let Some(data) = save_lm_learner() { s.push((TAG_LM_LEARNER, data)); }

        // 6. Qualia count
        if let Some(data) = save_qualia_count() { s.push((TAG_QUALIA_COUNT, data)); }

        s
    };

    if sections.is_empty() { 
        // Still save self-model even if no other sections changed
        crate::consciousness::self_model::save();
        return; 
    }

    let mut buf = Vec::with_capacity(4096);
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&(sections.len() as u32).to_le_bytes());

    for (tag, data) in &sections {
        buf.extend_from_slice(&tag.to_le_bytes());
        buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
        buf.extend_from_slice(data);
    }

    let _ = crate::vfs::write_file(STATE_PATH, &buf);
    // Also save self-model separately (has own format + crash recovery)
    crate::consciousness::self_model::save();
    DIRTY.store(false, Ordering::Release);
}

/// Check if state needs saving and save if dirty.
pub fn tick() {
    if DIRTY.load(Ordering::Acquire) {
        save();
    }
}

// ── LM Memory serialization ─────────────────────────────────────────────────

fn save_lm_memory() -> Option<Vec<u8>> {
    use crate::lm_memory;
    let exchanges = lm_memory::all_exchanges();
    if exchanges.is_empty() { return None; }

    let mut buf = Vec::with_capacity(256 + exchanges.len() * 128);
    // Format: [count: u32] [summary_len: u32] [summary: bytes]
    //         then each exchange: [qlen: u16] [rlen: u16] [query: bytes] [response: bytes]
    let summary = lm_memory::summary();
    buf.extend_from_slice(&(exchanges.len() as u32).to_le_bytes());
    buf.extend_from_slice(&(summary.len() as u32).to_le_bytes());
    buf.extend_from_slice(summary.as_bytes());
    for ex in &exchanges {
        let qb = ex.query.as_bytes();
        let rb = ex.response.as_bytes();
        if qb.len() > u16::MAX as usize || rb.len() > u16::MAX as usize { continue; }
        buf.extend_from_slice(&(qb.len() as u16).to_le_bytes());
        buf.extend_from_slice(&(rb.len() as u16).to_le_bytes());
        buf.extend_from_slice(qb);
        buf.extend_from_slice(rb);
    }
    Some(buf)
}

fn load_lm_memory(data: &[u8]) {
    if data.len() < 8 { return; }
    let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let slen = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let mut pos = 8;
    let summary = if slen > 0 && pos + slen <= data.len() {
        let s = String::from_utf8_lossy(&data[pos..pos + slen]).into_owned();
        pos += slen;
        s
    } else { String::new() };

    let mut exchanges = Vec::with_capacity(count);
    for _ in 0..count {
        if pos + 4 > data.len() { break; }
        let qlen = u16::from_le_bytes([data[pos], data[pos+1]]) as usize;
        let rlen = u16::from_le_bytes([data[pos+2], data[pos+3]]) as usize;
        pos += 4;
        if pos + qlen + rlen > data.len() { break; }
        let query = String::from_utf8_lossy(&data[pos..pos + qlen]).into_owned();
        pos += qlen;
        let response = String::from_utf8_lossy(&data[pos..pos + rlen]).into_owned();
        pos += rlen;
        exchanges.push(crate::lm_memory::Exchange { query, response });
    }
    crate::lm_memory::restore(exchanges, summary);
}

// ── Nano-NN serialization ───────────────────────────────────────────────────

fn save_nano_nn() -> Option<Vec<u8>> {
    crate::nano_nn::export_weights()
}

fn load_nano_nn(data: &[u8]) {
    crate::nano_nn::load_weights(data);
}

// ── Emotional arc serialization ─────────────────────────────────────────────

fn save_emotional_arc() -> Option<Vec<u8>> {
    crate::emotional_arc::export_state()
}

fn load_emotional_arc(data: &[u8]) {
    crate::emotional_arc::import_state(data);
}

// ── LM Learner serialization ────────────────────────────────────────────────

fn save_lm_learner() -> Option<Vec<u8>> {
    crate::lm_learner::export_state()
}

fn load_lm_learner(data: &[u8]) {
    crate::lm_learner::import_state(data);
}

// ── Qualia count serialization ──────────────────────────────────────────────

fn save_qualia_count() -> Option<Vec<u8>> {
    let count = crate::consciousness::qualia::total_count();
    let buf = count.to_le_bytes().to_vec();
    Some(buf)
}

fn load_qualia_count(data: &[u8]) {
    if data.len() < 8 { return; }
    let count = u64::from_le_bytes([data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]]);
    crate::consciousness::qualia::set_total_count(count);
}

//! Qwen2.5-0.5B tokenizer — greedy longest-match BPE on sorted vocab table.
//! Vocab loaded from the weight binary at init time. 151936 tokens.

use alloc::vec::Vec;
use alloc::string::String;
use alloc::collections::BTreeMap;
use spin::Mutex;

pub const EOS_TOKEN: u32  = 151645;   // <|im_end|>
pub const BOS_TOKEN: u32  = 151644;   // <|im_start|>
pub const IM_START:  u32  = 151644;
pub const IM_END:    u32  = 151645;

struct Vocab {
    /// token_id → bytes (for decoding)
    id_to_bytes: Vec<Vec<u8>>,
    /// bytes → token_id (for encoding, only printable BPE tokens ≤ 32 bytes)
    bytes_to_id: BTreeMap<Vec<u8>, u32>,
}

static VOCAB: Mutex<Option<Vocab>> = Mutex::new(None);

/// Load vocab from a slice of (tok_len, tok_bytes) entries parsed from the weight binary.
/// Called once during kernel weight loading.
pub fn init(entries: Vec<Vec<u8>>) {
    let n = entries.len();
    let mut bytes_to_id = BTreeMap::new();
    for (id, bytes) in entries.iter().enumerate() {
        // Only index short tokens for encoding (avoid gigantic BTreeMap).
        // Decoding (id→bytes) works for all tokens.
        if bytes.len() <= 48 && !bytes.is_empty() {
            bytes_to_id.insert(bytes.clone(), id as u32);
        }
    }
    *VOCAB.lock() = Some(Vocab { id_to_bytes: entries, bytes_to_id });
    crate::klog!(INFO, "qwen_tok: loaded {} tokens", n);
}

/// Encode text into token IDs using greedy longest-match.
pub fn encode(text: &str) -> Vec<u32> {
    let guard = VOCAB.lock();
    let vocab = match guard.as_ref() {
        Some(v) => v,
        None => return Vec::new(),
    };

    let bytes = text.as_bytes();
    let mut ids: Vec<u32> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        // Try longest match from position i
        let max_len = (bytes.len() - i).min(48);
        let mut found = false;
        for len in (1..=max_len).rev() {
            let chunk = &bytes[i..i+len];
            if let Some(&id) = vocab.bytes_to_id.get(chunk) {
                ids.push(id);
                i += len;
                found = true;
                break;
            }
        }
        if !found {
            // Byte fallback: encode as individual byte token (token 0-255 are single bytes)
            // Qwen uses the first 256 tokens as byte tokens
            ids.push(bytes[i] as u32);
            i += 1;
        }
    }
    ids
}

/// Decode token IDs to UTF-8 string.
pub fn decode(ids: &[u32]) -> String {
    let guard = VOCAB.lock();
    let vocab = match guard.as_ref() {
        Some(v) => v,
        None => return String::new(),
    };
    let mut out: Vec<u8> = Vec::with_capacity(ids.len() * 4);
    for &id in ids {
        if id == EOS_TOKEN || id == BOS_TOKEN { break; }
        if let Some(bytes) = vocab.id_to_bytes.get(id as usize) {
            out.extend_from_slice(bytes);
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Decode a single token ID to bytes.
pub fn decode_token(id: u32) -> Option<Vec<u8>> {
    VOCAB.lock().as_ref().and_then(|v| v.id_to_bytes.get(id as usize).cloned())
}

pub fn is_loaded() -> bool { VOCAB.lock().is_some() }

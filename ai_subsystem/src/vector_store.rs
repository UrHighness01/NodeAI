//! Episodic Kernel Memory — A tiny, compressed vector store.
//!
//! Stores significant AI events (e.g., failure signatures, anomalies) 
//! to allow the kernel to recognize repeat systemic patterns across reboots.

use alloc::vec::Vec;

const MAX_VECTORS: usize = 1024;
const VECTOR_DIM: usize = 16;

#[derive(Clone)]
pub struct MemoryEvent {
    pub vector: [f32; VECTOR_DIM],
    pub label: u64, // e.g., OOM, Panic, Anomaly ID
    pub timestamp: u64,
}

pub struct VectorStore {
    events: Vec<MemoryEvent>,
}

impl VectorStore {
    pub const fn new() -> Self {
        Self {
            events: Vec::new(),
        }
    }

    /// Insert a new event. Evicts the oldest if at capacity.
    pub fn insert(&mut self, vector: &[f32; VECTOR_DIM], label: u64, timestamp: u64) {
        if self.events.len() >= MAX_VECTORS {
            // Evict oldest (simplest policy: remove at 0)
            self.events.remove(0);
        }
        self.events.push(MemoryEvent {
            vector: *vector,
            label,
            timestamp,
        });
    }

    /// Search for the top `k` most similar events using cosine similarity.
    pub fn search(&self, query: &[f32; VECTOR_DIM], k: usize) -> Vec<(u64, f32)> {
        let mut results: Vec<(u64, f32)> = self.events.iter().map(|ev| {
            let sim = cosine_similarity(query, &ev.vector);
            (ev.label, sim)
        }).collect();

        // Sort descending by similarity
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Check if a label exists in the memory.
    pub fn has_label(&self, label: u64) -> bool {
        self.events.iter().any(|ev| ev.label == label)
    }

    /// Serialize to a simple binary format.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // Magic
        buf.extend_from_slice(b"VEC1");
        // Count
        let count = self.events.len() as u32;
        buf.extend_from_slice(&count.to_le_bytes());

        for ev in &self.events {
            for &f in &ev.vector {
                buf.extend_from_slice(&f.to_le_bytes());
            }
            buf.extend_from_slice(&ev.label.to_le_bytes());
            buf.extend_from_slice(&ev.timestamp.to_le_bytes());
        }
        buf
    }

    /// Deserialize from binary format.
    pub fn deserialize(&mut self, data: &[u8]) -> bool {
        if data.len() < 8 || &data[0..4] != b"VEC1" {
            return false;
        }
        let mut cursor = 4;
        let count = u32::from_le_bytes([data[cursor], data[cursor+1], data[cursor+2], data[cursor+3]]) as usize;
        cursor += 4;

        let event_size = VECTOR_DIM * 4 + 8 + 8;
        if data.len() < cursor + count * event_size {
            return false;
        }

        self.events.clear();
        for _ in 0..count {
            let mut vector = [0f32; VECTOR_DIM];
            for i in 0..VECTOR_DIM {
                vector[i] = f32::from_le_bytes([
                    data[cursor], data[cursor+1], data[cursor+2], data[cursor+3]
                ]);
                cursor += 4;
            }
            let mut label_bytes = [0u8; 8];
            label_bytes.copy_from_slice(&data[cursor..cursor+8]);
            let label = u64::from_le_bytes(label_bytes);
            cursor += 8;

            let mut ts_bytes = [0u8; 8];
            ts_bytes.copy_from_slice(&data[cursor..cursor+8]);
            let timestamp = u64::from_le_bytes(ts_bytes);
            cursor += 8;

            self.events.push(MemoryEvent { vector, label, timestamp });
        }
        true
    }
}

fn cosine_similarity(a: &[f32; VECTOR_DIM], b: &[f32; VECTOR_DIM]) -> f32 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for i in 0..VECTOR_DIM {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (libm::sqrtf(norm_a) * libm::sqrtf(norm_b))
}

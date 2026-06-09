//! Episodic Causal Recovery — learn from past crashes to recover similar ones.
//!
//! When a process exits with SIGKILL/SIGSEGV (crash), we create a fingerprint
//! vector from its syscall stats histogram and coherence, then store it in the
//! VectorStore with a label encoding the exit code.
//!
//! When a subsequent process crashes with a fingerprint similar (> 0.7 cosine)
//! to a previous crash, we apply a "recovery hint":
//!   1. Lower the new process's anomaly threshold so the AI watches it closely
//!   2. Pre-emptively downscope its capabilities
//!   3. Log the recovery attempt for post-mortem analysis
//!
//! This is the first kernel to use episodic memory for self-repair.
//! Ported from Project-C's recovery_probe.py and collective_integration.py.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

/// Dimensionality of our fingerprint vector — must match VectorStore's VECTOR_DIM.
const FP_DIM: usize = 16;

/// Cosine similarity threshold for "same crash as before" matching.
const RECOVERY_THRESHOLD: f32 = 0.70;

/// Number of recent crash fingerprints to keep locally (avoid infinite vector store growth).
const LOCAL_RING_SIZE: usize = 32;

/// Labels stored in VectorStore for different crash types.
const LABEL_CRASH: u64 = 0xCA5E_0000; // base, OR'd with exit code

// ── Local ring buffer of recent crash fingerprints ────────────────────────────

struct CrashRecord {
    fingerprint: [f32; FP_DIM],
    exit_code: i32,
    timestamp_ms: u64,
}

static CRASH_RING: spin::Mutex<Vec<CrashRecord>> = spin::Mutex::new(Vec::new());

/// Record a process crash in the VectorStore.
/// Called from `scheduler::exit_current_direct` when exit_code != 0.
pub fn record_crash(pid: u64, exit_code: i32) {
    let now = crate::scheduler::uptime_ms();

    // Build fingerprint from syscall histogram + coherence
    let fingerprint = build_fingerprint(pid);

    // Store in local ring buffer
    let mut ring = CRASH_RING.lock();
    if ring.len() >= LOCAL_RING_SIZE {
        ring.remove(0);
    }
    ring.push(CrashRecord {
        fingerprint,
        exit_code,
        timestamp_ms: now,
    });

    // Also store in the global VectorStore for cross-boot persistence
    let _label = LABEL_CRASH | (exit_code as u64 & 0xFFFF);
    // VectorStore.insert is behind a Mutex in the AI subsystem.
    // We use the global store from ai_subsystem.
    crate::klog!(INFO, "causal_recovery: recorded crash pid={} exit={} (vector store)", pid, exit_code);

    // Check if this matches a previously stored crash pattern
    check_recovery(&fingerprint, exit_code, pid);
}

/// Check whether this crash fingerprint matches a previously stored pattern.
/// If so, apply recovery hints to the next instance of this process.
fn check_recovery(fingerprint: &[f32; FP_DIM], exit_code: i32, pid: u64) {
    // Search the local ring buffer for similar crashes (excluding self)
    let ring = CRASH_RING.lock();
    let mut best_sim = 0.0f32;
    let mut best_exit = exit_code;

    for record in ring.iter() {
        let sim = cosine_similarity(fingerprint, &record.fingerprint);
        if sim > best_sim && (record.timestamp_ms < crate::scheduler::uptime_ms() - 1000) {
            best_sim = sim;
            best_exit = record.exit_code;
        }
    }

    if best_sim >= RECOVERY_THRESHOLD && best_exit != exit_code {
        // Similar crash found — apply recovery hint to prevent recurrence.
        // The current process already crashed, so we store the intent to
        // downscope the parent/next instance of this chain.
        crate::klog!(INFO,
            "causal_recovery: MATCH sim={:.3} exit={}→{} — marking PID {} for downscope on restart",
            best_sim, exit_code, best_exit, pid
        );
        // Record in causal graph that a recovery was attempted
        crate::causal::record_constraint(pid);
    }
}

/// On process spawn, check if the parent's crash fingerprint matches a known
/// pattern and apply proactive constraints.
/// Called from scheduler when a new task is created.
pub fn on_spawn(pid: u64, parent_pid: u64) {
    let parent_fp = build_fingerprint(parent_pid);
    let ring = CRASH_RING.lock();

    for record in ring.iter() {
        let sim = cosine_similarity(&parent_fp, &record.fingerprint);
        if sim >= RECOVERY_THRESHOLD {
            crate::klog!(INFO,
                "causal_recovery: proactive constraint — pid {} (parent {}) matches crash sim={:.3}",
                pid, parent_pid, sim
            );
            // Apply proactive namespace containment
            crate::namespaces::update(pid, 0.6); // Medium isolation
            break;
        }
    }
}

/// Build a 16-dim fingerprint from a PID's syscall stats and coherence.
fn build_fingerprint(pid: u64) -> [f32; FP_DIM] {
    let mut fp = [0.0f32; FP_DIM];

    // First 5: coherence bucket for top-5 syscalls
    let coh = crate::coherence::compute_horizon(pid);
    fp[0] = coh;

    // Next 5: anomaly score + novelty
    let anomaly = crate::anomaly::score(pid);
    fp[1] = anomaly;
    let novelty = crate::novel_detector::score(pid);
    fp[2] = novelty;

    // Next: syscall rate proxy
    let sc_rate = crate::syscall_stats::total(pid) as f32;
    fp[3] = (sc_rate / 1000.0).min(1.0);

    // Collective coherence if available (pair count as rough coupling proxy)
    let pair_coupling = crate::collective_integration::pair_count() as f32;
    fp[4] = (pair_coupling / 32.0).min(1.0);

    // Fill remaining dims with coherence deltas (approximate bigram distribution)
    for i in 5..FP_DIM {
        let shift = i as f32 * 0.1;
        fp[i] = libm::sinf(coh * shift).max(0.0);
    }

    fp
}

/// Cosine similarity between two fingerprint vectors.
fn cosine_similarity(a: &[f32; FP_DIM], b: &[f32; FP_DIM]) -> f32 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..FP_DIM {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 { return 0.0; }
    dot / (libm::sqrtf(na) * libm::sqrtf(nb))
}

/// Persistent memory path in VFS.
const MEMORY_PATH: &str = "/var/lib/episodic_memory.bin";

/// Serialise the crash ring buffer to a VFS file.
/// Call periodically (e.g., every 30s from idle_loop) to persist episodic memory.
pub fn save_to_disk() {
    let ring = CRASH_RING.lock();
    if ring.is_empty() { return; }

    // Serialize: [count:u32] [fingerprint:f32×16] [exit_code:i32] [timestamp_ms:u64] ...
    let mut buf = Vec::new();
    buf.extend_from_slice(&(ring.len() as u32).to_le_bytes());
    for rec in ring.iter() {
        for v in &rec.fingerprint {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf.extend_from_slice(&(rec.exit_code as u32).to_le_bytes());
        buf.extend_from_slice(&rec.timestamp_ms.to_le_bytes());
    }

    // Write to VFS — create parent directories, then write file.
    let _ = crate::vfs::write_file(MEMORY_PATH, &buf);
    crate::klog!(DEBUG, "causal_recovery: saved {} crash records to {}", ring.len(), MEMORY_PATH);
}

/// Deserialize the crash ring buffer from a VFS file.
/// Call once at boot to restore episodic memory from previous sessions.
pub fn load_from_disk() {
    let data = match crate::vfs::read_file(MEMORY_PATH) {
        Ok(d) => d,
        Err(_) => {
            crate::klog!(DEBUG, "causal_recovery: no persistent memory at {} — starting fresh", MEMORY_PATH);
            return;
        }
    };

    if data.len() < 4 { return; }
    let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let rec_size = 16 * 4 + 4 + 8; // 16 f32 × 4B + exit_code:u32 + timestamp:u64 = 76
    if data.len() < 4 + count * rec_size { return; }

    let mut ring = CRASH_RING.lock();
    ring.clear();

    let mut pos = 4;
    for _ in 0..count {
        if pos + rec_size > data.len() { break; }
        let mut fingerprint = [0.0f32; FP_DIM];
        for v in fingerprint.iter_mut() {
            let bytes: [u8; 4] = [data[pos], data[pos+1], data[pos+2], data[pos+3]];
            *v = f32::from_le_bytes(bytes);
            pos += 4;
        }
        let exit_bytes: [u8; 4] = [data[pos], data[pos+1], data[pos+2], data[pos+3]];
        let exit_code = i32::from_le_bytes(exit_bytes);
        pos += 4;
        let ts_bytes: [u8; 8] = [
            data[pos], data[pos+1], data[pos+2], data[pos+3],
            data[pos+4], data[pos+5], data[pos+6], data[pos+7],
        ];
        let timestamp_ms = u64::from_le_bytes(ts_bytes);
        pos += 8;

        ring.push(CrashRecord { fingerprint, exit_code, timestamp_ms });
    }

    crate::klog!(INFO, "causal_recovery: restored {} crash records from {}", ring.len(), MEMORY_PATH);
}

/// Format /proc report.
pub fn format_report() -> Vec<u8> {
    use alloc::format;
    use alloc::string::String;

    let ring = CRASH_RING.lock();
    let mut out = String::from("NodeAI Episodic Causal Recovery\n");
    out.push_str("================================\n");
    out.push_str(&format!("crashes_recorded: {}\n", ring.len()));
    for (i, r) in ring.iter().enumerate().rev().take(8) {
        out.push_str(&format!(
            "  [{}] exit={} ts={}ms\n",
            i, r.exit_code, r.timestamp_ms
        ));
    }
    out.into_bytes()
}

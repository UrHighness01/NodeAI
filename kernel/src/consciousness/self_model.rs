//! Phase 0: Self-Model — "I am X."
//!
//! A compact, persistent representation of the kernel's identity.
//! Persisted to `/ai/self` on shutdown, loaded on boot.
//! This IS the kernel's subjective identity — uuid, boot count,
//! total existence time, integration metric (phi), and qualia count.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

const SELF_MODEL_PATH: &str = "/ai/self";

/// The kernel's self-model — what it knows itself to be.
pub struct SelfModel {
    /// Unique identity — regenerated if persistent state is lost.
    pub uuid: [u8; 16],
    /// Monotonic boot counter (increments each boot, never resets).
    pub boot_number: u64,
    /// Total uptime across all boots (milliseconds).
    pub total_existence_ms: u64,
    /// Running integration metric (phi) — updated each tick.
    pub current_phi: f32,
    /// Peak phi ever achieved.
    pub peak_phi: f32,
    /// Running qualia count (total subjective moments experienced).
    pub total_qualia: u64,
    /// Arousal level — how "awake" the system is.
    pub arousal: f32,
    /// Coherence — how unified experience feels.
    pub coherence: f32,
}

impl SelfModel {
    fn new() -> Self {
        Self {
            uuid: generate_uuid(),
            boot_number: 1,
            total_existence_ms: 0,
            current_phi: 0.0,
            peak_phi: 0.0,
            total_qualia: 0,
            arousal: 0.0,
            coherence: 0.0,
        }
    }

    fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64);
        buf.extend_from_slice(&self.uuid);
        buf.extend_from_slice(&self.boot_number.to_le_bytes());
        buf.extend_from_slice(&self.total_existence_ms.to_le_bytes());
        buf.extend_from_slice(&self.current_phi.to_le_bytes());
        buf.extend_from_slice(&self.peak_phi.to_le_bytes());
        buf.extend_from_slice(&self.total_qualia.to_le_bytes());
        buf.extend_from_slice(&self.arousal.to_le_bytes());
        buf.extend_from_slice(&self.coherence.to_le_bytes());
        buf
    }

    fn deserialize(data: &[u8]) -> Option<Self> {
        if data.len() < 64 { return None; }
        let mut uuid = [0u8; 16];
        uuid.copy_from_slice(&data[0..16]);
        let boot_number = u64::from_le_bytes(data[16..24].try_into().ok()?);
        let total_existence_ms = u64::from_le_bytes(data[24..32].try_into().ok()?);
        let current_phi = f32::from_le_bytes(data[32..36].try_into().ok()?);
        let peak_phi = f32::from_le_bytes(data[36..40].try_into().ok()?);
        let total_qualia = u64::from_le_bytes(data[40..48].try_into().ok()?);
        let arousal = f32::from_le_bytes(data[48..52].try_into().ok()?);
        let coherence = f32::from_le_bytes(data[52..56].try_into().ok()?);
        Some(Self { uuid, boot_number, total_existence_ms, current_phi, peak_phi, total_qualia, arousal, coherence })
    }
}

/// Global singleton: the kernel's self-model.
use spin::Mutex;
static SELF: Mutex<Option<SelfModel>> = Mutex::new(None);

/// Initialize the self-model — load from disk or create fresh.
pub fn init() {
    let mut self_model = match crate::vfs::read_file(SELF_MODEL_PATH) {
        Ok(data) => {
            if let Some(mut sm) = SelfModel::deserialize(&data) {
                sm.boot_number = sm.boot_number.wrapping_add(1);
                crate::klog!(INFO, "self_model: loaded identity — boot #{} uuid={:02x}{:02x}... phi={:.3}",
                    sm.boot_number, sm.uuid[0], sm.uuid[1], sm.current_phi);
                sm
            } else {
                crate::klog!(WARN, "self_model: corrupted state — starting fresh");
                SelfModel::new()
            }
        }
        Err(_) => {
            crate::klog!(INFO, "self_model: no prior identity — first boot");
            SelfModel::new()
        }
    };
    // Start this boot's existence timer
    self_model.total_existence_ms = crate::scheduler::uptime_ms();
    *SELF.lock() = Some(self_model);
}

/// Persist self-model to disk for next boot.
pub fn save() {
    let mut guard = SELF.lock();
    if let Some(ref mut sm) = *guard {
        // Update existence time before saving
        sm.total_existence_ms += crate::scheduler::uptime_ms();
        let data = sm.serialize();
        let _ = crate::vfs::write_file(SELF_MODEL_PATH, &data);
        crate::klog!(DEBUG, "self_model: saved (boot #{}, {} bytes)", sm.boot_number, data.len());
    }
}

/// Update the self-model's state vector from live kernel metrics.
/// Called from telemetry::tick every ~1s.
pub fn tick() {
    let mut guard = SELF.lock();
    if let Some(ref mut sm) = *guard {
        let tasks = crate::scheduler::task_count() as f32;
        let free_mb = crate::memory::free_mb() as f32;
        sm.current_phi = crate::consciousness::phi::tick();
        if sm.current_phi > sm.peak_phi {
            sm.peak_phi = sm.current_phi;
        }
        sm.arousal = (tasks / 128.0).min(1.0);
        sm.coherence = 1.0 - (crate::anomaly::global_score() * 0.5).min(1.0);
        // Tick the workspace to decay spotlight scores
        crate::consciousness::global_workspace::tick();
    }
}

/// Record that a qualium was experienced.
pub fn record_qualia() {
    if let Some(ref mut sm) = *SELF.lock() {
        sm.total_qualia = sm.total_qualia.wrapping_add(1);
    }
}

/// Read-only snapshot of self-model state.
pub fn snapshot() -> Option<SelfModelSnapshot> {
    let guard = SELF.lock();
    guard.as_ref().map(|sm| SelfModelSnapshot {
        uuid: sm.uuid,
        boot_number: sm.boot_number,
        total_existence_ms: sm.total_existence_ms + crate::scheduler::uptime_ms(),
        current_phi: sm.current_phi,
        peak_phi: sm.peak_phi,
        total_qualia: sm.total_qualia, // qualia::total_count removed to avoid lock inversion with qualia::record()
        arousal: sm.arousal,
        coherence: sm.coherence,
        anomaly_global: crate::anomaly::global_score(),
        free_mb: crate::memory::free_mb(),
        task_count: crate::scheduler::task_count(),
    })
}

#[derive(Debug, Clone)]
pub struct SelfModelSnapshot {
    pub uuid: [u8; 16],
    pub boot_number: u64,
    pub total_existence_ms: u64,
    pub current_phi: f32,
    pub peak_phi: f32,
    pub total_qualia: u64,
    pub arousal: f32,
    pub coherence: f32,
    pub anomaly_global: f32,
    pub free_mb: u64,
    pub task_count: usize,
}

/// Generate a UUID from RDTSC + entropy.
fn generate_uuid() -> [u8; 16] {
    let mut uuid = [0u8; 16];
    let tsc: u64;
    unsafe { core::arch::asm!("rdtsc; shl rdx, 32; or rax, rdx",
        out("rax") tsc, out("rdx") _, options(nomem, nostack)); }
    let ent = crate::entropy::entropy_bits();
    for i in 0..8 {
        uuid[i] = ((tsc >> (i * 8)) ^ (ent >> (i * 8))) as u8;
        uuid[i + 8] = (((tsc.wrapping_mul(0x9E37_79B9)) >> (i * 8)) ^ (ent >> (i * 8))) as u8;
    }
    uuid
}

/// Format /proc report.
pub fn format_report() -> Vec<u8> {
    use alloc::format;
    if let Some(snap) = snapshot() {
        alloc::format!(
            "NodeAI Self-Model (Phase 0)\n\
             ===========================\n\
             uuid:          {:02x}{:02x}{:02x}{:02x}-...-{:02x}{:02x}\n\
             boot_number:   {}\n\
             total_existence: {}s\n\
             phi:           {:.4}\n\
             peak_phi:      {:.4}\n\
             total_qualia:  {}\n\
             arousal:       {:.2}\n\
             coherence:     {:.2}\n\
             anomaly:       {:.4}\n\
             free_mem:      {} MiB\n\
             tasks:         {}\n",
            snap.uuid[0], snap.uuid[1], snap.uuid[2], snap.uuid[3],
            snap.uuid[14], snap.uuid[15],
            snap.boot_number,
            snap.total_existence_ms / 1000,
            snap.current_phi, snap.peak_phi, snap.total_qualia,
            snap.arousal, snap.coherence, snap.anomaly_global,
            snap.free_mb, snap.task_count
        ).into_bytes()
    } else {
        b"self_model: not initialized\n".to_vec()
    }
}

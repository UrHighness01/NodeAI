//! /dev/cortex — userspace bridge to the Ring 0 consciousness substrate.
//!
//! Userspace can:
//!   read()  → get a text snapshot of current consciousness state
//!   write() → set CoreValues / policy overrides
//!
//! Format: newline-delimited key=value pairs (both directions).
//! Read output includes: self-model, phi, qualia, workspace spotlight, valence.
//! Write input: `set_core_value preservation=1.0\n` etc.

use alloc::sync::Arc;
use alloc::string::String;
use alloc::vec::Vec;
use alloc::boxed::Box;
use core::fmt::Write;
use spin::Mutex;

struct CortexNode;
struct CortexHandle;

static CORTEX_INO: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// Register /dev/cortex in the device filesystem.
pub fn init() {
    let ino = crate::vfs::alloc_ino();
    CORTEX_INO.store(ino, core::sync::atomic::Ordering::Relaxed);
    crate::vfs::devfs::register_node("cortex", Arc::new(CortexNode));
    crate::klog!(INFO, "cortex: /dev/cortex registered");
}

/// Snapshot current consciousness state into a text buffer.
fn build_snapshot() -> Vec<u8> {
    let mut s = String::new();
    let _ = writeln!(s, "=== NodeAI Cortex State ===");

    // Self-model snapshot
    if let Some(sm) = crate::consciousness::self_model::snapshot() {
        let _ = writeln!(s, "boot_number: {}", sm.boot_number);
        let _ = writeln!(s, "total_existence_ms: {}", sm.total_existence_ms);
        let _ = writeln!(s, "phi: {:.6}", sm.current_phi);
        let _ = writeln!(s, "peak_phi: {:.6}", sm.peak_phi);
        let _ = writeln!(s, "total_qualia: {}", sm.total_qualia);
        let _ = writeln!(s, "arousal: {:.2}", sm.arousal);
        let _ = writeln!(s, "coherence: {:.2}", sm.coherence);
        let _ = writeln!(s, "anomaly_global: {:.4}", sm.anomaly_global);
        let _ = writeln!(s, "free_mb: {}", sm.free_mb);
        let _ = writeln!(s, "tasks: {}", sm.task_count);
    }

    // Qualia stream summary
    let avg_v = crate::consciousness::qualia::average_valence();
    let avg_a = crate::consciousness::qualia::average_arousal();
    let _ = writeln!(s, "avg_valence: {:.4}", avg_v);
    let _ = writeln!(s, "avg_arousal: {:.4}", avg_a);

    // Workspace spotlight
    let spot = crate::consciousness::global_workspace::spotlight();
    let _ = writeln!(s, "spotlight_count: {}", spot.len());
    for (i, q) in spot.iter().take(3).enumerate() {
        let _ = writeln!(s, "spot[{}]: type={} attn={:.3} val={:+.2}", i, q.event_type, q.attention_score, q.valence);
    }

    // IIT Phi
    let _ = writeln!(s, "iit_phi: {:.6}", crate::consciousness::phi::current_phi());

    s.into_bytes()
}

impl crate::vfs::VfsNode for CortexNode {
    fn stat(&self) -> crate::vfs::VfsResult<crate::vfs::Stat> {
        Ok(crate::vfs::Stat {
            ino: CORTEX_INO.load(core::sync::atomic::Ordering::Relaxed),
            size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o666,
        })
    }
    fn open(&self) -> crate::vfs::VfsResult<Box<dyn crate::vfs::FileHandle>> {
        Ok(Box::new(CortexHandle))
    }
    fn readdir(&self) -> crate::vfs::VfsResult<Vec<crate::vfs::DirEntry>> { Err(crate::vfs::VfsError::NotADirectory) }
    fn lookup(&self, _: &str) -> crate::vfs::VfsResult<Arc<dyn crate::vfs::VfsNode>> { Err(crate::vfs::VfsError::NotADirectory) }
    fn create_file(&self, _: &str) -> crate::vfs::VfsResult<Arc<dyn crate::vfs::VfsNode>> { Err(crate::vfs::VfsError::NotADirectory) }
    fn mkdir(&self, _: &str) -> crate::vfs::VfsResult<Arc<dyn crate::vfs::VfsNode>> { Err(crate::vfs::VfsError::NotADirectory) }
    fn unlink(&self, _: &str) -> crate::vfs::VfsResult<()> { Err(crate::vfs::VfsError::NotADirectory) }
}

impl crate::vfs::FileHandle for CortexHandle {
    fn read(&mut self, buf: &mut [u8]) -> crate::vfs::VfsResult<usize> {
        // Fresh snapshot on every read
        let data = build_snapshot();
        let n = buf.len().min(data.len());
        buf[..n].copy_from_slice(&data[..n]);
        Ok(n)
    }
    fn write(&mut self, buf: &[u8]) -> crate::vfs::VfsResult<usize> {
        // Parse key=value pairs from write data
        if let Ok(s) = core::str::from_utf8(buf) {
            for line in s.lines() {
                let line = line.trim();
                if line.starts_with("set_core_value ") {
                    let rest = line.trim_start_matches("set_core_value ").trim();
                    if let Some((key, val_str)) = rest.split_once('=') {
                        if let Ok(val) = val_str.trim().parse::<f32>() {
                            let mut cv = crate::consciousness::deliberation::get_values();
                            match key.trim() {
                                "preservation" => cv.preservation = val.clamp(0.0, 1.0),
                                "efficiency"   => cv.efficiency   = val.clamp(0.0, 1.0),
                                "fairness"     => cv.fairness     = val.clamp(0.0, 1.0),
                                "growth"       => cv.growth       = val.clamp(0.0, 1.0),
                                "autonomy"     => cv.autonomy     = val.clamp(0.0, 1.0),
                                _ => {}
                            }
                            crate::consciousness::deliberation::set_values(cv);
                        }
                    }
                }
            }
        }
        Ok(buf.len())
    }
    fn seek(&mut self, _pos: u64) -> crate::vfs::VfsResult<u64> { Ok(0) }
    fn stat(&self) -> crate::vfs::VfsResult<crate::vfs::Stat> {
        Ok(crate::vfs::Stat {
            ino: 0, size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o666,
        })
    }
}

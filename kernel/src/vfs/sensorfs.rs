//! sensorfs — /dev/sensor/ synthetic VFS directory (Phase EW-0b).
//!
//! Mounts `/dev/sensor/` as a browsable directory where each registered sensor
//! appears as its own node. Reading a sensor node returns its latest reading.
//! Writing to a sensor node can configure it.
//!
//! Structure:
//!   /dev/sensor/              — directory (ls to list sensors)
//!   /dev/sensor/ambient_rf_0  — 2.4 GHz ambient sensor
//!   /dev/sensor/ambient_rf_1  — 5 GHz ambient sensor
//!   /dev/sensor/stats         — aggregate sensor statistics

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};
use super::{alloc_ino, DirEntry, FileHandle, Stat, VfsError, VfsNode, VfsResult};

/// Inode base for sensor nodes.
static SENSORFS_INO: AtomicU64 = AtomicU64::new(0);

/// Initialize sensorfs — mounts /dev/sensor as a directory.
pub fn init() {
    let ino = alloc_ino();
    SENSORFS_INO.store(ino, Ordering::Relaxed);

    if let Ok(parent) = crate::vfs::lookup("/dev") {
        // Mount our SensorDir over /dev/sensor
        crate::vfs::mount("/dev/sensor", Arc::new(SensorDir));
    }
    crate::klog!(INFO, "sensorfs: /dev/sensor/ directory registered");
}

/// The /dev/sensor/ directory node.
pub struct SensorDir;

impl VfsNode for SensorDir {
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat {
            ino: SENSORFS_INO.load(Ordering::Relaxed),
            size: 0, is_dir: true, nlink: 1, uid: 0, gid: 0, mode: 0o755,
        })
    }

    fn open(&self) -> VfsResult<Box<dyn FileHandle>> {
        Err(VfsError::NotADirectory)
    }

    fn readdir(&self) -> VfsResult<Vec<DirEntry>> {
        let mut entries = Vec::new();
        let stats = crate::sensor_cortex::stats();
        let base_ino = SENSORFS_INO.load(Ordering::Relaxed);

        entries.push(DirEntry { ino: base_ino + 1, name: String::from("stats"), is_dir: false });

        for i in 0..stats.num_sensors {
            entries.push(DirEntry {
                ino: base_ino + 2 + i as u64,
                name: alloc::format!("ambient_rf_{}", i),
                is_dir: false,
            });
        }
        Ok(entries)
    }

    fn lookup(&self, name: &str) -> VfsResult<Arc<dyn VfsNode>> {
        match name {
            "stats" => Ok(Arc::new(SensorStatsNode)),
            name if name.starts_with("ambient_rf_") => {
                Ok(Arc::new(SensorDataNode { label: String::from(name) }))
            }
            _ => Err(VfsError::NotFound),
        }
    }

    fn create_file(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn mkdir(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn unlink(&self, _: &str) -> VfsResult<()> { Err(VfsError::NotADirectory) }
}

/// /dev/sensor/stats — aggregate sensor statistics.
struct SensorStatsNode;

impl VfsNode for SensorStatsNode {
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: SENSORFS_INO.load(Ordering::Relaxed) + 1, size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o444 })
    }
    fn open(&self) -> VfsResult<Box<dyn FileHandle>> { Ok(Box::new(SensorStatsHandle)) }
    fn readdir(&self) -> VfsResult<Vec<DirEntry>> { Err(VfsError::NotADirectory) }
    fn lookup(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn create_file(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn mkdir(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn unlink(&self, _: &str) -> VfsResult<()> { Err(VfsError::NotADirectory) }
}

struct SensorStatsHandle;

impl FileHandle for SensorStatsHandle {
    fn read(&mut self, buf: &mut [u8]) -> VfsResult<usize> {
        let stats = crate::sensor_cortex::stats();
        let s = stats.fmt_report();
        let bytes = s.as_bytes();
        let n = bytes.len().min(buf.len());
        buf[..n].copy_from_slice(&bytes[..n]);
        Ok(n)
    }
    fn write(&mut self, _: &[u8]) -> VfsResult<usize> { Ok(0) }
    fn seek(&mut self, pos: u64) -> VfsResult<u64> { Ok(pos) }
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: SENSORFS_INO.load(Ordering::Relaxed) + 1, size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o444 })
    }
}

/// /dev/sensor/ambient_rf_N — per-sensor data node.
struct SensorDataNode {
    label: String,
}

impl VfsNode for SensorDataNode {
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: SENSORFS_INO.load(Ordering::Relaxed) + 2, size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o444 })
    }
    fn open(&self) -> VfsResult<Box<dyn FileHandle>> { Ok(Box::new(SensorDataHandle { label: self.label.clone() })) }
    fn readdir(&self) -> VfsResult<Vec<DirEntry>> { Err(VfsError::NotADirectory) }
    fn lookup(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn create_file(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn mkdir(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn unlink(&self, _: &str) -> VfsResult<()> { Err(VfsError::NotADirectory) }
}

struct SensorDataHandle {
    label: String,
}

impl FileHandle for SensorDataHandle {
    fn read(&mut self, buf: &mut [u8]) -> VfsResult<usize> {
        // Return sensor status data from the cortex
        let stats = crate::sensor_cortex::stats();
        let s = alloc::format!(
            "sensor:     {}\n\
             signals:    {}\n\
             jams:       {}\n\
             spectrum:   {} samples\n\
             timestamp:  {}s\n",
            self.label,
            stats.signals_detected,
            stats.jams_detected,
            stats.last_spectrum_count,
            crate::scheduler::uptime_ms() / 1000,
        );
        let bytes = s.as_bytes();
        let n = bytes.len().min(buf.len());
        buf[..n].copy_from_slice(&bytes[..n]);
        Ok(n)
    }
    fn write(&mut self, _: &[u8]) -> VfsResult<usize> { Ok(0) }
    fn seek(&mut self, pos: u64) -> VfsResult<u64> { Ok(pos) }
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: SENSORFS_INO.load(Ordering::Relaxed) + 2, size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o444 })
    }
}

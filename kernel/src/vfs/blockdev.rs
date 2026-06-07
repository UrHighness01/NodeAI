//! Block device VFS nodes — expose AHCI/NVMe drives as /dev/sdX and /dev/nvmeX.
//!
//! Each node supports raw byte-level read/write; userspace tools (mkfs, mount)
//! can build higher-level filesystems on top.

use alloc::{boxed::Box, sync::Arc, vec::Vec};
use spin::Mutex;
use super::{alloc_ino, FileHandle, Stat, VfsError, VfsNode, VfsResult};

// ── Block size ────────────────────────────────────────────────────────────────
const SECTOR_SIZE: u64 = 512;

// ── AHCI block device ─────────────────────────────────────────────────────────

pub struct AhciNode {
    ino:      u64,
    drive:    usize,
    size_bytes: u64,
}

impl AhciNode {
    pub fn new(drive_idx: usize, sector_count: u64) -> Arc<Self> {
        Arc::new(Self {
            ino:        alloc_ino(),
            drive:      drive_idx,
            size_bytes: sector_count * SECTOR_SIZE,
        })
    }
}

impl VfsNode for AhciNode {
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: self.ino, size: self.size_bytes, is_dir: false,
                  nlink: 1, uid: 0, gid: 0, mode: 0o660 })
    }
    fn open(&self) -> VfsResult<Box<dyn FileHandle>> {
        Ok(Box::new(AhciHandle { drive: self.drive, offset: Mutex::new(0), size: self.size_bytes }))
    }
    fn lookup(&self, _name: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotFound) }
    fn create_file(&self, _name: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::ReadOnly) }
    fn mkdir(&self, _name: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::ReadOnly) }
    fn unlink(&self, _name: &str) -> VfsResult<()> { Err(VfsError::ReadOnly) }
    fn readdir(&self) -> VfsResult<Vec<super::DirEntry>> { Err(VfsError::NotFound) }
}

struct AhciHandle {
    drive:  usize,
    offset: Mutex<u64>,
    size:   u64,
}

impl FileHandle for AhciHandle {
    fn read(&mut self, buf: &mut [u8]) -> VfsResult<usize> {
        let mut off = self.offset.lock();
        if *off >= self.size { return Ok(0); }
        let to_read = buf.len().min((self.size - *off) as usize);
        let lba   = *off / SECTOR_SIZE;
        let start = (*off % SECTOR_SIZE) as usize;
        let sectors = ((start + to_read + SECTOR_SIZE as usize - 1) / SECTOR_SIZE as usize) as u16;
        match crate::ahci::read_sectors(self.drive, lba, sectors) {
            Some(data) => {
                let end = (start + to_read).min(data.len());
                let n = end - start;
                buf[..n].copy_from_slice(&data[start..end]);
                *off += n as u64;
                Ok(n)
            }
            None => Err(VfsError::IoError),
        }
    }
    fn write(&mut self, buf: &[u8]) -> VfsResult<usize> {
        let mut off = self.offset.lock();
        if *off >= self.size { return Err(VfsError::IoError); }
        let lba     = *off / SECTOR_SIZE;
        let aligned = (buf.len() + SECTOR_SIZE as usize - 1) & !(SECTOR_SIZE as usize - 1);
        let mut sector_buf = alloc::vec![0u8; aligned];
        sector_buf[..buf.len()].copy_from_slice(buf);
        let sectors = (aligned / SECTOR_SIZE as usize) as u16;
        if crate::ahci::write_sectors(self.drive, lba, &sector_buf) {
            *off += buf.len() as u64;
            Ok(buf.len())
        } else {
            Err(VfsError::IoError)
        }
    }
    fn seek(&mut self, pos: u64) -> VfsResult<u64> {
        let mut off = self.offset.lock();
        *off = pos.min(self.size);
        Ok(*off)
    }
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: 0, size: self.size, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o660 })
    }
}

// ── NVMe block device ─────────────────────────────────────────────────────────

pub struct NvmeNode {
    ino:        u64,
    ctrl:       usize,
    size_bytes: u64,
}

impl NvmeNode {
    pub fn new(ctrl_idx: usize, lba_count: u64) -> Arc<Self> {
        Arc::new(Self {
            ino:        alloc_ino(),
            ctrl:       ctrl_idx,
            size_bytes: lba_count * 512,
        })
    }
}

impl VfsNode for NvmeNode {
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: self.ino, size: self.size_bytes, is_dir: false,
                  nlink: 1, uid: 0, gid: 0, mode: 0o660 })
    }
    fn open(&self) -> VfsResult<Box<dyn FileHandle>> {
        Ok(Box::new(NvmeHandle { ctrl: self.ctrl, offset: Mutex::new(0), size: self.size_bytes }))
    }
    fn lookup(&self, _name: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotFound) }
    fn create_file(&self, _name: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::ReadOnly) }
    fn mkdir(&self, _name: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::ReadOnly) }
    fn unlink(&self, _name: &str) -> VfsResult<()> { Err(VfsError::ReadOnly) }
    fn readdir(&self) -> VfsResult<Vec<super::DirEntry>> { Err(VfsError::NotFound) }
}

struct NvmeHandle {
    ctrl:   usize,
    offset: Mutex<u64>,
    size:   u64,
}

impl FileHandle for NvmeHandle {
    fn read(&mut self, buf: &mut [u8]) -> VfsResult<usize> {
        let mut off = self.offset.lock();
        if *off >= self.size { return Ok(0); }
        let to_read = buf.len().min((self.size - *off) as usize);
        let lba     = *off / 512;
        let sectors = ((to_read + 511) / 512) as u16;
        match crate::nvme::read_blocks(self.ctrl, lba, sectors) {
            Some(data) => {
                let start = (*off % 512) as usize;
                let n     = (data.len() - start).min(to_read);
                buf[..n].copy_from_slice(&data[start..start + n]);
                *off += n as u64;
                Ok(n)
            }
            None => Err(VfsError::IoError),
        }
    }
    fn write(&mut self, buf: &[u8]) -> VfsResult<usize> {
        let mut off = self.offset.lock();
        let lba     = *off / 512;
        let aligned = (buf.len() + 511) & !511;
        let mut sector_buf = alloc::vec![0u8; aligned];
        sector_buf[..buf.len()].copy_from_slice(buf);
        let sectors = (aligned / 512) as u16;
        if crate::nvme::write_blocks(self.ctrl, lba, &sector_buf) {
            *off += buf.len() as u64;
            Ok(buf.len())
        } else {
            Err(VfsError::IoError)
        }
    }
    fn seek(&mut self, pos: u64) -> VfsResult<u64> {
        let mut off = self.offset.lock();
        *off = pos.min(self.size);
        Ok(*off)
    }
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: 0, size: self.size, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o660 })
    }
}

/// Register all detected AHCI and NVMe drives in /dev.
/// Called from vfs::init() after devfs is mounted.
pub fn register_block_devices(_dev_root: &alloc::sync::Arc<dyn VfsNode>) {
    // AHCI drives → /dev/sda, /dev/sdb, ...
    // Default size: 64 GiB. A proper ATA IDENTIFY would give the real sector count.
    let ahci_count = crate::ahci::drive_count();
    for i in 0..ahci_count.min(26) {
        let node: Arc<dyn VfsNode> = AhciNode::new(i, 64 * 1024 * 1024 * 1024 / SECTOR_SIZE);
        let name = alloc::format!("sd{}", (b'a' + i as u8) as char);
        crate::vfs::devfs::register_node(&name, node);
        crate::klog!(INFO, "VFS: /dev/{} → AHCI drive {}", name, i);
    }

    // NVMe controllers → /dev/nvme0, /dev/nvme1, ...
    let nvme_count = crate::nvme::ctrl_count();
    for i in 0..nvme_count {
        let node: Arc<dyn VfsNode> = NvmeNode::new(i, 64 * 1024 * 1024 * 1024 / 512);
        let name = alloc::format!("nvme{}", i);
        crate::vfs::devfs::register_node(&name, node);
        crate::klog!(INFO, "VFS: /dev/{} → NVMe ctrl {}", name, i);
    }
}

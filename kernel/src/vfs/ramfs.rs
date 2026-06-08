//! ramfs — in-memory filesystem backed by BTreeMap — Phase 7.
//!
//! Suitable for the initial root filesystem, /tmp, and /proc.

use alloc::{
    boxed::Box, collections::BTreeMap, string::String, sync::Arc, vec::Vec,
};
use spin::Mutex;
use super::{
    alloc_ino, DirEntry, FileHandle, Stat, VfsError, VfsNode, VfsResult,
};
use core::sync::atomic::{AtomicU16, AtomicU32, Ordering};

// ── SymlinkNode ───────────────────────────────────────────────────────────────

/// An in-memory symbolic link node.  Stores the target path as a String.
pub struct SymlinkNode {
    ino:    u64,
    target: String,
}

impl SymlinkNode {
    pub fn new(target: &str) -> Arc<Self> {
        Arc::new(Self { ino: alloc_ino(), target: String::from(target) })
    }
}

impl VfsNode for SymlinkNode {
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat {
            ino: self.ino, size: self.target.len() as u64,
            is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o120777,
        })
    }
    fn open(&self) -> VfsResult<Box<dyn FileHandle>> { Err(VfsError::InvalidArgument) }
    fn readdir(&self) -> VfsResult<Vec<DirEntry>>    { Err(VfsError::NotADirectory) }
    fn lookup(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn create_file(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn mkdir(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn unlink(&self, _: &str) -> VfsResult<()> { Err(VfsError::NotADirectory) }
    fn readlink(&self) -> Option<String> { Some(self.target.clone()) }
}

// ── RamFile ───────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct RamFile {
    ino:  u64,
    data: Arc<Mutex<Vec<u8>>>,
    uid:  AtomicU32,
    gid:  AtomicU32,
    mode: AtomicU16,
}

impl RamFile {
    pub fn new() -> Arc<Self> {
        let cur_uid = crate::users::current_uid();
        let cur_gid = crate::users::get_user(cur_uid).map(|u| u.gid).unwrap_or(0);
        let umask = crate::users::umask();
        Arc::new(Self {
            ino: alloc_ino(),
            data: Arc::new(Mutex::new(Vec::new())),
            uid: AtomicU32::new(cur_uid),
            gid: AtomicU32::new(cur_gid),
            mode: AtomicU16::new(0o644 & !umask),
        })
    }

    pub fn new_with(content: Vec<u8>) -> Arc<Self> {
        let cur_uid = crate::users::current_uid();
        let cur_gid = crate::users::get_user(cur_uid).map(|u| u.gid).unwrap_or(0);
        let umask = crate::users::umask();
        Arc::new(Self {
            ino: alloc_ino(),
            data: Arc::new(Mutex::new(content)),
            uid: AtomicU32::new(cur_uid),
            gid: AtomicU32::new(cur_gid),
            mode: AtomicU16::new(0o644 & !umask),
        })
    }
}

impl VfsNode for RamFile {
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat {
            ino:    self.ino,
            size:   self.data.lock().len() as u64,
            is_dir: false,
            nlink:  1,
            uid:    self.uid.load(Ordering::Relaxed),
            gid:    self.gid.load(Ordering::Relaxed),
            mode:   self.mode.load(Ordering::Relaxed),
        })
    }

    fn open(&self) -> VfsResult<Box<dyn FileHandle>> {
        Ok(Box::new(RamFileHandle {
            data: self.data.lock().clone(),
            pos:  0,
            ino:  self.ino,
            back: Arc::clone(&self.data),
            uid:  self.uid.load(Ordering::Relaxed),
            gid:  self.gid.load(Ordering::Relaxed),
            mode: self.mode.load(Ordering::Relaxed),
        }))
    }

    fn readdir(&self) -> VfsResult<Vec<DirEntry>> { Err(VfsError::NotADirectory) }
    fn lookup(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn create_file(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn mkdir(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn unlink(&self, _: &str) -> VfsResult<()> { Err(VfsError::NotADirectory) }

    fn set_mode(&self, mode: u16) -> VfsResult<()> {
        self.mode.store(mode, Ordering::Relaxed);
        Ok(())
    }
    fn set_owner(&self, uid: u32, gid: u32) -> VfsResult<()> {
        self.uid.store(uid, Ordering::Relaxed);
        self.gid.store(gid, Ordering::Relaxed);
        Ok(())
    }
}

struct RamFileHandle {
    data: Vec<u8>,
    pos:  usize,
    ino:  u64,
    /// Shared data backing, for write-back on flush.
    back: Arc<Mutex<Vec<u8>>>,
    uid: u32,
    gid: u32,
    mode: u16,
}

// We hold a Mutex clone as back, which contains a raw Vec — this is Send.
// The handle itself is used exclusively on one CPU thread at a time.
unsafe impl Send for RamFileHandle {}
unsafe impl Sync for RamFileHandle {}

impl FileHandle for RamFileHandle {
    fn bytes_available(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read(&mut self, buf: &mut [u8]) -> VfsResult<usize> {
        let avail = self.data.len().saturating_sub(self.pos);
        let n = buf.len().min(avail);
        buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }

    fn write(&mut self, buf: &[u8]) -> VfsResult<usize> {
        let end = self.pos + buf.len();
        if end > self.data.len() { self.data.resize(end, 0); }
        self.data[self.pos..end].copy_from_slice(buf);
        self.pos = end;
        Ok(buf.len())
    }

    fn seek(&mut self, pos: u64) -> VfsResult<u64> {
        self.pos = pos as usize;
        Ok(pos)
    }

    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: self.ino, size: self.data.len() as u64, is_dir: false, nlink: 1,
                  uid: self.uid, gid: self.gid, mode: self.mode })
    }

    fn flush(&mut self) -> VfsResult<()> {
        *self.back.lock() = self.data.clone();
        Ok(())
    }

    fn truncate(&mut self, len: u64) -> VfsResult<()> {
        self.data.truncate(len as usize);
        if self.pos > self.data.len() { self.pos = self.data.len(); }
        Ok(())
    }
}

// ── RamDir ────────────────────────────────────────────────────────────────────

pub struct RamDir {
    ino:      u64,
    children: Mutex<BTreeMap<String, Arc<dyn VfsNode>>>,
    uid:  AtomicU32,
    gid:  AtomicU32,
    mode: AtomicU16,
}

impl RamDir {
    /// Create a fresh root directory (owned by root).
    pub fn new_root() -> Self {
        Self {
            ino: alloc_ino(),
            children: Mutex::new(BTreeMap::new()),
            uid: AtomicU32::new(0),
            gid: AtomicU32::new(0),
            mode: AtomicU16::new(0o755),
        }
    }

    pub fn new() -> Arc<Self> {
        let cur_uid = crate::users::current_uid();
        let cur_gid = crate::users::get_user(cur_uid).map(|u| u.gid).unwrap_or(0);
        let umask = crate::users::umask();
        Arc::new(Self {
            ino: alloc_ino(),
            children: Mutex::new(BTreeMap::new()),
            uid: AtomicU32::new(cur_uid),
            gid: AtomicU32::new(cur_gid),
            mode: AtomicU16::new(0o755 & !umask),
        })
    }
}

impl VfsNode for RamDir {
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat {
            ino: self.ino, size: 0, is_dir: true, nlink: 2,
            uid: self.uid.load(Ordering::Relaxed),
            gid: self.gid.load(Ordering::Relaxed),
            mode: self.mode.load(Ordering::Relaxed),
        })
    }

    fn open(&self) -> VfsResult<Box<dyn FileHandle>> {
        Err(VfsError::NotAFile)
    }

    fn readdir(&self) -> VfsResult<Vec<DirEntry>> {
        let ch = self.children.lock();
        Ok(ch.iter().map(|(name, node)| {
            let is_dir = node.stat().map(|s| s.is_dir).unwrap_or(false);
            let ino    = node.stat().map(|s| s.ino).unwrap_or(0);
            DirEntry { name: name.clone(), is_dir, ino }
        }).collect())
    }

    fn lookup(&self, name: &str) -> VfsResult<Arc<dyn VfsNode>> {
        self.children.lock().get(name).cloned().ok_or(VfsError::NotFound)
    }

    fn create_file(&self, name: &str) -> VfsResult<Arc<dyn VfsNode>> {
        let mut ch = self.children.lock();
        if ch.contains_key(name) { return Err(VfsError::Exists); }
        let f = RamFile::new() as Arc<dyn VfsNode>;
        ch.insert(String::from(name), f.clone());
        Ok(f)
    }

    fn mkdir(&self, name: &str) -> VfsResult<Arc<dyn VfsNode>> {
        let mut ch = self.children.lock();
        if ch.contains_key(name) { return Err(VfsError::Exists); }
        let d = RamDir::new() as Arc<dyn VfsNode>;
        ch.insert(String::from(name), d.clone());
        Ok(d)
    }

    fn unlink(&self, name: &str) -> VfsResult<()> {
        let mut ch = self.children.lock();
        ch.remove(name).map(|_| ()).ok_or(VfsError::NotFound)
    }

    fn set_mode(&self, mode: u16) -> VfsResult<()> {
        self.mode.store(mode, Ordering::Relaxed);
        Ok(())
    }
    fn set_owner(&self, uid: u32, gid: u32) -> VfsResult<()> {
        self.uid.store(uid, Ordering::Relaxed);
        self.gid.store(gid, Ordering::Relaxed);
        Ok(())
    }

    fn create_symlink(&self, name: &str, target: &str) -> VfsResult<Arc<dyn VfsNode>> {
        let mut ch = self.children.lock();
        if ch.contains_key(name) { return Err(VfsError::Exists); }
        let sym = SymlinkNode::new(target) as Arc<dyn VfsNode>;
        ch.insert(String::from(name), sym.clone());
        Ok(sym)
    }
}

impl RamDir {
    /// Insert an existing node under a new name (hard link — shares the same Arc).
    pub fn link_child(&self, name: &str, node: Arc<dyn VfsNode>) -> VfsResult<()> {
        let mut ch = self.children.lock();
        if ch.contains_key(name) { return Err(VfsError::Exists); }
        ch.insert(String::from(name), node);
        Ok(())
    }
}

//! /dev/llm — userspace LLM inference bridge (Phase CI-5).
//!
//! Provides a simple char device for communication between the kernel and
//! a userspace LLM inference daemon. The daemon reads queries from /dev/llm,
//! processes them (using Project-M/K weights), and writes responses back.
//!
//! This avoids kernel heap fragmentation and LLVM aliasing issues by
//! running neural inference entirely in userspace.

use alloc::sync::Arc;
use alloc::string::String;
use alloc::vec::Vec;
use alloc::boxed::Box;
use alloc::format;
use spin::Mutex;
use core::sync::atomic::{AtomicU64, Ordering};
use crate::vfs::{VfsNode, VfsResult, VfsError, FileHandle};

/// Pending query from user to daemon.
static PENDING_QUERY: Mutex<Option<String>> = Mutex::new(None);
/// Completed response from daemon to user.
static PENDING_RESPONSE: Mutex<Option<String>> = Mutex::new(None);
/// Whether a daemon is connected.
static DAEMON_CONNECTED: Mutex<bool> = Mutex::new(false);
/// Inode for /dev/llm
static LLM_INO: AtomicU64 = AtomicU64::new(0);

struct LlmNode;
struct LlmHandle;

/// Register /dev/llm in the device filesystem.
pub fn init() {
    let ino = crate::vfs::alloc_ino();
    LLM_INO.store(ino, Ordering::Relaxed);
    crate::vfs::devfs::register_node("llm", Arc::new(LlmNode));
    crate::klog!(INFO, "llm_bridge: /dev/llm registered — userspace LLM daemon interface");
}

/// Enqueue a query for the LLM daemon.
pub fn enqueue_query(query: &str) -> bool {
    let mut q = PENDING_QUERY.lock();
    if q.is_some() { return false; }
    *q = Some(String::from(query));
    *PENDING_RESPONSE.lock() = None;
    true
}

pub fn has_response() -> bool { PENDING_RESPONSE.lock().is_some() }
pub fn take_response() -> Option<String> { PENDING_RESPONSE.lock().take() }
pub fn is_daemon_connected() -> bool { *DAEMON_CONNECTED.lock() }

impl VfsNode for LlmNode {
    fn stat(&self) -> VfsResult<crate::vfs::Stat> {
        Ok(crate::vfs::Stat { ino: LLM_INO.load(Ordering::Relaxed), size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o666 })
    }
    fn open(&self) -> VfsResult<Box<dyn FileHandle>> { Ok(Box::new(LlmHandle)) }
    fn readdir(&self) -> VfsResult<Vec<crate::vfs::DirEntry>> { Err(VfsError::NotADirectory) }
    fn lookup(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn create_file(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn mkdir(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn unlink(&self, _: &str) -> VfsResult<()> { Err(VfsError::NotADirectory) }
}

impl FileHandle for LlmHandle {
    fn read(&mut self, buf: &mut [u8]) -> VfsResult<usize> {
        if *DAEMON_CONNECTED.lock() {
            if let Some(query) = PENDING_QUERY.lock().take() {
                let bytes = query.as_bytes();
                let n = bytes.len().min(buf.len());
                buf[..n].copy_from_slice(&bytes[..n]);
                return Ok(n);
            }
        }
        if let Some(resp) = PENDING_RESPONSE.lock().take() {
            let bytes = resp.as_bytes();
            let n = bytes.len().min(buf.len());
            buf[..n].copy_from_slice(&bytes[..n]);
            return Ok(n);
        }
        Ok(0)
    }

    fn write(&mut self, buf: &[u8]) -> VfsResult<usize> {
        let text = core::str::from_utf8(buf).unwrap_or("");
        let trimmed = text.trim();
        if trimmed == "--daemon-connect" { *DAEMON_CONNECTED.lock() = true; return Ok(buf.len()); }
        if trimmed == "--daemon-disconnect" { *DAEMON_CONNECTED.lock() = false; return Ok(buf.len()); }
        if trimmed == "--poll" || trimmed.is_empty() { return Ok(buf.len()); }
        if *DAEMON_CONNECTED.lock() {
            *PENDING_RESPONSE.lock() = Some(String::from(trimmed));
            return Ok(buf.len());
        }
        if PENDING_QUERY.lock().is_some() { return Ok(0); }
        *PENDING_QUERY.lock() = Some(String::from(trimmed));
        Ok(buf.len())
    }

    fn seek(&mut self, _pos: u64) -> VfsResult<u64> { Ok(0) }
    fn stat(&self) -> VfsResult<crate::vfs::Stat> {
        Ok(crate::vfs::Stat { ino: LLM_INO.load(Ordering::Relaxed), size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o666 })
    }
}

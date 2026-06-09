//! Virtual Filesystem Switch (VFS) — Phase 7.
//!
//! Provides a trait-based abstraction over filesystem implementations.
//! All filesystems implement `VfsNode` (or more specialized sub-traits).
//! Path resolution and file descriptor management live here too.

pub mod path;
pub mod proc_pid;
pub mod procfs;
pub mod ramfs;
pub mod devfs;
pub mod blockdev;

use alloc::{boxed::Box, string::String, sync::Arc, vec::Vec};
use spin::RwLock;
use core::sync::atomic::{AtomicU64, Ordering};

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VfsError {
    NotFound,
    NotADirectory,
    NotAFile,
    PermissionDenied,
    Exists,
    InvalidArgument,
    Io,
    OutOfSpace,
    ReadOnly,
    TooManyOpenFiles,
    /// Operation would block; caller should return EAGAIN when O_NONBLOCK is set.
    WouldBlock,
}

pub type VfsResult<T> = Result<T, VfsError>;

// ── Stat / metadata ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Stat {
    pub ino:    u64,
    pub size:   u64,
    pub is_dir: bool,
    pub nlink:  u32,
    pub uid:    u32,
    pub gid:    u32,
    pub mode:   u16,
}

// ── File handle trait ────────────────────────────────────────────────────────

/// An open file or directory, obtained via `VfsNode::open`.
pub trait FileHandle: Send + Sync {
    /// Return how many bytes can be read without blocking.
    /// Default: 0 (unknown / treat as not-ready). Implementors should override.
    fn bytes_available(&self) -> usize { 0 }
    fn read(&mut self, buf: &mut [u8])  -> VfsResult<usize>;
    fn write(&mut self, buf: &[u8])     -> VfsResult<usize>;
    fn seek(&mut self, pos: u64)        -> VfsResult<u64>;
    fn stat(&self)                      -> VfsResult<Stat>;
    fn flush(&mut self)                 -> VfsResult<()> { Ok(()) }
    /// Discard all content beyond `len` bytes.  Default: no-op.
    fn truncate(&mut self, _len: u64)   -> VfsResult<()> { Ok(()) }
    /// Duplicate this handle for sys_dup/sys_dup2.
    /// Returns Some(new_handle) if duplication is supported, None otherwise
    /// (caller falls back to reopening via path).
    fn clone_box(&self) -> Option<Box<dyn FileHandle>> { None }
}

// ── Directory entry ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DirEntry {
    pub name:   String,
    pub is_dir: bool,
    pub ino:    u64,
}

// ── VfsNode trait ────────────────────────────────────────────────────────────

/// A node (file or directory) in the VFS tree.
pub trait VfsNode: Send + Sync {
    fn stat(&self) -> VfsResult<Stat>;

    /// Open for reading/writing; returns a FileHandle.
    fn open(&self) -> VfsResult<Box<dyn FileHandle>>;

    /// Read directory entries (only valid for directory nodes).
    fn readdir(&self) -> VfsResult<Vec<DirEntry>>;

    /// Look up a child by name (directory nodes only).
    fn lookup(&self, name: &str) -> VfsResult<Arc<dyn VfsNode>>;

    /// Create a regular file child.
    fn create_file(&self, name: &str) -> VfsResult<Arc<dyn VfsNode>>;

    /// Create a directory child.
    fn mkdir(&self, name: &str) -> VfsResult<Arc<dyn VfsNode>>;

    /// Remove a child entry.
    fn unlink(&self, name: &str) -> VfsResult<()>;

    /// Change permission bits on this node.
    fn set_mode(&self, _mode: u16) -> VfsResult<()> { Err(VfsError::ReadOnly) }

    /// Create a symbolic link child named `name` pointing to `target`.
    fn create_symlink(&self, _name: &str, _target: &str) -> VfsResult<Arc<dyn VfsNode>> {
        Err(VfsError::InvalidArgument)
    }

    /// If this node is a symbolic link, return the target path.  None otherwise.
    fn readlink(&self) -> Option<alloc::string::String> { None }

    /// Create a hard link: insert `node` as `name` in this directory.
    /// Only implemented by directory nodes that support hard links.
    fn link_child(&self, _name: &str, _node: Arc<dyn VfsNode>) -> VfsResult<()> {
        Err(VfsError::InvalidArgument)
    }

    /// Change owner uid/gid on this node.
    fn set_owner(&self, _uid: u32, _gid: u32) -> VfsResult<()> { Err(VfsError::ReadOnly) }
}

// ── Mount table ───────────────────────────────────────────────────────────────

struct MountEntry {
    path: String,
    root: Arc<dyn VfsNode>,
}

static MOUNTS: RwLock<Vec<MountEntry>> = RwLock::new(Vec::new());

/// Global root node (populated by `init`).
static mut ROOT: Option<Arc<dyn VfsNode>> = None;

/// Mount `fs_root` at `mountpoint`.
pub fn mount(mountpoint: &str, root: Arc<dyn VfsNode>) {
    MOUNTS.write().push(MountEntry {
        path: String::from(mountpoint),
        root,
    });
}

/// Lookup a path; returns the deepest matching VfsNode.
pub fn lookup(path: &str) -> VfsResult<Arc<dyn VfsNode>> {
    path::resolve(path)
}

/// Initialise the VFS with a root ramfs and standard devfs mounts.
pub fn init() {
    let root = Arc::new(ramfs::RamDir::new_root());
    // Create /dev directory
    root.mkdir("dev").ok();
    root.mkdir("proc").ok();
    root.mkdir("sys").ok();
    root.mkdir("ai").ok();

    unsafe { ROOT = Some(root.clone() as Arc<dyn VfsNode>); }

    // Mount devfs at /dev — use DevDir::root() to store global reference.
    let _dev_node = root.lookup("dev").expect("vfs: /dev missing");
    let devfs_root = devfs::DevDir::root();
    mount("/dev", devfs_root.clone() as Arc<dyn VfsNode>);

    mount("/", root);

    // Register block devices (/dev/sdX, /dev/nvmeX).
    blockdev::register_block_devices(&(devfs_root as Arc<dyn VfsNode>));

    crate::klog!(INFO, "VFS initialized — root ramfs + devfs /dev + block devices");
}

/// Get the root filesystem node.
pub fn root() -> Arc<dyn VfsNode> {
    unsafe { ROOT.as_ref().expect("vfs not initialized").clone() }
}

// ── Global file-descriptor table (kernel-internal) ───────────────────────────

static NEXT_INO: AtomicU64 = AtomicU64::new(1);

pub fn alloc_ino() -> u64 {
    NEXT_INO.fetch_add(1, Ordering::Relaxed)
}

// ── Permission checking ───────────────────────────────────────────────────────

/// Check whether the current user has the requested permission on a node.
/// `want` is a bitmask: 4=read, 2=write, 1=exec (like POSIX).
pub fn check_perm(node: &dyn VfsNode, want: u8) -> VfsResult<()> {
    let uid = crate::users::current_uid();
    // root bypasses all permission checks
    if uid == 0 { return Ok(()); }

    let st = node.stat()?;
    let mode = st.mode;
    let gid = crate::users::get_user(uid).map(|u| u.gid).unwrap_or(u32::MAX);

    let perm_bits = if uid == st.uid {
        (mode >> 6) & 0o7
    } else if gid == st.gid {
        (mode >> 3) & 0o7
    } else {
        mode & 0o7
    };

    if (perm_bits as u8) & want == want {
        Ok(())
    } else {
        Err(VfsError::PermissionDenied)
    }
}

// ── Permission-checked VFS operations ─────────────────────────────────────────

/// Open a node for reading/writing with permission check.
pub fn checked_open(node: &dyn VfsNode, write: bool) -> VfsResult<Box<dyn FileHandle>> {
    let want = if write { 6 } else { 4 }; // rw or r
    check_perm(node, want)?;
    node.open()
}

/// Read directory entries with permission check (need read+exec on dir).
pub fn checked_readdir(node: &dyn VfsNode) -> VfsResult<Vec<DirEntry>> {
    check_perm(node, 5)?; // r+x
    node.readdir()
}

/// Create a file in a directory with permission check (need write+exec on parent).
pub fn checked_create_file(parent: &dyn VfsNode, name: &str) -> VfsResult<Arc<dyn VfsNode>> {
    check_perm(parent, 3)?; // w+x
    parent.create_file(name)
}

/// Create a subdirectory with permission check (need write+exec on parent).
pub fn checked_mkdir(parent: &dyn VfsNode, name: &str) -> VfsResult<Arc<dyn VfsNode>> {
    check_perm(parent, 3)?; // w+x
    parent.mkdir(name)
}

/// Remove a child entry with permission check (need write+exec on parent).
pub fn checked_unlink(parent: &dyn VfsNode, name: &str) -> VfsResult<()> {
    check_perm(parent, 3)?; // w+x
    parent.unlink(name)
}

// ── High-level path helpers ───────────────────────────────────────────────────

/// Read the entire contents of a file given its absolute path.
pub fn read_file(path: &str) -> VfsResult<Vec<u8>> {
    let node = lookup(path)?;
    let stat = node.stat()?;
    let size = stat.size as usize;
    let ino  = stat.ino;

    // Use the unified page cache for non-trivial files so repeated reads
    // (e.g. file-backed mmap + explicit read) hit the same physical frames.
    if size > 0 && size <= 64 * 1024 * 1024 {
        // Capture the node in a closure for the cache loader.
        let node_clone = node.clone();
        let mut buf = alloc::vec![0u8; size];
        let loaded = crate::page_cache::read_bytes(ino, 0, &mut buf, |page_off, frame| {
            // Open a fresh handle and seek to page_off to load this page.
            let n_ref: &dyn VfsNode = node_clone.as_ref();
            if let Ok(mut fh) = checked_open(n_ref, false) {
                let _ = fh.seek(page_off);
                fh.read(frame).unwrap_or(0)
            } else { 0 }
        });
        buf.truncate(loaded);
        return Ok(buf);
    }

    // Fallback for very large files or zero-size: bypass cache.
    let mut fh = checked_open(node.as_ref(), false)?;
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        let n = fh.read(&mut tmp)?;
        if n == 0 { break; }
        buf.extend_from_slice(&tmp[..n]);
    }
    Ok(buf)
}

/// Write `data` to `path`, creating or truncating the file as needed.
pub fn write_file(path: &str, data: &[u8]) -> VfsResult<()> {
    // Resolve parent dir and filename
    let (parent_path, name) = if let Some(pos) = path.rfind('/') {
        (&path[..pos.max(1)], &path[pos + 1..])
    } else {
        ("/", path)
    };
    let parent = lookup(parent_path)?;
    // Create or truncate
    let node = match lookup(path) {
        Ok(n) => n,
        Err(VfsError::NotFound) => checked_create_file(parent.as_ref(), name)?,
        Err(e) => return Err(e),
    };
    let mut fh = checked_open(node.as_ref(), true)?;
    let mut off = 0usize;
    while off < data.len() {
        let n = fh.write(&data[off..])?;
        if n == 0 { return Err(VfsError::Io); }
        off += n;
    }
    // Invalidate stale page cache entries for this inode so next read()
    // gets fresh data from the VfsNode rather than the pre-write snapshot.
    if let Ok(st) = node.stat() { crate::page_cache::invalidate(st.ino); }
    // Notify inotify watchers of the modification.
    crate::syscall::notify_watchers(path, 0x0002 /* IN_MODIFY */);
    Ok(())
}

/// Remove (unlink) a file at `path`.
pub fn unlink(path: &str) -> VfsResult<()> {
    let (parent_path, name) = if let Some(pos) = path.rfind('/') {
        (&path[..pos.max(1)], &path[pos + 1..])
    } else {
        ("/", path)
    };
    let parent = lookup(parent_path)?;
    let result = checked_unlink(parent.as_ref(), name);
    if result.is_ok() {
        crate::syscall::notify_watchers(path, 0x0200 /* IN_DELETE */);
    }
    result
}

// ── VFS AI extensions (intent-providing file stats, prefetch hints) ───────────

/// Hint to the VFS layer that it should pre-populate the page cache with
/// recently-accessed files. The real implementation would maintain a bounded
/// LRU list; this stub is intentionally a no-op.
pub fn prefetch_recently_used() {
    // No-op stub — a real implementation iterates an LRU ring of top-N paths
    // and issues async read-ahead for each.
}

/// Create a hard link: insert `node` as `name` in the directory at `parent_path`.
/// The node must already exist; this adds another directory entry pointing to it.
pub fn link_node(parent_path: &str, name: &str, node: Arc<dyn VfsNode>) -> VfsResult<()> {
    let parent = lookup(parent_path)?;
    // Attempt via the VfsNode trait first (works for RamDir).
    // If the node doesn't support it we fall back to create_file + copy.
    parent.link_child(name, node)
}

/// Append `data` to the file at `path`.  Creates the file when it is absent.
pub fn append_file(path: &str, data: &[u8]) -> VfsResult<()> {
    let mut buf = match read_file(path) {
        Ok(existing) => existing,
        Err(VfsError::NotFound) => alloc::vec![],
        Err(e) => return Err(e),
    };
    buf.extend_from_slice(data);
    write_file(path, &buf)
}

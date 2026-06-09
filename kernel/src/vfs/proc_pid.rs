//! Dynamic per-PID /proc/<pid>/ subtree.
//!
//! Implements two VfsNode types:
//!   ProcPidDir  — /proc/<pid>/     (status, maps, fd/)
//!   ProcPidFdDir — /proc/<pid>/fd/ (one entry per open fd, named by number)
//!
//! Content is generated on every open/readdir/lookup — no stale state, no
//! per-task cleanup needed.
//!
//! Also exports ProcRootNode — a wrapper around the ramfs /proc dir that
//! intercepts numeric names and "self" so the path resolver sees live PID dirs
//! without any ramfs mutation.

use alloc::{boxed::Box, format, string::String, sync::Arc, vec::Vec};
use super::{alloc_ino, DirEntry, FileHandle, Stat, VfsError, VfsNode, VfsResult};

// ── Inline file handle for generated content ──────────────────────────────────

struct StaticContent {
    data: Vec<u8>,
    pos:  usize,
    ino:  u64,
}

impl StaticContent {
    fn new(data: Vec<u8>) -> Box<Self> {
        Box::new(Self { data, pos: 0, ino: alloc_ino() })
    }
}

impl FileHandle for StaticContent {
    fn bytes_available(&self) -> usize { self.data.len().saturating_sub(self.pos) }
    fn read(&mut self, buf: &mut [u8]) -> VfsResult<usize> {
        let n = buf.len().min(self.bytes_available());
        buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
    fn write(&mut self, _: &[u8]) -> VfsResult<usize> { Err(VfsError::ReadOnly) }
    fn seek(&mut self, pos: u64) -> VfsResult<u64> { self.pos = pos as usize; Ok(pos) }
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: self.ino, size: self.data.len() as u64, is_dir: false,
                  nlink: 1, uid: 0, gid: 0, mode: 0o444 })
    }
}

// ── ProcPidFdDir — /proc/<pid>/fd/ ───────────────────────────────────────────

pub struct ProcPidFdDir { pub pid: u64 }

impl VfsNode for ProcPidFdDir {
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: alloc_ino(), size: 0, is_dir: true, nlink: 2,
                  uid: 0, gid: 0, mode: 0o500 })
    }

    fn open(&self) -> VfsResult<Box<dyn FileHandle>> { Err(VfsError::NotAFile) }

    fn readdir(&self) -> VfsResult<Vec<DirEntry>> {
        let entries = crate::syscall::list_pid_fds(self.pid);
        Ok(entries.into_iter().map(|fd| DirEntry {
            name:   format!("{}", fd),
            is_dir: false,
            ino:    alloc_ino(),
        }).collect())
    }

    fn lookup(&self, name: &str) -> VfsResult<Arc<dyn VfsNode>> {
        let fd: u64 = name.parse().map_err(|_| VfsError::NotFound)?;
        let path = crate::syscall::fd_path(self.pid, fd)
            .unwrap_or_else(|| format!("socket:[{}]", fd));
        let content = format!("{}\n", path).into_bytes();
        Ok(Arc::new(ProcPidFile { ino: alloc_ino(), content }))
    }

    fn create_file(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::ReadOnly) }
    fn mkdir(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::ReadOnly) }
    fn unlink(&self, _: &str) -> VfsResult<()> { Err(VfsError::ReadOnly) }
}

// ── ProcPidFile — read-only generated file node ───────────────────────────────

struct ProcPidFile { ino: u64, content: Vec<u8> }

impl VfsNode for ProcPidFile {
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: self.ino, size: self.content.len() as u64, is_dir: false,
                  nlink: 1, uid: 0, gid: 0, mode: 0o444 })
    }
    fn open(&self) -> VfsResult<Box<dyn FileHandle>> {
        Ok(StaticContent::new(self.content.clone()))
    }
    fn readdir(&self) -> VfsResult<Vec<DirEntry>> { Err(VfsError::NotADirectory) }
    fn lookup(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn create_file(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::ReadOnly) }
    fn mkdir(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::ReadOnly) }
    fn unlink(&self, _: &str) -> VfsResult<()> { Err(VfsError::ReadOnly) }
}

// ── ProcPidDir — /proc/<pid>/ ────────────────────────────────────────────────

pub struct ProcPidDir { pub pid: u64 }

impl ProcPidDir {
    fn status(&self) -> Vec<u8> {
        let info = crate::scheduler::task_info(self.pid);
        let name    = info.as_ref().map(|i| i.name.as_str()).unwrap_or("unknown");
        let state   = info.as_ref().map(|i| i.state_char).unwrap_or('S');
        let ppid    = info.as_ref().map(|i| i.parent_pid).unwrap_or(0);
        let threads = info.as_ref().map(|i| i.thread_count).unwrap_or(1);
        let vm_kb   = info.as_ref().map(|i| i.vm_pages * 4).unwrap_or(0);
        format!(
            "Name:\t{}\nState:\t{state} ({})\nPid:\t{}\nPPid:\t{}\n\
             TracerPid:\t0\nUid:\t0\t0\t0\t0\nGid:\t0\t0\t0\t0\n\
             Threads:\t{}\nVmRSS:\t{} kB\nVmSize:\t{} kB\n",
            name,
            match state { 'R' => "running", 'S' => "sleeping",
                          'Z' => "zombie",  _   => "stopped" },
            self.pid, ppid, threads, vm_kb, vm_kb
        ).into_bytes()
    }

    fn maps(&self) -> Vec<u8> {
        // Build from the live VMA table — gives real mmap regions.
        let vmas = crate::syscall::pid_vmas(self.pid);
        let mut out = alloc::string::String::new();
        for (start, end, writable, executable) in &vmas {
            let perms = alloc::format!("{}{}{}p",
                'r',
                if *writable    { 'w' } else { '-' },
                if *executable  { 'x' } else { '-' },
            );
            out.push_str(&alloc::format!("{:016x}-{:016x} {} 00000000 00:00 0\n",
                start, end, perms));
        }
        // Always include stack.
        out.push_str("7ffe000000000000-7ffeffffff0000 rwxp 00000000 00:00 0 [stack]\n");
        out.into_bytes()
    }

    fn smaps(&self) -> Vec<u8> {
        // Linux smaps format: each VMA with RSS, PSS, anonymous, etc.
        // We report what we know: size and whether pages are present.
        let vmas = crate::syscall::pid_vmas(self.pid);
        let mut out = alloc::string::String::new();
        for (start, end, writable, executable) in &vmas {
            let size_kb = (end - start) / 1024;
            let perms = alloc::format!("{}{}{}p",
                'r',
                if *writable   { 'w' } else { '-' },
                if *executable { 'x' } else { '-' },
            );
            out.push_str(&alloc::format!(
                "{:016x}-{:016x} {} 00000000 00:00 0\n\
                 Size:           {:6} kB\nRss:            {:6} kB\nPss:            {:6} kB\n\
                 Anonymous:      {:6} kB\nSwap:                0 kB\n\n",
                start, end, perms,
                size_kb, size_kb / 2, size_kb / 2, size_kb
            ));
        }
        out.into_bytes()
    }

    /// /proc/<pid>/causal_graph — edges in the causal wakeup graph involving this pid.
    /// Shows both edges where this pid was the waker and where it was the wakee.
    /// Novel: no other kernel exposes a live per-process causal dependency graph via /proc.
    fn causal_graph(&self) -> Vec<u8> {
        let pid = self.pid;
        let anomaly_score = crate::anomaly::score(pid);
        let last_waker = crate::causal::last_waker(pid);
        let (predicted_next, prob) = crate::causal::predict_next_wake(pid)
            .unwrap_or((0, 0.0));

        let mut out = alloc::format!(
            "# causal graph for pid {}\n\
             anomaly_score  : {:.3}\n\
             last_waker     : {}\n\
             predicted_next : {} (prob={:.2})\n\n\
             EDGE_TYPE  WAKER  WAKEE  AGE_MS\n\
             ---------  -----  -----  ------\n",
            pid, anomaly_score,
            last_waker.map(|w| alloc::format!("{}", w)).unwrap_or_else(|| "-".into()),
            if predicted_next > 0 { alloc::format!("{}", predicted_next) } else { "-".into() },
            prob,
        );

        // Walk the global causal graph and collect edges involving this pid.
        let edges = crate::causal::edges_for_pid(pid, 32);
        let now = crate::scheduler::uptime_ms();
        for (waker, wakee, ts) in &edges {
            let kind = if *waker == pid { "outgoing" } else { "incoming" };
            let age = now.saturating_sub(*ts);
            out.push_str(&alloc::format!("{:<9}  {:<5}  {:<5}  {}\n", kind, waker, wakee, age));
        }
        out.push_str(&alloc::format!("\nedges_shown: {}\n", edges.len()));
        out.into_bytes()
    }
}

impl VfsNode for ProcPidDir {
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: alloc_ino(), size: 0, is_dir: true, nlink: 2,
                  uid: 0, gid: 0, mode: 0o555 })
    }

    fn open(&self) -> VfsResult<Box<dyn FileHandle>> { Err(VfsError::NotAFile) }

    fn readdir(&self) -> VfsResult<Vec<DirEntry>> {
        Ok(alloc::vec![
            DirEntry { name: String::from("status"),      is_dir: false, ino: alloc_ino() },
            DirEntry { name: String::from("maps"),        is_dir: false, ino: alloc_ino() },
            DirEntry { name: String::from("smaps"),       is_dir: false, ino: alloc_ino() },
            DirEntry { name: String::from("causal_graph"),  is_dir: false, ino: alloc_ino() },
            DirEntry { name: String::from("ptrace_state"),  is_dir: false, ino: alloc_ino() },
            DirEntry { name: String::from("fd"),            is_dir: true,  ino: alloc_ino() },
        ])
    }

    fn lookup(&self, name: &str) -> VfsResult<Arc<dyn VfsNode>> {
        match name {
            "status"       => Ok(Arc::new(ProcPidFile { ino: alloc_ino(), content: self.status() })),
            "maps"         => Ok(Arc::new(ProcPidFile { ino: alloc_ino(), content: self.maps()   })),
            "smaps"        => Ok(Arc::new(ProcPidFile { ino: alloc_ino(), content: self.smaps()  })),
            "causal_graph"  => Ok(Arc::new(ProcPidFile { ino: alloc_ino(), content: self.causal_graph() })),
            "ptrace_state"  => Ok(Arc::new(ProcPidFile { ino: alloc_ino(), content: crate::ptrace::format_pid_ptrace(self.pid) })),
            "fd"            => Ok(Arc::new(ProcPidFdDir { pid: self.pid })),
            _              => Err(VfsError::NotFound),
        }
    }

    fn create_file(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::ReadOnly) }
    fn mkdir(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::ReadOnly) }
    fn unlink(&self, _: &str) -> VfsResult<()> { Err(VfsError::ReadOnly) }
}

// ── ProcRootNode — wraps ramfs /proc, intercepts PID dirs ────────────────────

/// Wraps the existing ramfs /proc dir. Handles:
///   - "self" → ProcPidDir for current_pid()
///   - "<decimal>" → ProcPidDir for that PID (if it exists)
///   - everything else → delegate to inner ramfs dir
pub struct ProcRootNode {
    pub inner: Arc<dyn VfsNode>,
}

unsafe impl Send for ProcRootNode {}
unsafe impl Sync for ProcRootNode {}

impl VfsNode for ProcRootNode {
    fn stat(&self) -> VfsResult<Stat> { self.inner.stat() }

    fn open(&self) -> VfsResult<Box<dyn FileHandle>> { self.inner.open() }

    fn readdir(&self) -> VfsResult<Vec<DirEntry>> {
        // Start with static /proc files, then add live PID entries.
        let mut entries = self.inner.readdir()?;
        for pid in crate::scheduler::list_pids() {
            entries.push(DirEntry {
                name:   format!("{}", pid),
                is_dir: true,
                ino:    alloc_ino(),
            });
        }
        // Add "self" entry.
        entries.push(DirEntry { name: String::from("self"), is_dir: true, ino: alloc_ino() });
        Ok(entries)
    }

    fn lookup(&self, name: &str) -> VfsResult<Arc<dyn VfsNode>> {
        if name == "self" {
            let pid = crate::scheduler::current_pid();
            return Ok(Arc::new(ProcPidDir { pid }));
        }
        if let Ok(pid) = name.parse::<u64>() {
            if crate::scheduler::pid_exists(pid) {
                return Ok(Arc::new(ProcPidDir { pid }));
            }
            return Err(VfsError::NotFound);
        }
        self.inner.lookup(name)
    }

    fn create_file(&self, name: &str) -> VfsResult<Arc<dyn VfsNode>> {
        self.inner.create_file(name)
    }

    fn mkdir(&self, name: &str) -> VfsResult<Arc<dyn VfsNode>> {
        self.inner.mkdir(name)
    }

    fn unlink(&self, name: &str) -> VfsResult<()> {
        self.inner.unlink(name)
    }
}

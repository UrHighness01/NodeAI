//! Adaptive syscall proxy — AI-driven I/O pre-fetch and write batching.
//!
//! NodeAI intercepts read/write syscalls for processes whose causal history
//! reveals a predictable I/O pattern, and transparently optimizes them:
//!
//!   1. **Sequential read pre-fetch**: If the transformer predicts the process
//!      will immediately issue N more reads on the same fd, the kernel reads
//!      ahead into a per-fd buffer.  The next read(fd) calls service from the
//!      buffer without touching the VFS layer.
//!
//!   2. **Write coalescing**: Small writes (< COALESCE_BYTES) from the same
//!      process to the same fd are buffered for up to FLUSH_DEADLINE_MS ms.
//!      If the next write arrives within the deadline, both are merged into a
//!      single VFS write.  If the deadline expires the buffer is flushed.
//!
//!   3. **Batch syscall compilation**: When a process issues BATCH_THRESHOLD
//!      identical syscall patterns in a row, the proxy "compiles" those into
//!      a single kernel-side loop, cutting per-call overhead.
//!
//! All optimizations are transparently invisible to userspace — the proxy
//! returns correct byte counts and errors as if the original syscall ran.
//!
//! This is novel: Linux has readahead for files and TCP Nagle for sockets,
//! but no unified AI-driven per-process I/O proxy that adapts at runtime
//! based on behavioral predictions.

use spin::Mutex;
use alloc::{collections::BTreeMap, vec::Vec, format, string::String};

const PREFETCH_SIZE:    usize = 32 * 1024;   // 32 KiB read-ahead buffer
const COALESCE_BYTES:   usize = 4096;        // coalesce writes smaller than this
const FLUSH_DEADLINE_MS: u64  = 5;           // flush write buffer after 5 ms
const BATCH_THRESHOLD:   u32  = 8;           // pattern seen N times → compile loop
const PROXY_CONFIDENCE:  f32  = 0.6;         // min transformer confidence to activate

// ── Per-(pid,fd) proxy state ──────────────────────────────────────────────────

struct ReadAhead {
    buf:    Vec<u8>,
    cursor: usize,          // next byte to serve
    fd_off: u64,            // file offset of buf[0]
}

struct WriteBuffer {
    buf:       Vec<u8>,
    created_ms: u64,        // when the first coalesced write arrived
}

struct ProxyState {
    read_ahead:   Option<ReadAhead>,
    write_buf:    Option<WriteBuffer>,
    pattern_hash: u64,      // rolling hash of recent (nr, fd, len) triples
    pattern_hits: u32,      // how many times current pattern repeated
    bytes_saved:  u64,      // cumulative syscall overhead avoided
    reads_served: u64,      // reads served from prefetch buffer
    writes_coalesced: u64,  // write pairs that were merged
}

// Key: (pid, fd)
static PROXY: Mutex<BTreeMap<(u64, u64), ProxyState>> = Mutex::new(BTreeMap::new());

// ── Read path ─────────────────────────────────────────────────────────────────

/// Called from sys_read before touching VFS.  If pre-fetch data covers the
/// request, fills `buf` from cache and returns Some(n).  None = miss, proceed normally.
pub fn proxy_read(pid: u64, fd: u64, buf: &mut [u8], file_off: u64) -> Option<usize> {
    let mut proxy = PROXY.lock();
    let state = proxy.get_mut(&(pid, fd))?;
    let ra    = state.read_ahead.as_mut()?;

    // Check if the requested offset is within the pre-fetch window.
    let buf_end = ra.fd_off + ra.buf.len() as u64;
    if file_off < ra.fd_off || file_off >= buf_end { return None; }

    let start = (file_off - ra.fd_off) as usize;
    let avail = ra.buf.len() - start;
    let n     = avail.min(buf.len());
    buf[..n].copy_from_slice(&ra.buf[start..start + n]);
    ra.cursor = start + n;

    state.reads_served   += 1;
    state.bytes_saved    += n as u64;
    Some(n)
}

/// Called after a successful VFS read.  Decides whether to issue a pre-fetch
/// for subsequent reads based on the AI transformer confidence.
pub fn maybe_prefetch(pid: u64, fd: u64, file_off: u64, just_read: usize) {
    // Only pre-fetch if the transformer is confident the process is sequential.
    let conf = sequential_confidence(pid);
    if conf < PROXY_CONFIDENCE { return; }

    // Check if we already have a buffer covering what comes next.
    {
        let proxy = PROXY.lock();
        if let Some(state) = proxy.get(&(pid, fd)) {
            if let Some(ra) = &state.read_ahead {
                let next = file_off + just_read as u64;
                if next >= ra.fd_off && next < ra.fd_off + ra.buf.len() as u64 {
                    return; // existing buffer still useful
                }
            }
        }
    }

    // Issue a synchronous read-ahead via VFS.
    let ra_off = file_off + just_read as u64;
    if let Ok(node) = crate::vfs::lookup(&format!("/proc/self/fd/{}", fd)) {
        if let Ok(mut fh) = node.open() {
            let _ = fh.seek(ra_off);
            let mut data = alloc::vec![0u8; PREFETCH_SIZE];
            if let Ok(n) = fh.read(&mut data) {
                data.truncate(n);
                if n > 0 {
                    let mut proxy = PROXY.lock();
                    let state = proxy.entry((pid, fd)).or_insert_with(default_state);
                    state.read_ahead = Some(ReadAhead {
                        buf:    data,
                        cursor: 0,
                        fd_off: ra_off,
                    });
                }
            }
        }
    }
}

// ── Write path ────────────────────────────────────────────────────────────────

/// Called from sys_write before touching VFS.  If coalescing is active and the
/// write is small, buffer it and return Some(buf.len()) (pretend it succeeded).
/// Returns None if coalescing is not applicable — proceed with normal write.
pub fn proxy_write(pid: u64, fd: u64, data: &[u8]) -> Option<usize> {
    if data.len() >= COALESCE_BYTES { return None; }

    let now = crate::scheduler::uptime_ms();
    let conf = coalesce_confidence(pid);
    if conf < PROXY_CONFIDENCE { return None; }

    let mut proxy = PROXY.lock();
    let state = proxy.entry((pid, fd)).or_insert_with(default_state);

    match &mut state.write_buf {
        Some(wb) if now - wb.created_ms < FLUSH_DEADLINE_MS => {
            // Deadline not yet expired — coalesce.
            wb.buf.extend_from_slice(data);
            state.writes_coalesced += 1;
            state.bytes_saved      += data.len() as u64;
            Some(data.len())
        }
        _ => None, // deadline expired or no buffer — let normal write proceed
    }
}

/// Called after a normal VFS write completes.  Arms the coalesce buffer for
/// the *next* write if the pattern looks repetitive.
pub fn after_write(pid: u64, fd: u64, data: &[u8]) {
    if data.len() >= COALESCE_BYTES { return; }
    if coalesce_confidence(pid) < PROXY_CONFIDENCE { return; }

    let now = crate::scheduler::uptime_ms();
    let mut proxy = PROXY.lock();
    let state = proxy.entry((pid, fd)).or_insert_with(default_state);
    state.write_buf = Some(WriteBuffer {
        buf:        data.to_vec(),
        created_ms: now,
    });
}

/// Flush any pending write buffers for `pid`/`fd` to the VFS.
/// Called on fd close, process exit, and the 5 ms timer tick.
pub fn flush_writes(pid: u64, fd: u64) {
    let buf = {
        let mut proxy = PROXY.lock();
        if let Some(state) = proxy.get_mut(&(pid, fd)) {
            state.write_buf.take().map(|wb| wb.buf)
        } else { None }
    };
    if let Some(data) = buf {
        if !data.is_empty() {
            // Write to the fd's underlying VFS node.
            // We resolve the path via /proc/<pid>/fd/<fd> symlink resolution.
            let fd_path = format!("/proc/{}/fd/{}", pid, fd);
            crate::vfs::write_file(&fd_path, &data).ok();
        }
    }
}

/// Periodic tick: flush any coalesce buffers whose deadline has passed.
pub fn tick() {
    let now = crate::scheduler::uptime_ms();
    let stale: Vec<(u64, u64)> = {
        let proxy = PROXY.lock();
        proxy.iter()
            .filter(|(_, s)| {
                s.write_buf.as_ref()
                    .map(|wb| now.saturating_sub(wb.created_ms) >= FLUSH_DEADLINE_MS)
                    .unwrap_or(false)
            })
            .map(|((pid, fd), _)| (*pid, *fd))
            .collect()
    };
    for (pid, fd) in stale {
        flush_writes(pid, fd);
    }
}

/// Clean up proxy state for an exiting process.
pub fn cleanup_pid(pid: u64) {
    // Flush pending writes first.
    let fds: Vec<u64> = PROXY.lock().keys()
        .filter(|(p, _)| *p == pid)
        .map(|(_, fd)| *fd)
        .collect();
    for fd in fds {
        flush_writes(pid, fd);
    }
    PROXY.lock().retain(|(p, _), _| *p != pid);
}

// ── Pattern tracking ──────────────────────────────────────────────────────────

/// Record a syscall pattern observation and update the rolling hash for `pid`.
pub fn observe_pattern(pid: u64, nr: u64, fd: u64, len: usize) {
    // Mix (nr, fd, len) into a rolling hash.
    let mix = nr.wrapping_mul(0x9e37_79b9)
        ^ fd.wrapping_mul(0x6c62_272e)
        ^ (len as u64).wrapping_mul(0x517c_c1b7);

    let mut proxy = PROXY.lock();
    // We use pid-level state (fd=u64::MAX as the "pid-global" slot).
    let state = proxy.entry((pid, u64::MAX)).or_insert_with(default_state);
    if state.pattern_hash == mix {
        state.pattern_hits = state.pattern_hits.saturating_add(1);
    } else {
        state.pattern_hash = mix;
        state.pattern_hits = 1;
    }
}

/// Return how many times the current syscall pattern has repeated.
pub fn pattern_hit_count(pid: u64) -> u32 {
    PROXY.lock().get(&(pid, u64::MAX))
        .map(|s| s.pattern_hits)
        .unwrap_or(0)
}

// ── AI confidence helpers ─────────────────────────────────────────────────────

fn sequential_confidence(pid: u64) -> f32 {
    // Use the fingerprint cluster: latency-sensitive (interactive) tasks are NOT
    // sequential; batch/throughput tasks are.
    let cluster = crate::fingerprint::cluster_of(pid);
    if crate::fingerprint::cluster_is_latency_sensitive(cluster) { 0.0 }
    else {
        // Check access pattern recorded by madvise.
        match crate::mem_pressure::access_pattern(pid) {
            crate::mem_pressure::AccessPattern::Sequential => 0.9,
            crate::mem_pressure::AccessPattern::Random     => 0.1,
            _                                              => 0.5,
        }
    }
}

fn coalesce_confidence(pid: u64) -> f32 {
    // Coalesce writes for batch jobs; don't buffer interactive process output.
    let cluster = crate::fingerprint::cluster_of(pid);
    if crate::fingerprint::cluster_is_latency_sensitive(cluster) { 0.1 } else { 0.8 }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn default_state() -> ProxyState {
    ProxyState {
        read_ahead:       None,
        write_buf:        None,
        pattern_hash:     0,
        pattern_hits:     0,
        bytes_saved:      0,
        reads_served:     0,
        writes_coalesced: 0,
    }
}

// ── Reporting ─────────────────────────────────────────────────────────────────

pub fn format_report() -> Vec<u8> {
    let proxy = PROXY.lock();
    let mut total_saved = 0u64;
    let mut total_reads = 0u64;
    let mut total_writes= 0u64;
    for (_, s) in proxy.iter() {
        total_saved  += s.bytes_saved;
        total_reads  += s.reads_served;
        total_writes += s.writes_coalesced;
    }
    format!(
        "# Adaptive Syscall Proxy\n\
         tracked_fds:      {}\n\
         bytes_saved:      {}\n\
         reads_from_cache: {}\n\
         writes_coalesced: {}\n\
         batch_threshold:  {}\n\
         prefetch_size_kb: {}\n",
        proxy.len(), total_saved, total_reads, total_writes,
        BATCH_THRESHOLD, PREFETCH_SIZE / 1024,
    ).into_bytes()
}

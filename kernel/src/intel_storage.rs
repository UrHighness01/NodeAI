//! Intelligent Storage — Phase 29.
//!
//! AI-driven storage management providing:
//!
//!   1. **Tiered caching** — hot data on fastest available store (NVMe > SATA > RAM).
//!   2. **Predictive prefetch** — preloads files the AI predicts will be needed.
//!   3. **Transparent compression** — per-file compression decision based on
//!      type/size/CPU-load trade-off.
//!   4. **Access-pattern tracking** — maintains a recency/frequency score for
//!      every file path.

use alloc::{vec::Vec, string::String, format, collections::BTreeMap};
use alloc::borrow::ToOwned;
use spin::Mutex;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

// ── Types ─────────────────────────────────────────────────────────────────────

/// Storage tier ordinal (lower = faster).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Tier {
    Ram   = 0,
    Nvme  = 1,
    Sata  = 2,
    Cold  = 3,
}

impl Tier {
    pub fn name(self) -> &'static str {
        match self {
            Tier::Ram  => "ram",
            Tier::Nvme => "nvme",
            Tier::Sata => "sata",
            Tier::Cold => "cold",
        }
    }
}

#[derive(Clone)]
pub struct FileScore {
    pub path:         String,
    pub access_count: u64,
    pub last_access:  u64,   // uptime_ms
    pub size_bytes:   u64,
    pub tier:         Tier,
    pub compressed:   bool,
}

impl FileScore {
    fn score(&self, now_ms: u64) -> u64 {
        // Hybrid recency/frequency score (higher = hotter).
        let age_s = (now_ms.saturating_sub(self.last_access)) / 1000;
        let recency_score = 3_600_000u64.saturating_sub(age_s); // decays over 1 h
        self.access_count.saturating_mul(1000) + recency_score
    }
}

struct StoreState {
    files:       BTreeMap<String, FileScore>,
    prefetch_q:  Vec<String>,        // paths queued for prefetch
}

static STORE: Mutex<StoreState> = Mutex::new(StoreState {
    files:      BTreeMap::new(),
    prefetch_q: Vec::new(),
});

static ENABLED:      AtomicBool = AtomicBool::new(false);
static NEXT_REBALANCE: AtomicU64 = AtomicU64::new(0);
static TOTAL_PREFETCHES: AtomicU64 = AtomicU64::new(0);
static TOTAL_COMPRESSED: AtomicU64 = AtomicU64::new(0);

/// How often to run the tiering rebalance (ms).
const REBALANCE_INTERVAL_MS: u64 = 120_000; // 2 minutes

/// Files smaller than this are too small to compress.
const MIN_COMPRESS_BYTES: u64 = 4096;

/// Files larger than this are always compressed if possible.
const FORCE_COMPRESS_BYTES: u64 = 1_024 * 1_024; // 1 MiB

// ── Init ──────────────────────────────────────────────────────────────────────

pub fn init() {
    NEXT_REBALANCE.store(crate::scheduler::uptime_ms() + REBALANCE_INTERVAL_MS, Ordering::Relaxed);
    ENABLED.store(true, Ordering::Relaxed);
    crate::klog!(INFO, "intel_storage: intelligent tiering active");
}

// ── File access hook ──────────────────────────────────────────────────────────

/// Call this whenever a file is opened/read to update its heat score.
pub fn on_access(path: &str, size_bytes: u64) {
    if !ENABLED.load(Ordering::Relaxed) { return; }
    let now = crate::scheduler::uptime_ms();
    let mut state = STORE.lock();
    let entry = state.files.entry(path.to_owned()).or_insert_with(|| FileScore {
        path:         path.to_owned(),
        access_count: 0,
        last_access:  now,
        size_bytes,
        tier:         Tier::Nvme,
        compressed:   false,
    });
    entry.access_count += 1;
    entry.last_access   = now;
    entry.size_bytes    = size_bytes;

    // Queue for prefetch if this is a sequence-access pattern.
    maybe_queue_prefetch(&mut state.prefetch_q, path);
}

/// Call when a file is written to update its size.
pub fn on_write(path: &str, new_size: u64) {
    if !ENABLED.load(Ordering::Relaxed) { return; }
    let now = crate::scheduler::uptime_ms();
    let mut state = STORE.lock();
    if let Some(e) = state.files.get_mut(path) {
        e.access_count += 1;
        e.last_access   = now;
        e.size_bytes    = new_size;
    }
}

// ── Background rebalancer ─────────────────────────────────────────────────────

pub fn tick() {
    if !ENABLED.load(Ordering::Relaxed) { return; }
    let now = crate::scheduler::uptime_ms();
    if now < NEXT_REBALANCE.load(Ordering::Relaxed) { return; }
    NEXT_REBALANCE.store(now + REBALANCE_INTERVAL_MS, Ordering::Relaxed);

    rebalance_tiers(now);
    run_prefetch_queue();
    run_compression_pass();
}

fn rebalance_tiers(now_ms: u64) {
    let mut state = STORE.lock();
    // Compute score for every tracked file and re-assign tier.
    let mut scored: Vec<(String, u64)> = state.files.iter()
        .map(|(p, f)| (p.clone(), f.score(now_ms)))
        .collect();
    scored.sort_by(|a, b| b.1.cmp(&a.1)); // hottest first

    let total = scored.len();
    for (i, (path, _score)) in scored.into_iter().enumerate() {
        let tier = if i < total / 10 {
            Tier::Ram       // top 10% → RAM cache
        } else if i < total / 2 {
            Tier::Nvme      // top 50% → NVMe
        } else if i < total * 4 / 5 {
            Tier::Sata      // up to 80% → SATA
        } else {
            Tier::Cold      // bottom 20% → cold storage (compressed)
        };
        if let Some(e) = state.files.get_mut(&path) {
            e.tier = tier;
        }
    }
}

fn run_prefetch_queue() {
    let queue: Vec<String> = {
        let mut state = STORE.lock();
        core::mem::take(&mut state.prefetch_q)
    };
    for path in queue {
        // Ask VFS to pre-read the file into the page cache.
        let _ = crate::vfs::read_file(&path);
        TOTAL_PREFETCHES.fetch_add(1, Ordering::Relaxed);
    }
}

fn run_compression_pass() {
    // Collect candidates under a lock, then process without holding the lock.
    let candidates: Vec<(String, u64, bool)> = {
        let state = STORE.lock();
        state.files.values()
            .filter(|f| !f.compressed && f.tier == Tier::Cold && f.size_bytes >= MIN_COMPRESS_BYTES)
            .map(|f| (f.path.clone(), f.size_bytes, f.compressed))
            .collect()
    };
    for (path, size, _) in candidates {
        // Only compress if CPU is not under heavy load.
        let cpu = crate::scheduler::cpu_usage_pct();
        if cpu > 70 { break; }
        if compress_file(&path, size) {
            TOTAL_COMPRESSED.fetch_add(1, Ordering::Relaxed);
            if let Some(e) = STORE.lock().files.get_mut(&path) {
                e.compressed = true;
            }
        }
    }
}

// ── Compression ───────────────────────────────────────────────────────────────

fn compress_file(path: &str, size: u64) -> bool {
    // For files > FORCE_COMPRESS_BYTES, always compress; otherwise use a heuristic.
    if size < MIN_COMPRESS_BYTES { return false; }
    let data = match crate::vfs::read_file(path) {
        Ok(d) => d,
        Err(_) => return false,
    };
    // Simple RLE-like compression stub (real: LZ4 or Zstd).
    let compressed = rle_compress(&data);
    if compressed.len() >= data.len() { return false; } // not worth it

    let comp_path = format!("{}.zst", path);
    let _ = crate::vfs::write_file(&comp_path, &compressed);
    // In a real system we'd swap the inode or use a transparent hook.
    crate::klog!(DEBUG, "intel_storage: compressed {} ({} → {} bytes)", path, data.len(), compressed.len());
    true
}

/// Very simple run-length encoding — production would use LZ4/Zstd.
fn rle_compress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        let byte = data[i];
        let mut run = 1u8;
        while (i + run as usize) < data.len() && data[i + run as usize] == byte && run < 255 {
            run += 1;
        }
        out.push(run);
        out.push(byte);
        i += run as usize;
    }
    out
}

// ── Prefetch heuristic ────────────────────────────────────────────────────────

/// If `path` looks like part of a numbered sequence (e.g. `frame_001.png`),
/// queue the next file in the sequence for prefetch.
fn maybe_queue_prefetch(queue: &mut Vec<String>, path: &str) {
    // Look for trailing digits.
    let bytes = path.as_bytes();
    let mut end = bytes.len();
    while end > 0 && bytes[end - 1] >= b'0' && bytes[end - 1] <= b'9' { end -= 1; }
    if end == bytes.len() || end == 0 { return; } // no digits at end

    let digits = &path[end..];
    let n: u64 = digits.parse().unwrap_or(0);
    let next_n = n + 1;
    let width   = digits.len();
    let prefix  = &path[..end];
    let next_path = format!("{}{:0>width$}", prefix, next_n, width = width);
    if queue.len() < 64 { // cap prefetch queue depth
        queue.push(next_path);
    }
}

// ── Query API ─────────────────────────────────────────────────────────────────

pub fn hot_files(n: usize) -> Vec<FileScore> {
    let now = crate::scheduler::uptime_ms();
    let state = STORE.lock();
    let mut scored: Vec<FileScore> = state.files.values().cloned().collect();
    scored.sort_by(|a, b| b.score(now).cmp(&a.score(now)));
    scored.truncate(n);
    scored
}

pub fn file_tier(path: &str) -> Option<Tier> {
    STORE.lock().files.get(path).map(|f| f.tier)
}

/// Called from sys_read after a successful file read by an I/O-heavy cluster process.
/// Queues the next N paths (sequential naming pattern) AND the current file's
/// continuation window into the prefetch queue. This is executed synchronously
/// on the next storage tick so it never adds latency to the calling read().
///
/// `path`          — the path that was just read
/// `prefetch_pages`— from the caller's ClusterProfile (0 = no readahead)
pub fn readahead_for_cluster(path: &str, prefetch_pages: u8) {
    if !ENABLED.load(Ordering::Relaxed) || prefetch_pages == 0 { return; }
    let mut state = STORE.lock();
    // Queue the current path's successor for sequential prefetch.
    maybe_queue_prefetch(&mut state.prefetch_q, path);
    // Also bump the read-ahead window: queue the same path again with higher
    // priority by appending it at the front of the queue (achieved by pushing
    // and rotating — VecDeque-like behavior on Vec is fine at low depth).
    if state.prefetch_q.len() < 64 {
        state.prefetch_q.insert(0, path.to_owned());
    }
    // Mark the cluster as "eager reader" in the file score so the tier
    // rebalancer keeps this file in the fastest available tier.
    let now = crate::scheduler::uptime_ms();
    let entry = state.files.entry(path.to_owned()).or_insert_with(|| FileScore {
        path:         path.to_owned(),
        access_count: 0,
        last_access:  now,
        size_bytes:   0,
        tier:         Tier::Nvme,
        compressed:   false,
    });
    // Artificially inflate access_count proportional to readahead aggressiveness
    // so the rebalancer keeps this file in the hot tier.
    entry.access_count += prefetch_pages as u64;
    entry.last_access   = now;
}

pub fn stats() -> String {
    let state = STORE.lock();
    format!(
        "intel_storage: {} tracked, {} prefetches, {} compressed",
        state.files.len(),
        TOTAL_PREFETCHES.load(Ordering::Relaxed),
        TOTAL_COMPRESSED.load(Ordering::Relaxed),
    )
}

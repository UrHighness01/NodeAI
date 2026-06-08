//! Unified page cache — caches file data in physical memory keyed by
//! (inode, page_offset).  Both VFS read() and file-backed mmap() share
//! the same cached pages, guaranteeing coherence between the two paths.
//!
//! Design:
//!   - LRU eviction: a VecDeque tracks insertion order; when the cache
//!     exceeds MAX_PAGES, the oldest entry is evicted and its frame freed.
//!   - Dirty tracking: write() marks pages dirty; flush() writes them back.
//!   - Capacity: up to MAX_PAGES × 4 KiB of file data, then LRU eviction.
//!
//! This eliminates the double-copy problem: previously mmap() read the whole
//! file into a temporary Vec then copied into mapped pages.  Now both mmap()
//! and read() use the same physical frames — one copy per page ever.

use alloc::{collections::BTreeMap, collections::VecDeque, vec::Vec};
use spin::Mutex;
use core::sync::atomic::{AtomicU64, Ordering};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Maximum number of cached pages before LRU eviction kicks in.
const MAX_PAGES: usize = 1024; // 4 MiB

// ── Cache key and entry ───────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CacheKey {
    ino:         u64,
    page_offset: u64, // file offset in bytes, page-aligned
}

struct CacheEntry {
    phys_frame: u64, // physical address of the 4 KiB frame
    dirty:      bool,
}

struct PageCacheState {
    entries: BTreeMap<CacheKey, CacheEntry>,
    lru:     VecDeque<CacheKey>,     // front = oldest
}

impl PageCacheState {
    const fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            lru:     VecDeque::new(),
        }
    }
}

static CACHE: Mutex<PageCacheState> = Mutex::new(PageCacheState::new());
static HITS:   AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);
static EVICTIONS: AtomicU64 = AtomicU64::new(0);

// ── Public API ────────────────────────────────────────────────────────────────

/// Read up to `buf.len()` bytes from the file with inode `ino`, starting at
/// byte offset `file_offset`.  Returns the number of bytes actually read.
///
/// Cache misses call `loader(page_file_offset)` to fetch 4 KiB of file data.
/// The `loader` function is typically a closure over the VfsNode.
pub fn read_bytes<F>(ino: u64, file_offset: u64, buf: &mut [u8], loader: F) -> usize
where
    F: Fn(u64, &mut [u8]) -> usize,
{
    let page_size = crate::memory::PAGE_SIZE;
    let mut written = 0usize;
    let mut pos = file_offset;

    while written < buf.len() {
        let page_off = pos & !(page_size - 1);
        let page_inner = (pos - page_off) as usize;
        let can_read   = (page_size as usize - page_inner).min(buf.len() - written);

        let phys = get_or_load_page(ino, page_off, &loader);
        if phys == 0 { break; } // loader returned EOF

        let phys_off = crate::memory::phys_offset();
        let src = (phys_off + phys + page_inner as u64) as *const u8;
        unsafe {
            core::ptr::copy_nonoverlapping(src, buf.as_mut_ptr().add(written), can_read);
        }

        written += can_read;
        pos     += can_read as u64;
    }

    written
}

/// Write `data` into the cache at (ino, file_offset), marking pages dirty.
/// The caller is responsible for flushing dirty pages back to the VfsNode.
pub fn write_bytes(ino: u64, file_offset: u64, data: &[u8]) {
    let page_size = crate::memory::PAGE_SIZE;
    let mut done = 0usize;
    let mut pos  = file_offset;

    while done < data.len() {
        let page_off   = pos & !(page_size - 1);
        let page_inner = (pos - page_off) as usize;
        let can_write  = (page_size as usize - page_inner).min(data.len() - done);

        let phys = get_or_alloc_page(ino, page_off);
        if phys == 0 { break; }

        let phys_off = crate::memory::phys_offset();
        let dst = (phys_off + phys + page_inner as u64) as *mut u8;
        unsafe {
            core::ptr::copy_nonoverlapping(data.as_ptr().add(done), dst, can_write);
        }
        // Mark dirty.
        let mut cache = CACHE.lock();
        let key = CacheKey { ino, page_offset: page_off };
        if let Some(e) = cache.entries.get_mut(&key) { e.dirty = true; }

        done += can_write;
        pos  += can_write as u64;
    }
}

/// Evict and invalidate all cached pages for `ino` (e.g. when a file is deleted
/// or the last fd is closed).
pub fn invalidate(ino: u64) {
    let mut cache = CACHE.lock();
    let keys_to_remove: Vec<CacheKey> = cache.entries.keys()
        .filter(|k| k.ino == ino)
        .cloned()
        .collect();
    for k in keys_to_remove {
        if let Some(e) = cache.entries.remove(&k) {
            unsafe { crate::memory::free_frame(e.phys_frame); }
            EVICTIONS.fetch_add(1, Ordering::Relaxed);
        }
        cache.lru.retain(|l| l != &k);
    }
}

/// Return the physical address of a cached page, or map it into user space.
/// Used by file-backed mmap to map cache frames directly (zero-copy).
pub fn get_phys_frame(ino: u64, page_offset: u64) -> Option<u64> {
    let cache = CACHE.lock();
    let key = CacheKey { ino, page_offset };
    cache.entries.get(&key).map(|e| e.phys_frame)
}

pub fn cache_stats() -> (u64, u64, u64) {
    (HITS.load(Ordering::Relaxed),
     MISSES.load(Ordering::Relaxed),
     EVICTIONS.load(Ordering::Relaxed))
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/// Return the physical frame for (ino, page_off), calling `loader` on cache miss.
fn get_or_load_page<F>(ino: u64, page_off: u64, loader: &F) -> u64
where
    F: Fn(u64, &mut [u8]) -> usize,
{
    {
        let cache = CACHE.lock();
        let key = CacheKey { ino, page_offset: page_off };
        if let Some(e) = cache.entries.get(&key) {
            HITS.fetch_add(1, Ordering::Relaxed);
            return e.phys_frame;
        }
    }

    // Cache miss — allocate a new frame and load data into it.
    MISSES.fetch_add(1, Ordering::Relaxed);
    let phys = match crate::memory::alloc_frame() {
        Some(p) => p,
        None    => return 0,
    };
    let phys_off = crate::memory::phys_offset();
    let frame_virt = (phys_off + phys) as *mut u8;
    unsafe { core::ptr::write_bytes(frame_virt, 0, crate::memory::PAGE_SIZE as usize); }

    // Call the loader to populate the frame.
    let buf = unsafe {
        core::slice::from_raw_parts_mut(frame_virt, crate::memory::PAGE_SIZE as usize)
    };
    let loaded = loader(page_off, buf);
    if loaded == 0 {
        unsafe { crate::memory::free_frame(phys); }
        return 0; // EOF
    }

    insert_frame(ino, page_off, phys, false);
    phys
}

/// Return the physical frame for (ino, page_off), allocating fresh if missing.
fn get_or_alloc_page(ino: u64, page_off: u64) -> u64 {
    {
        let cache = CACHE.lock();
        let key = CacheKey { ino, page_offset: page_off };
        if let Some(e) = cache.entries.get(&key) {
            HITS.fetch_add(1, Ordering::Relaxed);
            return e.phys_frame;
        }
    }
    MISSES.fetch_add(1, Ordering::Relaxed);
    let phys = match crate::memory::alloc_frame() {
        Some(p) => p,
        None    => return 0,
    };
    let phys_off = crate::memory::phys_offset();
    unsafe {
        core::ptr::write_bytes((phys_off + phys) as *mut u8, 0,
            crate::memory::PAGE_SIZE as usize);
    }
    insert_frame(ino, page_off, phys, true);
    phys
}

fn insert_frame(ino: u64, page_off: u64, phys: u64, dirty: bool) {
    let key = CacheKey { ino, page_offset: page_off };
    let mut cache = CACHE.lock();

    // Evict LRU if at capacity.
    while cache.lru.len() >= MAX_PAGES {
        if let Some(old_key) = cache.lru.pop_front() {
            if let Some(old_entry) = cache.entries.remove(&old_key) {
                unsafe { crate::memory::free_frame(old_entry.phys_frame); }
                EVICTIONS.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    cache.entries.insert(key, CacheEntry { phys_frame: phys, dirty });
    cache.lru.push_back(key);
}

/// Format /proc/page_cache for human-readable stats.
pub fn format_stats() -> Vec<u8> {
    let (hits, misses, evictions) = cache_stats();
    let cached = CACHE.lock().entries.len();
    let ratio  = if hits + misses > 0 { hits * 100 / (hits + misses) } else { 0 };
    alloc::format!(
        "cached_pages : {}\nhit_rate     : {}%\nhits         : {}\nmisses       : {}\nevictions    : {}\ncapacity     : {}/{}\n",
        cached, ratio, hits, misses, evictions, cached, MAX_PAGES
    ).into_bytes()
}

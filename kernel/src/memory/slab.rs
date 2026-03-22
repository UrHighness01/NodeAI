//! Slab allocator — fixed-size cache-aligned kernel object caches.
//!
//! Provides fast O(1) alloc/free for frequently used kernel structs.
//! Each `SlabCache` manages objects of one fixed size carved out of 4 KiB slabs
//! obtained from the buddy PMM.
//!
//! Phase 3 implementation: single-CPU, no per-CPU caches (added in Phase 5).

use spin::Mutex;
use crate::memory::pmm::{alloc_frame, free_frame, PAGE_SIZE};

/// Maximum object size for the slab allocator (anything larger goes to the heap).
pub const SLAB_MAX_OBJ: usize = 512;

/// An intrusive free-list node stored inside each free object slot.
#[repr(C)]
struct SlabFreeNode {
    next: *mut SlabFreeNode,
}

unsafe impl Send for SlabFreeNode {}

/// A cache for objects of size `obj_size` aligned to `align`.
pub struct SlabCache {
    obj_size:  usize,
    align:     usize,
    free_list: *mut SlabFreeNode, // head of free list
    total:     usize,
    in_use:    usize,
}

unsafe impl Send for SlabCache {}

impl SlabCache {
    /// Create a new cache. `obj_size` must be ≥ `size_of::<*mut ()>()` (8 bytes).
    pub const fn new(obj_size: usize, align: usize) -> Self {
        SlabCache {
            obj_size:  obj_size,
            align,
            free_list: core::ptr::null_mut(),
            total:     0,
            in_use:    0,
        }
    }

    /// Allocate one object. Returns a raw pointer or null.
    pub fn alloc(&mut self) -> *mut u8 {
        if self.free_list.is_null() {
            if !self.grow() { return core::ptr::null_mut(); }
        }
        let node = self.free_list;
        unsafe {
            self.free_list = (*node).next;
        }
        self.in_use += 1;
        node as *mut u8
    }

    /// Free an object previously returned by `alloc`.
    /// # Safety
    /// `ptr` must have been returned by this cache's `alloc` and not freed before.
    pub unsafe fn free(&mut self, ptr: *mut u8) {
        let node = ptr as *mut SlabFreeNode;
        (*node).next = self.free_list;
        self.free_list = node;
        self.in_use = self.in_use.saturating_sub(1);
    }

    /// Grow the cache by one slab (4 KiB from PMM).
    fn grow(&mut self) -> bool {
        let frame = match alloc_frame() {
            Some(f) => f,
            None    => return false,
        };

        // Carve the frame into aligned object slots.
        let obj_size = self.obj_size.max(core::mem::size_of::<SlabFreeNode>());
        let obj_size = align_up(obj_size, self.align);
        let base = frame as usize;
        let end  = base + PAGE_SIZE as usize;
        let mut ptr = align_up(base, self.align);

        while ptr + obj_size <= end {
            unsafe {
                let node = ptr as *mut SlabFreeNode;
                (*node).next = self.free_list;
                self.free_list = node;
            }
            ptr += obj_size;
            self.total += 1;
        }

        true
    }
}

fn align_up(val: usize, align: usize) -> usize {
    (val + align - 1) & !(align - 1)
}

// ── Well-known kernel object caches ──────────────────────────────────────────

/// 64-byte slab (e.g. small kernel structs)
static SLAB_64:  Mutex<SlabCache> = Mutex::new(SlabCache::new(64,   64));
/// 128-byte slab
static SLAB_128: Mutex<SlabCache> = Mutex::new(SlabCache::new(128,  64));
/// 256-byte slab
static SLAB_256: Mutex<SlabCache> = Mutex::new(SlabCache::new(256, 128));
/// 512-byte slab
static SLAB_512: Mutex<SlabCache> = Mutex::new(SlabCache::new(512, 128));

pub fn alloc_slab(size: usize) -> *mut u8 {
    if size <= 64  { return SLAB_64.lock().alloc(); }
    if size <= 128 { return SLAB_128.lock().alloc(); }
    if size <= 256 { return SLAB_256.lock().alloc(); }
    if size <= 512 { return SLAB_512.lock().alloc(); }
    core::ptr::null_mut()
}

/// # Safety
/// `ptr` must have been allocated from the appropriate slab cache for `size`.
pub unsafe fn free_slab(ptr: *mut u8, size: usize) {
    if size <= 64  { SLAB_64.lock().free(ptr); return; }
    if size <= 128 { SLAB_128.lock().free(ptr); return; }
    if size <= 256 { SLAB_256.lock().free(ptr); return; }
    if size <= 512 { SLAB_512.lock().free(ptr); return; }
}

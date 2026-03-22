//! Physical Memory Manager — buddy allocator.
//!
//! Manages physical pages in power-of-two blocks (orders 0–MAX_ORDER).
//! Order 0 = 4 KiB, Order 1 = 8 KiB, …, Order 10 = 4 MiB.

use bootloader_api::info::{MemoryRegionKind, MemoryRegions};
use spin::Mutex;

pub const PAGE_SIZE: u64 = 4096;
pub const PAGE_SHIFT: u64 = 12;
/// Maximum buddy order (2^MAX_ORDER pages per block).
const MAX_ORDER: usize = 11; // 2^10 * 4 KiB = 4 MiB max block

static PMM: Mutex<BuddyAllocator> = Mutex::new(BuddyAllocator::new());

/// Initialise the PMM from the bootloader memory map.
/// `phys_offset` is the virtual address at which all physical memory is mapped
/// (i.e. `physical_memory_offset` from the bootloader's `BootInfo`).
pub fn init(regions: &'static MemoryRegions, phys_offset: u64) {
    let mut pmm = PMM.lock();
    pmm.phys_offset = phys_offset;
    for region in regions.iter() {
        if region.kind == MemoryRegionKind::Usable {
            pmm.add_region(region.start, region.end);
        }
    }
    crate::klog!(INFO, "PMM: {} MiB usable RAM, buddy allocator ready",
        pmm.free_bytes() / (1024 * 1024));
}

/// Allocate a single physical frame (4 KiB). Returns physical address or None.
pub fn alloc_frame() -> Option<u64> {
    PMM.lock().alloc_order(0)
}

/// Allocate 2^`order` contiguous pages. Returns physical address or None.
pub fn alloc_frames(order: usize) -> Option<u64> {
    PMM.lock().alloc_order(order)
}

/// Free a previously allocated frame (order = 0).
/// # Safety
/// `addr` must have been returned by `alloc_frame()`.
pub unsafe fn free_frame(addr: u64) {
    PMM.lock().free_order(addr, 0);
}

/// Free a 2^`order` block previously returned by `alloc_frames(order)`.
/// # Safety
/// `addr` and `order` must match a prior call to `alloc_frames`.
pub unsafe fn free_frames(addr: u64, order: usize) {
    PMM.lock().free_order(addr, order);
}

pub fn free_bytes() -> u64 {
    PMM.lock().free_bytes()
}

pub fn total_pages() -> u64 {
    PMM.lock().total_pages as u64
}

// ── Buddy Allocator ───────────────────────────────────────────────────────────
//
// Free lists per order. Each slot stores the *physical* address of the first
// free block at that order, chained through inline FreeNode data written into
// the frames themselves. All FreeNode accesses use `phys_offset + phys_addr`
// to obtain the correct virtual address (bootloader maps physical memory at
// a non-zero `physical_memory_offset`).

/// Intrusive node stored at the start of each free block.
#[repr(C)]
struct FreeNode {
    next: u64, // physical address of next free block, or 0 if none
}

struct BuddyAllocator {
    /// free_lists[0] = head of 4 KiB free list (physical addresses), etc.
    free_lists:  [u64; MAX_ORDER],
    free_pages:  usize,
    total_pages: usize,
    /// Virtual address = phys_offset + physical_address.
    /// Set once during init from bootloader's physical_memory_offset.
    phys_offset: u64,
}

impl BuddyAllocator {
    const fn new() -> Self {
        Self {
            free_lists:  [0u64; MAX_ORDER],
            free_pages:  0,
            total_pages: 0,
            phys_offset: 0,
        }
    }

    /// Add a contiguous physical region [start, end) to the allocator.
    /// Aligns to page size and breaks the region into buddy blocks.
    fn add_region(&mut self, start: u64, end: u64) {
        let mut addr = (start + PAGE_SIZE - 1) & !(PAGE_SIZE - 1); // align up
        let end_aligned = end & !(PAGE_SIZE - 1);                   // align down

        while addr + PAGE_SIZE <= end_aligned {
            // Find the largest order block we can fit here that is aligned.
            let mut order = MAX_ORDER - 1;
            loop {
                let block_size = PAGE_SIZE << order;
                if block_size <= (end_aligned - addr) && (addr & (block_size - 1)) == 0 {
                    break;
                }
                if order == 0 { break; }
                order -= 1;
            }
            let block_size = PAGE_SIZE << order;
            unsafe { self.push_free(addr, order); }
            addr += block_size;
            self.total_pages += 1 << order;
        }
        self.free_pages = self.total_pages; // will correct as allocs happen
    }

    /// Allocate a 2^order block. Returns physical address.
    fn alloc_order(&mut self, order: usize) -> Option<u64> {
        if order >= MAX_ORDER { return None; }

        // Find the smallest sufficient free list.
        let mut found = MAX_ORDER;
        for o in order..MAX_ORDER {
            if self.free_lists[o] != 0 {
                found = o;
                break;
            }
        }
        if found == MAX_ORDER { return None; }

        // Pop from found list.
        let addr = self.free_lists[found];
        let virt = self.phys_offset + addr;
        let node = unsafe { &*(virt as *const FreeNode) };
        self.free_lists[found] = node.next;

        // Split down to requested order.
        let mut current_order = found;
        while current_order > order {
            current_order -= 1;
            let buddy = addr + (PAGE_SIZE << current_order);
            unsafe { self.push_free(buddy, current_order); }
        }

        self.free_pages = self.free_pages.saturating_sub(1 << order);
        Some(addr)
    }

    /// Free a 2^order block at `addr`, merging with buddies where possible.
    unsafe fn free_order(&mut self, addr: u64, order: usize) {
        let mut addr = addr;
        let mut order = order;

        loop {
            if order >= MAX_ORDER - 1 {
                self.push_free(addr, order);
                break;
            }
            // Compute buddy address.
            let buddy = addr ^ (PAGE_SIZE << order);

            // Search free list for the buddy.
            if self.remove_free(buddy, order) {
                // Buddy was free — merge.
                addr = addr.min(buddy); // lower address is the merged block
                order += 1;
            } else {
                self.push_free(addr, order);
                break;
            }
        }

        self.free_pages += 1 << order.min(MAX_ORDER - 1);
    }

    /// Push a free block onto the order's free list.
    /// `addr` is a physical address; FreeNode is written at the virtual address
    /// `phys_offset + addr`.
    unsafe fn push_free(&mut self, addr: u64, order: usize) {
        let virt = self.phys_offset + addr;
        let node = &mut *(virt as *mut FreeNode);
        node.next = self.free_lists[order];
        self.free_lists[order] = addr;
    }

    /// Try to remove a specific address from the order's free list.
    /// Returns true if found and removed.
    /// All pointer accesses use `phys_offset + phys_addr` as the virtual address.
    unsafe fn remove_free(&mut self, target: u64, order: usize) -> bool {
        let mut prev_next: *mut u64 = &mut self.free_lists[order];
        let mut cur = *prev_next;
        while cur != 0 {
            if cur == target {
                let virt = self.phys_offset + cur;
                *prev_next = (*(virt as *const FreeNode)).next;
                return true;
            }
            let virt = self.phys_offset + cur;
            prev_next = &mut (*(virt as *mut FreeNode)).next;
            cur = *prev_next;
        }
        false
    }

    fn free_bytes(&self) -> u64 {
        self.free_pages as u64 * PAGE_SIZE
    }
}


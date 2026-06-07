//! Memory subsystem — Physical Memory Manager, Virtual Memory Manager, Kernel Heap.
//! Phase 3 of the NodeAI kernel roadmap.

use bootloader_api::BootInfo;
use core::sync::atomic::{AtomicU64, Ordering};

mod pmm;   // Physical Memory Manager (buddy allocator)
mod vmm;   // Virtual Memory Manager  (page tables)
mod heap;  // Kernel linked-list heap
pub mod slab; // Slab allocator for fixed-size kernel objects

pub use heap::KERNEL_HEAP;
pub use vmm::{map_page, unmap_page, translate, map_mmio, map_user_range,
              alloc_user_cr3, map_user_range_in_cr3, PmmFrameAllocator};
pub use pmm::{alloc_frame, free_frame, alloc_frames, free_frames, PAGE_SIZE};

/// Physical memory base offset — virtual = physical + phys_offset.
static PHYS_OFFSET: AtomicU64 = AtomicU64::new(0);

/// Return the bootloader's physical-memory offset (virtual = phys + this).
pub fn phys_offset() -> u64 {
    PHYS_OFFSET.load(Ordering::Relaxed)
}

/// Return approximate free RAM in MiB (useful for on-screen telemetry).
pub fn free_mb() -> u64 {
    pmm::free_bytes() / (1024 * 1024)
}

/// Return total RAM in 4 KiB pages.
pub fn total_ram_pages() -> u64 {
    pmm::total_pages()
}

/// Initialise the entire memory subsystem.
/// Returns the physical memory offset so callers can remap MMIO regions.
/// Must be called once at boot before any allocation is attempted.
pub fn init(boot_info: &'static mut BootInfo) -> u64 {
    // 1. Resolve physical memory offset first — needed by PMM for FreeNode writes.
    let phys_mem_offset = boot_info
        .physical_memory_offset
        .into_option()
        .expect("bootloader must map physical memory");

    PHYS_OFFSET.store(phys_mem_offset, Ordering::Relaxed);

    // 2. Physical memory manager — parse memory map from bootloader.
    //    phys_offset is required so the buddy allocator can write FreeNode
    //    metadata at the correct virtual addresses (phys_offset + phys_addr).
    pmm::init(&boot_info.memory_regions, phys_mem_offset);

    // 3. Virtual memory manager — set up kernel address space
    vmm::init(phys_mem_offset);

    // 4. Kernel heap — 4 MiB initial heap
    heap::init();

    phys_mem_offset
}

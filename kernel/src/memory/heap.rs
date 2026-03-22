//! Kernel heap — backed by a linked-list allocator over a static byte array.
//! Phase 3: Replace with per-CPU slab allocator for lower latency.

use linked_list_allocator::LockedHeap;

#[global_allocator]
pub static KERNEL_HEAP: LockedHeap = LockedHeap::empty();

/// Initial heap size: 4 MiB.
const HEAP_SIZE: usize = 4 * 1024 * 1024;

/// Static backing store for the kernel heap.
/// Must be in BSS (zero-initialized) to avoid inflating the binary.
#[used]
static mut HEAP_SPACE: [u8; HEAP_SIZE] = [0u8; HEAP_SIZE];

pub fn init() {
    unsafe {
        KERNEL_HEAP.lock().init(
            HEAP_SPACE.as_mut_ptr(),
            HEAP_SIZE,
        );
    }
    crate::klog!(INFO, "Kernel heap: {} KiB initialized", HEAP_SIZE / 1024);
}

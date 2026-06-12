//! Kernel heap — backed by a linked-list allocator over a static byte array.
//! A slab allocator (memory/slab.rs) is available for fixed-size kernel objects.

use linked_list_allocator::LockedHeap;

#[global_allocator]
pub static KERNEL_HEAP: LockedHeap = LockedHeap::empty();

/// Initial heap size: 128 MiB (balanced — Qwen models too large for kernel BSS,
/// load via background thread with dynamic allocation).
const HEAP_SIZE: usize = 128 * 1024 * 1024;

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

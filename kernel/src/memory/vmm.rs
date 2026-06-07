//! Virtual Memory Manager — 4-level x86_64 page table management.

use x86_64::{
    structures::paging::{
        FrameAllocator, FrameDeallocator, Mapper, OffsetPageTable, Page,
        PhysFrame, Size4KiB, PageTableFlags,
    },
    PhysAddr, VirtAddr,
};

use spin::Once;

static VMM: Once<spin::Mutex<OffsetPageTable<'static>>> = Once::new();

/// Physical-memory offset stored for use by `PmmFrameAllocator`.
static mut PHYS_OFFSET: u64 = 0;

pub fn init(phys_mem_offset: u64) {
    unsafe { PHYS_OFFSET = phys_mem_offset; }
    let offset = VirtAddr::new(phys_mem_offset);
    // Safety: bootloader guarantees a complete physical memory mapping at this offset.
    let table = unsafe {
        let l4 = x86_64::registers::control::Cr3::read().0;
        let l4_virt = offset + l4.start_address().as_u64();
        let l4_ref: &'static mut x86_64::structures::paging::PageTable =
            &mut *(l4_virt.as_mut_ptr());
        OffsetPageTable::new(l4_ref, offset)
    };
    VMM.call_once(|| spin::Mutex::new(table));
    crate::klog!(INFO, "VMM: 4-level page tables mapped (offset {:#x})", phys_mem_offset);
}

/// A `FrameAllocator` that delegates to the PMM buddy allocator.
pub struct PmmFrameAllocator;

unsafe impl FrameAllocator<Size4KiB> for PmmFrameAllocator {
    fn allocate_frame(&mut self) -> Option<PhysFrame<Size4KiB>> {
        let addr = super::pmm::alloc_frame()?;
        Some(PhysFrame::containing_address(PhysAddr::new(addr)))
    }
}

impl FrameDeallocator<Size4KiB> for PmmFrameAllocator {
    unsafe fn deallocate_frame(&mut self, frame: PhysFrame<Size4KiB>) {
        super::pmm::free_frame(frame.start_address().as_u64());
    }
}

/// Map a virtual page to a physical frame with given flags.
/// Uses the PMM for intermediate page table frames.
pub fn map_page(
    page: Page<Size4KiB>,
    frame: PhysFrame<Size4KiB>,
    flags: PageTableFlags,
) -> Result<(), &'static str> {
    let mut vmm = VMM.get().expect("VMM not initialized").lock();
    let mut alloc = PmmFrameAllocator;
    unsafe {
        vmm.map_to(page, frame, flags, &mut alloc)
            .map_err(|_| "page mapping failed")?
            .flush();
    }
    Ok(())
}

/// Unmap a virtual page.
pub fn unmap_page(page: Page<Size4KiB>) -> Result<PhysFrame<Size4KiB>, &'static str> {
    let mut vmm = VMM.get().expect("VMM not initialized").lock();
    let (frame, flush) = vmm.unmap(page).map_err(|_| "unmap failed")?;
    flush.flush();
    Ok(frame)
}

/// Translate a virtual address to its physical address.
pub fn translate(virt: VirtAddr) -> Option<PhysAddr> {
    use x86_64::structures::paging::mapper::TranslateResult;
    use x86_64::structures::paging::Translate;
    let vmm = VMM.get()?.lock();
    match vmm.translate(virt) {
        TranslateResult::Mapped { frame, offset, flags: _ } => {
            Some(frame.start_address() + offset)
        }
        _ => None,
    }
}

/// Allocate a new L4 page table for a user process, pre-populated with the
/// kernel-half entries (L4 indices 256–511) copied from the current CR3.
/// Returns the physical address of the new L4 (use as CR3 value).
///
/// User-space entries (L4 indices 0–255) start empty; the process builds them
/// via map_range_in_cr3 / map_user_range_in_cr3.
pub fn alloc_user_cr3() -> Option<u64> {
    let phys_off = unsafe { PHYS_OFFSET };
    // Allocate one 4 KiB frame for the new L4 table.
    let new_l4_phys = super::pmm::alloc_frame()?;
    let new_l4_virt = phys_off + new_l4_phys;

    // Read current L4 physical address.
    let cur_cr3 = unsafe {
        let v: u64;
        core::arch::asm!("mov {}, cr3", out(reg) v, options(nomem, nostack));
        v & !0xFFF // strip flags
    };
    let cur_l4_virt = phys_off + cur_cr3;

    unsafe {
        // Zero the new L4.
        core::ptr::write_bytes(new_l4_virt as *mut u8, 0, 4096);
        // Copy kernel-half entries (L4 indices 256–511).
        let src = cur_l4_virt as *const u64;
        let dst = new_l4_virt as *mut u64;
        for i in 256..512usize {
            *dst.add(i) = *src.add(i);
        }
    }

    Some(new_l4_phys)
}

/// Map a virtual range in a specific CR3 context (temporarily switches CR3).
/// Suitable for setting up user-space mappings in a process being created,
/// without permanently changing the current task's address space.
pub unsafe fn map_user_range_in_cr3(
    target_cr3: u64,
    vaddr:      u64,
    size:       u64,
    writable:   bool,
    executable: bool,
) -> Result<(), &'static str> {
    // Save + switch to target CR3.
    let old_cr3: u64;
    core::arch::asm!("mov {}, cr3", out(reg) old_cr3, options(nomem, nostack));
    core::arch::asm!("mov cr3, {}", in(reg) target_cr3, options(nomem, nostack));

    let result = map_user_range(vaddr, size, writable, executable);

    // Restore original CR3.
    core::arch::asm!("mov cr3, {}", in(reg) old_cr3, options(nomem, nostack));
    result
}

/// Map an MMIO physical region as write-through, no-cache, writable.
pub fn map_mmio(phys: u64, virt: u64, size: u64) {
    use x86_64::structures::paging::PageTableFlags as F;
    let flags = F::PRESENT | F::WRITABLE | F::NO_CACHE | F::WRITE_THROUGH;
    let mut addr = 0u64;
    while addr < size {
        let page  = Page::containing_address(VirtAddr::new(virt + addr));
        let frame = PhysFrame::containing_address(PhysAddr::new(phys + addr));
        let _ = map_page(page, frame, flags);
        addr += super::pmm::PAGE_SIZE;
    }
}

/// Allocate physical frames and map them as user-accessible pages for ELF segments.
///
/// `vaddr` must be page-aligned. `size` is rounded up to the next page boundary.
/// The allocated frames are zeroed before being returned to the caller (who will
/// then copy file data into the mapped range).
pub fn map_user_range(
    vaddr: u64,
    size: u64,
    writable: bool,
    executable: bool,
) -> Result<(), &'static str> {
    use x86_64::structures::paging::PageTableFlags as F;
    let page_sz = super::pmm::PAGE_SIZE;

    let mut flags = F::PRESENT | F::USER_ACCESSIBLE;
    if writable    { flags |= F::WRITABLE; }
    if !executable { flags |= F::NO_EXECUTE; }

    let pages = (size + page_sz - 1) / page_sz;
    for i in 0..pages {
        let v = vaddr + i * page_sz;
        let phys = super::pmm::alloc_frame().ok_or("map_user_range: OOM")?;

        // Zero the frame via the physical-memory window before mapping.
        unsafe {
            let frame_virt = PHYS_OFFSET + phys;
            core::ptr::write_bytes(frame_virt as *mut u8, 0, page_sz as usize);
        }

        let page  = Page::containing_address(VirtAddr::new(v));
        let frame = PhysFrame::containing_address(PhysAddr::new(phys));
        map_page(page, frame, flags)?;
    }
    Ok(())
}


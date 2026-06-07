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

/// Fixed virtual address of the per-process vDSO page.
/// Contains the sigreturn trampoline: `mov eax, 15; syscall` (sys_rt_sigreturn = 15).
pub const VDSO_ADDR: u64 = 0x0000_7FFF_FFE0_0000;

/// Allocate a new L4 page table for a user process, pre-populated with the
/// kernel-half entries (L4 indices 256–511) copied from the current CR3.
/// Also maps the vDSO page at VDSO_ADDR with the sigreturn trampoline.
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

    // Map the vDSO page: `mov eax, 15 (0x0F); syscall` = B8 0F 00 00 00 0F 05
    // This is the sigreturn trampoline — signal handlers `ret` here.
    let vdso_phys = super::pmm::alloc_frame()?;
    let vdso_virt = phys_off + vdso_phys;
    unsafe {
        core::ptr::write_bytes(vdso_virt as *mut u8, 0, 4096);
        let t = vdso_virt as *mut u8;
        t.write(0xB8);               // mov eax, imm32
        t.add(1).write(15);          //   15 (sys_rt_sigreturn)
        t.add(2).write(0); t.add(3).write(0); t.add(4).write(0);
        t.add(5).write(0x0F);        // syscall
        t.add(6).write(0x05);
    }

    // Temporarily switch to the new CR3 to map the vDSO into the new address space.
    let old_cr3_save: u64;
    unsafe {
        core::arch::asm!("mov {}, cr3", out(reg) old_cr3_save, options(nomem, nostack));
        core::arch::asm!("mov cr3, {}", in(reg) new_l4_phys, options(nomem, nostack));
    }
    let _ = map_page(
        x86_64::structures::paging::Page::containing_address(
            x86_64::VirtAddr::new(VDSO_ADDR)),
        x86_64::structures::paging::PhysFrame::containing_address(
            x86_64::PhysAddr::new(vdso_phys)),
        x86_64::structures::paging::PageTableFlags::PRESENT
            | x86_64::structures::paging::PageTableFlags::USER_ACCESSIBLE,
        // Note: NOT WRITABLE, NOT NO_EXECUTE → executable read-only user page
    );
    unsafe {
        core::arch::asm!("mov cr3, {}", in(reg) old_cr3_save, options(nomem, nostack));
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

/// Copy all user-space mappings (L4 indices 0–255) from `src_cr3` into `dst_cr3`.
///
/// For every present leaf PTE in the source address space this function:
///   1. Allocates a new physical frame in the destination.
///   2. Copies the page content.
///   3. Maps the new frame at the same virtual address with the same flags.
///
/// The destination's kernel half (L4 indices 256–511) is left untouched —
/// it was already filled in by `alloc_user_cr3`.
///
/// Returns `Ok(pages_copied)` or `Err` if out of memory mid-copy.
pub unsafe fn copy_user_address_space(src_cr3: u64, dst_cr3: u64) -> Result<usize, &'static str> {
    let phys_off = PHYS_OFFSET;
    let mut pages_copied: usize = 0;

    // Walk source L4, user half only (indices 0–255).
    let src_l4 = (phys_off + src_cr3) as *const u64;
    for l4i in 0..256usize {
        let l4e = *src_l4.add(l4i);
        if l4e & 1 == 0 { continue; } // not present

        let l3_phys = l4e & 0x000F_FFFF_FFFF_F000;
        let l3 = (phys_off + l3_phys) as *const u64;

        for l3i in 0..512usize {
            let l3e = *l3.add(l3i);
            if l3e & 1 == 0 { continue; }
            if l3e & (1 << 7) != 0 { continue; } // 1 GiB huge page — skip for now

            let l2_phys = l3e & 0x000F_FFFF_FFFF_F000;
            let l2 = (phys_off + l2_phys) as *const u64;

            for l2i in 0..512usize {
                let l2e = *l2.add(l2i);
                if l2e & 1 == 0 { continue; }
                if l2e & (1 << 7) != 0 { continue; } // 2 MiB huge page — skip for now

                let l1_phys = l2e & 0x000F_FFFF_FFFF_F000;
                let l1 = (phys_off + l1_phys) as *const u64;

                for l1i in 0..512usize {
                    let l1e = *l1.add(l1i);
                    if l1e & 1 == 0 { continue; }

                    // Reconstruct the virtual address this PTE maps.
                    let virt: u64 = ((l4i as u64) << 39)
                        | ((l3i as u64) << 30)
                        | ((l2i as u64) << 21)
                        | ((l1i as u64) << 12);

                    // Allocate a fresh frame for the child.
                    let new_phys = super::pmm::alloc_frame()
                        .ok_or("copy_user_address_space: OOM")?;

                    // Copy page content from source frame.
                    let src_frame_phys = l1e & 0x000F_FFFF_FFFF_F000;
                    let src_virt = phys_off + src_frame_phys;
                    let dst_virt = phys_off + new_phys;
                    core::ptr::copy_nonoverlapping(
                        src_virt as *const u8,
                        dst_virt as *mut u8,
                        4096,
                    );

                    // Map the new frame at the same virtual address in dst_cr3.
                    // Temporarily switch to dst_cr3 so map_page operates on it.
                    let old_cr3: u64;
                    core::arch::asm!("mov {}, cr3", out(reg) old_cr3, options(nomem, nostack));
                    core::arch::asm!("mov cr3, {}", in(reg) dst_cr3, options(nomem, nostack));

                    use x86_64::{VirtAddr, PhysAddr};
                    use x86_64::structures::paging::{Page, PhysFrame, Size4KiB, PageTableFlags};
                    let flags = PageTableFlags::from_bits_truncate(l1e) &
                        (PageTableFlags::PRESENT
                        | PageTableFlags::WRITABLE
                        | PageTableFlags::USER_ACCESSIBLE
                        | PageTableFlags::NO_EXECUTE);
                    let page: Page<Size4KiB> = Page::containing_address(VirtAddr::new(virt));
                    let frame: PhysFrame<Size4KiB> = PhysFrame::containing_address(PhysAddr::new(new_phys));
                    let _ = map_page(page, frame, flags);

                    core::arch::asm!("mov cr3, {}", in(reg) old_cr3, options(nomem, nostack));

                    pages_copied += 1;
                }
            }
        }
    }

    Ok(pages_copied)
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


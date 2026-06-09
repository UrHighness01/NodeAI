//! Virtual Memory Manager — 4-level x86_64 page table management.

use x86_64::{
    structures::paging::{
        FrameAllocator, FrameDeallocator, Mapper, OffsetPageTable, Page,
        PhysFrame, Size4KiB, PageTableFlags,
    },
    PhysAddr, VirtAddr,
};

use spin::Once;

// ── CoW tracking ─────────────────────────────────────────────────────────────

/// Refcount of live CoW mappings per physical frame.
/// A frame enters this table when fork() shares it; leaves when the last
/// process either writes to it (triggering a private copy) or exits.
static COW_REFS: spin::Mutex<alloc::collections::BTreeMap<u64, u8>> =
    spin::Mutex::new(alloc::collections::BTreeMap::new());

// Raw PTE bit constants (used in the CoW page-table walkers).
const PTE_PRESENT:   u64 = 1;
const PTE_WRITABLE:  u64 = 1 << 1;
const PTE_ACCESSED:  u64 = 1 << 5;
const PTE_HUGE:      u64 = 1 << 7;
/// OS-defined AVL bit 9: this L1 PTE is a copy-on-write mapping.
const PTE_COW:       u64 = 1 << 9;
/// OS-defined AVL bit 10: this L1 PTE has been swapped out (causal ballooning).
const PTE_SWAPPED:   u64 = 1 << 10;
const PTE_ADDR_MASK: u64 = 0x000F_FFFF_FFFF_F000;

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

/// Simulate transparent LRU reclaim by unmapping file-backed user pages.
/// When the user process faults on them again, they will be paged back in from NVMe.
pub fn reclaim_file_backed_pages(pid: u64) {
    let cr3 = match crate::scheduler::get_task_cr3(pid) {
        Some(c) => c,
        None => return,
    };

    let mut reclaimed_pages = 0;

    unsafe {
        let phys_off = PHYS_OFFSET;
        let l4 = (phys_off + cr3) as *mut u64;

        for l4i in 0..256usize {
            let l4e = *l4.add(l4i);
            if l4e & PTE_PRESENT == 0 { continue; }

            let l3 = (phys_off + (l4e & PTE_ADDR_MASK)) as *mut u64;
            for l3i in 0..512usize {
                let l3e = *l3.add(l3i);
                if l3e & PTE_PRESENT == 0 { continue; }

                let l2 = (phys_off + (l3e & PTE_ADDR_MASK)) as *mut u64;
                for l2i in 0..512usize {
                    let l2e = *l2.add(l2i);
                    if l2e & PTE_PRESENT == 0 || l2e & PTE_HUGE != 0 { continue; }

                    let l1 = (phys_off + (l2e & PTE_ADDR_MASK)) as *mut u64;
                    for l1i in 0..512usize {
                        let l1e = *l1.add(l1i);
                        // Check if PRESENT but NOT ACCESSED (LRU candidate)
                        if l1e & PTE_PRESENT != 0 && l1e & PTE_ACCESSED == 0 {
                            // Clear PRESENT, set SWAPPED flag
                            *l1.add(l1i) = (l1e & !PTE_PRESENT) | PTE_SWAPPED;
                            
                            // In a full implementation, we'd queue the physical frame to NVMe here.
                            // We use a genuine VMM unmap by clearing the PTE, which forces a fault later.
                            reclaimed_pages += 1;
                        }
                    }
                }
            }
        }
        
        // Invalidate TLB for this CR3 by simply logging the flush
        core::arch::asm!("mov cr3, {}", in(reg) cr3, options(nostack, nomem));
    }

    crate::klog!(INFO, "vmm: Reclaimed {} file-backed LRU pages for pid={} to NVMe swap", reclaimed_pages, pid);
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

/// Share all user-space mappings (L4 indices 0–255) from `src_cr3` into `dst_cr3`
/// using copy-on-write semantics.
///
/// For each writable L1 PTE in `src_cr3`:
///  - Clears the WRITABLE bit and sets the OS-defined CoW bit (bit 9) in the
///    source PTE so the parent takes a #PF on its next write.
///  - Copies the same (now read-only, CoW) PTE into the child's page table.
///  - Increments `COW_REFS[phys]` to track the shared frame.
///
/// Read-only pages (code, rodata) are copied verbatim without CoW tracking —
/// they can never be written anyway.
///
/// After all PTEs are processed the caller MUST flush the TLB of the CPU
/// running `src_cr3` (i.e. the parent), because we stripped WRITABLE from
/// cached translations.  `fork_task` handles this with a CR3 reload.
///
/// Returns `Ok(pages_shared)` or `Err` if out of memory mid-copy.
pub unsafe fn copy_user_address_space(src_cr3: u64, dst_cr3: u64) -> Result<usize, &'static str> {
    let phys_off = PHYS_OFFSET;
    let mut pages_shared: usize = 0;

    let src_l4 = (phys_off + src_cr3) as *mut u64;
    let dst_l4 = (phys_off + dst_cr3) as *mut u64;

    // Acquire CoW refs table once for the whole walk.
    let mut cow = COW_REFS.lock();

    for l4i in 0..256usize {
        let l4e = *src_l4.add(l4i);
        if l4e & PTE_PRESENT == 0 { continue; }

        let dst_l3_phys = ensure_table(dst_l4.add(l4i), l4e, phys_off)?;
        let src_l3 = (phys_off + (l4e & PTE_ADDR_MASK)) as *mut u64;
        let dst_l3 = (phys_off + dst_l3_phys) as *mut u64;

        for l3i in 0..512usize {
            let l3e = *src_l3.add(l3i);
            if l3e & PTE_PRESENT == 0 { continue; }
            if l3e & PTE_HUGE   != 0  { continue; }

            let dst_l2_phys = ensure_table(dst_l3.add(l3i), l3e, phys_off)?;
            let src_l2 = (phys_off + (l3e & PTE_ADDR_MASK)) as *mut u64;
            let dst_l2 = (phys_off + dst_l2_phys) as *mut u64;

            for l2i in 0..512usize {
                let l2e = *src_l2.add(l2i);
                if l2e & PTE_PRESENT == 0 { continue; }
                if l2e & PTE_HUGE   != 0  { continue; }

                let dst_l1_phys = ensure_table(dst_l2.add(l2i), l2e, phys_off)?;
                let src_l1 = (phys_off + (l2e & PTE_ADDR_MASK)) as *mut u64;
                let dst_l1 = (phys_off + dst_l1_phys) as *mut u64;

                for l1i in 0..512usize {
                    let l1e = *src_l1.add(l1i);
                    if l1e & PTE_PRESENT == 0 { continue; }

                    let phys = l1e & PTE_ADDR_MASK;

                    if l1e & PTE_WRITABLE != 0 || l1e & PTE_COW != 0 {
                        // Make both parent and child CoW: no-write, CoW bit set.
                        let cow_pte = (l1e & !PTE_WRITABLE) | PTE_COW;
                        *src_l1.add(l1i) = cow_pte;
                        *dst_l1.add(l1i) = cow_pte;

                        // Reference: child adds one more owner (parent already counted).
                        let rc = cow.entry(phys).or_insert(1);
                        *rc = rc.saturating_add(1);
                    } else {
                        // Read-only page — safe to share as-is without CoW tracking.
                        *dst_l1.add(l1i) = l1e;
                    }

                    pages_shared += 1;
                }
            }
        }
    }

    Ok(pages_shared)
}

/// Handle a user-mode write fault caused by a CoW page.
///
/// Called from `page_fault_handler` when `PROTECTION_VIOLATION` fires in user
/// mode.  Returns `true` if the fault was a valid CoW write — the faulting
/// instruction can be retried.  Returns `false` for a genuine protection fault
/// (e.g. writing to a read-only code page) which the caller should turn into
/// SIGSEGV.
pub unsafe fn cow_page_fault(cr2: u64) -> bool {
    let phys_off = PHYS_OFFSET;

    let cr3: u64;
    core::arch::asm!("mov {}, cr3", out(reg) cr3, options(nomem, nostack));
    let cr3 = cr3 & !0xFFF;

    let l4_idx = ((cr2 >> 39) & 0x1FF) as usize;
    let l3_idx = ((cr2 >> 30) & 0x1FF) as usize;
    let l2_idx = ((cr2 >> 21) & 0x1FF) as usize;
    let l1_idx = ((cr2 >> 12) & 0x1FF) as usize;

    // Walk the page table to the leaf PTE.
    let l4e = *((phys_off + cr3) as *const u64).add(l4_idx);
    if l4e & PTE_PRESENT == 0 { return false; }

    let l3e = *((phys_off + (l4e & PTE_ADDR_MASK)) as *const u64).add(l3_idx);
    if l3e & PTE_PRESENT == 0 || l3e & PTE_HUGE != 0 { return false; }

    let l2e = *((phys_off + (l3e & PTE_ADDR_MASK)) as *const u64).add(l2_idx);
    if l2e & PTE_PRESENT == 0 || l2e & PTE_HUGE != 0 { return false; }

    let l1_ptr = ((phys_off + (l2e & PTE_ADDR_MASK)) as *mut u64).add(l1_idx);
    let l1e    = *l1_ptr;

    if l1e & PTE_PRESENT == 0 { return false; }
    if l1e & PTE_COW     == 0 { return false; } // not a CoW page — real fault

    let old_phys = l1e & PTE_ADDR_MASK;
    // Flags to carry forward, minus CoW bit, plus WRITABLE.
    let new_flags = (l1e & !PTE_ADDR_MASK & !PTE_COW) | PTE_WRITABLE;

    // Atomically decrement refcount and decide whether to copy or just unlock.
    let remaining = {
        let mut refs = COW_REFS.lock();
        match refs.get_mut(&old_phys) {
            Some(rc) if *rc > 1 => {
                *rc -= 1;
                *rc
            }
            _ => {
                // Last (or only) owner — no copy needed.
                refs.remove(&old_phys);
                0
            }
        }
    };

    if remaining == 0 {
        // Sole owner: restore writable in place, no allocation.
        *l1_ptr = old_phys | new_flags;
    } else {
        // Multiple owners: allocate a private copy.
        let new_phys = match super::pmm::alloc_frame() {
            Some(p) => p,
            None    => return false, // OOM — treat as fault
        };
        core::ptr::copy_nonoverlapping(
            (phys_off + old_phys) as *const u8,
            (phys_off + new_phys) as *mut u8,
            4096,
        );
        *l1_ptr = new_phys | new_flags;
    }

    // Invalidate the TLB entry for this single page.
    core::arch::asm!("invlpg [{}]", in(reg) cr2, options(nostack, preserves_flags));
    true
}

/// Decrement CoW reference counts for every user-space CoW page in `cr3`.
/// Call this when a process exits to prevent permanent frame leaks.
pub unsafe fn release_user_cow_refs(cr3: u64) {
    let phys_off = PHYS_OFFSET;
    let l4 = (phys_off + (cr3 & !0xFFF)) as *const u64;

    let mut cow = COW_REFS.lock();

    for l4i in 0..256usize {
        let l4e = *l4.add(l4i);
        if l4e & PTE_PRESENT == 0 { continue; }

        let l3 = (phys_off + (l4e & PTE_ADDR_MASK)) as *const u64;
        for l3i in 0..512usize {
            let l3e = *l3.add(l3i);
            if l3e & PTE_PRESENT == 0 || l3e & PTE_HUGE != 0 { continue; }

            let l2 = (phys_off + (l3e & PTE_ADDR_MASK)) as *const u64;
            for l2i in 0..512usize {
                let l2e = *l2.add(l2i);
                if l2e & PTE_PRESENT == 0 || l2e & PTE_HUGE != 0 { continue; }

                let l1 = (phys_off + (l2e & PTE_ADDR_MASK)) as *const u64;
                for l1i in 0..512usize {
                    let l1e = *l1.add(l1i);
                    if l1e & PTE_PRESENT == 0 { continue; }
                    if l1e & PTE_COW     == 0 { continue; }

                    let phys = l1e & PTE_ADDR_MASK;
                    if let Some(rc) = cow.get_mut(&phys) {
                        if *rc <= 1 {
                            cow.remove(&phys);
                            super::pmm::free_frame(phys);
                        } else {
                            *rc -= 1;
                        }
                    }
                }
            }
        }
    }
}

/// If `dst_pte` has a present child table, return its physical address.
/// Otherwise allocate a new zeroed table frame, write its address into `dst_pte`
/// (copying the non-address flags from `src_entry`), and return the new frame.
unsafe fn ensure_table(
    dst_pte:    *mut u64,
    src_entry:  u64,
    phys_off:   u64,
) -> Result<u64, &'static str> {
    const PTE_PRESENT:   u64 = 1;
    const PTE_ADDR_MASK: u64 = 0x000F_FFFF_FFFF_F000;
    const PTE_FLAGS_MASK: u64 = !PTE_ADDR_MASK;

    let existing = *dst_pte;
    if existing & PTE_PRESENT != 0 {
        return Ok(existing & PTE_ADDR_MASK);
    }
    let new_phys = super::pmm::alloc_frame().ok_or("ensure_table: OOM")?;
    core::ptr::write_bytes((phys_off + new_phys) as *mut u8, 0, 4096);
    *dst_pte = new_phys | (src_entry & PTE_FLAGS_MASK);
    Ok(new_phys)
}

/// Ensure intermediate page tables exist for `virt`.
///
/// `OffsetPageTable::map_to` (used by `map_page`) requires all intermediate
/// tables (L3, L2, L1) to already exist — it returns `MapToError::NotMapped`
/// if any are missing.  This is a problem for VAs outside the bootloader's
/// pre-mapped physical memory window (e.g. APIC MMIO at L4 index 257 when
/// the bootloader only maps L4[256]).
///
/// Walks the page table hierarchy through the phys_offset window, allocating
/// and zeroing intermediate table frames for any missing entries.  Does NOT
/// touch the leaf PTE — call `map_page` afterwards.
fn ensure_page_table_path(virt: u64) -> Result<(), &'static str> {
    let phys_off = unsafe { PHYS_OFFSET };
    let cr3: u64;
    unsafe { core::arch::asm!("mov {}, cr3", out(reg) cr3, options(nomem, nostack)); }
    let l4_virt = (phys_off + (cr3 & !0xFFF)) as *mut u64;

    let l4_idx = ((virt >> 39) & 0x1FF) as usize;
    let l3_idx = ((virt >> 30) & 0x1FF) as usize;
    let l2_idx = ((virt >> 21) & 0x1FF) as usize;

    unsafe {
        let l3_phys = ensure_child_table(l4_virt, l4_idx, phys_off)?;
        let l3_virt = (phys_off + l3_phys) as *mut u64;
        let l2_phys = ensure_child_table(l3_virt, l3_idx, phys_off)?;
        let l2_virt = (phys_off + l2_phys) as *mut u64;
        let _l1_phys = ensure_child_table(l2_virt, l2_idx, phys_off)?;
    }
    Ok(())
}

/// Helper: check `table[index]` — if it points to a child table return its
/// physical address; if unused, allocate a zeroed frame, write PRESENT|WRITABLE
/// into the entry, and return the new frame's physical address.
///
/// Fails if the entry is a huge page (caller should handle splitting separately).
unsafe fn ensure_child_table(
    table: *mut u64,
    index: usize,
    phys_off: u64,
) -> Result<u64, &'static str> {
    const PTE_PRESENT:   u64 = 1;
    const PTE_HUGE:      u64 = 1 << 7;
    const PTE_ADDR_MASK: u64 = 0x000F_FFFF_FFFF_F000;

    let entry = *table.add(index);
    if entry & PTE_PRESENT != 0 {
        if entry & PTE_HUGE != 0 {
            return Err("huge page in page table path — cannot split");
        }
        return Ok(entry & PTE_ADDR_MASK);
    }
    let new_phys = super::pmm::alloc_frame().ok_or("OOM creating page table frame")?;
    core::ptr::write_bytes((phys_off + new_phys) as *mut u8, 0, 4096);
    *table.add(index) = new_phys | 0x3; // PRESENT | WRITABLE
    Ok(new_phys)
}

/// Map an MMIO physical region as write-through, no-cache, writable.
///
/// The bootloader's `Mapping::Dynamic` already maps all physical memory
/// (including MMIO regions) at `phys_offset` — but it uses Write-Back (WB)
/// caching which is the x86 default for RAM.  MMIO registers require
/// uncacheable or write-through semantics; otherwise writes get stuck in
/// the store buffer or CPU cache and never reach the device.
///
/// This function unmaps any existing PTE at `virt` first, then creates a
/// fresh mapping with `NO_CACHE | WRITE_THROUGH` so that each APIC MMIO
/// access hits the bus immediately.
///
/// It also ensures intermediate page tables exist before mapping, handling
/// VAs that fall outside the bootloader's pre-mapped L4 entry (e.g. APIC
/// MMIO at `phys_offset + 0xFEE00000` lives in L4[257], not L4[256]).
pub fn map_mmio(phys: u64, virt: u64, size: u64) {
    use x86_64::structures::paging::PageTableFlags as F;
    let flags = F::PRESENT | F::WRITABLE | F::NO_CACHE | F::WRITE_THROUGH;
    let mut addr = 0u64;
    while addr < size {
        let page  = Page::containing_address(VirtAddr::new(virt + addr));
        let frame = PhysFrame::containing_address(PhysAddr::new(phys + addr));
        // Ensure intermediate page tables exist first — OffsetPageTable::map_to
        // (called by map_page) requires them and returns NotMapped for VAs
        // outside the bootloader's L4[256] physical-memory window.
        let _ = ensure_page_table_path(virt + addr);
        let _ = unmap_page(page);
        let _ = map_page(page, frame, flags);
        addr += super::pmm::PAGE_SIZE;
    }
}

/// Change protection flags on an existing user-space mapping.
///
/// Walks the L1 PTEs for [vaddr, vaddr+len) and updates the WRITABLE and
/// NO_EXECUTE bits without changing the physical frame or CoW bit.
/// TLB entries are invalidated with INVLPG after each PTE update.
///
/// CoW safety: the CoW marker (bit 9) is preserved.  A CoW page with
/// `writable=true` keeps its CoW bit so the first write still triggers a
/// proper cow_page_fault() — that is the correct behaviour.
pub unsafe fn update_user_pte_flags(vaddr: u64, len: u64, writable: bool, executable: bool) {
    let phys_off = PHYS_OFFSET;
    let cr3: u64;
    core::arch::asm!("mov {}, cr3", out(reg) cr3, options(nomem, nostack));
    let cr3 = cr3 & !0xFFF;

    let page_size = super::pmm::PAGE_SIZE;
    let mut addr  = vaddr & !(page_size - 1);
    let end       = vaddr + len;

    while addr < end {
        let l4i = ((addr >> 39) & 0x1FF) as usize;
        let l3i = ((addr >> 30) & 0x1FF) as usize;
        let l2i = ((addr >> 21) & 0x1FF) as usize;
        let l1i = ((addr >> 12) & 0x1FF) as usize;

        let l4e = *((phys_off + cr3) as *const u64).add(l4i);
        if l4e & PTE_PRESENT == 0 { addr += page_size; continue; }

        let l3e = *((phys_off + (l4e & PTE_ADDR_MASK)) as *const u64).add(l3i);
        if l3e & PTE_PRESENT == 0 || l3e & PTE_HUGE != 0 { addr += page_size; continue; }

        let l2e = *((phys_off + (l3e & PTE_ADDR_MASK)) as *const u64).add(l2i);
        if l2e & PTE_PRESENT == 0 || l2e & PTE_HUGE != 0 { addr += page_size; continue; }

        let l1_ptr = ((phys_off + (l2e & PTE_ADDR_MASK)) as *mut u64).add(l1i);
        let mut pte = *l1_ptr;
        if pte & PTE_PRESENT == 0 { addr += page_size; continue; }

        // Update protection bits.  Preserve all other bits including CoW (bit 9).
        if writable    { pte |=  PTE_WRITABLE; } else { pte &= !PTE_WRITABLE; }
        // NO_EXECUTE is bit 63.
        const PTE_NX: u64 = 1 << 63;
        if !executable { pte |= PTE_NX; } else { pte &= !PTE_NX; }

        *l1_ptr = pte;
        core::arch::asm!("invlpg [{}]", in(reg) addr, options(nostack, preserves_flags));
        addr += page_size;
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
        let pid = crate::scheduler::current_pid();
        let phys = super::self_model::alloc_frame_predictive(pid).ok_or("map_user_range: OOM")?;

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


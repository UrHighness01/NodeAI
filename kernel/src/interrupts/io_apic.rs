//! I/O APIC driver — routes ISA IRQs to IDT vectors.
//!
//! The I/O APIC lives at physical 0xFEC0_0000 (ACPI MADT default).
//! After VMM init we can reach it at `phys_offset + IOAPIC_PHYS`.
//! We programme exactly one entry: IRQ1 (PS/2 keyboard) → vector 0x21.

/// Standard I/O APIC physical base address.
const IOAPIC_PHYS: u64 = 0xFEC0_0000;

/// I/O APIC MMIO offsets.
const IOREGSEL: u64 = 0x00;
const IOWIN:    u64 = 0x10;

/// IOREDTBL entries start at register index 0x10.
/// Entry N occupies registers 0x10+2*N (low) and 0x11+2*N (high).
const IOREDTBL_BASE: u32 = 0x10;

static mut IOAPIC_VIRT: u64 = 0;

/// Call once AFTER VMM is set up so the physical region is accessible.
pub fn init(phys_offset: u64) {
    unsafe {
        IOAPIC_VIRT = phys_offset + IOAPIC_PHYS;

        let vector = super::apic::KEYBOARD_VECTOR as u32;

        // IRQ1 → keyboard
        ioapic_write(IOREDTBL_BASE + 2 * 1 + 1, 0x0000_0000);
        ioapic_write(IOREDTBL_BASE + 2 * 1, vector);

        // IRQ12 → PS/2 mouse
        let mouse_vec = super::apic::MOUSE_VECTOR as u32;
        ioapic_write(IOREDTBL_BASE + 2 * 12 + 1, 0x0000_0000);
        ioapic_write(IOREDTBL_BASE + 2 * 12,     mouse_vec);
    }
    crate::klog!(INFO, "I/O APIC: IRQ1 → vector {:#x}, IRQ12 → vector {:#x}",
        super::apic::KEYBOARD_VECTOR, super::apic::MOUSE_VECTOR);
}

unsafe fn ioapic_write(index: u32, value: u32) {
    let sel = (IOAPIC_VIRT + IOREGSEL) as *mut u32;
    let win = (IOAPIC_VIRT + IOWIN)    as *mut u32;
    core::ptr::write_volatile(sel, index);
    core::ptr::write_volatile(win, value);
}


//! ACPI parsing — Phase 2.
//!
//! Parses RSDP → RSDT/XSDT → MADT, HPET, FADT tables.
//! Provides: CPU count, IOAPIC addresses, power management registers.

use spin::Once;

mod rsdp;
mod madt;
mod fadt;

pub use madt::{CpuInfo, IoApicInfo, MadtData};
pub use fadt::pm_info;

/// Globally accessible ACPI data after init.
static ACPI: Once<AcpiInfo> = Once::new();

pub struct AcpiInfo {
    pub sdt_root:    u64,
    pub cpu_count:   usize,
    pub io_apic_addr: Option<u32>,
    pub hpet_addr:   Option<u64>,
    pub lapic_addr:  u64,
}

/// Boot-time ACPI initialisation.
/// `rsdp_addr` is the physical address of the RSDP (from bootloader).
/// `phys_offset` is the virtual base at which all physical memory is mapped.
pub fn init(rsdp_addr: u64, phys_offset: u64) {
    crate::klog!(INFO, "ACPI: parsing tables from RSDP @ {:#x}", rsdp_addr);

    let sdt_root = match unsafe { rsdp::locate_and_validate(rsdp_addr, phys_offset) } {
        Ok(addr) => {
            crate::klog!(INFO, "ACPI: SDT root @ {:#x}", addr);
            addr
        }
        Err(e) => {
            crate::klog!(WARN, "ACPI: RSDP validation failed — {}", e);
            return;
        }
    };

    // ── MADT ─────────────────────────────────────────────────────────────────
    let (cpu_count, io_apic_addr, lapic_addr) = unsafe {
        if let Some(madt_phys) = madt::find_table(sdt_root, b"APIC", phys_offset) {
            let data = madt::parse(madt_phys, phys_offset);
            let ioa  = data.io_apics().first().map(|a| a.base_addr);
            let lc   = data.cpu_count();
            let la   = data.lapic_addr();
            crate::klog!(INFO, "ACPI: {} CPU(s), I/O APIC @ {:?}, LAPIC @ {:#x}",
                lc, ioa, la);
            (lc, ioa, la)
        } else {
            crate::klog!(WARN, "ACPI: MADT not found in SDT");
            (1, None, crate::interrupts::LOCAL_APIC_BASE)
        }
    };

    // ── HPET ─────────────────────────────────────────────────────────────────
    let hpet_addr = unsafe {
        if let Some(hpet_phys) = madt::find_table(sdt_root, b"HPET", phys_offset) {
            // HPET base address is at offset 44 in the table (after the standard
            // ACPI header). hpet_phys is physical → add phys_offset for the virtual ptr.
            let base = core::ptr::read_unaligned((phys_offset + hpet_phys + 44) as *const u64);
            crate::klog!(INFO, "ACPI: HPET @ {:#x}", base);
            Some(base)
        } else {
            crate::klog!(INFO, "ACPI: no HPET table");
            None
        }
    };

    // ── FADT ─────────────────────────────────────────────────────────────────
    unsafe {
        if let Some(fadt_phys) = madt::find_table(sdt_root, b"FACP", phys_offset) {
            fadt::parse(fadt_phys, phys_offset);
        }
    }

    ACPI.call_once(|| AcpiInfo {
        sdt_root,
        cpu_count,
        io_apic_addr,
        hpet_addr,
        lapic_addr,
    });
}

/// Returns ACPI info if `init()` has been called.
pub fn info() -> Option<&'static AcpiInfo> {
    ACPI.get()
}


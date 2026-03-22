//! MADT (Multiple APIC Description Table) parsing.
//! Provides CPU and I/O APIC topology.

/// Max CPUs / I/O APICs we track.
const MAX_CPUS: usize    = 256;
const MAX_IO_APICS: usize = 8;

/// Information about a logical CPU parsed from MADT.
#[derive(Debug, Clone, Copy)]
pub struct CpuInfo {
    pub acpi_id:  u8,
    pub apic_id:  u8,
    pub enabled:  bool,
}

/// Information about an I/O APIC parsed from MADT.
#[derive(Debug, Clone, Copy)]
pub struct IoApicInfo {
    pub id:         u8,
    pub base_addr:  u32,
    pub gsi_base:   u32,
}

/// Interrupt source override — maps legacy ISA IRQ to GSI.
#[derive(Debug, Clone, Copy)]
pub struct IrqOverride {
    pub bus:       u8,
    pub irq:       u8,
    pub gsi:       u32,
    pub flags:     u16,
}

/// Parsed MADT data.
pub struct MadtData {
    cpus:       [CpuInfo;    MAX_CPUS],
    cpu_count:  usize,
    io_apics:   [IoApicInfo; MAX_IO_APICS],
    io_count:   usize,
    lapic_addr: u64, // local APIC base (may differ from IA32_APIC_BASE)
}

impl MadtData {
    pub fn cpus(&self) -> &[CpuInfo] {
        &self.cpus[..self.cpu_count]
    }
    pub fn io_apics(&self) -> &[IoApicInfo] {
        &self.io_apics[..self.io_count]
    }
    pub fn lapic_addr(&self) -> u64 { self.lapic_addr }
    pub fn cpu_count(&self) -> usize { self.cpu_count }
}

// ── ACPI table header ─────────────────────────────────────────────────────────

#[repr(C, packed)]
struct AcpiHeader {
    signature:  [u8; 4],
    length:     u32,
    revision:   u8,
    checksum:   u8,
    oem_id:     [u8; 6],
    oem_table:  [u8; 8],
    oem_rev:    u32,
    creator_id: u32,
    creator_rev:u32,
}

// ── MADT layout ───────────────────────────────────────────────────────────────

#[repr(C, packed)]
struct MadtHeader {
    header:      AcpiHeader,
    lapic_addr:  u32,
    flags:       u32,  // bit 0 = dual 8259 present
}

/// Parse the MADT at **physical** address `madt_phys`.
/// `phys_offset` maps physical → virtual: `virt = phys_offset + phys`.
/// # Safety
/// `madt_phys` must be a valid physical address of a MADT.
pub unsafe fn parse(madt_phys: u64, phys_offset: u64) -> MadtData {
    let madt_virt = phys_offset + madt_phys;
    let hdr = &*(madt_virt as *const MadtHeader);
    let length = { hdr.header.length } as usize;
    let lapic_base = { hdr.lapic_addr } as u64;

    let mut data = MadtData {
        cpus:       [CpuInfo { acpi_id: 0, apic_id: 0, enabled: false }; MAX_CPUS],
        cpu_count:  0,
        io_apics:   [IoApicInfo { id: 0, base_addr: 0, gsi_base: 0 }; MAX_IO_APICS],
        io_count:   0,
        lapic_addr: lapic_base,
    };

    // the MADT body's byte range: right after the 44-byte fixed header.
    let body_start = madt_virt as usize + core::mem::size_of::<MadtHeader>();
    let body_end   = madt_virt as usize + length;
    let mut offset = body_start;

    while offset + 2 <= body_end {
        let rec_type = *(offset as *const u8);
        let rec_len  = *((offset + 1) as *const u8) as usize;
        if rec_len < 2 { break; }

        match rec_type {
            // Type 0 — Processor Local APIC
            0 if rec_len >= 8 => {
                let acpi_id = *((offset + 2) as *const u8);
                let apic_id = *((offset + 3) as *const u8);
                let flags   = *((offset + 4) as *const u32);
                if data.cpu_count < MAX_CPUS {
                    data.cpus[data.cpu_count] = CpuInfo {
                        acpi_id,
                        apic_id,
                        enabled: (flags & 1) != 0,
                    };
                    data.cpu_count += 1;
                }
            }

            // Type 1 — I/O APIC
            1 if rec_len >= 12 => {
                let id        = *((offset + 2) as *const u8);
                let base_addr = *((offset + 4) as *const u32);
                let gsi_base  = *((offset + 8) as *const u32);
                if data.io_count < MAX_IO_APICS {
                    data.io_apics[data.io_count] = IoApicInfo {
                        id,
                        base_addr: { base_addr },
                        gsi_base:  { gsi_base },
                    };
                    data.io_count += 1;
                }
            }

            // Type 5 — Local APIC Address Override (64-bit LAPIC address)
            5 if rec_len >= 12 => {
                let addr = *((offset + 4) as *const u64);
                data.lapic_addr = { addr };
            }

            _ => {} // type 2 (overrides), 3, 4, etc. — skip for now
        }

        offset += rec_len;
    }

    data
}

// ── SDT root walk helpers ─────────────────────────────────────────────────────

/// Given an XSDT/RSDT root **physical** address, find the physical address of
/// a child table by 4-byte signature.  Returns the child **physical** address.
/// `phys_offset` maps physical → virtual for all dereferences.
/// # Safety
/// `xsdt_phys` must be a valid physical address of the system table root.
pub unsafe fn find_table(xsdt_phys: u64, sig: &[u8; 4], phys_offset: u64) -> Option<u64> {
    let xsdt_virt = phys_offset + xsdt_phys;
    let hdr = &*(xsdt_virt as *const AcpiHeader);
    let length = { hdr.length } as usize;
    let revision = hdr.revision;

    let entry_size: usize = if revision >= 2 { 8 } else { 4 };
    let entries_start = xsdt_virt as usize + core::mem::size_of::<AcpiHeader>();
    let entries_end   = xsdt_virt as usize + length;
    let mut pos = entries_start;

    while pos + entry_size <= entries_end {
        // Entry in the table is a physical address of the child table.
        let child_phys: u64 = if entry_size == 8 {
            *(pos as *const u64)
        } else {
            *(pos as *const u32) as u64
        };

        if child_phys != 0 {
            let child_virt = phys_offset + child_phys;
            let child_hdr = child_virt as *const AcpiHeader;
            if &(*child_hdr).signature == sig {
                return Some(child_phys); // return physical address
            }
        }

        pos += entry_size;
    }

    None
}


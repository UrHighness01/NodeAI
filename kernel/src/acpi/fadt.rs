//! FADT (Fixed ACPI Description Table) — power management register locations.
//!
//! We extract the PM1a/PM1b control blocks used for ACPI shutdown (SLP_EN).

use spin::Once;

/// Power management port info.
pub struct PmInfo {
    pub pm1a_ctrl: u32,
    pub pm1b_ctrl: u32,
    pub slp_typa:  u16,
    pub slp_typb:  u16,
}

static PM_INFO: Once<PmInfo> = Once::new();

/// Returns power management info if FADT has been parsed.
pub fn pm_info() -> Option<&'static PmInfo> {
    PM_INFO.get()
}

#[repr(C, packed)]
struct FadtRaw {
    // Standard ACPI header (36 bytes)
    signature:      [u8; 4],
    length:         u32,
    revision:       u8,
    checksum:       u8,
    oem_id:         [u8; 6],
    oem_table:      [u8; 8],
    oem_rev:        u32,
    creator_id:     u32,
    creator_rev:    u32,
    // FADT fields
    fw_ctrl:        u32,
    dsdt:           u32,
    _reserved:      u8,
    preferred_pm:   u8,
    sci_int:        u16,
    smi_cmd:        u32,
    acpi_enable:    u8,
    acpi_disable:   u8,
    _s4bios:        u8,
    _pstate:        u8,
    pm1a_evt_blk:   u32,
    pm1b_evt_blk:   u32,
    pm1a_cnt_blk:   u32,
    pm1b_cnt_blk:   u32,
    // ... more fields follow but we only need the above
}

/// Parse the FADT at physical address `fadt_phys`.
/// `phys_offset` maps physical → virtual for the dereference.
/// # Safety
/// `fadt_phys` must be a valid physical address of a FADT.
pub unsafe fn parse(fadt_phys: u64, phys_offset: u64) {
    let fadt = &*((phys_offset + fadt_phys) as *const FadtRaw);
    let pm1a = { fadt.pm1a_cnt_blk };
    let pm1b = { fadt.pm1b_cnt_blk };

    crate::klog!(INFO, "ACPI: FADT PM1a_CNT={:#x} PM1b_CNT={:#x}", pm1a, pm1b);

    PM_INFO.call_once(|| PmInfo {
        pm1a_ctrl: pm1a,
        pm1b_ctrl: pm1b,
        slp_typa:  0,
        slp_typb:  0,
    });
}

/// Trigger ACPI shutdown via PM1 control register.
/// # Safety
/// Caller ensures this is intentional (called from power-off path).
pub unsafe fn acpi_shutdown() {
    use x86_64::instructions::port::Port;
    if let Some(info) = pm_info() {
        if info.pm1a_ctrl != 0 {
            let val: u16 = (info.slp_typa << 10) | (1 << 13); // SLP_EN
            Port::<u16>::new(info.pm1a_ctrl as u16).write(val);
        }
    }
}

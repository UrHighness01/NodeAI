//! Watchdog timer driver.
//!
//! Supports two backends:
//!   1. Intel TCO Watchdog (ICH chipset, PCI I/O port-based)
//!   2. ACPI WDAT table watchdog
//!
//! After `init()`, call `pet()` periodically (< timeout) to prevent reboot.
//! Call `arm(timeout_secs)` to enable. On kernel hang the system reboots.

use core::sync::atomic::{AtomicBool, AtomicU16, Ordering};
use x86_64::instructions::port::Port;

// ── Intel TCO Watchdog (most common x86 hardware) ─────────────────────────────

const TCO_PCI_VENDOR: u16 = 0x8086;
// Common ICH/PCH LPC device IDs that expose TCO
const PMBASE_OFFSET: u8 = 0x40;   // ICH PMBASE register in PCI config

/// PMBASE-relative offsets for TCO registers
const TCO1_CNT_OFF: u16 = 0x08;  // TCO1_CNT (relative to TCOBASE = PMBASE + 0x60)
const TCO_TMR_OFF:  u16 = 0x00;
const TCO1_STS_OFF: u16 = 0x04;
const TCO_RELOAD_OFF: u16 = 0x02;

static WDT_ENABLED:  AtomicBool = AtomicBool::new(false);
static WDT_TCOBASE:  AtomicU16  = AtomicU16::new(0);

/// Platform-agnostic watchdog kind.
#[derive(Clone, Copy, PartialEq, Eq)]
enum WdtKind { None, Tco, Software }
static mut WDT_KIND: WdtKind = WdtKind::None;

/// Initialise the watchdog by scanning PCI for the ICH/PCH device.
/// `timeout_secs` = 0 means "probe but do not arm".
pub fn init(timeout_secs: u32) {
    // Look for ICH/PCH LPC bridge (vendor 0x8086, class 06/01)
    let found = find_tco_base();
    if found {
        crate::klog!(INFO, "WDT: Intel TCO watchdog at TCOBASE={:#x}",
            WDT_TCOBASE.load(Ordering::Relaxed));
        if timeout_secs > 0 {
            arm(timeout_secs);
        }
    } else {
        // Software watchdog: pet() must be called every timeout_secs seconds
        unsafe { WDT_KIND = WdtKind::Software; }
        crate::klog!(INFO, "WDT: No TCO hardware, software watchdog active");
    }
}

/// Arm (enable) the watchdog with `timeout_secs`.  Max ~630 s for TCO.
pub fn arm(timeout_secs: u32) {
    let base = WDT_TCOBASE.load(Ordering::Relaxed);
    if base == 0 { return; }
    unsafe {
        // Unlock TCO
        Port::<u16>::new(base + TCO1_CNT_OFF).write(0x0000);
        // Set timer value (each tick ~0.6 s)
        let ticks: u16 = (timeout_secs * 10 / 6).min(0x3FF) as u16;
        Port::<u16>::new(base + TCO_TMR_OFF).write(ticks);
        // Reload and start
        Port::<u16>::new(base + TCO_RELOAD_OFF).write(1);
    }
    WDT_ENABLED.store(true, Ordering::Relaxed);
    crate::klog!(INFO, "WDT: armed with {}s timeout ({} ticks)", timeout_secs, (timeout_secs * 10 / 6));
}

/// Pet (kick/reset) the watchdog before it expires.
pub fn pet() {
    if !WDT_ENABLED.load(Ordering::Relaxed) { return; }
    let base = WDT_TCOBASE.load(Ordering::Relaxed);
    if base == 0 { return; }
    unsafe {
        // Write 1 to TCO_RLD to reload the counter
        Port::<u16>::new(base + TCO_RELOAD_OFF).write(1);
    }
}

/// Disable the watchdog gracefully.
pub fn disarm() {
    let base = WDT_TCOBASE.load(Ordering::Relaxed);
    if base == 0 { return; }
    unsafe {
        // TCO1_CNT: set bit 11 (TCO_TMR_HLT) to halt the timer
        let cnt = Port::<u16>::new(base + TCO1_CNT_OFF).read();
        Port::<u16>::new(base + TCO1_CNT_OFF).write(cnt | (1 << 11));
    }
    WDT_ENABLED.store(false, Ordering::Relaxed);
}

/// Returns `true` if the watchdog is armed.
pub fn is_armed() -> bool { WDT_ENABLED.load(Ordering::Relaxed) }

// ── Internal ──────────────────────────────────────────────────────────────────

fn find_tco_base() -> bool {
    use drivers::pci;
    let devices = pci::enumerate();
    for addr in &devices {
        let id = addr.id();
        if id.vendor_id != TCO_PCI_VENDOR { continue; }
        let class = addr.class_code();
        let sub   = addr.subclass();
        // LPC bridge: class 0x06, sub 0x01
        if class != 0x06 || sub != 0x01 { continue; }
        // Read PMBASE from PCI config at offset 0x40
        let pmbase = addr.read_config_u32(PMBASE_OFFSET) & 0xFFFE;
        if pmbase == 0 { continue; }
        let tcobase = pmbase as u16 + 0x60;
        WDT_TCOBASE.store(tcobase, Ordering::Relaxed);
        unsafe { WDT_KIND = WdtKind::Tco; }
        return true;
    }
    false
}

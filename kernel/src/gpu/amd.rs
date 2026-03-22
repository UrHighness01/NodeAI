//! AMD Radeon display driver stub — Phase 27.
//!
//! Performs PCI detection for AMD/ATI display adapters and provides the same
//! minimal API as i915.rs.  Full command-processor and modesetting support
//! is deferred to Phase 28+.

use spin::Mutex;

const AMD_VENDOR: u16 = 0x1002;
const DISPLAY_CLASS: u8 = 0x03;

// ── Radeon register offsets (DCE) ─────────────────────────────────────────────
const D1CRTC_H_TOTAL:         u32 = 0x6000;
const D1CRTC_V_TOTAL:         u32 = 0x6020;
const D1CRTC_CONTROL:         u32 = 0x6080;
const D1GRPH_PRIMARY_SURFACE_ADDRESS: u32 = 0x6110;
const D1GRPH_PITCH:           u32 = 0x6120;

pub struct AmdInfo {
    pub mmio_base:  u64,
    pub fb_base:    u64,
    pub fb_stride:  u32,
    pub width:      u32,
    pub height:     u32,
}

static AMD: Mutex<Option<AmdInfo>> = Mutex::new(None);

pub fn probe(phys_offset: u64) -> bool {
    let devices = drivers::pci::enumerate();
    for addr in &devices {
        let id = addr.id();
        if id.vendor_id != AMD_VENDOR { continue; }
        if addr.class_code() != DISPLAY_CLASS { continue; }
        if addr.subclass() == 2 { continue; }

        addr.enable_bus_master();

        let bar0_phys = addr.bar_mmio_base(0);
        if bar0_phys == 0 { continue; }
        let mmio = phys_offset + bar0_phys;

        unsafe {
            let htotal = core::ptr::read_volatile((mmio + D1CRTC_H_TOTAL as u64) as *const u32);
            let vtotal = core::ptr::read_volatile((mmio + D1CRTC_V_TOTAL as u64) as *const u32);
            let w = (htotal & 0x3FFF) + 1;
            let h = (vtotal & 0x3FFF) + 1;

            let fb_phys = core::ptr::read_volatile(
                (mmio + D1GRPH_PRIMARY_SURFACE_ADDRESS as u64) as *const u32,
            ) as u64;
            let stride = core::ptr::read_volatile(
                (mmio + D1GRPH_PITCH as u64) as *const u32,
            );

            crate::klog!(INFO, "AMD: {}×{} fb={:#x}", w, h, fb_phys);
            *AMD.lock() = Some(AmdInfo {
                mmio_base: mmio,
                fb_base:   phys_offset + fb_phys,
                fb_stride: stride,
                width: w, height: h,
            });
        }
        return true;
    }
    false
}

pub fn is_available() -> bool { AMD.lock().is_some() }
pub fn resolution() -> Option<(u32, u32)> {
    AMD.lock().as_ref().map(|i| (i.width, i.height))
}
pub fn fb_phys() -> Option<u64> {
    AMD.lock().as_ref().map(|i| i.fb_base)
}

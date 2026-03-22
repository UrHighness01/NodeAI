//! Intel i915 integrated graphics driver stub — Phase 27.
//!
//! Implements:
//!   - PCI discovery: vendor=0x8086, class=0x03 (display)
//!   - GTTMMADR BAR0 mapping
//!   - Pipe A enable / disable via PIPECONF register
//!   - Read current mode (HTOTAL / VTOTAL)
//!   - Linear framebuffer base address retrieval (for VBE hand-off)
//!
//! Full modesetting (DPLL programming, panel power, FDI links) is a future task.

use spin::Mutex;
use alloc::vec::Vec;

const INTEL_VENDOR: u16 = 0x8086;
const DISPLAY_CLASS: u8 = 0x03;

// ── GTT MMIO register offsets ─────────────────────────────────────────────────
const PIPEACONF:    u32 = 0x0007_0008;
const PIPEA_HTOTAL: u32 = 0x0006_0000;
const PIPEA_VTOTAL: u32 = 0x0006_0004;
const DSPABASE:     u32 = 0x0007_0184; // Display A base (legacy)
const DSPASTRIDE:   u32 = 0x0007_0188;
const DSPACNTR:     u32 = 0x0007_0180;

// PIPECONF bits
const PIPECONF_ENABLE: u32 = 1 << 31;
const PIPECONF_STATE:  u32 = 1 << 30;

// DSPACNTR bits
const DSPCNTR_ENABLE: u32 = 1 << 31;

pub struct I915Info {
    pub mmio_base:   u64,
    pub fb_base:     u64,
    pub fb_stride:   u32,
    pub width:       u32,
    pub height:      u32,
    pub pipe_active: bool,
}

static I915: Mutex<Option<I915Info>> = Mutex::new(None);

pub fn probe(phys_offset: u64) -> bool {
    let devices = drivers::pci::enumerate();
    for addr in &devices {
        let id = addr.id();
        if id.vendor_id != INTEL_VENDOR { continue; }
        if addr.class_code() != DISPLAY_CLASS { continue; }
        // subclass 0=VGA, 1=XGA, 2=3D (skip), 80=Other
        let sub = addr.subclass();
        if sub == 2 { continue; } // skip pure 3D cards

        addr.enable_bus_master();

        let bar0_phys = addr.bar_mmio_base(0);
        if bar0_phys == 0 { continue; }
        let mmio = phys_offset + bar0_phys;

        unsafe {
            let htotal = core::ptr::read_volatile((mmio + PIPEA_HTOTAL as u64) as *const u32);
            let vtotal = core::ptr::read_volatile((mmio + PIPEA_VTOTAL as u64) as *const u32);
            let active_h = (htotal & 0x0FFF) + 1;
            let active_v = (vtotal & 0x0FFF) + 1;

            let pipeconf = core::ptr::read_volatile((mmio + PIPEACONF as u64) as *const u32);
            let pipe_active = pipeconf & PIPECONF_ENABLE != 0;

            let fb_base  = core::ptr::read_volatile((mmio + DSPABASE as u64) as *const u32) as u64;
            let fb_stride= core::ptr::read_volatile((mmio + DSPASTRIDE as u64) as *const u32);

            let info = I915Info {
                mmio_base: mmio,
                fb_base:   phys_offset + fb_base,
                fb_stride,
                width:     active_h,
                height:    active_v,
                pipe_active,
            };
            crate::klog!(INFO,
                "i915: {}×{} pipe_active={} fb={:#x}",
                active_h, active_v, pipe_active, fb_base);
            *I915.lock() = Some(info);
        }
        return true;
    }
    false
}

pub fn is_available() -> bool { I915.lock().is_some() }

pub fn resolution() -> Option<(u32, u32)> {
    I915.lock().as_ref().map(|i| (i.width, i.height))
}

pub fn fb_phys() -> Option<u64> {
    I915.lock().as_ref().map(|i| i.fb_base)
}

/// Enable display plane A (no-op if already enabled).
pub fn enable_display() {
    if let Some(ref info) = *I915.lock() {
        unsafe {
            let mmio = info.mmio_base;
            let cntr = core::ptr::read_volatile((mmio + DSPACNTR as u64) as *const u32);
            core::ptr::write_volatile(
                (mmio + DSPACNTR as u64) as *mut u32,
                cntr | DSPCNTR_ENABLE,
            );
        }
    }
}

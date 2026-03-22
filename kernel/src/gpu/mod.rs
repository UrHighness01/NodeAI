//! GPU / DRM-KMS abstraction — Phase 27.
//!
//! Probes for Intel i915 and AMD Radeon, exposes a unified API:
//!   - `init(phys_offset)` — probe all GPU vendors
//!   - `resolution() -> Option<(u32,u32)>` — active display resolution
//!   - `fb_phys() -> Option<u64>` — linear framebuffer physical address
//!   - `is_available() -> bool`
//!
//! The framebuffer returned here can be mapped into the VBE/VESA region
//! used by the existing `framebuffer` module.

pub mod i915;
pub mod amd;

use core::sync::atomic::{AtomicBool, Ordering};

static GPU_READY: AtomicBool = AtomicBool::new(false);

/// Probe PCI for supported GPUs. Should be called once during kernel_main.
pub fn init(phys_offset: u64) {
    let found_i915 = i915::probe(phys_offset);
    let found_amd  = amd::probe(phys_offset);

    if found_i915 || found_amd {
        GPU_READY.store(true, Ordering::Relaxed);
        if let Some((w, h)) = resolution() {
            crate::klog!(INFO, "GPU: display {}×{} ready", w, h);
        }
    } else {
        crate::klog!(WARN, "GPU: no supported display adapter found");
    }
}

/// Returns true if a GPU was detected and initialised.
pub fn is_available() -> bool { GPU_READY.load(Ordering::Relaxed) }

/// Returns the resolution of the primary display.
pub fn resolution() -> Option<(u32, u32)> {
    i915::resolution().or_else(|| amd::resolution())
}

/// Returns the physical base of the linear framebuffer.
pub fn fb_phys() -> Option<u64> {
    i915::fb_phys().or_else(|| amd::fb_phys())
}

//! VirtIO driver family — primary drivers for VirtualBox/QEMU testing.
//! Implements VirtIO spec 1.2 (MMIO + PCI transport).

pub mod blk;   // VirtIO Block — disk I/O
pub mod net;   // VirtIO Net  — networking
pub mod gpu;   // VirtIO GPU  — framebuffer display

// ── Virtqueue ─────────────────────────────────────────────────────────────────

/// A VirtIO virtqueue descriptor.
#[derive(Debug, Default, Clone, Copy)]
#[repr(C)]
pub struct VirtqDesc {
    /// Physical address of the buffer.
    pub addr: u64,
    /// Length of the buffer in bytes.
    pub len: u32,
    /// Flags: NEXT (0x1), WRITE (0x2), INDIRECT (0x4).
    pub flags: u16,
    /// Index of the next descriptor in the chain (if NEXT flag set).
    pub next: u16,
}

/// A VirtIO used-ring element (id + written length).
#[derive(Debug, Default, Clone, Copy)]
#[repr(C)]
pub struct VirtqUsedElem {
    /// Descriptor table index of the head of the completed chain.
    pub id:  u32,
    /// Total bytes written into the buffer(s) by the device.
    pub len: u32,
}

/// VirtIO device status register values.
pub mod status {
    pub const ACKNOWLEDGE: u8 = 1;
    pub const DRIVER: u8      = 2;
    pub const DRIVER_OK: u8   = 4;
    pub const FEATURES_OK: u8 = 8;
    pub const FAILED: u8      = 128;
}

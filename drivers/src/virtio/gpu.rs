//! VirtIO GPU driver — Phase 12a.
//!
//! Implements the VirtIO 1.2 GPU device (PCI device ID 0x1050) for QEMU/VirtualBox.
//! Uses the legacy I/O-port transport for simplicity (matching our blk/net approach).
//!
//! Boot flow:
//!   1. `init()` — PCI probe, device reset, feature negotiation.
//!   2. `setup_framebuffer(width, height)` — create a host-side 2D resource, attach
//!      guest-allocated BGRA pixel pages, set as scanout.
//!   3. Caller writes pixels into the returned `*mut u8` slice.
//!   4. `flush()` — transfer + flush to push pixels to the virtual monitor.

use alloc::alloc::{alloc_zeroed, Layout};
use super::{VirtqDesc, status};
use crate::pci::PciAddress;

// ── PCI IDs ───────────────────────────────────────────────────────────────────
pub const VIRTIO_GPU_VENDOR:  u16 = 0x1AF4;
pub const VIRTIO_GPU_DEVICE:  u16 = 0x1050; // modern VirtIO GPU

// ── VirtIO-GPU command types ──────────────────────────────────────────────────
const VIRTIO_GPU_CMD_RESOURCE_CREATE_2D:      u32 = 0x0101;
const VIRTIO_GPU_CMD_SET_SCANOUT:             u32 = 0x0103;
const VIRTIO_GPU_CMD_RESOURCE_FLUSH:          u32 = 0x0104;
const VIRTIO_GPU_CMD_TRANSFER_TO_HOST_2D:     u32 = 0x0105;
const VIRTIO_GPU_CMD_RESOURCE_ATTACH_BACKING: u32 = 0x0106;

// ── Pixel format ─────────────────────────────────────────────────────────────
const VIRTIO_GPU_FORMAT_B8G8R8X8_UNORM: u32 = 2;

// ── Queue layout ─────────────────────────────────────────────────────────────
const QUEUE_SIZE: usize = 16;

#[repr(C, align(4096))]
struct VirtQueue {
    desc:        [VirtqDesc; QUEUE_SIZE],
    avail_flags: u16,
    avail_idx:   u16,
    avail_ring:  [u16; QUEUE_SIZE],
    _pad: [u8; 4096
        - (core::mem::size_of::<VirtqDesc>() * QUEUE_SIZE + 4 + QUEUE_SIZE * 2) % 4096],
    used_flags:  u16,
    used_idx:    u16,
}

// ── VirtIO-GPU command headers ────────────────────────────────────────────────

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct GpuCtrlHdr {
    cmd_type: u32,
    flags:    u32,
    fence_id: u64,
    ctx_id:   u32,
    padding:  u32,
}

#[repr(C)]
struct GpuRect { x: u32, y: u32, w: u32, h: u32 }

#[repr(C)]
struct GpuResourceCreate2d {
    hdr:         GpuCtrlHdr,
    resource_id: u32,
    format:      u32,
    width:        u32,
    height:       u32,
}

#[repr(C)]
struct GpuResourceAttachBacking {
    hdr:         GpuCtrlHdr,
    resource_id: u32,
    nr_entries:  u32,
}

#[repr(C)]
struct GpuMemEntry { addr: u64, length: u32, padding: u32 }

#[repr(C)]
struct GpuSetScanout {
    hdr:         GpuCtrlHdr,
    rect:        GpuRect,
    scanout_id:  u32,
    resource_id: u32,
}

#[repr(C)]
struct GpuTransferToHost2d {
    hdr:         GpuCtrlHdr,
    rect:        GpuRect,
    offset:      u64,
    resource_id: u32,
    padding:     u32,
}

#[repr(C)]
struct GpuResourceFlush {
    hdr:         GpuCtrlHdr,
    rect:        GpuRect,
    resource_id: u32,
    padding:     u32,
}

// ── Driver struct ─────────────────────────────────────────────────────────────
pub struct VirtioGpu {
    io_base:    u16,
    ctrl_queue: *mut VirtQueue,
    desc_next:  u16,
    avail_idx:  u16,
    /// Guest-side pixel buffer (BGRA, 4 bytes per pixel).
    pub fb_ptr:      *mut u8,
    pub fb_width:    u32,
    pub fb_height:   u32,
    resource_id: u32,
}

unsafe impl Send for VirtioGpu {}

impl VirtioGpu {
    /// Probe PCI bus for a VirtIO-GPU device and initialise it.
    /// Returns `None` if none is found or initialisation fails.
    pub unsafe fn init(addr: PciAddress) -> Option<Self> {
        let id = addr.id();
        if id.vendor_id != VIRTIO_GPU_VENDOR || id.device_id != VIRTIO_GPU_DEVICE {
            return None;
        }
        if !addr.bar_is_io(0) { return None; }
        addr.enable_bus_master();
        let io = addr.bar_io_base(0);

        use x86_64::instructions::port::Port;

        // Reset
        Port::<u8>::new(io + 18).write(0);
        Port::<u8>::new(io + 18).write(status::ACKNOWLEDGE | status::DRIVER);

        // Negotiate features (accept all for simplicity)
        let _feat = Port::<u32>::new(io).read();
        Port::<u32>::new(io + 4).write(0);
        Port::<u8>::new(io + 18).write(
            status::ACKNOWLEDGE | status::DRIVER | status::FEATURES_OK);

        // Set up control queue (queue 0)
        Port::<u16>::new(io + 14).write(0);
        let layout = Layout::from_size_align(4096, 4096).unwrap();
        let ctrl_q = alloc_zeroed(layout) as *mut VirtQueue;
        Port::<u32>::new(io + 8).write(ctrl_q as u32 / 4096);

        Port::<u8>::new(io + 18).write(
            status::ACKNOWLEDGE | status::DRIVER | status::FEATURES_OK | status::DRIVER_OK);

        Some(Self {
            io_base:    io,
            ctrl_queue: ctrl_q,
            desc_next:  0,
            avail_idx:  0,
            fb_ptr:     core::ptr::null_mut(),
            fb_width:   0,
            fb_height:  0,
            resource_id: 1,
        })
    }

    /// Allocate a guest framebuffer and program the GPU to display it.
    /// `width` × `height` in pixels (BGRX 32-bit).
    /// Returns the virtual address of the pixel buffer on success.
    pub unsafe fn setup_framebuffer(&mut self, width: u32, height: u32)
        -> Option<*mut u8>
    {
        self.fb_width  = width;
        self.fb_height = height;
        let bytes = (width * height * 4) as usize;

        // Allocate page-aligned guest memory for pixels
        let pages  = (bytes + 4095) / 4096;
        let layout = Layout::from_size_align(pages * 4096, 4096).unwrap();
        let fb_virt = alloc_zeroed(layout);
        self.fb_ptr = fb_virt;

        let mut resp_hdr = GpuCtrlHdr::default();

        // 1. RESOURCE_CREATE_2D
        let mut create = GpuResourceCreate2d {
            hdr:         GpuCtrlHdr { cmd_type: VIRTIO_GPU_CMD_RESOURCE_CREATE_2D,
                                      ..Default::default() },
            resource_id: self.resource_id,
            format:      VIRTIO_GPU_FORMAT_B8G8R8X8_UNORM,
            width,
            height,
        };
        self.send_cmd(
            &mut create as *mut _ as *mut u8, core::mem::size_of::<GpuResourceCreate2d>(),
            &mut resp_hdr as *mut _ as *mut u8, core::mem::size_of::<GpuCtrlHdr>());

        // 2. RESOURCE_ATTACH_BACKING — describe guest pages
        let ab_size = core::mem::size_of::<GpuResourceAttachBacking>()
                    + core::mem::size_of::<GpuMemEntry>();
        let ab_layout = Layout::from_size_align(ab_size, 8).unwrap();
        let ab_buf = alloc_zeroed(ab_layout);
        let ab_hdr = ab_buf as *mut GpuResourceAttachBacking;
        (*ab_hdr).hdr.cmd_type = VIRTIO_GPU_CMD_RESOURCE_ATTACH_BACKING;
        (*ab_hdr).resource_id  = self.resource_id;
        (*ab_hdr).nr_entries   = 1;
        let entry_ptr = ab_buf.add(core::mem::size_of::<GpuResourceAttachBacking>())
                               as *mut GpuMemEntry;
        entry_ptr.write(GpuMemEntry {
            addr: fb_virt as u64, length: bytes as u32, padding: 0,
        });
        self.send_cmd(ab_buf, ab_size,
                      &mut resp_hdr as *mut _ as *mut u8, core::mem::size_of::<GpuCtrlHdr>());

        // 3. SET_SCANOUT — bind resource to display 0
        let mut scanout = GpuSetScanout {
            hdr:         GpuCtrlHdr { cmd_type: VIRTIO_GPU_CMD_SET_SCANOUT,
                                      ..Default::default() },
            rect:        GpuRect { x: 0, y: 0, w: width, h: height },
            scanout_id:  0,
            resource_id: self.resource_id,
        };
        self.send_cmd(
            &mut scanout as *mut _ as *mut u8, core::mem::size_of::<GpuSetScanout>(),
            &mut resp_hdr as *mut _ as *mut u8, core::mem::size_of::<GpuCtrlHdr>());

        Some(fb_virt)
    }

    /// Push pixel buffer to the host display.
    pub unsafe fn flush(&mut self) {
        let (w, h) = (self.fb_width, self.fb_height);
        let mut resp_hdr = GpuCtrlHdr::default();

        // TRANSFER_TO_HOST_2D
        let mut xfer = GpuTransferToHost2d {
            hdr:         GpuCtrlHdr { cmd_type: VIRTIO_GPU_CMD_TRANSFER_TO_HOST_2D,
                                      ..Default::default() },
            rect:        GpuRect { x: 0, y: 0, w, h },
            offset:      0,
            resource_id: self.resource_id,
            padding:     0,
        };
        self.send_cmd(
            &mut xfer as *mut _ as *mut u8, core::mem::size_of::<GpuTransferToHost2d>(),
            &mut resp_hdr as *mut _ as *mut u8, core::mem::size_of::<GpuCtrlHdr>());

        // RESOURCE_FLUSH
        let mut flush_cmd = GpuResourceFlush {
            hdr:         GpuCtrlHdr { cmd_type: VIRTIO_GPU_CMD_RESOURCE_FLUSH,
                                      ..Default::default() },
            rect:        GpuRect { x: 0, y: 0, w, h },
            resource_id: self.resource_id,
            padding:     0,
        };
        self.send_cmd(
            &mut flush_cmd as *mut _ as *mut u8, core::mem::size_of::<GpuResourceFlush>(),
            &mut resp_hdr as *mut _ as *mut u8, core::mem::size_of::<GpuCtrlHdr>());
    }

    /// Submit a single request→response command pair on the control queue.
    unsafe fn send_cmd(&mut self, req: *mut u8, req_len: usize,
                        resp: *mut u8, resp_len: usize) {
        let q = &mut *self.ctrl_queue;

        let di_req  = self.desc_next as usize % QUEUE_SIZE;
        let di_resp = (self.desc_next + 1) as usize % QUEUE_SIZE;

        q.desc[di_req] = VirtqDesc {
            addr:  req as u64,
            len:   req_len as u32,
            flags: 0x1, // NEXT
            next:  di_resp as u16,
        };
        q.desc[di_resp] = VirtqDesc {
            addr:  resp as u64,
            len:   resp_len as u32,
            flags: 0x2, // WRITE
            next:  0,
        };

        let ai = self.avail_idx as usize % QUEUE_SIZE;
        q.avail_ring[ai] = di_req as u16;
        core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
        q.avail_idx = q.avail_idx.wrapping_add(1);
        self.desc_next  = self.desc_next.wrapping_add(2);
        self.avail_idx  = self.avail_idx.wrapping_add(1);

        // Notify device: queue 0
        core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
        x86_64::instructions::port::Port::<u16>::new(self.io_base + 16).write(0);

        // Poll for used-ring advance (synchronous, no IRQ needed)
        let mut spins = 0u32;
        loop {
            core::sync::atomic::fence(core::sync::atomic::Ordering::Acquire);
            if (*self.ctrl_queue).used_idx != q.avail_idx.wrapping_sub(1) { break; }
            spins += 1;
            if spins > 1_000_000 { break; } // timeout guard
            x86_64::instructions::nop();
        }
    }
}

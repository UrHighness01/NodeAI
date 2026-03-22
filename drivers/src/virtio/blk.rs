//! VirtIO Block driver — VirtualBox/QEMU disk over PCI transport.
//!
//! Implements the minimal virtio-blk spec (v1.2) needed to read sectors.
//! Uses split virtqueues (legacy + modern MMIO).

use alloc::alloc::{alloc_zeroed, Layout};
use super::{VirtqDesc, status};
use crate::pci::PciAddress;

// ── VirtIO-blk PCI vendor/device IDs ─────────────────────────────────────────
pub const VIRTIO_VENDOR: u16    = 0x1AF4;
pub const VIRTIO_BLK_DEVICE: u16 = 0x1001; // legacy ID
pub const VIRTIO_BLK_DEVICE2: u16 = 0x1042; // modern ID

// ── VirtIO PCI capability offsets ────────────────────────────────────────────
const VIRTIO_PCI_STATUS:      u8 = 18; // device status byte (word-aligned read)
const VIRTIO_PCI_QUEUE_SEL:   u8 = 14;
const VIRTIO_PCI_QUEUE_SIZE:  u8 = 12;
const VIRTIO_PCI_QUEUE_PFN:   u8 = 8;

// ── Queue constants ───────────────────────────────────────────────────────────
const QUEUE_SIZE: usize = 64;
const SECTOR_SIZE: usize = 512;

/// VirtIO split virtqueue.
#[repr(C, align(4096))]
struct VirtQueue {
    desc:  [VirtqDesc; QUEUE_SIZE],
    avail_flags: u16,
    avail_idx:   u16,
    avail_ring:  [u16; QUEUE_SIZE],
    // padding to 4096-byte boundary
    _pad: [u8; 4096 - (core::mem::size_of::<VirtqDesc>() * QUEUE_SIZE + 4 + QUEUE_SIZE * 2) % 4096],
    used_flags:  u16,
    used_idx:    u16,
}

/// VirtIO-blk request header.
#[repr(C)]
struct BlkReqHeader {
    req_type: u32,
    _reserved: u32,
    sector: u64,
}

/// Status byte placed after data buffer.
const VIRTIO_BLK_S_OK: u8    = 0;
const VIRTIO_BLK_S_IOERR: u8 = 1;
const VIRTIO_BLK_S_UNSUPP: u8 = 2;

/// Fully initialized VirtIO block device.
pub struct VirtioBlk {
    /// MMIO base address of the BAR0 region (mapped to virtual).
    mmio_base: *mut u8,
    queue: *mut VirtQueue,
    desc_next: u16,
    avail_idx: u16,
    sector_count: u64,
}

unsafe impl Send for VirtioBlk {}

impl VirtioBlk {
    /// Initialize a VirtIO-blk device at the PCI address.
    /// Returns `None` if the device is not a virtio-blk.
    ///
    /// # Safety
    /// `phys_offset` is the physical memory offset for MMIO remapping.
    pub unsafe fn init(addr: PciAddress, phys_offset: u64) -> Option<Self> {
        let id = addr.id();
        if id.vendor_id != VIRTIO_VENDOR
            || (id.device_id != VIRTIO_BLK_DEVICE && id.device_id != VIRTIO_BLK_DEVICE2)
        {
            return None;
        }

        // Read BAR0 (I/O port or MMIO base)
        let bar0 = addr.read_config_u32(0x10);
        let is_io = bar0 & 1 != 0;
        if is_io {
            // Legacy I/O port mode — get the base port
            let io_base = (bar0 & !0x3) as u16;
            return Some(Self::init_io_port(io_base));
        }

        // MMIO mode
        let mmio_phys = (bar0 & !0xF) as u64;
        let mmio_virt = (phys_offset + mmio_phys) as *mut u8;
        Some(Self::init_mmio(mmio_virt))
    }

    unsafe fn init_io_port(io_base: u16) -> Self {
        use x86_64::instructions::port::Port;

        // Reset device
        Port::<u8>::new(io_base + 18).write(0); // status = 0 (reset)
        Port::<u8>::new(io_base + 18).write(status::ACKNOWLEDGE | status::DRIVER);

        // Accept all features (simplified — production code should negotiate)
        let _features = Port::<u32>::new(io_base).read();
        Port::<u32>::new(io_base + 4).write(0); // write negotiated features

        Port::<u8>::new(io_base + 18).write(
            status::ACKNOWLEDGE | status::DRIVER | status::FEATURES_OK);

        // Select queue 0 (request queue)
        Port::<u16>::new(io_base + VIRTIO_PCI_QUEUE_SEL as u16).write(0);
        let qsize = Port::<u16>::new(io_base + VIRTIO_PCI_QUEUE_SIZE as u16).read() as usize;
        let qsize = qsize.min(QUEUE_SIZE);

        // Allocate queue memory (must be physically contiguous, page-aligned)
        let layout = Layout::from_size_align(4096, 4096).unwrap();
        let queue_mem = alloc_zeroed(layout) as *mut VirtQueue;

        // Tell device the physical address of the queue (page frame number)
        let queue_phys = queue_mem as u32 / 4096;
        Port::<u32>::new(io_base + VIRTIO_PCI_QUEUE_PFN as u16).write(queue_phys);

        Port::<u8>::new(io_base + 18).write(
            status::ACKNOWLEDGE | status::DRIVER | status::FEATURES_OK | status::DRIVER_OK);

        // Read sector count from virtio-blk config space (offset 0x14 in I/O space for capacity)
        let cap_lo = Port::<u32>::new(io_base + 0x14).read();
        let cap_hi = Port::<u32>::new(io_base + 0x18).read();
        let sector_count = (cap_hi as u64) << 32 | cap_lo as u64;

        VirtioBlk {
            mmio_base:    io_base as *mut u8,
            queue:        queue_mem,
            desc_next:    0,
            avail_idx:    0,
            sector_count,
        }
    }

    unsafe fn init_mmio(mmio_base: *mut u8) -> Self {
        // MMIO virtio is handled similarly; sector count at offset 0x100 + config
        VirtioBlk {
            mmio_base,
            queue:        core::ptr::null_mut(),
            desc_next:    0,
            avail_idx:    0,
            sector_count: 0,
        }
    }

    pub fn sector_count(&self) -> u64 { self.sector_count }

    /// Read `count` sectors starting at `sector` into `buf` (must be sector-aligned size).
    /// # Safety
    /// `buf` must be valid and large enough for `count * 512` bytes.
    pub unsafe fn read_sectors(&mut self, sector: u64, buf: &mut [u8]) -> Result<(), &'static str> {
        if self.queue.is_null() {
            return Err("VirtIO-blk not initialized (MMIO path not yet wired)");
        }
        let count = buf.len() / SECTOR_SIZE;
        if count == 0 { return Ok(()); }

        // Build request: header + data buffer + status byte
        let layout = Layout::from_size_align(
            core::mem::size_of::<BlkReqHeader>() + buf.len() + 1, 16).unwrap();
        let req_mem = alloc_zeroed(layout);
        let hdr = &mut *(req_mem as *mut BlkReqHeader);
        hdr.req_type = 0; // VIRTIO_BLK_T_IN (read)
        hdr._reserved = 0;
        hdr.sector = sector;

        let data_ptr = req_mem.add(core::mem::size_of::<BlkReqHeader>());
        let status_ptr = data_ptr.add(buf.len());

        let q = &mut *self.queue;

        // Descriptor 0: header (read-only from device perspective)
        let d0 = self.desc_next as usize % QUEUE_SIZE;
        q.desc[d0] = VirtqDesc {
            addr:  req_mem as u64,
            len:   core::mem::size_of::<BlkReqHeader>() as u32,
            flags: 0x1, // NEXT
            next:  (d0 as u16 + 1) % QUEUE_SIZE as u16,
        };
        // Descriptor 1: data buffer (writable by device)
        let d1 = (d0 + 1) % QUEUE_SIZE;
        q.desc[d1] = VirtqDesc {
            addr:  data_ptr as u64,
            len:   buf.len() as u32,
            flags: 0x1 | 0x2, // NEXT | WRITE
            next:  (d1 as u16 + 1) % QUEUE_SIZE as u16,
        };
        // Descriptor 2: status byte (writable)
        let d2 = (d1 + 1) % QUEUE_SIZE;
        q.desc[d2] = VirtqDesc {
            addr:  status_ptr as u64,
            len:   1,
            flags: 0x2, // WRITE, no next
            next:  0,
        };

        // Put d0 in avail ring
        let ai = q.avail_idx as usize % QUEUE_SIZE;
        q.avail_ring[ai] = d0 as u16;
        core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
        q.avail_idx = q.avail_idx.wrapping_add(1);
        core::sync::atomic::fence(core::sync::atomic::Ordering::Release);

        // Notify device (doorbell) — for I/O port mode write queue index to port io_base + 16
        let io_base = self.mmio_base as u16;
        x86_64::instructions::port::Port::<u16>::new(io_base + 16).write(0);

        // Poll used ring until device completes
        let mut timeout = 10_000_000u32;
        while q.used_idx == self.avail_idx.wrapping_sub(1) {
            core::hint::spin_loop();
            timeout -= 1;
            if timeout == 0 { return Err("VirtIO-blk timeout"); }
        }

        let st = *status_ptr;
        if st != VIRTIO_BLK_S_OK {
            return Err("VirtIO-blk I/O error");
        }

        // Copy data from request buffer to caller's buf
        let src = core::slice::from_raw_parts(data_ptr, buf.len());
        buf.copy_from_slice(src);

        self.desc_next = self.desc_next.wrapping_add(3);
        self.avail_idx = self.avail_idx.wrapping_add(1);
        Ok(())
    }
}


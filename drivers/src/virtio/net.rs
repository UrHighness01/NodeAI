//! VirtIO-net driver — Phase 9.
//!
//! Implements transmit and receive over a virtio-net PCI device.
//! Uses the same split-virtqueue infrastructure as virtio-blk.

use super::{VirtqDesc, VirtqUsedElem, status};
use crate::pci::PciAddress;

pub const VIRTIO_NET_VENDOR: u16 = 0x1AF4;
pub const VIRTIO_NET_DEVICE: u16 = 0x1000; // legacy network device
pub const VIRTIO_NET_DEVICE2: u16 = 0x1041; // modern

/// Must match the queue size reported by the VirtualBox VirtIO-net device.
/// VirtualBox 7.x reports 1024 for both RX and TX queues.
const QUEUE_SIZE: usize = 1024;
const ETH_MTU:    usize = 1514; // max Ethernet II frame without VLAN

// Indices of the two standard virtio-net queues
const RX_QUEUE: u16 = 0;
const TX_QUEUE: u16 = 1;

/// 10-byte virtio-net header prepended to every frame (legacy, without MRG_RXBUF).
/// IMPORTANT: only 10 bytes — num_buffers is NOT present without VIRTIO_NET_F_MRG_RXBUF.
#[repr(C)]
struct NetHdr {
    flags:       u8,
    gso_type:    u8,
    hdr_len:     u16,
    gso_size:    u16,
    csum_start:  u16,
    csum_offset: u16,
    // num_buffers is omitted — not negotiated, device writes exactly 10 bytes
}

/// Virtio split virtqueue (shared between RX and TX).
#[repr(C, align(4096))]
struct VirtQueue {
    desc:        [VirtqDesc; QUEUE_SIZE],
    avail_flags: u16,
    avail_idx:   u16,
    avail_ring:  [u16; QUEUE_SIZE],
    _pad: [u8; 4096 - (core::mem::size_of::<VirtqDesc>() * QUEUE_SIZE + 4 + QUEUE_SIZE * 2) % 4096],
    used_flags:  u16,
    used_idx:    u16,
    /// Used ring entries — device fills these as it consumes avail descriptors.
    used_ring:   [VirtqUsedElem; QUEUE_SIZE],
}

pub struct VirtioNet {
    io_base:          u16,
    rx_queue:         *mut VirtQueue,
    tx_queue:         *mut VirtQueue,
    rx_desc_next:     u16,
    tx_desc_next:     u16,
    tx_avail_idx:     u16,
    rx_avail_idx:     u16,
    /// Tracks how many used-ring entries we have already processed.
    last_rx_used_idx: u16,
    /// Physical memory offset: physical_address + phys_offset = virtual_address.
    phys_offset: u64,
    /// Allocates 2^order contiguous physical pages; returns physical address.
    alloc_phys: fn(order: usize) -> Option<u64>,
    pub mac: [u8; 6],
}

unsafe impl Send for VirtioNet {}

impl VirtioNet {
    /// Initialise from a PCI address.
    /// `phys_offset`: bootloader physical-memory offset (virt = phys + phys_offset).
    /// `alloc_phys`: allocates 2^order contiguous physical pages, returns physical address.
    pub unsafe fn init(
        addr: PciAddress,
        phys_offset: u64,
        alloc_phys: fn(usize) -> Option<u64>,
    ) -> Option<Self> {
        let id = addr.id();
        if id.vendor_id != VIRTIO_NET_VENDOR
            || (id.device_id != VIRTIO_NET_DEVICE && id.device_id != VIRTIO_NET_DEVICE2)
        {
            return None;
        }
        if !addr.bar_is_io(0) { return None; } // Only legacy I/O port mode for now
        addr.enable_bus_master();
        let io = addr.bar_io_base(0);
        Some(Self::init_io(io, phys_offset, alloc_phys))
    }

    unsafe fn init_io(io: u16, phys_offset: u64, alloc_phys: fn(usize) -> Option<u64>) -> Self {
        use x86_64::instructions::port::Port;

        // ── Strict VirtIO legacy (pre-1.0) init sequence ──
        // Step 1: Reset the device
        Port::<u8>::new(io + 18).write(0);
        // Step 2: Acknowledge we found it
        Port::<u8>::new(io + 18).write(status::ACKNOWLEDGE);
        // Step 3: Tell it we're driving it
        Port::<u8>::new(io + 18).write(status::ACKNOWLEDGE | status::DRIVER);
        // Step 4: Read and accept features
        //   Accept VIRTIO_NET_F_MAC (bit 5) so the MAC in config space is valid.
        //   Reject all other features (BIG-endian, checksum offload, GSO, etc.)
        let dev_feat = Port::<u32>::new(io).read();
        let accept   = dev_feat & (1 << 5); // VIRTIO_NET_F_MAC only
        Port::<u32>::new(io + 4).write(accept);
        // Step 5: (NO FEATURES_OK needed for legacy devices; skip it)

        // Read MAC from config space (offset 0x14 in I/O space)
        let mut mac = [0u8; 6];
        for (i, b) in mac.iter_mut().enumerate() {
            *b = Port::<u8>::new(io + 0x14 + i as u16).read();
        }

        // Step 6: Set up virtqueues using PHYSICAL frames so DMA addresses are correct.
        // VirtQueue with QUEUE_SIZE=1024 needs 7 pages (28 KiB); we use order=3 (8 pages).
        let alloc_queue = |queue_idx: u16| -> *mut VirtQueue {
            Port::<u16>::new(io + 14).write(queue_idx);
            // Read back the queue size the device advertises.
            let dev_qs = Port::<u16>::new(io + 12).read();
            // Panic loudly if device queue size doesn't match our compile-time constant.
            // Check logs: if you see this panic, change QUEUE_SIZE to match dev_qs.
            assert_eq!(dev_qs, QUEUE_SIZE as u16,
                "VirtIO-net: device queue size != QUEUE_SIZE");
            let q_phys = alloc_phys(3).expect("OOM for VirtIO queue");
            let q_virt = phys_offset + q_phys;
            // Zero the queue so all indices start at 0 and descriptors are empty.
            core::ptr::write_bytes(q_virt as *mut u8, 0, 8 * 4096);
            // Tell the device the physical page frame number.
            Port::<u32>::new(io + 8).write((q_phys / 4096) as u32);
            q_virt as *mut VirtQueue
        };

        // Set up RX queue (queue 0) then TX queue (queue 1)
        let rx_q = alloc_queue(RX_QUEUE);
        let tx_q = alloc_queue(TX_QUEUE);

        // Suppress all VirtIO PCI INTx interrupts — we use poll-only mode.
        // VRING_AVAIL_F_NO_INTERRUPT (bit 0) tells the device: do NOT send
        // an interrupt when you add an entry to the used ring.
        // NOTE: We deliberately do NOT set this flag here because some
        // VirtIO implementations (including VirtualBox) misinterpret it as
        // "don't process queue descriptors" rather than only suppressing
        // the interrupt signal.  The interrupt itself is harmless: it fires
        // on the PCI IRQ line which is masked in the I/O APIC, so the CPU
        // never sees it.  The ISR register is drained in poll_rx to keep
        // the device happy.

        // Step 7: Mark driver as ready
        Port::<u8>::new(io + 18).write(
            status::ACKNOWLEDGE | status::DRIVER | status::DRIVER_OK);

        // Read and discard ISR register (io+19) to clear any spurious interrupt
        // that may have been asserted in the window before we set avail_flags.
        let _ = Port::<u8>::new(io + 19).read();

        let mut net = Self {
            io_base:          io,
            rx_queue:         rx_q,
            tx_queue:         tx_q,
            rx_desc_next:     0,
            tx_desc_next:     0,
            tx_avail_idx:     0,
            rx_avail_idx:     0,
            last_rx_used_idx: 0,
            phys_offset,
            alloc_phys,
            mac,
        };

        // Pre-populate RX descriptors with receive buffers
        net.refill_rx();
        net
    }

    /// Provide the device with receive buffers (physical frames for correct DMA).
    unsafe fn refill_rx(&mut self) {
        let q = self.rx_queue; // raw pointer
        let hdr_size = core::mem::size_of::<NetHdr>();
        let buf_pages = (ETH_MTU + hdr_size + 4095) / 4096;
        let buf_order = buf_pages.next_power_of_two().trailing_zeros() as usize;
        for _ in 0..(QUEUE_SIZE / 2) {
            let buf_phys = (self.alloc_phys)(buf_order).expect("OOM for RX buffer");
            let buf_virt = self.phys_offset + buf_phys;
            core::ptr::write_bytes(buf_virt as *mut u8, 0, (1 << buf_order) * 4096);
            let d = self.rx_desc_next as usize % QUEUE_SIZE;
            core::ptr::write_volatile(&mut (*q).desc[d], VirtqDesc {
                addr:  buf_phys,
                len:   (ETH_MTU + hdr_size) as u32,
                flags: 0x2, // WRITE
                next:  0,
            });
            let ai = self.rx_avail_idx as usize % QUEUE_SIZE;
            core::ptr::write_volatile(&mut (*q).avail_ring[ai], d as u16);
            core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
            let new_avail = core::ptr::read_volatile(&(*q).avail_idx).wrapping_add(1);
            core::ptr::write_volatile(&mut (*q).avail_idx, new_avail);
            self.rx_desc_next  = self.rx_desc_next.wrapping_add(1);
            self.rx_avail_idx  = self.rx_avail_idx.wrapping_add(1);
        }
        // Notify device: queue 0 has new buffers
        core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
        x86_64::instructions::port::Port::<u16>::new(self.io_base + 16).write(0);
    }

    /// Return diagnostic info about the RX queue for serial logging.
    /// Returns (used_idx, avail_idx, desc0_addr, desc0_flags, last_rx_used_idx).
    pub unsafe fn rx_debug_state(&self) -> (u16, u16, u64, u16, u16) {
        let q = self.rx_queue;
        let used_idx  = core::ptr::read_volatile(&(*q).used_idx);
        let avail_idx = core::ptr::read_volatile(&(*q).avail_idx);
        let desc0_addr  = core::ptr::read_volatile(&(*q).desc[0].addr);
        let desc0_flags = core::ptr::read_volatile(&(*q).desc[0].flags);
        (used_idx, avail_idx, desc0_addr, desc0_flags, self.last_rx_used_idx)
    }

    /// Transmit a raw Ethernet frame.
    pub unsafe fn transmit(&mut self, frame: &[u8]) -> Result<(), &'static str> {
        if frame.len() > ETH_MTU { return Err("frame too large"); }

        let hdr_size = core::mem::size_of::<NetHdr>();
        let total = hdr_size + frame.len();
        // Allocate a physical page for the TX buffer so DMA addresses are correct.
        let buf_order = ((total + 4095) / 4096).next_power_of_two().trailing_zeros() as usize;
        let buf_phys = (self.alloc_phys)(buf_order).ok_or("OOM for TX buffer")?;
        let buf_virt = self.phys_offset + buf_phys;
        // Zero the virtio-net header, then copy frame data.
        core::ptr::write_bytes(buf_virt as *mut u8, 0, hdr_size);
        core::ptr::copy_nonoverlapping(frame.as_ptr(), (buf_virt + hdr_size as u64) as *mut u8, frame.len());

        let q = &mut *self.tx_queue;
        let d = self.tx_desc_next as usize % QUEUE_SIZE;
        q.desc[d] = VirtqDesc {
            addr:  buf_phys,                          // physical address for DMA
            len:   total as u32,
            flags: 0,
            next:  0,
        };
        let ai = self.tx_avail_idx as usize % QUEUE_SIZE;
        q.avail_ring[ai] = d as u16;
        core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
        q.avail_idx = q.avail_idx.wrapping_add(1);
        core::sync::atomic::fence(core::sync::atomic::Ordering::Release);

        // Doorbell: notify TX queue (queue 1)
        x86_64::instructions::port::Port::<u16>::new(self.io_base + 16).write(1);

        self.tx_desc_next = self.tx_desc_next.wrapping_add(1);
        self.tx_avail_idx = self.tx_avail_idx.wrapping_add(1);
        Ok(())
    }

    /// Poll for received frames. Calls `handler(frame_data)` for each.
    pub unsafe fn poll_rx(&mut self, mut handler: impl FnMut(&[u8])) {        // Read and clear ISR (io+19) so the PCI INTx line is de-asserted.
        // Even in poll-only mode, the device may assert the interrupt once
        // before seeing our avail_flags=1; clearing it here keeps the IRQ line low.
        let _ = x86_64::instructions::port::Port::<u8>::new(self.io_base + 19).read();

        let q = self.rx_queue; // raw pointer; do NOT create &mut — device also writes
        let hdr_size = core::mem::size_of::<NetHdr>();
        loop {
            // Ensure we see all DMA writes the device committed before updating used_idx.
            // read_volatile forces the compiler to always load from memory (no caching).
            core::sync::atomic::fence(core::sync::atomic::Ordering::Acquire);
            let used_idx = core::ptr::read_volatile(&(*q).used_idx);
            if used_idx == self.last_rx_used_idx { break; }

            let ui  = self.last_rx_used_idx as usize % QUEUE_SIZE;
            // read_volatile for used_ring entries written by the device
            let used_id  = core::ptr::read_volatile(&(*q).used_ring[ui].id);
            let used_len = core::ptr::read_volatile(&(*q).used_ring[ui].len);
            let desc_id  = used_id  as usize % QUEUE_SIZE;
            let pkt_len  = used_len as usize;

            if pkt_len > hdr_size {
                let buf_phys = core::ptr::read_volatile(&(*q).desc[desc_id].addr);
                let buf_virt = self.phys_offset + buf_phys;
                let payload = core::slice::from_raw_parts(
                    (buf_virt + hdr_size as u64) as *const u8,
                    pkt_len - hdr_size,
                );
                handler(payload);
            }

            // Re-submit this descriptor so the device can reuse the buffer.
            let ai = self.rx_avail_idx as usize % QUEUE_SIZE;
            core::ptr::write_volatile(&mut (*q).avail_ring[ai], desc_id as u16);
            core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
            let new_avail = core::ptr::read_volatile(&(*q).avail_idx).wrapping_add(1);
            core::ptr::write_volatile(&mut (*q).avail_idx, new_avail);
            core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
            // Doorbell: notify RX queue (queue 0)
            x86_64::instructions::port::Port::<u16>::new(self.io_base + 16).write(0);

            self.rx_avail_idx     = self.rx_avail_idx.wrapping_add(1);
            self.last_rx_used_idx = self.last_rx_used_idx.wrapping_add(1);
        }
    }
}

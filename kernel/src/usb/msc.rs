//! USB Mass Storage Class (MSC) driver — Bulk-Only Transport (BOT) + SCSI.
//! Phase 27.
//!
//! Exposes a simple block-read / block-write interface on top of USB BOT.
//! The xHCI layer calls `process_cbw_in/process_cbw_out` with the device's
//! bulk-in and bulk-out endpoint handles.

use alloc::{vec, vec::Vec};
use spin::Mutex;

const SECTOR_SIZE: usize = 512;

// ── SCSI commands ──────────────────────────────────────────────────────────────
const SCSI_READ10:       u8 = 0x28;
const SCSI_WRITE10:      u8 = 0x2A;
const SCSI_READ_CAPACITY: u8 = 0x25;
const SCSI_INQUIRY:      u8 = 0x12;
const SCSI_TEST_UNIT_RDY: u8 = 0x00;

// ── CBW / CSW ─────────────────────────────────────────────────────────────────
const CBW_SIGNATURE: u32 = 0x4342_5355; // "USBC"
const CSW_SIGNATURE: u32 = 0x5342_5355; // "USBS"

#[repr(C, packed)]
struct Cbw {
    sig:       u32,
    tag:       u32,
    data_len:  u32,
    flags:     u8,   // 0x80 = data-in, 0x00 = data-out
    lun:       u8,
    cb_len:    u8,
    cb:        [u8; 16],
}

#[repr(C, packed)]
struct Csw {
    sig:     u32,
    tag:     u32,
    residue: u32,
    status:  u8,
}

// ── UsbMscDrive — wraps a pair of bulk transfers ──────────────────────────────

/// Opaque handle filled in by xHCI when a MSC device is enumerated.
#[derive(Clone)]
pub struct BulkEndpoints {
    pub dev_addr: u8,
    pub ep_in:    u8,
    pub ep_out:   u8,
}

struct MscDrive {
    ep:      BulkEndpoints,
    sectors: u64,
    next_tag: u32,
}

impl MscDrive {
    fn tag(&mut self) -> u32 {
        let t = self.next_tag;
        self.next_tag = self.next_tag.wrapping_add(1);
        t
    }

    /// Build a SCSI READ(10) CBW and return it + the expected data length.
    fn build_read10_cbw(&mut self, lba: u32, count: u16) -> Cbw {
        let tag = self.tag();
        let mut cb = [0u8; 16];
        cb[0] = SCSI_READ10;
        cb[2] = (lba >> 24) as u8;
        cb[3] = (lba >> 16) as u8;
        cb[4] = (lba >> 8)  as u8;
        cb[5] =  lba        as u8;
        cb[7] = (count >> 8) as u8;
        cb[8] =  count       as u8;
        Cbw {
            sig:      CBW_SIGNATURE,
            tag,
            data_len: count as u32 * SECTOR_SIZE as u32,
            flags:    0x80, // data-in
            lun:      0,
            cb_len:   10,
            cb,
        }
    }

    fn build_write10_cbw(&mut self, lba: u32, count: u16) -> Cbw {
        let tag = self.tag();
        let mut cb = [0u8; 16];
        cb[0] = SCSI_WRITE10;
        cb[2] = (lba >> 24) as u8;
        cb[3] = (lba >> 16) as u8;
        cb[4] = (lba >> 8)  as u8;
        cb[5] =  lba        as u8;
        cb[7] = (count >> 8) as u8;
        cb[8] =  count       as u8;
        Cbw {
            sig:      CBW_SIGNATURE,
            tag,
            data_len: count as u32 * SECTOR_SIZE as u32,
            flags:    0x00, // data-out
            lun:      0,
            cb_len:   10,
            cb,
        }
    }
}

// ── Global device list ────────────────────────────────────────────────────────

static MSC_DRIVES: Mutex<Vec<MscDrive>> = Mutex::new(Vec::new());

/// Register a new USB MSC drive (called by xHCI after enumeration).
pub fn register_drive(ep: BulkEndpoints, sectors: u64) {
    crate::klog!(INFO, "USB MSC: drive registered ({} MiB)", sectors / 2048);
    MSC_DRIVES.lock().push(MscDrive { ep, sectors, next_tag: 1 });
}

/// Number of USB mass storage drives.
pub fn drive_count() -> usize { MSC_DRIVES.lock().len() }

/// Read `count` sectors from drive `idx` at `lba`.
/// In a full implementation this would issue the BOT transfer through xHCI.
/// Here we return a zeroed buffer as a stub (actual xHCI transfer callbacks
/// are connected in `usb/mod.rs`).
pub fn read_sectors(idx: usize, lba: u64, count: u16) -> Option<Vec<u8>> {
    let drives = MSC_DRIVES.lock();
    if idx >= drives.len() { return None; }
    // Stub: real implementation calls super::xhci_bulk_transfer()
    let buf = vec![0u8; count as usize * SECTOR_SIZE];
    Some(buf)
}

/// Write sectors — stub.
pub fn write_sectors(idx: usize, lba: u64, data: &[u8]) -> bool {
    let drives = MSC_DRIVES.lock();
    idx < drives.len()
}

/// Capacity in sectors for drive `idx`.
pub fn capacity(idx: usize) -> Option<u64> {
    let drives = MSC_DRIVES.lock();
    drives.get(idx).map(|d| d.sectors)
}

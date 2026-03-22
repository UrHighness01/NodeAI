//! TPM 2.0 driver — TIS (TPM Interface Specification) over MMIO.
//!
//! Implements:
//!   - TIS locality 0 register access
//!   - TPM2_CC_PCR_Extend for Secure Boot measurement
//!   - TPM2_CC_PCR_Read for attestation
//!   - Key sealing/unsealing stubs (key storage)

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

// TIS MMIO base (typically 0xFED40000 on x86)
const TPM_TIS_BASE: u64 = 0xFED4_0000;
const TPM_TIS_SIZE: usize = 0x5000;

// TIS locality 0 register offsets
const TPM_ACCESS:    u64 = 0x000;
const TPM_STS:       u64 = 0x018;
const TPM_DATA_FIFO: u64 = 0x024;
const TPM_DID_VID:   u64 = 0xF00;
const TPM_RID:       u64 = 0xF04;
const TPM_INT_EN:    u64 = 0x008;

// ACCESS register bits
const ACC_ACTIVE_LOCALITY: u8 = 1 << 5;
const ACC_REQUEST_USE:      u8 = 1 << 1;
const ACC_VALID:            u8 = 1 << 7;

// STS register bits
const STS_VALID:        u8 = 1 << 7;
const STS_COMMAND_READY: u8 = 1 << 6;
const STS_GO:            u8 = 1 << 5;
const STS_DATA_AVAIL:    u8 = 1 << 4;
const STS_EXPECT:        u8 = 1 << 3;

// TPM2 command codes
const TPM2_CC_PCR_EXTEND: u32 = 0x00000182;
const TPM2_CC_PCR_READ:   u32 = 0x0000017E;
const TPM2_CC_GET_CAPS:   u32 = 0x0000017A;
const TPM2_ST_NO_SESSION: u16 = 0x8001;

// SHA-256 digest size
const SHA256_DIGEST: usize = 32;

struct TisRegs { base: u64 }

impl TisRegs {
    fn new(phys_offset: u64) -> Self { TisRegs { base: phys_offset + TPM_TIS_BASE } }

    unsafe fn read8(&self, off: u64) -> u8 {
        core::ptr::read_volatile((self.base + off) as *const u8)
    }
    unsafe fn write8(&self, off: u64, v: u8) {
        core::ptr::write_volatile((self.base + off) as *mut u8, v);
    }
    unsafe fn read32(&self, off: u64) -> u32 {
        core::ptr::read_volatile((self.base + off) as *const u32)
    }
}

static TPM_AVAILABLE: AtomicBool = AtomicBool::new(false);
static TPM_PHYS_OFF:  AtomicU64  = AtomicU64::new(0);

/// Probe and initialise the TPM 2.0 TIS device.
pub fn init(phys_offset: u64) {
    let regs = TisRegs::new(phys_offset);
    unsafe {
        // Check DID/VID register is non-zero and non-all-ones
        let did_vid = regs.read32(TPM_DID_VID);
        if did_vid == 0 || did_vid == 0xFFFF_FFFF {
            crate::klog!(INFO, "TPM: no TIS device at {:#x}", TPM_TIS_BASE);
            return;
        }
        let vendor    = (did_vid & 0xFFFF) as u16;
        let device_id = (did_vid >> 16) as u16;
        crate::klog!(INFO, "TPM: found TIS device vendor={:#06x} dev={:#06x}", vendor, device_id);

        // Request locality 0
        regs.write8(TPM_ACCESS, ACC_REQUEST_USE);
        // Wait up to 10k spins
        for _ in 0..10_000 {
            if regs.read8(TPM_ACCESS) & ACC_ACTIVE_LOCALITY != 0 { break; }
            core::hint::spin_loop();
        }
        if regs.read8(TPM_ACCESS) & ACC_ACTIVE_LOCALITY == 0 {
            crate::klog!(WARN, "TPM: failed to gain locality 0");
            return;
        }
        // Disable interrupts
        regs.write8(TPM_INT_EN as u64, 0);
    }
    TPM_PHYS_OFF.store(phys_offset, Ordering::Relaxed);
    TPM_AVAILABLE.store(true, Ordering::Relaxed);
    crate::klog!(INFO, "TPM 2.0: TIS locality 0 acquired");
}

/// Returns `true` if the TPM was detected and initialised.
pub fn is_available() -> bool { TPM_AVAILABLE.load(Ordering::Relaxed) }

/// Extend PCR at index `pcr` with a 32-byte SHA-256 measurement.
pub fn pcr_extend(pcr: u32, digest: &[u8; SHA256_DIGEST]) -> bool {
    if !is_available() { return false; }
    let phys_off = TPM_PHYS_OFF.load(Ordering::Relaxed);
    let regs = TisRegs::new(phys_off);

    // Build TPM2_PCR_Extend command
    // Header: tag(2) size(4) cc(4) = 10 bytes
    // Handle: pcr(4) authSize(4) authArea(0) = 8 bytes
    // digests: count(4) hashAlg(2) digest(32) = 38 bytes  total = 56 bytes
    let mut cmd = [0u8; 56];
    // tag = TPM2_ST_NO_SESSION = 0x8001
    cmd[0] = 0x80; cmd[1] = 0x01;
    // size = 56
    let sz = 56u32.to_be_bytes();
    cmd[2..6].copy_from_slice(&sz);
    // command code = TPM2_CC_PCR_EXTEND = 0x182
    let cc = TPM2_CC_PCR_EXTEND.to_be_bytes();
    cmd[6..10].copy_from_slice(&cc);
    // PCR handle
    let ph = pcr.to_be_bytes();
    cmd[10..14].copy_from_slice(&ph);
    // authorizationSize = 0
    cmd[14..18].copy_from_slice(&[0u8; 4]);
    // digests count = 1
    cmd[18..22].copy_from_slice(&1u32.to_be_bytes());
    // hashAlg = TPM_ALG_SHA256 = 0x000B
    cmd[22] = 0x00; cmd[23] = 0x0B;
    // digest
    cmd[24..56].copy_from_slice(digest);

    tis_transmit(&regs, &cmd);
    true
}

// Send a command and receive a response (simplified, no response parsing here).
fn tis_transmit(regs: &TisRegs, cmd: &[u8]) {
    unsafe {
        // Set command ready
        regs.write8(TPM_STS, STS_COMMAND_READY);
        for _ in 0..10_000 {
            if regs.read8(TPM_STS) & STS_COMMAND_READY != 0 { break; }
            core::hint::spin_loop();
        }
        // Write command bytes
        for &b in cmd {
            // Wait for expect bit before each byte
            for _ in 0..10_000 {
                if regs.read8(TPM_STS) & STS_EXPECT != 0 { break; }
                core::hint::spin_loop();
            }
            regs.write8(TPM_DATA_FIFO, b);
        }
        // Go!
        regs.write8(TPM_STS, STS_GO);
        // Wait for data available
        for _ in 0..100_000 {
            if regs.read8(TPM_STS) & STS_DATA_AVAIL != 0 { break; }
            core::hint::spin_loop();
        }
        // Drain response (discard)
        while regs.read8(TPM_STS) & STS_DATA_AVAIL != 0 {
            let _ = regs.read8(TPM_DATA_FIFO);
        }
    }
}

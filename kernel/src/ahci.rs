//! AHCI (Serial ATA) driver — Phase 27.
//!
//! Implements:
//!   - PCI discovery of AHCI controllers (class 0x01 / sub 0x06)
//!   - HBA memory-mapped register access
//!   - Command list + FIS receive area setup
//!   - IDENTIFY DEVICE (ATA opcode 0xEC)
//!   - 48-bit LBA DMA READ (ATA opcode 0x25) + WRITE (0x35)
//!
//! The driver exposes a block device interface compatible with the VFS block
//! layer so VirtIO-blk and AHCI disks look identical to higher layers.

use alloc::{vec, vec::Vec, boxed::Box};
use spin::{Mutex, Once};
use core::sync::atomic::{AtomicBool, Ordering};

/// Translate a kernel virtual address to physical. Tries the VMM page-table
/// walk first; falls back to the phys_offset subtraction for identity-mapped
/// addresses (MMIO window, early boot structs).
#[inline]
fn va_to_pa(va: u64, phys_off: u64) -> u64 {
    crate::memory::translate(x86_64::VirtAddr::new(va))
        .map(|p| p.as_u64())
        .unwrap_or_else(|| va.saturating_sub(phys_off))
}

// ── PCI IDs ────────────────────────────────────────────────────────────────────
const AHCI_CLASS:    u8 = 0x01;
const AHCI_SUBCLASS: u8 = 0x06;

// ── HBA register offsets ──────────────────────────────────────────────────────
const HBA_CAP:    u32 = 0x000; // Host capability
const HBA_GHC:    u32 = 0x004; // Global host control
const HBA_IS:     u32 = 0x008; // Interrupt status
const HBA_PI:     u32 = 0x00C; // Ports implemented
const HBA_VS:     u32 = 0x010; // AHCI version

// Port register base = 0x100 + port * 0x80
const PORT_SIZE: u32 = 0x80;
const PORT_BASE: u32 = 0x100;

// Per-port register offsets
const P_CLB:    u32 = 0x00; // Command list base address (low)
const P_CLBU:   u32 = 0x04; // Command list base address (high)
const P_FB:     u32 = 0x08; // FIS base address (low)
const P_FBU:    u32 = 0x0C; // FIS base address (high)
const P_IS:     u32 = 0x10; // Interrupt status
const P_IE:     u32 = 0x14; // Interrupt enable
const P_CMD:    u32 = 0x18; // Command and status
const P_TFD:    u32 = 0x20; // Task file data
const P_SIG:    u32 = 0x24; // Signature
const P_SSTS:   u32 = 0x28; // SATA status
const P_SCTL:   u32 = 0x2C; // SATA control
const P_SERR:   u32 = 0x30; // SATA error
const P_SACT:   u32 = 0x34; // SATA active
const P_CI:     u32 = 0x38; // Command issue

// CMD register bits
const CMD_ST:   u32 = 1 << 0;  // Start
const CMD_FRE:  u32 = 1 << 4;  // FIS Receive Enable
const CMD_FR:   u32 = 1 << 14; // FIS Receive Running
const CMD_CR:   u32 = 1 << 15; // Command List Running

// GHC bits
const GHC_AHCI_ENABLE: u32 = 1 << 31;
const GHC_RESET:       u32 = 1 << 0;

// Task file status bits
const TFD_STS_BSY: u8 = 1 << 7;
const TFD_STS_DRQ: u8 = 1 << 3;

// FIS types
const FIS_TYPE_REG_H2D: u8 = 0x27; // Register FIS — host to device

// ATA commands
const ATA_CMD_IDENTIFY:    u8 = 0xEC;
const ATA_CMD_READ_DMA_EX: u8 = 0x25;
const ATA_CMD_WRITE_DMA_EX: u8 = 0x35;

const SECTOR_SIZE: usize = 512;

// ── FIS and command structures (repr(C) for HW access) ────────────────────────

#[repr(C, packed)]
struct FisRegH2D {
    fis_type:   u8,        // 0x27
    pmport_c:   u8,        // C=1 → command register
    command:    u8,
    feature_lo: u8,
    lba0:       u8,
    lba1:       u8,
    lba2:       u8,
    device:     u8,
    lba3:       u8,
    lba4:       u8,
    lba5:       u8,
    feature_hi: u8,
    count_lo:   u8,
    count_hi:   u8,
    icc:        u8,
    control:    u8,
    _res:       [u8; 4],
}

#[repr(C, align(128))]
struct CommandHeader {
    flags:   u16,  // CFL[4:0], other
    prdtl:   u16,  // PRDT entry count
    prdbc:   u32,  // Physical region descriptor byte count (set by HW)
    ctba:    u32,  // Command table base address (low)
    ctbau:   u32,  // Command table base address (high)
    _res:    [u32; 4],
}

#[repr(C, align(128))]
struct PrdtEntry {
    dba:    u32,   // Data base address (low)
    dbau:   u32,   // Data base address (high)
    _res:   u32,
    dbc:    u32,   // Byte count (bit 31 = interrupt on completion)
}

#[repr(C)]
struct CommandTable {
    cfis:  [u8; 64],       // Command FIS
    acmd:  [u8; 16],       // ATAPI command (unused for ATA)
    _res:  [u8; 48],
    prdt:  [PrdtEntry; 1], // 1 PRDT entry (enough for 1 sector)
}

// Fixed-size receive FIS area (256 bytes)
#[repr(C, align(256))]
struct RecvFis { data: [u8; 256] }

// ── Port state ─────────────────────────────────────────────────────────────────

struct AhciPort {
    regs_base:    u64,        // VA of HBA MMIO
    port_idx:     usize,
    phys_off:     u64,        // physical-to-virtual offset
    cmd_list:     Box<[CommandHeader; 32]>,
    cmd_list_pa:  u64,        // physical address of cmd_list
    recv_fis:     Box<RecvFis>,
    recv_fis_pa:  u64,        // physical address of recv_fis
    cmd_table:    Box<CommandTable>,
    cmd_table_pa: u64,        // physical address of cmd_table
    sectors:      u64,
    model:        [u8; 40],
}

impl AhciPort {
    fn port_reg(&self, off: u32) -> u64 {
        self.regs_base + PORT_BASE as u64 + self.port_idx as u64 * PORT_SIZE as u64 + off as u64
    }

    unsafe fn pr32(&self, off: u32) -> u32 {
        core::ptr::read_volatile(self.port_reg(off) as *const u32)
    }
    unsafe fn pw32(&self, off: u32, v: u32) {
        core::ptr::write_volatile(self.port_reg(off) as *mut u32, v)
    }

    /// Stop command engine and FIS receiving.
    unsafe fn stop(&self) {
        let cmd = self.pr32(P_CMD);
        self.pw32(P_CMD, cmd & !(CMD_ST | CMD_FRE));
        let deadline = crate::scheduler::uptime_ms() + 500;
        while crate::scheduler::uptime_ms() < deadline {
            let c = self.pr32(P_CMD);
            if c & (CMD_FR | CMD_CR) == 0 { break; }
            core::hint::spin_loop();
        }
    }

    /// Set command list and FIS base addresses, start engine.
    unsafe fn start(&mut self) {
        self.pw32(P_CLB,  self.cmd_list_pa as u32);
        self.pw32(P_CLBU, (self.cmd_list_pa >> 32) as u32);
        self.pw32(P_FB,   self.recv_fis_pa as u32);
        self.pw32(P_FBU,  (self.recv_fis_pa >> 32) as u32);
        // Clear errors
        self.pw32(P_SERR, 0xFFFF_FFFF);
        self.pw32(P_IS,   0xFFFF_FFFF);
        // Enable FIS receive + start
        let cmd = self.pr32(P_CMD) | CMD_FRE;
        self.pw32(P_CMD, cmd);
        let cmd = self.pr32(P_CMD) | CMD_ST;
        self.pw32(P_CMD, cmd);
    }

    /// Issue a command in slot 0 and wait for completion. Returns true on success.
    unsafe fn issue_cmd(&self) -> bool {
        // Clear IS
        self.pw32(P_IS, 0xFFFF_FFFF);
        // Issue slot 0
        self.pw32(P_CI, 1);
        let deadline = crate::scheduler::uptime_ms() + 5000;
        loop {
            if crate::scheduler::uptime_ms() >= deadline { return false; }
            if self.pr32(P_CI) & 1 == 0 { break; }
            core::hint::spin_loop();
        }
        (self.pr32(P_TFD) >> 1) & 1 == 0 // no error bit
    }

    /// Identify the drive; populate `sectors` and `model`.
    unsafe fn identify(&mut self) -> bool {
        let mut buf = [0u8; SECTOR_SIZE];

        // Build command table
        let ct = &mut *self.cmd_table;
        ct.cfis = [0u8; 64];
        let fis = &mut *(ct.cfis.as_mut_ptr() as *mut FisRegH2D);
        fis.fis_type  = FIS_TYPE_REG_H2D;
        fis.pmport_c  = 0x80;
        fis.command   = ATA_CMD_IDENTIFY;
        fis.device    = 0;

        // PRDT: 512 bytes from buf
        let buf_phys = va_to_pa(buf.as_ptr() as u64, self.phys_off);
        ct.prdt[0] = PrdtEntry {
            dba:  buf_phys as u32,
            dbau: (buf_phys >> 32) as u32,
            _res: 0,
            dbc:  (SECTOR_SIZE as u32 - 1) | (1 << 31),
        };

        // Command header slot 0
        let ct_phys = va_to_pa(ct as *const _ as u64, self.phys_off);
        self.cmd_list[0] = CommandHeader {
            flags:  (core::mem::size_of::<FisRegH2D>() / 4) as u16,
            prdtl:  1,
            prdbc:  0,
            ctba:   ct_phys as u32,
            ctbau:  (ct_phys >> 32) as u32,
            _res:   [0; 4],
        };

        if !self.issue_cmd() { return false; }

        // Parse identify data
        // Words 100-103: 48-bit total addressable sectors
        let words = buf.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect::<Vec<_>>();
        self.sectors = (words[100] as u64)
            | ((words[101] as u64) << 16)
            | ((words[102] as u64) << 32)
            | ((words[103] as u64) << 48);

        // Words 27-46: model string (40 bytes, each word big-endian)
        for i in 0..20usize {
            let w = words[27 + i];
            self.model[i * 2]     = (w >> 8) as u8;
            self.model[i * 2 + 1] = (w & 0xFF) as u8;
        }
        true
    }

    /// Read `count` sectors starting at `lba` into `buf`.
    pub unsafe fn read_sectors(&mut self, lba: u64, count: u16, buf: &mut [u8]) -> bool {
        let ct = &mut *self.cmd_table;
        ct.cfis = [0u8; 64];
        let fis = &mut *(ct.cfis.as_mut_ptr() as *mut FisRegH2D);
        fis.fis_type  = FIS_TYPE_REG_H2D;
        fis.pmport_c  = 0x80;
        fis.command   = ATA_CMD_READ_DMA_EX;
        fis.device    = 1 << 6; // LBA mode
        fis.lba0      = lba as u8;
        fis.lba1      = (lba >> 8)  as u8;
        fis.lba2      = (lba >> 16) as u8;
        fis.lba3      = (lba >> 24) as u8;
        fis.lba4      = (lba >> 32) as u8;
        fis.lba5      = (lba >> 40) as u8;
        fis.count_lo  = count as u8;
        fis.count_hi  = (count >> 8) as u8;

        let buf_phys = va_to_pa(buf.as_ptr() as u64, self.phys_off);
        let bytes = count as u32 * SECTOR_SIZE as u32;
        ct.prdt[0] = PrdtEntry {
            dba:  buf_phys as u32,
            dbau: (buf_phys >> 32) as u32,
            _res: 0,
            dbc:  (bytes - 1) | (1 << 31),
        };

        let ct_phys = va_to_pa(ct as *const _ as u64, self.phys_off);
        self.cmd_list[0] = CommandHeader {
            flags:  (core::mem::size_of::<FisRegH2D>() / 4) as u16 | (1 << 6), // write=0
            prdtl:  1,
            prdbc:  0,
            ctba:   ct_phys as u32,
            ctbau:  (ct_phys >> 32) as u32,
            _res:   [0; 4],
        };

        self.issue_cmd()
    }

    /// Write `count` sectors starting at `lba` from `buf`.
    pub unsafe fn write_sectors(&mut self, lba: u64, count: u16, buf: &[u8]) -> bool {
        let ct = &mut *self.cmd_table;
        ct.cfis = [0u8; 64];
        let fis = &mut *(ct.cfis.as_mut_ptr() as *mut FisRegH2D);
        fis.fis_type  = FIS_TYPE_REG_H2D;
        fis.pmport_c  = 0x80;
        fis.command   = ATA_CMD_WRITE_DMA_EX;
        fis.device    = 1 << 6;
        fis.lba0      = lba as u8;
        fis.lba1      = (lba >> 8)  as u8;
        fis.lba2      = (lba >> 16) as u8;
        fis.lba3      = (lba >> 24) as u8;
        fis.lba4      = (lba >> 32) as u8;
        fis.lba5      = (lba >> 40) as u8;
        fis.count_lo  = count as u8;
        fis.count_hi  = (count >> 8) as u8;

        let buf_phys = va_to_pa(buf.as_ptr() as u64, self.phys_off);
        let bytes = count as u32 * SECTOR_SIZE as u32;
        ct.prdt[0] = PrdtEntry {
            dba:  buf_phys as u32,
            dbau: (buf_phys >> 32) as u32,
            _res: 0,
            dbc:  (bytes - 1) | (1 << 31),
        };

        let ct_phys = va_to_pa(ct as *const _ as u64, self.phys_off);
        self.cmd_list[0] = CommandHeader {
            flags:  (core::mem::size_of::<FisRegH2D>() / 4) as u16 | (1 << 6), // write=1
            prdtl:  1,
            prdbc:  0,
            ctba:   ct_phys as u32,
            ctbau:  (ct_phys >> 32) as u32,
            _res:   [0; 4],
        };

        self.issue_cmd()
    }
}

// ── Driver global state ────────────────────────────────────────────────────────

static AHCI_PORTS: Mutex<Vec<AhciPort>> = Mutex::new(Vec::new());

/// Probe all PCI AHCI controllers and initialise their ports.
pub fn init(phys_offset: u64) {
    let devices = drivers::pci::enumerate();
    let mut port_count = 0usize;

    for addr in &devices {
        let cls = addr.class_code();
        let sub = addr.subclass();
        if cls != AHCI_CLASS || sub != AHCI_SUBCLASS { continue; }

        addr.enable_bus_master();

        // BAR5 = AHCI HBA memory-mapped base (physical)
        let hba_phys = addr.bar_mmio_base(5);
        if hba_phys == 0 { continue; }
        let hba_va = phys_offset + hba_phys;

        unsafe {
            let ghc = core::ptr::read_volatile((hba_va + HBA_GHC as u64) as *const u32);
            // Enable AHCI mode
            core::ptr::write_volatile(
                (hba_va + HBA_GHC as u64) as *mut u32,
                ghc | GHC_AHCI_ENABLE,
            );

            // Which ports are implemented?
            let pi = core::ptr::read_volatile((hba_va + HBA_PI as u64) as *const u32);
            let cap = core::ptr::read_volatile((hba_va + HBA_CAP as u64) as *const u32);
            let max_ports = ((cap >> 0) & 0x1F) as usize + 1;

            for port_idx in 0..max_ports.min(32) {
                if pi & (1 << port_idx) == 0 { continue; }

                let port_reg = hba_va + PORT_BASE as u64 + port_idx as u64 * PORT_SIZE as u64;
                // Check SATA status: DET=1 and IPM=1
                let ssts = core::ptr::read_volatile((port_reg + P_SSTS as u64) as *const u32);
                let det = ssts & 0x0F;
                let ipm = (ssts >> 8) & 0x0F;
                if det != 3 || ipm != 1 { continue; }

                let sig = core::ptr::read_volatile((port_reg + P_SIG as u64) as *const u32);
                // 0x00000101 = ATA drive, 0xEB140101 = ATAPI
                if sig != 0x0000_0101 { continue; }

                let cmd_list_box: Box<[CommandHeader; 32]> = Box::new(unsafe { core::mem::zeroed() });
                let recv_fis_box: Box<RecvFis> = Box::new(RecvFis { data: [0; 256] });
                let cmd_table_box: Box<CommandTable> = Box::new(unsafe { core::mem::zeroed() });
                let cl_pa  = va_to_pa(cmd_list_box.as_ptr()  as u64, phys_offset);
                let rf_pa  = va_to_pa(&*recv_fis_box          as *const RecvFis as u64, phys_offset);
                let ct_pa  = va_to_pa(&*cmd_table_box         as *const CommandTable as u64, phys_offset);
                let cmd_list = Box::new(unsafe { core::mem::zeroed() });
                let recv_fis = Box::new(RecvFis { data: [0; 256] });
                let cmd_table = Box::new(unsafe { core::mem::zeroed() });
                let cmd_list_pa = va_to_pa(cmd_list.as_ptr() as u64, phys_offset);
                let recv_fis_pa = va_to_pa(&*recv_fis as *const RecvFis as u64, phys_offset);
                let cmd_table_pa = va_to_pa(cmd_table.as_ptr() as u64, phys_offset);

                let mut port = AhciPort {
                    regs_base:    hba_va,
                    port_idx,
                    phys_off:     phys_offset,
                    cmd_list,
                    cmd_list_pa,
                    recv_fis,
                    recv_fis_pa,
                    cmd_table,
                    cmd_table_pa,
                    sectors:   0,
                    model:     [b' '; 40],
                };

                port.stop();
                port.start();

                if port.identify() {
                    let model = core::str::from_utf8(&port.model).unwrap_or("?");
                    let secs  = port.sectors;
                    crate::klog!(INFO,
                        "AHCI: port {} — {} ({} MiB)",
                        port_idx,
                        model.trim(),
                        secs / 2048);
                    AHCI_PORTS.lock().push(port);
                    port_count += 1;
                }
            }
        }
    }

    if port_count > 0 {
        crate::klog!(INFO, "AHCI: {} drive(s) ready", port_count);
    }
}

/// Returns number of detected AHCI drives.
pub fn drive_count() -> usize { AHCI_PORTS.lock().len() }

/// Read `count` 512-byte sectors from drive `drive_idx` at `lba`.
pub fn read_sectors(drive_idx: usize, lba: u64, count: u16) -> Option<Vec<u8>> {
    let mut buf = vec![0u8; count as usize * SECTOR_SIZE];
    let mut ports = AHCI_PORTS.lock();
    let port = ports.get_mut(drive_idx)?;
    unsafe {
        if port.read_sectors(lba, count, &mut buf) {
            Some(buf)
        } else {
            None
        }
    }
}

/// Write `data` (must be a multiple of 512 bytes) to drive `drive_idx` at `lba`.
pub fn write_sectors(drive_idx: usize, lba: u64, data: &[u8]) -> bool {
    let count = (data.len() / SECTOR_SIZE) as u16;
    let mut ports = AHCI_PORTS.lock();
    if let Some(port) = ports.get_mut(drive_idx) {
        unsafe { port.write_sectors(lba, count, data) }
    } else {
        false
    }
}

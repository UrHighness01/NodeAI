//! NVMe driver — Phase 27.
//!
//! Implements:
//!   - PCI discovery  (class 0x01, subclass 0x08, progif 0x02)
//!   - Controller initialisation (CAP / CC / CSTS / AQA / ASQ / ACQ)
//!   - Admin queue: Identify Controller + Identify Namespace
//!   - I/O queue creation
//!   - NVM Read (opcode 0x02) and Write (opcode 0x01)
//!
//! Only one namespace (NSID=1) and one I/O queue are supported for now.

use alloc::{vec, vec::Vec, boxed::Box, string::String, borrow::ToOwned};
use spin::Mutex;

// ── PCI identity ───────────────────────────────────────────────────────────────
const NVME_CLASS:    u8 = 0x01;
const NVME_SUBCLASS: u8 = 0x08;
const NVME_PROGIF:   u8 = 0x02;

// ── Controller registers (BAR0 offsets) ───────────────────────────────────────
const CAP:    u64 = 0x00;
const VS:     u64 = 0x08;
const INTMS:  u64 = 0x0C;
const INTMC:  u64 = 0x10;
const CC:     u64 = 0x14;
const CSTS:   u64 = 0x1C;
const AQA:    u64 = 0x24;
const ASQ:    u64 = 0x28;
const ACQ:    u64 = 0x30;

// CC bits
const CC_EN:    u32 = 1 << 0;
const CC_CSS_NVM: u32 = 0 << 4;
const CC_MPS:   u32 = 0 << 7;   // host page size = 4K (MPSMIN)
const CC_AQS:   u32 = 6 << 16;  // admin submission queue entry = 64 bytes (2^6)
const CC_AQC:   u32 = 4 << 20;  // admin completion queue entry = 16 bytes (2^4)
const CC_IOCQS: u32 = 6 << 20;  // I/O CQ entry = 16 bytes
const CC_IOCSS: u32 = 6 << 16;  // I/O SQ entry = 64 bytes; reused field

const CSTS_RDY: u32 = 1 << 0;

// Queue depths
const ADMIN_QUEUE_DEPTH: usize = 16;
const IO_QUEUE_DEPTH:    usize = 64;
const SQE_SIZE: usize = 64;
const CQE_SIZE: usize = 16;

// ── NVMe SQE / CQE structures ─────────────────────────────────────────────────

#[repr(C)]
struct Sqe {
    cdw0:  u32,   // opcode[7:0] fuse[9:8] psdt[15:14] cid[31:16]
    nsid:  u32,
    cdw2:  u32,
    cdw3:  u32,
    mptr:  u64,
    prp1:  u64,
    prp2:  u64,
    cdw10: u32,
    cdw11: u32,
    cdw12: u32,
    cdw13: u32,
    cdw14: u32,
    cdw15: u32,
}

#[repr(C)]
struct Cqe {
    dw0:   u32,
    dw1:   u32,
    sq_head: u16,
    sq_id:   u16,
    cid:     u16,
    status:  u16,  // bit 0 = phase tag
}

// ── Admin opcodes ──────────────────────────────────────────────────────────────
const ADMIN_DELETE_IOSQ: u8 = 0x00;
const ADMIN_CREATE_IOSQ: u8 = 0x01;
const ADMIN_DELETE_IOCQ: u8 = 0x04;
const ADMIN_CREATE_IOCQ: u8 = 0x05;
const ADMIN_IDENTIFY:    u8 = 0x06;
const ADMIN_SET_FEAT:    u8 = 0x09;

// I/O opcodes
const IO_WRITE: u8 = 0x01;
const IO_READ:  u8 = 0x02;

// ── Controller state ───────────────────────────────────────────────────────────

struct Queue {
    sq:     Box<[Sqe]>,
    cq:     Box<[Cqe]>,
    sq_tail: u16,
    cq_head: u16,
    phase:   u8,
    db_sq:   u64,  // VA of SQ doorbell
    db_cq:   u64,  // VA of CQ doorbell
    next_cid: u16,
}

impl Queue {
    fn new(depth: usize, db_sq: u64, db_cq: u64) -> Self {
        let sq = unsafe {
            let mut v: Vec<Sqe> = Vec::with_capacity(depth);
            v.set_len(depth);
            core::ptr::write_bytes(v.as_mut_ptr(), 0, depth);
            v.into_boxed_slice()
        };
        let cq = unsafe {
            let mut v: Vec<Cqe> = Vec::with_capacity(depth);
            v.set_len(depth);
            core::ptr::write_bytes(v.as_mut_ptr(), 0, depth);
            v.into_boxed_slice()
        };
        Queue { sq, cq, sq_tail: 0, cq_head: 0, phase: 1, db_sq, db_cq, next_cid: 1 }
    }

    fn sq_phys(&self, phys_off: u64) -> u64 {
        self.sq.as_ptr() as u64 - phys_off
    }
    fn cq_phys(&self, phys_off: u64) -> u64 {
        self.cq.as_ptr() as u64 - phys_off
    }

    /// Submit a SQE and ring the tail doorbell. Returns CID.
    unsafe fn submit(&mut self, mut sqe: Sqe) -> u16 {
        let cid = self.next_cid;
        self.next_cid = self.next_cid.wrapping_add(1).max(1);
        sqe.cdw0 = (sqe.cdw0 & 0x0000_FFFF) | ((cid as u32) << 16);
        let slot = self.sq_tail as usize;
        self.sq[slot] = sqe;
        self.sq_tail = (self.sq_tail + 1) % self.sq.len() as u16;
        core::ptr::write_volatile(self.db_sq as *mut u32, self.sq_tail as u32);
        cid
    }

    /// Wait for a completion entry matching `cid`. Returns status word.
    unsafe fn complete(&mut self, cid: u16) -> u16 {
        let deadline = crate::scheduler::uptime_ms() + 5000;
        loop {
            if crate::scheduler::uptime_ms() >= deadline { return 0xFFFF; }
            let cqe = &self.cq[self.cq_head as usize];
            let phase_ok = (cqe.status & 1) == self.phase as u16;
            if phase_ok && cqe.cid == cid {
                let status = cqe.status >> 1;
                self.cq_head = (self.cq_head + 1) % self.cq.len() as u16;
                if self.cq_head == 0 { self.phase ^= 1; }
                // Ring CQ head doorbell
                core::ptr::write_volatile(self.db_cq as *mut u32, self.cq_head as u32);
                return status;
            }
            core::hint::spin_loop();
        }
    }
}

struct NvmeCtrl {
    base:       u64,   // Controller BAR0 VA
    phys_off:   u64,
    admin_q:    Queue,
    io_q:       Queue,
    ns_blocks:  u64,
    block_shift: u32,
    lbads:      u32,   // log2(block size)
}

impl NvmeCtrl {
    unsafe fn r32(&self, off: u64) -> u32 {
        core::ptr::read_volatile((self.base + off) as *const u32)
    }
    unsafe fn w32(&self, off: u64, v: u32) {
        core::ptr::write_volatile((self.base + off) as *mut u32, v)
    }
    unsafe fn r64(&self, off: u64) -> u64 {
        core::ptr::read_volatile((self.base + off) as *const u64)
    }
    unsafe fn w64(&self, off: u64, v: u64) {
        core::ptr::write_volatile((self.base + off) as *mut u64, v)
    }

    /// Stride between doorbell registers (in bytes = 4 << dstrd).
    unsafe fn doorbell_step(&self) -> u64 {
        let cap = self.r64(CAP);
        let dstrd = ((cap >> 32) & 0xF) as u64;
        4u64 << dstrd
    }
    unsafe fn sq_db(&self, qid: u16) -> u64 {
        let step = self.doorbell_step();
        self.base + 0x1000 + (2 * qid as u64) * step
    }
    unsafe fn cq_db(&self, qid: u16) -> u64 {
        let step = self.doorbell_step();
        self.base + 0x1000 + (2 * qid as u64 + 1) * step
    }

    /// Admin identify (CNS=1 → controller, CNS=0 → namespace NSID=1).
    unsafe fn identify(&mut self, cns: u32, nsid: u32) -> Option<Vec<u8>> {
        let mut buf = vec![0u8; 4096];
        let prp1 = buf.as_ptr() as u64 - self.phys_off;
        let sqe = Sqe {
            cdw0:  ADMIN_IDENTIFY as u32,
            nsid,
            prp1,
            cdw10: cns,
            ..unsafe { core::mem::zeroed() }
        };
        let cid = self.admin_q.submit(sqe);
        let st = self.admin_q.complete(cid);
        if st == 0 { Some(buf) } else { None }
    }

    /// Create I/O completion queue (admin opcode 0x05).
    unsafe fn create_iocq(&mut self) -> bool {
        let cq_phys = self.io_q.cq_phys(self.phys_off);
        let sqe = Sqe {
            cdw0:  ADMIN_CREATE_IOCQ as u32,
            prp1:  cq_phys,
            cdw10: ((IO_QUEUE_DEPTH as u32 - 1) << 16) | 1, // QSIZE | QID=1
            cdw11: 1,  // PC=1 (contiguous), IEN=0
            ..unsafe { core::mem::zeroed() }
        };
        let cid = self.admin_q.submit(sqe);
        self.admin_q.complete(cid) == 0
    }

    /// Create I/O submission queue (admin opcode 0x01).
    unsafe fn create_iosq(&mut self) -> bool {
        let sq_phys = self.io_q.sq_phys(self.phys_off);
        let sqe = Sqe {
            cdw0:  ADMIN_CREATE_IOSQ as u32,
            prp1:  sq_phys,
            cdw10: ((IO_QUEUE_DEPTH as u32 - 1) << 16) | 1, // QSIZE | QID=1
            cdw11: (1 << 16) | 1,  // CQID=1, PC=1
            ..unsafe { core::mem::zeroed() }
        };
        let cid = self.admin_q.submit(sqe);
        self.admin_q.complete(cid) == 0
    }

    /// Read `nlba` logical blocks from `slba` into `buf`.
    unsafe fn read_blocks(&mut self, slba: u64, nlba: u16, buf: &mut [u8]) -> bool {
        let prp1 = buf.as_ptr() as u64 - self.phys_off;
        let sqe = Sqe {
            cdw0:  IO_READ as u32,
            nsid:  1,
            prp1,
            cdw10: slba as u32,
            cdw11: (slba >> 32) as u32,
            cdw12: nlba as u32 - 1,
            ..unsafe { core::mem::zeroed() }
        };
        let cid = self.io_q.submit(sqe);
        self.io_q.complete(cid) == 0
    }

    /// Write `nlba` logical blocks from `buf` to `slba`.
    unsafe fn write_blocks(&mut self, slba: u64, nlba: u16, buf: &[u8]) -> bool {
        let prp1 = buf.as_ptr() as u64 - self.phys_off;
        let sqe = Sqe {
            cdw0:  IO_WRITE as u32,
            nsid:  1,
            prp1,
            cdw10: slba as u32,
            cdw11: (slba >> 32) as u32,
            cdw12: nlba as u32 - 1,
            ..unsafe { core::mem::zeroed() }
        };
        let cid = self.io_q.submit(sqe);
        self.io_q.complete(cid) == 0
    }
}

// ── Global state ───────────────────────────────────────────────────────────────

static NVME_CTRLS: Mutex<Vec<NvmeCtrl>> = Mutex::new(Vec::new());

/// Probe PCI for NVMe controllers and initialise them.
pub fn init(phys_offset: u64) {
    let devices = drivers::pci::enumerate();
    let mut count = 0usize;

    for addr in &devices {
        if addr.class_code() != NVME_CLASS || addr.subclass() != NVME_SUBCLASS {
            continue;
        }

        addr.enable_bus_master();

        let bar0_phys = addr.bar_mmio_base(0);
        if bar0_phys == 0 { continue; }
        let base = phys_offset + bar0_phys;

        unsafe {
            // Disable controller before init
            let cur_cc = core::ptr::read_volatile((base + CC) as *const u32);
            core::ptr::write_volatile((base + CC) as *mut u32, cur_cc & !CC_EN);
            let deadline = crate::scheduler::uptime_ms() + 500;
            loop {
                let csts = core::ptr::read_volatile((base + CSTS) as *const u32);
                if csts & CSTS_RDY == 0 { break; }
                if crate::scheduler::uptime_ms() >= deadline { break; }
                core::hint::spin_loop();
            }

            // Compute doorbell step from CAP
            let cap = core::ptr::read_volatile((base + CAP) as *const u64);
            let dstrd = ((cap >> 32) & 0xF) as u64;
            let db_step = 4u64 << dstrd;
            let db_asq = base + 0x1000;                  // Admin SQ doorbell (QID=0)
            let db_acq = base + 0x1000 + db_step;        // Admin CQ doorbell

            // Allocate admin queues
            let admin_sq_mem: Box<[Sqe]> = {
                let mut v: Vec<Sqe> = Vec::with_capacity(ADMIN_QUEUE_DEPTH);
                v.set_len(ADMIN_QUEUE_DEPTH);
                core::ptr::write_bytes(v.as_mut_ptr(), 0, ADMIN_QUEUE_DEPTH);
                v.into_boxed_slice()
            };
            let admin_cq_mem: Box<[Cqe]> = {
                let mut v: Vec<Cqe> = Vec::with_capacity(ADMIN_QUEUE_DEPTH);
                v.set_len(ADMIN_QUEUE_DEPTH);
                core::ptr::write_bytes(v.as_mut_ptr(), 0, ADMIN_QUEUE_DEPTH);
                v.into_boxed_slice()
            };
            let asq_phys = admin_sq_mem.as_ptr() as u64 - phys_offset;
            let acq_phys = admin_cq_mem.as_ptr() as u64 - phys_offset;

            let admin_q = Queue {
                sq: admin_sq_mem,
                cq: admin_cq_mem,
                sq_tail: 0, cq_head: 0, phase: 1,
                db_sq: db_asq, db_cq: db_acq, next_cid: 1,
            };

            // Allocate I/O queues
            let db_iosq = base + 0x1000 + 2 * db_step;
            let db_iocq = base + 0x1000 + 3 * db_step;
            let io_q = Queue::new(IO_QUEUE_DEPTH, db_iosq, db_iocq);

            // Configure admin queue depth + base addresses
            core::ptr::write_volatile(
                (base + AQA) as *mut u32,
                ((ADMIN_QUEUE_DEPTH as u32 - 1) << 16) | (ADMIN_QUEUE_DEPTH as u32 - 1),
            );
            core::ptr::write_volatile((base + ASQ) as *mut u64, asq_phys);
            core::ptr::write_volatile((base + ACQ) as *mut u64, acq_phys);

            // Enable controller
            core::ptr::write_volatile(
                (base + CC) as *mut u32,
                CC_EN | CC_CSS_NVM | CC_MPS | (6 << 16) | (4 << 20),
            );
            let deadline = crate::scheduler::uptime_ms() + 2000;
            loop {
                let csts = core::ptr::read_volatile((base + CSTS) as *const u32);
                if csts & CSTS_RDY != 0 { break; }
                if crate::scheduler::uptime_ms() >= deadline { continue; }
                core::hint::spin_loop();
            }

            let mut ctrl = NvmeCtrl {
                base,
                phys_off: phys_offset,
                admin_q,
                io_q,
                ns_blocks: 0,
                block_shift: 9,
                lbads: 9,
            };

            // Identify controller to get model string
            let model_s = if let Some(id_ctrl) = ctrl.identify(1, 0) {
                let mn = &id_ctrl[24..64];
                core::str::from_utf8(mn).unwrap_or("?").trim().to_owned()
            } else {
                String::from("Unknown NVMe")
            };

            // Identify namespace 1
            if let Some(id_ns) = ctrl.identify(0, 1) {
                let mut ns_bytes = [0u8; 8];
                ns_bytes.copy_from_slice(&id_ns[0..8]);
                ctrl.ns_blocks = u64::from_le_bytes(ns_bytes);
                // LBAF[0] (bytes 128..132): lbads at byte 3 of each LBAF entry (shift)
                let lbads = id_ns[130] & 0xFF;
                ctrl.lbads = lbads as u32;
                ctrl.block_shift = lbads as u32;
            }

            // Create I/O queues
            ctrl.create_iocq();
            ctrl.create_iosq();

            let gib = (ctrl.ns_blocks << ctrl.lbads) >> 30;
            crate::klog!(INFO, "NVMe: {} — {} GiB (block_shift={})", model_s, gib, ctrl.lbads);
            NVME_CTRLS.lock().push(ctrl);
            count += 1;
        }
    }

    if count > 0 {
        crate::klog!(INFO, "NVMe: {} controller(s) ready", count);
    }
}

/// Number of NVMe controllers found.
pub fn ctrl_count() -> usize { NVME_CTRLS.lock().len() }

/// Read `count` blocks from controller `idx` namespace 1 starting at `lba`.
pub fn read_blocks(idx: usize, lba: u64, count: u16) -> Option<Vec<u8>> {
    let mut ctrls = NVME_CTRLS.lock();
    let ctrl = ctrls.get_mut(idx)?;
    let block_size = 1usize << ctrl.lbads;
    let mut buf = vec![0u8; count as usize * block_size];
    unsafe {
        if ctrl.read_blocks(lba, count, &mut buf) {
            Some(buf)
        } else {
            None
        }
    }
}

/// Write `data` to controller `idx` namespace 1 at `lba`.
pub fn write_blocks(idx: usize, lba: u64, data: &[u8]) -> bool {
    let mut ctrls = NVME_CTRLS.lock();
    if let Some(ctrl) = ctrls.get_mut(idx) {
        let block_size = 1usize << ctrl.lbads;
        let count = (data.len() / block_size) as u16;
        unsafe { ctrl.write_blocks(lba, count, data) }
    } else {
        false
    }
}

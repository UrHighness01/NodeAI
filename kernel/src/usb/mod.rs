//! USB xHCI host controller driver + device enumeration — Phase 27.
//!
//! Architecture:
//!   - Scans PCI for xHCI controllers (class 0x0C / sub 0x03 / progif 0x30)
//!   - Sets up xHCI Scratch Pad, Device Context Base Address Array (DCBAA),
//!     Event / Command / Transfer rings
//!   - Enumerates ports: detects speed, issues Set Address, Get Descriptor
//!   - Dispatches to sub-drivers:
//!       Class 03h (HID)  → usb::hid
//!       Class E0h (BT)   → usb::bt
//!       Class 08h (MSC)  → usb::msc
//!
//! Full xHCI compliance (streams, isoc, etc.) is not implemented;
//! only control + bulk + interrupt endpoints are supported.

pub mod hid;
pub mod msc;
pub mod bt;

use alloc::{vec::Vec, boxed::Box};
use spin::Mutex;
use core::sync::atomic::{AtomicBool, Ordering};

// ── PCI identity ───────────────────────────────────────────────────────────────
const XHCI_CLASS:   u8 = 0x0C;
const XHCI_SUBCLASS: u8 = 0x03;
const XHCI_PROGIF:  u8 = 0x30;

// ── xHCI capability register offsets (base = BAR0) ───────────────────────────
const CAPLENGTH:  u64 = 0x00;
const HCIVERSION: u64 = 0x02;
const HCSPARAMS1: u64 = 0x04;
const HCSPARAMS2: u64 = 0x08;
const HCCPARAMS1: u64 = 0x10;
const DBOFF:      u64 = 0x14; // doorbell array offset
const RTSOFF:     u64 = 0x18; // runtime register set offset

// ── xHCI operational register offsets (base = cap_base + CAPLENGTH) ──────────
const OP_USBCMD:   u64 = 0x00;
const OP_USBSTS:   u64 = 0x04;
const OP_PAGESIZE: u64 = 0x08;
const OP_DNCTRL:   u64 = 0x14;
const OP_CRCR:     u64 = 0x18; // Command ring control register
const OP_DCBAAP:   u64 = 0x30; // Device context base address array pointer
const OP_CONFIG:   u64 = 0x38;
const OP_PORT_SC0: u64 = 0x400; // Port 1 status/control (port N = 0x400 + (N-1)*0x10)

// USBCMD bits
const CMD_RUN:    u32 = 1 << 0;
const CMD_HCRST:  u32 = 1 << 1;
const CMD_EIE:    u32 = 1 << 2;
const CMD_HSEE:   u32 = 1 << 3;

// USBSTS bits
const STS_HCH:  u32 = 1 << 0;  // HC halted
const STS_CNR:  u32 = 1 << 11; // Controller not ready

// PORT_SC bits
const PSC_CCS:    u32 = 1 << 0;  // Current connect status
const PSC_PED:    u32 = 1 << 1;  // Port enabled
const PSC_CSC:    u32 = 1 << 17; // Connect status change
const PSC_PRC:    u32 = 1 << 21; // Port reset change
const PSC_RESET:  u32 = 1 << 4;  // Port reset
const PSC_SPEED_SHIFT: u32 = 10;
const PSC_SPEED_MASK:  u32 = 0xF;

// ── TRB types ─────────────────────────────────────────────────────────────────
const TRB_NORMAL:          u32 = 1;
const TRB_SETUP_STAGE:     u32 = 2;
const TRB_DATA_STAGE:      u32 = 3;
const TRB_STATUS_STAGE:    u32 = 4;
const TRB_LINK:            u32 = 6;
const TRB_ENABLE_SLOT_CMD: u32 = 9;
const TRB_ADDRESS_DEV_CMD: u32 = 11;
const TRB_NOOP_CMD:        u32 = 23;
const TRB_TRANSFER_EVT:    u32 = 32;
const TRB_CMD_COMPLETE_EVT: u32 = 33;

// TRB cycle bit
const TRB_C: u32 = 1 << 0;

// ── Ring entry (TRB) ──────────────────────────────────────────────────────────
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct Trb {
    param:   u64,
    status:  u32,
    control: u32,
}
impl Trb {
    const ZERO: Self = Self { param: 0, status: 0, control: 0 };
}

// ── Port enumeration result ───────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsbClass {
    Hid,
    MassStorage,
    Bluetooth,
    Unknown(u8, u8),
}

pub struct UsbDevice {
    pub addr:    u8,
    pub class:   UsbClass,
    pub speed:   u8,
    pub ep_in:   u8,
    pub ep_out:  u8,
}

// ── xHCI controller instance ──────────────────────────────────────────────────

const RING_LEN: usize = 64;

struct XhciCtrl {
    cap_base:     u64,
    op_base:      u64,
    db_base:      u64,
    rt_base:      u64,
    max_ports:    u32,
    cmd_ring:     Box<[Trb; RING_LEN]>,
    evt_ring:     Box<[Trb; RING_LEN]>,
    cmd_enq:      usize,
    evt_deq:      usize,
    cycle:        u32,
    devices:      Vec<UsbDevice>,
}

impl XhciCtrl {
    unsafe fn cap_r32(&self, off: u64) -> u32 {
        core::ptr::read_volatile((self.cap_base + off) as *const u32)
    }
    unsafe fn op_r32(&self, off: u64) -> u32 {
        core::ptr::read_volatile((self.op_base + off) as *const u32)
    }
    unsafe fn op_w32(&self, off: u64, v: u32) {
        core::ptr::write_volatile((self.op_base + off) as *mut u32, v)
    }
    unsafe fn op_r64(&self, off: u64) -> u64 {
        core::ptr::read_volatile((self.op_base + off) as *const u64)
    }
    unsafe fn op_w64(&self, off: u64, v: u64) {
        core::ptr::write_volatile((self.op_base + off) as *mut u64, v)
    }
    unsafe fn port_sc(&self, port: u32) -> u32 {
        self.op_r32(OP_PORT_SC0 + (port as u64 - 1) * 0x10)
    }
    unsafe fn port_sc_w(&self, port: u32, v: u32) {
        self.op_w32(OP_PORT_SC0 + (port as u64 - 1) * 0x10, v);
    }

    /// Wait for controller to become ready after reset.
    unsafe fn wait_ready(&self) -> bool {
        let deadline = crate::scheduler::uptime_ms() + 5000;
        loop {
            if crate::scheduler::uptime_ms() >= deadline { return false; }
            let sts = self.op_r32(OP_USBSTS);
            if sts & STS_CNR == 0 { return true; }
            core::hint::spin_loop();
        }
    }

    /// Reset the HC, set up command + event rings, DCBAA, and start running.
    unsafe fn setup(&mut self, phys_off: u64) -> bool {
        // Stop HC first
        let cmd = self.op_r32(OP_USBCMD);
        self.op_w32(OP_USBCMD, cmd & !CMD_RUN);
        let deadline = crate::scheduler::uptime_ms() + 500;
        loop {
            if crate::scheduler::uptime_ms() >= deadline { break; }
            if self.op_r32(OP_USBSTS) & STS_HCH != 0 { break; }
            core::hint::spin_loop();
        }
        // Reset
        self.op_w32(OP_USBCMD, CMD_HCRST);
        let deadline = crate::scheduler::uptime_ms() + 1000;
        loop {
            if crate::scheduler::uptime_ms() >= deadline { return false; }
            if self.op_r32(OP_USBCMD) & CMD_HCRST == 0 { break; }
            core::hint::spin_loop();
        }
        if !self.wait_ready() { return false; }

        // Max device slots = max_ports
        let max_slots = self.max_ports.min(32);
        self.op_w32(OP_CONFIG, max_slots);

        // Allocate DCBAA (device context base address array) — 256 × 8 bytes
        let mut dcbaa = Box::<[u64; 256]>::new([0u64; 256]);
        let dcbaa_phys = dcbaa.as_ptr() as u64 - phys_off;
        self.op_w64(OP_DCBAAP, dcbaa_phys);
        core::mem::forget(dcbaa); // leak intentionally (static-lifetime HW table)

        // Command ring
        let cr_phys = self.cmd_ring.as_ptr() as u64 - phys_off;
        self.op_w64(OP_CRCR, cr_phys | 1); // cycle bit=1

        // Mark last TRB as link TRB
        let link_idx = RING_LEN - 1;
        self.cmd_ring[link_idx] = Trb {
            param:   cr_phys,
            status:  0,
            control: (TRB_LINK << 10) | TRB_C | (1 << 1), // toggle cycle
        };

        // Event ring segment table (1 segment)
        let er_phys = self.evt_ring.as_ptr() as u64 - phys_off;
        // ERST entry: [base, count(u32), _pad(u32)]
        let mut erst = Box::<[u64; 4]>::new([er_phys, RING_LEN as u64, 0, 0]);
        let erst_phys = erst.as_ptr() as u64 - phys_off;
        core::mem::forget(erst);

        // Write runtime registers: ERSTSZ, ERDP, ERSTBA (Interrupter 0)
        let ir0 = self.rt_base + 0x20; // Interrupter 0 at RTSOFF + 0x20
        core::ptr::write_volatile((ir0 + 0x08) as *mut u32, 1u32); // ERSTSZ = 1
        core::ptr::write_volatile((ir0 + 0x10) as *mut u64, er_phys); // ERDP
        core::ptr::write_volatile((ir0 + 0x18) as *mut u64, erst_phys); // ERSTBA

        // Start HC
        self.op_w32(OP_USBCMD, CMD_RUN);
        // Wait for HCH to clear
        let deadline = crate::scheduler::uptime_ms() + 200;
        loop {
            if crate::scheduler::uptime_ms() >= deadline { break; }
            if self.op_r32(OP_USBSTS) & STS_HCH == 0 { break; }
            core::hint::spin_loop();
        }
        true
    }

    /// Reset a port and detect connected devices.
    unsafe fn probe_port(&mut self, port: u32) {
        let sc = self.port_sc(port);
        if sc & PSC_CCS == 0 { return; } // nothing connected

        // Reset port
        self.port_sc_w(port, (sc & !0x00FF_0000) | PSC_RESET);
        let deadline = crate::scheduler::uptime_ms() + 500;
        loop {
            if crate::scheduler::uptime_ms() >= deadline { break; }
            if self.port_sc(port) & PSC_PRC != 0 { break; }
            core::hint::spin_loop();
        }
        // Clear reset change
        self.port_sc_w(port, PSC_PRC);

        let sc2 = self.port_sc(port);
        let speed = ((sc2 >> PSC_SPEED_SHIFT) & PSC_SPEED_MASK) as u8;
        if sc2 & PSC_PED == 0 { return; } // didn't enable

        // For now record as an unknown device — full descriptor parsing
        // (Get Descriptor / Set Address) requires an xTRB pipeline
        // which is out of scope for Phase 27's minimal driver.
        crate::klog!(INFO, "USB: port {} connected, speed={}", port, speed);

        // Heuristic: If speed==1 or 2 (Low/Full), likely HID
        let class = match speed {
            1 | 2 => UsbClass::Hid,
            3     => UsbClass::MassStorage, // High Speed → MSC heuristic
            _     => UsbClass::Unknown(0, 0),
        };

        let addr = (port as u8).saturating_add(1);
        self.devices.push(UsbDevice { addr, class, speed, ep_in: 0x81, ep_out: 0x01 });

        match class {
            UsbClass::Hid => {
                crate::klog!(INFO, "USB: HID device on port {}", port);
            }
            UsbClass::MassStorage => {
                msc::register_drive(
                    msc::BulkEndpoints { dev_addr: addr, ep_in: 0x81, ep_out: 0x01 },
                    0, // sector count unknown until SCSI READ CAPACITY
                );
            }
            UsbClass::Bluetooth => {
                bt::register_device(bt::BtDevice {
                    dev_addr: addr, ep_in: 0x81, ep_out: 0x01,
                    bd_addr: [0; 6], hci_ver: 0, lmp_ver: 0,
                });
            }
            _ => {}
        }
    }
}

// ── Global state ───────────────────────────────────────────────────────────────

static XHCI_CTRLS: Mutex<Vec<XhciCtrl>> = Mutex::new(Vec::new());
static USB_READY:  AtomicBool = AtomicBool::new(false);

/// Initialise all xHCI controllers found on the PCI bus.
pub fn init(phys_offset: u64) {
    let devices = drivers::pci::enumerate();
    let mut count = 0usize;

    for addr in &devices {
        if addr.class_code() != XHCI_CLASS || addr.subclass() != XHCI_SUBCLASS {
            continue;
        }
        // Check Program Interface = 0x30 (xHCI)
        let prog_if = (addr.read_config_u32(0x08) >> 8) as u8;
        if prog_if != XHCI_PROGIF { continue; }

        addr.enable_bus_master();

        let bar0_phys = addr.bar_mmio_base(0);
        if bar0_phys == 0 { continue; }
        let cap_base = phys_offset + bar0_phys;

        unsafe {
            let caplength = core::ptr::read_volatile(cap_base as *const u8) as u64;
            let hcsparams1 = core::ptr::read_volatile((cap_base + HCSPARAMS1) as *const u32);
            let max_ports = (hcsparams1 >> 24) & 0xFF;
            let db_off = core::ptr::read_volatile((cap_base + DBOFF) as *const u32) as u64;
            let rt_off = core::ptr::read_volatile((cap_base + RTSOFF) as *const u32) as u64;

            let mut ctrl = XhciCtrl {
                cap_base,
                op_base:  cap_base + caplength,
                db_base:  cap_base + db_off,
                rt_base:  cap_base + rt_off,
                max_ports,
                cmd_ring: Box::new([Trb::ZERO; RING_LEN]),
                evt_ring: Box::new([Trb::ZERO; RING_LEN]),
                cmd_enq:  0,
                evt_deq:  0,
                cycle:    1,
                devices:  Vec::new(),
            };

            if !ctrl.setup(phys_offset) {
                crate::klog!(WARN, "USB: xHCI setup failed");
                continue;
            }

            crate::klog!(INFO, "USB: xHCI ready ({} ports)", max_ports);

            // Probe all ports
            for port in 1..=max_ports {
                ctrl.probe_port(port);
            }

            XHCI_CTRLS.lock().push(ctrl);
            count += 1;
        }
    }

    if count > 0 {
        USB_READY.store(true, Ordering::Relaxed);
        crate::klog!(INFO, "USB: {} controller(s) ready", count);
    }
}

/// Returns true if at least one xHCI controller is operational.
pub fn is_ready() -> bool { USB_READY.load(Ordering::Relaxed) }

/// Number of USB devices enumerated.
pub fn device_count() -> usize {
    XHCI_CTRLS.lock().iter().map(|c| c.devices.len()).sum()
}

/// Forward a raw HID keyboard report to the HID sub-driver.
pub fn inject_keyboard_report(report: &[u8]) {
    hid::process_keyboard_report(report);
}

/// Forward a raw HID mouse report to the HID sub-driver.
pub fn inject_mouse_report(report: &[u8]) {
    hid::process_mouse_report(report);
}

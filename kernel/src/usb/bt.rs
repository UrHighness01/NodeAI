//! Bluetooth HCI stub (enumeration only, no data transfer).
//!
//! Provides a minimal HCI command/event framework for USB Bluetooth dongles
//! (class 0xE0 / sub 0x01 / progif 0x01).  Full L2CAP and profile support
//! is not yet implemented; this layer handles:
//!
//!   - HCI Reset
//!   - HCI Read Local Version Information
//!   - HCI LE Set Advertising Data (basic BLE advertisement)
//!   - Event dispatch to registered listeners

use alloc::vec::Vec;
use spin::Mutex;

// ── HCI packet types ──────────────────────────────────────────────────────────
const HCI_COMMAND_PACKET: u8 = 0x01;
const HCI_ACL_PACKET:     u8 = 0x02;
const HCI_EVENT_PACKET:   u8 = 0x04;

// ── HCI opcodes (OGF << 10 | OCF) ────────────────────────────────────────────
const HCI_RESET:                       u16 = 0x0C03;
const HCI_READ_LOCAL_VERSION:          u16 = 0x1001;
const HCI_LE_SET_ADV_DATA:             u16 = 0x2008;
const HCI_LE_SET_ADV_ENABLE:           u16 = 0x200A;

// ── HCI events ─────────────────────────────────────────────────────────────────
const EVT_COMMAND_COMPLETE: u8 = 0x0E;
const EVT_COMMAND_STATUS:   u8 = 0x0F;
const EVT_LE_META:          u8 = 0x3E;

// ── Device state ──────────────────────────────────────────────────────────────

pub struct BtDevice {
    pub dev_addr:  u8,   // USB device address
    pub ep_in:     u8,   // Interrupt IN endpoint (HCI events)
    pub ep_out:    u8,   // Bulk OUT endpoint (HCI commands / ACL data)
    pub bd_addr:   [u8; 6],
    pub hci_ver:   u8,
    pub lmp_ver:   u8,
}

struct BtState {
    devices: Vec<BtDevice>,
    /// Raw event log (last 64 events, 255 bytes each max)
    event_log: Vec<Vec<u8>>,
}

static BT_STATE: Mutex<BtState> = Mutex::new(BtState {
    devices:   Vec::new(),
    event_log: Vec::new(),
});

// ── HCI frame builder ─────────────────────────────────────────────────────────

/// Build a HCI command packet: [type=0x01, opcode_lo, opcode_hi, param_len, params...]
pub fn build_hci_command(opcode: u16, params: &[u8]) -> Vec<u8> {
    let mut pkt = Vec::with_capacity(4 + params.len());
    pkt.push(HCI_COMMAND_PACKET);
    pkt.push(opcode as u8);
    pkt.push((opcode >> 8) as u8);
    pkt.push(params.len() as u8);
    pkt.extend_from_slice(params);
    pkt
}

/// Build a HCI Reset command.
pub fn cmd_reset() -> Vec<u8> { build_hci_command(HCI_RESET, &[]) }

/// Build a HCI Read Local Version command.
pub fn cmd_read_local_version() -> Vec<u8> { build_hci_command(HCI_READ_LOCAL_VERSION, &[]) }

/// Build HCI LE Set Advertising Data (31-byte AD payload).
pub fn cmd_le_set_adv_data(ad: &[u8; 31]) -> Vec<u8> {
    let mut params = [0u8; 32];
    params[0] = 31;
    params[1..].copy_from_slice(ad);
    build_hci_command(HCI_LE_SET_ADV_DATA, &params)
}

/// Build HCI LE Set Advertising Enable.
pub fn cmd_le_set_adv_enable(enable: bool) -> Vec<u8> {
    build_hci_command(HCI_LE_SET_ADV_ENABLE, &[enable as u8])
}

// ── Event processing ──────────────────────────────────────────────────────────

/// Process a raw HCI event packet received from the device.
pub fn process_event(dev_idx: usize, pkt: &[u8]) {
    if pkt.len() < 3 { return; }
    let evt_code = pkt[1];
    let _param_len = pkt[2] as usize;
    let params = if pkt.len() > 3 { &pkt[3..] } else { &[] };

    match evt_code {
        EVT_COMMAND_COMPLETE => {
            if params.len() >= 3 {
                let opcode = u16::from_le_bytes([params[1], params[2]]);
                let status = params[3];
                crate::klog!(DEBUG, "BT[{}]: cmd_complete op={:#06X} st={}", dev_idx, opcode, status);

                // If this was the reset response, read version
                if opcode == HCI_RESET && status == 0 {
                    // Next step: send Read Local Version
                }
                if opcode == HCI_READ_LOCAL_VERSION && status == 0 && params.len() >= 9 {
                    let hci_ver  = params[4];
                    let lmp_ver  = params[8];
                    let mut state = BT_STATE.lock();
                    if let Some(dev) = state.devices.get_mut(dev_idx) {
                        dev.hci_ver = hci_ver;
                        dev.lmp_ver = lmp_ver;
                    }
                    crate::klog!(INFO, "BT[{}]: HCI v{} LMP v{}", dev_idx, hci_ver, lmp_ver);
                }
            }
        }
        EVT_COMMAND_STATUS => {
            if params.len() >= 1 {
                let status = params[0];
                if status != 0 {
                    crate::klog!(WARN, "BT[{}]: cmd_status error {}", dev_idx, status);
                }
            }
        }
        EVT_LE_META => {
            if params.len() >= 1 {
                let sub = params[0];
                crate::klog!(DEBUG, "BT[{}]: LE meta subevent={}", dev_idx, sub);
            }
        }
        _ => {}
    }

    // Log raw event
    let mut state = BT_STATE.lock();
    if state.event_log.len() >= 64 { state.event_log.remove(0); }
    state.event_log.push(pkt.to_vec());
}

// ── Public API ─────────────────────────────────────────────────────────────────

/// Register a USB Bluetooth device (called by usb/mod.rs on enumeration).
pub fn register_device(dev: BtDevice) {
    crate::klog!(INFO, "BT: device registered (USB addr {})", dev.dev_addr);
    BT_STATE.lock().devices.push(dev);
}

/// Number of registered BT devices.
pub fn device_count() -> usize { BT_STATE.lock().devices.len() }

/// Get BD_ADDR of device `idx`.
pub fn bd_addr(idx: usize) -> Option<[u8; 6]> {
    BT_STATE.lock().devices.get(idx).map(|d| d.bd_addr)
}

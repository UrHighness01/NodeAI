//! Intel iwlwifi driver — device probe, registers, PCI config. Tx/Rx stubbed.
//!
//! Handles the PCIe side of Intel WiFi adapters (iwlwifi family):
//!   - PCI detection by device ID list (vendor 0x8086, class 0x02 / sub 0x80)
//!   - CSR register map (BAR0 MMIO)
//!   - Load firmware from VFS `/lib/firmware/iwlwifi.ucode`
//!   - Basic initialisation sequence (APMG, ICT)
//!
//! The 802.11 MAC layer and association state machine are in `wifi/mod.rs`.
//! Full tx/rx DMA rings are still stubbed (device enumeration only).

use alloc::vec::Vec;
use spin::Mutex;
use core::sync::atomic::{AtomicBool, Ordering};

const INTEL_VENDOR: u16 = 0x8086;
// Common Intel WiFi device IDs (partial list)
const IWLWIFI_DEV_IDS: &[u16] = &[
    0x095A, 0x095B, // Intel Wireless 7265
    0x24F3, 0x24F4, // Intel Wireless 8260
    0x2526, 0x2723, // Intel Wi-Fi 6 AX200/AX201
    0x7A70, 0x7A74, // Intel Wi-Fi 6E AX211
    0x51F0, 0x51F1, // Intel Wi-Fi 6 AX201 (Ice Lake)
];

// ── CSR registers (BAR0 offsets) ──────────────────────────────────────────────
const CSR_HW_IF_CONFIG_REG:   u32 = 0x000;
const CSR_INT_COALESCING:     u32 = 0x004;
const CSR_INT:                u32 = 0x008;
const CSR_INT_MASK:           u32 = 0x00C;
const CSR_FH_INT_STATUS:      u32 = 0x010;
const CSR_GPIO_IN:            u32 = 0x018;
const CSR_RESET:              u32 = 0x020;
const CSR_GP_CNTRL:           u32 = 0x024;
const CSR_HW_REV:             u32 = 0x028;
const CSR_UCODE_DRV_GP1:      u32 = 0x054;
const CSR_UCODE_DRV_GP1_SET:  u32 = 0x058;
const CSR_UCODE_DRV_GP1_CLR:  u32 = 0x05C;
const CSR_APMG_CLK_CTRL:      u32 = 0x068;
const CSR_APMG_PS_CTRL:       u32 = 0x06C;

// CSR_RESET bits
const CSR_RESET_REG_FLAG_SW_RESET:  u32 = 1 << 7;
const CSR_RESET_REG_FLAG_NEVO_RESET: u32 = 1 << 31;

// CSR_GP_CNTRL bits
const CSR_GP_CNTRL_REG_FLAG_INIT_DONE: u32 = 1 << 9;
const CSR_GP_CNTRL_REG_FLAG_MAC_CLOCK_READY: u32 = 1 << 0;

// ── Device instance ───────────────────────────────────────────────────────────

pub struct IwlDevice {
    pub mmio_base: u64,
    pub dev_id:    u16,
    pub hw_rev:    u32,
    pub fw_loaded: bool,
}

impl IwlDevice {
    unsafe fn r32(&self, reg: u32) -> u32 {
        core::ptr::read_volatile((self.mmio_base + reg as u64) as *const u32)
    }
    unsafe fn w32(&self, reg: u32, v: u32) {
        core::ptr::write_volatile((self.mmio_base + reg as u64) as *mut u32, v)
    }

    /// Software reset the device.
    unsafe fn sw_reset(&self) {
        self.w32(CSR_RESET, CSR_RESET_REG_FLAG_SW_RESET);
        // Short busy-wait (~10ms) for reset to take effect
        let deadline = crate::scheduler::uptime_ms() + 10;
        while crate::scheduler::uptime_ms() < deadline { core::hint::spin_loop(); }
    }

    /// Enable APMG (Advanced Power Management for Gen) clock.
    unsafe fn apmg_enable(&self) {
        self.w32(CSR_APMG_CLK_CTRL, 0x0000_0002); // DMA clock enable
        let ps = self.r32(CSR_APMG_PS_CTRL);
        self.w32(CSR_APMG_PS_CTRL, ps & !(1 << 1)); // clear SLP_CTRL
    }

    /// Wait for MAC clock ready.
    unsafe fn wait_mac_ready(&self) -> bool {
        let deadline = crate::scheduler::uptime_ms() + 200;
        loop {
            if crate::scheduler::uptime_ms() >= deadline { return false; }
            if self.r32(CSR_GP_CNTRL) & CSR_GP_CNTRL_REG_FLAG_MAC_CLOCK_READY != 0 {
                return true;
            }
            core::hint::spin_loop();
        }
    }

    /// Load firmware from VFS `/lib/firmware/iwlwifi.ucode`.
    /// Returns true if firmware was found and parsed (loading into device deferred).
    fn load_firmware(&mut self) -> bool {
        match crate::vfs::read_file("/lib/firmware/iwlwifi.ucode") {
            Ok(data) if data.len() > 64 => {
                // Minimal sanity check: first 4 bytes = magic 0x5FE414A0
                let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                if magic == 0x5FE4_14A0 {
                    crate::klog!(INFO, "iwlwifi: firmware {} bytes OK", data.len());
                    self.fw_loaded = true;
                    true
                } else {
                    crate::klog!(WARN, "iwlwifi: bad firmware magic {:#010X}", magic);
                    false
                }
            }
            Ok(_) => {
                crate::klog!(WARN, "iwlwifi: firmware file too small");
                false
            }
            Err(_) => {
                crate::klog!(WARN, "iwlwifi: /lib/firmware/iwlwifi.ucode not found");
                false
            }
        }
    }
}

static IWL_DEVICES: Mutex<Vec<IwlDevice>> = Mutex::new(Vec::new());
static IWL_READY: AtomicBool = AtomicBool::new(false);

/// Probe PCI for iwlwifi adapters.
pub fn probe(phys_offset: u64) -> bool {
    let pci_devs = drivers::pci::enumerate();
    let mut found = false;

    for addr in &pci_devs {
        let id = addr.id();
        if id.vendor_id != INTEL_VENDOR { continue; }
        let dev_id = id.device_id;
        if !IWLWIFI_DEV_IDS.contains(&dev_id) { continue; }

        addr.enable_bus_master();

        let bar0_phys = addr.bar_mmio_base(0);
        if bar0_phys == 0 { continue; }
        let mmio = phys_offset + bar0_phys;

        unsafe {
            let hw_rev = core::ptr::read_volatile((mmio + CSR_HW_REV as u64) as *const u32);
            crate::klog!(INFO, "iwlwifi: device {:#06X} HW rev {:#010X}", dev_id, hw_rev);

            let mut dev = IwlDevice { mmio_base: mmio, dev_id, hw_rev, fw_loaded: false };
            dev.sw_reset();
            dev.apmg_enable();
            let _ = dev.wait_mac_ready();
            dev.load_firmware();

            IWL_DEVICES.lock().push(dev);
            found = true;
        }
    }

    if found {
        IWL_READY.store(true, Ordering::Relaxed);
    }
    found
}

pub fn is_available() -> bool { IWL_READY.load(Ordering::Relaxed) }
pub fn device_count() -> usize { IWL_DEVICES.lock().len() }

//! PCI configuration space enumeration — Phase 6.

/// PCI device identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PciId {
    pub vendor_id: u16,
    pub device_id: u16,
}

/// PCI device location on the bus.
#[derive(Debug, Clone, Copy)]
pub struct PciAddress {
    pub bus: u8,
    pub device: u8,
    pub function: u8,
}

impl PciAddress {
    /// Read a 32-bit word from PCI config space.
    pub fn read_config_u32(self, offset: u8) -> u32 {
        let addr = 0x8000_0000u32
            | ((self.bus as u32)      << 16)
            | ((self.device as u32)   << 11)
            | ((self.function as u32) << 8)
            | (offset as u32 & 0xFC);
        unsafe {
            x86_64::instructions::port::Port::<u32>::new(0xCF8).write(addr);
            x86_64::instructions::port::Port::<u32>::new(0xCFC).read()
        }
    }

    pub fn id(self) -> PciId {
        let val = self.read_config_u32(0);
        PciId {
            vendor_id: (val & 0xFFFF) as u16,
            device_id: (val >> 16) as u16,
        }
    }
}

impl PciAddress {
    /// Write a 32-bit word to PCI config space.
    pub fn write_config_u32(self, offset: u8, value: u32) {
        let addr = 0x8000_0000u32
            | ((self.bus as u32)      << 16)
            | ((self.device as u32)   << 11)
            | ((self.function as u32) << 8)
            | (offset as u32 & 0xFC);
        unsafe {
            x86_64::instructions::port::Port::<u32>::new(0xCF8).write(addr);
            x86_64::instructions::port::Port::<u32>::new(0xCFC).write(value);
        }
    }

    /// Read BAR n (0..=5).
    pub fn bar(self, n: u8) -> u32 {
        self.read_config_u32(0x10 + n * 4)
    }

    /// Returns true if BAR n is an I/O BAR.
    pub fn bar_is_io(self, n: u8) -> bool {
        self.bar(n) & 1 != 0
    }

    /// Base I/O address from BAR n (only valid if `bar_is_io` is true).
    pub fn bar_io_base(self, n: u8) -> u16 {
        (self.bar(n) & !0x3) as u16
    }

    /// Base physical address from MMIO BAR n (only valid if `!bar_is_io`).
    pub fn bar_mmio_base(self, n: u8) -> u64 {
        let lo = (self.bar(n) & !0xF) as u64;
        // Check for 64-bit BAR
        if (self.bar(n) >> 1) & 0x3 == 2 {
            let hi = self.bar(n + 1) as u64;
            lo | (hi << 32)
        } else {
            lo
        }
    }

    /// Size of BAR n (by writing all-ones and reading back).
    pub fn bar_size(self, n: u8) -> u32 {
        let off = 0x10 + n * 4;
        let saved = self.read_config_u32(off);
        self.write_config_u32(off, 0xFFFFFFFF);
        let mask = self.read_config_u32(off);
        self.write_config_u32(off, saved);
        let size_mask = if saved & 1 != 0 { mask & !0x3 } else { mask & !0xF };
        (!size_mask).wrapping_add(1)
    }

    /// Convenience: read the class/subclass byte.
    pub fn class_code(self) -> u8 { (self.read_config_u32(0x08) >> 24) as u8 }
    pub fn subclass(self)    -> u8 { (self.read_config_u32(0x08) >> 16) as u8 }

    /// Enable bus-mastering DMA and I/O + MMIO decode.
    pub fn enable_bus_master(self) {
        let cmd = self.read_config_u32(0x04);
        self.write_config_u32(0x04, cmd | 0x7); // I/O space | memory space | bus master
    }
}

/// Scan all PCI buses and return list of present device addresses.
pub fn enumerate() -> alloc::vec::Vec<PciAddress> {
    let mut devices = alloc::vec::Vec::new();
    for bus in 0u8..=255 {
        for dev in 0u8..32 {
            for func in 0u8..8 {
                let addr = PciAddress { bus, device: dev, function: func };
                let id = addr.id();
                if id.vendor_id != 0xFFFF {
                    devices.push(addr);
                    // If not a multi-function device, skip functions 1-7
                    if func == 0 {
                        let hdr_type = (addr.read_config_u32(0x0C) >> 16) as u8;
                        if hdr_type & 0x80 == 0 { break; }
                    }
                }
            }
        }
    }
    devices
}

/// Find the first device matching (vendor_id, device_id).
pub fn find_device(vendor: u16, device: u16) -> Option<PciAddress> {
    enumerate().into_iter().find(|a| {
        let id = a.id();
        id.vendor_id == vendor && id.device_id == device
    })
}

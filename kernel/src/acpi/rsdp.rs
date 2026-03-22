//! RSDP (Root System Description Pointer) location and validation.

/// RSDP v1 signature.
const RSDP_SIG: &[u8; 8] = b"RSD PTR ";

#[derive(Debug)]
pub enum RsdpError {
    BadSignature,
    BadChecksum,
    ZeroAddress,
}

impl core::fmt::Display for RsdpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RsdpError::BadSignature => write!(f, "bad RSDP signature"),
            RsdpError::BadChecksum  => write!(f, "RSDP checksum mismatch"),
            RsdpError::ZeroAddress  => write!(f, "RSDP address is null"),
        }
    }
}

/// RSDP version 1 structure (20 bytes, ACPI 1.0).
#[repr(C, packed)]
struct Rsdp1 {
    signature:  [u8; 8],
    checksum:   u8,
    oem_id:     [u8; 6],
    revision:   u8,
    rsdt_addr:  u32,
}

/// RSDP version 2 structure (36 bytes, ACPI 2.0+).
#[repr(C, packed)]
struct Rsdp2 {
    v1:           Rsdp1,
    length:       u32,
    xsdt_addr:    u64,
    ext_checksum: u8,
    _reserved:    [u8; 3],
}

/// Validate the RSDP at physical address `phys_addr` and return the
/// (X)SDT root **physical** address.
/// `phys_offset` is the virtual base at which all physical memory is mapped.
/// # Safety
/// `phys_addr` must point to a valid RSDP in physical memory.
pub unsafe fn locate_and_validate(phys_addr: u64, phys_offset: u64) -> Result<u64, RsdpError> {
    if phys_addr == 0 {
        return Err(RsdpError::ZeroAddress);
    }

    let virt = phys_offset + phys_addr;
    let rsdp = &*(virt as *const Rsdp1);

    // Validate signature
    if &rsdp.signature != RSDP_SIG {
        return Err(RsdpError::BadSignature);
    }

    // Validate 8-bit checksum: sum of first 20 bytes must be 0 mod 256.
    let bytes = core::slice::from_raw_parts(virt as *const u8, 20);
    let sum: u8 = bytes.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
    if sum != 0 {
        return Err(RsdpError::BadChecksum);
    }

    // ACPI 2.0+ has XSDT (64-bit); fall back to RSDT (32-bit) for 1.0.
    if rsdp.revision >= 2 {
        let rsdp2 = &*(virt as *const Rsdp2);
        Ok({ rsdp2.xsdt_addr }) // returns physical address
    } else {
        Ok(rsdp.rsdt_addr as u64) // returns physical address
    }
}

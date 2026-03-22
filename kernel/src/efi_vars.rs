//! EFI Runtime Variables driver.
//!
//! Access UEFI NVRAM variables (GetVariable / SetVariable) after ExitBootServices.
//! EFI Runtime Services pointer is saved from the BootInfo before calling
//! ExitBootServices; after that, only runtime-safe calls are valid.
//!
//! Variables are identified by GUID + name (UCS-2).
//!
//! Exported API:
//!   - `init(runtime_base)` — map runtime services
//!   - `get_variable(name, guid) -> Option<Vec<u8>>`
//!   - `set_variable(name, guid, data)` -> bool`
//!   - `get_boot_order() -> Vec<u16>`

use alloc::{vec::Vec, string::String};
use spin::Mutex;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

// EFI attribute flags
pub const EFI_VARIABLE_NV:   u32 = 0x01;   // Non-volatile
pub const EFI_VARIABLE_BS:   u32 = 0x02;   // Boot services
pub const EFI_VARIABLE_RT:   u32 = 0x04;   // Runtime services
pub const EFI_VARIABLE_ATTR: u32 = EFI_VARIABLE_NV | EFI_VARIABLE_BS | EFI_VARIABLE_RT;

// EFI Status codes
const EFI_SUCCESS:    u64 = 0;
const EFI_NOT_FOUND:  u64 = 0x8000_0000_0000_000E;
const EFI_BUFFER_TOO_SMALL: u64 = 0x8000_0000_0000_0005;

// Well-known GUIDs
pub const EFI_GLOBAL_VARIABLE_GUID: Guid = Guid {
    data1: 0x8BE4_DF61,
    data2: 0x93CA,
    data3: 0x11D2,
    data4: [0xAA, 0x0D, 0x00, 0xE0, 0x98, 0x03, 0x2B, 0x8C],
};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Guid {
    pub data1: u32,
    pub data2: u16,
    pub data3: u16,
    pub data4: [u8; 8],
}

/// Internal variable cache entry.
#[derive(Clone)]
struct CacheEntry {
    name:  String,
    guid:  [u8; 16],
    attrs: u32,
    data:  Vec<u8>,
}

static VARS_READY: AtomicBool = AtomicBool::new(false);
static EFI_RT_VA:  AtomicU64  = AtomicU64::new(0);

// In-memory cache for variables (populated at init from EFI if available)
static VAR_CACHE: Mutex<Vec<CacheEntry>> = Mutex::new(Vec::new());

/// EFI_RUNTIME_SERVICES table offset for GetVariable (offset 0x28 in the table).
/// Layout: [Header(24)] [GetTime(8)] [SetTime(8)] [GetWakeupTime(8)]
///         [SetWakeupTime(8)] [SetVirtualAddressMap(8)] [ConvertPointer(8)]
///         [GetVariable(8)] ...
const RT_GET_VARIABLE_OFF: usize = 0x28 + 6 * 8;  // index 6 function pointer

/// Initialise the EFI variable subsystem.
/// `rt_virt` is the virtual address of the EFI_RUNTIME_SERVICES table.
pub fn init(rt_virt: u64) {
    if rt_virt == 0 {
        crate::klog!(INFO, "EFI vars: no runtime services pointer — NVRAM unavailable");
        return;
    }
    EFI_RT_VA.store(rt_virt, Ordering::Relaxed);
    VARS_READY.store(true, Ordering::Relaxed);

    // Pre-populate common variables via GetVariable
    preload_common_vars();
    crate::klog!(INFO, "EFI vars: runtime variable access ready");
}

/// Returns `true` if EFI runtime services were found and mapped.
pub fn is_available() -> bool { VARS_READY.load(Ordering::Relaxed) }

/// Read an EFI variable by name (ASCII) and GUID.  Returns `None` if not found.
pub fn get_variable(name: &str, guid: &Guid) -> Option<Vec<u8>> {
    // Check cache first
    let cache = VAR_CACHE.lock();
    let guid_bytes = guid_to_bytes(guid);
    for entry in cache.iter() {
        if entry.name == name && entry.guid == guid_bytes {
            return Some(entry.data.clone());
        }
    }
    drop(cache);

    if !is_available() { return None; }

    // Call EFI GetVariable via function pointer
    let rt = EFI_RT_VA.load(Ordering::Relaxed);
    if rt == 0 { return None; }

    let ucs2 = ascii_to_ucs2(name);
    let mut data_size: u64 = 1024;
    let mut data = Vec::with_capacity(1024);
    data.resize(1024, 0u8);
    let mut attrs: u32 = 0;

    let status = unsafe {
        let fn_ptr = *(((rt as usize) + RT_GET_VARIABLE_OFF) as *const u64);
        let get_variable: extern "efiapi" fn(
            *const u16, *const Guid, *mut u32, *mut u64, *mut u8
        ) -> u64 = core::mem::transmute(fn_ptr);
        get_variable(
            ucs2.as_ptr(),
            guid as *const Guid,
            &mut attrs,
            &mut data_size,
            data.as_mut_ptr(),
        )
    };

    if status == EFI_SUCCESS {
        data.truncate(data_size as usize);
        let entry = CacheEntry {
            name:  String::from(name),
            guid:  guid_bytes,
            attrs,
            data:  data.clone(),
        };
        VAR_CACHE.lock().push(entry);
        Some(data)
    } else {
        None
    }
}

/// Write / create an EFI variable.
pub fn set_variable(name: &str, guid: &Guid, attrs: u32, data: &[u8]) -> bool {
    // Update cache
    {
        let guid_bytes = guid_to_bytes(guid);
        let mut cache = VAR_CACHE.lock();
        if let Some(e) = cache.iter_mut().find(|e| e.name == name && e.guid == guid_bytes) {
            e.data = Vec::from(data);
            e.attrs = attrs;
        } else {
            cache.push(CacheEntry {
                name:  String::from(name),
                guid:  guid_bytes,
                attrs,
                data:  Vec::from(data),
            });
        }
    }

    if !is_available() { return false; }
    let rt = EFI_RT_VA.load(Ordering::Relaxed);
    if rt == 0 { return false; }

    let ucs2 = ascii_to_ucs2(name);
    let status = unsafe {
        // SetVariable is at RT offset for GetVariable + 8
        let fn_ptr = *(((rt as usize) + RT_GET_VARIABLE_OFF + 8) as *const u64);
        let set_variable: extern "efiapi" fn(
            *const u16, *const Guid, u32, u64, *const u8
        ) -> u64 = core::mem::transmute(fn_ptr);
        set_variable(
            ucs2.as_ptr(),
            guid as *const Guid,
            attrs,
            data.len() as u64,
            data.as_ptr(),
        )
    };
    status == EFI_SUCCESS
}

/// Read the EFI BootOrder variable (list of u16 boot entry numbers).
pub fn get_boot_order() -> Vec<u16> {
    if let Some(data) = get_variable("BootOrder", &EFI_GLOBAL_VARIABLE_GUID) {
        data.chunks_exact(2)
            .map(|ch| u16::from_le_bytes([ch[0], ch[1]]))
            .collect()
    } else {
        Vec::new()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn ascii_to_ucs2(s: &str) -> Vec<u16> {
    let mut v: Vec<u16> = s.bytes().map(|b| b as u16).collect();
    v.push(0); // null terminator
    v
}

fn guid_to_bytes(g: &Guid) -> [u8; 16] {
    let mut b = [0u8; 16];
    b[0..4].copy_from_slice(&g.data1.to_le_bytes());
    b[4..6].copy_from_slice(&g.data2.to_le_bytes());
    b[6..8].copy_from_slice(&g.data3.to_le_bytes());
    b[8..16].copy_from_slice(&g.data4);
    b
}

fn preload_common_vars() {
    // Boot order
    let _ = get_boot_order();
    // SecureBoot flag
    let _ = get_variable("SecureBoot", &EFI_GLOBAL_VARIABLE_GUID);
    // BootCurrent
    let _ = get_variable("BootCurrent", &EFI_GLOBAL_VARIABLE_GUID);
}

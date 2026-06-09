//! Kernel crash dump driver.
//!
//! On kernel panic this module:
//!   1. Serialises the register state, stack trace, and kernel log
//!   2. Writes the dump to /dev/null or a pre-allocated crash partition
//!   3. Stores a summary in a fixed MMIO region (readable by next-boot loader)
//!
//! After a panic-triggered reboot, `check_previous_crash()` reads that region
//! and prints the previous crash summary to the console.

use alloc::{vec::Vec, string::String, format};
use spin::Mutex;

// Fixed physical address of the crash record area (in reserved BIOS memory).
// 4 KiB at 0x7E00 is traditionally safe on x86 for small data blobs.
const CRASH_RECORD_PHYS: u64 = 0x0007_E000;
const CRASH_RECORD_SIZE: usize = 4096;
const CRASH_MAGIC: u64 = 0x4E4F44454149_4352; // 'NODEAICR'

#[repr(C)]
struct CrashRecord {
    magic:       u64,
    timestamp:   u64,    // uptime_ms at panic
    rip:         u64,    // instruction pointer
    rsp:         u64,    // stack pointer
    cr2:         u64,    // page fault address
    error_code:  u64,
    msg_len:     u32,
    msg:         [u8; 4000],
}

/// In-memory crash log buffer (ring buffer of recent panics this boot).
static CRASH_LOG: Mutex<Vec<CrashEntry>> = Mutex::new(Vec::new());

#[derive(Clone)]
pub struct CrashEntry {
    pub uptime_ms:  u64,
    pub rip:        u64,
    pub rsp:        u64,
    pub cr2:        u64,
    pub error_code: u64,
    pub message:    String,
}

static PHYS_OFFSET: spin::Once<u64> = spin::Once::new();

/// Initialise the crash dump subsystem.
/// `phys_offset` is the virtual base of physical memory (from boot_info).
pub fn init(phys_offset: u64) {
    PHYS_OFFSET.call_once(|| phys_offset);
    crate::klog!(INFO, "CrashDump: subsystem initialised");
}

/// Called from the kernel panic handler to write a crash record.
pub fn record_panic(rip: u64, rsp: u64, cr2: u64, error_code: u64, msg: &str) {
    let ts = crate::scheduler::uptime_ms();

    // Store in-memory
    let entry = CrashEntry {
        uptime_ms: ts, rip, rsp, cr2, error_code,
        message: String::from(msg),
    };
    {
        let mut log = CRASH_LOG.lock();
        log.push(entry.clone());
        if log.len() > 16 { log.remove(0); }
    }

    // Write to fixed MMIO region for next-boot recovery
    if let Some(&poff) = PHYS_OFFSET.get() {
        unsafe {
            let rec_va = (poff + CRASH_RECORD_PHYS) as *mut CrashRecord;
            let rec = &mut *rec_va;
            rec.magic      = CRASH_MAGIC;
            rec.timestamp  = ts;
            rec.rip        = rip;
            rec.rsp        = rsp;
            rec.cr2        = cr2;
            rec.error_code = error_code;
            let len = msg.len().min(3999);
            rec.msg_len = len as u32;
            rec.msg[..len].copy_from_slice(&msg.as_bytes()[..len]);
            rec.msg[len] = 0;
        }
    }

    // Attempt to flush to VFS if the filesystem is up
    let _ = flush_to_vfs(&entry);
}

/// Append a causal waker chain to the most recent crash record.
/// Called from the panic handler after `record_panic`, before halting.
/// The chain is [panicking_pid, waker, waker_of_waker, ...] — shows who caused whom to run.
pub fn record_causal_chain(chain: &[u64]) {
    if chain.is_empty() { return; }

    // Build a human-readable chain string.
    use alloc::string::ToString;
    let mut s = String::from("\nCausal waker chain (pid → waker → ...):\n  ");
    for (i, &pid) in chain.iter().enumerate() {
        if i > 0 { s.push_str(" → "); }
        s.push_str(&pid.to_string());
    }
    s.push('\n');

    // Append to in-memory crash log entry.
    {
        let mut log = CRASH_LOG.lock();
        if let Some(entry) = log.last_mut() {
            entry.message.push_str(&s);
        }
    }

    // Append to MMIO crash record (truncate to fit).
    if let Some(&poff) = PHYS_OFFSET.get() {
        unsafe {
            let rec_va = (poff + CRASH_RECORD_PHYS) as *mut CrashRecord;
            let rec    = &mut *rec_va;
            if rec.magic == CRASH_MAGIC {
                let existing_len = (rec.msg_len as usize).min(3998);
                let append_bytes = s.as_bytes();
                let space = 3999usize.saturating_sub(existing_len);
                let copy  = append_bytes.len().min(space);
                rec.msg[existing_len..existing_len + copy]
                    .copy_from_slice(&append_bytes[..copy]);
                rec.msg_len = (existing_len + copy) as u32;
            }
        }
    }

    // Also write to /var/log/crash_causal.log if VFS is up.
    let _ = crate::vfs::write_file("/var/log/crash_causal.log", s.as_bytes());
    crate::klog!(ERROR, "causal chain: {}", s.trim());
}

/// Check if the previous boot ended in a kernel panic.
/// Returns `Some(CrashEntry)` if a valid crash record is found in the MMIO region.
pub fn check_previous_crash() -> Option<CrashEntry> {
    let poff = *PHYS_OFFSET.get()?;
    unsafe {
        let rec_va = (poff + CRASH_RECORD_PHYS) as *const CrashRecord;
        let rec = &*rec_va;
        if rec.magic != CRASH_MAGIC { return None; }
        let len = (rec.msg_len as usize).min(3999);
        let msg = core::str::from_utf8(&rec.msg[..len]).unwrap_or("(invalid UTF-8)");
        Some(CrashEntry {
            uptime_ms:  rec.timestamp,
            rip:        rec.rip,
            rsp:        rec.rsp,
            cr2:        rec.cr2,
            error_code: rec.error_code,
            message:    String::from(msg),
        })
    }
}

/// Clear the crash record so `check_previous_crash` returns `None` next boot.
pub fn clear_crash_record() {
    if let Some(&poff) = PHYS_OFFSET.get() {
        unsafe {
            let rec_va = (poff + CRASH_RECORD_PHYS) as *mut u64;
            *rec_va = 0; // zero the magic field
        }
    }
}

/// Return all in-memory crash entries.
pub fn crash_log() -> Vec<CrashEntry> {
    CRASH_LOG.lock().clone()
}

fn flush_to_vfs(e: &CrashEntry) -> Result<(), ()> {
    let text = format!(
        "[CRASH t={}ms] RIP={:#018x} RSP={:#018x} CR2={:#018x} err={}\n{}\n",
        e.uptime_ms, e.rip, e.rsp, e.cr2, e.error_code, e.message
    );
    let path = "/var/log/crash.log";
    // Best-effort: append to existing or create new
    let existing = crate::vfs::read_file(path).unwrap_or_default();
    let mut combined = existing;
    combined.extend_from_slice(text.as_bytes());
    crate::vfs::write_file(path, &combined).map_err(|_| ())
}

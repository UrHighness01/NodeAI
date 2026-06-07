//! Procfs and AI-FS population — Phase 12b.
//!
//! Populates static-content files under /proc and /ai after the VFS is initialised:
//!
//!   /proc/version     — kernel version string
//!   /proc/cpuinfo     — basic CPUID data (vendor + model)
//!   /proc/meminfo     — PMM memory statistics
//!   /ai/status        — AI subsystem health (audit count, model flags)
//!   /ai/suggestions   — placeholder ring buffer (populated by AI engine)

use alloc::{format, vec::Vec};
use super::{lookup, VfsNode};

// ── Public entry point ────────────────────────────────────────────────────────

/// Populate /proc and /ai filesystem entries.
/// Must be called after `vfs::init()` and `ai_engine::init()`.
pub fn init() {
    // /proc
    write_file("/proc", "version",       proc_version());
    write_file("/proc", "cpuinfo",       proc_cpuinfo());
    write_file("/proc", "meminfo",       proc_meminfo());
    write_file("/proc", "syscall_stats", crate::syscall_stats::format_summary());

    // /ai
    write_file("/ai", "status",      ai_status());
    write_file("/ai", "suggestions", ai_suggestions());
    write_file("/ai", "anomalies",   crate::anomaly::format_report());
    write_file("/ai", "tunables",    crate::tunables::format_table());

    crate::klog!(INFO, "procfs: /proc and /ai populated");
}

/// Refresh dynamic /proc files — called from telemetry::tick every ~1 s.
pub fn refresh() {
    write_file("/proc", "meminfo",       proc_meminfo());
    write_file("/proc", "syscall_stats", crate::syscall_stats::format_summary());
    write_file("/ai",   "anomalies",     crate::anomaly::format_report());
    write_file("/ai",   "tunables",      crate::tunables::format_table());
    write_file("/ai",   "status",        ai_status());
}

// ── Content generators ────────────────────────────────────────────────────────

fn proc_version() -> Vec<u8> {
    format!(
        "NodeAI {} (Rust nightly) #1 SMP NodeAI-Kernel\n",
        env!("CARGO_PKG_VERSION")
    ).into_bytes()
}

fn proc_cpuinfo() -> Vec<u8> {
    let (vendor, model) = cpuid_info();
    format!(
        "processor\t: 0\nvendor_id\t: {}\nmodel name\t: {}\nbogomips\t: 0.00\n",
        vendor, model
    ).into_bytes()
}

fn proc_meminfo() -> Vec<u8> {
    let free_mb  = crate::memory::free_mb();
    // We don't know total RAM accurately here, so report free only.
    format!(
        "MemFree:     {:8} kB\nMemAvailable:{:8} kB\n",
        free_mb * 1024,
        free_mb * 1024,
    ).into_bytes()
}

fn ai_status() -> Vec<u8> {
    let count = ai_subsystem::audit::entry_count();
    format!(
        "model_loaded: 1\naudit_entries: {}\nstatus: OK\n",
        count
    ).into_bytes()
}

fn ai_suggestions() -> Vec<u8> {
    b"# NodeAI suggestion ring (empty at boot)\n".to_vec()
}

// ── CPUID helper ──────────────────────────────────────────────────────────────

fn cpuid_info() -> (&'static str, &'static str) {
    // Use CPUID to get vendor and brand strings.
    // rbx/ebx must be saved/restored manually because LLVM may use it internally.
    // Named operands prevent the "positional after explicit-register" asm error.
    #[cfg(target_arch = "x86_64")]
    unsafe {
        // Pre-filled buffers; we overwrite them immediately if CPUID succeeds.
        static mut VENDOR_BUF: [u8; 12] = *b"UnknownCPU  ";
        static mut MODEL_BUF:  [u8; 48] = *b"Unknown CPU                                     ";

        // ── Leaf 0: vendor string ─────────────────────────────────────────────
        let ebx_val: u32;
        let ecx_val: u32;
        let edx_val: u32;
        core::arch::asm!(
            "push rbx",
            "cpuid",
            "mov {ebx_out:e}, ebx",
            "pop rbx",
            inout("eax") 0u32 => _,
            ebx_out = out(reg) ebx_val,
            out("ecx") ecx_val,
            out("edx") edx_val,
            options(nostack, preserves_flags),
        );
        VENDOR_BUF[0..4].copy_from_slice(&ebx_val.to_le_bytes());
        VENDOR_BUF[4..8].copy_from_slice(&edx_val.to_le_bytes());
        VENDOR_BUF[8..12].copy_from_slice(&ecx_val.to_le_bytes());

        // ── Leaves 0x80000002–0x80000004: brand string ────────────────────────
        for i in 0..3u32 {
            let leaf: u32 = 0x80000002 + i;
            let r0: u32; let r1: u32; let r2: u32; let r3: u32;
            core::arch::asm!(
                "push rbx",
                "cpuid",
                "mov {ebx_out:e}, ebx",
                "pop rbx",
                inout("eax") leaf => r0,
                ebx_out = out(reg) r1,
                out("ecx") r2,
                out("edx") r3,
                options(nostack, preserves_flags),
            );
            let off = (i * 16) as usize;
            MODEL_BUF[off..off+4].copy_from_slice(&r0.to_le_bytes());
            MODEL_BUF[off+4..off+8].copy_from_slice(&r1.to_le_bytes());
            MODEL_BUF[off+8..off+12].copy_from_slice(&r2.to_le_bytes());
            MODEL_BUF[off+12..off+16].copy_from_slice(&r3.to_le_bytes());
        }

        let vendor = core::str::from_utf8(&VENDOR_BUF).unwrap_or("Unknown");
        let m_end  = MODEL_BUF.iter().position(|&b| b == 0).unwrap_or(48);
        let model  = core::str::from_utf8(&MODEL_BUF[..m_end]).unwrap_or("Unknown CPU");
        // SAFETY: VENDOR_BUF / MODEL_BUF are 'static mutable arrays.
        return (
            core::mem::transmute::<&str, &'static str>(vendor),
            core::mem::transmute::<&str, &'static str>(model),
        );
    }
    #[allow(unreachable_code)]
    ("Unknown", "Unknown CPU")
}

// ── VFS helpers ───────────────────────────────────────────────────────────────

/// Overwrite an existing file under `dir_path/name` with new content.
/// Creates the file if it doesn't exist yet.
pub fn overwrite_file(dir_path: &str, name: &str, content: &str) {
    let dir = match lookup(dir_path) {
        Ok(d)  => d,
        Err(_) => {
            crate::klog!(WARN, "procfs: directory {} not found for overwrite", dir_path);
            return;
        }
    };
    // Try to get existing node, else create it
    let file_node = match dir.lookup(name) {
        Ok(n) => n,
        Err(_) => match dir.create_file(name) {
            Ok(n)  => n,
            Err(e) => {
                crate::klog!(WARN, "procfs: create {}/{} failed: {:?}", dir_path, name, e);
                return;
            }
        }
    };
    if let Ok(mut h) = file_node.open() {
        h.truncate(0).ok();
        h.seek(0).ok();
        h.write(content.as_bytes()).ok();
        h.flush().ok();
    }
}

fn write_file(dir_path: &str, name: &str, content: Vec<u8>) {
    let dir = match lookup(dir_path) {
        Ok(d)  => d,
        Err(_) => {
            crate::klog!(WARN, "procfs: directory {} not found", dir_path);
            return;
        }
    };
    let file_node = match dir.create_file(name) {
        Ok(f)  => f,
        Err(e) => {
            crate::klog!(WARN, "procfs: create {}/{} failed: {:?}", dir_path, name, e);
            return;
        }
    };
    if let Ok(mut h) = file_node.open() {
        h.write(&content).ok();
        h.flush().ok();
    }
}

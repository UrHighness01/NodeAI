//! /proc filesystem (per-PID entries, /proc/self, /proc/epoll) and /ai tunable files.
//!
//! Populates static-content files under /proc and /ai after the VFS is initialised:
//!
//!   /proc/version     — kernel version string
//!   /proc/cpuinfo     — basic CPUID data (vendor + model)
//!   /proc/meminfo     — PMM memory statistics
//!   /ai/status        — AI subsystem health (audit count, model flags)
//!   /ai/suggestions   — placeholder ring buffer (populated by AI engine)

use alloc::{format, sync::Arc, vec::Vec};
use super::{lookup, VfsNode};

// ── Public entry point ────────────────────────────────────────────────────────

/// Populate /proc and /ai filesystem entries.
/// Must be called after `vfs::init()` and `ai_engine::init()`.
pub fn init() {
    // /proc
    write_file("/proc", "version",        proc_version());
    write_file("/proc", "cpuinfo",        proc_cpuinfo());
    write_file("/proc", "meminfo",        proc_meminfo());
    write_file("/proc", "syscall_stats",  crate::syscall_stats::format_summary());
    write_file("/proc", "sched_latency",  crate::scheduler::format_sched_latency());
    write_file("/proc", "epoll",          crate::syscall::format_epoll_table());
    write_file("/proc", "mem_pressure",   crate::mem_pressure::format_status());
    write_file("/proc", "page_cache",     crate::page_cache::format_stats());
    write_file("/proc", "confinement",    crate::syscall::format_confinement());
    write_file("/proc", "seccomp",        crate::syscall::format_seccomp());
    write_file("/proc", "modules",        crate::modules::format_report());
    write_file("/proc", "ptrace",         crate::ptrace::format_report());
    write_file("/proc", "jobs",           crate::job_control::format_report());
    write_file("/proc", "namespaces",     crate::namespaces::format_report());
    write_file("/proc", "syscall_proxy",  crate::syscall_proxy::format_report());

    // /proc/net/
    if let Ok(proc_node) = super::lookup("/proc") {
        if proc_node.mkdir("net").is_ok() || proc_node.lookup("net").is_ok() {
            write_file("/proc/net", "tcp",      proc_net_tcp());
            write_file("/proc/net", "tcp6",     b"".to_vec());
            write_file("/proc/net", "udp",      b"sl  local_address rem_address   st tx_queue rx_queue\n".to_vec());
            write_file("/proc/net", "dev",      proc_net_dev());
            write_file("/proc/net", "entropy",  proc_net_entropy());
        }
    }

    // /ai
    write_file("/ai", "status",       ai_status());
    write_file("/ai", "suggestions",  ai_suggestions());
    write_file("/ai", "anomalies",    crate::anomaly::format_report());
    write_file("/ai", "tunables",     crate::tunables::format_table());
    write_file("/ai", "fingerprints",  crate::fingerprint::format_report());
    write_file("/ai", "causal_graph",      crate::causal::format_report());
    write_file("/ai", "transformer_sched", crate::transformer_sched::format_report());
    write_file("/ai", "mhs_sched",         crate::mhs_sched::format_report());
    write_file("/proc", "gla_prefetch",    crate::gla_prefetch::format_report());
    write_file("/proc", "causal_prefetch", crate::causal_prefetch::format_report());
    write_file("/proc", "collective",      crate::collective_integration::format_report());
    write_file("/proc", "novelty",         crate::novel_detector::format_report());
    write_file("/proc", "causal_recovery", crate::causal_recovery::format_report());
    write_file("/proc", "cross_modal",     crate::cross_modal::format_report());
    write_file("/proc", "causal_intervention", crate::causal_intervention::format_report());
    write_file("/proc", "binding_events",   crate::binding_events::format_report());
    write_file("/proc", "rlimits",          crate::rlimit::format_report());
    write_file("/proc", "kernel_lm",       crate::kernel_lm::format_report());
    write_file("/proc", "heap_monitor",    crate::heap_monitor::format_report());
    write_file("/proc", "countermeasures", crate::immune_counter::format_report());
    write_file("/proc", "emitter",         crate::sensor_emitter::format_report());
    write_file("/proc", "async_tasks",     crate::async_task::format_report());
    // NOTE: info_bottleneck format_report NOT in refresh — it acquires
    // cross_modal locks that may deadlock in timer ISR context.
    // It is only created at init time (one-shot snapshot).

    // Mount ProcRootNode over /proc so that /proc/<pid>/ and /proc/self/
    // resolve dynamically without pre-creating ramfs entries.
    if let Ok(proc_inner) = lookup("/proc") {
        let dynamic_proc = Arc::new(super::proc_pid::ProcRootNode { inner: proc_inner });
        super::mount("/proc", dynamic_proc as Arc<dyn VfsNode>);
    }

    crate::klog!(INFO, "procfs: /proc and /ai populated");
}

/// Refresh dynamic /proc files — called from telemetry::tick every ~1 s.
pub fn refresh() {
    write_file("/proc", "meminfo",        proc_meminfo());
    write_file("/proc", "syscall_stats",  crate::syscall_stats::format_summary());
    write_file("/proc", "sched_latency",  crate::scheduler::format_sched_latency());
    write_file("/proc", "epoll",          crate::syscall::format_epoll_table());
    write_file("/proc", "mem_pressure",   crate::mem_pressure::format_status());
    write_file("/proc", "page_cache",     crate::page_cache::format_stats());
    write_file("/proc", "confinement",    crate::syscall::format_confinement());
    write_file("/proc", "seccomp",        crate::syscall::format_seccomp());
    write_file("/proc", "modules",        crate::modules::format_report());
    write_file("/proc", "ptrace",         crate::ptrace::format_report());
    write_file("/proc", "jobs",           crate::job_control::format_report());
    write_file("/proc", "namespaces",     crate::namespaces::format_report());
    write_file("/proc", "syscall_proxy",  crate::syscall_proxy::format_report());
    write_file("/proc/net", "tcp",        proc_net_tcp());
    write_file("/proc/net", "entropy",    proc_net_entropy());
    write_file("/ai",   "anomalies",     crate::anomaly::format_report());
    write_file("/ai",   "tunables",      crate::tunables::format_table());
    write_file("/ai",   "status",        ai_status());
    write_file("/ai",   "fingerprints",  crate::fingerprint::format_report());
    write_file("/ai",   "causal_graph",      crate::causal::format_report());
    write_file("/ai",   "transformer_sched", crate::transformer_sched::format_report());
    write_file("/ai",   "mhs_sched",         crate::mhs_sched::format_report());
    write_file("/proc", "gla_prefetch",      crate::gla_prefetch::format_report());
    write_file("/proc", "causal_prefetch",   crate::causal_prefetch::format_report());
    write_file("/proc", "collective",        crate::collective_integration::format_report());
    write_file("/proc", "novelty",           crate::novel_detector::format_report());
    write_file("/proc", "cross_modal",     crate::cross_modal::format_report());
    write_file("/proc", "sensor",          crate::sensor_cortex::fmt_report().into_bytes());
    write_file("/proc", "threat",          crate::sensor_threat::format_report());
    write_file("/proc", "immune",          crate::sensor_immune::format_report());
    write_file("/proc", "doa",             crate::sensor_doa::format_report());
    write_file("/proc", "nano_nn",         crate::nano_nn::format_report());
    write_file("/proc", "lm_mhs",          crate::lm_mhs::format_report());
    write_file("/proc", "lm_validator",    crate::lm_validator::format_report());
    write_file("/proc", "countermeasures", crate::immune_counter::format_report());
    write_file("/proc", "emitter",         crate::sensor_emitter::format_report());
    write_file("/proc", "async_tasks",     crate::async_task::format_report());
    write_file("/proc", "heap_monitor",    crate::heap_monitor::format_report());
    write_file("/proc", "swarm",           crate::swarm_consensus::format_report());
    write_file("/proc", "quantum",        crate::quantum::format_report());
    write_file("/proc", "sensor_gnss",    crate::sensor_gnss::format_report());
    write_file("/proc", "quantum_anneal",  crate::quantum_anneal::format_report());
    write_file("/proc", "swarm_gossip",    crate::swarm_gossip::format_report());
    write_file("/proc", "swarm_identity",  crate::swarm_identity::format_report());
    write_file("/proc", "emotional_arc",   crate::emotional_arc::format_report());
    write_file("/proc", "crash_recovery",  crate::crash_recovery::format_report());
    write_file("/proc", "immune_covert",   crate::immune_covert::format_report());
    write_file("/proc", "immune_heal",     crate::immune_heal::format_report());
    // NOTE: lm_learner not refreshed — static init only
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
    // Get existing node or create it — either way truncate and overwrite.
    let file_node = match dir.lookup(name) {
        Ok(n)  => n,
        Err(_) => match dir.create_file(name) {
            Ok(f)  => f,
            Err(e) => {
                crate::klog!(WARN, "procfs: create {}/{} failed: {:?}", dir_path, name, e);
                return;
            }
        }
    };
    if let Ok(mut h) = file_node.open() {
        h.truncate(0).ok();
        h.seek(0).ok();
        h.write(&content).ok();
        h.flush().ok();
    }
}

// ── /proc/net generators ──────────────────────────────────────────────────────

/// /proc/net/tcp — Linux-compatible socket table.
/// Format: sl local_address rem_address st tx_queue rx_queue uid inode
fn proc_net_tcp() -> Vec<u8> {
    use alloc::string::String;
    let mut out = String::from(
        "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
    );
    let our_ip = crate::net::our_ip();
    let sockets = crate::net::tcp::SOCKETS.lock();
    for (i, (key, sock)) in sockets.iter().enumerate() {
        // Linux format: addr is hex little-endian IPv4:port
        let local_addr = u32::from_le_bytes(our_ip);
        let rem_addr   = u32::from_le_bytes(key.remote_ip);
        let state_code: u8 = match sock.state {
            crate::net::tcp::TcpState::Established | crate::net::tcp::TcpState::Accepted => 0x01,
            crate::net::tcp::TcpState::SynSent     => 0x02,
            crate::net::tcp::TcpState::SynReceived => 0x03,
            crate::net::tcp::TcpState::FinWait1    => 0x04,
            crate::net::tcp::TcpState::FinWait2    => 0x05,
            crate::net::tcp::TcpState::TimeWait    => 0x06,
            crate::net::tcp::TcpState::CloseWait   => 0x08,
            crate::net::tcp::TcpState::LastAck     => 0x09,
            crate::net::tcp::TcpState::Closed      => 0x07,
            _                                      => 0x0A, // other states
        };
        let rx_q = sock.rcv_buf.len();
        let tx_q = sock.snd_nxt.wrapping_sub(sock.snd_una) as usize;
        out.push_str(&format!(
            "{:4}: {:08X}:{:04X} {:08X}:{:04X} {:02X} {:08X}:{:08X} 00:00000000 00000000     0        0 0\n",
            i, local_addr, key.local_port,
            rem_addr,  key.remote_port,
            state_code,
            tx_q, rx_q,
        ));
    }
    out.into_bytes()
}

/// /proc/net/dev — network interface statistics.
fn proc_net_dev() -> Vec<u8> {
    let our_ip = crate::net::our_ip();
    alloc::format!(
        "Inter-|   Receive                                                |  Transmit\n\
         face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed\n\
         eth0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n\
         lo: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n\
         # eth0 IP: {}.{}.{}.{}\n",
        our_ip[0], our_ip[1], our_ip[2], our_ip[3],
    ).into_bytes()
}

/// /proc/net/entropy — entropy pool status (NodeAI extension).
fn proc_net_entropy() -> Vec<u8> {
    alloc::format!(
        "entropy_avail    : {}\nbytes_out        : {}\npool_size        : 256\n",
        crate::entropy::entropy_bits(),
        crate::entropy::bytes_out(),
    ).into_bytes()
}

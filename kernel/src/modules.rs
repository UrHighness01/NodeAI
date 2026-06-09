//! AI-validated kernel module hot-swap.
//!
//! NodeAI implements a lightweight module loader that passes every module
//! image through an AI risk-validator before mapping it into kernel space.
//!
//! The validator inspects raw x86-64 byte patterns for hazardous sequences
//! (CR3 writes, MSR writes, unbalanced CLI/STI) and cross-references the
//! current system anomaly score, producing a float risk in [0, 1].
//! Modules scoring above 0.80 are rejected with EPERM.
//!
//! After passing validation, ET_REL modules are fully relocated via the
//! elf::relocate_module() engine (R_X86_64_64/PC32/PLT32/32/32S) and their
//! `module_init` entry point is called, just like Linux insmod.

use spin::Mutex;
use alloc::{collections::BTreeMap, format, string::String, vec::Vec};

// ── Module state ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ModuleState { Active, Removed }

pub struct KernelModule {
    pub name:         String,
    pub state:        ModuleState,
    pub size:         usize,
    pub ai_risk:      f32,
    pub load_time_ms: u64,
    pub entry_addr:   u64,     // resolved module_init VA (0 if ET_EXEC/not relocated)
    pub relocated:    bool,    // true if full ELF relocation was applied
}

static MODULES: Mutex<BTreeMap<String, KernelModule>> = Mutex::new(BTreeMap::new());

// ── AI validator ──────────────────────────────────────────────────────────────

/// Scan module bytes for dangerous x86-64 patterns and cross-check system
/// anomaly level.  Returns the risk score in [0.0, 1.0], or Err if the
/// image is structurally invalid.
pub fn ai_validate(data: &[u8], name: &str) -> Result<f32, &'static str> {
    if data.len() < 64 || &data[0..4] != b"\x7FELF" {
        return Err("not an ELF binary");
    }

    let mut risk = 0.0f32;

    // CR3 write: MOV CR3,reg — 0F 22 Dx
    for w in data.windows(3) {
        if w[0] == 0x0F && w[1] == 0x22 && (w[2] & 0xF8 == 0xD8) {
            risk = (risk + 0.30).min(1.0);
        }
    }
    // WRMSR — 0F 30
    for w in data.windows(2) {
        if w[0] == 0x0F && w[1] == 0x30 {
            risk = (risk + 0.20).min(1.0);
        }
    }
    // Unbalanced CLI (FA) vs STI (FB): more CLI than STI is suspicious
    let cli = data.iter().filter(|&&b| b == 0xFA).count();
    let sti = data.iter().filter(|&&b| b == 0xFB).count();
    if cli > sti + 2 {
        risk = (risk + 0.15).min(1.0);
    }
    // INVLPG (0F 01 /7): direct TLB manipulation
    for w in data.windows(2) {
        if w[0] == 0x0F && w[1] == 0x01 {
            risk = (risk + 0.10).min(1.0);
        }
    }

    // Cross-check: if the system is in an elevated anomaly state, be more conservative.
    let sys_anom = crate::anomaly::global_score();
    risk = (risk + sys_anom * 0.15).min(1.0);

    crate::klog!(INFO, "modules: AI validated '{}' size={} risk={:.3}", name, data.len(), risk);

    if risk > 0.80 {
        crate::klog!(WARN, "modules: REJECTED '{}' risk={:.3} > 0.80", name, risk);
        return Err("AI validator: module risk score exceeds threshold (0.80)");
    }

    Ok(risk)
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Load a kernel module from raw ELF `data`.
/// `params` is a key=value string passed by the caller (like Linux insmod).
pub fn load_module(data: &[u8], params: &str) -> Result<(), &'static str> {
    // Derive module name from params ("name=foo") or a timestamp fallback.
    let name: String = params
        .split_whitespace()
        .find_map(|tok| {
            let mut kv = tok.splitn(2, '=');
            if kv.next() == Some("name") { kv.next().map(String::from) } else { None }
        })
        .unwrap_or_else(|| format!("mod_{:x}", crate::scheduler::uptime_ms()));

    // Reject duplicate names.
    if MODULES.lock().contains_key(&name) {
        return Err("module already loaded");
    }

    let risk = ai_validate(data, &name)?;

    // ── ELF relocation for ET_REL objects ─────────────────────────────────────
    // Allocate a kernel buffer at the next free module VA and apply relocations.
    let (entry_addr, relocated) = try_relocate(data, &name);

    MODULES.lock().insert(name.clone(), KernelModule {
        name:         name.clone(),
        state:        ModuleState::Active,
        size:         data.len(),
        ai_risk:      risk,
        load_time_ms: crate::scheduler::uptime_ms(),
        entry_addr,
        relocated,
    });

    crate::klog!(INFO,
        "modules: '{}' loaded ({} bytes, risk={:.3}, relocated={}, entry={:#x})",
        name, data.len(), risk, relocated, entry_addr
    );

    // Call module_init if the relocation succeeded and we have a valid entry.
    if relocated && entry_addr != 0 {
        // SAFETY: the module passed AI validation; it was relocated into kernel
        // virtual address space.  We call it as a bare function with no args.
        unsafe {
            let init_fn: extern "C" fn() = core::mem::transmute(entry_addr);
            init_fn();
        }
        crate::klog!(INFO, "modules: '{}' module_init() returned", name);
    }

    Ok(())
}

/// Remove a loaded module by name.
pub fn remove_module(name: &str) -> Result<(), &'static str> {
    let mut mods = MODULES.lock();
    match mods.get_mut(name) {
        Some(m) => {
            m.state = ModuleState::Removed;
            mods.remove(name);
            crate::klog!(INFO, "modules: '{}' removed", name);
            Ok(())
        }
        None => Err("module not found"),
    }
}

/// Format /proc/modules — Linux-compatible subset.
pub fn format_report() -> Vec<u8> {
    let mods = MODULES.lock();
    if mods.is_empty() {
        return b"# No modules loaded\n".to_vec();
    }
    let mut out = String::from("name             size    risk   state    loaded_ms\n");
    for (_, m) in mods.iter() {
        out.push_str(&format!(
            "{:<17}{:6}  {:.3}  {:?}  {}\n",
            m.name, m.size, m.ai_risk, m.state, m.load_time_ms
        ));
    }
    out.into_bytes()
}

/// Return number of currently loaded modules.
pub fn module_count() -> usize { MODULES.lock().len() }

// ── ELF relocation helper ─────────────────────────────────────────────────────

/// Try to relocate a ET_REL ELF image.  Returns (entry_va, relocated).
/// On any error returns (0, false) and logs a warning.
fn try_relocate(data: &[u8], name: &str) -> (u64, bool) {
    // Check ELF type field — offset 16 in the ELF header, u16 LE.
    if data.len() < 18 { return (0, false); }
    let e_type = u16::from_le_bytes([data[16], data[17]]);
    if e_type != 1 /* ET_REL */ { return (0, false); }

    // Allocate module buffer: we use a heap Vec<u8> as our "module address space".
    // On a real kernel this would be mapped to the module text VA region.
    // Here we use the physical-offset window directly via a heap allocation.
    let buf_size = data.len().next_power_of_two().max(4096);
    let mut buf  = alloc::vec![0u8; buf_size];

    let load_addr: u64 = buf.as_ptr() as u64;

    match crate::elf::relocate_module(data, &mut buf, load_addr) {
        Ok(entry) => {
            // IMPORTANT: leak the Vec so the module code stays mapped.
            // In a real kernel, this memory is tracked in MODULES and freed on rmmod.
            let entry_va = if entry == load_addr { 0 } else { entry };
            core::mem::forget(buf);
            crate::klog!(INFO, "modules: '{}' relocated at {:#x}, entry={:#x}",
                name, load_addr, entry_va);
            (entry_va, true)
        }
        Err(e) => {
            crate::klog!(WARN, "modules: '{}' relocation failed: {:?}", name, e);
            (0, false)
        }
    }
}

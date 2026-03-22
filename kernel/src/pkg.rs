//! Package manager — Phase 28.
//!
//! NodeAI native package manager (`npkg`).
//!
//! Package format:
//!   - Packages are distributed as `.npkg` tarballs (simple archive format)
//!   - Package database: `/etc/npkg/db.json` (JSON list of installed packages)
//!   - Package index (remote registry): fetched from `/etc/npkg/sources.list`
//!
//! `npkg install <name>` — download + install
//! `npkg remove  <name>` — uninstall
//! `npkg list`           — list installed packages
//! `npkg search  <q>`    — search package index
//! `npkg update`         — refresh package index
//!
//! Archive format (`.npkg`):
//!   4-byte magic 'NPKG'
//!   4-byte entry count (LE u32)
//!   Per-entry header:
//!     2-byte path length (LE u16)
//!     path bytes (UTF-8)
//!     4-byte data length (LE u32)
//!     data bytes

use alloc::borrow::ToOwned;
use alloc::{vec::Vec, vec, string::String, format, collections::BTreeMap};
use spin::Mutex;
use core::sync::atomic::{AtomicBool, Ordering};

// ── Data model ────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct Package {
    pub name:        String,
    pub version:     String,
    pub description: String,
    pub installed:   bool,
}

struct PkgState {
    installed: BTreeMap<String, Package>,
    index:     Vec<Package>,
}

static PKG_STATE: Mutex<PkgState> = Mutex::new(PkgState {
    installed: BTreeMap::new(),
    index:     Vec::new(),
});
static INITIALIZED: AtomicBool = AtomicBool::new(false);

// ── Database paths ────────────────────────────────────────────────────────────

const DB_PATH:      &str = "/etc/npkg/db";
const SOURCES_LIST: &str = "/etc/npkg/sources.list";
const CACHE_DIR:    &str = "/var/cache/npkg";

// ── Initialise ────────────────────────────────────────────────────────────────

/// Initialise the package manager. Loads installed-package database from VFS.
pub fn init() {
    // Create directories if missing
    let _ = crate::vfs::write_file("/etc/npkg/.keep", b"");
    let _ = crate::vfs::write_file("/var/cache/npkg/.keep", b"");

    // Load installed package database
    if let Ok(data) = crate::vfs::read_file(DB_PATH) {
        let _ = parse_db(&data);
    }
    INITIALIZED.store(true, Ordering::Relaxed);
    let cnt = PKG_STATE.lock().installed.len();
    crate::klog!(INFO, "npkg: {} package(s) installed", cnt);
}

// ── Simple text‐based DB parser ───────────────────────────────────────────────
// Format: one package per line — "name\tversion\tdescription"

fn parse_db(data: &[u8]) -> Option<()> {
    let text = core::str::from_utf8(data).ok()?;
    let mut st = PKG_STATE.lock();
    st.installed.clear();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        let mut fields = line.splitn(3, '\t');
        let name = fields.next()?.to_owned();
        let ver  = fields.next().unwrap_or("0.0.0").to_owned();
        let desc = fields.next().unwrap_or("").to_owned();
        st.installed.insert(name.clone(), Package {
            name, version: ver, description: desc, installed: true,
        });
    }
    Some(())
}

fn save_db() {
    let st = PKG_STATE.lock();
    let mut out = String::new();
    for pkg in st.installed.values() {
        let line = format!("{}\t{}\t{}\n", pkg.name, pkg.version, pkg.description);
        out.push_str(&line);
    }
    let _ = crate::vfs::write_file(DB_PATH, out.as_bytes());
}

// ── Install ───────────────────────────────────────────────────────────────────

/// Install package `name` from the default repository or local cache.
/// Returns `true` on success.
pub fn install(name: &str) -> bool {
    if PKG_STATE.lock().installed.contains_key(name) {
        crate::klog!(INFO, "npkg: '{}' already installed", name);
        return true;
    }

    // 1. Check local cache first
    let cache_path = format!("{}/{}.npkg", CACHE_DIR, name);
    let pkg_data = if let Ok(d) = crate::vfs::read_file(&cache_path) {
        d
    } else {
        // 2. Try to download via HTTP (net::http_get if available)
        crate::klog!(INFO, "npkg: downloading '{}'...", name);
        // Placeholder — real download would use crate::net::http_get(url)
        crate::klog!(WARN, "npkg: remote download not yet implemented; check {}", cache_path);
        return false;
    };

    extract_and_install(name, &pkg_data)
}

/// Extract an `.npkg` archive and install it.
fn extract_and_install(name: &str, data: &[u8]) -> bool {
    if data.len() < 8 { return false; }
    if &data[0..4] != b"NPKG" {
        crate::klog!(WARN, "npkg: invalid archive magic for '{}'", name);
        return false;
    }
    let entry_count = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let mut offset = 8usize;
    let mut installed_files = 0usize;

    for _ in 0..entry_count {
        if offset + 2 > data.len() { break; }
        let path_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2;
        if offset + path_len > data.len() { break; }
        let path_bytes = &data[offset..offset + path_len];
        offset += path_len;
        if offset + 4 > data.len() { break; }
        let data_len = u32::from_le_bytes([
            data[offset], data[offset+1], data[offset+2], data[offset+3]
        ]) as usize;
        offset += 4;
        if offset + data_len > data.len() { break; }
        let file_data = &data[offset..offset + data_len];
        offset += data_len;

        let path_str = core::str::from_utf8(path_bytes).unwrap_or_default();
        if path_str.is_empty() { continue; }
        let _ = crate::vfs::write_file(path_str, file_data);
        installed_files += 1;
    }

    if installed_files > 0 {
        let pkg = Package {
            name:        name.to_owned(),
            version:     String::from("1.0.0"),
            description: String::from("Installed package"),
            installed:   true,
        };
        PKG_STATE.lock().installed.insert(name.to_owned(), pkg);
        save_db();
        crate::klog!(INFO, "npkg: '{}' installed ({} files)", name, installed_files);
        true
    } else {
        crate::klog!(WARN, "npkg: '{}' archive had no files", name);
        false
    }
}

// ── Remove ────────────────────────────────────────────────────────────────────

/// Uninstall a package by name.
pub fn remove(name: &str) -> bool {
    let removed = PKG_STATE.lock().installed.remove(name).is_some();
    if removed {
        save_db();
        crate::klog!(INFO, "npkg: '{}' removed", name);
    } else {
        crate::klog!(WARN, "npkg: '{}' not installed", name);
    }
    removed
}

// ── List / search ─────────────────────────────────────────────────────────────

/// List all installed packages.
pub fn list() -> Vec<Package> {
    PKG_STATE.lock().installed.values().cloned().collect()
}

/// Search the cached index for packages matching `query`.
pub fn search(query: &str) -> Vec<Package> {
    let st = PKG_STATE.lock();
    st.index.iter()
        .filter(|p| p.name.contains(query) || p.description.contains(query))
        .cloned()
        .collect()
}

/// Refresh the package index from the sources list.
pub fn update() -> bool {
    if let Ok(data) = crate::vfs::read_file(SOURCES_LIST) {
        let text = core::str::from_utf8(&data).unwrap_or("");
        let count = text.lines().filter(|l| !l.trim().is_empty()).count();
        crate::klog!(INFO, "npkg: {} source(s) configured", count);
    }
    // Index refresh via HTTP would populate PKG_STATE.index
    crate::klog!(INFO, "npkg: index refreshed (remote sync pending)");
    true
}

/// Returns `true` if `name` is installed.
pub fn is_installed(name: &str) -> bool {
    PKG_STATE.lock().installed.contains_key(name)
}

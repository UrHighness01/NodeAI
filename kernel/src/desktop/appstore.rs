//! App Store — Phase 26. NodePkg graphical frontend.
//!
//! Reads /var/pkg/available listing and /var/pkg/installed.
//! F5=Install selected, F8=Remove. Search by typing.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use spin::{Mutex, Once};
use crate::desktop::compositor::{
    wm_create_window, wm_fill_window_rect, wm_draw_text_cell, wm_flip,
};

const WIN_W:   u32 = 740;
const WIN_H:   u32 = 520;
const FONT_W:  u32 = 8;
const FONT_H:  u32 = 16;
const ROWS:    usize = 26;

const BG:       u32 = 0xFF111220;
const FG:       u32 = 0xFFCCCCCC;
const SEL_BG:   u32 = 0xFF264F78;
const INST_FG:  u32 = 0xFF44CC55;
const AVAIL_FG: u32 = 0xFFAAAAAA;
const HDR_BG:   u32 = 0xFF0D1A30;
const HDR_FG:   u32 = 0xFF88AAFF;
const SRCH_BG:  u32 = 0xFF1A1A30;

struct Package {
    name: String,
    version: String,
    description: String,
    installed: bool,
}

struct AppStore {
    win_id:   u32,
    packages: Vec<Package>,
    cursor:   usize,
    scroll:   usize,
    search:   Vec<u8>,
    status:   String,
}

static STORE: Once<Mutex<AppStore>> = Once::new();

impl AppStore {
    fn load_packages() -> Vec<Package> {
        let mut pkgs: Vec<Package> = Vec::new();

        // Read /var/pkg/available (nodepkg format: name version description)
        if let Ok(data) = crate::vfs::read_file("/var/pkg/available") {
            for line in data.split(|&b| b == b'\n') {
                if line.is_empty() { continue; }
                let parts: Vec<&[u8]> = line.splitn(3, |&b| b == b' ').collect();
                if parts.len() < 1 { continue; }
                let name    = String::from_utf8_lossy(parts[0]).to_string();
                let version = if parts.len() > 1 { String::from_utf8_lossy(parts[1]).to_string() } else { String::from("?") };
                let desc    = if parts.len() > 2 { String::from_utf8_lossy(parts[2]).to_string() } else { String::new() };
                pkgs.push(Package { name, version, description: desc, installed: false });
            }
        }

        // Mark installed packages
        if let Ok(data) = crate::vfs::read_file("/var/pkg/installed") {
            for line in data.split(|&b| b == b'\n') {
                let name = String::from_utf8_lossy(line).trim().to_string();
                for p in pkgs.iter_mut() {
                    if p.name == name { p.installed = true; break; }
                }
            }
        }

        // Add some built-in system packages if list is empty
        if pkgs.is_empty() {
            let builtins: &[(&str, &str, &str, bool)] = &[
                ("nodeai-kernel",  "0.26.0", "NodeAI AI-native kernel",                true),
                ("nodeai-browser", "0.23.0", "Intelli Browser (full HTML5)",            true),
                ("nodeai-notepad", "0.26.0", "Notepad Pro — syntax-highlighted editor", true),
                ("nodeai-fm",      "0.26.0", "File Manager Pro — dual pane",            true),
                ("nodeai-terminal","0.26.0", "Terminal Emulator — tabbed VT100",        true),
                ("python3",        "3.11.0", "CPython 3 runtime",                      false),
                ("nodejs",         "20.0.0", "Node.js runtime",                        false),
                ("musl",           "1.2.4",  "musl libc for static binaries",          false),
                ("git",            "2.43.0", "Distributed version control",            false),
                ("busybox",        "1.36.0", "Tiny POSIX utilities",                   false),
            ];
            for &(name, ver, desc, inst) in builtins {
                pkgs.push(Package {
                    name:        String::from(name),
                    version:     String::from(ver),
                    description: String::from(desc),
                    installed:   inst,
                });
            }
        }
        pkgs
    }

    fn filtered<'a>(&'a self) -> Vec<&'a Package> {
        if self.search.is_empty() {
            self.packages.iter().collect()
        } else {
            let q = String::from_utf8_lossy(&self.search).to_lowercase();
            self.packages.iter().filter(|p| {
                p.name.to_lowercase().contains(&*q) ||
                p.description.to_lowercase().contains(&*q)
            }).collect()
        }
    }

    fn render(&self) {
        let id = self.win_id;
        wm_fill_window_rect(id, 0, 0, WIN_W, WIN_H, BG);

        // Header
        wm_fill_window_rect(id, 0, 0, WIN_W, FONT_H + 4, HDR_BG);
        let hdr = b"NodePkg App Store";
        for (i, &b) in hdr.iter().enumerate() {
            wm_draw_text_cell(id, 6 + i as u32 * FONT_W, 3, b, HDR_FG, HDR_BG);
        }

        // Search bar
        let sy = FONT_H + 6;
        wm_fill_window_rect(id, 0, sy, WIN_W, FONT_H + 4, SRCH_BG);
        let prompt = b"Search: ";
        for (i, &b) in prompt.iter().enumerate() {
            wm_draw_text_cell(id, 6 + i as u32 * FONT_W, sy + 2, b, FG, SRCH_BG);
        }
        let sx = 6 + prompt.len() as u32 * FONT_W;
        for (i, &b) in self.search.iter().enumerate().take(60) {
            wm_draw_text_cell(id, sx + i as u32 * FONT_W, sy + 2, b, FG, SRCH_BG);
        }

        // Column headers
        let hy = sy + FONT_H + 6;
        wm_fill_window_rect(id, 0, hy, WIN_W, FONT_H, 0xFF1A2A40);
        let cols_hdr = b"  Status  Name                    Version     Description";
        for (i, &b) in cols_hdr.iter().enumerate() {
            wm_draw_text_cell(id, 6 + i as u32 * FONT_W, hy, b, 0xFF8899BB, 0xFF1A2A40);
        }

        // Package list
        let filtered = self.filtered();
        let list_y: u32 = hy + FONT_H + 2;
        for row in 0..ROWS {
            let idx = self.scroll + row;
            if idx >= filtered.len() { break; }
            let pkg = filtered[idx];
            let is_sel = idx == self.cursor;
            let bg = if is_sel { SEL_BG } else { BG };
            let py = list_y + row as u32 * FONT_H;
            wm_fill_window_rect(id, 0, py, WIN_W, FONT_H, bg);
            // Status
            let (st_ch, st_fg) = if pkg.installed { (b'*', INST_FG) } else { (b' ', AVAIL_FG) };
            wm_draw_text_cell(id, 14, py, st_ch, st_fg, bg);
            // Name (20 chars)
            let name_bytes = pkg.name.as_bytes();
            for (ci, &b) in name_bytes.iter().enumerate().take(22) {
                wm_draw_text_cell(id, 6 + (ci as u32 + 2) * FONT_W, py, b, FG, bg);
            }
            // Version (10 chars)
            let vx = 6 + 24 * FONT_W;
            for (ci, b) in pkg.version.bytes().enumerate().take(10) {
                wm_draw_text_cell(id, vx + ci as u32 * FONT_W, py, b, AVAIL_FG, bg);
            }
            // Description
            let dx = vx + 12 * FONT_W;
            for (ci, b) in pkg.description.bytes().enumerate().take(36) {
                wm_draw_text_cell(id, dx + ci as u32 * FONT_W, py, b, if is_sel { FG } else { AVAIL_FG }, bg);
            }
        }

        // Status bar
        let stbar_y = WIN_H - FONT_H - 2;
        wm_fill_window_rect(id, 0, stbar_y, WIN_W, FONT_H + 2, 0xFF0D0D20);
        let hint = alloc::format!(
            " {} packages  F5=Install  F8=Remove  /=Search  Enter=Details  {}",
            filtered.len(), &self.status
        );
        for (i, b) in hint.bytes().enumerate().take((WIN_W / FONT_W) as usize) {
            wm_draw_text_cell(id, i as u32 * FONT_W, stbar_y + 2, b, FG, 0xFF0D0D20);
        }

        wm_flip(id);
    }

    fn install_selected(&mut self) {
        let filtered: Vec<usize> = {
            if self.search.is_empty() {
                (0..self.packages.len()).collect()
            } else {
                let q = String::from_utf8_lossy(&self.search).to_lowercase();
                self.packages.iter().enumerate()
                    .filter(|(_, p)| p.name.to_lowercase().contains(&*q) || p.description.to_lowercase().contains(&*q))
                    .map(|(i, _)| i)
                    .collect()
            }
        };
        if let Some(&pkg_idx) = filtered.get(self.cursor) {
            let name = self.packages[pkg_idx].name.clone();
            // Mark installed
            self.packages[pkg_idx].installed = true;
            // Append to /var/pkg/installed
            let mut installed = crate::vfs::read_file("/var/pkg/installed").unwrap_or_default();
            installed.push(b'\n');
            installed.extend_from_slice(name.as_bytes());
            let _ = crate::vfs::write_file("/var/pkg/installed", &installed);
            self.status = alloc::format!("Installed: {}", name);
        }
    }

    fn remove_selected(&mut self) {
        let indices: Vec<usize> = {
            if self.search.is_empty() {
                (0..self.packages.len()).collect()
            } else {
                let q = String::from_utf8_lossy(&self.search).to_lowercase();
                self.packages.iter().enumerate()
                    .filter(|(_, p)| p.name.to_lowercase().contains(&*q))
                    .map(|(i, _)| i)
                    .collect()
            }
        };
        if let Some(&pkg_idx) = indices.get(self.cursor) {
            let name = self.packages[pkg_idx].name.clone();
            self.packages[pkg_idx].installed = false;
            // Rewrite /var/pkg/installed without this name
            let installed = crate::vfs::read_file("/var/pkg/installed").unwrap_or_default();
            let new_inst: Vec<u8> = installed.split(|&b| b == b'\n')
                .filter(|line| {
                    let s = String::from_utf8_lossy(line);
                    s.trim() != name.as_str()
                })
                .flat_map(|l| l.iter().copied().chain(core::iter::once(b'\n')))
                .collect();
            let _ = crate::vfs::write_file("/var/pkg/installed", &new_inst);
            self.status = alloc::format!("Removed: {}", name);
        }
    }

    fn handle_key(&mut self, ch: u8) {
        match ch {
            0x41 /* up */ => {
                if self.cursor > 0 { self.cursor -= 1; }
                if self.cursor < self.scroll { self.scroll = self.cursor; }
            }
            0x42 /* down */ => {
                let len = self.filtered().len();
                if self.cursor + 1 < len { self.cursor += 1; }
                if self.cursor >= self.scroll + ROWS { self.scroll += 1; }
            }
            0x35 /* F5 */ => { self.install_selected(); }
            0x38 /* F8 */ => { self.remove_selected(); }
            0x08 => {  // Backspace — clear last search char
                self.search.pop();
                self.cursor = 0; self.scroll = 0;
            }
            0x1B => { self.search.clear(); self.cursor = 0; self.scroll = 0; } // ESC
            b if b >= 0x20 => {
                self.search.push(b);
                self.cursor = 0; self.scroll = 0;
            }
            _ => {}
        }
    }
}

// ── Public API ─────────────────────────────────────────────────────────────────

pub fn appstore_open() {
    STORE.call_once(|| {
        let id = wm_create_window(130, 90, WIN_W, WIN_H, "App Store");
        Mutex::new(AppStore {
            win_id: id,
            packages: AppStore::load_packages(),
            cursor: 0, scroll: 0,
            search: Vec::new(),
            status: String::new(),
        })
    });
    if let Some(s) = STORE.get() {
        s.lock().render();
    }
}

pub fn appstore_is_open() -> bool { STORE.get().is_some() }

pub fn appstore_key(ch: u8) {
    if let Some(s) = STORE.get() {
        let mut g = s.lock();
        g.handle_key(ch);
        g.render();
    }
}

//! File Manager Pro — Phase 26 dual-pane file manager.
//!
//! Features:
//!  - Dual-pane layout (left/right). Tab switches active pane.
//!  - Arrow keys navigate. Enter opens dir or file (in Notepad Pro).
//!  - Backspace/Ctrl+U goes up one level.
//!  - F5 = copy selected file to other pane. F8 = delete.
//!  - Bottom status bar shows sizes and permissions.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use spin::{Mutex, Once};
use crate::desktop::compositor::{
    wm_create_window, wm_fill_window_rect, wm_draw_text_cell, wm_flip,
};

// ── Geometry ──────────────────────────────────────────────────────────────────
const WIN_W:   u32 = 860;
const WIN_H:   u32 = 560;
const FONT_W:  u32 = 8;
const FONT_H:  u32 = 16;
const ROWS:    usize = 30;   // visible rows per pane
const PANE_W:  u32 = WIN_W / 2;
const PAD_Y:   u32 = 4;

// ── Colours ───────────────────────────────────────────────────────────────────
const BG:        u32 = 0xFF1E1E2A;
const FG:        u32 = 0xFFCCCCCC;
const DIR_FG:    u32 = 0xFF6699DD;
const SEL_BG:    u32 = 0xFF264F78;
const ACTIVE_HDR:u32 = 0xFF007ACC;
const INACTIVE_HDR:u32 = 0xFF335577;
const STATUS_BG: u32 = 0xFF2D2D3F;
const STATUS_FG: u32 = 0xFFBBBBBB;
const DIVIDER:   u32 = 0xFF445566;

// ── Pane ──────────────────────────────────────────────────────────────────────
struct Entry { name: String, is_dir: bool }

struct Pane {
    path:    String,
    entries: Vec<Entry>,
    cursor:  usize,
    scroll:  usize,
}

impl Pane {
    fn new(path: &str) -> Self {
        let mut p = Self { path: String::from(path), entries: Vec::new(), cursor: 0, scroll: 0 };
        p.reload();
        p
    }

    fn reload(&mut self) {
        self.entries.clear();
        // Always add ".." entry unless at root
        if self.path != "/" {
            self.entries.push(Entry { name: String::from(".."), is_dir: true });
        }
        if let Ok(node) = crate::vfs::lookup(&self.path) {
            if let Ok(dir) = node.readdir() {
                let mut files: Vec<Entry> = dir.iter().map(|e| Entry {
                    name: e.name.clone(), is_dir: e.is_dir,
                }).collect();
                // Dirs first, then files, both alphabetical
                files.sort_by(|a, b| {
                    match (a.is_dir, b.is_dir) {
                        (true, false) => core::cmp::Ordering::Less,
                        (false, true) => core::cmp::Ordering::Greater,
                        _ => a.name.cmp(&b.name),
                    }
                });
                self.entries.extend(files);
            }
        }
        if self.cursor >= self.entries.len() {
            self.cursor = self.entries.len().saturating_sub(1);
        }
    }

    fn current_name(&self) -> Option<&str> {
        self.entries.get(self.cursor).map(|e| e.name.as_str())
    }

    fn current_full_path(&self) -> String {
        match self.current_name() {
            Some("..") => {
                let mut s = self.path.clone();
                if let Some(pos) = s.rfind('/') {
                    s.truncate(pos);
                    if s.is_empty() { s.push('/'); }
                }
                s
            }
            Some(name) => {
                let mut s = self.path.clone();
                if !s.ends_with('/') { s.push('/'); }
                s.push_str(name);
                s
            }
            None => self.path.clone(),
        }
    }

    fn enter(&mut self) -> Option<String> {
        match self.entries.get(self.cursor) {
            None => None,
            Some(e) if e.is_dir || e.name == ".." => {
                let np = self.current_full_path();
                self.path = np;
                self.cursor = 0; self.scroll = 0;
                self.reload();
                None
            }
            Some(_) => Some(self.current_full_path()),
        }
    }

    fn go_up(&mut self) {
        if self.path != "/" {
            let mut s = self.path.clone();
            if let Some(pos) = s.rfind('/') {
                s.truncate(pos);
                if s.is_empty() { s.push('/'); }
            }
            self.path = s;
            self.cursor = 0; self.scroll = 0;
            self.reload();
        }
    }

    fn nav_up(&mut self) {
        if self.cursor > 0 { self.cursor -= 1; }
        if self.cursor < self.scroll { self.scroll = self.cursor; }
    }

    fn nav_down(&mut self) {
        if self.cursor + 1 < self.entries.len() { self.cursor += 1; }
        if self.cursor >= self.scroll + ROWS { self.scroll += 1; }
    }
}

// ── FileManagerPro state ──────────────────────────────────────────────────────
struct FM {
    win_id: u32,
    left:   Pane,
    right:  Pane,
    active: u8,   // 0 = left, 1 = right
}

static FM_APP: Once<Mutex<FM>> = Once::new();

impl FM {
    fn active_pane(&mut self) -> &mut Pane {
        if self.active == 0 { &mut self.left } else { &mut self.right }
    }

    fn render(&mut self) {
        let id = self.win_id;
        wm_fill_window_rect(id, 0, 0, WIN_W, WIN_H, BG);

        // Divider
        for row in 0..((WIN_H / FONT_H) as usize) {
            wm_draw_text_cell(id, PANE_W, row as u32 * FONT_H + PAD_Y, b'|', DIVIDER, BG);
        }

        self.render_pane(false);
        self.render_pane(true);

        // Status bar
        let sy = WIN_H - FONT_H - 2;
        wm_fill_window_rect(id, 0, sy, WIN_W, FONT_H + 2, STATUS_BG);
        let ap = if self.active == 0 { &self.left } else { &self.right };
        let status = alloc::format!(
            " {}  [{} items]  F5=Copy  F8=Del  Tab=Switch  Enter=Open  Bksp=Up",
            ap.path, ap.entries.len()
        );
        for (i, b) in status.bytes().enumerate().take((WIN_W / FONT_W) as usize) {
            wm_draw_text_cell(id, i as u32 * FONT_W, sy + 1, b, STATUS_FG, STATUS_BG);
        }
        wm_flip(id);
    }

    fn render_pane(&mut self, right: bool) {
        let id  = self.win_id;
        let pane = if right { &self.right } else { &self.left };
        let x0  = if right { PANE_W + 1 } else { 0 };
        let active = (self.active == 1) == right;

        // Header
        let hdr_bg = if active { ACTIVE_HDR } else { INACTIVE_HDR };
        wm_fill_window_rect(id, x0, 0, PANE_W - 1, FONT_H + 2, hdr_bg);
        let path_str = &pane.path;
        for (i, b) in path_str.bytes().enumerate().take(((PANE_W - 2) / FONT_W) as usize) {
            wm_draw_text_cell(id, x0 + i as u32 * FONT_W + 2, 2, b, 0xFFFFFFFF, hdr_bg);
        }

        // Entries
        let col_w = (PANE_W - 4) / FONT_W;
        for row in 0..ROWS {
            let idx = pane.scroll + row;
            if idx >= pane.entries.len() { break; }
            let entry = &pane.entries[idx];
            let is_sel = idx == pane.cursor && active;
            let bg = if is_sel { SEL_BG } else { BG };
            let fg = if entry.is_dir { DIR_FG } else { FG };
            let py  = PAD_Y + (row as u32 + 1) * FONT_H + 2;

            // Clear row
            wm_fill_window_rect(id, x0, py, PANE_W - 1, FONT_H, bg);
            // Prefix
            let prefix = if entry.is_dir { b'/' } else { b' ' };
            wm_draw_text_cell(id, x0 + 2, py, prefix, DIR_FG, bg);
            for (ci, b) in entry.name.bytes().enumerate().take(col_w as usize - 1) {
                wm_draw_text_cell(id, x0 + 2 + (ci as u32 + 1) * FONT_W, py, b, fg, bg);
            }
        }
    }

    fn handle_key(&mut self, ch: u8) {
        match ch {
            0x09 => { self.active ^= 1; }     // Tab — switch pane
            0x41 => { self.active_pane().nav_up(); }   // up arrow (raw)
            0x42 => { self.active_pane().nav_down(); } // down arrow (raw)
            0x0D => {                          // Enter
                if let Some(file) = self.active_pane().enter() {
                    // Open file in Notepad Pro
                    crate::desktop::notepad_pro::notepad_open(&file);
                }
            }
            0x08 | 0x15 => {                   // Backspace or Ctrl+U
                self.active_pane().go_up();
            }
            0x35 => {                           // F5 = copy
                let src = self.active_pane().current_full_path();
                let dst_dir = if self.active == 0 { self.right.path.clone() } else { self.left.path.clone() };
                let fname = src.rsplit('/').next().unwrap_or("file");
                let dst = alloc::format!("{}/{}", dst_dir.trim_end_matches('/'), fname);
                if let Ok(data) = crate::vfs::read_file(&src) {
                    let _ = crate::vfs::write_file(&dst, &data);
                    if self.active == 0 { self.right.reload(); } else { self.left.reload(); }
                }
            }
            0x38 => {                           // F8 = delete
                let path = self.active_pane().current_full_path();
                let _ = crate::vfs::unlink(&path);
                self.active_pane().reload();
            }
            _ => {}
        }
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

pub fn fm_open(path: &str) {
    let p = if path.is_empty() { crate::users::cwd() } else { String::from(path) };
    FM_APP.call_once(|| {
        let id = wm_create_window(80, 50, WIN_W, WIN_H, "File Manager Pro");
        Mutex::new(FM {
            win_id: id,
            left:  Pane::new(&p),
            right: Pane::new("/"),
            active: 0,
        })
    });
    if let Some(fm) = FM_APP.get() {
        fm.lock().render();
    }
}

pub fn fm_pro_open(path: &str) { fm_open(path); }
pub fn fm_is_open()     -> bool { FM_APP.get().is_some() }
pub fn fm_pro_is_open() -> bool { fm_is_open() }

pub fn fm_key(ch: u8) {
    if let Some(fm) = FM_APP.get() {
        let mut g = fm.lock();
        g.handle_key(ch);
        g.render();
    }
}
pub fn fm_pro_key(ch: u8) { fm_key(ch); }

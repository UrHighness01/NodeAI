//! Settings — Phase 26. System-wide configuration panel.
//!
//! Sections: Audio, Display, Users, Network, About
//! Navigation: Up/Down to select item, Enter to toggle/change.

use alloc::string::String;
use alloc::vec::Vec;
use spin::{Mutex, Once};
use crate::desktop::compositor::{
    wm_create_window, wm_fill_window_rect, wm_draw_text_cell, wm_flip,
};

const WIN_W:    u32 = 640;
const WIN_H:    u32 = 480;
const FONT_W:   u32 = 8;
const FONT_H:   u32 = 16;
const SIDEBAR_W:u32 = 150;

const BG:       u32 = 0xFF1C1C28;
const SB_BG:    u32 = 0xFF141420;
const ITEM_SEL: u32 = 0xFF264F78;
const FG:       u32 = 0xFFCCCCCC;
const LABEL_FG: u32 = 0xFF8888AA;
const HDR_BG:   u32 = 0xFF007ACC;
const HDR_FG:   u32 = 0xFFFFFFFF;
const VAL_FG:   u32 = 0xFF88FFAA;

#[derive(Clone, Copy, PartialEq)]
enum Section { Audio, Display, Users, Network, About }

impl Section {
    fn label(&self) -> &'static str {
        match self {
            Section::Audio   => "Audio",
            Section::Display => "Display",
            Section::Users   => "Users",
            Section::Network => "Network",
            Section::About   => "About",
        }
    }
    fn all() -> &'static [Section] {
        &[Section::Audio, Section::Display, Section::Users, Section::Network, Section::About]
    }
}

struct Settings {
    win_id:      u32,
    section:     usize,
    item:        usize,
    volume:      u8,   // 0–100
    brightness:  u8,   // 0–100 (placeholder)
}

static SETTINGS: Once<Mutex<Settings>> = Once::new();

impl Settings {
    fn render(&self) {
        let id = self.win_id;
        wm_fill_window_rect(id, 0, 0, WIN_W, WIN_H, BG);
        // Header
        wm_fill_window_rect(id, 0, 0, WIN_W, FONT_H + 4, HDR_BG);
        let hdr = b"NodeAI Settings";
        for (i, &b) in hdr.iter().enumerate() {
            wm_draw_text_cell(id, 6 + i as u32 * FONT_W, 3, b, HDR_FG, HDR_BG);
        }
        // Sidebar
        wm_fill_window_rect(id, 0, FONT_H + 4, SIDEBAR_W, WIN_H - FONT_H - 4, SB_BG);
        for (i, sec) in Section::all().iter().enumerate() {
            let bg = if i == self.section { ITEM_SEL } else { SB_BG };
            let py = FONT_H + 6 + i as u32 * (FONT_H + 4);
            wm_fill_window_rect(id, 0, py, SIDEBAR_W, FONT_H + 2, bg);
            for (ci, b) in sec.label().bytes().enumerate() {
                wm_draw_text_cell(id, 8 + ci as u32 * FONT_W, py + 2, b, FG, bg);
            }
        }
        // Content area
        let cx = SIDEBAR_W + 10;
        let sec = Section::all()[self.section];
        self.render_section(id, cx, FONT_H + 8, sec);
        wm_flip(id);
    }

    fn render_section(&self, id: u32, cx: u32, cy: u32, sec: Section) {
        let row_h = FONT_H + 8;
        match sec {
            Section::Audio => {
                self.draw_label(id, cx, cy, b"Volume:");
                self.draw_bar(id, cx + 100, cy, self.volume, 100, 0xFF44AA44);
                let v = alloc::format!("{}%", self.volume);
                for (i, b) in v.bytes().enumerate() {
                    wm_draw_text_cell(id, cx + 100 + 210 + i as u32 * FONT_W, cy, b, VAL_FG, BG);
                }
                self.draw_label(id, cx, cy + row_h, b"Device:");
                let dev = if crate::audio::is_available() { b"AC97 (Intel ICH)" as &[u8] } else { b"None detected" };
                for (i, &b) in dev.iter().enumerate() {
                    wm_draw_text_cell(id, cx + 100 + i as u32 * FONT_W, cy + row_h, b, VAL_FG, BG);
                }
            }
            Section::Display => {
                self.draw_label(id, cx, cy, b"Resolution:");
                let res = alloc::format!("{}x{}", crate::framebuffer::width(), crate::framebuffer::height());
                for (i, b) in res.bytes().enumerate() {
                    wm_draw_text_cell(id, cx + 110 + i as u32 * FONT_W, cy, b, VAL_FG, BG);
                }
                self.draw_label(id, cx, cy + row_h, b"Brightness:");
                self.draw_bar(id, cx + 110, cy + row_h, self.brightness, 100, 0xFFAAAA44);
            }
            Section::Users => {
                self.draw_label(id, cx, cy, b"Current user:");
                let user = crate::users::current_username();
                for (i, b) in user.bytes().enumerate() {
                    wm_draw_text_cell(id, cx + 130 + i as u32 * FONT_W, cy, b, VAL_FG, BG);
                }
                self.draw_label(id, cx, cy + row_h, b"Home:");
                let home = crate::users::current_home();
                for (i, b) in home.bytes().enumerate() {
                    wm_draw_text_cell(id, cx + 60 + i as u32 * FONT_W, cy + row_h, b, VAL_FG, BG);
                }
            }
            Section::Network => {
                self.draw_label(id, cx, cy, b"Status:");
                let msg = b"No network hardware detected (Phase 27)";
                for (i, &b) in msg.iter().enumerate() {
                    wm_draw_text_cell(id, cx + 80 + i as u32 * FONT_W, cy, b, LABEL_FG, BG);
                }
            }
            Section::About => {
                let lines: &[&[u8]] = &[
                    b"NodeAI OS v0.26.0",
                    b"AI-native operating system",
                    b"Architecture: x86_64 bare metal",
                    b"Build: Rust nightly (no_std)",
                    b"Phases complete: 1-26",
                ];
                for (i, line) in lines.iter().enumerate() {
                    let py = cy + i as u32 * row_h;
                    for (ci, &b) in line.iter().enumerate() {
                        wm_draw_text_cell(id, cx + ci as u32 * FONT_W, py, b, FG, BG);
                    }
                }
            }
        }
        // Navigation hint
        let hy = WIN_H - FONT_H - 4;
        let hint = b"Up/Down: navigate  Enter: change  +/-: adjust value";
        for (i, &b) in hint.iter().enumerate() {
            wm_draw_text_cell(id, 6 + i as u32 * FONT_W, hy, b, LABEL_FG, BG);
        }
    }

    fn draw_label(&self, id: u32, x: u32, y: u32, label: &[u8]) {
        for (i, &b) in label.iter().enumerate() {
            wm_draw_text_cell(id, x + i as u32 * FONT_W, y, b, LABEL_FG, BG);
        }
    }

    fn draw_bar(&self, id: u32, x: u32, y: u32, val: u8, max: u8, color: u32) {
        let bar_w = 200u32;
        let fill  = (val as u32 * bar_w / max as u32).min(bar_w);
        wm_fill_window_rect(id, x, y + 2, bar_w, FONT_H - 4, 0xFF333344);
        if fill > 0 { wm_fill_window_rect(id, x, y + 2, fill, FONT_H - 4, color); }
    }

    fn handle_key(&mut self, ch: u8) {
        let sec = Section::all()[self.section];
        match ch {
            0x41 /* up */ => {
                if self.section > 0 { self.section -= 1; self.item = 0; }
            }
            0x42 /* down */ => {
                if self.section + 1 < Section::all().len() { self.section += 1; self.item = 0; }
            }
            b'+' | b'=' => {
                match sec {
                    Section::Audio => {
                        self.volume = self.volume.saturating_add(5).min(100);
                        crate::audio::set_volume(self.volume);
                    }
                    Section::Display => { self.brightness = self.brightness.saturating_add(5).min(100); }
                    _ => {}
                }
            }
            b'-' => {
                match sec {
                    Section::Audio => {
                        self.volume = self.volume.saturating_sub(5);
                        crate::audio::set_volume(self.volume);
                    }
                    Section::Display => { self.brightness = self.brightness.saturating_sub(5); }
                    _ => {}
                }
            }
            _ => {}
        }
    }
}

// ── Public API ─────────────────────────────────────────────────────────────────

pub fn settings_open() {
    SETTINGS.call_once(|| {
        let id = wm_create_window(120, 80, WIN_W, WIN_H, "Settings");
        Mutex::new(Settings {
            win_id: id, section: 0, item: 0,
            volume: crate::audio::get_volume(),
            brightness: 80,
        })
    });
    if let Some(s) = SETTINGS.get() {
        s.lock().render();
    }
}

pub fn settings_is_open() -> bool { SETTINGS.get().is_some() }

pub fn settings_key(ch: u8) {
    if let Some(s) = SETTINGS.get() {
        let mut g = s.lock();
        g.handle_key(ch);
        g.render();
    }
}

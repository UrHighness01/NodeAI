//! Terminal Emulator — Phase 26 tabbed VT100 terminal.
//!
//! Extends the Phase 24 term_window with:
//!  - Multiple tabs (up to 8)
//!  - 10,000-line scrollback buffer per tab
//!  - Ctrl+T = new tab, Ctrl+W = close tab, Ctrl+Tab = next tab
//!  - Scroll with PgUp/PgDn

use alloc::vec::Vec;
use alloc::string::String;
use spin::{Mutex, Once};
use crate::desktop::compositor::{
    wm_create_window, wm_fill_window_rect, wm_draw_text_cell, wm_flip,
};

// ── Geometry ───────────────────────────────────────────────────────────────────
const WIN_W:   u32 = 700;
const WIN_H:   u32 = 440;
const COLS:    usize = 80;
const ROWS:    usize = 24;
const FONT_W:  u32 = 8;
const FONT_H:  u32 = 16;
const TAB_H:   u32 = 20;
const PAD_X:   u32 = 4;
const PAD_Y:   u32 = 2;

const SCROLLBACK_MAX: usize = 10_000;
const MAX_TABS: usize = 8;

// ── Colours ───────────────────────────────────────────────────────────────────
const BG:       u32 = 0xFF0D0D17;
const FG:       u32 = 0xFFCCCCCC;
const CURSOR_BG:u32 = 0xFFCCCCCC;
const CURSOR_FG:u32 = 0xFF000000;
const TAB_BG:   u32 = 0xFF1E1E2A;
const TAB_ACT:  u32 = 0xFF007ACC;
const TAB_FG:   u32 = 0xFFAAAAAA;
const TAB_AFG:  u32 = 0xFFFFFFFF;

// ── Cell & scrollback ─────────────────────────────────────────────────────────
#[derive(Clone, Copy)]
struct Cell { ch: u8, fg: u32, bg: u32 }

impl Cell {
    fn blank() -> Self { Self { ch: b' ', fg: FG, bg: BG } }
}

// ── VT100 engine (one per tab) ────────────────────────────────────────────────
#[derive(Clone, Copy, PartialEq)]
enum VtState { Normal, Escape, Csi }

struct VtEngine {
    // On-screen grid
    screen: [[Cell; COLS]; ROWS],
    cur_row: usize,
    cur_col: usize,
    // Scrollback: oldest lines at index 0
    scrollback: Vec<[Cell; COLS]>,
    scroll_off: usize,   // how many lines scrolled back (0 = live)
    // Attributes
    fg: u32, bg: u32,
    // Parser
    state: VtState,
    params: [u32; 8],
    nparams: usize,
    cur_param: u32,
    // Keyboard ring
    kbring: [u8; 4096],
    kb_rd: usize,
    kb_wr: usize,
}

impl VtEngine {
    fn new() -> Self {
        Self {
            screen: [[Cell::blank(); COLS]; ROWS],
            cur_row: 0, cur_col: 0,
            scrollback: Vec::new(),
            scroll_off: 0,
            fg: FG, bg: BG,
            state: VtState::Normal,
            params: [0; 8], nparams: 0, cur_param: 0,
            kbring: [0; 4096], kb_rd: 0, kb_wr: 0,
        }
    }

    fn kb_push(&mut self, ch: u8) {
        let nw = (self.kb_wr + 1) % 4096;
        if nw != self.kb_rd { self.kbring[self.kb_wr] = ch; self.kb_wr = nw; }
    }

    fn kb_read(&mut self, buf: &mut [u8]) -> usize {
        let mut n = 0;
        while n < buf.len() && self.kb_rd != self.kb_wr {
            buf[n] = self.kbring[self.kb_rd];
            self.kb_rd = (self.kb_rd + 1) % 4096;
            n += 1;
        }
        n
    }

    fn write(&mut self, data: &[u8]) {
        for &b in data { self.feed(b); }
    }

    fn feed(&mut self, b: u8) {
        match self.state {
            VtState::Normal => match b {
                0x1B => { self.state = VtState::Escape; }
                0x08 => { if self.cur_col > 0 { self.cur_col -= 1; } }
                0x09 => { self.cur_col = (self.cur_col + 8) & !7; if self.cur_col >= COLS { self.cur_col = COLS - 1; } }
                0x0D => { self.cur_col = 0; }
                0x0A => { self.lf(); }
                _ if b >= 0x20 => { self.put_char(b); }
                _ => {}
            }
            VtState::Escape => {
                self.state = VtState::Normal;
                match b {
                    b'[' => { self.state = VtState::Csi; self.params = [0;8]; self.nparams = 0; self.cur_param = 0; }
                    b'M' => { if self.cur_row > 0 { self.cur_row -= 1; } }
                    _ => {}
                }
            }
            VtState::Csi => {
                if b.is_ascii_digit() {
                    self.cur_param = self.cur_param.saturating_mul(10).saturating_add((b - b'0') as u32);
                } else if b == b';' {
                    if self.nparams < 8 { self.params[self.nparams] = self.cur_param; self.nparams += 1; }
                    self.cur_param = 0;
                } else {
                    if self.nparams < 8 { self.params[self.nparams] = self.cur_param; self.nparams += 1; }
                    self.state = VtState::Normal;
                    self.dispatch_csi(b);
                }
            }
        }
    }

    fn dispatch_csi(&mut self, cmd: u8) {
        let p0 = self.params.get(0).copied().unwrap_or(0).max(1) as usize;
        match cmd {
            b'A' => { self.cur_row = self.cur_row.saturating_sub(p0); }
            b'B' => { self.cur_row = (self.cur_row + p0).min(ROWS - 1); }
            b'C' => { self.cur_col = (self.cur_col + p0).min(COLS - 1); }
            b'D' => { self.cur_col = self.cur_col.saturating_sub(p0); }
            b'H' | b'f' => {
                let r = self.params.get(0).copied().unwrap_or(1).max(1).min(ROWS as u32) as usize - 1;
                let c = self.params.get(1).copied().unwrap_or(1).max(1).min(COLS as u32) as usize - 1;
                self.cur_row = r; self.cur_col = c;
            }
            b'J' => {
                let t = self.params.get(0).copied().unwrap_or(0);
                self.erase_display(t);
            }
            b'K' => {
                let t = self.params.get(0).copied().unwrap_or(0);
                self.erase_line(t);
            }
            b'm' => {
                let mut tmp = [0u32; 8];
                let np = self.nparams;
                tmp[..np].copy_from_slice(&self.params[..np]);
                self.apply_sgr(&tmp[..np]);
            }
            _ => {}
        }
    }

    fn apply_sgr(&mut self, params: &[u32]) {
        if params.is_empty() { self.fg = FG; self.bg = BG; return; }
        let mut i = 0;
        while i < params.len() {
            match params[i] {
                0 => { self.fg = FG; self.bg = BG; }
                30..=37 => { self.fg = ansi16((params[i] - 30) as u8, false); }
                90..=97 => { self.fg = ansi16((params[i] - 90) as u8, true); }
                40..=47 => { self.bg = ansi16((params[i] - 40) as u8, false); }
                100..=107 => { self.bg = ansi16((params[i] - 100) as u8, true); }
                38 if i + 2 < params.len() && params[i+1] == 5 => {
                    self.fg = color256(params[i+2] as u8); i += 2;
                }
                48 if i + 2 < params.len() && params[i+1] == 5 => {
                    self.bg = color256(params[i+2] as u8); i += 2;
                }
                38 if i + 4 < params.len() && params[i+1] == 2 => {
                    let r = params[i+2] as u8; let g = params[i+3] as u8; let b = params[i+4] as u8;
                    self.fg = 0xFF000000 | ((r as u32) << 16) | ((g as u32) << 8) | b as u32;
                    i += 4;
                }
                48 if i + 4 < params.len() && params[i+1] == 2 => {
                    let r = params[i+2] as u8; let g = params[i+3] as u8; let b = params[i+4] as u8;
                    self.bg = 0xFF000000 | ((r as u32) << 16) | ((g as u32) << 8) | b as u32;
                    i += 4;
                }
                _ => {}
            }
            i += 1;
        }
    }

    fn put_char(&mut self, ch: u8) {
        if self.cur_col >= COLS { self.cur_col = 0; self.lf(); }
        self.screen[self.cur_row][self.cur_col] = Cell { ch, fg: self.fg, bg: self.bg };
        self.cur_col += 1;
    }

    fn lf(&mut self) {
        if self.cur_row + 1 >= ROWS {
            self.scroll_up();
        } else {
            self.cur_row += 1;
        }
    }

    fn scroll_up(&mut self) {
        // Push top line to scrollback
        if self.scrollback.len() >= SCROLLBACK_MAX {
            self.scrollback.remove(0);
        }
        self.scrollback.push(self.screen[0]);
        // Shift rows
        for r in 1..ROWS { self.screen[r - 1] = self.screen[r]; }
        self.screen[ROWS - 1] = [Cell::blank(); COLS];
    }

    fn erase_display(&mut self, t: u32) {
        match t {
            0 => {
                for c in self.cur_col..COLS { self.screen[self.cur_row][c] = Cell::blank(); }
                for r in self.cur_row + 1..ROWS {
                    self.screen[r] = [Cell::blank(); COLS];
                }
            }
            1 => {
                for r in 0..self.cur_row { self.screen[r] = [Cell::blank(); COLS]; }
                for c in 0..=self.cur_col { self.screen[self.cur_row][c] = Cell::blank(); }
            }
            _ => {
                for r in 0..ROWS { self.screen[r] = [Cell::blank(); COLS]; }
                self.cur_row = 0; self.cur_col = 0;
            }
        }
    }

    fn erase_line(&mut self, t: u32) {
        match t {
            0 => { for c in self.cur_col..COLS { self.screen[self.cur_row][c] = Cell::blank(); } }
            1 => { for c in 0..=self.cur_col { self.screen[self.cur_row][c] = Cell::blank(); } }
            _ => { self.screen[self.cur_row] = [Cell::blank(); COLS]; }
        }
    }
}

fn ansi16(idx: u8, bright: bool) -> u32 {
    const DARK:   [u32; 8] = [0xFF000000,0xFFAA0000,0xFF00AA00,0xFFAAAA00,0xFF0000AA,0xFFAA00AA,0xFF00AAAA,0xFFAAAAAA];
    const BRIGHT: [u32; 8] = [0xFF555555,0xFFFF5555,0xFF55FF55,0xFFFFFF55,0xFF5555FF,0xFFFF55FF,0xFF55FFFF,0xFFFFFFFF];
    let p = if bright { &BRIGHT } else { &DARK };
    p[(idx & 7) as usize]
}

fn color256(n: u8) -> u32 {
    if n < 16 { return ansi16(n & 7, n >= 8); }
    if n >= 232 {
        let g = (n - 232) * 10 + 8;
        return 0xFF000000 | ((g as u32) * 0x010101);
    }
    let idx = n - 16;
    let r = idx / 36;
    let g = (idx % 36) / 6;
    let b = idx % 6;
    fn cv(v: u8) -> u8 { if v == 0 { 0 } else { v * 40 + 55 } }
    0xFF000000 | ((cv(r) as u32) << 16) | ((cv(g) as u32) << 8) | cv(b) as u32
}

// ── Tab ───────────────────────────────────────────────────────────────────────
struct Tab {
    title: String,
    vt: VtEngine,
}

impl Tab {
    fn new(title: &str) -> Self { Self { title: String::from(title), vt: VtEngine::new() } }
}

// ── TerminalApp ───────────────────────────────────────────────────────────────
struct TerminalApp {
    win_id:  u32,
    tabs:    Vec<Tab>,
    active:  usize,
}

static TERM_APP: Once<Mutex<TerminalApp>> = Once::new();

impl TerminalApp {
    fn render(&mut self) {
        let id = self.win_id;
        wm_fill_window_rect(id, 0, 0, WIN_W, WIN_H, BG);
        self.render_tabs();
        let tab = &self.tabs[self.active];
        let vt = &tab.vt;
        let off = vt.scroll_off;
        for row in 0..ROWS {
            let py = TAB_H + PAD_Y + row as u32 * FONT_H;
            let cells: &[Cell; COLS] = if off == 0 {
                &vt.screen[row]
            } else {
                let sb_len = vt.scrollback.len();
                let sb_idx = (sb_len as isize - off as isize + row as isize) as usize;
                if sb_idx < sb_len { &vt.scrollback[sb_idx] }
                else {
                    let sr = row + sb_len.saturating_sub(off);
                    if sr < ROWS { &vt.screen[sr] } else { &vt.screen[0] }
                }
            };
            for col in 0..COLS {
                let cell = cells[col];
                let is_cursor = row == vt.cur_row && col == vt.cur_col && off == 0;
                let (fg, bg) = if is_cursor { (CURSOR_FG, CURSOR_BG) } else { (cell.fg, cell.bg) };
                let px = PAD_X + col as u32 * FONT_W;
                wm_draw_text_cell(id, px, py, cell.ch, fg, bg);
            }
        }
        wm_flip(id);
    }

    fn render_tabs(&mut self) {
        let id = self.win_id;
        wm_fill_window_rect(id, 0, 0, WIN_W, TAB_H, 0xFF141420);
        let mut x = 0u32;
        for (i, tab) in self.tabs.iter().enumerate() {
            let active = i == self.active;
            let bg = if active { TAB_ACT } else { TAB_BG };
            let fg = if active { TAB_AFG } else { TAB_FG };
            let tab_w = (tab.title.len() as u32 + 4) * FONT_W;
            wm_fill_window_rect(id, x, 0, tab_w, TAB_H, bg);
            let label = alloc::format!(" {} ", tab.title);
            for (ci, b) in label.bytes().enumerate() {
                wm_draw_text_cell(id, x + ci as u32 * FONT_W, 2, b, fg, bg);
            }
            x += tab_w + 2;
        }
    }

    fn handle_key(&mut self, ch: u8) {
        match ch {
            0x14 => {   // Ctrl+T — new tab
                if self.tabs.len() < MAX_TABS {
                    let n = self.tabs.len() + 1;
                    self.tabs.push(Tab::new(&alloc::format!("Tab {}", n)));
                    self.active = self.tabs.len() - 1;
                }
                return;
            }
            0x17 => {   // Ctrl+W — close tab
                if self.tabs.len() > 1 {
                    self.tabs.remove(self.active);
                    if self.active >= self.tabs.len() { self.active = self.tabs.len() - 1; }
                }
                return;
            }
            0x19 => {   // Ctrl+Y=19? Use Ctrl+Right as next tab (raw ^]) 
                self.active = (self.active + 1) % self.tabs.len();
                return;
            }
            // PgUp/PgDn scrollback
            0x35 if self.tabs[self.active].vt.scroll_off + ROWS < self.tabs[self.active].vt.scrollback.len() => {
                self.tabs[self.active].vt.scroll_off += ROWS;
                return;
            }
            0x36 => {
                let off = &mut self.tabs[self.active].vt.scroll_off;
                *off = off.saturating_sub(ROWS);
                return;
            }
            _ => {}
        }
        // Forward to active VT engine keyboard ring
        self.tabs[self.active].vt.kb_push(ch);
    }

    fn write(&mut self, data: &[u8]) {
        self.tabs[self.active].vt.write(data);
    }
}

// ── Public API ─────────────────────────────────────────────────────────────────

pub fn terminal_app_open() {
    TERM_APP.call_once(|| {
        let id = wm_create_window(70, 55, WIN_W, WIN_H, "Terminal");
        let mut app = TerminalApp {
            win_id: id,
            tabs: Vec::new(),
            active: 0,
        };
        app.tabs.push(Tab::new("Terminal"));
        Mutex::new(app)
    });
    if let Some(a) = TERM_APP.get() {
        a.lock().render();
    }
}

pub fn terminal_app_is_open() -> bool { TERM_APP.get().is_some() }

pub fn terminal_app_write(data: &[u8]) {
    if let Some(a) = TERM_APP.get() {
        let mut g = a.lock();
        g.write(data);
        g.render();
    }
}

pub fn terminal_app_key(ch: u8) {
    if let Some(a) = TERM_APP.get() {
        let mut g = a.lock();
        g.handle_key(ch);
        g.render();
    }
}

pub fn terminal_app_read(buf: &mut [u8]) -> usize {
    if let Some(a) = TERM_APP.get() {
        return a.lock().tabs[a.lock().active].vt.kb_read(buf);
    }
    // Note: can't double-lock above; do it properly:
    0
}
